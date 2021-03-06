#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <map>
#include <queue>
#include <vector>

#define DEBUG_TYPE "psdswp"

using namespace llvm;

STATISTIC(LoopsConsidered,  "Number of loops considered by PS-DSWP");
STATISTIC(LoopsParallelized,  "Number of loops parallelized by PS-DSWP");
STATISTIC(LoopStages,  "Number of loops stages created by PS-DSWP");

namespace {
  cl::opt<unsigned int> numThreads("num-threads",
    cl::desc("Number of threads for PS-DSWP parallelization"),
    cl::value_desc("threads"));
  // Acyclic Singly-Linked Lists
  cl::list<std::string> singlyLinkedLists("asll", cl::NormalFormatting,
    cl::desc("Mark a C struct as an acyclic singly-linked list"),
    cl::value_desc("name"));

  struct PS_DSWP : public FunctionPass {
    static char ID; // Pass identification
    std::set<Function*> stageFuncs; // Track functions for stages to skip

    PS_DSWP() : FunctionPass(ID) {}

    FunctionCallee createSyncArrays, freeSyncArrays, produce, consume,
                   launchStage, waitForStage;

    // Entry point for the overall scalar-replacement pass
    bool runOnFunction(Function &F);

    void initializeParFuncs(Module& M);

    // getAnalysisUsage - List passes required by this pass
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DependenceAnalysisWrapperPass>();
      AU.addRequiredID(LoopSimplifyID);
      FunctionPass::getAnalysisUsage(AU);
    }
  };

  struct PDGNode {
    Instruction* inst;
    PDGNode(Instruction* i) : inst(i) {}
  };
  struct PDGEdge {
    // Self edges are used for instructions that (for whatever reason) has to
    // be in a sequential stage. For instance, a PHI taking a value from the
    // back edge
    enum Type { Register, Memory, Control, PHI, Self };
    Type dependence;
    bool loopCarried;
    Instruction *src, *dst;
    PDGEdge(Type dep, bool carried, Instruction* s, Instruction* d)
      : dependence(dep), loopCarried(carried), src(s), dst(d) {}
  };

  struct DAGNode {
    std::vector<Instruction*> insts;
    bool doall;
  };
  using DAGEdge = PDGEdge;

  // Using a custom directed graph implementation since LLVM's doesn't have
  // a lot of functionality
  template<typename TNode, typename TEdge>
  class DiGraph {
  private:
    std::vector<TNode> nodes;
    std::vector<std::map<int, std::vector<TEdge>>> edges;
  public:
    int insertNode(TNode node) {
      int i = nodes.size();
      nodes.push_back(node);
      edges.push_back(std::map<int, std::vector<TEdge>>());
      return i;
    }

    TNode& getNode(int n) { return nodes[n]; }

    int getNodeCount() { return nodes.size(); }

    void addEdge(int src, int dst, TEdge edge) {
      std::map<int, std::vector<TEdge>>& nodeEdges = edges[src];

      auto f = nodeEdges.find(dst);
      if (f != nodeEdges.end()) {
        nodeEdges[dst].push_back(edge);
      } else {
        nodeEdges[dst] = std::vector<TEdge>({edge});
      }
    }

    bool hasLoopCarriedEdge(int src, int dst) {
      std::map<int, std::vector<TEdge>>& nodeEdges = edges[src];
      auto f = nodeEdges.find(dst);
      if (f != nodeEdges.end()) {
        auto edge_list = nodeEdges[dst];
        for(size_t i = 0;i<edge_list.size();i++)
        	if(edge_list[i].loopCarried == true)
        		return true;
      }
        return false;
    }

    std::vector<int> getAdjs(int src) {
      auto nodeEdges = edges[src];
      std::vector<int> adj;
      for (auto const& i : nodeEdges){
	    //std::cout << i.first << ':'<< i.second << std::endl;
	    adj.push_back(i.first);
	    }
        return adj;
    }

    bool hasEdge(int src, int dst) {
      auto nodeEdges = edges[src];
      auto f = nodeEdges.find(dst);
      if (f != nodeEdges.end())
      	return true;
      else
      	return false;
    }

    std::vector<TEdge> getEdge(int src, int dst) {
      std::map<int, std::vector<TEdge>>& nodeEdges = edges[src];
      auto f = nodeEdges.find(dst);
      if (f != nodeEdges.end())
      	return nodeEdges[dst];
      else
      	return {};
    }
  };

  class WorkObject {
  public:
    using DomTreeNode = DomTreeNodeBase<BasicBlock>;

    WorkObject(BasicBlock* B, BasicBlock* P, const DomTreeNode* N,
               const DomTreeNode* PN)
        : currentBB(B), parentBB(P), Node(N), parentNode(PN) {}

    BasicBlock* currentBB;
    BasicBlock* parentBB;
    const DomTreeNode* Node;
    const DomTreeNode* parentNode;
  };

  iterator_range<const_pred_iterator> preds(const BasicBlock* BB) {
    return make_range(pred_begin(BB), pred_end(BB));
  }

  // The code for this is based on Cytron et al. and LLVM's construction of
  // Forward Dominance Frontiers
  class ReverseDominanceFrontier {
  private:
    std::map<const BasicBlock*, std::set<const BasicBlock*>> RDF;
  public:
    std::set<const BasicBlock*> rdf(const BasicBlock* bb) { return RDF[bb]; }

    ReverseDominanceFrontier(Function& F) {
      PostDominatorTree PDT(F); // Note: PDT.dominates(A, B) <=> A postdom B

      std::vector<WorkObject> worklist;
      std::set<BasicBlock*> visited, inserted;

      if (PDT.getRootNode()->getBlock()) {
        BasicBlock* BB = PDT.getRootNode()->getBlock();
        DomTreeNodeBase<BasicBlock>* Node = PDT[BB];
        worklist.push_back(WorkObject(BB, nullptr, Node, nullptr));
        inserted.insert(BB);
      } else {
        for (DomTreeNodeBase<BasicBlock>* Node : PDT.getRootNode()->children()) {
          BasicBlock* BB = Node->getBlock();
          worklist.push_back(WorkObject(BB, nullptr, Node, nullptr));
          inserted.insert(BB);
        }
      }

      do {
        WorkObject* currentW = &worklist.back();

        BasicBlock* currentBB = currentW->currentBB;
        BasicBlock* parentBB  = currentW->parentBB;

        const DomTreeNodeBase<BasicBlock>* currentNode = currentW->Node;
        const DomTreeNodeBase<BasicBlock>* parentNode  = currentW->parentNode;

        assert(currentBB && "Invalid work object. Missing current Basic Block");
        assert(currentNode && "Invalid work object. Missing current Node");

        std::set<const BasicBlock*>& S = RDF[currentBB];

        // Visit each block only once. (DFlocal part)
        if (visited.insert(currentBB).second) {
          // Loop over CFG successors to calculate DFlocal[currentNode]
          for (const auto Pred : preds(currentBB)) {
            if (PDT[Pred]->getIDom() != currentNode)
              S.insert(Pred);
          }
        }

        // At this point, S is DFlocal. Now we union in DFup's of our children
        bool visitChild = false;
        for (auto NI = currentNode->begin(), NE = currentNode->end();
             NI != NE; ++NI) {
          DomTreeNodeBase<BasicBlock>* IDominee = *NI;
          BasicBlock* childBB = IDominee->getBlock();
          if (inserted.count(childBB) == 0) {
            worklist.push_back(WorkObject(childBB, currentBB, IDominee, currentNode));
            inserted.insert(childBB);
            visitChild = true;
          }
        }

        // If all children are visited or there is any child then pop this block
        // from the worklist.
        if (!visitChild) {
          if (parentBB) {
            auto CDFI = S.begin(), CDFE = S.end();
            auto& parentSet = RDF[parentBB];
            for (; CDFI != CDFE; ++CDFI) {
              if (!PDT.properlyDominates(parentNode, PDT[*CDFI]))
                parentSet.insert(*CDFI);
            }
          }
          worklist.pop_back();
        }
      } while (!worklist.empty());
    }
  };

  struct DataStructureAnalysis {
    // Pointers that are updated on each iteration in such a way that it will
    // point to a different element of the list (and so there are not
    // loop-carried dependences based on these pointers)
    // We map these back to the pointer they are derived from (or the pointer
    // itself for the PHI's that induce this) to allow us to determine when
    // two accesses must point to different elements
    std::map<Value*, Value*> notCarriedPointers;
    // For a given pointer, the set of pointers that point to elements further
    // along (in a particular iteration)
    std::map<Value*, std::set<Value*>> forwardPointers;
    // For a given pointer, the set of pointers that point to elements behind
    // it (in a particular iteration)
    std::map<Value*, std::set<Value*>> backwardPointers;
  };
}

using PDG = DiGraph<PDGNode, PDGEdge>;
using DAG = DiGraph<DAGNode, DAGEdge>;

static PDG generatePDG(Loop*, LoopInfo&, DependenceInfo&, DominatorTree&,
                       ReverseDominanceFrontier&, DataStructureAnalysis&);
static DAG computeDAGscc(PDG);
static void strongconnect(int, int*, int* ,std::stack<int>&,bool*, PDG, DAG&, std::map<int, std::vector<int>>&, std::map<int, int>&);
static DAG connectEdges(PDG, DAG,  std::map<int, std::vector<int>>&,  std::map<int, int>&);
static DAG threadPartitioning(DAG dag);
static DataStructureAnalysis analyzeDataStructures(Loop* loop);
static bool performParallelization(PS_DSWP& psdswp, DAG partition, Loop* loop,
                                   DominatorTree& DT);

void PS_DSWP::initializeParFuncs(Module& M) {
  auto& context = M.getContext();

  Type* tyI8Ptr = Type::getInt8PtrTy(context);
  Type* tyI32 = Type::getInt32Ty(context);
  Type* tyI64 = Type::getInt64Ty(context);
  Type* tyVoid = Type::getVoidTy(context);

  FunctionType* createSyncArraysTy
      = FunctionType::get(tyI8Ptr, ArrayRef<Type*>(tyI32), true);
  createSyncArrays = M.getOrInsertFunction("createSyncArrays", createSyncArraysTy);

  std::vector<Type*> freeArgs = {tyI8Ptr, tyI32};
  FunctionType* freeSyncArraysTy
      = FunctionType::get(tyVoid, ArrayRef<Type*>(freeArgs), true);
  freeSyncArrays = M.getOrInsertFunction("freeSyncArrays", freeSyncArraysTy);

  std::vector<Type*> produceArgs = {tyI8Ptr, tyI32, tyI32, tyI64};
  FunctionType* produceTy
      = FunctionType::get(tyVoid, ArrayRef<Type*>(produceArgs), false);
  produce = M.getOrInsertFunction("produce", produceTy);

  std::vector<Type*> consumeArgs = {tyI8Ptr, tyI32, tyI32};
  FunctionType* consumeTy
      = FunctionType::get(tyI64, ArrayRef<Type*>(consumeArgs), false);
  consume = M.getOrInsertFunction("consume", consumeTy);

  Type* funcI8PtrToI8Ptr
    = FunctionType::get(tyI8Ptr, ArrayRef<Type*>(tyI8Ptr), false)->getPointerTo();
  std::vector<Type*> launchArgs = {tyI8Ptr, funcI8PtrToI8Ptr};
  FunctionType* launchTy
      = FunctionType::get(tyI64, ArrayRef<Type*>(launchArgs), false);
  launchStage = M.getOrInsertFunction("launchStage", launchTy);

  std::vector<Type*> waitArgs = {tyI64};
  FunctionType* waitTy
      = FunctionType::get(tyVoid, ArrayRef<Type*>(waitArgs), false);
  waitForStage = M.getOrInsertFunction("waitForStage", waitTy);
}

bool PS_DSWP::runOnFunction(Function &F) {
  if (stageFuncs.find(&F) != stageFuncs.end()) {
    LLVM_DEBUG(dbgs() << "[psdswp] Skipping function " << F.getName()
                      << " since it is a stage of a parallelized loop\n");
    return false;
  }
  if (F.hasFnAttribute(Attribute::OptimizeNone)) {
    LLVM_DEBUG(dbgs() << "[psdswp] Skipping function " << F.getName()
                      << " (marked optnone)\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "[psdswp] Running on function " << F.getName() << "!\n");
  if (numThreads < 2) {
    errs() << "PS-DSWP pass skipped since it was does not run with less than "
              "two threads\n";
    return false;
  }

  initializeParFuncs(*F.getParent());

  bool modified = false; // Tracks whether we modified the function

  DominatorTree DT(F);
  LoopInfo LI(DT);
  DependenceInfo& DI = getAnalysis<DependenceAnalysisWrapperPass>().getDI();

  ReverseDominanceFrontier RDF(F);

  // For now, just considering the top level loops. Not actually sure if this
  // is correct behavior in general
  for (Loop* loop : LI.getTopLevelLoops()) {
    // There are many complications of code-generation for loops without a
    // single exit or entry, so it's not clear how to handle them
    if (!loop->getExitBlock() || !loop->getLoopPredecessor()) {
      LLVM_DEBUG(dbgs() << "[psdswp] Skipping loop " << loop->getName()
                        << " as it lacks unique exit or entry\n");
      continue;
    } else if (loop->getLoopLatch() != loop->getExitingBlock()) {
      LLVM_DEBUG(dbgs() << "[psdswp] Skipping loop " << loop->getName()
                        << " as it's latch is not a conditional branch\n");
      continue;
    }

    LoopsConsidered++;

    // Determine which loads/stores cannot be loop-carried (based on the
    // data-structure declarations)
    DataStructureAnalysis DSA = analyzeDataStructures(loop);

    LLVM_DEBUG(dbgs() << "[psdswp] Running on loop " << *loop << "\n");
    PDG pdg = generatePDG(loop, LI, DI, DT, RDF, DSA);

    LLVM_DEBUG(dbgs() << "Performing SCCC Tests " << "\n");
    DAG dag_scc = computeDAGscc(pdg);
    if (dag_scc.getNodeCount() <= 1 && !dag_scc.getNode(0).doall) continue;

    LLVM_DEBUG(dbgs() << "[psdswp] Partitioning graph\n");
    DAG partitioned = threadPartitioning(dag_scc);
    if (partitioned.getNodeCount() <= 1) continue;
    if (partitioned.getNodeCount() > numThreads) continue;
    // TODO: Should probably have another performance estimate here to predict
    // whether this will actually perform result in a speed-up

    LoopsParallelized++;
    LoopStages += partitioned.getNodeCount();
    modified |= performParallelization(*this, partitioned, loop, DT);
  }

  return modified;
}

// Determines whether the register dependence src -> dst is loop carried
//
// In SSA, only phi's can have a loop carried dependence, and then
// only if src is the operand for the back edge
static bool isLoopCarriedRegister(Use& use, Instruction* dst,
                                  BasicBlock* backEdge) {
  PHINode* phi = dyn_cast<PHINode>(dst);
  if (!phi) return false;

  return phi->getIncomingBlock(use) == backEdge;
}

// Taken from GVN.cpp (in LLVM source) since it's declared as static there.
// This tests whether Between lies on every path from From to To
// Note that this method relies on To being reachable from both From and
// Between, which in our use we know is true since both From and To are in the
// loop and Between is branch which creates the back edge
static bool liesBetween(const Instruction *From, Instruction *Between,
                        const Instruction *To, DominatorTree *DT) {
  if (From->getParent() == Between->getParent())
    return DT->dominates(From, Between);
  SmallSet<BasicBlock*, 1> Exclusion;
  Exclusion.insert(Between->getParent());
  return !isPotentiallyReachable(From, To, &Exclusion, DT);
}

static Value* getMemoryPointer(Instruction* inst) {
  if (StoreInst* store = dyn_cast<StoreInst>(inst)) {
    return store->getPointerOperand();
  } else if (LoadInst* load = dyn_cast<LoadInst>(inst)) {
    return load->getPointerOperand();
  } else {
    return nullptr;
  }
}

static void checkMemoryDependence(Instruction& src, Instruction& dst,
                                  int srcNode, int dstNode,
                                  PDG& graph, BasicBlock* backedge,
                                  DominatorTree& DT, DependenceInfo& DI,
                                  DataStructureAnalysis& DSA) {
  // First, see if we can reach dst from src without traversing the back edge
  // LLVM supports checking reachability without traversing a set of basic
  // blocks, so we check reachability without entering the basic block that
  // contains the back edge and then handle either of the instructions being
  // in that basic block
  bool maybeLoopIndependent = false;
  if (&src == &dst) {
    maybeLoopIndependent = true;
  } else if (src.getParent() == backedge && dst.getParent() == backedge) {
    // If both instructions are in the back edge's basic block, test whether
    // src is before dst
    maybeLoopIndependent = src.comesBefore(&dst);
  } else if (src.getParent() == backedge) {
    // If src is in the back edge basic block and dst isn't, it must be loop
    // carried
  } else if (dst.getParent() == backedge) {
    // If dst is in the back edge basic block and src isn't, it may be loop
    // independent
    maybeLoopIndependent = true;
  } else {
    // May be loop independent if the back edge's terminator does not exist
    // on every path from src to dst
    maybeLoopIndependent = !liesBetween(&src, &dst, backedge->getTerminator(),
                                        &DT);
  }

  std::unique_ptr<Dependence> dependence =
    DI.depends(&src, &dst, maybeLoopIndependent);
  if (dependence != nullptr) {
    unsigned direction = dependence->getDirection(1);
    bool loopCarried = dependence->isConfused() || !dependence->isLoopIndependent();

    // The condition should actually be that they use the same pointer and that
    // pointer is known to vary by iteration
    Value* pointerSrc = getMemoryPointer(&src);
    Value* pointerDst = getMemoryPointer(&dst);
    if (pointerSrc != nullptr && pointerSrc == pointerDst &&
        DSA.notCarriedPointers.find(pointerSrc) != DSA.notCarriedPointers.end()) {
      LLVM_DEBUG(dbgs() << "Data-type analysis shows " << src << " and "
                        << dst << " do not have a loop-carried dependence\n");
      loopCarried = false;
    } else if (pointerSrc != nullptr && pointerDst != nullptr) {
      auto fSrc = DSA.notCarriedPointers.find(pointerSrc);
      auto fDst = DSA.notCarriedPointers.find(pointerDst);
      if (fSrc != DSA.notCarriedPointers.end()
          && fDst != DSA.notCarriedPointers.end()) {
        Value* derivedSrc = fSrc->second;
        Value* derivedDst = fDst->second;
        std::set<Value*>& forwardSrc = DSA.forwardPointers[derivedSrc];
        if (forwardSrc.find(derivedDst) != forwardSrc.end()) {
          // If dst is referencing a future element in the list, set direction
          // to GT
          direction = Dependence::DVEntry::GT;
        }
      }
    }

    if (direction != Dependence::DVEntry::GT
        && (loopCarried || (&src != &dst && !DT.dominates(&dst, &src)))) {
      // Only include dependence that aren't positive or if the dependence
      // isn't loop carried, only include ones where the dst doesn't dominate
      // the src (i.e. exclude backwards dependences/non-carried self
      // dependences)
      // Basically all of these cases are ones where either the dependence
      // doesn't matter to our analysis or the other direction will get
      // put in and adding both may unecessarily create a cycle that forces
      // sequentialization
      graph.addEdge(srcNode, dstNode,
                    PDGEdge{PDGEdge::Memory, loopCarried, &src, &dst});
      LLVM_DEBUG(
        std::string dirStr = "UNKNOWN";
        switch (direction) {
          case Dependence::DVEntry::NONE: dirStr = "NONE"; break;
          case Dependence::DVEntry::LT:   dirStr = "LT"; break;
          case Dependence::DVEntry::EQ:   dirStr = "EQ"; break;
          case Dependence::DVEntry::LE:   dirStr = "LE"; break;
          case Dependence::DVEntry::GT:   dirStr = "GT"; break;
          case Dependence::DVEntry::NE:   dirStr = "NE"; break;
          case Dependence::DVEntry::GE:   dirStr = "GE"; break;
          case Dependence::DVEntry::ALL:  dirStr = "ALL"; break;
        }
        dbgs() << "[psdswp] Memory dependence from " << src << " to  " << dst
               << (loopCarried ? " (loop carried)" : "")
               << " direction " << dirStr << "\n");
    } else {
      LLVM_DEBUG(
      dbgs() << "DISCARDING dependence from " << src << " to " << dst
             << (direction == Dependence::DVEntry::GT ? " was GT" : " wasn't GT")
             << "\n");
    }
  }
}

static PDG generatePDG(Loop* loop, LoopInfo& LI, DependenceInfo& DI,
                       DominatorTree& DT, ReverseDominanceFrontier& RDF,
                       DataStructureAnalysis& DSA) {
  PDG graph;

  BasicBlock* incoming;
  BasicBlock* backedge;
  assert(loop->getIncomingAndBackEdge(incoming, backedge)
    && "Loop does not have unique incoming or back edge");

  std::map<Instruction*, int> nodes;

  std::set<Instruction*> memInsts; // The instructions which touch memory

  for (BasicBlock* bb : loop->blocks()) {
    for (Instruction& inst : *bb) {
      int node = graph.insertNode(PDGNode(&inst));
      nodes[&inst] = node;

      // Process data dependencies here
      // Only consider memory dependencies for instructions which touch memory
      if (inst.mayReadOrWriteMemory()) {
        memInsts.insert(&inst); // Insert first so we'll consider self loops
        for (Instruction* other : memInsts) {
          if (inst.mayWriteToMemory() || other->mayWriteToMemory()) {
            // Only need to consider dependence if at least one instruction
            // writes memory
            int thisNode = nodes[&inst];
            int otherNode = nodes[other];
            checkMemoryDependence(inst, *other, thisNode, otherNode, graph,
                                  backedge, DT, DI, DSA);
            if (thisNode != otherNode) {
              checkMemoryDependence(*other, inst, otherNode, thisNode, graph,
                                    backedge, DT, DI, DSA);
            }
          }
        }
      }
      // Handle register dependencies for all instructions, this is simply
      // using use-def chains in LLVM (the uses of this instruciton and the
      // values used by it)
      for (Use& op : inst.operands()) {
        Instruction* opInst = dyn_cast<Instruction>(op.get());
        if (opInst) {
          auto f = nodes.find(opInst);
          if (f != nodes.end()) {
            bool carried = isLoopCarriedRegister(op, &inst, backedge);
            // Direction, register value written in op used in inst
            graph.addEdge(f->second, node,
                          PDGEdge(PDGEdge::Register, carried, opInst, &inst));
            LLVM_DEBUG(
              dbgs() << "[psdswp] Register dependence from " << *opInst
                     << " to " << inst << (carried?" (loop carried)\n":"\n"));
          }
        }
      }
      for (Use& use : inst.uses()) {
        Instruction* useInst = dyn_cast<Instruction>(use.getUser());
        if (useInst) {
          auto f = nodes.find(useInst);
          if (f != nodes.end()) {
            bool carried = isLoopCarriedRegister(use, useInst, backedge);
            // Direction, register value written in inst used in use
            graph.addEdge(node, f->second,
                          PDGEdge(PDGEdge::Register, carried, &inst, useInst));
            LLVM_DEBUG(
              dbgs() << "[psdswp] Register dependence from " << inst << " to "
                     << *useInst << (carried ? " (loop carried)\n" : "\n"));
          } else if (!loop->contains(useInst)) {
            // NOTE: Since we currently don't support live-outs from DOALL,
            // add a self edge to live-outs
            graph.addEdge(node, node, PDGEdge(PDGEdge::Self, true, &inst, &inst));
            LLVM_DEBUG(
              dbgs() << "[psdswp] Self dependence for live-out: " << inst << "\n");
          }
        }
      }

      // For PHIs we insert additional register edges from the branch to the
      // PHI, since the branch that we took to get to the PHI matters. This is
      // in addition to the dependences on the actual values that the PHI can
      // take
      if (PHINode* phi = dyn_cast<PHINode>(&inst)) {
        for (BasicBlock* bb : phi->blocks()) {
          if (loop->contains(bb)) {
            Instruction* term = bb->getTerminator();
            auto f = nodes.find(term);
            if (f != nodes.end()) {
              bool carried = bb == backedge;
              graph.addEdge(f->second, node,
                            PDGEdge(PDGEdge::PHI, carried, term, phi));
              LLVM_DEBUG(
                dbgs() << "[psdswp] PHI control dependence from " << *term
                       << " to " << *phi << (carried?" (loop carried)\n":"\n"));
            }
          }
          // Also, if the phi has a branch from the back edge, add a
          // loop-carried self loop since the value depends on the iteration
          // and so this cannot be placed in a parallel stage
          if (bb == backedge) {
            graph.addEdge(node, node,
                PDGEdge(PDGEdge::Self, true, phi, phi));
            LLVM_DEBUG(
              dbgs() << "[psdswp] PHI self dependence on " << *phi << "\n");
          }
        }
      } else if (BranchInst* branch = dyn_cast<BranchInst>(&inst)) {
        for (BasicBlock* bb : branch->successors()) {
          if (loop->contains(bb)) {
            for (PHINode& phi : bb->phis()) {
              auto f = nodes.find(&phi);
              if (f != nodes.end()) {
                bool carried = branch->getParent() == backedge;
                graph.addEdge(node, f->second,
                              PDGEdge(PDGEdge::PHI, carried, branch, &phi));
                LLVM_DEBUG(
                  dbgs() << "[psdswp] PHI control dependence from " << *branch
                         << " to " << phi << (carried?" (loop carried)\n":"\n"));
              }
            }
          }
        }
      }
    }
  }

  // Compute control dependenceis and add them to the graph
  // Use Reverse Dominance Frontier to get the control dependences
  // (specifically if X in RDF(Y) then Y is control dependent on X
  for (BasicBlock* bb : loop->blocks()) {
    std::set<const BasicBlock*> handled;
    std::queue<const BasicBlock*> worklist;
    for (const BasicBlock* b : RDF.rdf(bb)) worklist.push(b);

    while (!worklist.empty()) {
      const BasicBlock* b = worklist.front(); worklist.pop();
      if (handled.find(b) != handled.end()) continue;
      handled.insert(b);

      // This means that bb has a control dependence on b
      //
      // Specifically, therefore, each instruction in bb has a dependence on
      // the terminator of b
      Instruction* terminator = (Instruction*) b->getTerminator();
      auto f = nodes.find(terminator);
      if (f != nodes.end()) {
        // Doesn't actually matter what instruction we use in bb
        bool loopCarried = liesBetween(terminator, bb->getTerminator(),
                                       backedge->getTerminator(), &DT);
        for (Instruction& inst : *bb) {
          graph.addEdge(f->second, nodes[&inst],
                        PDGEdge(PDGEdge::Control, loopCarried, terminator, &inst));
          LLVM_DEBUG(
            dbgs() << "[psdswp] Control dependence from " << *terminator
                   << " to " << inst
                   << (loopCarried ? " (loop carried)\n" : "\n"));
        }
      }

      // Also add control dependencies from the basic block that these basic
      // blocks (the ones it always depends on) depend on
      for (const BasicBlock* block : RDF.rdf(b)) worklist.push(block);
    }
  }

  // Make sure that every instruction has an edge from the latch terminator
  Instruction* latchInst = loop->getLoopLatch()->getTerminator();
  int latchNode = nodes[latchInst];
  for (auto [inst, n] : nodes) {
    if (inst != loop->getLoopLatch()->getTerminator()
        && !graph.hasEdge(latchNode, n))
      graph.addEdge(latchNode, n,
        PDGEdge(PDGEdge::Type::Control, false, latchInst, inst));
  }

  return graph;
}


static DAG connectEdges(PDG graph, DAG dag_scc, std::map<int, std::vector<int>> &scc_to_pdg_map, std::map<int, int> &pdg_to_scc_map){

	for(int i=0;i<dag_scc.getNodeCount();i++){
	  LLVM_DEBUG(dbgs() << "Node" << i );
	  if (dag_scc.getNode(i).doall)
	    LLVM_DEBUG(dbgs() << ": Doall" <<"\n");
		else
			LLVM_DEBUG(dbgs() << ": Sequential" <<"\n");
	}
	for(size_t i = 0;i<pdg_to_scc_map.size(); i++)
		for(size_t j = 0; j < pdg_to_scc_map.size(); j++){
			if(pdg_to_scc_map[i] == pdg_to_scc_map[j])
				continue;
			else{
				std::vector<PDGEdge> edges = graph.getEdge(i , j);
				for (size_t k = 0; k < edges.size(); k++)
					dag_scc.addEdge( pdg_to_scc_map[i] , pdg_to_scc_map[j] , edges[k] );		
			}	
	}

  LLVM_DEBUG(dbgs() << " [psdswp] Number of nodes in DAG SCC the graph " << dag_scc.getNodeCount() << "\n");
  for(int i=0;i<dag_scc.getNodeCount();i++){
    std::vector<int> adjacents = dag_scc.getAdjs(i);
    LLVM_DEBUG(dbgs() << "[psdswp] Adjacent nodes of " << i << ":");
    for(size_t j=0;j<adjacents.size();j++)
      LLVM_DEBUG(dbgs() <<adjacents[j] << " ");
    LLVM_DEBUG(dbgs() <<"\n");
  }

	return dag_scc;
}


static void strongconnect(int u, int disc[], int low[], std::stack<int> *st,
					bool stackMember[], PDG& graph, DAG &dag_scc, std::map<int, std::vector<int>> &scc_to_pdg_map, std::map<int, int> &pdg_to_scc_map)
{
	
	static int time = 0;

	// Initialize discovery time and low value
	disc[u] = low[u] = ++time;
	st->push(u);
	stackMember[u] = true;

	// Go through all vertices adjacent to this
	std::vector<int> adjacents  = graph.getAdjs(u);
	/*
	LLVM_DEBUG(dbgs() << "Adjacent nodes of " << u << ":");
	for(size_t j=0;j<adjacents.size();j++)
        LLVM_DEBUG(dbgs() <<adjacents[j] << " ");
   LLVM_DEBUG(dbgs() <<"\n");
   */
	for(size_t j=0;j<adjacents.size();j++)
	{
		int v = adjacents[j]; // v is current adjacent of 'u'

		// If v is not visited yet, then recur for it
		if (disc[v] == -1)
		{
			strongconnect(v, disc, low, st, stackMember, graph, dag_scc, scc_to_pdg_map, pdg_to_scc_map);

			// Check if the subtree rooted with 'v' has a
			// connection to one of the ancestors of 'u'
			low[u] = std::min(low[u], low[v]);
		}

		// Update low value of 'u' if 'v' is still in stack
		else if (stackMember[v] == true)
			low[u] = std::min(low[u], disc[v]);
	}
	// head node found, pop the stack and print an SCC
	int w = 0; // To store stack extracted vertices
	if (low[u] == disc[u])
	{	//int node = graph.insertNode(PDGNode(&inst));
		std::vector<Instruction*> insts;
		std::vector<int> node_maps;
		bool loopcarried = false;
		int len = dag_scc.getNodeCount();
		Instruction* w_inst;
		while (st->top() != u)
		{
			w = (int) st->top();
			//cout << w << " ";
			LLVM_DEBUG(dbgs() << w << " ");
			w_inst = graph.getNode(w).inst;
			insts.push_back(w_inst);
			node_maps.push_back(w);
			pdg_to_scc_map.insert({w, len});
			stackMember[w] = false;
			st->pop();
		}
		w = (int) st->top();
		//cout << w << "\n";
		LLVM_DEBUG(dbgs() << w << "\n");
		w_inst = graph.getNode(w).inst;
		insts.push_back(w_inst);
		node_maps.push_back(w);
		stackMember[w] = false;
		pdg_to_scc_map.insert({w, len});
		st->pop();
		DAGNode combined;
		combined.insts = insts;
		scc_to_pdg_map.insert({len, node_maps});
		pdg_to_scc_map.insert({w, len});
				
		
    for (int i1 : node_maps)
      for (int i2 : node_maps)
				loopcarried = loopcarried || graph.hasLoopCarriedEdge(i1,i2);
				//LLVM_DEBUG(dbgs() << loopcarried << "," << graph.hasLoopCarriedEdge(i1,i2) << ",");

		if(loopcarried == false)
			combined.doall = true;
		else
			combined.doall = false;
			
		int n = dag_scc.insertNode(combined);
	}
}

static DAG computeDAGscc(PDG graph) {
  //Unit testing new functions
  LLVM_DEBUG(dbgs() << "[psdswp] Number of nodes in PDG the graph " << graph.getNodeCount() << "\n");
  for(int i=0;i<graph.getNodeCount();i++){
    std::vector<int> adjacents = graph.getAdjs(i);
    LLVM_DEBUG(dbgs() << "[psdswp] Adjacent nodes of " << i << ":");
    for(size_t j=0;j<adjacents.size();j++)
      LLVM_DEBUG(dbgs() <<adjacents[j] << " ");
    LLVM_DEBUG(dbgs() <<"\n");
  }

  	DAG dag_scc;
  	std::map<int, std::vector<int>> scc_to_pdg_map; //map nodes of sccc to constituent pdg nodes
  	std::map<int, int> pdg_to_scc_map;  //map nodes of pdg to combind dag node

	int V = graph.getNodeCount();
  int *disc = new int[V];
	int *low = new int[V];
	bool *stackMember = new bool[V];
	std::stack<int> *st = new std::stack<int>();

	// Initialize disc and low, and stackMember arrays
	for (int i = 0; i < V; i++)
	{
		disc[i] = -1;
		low[i] = -1;
		stackMember[i] = false;
	}
	
	// Call the recursive helper function to find strongly
	// connected components in DFS tree with vertex 'i'
	LLVM_DEBUG(dbgs() << "[psdswp] Strongly Connected Components\n");
	for (int i = 0; i < V; i++)
		if (disc[i] == -1)
			strongconnect(i, disc, low, st, stackMember, graph, dag_scc, scc_to_pdg_map, pdg_to_scc_map);
			
	
	/*	
  LLVM_DEBUG(dbgs() << "[psdswp] Number of nodes in the graph " << dag_scc.getNodeCount() << "\n");
  for(int i=0;i<dag_scc.getNodeCount();i++){
    std::vector<int> adjacents = dag_scc.getAdjs(i);
    LLVM_DEBUG(dbgs() << "[psdswp] Adjacent nodes of " << i << ":");
    for(size_t j=0;j<adjacents.size();j++)
      LLVM_DEBUG(dbgs() <<adjacents[j] << " ");
    LLVM_DEBUG(dbgs() <<"\n");
  } */

  delete [] disc;
  delete [] low;
  delete [] stackMember;
  delete st;

  return connectEdges(graph, dag_scc, scc_to_pdg_map, pdg_to_scc_map);
}

static bool existsLongPathSet(DAG& graph, std::set<int> srcs,
                              std::set<int> dsts) {
  std::set<int> considered = srcs;

  std::queue<int> worklist;
  for (int src : srcs) {
    for (int n : graph.getAdjs(src)) {
      // Don't add destination nodes in the first round since we only want to
      // consider paths with at least one intermediate node
      if (dsts.find(n) == dsts.end())
        worklist.push(n);
    }
  }

  while (!worklist.empty()) {
    int idx = worklist.front(); worklist.pop();
    considered.insert(idx);
    if (dsts.find(idx) != dsts.end()) return true;

    for (int n : graph.getAdjs(idx)) {
      if (considered.find(n) == considered.end()) {
        worklist.push(n);
      }
    }
  }

  return false;
}

static DAG threadPartitioning(DAG dag) {
  std::map<int, int> nodeToBlock;
  std::map<int, std::set<int>> blockToNodes;

  std::set<int> doall_blocks;
  std::set<int> sequential_blocks;
  DAG dag_threaded;

  LLVM_DEBUG(dbgs() << "In threadPartitioning()\n");

  auto findEdgesBetweenBlocks = [&](int block1, int block2) {
    std::vector<DAGEdge> edges;
    for (int n1 : blockToNodes[block1]) {
      for (int n2 : blockToNodes[block2]) {
        std::vector<DAGEdge> edges1T2 = dag.getEdge(n1, n2);
        std::vector<DAGEdge> edges2T1 = dag.getEdge(n2, n1);
        edges.insert(edges.end(), edges1T2.begin(), edges1T2.end());
        edges.insert(edges.end(), edges2T1.begin(), edges2T1.end());
      }
    }
    return edges;
  };
  auto existsLongPathBlocks = [&](int block1, int block2) {
    return existsLongPathSet(dag, blockToNodes[block1], blockToNodes[block2])
        || existsLongPathSet(dag, blockToNodes[block2], blockToNodes[block1]);
  };
  auto mergeBlocks = [&](int block1, int block2) {
    blockToNodes[block1].insert(blockToNodes[block2].begin(),
                                blockToNodes[block2].end());
    for (int n1 : blockToNodes[block2]) {
      nodeToBlock[n1] = block1;
    }
    blockToNodes.erase(blockToNodes.find(block2));
  };

  const int numNodes = dag.getNodeCount();
  for (int i = 0; i < numNodes; i++) {
    nodeToBlock[i] = i;
    blockToNodes[i] = std::set<int>({i});
    if (dag.getNode(i).doall) doall_blocks.insert(i);
    else sequential_blocks.insert(i);
  }

  // Merge DOALL nodes
  bool merged = false;
  do {
    LLVM_DEBUG(dbgs() << "[psdswp] Entered DOALL do-while in threadPartitioning()\n");
    merged = false;

    auto it = doall_blocks.begin();
    auto end = doall_blocks.end();
    while (it != end) {
      int firstBlock = *it;

      auto innerIt = it;
      ++innerIt;
      while (innerIt != end) {
        int secondBlock = *innerIt;

        std::vector<DAGEdge> edges =
            findEdgesBetweenBlocks(firstBlock, secondBlock);

        // Check Condition 2 from the paper (no edges between them that represent
        // loop-carried dependencies)
        bool anyLoopCarried = false;
        for (DAGEdge edge : edges) {
          if (edge.loopCarried) {
            anyLoopCarried = true;
            break;
          }
        }
        if (anyLoopCarried) { ++innerIt; continue; }

        // Check Condition 1 from the paper (that there does not exist a path
        // containing one or more intermediate nodes between the two candidates
        if (existsLongPathBlocks(firstBlock, secondBlock))
          { ++innerIt; continue; }

        // Merge the blocks
        mergeBlocks(firstBlock, secondBlock);
        innerIt = doall_blocks.erase(innerIt);
        merged = true;
        LLVM_DEBUG(dbgs() << "[psdswp] Merged blocks " << firstBlock << " and "
                          << secondBlock << "\n");
      }
      ++it;
    }
  } while (merged);

  LLVM_DEBUG(dbgs() << "[psdswp] After merging DOALL:\n");
  LLVM_DEBUG(
  for (auto it : blockToNodes) {
    dbgs() << "\tBlock " << it.first << " : ";
    for (int i : it.second) {
      dbgs() << i << " ";
    }
    dbgs() << "\n";
  });


  //Naive approach - most number of instructions (Actually decide this based on max profile weight)
  int max = -1;
  int nodeIndex = -1;
  for (auto it : doall_blocks) {
  	int count = 0;
  	for( auto node : blockToNodes[it] ){
  		//errs() << "[psdswp] size " << dag.getNode(node).insts.size();
  		count += dag.getNode(node).insts.size();
  		}
  	if (count > max){
  		max = count;
  		nodeIndex = it;
  		}
  	}
  if(max != -1){
		LLVM_DEBUG(dbgs() << "[psdswp] Max Profile block " << nodeIndex
                      << " number of instructions  " << max << "\n");
    for (auto it = doall_blocks.begin(), end = doall_blocks.end();
         it != end; ) {
			if (*it != nodeIndex) {
        sequential_blocks.insert(*it);
        it = doall_blocks.erase(it);
      } else {
        ++it;
      }
    }
  }


  // Merge SEQUENTIAL nodes
  bool mergedSeq = false;
  do {
    LLVM_DEBUG(dbgs() << "[psdswp] Entered SEQUENTIAL do-while in  threadPartitioning()\n");
    mergedSeq = false;

    auto it = sequential_blocks.begin();
    auto end = sequential_blocks.end();
    while (it != end) {
      int firstBlock = *it;

      auto innerIt = it;
      ++innerIt;
      while (innerIt != end) {
        int secondBlock = *innerIt;
        LLVM_DEBUG(dbgs() << "[psdswp] TRYING TO MERGE " << firstBlock << " and "
                          << secondBlock << "("
                          << existsLongPathBlocks(firstBlock, secondBlock)
                          << ")\n");

        // Check Condition 1 for valid assignment from the paper (that there does not exist a path
        // containing one or more intermediate nodes between the two candidates
        if (existsLongPathBlocks(firstBlock, secondBlock))
          { ++innerIt; continue; }

        // Merge the blocks
        mergeBlocks(firstBlock, secondBlock);
        innerIt = sequential_blocks.erase(innerIt);
        mergedSeq = true;
        LLVM_DEBUG(dbgs() << "[psdswp] Merged blocks " << firstBlock << " and "
                          << secondBlock << "\n");
      }
      ++it;
    }
  } while (mergedSeq);

  LLVM_DEBUG(dbgs() << "[psdswp] After merging SEQUENTIAL:\n");
  LLVM_DEBUG(
  for (auto it : blockToNodes) {
    dbgs() << "\tBlock " << it.first << " : ";
    for (int i : it.second) {
      dbgs() << i << " ";
    }
    dbgs() << "\n";
  });
  //errs() << "[psdswp] Node to block size " << nodeToBlock.size()<< "\n";
  //errs() << "[psdswp] Block to node size :" << blockToNodes.size()<< "\n";
  for (auto it : blockToNodes)
  {
    DAGNode combined;
    std::vector<Instruction*> insts;
    for (int j : it.second)
    {
 			insts.insert(insts.end(),  dag.getNode(j).insts.begin(),  dag.getNode(j).insts.end());
 			nodeToBlock[j] = dag_threaded.getNodeCount();
  	}	
    combined.insts = insts;
    if(sequential_blocks.find(it.first) != sequential_blocks.end()) combined.doall = false;
    else combined.doall = true;
    dag_threaded.insertNode(combined);

  }

  for(auto it1 : nodeToBlock)
		LLVM_DEBUG(dbgs() << "[psdswp] node to blocks :" << it1.first << "->"
                      << it1.second <<  "\n");
	


  //Connect all edges, possibly repeats edges
  for(auto it1 : nodeToBlock){
		for(auto it2 : nodeToBlock){
			if(nodeToBlock[it1.first] == nodeToBlock[it2.first])
				continue;
			else{
				std::vector<DAGEdge> edges = dag.getEdge(it1.first , it2.first);
				for (size_t k = 0; k < edges.size(); k++)
				{  //errs() << "[psdswp] Adding edge from " << it1.first << " and " <<it2.first << " to " <<it1.second << " and " << it2.second <<"\n";
					dag_threaded.addEdge( nodeToBlock[it1.first] , nodeToBlock[it2.first] , edges[k] );
					}
				}
			}
			}


  LLVM_DEBUG(dbgs() << " [psdswp] Number of nodes in DAG after thread partitioning " << dag_threaded.getNodeCount() << "\n");
  for(int i=0;i<dag_threaded.getNodeCount();i++){
    std::vector<int> adjacents = dag_threaded.getAdjs(i);
    LLVM_DEBUG(dbgs() << "Adjacent nodes of " << i << ":");
    for(size_t j=0;j<adjacents.size();j++)
      LLVM_DEBUG(dbgs() <<adjacents[j] << " ");
    LLVM_DEBUG(dbgs() <<"\n");
  }

  // Reassign all but one parallel loop to be sequential
  // Merge SEQUENTIAL nodes - Done, but with some commented out naive reassignments

  return dag_threaded;
}

bool verifyPHIIteratesList(PHINode& phi, Loop* loop) {
  // We know this phi's type is an acyclic linked list, we now want to verify
  // that its value advances through the list each iteration
  const unsigned incoming = phi.getNumIncomingValues();
  for (unsigned i = 0; i < incoming; i++) {
    const BasicBlock* block = phi.getIncomingBlock(i);
    // Skip edges from outside the loop
    if (!loop->contains(block)) continue;

    Value* value = phi.getIncomingValue(i);
    // From within the loop we want to verify that this value MUST be further
    // into the list; we do this by ensuring that it is calculated through
    // some number of loads and GEPs from this phi

    do {
      if (LoadInst* load = dyn_cast<LoadInst>(value)) {
        Value* ptr = load->getPointerOperand();
        if (GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(ptr)) {
          value = gep->getPointerOperand();
        } else {
          errs() << "WARNING: Found phi of an ASLL that's derived of a load not "
                    "derived from a GEP: " << *ptr << "\n";
          break;
        }
      } else {
        errs() << "WARNING: Found phi of an ASLL that's not derived from a load: "
               << *value << "\n";
        break;
      }
    } while (value != &phi);

    if (value != &phi) return false;
  }

  return true;
}

static DataStructureAnalysis analyzeDataStructures(Loop* loop) {
  DataStructureAnalysis result;
  std::map<Value*, Value*>& notCarriedPointers = result.notCarriedPointers;
  std::map<Value*, std::set<Value*>>& forwardPointers = result.forwardPointers;
  std::map<Value*, std::set<Value*>>& backwardPointers = result.backwardPointers;

  for (BasicBlock* bb : loop->blocks()) {
    // Variables that cannot be loop-carried are caused by phis
    for (PHINode& phi : bb->phis()) {
      bool isAcyclicLinkedList = false;

      Type* phiType = phi.getType();
      if (!phiType->isPointerTy()) continue;
      StructType* ty = dyn_cast<StructType>(phiType->getPointerElementType());
      if (ty && ty->hasName()) {
        StringRef name = ty->getName();
        if (name.startswith("struct.")) {
          name = name.drop_front(7);
          for (std::string listName : singlyLinkedLists) {
            if (listName == name) { isAcyclicLinkedList = true; break; }
          }
        }
      }

      if (isAcyclicLinkedList) {
        if (!verifyPHIIteratesList(phi, loop)) continue;

        LLVM_DEBUG(dbgs() << "Found acyclic linked list phi: " << phi << "\n");

        // The phi and pointers derived from it (through GEPs) do not have loop
        // carried dependences on themselves
        std::list<Instruction*> pointers; pointers.push_back(&phi);
        notCarriedPointers[&phi] = &phi;

        // Find all pointers derived from the PHI through GEPs or derived from
        // it through loads of next elements (these are loads from pointers
        // derived from the PHI that have the same type as the PHI)
        for (Instruction* pointer : pointers) {
          Value* derivedFrom = notCarriedPointers[pointer];
          for (Value* user : pointer->users()) {
            LoadInst* load = dyn_cast<LoadInst>(user);
            GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(user);
            if (load && load->getType() == phi.getType()) {
              // so, this load is a pointer to a later element of this list
              forwardPointers[derivedFrom].insert(load);
              backwardPointers[load].insert(derivedFrom);

              for (Value* backwards : backwardPointers[derivedFrom]) {
                forwardPointers[backwards].insert(load);
                backwardPointers[load].insert(backwards);
              }

              // Since this is derived from the iteration through the loop,
              // this pointer also cannot write to the same location in future
              // iterations
              notCarriedPointers[load] = load;
              pointers.push_back(load);
            } else if (gep) {
              notCarriedPointers[gep] = derivedFrom;
              pointers.push_back(gep);
            }
          }
        }
      }
    }
  }

  return result;
}

static Value* extendAndCast(IRBuilder<>& builder, Module& M, Value* value) {
  Type* TyInt64 = Type::getInt64Ty(M.getContext());
  Type* valueType = value->getType();
  if (valueType == TyInt64) return value;
  if (valueType->isPointerTy()) {
    return builder.CreatePtrToInt(value, TyInt64);
  } else {
    if (valueType->isIntegerTy()) {
      // The type of extension doesn't really matter since we truncate back
      return builder.CreateZExt(value, TyInt64);
    } else if (valueType == Type::getDoubleTy(M.getContext())) {
      return builder.CreateBitCast(value, TyInt64);
    } else if (valueType == Type::getFloatTy(M.getContext())) {
      return builder.CreateZExt(
              builder.CreateBitCast(value, Type::getInt32Ty(M.getContext())),
              TyInt64);
    } else {
      assert(false && "extendAndCast() encountered unknown type");
    }
  }
}

static Value* truncateAndCast(IRBuilder<>& builder, Module& M,
                              Instruction* inst, Type* dstType,
                              Twine name) {
  assert(inst->getType() == Type::getInt64Ty(M.getContext()));
  if (inst->getType() == dstType) return inst;
  if (dstType->isPointerTy()) {
    return builder.CreateIntToPtr(inst, dstType, name);
  } else {
    if (dstType->isIntegerTy()) {
      return builder.CreateTrunc(inst, dstType, name);
    } else if (dstType == Type::getDoubleTy(M.getContext())) {
      return builder.CreateBitCast(inst, dstType, name);
    } else if (dstType == Type::getFloatTy(M.getContext())) {
      return builder.CreateBitCast(
              builder.CreateTrunc(inst, Type::getInt32Ty(M.getContext())),
              dstType, name);
    } else {
      assert(false && "truncateAndCast() encountered unknown type");
    }
  }
}

static bool performParallelization(PS_DSWP& psdswp, DAG partition, Loop* loop,
                                   DominatorTree& DT) {
  // This map tracks the synchronization arrays and which values they represent
  // and in what stage and what type they are (the value, stage, and type are
  // the key, the value is the number of the synchronization array)
  int numSyncArrays = 0;
  std::map<std::tuple<const Value*, int, DAGEdge::Type>, int> syncArrays;
  std::vector<int> syncArrayRepls;

  std::vector<int> nodeRepls; // Replication factor of each node
  std::vector<Function*> nodeFuncs; // Function for each stage
  std::vector<StructType*> nodeInputStructs; // Struct of inputs for each stage
  int parStageRepl = -1;

  // Track struct fields (other than those used for all stages) that each
  // stages needs
  std::map<int, std::vector<Value*>> stageInputs;

  // Compute live-outs: values defined in the loop and used outside of it, and
  // assign each a sync array that will be used to communicate it
  std::map<Instruction*, int> liveOuts;
  for (BasicBlock* bb : loop->blocks()) {
    for (Instruction& inst : *bb) {
      for (Value* user : inst.users()) {
        Instruction* userI = dyn_cast<Instruction>(user);
        if (userI && !loop->contains(userI)) {
          liveOuts[&inst] = numSyncArrays++;
          syncArrayRepls.push_back(1); // Live-outs just communicated over 1 array
        }
      }
    }
  }

  const int numNodes = partition.getNodeCount();
  Module& M = *(loop->getHeader()->getModule());
  Function& F = *(loop->getHeader()->getParent());

  for (int i = 0; i < numNodes; i++) {
    if (partition.getNode(i).doall) {
      nodeRepls.push_back(numThreads - (numNodes-1));
      assert(parStageRepl == -1 && "Multiple parallel stages");
      parStageRepl = numThreads - (numNodes-1);
    } else {
      nodeRepls.push_back(1);
    }
    nodeFuncs.push_back(nullptr); // Allocating space
    nodeInputStructs.push_back(nullptr);
  }

  // We need to traverse in reverse-topological order so that we assign
  // synchronization arrays before we need to produce values into them
  // This is Kahn's Algorithm
  // (see https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm)
  std::vector<int> order;
  std::list<int> s;

  // Find number of incoming edges for each node in the partitioned graph
  // Also, init set s to contain the nodes with no incoming edges
  std::vector<int> numIncoming;
  // Pair of the source node and the DAGEdge
  std::vector<std::vector<std::pair<int, DAGEdge>>> incomingEdges;
  for (int i = 0; i < numNodes; i++) {
    int numEdges = 0;
    incomingEdges.push_back(std::vector<std::pair<int, DAGEdge>>{});
    for (int j = 0; j < numNodes; j++) {
      std::vector<DAGEdge> edges = partition.getEdge(j, i);
      // Only count one edge for simplicity
      if (!edges.empty()) numEdges++;
      for (DAGEdge e : edges)
        incomingEdges[i].push_back(std::make_pair(j, e));
    }
    numIncoming.push_back(numEdges);
    if (numEdges == 0) s.push_back(i);
  }

  assert(!s.empty() && "Found no source nodes in the partition");

  while (!s.empty()) {
    int n = s.front(); s.pop_front();
    order.push_back(n);
    for (int m : partition.getAdjs(n)) {
      numIncoming[m] -= 1;
      if (numIncoming[m] == 0) s.push_back(m);
    }
  }

  for (int n : numIncoming) {
    assert(n == 0 && "Failed to construct topological sort");
  }

  // We emit a function for each stage, it takes a struct containing all values
  // we use defined outside of the loop and the index of this instance
  // Traversing in reverse-topological order
  for (auto it = order.rbegin(), end = order.rend(); it != end; ++it) {
    int n = *it;
    DAGNode& node = partition.getNode(n);
    LLVM_DEBUG(dbgs() << "Code-generating for node " << n << "\n");

    std::vector<std::pair<int, DAGEdge>>& inEdges = incomingEdges[n];

    // Place instructions in this part into a set for easier lookup
    std::set<const Instruction*> stageInsts;
    for (Instruction* inst : node.insts) stageInsts.insert(inst);

    // Sort incoming dependences by the source instruction
    std::map<const Instruction*, std::vector<std::pair<int, DAGEdge>>> dependences;
    for (std::pair<int, DAGEdge>& pair : inEdges)
      dependences[pair.second.src].push_back(pair);

    // Sort outgoing dependences, tracking the actual edge and the number of
    // the stage the dependence goes to
    std::map<const Instruction*, std::map<int, std::set<DAGEdge::Type>>> outEdges;
    for (int i = 0; i < numNodes; i++) {
      std::vector<DAGEdge> edges = partition.getEdge(n, i);
      for (DAGEdge e : edges) {
        outEdges[e.src][i].insert(e.dependence);
      }
    }

    // Find all values from outside the loop-nest that are needed in this
    // partition. These become arguments to the node function
    std::set<Value*> outsideValues;
    for (Instruction* inst : node.insts) {
      for (Value* v : inst->operands()) {
        Argument* arg = dyn_cast<Argument>(v);
        Instruction* ins = dyn_cast<Instruction>(v);
        // Any value other than an arg or an instruction is either global or
        // constant (or a basic block, which is technically the later)
        if (arg || (ins && !loop->contains(ins))) {
          outsideValues.insert(v);
        }
      }
    }

    // Function's input struct
    std::vector<Type*> structFields; std::vector<Value*> structValues;

    // Add the instance number as the first argument
    structFields.push_back(Type::getInt32Ty(M.getContext()));
    // And an i8* (void*) for the synchronization arrays
    structFields.push_back(Type::getInt8Ty(M.getContext())->getPointerTo());

    for (Value* val : outsideValues) {
      structFields.push_back(val->getType());
      structValues.push_back(val);
    }

    stageInputs[n] = structValues;

    StructType* inputStructTy = StructType::create(M.getContext(),
         ArrayRef<Type*>(structFields),
         (loop->getName() + ".par.stage" + std::to_string(n) + ".args").str());
    PointerType* voidPtr = Type::getInt8Ty(M.getContext())->getPointerTo();
    // Takes and returns i8* to match POSIX pthread launch (which technically
    // takes void*, but LLVM interprets as i8*)
    FunctionType* funcType =
      FunctionType::get(voidPtr, ArrayRef<Type*>(voidPtr), false);
    Function* func =
      Function::Create(funcType,
        GlobalValue::LinkageTypes::ExternalLinkage, // unecessary?
        loop->getName() + ".par.stage" + std::to_string(n), M);
    nodeFuncs[n] = func;
    nodeInputStructs[n] = inputStructTy;
    psdswp.stageFuncs.insert(func);

    BasicBlock* entry = BasicBlock::Create(M.getContext(), "entry", func);
    BasicBlock* exit = BasicBlock::Create(M.getContext(), "exit", func);

    // Cast argument to struct type
    CastInst* inputPtr = BitCastInst::CreatePointerCast(func->getArg(0),
                            inputStructTy->getPointerTo(), "stage.arg",
                            entry);

    // Load the values out of the struct
    ValueToValueMapTy vmap;
    ValueMapper mapper(vmap);
    int i = 2;
    Type* tyI32 = Type::getInt32Ty(M.getContext());

    std::vector<Value*> gepIdx;
    gepIdx.push_back(ConstantInt::get(tyI32, 0));
    gepIdx.push_back(ConstantInt::get(tyI32, 0));

    for (Value* value : structValues) {
      gepIdx[1] = ConstantInt::get(tyI32, i);
      GetElementPtrInst* gep =
        GetElementPtrInst::Create(inputStructTy, inputPtr,
                                  ArrayRef<Value*>(gepIdx), "", entry);
      vmap[value] = new LoadInst(value->getType(), gep, value->getName(),
                                 entry);
      i++;
    }

    // And load the index (at index 0) and the synchronization arrays (index 1)
    gepIdx[1] = ConstantInt::get(tyI32, 0);
    LoadInst* instance =
      new LoadInst(tyI32,
        GetElementPtrInst::Create(inputStructTy, inputPtr,
                                  ArrayRef<Value*>(gepIdx), "", entry),
        "par.inst", entry);
    gepIdx[1] = ConstantInt::get(tyI32, 1);
    LoadInst* arrays =
      new LoadInst(Type::getInt8Ty(M.getContext())->getPointerTo(),
        GetElementPtrInst::Create(inputStructTy, inputPtr,
                                  ArrayRef<Value*>(gepIdx), "", entry),
        "par.arrays", entry);

    // The actual code-gen
    // Strategy: copy all instructions from the loop into this new function,
    // but removing or replacing the instructions not in this part of the
    // partition. We remove instructions have no instruction in this part has
    // a dependence on, and replace instructions that we do have dependences on
    // with consume() operations

    // Register the function exit block to be the exit block of the loop
    // and the entry block of the function to be the loop's entry
    assert(loop->getExitBlock() && "Loop doesn't have unique exit block");
    assert(loop->getLoopPredecessor() && "Loop doesn't have unique entry");
    vmap[loop->getExitBlock()] = exit;
    vmap[loop->getLoopPredecessor()] = entry;

    std::vector<Instruction*> toRemap;

    // Create a counter of the iterations (used in producing to parallel stages)
    // We will actually place this instruction (and add the increment handler)
    // after the rest of the code is in-place
    PHINode* iterCounter = PHINode::Create(Type::getInt32Ty(M.getContext()), 2,
                            "count.iters");
    BinaryOperator* iterInc = BinaryOperator::Create(Instruction::BinaryOps::Add,
                            iterCounter,
                            ConstantInt::get(Type::getInt32Ty(M.getContext()), 1),
                            "iter.inc");
    BinaryOperator* iterMod = BinaryOperator::Create(Instruction::BinaryOps::URem,
                            iterInc,
                            ConstantInt::get(Type::getInt32Ty(M.getContext()),
                              parStageRepl),
                            "iter.mod");

    // And then process all the instructions
    for (const BasicBlock* bb : loop->blocks()) {
      BasicBlock* newBB = BasicBlock::Create(M.getContext(), bb->getName(), func);
      vmap[bb] = newBB;
      IRBuilder builder(newBB);
      // Track the most recent phi's position so we can insert the next phi directly
      // after it (not after any produce calls it may have generated, since LLVM
      // doesn't allow that)
      PHINode* lastPHI = nullptr;
      for (const Instruction& inst : *bb) {
        if (stageInsts.find(&inst) != stageInsts.end()) {
          Instruction* copy = inst.clone();
          vmap[&inst] = copy;
          toRemap.push_back(copy);

          // Insert the node, just use builder for non-phi nodes and insert it
          // after the latest phi for phi's
          if (PHINode* phi = dyn_cast<PHINode>(copy)) {
            if (lastPHI) copy->insertAfter(lastPHI);
            else newBB->getInstList().push_front(copy);
            lastPHI = phi;
          } else {
            builder.Insert(copy);
          }

          auto f = outEdges.find(&inst);
          if (f != outEdges.end()) {
            std::map<int, std::set<DAGEdge::Type>> outs = f->second;
            for (auto const& [toStage, edgeTypes] : outs) {
              // For the replication to transmit to:
              // If communicating from a sequential stage to a parallel stage
              // use the "iteration counter"
              // If communicating between sequential stages just use 0
              // If communicating from a parallel stage to a sequential stage
              // use the instance number of this thread
              Value* toInst = (nodeRepls[toStage] == 1
                               ? (nodeRepls[n] == 1
                                  ? (Value*) builder.getInt32(0)
                                  : (Value*) instance)
                               : (Value*) iterCounter);
              std::vector<Value*> produceArgs
                = {arrays, nullptr, toInst, nullptr};
              if (edgeTypes.find(DAGEdge::Type::Register) != edgeTypes.end()) {
                // Just produce the value into the sync array
                produceArgs[1] = builder.getInt32(
                  syncArrays[std::make_tuple(&inst, toStage, DAGEdge::Type::Register)]);
                produceArgs[3] = extendAndCast(builder, M, copy);
                builder.CreateCall(psdswp.produce, ArrayRef<Value*>(produceArgs));
              }
              if (edgeTypes.find(DAGEdge::Type::Memory) != edgeTypes.end()) {
                // Just produce a value into the sync array
                produceArgs[1] = builder.getInt32(
                  syncArrays[std::make_tuple(&inst, toStage, DAGEdge::Type::Memory)]);
                produceArgs[3] = builder.getInt64(0);
                builder.CreateCall(psdswp.produce, ArrayRef<Value*>(produceArgs));
              }
              if (edgeTypes.find(DAGEdge::Type::Control) != edgeTypes.end()) {
                produceArgs[1] = builder.getInt32(
                  syncArrays[std::make_tuple(&inst, toStage, DAGEdge::Type::Control)]);
                // If this is the loop latch, produce for the next iteration
                if (&inst == loop->getLoopLatch()->getTerminator()
                    && nodeRepls[toStage] != 1)
                  produceArgs[2] = iterMod;
                // Produce the condition into the sync array
                Value* cond;
                if (BranchInst* branch = dyn_cast<BranchInst>(copy)) {
                  assert(branch->isConditional()
                    && "Control dependence from unconditional branch");
                  cond = branch->getCondition();
                } else if (SwitchInst* swtch = dyn_cast<SwitchInst>(copy)) {
                  cond = swtch->getCondition();
                } else {
                  assert(false
                    && "Control dependence from instruction not branch or switch");
                }
                Value* newCond = mapper.mapValue(*cond);
                // Insert code before the branch
                produceArgs[3] = CastInst::CreateZExtOrBitCast(
                    newCond, Type::getInt64Ty(M.getContext()), "", copy);
                CallInst::Create(psdswp.produce, ArrayRef<Value*>(produceArgs),
                                 "", copy);
              } else if (edgeTypes.find(DAGEdge::Type::PHI) != edgeTypes.end()) {
                const BranchInst* branch = dyn_cast<BranchInst>(&inst);
                if (branch && branch->isUnconditional()) {}
                else {
                  // Send condition
                  produceArgs[1] = builder.getInt32(
                    syncArrays[std::make_tuple(&inst, toStage, DAGEdge::Type::PHI)]);
                  Value* cond;
                  if (branch) { cond = branch->getCondition(); }
                  else if (const SwitchInst* swtch = dyn_cast<SwitchInst>(&inst)) {
                    cond = swtch->getCondition();
                  }
                  Value* newCond = mapper.mapValue(*cond);
                  produceArgs[3] = CastInst::CreateZExtOrBitCast(
                      newCond, Type::getInt64Ty(M.getContext()), "", copy);
                  CallInst::Create(psdswp.produce, ArrayRef<Value*>(produceArgs),
                                   "", copy);
                }
              }
              if (edgeTypes.find(DAGEdge::Type::Self) != edgeTypes.end()) {
                LLVM_DEBUG(dbgs() << "Ignoring self edge in code-gen");
              }
            }
          }
        } else {
          auto f = dependences.find(&inst);
          if (f == dependences.end()) {
            // Ignore the instruction unless it's a terminator
            if (inst.isTerminator()) {
              if (const BranchInst* branch = dyn_cast<BranchInst>(&inst)) {
                // For unconditional branches, just emit them and remap them
                if (branch->isUnconditional()) {
                  Instruction* copy = inst.clone();
                  builder.Insert(copy);
                  toRemap.push_back(copy);
                } else {
                  // For conditional branches pick one direction that does not
                  // traverse a back-edge
                  assert(branch->getNumSuccessors() == 2
                        && "Conditional branch with more than 2 successors");
                  BasicBlock* succ1 = branch->getSuccessor(0);
                  BasicBlock* succ2 = branch->getSuccessor(1);
                  if (succ1 != bb && !DT.dominates(succ1, bb)) {
                    toRemap.push_back(builder.CreateBr(succ1));
                  } else {
                    assert(succ2 != bb && !DT.dominates(succ2, bb)
                      && "Unhandled: branch containing two back-edges");
                    toRemap.push_back(builder.CreateBr(succ2));
                  }
                }
              } else if (const SwitchInst* swtch = dyn_cast<SwitchInst>(&inst)) {
                // Pick one branch that does not traverse a back-edge
                unsigned numSuccs = swtch->getNumSuccessors();
                bool found = false;
                for (unsigned i = 0; i < numSuccs; i++) {
                  BasicBlock* succ = swtch->getSuccessor(i);
                  if (succ != bb && !DT.dominates(succ, bb)) {
                    toRemap.push_back(builder.CreateBr(succ));
                    found = true;
                    break;
                  }
                }
                assert(found && "Encountered switch with only back edges");
              } else {
                assert(false && "Found loop terminator not a branch or switch");
              }
            }
          } else {
            std::vector<std::pair<int, DAGEdge>>& edges = f->second;
            bool hasRegDependence = false;
            bool hasMemDependence = false;
            bool hasControlDependence = false;
            bool hasPhiDependence = false;

            int fromReplReg = 0;
            int fromReplMem = 0;
            int fromReplControl = 0;
            int fromReplPhi = 0;

            for (std::pair<int, DAGEdge>& pair : edges) {
              DAGEdge& e = pair.second;
              switch (e.dependence) {
                case DAGEdge::Type::Register: {
                  assert((fromReplReg == 0
                          || fromReplReg == nodeRepls[pair.first])
                    && "Same dependence from stages of different replications");
                  hasRegDependence = true;
                  fromReplReg = nodeRepls[pair.first];
                  break; }
                case DAGEdge::Type::Memory: {
                  assert((fromReplMem == 0
                          || fromReplMem == nodeRepls[pair.first])
                    && "Same dependence from stages of different replications");
                  hasMemDependence = true;
                  fromReplMem = nodeRepls[pair.first];
                  break; }
                case DAGEdge::Type::Control: {
                  assert((fromReplControl == 0
                          || fromReplControl == nodeRepls[pair.first])
                    && "Same dependence from stages of different replications");
                  hasControlDependence = true;
                  fromReplControl = nodeRepls[pair.first];
                  break; }
                case DAGEdge::Type::PHI: {
                  assert((fromReplControl == 0
                          || fromReplControl == nodeRepls[pair.first])
                    && "Same dependence from stages of different replications");
                  hasPhiDependence = true;
                  fromReplPhi = nodeRepls[pair.first];
                  break; }
                case DAGEdge::Type::Self: {
                  LLVM_DEBUG(dbgs() << "Ignoring self edge");
                  break; }
              }
            }
            if(hasRegDependence) {
              assert((nodeRepls[n] == 1 || fromReplReg == 1)
                  && "Communication between parallel stages not supported");
              int syncArrayNum = numSyncArrays++;
              // If the dependence comes from a parallel stage, we have an
              // array for each instance of the parallel stage to enforce
              // the correct ordering
              syncArrayRepls.push_back(fromReplReg == 1 ? nodeRepls[n] : fromReplReg);
              syncArrays[std::make_tuple(&inst, n, DAGEdge::Type::Register)] = syncArrayNum;
              ConstantInt* syncArrayArg = builder.getInt32(syncArrayNum);
              std::vector<Value*> consumeArgs = {arrays, syncArrayArg,
                                                  fromReplReg == 1
                                                    ? (Value*) instance
                                                    : (Value*) iterCounter};
              CallInst* consumeInst
                = builder.CreateCall(psdswp.consume, ArrayRef<Value*>(consumeArgs));
              vmap[&inst] = truncateAndCast(builder, M, consumeInst,
                                            inst.getType(), inst.getName());
            }
            if (hasMemDependence) {
              assert((nodeRepls[n] == 1 || fromReplMem == 1)
                  && "Communication between parallel stages not supported");
              // For memory we consume just to signal that the memory
              // operation depended on has occured
              int syncArrayNum = numSyncArrays++;
              syncArrayRepls.push_back(fromReplMem == 1 ? nodeRepls[n] : fromReplMem);
              syncArrays[std::make_tuple(&inst, n, DAGEdge::Type::Memory)] = syncArrayNum;
              ConstantInt* syncArrayArg = builder.getInt32(syncArrayNum);
              std::vector<Value*> consumeArgs = {arrays, syncArrayArg,
                                                  fromReplMem == 1
                                                    ? (Value*) instance
                                                    : (Value*) iterCounter};
              CallInst* consumeInst
                = builder.CreateCall(psdswp.consume, ArrayRef<Value*>(consumeArgs));
            }
            if (hasControlDependence) {
              assert((nodeRepls[n] == 1 || fromReplControl == 1)
                  && "Communication between parallel stages not supported");
              // For control dependences, this instruction must be a
              // conditional branch and we consume the condition to use
              Value* cond;

              if (const BranchInst* branch = dyn_cast<BranchInst>(&inst)) {
                assert(branch->isConditional()
                  && "Control dependence is from an unconditional branch");
                cond = branch->getCondition();
              } else if (const SwitchInst* swtch = dyn_cast<SwitchInst>(&inst)) {
                cond = swtch->getCondition();
              } else {
                assert(false
                  && "Control dependence from non-branch or switch");
              }

              if (vmap.find(cond) == vmap.end()) {
                int syncArrayNum = numSyncArrays++;
                syncArrayRepls.push_back(fromReplControl == 1 ? nodeRepls[n] : fromReplControl);
                syncArrays[std::make_tuple(&inst, n, DAGEdge::Type::Control)] = syncArrayNum;
                ConstantInt* syncArrayArg = builder.getInt32(syncArrayNum);
                std::vector<Value*> consumeArgs = {arrays, syncArrayArg,
                                                    fromReplControl == 1
                                                      ? (Value*) instance
                                                      : (Value*) iterCounter};
                CallInst* consumeInst
                  = builder.CreateCall(psdswp.consume, ArrayRef<Value*>(consumeArgs));
                Value* newCond = truncateAndCast(builder, M, consumeInst,
                                              cond->getType(),
                                              "");
                vmap[cond] = newCond;
              }

              Instruction* copy = inst.clone();
              toRemap.push_back(copy);
              builder.Insert(copy);
            } else if (hasPhiDependence) {
              // If we insert the branch (because of a control dependence)
              // we don't need to handle the PHI specially anymore, since the
              // way we handle it is just to insert the branch like a control
              // dependence
              assert((nodeRepls[n] == 1 || fromReplPhi == 1)
                  && "Communication between parallel stages not supported");

              const BranchInst* branch = dyn_cast<BranchInst>(&inst);
              if (branch && branch->isUnconditional()) {
                Instruction* copy = inst.clone();
                toRemap.push_back(copy);
                builder.Insert(copy);
              } else {
                Value* cond;
                if (const BranchInst* branch = dyn_cast<BranchInst>(&inst)) {
                  cond = branch->getCondition();
                } else if (const SwitchInst* swtch = dyn_cast<SwitchInst>(&inst)) {
                  cond = swtch->getCondition();
                } else {
                  assert(false && "PHI dependence from non branch or switch");
                }
                if (vmap.find(cond) == vmap.end()) {
                  int syncArrayNum = numSyncArrays++;
                  syncArrayRepls.push_back(fromReplControl == 1
                                            ? nodeRepls[n]
                                            : fromReplPhi);
                  syncArrays[std::make_tuple(&inst, n, DAGEdge::Type::Control)]
                      = syncArrayNum;
                  ConstantInt* syncArrayArg = builder.getInt32(syncArrayNum);
                  std::vector<Value*> consumeArgs = {arrays, syncArrayArg,
                                                      fromReplControl == 1
                                                        ? (Value*) instance
                                                        : (Value*) iterCounter};
                  CallInst* consumeInst
                    = builder.CreateCall(psdswp.consume, ArrayRef<Value*>(consumeArgs));
                  Value* newCond = truncateAndCast(builder, M, consumeInst,
                                                Type::getInt1Ty(M.getContext()),
                                                "");
                  vmap[cond] = newCond;
                }

                Instruction* copy = inst.clone();
                toRemap.push_back(copy);
                builder.Insert(copy);
              }
            }
          }
        }
      }
    }

    // Place the iteration counter now, at the head of the loop header
    BasicBlock* newHeader = static_cast<BasicBlock*>(mapper.mapValue(*loop->getHeader()));
    BasicBlock* newLatch = static_cast<BasicBlock*>(mapper.mapValue(*loop->getLoopLatch()));
    iterCounter->insertBefore(newHeader->getFirstNonPHI());
    iterMod->insertBefore(newHeader->getFirstNonPHI());
    iterInc->insertBefore(iterMod);
    iterCounter->addIncoming(
        ConstantInt::get(Type::getInt32Ty(M.getContext()), 0),
        entry);
    iterCounter->addIncoming(iterMod, newLatch);

    // Add branch from entry to exit and loop header
    // To do this, consume a value from the synchronization array for the
    // loop's latch instruction (if the stage is parallel) and for sequential
    // stages the loop's guard (or lack of one) guarantees at least one
    // iteration will be executed
    Instruction* latch = loop->getLoopLatch()->getTerminator();
    if (nodeRepls[n] != 1) {
      int syncArrayNum = syncArrays[std::make_tuple(latch, n, DAGEdge::Type::Control)];
      IRBuilder builder(entry);
      ConstantInt* syncArrayArg = builder.getInt32(syncArrayNum);
      std::vector<Value*> consumeArgs = {arrays, syncArrayArg, instance};
      CallInst* consumeInst
          = builder.CreateCall(psdswp.consume, ArrayRef<Value*>(consumeArgs));
      Value* cond = truncateAndCast(builder, M, consumeInst,
                                    Type::getInt1Ty(M.getContext()), "");
      BranchInst* branch = dyn_cast<BranchInst>(latch->clone());
      assert(branch && "Loop latch terminator not a branch instruction");
      assert(branch->isConditional() && "Loop latch terminator is not conditional");
      const int numSuccessors = branch->getNumSuccessors();
      for (unsigned i = 0; i < numSuccessors; i++) {
        branch->setSuccessor(i,
          static_cast<BasicBlock*>(mapper.mapValue(*branch->getSuccessor(i))));
      }
      branch->setCondition(cond);
      builder.Insert(branch);
    } else {
      // For sequential stages just jump directly to the loop
      IRBuilder builder(entry);
      builder.CreateBr(static_cast<BasicBlock*>(mapper.mapValue(*loop->getHeader())));
    }

    // Add communication of live outs
    IRBuilder builder(exit);
    for (Instruction* inst : node.insts) {
      auto f = liveOuts.find(inst);
      if (f == liveOuts.end()) continue;
      assert(nodeRepls[n] == 1 && "Live-outs from DOALL stages not supported");
      std::vector<Value*> produceArgs =
        {arrays, builder.getInt32(f->second), builder.getInt32(0),
         extendAndCast(builder, M, mapper.mapValue(*inst))};
      builder.CreateCall(psdswp.produce, ArrayRef<Value*>(produceArgs));
    }

    // Add return NULL
    builder.CreateRet(ConstantPointerNull::get(voidPtr));

    // Remap all instructions that need it (we have to do this after all
    // instruction are in place to handle phi's)
    for (Instruction* inst : toRemap) mapper.remapInstruction(*inst);

    LLVM_DEBUG(
      dbgs() << "\n\n\n===============";
      dbgs() << *func;
      dbgs() << "===============\n\n\n");
  }

  // Code-gen to launch and finish the pipeline, we do this at the end of the
  // loop's pre-header, replacing the branch from that block into the
  // header with a branch into a new block for this launch/finish, and deleting
  // the loop body itself

  BranchInst* loopLatchInst = dyn_cast<BranchInst>(loop->getLoopLatch()->getTerminator());
  assert(loopLatchInst && "Loop latch terminator is not a branch");
  bool exitsOn0 = loopLatchInst->getSuccessor(1) != loop->getHeader();

  BasicBlock* newLoop = BasicBlock::Create(M.getContext(), "loop.par", &F);
  // Block for getting live-outs
  BasicBlock* liveOutConsumes = BasicBlock::Create(M.getContext(), "loop.outs", &F);
  IRBuilder builder(newLoop);
  assert(loop->getLoopPreheader() && "Loop doesn't have a pre-header");
  assert(loop->getExitBlock() && "Loop doesn't have unique exit block");
  BranchInst* branch = dyn_cast<BranchInst>(loop->getLoopPreheader()->getTerminator());
  assert(branch && branch->isUnconditional()
         && "Loop pre-header is not unconditional branch");
  branch->setSuccessor(0, newLoop);
  // Add branch to the exit block, and set the builder to insert before it
  builder.SetInsertPoint(BranchInst::Create(liveOutConsumes, newLoop));

  IRBuilder outsBuilder(liveOutConsumes);
  outsBuilder.SetInsertPoint(BranchInst::Create(loop->getExitBlock(), liveOutConsumes));

  // To launch the pipeline:
  // (1) Create the synchronization arrays
  std::vector<Value*> createArgs;
  createArgs.push_back(builder.getInt32(numSyncArrays));
  for (int repl : syncArrayRepls) {
    if (repl <= 0) {
      errs() << "syncArray repl error\n";
      abort();
    }
    createArgs.push_back(builder.getInt32(repl));
  }
  CallInst* syncArray =
    builder.CreateCall(psdswp.createSyncArrays, ArrayRef<Value*>(createArgs),
                       "sync.array");

  // Now, replace uses of liveOuts with consumes
  {
    for (auto [inst, syncArrayNum] : liveOuts) {
      std::vector<Value*> consumeArgs = {syncArray,
                                         builder.getInt32(syncArrayNum),
                                         builder.getInt32(0)};
      CallInst* consumeInst = outsBuilder.CreateCall(psdswp.consume,
                                ArrayRef<Value*>(consumeArgs));
      inst->replaceAllUsesWith(truncateAndCast(outsBuilder, M, consumeInst,
                                               inst->getType(), inst->getName()));
    }
  }

  // Now we delete the loop (first go through and drop references to avoid
  // dependence issues as we go through and delete)
  for (BasicBlock* bb : loop->blocks()) {
    bb->dropAllReferences();
  }
  for (BasicBlock* bb : loop->blocks()) {
    bb->eraseFromParent();
  }

  // (2) Signaling the first iteration of any stage which needs it (these are
  // parallel stages, and we only signal the first instance)
  // This must be done before we create any of the stages since it otherwise
  // produces a race condition with the main stage finishing the loop before
  // this code is executed
  {
    std::vector<Value*> produceArgs = {syncArray,
                                       builder.getInt32(0), // placeholder
                                       builder.getInt32(0),
                                       builder.getInt64(exitsOn0 ? 1 : 0)};
    for (int n = 0; n < numNodes; n++) {
      if (nodeRepls[n] != 1) {
        auto f = syncArrays.find(std::make_tuple(loopLatchInst, n, DAGEdge::Type::Control));
        assert(f != syncArrays.end() && "Parallel stage without dependence on latch");
        int syncArray = f->second;
        produceArgs[1] = builder.getInt32(syncArray);
        builder.CreateCall(psdswp.produce, ArrayRef<Value*>(produceArgs));
      }
    }
  }

  // (3) Create structs for each stage instance and launch the stage instance
  std::vector<std::vector<CallInst*>> threadIds;
  for (int n = 0; n < numNodes; n++) {
    threadIds.push_back(std::vector<CallInst*>{});

    const int instances = nodeRepls[n];
    Function* nodeFunc = nodeFuncs[n];
    StructType* nodeInput = nodeInputStructs[n];
    for (int inst = 0; inst < instances; inst++) {
      AllocaInst* input = builder.CreateAlloca(nodeInput);

      // Add instance number
      std::vector<Value*> gepIndices = {builder.getInt32(0), builder.getInt32(0)};
      builder.CreateStore(
        builder.getInt32(inst),
        builder.CreateGEP(nodeInput, input, ArrayRef<Value*>(gepIndices)));

      // Add synchronization arrays
      gepIndices[1] = builder.getInt32(1);
      builder.CreateStore(syncArray,
        builder.CreateGEP(nodeInput, input, ArrayRef<Value*>(gepIndices)));

      // Add other arguments
      int i = 2;
      for (Value* arg : stageInputs[n]) {
        gepIndices[1] = builder.getInt32(i);
        builder.CreateStore(arg,
          builder.CreateGEP(nodeInput, input, ArrayRef<Value*>(gepIndices)));
        i++;
      }

      // Now, we launch the stage instance
      std::vector<Value*> launchArgs = {
        builder.CreateBitCast(input, Type::getInt8PtrTy(M.getContext())),
        nodeFunc
      };
      threadIds[n].push_back(
        builder.CreateCall(psdswp.launchStage, ArrayRef<Value*>(launchArgs)));
    }
  }

  // Now, we wait for it to complete by
  // (4) Wait for the source node(s) to complete
  for (int n = 0; n < numNodes; n++) {
    if (incomingEdges[n].empty()) { // Source node
      const int instances = nodeRepls[n];
      for (int inst = 0; inst < instances; inst++) {
        builder.CreateCall(psdswp.waitForStage,
                           ArrayRef<Value*>(threadIds[n][inst]));
      }
    }
  }

  // (5) Send signals for every instance of every stage on the loop condition
  // to exit the loop
  std::vector<Value*> produceArgs = {syncArray,
                                     builder.getInt32(0), // placeholder
                                     builder.getInt32(0), // placeholder
                                     builder.getInt64(exitsOn0 ? 0 : 1)};
  for (int n = 0; n < numNodes; n++) {
    auto f = syncArrays.find(std::make_tuple(loopLatchInst, n, DAGEdge::Type::Control));
    if (f != syncArrays.end()) {
      int syncArray = f->second;
      int repl = nodeRepls[n];
      for (int inst = 0; inst < repl; inst++) {
        produceArgs[1] = builder.getInt32(syncArray);
        produceArgs[2] = builder.getInt32(inst);
        builder.CreateCall(psdswp.produce, ArrayRef<Value*>(produceArgs));
      }
    }
  }

  // (6) Wait for all other instances to finish
  for (int n = 0; n < numNodes; n++) {
    if (!incomingEdges[n].empty()) { // Non-source node
      const int instances = nodeRepls[n];
      for (int inst = 0; inst < instances; inst++) {
        builder.CreateCall(psdswp.waitForStage,
                           ArrayRef<Value*>(threadIds[n][inst]));
      }
    }
  }

  // (7) Free the synchronization arrays (after consuming live-outs)
  std::vector<Value*> freeArgs;
  freeArgs.push_back(syncArray);
  freeArgs.push_back(outsBuilder.getInt32(numSyncArrays));
  for (int repl : syncArrayRepls) {
    freeArgs.push_back(outsBuilder.getInt32(repl));
  }
  outsBuilder.CreateCall(psdswp.freeSyncArrays, ArrayRef<Value*>(freeArgs));

  return true;
}

char PS_DSWP::ID = 0;

static RegisterPass<PS_DSWP> X("psdswp",
                "Parallel Stage, Decoupled Software Pipelining",
                true /* Can modify the CFG */,
                true /* Transformation Pass */);
