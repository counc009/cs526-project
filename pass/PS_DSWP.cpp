#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils.h"

#include <map>
#include <queue>
#include <vector>

#define DEBUG_TYPE "psdswp"

using namespace llvm;

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
    PS_DSWP() : FunctionPass(ID) {}

    // Entry point for the overall scalar-replacement pass
    bool runOnFunction(Function &F);

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
    enum Type { Register, Memory, Control };
    Type dependence;
    bool loopCarried;
    PDGEdge(Type dep, bool carried) : dependence(dep), loopCarried(carried) {}
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
    std::map<BasicBlock*, std::set<const BasicBlock*>> RDF;
  public:
    std::set<const BasicBlock*> rdf(BasicBlock* bb) { return RDF[bb]; }

    ReverseDominanceFrontier(Function& F) {
      PostDominatorTree PDT(F); // Note: PDT.dominates(A, B) <=> A postdom B
      DomTreeNodeBase<BasicBlock>* Node = PDT[PDT.getRoot()];
      BasicBlock* BB = Node->getBlock();

      std::vector<WorkObject> worklist;
      std::set<BasicBlock*> visited;
      
      worklist.push_back(WorkObject(BB, nullptr, Node, nullptr));
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
          if (visited.count(childBB) == 0) {
            worklist.push_back(WorkObject(childBB, currentBB, IDominee, currentNode));
            visitChild = true;
          }
        }

        // If all children are visited or there is any child then pop this block
        // from the worklist.
        if (!visitChild) {
          if (!parentBB) break;

          auto CDFI = S.begin(), CDFE = S.end();
          auto& parentSet = RDF[parentBB];
          for (; CDFI != CDFE; ++CDFI) {
            if (!PDT.properlyDominates(parentNode, PDT[*CDFI]))
              parentSet.insert(*CDFI);
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

bool PS_DSWP::runOnFunction(Function &F) {
  LLVM_DEBUG(dbgs() << "[psdswp] Running on function " << F.getName() << "!\n");
  if (numThreads < 2) {
    errs() << "PS-DSWP pass skipped since it was does not run with less than "
              "two threads\n";
    return false;
  }

  bool modified = false; // Tracks whether we modified the function

  DominatorTree DT(F);
  LoopInfo LI(DT);
  DependenceInfo& DI = getAnalysis<DependenceAnalysisWrapperPass>().getDI();

  ReverseDominanceFrontier RDF(F);

  // For now, just considering the top level loops. Not actually sure if this
  // is correct behavior in general
  for (Loop* loop : LI.getTopLevelLoops()) {
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
    
    // TODO: code-gen, etc.
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
    bool loopCarried = !dependence->isLoopIndependent() || !maybeLoopIndependent;

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
        std::set<Value*>& forwardDst = DSA.forwardPointers[derivedDst];
        if (forwardDst.find(derivedSrc) != forwardDst.end()) {
          // If dst is referencing a prior element in the list, set direction
          // to LT
          direction = Dependence::DVEntry::LT;
        }
      }
    }

    if (direction != Dependence::DVEntry::LT
        && (loopCarried || (&src != &dst && !DT.dominates(&dst, &src)))) {
      // Only include dependence that aren't negative or if the dependence
      // isn't loop carried, only include ones where the dst doesn't dominate
      // the src (i.e. exclude backwards dependences/non-carried self
      // dependences)
      // Basically all of these cases are ones where either the dependence
      // doesn't matter to our analysis or the other direction will get
      // put in and adding both may unecessarily create a cycle that forces
      // sequentialization
      graph.addEdge(srcNode, dstNode, PDGEdge{PDGEdge::Memory, loopCarried});
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
             << (direction == Dependence::DVEntry::LT ? " was LT" : " wasn't LT")
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
            graph.addEdge(f->second, node, PDGEdge(PDGEdge::Register, carried));
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
            graph.addEdge(node, f->second, PDGEdge(PDGEdge::Register, carried));
            LLVM_DEBUG(
              dbgs() << "[psdswp] Register dependence from " << inst << " to "
                     << *useInst << (carried ? " (loop carried)\n" : "\n")); 
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
              graph.addEdge(f->second, node, PDGEdge(PDGEdge::Register, carried));
              LLVM_DEBUG(
                dbgs() << "[psdswp] PHI control dependence from " << *term
                       << " to " << *phi << (carried?" (loop carried)\n":"\n"));
            }
          }
        }
      } else if (BranchInst* branch = dyn_cast<BranchInst>(&inst)) {
        for (BasicBlock* bb : branch->successors()) {
          if (loop->contains(bb)) {
            for (PHINode& phi : bb->phis()) {
              auto f = nodes.find(&phi);
              if (f != nodes.end()) {
                bool carried = branch->getParent() == backedge;
                graph.addEdge(node, f->second, PDGEdge(PDGEdge::Register, carried));
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
    for (const BasicBlock* b : RDF.rdf(bb)) {
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
                        PDGEdge(PDGEdge::Control, loopCarried));
          LLVM_DEBUG(
            dbgs() << "[psdswp] Control dependence from " << *terminator
                   << " to " << inst
                   << (loopCarried ? " (loop carried)\n" : "\n"));
        }
      }
    }
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
					bool stackMember[], PDG graph, DAG &dag_scc, std::map<int, std::vector<int>> &scc_to_pdg_map, std::map<int, int> &pdg_to_scc_map)
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

  errs() << "In threadPartitioning()\n";

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
    errs() << "[psdswp] Entered DOALL do-while in threadPartitioning()\n";
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
        errs() << "[psdswp] Merged blocks " << firstBlock << " and " << secondBlock << "\n";
      }
      ++it;
    }
  } while (merged);

  errs() << "[psdswp] After merging DOALL:\n";
  for (auto it : blockToNodes) {
    errs() << "\tBlock " << it.first << " : ";
    for (int i : it.second) {
      errs() << i << " ";
    }
    errs() << "\n";
  }
  
  
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
		errs() << "[psdswp] Max Profile block " << nodeIndex << " number of instructions  " << max << "\n";
		for (auto it : doall_blocks) {
			if(it!=nodeIndex)
				{
				errs() << "[psdswp] Converting block from doall to sequential " << it << "\n";
				doall_blocks.erase(it);
				sequential_blocks.insert(it);
				}
  	}
  }

  
  // Merge SEQUENTIAL nodes
  bool mergedSeq = false;
  do {
    errs() << "[psdswp] Entered SEQUENTIAL do-while in  threadPartitioning()\n";
    mergedSeq = false;

    auto it = sequential_blocks.begin();
    auto end = sequential_blocks.end();
    while (it != end) {
      int firstBlock = *it;
      
      auto innerIt = it;
      ++innerIt;
      while (innerIt != end) {
        int secondBlock = *innerIt;
        errs() << "[psdswp] TRYING TO MERGE " << firstBlock << " and "
               << secondBlock << "(" << existsLongPathBlocks(firstBlock, secondBlock) << ")\n";
        
        // Check Condition 1 for valid assignment from the paper (that there does not exist a path
        // containing one or more intermediate nodes between the two candidates
        if (existsLongPathBlocks(firstBlock, secondBlock))
          { ++innerIt; continue; }
        
        // Merge the blocks
        mergeBlocks(firstBlock, secondBlock);
        innerIt = sequential_blocks.erase(innerIt);
        mergedSeq = true;
        errs() << "[psdswp] Merged blocks " << firstBlock << " and " << secondBlock << "\n";
      }
      ++it;
    }
  } while (mergedSeq);
  
  errs() << "[psdswp] After merging SEQUENTIAL:\n";
  for (auto it : blockToNodes) {
    errs() << "\tBlock " << it.first << " : ";
    for (int i : it.second) {
      errs() << i << " ";
    }
    errs() << "\n";
  }
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
		errs() << "[psdswp] node to blocks :" << it1.first << "->"<< it1.second <<  "\n";
	


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

char PS_DSWP::ID = 0;

static RegisterPass<PS_DSWP> X("psdswp",
                "Parallel Stage, Decoupled Software Pipelining",
                true /* Can modify the CFG */,
                true /* Transformation Pass */);
