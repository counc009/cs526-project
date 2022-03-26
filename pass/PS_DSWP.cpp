#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Transforms/Utils.h"

#include <map>
#include <vector>

#define DEBUG_TYPE "psdswp"

using namespace llvm;

namespace {
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

    void addEdge(int src, int dst, TEdge edge) {
      std::map<int, std::vector<TEdge>>& nodeEdges = edges[src];

      auto f = nodeEdges.find(dst);
      if (f != nodeEdges.end()) {
        nodeEdges[dst].push_back(edge);
      } else {
        nodeEdges[dst] = std::vector<TEdge>({edge});
      }
    }
    
    TEdge* getEdge(int src, int dst) {
      std::map<int, TEdge>& nodeEdges = edges[src];
      auto f = nodeEdges.find(dst);
      if (f != nodeEdges.end()) return &(*f);
      else return nullptr;
    }
  };
}

using PDG = DiGraph<PDGNode, PDGEdge>;
using DAG = DiGraph<DAGNode, DAGEdge>;

static PDG generatePDG(Loop*, LoopInfo&, DependenceInfo&);
static DAG computeDAGscc(PDG);

bool PS_DSWP::runOnFunction(Function &F) {
  LLVM_DEBUG(errs() << "Running PS-DSWP on function " << F.getName() << "!\n");
  
  bool modified = false; // Tracks whether we modified the function

  DominatorTree DT(F);
  LoopInfo LI(DT);
  DependenceInfo& DI = getAnalysis<DependenceAnalysisWrapperPass>().getDI();

  // For now, just considering the top level loops. Not actually sure if this
  // is correct behavior in general
  for (Loop* loop : LI.getTopLevelLoops()) {
    LLVM_DEBUG(errs() << "\tRunning on loop " << *loop << "\n");
    PDG pdg = generatePDG(loop, LI, DI);
    DAG dag_scc = computeDAGscc(pdg);
    // TODO
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

static PDG generatePDG(Loop* loop, LoopInfo& LI, DependenceInfo& DI) {
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
        memInsts.insert(&inst);
        for (Instruction* other : memInsts) {
          // TODO: Is there a data dependence between these instructions?
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
              errs() << "Dependence from " << *opInst << " to " << inst 
                     << (carried ? " (loop carried)\n" : "\n")); 
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
              errs() << "Dependence from " << inst << " to " << *useInst
                     << (carried ? " (loop carried)\n" : "\n")); 
          }
        }
      }
    }
  }

  // Handle control flow dependencies here (TODO)

  return DiGraph<PDGNode, PDGEdge>();
}

static DAG computeDAGscc(PDG pdg) {
  // TODO
  return DiGraph<DAGNode, DAGEdge>();
}

char PS_DSWP::ID = 0;

static RegisterPass<PS_DSWP> X("psdswp",
                "Parallel Stage, Decoupled Software Pipelining",
                true /* Can modify the CFG */,
                true /* Transformation Pass */);
