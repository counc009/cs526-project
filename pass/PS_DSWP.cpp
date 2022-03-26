#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/DDG.h"
#include "llvm/Analysis/DDGPrinter.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Transforms/Utils.h"

#include <vector>

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
  };
  struct PDGEdge {
    enum Type { Register, Memory, Control };
    bool loopCarried;
  };

  struct DAGNode {
    std::vector<Instruction*> insts;
    bool doall;
  };
  using DAGEdge = PDGEdge;
}

using PDG = DirectedGraph<PDGNode, PDGEdge>;
using DAG = DirectedGraph<DAGNode, DAGEdge>;

static PDG generatePDG(Loop*);
static DAG computeDAGscc(PDG);

bool PS_DSWP::runOnFunction(Function &F) {
  errs() << "Running PS-DSWP on function " << F.getName() << "!\n";
  
  bool modified = false; // Tracks whether we modified the function

  DominatorTree DT(F);
  LoopInfo LI(DT);
  DependenceInfo& DI = getAnalysis<DependenceAnalysisWrapperPass>().getDI();

  // For now, just considering the top level loops. Not actually sure if this
  // is correct behavior in general
  for (Loop* loop : LI.getTopLevelLoops()) {
    errs() << "\tRunning on loop " << *loop << "\n";
    PDG pdg = generatePDG(loop);
    DAG dag_scc = computeDAGscc(pdg);
    // TODO
  }

  return modified;
}

static PDG generatePDG(Loop* loop) {
  // TODO
  return DirectedGraph<PDGNode, PDGEdge>();
}

static DAG computeDAGscc(PDG pdg) {
  // TODO
  return DirectedGraph<DAGNode, DAGEdge>();
}

char PS_DSWP::ID = 0;

static RegisterPass<PS_DSWP> X("psdswp",
                "Parallel Stage, Decoupled Software Pipelining",
                true /* Can modify the CFG */,
                true /* Transformation Pass */);
