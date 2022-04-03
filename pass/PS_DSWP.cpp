#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
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
      	return &(*f);
      else 
      	return nullptr;
    }
    
    TEdge* getEdge(int src, int dst) {
      std::map<int, TEdge>& nodeEdges = edges[src];
      auto f = nodeEdges.find(dst);
      if (f != nodeEdges.end()) return &(*f);
      else return nullptr;
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
}

using PDG = DiGraph<PDGNode, PDGEdge>;
using DAG = DiGraph<DAGNode, DAGEdge>;

static PDG generatePDG(Loop*, LoopInfo&, DependenceInfo&, DominatorTree&,
                       ReverseDominanceFrontier&);
static DAG computeDAGscc(PDG);
static void strongconnect(int, int*, int* ,std::stack<int>&,bool*, PDG, DAG&, std::map<int, std::vector<int>>&);

bool PS_DSWP::runOnFunction(Function &F) {
  LLVM_DEBUG(dbgs() << "[psdswp] Running on function " << F.getName() << "!\n");
  
  bool modified = false; // Tracks whether we modified the function

  DominatorTree DT(F);
  LoopInfo LI(DT);
  DependenceInfo& DI = getAnalysis<DependenceAnalysisWrapperPass>().getDI();

  ReverseDominanceFrontier RDF(F);

  // For now, just considering the top level loops. Not actually sure if this
  // is correct behavior in general
  for (Loop* loop : LI.getTopLevelLoops()) {
    LLVM_DEBUG(dbgs() << "[psdswp] Running on loop " << *loop << "\n");
    PDG pdg = generatePDG(loop, LI, DI, DT, RDF);
    LLVM_DEBUG(dbgs() << "Performing SCCC Tests " << "\n");
    DAG dag_scc = computeDAGscc(pdg);
    // TODO: partition, code-gen, etc.
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

static void checkMemoryDependence(Instruction& src, Instruction& dst,
                                  int srcNode, int dstNode,
                                  PDG& graph, BasicBlock* backedge,
                                  DominatorTree& DT, DependenceInfo& DI) {
  // First, see if we can reach dst from src without traversing the back edge
  // LLVM supports checking reachability without traversing a set of basic
  // blocks, so we check reachability without entering the basic block that
  // contains the back edge and then handle either of the instructions being
  // in that basic block
  bool maybeLoopIndependent = false;
  if (src.getParent() == backedge && dst.getParent() == backedge) {
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
             << (!dependence->isLoopIndependent() ? " (loop carried)" : "")
             << " direction " << dirStr << "\n");
    if (direction != Dependence::DVEntry::LT) {
      // Only include dependence that aren't negative
      graph.addEdge(srcNode, dstNode, PDGEdge{PDGEdge::Memory,
                                        !dependence->isLoopIndependent()});
    }
  }
}

static PDG generatePDG(Loop* loop, LoopInfo& LI, DependenceInfo& DI,
                       DominatorTree& DT, ReverseDominanceFrontier& RDF) {
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
                                  backedge, DT, DI);
            if (thisNode != otherNode) {
              checkMemoryDependence(*other, inst, otherNode, thisNode, graph,
                                    backedge, DT, DI);
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

static void strongconnect(int u, int disc[], int low[], std::stack<int> *st,
					bool stackMember[], PDG graph, DAG &dag_scc, std::map<int, std::vector<int>> &pdg_to_scc)
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
			strongconnect(v, disc, low, st, stackMember, graph, dag_scc, pdg_to_scc);

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
		st->pop();
		DAGNode combined;
		combined.insts = insts;
		pdg_to_scc.insert({len, node_maps});
		/*
		for(size_t i1 = 0; i1< node_maps.size();i1++)
			for(size_t i2 = 0; i2< node_maps.size();i2++)
				{
				PDGEdge* edge = graph.getEdge(i1,i2);
				if(edge==nullptr)
					continue;
				else
					if(edge->loopCarried==true)
						loopcarried = true;	
				}
		if(loopcarried == false)
			combined.doall = true;
		else
			combined.doall = false;
			*/
		int n = dag_scc.insertNode(combined);
	}
}

static DAG computeDAGscc(PDG graph) {
  //Unit testing new functions
  LLVM_DEBUG(dbgs() << "Number of nodes in the graph " << graph.getNodeCount() << "\n");
  for(int i=0;i<graph.getNodeCount();i++){
    std::vector<int> adjacents = graph.getAdjs(i);
    LLVM_DEBUG(dbgs() << "Adjacent nodes of " << i << ":");
  for(size_t j=0;j<adjacents.size();j++)
    LLVM_DEBUG(dbgs() <<adjacents[j] << " ");
   LLVM_DEBUG(dbgs() <<"\n");
  }
  
  	DAG dag_scc;
  	std::map<int, std::vector<int>> pdg_to_scc;
  
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
	for (int i = 0; i < V; i++)
		if (disc[i] == -1)
			strongconnect(i, disc, low, st, stackMember, graph, dag_scc, pdg_to_scc);
	
  LLVM_DEBUG(dbgs() << "Number of nodes in the graph " << dag_scc.getNodeCount() << "\n");
  for(int i=0;i<dag_scc.getNodeCount();i++){
    std::vector<int> adjacents = dag_scc.getAdjs(i);
    LLVM_DEBUG(dbgs() << "Adjacent nodes of " << i << ":");
  for(size_t j=0;j<adjacents.size();j++)
    LLVM_DEBUG(dbgs() <<adjacents[j] << " ");
   LLVM_DEBUG(dbgs() <<"\n");
  }
  
  for (auto const& i : pdg_to_scc){
	    LLVM_DEBUG(dbgs() << i.first << ':');
	    for(size_t x=0;x<i.second.size();x++)
	    	LLVM_DEBUG(dbgs() << i.second[x] << " ");
	    LLVM_DEBUG(dbgs() <<"\n");
	    	
	}

  return DiGraph<DAGNode, DAGEdge>();
}


char PS_DSWP::ID = 0;

static RegisterPass<PS_DSWP> X("psdswp",
                "Parallel Stage, Decoupled Software Pipelining",
                true /* Can modify the CFG */,
                true /* Transformation Pass */);
