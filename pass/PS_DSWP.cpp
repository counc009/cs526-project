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
}

using PDG = DiGraph<PDGNode, PDGEdge>;
using DAG = DiGraph<DAGNode, DAGEdge>;

static PDG generatePDG(Loop*, LoopInfo&, DependenceInfo&, DominatorTree&,
                       ReverseDominanceFrontier&, std::set<Instruction*>&);
static DAG computeDAGscc(PDG);
static void strongconnect(int, int*, int* ,std::stack<int>&,bool*, PDG, DAG&, std::map<int, std::vector<int>>&, std::map<int, int>&);
static DAG connectEdges(PDG, DAG,  std::map<int, std::vector<int>>&,  std::map<int, int>&);
static DAG threadPartitioning(DAG dag);
static std::set<Instruction*> analyzeDataStructures(Loop* loop);

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
    std::set<Instruction*> nonLoopCarried = analyzeDataStructures(loop);

    LLVM_DEBUG(dbgs() << "[psdswp] Running on loop " << *loop << "\n");
    PDG pdg = generatePDG(loop, LI, DI, DT, RDF, nonLoopCarried);

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

static void checkMemoryDependence(Instruction& src, Instruction& dst,
                                  int srcNode, int dstNode,
                                  PDG& graph, BasicBlock* backedge,
                                  DominatorTree& DT, DependenceInfo& DI,
                                  std::set<Instruction*>& nonLoopCarried) {
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
    bool loopCarried = !dependence->isLoopIndependent();
    if (nonLoopCarried.find(&src) != nonLoopCarried.end()
        || nonLoopCarried.find(&dst) != nonLoopCarried.end()) {
      loopCarried = false;
    }
    if (direction != Dependence::DVEntry::LT
        && (loopCarried || (&src != &dst && !DT.dominates(&dst, &src)))) {
      // Only include dependence that aren't negative or if the dependence
      // isn't loop carried, only include ones where the dst doesn't dominate
      // the src (i.e. exclude backwards dependences/non-carried self
      // dependences)
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
    }
  }
}

static PDG generatePDG(Loop* loop, LoopInfo& LI, DependenceInfo& DI,
                       DominatorTree& DT, ReverseDominanceFrontier& RDF,
                       std::set<Instruction*>& nonLoopCarried) {
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
                                  backedge, DT, DI, nonLoopCarried);
            if (thisNode != otherNode) {
              checkMemoryDependence(*other, inst, otherNode, thisNode, graph,
                                    backedge, DT, DI, nonLoopCarried);
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

static bool existsLongPath(DAG& graph, int src, int dst) {
  std::set<int> considered;
  considered.insert(src);
  
  std::queue<int> worklist;
  for (int n : graph.getAdjs(src)) {
    // Don't add the destination node in the first round since we only want
    // to consider paths with at least one intermediate node
    if (n != dst)
      worklist.push(n);
  }

  while (!worklist.empty()) {
    int idx = worklist.front(); worklist.pop();
    considered.insert(idx);
    if (idx == dst) return true;

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
    for (int n1 : blockToNodes[block1]) {
      for (int n2 : blockToNodes[block2]) {
        if (existsLongPath(dag, n1, n2) || existsLongPath(dag, n2, n1))
          return false;
      }
    }
    return true;
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
  
  /*
  //Naive approach - most number of nodes (Actually decide this based on max profile weight)
  int max = -1;
  int nodeIndex = -1;
  for(size_t i = 0;i<blockToNodes.size(); i++)
  	if(blockToNodes[i].size() > max){
  		max = blockToNodes[i].size();
  		nodeIndex = i;
  		}
  		
  for(size_t i = 0;i<blockToNodes.size(); i++)
  	if(i!=nodeIndex)
  		doall_blocks.erase(i);
  		sequential_blocks.insert(i);
  */
  
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
  
  for(size_t i = 0;i<blockToNodes.size(); i++)
  {
    DAGNode combined;
    std::vector<Instruction*> insts;
    for(size_t j = 0;j<blockToNodes[i].size();j++)
    {
 			insts.insert(insts.end(),  dag.getNode(j).insts.begin(),  dag.getNode(j).insts.end());
  	}	
    combined.insts = insts;
    if(sequential_blocks.find(i) != sequential_blocks.end()) combined.doall = false;
    else combined.doall = true;
    dag_threaded.insertNode(combined);

  }
  
  //Connect all edges, possibly repeats edges 
  for(size_t i = 0;i<nodeToBlock.size(); i++)
		for(size_t j = 0; j < nodeToBlock.size(); j++){
			if(nodeToBlock[i] == nodeToBlock[j])
				continue;
			else{
				std::vector<DAGEdge> edges = dag.getEdge(i , j);
				for (size_t k = 0; k < edges.size(); k++)
					dag_threaded.addEdge( nodeToBlock[i] , nodeToBlock[j] , edges[k] );
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

static std::set<Instruction*> analyzeDataStructures(Loop* loop) {
  std::set<Instruction*> cannotBeLoopCarried;
  std::set<Value*> notCarriedPointers;
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
        LLVM_DEBUG(dbgs() << "Found acyclic linked list phi: " << phi << "\n");
        std::list<Instruction*> worklist;
        worklist.push_back(&phi);

        while (!worklist.empty()) {
          Instruction* current = worklist.front();
          worklist.pop_front();

          if (notCarriedPointers.find(current) != notCarriedPointers.end())
            continue;

          if (!current->mayReadOrWriteMemory()) {
            notCarriedPointers.insert(current);
            for (Value* user : current->users()) {
              Instruction* inst = dyn_cast<Instruction>(user);
              if (!inst) continue;
              worklist.push_back(inst);
            }
          } else if (StoreInst* store = dyn_cast<StoreInst>(current)) {
            if (notCarriedPointers.find(store->getPointerOperand())
                != notCarriedPointers.end()) {
              cannotBeLoopCarried.insert(store);
              LLVM_DEBUG(dbgs() << "Found store that cannot have loop-carried "
                                   "dependence: " << *store << "\n");
            }
          } else if (LoadInst* load = dyn_cast<LoadInst>(current)) {
            if (notCarriedPointers.find(load->getPointerOperand())
                != notCarriedPointers.end()) {
              cannotBeLoopCarried.insert(load);
              LLVM_DEBUG(dbgs() << "Found load that cannot have loop-carried "
                                   "dependence: " << *load << "\n");
            }
          }
        }
      }
    }
  }
  return cannotBeLoopCarried;
}

char PS_DSWP::ID = 0;

static RegisterPass<PS_DSWP> X("psdswp",
                "Parallel Stage, Decoupled Software Pipelining",
                true /* Can modify the CFG */,
                true /* Transformation Pass */);
