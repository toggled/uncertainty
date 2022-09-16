#include "MonteCarloBFS.h"



MonteCarloBFS::MonteCarloBFS()
{
	Graph();
}

MonteCarloBFS::MonteCarloBFS(DirectedGraph& graph)
{
	Graph();
	reference_graph = graph;
	// Get edge probability map
	weights = boost::get(boost::edge_weight_t(), reference_graph);
}
// Returns 1 if target is reachable from source
int MonteCarloBFS::run(VertexDescr source, VertexDescr target)
{
	std::queue<VertexDescr> worklist;
	std::set<VertexDescr> explored;
	OutEdgeIter ei, ei_end;
	VertexDescr v, w;

	// Add source in worklist
	worklist.push(source);
	explored.insert(source);

	if (source == target) {
		return 1;
	}

	while (!worklist.empty())
	{
		v = worklist.front();
		worklist.pop();
		// Iterate through all outgoing edges
		if (boost::out_degree(v, reference_graph) != 0) {
			for (boost::tie(ei, ei_end) = boost::out_edges(v, reference_graph); ei != ei_end; ++ei) {
				w = boost::target(*ei, reference_graph);
				// Sample to check if edge (v,w) exists
				if (checkExist(get(weights, edge(v, w, reference_graph).first))) {
					// Edge exists
					//add_edge(v, w, 1, directed_graph);	// Add to graph
					// Check if w is target
					if (w == target) {
						// Reached target
						return 1;
					}
					// if not explored, add to worklist
					if (explored.count(w) == 0) {
						worklist.push(w);
						explored.insert(w);
					}
				}
				else {
					// Edge does not exist
					// Do not add any edge
				}
			}
		}
	}
	// Target not found
	return 0;
}


int MonteCarloBFS::run_forRSS(VertexDescr source, VertexDescr target, std::set<Edge_s_t> e1, std::set<Edge_s_t> e2)
{
	std::queue<VertexDescr> worklist;
	std::set<VertexDescr> explored;
	OutEdgeIter ei, ei_end;
	VertexDescr v, w;

	// Add source in worklist
	worklist.push(source);
	explored.insert(source);

	if (source == target) {
		return 1;
	}

	while (!worklist.empty())
	{
		v = worklist.front();
		worklist.pop();
		// Iterate through all outgoing edges
		if (boost::out_degree(v, reference_graph) != 0) {
			for (boost::tie(ei, ei_end) = boost::out_edges(v, reference_graph); ei != ei_end; ++ei) {
				w = boost::target(*ei, reference_graph);
				if (e1.count(std::make_pair(v, w)) == 0) {
					// (v,w) not in non-included edges
					if (e1.count(std::make_pair(v, w)) == 0) {
						if (w == target) {
							// Reached target
							return 1;
						}
						// if not explored, add to worklist
						if (explored.count(w) == 0) {
							worklist.push(w);
							explored.insert(w);
						}
					}
					else {
						// Sample to check if edge (v,w) exists
						if (checkExist(get(weights, edge(v, w, reference_graph).first))) {
							// Edge exists
							//add_edge(v, w, 1, directed_graph);	// Add to graph
							// Check if w is target
							if (w == target) {
								// Reached target
								return 1;
							}
							// if not explored, add to worklist
							if (explored.count(w) == 0) {
								worklist.push(w);
								explored.insert(w);
							}
						}
					}
				}
			}
		}
	}
	// Target not found
	return 0;
}


MonteCarloBFS::~MonteCarloBFS()
{
}
