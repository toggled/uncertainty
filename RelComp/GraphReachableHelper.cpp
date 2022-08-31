#include "GraphReachableHelper.h"



GraphReachableHelper::GraphReachableHelper()
{
}

// Gets all possible vertices on all paths from source to target
std::set<VertexDescr> GraphReachableHelper::getReachableSet(DirectedGraph g, VertexDescr source, VertexDescr target)
{
	reachable.clear();
	explored.clear();
	if (checkReachable(g, source, target)) {
		reachable.insert(source);
	}
	return reachable;
}

bool GraphReachableHelper::checkReachable(DirectedGraph g, VertexDescr v, VertexDescr target)
{
	OutEdgeIter out_i, out_end;
	VertexDescr w;
	bool reachableFlag = false;
	// Set v as explored
	explored.insert(v);
	// Return true if reached target
	if (v == target) {
		return true;
	}
	// Return false if reached leaf node
	if (boost::out_degree(v, g) == 0) {
		return false;
	}
	for (boost::tie(out_i, out_end) = boost::out_edges(v, g); out_i != out_end; ++out_i) {
		w = boost::target(*out_i, g);
		if (explored.count(w) != 0 && w != target) {
			// Skip if w is explored and is not target
			continue;
		}
		else if (checkReachable(g, w, target)) {
			reachable.insert(w);
			reachableFlag = true;
		}
	}
	if (reachableFlag) {
		return true;
	}
	else {
		return false;
	}
}


GraphReachableHelper::~GraphReachableHelper()
{
}
