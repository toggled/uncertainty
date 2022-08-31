#pragma once
#include <queue>
#include "Graph.h"
class MonteCarloBFS :
	public Graph
{
protected:
	DirectedGraph reference_graph;
	boost::property_map<DirectedGraph, boost::edge_weight_t>::type weights;
public:
	MonteCarloBFS();
	MonteCarloBFS(DirectedGraph& graph);
	int run(VertexDescr source, VertexDescr target);
	int run_forRSS(VertexDescr source, VertexDescr target, std::set<Edge_s_t> e1, std::set<Edge_s_t> e2);
	~MonteCarloBFS();
};

