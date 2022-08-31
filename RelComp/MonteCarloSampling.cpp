#include "MonteCarloSampling.h"



void MonteCarloSampling::instantiate(DirectedGraph& reference_graph)
{
	std::pair<EdgeIter, EdgeIter> ep;
	VertexDescr u, v;
	boost::property_map<DirectedGraph, boost::edge_weight_t>::type weights = boost::get(boost::edge_weight_t(), reference_graph);

	// Prevent assert error
	u = num_vertices(reference_graph);
	v = u + 1;
	add_edge(u, v, directed_graph);
	clear_vertex(u, directed_graph);
	clear_vertex(v, directed_graph);


	for (ep = edges(reference_graph); ep.first != ep.second; ++ep.first) {
		u = source(*ep.first, reference_graph);
		v = target(*ep.first, reference_graph);

		if (checkExist(get(weights, *ep.first))) {
			// Edge exists
			add_edge(u, v, 1, directed_graph);
		}
		else {
			// Edge does not exist
			// Do not add any edge
		}
	}
}

MonteCarloSampling::MonteCarloSampling(DirectedGraph& reference_graph)
{
	Graph();
	instantiate(reference_graph);
}


MonteCarloSampling::~MonteCarloSampling()
{
}
