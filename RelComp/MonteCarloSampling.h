#pragma once
#include "Graph.h"
class MonteCarloSampling :
	public Graph
{
protected:
	void instantiate(DirectedGraph& reference_graph);
public:
	MonteCarloSampling(DirectedGraph& reference_graph);
	~MonteCarloSampling();
};

