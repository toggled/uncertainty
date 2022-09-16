#pragma once
#include <stack>
#include <queue>
#include <unordered_set>
#include <functional>
#include "Common.h"
#include "Graph.h"
#include "MonteCarloBFS.h"
#include "Randomiser.h"
#include <string>
#include <vector>

class RSS
{
private:
	boost::property_map<DirectedGraph, boost::edge_weight_t>::type weights;
	Graph reference_graph;
	DirectedGraph dg;
	VertexDescr t;
	
	// Here we set the r in RSS as 4 in this implementation since the outgoing edge number is small in our current experiments
	// One can replace it with any other number by enriching the following states when the outgoing edge number are huge,
	// and reset the threshold (now is 4) in the .cpp file.

	std::vector<std::string> states = {"0001","0010","0011","0100","0101","0110","0111","1000","1001","1010","1011","1100","1101","1110","1111","0000"};

public:
	RSS();
	RSS(Graph graph);
	void setTarget(VertexDescr target);
	double findReliability_RSS(std::unordered_set<VertexDescr> sv_map, std::queue<Edge_s_t> si, size_t n, int flag, std::vector<VertexDescr> nodes);
	std::queue<Edge_s_t> getOutgoingNeighbours_RSS(VertexDescr w, std::queue<Edge_s_t> si);
	double samplingR_RSS(std::unordered_set<VertexDescr> sv_map, std::queue<Edge_s_t> si);
	std::queue<Edge_s_t> getInitStack_RSS(VertexDescr source);
	// when the edges to be explored is less than r, it calls RHH
	// In RHH (for RSS), if we found the number of unexplored edges is more than r, we call back RSS
	double findReliability_RHH_forRSS(std::unordered_set<VertexDescr> sv_map, std::queue<Edge_s_t> si, size_t n, bool flag, VertexDescr node);
	~RSS();
};