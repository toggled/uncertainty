#pragma once
#include <boost/dynamic_bitset.hpp>
#include <map>
#include "MonteCarloSampling.h"
#include "FileIO.h"
#include "Graph.h"

class MonteCarloSamplingDFSSharing
{
private:
	std::vector<std::vector<std::vector<char> > > bfs_sharing_data;
	std::vector<std::vector<boost::dynamic_bitset<> > > bfs_sharing_map;
	boost::dynamic_bitset<> source_target_bitmap;
public:
	MonteCarloSamplingDFSSharing(Graph original_graph, int k);
	MonteCarloSamplingDFSSharing(std::string file_name);
	~MonteCarloSamplingDFSSharing();
	void generateSourceTargetBitmap(VertexDescr source_node, VertexDescr target_node, Graph reference);
	bool checkReachable(size_t k_index);
};

