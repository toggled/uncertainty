#pragma once
#include <boost/dynamic_bitset.hpp>
#include <queue>
#include "Common.h"
#include "FileIO.h"
#include "Graph.h"

class BFSSharing
{
private:
	// k worlds
	size_t k;
	// Graph
	DirectedGraph g;
	// Size of graph
	size_t size;
	// Offline compact world data
	std::string compact_world_file;
	std::map<VertexDescr, std::map<VertexDescr, std::string>> index;
	// Set of explored vertices
	std::set<VertexDescr> explored_set;
	// Bit vector map
	std::map<VertexDescr, boost::dynamic_bitset<> > vertex_bitmap;
	boost::dynamic_bitset<> getEdgeBitVector(std::vector<char> bit_vector);
	// Update(), algorithm #3
	void update(VertexDescr v, VertexDescr u, std::set<VertexDescr> U);
	std::string bitEncode(std::vector<char> raw);
	boost::dynamic_bitset<> bitDecode(std::string encoded);
public:
	size_t getK();
	BFSSharing(Graph original_graph, int k, int i);
	BFSSharing(std::string offline_sampled_world);
	double getReliability(Graph original_graph, VertexDescr source_node, VertexDescr target_node);
	boost::dynamic_bitset<> getFinalBitVector(Graph original_graph, VertexDescr source_node, VertexDescr target_node);
	~BFSSharing();
	static void generateHashFile(Graph original_graph, std::string index_file_name);
};