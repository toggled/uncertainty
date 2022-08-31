#pragma once
#include <iostream>
#if defined(__has_include)
#if __has_include(<filesystem>)
#include <filesystem>
#endif
#if __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
#endif
#endif
#include <boost/graph/graphviz.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/random.hpp>
#include <numeric>
#include <queue>
#include "Common.h"
#include "GraphVisitor.h"
#include "GraphAllPathsHelper.h"
#include "Mapping.h"
#include "Randomiser.h"


class Graph
{
protected:
	DirectedGraph directed_graph;
	std::string graph_name;
	void add_edge(VertexDescr origin, VertexDescr destination, DirectedGraph &g);
	void add_edge(VertexDescr origin, VertexDescr destination, double probability, DirectedGraph &g);
	void instantiate(std::list<Mapping> mapping_list);
public:
	Graph();
	Graph(DirectedGraph graph);
	Graph(std::list<Mapping> mapping_list);
	Graph(std::list<Mapping> mapping_list, std::string file_path);
	~Graph();
	DirectedGraph * getGraph();
	bool checkExist(double probability);
	bool checkReachable(VertexDescr source_node, VertexDescr target_node);
	void printEdgeList();
	void printGraph();
	void printStats();
	size_t getNumOutEdges();
	size_t getNumOutEdges(int vertex_index);
	VertexDescr getRandomVertex();
	std::string getGraphName();
	std::vector<std::vector<VertexDescr>> getUniquePathsFromSource(VertexDescr source, VertexDescr target, DirectedGraph const& g);
	std::vector<VertexDescr> getBFSKDistanceAwayVertices(int threshold, VertexDescr source, DirectedGraph const& g);
	std::vector<VertexDescr> getDFSKDistanceAwayVertices(int threshold, VertexDescr source, DirectedGraph const& g);
	std::vector<std::pair<VertexDescr, VertexDescr>> getKUniquePairsSourceTargetVertices(size_t k, int threshold, DirectedGraph const& g);
	Graph simplify(VertexDescr source);
};

