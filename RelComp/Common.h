#pragma once
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/random.hpp>
#include "Constants.h"

// Graph Definitions
typedef boost::property<boost::edge_weight_t, double> EdgeWeightProperty;
typedef boost::adjacency_list<boost::listS, boost::vecS, boost::bidirectionalS, boost::property<boost::vertex_index_t, int>, EdgeWeightProperty > DirectedGraph;
typedef boost::graph_traits<DirectedGraph>::vertex_descriptor VertexDescr;
typedef std::pair<VertexDescr, VertexDescr> Edge_s_t;
typedef boost::graph_traits<DirectedGraph>::edge_descriptor EdgeDescr;
typedef boost::graph_traits<DirectedGraph>::edge_iterator EdgeIter;
typedef boost::graph_traits<DirectedGraph>::vertex_iterator VertexIter;
typedef boost::graph_traits<DirectedGraph>::in_edge_iterator InEdgeIter;
typedef boost::graph_traits<DirectedGraph>::out_edge_iterator OutEdgeIter;

//typedef boost::adjacency_list<boost::listS, boost::vecS, boost::undirectedS, boost::no_property, EdgeWeightProperty > UndirectedGraph;