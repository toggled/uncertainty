//
//  Bag.h
//  UncertainGraph
//
//  Created by Silviu Maniu on 4/2/13.
//  Copyright (c) 2013 Silviu Maniu. All rights reserved.
//

#ifndef __UncertainGraph__Bag__
#define __UncertainGraph__Bag__

#include <iostream>
#include <fstream>
#include <math.h>
#include <limits>

#include <boost/config.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/adjacency_list.hpp>

#include "definitions.h"
#include "ProbabilisticGraph.h"
#include "DistanceDistribution.h"
#include "EvalNode.h"

//typedef boost::property<boost::edge_weight_t,float> EdgeWeightProperty;
typedef boost::property<boost::edge_weight_t, double> EdgeWeightProperty;
typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, boost::no_property, EdgeWeightProperty> ProbGraph;
typedef boost::graph_traits<ProbGraph>::vertex_descriptor vertex_descriptor;
typedef boost::graph_traits<ProbGraph>::edge_descriptor edge_descriptor;
typedef std::pair<NodeIdType, NodeIdType> Edge;

typedef std::unordered_map<NodeIdType, std::unordered_map<NodeIdType, DistanceDistribution*>*> DistanceMap;
typedef std::unordered_map<NodeIdType, std::unordered_map<NodeIdType, EvalNode*>*> TreeMap;

class Bag{
  std::string* bag_name;
  std::unordered_set<NodeIdType>* nodes;
  std::unordered_set<NodeIdType>* covered_nodes;
  std::unordered_set<NodeIdType>* uncovered_nodes;
  std::unordered_set<Bag*>* children;
  Bag* parent;
  DistanceMap* distance_map;
  TreeMap* tree_map;
  DistanceMap* uncompressed_distances;
  TreeMap computed_trees;
  EdgeMap edges;
  EdgeMap computed_edges;
  
  int number_edges;
  int number_computed_edges;
  int number_computed_trees;
  int number_distances;
  int number_trees;
  int level;
  int type = TreeDecNode;
  bool contains_query_nodes;
public:
  Bag(std::string* bag_name);
  Bag() {};
  ~Bag(){ delete nodes; delete covered_nodes; delete children; delete uncovered_nodes; delete distance_map; delete bag_name;};
  std::string* get_id() {return bag_name;};
  void set_id(std::string* bag_id) {delete bag_name; bag_name = bag_id;};
  void add_node_to_bag(NodeIdType node);
  void cover_node(NodeIdType node);
  void uncover_node(NodeIdType node);
  void add_edge_to_bag(NodeIdType node, EdgeType* edge);
  std::unordered_set<NodeIdType>* get_bag_nodes();
  std::unordered_set<NodeIdType>* get_uncovered_bag_nodes();
  std::unordered_set<NodeIdType>* get_covered_bag_nodes();
  bool has_node(NodeIdType node);
  EdgeType* find_edge(NodeIdType first, NodeIdType second);
  bool has_uncovered_node(NodeIdType node);
  void add_to_children(Bag* bag);
  void remove_from_children(Bag* bag);
  std::unordered_set<Bag*>* get_children() {return children;};
  void set_parent(Bag* bag);
  Bag* get_parent() {return parent;};
  int get_level() {return level;};
  int get_width() {return nodes->size()-1;};
  int get_number_edges() {return number_edges;};
  int get_number_computed_edges() {return number_computed_edges;};
  int get_number_nodes() {return (int)nodes->size();};
  EdgeMap* get_edges() {return &edges;};
  EdgeMap* get_computed_edges() {return &computed_edges;};
  TreeMap* get_computed_trees() {return &computed_trees;};
  std::unordered_set<NodeIdType>* get_nodes() {return nodes;};
  DistanceMap* get_computed_distances();
  TreeMap* get_tree_map();
  void set_level(int bag_level) {level=bag_level;};
  void list_bag_nodes();
  void set_number_of_edges(int number_edges);
  void write_bag_to_file(std::ofstream &file);
  void write_bag_to_file_root(std::ofstream &file);

  void load_bag_from_file(std::ifstream &file);
  void load_spqr_bag_from_file(std::ifstream &file);
  void compute_all_paths(int samples=10000);
  void uncompress_paths(NodeIdType src, NodeIdType tgt);
  void clear_computed_edges();
  void clear_computed_trees();
  EdgeType* find_computed_edge(NodeIdType first, NodeIdType second);
  EvalNode* find_computed_tree(NodeIdType first, NodeIdType second);
  EdgeType* find_any_edge(NodeIdType first, NodeIdType second);
  void add_computed_edge(NodeIdType node, EdgeType* edge);
  void add_computed_tree(NodeIdType node, EvalNode* tree);
  void replace_computed_tree(NodeIdType node, EvalNode* tree);
  bool has_query_nodes() {return contains_query_nodes;};
  void set_has_query_nodes(bool q) {contains_query_nodes=q;};
  double get_possible_worlds();
private:
  void compute_edge(NodeIdType node_first, NodeIdType node_second, int samples=10000);
  void compute_new_edge(NodeIdType node_first, NodeIdType node_second,NodeIdType src=-1, NodeIdType tgt=-1);
  void compute_tree(NodeIdType node_first, NodeIdType node_second);
  EvalNode* compute_tree_branch(NodeIdType src, NodeIdType tgt);
  EvalNode* get_edge_tree(NodeIdType src, NodeIdType tgt);
};

#endif /* defined(__UncertainGraph__Bag__) */
