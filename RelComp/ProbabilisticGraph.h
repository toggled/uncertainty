//
//  Graph.h
//  UncertainGraph
//
//  Created by Silviu Maniu on 1/2/13.
//  Copyright (c) 2013 Silviu Maniu. All rights reserved.
//

#ifndef __UncertainGraph__Graph__
#define __UncertainGraph__Graph__

#include <iostream>
#include "definitions.h"

class ProbabilisticGraph{
    EdgeMap outgoing;
    EdgeMap incoming;
    EdgeMap undirected;
    unsigned long number_edges;
    
    DistanceDistribution* undirected_distribution;
public:
    ProbabilisticGraph(std::ifstream& input_stream, bool isWeighted = false);
    ~ProbabilisticGraph() {delete undirected_distribution;};
    bool is_node(NodeIdType node_id);
    std::unordered_map<NodeIdType, EdgeType*>* get_outgoing_nodes(NodeIdType node_id);
    std::unordered_map<NodeIdType, EdgeType*>* get_incoming_nodes(NodeIdType node_id);
    std::unordered_map<NodeIdType, EdgeType*>* get_neighbours(NodeIdType node_id);
    std::vector<NodeIdType> get_node_vector();
    void add_edge(NodeIdType key_first, NodeIdType key_second, DistanceDistribution* value);
    void add_weighted_edge(NodeIdType key_first, NodeIdType key_second, double weight, DistanceDistribution* value);
    EdgeType* get_edge(NodeIdType key_first, NodeIdType key_second);
    void remove_edge(NodeIdType key_first, NodeIdType key_second);
    void add_undirected_edge(NodeIdType key_first, NodeIdType key_second, DistanceDistribution* value, double weight=1.0);
    void remove_node(NodeIdType node);
    void remove_edge(NodeIdType key_first, EdgeType* edge);
    unsigned long get_number_nodes() {return (unsigned long)undirected.size();};
    unsigned long get_number_edges() {return number_edges;};
    void write_graph(std::ofstream& output_stream);
private:
    void add_node(NodeIdType key);
};

#endif /* defined(__UncertainGraph__Graph__) */
