//
//  TreeDecomposition.h
//  UncertainGraph
//
//  Created by Silviu Maniu on 4/2/13.
//  Copyright (c) 2013 Silviu Maniu. All rights reserved.
//

#ifndef __UncertainGraph__TreeDecomposition__
#define __UncertainGraph__TreeDecomposition__

#include "definitions.h"
#include "Bag.h"
#include "ProbabilisticGraph.h"

class TreeDecomposition{
    std::vector<Bag*>* bags;
    Bag* root_bag;
    ProbabilisticGraph* graph;
    int number_bags, height, width, treewidth;
    std::unordered_map<unsigned long, std::unordered_set<NodeIdType>*> degrees;
    std::unordered_map<NodeIdType, unsigned long> node_degrees;
    std::unordered_set<NodeIdType>* covered;
    std::unordered_map<std::string,Bag*> bag_map;
    std::unordered_map<int,std::vector<Bag*>*> level_map;
    
    DistanceDistribution* undirected_distribution;

public:
    TreeDecomposition(ProbabilisticGraph* graph);
    TreeDecomposition(std::string path_prefix, std::string graph_name);
    ~TreeDecomposition() { delete bags; delete root_bag;};
    void decompose_graph(int width);
    void load_decomposition(std::string path_prefix, std::string graph_name);
    void write_decomposition();
    void write_decomposition_tot(NodeIdType source, NodeIdType target, std::string graph_name);

    Bag* get_root() {return root_bag;};
    void propagate_computations();
    int redo_computations(NodeIdType src=-1, NodeIdType tgt=-1);
private:
    void create_tree();
    bool link_bags(Bag* child, Bag* parent);
    void propagate_children(Bag* bag);
    void reduce_bag_from_node(NodeIdType node);
    void remove_from_degrees(NodeIdType node);
    void move_edges_to_bag(Bag* bag);
    void increase_degree(NodeIdType node, unsigned long degree);
    bool test_for_clique(NodeIdType node);
};

#endif /* defined(__UncertainGraph__TreeDecomposition__) */
