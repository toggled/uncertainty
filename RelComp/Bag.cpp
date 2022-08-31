//
//  Bag.cpp
//  UncertainGraph
//
//  Created by Silviu Maniu on 4/2/13.
//  Copyright (c) 2013 Silviu Maniu. All rights reserved.
//

#include "Bag.h"
#include "ShortestPathHeapSampler.h"

#include <cmath>

Bag::Bag(std::string* bbag_name){
    this->bag_name = bbag_name;
    parent = NULL;
    nodes = new std::unordered_set<NodeIdType>();
    covered_nodes = new std::unordered_set<NodeIdType>();
    uncovered_nodes = new std::unordered_set<NodeIdType>();
    children = new std::unordered_set<Bag*>();
    distance_map = new DistanceMap();
    tree_map = new TreeMap();
    number_edges = 0;
    number_distances = 0;
    number_computed_edges = 0;
    number_computed_trees = 0;
    number_trees = 0;
    level = -1;
    contains_query_nodes = false;
    uncompressed_distances = nullptr;
};

void Bag::add_node_to_bag(NodeIdType node){
    nodes->insert(node);
};

void Bag::cover_node(NodeIdType node){
    covered_nodes->insert(node);
    uncovered_nodes->erase(node);
}

void Bag::uncover_node(NodeIdType node){
    uncovered_nodes->insert(node);
    covered_nodes->erase(node);
}

void Bag::add_edge_to_bag(NodeIdType node, EdgeType* edge){
    if(edges.find(node)==edges.end())
        edges[node] = new std::unordered_map<NodeIdType, EdgeType*>();
    
    (*edges[node])[edge->first]=edge;
    
    number_edges++;
}

void Bag::add_computed_edge(NodeIdType node, EdgeType* edge){
    if(computed_edges.find(node)== computed_edges.end())
        computed_edges[node] = new std::unordered_map<NodeIdType, EdgeType*>();
    else{
        if(computed_edges[node]->find(edge->first)!=computed_edges[node]->end())
            delete (*computed_edges[node])[edge->first];
    }
    (*computed_edges[node])[edge->first]=edge;
    number_computed_edges++;
}

void Bag::add_computed_tree(NodeIdType node, EvalNode* tree){
    if(computed_trees.find(node)==computed_trees.end())
        computed_trees[node] = new std::unordered_map<NodeIdType, EvalNode*>();
    else{
        if(computed_trees[node]->find(tree->get_tgt())!=\
           computed_trees[node]->end())
            delete (*computed_trees[node])[tree->get_tgt()];
    }
    (*computed_trees[node])[tree->get_tgt()] = tree;
    number_computed_trees++;
}

void Bag::replace_computed_tree(NodeIdType node, EvalNode* tree){
    (*computed_trees[node])[tree->get_tgt()] = tree;
}

std::unordered_set<NodeIdType>* Bag::get_bag_nodes(){
    return nodes;
}

bool Bag::has_node(NodeIdType node){
    bool return_value = false;
    if(nodes->find(node)!=nodes->end())
        return_value = true;
    return return_value;
}

EdgeType* Bag::find_edge(NodeIdType first, NodeIdType second){
    if(edges.find(first)!=edges.end()){
        if(edges[first]->find(second)!=edges[first]->end())
            return (*edges[first])[second];
    }
    return nullptr;
}

EdgeType* Bag::find_computed_edge(NodeIdType first, NodeIdType second){
    if(computed_edges.find(first)!=computed_edges.end()){
        if(computed_edges[first]->find(second)!=computed_edges[first]->end())
            return (*computed_edges[first])[second];
    }
    return nullptr;
}

EvalNode* Bag::find_computed_tree(NodeIdType first, NodeIdType second){
    if(computed_trees.find(first)!=computed_trees.end()){
        if(computed_trees[first]->find(second)!=computed_trees[first]->end())
            return (*computed_trees[first])[second];
    }
    return nullptr;
}

EdgeType* Bag::find_any_edge(NodeIdType first, NodeIdType second){
    EdgeType* edge = find_computed_edge(first, second);
    if(edge==nullptr) edge = find_edge(first, second);
    return edge;
}

bool Bag::has_uncovered_node(NodeIdType node){
    bool return_value = false;
    if(uncovered_nodes->find(node)!=uncovered_nodes->end())
        return_value = true;
    return return_value;
}

std::unordered_set<NodeIdType>* Bag::get_uncovered_bag_nodes(){
    return uncovered_nodes;
}

std::unordered_set<NodeIdType>* Bag::get_covered_bag_nodes(){
    return covered_nodes;
}

void Bag::list_bag_nodes(){
    for(NodeIdType node:*nodes) std::cout << node << " ";
    std::cout << "\n";
}

void Bag::load_spqr_bag_from_file(std::ifstream &file){
    delete bag_name;
    std::string* b_name = new std::string();
    file >> *b_name;
    bag_name = b_name;
    file >> type;
    unsigned long number_nodes;
    file >> number_nodes;
    //std::cout << nodes << std::endl;
    for(unsigned long i=0;i<number_nodes;i++){
        NodeIdType node;
        file >> node;
        add_node_to_bag(node);
        cover_node(node);
    }
}

void Bag::load_bag_from_file(std::ifstream &file){
    delete bag_name;
    std::string* b_name = new std::string();
    file >> *b_name;
    bag_name = b_name;
    file >> level;
    //std::cout << level << std::endl;
    unsigned long nnodes;
    file >> nnodes;
    //std::cout << nodes << std::endl;
    for(unsigned long i=0;i<nnodes;i++){
        NodeIdType node;
        file >> node;
        add_node_to_bag(node);
        uncover_node(node);
    }
    file >> nnodes;
    for(unsigned long i=0;i<nnodes;i++){
        NodeIdType node;
        file >> node;
        cover_node(node);
        uncovered_nodes->erase(node);
    }
    //std::cout << nodes << std::endl;
    unsigned long nedges;
    file >> nedges;
    //std::cout << edges << std::endl;
    NodeIdType first, second;
    for(unsigned long i=0;i<nedges;i++){
        file >> first >> second;
        //std::cout << i << " " << first << " " << second << std::endl;
        EdgeType* edge = new EdgeType;
        edge->first = second;
        DistanceDistribution* distribution = new DistanceDistribution();
        file >> *distribution;
        //std::cout << distribution->get_max_distance() << std::endl;
        edge->second = distribution;
        edge->third = false;
        add_edge_to_bag(first, edge);
    }
    file >> nedges;
    //std::cout << edges << std::endl;
    for(unsigned long i=0;i<nedges;i++){
        file >> first >> second;
        //std::cout << i << " " << first << " " << second << std::endl;
        EdgeType* edge = new EdgeType;
        edge->first = second;
        DistanceDistribution* distribution = new DistanceDistribution();
        file >> *distribution;
        //std::cout << distribution->get_max_distance() << std::endl;
        edge->second = distribution;
        edge->third = false;
        add_computed_edge(first, edge);
    }
    
    unsigned long ntrees;
    file >> ntrees;
    for(unsigned long i=0;i<ntrees;i++){
        std::vector<EvalNode*> vec_comp_trees;
        NodeIdType nfrom, nto;
        file >> nfrom >> nto;
        unsigned long tnodes;
        file >> tnodes;
        for(unsigned long j=0;j<tnodes;j++){
            int id;
            int ttype;
            int toper;
            NodeIdType tsrc, ttgt;
            file >> id >> ttype >> toper >> tsrc >> ttgt;
            EvalNode* tnode = new EvalNode((EvalNodeType)ttype, (EvalNodeOper)\
                                           toper, tsrc, ttgt);
            vec_comp_trees.push_back(tnode);
        }
        unsigned long tedges;
        file >> tedges;
        for(unsigned long j=0;j<tedges;j++){
            int src, tgt, pos;
            file >> src >> tgt >> pos;
            if(pos==0) vec_comp_trees.at(src)->set_left(vec_comp_trees.at(tgt));
            else vec_comp_trees.at(src)->set_right(vec_comp_trees.at(tgt));
        }
        add_computed_tree(nfrom, vec_comp_trees.at(0));
    }
    
    file >> number_distances;
    for(int i=0;i<number_distances;i++){
        NodeIdType first, second;
        int distributions;
        file >> first >> second;
        if(distance_map->find(first)==distance_map->end())
            (*distance_map)[first] = new std::unordered_map<NodeIdType, DistanceDistribution*>();
        if((*distance_map)[first]->find(second)==(*distance_map)[first]->end())
            (*(*distance_map)[first])[second] = new DistanceDistribution();
        DistanceDistribution* distance_distribution = (*(*distance_map)[first])[second];
        file >> distributions;
        for(int j=0;j<distributions;j++){
            int distance;
            float probability;
            file >> distance >> probability;
            distance_distribution->add_to_distribution(distance, probability);
        }
    }
    
    number_trees = 0;
    file >> ntrees;
    for(unsigned long i=0;i<ntrees;i++){
        std::vector<EvalNode*> vec_map_trees;
        NodeIdType nfrom, nto;
        file >> nfrom >> nto;
        unsigned long tnodes;
        file >> tnodes;
        for(unsigned long j=0;j<tnodes;j++){
            int id;
            int ttype;
            int toper;
            NodeIdType tsrc, ttgt;
            file >> id >> ttype >> toper >> tsrc >> ttgt;
            EvalNode* tnode = new EvalNode((EvalNodeType)ttype, (EvalNodeOper)\
                                           toper, tsrc, ttgt);
            vec_map_trees.push_back(tnode);
        }
        unsigned long tedges;
        file >> tedges;
        for(unsigned long j=0;j<tedges;j++){
            int src, tgt, pos;
            file >> src >> tgt >> pos;
            if(pos==0) vec_map_trees.at(src)->set_left(vec_map_trees.at(tgt));
            else vec_map_trees.at(src)->set_right(vec_map_trees.at(tgt));
        }
        if(tree_map->find(nfrom)==tree_map->end())
            (*tree_map)[nfrom] = new std::unordered_map<NodeIdType, EvalNode*>();
        (*(*tree_map)[nfrom])[nto] = vec_map_trees.at(0);
        number_trees++;
    }
}

void Bag::write_bag_to_file(std::ofstream &file){
    file << *bag_name << "\n";
    file << level << "\n";
    file << nodes->size() << "\n";
    for(NodeIdType node:*nodes)
        file << node << "\t";
    file << "\n" << covered_nodes->size() << "\n";
    for(NodeIdType node:*covered_nodes)
        file << node << "\t";
    file << "\n" << number_edges << "\n";
    for(NodeIdType node:*nodes){
        if(edges.find(node)!=edges.end())
            for(auto edge:*(edges[node])){
                file << node << "\t" << edge.first << "\n";
                file << *edge.second->second;
            }
    }
    file << number_computed_edges << "\n";
    for(auto edge_list:computed_edges){
        NodeIdType node = edge_list.first;
        for(auto edge:*edge_list.second){
            file << node << "\t" << edge.first << "\n";
            file << *edge.second->second;
        }
    }
    
    file<< number_computed_trees << "\n";
    for(auto tree_list:computed_trees){
        NodeIdType node = tree_list.first;
        for(auto tree:*tree_list.second){
            file << node << "\t" << tree.first << "\n";
            std::vector<std::string> tree_nodes;
            std::vector<std::string> tree_edges;
            int nid = -1;
            tree.second->print(nid, tree_nodes, tree_edges);
            file << tree_nodes.size() << "\n";
            for(auto val:tree_nodes) file << val;
            file << tree_edges.size() << "\n";
            for(auto val:tree_edges) file << val;
        }
    }
    
    file << number_distances << std::endl;
    for(NodeIdType node:*nodes){
        if(distance_map->find(node)!=distance_map->end()){
            for(auto iterator=(*distance_map)[node]->begin();iterator!=(*distance_map)[node]->end();++iterator){
                file << node << "\t" << iterator->first << std::endl;
                file << *iterator->second;
            }
        }
    }
    
    file << number_trees << std::endl;

    for(NodeIdType node:*nodes){
        if(tree_map->find(node)!=tree_map->end()){
            for(auto iterator=(*tree_map)[node]->begin();iterator!=(*tree_map)[node]->end();++iterator){
                file << node << "\t" << iterator->first << std::endl;
                std::vector<std::string> tree_nodes;
                std::vector<std::string> tree_edges;
                int nid = -1;
                iterator->second->print(nid, tree_nodes, tree_edges);
                file << tree_nodes.size() << "\n";
                for(auto val:tree_nodes) file << val;
                file << tree_edges.size() << "\n";
                for(auto val:tree_edges) file << val;
            }
        }
    }
    
}


void Bag::write_bag_to_file_root(std::ofstream &file){
    for(auto edge_list:computed_edges){
        NodeIdType node = edge_list.first;
        for(auto edge:*edge_list.second){
            file << node << "\t" << edge.first <<"\t"<< *edge.second->second;
        }
    }

    file<< number_computed_trees << "\n";
    for(auto tree_list:computed_trees){
        NodeIdType node = tree_list.first;
        for(auto tree:*tree_list.second){
            file << node << "3\t" << tree.first << "\n";
            std::vector<std::string> tree_nodes;
            std::vector<std::string> tree_edges;
            int nid = -1;
            tree.second->print(nid, tree_nodes, tree_edges);
            file << tree_nodes.size() << "4-\n";
            for(auto val:tree_nodes) file << val;
            file << tree_edges.size() << "5-\n";
            for(auto val:tree_edges) file << val;
        }
    }

    file << number_distances << std::endl;
    for(NodeIdType node:*nodes){
        if(distance_map->find(node)!=distance_map->end()){
            for(auto iterator=(*distance_map)[node]->begin();iterator!=(*distance_map)[node]->end();++iterator){
                file << node << "6\t" << iterator->first << std::endl;
                file << *iterator->second;
            }
        }
    }

    file << number_trees << std::endl;

    for(NodeIdType node:*nodes){
        if(tree_map->find(node)!=tree_map->end()){
            for(auto iterator=(*tree_map)[node]->begin();iterator!=(*tree_map)[node]->end();++iterator){
                file << node << "\t" << iterator->first << std::endl;
                std::vector<std::string> tree_nodes;
                std::vector<std::string> tree_edges;
                int nid = -1;
                iterator->second->print(nid, tree_nodes, tree_edges);
                file << tree_nodes.size() << "\n";
                for(auto val:tree_nodes) file << val;
                file << tree_edges.size() << "\n";
                for(auto val:tree_edges) file << val;
            }
        }
    }

}


void Bag::compute_all_paths(int samples){
    number_distances = 0;
    delete distance_map;
    number_trees = 0;
    delete tree_map;
    distance_map = new DistanceMap();
    tree_map = new TreeMap();
    for(NodeIdType node_first:*uncovered_nodes){
        for(NodeIdType node_second:*uncovered_nodes){
            if(node_first!=node_second){
                compute_edge(node_first, node_second, samples);
                #if LIN==1
                compute_tree(node_first, node_second);
                #endif
            }
        }
    }
    //std::cout << "\t\tbag " << *bag_name << " ne:" << number_distances << std::endl;
}

void Bag::compute_edge(NodeIdType node_first, NodeIdType node_second,\
                       int samples){
    DistanceDistribution* distance_new = new DistanceDistribution();
    EdgeType* edge_direct = find_any_edge(node_first, node_second);
    if(edge_direct!=nullptr){
        distance_new->combine_distribution(edge_direct->second);
    }
    if(type==TreeDecNode){
        //this is for tree decompositions
        NodeIdType covered = *covered_nodes->begin();
        
        EdgeType *edge_to, *edge_from;
        edge_to = find_any_edge(node_first, covered);
        edge_from = find_any_edge(covered, node_second);
        
        if((edge_to!=nullptr)&&(edge_from!=nullptr)){
            distance_new->combine_distribution(propagate_distribution(edge_from->second, edge_to->second));
        }
        
    }
    else if(type==SNode){
        //serial Node for tree decomposition
        //TODO: replace sample by direct computation
        ShortestPathSampler sampler;
        delete distance_new;
        distance_new = sampler.sample(this, node_first, node_second, samples);
    }
    else if(type==RNode){
        //rigid Node for tree decomposition
        ShortestPathSampler sampler;
        delete distance_new;
        distance_new = sampler.sample(this, node_first, node_second, samples);
    }
    if((distance_new->get_max_distance()>0)&&(distance_new->get_size()>0)){
        if(distance_map->find(node_first)==distance_map->end())
            (*distance_map)[node_first] = new std::unordered_map<NodeIdType, DistanceDistribution*>();
        (*(*distance_map)[node_first])[node_second]=distance_new;
        number_distances++;
    }
    else{
        delete distance_new;
    }
}

void Bag::compute_new_edge(NodeIdType node_first, NodeIdType node_second, NodeIdType src, NodeIdType tgt){
    NodeIdType covered = *covered_nodes->begin();
    EdgeType* edge_direct = find_any_edge(node_first, node_second);
    EdgeType *edge_to, *edge_from;
    if((covered!=src)&&(covered!=tgt)){
        edge_to = find_any_edge(node_first, covered);
        edge_from = find_any_edge(covered, node_second);
    }
    else{
        edge_to = edge_from = nullptr;
    }
    DistanceDistribution* distance_new = new DistanceDistribution();
    if(edge_direct!=nullptr){
        distance_new->combine_distribution(edge_direct->second);
    }
    if((edge_to!=nullptr)&&(edge_from!=nullptr)){
        DistanceDistribution* prop = propagate_distribution(edge_from->second, edge_to->second);
        distance_new->combine_distribution(prop);
        delete prop;
    }
    if((distance_new->get_max_distance()>0)&&(distance_new->get_size()>0)){
        if(uncompressed_distances->find(node_first)==uncompressed_distances->end())
            (*uncompressed_distances)[node_first] = new std::unordered_map<NodeIdType, DistanceDistribution*>();
        (*(*uncompressed_distances)[node_first])[node_second]=distance_new;
    }
    else{
        delete distance_new;
    }
    
}

void Bag::compute_tree(NodeIdType node_first, NodeIdType node_second){
    NodeIdType covered = *covered_nodes->begin();
    
    //compute
    EvalNode *tree_to = compute_tree_branch(node_first, covered);
    EvalNode *tree_from = compute_tree_branch(covered, node_second);
    EvalNode *tree_cov  = nullptr;
    if(tree_to!=nullptr&&tree_from!=nullptr){
        tree_cov = new EvalNode(NODETYPE_INTER, NODEOPER_SUM, node_first,\
                                node_second);
        tree_cov->set_left(tree_to);
        tree_cov->set_right(tree_from);
    }
    EvalNode *tree_dir = compute_tree_branch(node_first, node_second);
    EvalNode* tree = nullptr;
    if(tree_dir!=nullptr&&tree_cov!=nullptr){
        tree = new EvalNode(NODETYPE_INTER, NODEOPER_MIN, node_first,\
                            node_second);
        tree->set_left(tree_dir);
        tree->set_right(tree_cov);
    }
    else if(tree_dir!=nullptr)
        tree = tree_dir;
    else if(tree_cov!=nullptr)
        tree = tree_cov;
    
    //add
    if(tree!=nullptr){
        if(tree_map->find(node_first)==tree_map->end())
            (*tree_map)[node_first] = new std::unordered_map<NodeIdType, EvalNode*>();
        (*(*tree_map)[node_first])[node_second] = tree;
        number_trees++;
    }
}

void Bag::uncompress_paths(NodeIdType, NodeIdType ){
    if(uncompressed_distances!=nullptr){
        for(auto iter_first:*uncompressed_distances){
            for(auto iter_second:*iter_first.second){
                delete iter_second.second;
            }
            iter_first.second->clear();
            delete iter_first.second;
        }
        uncompressed_distances->clear();
        delete uncompressed_distances;
    }
    uncompressed_distances = new DistanceMap();
    
    for(auto neigh:computed_edges){
        NodeIdType source = neigh.first;
        for(auto neigh_list:*neigh.second){
            NodeIdType target = neigh_list.first;
            DistanceDistribution *distance_new = new DistanceDistribution();
            distance_new->copy_from_distribution(neigh_list.second->second);
            if(uncompressed_distances->find(source)==uncompressed_distances->end())
                (*uncompressed_distances)[source] = new std::unordered_map<NodeIdType, DistanceDistribution*>();
            (*(*uncompressed_distances)[source])[target]=distance_new;
        }
    }
}

void Bag::set_number_of_edges(int nnumber_edges){
    this->number_edges = nnumber_edges;
}

void Bag::add_to_children(Bag* bag){
    children->insert(bag);
}

void Bag::remove_from_children(Bag* bag){
    children->erase(bag);
}

void Bag::set_parent(Bag* bag){
    parent = bag;
}

void Bag::clear_computed_edges(){
    number_computed_edges=0;
    for(auto iter:computed_edges){
        for(auto iter_second:*iter.second){
            delete iter_second.second->second;
            delete iter_second.second;
        }
        delete iter.second;
    }
    computed_edges.clear();
}

void Bag::clear_computed_trees(){
    number_computed_trees=0;
    computed_trees.clear();
}

DistanceMap* Bag::get_computed_distances(){
    if(contains_query_nodes)
        return uncompressed_distances;
    else
        return distance_map;
}

TreeMap* Bag::get_tree_map(){
    return tree_map;
}

double Bag::get_possible_worlds(){
    double possible_worlds = 0;
    for(auto node_id:*nodes){
        std::unordered_set<NodeIdType> visited_outgoing;
        auto outgoing_computed = get_computed_edges();
        if(outgoing_computed->find(node_id)!=outgoing_computed->end()){
            for(auto edge:*outgoing_computed->operator[](node_id)){
                possible_worlds += log(1.0+(double)edge.second->second->get_size());
            }
        }
        auto outgoing = get_edges();
        if(outgoing->find(node_id)!=outgoing->end()){
            for(auto edge:*outgoing->operator[](node_id)){
                if(visited_outgoing.find(edge.first)==visited_outgoing.end()){
                    possible_worlds += log(1.0+(double)edge.second->second->get_size());
                }
            }
        }
    }
    return possible_worlds;
}

EvalNode* Bag::compute_tree_branch(NodeIdType src, NodeIdType tgt){
    EdgeType* edge = find_edge(src, tgt);
    EvalNode* tree = find_computed_tree(src, tgt);
    EvalNode* tree_out = nullptr;
    if(tree!=nullptr&&edge!=nullptr){
        EvalNode* etree = new EvalNode(NODETYPE_LEAF, NODEOPER_NONE, src, tgt);
        tree_out = new EvalNode(NODETYPE_INTER, NODEOPER_MIN, src, tgt);
        tree_out->set_left(etree);
        tree_out->set_right(tree);
    }
    else if(tree!=nullptr){
        tree_out = tree;
    }
    else if(edge!=nullptr){
        tree_out = new EvalNode(NODETYPE_LEAF, NODEOPER_NONE, src, tgt);
    }
    return tree_out;
}





