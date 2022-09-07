    //
//  TreeDecomposition.cpp
//  UncertainGraph
//
//  Created by Silviu Maniu on 4/2/13.
//  Copyright (c) 2013 Silviu Maniu. All rights reserved.
//

#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <string>

#include "TreeDecomposition.h"

TreeDecomposition::TreeDecomposition(ProbabilisticGraph* ggraph){
    bags = new std::vector<Bag*>();
    covered = new std::unordered_set<NodeIdType>();
    this->graph = ggraph;
    number_bags = 0;
    undirected_distribution = new DistanceDistribution();
    undirected_distribution->add_to_distribution(1, 1.0f);
};

TreeDecomposition::TreeDecomposition(std::string path_prefix, std::string graph_name){
    bags = new std::vector<Bag*>();
    covered = new std::unordered_set<NodeIdType>();
    number_bags = 0;
    load_decomposition(path_prefix, graph_name);
}

void TreeDecomposition::decompose_graph(int wwidth){
    //node degree map
    width = wwidth;
    std::cout << "decomposing graph (w=" << wwidth << ")... " << std::flush;
    timestamp_t t0, t1;
    float time_msec;
    t0 = get_timestamp();
    for(NodeIdType node_id:graph->get_node_vector()){
        unsigned long degree = graph->get_neighbours(node_id)->size();
        if(degrees.find(degree)==degrees.end()) degrees[degree]=new std::unordered_set<NodeIdType>();
        degrees[degree]->insert(node_id);
        node_degrees[node_id]=degree;
    }
    //main loop of tree decomposition
    for(int d=1; d<=width; d++)
        while((degrees.find(d)!=degrees.end())&&(degrees[d]->size()>0)){
            //std::cout << "reducing node " << *(degrees[d]->begin()) << "\n";
            //if(test_for_clique(*(degrees[d]->begin())))
                reduce_bag_from_node(*(degrees[d]->begin()));
            //else
                //making sure it does not go into an infinite loop
                //degrees[d]->erase(*(degrees[d]->begin()));
        }
    t1 = get_timestamp();
    time_msec = (t1-t0)/1000000.0L;
    std::cout << "done in " << time_msec << "(sec)" << std::endl;
    std::cout << "found " << bags->size() << " bags" << std::endl;
    std::cout << "creating root bag... " << std::flush;
    t0 = get_timestamp();
    //writing the root bag
    std::ostringstream bag_name;
    bag_name << "root";
    std::string* bag_name_str = new std::string(bag_name.str());
    Bag* bag = new Bag(bag_name_str);
    for(NodeIdType node:graph->get_node_vector()){
        if(covered->find(node)==covered->end()){
            bag->add_node_to_bag(node);
            bag->cover_node(node);
        }
    }
    root_bag = bag;
    //add bag to list
    //std::cout << std::endl << "root bag" << std::endl;
    //bag->list_bag_nodes();
    move_edges_to_bag(bag);
    bag_map["root"] = bag;
    t1 = get_timestamp();
    time_msec = (t1-t0)/1000000.0L;
    std::cout << "done in " << time_msec << "(sec)" << std::endl;
    
    //creating the tree;
    t0 = get_timestamp();
    std::cout << "creating tree links... " << std::flush;
    create_tree();
    t1 = get_timestamp();
    time_msec = (t1-t0)/1000000.0L;
    std::cout << "done in " << time_msec << "(sec)" << std::endl;
    std::cout << "width=" << width << " height=" << height << std::endl;
    //propagating upwards
    t0 = get_timestamp();
    std::cout << "computing and propagating... " << std::flush;
    propagate_computations();
    t1 = get_timestamp();
    time_msec = (t1-t0)/1000000.0L;
    std::cout << "done in " << time_msec << "(sec)" << std::endl;
    //writing decomposition
    t0 = get_timestamp();
    std::cout << "writing to disk... " << std::flush;
    write_decomposition();
    t1 = get_timestamp();
    time_msec = (t1-t0)/1000000.0L;
    std::cout << "done in " << time_msec << "(sec)" << std::endl;
};

void TreeDecomposition::reduce_bag_from_node(NodeIdType node){
    std::ostringstream bag_name;
    bag_name << "bag_" << (number_bags)++;
    std::string* bag_name_str = new std::string(bag_name.str());
    Bag* bag = new Bag(bag_name_str);
    std::vector<NodeIdType> outside_nodes;
    //add all nodes in bag and remove the original node from the graph
    for(auto edge:*(graph->get_neighbours(node))){
        if((covered->find(edge.first)==covered->end())&&(edge.first!=node)){
            bag->add_node_to_bag(edge.first);
            bag->uncover_node(edge.first);
            outside_nodes.push_back(edge.first);
        }
    }
    
    bag->add_node_to_bag(node);
    remove_from_degrees(node);
    bag->cover_node(node);
    graph->remove_node(node);
    covered->insert(node);
    //add clique between outside_nodes;
    for(NodeIdType node_first:outside_nodes){
        unsigned long degree_node_first = node_degrees[node_first];
        for(NodeIdType node_second:outside_nodes){
            if(node_first<node_second){
                unsigned long degree_node_second = node_degrees[node_second];
                bool found = false;
                auto neighbourhood = graph->get_neighbours(node_first);
                if(neighbourhood!=nullptr){
                    for(auto node_neigh:*neighbourhood){
                        if(node_neigh.first==node_second){
                            found = true;
                            break;
                        }
                }
                }
                if(!found){
                    graph->add_undirected_edge(node_first, node_second, undirected_distribution);
                    graph->add_undirected_edge(node_second, node_first, undirected_distribution);
                    increase_degree(node_first, degree_node_first);
                    increase_degree(node_second, degree_node_second);
                }
            }
        }
    }
    //add edges to bag
    move_edges_to_bag(bag);
    bags->push_back(bag);
    bag_map[*bag_name_str] = bag;
};

void TreeDecomposition::remove_from_degrees(NodeIdType node){
    unsigned long degree = node_degrees[node];
    for(auto node_neigh:*(graph->get_neighbours(node))){
        NodeIdType node_out = node_neigh.first;
        if(covered->find(node_out)==covered->end()){
            unsigned long degree_out = node_degrees[node_out];
            node_degrees.erase(node_out);
            degrees[degree_out]->erase(node_out);
            degree_out--;
                node_degrees[node_out] = degree_out;
                if(degrees.find(degree_out)==degrees.end())
                    degrees[degree_out] = new std::unordered_set<NodeIdType>();
                degrees[degree_out]->insert(node_out);
                //std::cout << "\t changed node " << node_out << " deg: " << degree_out+1 << "->" << degree_out << "\n";
        }
    }
    //std::cout << "\t removed node " << node << " deg: " << degree << "\n";
    node_degrees.erase(node);
    degrees[degree]->erase(node);
};

void TreeDecomposition::move_edges_to_bag(Bag *bag){
    int number_edges = 0;
    for(NodeIdType node:*(bag->get_bag_nodes()))
        for(auto edge:*(graph->get_outgoing_nodes(node)))
            if((bag->has_node(edge.first))&&(!edge.second->third)){
                number_edges++;
                bag->add_edge_to_bag(node, edge.second);
                edge.second->third = true;
            }
    bag->set_number_of_edges(number_edges);
}

void TreeDecomposition::write_decomposition(){
    std::cout<<"write_decomposition()-start\n";
    std::ofstream file_bags("bags");
    for(Bag* bag:*(bags)) bag->write_bag_to_file(file_bags);
    file_bags.close();
    std::ofstream file_root("root");
    root_bag->write_bag_to_file(file_root);
    file_root.close();
    std::ofstream file("index");
    file << number_bags;
    file.close();
    std::ofstream file_tree("tree");
    for(Bag* bag:*bags){
        std::string* parent_id;
        if(bag->get_parent()==nullptr) parent_id = new std::string("nullptr");
        else parent_id=bag->get_parent()->get_id();
        file_tree << *(bag->get_id()) << "\t" << *parent_id << std::endl;
    }
    file_tree.close();
    std::ofstream file_stats("stats");
    file_stats << treewidth << " " << bags->size() << " " << width << " " << height << " " << root_bag->get_number_nodes() << " " << root_bag->get_number_edges() << " " << root_bag->get_number_computed_edges() << std::endl;
    file_stats.close();
    std::cout<<"write_decomposition()-end\n";

}


/*void TreeDecomposition::write_decomposition_tot( NodeIdType source, NodeIdType target, ){
	std::string s1 = std::to_string(source);
	std::string t1 = std::to_string(target);

    std::ofstream file_bags("bags_final_"+s1+"_"+t1);
    for(Bag* bag:*(bags)) bag->write_bag_to_file(file_bags);
    file_bags.close();
    std::ofstream file_root("root_final_"+s1+"_"+t1);
    root_bag->write_bag_to_file(file_root);
    file_root.close();
    std::ofstream file("index_final_"+s1+"_"+t1);
    file << number_bags;
    file.close();
    std::ofstream file_tree("tree_final_"+s1+"_"+t1);
    for(Bag* bag:*bags){
        std::string* parent_id;
        if(bag->get_parent()==nullptr) parent_id = new std::string("nullptr");
        else parent_id=bag->get_parent()->get_id();
        file_tree << *(bag->get_id()) << "\t" << *parent_id << std::endl;
    }
    file_tree.close();
    std::ofstream file_stats("stats_final_"+s1+"_"+t1);
    file_stats << treewidth << " " << bags->size() << " " << width << " " << height << " " << root_bag->get_number_nodes() << " " << root_bag->get_number_edges() << " " << root_bag->get_number_computed_edges() << std::endl;
    file_stats.close();
}


*/
void TreeDecomposition::write_decomposition_tot( NodeIdType source, NodeIdType target, std::string graph_name){
	// TODO: Modify graph_name location. (to write in decomp/)
    std::string s1 = std::to_string(source);
	std::string t1 = std::to_string(target);
    std::cout<<"Output filename: "<<graph_name+"_query_subgraph_"+s1+"_"+t1+".txt"<<"\n";
    std::ofstream file_root(graph_name+"_query_subgraph_"+s1+"_"+t1+".txt");
    root_bag->write_bag_to_file(file_root);
    file_root.close();

}


void TreeDecomposition::load_decomposition(std::string path_prefix, std::string graph_name){
    bag_map.clear();
    std::ostringstream index_name;
    index_name << path_prefix << "/"<<graph_name<<"_index.txt";
    std::cout<<"index_name: "<<index_name.str()<<"\n";
    std::ifstream file_index(index_name.str());
    file_index >> number_bags;
    file_index.close();
    std::ostringstream bags_name;
    bags_name << path_prefix << "/"<<graph_name<<"_bags.txt";
    std::ifstream file_bags(bags_name.str());
    for(int i=0;i<number_bags;i++){
        std::string* b_name = new std::string("default");
        Bag* bag = new Bag(b_name);
        bag->load_bag_from_file(file_bags);
        bags->push_back(bag);
        bag_map[*bag->get_id()]=bag;
        int level = bag->get_level();
        if(level_map.find(level)==level_map.end())
            level_map[level] = new std::vector<Bag*>();
        level_map[level]->push_back(bag);
    }
    file_bags.close();
    std::ostringstream root_file_name;
    root_file_name << path_prefix << "/"<<graph_name<<"_root.txt";
    std::ifstream file_root(root_file_name.str());
    std::string* root_name = new std::string("root");
    root_bag = new Bag(root_name);
    root_bag->load_bag_from_file(file_root);
    bag_map["root"]=root_bag;
    bag_map["nullptr"]=nullptr;
    file_root.close();
    std::ostringstream tree_file_name;
    tree_file_name << path_prefix << "/"<<graph_name<<"_tree.txt";
    std::ifstream file_tree(tree_file_name.str());
    std::string child_bag, parent_bag;
    height = 0;
    width = 0;
    while(file_tree>>child_bag>>parent_bag){
        if(parent_bag=="root") width++;
        if(height<bag_map[child_bag]->get_level()) height = bag_map[child_bag]->get_level();
        bag_map[child_bag]->set_parent(bag_map[parent_bag]);
        if(bag_map[parent_bag]!=nullptr)
            bag_map[parent_bag]->add_to_children(bag_map[child_bag]);
    }
    file_tree.close();
}

void TreeDecomposition::create_tree(){
    //
    unsigned long num_bags = bags->size();     
    for(unsigned long i=0;i<num_bags;i++){
        //std::cout << "linking bag " << *((*iterBags)->get_id()) << std::endl;
        Bag* child = bags->at(i);
        bool found = false;
        for(unsigned long j=i+1;j<num_bags;j++){
            Bag* parent = bags->at(j);
            if(link_bags(child,parent)){
                found = true;
                break;
            }
        }
        if(!found){
            root_bag->add_to_children(child);
            child->set_parent(root_bag);
        }
        
    }
    //computing the levels
    width = 0;
    height = 0;
    for(Bag* bag:*bags) if(bag->get_parent()==nullptr) bag->set_level(0);
    root_bag->set_level(0);
    for(Bag* bag:*root_bag->get_children()) bag->set_level(1);
    width = (int) root_bag->get_children()->size();
    bool found=false;
    std::unordered_set<Bag*> visited;
    do{
        found=false;
        for(Bag* bag:*bags){
            if((bag->get_level()!=-1)&&(visited.find(bag)==visited.end())){
                if(bag->get_level()>height) height = bag->get_level();
                for(Bag* child_bag:*bag->get_children()){
                    if(child_bag->get_level()==-1){
                        found=true;
                        child_bag->set_level(bag->get_level()+1);
                        visited.insert(bag);
                    }
                }
            }
        }
    }while(found);
    level_map[0] = new std::vector<Bag*>();
    level_map[0]->push_back(root_bag);
    for(Bag* bag:*bags){
        int level=bag->get_level();
        if(level_map.find(level)==level_map.end())
            level_map[level] = new std::vector<Bag*>();
        level_map[level]->push_back(bag);
    }
}

bool TreeDecomposition::link_bags(Bag *child, Bag *parent){
    bool linked=false;
    bool found_all = true;
    std::unordered_set<NodeIdType>* uncovered_nodes = child->get_uncovered_bag_nodes();
    for(NodeIdType node:*uncovered_nodes){
        //std::cout << "\t\t" <<" uncovered node " << node << std::endl;
        if(!parent->has_node(node)){
            found_all = false;
            break;
        }
    }
    
    if(found_all){
        //std::cout << "\tparent bag " << *(parent->get_id()) << std::endl;
        parent->add_to_children(child);
        child->set_parent(parent);
        linked = true;
    }
    return linked;
}

void TreeDecomposition::propagate_computations(){
    for(int level=height;level>0;level--){
        unsigned long num_bags = 0;
        if(level_map.find(level)!=level_map.end())
            num_bags = level_map[level]->size();
        #pragma omp parallel for
        for(unsigned long i=0;i<num_bags;i++){
            Bag* bag = level_map[level]->at(i);
            propagate_children(bag);
            if(bag->get_level()>0) bag->compute_all_paths();
        }
    }
    propagate_children(root_bag);
}

int TreeDecomposition::redo_computations(NodeIdType src, NodeIdType tgt){
    int hit_bags = 0;
    unsigned long num_bags = bags->size();
    for(unsigned long i=0;i<num_bags;i++) bags->at(i)->set_has_query_nodes(false);
    for(int level=height;level>0;level--){
        num_bags = 0;
        if(level_map.find(level)!=level_map.end())
            num_bags = level_map[level]->size();
        for(unsigned long i=0;i<num_bags;i++){
            Bag* bag = level_map[level]->at(i);
            if(bag->get_covered_bag_nodes()->find(src)!=bag->get_nodes()->end()){
                bag->set_has_query_nodes(true);
                Bag* p_bag = bag->get_parent();
                if(p_bag!=nullptr) p_bag->set_has_query_nodes(true);
            }
            if(bag->get_covered_bag_nodes()->find(tgt)!=bag->get_nodes()->end()){
                bag->set_has_query_nodes(true);
                Bag* p_bag = bag->get_parent();
                if(p_bag!=nullptr) p_bag->set_has_query_nodes(true);
            }
            if(bag->has_query_nodes()){
                hit_bags++;
                Bag* p_bag = bag->get_parent();
                if(p_bag!=nullptr) p_bag->set_has_query_nodes(true);
                propagate_children(bag);
                if(bag->get_level()>0){
                    //propagate all the uncompressed edges
                    for(auto neigh_list:*bag->get_edges()){
                        NodeIdType source = neigh_list.first;
                        for(auto neigh:*neigh_list.second){
                            NodeIdType target = neigh.first;
                            EdgeType* edge = bag->find_computed_edge(source, target);
                            if(edge==nullptr){
                                DistanceDistribution* dist_new = new DistanceDistribution();
                                dist_new->copy_from_distribution(neigh.second->second);
                                EdgeType* new_edge = new EdgeType;
                                new_edge->first = target;
                                new_edge->second = dist_new;
                                bag->add_computed_edge(source, new_edge);
                            }
                        }
                    }
                    bag->uncompress_paths(src, tgt);
                }
            }
        }
    }
    propagate_children(root_bag);
    return hit_bags;
}

void TreeDecomposition::propagate_children(Bag* bag){
    //computed edges
    bag->clear_computed_edges();
    bag->clear_computed_trees();
    for(Bag* child_bag:*bag->get_children()){
        std::string bag_orig_name = *bag->get_id();
        std::string bag_name = *child_bag->get_id();
        //std::cout << "\tpropagating " << bag_name << "->" << bag_orig_name << std::endl;
        DistanceMap* distances = child_bag->get_computed_distances();
        TreeMap* trees = child_bag->get_tree_map();
        //compute distances
        for(auto iter:*distances){
            NodeIdType first = iter.first;
            for(auto iter_dist:*iter.second){
                DistanceDistribution* distrib = new DistanceDistribution();
                NodeIdType second = iter_dist.first;
                distrib->copy_from_distribution(iter_dist.second);
                EdgeType* edge = nullptr;
                edge = bag->find_computed_edge(first, second);
                if(edge!=nullptr){
                    edge->second->combine_distribution(distrib);
                    delete distrib;
                }
                else{
                    DistanceDistribution* dist_new = new DistanceDistribution();
                    EdgeType* old_edge = bag->find_edge(first, second);
                    if(old_edge!=nullptr){
                        dist_new->copy_from_distribution(old_edge->second);
                    }
                    EdgeType* new_edge = new EdgeType;
                    new_edge->first = second;
                    new_edge->second = dist_new;
                    new_edge->second->combine_distribution(distrib);
                    delete distrib;
                    new_edge->third = false;
                    int max_distance = new_edge->second->get_max_distance();
                    if(max_distance>0) bag->add_computed_edge(first, new_edge);
                    else { delete dist_new; delete new_edge;}
                }
            }
        }
        //compute trees
        #if LIN==1
		//std::cout << "Case: LIN=1" << std::endl;
        for(auto iter:*trees){
			//std::cout << "Tree loop" << std::endl;
            NodeIdType first = iter.first;
            for(auto iter_tree:*iter.second){
                NodeIdType second = iter_tree.first;
                EvalNode* new_tree = nullptr;
                EvalNode* old_tree = bag->find_computed_tree(first, second);
                if(old_tree!=nullptr){
                    new_tree = new EvalNode(NODETYPE_INTER, NODEOPER_MIN,\
                                            first, second);
                    new_tree->set_left(iter_tree.second);
                    new_tree->set_right(old_tree);
                    bag->replace_computed_tree(first, new_tree);
                }
                else{
                    EdgeType* old_edge = bag->find_edge(first, second);
                    if(old_edge!=nullptr){
                        old_tree =  new EvalNode(NODETYPE_LEAF, NODEOPER_NONE,\
                                                 first, second);
                        new_tree = new EvalNode(NODETYPE_INTER, NODEOPER_MIN,\
                                                first, second);
                        new_tree->set_left(iter_tree.second);
                        new_tree->set_right(old_tree);
                    }
                    else
                        new_tree = iter_tree.second;
                    bag->add_computed_tree(first, new_tree);
                }
            }
        }
		std::cout << "Tree size: " << trees->size() << std::endl;
        #endif
    }
}

void TreeDecomposition::increase_degree(NodeIdType node, unsigned long degree){
    degrees[degree]->erase(node);
    node_degrees.erase(node);
    unsigned long degree_new = degree + 1;
    if(degrees.find(degree_new)==degrees.end())
        degrees[degree_new] = new std::unordered_set<NodeIdType>();
    degrees[degree_new]->insert(node);
    node_degrees[node] = degree_new;
}

bool TreeDecomposition::test_for_clique(NodeIdType node){
    bool clique = true;
    std::vector<NodeIdType> outside_nodes;
    for(auto edge:*(graph->get_neighbours(node)))
        if(covered->find(edge.first)==covered->end())
            outside_nodes.push_back(edge.first);
    for(NodeIdType node_first:outside_nodes){
        for(NodeIdType node_second:outside_nodes){
            if(node_first!=node_second){
                bool found = false;
                for(auto node_neigh:*(graph->get_neighbours(node_first))){
                    if(node_neigh.first==node_second){
                        found = true;
                        break;
                    }
                }
                if(!found){
                    clique = false;
                    break;
                }
            }
        }
    }
    
    return clique;
}
