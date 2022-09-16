//
//  ProbabilisticGraph.cpp
//  UncertainProbabilisticGraph
//
//  Created by Silviu Maniu on 1/2/13.
//  Copyright (c) 2013 Silviu Maniu. All rights reserved.
//

#include "ProbabilisticGraph.h"

#include <fstream>

ProbabilisticGraph::ProbabilisticGraph(std::ifstream& input_stream, bool isWeighted){
    undirected_distribution = new DistanceDistribution();
    undirected_distribution->add_to_distribution(1, 1.0f);
    number_edges = 0;
    NodeIdType id_first, id_second;
    if (isWeighted){
        double weight;
        double probability;
        while(input_stream >> id_first >> id_second >> weight>>probability){
            //*input_stream >> value;
            DistanceDistribution* original = new DistanceDistribution();
            //input_stream >> *original;
            
            // input_stream >> probability;
            original->add_to_distribution(1, probability);
            //std::cout << "DistanceDistribution: " << *original << std::endl;
            // add_edge(id_first, id_second, original);
            add_weighted_edge(id_first, id_second, weight, original);
            add_undirected_edge(id_first, id_second, undirected_distribution);
            add_undirected_edge(id_second, id_first, undirected_distribution);
            number_edges++;
        }while(!(input_stream.eof()));
        std::cout<<"num of edges: "<<number_edges<<"\n";
    }
    else{
        while(input_stream >> id_first >> id_second){
                //*input_stream >> value;
                DistanceDistribution* original = new DistanceDistribution();
                //input_stream >> *original;
                double probability;
                input_stream >> probability;
                original->add_to_distribution(1, probability);
                //std::cout << "DistanceDistribution: " << *original << std::endl;
                add_edge(id_first, id_second, original);
                add_undirected_edge(id_first, id_second, undirected_distribution);
                add_undirected_edge(id_second, id_first, undirected_distribution);
                number_edges++;
            }while(!(input_stream.eof()));
    }
    
};

void ProbabilisticGraph::write_graph(std::ofstream &output_stream){
    for(auto edge_list:outgoing){
        NodeIdType node_first = edge_list.first;
        for(auto edge:*edge_list.second){
            NodeIdType node_second = edge.first;
            output_stream << node_first << "\t" << node_second << std::endl;
            output_stream << *edge.second->second;
        }
    }
}

bool ProbabilisticGraph::is_node(NodeIdType node_id){
    EdgeMap::const_iterator find_iter = undirected.find(node_id);
    if(find_iter!=undirected.end()) return true;
    else return false;
};

std::unordered_map<NodeIdType, EdgeType*>* ProbabilisticGraph::get_outgoing_nodes(NodeIdType node_id){
    std::unordered_map<NodeIdType, EdgeType*>* rtval = nullptr;
    if(outgoing.find(node_id)!=outgoing.end()) rtval = outgoing[node_id];
    return rtval;
};

std::unordered_map<NodeIdType, EdgeType*>* ProbabilisticGraph::get_incoming_nodes(NodeIdType node_id){
    std::unordered_map<NodeIdType, EdgeType*>* rtval = nullptr;
    if(incoming.find(node_id)!=incoming.end()) rtval = incoming[node_id];
    return rtval;
};

std::unordered_map<NodeIdType, EdgeType*>* ProbabilisticGraph::get_neighbours(NodeIdType node_id){
    return undirected[node_id];
};

void ProbabilisticGraph::add_edge(NodeIdType key_first, NodeIdType key_second, DistanceDistribution* value){
    EdgeType* out= new EdgeType;
    out->first = key_second;
    out->second = value;
    out->third = false;
    EdgeType* in = new EdgeType;
    in->first = key_first;
    in->second = value;
    in->third = false;
    add_node(key_first);
    add_node(key_second);
    (*outgoing[key_first])[out->first]=out;
    (*incoming[key_second])[in->first]=in;
};
void  ProbabilisticGraph::add_weighted_edge(NodeIdType key_first, NodeIdType key_second, double weight, DistanceDistribution* value){
    EdgeType* out= new EdgeType;
    out->first = key_second;
    out->second = value;
    out->third = false;
    out->weight = weight;
    EdgeType* in = new EdgeType;
    in->first = key_first;
    in->second = value;
    in->third = false;
    in->weight = weight;
    add_node(key_first);
    add_node(key_second);
    (*outgoing[key_first])[out->first]=out;
    (*incoming[key_second])[in->first]=in;
}


EdgeType* ProbabilisticGraph::get_edge(NodeIdType key_first, NodeIdType key_second){
    EdgeType* edge = nullptr;
    if((outgoing.find(key_first)!=outgoing.end())&&((*outgoing[key_first]).find(key_second)!=(*outgoing[key_first]).end()))
       edge = (*outgoing[key_first])[key_second];
    return edge;
}

void ProbabilisticGraph::remove_edge(NodeIdType key_first, NodeIdType key_second){
    delete (*outgoing[key_first])[key_second]->second;
    delete (*outgoing[key_first])[key_second];
    delete (*incoming[key_second])[key_first];
    (*outgoing[key_first]).erase(key_second);
    (*incoming[key_second]).erase(key_first);
    number_edges--;
    if(outgoing[key_first]->size()==0) outgoing.erase(key_first);
    if(incoming[key_second]->size()==0) incoming.erase(key_second);
}

void ProbabilisticGraph::add_undirected_edge(NodeIdType key_first, NodeIdType key_second, DistanceDistribution* value, double weight){
    EdgeType* out = new EdgeType;
    out->first = key_second;
    out->second = value;
    out->third = false;
    out->weight = weight;
    (*undirected[key_first])[out->first]=out;
};

void ProbabilisticGraph::add_node(NodeIdType key){
    if(!is_node(key)){
        outgoing[key] = new std::unordered_map<NodeIdType, EdgeType*>();
        incoming[key] = new std::unordered_map<NodeIdType, EdgeType*>();
        undirected[key] = new std::unordered_map<NodeIdType, EdgeType*>();
    }
};

std::vector<NodeIdType> ProbabilisticGraph::get_node_vector(){
    std::vector<NodeIdType> nodes;
    for(auto kv:outgoing) nodes.push_back(kv.first);
    return nodes;
};

void ProbabilisticGraph::remove_node(NodeIdType node){
    undirected.erase(node);
};

void ProbabilisticGraph::remove_edge(NodeIdType node, EdgeType *edge){
    outgoing[node]->erase(edge->first);
}



