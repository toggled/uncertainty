//
//  ShortestPathTreeSampler.cpp
//  uncertain_graph
//
//  Created by Silviu Maniu on 29/4/14.
//  Copyright (c) 2014 Silviu Maniu. All rights reserved.
//

#include "ShortestPathTreeSampler.h"

void ShortestPathTreeSampler::sample_outgoing_edges(Bag *bag,\
                                                    weighted_id node_handle,\
                                                    std::unordered_map<NodeIdType,std::unordered_map<NodeIdType,int>>& sampled){
    std::unordered_set<NodeIdType> visited_outgoing;
    auto outgoing_computed = bag->get_computed_trees();
    if(outgoing_computed->find(node_handle.id)!=outgoing_computed->end()){
        for(auto edge:*(*outgoing_computed)[node_handle.id]){
            if(visited_nodes.find(edge.first)==visited_nodes.end()){
                int s_distance = edge.second->eval(graph, sampled);
                sampled_computed++;
                visited_outgoing.insert(edge.first);
                if(s_distance!=std::numeric_limits<int>::max())
                    relax(node_handle, edge.first, s_distance);
            }
        }
    }
    auto outgoing = bag->get_edges();
    if(outgoing->find(node_handle.id)!=outgoing->end()){
        for(auto edge:*outgoing->operator[](node_handle.id)){
            if((visited_nodes.find(edge.first)==visited_nodes.end())&&(visited_outgoing.find(edge.first)==visited_outgoing.end())){
                int s_distance = edge.second->second->sample_distance();
                sampled_original++;
                visited_outgoing.insert(edge.first);
                if(s_distance!=std::numeric_limits<int>::max())
                    relax(node_handle, edge.second->first, s_distance);
            }
        }
    }
}
