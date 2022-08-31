//
//  ShortestPathSampler.cpp
//  uncertain_graph
//
//  Created by Silviu Maniu on 26/2/13.
//  Copyright (c) 2013 Silviu Maniu. All rights reserved.
//

#include "ShortestPathHeapSampler.h"

DistanceDistribution* ShortestPathSampler::sample(Bag *bag, NodeIdType source, NodeIdType target, unsigned long samples){
    sampled_computed = 0;
    sampled_original = 0;
	reached = 0;
    DistanceDistribution* dist = new DistanceDistribution();
    //unsigned long samples = (max_diameter*max_diameter)/(2.0f*epsilon*epsilon)*log(2.0f/delta);
    //std::cout << "samples " << samples << std::endl << std::flush;
    for(unsigned long sample=1; sample<=samples; sample++){
        std::unordered_map<NodeIdType,std::unordered_map<NodeIdType,int>>\
        sampled;
        queue = new boost::heap::fibonacci_heap<weighted_id>();
        queue_nodes.clear();
        visited_nodes.clear();
        queue_nodes[source] = queue->push(weighted_id(source, 0));
        int distance = std::numeric_limits<int>::max();
        //std::cout << "Sample " << sample << " pair (" << source << "," << target << ")" << std::endl;
        while(queue->size()>0){
            auto wn = queue->top();
            //std::cout << "\t node " << wn.id << " d=" << wn.distance << " q.size=" << this->queue->size() << std::endl;
            if(wn.id==target){
				reached++;
                distance = wn.distance;
                break;
            }
            else if(wn.distance==std::numeric_limits<int>::max()){
                break;
            }
            sample_outgoing_edges(bag, wn, sampled);
            queue->pop();
            visited_nodes.insert(wn.id);
        }
        for(int d=1;d<=dist->get_max_distance();d++){
            float dist_probability = dist->get_probability(d);
            float new_probability=0.0f;
            if(d!=distance) new_probability = (dist_probability*((float)sample-1.0f))/(float)sample;
            if(new_probability!=0) dist->set_in_distribution(d, new_probability);
        }
        if(distance!=std::numeric_limits<int>::max()){
            float new_probability = (dist->get_probability(distance)*((float)sample-1.0f)+1.0f)/(float)sample;
            if(new_probability!=0) dist->set_in_distribution(distance, new_probability);
        }
        delete queue;
    }
    return dist;
};

void ShortestPathSampler::sample_outgoing_edges(Bag *bag,\
                                                weighted_id node_handle,\
                                                std::unordered_map<NodeIdType,std::unordered_map<NodeIdType,int>>& sampled){
    std::unordered_set<NodeIdType> visited_outgoing;
    auto outgoing_computed = bag->get_computed_edges();
    if(outgoing_computed->find(node_handle.id)!=outgoing_computed->end()){
        for(auto edge:*outgoing_computed->operator[](node_handle.id)){
            if(visited_nodes.find(edge.first)==visited_nodes.end()){
                int s_distance = edge.second->second->sample_distance();
                sampled_computed++;
                visited_outgoing.insert(edge.first);
                if(s_distance!=std::numeric_limits<int>::max())
                    relax(node_handle, edge.second->first, s_distance);
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

void ShortestPathSampler::relax(ShortestPathSampler::weighted_id node_handle, NodeIdType tgt, int sampled_distance){
    if(queue_nodes.find(tgt)==queue_nodes.end()){
        queue_nodes[tgt] = queue->push(weighted_id(tgt, std::numeric_limits<int>::max()));
    }
    int prev_dist = (*queue_nodes[tgt]).distance;
    int new_dist = node_handle.distance+sampled_distance;
    if(new_dist<prev_dist){
        auto handle = queue_nodes[tgt];
        weighted_id wid(tgt,new_dist);
        queue->increase(handle, wid);
    }
}
