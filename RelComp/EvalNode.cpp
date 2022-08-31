//
//  EvalNode.cpp
//  uncertain_graph
//
//  Created by Silviu Maniu on 25/4/14.
//  Copyright (c) 2014 Silviu Maniu. All rights reserved.
//

#include "EvalNode.h"

#include <iostream>
#include <sstream>

int EvalNode::eval(ProbabilisticGraph *graph,std::unordered_map<NodeIdType,\
                      std::unordered_map<NodeIdType,int>>& sampled){
    int value = std::numeric_limits<int>::max();
    if(type==NODETYPE_INTER){
        int value_right = right->eval(graph, sampled);
        int value_left = std::numeric_limits<int>::max();
        if(oper==NODEOPER_MIN){
            value_left = left->eval(graph, sampled);
            value = value_right<value_left?value_right:value_left;
        }
        else{
            if(value_right!=std::numeric_limits<int>::max()){
                value_left = left->eval(graph, sampled);
                if(value_left!=std::numeric_limits<int>::max())
                    value = value_right+value_left;
            }
        }
    }
    else{
        bool is_sampled = false;
        if(sampled.find(src)!=sampled.end())
            if(sampled[src].find(tgt)!=sampled[src].end()){
                is_sampled = true;
                value = sampled[src][tgt];
            }
        if(!is_sampled){
            EdgeType* edge = graph->get_edge(src, tgt);
            if(edge!=nullptr){
                value = edge->second->sample_distance();
                sampled[src][tgt] = value;
            }
        }
    }
    return value;
}

void EvalNode::print(int& nid, std::vector<std::string> &nodes,\
                     std::vector<std::string> &edges){
    std::ostringstream node;
    nid++;
    unsigned long cur_id = nid;
    node << nid << "\t" << type << "\t" << oper << "\t" << src << "\t"\
    << tgt << std::endl;
    nodes.push_back(node.str());
    if(left!=nullptr){
        std::ostringstream edge;
        edge << cur_id << "\t" << nid+1 << "\t0" << std::endl;
        edges.push_back(edge.str());
        left->print(nid, nodes, edges);
    }
    if(right!=nullptr){
        std::ostringstream edge;
        edge << cur_id << "\t" << nid+1 << "\t1" << std::endl;
        edges.push_back(edge.str());
        right->print(nid, nodes, edges);
    }
}
