//
//  ShortestPathSampler.h
//  uncertain_graph
//
//  Created by Silviu Maniu on 26/2/13.
//  Copyright (c) 2013 Silviu Maniu. All rights reserved.
//

#ifndef __uncertain_graph__ShortestPathSamplerHeap__
#define __uncertain_graph__ShortestPathSamplerHeap__

#include <iostream>

#include "definitions.h"
#include "Bag.h"
#include "DistanceDistribution.h"
#include "Graph.h"

#include <limits.h>

#include <boost/heap/fibonacci_heap.hpp>
#include <boost/config.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/adjacency_list.hpp>

class ShortestPathSampler{
protected:
  struct weighted_id{
    NodeIdType id;
    int distance;
    bool operator<(const weighted_id &a) const {
      return distance>a.distance?true:(distance<a.distance?false:id>a.id);
    }
    weighted_id(NodeIdType i, double s) : id(i), distance(s) {}
  };
  boost::heap::fibonacci_heap<weighted_id>* queue;
  std::unordered_map<NodeIdType, boost::heap::fibonacci_heap<weighted_id>::handle_type> queue_nodes;
  std::unordered_set<NodeIdType> visited_nodes;
  int sampled_computed;
  int sampled_original;
  int reached;
  ProbabilisticGraph* graph;
  
public:
  DistanceDistribution* sample(Bag *bag, NodeIdType source, NodeIdType target, unsigned long samples=1000);
  void set_graph(ProbabilisticGraph* pgraph) {graph = pgraph;};
  int get_sampled_computed() {return sampled_computed;};
  int get_sampled_original() {return sampled_original;};
  int get_reached() { return reached; };
protected:
  virtual void sample_outgoing_edges(Bag *bag, weighted_id node_handle,\
                             std::unordered_map<NodeIdType,std::unordered_map<NodeIdType,int>>& sampled);
  void relax(weighted_id node_handle, NodeIdType tgt,\
             int sampled_distance);
};

#endif /* defined(__uncertain_graph__ShortestPathSampler__) */
