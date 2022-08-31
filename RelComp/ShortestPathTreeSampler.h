//
//  ShortestPathTreeSampler.h
//  uncertain_graph
//
//  Created by Silviu Maniu on 29/4/14.
//  Copyright (c) 2014 Silviu Maniu. All rights reserved.
//

#ifndef __uncertain_graph__ShortestPathTreeSampler__
#define __uncertain_graph__ShortestPathTreeSampler__

#include "ShortestPathHeapSampler.h"

class ShortestPathTreeSampler:public ShortestPathSampler{
protected:
  void sample_outgoing_edges(Bag *bag, weighted_id node_handle,\
                             std::unordered_map<NodeIdType,std::unordered_map<NodeIdType,int>>& sampled);
};

#endif /* defined(__uncertain_graph__ShortestPathTreeSampler__) */
