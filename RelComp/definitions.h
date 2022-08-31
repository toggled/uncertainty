//
//  definitions.h
//  UncertainGraph
//
//  Created by Silviu Maniu on 1/2/13.
//  Copyright (c) 2013 Silviu Maniu. All rights reserved.
//

#ifndef UncertainGraph_definitions_h
#define UncertainGraph_definitions_h

#include <unordered_map>
#include <unordered_set>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "DistanceDistribution.h"

static const float delta = 0.05f;
static const float epsilon = 0.05f;
static const float max_diameter = 3.0f;

typedef unsigned long long timestamp_t;

//Graph representation type definitions
typedef int NodeIdType;
typedef float ValueType;
typedef struct{
    NodeIdType first;
    DistanceDistribution* second;
    bool third;
} EdgeType;
typedef struct{
    
} EvalEdgeTree;
//typedef std::pair<NodeIdType, ValueType> EdgeType;
typedef std::unordered_set<NodeIdType> NodeSet;
typedef std::unordered_map<NodeIdType, std::unordered_map<NodeIdType,EdgeType*>*> EdgeMap;
enum SPQRNodeType {SNode, PNode, RNode, TreeDecNode};
const std::string nodetypes[] = {"serial", "parallel", "rigid", "decomposition"};
const std::string computations[] = {"sampling", "exact", "sampling", "all-pairs"};

static timestamp_t get_timestamp ()
{
    return  0;
}

#endif
