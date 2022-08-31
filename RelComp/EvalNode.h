//
//  EvalNode.h
//  uncertain_graph
//
//  Created by Silviu Maniu on 25/4/14.
//  Copyright (c) 2014 Silviu Maniu. All rights reserved.
//

#ifndef __uncertain_graph__EvalNode__
#define __uncertain_graph__EvalNode__

#include "definitions.h"
#include "ProbabilisticGraph.h"

#include <unordered_map>

enum EvalNodeType{NODETYPE_LEAF, NODETYPE_INTER};
enum EvalNodeOper{NODEOPER_NONE, NODEOPER_MIN, NODEOPER_SUM};

class EvalNode{
  unsigned long id;
  EvalNodeType type;
  EvalNodeOper oper;
  NodeIdType src;
  NodeIdType tgt;
  EvalNode* left;
  EvalNode* right;
public:
  EvalNode(EvalNodeType node_type, EvalNodeOper node_oper,\
           NodeIdType edge_src,\
           NodeIdType edge_tgt) : type(node_type),\
           oper(node_oper), src(edge_src), tgt(edge_tgt),\
           left(nullptr), right(nullptr) {};
  void set_left(EvalNode* node_left){ left = node_left;};
  void set_right(EvalNode* node_right){ right = node_right; };
  EvalNode* get_left() {return left;};
  EvalNode* get_right() {return right;};
  NodeIdType get_src() {return src;};
  NodeIdType get_tgt() {return tgt;};
  int eval(ProbabilisticGraph* graph, std::unordered_map<NodeIdType,\
              std::unordered_map<NodeIdType, int>>& sampled);
  void print(int& nid, std::vector<std::string>& nodes,\
             std::vector<std::string>& edges);
};


#endif /* defined(__uncertain_graph__EvalNode__) */
