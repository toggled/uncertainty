#pragma once
#include "Common.h"
class GraphReachableHelper
{
private: 
	std::set<VertexDescr> explored;
	std::set<VertexDescr> reachable;
public:
	GraphReachableHelper();
	std::set<VertexDescr> getReachableSet(DirectedGraph g, VertexDescr source, VertexDescr target);
	bool checkReachable(DirectedGraph g, VertexDescr v, VertexDescr target);
	~GraphReachableHelper();
};

