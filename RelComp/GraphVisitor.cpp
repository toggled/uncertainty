#include "GraphVisitor.h"

void GraphVisitor::initialize_vertex(const VertexDescr & s, const DirectedGraph & g) const
{
}

void GraphVisitor::discover_vertex(const VertexDescr & s, const DirectedGraph & g) const
{
}

void GraphVisitor::examine_vertex(const VertexDescr & s, const DirectedGraph & g) const
{
}

void GraphVisitor::examine_edge(const EdgeDescr & e, const DirectedGraph & g) const
{
}

void GraphVisitor::edge_relaxed(const EdgeDescr & e, const DirectedGraph & g) const
{
}

void GraphVisitor::edge_not_relaxed(const EdgeDescr & e, const DirectedGraph & g) const
{
}

void GraphVisitor::finish_vertex(const VertexDescr & s, const DirectedGraph & g) const
{
	if (destination_vertex_m == s)
		throw(2);
}
