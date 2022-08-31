#pragma once
#include "Common.h"

// Visitor that throw an exception when finishing the destination vertex
class GraphVisitor : public boost::default_bfs_visitor {
protected:
	VertexDescr destination_vertex_m;
public:
	GraphVisitor(VertexDescr destination_vertex_l)
		: destination_vertex_m(destination_vertex_l) {};

	void initialize_vertex(const VertexDescr &s, const DirectedGraph &g) const;
	void discover_vertex(const VertexDescr &s, const DirectedGraph &g) const;
	void examine_vertex(const VertexDescr &s, const DirectedGraph &g) const;
	void examine_edge(const EdgeDescr &e, const DirectedGraph &g) const;
	void edge_relaxed(const EdgeDescr &e, const DirectedGraph &g) const;
	void edge_not_relaxed(const EdgeDescr &e, const DirectedGraph &g) const;
	void finish_vertex(const VertexDescr &s, const DirectedGraph &g) const;
};