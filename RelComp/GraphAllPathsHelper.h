#pragma once
#include "Common.h"
class GraphAllPathsHelper
{
public:
	GraphAllPathsHelper();
	~GraphAllPathsHelper();
	template <typename Report> static void all_paths_helper(VertexDescr from, VertexDescr to, DirectedGraph const & g, std::vector<VertexDescr>& path, Report const& callback);
	template <typename Report> static void all_paths(VertexDescr from, VertexDescr to, DirectedGraph const & g, Report const& callback);
};

template<typename Report>
inline void GraphAllPathsHelper::all_paths_helper(VertexDescr from, VertexDescr to, DirectedGraph const & g, std::vector<VertexDescr>& path, Report const & callback)
{
	path.push_back(from);

	if (from == to) {
		callback(path);
	}
	else {
		for (auto out : make_iterator_range(out_edges(from, g))) {
			auto v = target(out, g);
			if (path.end() == std::find(path.begin(), path.end(), v)) {
				all_paths_helper(v, to, g, path, callback);
			}
		}
	}

	path.pop_back();
}

template<typename Report>
inline void GraphAllPathsHelper::all_paths(VertexDescr from, VertexDescr to, DirectedGraph const & g, Report const & callback)
{
	std::vector<VertexDescr> state;
	all_paths_helper(from, to, g, state, callback);
}
