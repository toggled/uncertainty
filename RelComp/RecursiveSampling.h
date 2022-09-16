#pragma once
#include <stack>
#include <queue>
#include <unordered_set>
#include <functional>
#include "Common.h"
#include "Graph.h"
#include "MonteCarloBFS.h"
#include "Randomiser.h"

template <typename T>
inline void hash_combine(std::size_t &seed, const T &val) {
	seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// auxiliary generic functions to create a hash value using a seed
template <typename T> inline void hash_val(std::size_t &seed, const T &val) {
	hash_combine(seed, val);
}
template <typename T, typename... Types>
inline void hash_val(std::size_t &seed, const T &val, const Types &... args) {
	hash_combine(seed, val);
	hash_val(seed, args...);
}

template <typename... Types>
inline std::size_t hash_val(const Types &... args) {
	std::size_t seed = 0;
	hash_val(seed, args...);
	return seed;
}

struct pair_hash {
	template <class T1, class T2>
	std::size_t operator()(const std::pair<T1, T2> &p) const {
		return hash_val(p.first, p.second);
	}
};

class RecursiveSampling
{
private:
	boost::property_map<DirectedGraph, boost::edge_weight_t>::type weights;
	Graph reference_graph;
	DirectedGraph dg;
	VertexDescr t;
	std::queue<Edge_s_t> getOutgoingNeighbours(VertexDescr v);
public:
	RecursiveSampling();
	RecursiveSampling(Graph graph);
	//double findReliability_RB(Graph& g, std::set<Edge_s_t> e1, std::set<Edge_s_t> e2, std::queue<VertexDescr> sv, std::queue<std::queue<Edge_s_t>> si, size_t n);
	double findReliability_RB(Graph& g, std::unordered_set<Edge_s_t,pair_hash> e1, std::unordered_set<Edge_s_t, pair_hash> e2, std::queue<VertexDescr> sv, std::queue<std::queue<Edge_s_t>> si, size_t n);
	//double findReliability_RHH(std::set<Edge_s_t> e1, std::set<Edge_s_t> e2, std::queue<VertexDescr> sv, std::queue<std::queue<Edge_s_t>> si, size_t n);
	double findReliability_RHH(std::unordered_set<Edge_s_t, pair_hash> e1, std::unordered_set<Edge_s_t, pair_hash> e2, std::queue<VertexDescr> sv, std::unordered_set<VertexDescr> sv_map, std::queue<std::queue<Edge_s_t>> si, size_t n);
	std::pair<std::queue<VertexDescr>, std::queue<std::queue<Edge_s_t>>> getInitStack(VertexDescr source);
	//std::pair<Edge_s_t, bool> nextEdge(std::set<Edge_s_t> e1, std::set<Edge_s_t> e2, std::queue<VertexDescr> sv, std::queue<std::queue<Edge_s_t>> si);
	std::pair<Edge_s_t, bool> nextEdge(std::unordered_set<Edge_s_t, pair_hash> e1, std::unordered_set<Edge_s_t, pair_hash> e2, std::queue<VertexDescr> sv, std::queue<std::queue<Edge_s_t>> si);
	void setTarget(VertexDescr target);
	//double samplingR(std::set<Edge_s_t> e1, std::set<Edge_s_t> e2, VertexDescr v, double pr, double q);
	double samplingR(std::unordered_set<Edge_s_t, pair_hash> e1, std::unordered_set<Edge_s_t, pair_hash> e2, std::queue<VertexDescr> sv, std::unordered_set<VertexDescr> sv_map, std::queue<std::queue<Edge_s_t>> si);
	double samplingR_1(std::unordered_set<Edge_s_t, pair_hash> e1, std::unordered_set<Edge_s_t, pair_hash> e2, std::queue<VertexDescr> sv, std::unordered_set<VertexDescr> sv_map, std::queue<std::queue<Edge_s_t>> si);
	std::queue<Edge_s_t> getOutgoingNeighbours1(VertexDescr v, std::unordered_set<VertexDescr> sv);
	double findReliability_RHH_plus(std::unordered_set<VertexDescr> sv_map, std::queue<Edge_s_t> si, size_t n, bool flag, VertexDescr node);
	std::queue<Edge_s_t> getOutgoingNeighbours_plus(VertexDescr w, std::queue<Edge_s_t> si);
	double samplingR_plus(std::unordered_set<VertexDescr> sv_map, std::queue<Edge_s_t> si);
	std::queue<Edge_s_t> getInitStack_plus(VertexDescr source);
	~RecursiveSampling();
};

