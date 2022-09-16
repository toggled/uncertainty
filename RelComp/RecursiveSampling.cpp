#include "RecursiveSampling.h"
#include <algorithm>
#include <random>

auto rng = std::default_random_engine{};


// Gets all out edges of a vertex v
std::queue<Edge_s_t> RecursiveSampling::getOutgoingNeighbours(VertexDescr v)
{
	std::queue<Edge_s_t> i;
	// Check if there are any out edges from w
	if (boost::out_degree(v, dg) != 0) {
		OutEdgeIter out_i, out_end;
		for (boost::tie(out_i, out_end) = boost::out_edges(v, dg); out_i != out_end; ++out_i) {
			i.push(std::make_pair(v, boost::target(*out_i, dg)));
		}
	}
	else {
		// Empty queue
	}
	return i;
}


std::queue<Edge_s_t> RecursiveSampling::getOutgoingNeighbours1(VertexDescr v, std::unordered_set<VertexDescr> sv)
{
	std::queue<Edge_s_t> i;
	// Check if there are any out edges from w
	if (boost::out_degree(v, dg) != 0) {
		OutEdgeIter out_i, out_end;
		for (boost::tie(out_i, out_end) = boost::out_edges(v, dg); out_i != out_end; ++out_i) {
			if (sv.count(boost::target(*out_i, dg))==0)
				i.push(std::make_pair(v, boost::target(*out_i, dg)));
		}
	}
	else {
		// Empty queue
	}
	return i;
}

RecursiveSampling::RecursiveSampling()
{
}

RecursiveSampling::RecursiveSampling(Graph graph)
{
	reference_graph = graph;
	dg = *graph.getGraph();
	weights = boost::get(boost::edge_weight_t(), dg);
}

double RecursiveSampling::findReliability_RB(Graph& g, std::unordered_set<Edge_s_t, pair_hash> e1, std::unordered_set<Edge_s_t, pair_hash> e2, std::queue<VertexDescr> sv, std::queue<std::queue<Edge_s_t>> si, size_t n)
{
	std::pair<Edge_s_t, bool> edge_bool_pair;
	Edge_s_t e;
	//std::cout << "-> " << sv.top() << std::endl;	// Debug
	// Apply non-recursive sampling estimate if n <= threshold
	if (n <= constants::kRecursiveSamplingThreshold) {		
		// Use Monte Carlo
		int reliability = 0;
		if (n == 0) {
			n = 1;
		}
		MonteCarloBFS g_j = MonteCarloBFS(*g.getGraph());
		//std::cout << "n = " << n << std::endl;
		for (size_t j = 0; j < n; j++) {
			// Compute reliability of Gj
			// And update reliability
			reliability += g_j.run(sv.front(), t);
		}
		return reliability / (double) n;
	}
	// Return 1 if E1 contains a d-path from s to t
	if (sv.front() == t) {
		//std::cout << "Leaf node found: Return 1" << std::endl;	// Debug
		return 1;
	}
	edge_bool_pair = nextEdge(e1, e2, sv, si);
	// Return 0 if E2 contains a d-cut from s to t
	if (edge_bool_pair.second == false) {
		//std::cout << "Leaf node found: Return 0" << std::endl;	// Debug
		return 0;
	}
	e = edge_bool_pair.first;
	// E1 U e
	std::unordered_set<Edge_s_t, pair_hash> e1_u_e(e1);
	e1_u_e.insert(e);
	// E2 U e
	std::unordered_set<Edge_s_t, pair_hash> e2_u_e(e2);
	e2_u_e.insert(e);
	Graph g_e2 = g;
	remove_edge(edge(e.first, e.second, *g_e2.getGraph()).first, *g_e2.getGraph());
	// Sv push w
	VertexDescr w = e.second;
	std::queue<VertexDescr> sv_w(sv);
	sv_w.push(w);
	// Si push all outgoing edges
	std::queue<std::queue<Edge_s_t>> si_1(si);
	si_1.push(getOutgoingNeighbours(w));
	// Edge Probability
	double edge_probability = get(weights, boost::edge(sv.front(), w, dg).first);

	return edge_probability * findReliability_RB(g, e1_u_e, e2, sv_w, si_1, std::round(n*edge_probability)) + (1 - edge_probability)*findReliability_RB(g_e2, e1, e2_u_e, sv, si, n - std::round(n*edge_probability));
}
double RecursiveSampling::findReliability_RHH(std::unordered_set<Edge_s_t, pair_hash> e1, std::unordered_set<Edge_s_t, pair_hash> e2, std::queue<VertexDescr> sv, std::unordered_set<VertexDescr> sv_map, std::queue<std::queue<Edge_s_t>> si, size_t n)
{
	//std::cout <<std::endl<< "+++++++++++++++++++++++++++++++++++++" << std::endl;
	//std::cout << "1:" <<n << std::endl;
	//std::cout << "Current first node: " << sv.front() << std::endl;
	//std::cout << "Current last node: " << sv.back() << std::endl;
	//auto stopping = 0;
	//std::cin >> stopping;

	// Return 1 if E1 contains a d-path from s to t
	if (sv.back() == t) {
		//std::cout << "Leaf node found: Return 1" << std::endl;	// Debug
		return 1;
	}

	std::pair<Edge_s_t, bool> edge_bool_pair;
	Edge_s_t e;

	// Apply non-recursive sampling estimate if n <= threshold
	if (n <= constants::kRecursiveSamplingThreshold) {
		//std::cout << "Entering RHH..." << std::endl;
		//std::cout << "Less than threshold" << std::endl;
		if (n == 0) {
			return 0;
		}
		size_t rhh = 0;
		for (size_t i = 0; i < n; i++) {				// Run n times
			rhh += samplingR_1(e1, e2, sv, sv_map,si);	// Returns 1 or 0
		}
		//std::cout << "Result is:" << rhh<< "by" << n <<std::endl;
		return rhh / (double)n;
	}
	
	edge_bool_pair = nextEdge(e1, e2, sv, si);
	e = edge_bool_pair.first;

	//while (sv_map.count(e.second) == 1)
	//{
		//e1.insert(e);
		//edge_bool_pair = nextEdge(e1, e2, sv, si);
		//e = edge_bool_pair.first;
	//}
	// Return 0 if E2 contains a d-cut from s to t
	if (edge_bool_pair.second == false) {
		//std::cout << "Leaf node found: Return 0" << std::endl;	// Debug
		return 0;
	}

	

	// E1 U e
	std::unordered_set<Edge_s_t, pair_hash> e1_u_e(e1);
	e1_u_e.insert(e);
	// E2 U e
	std::unordered_set<Edge_s_t, pair_hash> e2_u_e(e2);
	e2_u_e.insert(e);
	
	VertexDescr w = e.second;

	
	// Edge Probability
	//std::cout << "Next edge: " <<e.first<<", "<<w<<std::endl;
	double edge_probability = get(weights, boost::edge(e.first, w, dg).first);
	//std::cout << "Prob:" << edge_probability << " and n:"<<n << std::endl;
	//std::cout << "--------------------------" << std::endl;

	if (sv_map.count(w) == 1)
	{
		// the ending node has been reached
		return edge_probability * findReliability_RHH(e1_u_e, e2, sv, sv_map, si, std::round(n*edge_probability)) + (1 - edge_probability)*findReliability_RHH(e1, e2_u_e, sv, sv_map, si, n - std::round(n*edge_probability));
	}

	// Sv push w
	std::queue<VertexDescr> sv_w(sv);
	sv_w.push(w);

	std::unordered_set<VertexDescr> sv_map_w(sv_map);
	sv_map_w.insert(w);

	// Si push all outgoing edges
	std::queue<std::queue<Edge_s_t>> si_1(si);
	si_1.push(getOutgoingNeighbours1(w,sv_map_w));
	

	return edge_probability * findReliability_RHH(e1_u_e, e2, sv_w, sv_map_w, si_1, std::round(n*edge_probability)) + (1 - edge_probability)*findReliability_RHH(e1, e2_u_e, sv, sv_map, si, n - std::round(n*edge_probability));
}
// Returns the initial stack of sv and si given a source vertex
// queue for BFS-based, stack for DFS based
std::pair<std::queue<VertexDescr>, std::queue<std::queue<Edge_s_t>>> RecursiveSampling::getInitStack(VertexDescr source)
{
	std::queue<VertexDescr> sv;
	std::queue<std::queue<Edge_s_t>> si;
	sv.push(source);
	si.push(getOutgoingNeighbours(source));
	return std::make_pair(sv,si);
}
// return a pair of edge and bool. bool == false if no edges found
std::pair<Edge_s_t, bool> RecursiveSampling::nextEdge(std::unordered_set<Edge_s_t, pair_hash> e1, std::unordered_set<Edge_s_t, pair_hash> e2, std::queue<VertexDescr> sv, std::queue<std::queue<Edge_s_t>> si)
{
	//VertexDescr v, v_i;
	//std::queue<Edge_s_t> neighbours;
	Edge_s_t e;
	//bool pop_flag = true;	// Only pop si and sv if true
	//int counting = 0;
	while (!sv.empty()) {
		//pop_flag = true;
		//v = sv.top();
		while (!si.front().empty()) {
			//v_i = si.top().front().second;
			e = si.front().front();
			si.front().pop();
			// Check that e does not belong in e2
			if (e2.count(e) == 0) {
				// Check that e belongs in e1
				if (e1.count(e) == 0) {
					//sv.push(v_i);
					//si.push(getOutgoingNeighbours(v_i));
					//pop_flag = false;
					//break;
					//continue;
				//}
				//else {
					return std::make_pair(e, true);
				}
			}
		}
		sv.pop();
		si.pop();
		//if (pop_flag || si.top().empty()) {
			//sv.pop();
			//si.pop();
			//counting++;
			//if (counting > 50)
			//	break;
		//}
	}
	// No edge found
	return std::make_pair(std::make_pair(-1, -1), false);
}

void RecursiveSampling::setTarget(VertexDescr target)
{
	t = target;
}

// RHH
double RecursiveSampling::samplingR(std::unordered_set<Edge_s_t, pair_hash> e1, std::unordered_set<Edge_s_t, pair_hash> e2, std::queue<VertexDescr> sv, std::unordered_set<VertexDescr> sv_map, std::queue<std::queue<Edge_s_t>> si)
{
	//std::cout << "Enter sampleR" << std::endl;
	std::pair<Edge_s_t, bool> edge_bool_pair;
	Edge_s_t e;
	//std::stack<VertexDescr> sv;
	//std::stack<std::queue<Edge_s_t>> si;
	double edge_probability;
	VertexDescr w;
	// Return if E1 contains a d-path from s to t
	if (sv.back() == t) {
		//return pr / q;
		return 1;
	}
	// Find next edge
	//sv.push(v);
	//si.push(getOutgoingNeighbours(v));
	edge_bool_pair = nextEdge(e1, e2, sv, si);
	// Return 0 if E2 contains a d-cut from s to t
	if (edge_bool_pair.second == false) {
		//std::cout << "Leaf node found: Return 0" << std::endl;	// Debug
		return 0;
	}
	e = edge_bool_pair.first;
	w = e.second;
	
	// Get edge probability
	std::cout << "Next edge (sample R): " << e.first << ", " << w << std::endl;
	edge_probability = get(weights, boost::edge(e.first, w, dg).first);
	std::cout << "Prob:" << edge_probability << std::endl;
	std::cout << "--------------------------" << std::endl;
	
	// Check if exists
	if (Randomiser::getProbability() <= edge_probability) {
		std::cout << "000000" << std::endl;
		if (w == t)
			return 1;
		// E1 U e
		std::cout << "111111" << std::endl;
		std::unordered_set<Edge_s_t,pair_hash> e1_u_e(e1);
		e1_u_e.insert(e);

		if (sv_map.count(w) == 1)
			return samplingR(e1_u_e, e2, sv, sv_map, si);

		std::cout << "22222222" << std::endl;
		std::queue<VertexDescr> sv_w(sv);
		sv_w.push(w);
		std::unordered_set<VertexDescr> sv_map_w(sv_map);
		sv_map_w.insert(w);
		// Si push all outgoing edges
		std::queue<std::queue<Edge_s_t>> si_1(si);
		si_1.push(getOutgoingNeighbours1(w,sv_map_w));
		std::cout << "333333333" << std::endl;
		
		//return samplingR(e1_u_e, e2, w, edge_probability*pr, edge_probability*q);
		return samplingR(e1_u_e, e2, sv_w, sv_map_w, si_1);
	}
	else {
		// E2 U e
		std::unordered_set<Edge_s_t, pair_hash> e2_u_e(e2);
		e2_u_e.insert(e);
		std::cout << "444444444" << std::endl;
		//return samplingR(e1, e2_u_e, v, (1 - edge_probability)*pr, (1 - edge_probability)*q);
		return samplingR(e1, e2_u_e, sv, sv_map, si);
	}
}



double RecursiveSampling::samplingR_1(std::unordered_set<Edge_s_t, pair_hash> e1, std::unordered_set<Edge_s_t, pair_hash> e2, std::queue<VertexDescr> sv, std::unordered_set<VertexDescr> sv_map, std::queue<std::queue<Edge_s_t>> si)
{
	Edge_s_t e;
	//std::cout << "Inner MC." << std::endl;
	while (!sv.empty()) {
		while (!si.front().empty()) {
			e = si.front().front();
			si.front().pop();
			// Check that e does not belong in e2
			if (e2.count(e) == 0) {
				// Check that e belongs in e1
				if (e1.count(e) == 0) {
		            // a new edge to explore
					//std::cout << "a new edge to explore: " << e.first << "  " << e.second << std::endl;
					//auto marking=0;
					//std::cin >> marking;
					if (sv_map.count(e.second) == 0)
					{
						double edge_probability = get(weights, boost::edge(e.first, e.second, dg).first);
						// sample the existence
						if (Randomiser::getProbability() <= edge_probability)
						{
							//exist
							if (e.second == t)
								//if reach the target
								return 1;
							e1.insert(e);
							sv.push(e.second);
							sv_map.insert(e.second);
							si.push(getOutgoingNeighbours1(e.second, sv_map));
						}
						else
						{
							e2.insert(e);
						}
					}
					else
					{
						e1.insert(e);
					}
				}
			}
		}
		sv.pop();
		si.pop();
	}
	return 0;
}


double RecursiveSampling::findReliability_RHH_plus(std::unordered_set<VertexDescr> sv_map, std::queue<Edge_s_t> si, size_t n, bool flag, VertexDescr node)
{
	//std::cout << "--------------------------" << std::endl;
	//std::cout << "Entering RHH. Current node is: " << node << std::endl;
	// Return 1 if E1 contains a d-path from s to t
	if (flag) {
		if (node == t)
		{
			//std::cout << "      path found with n="<< n << std::endl;
			return 1;
		}
		
		sv_map.insert(node);
		si = getOutgoingNeighbours_plus(node, si);
	}


	// no edge to expand
	// find a s-t cut
	if (si.empty())
	{
		return 0;
	}

	

	// Apply non-recursive sampling estimate if n <= threshold
	if (n <= constants::kRecursiveSamplingThreshold) {
		//std::cout << "Entering RHH..." << std::endl;
		//std::cout << "   Less than threshold. n=" << n << std::endl;
		if (n == 0) {
			return 0;
		}
		size_t rhh = 0;
		for (size_t i = 0; i < n; i++) {				// Run n times
			rhh += samplingR_plus(sv_map, si);	// Returns 1 or 0
			//std::cout << "      " << i << ":" << rhh << std::endl;
		}
		//std::cout << "   Result is:" << rhh<< " by " << n <<std::endl<<"   ------------------------"<<std::endl;
		return rhh / (double)n;
	}

	// the next edge to explore
	Edge_s_t e=si.front();
	si.pop();
	while (sv_map.count(e.second) == 1)
	{
		if (si.empty())
			return 0;
		e = si.front();
		si.pop();
	}

	double edge_probability = get(weights, boost::edge(e.first, e.second, dg).first);
	return edge_probability * findReliability_RHH_plus(sv_map, si, std::round(n*edge_probability), true, e.second) + (1 - edge_probability)*findReliability_RHH_plus(sv_map, si, n - std::round(n*edge_probability), false, e.second);
}


std::queue<Edge_s_t> RecursiveSampling::getOutgoingNeighbours_plus(VertexDescr w, std::queue<Edge_s_t> si)
{
	//std::vector<Edge_s_t> temp;

	// Check if there are any out edges from w
	if (boost::out_degree(w, dg) != 0) {
		OutEdgeIter out_i, out_end;
		for (boost::tie(out_i, out_end) = boost::out_edges(w, dg); out_i != out_end; ++out_i) {
			//temp.push_back(e);
			si.push(std::make_pair(w, boost::target(*out_i, dg)));
		}
		// randomize
		//std::shuffle(std::begin(temp), std::end(temp), rng);
		//for (auto it : temp)
		//{
			//si.push(it);
		//}
	}

	return si;
}


double RecursiveSampling::samplingR_plus(std::unordered_set<VertexDescr> sv_map, std::queue<Edge_s_t> si)
{
	Edge_s_t e;
	VertexDescr v,v1;
	// MC
	//std::queue<VertexDescr> temp;
	while (!si.empty())
	{
		// still have edge to expand
		e = si.front();
		si.pop();

		// Check that if the end node has been reach
		if (sv_map.count(e.second) == 1)
			continue;

		// sample the existence
		if (Randomiser::getProbability() < get(weights, boost::edge(e.first, e.second, dg).first))
		{
			//exist
			if (e.second == t)
				//if reach the target
				return 1;
			sv_map.insert(e.second);
			si = getOutgoingNeighbours_plus(e.second, si);
			//temp.push(e.second);
		}
	}
	return 0;

	//std::cout << "    Yes." << std::endl;
	//OutEdgeIter out_i, out_end;
	// 	while (!temp.empty())
	// 	{
	// 		v = temp.front();
	// 		temp.pop();
	// 		if (boost::out_degree(v, dg) != 0) {
	// 			for (boost::tie(out_i, out_end) = boost::out_edges(v, dg); out_i != out_end; ++out_i) {
	// 				v1 = boost::target(*out_i, dg);
	// 				if (Randomiser::getProbability() <= get(weights, boost::edge(v, v1, dg).first))
	// 				{
	// 					//exist
	// 					if (v1 == t)
	// 						//if reach the target
	// 						return 1;
	// 					if (sv_map.count(v1) == 0)
	// 					{
	// 						sv_map.insert(v1);
	// 						//si = getOutgoingNeighbours_plus(e.second, si);
	// 						temp.push(v1);
	// 					}

	// 				}
	// 			}
	// 		}

	// 	}
	
}

std::queue<Edge_s_t> RecursiveSampling::getInitStack_plus(VertexDescr source)
{
	std::queue<Edge_s_t> si;
	si = getOutgoingNeighbours_plus(source, si);
	return si;
}

RecursiveSampling::~RecursiveSampling()
{
}
