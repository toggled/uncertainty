#include "Rss.h"
#include <algorithm>
#include <random>

//auto rng = std::default_random_engine{};

RSS::RSS()
{
}

RSS::RSS(Graph graph)
{
	reference_graph = graph;
	dg = *graph.getGraph();
	weights = boost::get(boost::edge_weight_t(), dg);
}

void RSS::setTarget(VertexDescr target)
{
	t = target;
}

double RSS::findReliability_RSS(std::unordered_set<VertexDescr> sv_map, std::queue<Edge_s_t> si, size_t n, int flag, std::vector<VertexDescr> nodes)
{
	//std::cout << "--------------------------" << std::endl;
	
	//std::cout << flag << ":" << states[flag]<<std::endl;


	for (int i=0; i<4; i++)
	{
		//std::cout << "     " << nodes[i] << std::endl;
		//std::cout << "     " << states[flag][i] << std::endl;
		if (states[flag][i] == '1')
		{
			//std::cout << "       Yes" << std::endl;
			if (nodes[i] == t)
			{
				// Return 1 if a d-path from s to t is found
				//std::cout << "find s-t path with n=" << n << std::endl;
				return 1;
			}
				
			sv_map.insert(nodes[i]);
			si = getOutgoingNeighbours_RSS(nodes[i], si);
		}

	}

	//int a;
	//std::cin >> a;

	// no edge to expand
	// find a s-t cut
	if (si.empty())
	{
		//std::cout << "find s-t cut with n=" << n << std::endl;
		return 0;
	}



	// Apply non-recursive sampling estimate if n <= threshold
	if (n <= constants::kRecursiveSamplingThreshold) {
		//std::cout << "Entering RHH..." << std::endl;
		//std::cout << "   Less than threshold. n=" << n << std::endl;
		if (n <= 0) {
			return 0;
		}
		size_t rhh = 0;
		for (size_t i = 0; i < n; i++) {				// Run n times
			rhh += samplingR_RSS(sv_map, si);	// Returns 1 or 0
												//std::cout << "      " << i << ":" << rhh << std::endl;
		}
		//std::cout << "   Result is:" << rhh<< " by " << n <<std::endl<<"   ------------------------"<<std::endl;
		return rhh / (double)n;
	}

	// the next edge to explore
	nodes.clear();
	Edge_s_t e;
	std::vector<Edge_s_t> edges;
	while (nodes.size() < 4)
	{
		if (si.empty())
		{
			// less than r edges to explore
			// call RHH
			for (auto e1 : edges)
				si.push(e1);
			return findReliability_RHH_forRSS(sv_map, si, n, false, t);
		}
		e = si.front();
		si.pop();
		if (sv_map.count(e.second) == 1)
			continue;
		nodes.push_back(e.second);
		edges.push_back(e);
	}

	double reliablity=0;
	size_t temp_n=n;
	double temp_prob = 1;
	std::vector<double> probabilities;
	for (int i = 0; i < 4; i++)
		probabilities.push_back(get(weights, boost::edge(edges[i].first, edges[i].second, dg).first));

	for (int i = 0; i < 16; i++)
	{
		if (i == 15)
		{
			reliablity += temp_prob * findReliability_RSS(sv_map, si, temp_n, 15, nodes);
			break;
		}
		double prob = 1;
		for (int j = 0; j < 4; j++)
		{
			if (states[i][j] == '1')
				prob = prob * probabilities[j];
			else
				prob = prob * (1 - probabilities[j]);
		}
		size_t this_n = std::round(n*prob);
		temp_n -= this_n;
		temp_prob -= prob;
		reliablity += prob * findReliability_RSS(sv_map, si, this_n, i, nodes);
	}

	return reliablity;
}


std::queue<Edge_s_t> RSS::getOutgoingNeighbours_RSS(VertexDescr w, std::queue<Edge_s_t> si)
{
	//std::vector<Edge_s_t> temp;

	// Check if there are any out edges from w
	if (boost::out_degree(w, dg) != 0) {
		OutEdgeIter out_i, out_end;
		for (boost::tie(out_i, out_end) = boost::out_edges(w, dg); out_i != out_end; ++out_i) {
			//temp.push_back(e);
			si.push(std::make_pair(w, boost::target(*out_i, dg)));
		}
	}

	return si;
}


double RSS::findReliability_RHH_forRSS(std::unordered_set<VertexDescr> sv_map, std::queue<Edge_s_t> si, size_t n, bool flag, VertexDescr node)
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
		si = getOutgoingNeighbours_RSS(node, si);
	}


	// no edge to expand
	// find a s-t cut
	if (si.empty())
	{
		return 0;
	}

	if (si.size() > 4)
	{
		// more than r edges to be explore
		// call RSS
		return findReliability_RSS(sv_map, si, n, 15, std::vector<VertexDescr>());
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
			rhh += samplingR_RSS(sv_map, si);	// Returns 1 or 0
												//std::cout << "      " << i << ":" << rhh << std::endl;
		}
		//std::cout << "   Result is:" << rhh<< " by " << n <<std::endl<<"   ------------------------"<<std::endl;
		return rhh / (double)n;
	}

	// the next edge to explore
	Edge_s_t e = si.front();
	si.pop();
	while (sv_map.count(e.second) == 1)
	{
		if (si.empty())
			return 0;
		e = si.front();
		si.pop();
	}

	double edge_probability = get(weights, boost::edge(e.first, e.second, dg).first);
	return edge_probability * findReliability_RHH_forRSS(sv_map, si, std::round(n*edge_probability), true, e.second) + (1 - edge_probability)*findReliability_RHH_forRSS(sv_map, si, n - std::round(n*edge_probability), false, e.second);
}


double RSS::samplingR_RSS(std::unordered_set<VertexDescr> sv_map, std::queue<Edge_s_t> si)
{
	Edge_s_t e;
	VertexDescr v, v1;
	// MC
	std::queue<VertexDescr> temp;
	while (!si.empty())
	{
		// still have edge to expand
		e = si.front();
		si.pop();

		// Check that if the end node has been reach
		if (sv_map.count(e.second) == 1)
			continue;

		// sample the existence
		if (Randomiser::getProbability() <= get(weights, boost::edge(e.first, e.second, dg).first))
		{
			//exist
			if (e.second == t)
				//if reach the target
				return 1;
			sv_map.insert(e.second);
			//si = getOutgoingNeighbours_plus(e.second, si);
			temp.push(e.second);
		}
	}

	//std::cout << "    Yes." << std::endl;
	OutEdgeIter out_i, out_end;
	while (!temp.empty())
	{
		v = temp.front();
		temp.pop();
		if (boost::out_degree(v, dg) != 0) {
			for (boost::tie(out_i, out_end) = boost::out_edges(v, dg); out_i != out_end; ++out_i) {
				v1 = boost::target(*out_i, dg);
				if (Randomiser::getProbability() <= get(weights, boost::edge(v, v1, dg).first))
				{
					//exist
					if (v1 == t)
						//if reach the target
						return 1;
					if (sv_map.count(v1) == 0)
					{
						sv_map.insert(v1);
						//si = getOutgoingNeighbours_plus(e.second, si);
						temp.push(v1);
					}

				}
			}
		}

	}
	return 0;
}

std::queue<Edge_s_t> RSS::getInitStack_RSS(VertexDescr source)
{
	std::queue<Edge_s_t> si;
	si = getOutgoingNeighbours_RSS(source, si);
	return si;
}

RSS::~RSS()
{
}
