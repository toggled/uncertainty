#include "LazyPropagation.h"



LazyPropagation::LazyPropagation()
{
}

double LazyPropagation::lazySample(VertexDescr source, VertexDescr target, DirectedGraph g, size_t sample_size)
{
	// Custom comparator
	struct Comp
	{
		bool operator()(const std::pair<VertexDescr, size_t>& s1, const std::pair<VertexDescr, size_t>& s2)
		{
			return s1.second > s2.second;
		}
	};	


	VertexDescr v, nbr;
	std::map<VertexDescr, size_t> c_v;
	std::queue<VertexDescr> h;
	std::map<VertexDescr, std::priority_queue<std::pair<VertexDescr, size_t>, std::vector<std::pair<VertexDescr, size_t>>, Comp>> h_v;
	std::set<VertexDescr> visited;
	OutEdgeIter ei, ei_end;
	bool reached_flag = false;

	// Get edge probability map
	boost::property_map<DirectedGraph, boost::edge_weight_t>::type weights = boost::get(boost::edge_weight_t(), g);

	size_t s = 0;

	for (size_t i = 1; i <= sample_size; i++) {
		h.push(source);	// traversal frontier
		visited.insert(source);	// Mark source as visited
		while (!h.empty()) {
			v = h.front();
			h.pop();
			// if v is not initialised then
			if (c_v.count(v) == 0) {
				c_v[v] = 0;
				//h_v[v] = std::priority_queue<std::pair<VertexDescr, size_t>, std::vector<std::pair<VertexDescr, size_t>>, Comp()>();
				// iterate through all neighbours
				for (boost::tie(ei, ei_end) = boost::out_edges(v, g); ei != ei_end; ++ei) {
					nbr = boost::target(*ei, g);
					// add each neighbour to h_v
					h_v[v].push(std::make_pair(nbr, Randomiser::geometric_dist(get(weights, edge(v, nbr, g).first))));
				}
			}
			// While X(nbr) == num of times v been visited
			while (!h_v[v].empty() && h_v[v].top().second <= c_v[v]) {
				nbr = h_v[v].top().first;
				//std::cout << "Heap.front: " << h_v[v].front().first << " - " << h_v[v].front().second << std::endl;
				h_v[v].pop();
				//std::cout << "Heap after pop: " << h_v[v].front().first << " - " << h_v[v].front().second << std::endl;

				// Add nbr if nbr is not visited
				if (visited.count(nbr) == 0) {
					h.push(nbr);
					visited.insert(nbr);	// Mark nbr as visited
				}
				
				size_t rv = Randomiser::geometric_dist(get(weights, edge(v, nbr, g).first));
				
				h_v[v].push(std::make_pair(nbr, rv + c_v[v] + 1));

				// Early stopping condition
				if (nbr == target) {
					s = s + 1;
					reached_flag = true;
					break;
				}
			}
			c_v[v] = c_v[v] + 1;
			// Early stopping condition
			if (reached_flag) {
				break;
			}
		}
		// Reset all visited vertex to not visited
		//c_v.clear();
		//h_v.clear();
		visited.clear();
		h = std::queue<VertexDescr>();	// Clear worklist queue
		reached_flag = false;
	}
	return s / (double) sample_size;
}


LazyPropagation::~LazyPropagation()
{
}
