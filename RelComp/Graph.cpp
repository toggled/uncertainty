#include "Graph.h"

Graph::Graph()
{
	// Create empty graph
	directed_graph = DirectedGraph();
}

Graph::Graph(DirectedGraph graph)
{
	directed_graph = graph;
}

Graph::Graph(std::list<Mapping> mapping_list)
{
	directed_graph = DirectedGraph();
	instantiate(mapping_list);
}

Graph::Graph(std::list<Mapping> mapping_list, std::string file_path)
{
	directed_graph = DirectedGraph();
	instantiate(mapping_list);
	// Get graph name from file path
	graph_name = std::experimental::filesystem::path(file_path).stem().string();
}


Graph::~Graph()
{
}

DirectedGraph * Graph::getGraph()
{
	return &directed_graph;
}

bool Graph::checkReachable(VertexDescr source_node, VertexDescr target_node)
{
	// Visitor to throw an exeption when the target is reached
	GraphVisitor vis(vertex(target_node, directed_graph));

	// Throws exception when target is reached
	try {
		breadth_first_search(directed_graph, vertex(source_node, directed_graph), visitor(vis));
	}
	catch (int exception) {
		//std::cout << exception << ": Target vertex found." << std::endl << std::endl;
		return true;
	}
	return false;
}

void Graph::printEdgeList()
{
	std::pair<EdgeIter, EdgeIter> ei = edges(directed_graph);

	std::cout << "Number of edges = " << num_edges(directed_graph) << "\n";
	std::cout << "Edge list:\n";

	typedef boost::property_map<DirectedGraph, boost::edge_weight_t>::type WeightMap;
	WeightMap weights = get(boost::edge_weight, directed_graph);

	for (EdgeIter it = ei.first; it != ei.second; ++it)
	{
		std::cout << "Edge Probability " << *it << ": " << get(weights, *it) << std::endl;
	}

	std::cout << std::endl;
}

void Graph::printGraph()
{
	boost::dynamic_properties dp;
	dp.property("label", get(boost::edge_weight, directed_graph));
	dp.property("node_id", get(boost::vertex_index, directed_graph));
	std::ofstream dotFile("graph.dot");
	boost::write_graphviz_dp(dotFile, directed_graph, dp);
}

// Prints Mean, SD, Quartile of Edge Prob
void Graph::printStats()
{
	std::cout << "Number of nodes = " << num_vertices(directed_graph) << "\n";
	std::cout << "Number of edges = " << num_edges(directed_graph) << "\n";
	// Get all edge prob in vector
	std::pair<EdgeIter, EdgeIter> ei = edges(directed_graph);
	std::vector<double> probabilities;

	typedef boost::property_map<DirectedGraph, boost::edge_weight_t>::type WeightMap;
	WeightMap weights = get(boost::edge_weight, directed_graph);
	for (EdgeIter it = ei.first; it != ei.second; ++it)
	{
		probabilities.push_back(get(weights, *it));
	}

	// Print mean
	double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
	double mean = sum / probabilities.size();
	std::cout << "Mean = " << mean << std::endl;

	// Print SD
	std::vector<double> diff(probabilities.size());
	std::transform(probabilities.begin(), probabilities.end(), diff.begin(), [mean](double x) { return x - mean; });
	double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
	double stdev = std::sqrt(sq_sum / probabilities.size());
	std::cout << "S.D. = " << stdev << std::endl;

	// Print Quartile
	auto const Q1 = probabilities.size() / 4;
	auto const Q2 = probabilities.size() / 2;
	auto const Q3 = Q1 + Q2;

	std::nth_element(probabilities.begin(), probabilities.begin() + Q1, probabilities.end());
	std::nth_element(probabilities.begin() + Q1 + 1, probabilities.begin() + Q2, probabilities.end());
	std::nth_element(probabilities.begin() + Q2 + 1, probabilities.begin() + Q3, probabilities.end());

	std::cout << "Q1 = " << probabilities[Q1] << std::endl;
	std::cout << "Q2 = " << probabilities[Q2] << std::endl;
	std::cout << "Q3 = " << probabilities[Q3] << std::endl;
}

size_t Graph::getNumOutEdges()
{
	// Get user input on number of edges
	int vertex_index = 0;
	std::cout << "Input vertex index: ";
	std::cin >> vertex_index;

	return getNumOutEdges(vertex_index);
}

size_t Graph::getNumOutEdges(int vertex_index)
{
	VertexDescr u = vertex(vertex_index, directed_graph);

	std::cout << "Vertex #" << vertex_index << ": " << out_degree(u, directed_graph) << std::endl << std::endl;
	/*
	// Debug print messages
	std::cout << "Number of vertices: " << num_vertices(directed_graph) << std::endl;
	std::cout << "List of Edges: " << std::endl;

	typename boost::graph_traits < DirectedGraph >::out_edge_iterator ei, ei_end;
	for (boost::tie(ei, ei_end) = out_edges(u, directed_graph); ei != ei_end; ++ei) {
		auto source = boost::source(*ei, directed_graph);
		auto target = boost::target(*ei, directed_graph);
		std::cout << "There is an edge from " << source << " to " << target << std::endl;
	}
	*/
	return out_degree(u, directed_graph);
}

VertexDescr Graph::getRandomVertex()
{
	auto seed = std::random_device()();
	boost::mt19937 gen(seed);
	return boost::random_vertex(directed_graph, gen);
}

std::string Graph::getGraphName()
{
	return graph_name;
}

/*
 * Returns a vector with unique paths from source to target
 */
std::vector<std::vector<VertexDescr>> Graph::getUniquePathsFromSource(VertexDescr source, VertexDescr target, DirectedGraph const & g)
{
	std::vector<std::vector<VertexDescr>> paths;

	GraphAllPathsHelper::all_paths(source, target, g, [&](std::vector<VertexDescr> const& path) {
		paths.push_back(path);
	});
	// Debug
	/*for (std::vector<VertexDescr> path : paths) {
		for (auto v : path)
			std::cout << get(boost::vertex_index, g, v) << " ";
		std::cout << "\n";
	}*/
	return paths;
}

std::vector<VertexDescr> Graph::getBFSKDistanceAwayVertices(int threshold, VertexDescr source, DirectedGraph const & g)
{
	std::vector<double> distances(num_vertices(g));
	std::vector<boost::default_color_type> colormap(num_vertices(g));
	std::vector<VertexDescr> reachable_nodes;

	// Run BFS and record all distances from the source node
	breadth_first_search(g, source,
		visitor(make_bfs_visitor(boost::record_distances(distances.data(), boost::on_tree_edge())))
		.color_map(colormap.data())
	);

	for (auto vd : boost::make_iterator_range(vertices(g)))
		if (colormap.at(vd) == boost::default_color_type{})
			distances.at(vd) = -1;

	distances[source] = -2;

	boost::filtered_graph<DirectedGraph, boost::keep_all, std::function<bool(VertexDescr)>>
	//fg(g, {}, [&](VertexDescr vd) { return distances[vd] != -1 && distances[vd] <= threshold; });
	fg(g, {}, [&](VertexDescr vd) { return distances[vd] == threshold; });
	// Print edge list
	//std::cout << "filtered out-edges:" << std::endl;
	//std::cout << "Source Vertex: " << source << std::endl;
	
	//// Edge method
	//auto ei = boost::edges(fg);
	////typedef boost::property_map<DirectedGraph, boost::edge_weight_t>::type WeightMap;
	////WeightMap weights = get(boost::edge_weight, fg);

	//for (auto it = ei.first; it != ei.second; ++it)
	//{
	//	if (source != target(*it, g) && (distances[boost::target(*it, g)] == threshold)) {
	//		//std::cout << "Edge Probability " << *it << ": " << get(weights, *it) << std::endl;
	//		//std::cout << "Distance(" << boost::source(*it, g) << "): " << distances[boost::source(*it, g)] << std::endl;
	//		std::cout << "Distance(" << boost::target(*it, g) << "): " << distances[boost::target(*it, g)] << std::endl;
	//		reachable_nodes.push_back(target(*it, g));	// Add target vertex to vector
	//	}
	//}

	// Vertice method
	auto vi = boost::vertices(fg);
	for (auto it = vi.first; it != vi.second; ++it)
	{
		if (source != *it && (distances[*it] == threshold)) {
			//std::cout << "Distance(" << *it << "): " << distances[*it] << std::endl;
			reachable_nodes.push_back(*it);	// Add target vertex to vector
		}
	}


	// Remove duplicates
	sort(reachable_nodes.begin(), reachable_nodes.end());
	reachable_nodes.erase(unique(reachable_nodes.begin(), reachable_nodes.end()), reachable_nodes.end());

	return reachable_nodes;
}

std::vector<VertexDescr> Graph::getDFSKDistanceAwayVertices(int threshold, VertexDescr source, DirectedGraph const & g)
{
	std::vector<double> distances(num_vertices(g));
	std::vector<boost::default_color_type> colormap(num_vertices(g));
	std::vector<VertexDescr> reachable_nodes;

	auto stop_when = [&](DirectedGraph::vertex_descriptor vd, ...) { return distances.at(vd) >= threshold; };
	// Run DFS
	depth_first_visit(g, source,
		make_dfs_visitor(boost::record_distances(distances.data(), boost::on_tree_edge())),
		colormap.data(),
		stop_when
	);

	for (auto vd : boost::make_iterator_range(vertices(g)))
		if (colormap.at(vd) == boost::default_color_type{})
			distances.at(vd) = -1;

	distances[source] = -2;

	boost::filtered_graph<DirectedGraph, boost::keep_all, std::function<bool(VertexDescr)>>
		fg(g, {}, [&](VertexDescr vd) { 
			return distances[vd] != -1 && distances[vd] <= threshold;
	});

	// Print edge list
	//std::cout << "filtered out-edges:" << std::endl;
	//std::cout << "Source Vertex: " << source << std::endl;
	
	auto ei = boost::edges(fg);

	typedef boost::property_map<DirectedGraph, boost::edge_weight_t>::type WeightMap;
	WeightMap weights = get(boost::edge_weight, fg);

	for (auto it = ei.first; it != ei.second; ++it)
	{
		if (source != target(*it, g) && distances[boost::source(*it, g)] < distances[boost::target(*it, g)]) {
			std::cout << "Edge Probability " << *it << ": " << get(weights, *it) << std::endl;
			/*std::cout << "Distance(" << boost::source(*it, g) << "): " << distances[boost::source(*it, g)] << std::endl;
			std::cout << "Distance(" << boost::target(*it, g) << "): " << distances[boost::target(*it, g)] << std::endl;*/
			reachable_nodes.push_back(target(*it, g));	// Add target vertex to vector
		}
	}

	// Remove duplicates
	sort(reachable_nodes.begin(), reachable_nodes.end());
	reachable_nodes.erase(unique(reachable_nodes.begin(), reachable_nodes.end()), reachable_nodes.end());

	return reachable_nodes;
}

/**
 * Gets 100 random pairs of Source-Target vertices
 */
std::vector<std::pair<VertexDescr, VertexDescr>> Graph::getKUniquePairsSourceTargetVertices(size_t k, int threshold, DirectedGraph const & g)
{
	std::vector<std::pair<VertexDescr, VertexDescr>> source_target_pairs;
	std::vector<VertexDescr> v_list;
	VertexDescr source;
	VertexDescr prev_source = -1;
	int progress_flag = 0;
	int curr_threshold = threshold;
	int tries = 0;
	std::cout << "Generating Source-Target vertices...";
	// Hunt until k pairs are found
	while (source_target_pairs.size() < k) {
		source = getRandomVertex();
		// Prevent same vertex being picked
		if (source == prev_source) {
			continue;
		}
		prev_source = source;
		v_list = getBFSKDistanceAwayVertices(curr_threshold, source, g);
		// Check if any target vertex exist, if not, do not add
		if (v_list.size() != 0) {
			source_target_pairs.push_back(std::make_pair(source, Randomiser::getRandomVertexFromVector(v_list)));
		}

		// Remove duplicates
		sort(source_target_pairs.begin(), source_target_pairs.end());
		source_target_pairs.erase(unique(source_target_pairs.begin(), source_target_pairs.end()), source_target_pairs.end());

		// Progress marker
		if (progress_flag == 0 && source_target_pairs.size() >= k / 4) {
			std::cout << source_target_pairs.size() << "%...";
			progress_flag++;
		}

		else if (progress_flag == 1 && source_target_pairs.size() >= k / 2) {
			std::cout << source_target_pairs.size() << "%...";
			progress_flag++;
		}
		else if (progress_flag == 2 && source_target_pairs.size() == 3 * k / 4) {
			std::cout << source_target_pairs.size() << "%...";
			progress_flag++;
		}

		// Add counter
		tries++;

		// If failed to find after 100000 tries, reduce threshold
		if (tries == 100000) {
			std::cout << "Failed to find " << curr_threshold << "-hop pairs...";
			curr_threshold--;
			tries = 0;
			if (curr_threshold < 1) {
				std::cout << "Failed to find k pairs...";
			}
			else {
				std::cout << "Trying " << curr_threshold << "-hop search instead...";
			}
		}
	}
	std::cout << "Done" << std::endl << std::endl;
	/*
	// Print list of pairs
	std::cout << "Result: " << std::endl;
	for (std::pair<VertexDescr, VertexDescr> source_target : source_target_pairs) {
		std::cout << source_target.first << "-" << source_target.second << std::endl;
	}
	*/
	return source_target_pairs;
}
// Returns the subgraph of all vertices connected to source
 Graph Graph::simplify(VertexDescr source)
 {
	 VertexDescr v, w;
	 DirectedGraph dg;
	 OutEdgeIter ei, ei_end;
	 std::set<VertexDescr> explored;
	 std::queue<VertexDescr> worklist;
	 worklist.push(source);
	 
	 // Get edge probability map
	 boost::property_map<DirectedGraph, boost::edge_weight_t>::type weights = boost::get(boost::edge_weight_t(), directed_graph);

	 while (!worklist.empty())
	 {
		 v = worklist.front();
		 worklist.pop();
		 if (explored.count(v) != 0) {
			 // v has been explored before
			 continue;
		 }
		 explored.insert(v);
		 // Check that it is not a leaf node
		 if (boost::out_degree(v, directed_graph) != 0) {
			 for (boost::tie(ei, ei_end) = boost::out_edges(v, directed_graph); ei != ei_end; ++ei) {
				 w = boost::target(*ei, directed_graph);
				 worklist.push(w);
				 boost::add_edge(v, w, get(weights, edge(v, w, directed_graph).first), dg);
			 }
		 }

	 }
	 Graph simple_graph = Graph(dg);
	 return simple_graph;
 }

void Graph::add_edge(VertexDescr origin, VertexDescr destination, DirectedGraph & g)
{
	boost::add_edge(origin, destination, g);
}

/**
 * This is used to generate the reference graph containing edge probabilities
 */
void Graph::add_edge(VertexDescr origin, VertexDescr destination, double probability, DirectedGraph& g) {
	boost::add_edge(origin, destination, probability, g);
}

bool Graph::checkExist(double probability)
{
	//  If x < prob(edge), Edge exist
	if (Randomiser::getProbability() < probability) {
		return true;
	}
	return false;
}

void Graph::instantiate(std::list<Mapping> mapping_list) {
	for (Mapping map : mapping_list) {
		add_edge(map.getOrigin(), map.getDistination(), map.getProbability(), directed_graph);
	}
}