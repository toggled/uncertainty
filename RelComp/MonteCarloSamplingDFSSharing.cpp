#include "MonteCarloSamplingDFSSharing.h"

/*
 * Generates the binary containing all the reachables states for k worlds
 * Sets the variable source_target_bitmap
 */
void MonteCarloSamplingDFSSharing::generateSourceTargetBitmap(VertexDescr source_node, VertexDescr target_node, Graph reference)
{
	// Previous vertex before current vertex
	std::pair<VertexDescr, boost::dynamic_bitset <>> prev_vertex;
	boost::dynamic_bitset <> vertex_bitmap;
	// Vector containing all intermediate vertex
	std::map<VertexDescr, boost::dynamic_bitset <>> temp;
	// Set Source vertex as all "1"s
	temp[source_node] = bfs_sharing_map[0][0];
	temp[source_node] = temp[source_node].set();
	// Debug 
	//std::cout << "Source Node Bitmap: " << temp[source_node] << std::endl;
	// Get bitmap of intermediate nodes to target
	std::vector<std::vector<VertexDescr>> paths = reference.getUniquePathsFromSource(source_node, target_node, *reference.getGraph());
	// Find all paths to intermediate
	for (std::vector<VertexDescr> path : paths) {
		prev_vertex = std::make_pair(source_node, temp[source_node]);
		for (VertexDescr vertex : path) {
			// Find bitmap for vertex
			vertex_bitmap = bfs_sharing_map[prev_vertex.first][vertex] & prev_vertex.second;
			// Check if vertex already in map
			if (temp.find(vertex) != temp.end()) {
				// OR the bitmap
				temp[vertex] = temp[vertex] | vertex_bitmap;
			}
			else {
				// Create a key in temp map
				temp[vertex] = vertex_bitmap;
			}
			prev_vertex = std::make_pair(vertex, temp[vertex]);
		}
	}
	source_target_bitmap = temp[target_node];
}

/*
 * Only used in creating the BFS Sharing Map
 */
MonteCarloSamplingDFSSharing::MonteCarloSamplingDFSSharing(Graph original_graph, int k)
{
	// Due to memory constraint, sample by edge and save to file
	DirectedGraph g = *original_graph.getGraph();
	size_t size = num_vertices(g);
	std::string file_name = original_graph.getGraphName() + "_BFS_Sharing_" + std::to_string(k) + ".txt";
	std::vector<char> bit_vector;
	std::pair<EdgeDescr, bool> ep;
	bool is_zero = true;
	boost::property_map<DirectedGraph, boost::edge_weight_t>::type weights = boost::get(boost::edge_weight_t(), g);
	std::cout << "Writing file...";
	// Create the file, save the size of graph
	FileIO::saveFile(file_name, size, k);

	// Source vertex
	for (size_t x = 0; x < size; x++) {
		// Target vertex
		for (size_t y = 0; y < size; y++) {
			// Sample if edge exist in kth world
			for (int i = 0; i < k; i++) {
				// Get edge of node
				ep = boost::edge(x, y, g);
				// Determine if edge exists
				if (ep.second) {
					if (original_graph.checkExist(get(weights, ep.first))) {
						is_zero = false;
						bit_vector.push_back('1');
					}
					else {
						bit_vector.push_back('0');
					}
				}
				else {
					bit_vector.push_back('0');
				}
			}
			// Write to file
			if (is_zero) {
				bit_vector.clear();
				bit_vector.push_back('0');
			}
			FileIO::appendToFile(bit_vector, file_name);
			bit_vector.clear();
			is_zero = true;
		}
	}
	std::cout << "Done";
}

/*
 * This is used in sampling
 */
MonteCarloSamplingDFSSharing::MonteCarloSamplingDFSSharing(std::string file_name)
{
	//Broke

	//bfs_sharing_data = FileIO::readBFSSharingFile(file_name);

	// Building the map from the data
	std::vector<boost::dynamic_bitset<>> target_vector;
	for (std::vector<std::vector<char>> bfs_sharing_data_target_vector : bfs_sharing_data) {
		for (std::vector<char> bfs_sharing_data_char_map : bfs_sharing_data_target_vector) {
			target_vector.push_back(boost::dynamic_bitset<>(std::string(bfs_sharing_data_char_map.begin(), bfs_sharing_data_char_map.end())));
		}
		bfs_sharing_map.push_back(target_vector);
		target_vector.clear();
	}
}

MonteCarloSamplingDFSSharing::~MonteCarloSamplingDFSSharing()
{
}

/*
 * Returns whether the target vertex is reachable for kth world
 */
bool MonteCarloSamplingDFSSharing::checkReachable(size_t k_index)
{
	return source_target_bitmap[k_index];
}
