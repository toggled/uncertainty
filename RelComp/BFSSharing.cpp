#include "BFSSharing.h"

// Returns a bit vector given the raw form
boost::dynamic_bitset<> BFSSharing::getEdgeBitVector(std::vector<char> bit_vector)
{
	if (bit_vector.size() == 1 && bit_vector[0] == '0') {
		return boost::dynamic_bitset<>(k);
	}
	return boost::dynamic_bitset<>(std::string(bit_vector.begin(), bit_vector.end()));
}

// Algorithm #3
void BFSSharing::update(VertexDescr v, VertexDescr u, std::set<VertexDescr> U)
{
	vertex_bitmap[u] = vertex_bitmap[u] | (vertex_bitmap[v] & bitDecode(index[v][u]));

	VertexDescr w, x;
	std::queue<VertexDescr> q;
	std::set<VertexDescr> updated;
	OutEdgeIter e_i, e_out;
	boost::dynamic_bitset<> new_bit_vector;
	q.push(u);
	while (!q.empty())
	{
		w = q.front(); q.pop();
		for (boost::tie(e_i, e_out) = boost::out_edges(w, g); e_i != e_out; ++e_i) {
			x = boost::target(*e_i, g);
			// Check that x is explored and not updated
			if (U.count(x) != 0 && updated.count(x) == 0) {
				new_bit_vector = vertex_bitmap[x] | (vertex_bitmap[w] & bitDecode(index[w][x]));
				// Mark x as updated
				updated.insert(x);
				// Enqueue x if bit vector changed
				if (new_bit_vector != vertex_bitmap[x]) {
					vertex_bitmap[x] = new_bit_vector;
					q.push(x);
				}
			}
		}
	}
}
/* 
 * Converts binary data to ASCI
 */
std::string BFSSharing::bitEncode(std::vector<char> raw)
{
	boost::dynamic_bitset<> bit(std::string(raw.begin(), raw.end()));
	size_t padding = 7 - bit.size() % 7;
	std::bitset<7> bs;
	std::string s;
	std::string encoded = "";
	//std::cout << "Original Bit: " << bit << std::endl;
	// Special Case: All '0' bits
	if (bit.count() == 0) {
		bit.resize(7);
	}
	// Pad the bit to bytes
	if (bit.size() % 7 != 0) {
		bit.resize(bit.size() + padding);
	}

	to_string(bit, s);
	std::istringstream in(s);
	while (in >> bs) {
		encoded += char(bs.to_ulong() + 64);
	}
	//std::cout << "Bit: " << bit << std::endl;
	//std::cout << "Encoded string: " << encoded << std::endl;
	return encoded;
}

/*
 * Converts ASCI to binary
 */
boost::dynamic_bitset<> BFSSharing::bitDecode(std::string encoded)
{
	std::string decoded_raw = "";
	for (char x : encoded) {
		decoded_raw += std::bitset<7>(x - 64).to_string();
	}
	boost::dynamic_bitset<> decoded(decoded_raw);
	decoded.resize(k);
	//std::cout << "Decoded: " << decoded << std::endl;
	return decoded;
}

size_t BFSSharing::getK()
{
	return k;
}

/*
 * Only used in creating the BFS Sharing Map
 * New Version. Replaces MonteCarloSamplingDFSSharing
 */
BFSSharing::BFSSharing(Graph original_graph, int k, int i)
{
	// Due to memory constraint, sample by edge and save to file
	DirectedGraph g = *original_graph.getGraph();
	size_t size = num_vertices(g);
	std::string file_name = original_graph.getGraphName() + "/" + std::to_string(i) + "/" + std::to_string(k) + ".txt";
	//std::cout << "File Name: " << file_name << std::endl;
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
		if (boost::out_degree(x, g) != 0) {
			OutEdgeIter out_i, out_end;
			size_t y;
			// Cycle through all outgoing edges of x
			for (boost::tie(out_i, out_end) = boost::out_edges(x, g); out_i != out_end; ++out_i) {
				y = boost::target(*out_i, g);
				// Get edge of node
				ep = boost::edge(x, y, g);
				// Sample if edge exist in kth world
				for (int i = 0; i < k; i++) {
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
				FileIO::appendToFile(x, y, bitEncode(bit_vector), file_name);
				bit_vector.clear();
				is_zero = true;
			}
		}
	}
	std::cout << "Done";
}

BFSSharing::BFSSharing(std::string offline_sampled_world)
{
	// Load the offline sampled world
	compact_world_file = offline_sampled_world;
	std::pair<size_t, size_t> size_k_pair = FileIO::readBFSSharingFile(compact_world_file);
	size = size_k_pair.first;
	k = size_k_pair.second;
	index = FileIO::readFullBFSSharingFile(compact_world_file);

	std::cout << std::endl << "k = " << k << std::endl << std::endl;
}

double BFSSharing::getReliability(Graph original_graph, VertexDescr source_node, VertexDescr target_node)
{
	getFinalBitVector(original_graph, source_node, target_node);
	return vertex_bitmap[target_node].count() / (double) k;
}

boost::dynamic_bitset<> BFSSharing::getFinalBitVector(Graph original_graph, VertexDescr source_node, VertexDescr target_node)
{
	g = *original_graph.getGraph();
	VertexDescr v, ingoing_vertex, outgoing_vertex;
	OutEdgeIter ei, ei_end, out_i, out_end;
	InEdgeIter in_i, in_end;
	// Clear all pre-existing values
	explored_set.clear();
	vertex_bitmap.clear();
	// Add source vertex to explored
	explored_set.insert(source_node);
	// Init bit vectors
	vertex_bitmap[source_node] = boost::dynamic_bitset<>(k).set();	// All '1'
	vertex_bitmap[target_node] = boost::dynamic_bitset<>(k);		// All '0'
																	// Assign v
	v = source_node;
	// Init worklist
	std::queue<VertexDescr> worklist;
	for (boost::tie(ei, ei_end) = boost::out_edges(v, g); ei != ei_end; ++ei) {
		worklist.push(boost::target(*ei, g));
	}

	// BFS Sharing
	while (!worklist.empty()) {
		v = worklist.front(); worklist.pop();
		// Check if vertex is explored already, if yes then skip
		if (explored_set.count(v) != 0) {
			continue;
		}
		// Add vertex as explored
		explored_set.insert(v);
		// Init bit vector
		vertex_bitmap[v] = boost::dynamic_bitset<>(k);
		// Calculate bit vector for v
		for (boost::tie(in_i, in_end) = boost::in_edges(v, g); in_i != in_end; ++in_i) {
			ingoing_vertex = boost::source(*in_i, g);
			// Check that ingoing_vertex is explored
			if (explored_set.count(ingoing_vertex) != 0) {
				// AND the source and edge then OR all ingoing results 
				vertex_bitmap[v] = vertex_bitmap[v] | (vertex_bitmap[ingoing_vertex] & bitDecode(index[ingoing_vertex][v]));
			}
		}
		// Update all outgoing vertices of v
		for (boost::tie(out_i, out_end) = boost::out_edges(v, g); out_i != out_end; ++out_i) {
			outgoing_vertex = boost::target(*out_i, g);
			// If outgoing vertex is not explored, add to worklist and skip
			if (explored_set.count(outgoing_vertex) == 0) {
				// Only add if current bit vector is not 0, i.e. impossible
				if (vertex_bitmap[v].count() != 0) {
					worklist.push(outgoing_vertex);
				}
				continue;
			}
			// Call update
			update(v, outgoing_vertex, explored_set);
		}
	}

	/*
	// Debug
	std::cout << std::endl;
	for (std::pair<VertexDescr, boost::dynamic_bitset<>> pair : vertex_bitmap) {
	std::cout << "Bit vector of " << pair.first << ": " << pair.second << std::endl ;
	}
	std::cout << std::endl;
	/**/
	// Return final bit vector
	return vertex_bitmap[target_node];
}

BFSSharing::~BFSSharing()
{
}
// Generates hash file from the BFS Sharing file
void BFSSharing::generateHashFile(Graph original_graph, std::string index_file_name)
{
	std::string hash_file_name = original_graph.getGraphName() + "_BFS_Sharing_hash.txt";
	FileIO::generateBFSHash(index_file_name, hash_file_name);
}
