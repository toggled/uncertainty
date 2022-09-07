#include "FileIO.h"

#if defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
std::string FileIO::output_path = "../../../../Results/";
#else
std::string FileIO::output_path = "";
#endif
std::string FileIO::dir_path = "";


FileIO::FileIO()
{
}


FileIO::~FileIO()
{
}
/**
 * Reads a file with the format 
 * 
 */
std::list<Mapping> FileIO::readFile(std::string file_path)
{
	// Assumed to be an incomplete file that needs to be generated
	return readFile(file_path, 0);
}

/**
 * Mode 0: Read file and discard its edge probability. Set all edge probabilities to 0.0
 * Mode 1: Read file in read-only
 */
std::list<Mapping> FileIO::readFile(std::string file_path, int mode)
{
	std::list<Mapping> mapping_list;
	std::fstream file;
	file.open(file_path);
	std::string content;
	std::string line;
	while (getline(file, line)) {
		std::stringstream sin(line);
		int node1, node2;
		double probability = 0.0;
		if (mode == 0) {
			while (sin >> node1 >> node2) {
				mapping_list.push_back(Mapping{ node1, node2, probability });
			}
		}
		else if (mode == 1) {
			while (sin >> node1 >> node2 >> probability) {
				mapping_list.push_back(Mapping{ node1, node2, probability });
			}
		}
	}
	return mapping_list;
}

// Returns the size and k of the graph contained
std::pair<size_t, size_t> FileIO::readBFSSharingFile(std::string file_name)
{
	std::fstream file;
	file.open(file_name);
	size_t num_vertices, k;
	std::string line;
	getline(file, line);
	std::stringstream sin(line);
	// Get number of vertices
	sin >> num_vertices;
	getline(file, line);
	//sin = std::stringstream (line);
	sin.str(line.c_str());
	// Get k
	sin >> k;
	return std::make_pair(num_vertices, k);
}
// Loads the entire BFS SHaring file into memory
std::map<VertexDescr, std::map<VertexDescr, std::string>> FileIO::readFullBFSSharingFile(std::string file_name)
{
	std::map<VertexDescr, std::map<VertexDescr, std::string>> index;
	std::ifstream file;
	file.open(file_name);
	std::string line, encoded_string;
	VertexDescr source, target;
	// SKip first 2 lines
	for (int i = 0; i < 2; i++) {
		getline(file, line);
	}
	while (getline(file, line)) {
		std::stringstream ss(line);
		ss >> source >> target >> encoded_string;
		index[source][target] = encoded_string;
	}
	return index;
}

// Read the bit vector of the edge from source to target from file
std::string FileIO::readBFSSharingFileGetEdgeBitVector(std::string file_name, size_t source, size_t target, size_t size)
{
	std::fstream file;
	file.open(file_name);
	size_t target_line = source * size + target + 2;
	std::string encoded_string;
	for (size_t line_number = 0; line_number <= target_line; line_number++)
		getline(file, encoded_string);
	return encoded_string;
}

/*
 * Reads a DBLP dataset with parallel edge probabilities
 * Condenses the parrallel probabitlies and
 * returns a list of Mapping for output
 */
std::list<Mapping> FileIO::readDBLPFile(std::string file_path)
{
	std::list<Mapping> mapping_list;
	std::fstream file;
	file.open(file_path);
	std::string line;
	int node1, node2;
	double probability;
	int not_used;
	std::vector<double> parallel_probabilities;
	while (getline(file, line)) {
		parallel_probabilities.clear();
		std::istringstream sin(line);
		sin >> node1 >> node2;
		while (sin >> probability >> not_used) {
			// Add probability into vector of doubles
			parallel_probabilities.push_back(probability);
		}
		// Condense parallel edge probabilities and add into Mapping List
		mapping_list.push_back(Mapping{ node1, node2, ProbabilityFactory::condenseEdgeProbabilities(parallel_probabilities) });
	}
	return mapping_list;
}

void FileIO::readSourceTargetFile(std::string file_path, std::vector<std::pair<VertexDescr, VertexDescr>>& source_target_pairs)
{
	// std::vector<std::pair<VertexDescr, VertexDescr>> source_target_pairs;
	std::fstream file;
	file.open(file_path);
	std::string content;
	std::string line;
	while (getline(file, line)) {
		std::stringstream sin(line);
		VertexDescr source, target;
			while (sin >> source >> target) {
				source_target_pairs.push_back(std::make_pair(source, target));
			}
	}
	file.close();
	// return std::move(source_target_pairs);
}
// Given a BFS Sharing file, generate hash file for BFS Sharing
void FileIO::generateBFSHash(std::string index_file_name, std::string hash_file_name)
{
	std::vector<std::pair<VertexDescr, size_t>> v_l;
	size_t line_number = 2;		// offset = 2
	std::string line_content;
	VertexDescr curr;
	VertexDescr prev = -1;
	std::fstream file;
	file.open(index_file_name);
	std::cout << "Generating Hash file...";
	// Skip first 2 lines
	for (int i = 0; i < 2; i++) {
		getline(file, line_content);
	}
	while (getline(file, line_content)) {
		line_number += 1;
		std::stringstream content(line_content);
		content >> curr;
		// Start of next source node
		if (curr != prev) {
			FileIO::appendToHashFile(curr, line_number, hash_file_name);
			prev = curr;
		}
	}
	std::cout << "Done" << std::endl << std::endl;
}

/*
 * Takes in csv file and get the Optimal K for that file
 */
std::string FileIO::getFilePath()
{
	std::cout << "Choose file to read" << std::endl;
	std::string path = dir_path;
	std::vector<std::string> file_path;
	int i = 1;
	int option;
	try {
		for (auto & p : std::experimental::filesystem::directory_iterator(path)) {	// std::filesystem for C++17
			file_path.push_back(p.path().string());
			std::cout << i << " - " << p.path().string().substr(path.length()) << std::endl;
			i++;
		}
		std::cout << "Option: ";
		std::cin >> option;
	}
	catch (const std::experimental::filesystem::filesystem_error& e) {
		std::cout << "Error (" << e.code() << "): " << e.what() << std::endl << std::endl;
		exit(1);
	}
	return file_path[option-1];
}

void FileIO::saveFile(std::list<Mapping> mapping_list)
{
	std::string file_name;
	std::ofstream file;

	// Generate the new probabilities
	mapping_list = generateProbabilities(mapping_list);

	// Gets user input on file name and create the file
	std::cout << "Input file name: ";
	std::cin >> file_name;
	file.open(file_name);

	// Write into file
	std::cout << "Writing file...";
	for (Mapping map : mapping_list) {
		file << map.getOrigin() << " " << map.getDistination() << " " << map.getProbability() << std::endl;
	}
	std::cout << "Done" << std::endl;
}

/**
 * Saves a text file given source-target pairs with the following format
 * [source] [target]
 */
void FileIO::saveFile(std::vector<std::pair<VertexDescr, VertexDescr>> source_target_pairs)
{
	std::string file_name;
	std::ofstream file;

	// Gets user input on file name and create the file
	std::cout << "Input file name: ";
	std::cin >> file_name;
	file.open(file_name);

	// Write into file
	std::cout << "Writing file...";
	for (std::pair<VertexDescr, VertexDescr> source_target : source_target_pairs) {
		file << source_target.first << " " << source_target.second << " " << std::endl;
	}
	std::cout << "Done" << std::endl;
}

/**
* Saves a text file given k-reliability pairs with the following format
* [k],[reliability]
*/
void FileIO::saveFile(std::vector<std::pair<int, double>> k_reliability_pairs)
{
	std::string file_name;

	// Gets user input on file name and create the file
	std::cout << std::endl << "Input file name: ";
	std::cin >> file_name;

	saveFile(k_reliability_pairs, file_name);
}

/**
* Saves a text file given k-reliability pairs with the following format
* [k],[reliability]
*/
void FileIO::saveFile(std::vector<std::pair<int, double>> k_reliability_pairs, std::string file_name)
{
	std::ofstream file;

	file.open(file_name);

	// Write into file
	std::cout << "Writing file...";
	for (std::pair<int, double> k_reliability : k_reliability_pairs) {
		file << k_reliability.first << "," << k_reliability.second << " " << std::endl;
	}
	std::cout << "Done" << std::endl;
}

// Create the file for BFS Sharing Offline Compact Sampled World
void FileIO::saveFile(std::string file_name, size_t num_vertices, size_t k) {
	std::ofstream file;
	file.open(file_name);
	file << num_vertices << std::endl;
	file << k << std::endl;
}

void FileIO::appendResultstoFile(size_t k, double reliability, long long time, size_t memory, std::string file_name)
{
	std::ofstream file;
	file.open(file_name, std::ios_base::app);
	file << k << "," << reliability << "," << time << "," << memory << std::endl;
}

// Appends the bitmap to the file
void FileIO::appendToFile(std::vector<char> bit_vector, std::string file_name) {
	std::ofstream file;
	file.open(file_name, std::ios_base::app);
	file << std::string(bit_vector.begin(), bit_vector.end()) << std::endl;
}

// Appends the bitmap to the file
void FileIO::appendToFile(VertexDescr source, VertexDescr target, std::string encoded_vector, std::string file_name)
{
	std::ofstream file;
	file.open(file_name, std::ios_base::app);
	file << source << "\t" << target << "\t" << encoded_vector << std::endl;
}
// Writes the line number of the source for the BFS Sharing file
void FileIO::appendToHashFile(VertexDescr source, size_t line_number, std::string file_name)
{
	std::ofstream file;
	file.open(file_name, std::ios_base::app);
	file << source << "\t" << line_number << std::endl;
}

std::list<Mapping> FileIO::generateProbabilities(std::list<Mapping> mapping_list)
{
	// Gets user input for method to generate probabilities for the dataset
	int option = 0;
	std::cout << std::endl << "Choose method of generating edge probabilities" << std::endl;
	std::cout << kUnifrom << " - Uniform Probability Method" << std::endl;
	std::cout << kOneByOutDegree << " - One by Out Degree Method" << std::endl;
	std::cout << kCondenseEdgeProbabilities << " - Condense edge probabilties" << std::endl;
	std::cout << kUndirectedToDirected << " - Convert to Bidirectional Graph (For undirected graphs)" << std::endl;
	std::cout << kDirectedToUndirected << " - Convert to undirectional Graph (from directed graphs)" << std::endl;
	std::cout << kAsis << " - Do nothing" << std::endl;
	std::cout << "Option: ";
	std::cin >> option;
	return generateProbabilities(mapping_list, option);
}

std::list<Mapping> FileIO::generateProbabilities(std::list<Mapping> mapping_list, int method)
{
	// Sort the mappings
	mapping_list.sort();

	switch (method) {
	case kUnifrom:
		mapping_list.unique();	// Remove duplicates
		mapping_list = ProbabilityFactory::uniformProbabilty(mapping_list);
		break;
	case kOneByOutDegree:
		mapping_list.unique();	// Remove duplicates
		mapping_list = ProbabilityFactory::oneByOutDegreeMethod(mapping_list);
		break;
	case kCondenseEdgeProbabilities:
		mapping_list = ProbabilityFactory::condenseEdgeProbabilities(mapping_list);
		break;
	case kUndirectedToDirected:
		mapping_list = ProbabilityFactory::convertToDirected(mapping_list);
		mapping_list.sort();
		break;
	case kDirectedToUndirected:
		mapping_list.unique();	// Remove duplicates
		mapping_list = ProbabilityFactory::convertToUndirected(mapping_list);
		mapping_list.sort();
		break;
	case kAsis:
		break;
	default:
		std::cout << "Error: Unknown method specified." << std::endl;
		break;
	}
	return mapping_list;
}

void FileIO::setDirPath(std::string new_dir_path)
{
	dir_path = new_dir_path;
}
