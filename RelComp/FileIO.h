#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#if defined(__has_include)
#if __has_include(<filesystem>)
# include <filesystem>
#endif
#if __has_include(<experimental/filesystem>)
# include <experimental/filesystem>
#endif
#endif

#include "Mapping.h"
#include "ProbabilityFactory.h"

enum ProbabilityMethod { 
	kUnifrom, 
	kOneByOutDegree,
	kCondenseEdgeProbabilities,
	kUndirectedToDirected,
	kDirectedToUndirected,
	kAsis
};

class FileIO
{
private:
	static std::string dir_path;
	static std::string output_path;
public:
	FileIO();
	~FileIO();
	static std::list<Mapping> readFile(std::string file_path);
	static std::list<Mapping> readFile(std::string file_path, int mode);
	static std::pair<size_t, size_t> readBFSSharingFile(std::string file_name);
	static std::map<VertexDescr, std::map<VertexDescr, std::string>> readFullBFSSharingFile(std::string file_name);
	static std::string readBFSSharingFileGetEdgeBitVector(std::string file_name, size_t source, size_t target, size_t size);
	static std::list<Mapping> readDBLPFile(std::string file_path);
	static std::vector<std::pair<VertexDescr, VertexDescr>> readSourceTargetFile(std::string file_path);
	static std::string getFilePath();
	static void generateBFSHash(std::string index_file_name, std::string hash_file_name);
	static void saveFile(std::list<Mapping> mapping_list);
	static void saveFile(std::vector<std::pair<VertexDescr, VertexDescr>> source_target_pairs);
	static void saveFile(std::vector<std::pair<int, double>> k_reliability_pairs);
	static void saveFile(std::vector<std::pair<int, double>> k_reliability_pairs, std::string file_name);
	static void saveFile(std::string file_name, size_t num_vertices, size_t k);
	static void appendResultstoFile(size_t k, double reliability, long long time, size_t memory, std::string file_name);
	static void appendToFile(std::vector<char> bit_vector, std::string file_name);
	static void appendToFile(VertexDescr source, VertexDescr target, std::string encoded_vector, std::string file_name);
	static void appendToHashFile(VertexDescr source, size_t line_number, std::string file_name);
	static std::list<Mapping> generateProbabilities(std::list<Mapping> mapping_list);
	static std::list<Mapping> generateProbabilities(std::list<Mapping> mapping_list, int method);
	static void setDirPath(std::string new_dir_path);
};

