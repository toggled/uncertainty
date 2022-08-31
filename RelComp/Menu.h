#pragma once
#include <chrono>
#include <iostream>
#include "Graph.h"
#include "FileIO.h"
#include "Constants.h"
#include "MemoryMonitor.h"

// ProbTree
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

#include "definitions.h"
#include "ProbabilisticGraph.h"
#include "TreeDecomposition.h"
#include "IndSubgraph.h"
#include "ConvergenceHelper.h"
#include "ShortestPathHeapSampler.h"
#include "ShortestPathTreeSampler.h"




#define MAX_NODES   2000000

class Menu
{
public:
	Menu();
	~Menu();
	static Graph readGraph(std::string file_path);
	static void graphMenu(Graph& graph);
	static void findKMonteCarlo(Graph& graph);
	static void createMonteCarloBFSSharingFile(Graph& graph);
	static void createBFSHashfile(Graph& graph);
	static void findKBFSSharing(Graph& graph);
	static void findkRecursiveSampling_RB(Graph& graph);
	static void findkRecursiveSampling_RHH(Graph& graph);
	static void findkRSS(Graph& graph);
	static void findkLazyPropagation(Graph& graph);
	static void findkProbTree(std::string file_name_decomp, std::string graph_name, std::string source_target_pairs);
	static void writeProbTree(std::string file_name_decomp, std::string graph_name, std::string source_target_pairs, IndSubgraph t);

	static void debugCommand(Graph& graph);
};

