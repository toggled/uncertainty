#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include "Menu.h"
#include "FileIO.h"
using namespace std;


int main(int argc, char* argv[]) {

	if(argc < 4){
		cout << "usage " << argv[0] << endl
				<< " -i input file" << endl
				<< " -d directory" << endl
				<< " -s source-target file" << endl
				<< " -m method [1= TreeDecomposition, 2= IndSubgraph, 3=Reachability]"<< endl;
		return 0;
	}

	std::string graph_name;
	std::string file_name_decomp;
	std::string source_target_pairs;
	int c, metric=0;

	while ((c = getopt(argc, argv, "i:d:s:m:")) != -1) {
		switch(c) {
		case 'i':
			graph_name=strdup(optarg);
			break;
		case 'd':
			file_name_decomp=strdup(optarg); //set path to folder containing bags etc.
			break;
		case 's':
			source_target_pairs=strdup(optarg);
			break;
		case 'm':
			metric = atoi(optarg);
			break;
		}
	}

	//Load graph
	std::ifstream input_stream(graph_name.c_str());

	ProbabilisticGraph g(input_stream);
	if(metric==1){
		cout << "Computing bags using method TreeDecomposition..."<< endl;
		TreeDecomposition t(&g);
		t.decompose_graph(2);
		t.write_decomposition();
		cout << "Done!"<< endl;
	}
	if(metric==2){
		cout << "Computing bags using method IndSubgraph..."<< endl;
		IndSubgraph t(&g);
		t.decompose_graph(2, graph_name);
		cout << "Done!"<< endl;
	}
	if(metric==3){
		cout << "Computing reachability..."<< endl;
		Menu::findkProbTree(file_name_decomp, graph_name, source_target_pairs);
		cout << "Done!"<< endl;
	}
	if(metric==4){
		cout << "Writing root bag..."<< endl;
		//Menu::writeProbTree(file_name_decomp, graph_name, source_target_pairs);
		cout << "Done!"<< endl;
	}
	if(metric==5){
		cout << "Computing and Writing subgraph for source target list..."<< endl;
		IndSubgraph t(&g);
		t.decompose_graph(2, graph_name);
		Menu::writeProbTree(file_name_decomp, graph_name, source_target_pairs, t);
		cout << "**Done!"<< endl;
	}
	return 0;
}

