#include "Menu.h"

Menu::Menu()
{
}


Menu::~Menu()
{
}



// Use FWD with w=2
void Menu::findkProbTree(std::string file_name_decomp, std::string graph_name, std::string source_target_file)
{
	//std::cout << "Input dir of Index: " << std::endl << std::flush;
	// Get dir of index
	//std::string file_name_decomp("./");
	TreeDecomposition decomp(file_name_decomp, graph_name);
	Bag* root_bag = decomp.get_root();

	std::cout << std::endl << "Reading Source-Target file..." << std::endl;
	std::vector<std::pair<VertexDescr, VertexDescr>> source_target_pairs = FileIO::readSourceTargetFile(source_target_file);

	// Get graph name
	std::cout << "Graph name: ";
	//std::string graph_name="test_graph.txt";


	int samples = 0;
	NodeIdType source, target;
	int pairs = 0;
	ShortestPathSampler sampler;
	double reliability = 0.0;
	std::vector<double> reliability_k, reliability_j;
	double avg_r = 0.0;
	double diff_sq_sum = 0.0;
	bool write_flag = true;

	// Start up Memory Monitor daemon thread
	MemoryMonitor mm = MemoryMonitor();
	std::thread t1(&MemoryMonitor::updatePeakMemory, std::ref(mm));
	t1.detach();

	while (samples < constants::kMaximumRound) {
		// Step up k
		samples += constants::kKStepUp;
		std::cout << std::endl << "k = " << samples << std::endl;

		// Reset var
		reliability_k.clear();

		for (size_t i = 0; i < source_target_pairs.size(); i++) {
			source = source_target_pairs[i].first;
			target = source_target_pairs[i].second;

			// Reset var
			reliability_j.clear();
			diff_sq_sum = 0.0;
			write_flag = true;

			for (int j = 0; j < constants::kRepeatForVariance; j++) {
				std::cout << j << "th iteration" << std::endl;

				// Start time
				auto start = std::chrono::high_resolution_clock::time_point::max();
				auto finish = std::chrono::high_resolution_clock::time_point::max();
				start = std::chrono::high_resolution_clock::now();
				mm.startMonitoring();

				NodeIdType src, tgt;
				src = tgt = -1;
				bool good_tree = true;
				if (!root_bag->has_node(source)) src = source;
				if (!root_bag->has_node(target)) tgt = target;
				int hit_bags = 0;
				try {
					if ((src != -1) || (tgt != -1)) hit_bags = decomp.redo_computations(src, tgt);
				}
				catch (int e) {
					std::cerr << "exception " << e << "caught in " << src << "->" << \
						tgt << " - skipping" << std::endl;
					good_tree = false;
				}
				std::cout << "s-t pairs: " << source << "\t" << target << std::endl << std::flush;
				DistanceDistribution *dist = nullptr;
				if (good_tree) {
					try {
						dist = sampler.sample(root_bag, source, target, samples);
					}
					catch (int e) {
						std::cerr << "exception " << e << "caught in " << src << "->"\
							<< tgt << " - skipping" << std::endl;
						dist = new DistanceDistribution();
					}
				}
				else
					dist = new DistanceDistribution();
				reliability = sampler.get_reached() / (double)samples;

				// Stop time
				finish = std::chrono::high_resolution_clock::now();
				//auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish - start).count();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();

				std::cout << "Reliability Estimator, R^ (" << source << ", " << target << ") = " << reliability << std::endl;
				std::cout << "Execution time = " << duration << " ms" << std::endl << std::endl;

				if (write_flag) {
					// Write interim results into csv
					FileIO::appendResultstoFile(samples, reliability, duration, mm.getPeakMemory(), graph_name + "_ProbTree_k_" + std::to_string(i) + ".csv");
					write_flag = false;
				}
				
				// Add r to vector
				reliability_j.push_back(reliability);

				delete dist;
			}

			// Variance calculation
			avg_r = ConvergenceHelper::getAvgReliability(reliability_j);
			for (int j = 0; j < constants::kRepeatForVariance; j++) {
				auto difference_sq = pow(reliability_j[j] - avg_r, 2);
				diff_sq_sum += difference_sq;
			}
			FileIO::appendResultstoFile(samples, diff_sq_sum / (constants::kRepeatForVariance - 1), 0, i, graph_name + "_ProbTree_variance.csv");
		}
	}
	mm.stopMonitoring();
}


void Menu::writeProbTree(std::string file_name_decomp, std::string graph_name, std::string source_target_file, 	IndSubgraph t)
{
	//std::cout << "Input dir of Index: " << std::endl << std::flush;
	// Get dir of index
	//std::string file_name_decomp("./");
	TreeDecomposition decomp(file_name_decomp, graph_name);
	//IndSubgraph decomp(t);
	Bag* root_bag = decomp.get_root();

	std::cout << std::endl << "Reading Source-Target file..." << std::endl;
	std::vector<std::pair<VertexDescr, VertexDescr>> source_target_pairs = FileIO::readSourceTargetFile(source_target_file);

	int samples = 0;
	NodeIdType source, target;

	// Start up Memory Monitor daemon thread
	MemoryMonitor mm = MemoryMonitor();
	std::thread t1(&MemoryMonitor::updatePeakMemory, std::ref(mm));
	t1.detach();

	for (size_t i = 0; i < source_target_pairs.size(); i++) {
		source = source_target_pairs[i].first;
		target = source_target_pairs[i].second;

		std::cout << "s-t pairs: " << source << "\t" << target << std::endl << std::flush;

		// Start time
		auto start = std::chrono::high_resolution_clock::time_point::max();
		auto finish = std::chrono::high_resolution_clock::time_point::max();
		start = std::chrono::high_resolution_clock::now();
		mm.startMonitoring();
		// here
		NodeIdType src, tgt;
		src = tgt = -1;
		if (!root_bag->has_node(source)) src = source;
		if (!root_bag->has_node(target)) tgt = target;
		int hit_bags = 0;
		try {
			if ((src != -1) || (tgt != -1)) hit_bags = decomp.redo_computations(src, tgt);
			decomp.write_decomposition_tot(source, target, graph_name);
		}
		catch (int e) {
			std::cerr << "exception " << e << "caught in " << src << "->" << \
				tgt << " - skipping" << std::endl;
		}

		// Stop time
		finish = std::chrono::high_resolution_clock::now();
		//auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish - start).count();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
		std::cout<<"duration:"<<duration<<"ms   memory:  "<<mm.getPeakMemory()<<std::endl;
	}

	mm.stopMonitoring();
}


/*
 * Contains debug commands
 */
