#include "ProbabilityFactory.h"



ProbabilityFactory::ProbabilityFactory()
{
}


ProbabilityFactory::~ProbabilityFactory()
{
}

std::list<Mapping> ProbabilityFactory::uniformProbabilty(std::list<Mapping> mapping_list)
{
	int size;
	double probability;
	std::cout << "Number of probabilities to input: ";
	std::cin >> size;
	std::vector<double> probabilities;
	for (int i = 0; i < size; i++) {
		std::cout << "Probability #" << i + 1 << ": ";
		std::cin >> probability;
		probabilities.push_back(probability);
	}
	std::cout << "Generating edge probabilities...";
	Randomiser::uniform_dist(mapping_list, size, probabilities);
	std::cout << "Done" << std::endl;
	return mapping_list;
}

std::list<Mapping> ProbabilityFactory::oneByOutDegreeMethod(std::list<Mapping> mapping_list)
{
	// Generate a temp DirectedGraph
	Graph temp(mapping_list);

	// Generate probabilities
	for (Mapping& map : mapping_list) {
		map.setProbability(1.0 / temp.getNumOutEdges(map.getOrigin()));
	}

	return mapping_list;
}

// Assumes list is sorted
std::list<Mapping> ProbabilityFactory::condenseEdgeProbabilities(std::list<Mapping> mapping_list)
{
	std::vector<Mapping> v_mapping;
	std::list<Mapping> processed_mapping_list;
	size_t start = 0;
	size_t end = 0;
	double probability_product = 1.0;

	// Convert list<> to vector<>
	v_mapping.reserve(mapping_list.size());
	std::copy(std::begin(mapping_list), std::end(mapping_list), std::back_inserter(v_mapping));

	// Checks if there are duplicates, if there is, condenses edge probabilities using the formulua 1-(1-p1)(1-p2)...
	for (size_t i = 0; i < v_mapping.size() - 1; i++) {
		// Checks if new set
		if (end == 0) {
			start = i;
			end = i;
		}

		// Checks if the current same as the next
		if (v_mapping[i].getOrigin() == v_mapping[i + 1].getOrigin()
			&& v_mapping[i].getDistination() == v_mapping[i + 1].getDistination()) {
			end = i + 1;
		}

		else {
			// Generate condensed edge probability
			for (size_t map_id = start; map_id <= end; map_id++) {
				probability_product = probability_product * (1 - v_mapping[map_id].getProbability());
			}
			// Add the mapping into the list
			processed_mapping_list.push_back(Mapping(v_mapping[start].getOrigin(), 
												v_mapping[start].getDistination(),
												1 - probability_product));
			// Reset temp variables
			start = 0;
			end = 0;
			probability_product = 1;
		}
	}
	// Account for last element
	if (end == 0) {
		start = v_mapping.size() - 1;
		end = v_mapping.size() - 1;
	}
	for (size_t map_id = start; map_id <= end; map_id++) {
		probability_product = probability_product * (1 - v_mapping[map_id].getProbability());
	}
	// Add the mapping into the list
	processed_mapping_list.push_back(Mapping(v_mapping[start].getOrigin(),
		v_mapping[start].getDistination(),
		1 - probability_product));
	return processed_mapping_list;
}

/* 
 * Returns a condensed edge probability given all parallel edge probabilities
 */
double ProbabilityFactory::condenseEdgeProbabilities(std::vector<double> probabilities)
{
	// Formula: 1-(1-p1)(1-p2)...
	double product = 1.0;
	for (double p : probabilities) {
		product *= 1 - p;
	}
	return 1 - product;
}

// Converts an undirected graph to a directed graph
std::list<Mapping> ProbabilityFactory::convertToDirected(const std::list<Mapping>& mapping_list)
{
	std::list<Mapping> bidirected_mapping_list;
	for (Mapping map : mapping_list) {
		bidirected_mapping_list.push_back(Mapping(map.getOrigin(), map.getDistination(), map.getProbability()));
		if (map.getOrigin() != map.getDistination()) {
			bidirected_mapping_list.push_back(Mapping(map.getDistination(), map.getOrigin(), map.getProbability()));
		}
	}
	return bidirected_mapping_list;
}

std::list<Mapping> ProbabilityFactory::convertToUndirected(std::list<Mapping>& mapping_list)
{
	for (auto m : boost::adaptors::reverse(mapping_list)) {
		if (m.getOrigin() == m.getDistination()) {
			continue;	// skip
		}
		if (m.getOrigin() == 1788 && m.getDistination() == 1786) {
			std::cout << m.getOrigin() << ", " << m.getDistination() << std::endl;
		}
		for (auto m2 : mapping_list) {
			if (m2.getOrigin() == 1786 && m2.getDistination() == 1788 && m.getOrigin() == 1788 && m.getDistination() == 1786) {
				std::cout << "m2 !!!" << std::endl;
				std::cout << m.getOrigin() << ", " << m.getDistination() << std::endl;
			}
			if (m.getDistination() == m2.getOrigin() && m.getOrigin() == m2.getDistination()) {
				if (m.getOrigin() == 1788) {
					std::cout << m.getOrigin() << ", " << m.getDistination() << std::endl;
					std::cout << m2.getOrigin() << ", " << m2.getDistination() << std::endl;
					std::cout << "Removed" << std::endl;
				}
				mapping_list.remove(m);
			}
		}
	}
	return mapping_list;
}

