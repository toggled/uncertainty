#pragma once
#include <boost/range/adaptor/reversed.hpp>
#include <list>
#include <iostream>
#include <vector>
#include "Graph.h"
#include "Mapping.h"
#include "Randomiser.h"
class ProbabilityFactory
{
public:
	ProbabilityFactory();
	~ProbabilityFactory();
	static std::list<Mapping> uniformProbabilty(std::list<Mapping> mapping_list);
	static std::list<Mapping> oneByOutDegreeMethod(std::list<Mapping> mapping_list);
	static std::list<Mapping> condenseEdgeProbabilities(std::list<Mapping> mapping_list);
	static double condenseEdgeProbabilities(std::vector<double> probabilities);
	static std::list<Mapping> convertToDirected(const std::list<Mapping>& mapping_list);
	static std::list<Mapping> convertToUndirected(std::list<Mapping>& mapping_list);
};

