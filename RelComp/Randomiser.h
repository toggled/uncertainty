#pragma once
#include <chrono>
#include <list>
#include <random>
#include "Common.h"
#include "Mapping.h"

class Randomiser
{
public:
	Randomiser();
	~Randomiser();
	static void uniform_dist(std::list<Mapping>& mapping_list, int bins, std::vector<double> probability);
	static size_t geometric_dist(double edge_probabiltiy);
	static double getProbability();
	static VertexDescr getRandomVertexFromVector(std::vector<VertexDescr> v_list);
};

