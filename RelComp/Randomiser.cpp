#include "Randomiser.h"



Randomiser::Randomiser()
{
}


Randomiser::~Randomiser()
{
}

/**
 * Takes in parameter bins and generates an uniform distribution
 */
void Randomiser::uniform_dist(std::list<Mapping>& mapping_list, int bins, std::vector<double> probability)
{
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(1, bins);
	for (Mapping& map : mapping_list) {
		map.setProbability(probability[distribution(generator)-1]);
	}
}

// Generates a geometric r.v
 size_t Randomiser::geometric_dist(double edge_probabiltiy)
 {
	 std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
	 // P=1 case
	 if (edge_probabiltiy == 1.0) {
		 return 0;
	 }
	 std::geometric_distribution<size_t> distribution(edge_probabiltiy);
	 return distribution(generator);
 }

/**
 * Generates a probabiity from 0 to 1
 */
double Randomiser::getProbability()
{
	std::mt19937 rng;
	rng.seed(std::random_device()());
	std::uniform_real_distribution<> dist(0, std::nextafter(1, std::numeric_limits<double>::max())); // distribution in range [0, 1]
	return dist(rng);
}

/**
 * Returns a Vertex id from a given vector of vertices
 */
 VertexDescr Randomiser::getRandomVertexFromVector(std::vector<VertexDescr> v_list)
 {
	 // Checks if size of vector == 1
	 if (v_list.size() == 1) {
		 return v_list[0];
	 }
	 std::default_random_engine generator;
	 std::uniform_int_distribution<size_t> distribution(1, v_list.size());
	 return v_list[distribution(generator)-1];
 }

