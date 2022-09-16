#pragma once
class Mapping
{
private:
	int origin, destination;
	double weight;
	double probability;
public:
	Mapping(int origin, int destination, double probability);
	Mapping(int origin, int destination, double weight, double probability);
	~Mapping();
	int getOrigin();
	int getDistination();
	double getProbability();
	double getWeight();
	void setProbability(double new_probability);
	void setWeight(double new_weight);
	bool operator<(const Mapping& rhs);
	bool operator== (const Mapping rhs);
};
