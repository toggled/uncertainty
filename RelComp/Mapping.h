#pragma once
class Mapping
{
private:
	int origin, destination;
	double probability;
public:
	Mapping(int origin, int destination, double probability);
	~Mapping();
	int getOrigin();
	int getDistination();
	double getProbability();
	void setProbability(double new_probability);
	bool operator<(const Mapping& rhs);
	bool operator== (const Mapping rhs);
};
