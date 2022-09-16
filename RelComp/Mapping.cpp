#include "Mapping.h"

Mapping::Mapping(int origin, int destination, double probability)
{
	Mapping::origin = origin;
	Mapping::destination = destination;
	Mapping::probability = probability;
}
Mapping::Mapping(int origin, int destination, double weight, double probability)
{
	Mapping::origin = origin;
	Mapping::destination = destination;
	Mapping::weight = weight;
	Mapping::probability = probability;
}

Mapping::~Mapping() {
}

int Mapping::getOrigin()
{
	return Mapping::origin;
}

int Mapping::getDistination()
{
	return Mapping::destination;
}

double Mapping::getProbability()
{
	return Mapping::probability;
}

double Mapping::getWeight()
{
	return Mapping::weight;
}

void Mapping::setProbability(double new_probability)
{
	Mapping::probability = new_probability;
}

void Mapping::setWeight(double new_weight)
{
	Mapping::weight = new_weight;
}

bool Mapping::operator<(const Mapping & rhs)
{
	if (origin == rhs.origin)
		return (destination < rhs.destination);
	else
		return (origin < rhs.origin);
}

bool Mapping::operator==(const Mapping rhs)
{
	return ((origin == rhs.origin) && (destination == rhs.destination));
}
