#include "Mapping.h"

Mapping::Mapping(int origin, int destination, double probability)
{
	Mapping::origin = origin;
	Mapping::destination = destination;
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

void Mapping::setProbability(double new_probability)
{
	Mapping::probability = new_probability;
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
