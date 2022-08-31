#pragma once
#include <cmath>
#include <vector>

class ConvergenceHelper
{
public:
	ConvergenceHelper();
	~ConvergenceHelper();
	static double getAvgReliability(std::vector<double> r);
	static double getStdReliabilityMC(double r, std::vector<int> i_st);
};

