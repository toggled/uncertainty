#include "ConvergenceHelper.h"

ConvergenceHelper::ConvergenceHelper()
{
}


ConvergenceHelper::~ConvergenceHelper()
{
}

/*
 * returns average r given a vector of r
 */
double ConvergenceHelper::getAvgReliability(std::vector<double> r)
{
	double subtotal = 0.0;
	for (double r_i : r) {
		subtotal += r_i;
	}
	return subtotal / (double)r.size();
}

/*
 * Returns std R for MC Sampling
 */
double ConvergenceHelper::getStdReliabilityMC(double r, std::vector<int> i_st)
{
	double subtotal = 0.0;
	for (int i : i_st) {
		subtotal += pow((i - r), 2);
	}
	return sqrt(subtotal) / (double)i_st.size();
}
