#pragma once
#include <boost/graph/transitive_closure.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <algorithm>
#include <cmath>
#include "Common.h"
#include "Randomiser.h"

class LazyPropagation
{
public:
	LazyPropagation();
	double lazySample(VertexDescr source, VertexDescr target, DirectedGraph g, size_t sample_size);
	~LazyPropagation();
};

