//
//  DistanceDistribution.h
//  UncertainGraph
//
//  Created by Silviu Maniu on 9/2/13.
//  Copyright (c) 2013 Silviu Maniu. All rights reserved.
//

#ifndef __UncertainGraph__DistanceDistribution__
#define __UncertainGraph__DistanceDistribution__

#include <iostream>
#include <ostream>
#include <istream>
#include <unordered_map>
#include <cstdlib>
#include <limits>
#include <random>

class DistanceDistribution{
    std::unordered_map<int,float>* distribution;
    int max_distance;
public:
    DistanceDistribution();
    ~DistanceDistribution();
    void add_to_distribution(int distance, float probability);
    void set_in_distribution(int distance, float probability);
    float get_probability(int distance);
    int sample_distance();
    int get_max_distance() {return max_distance;};
    unsigned long get_size() {return distribution->size();};
    //void write_to_stream(std::ostream* out_stream);
    //void load_from_stream(std::istream* in_stream);
    //Combination operators
    void combine_distribution(DistanceDistribution* other);
    void copy_from_distribution(DistanceDistribution* other);
    friend DistanceDistribution* propagate_distribution(DistanceDistribution* right, DistanceDistribution* left);
    //Output and input operators
    friend std::ostream& operator<<(std::ostream &out, DistanceDistribution &distribution);
    friend std::istream& operator>>(std::istream &in, DistanceDistribution &distribution);
};

#endif /* defined(__UncertainGraph__DistanceDistribution__) */
