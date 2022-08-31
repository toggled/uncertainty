//
//  DistanceDistribution.cpp
//  UncertainGraph
//
//  Created by Silviu Maniu on 9/2/13.
//  Copyright (c) 2013 Silviu Maniu. All rights reserved.
//

#include "DistanceDistribution.h"

DistanceDistribution::DistanceDistribution() {
	distribution = new std::unordered_map<int, float>();
	max_distance = 0;
}

DistanceDistribution::~DistanceDistribution() {
	distribution->clear();
	delete distribution;
}

void DistanceDistribution::add_to_distribution(int distance, float probability) {
	float previous_probability = 0.0f;
	if (this->distribution->find(distance) != this->distribution->end())
		previous_probability = this->distribution->operator[](distance);
	this->distribution->operator[](distance) = previous_probability + probability;
	if (distance>this->max_distance) this->max_distance = distance;
}

void DistanceDistribution::set_in_distribution(int distance, float probability) {
	this->distribution->operator[](distance) = probability;
	if (distance>this->max_distance) this->max_distance = distance;
}

float DistanceDistribution::get_probability(int distance) {
	if (this->distribution->find(distance) != this->distribution->end())
		return this->distribution->operator[](distance);
	else
		return 0.0f;
}

int DistanceDistribution::sample_distance() {
	float cumulated = 0.0f;
	float prob = ((float)rand() / (RAND_MAX));
	for (auto pair : *distribution) {
		float dist_prob = pair.second;
		cumulated += dist_prob;
		if (prob <= cumulated) return pair.first;
	}
	return std::numeric_limits<int>::max();
}

std::ostream& operator<<(std::ostream &out, DistanceDistribution &distribution) {
	out << distribution.get_size() << std::endl;
	for (auto iterator = distribution.distribution->begin(); iterator != distribution.distribution->end(); ++iterator) {
		out << iterator->first << "\t" << iterator->second << std::endl;
	}
	return out;
}

std::istream& operator>>(std::istream &in, DistanceDistribution &distribution) {
	int distribution_size = 0;
	in >> distribution_size;
	for (int i = 0; i<distribution_size; i++) {
		float f_distance;
		float probability;
		in >> f_distance >> probability;
		int distance = (int)f_distance;
		//std::cout << distance << " " << probability << std::endl;
		distribution.add_to_distribution(distance, probability);
	}
	return in;
}

void DistanceDistribution::combine_distribution(DistanceDistribution* other) {
	float remaining = 1.0f;
	int max_dist = max_distance<other->max_distance ? other->max_distance : max_distance;
	float right_probability = 0.0f;
	float left_probability = 0.0f;
	for (int d = 1; d <= max_dist; d++) {
		float rewinded_right = this->get_probability(d) / (1.0f - right_probability);
		float rewinded_left = other->get_probability(d) / (1.0f - left_probability);
		right_probability += this->get_probability(d);
		left_probability += other->get_probability(d);
		float combined_probability = remaining * (rewinded_right + (1.0f - rewinded_right)*rewinded_left);
		if (combined_probability>0) {
			this->set_in_distribution(d, combined_probability);
			remaining -= combined_probability;
		}
	}
}

DistanceDistribution* propagate_distribution(DistanceDistribution* right, DistanceDistribution* left) {
	DistanceDistribution* new_dist = new DistanceDistribution();
	for (auto first_entry : *right->distribution) {
		for (auto second_entry : *left->distribution) {
			int new_distance = first_entry.first + second_entry.first;
			float new_probability = first_entry.second * second_entry.second;
			new_dist->add_to_distribution(new_distance, new_probability);
		}
	}
	return new_dist;
}

void DistanceDistribution::copy_from_distribution(DistanceDistribution *other) {
	this->max_distance = other->get_max_distance();
	this->distribution->clear();
	delete this->distribution;
	this->distribution = new std::unordered_map<int, float>;
	for (int d = 1; d <= this->max_distance; d++) {
		float probability = other->get_probability(d);
		if (probability>0) this->distribution->operator[](d) = probability;
	}
}
