#ifndef RANDOM_MT_H
#define RANDOM_MT_H

#include <random>
#include <chrono>

namespace Random
{
	inline std::mt19937 generate() {
		std::random_device rd{};
		std::seed_seq ss{ static_cast<std::seed_seq::result_type>(
				std::chrono::steady_clock::now().time_since_epoch().count()
				) }; // seed by clock event
		return std::mt19937{ ss };
	}
	inline std::mt19937 mt{ generate() };
	inline double get(double min, double max) {
		return std::uniform_real_distribution{min,max}(mt);
	}
	inline bool getError(double p) { return Random::get(0.0, 1.0) <= p ? true : false; }
}

#endif
