#ifndef MEASUREMENT_NOISE_H
#define MEASUREMENT_NOISE_H

#include <random>
#include <vector>

// Apply iid bit-flip noise to a syndrome vector.
std::vector<bool> add_measurement_noise(
    const std::vector<bool>& syndrome,
    double p,
    std::mt19937& rng
);

#endif
