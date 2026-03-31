#include "noise/measurement_noise.h"

#include <stdexcept>

std::vector<bool> add_measurement_noise(
    const std::vector<bool>& syndrome,
    double p,
    std::mt19937& rng
)
{
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("measurement noise p must be in [0,1]");
    }

    std::bernoulli_distribution dist(p);

    std::vector<bool> noisy = syndrome;

    for (std::size_t i = 0; i < noisy.size(); ++i) {
        if (dist(rng)) {
            noisy[i] = noisy[i] ^ true;
        }
    }

    return noisy;
}
