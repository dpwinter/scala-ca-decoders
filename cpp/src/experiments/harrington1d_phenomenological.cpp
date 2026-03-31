#include "experiments/harrington1d_phenomenological.h"

#include "codes/repetition_code.h"
#include "experiments/lifetime_utils.h"
#include "models/harrington1d.h"
#include "noise/measurement_noise.h"

#include <random>
#include <stdexcept>
#include <vector>

namespace {

// Match legacy applyCorrections(code, cors, L):
// - R flips qubit i+1
// - L flips qubit i
void apply_corrections(
    RepetitionCode& code,
    const std::vector<Harrington1DLocation>& corrections
)
{
    const int L = static_cast<int>(code.distance());

    for (int i = 0; i < L; ++i) {
        if (corrections[i] == H1D_R) {
            code.flip_qubit((i + 1) % L);
        } else if (corrections[i] == H1D_L) {
            code.flip_qubit(i);
        }
    }
}

}  // namespace

std::vector<int> run_harrington1d_phenomenological_lifetimes(
    std::size_t distance,
    int samples,
    int max_steps,
    double p_data,
    double p_meas,
    int U,
    double fN,
    double fC,
    unsigned int seed
)
{
    if (distance == 0) {
        throw std::invalid_argument("distance must be positive");
    }
    if (samples <= 0) {
        throw std::invalid_argument("samples must be positive");
    }
    if (max_steps < 0) {
        throw std::invalid_argument("max_steps must be non-negative");
    }
    if (p_data < 0.0 || p_data > 1.0) {
        throw std::invalid_argument("p_data must be in [0,1]");
    }
    if (p_meas < 0.0 || p_meas > 1.0) {
        throw std::invalid_argument("p_meas must be in [0,1]");
    }
    if (U <= 0) {
        throw std::invalid_argument("U must be positive");
    }
    if (fN < 0.0 || fN > 1.0) {
        throw std::invalid_argument("fN must be in [0,1]");
    }
    if (fC < 0.0 || fC > 1.0) {
        throw std::invalid_argument("fC must be in [0,1]");
    }

    std::mt19937 rng(seed);
    std::vector<int> lifetimes;
    lifetimes.reserve(static_cast<std::size_t>(samples));

    for (int n = 0; n < samples; ++n) {
        Harrington1D decoder(static_cast<int>(distance), U, fC, fN);
        RepetitionCode code(distance);

        int lifetime = 0;

        while (!code.has_logical_error()) {
            code.apply_data_noise(p_data, rng);

            auto syndrome = code.syndrome();

            if (p_meas > 0.0) {
                syndrome = add_measurement_noise(syndrome, p_meas, rng);
            }

            const auto corrections = decoder.step(syndrome);
            apply_corrections(code, corrections);

            ++lifetime;

            if (max_steps > 0 && lifetime >= max_steps) {
                break;
            }
        }

        lifetimes.push_back(lifetime);
    }

    return lifetimes;
}
