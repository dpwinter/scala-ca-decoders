#include "experiments/scala1d_phenomenological.h"

#include "codes/repetition_code.h"
#include "experiments/lifetime_utils.h"
#include "models/scala1d.h"
#include "noise/measurement_noise.h"

#include <random>
#include <stdexcept>
#include <vector>

std::vector<int> run_scala1d_phenomenological_lifetimes(
    std::size_t distance,
    int samples,
    int max_steps,
    int reset_time,
    double p_data,
    double p_meas,
    double p_signal,
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
    if (reset_time < 0) {
        throw std::invalid_argument("reset_time must be non-negative");
    }
    if (p_data < 0.0 || p_data > 1.0) {
        throw std::invalid_argument("p_data must be in [0,1]");
    }
    if (p_meas < 0.0 || p_meas > 1.0) {
        throw std::invalid_argument("p_meas must be in [0,1]");
    }
    if (p_signal < 0.0 || p_signal > 1.0) {
        throw std::invalid_argument("p_signal must be in [0,1]");
    }

    std::mt19937 rng(seed);
    std::vector<int> lifetimes;
    lifetimes.reserve(static_cast<std::size_t>(samples));

    for (int sample = 0; sample < samples; ++sample) {
        RepetitionCode code(distance);
        Scala1D decoder(distance);

        int t = 0;

        while (!code.has_logical_error()) {
            code.apply_data_noise(p_data, rng);

            const auto syn = code.syndrome();
            const auto syn_obs = add_measurement_noise(syn, p_meas, rng);

            decoder.step(code, syn_obs, p_signal, rng);

            ++t;

            if (reset_time > 0 && (t % reset_time == 0)) {
                decoder.reset();
            }

            if (max_steps > 0 && t >= max_steps) {
                break;
            }
        }

        lifetimes.push_back(t);
    }

    return lifetimes;
}
