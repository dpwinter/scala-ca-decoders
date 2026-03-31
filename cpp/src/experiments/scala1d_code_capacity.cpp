#include "experiments/scala1d_code_capacity.h"

#include "codes/repetition_code.h"
#include "models/scala1d.h"

#include <random>
#include <stdexcept>
#include <vector>

std::vector<std::uint8_t> run_scala1d_code_capacity(
    std::size_t distance,
    int samples,
    int max_steps,
    double p_data,
    unsigned int seed
)
{
    if (distance == 0) {
        throw std::invalid_argument("distance must be positive");
    }
    if (samples <= 0) {
        throw std::invalid_argument("samples must be positive");
    }
    if (max_steps <= 0) {
        throw std::invalid_argument("max_steps must be positive");
    }
    if (p_data < 0.0 || p_data > 1.0) {
        throw std::invalid_argument("p_data must be in [0,1]");
    }

    std::mt19937 rng(seed);
    std::vector<std::uint8_t> outcomes;
    outcomes.reserve(static_cast<std::size_t>(samples));

    for (int sample = 0; sample < samples; ++sample) {
        RepetitionCode code(distance);
        Scala1D decoder(distance);

        // Apply iid data noise once at t = 0.
        code.apply_data_noise(p_data, rng);

        // Unresolved runs are counted as failures.
        bool log_err = true;

        for (int t = 0; t < max_steps; ++t) {
            const auto syn = code.syndrome();

            bool has_defect = false;
            for (bool s : syn) {
                if (s) {
                    has_defect = true;
                    break;
                }
            }

            if (!has_defect) {
                log_err = code.has_logical_error();
                break;
            }

            // Deterministic decoding: no further noise after t = 0.
            decoder.step(code, syn, 0.0, rng);
        }

        outcomes.push_back(log_err ? 1 : 0);
    }

    return outcomes;
}
