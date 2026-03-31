#include "experiments/harrington1d_code_capacity.h"

#include "codes/repetition_code.h"
#include "models/harrington1d.h"

#include <random>
#include <stdexcept>
#include <vector>

namespace {

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

std::vector<std::uint8_t> run_harrington1d_code_capacity(
    std::size_t distance,
    int samples,
    int max_steps,
    double p_data,
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
    if (max_steps <= 0) {
        throw std::invalid_argument("max_steps must be positive");
    }
    if (p_data < 0.0 || p_data > 1.0) {
        throw std::invalid_argument("p_data must be in [0,1]");
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
    std::vector<std::uint8_t> outcomes;
    outcomes.reserve(static_cast<std::size_t>(samples));

    for (int n = 0; n < samples; ++n) {
        Harrington1D decoder(static_cast<int>(distance), U, fC, fN);
        RepetitionCode code(distance);

        // Single-shot data noise at t = 0.
        code.apply_data_noise(p_data, rng);

        // Unresolved runs count as failures.
        bool log_err = true;

        for (int t = 0; t < max_steps; ++t) {
            const auto syndrome = code.syndrome();

            bool has_defect = false;
            for (bool s : syndrome) {
                if (s) {
                    has_defect = true;
                    break;
                }
            }

            if (!has_defect) {
                log_err = code.has_logical_error();
                break;
            }

            const auto corrections = decoder.step(syndrome);
            apply_corrections(code, corrections);
        }

        outcomes.push_back(log_err ? 1 : 0);
    }

    return outcomes;
}
