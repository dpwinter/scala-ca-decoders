#include "experiments/outcome_utils.h"

#include "third_party/npy.hpp"

#include <filesystem>
#include <numeric>
#include <stdexcept>

double logical_error_rate(const std::vector<std::uint8_t>& outcomes)
{
    if (outcomes.empty()) {
        throw std::invalid_argument("cannot compute logical error rate of empty outcome vector");
    }

    const double total = std::accumulate(outcomes.begin(), outcomes.end(), 0.0);
    return total / static_cast<double>(outcomes.size());
}

void write_outcomes_npy(
    const std::vector<std::uint8_t>& outcomes,
    const std::string& output_path
)
{
    if (output_path.empty()) {
        throw std::invalid_argument("output_path must not be empty");
    }

    std::filesystem::path path(output_path);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    npy::npy_data<std::uint8_t> data{outcomes};
    data.shape = {outcomes.size()};
    npy::write_npy(output_path, data);
}
