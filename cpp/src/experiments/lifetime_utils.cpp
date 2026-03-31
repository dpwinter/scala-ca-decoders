#include "experiments/lifetime_utils.h"

#include "third_party/npy.hpp"

#include <filesystem>
#include <numeric>
#include <stdexcept>

double mean_lifetime(const std::vector<int>& lifetimes)
{
    if (lifetimes.empty()) {
        throw std::invalid_argument("cannot compute mean of empty lifetime vector");
    }

    const double total = std::accumulate(lifetimes.begin(), lifetimes.end(), 0.0);
    return total / static_cast<double>(lifetimes.size());
}

void write_lifetimes_npy(
    const std::vector<int>& lifetimes,
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

    npy::npy_data<int> data{lifetimes};
    data.shape = {lifetimes.size()};
    npy::write_npy(output_path, data);
}
