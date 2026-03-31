#ifndef OUTCOME_UTILS_H
#define OUTCOME_UTILS_H

#include <cstdint>
#include <string>
#include <vector>

double logical_error_rate(const std::vector<std::uint8_t>& outcomes);

void write_outcomes_npy(
    const std::vector<std::uint8_t>& outcomes,
    const std::string& output_path
);

#endif
