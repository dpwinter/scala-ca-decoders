#ifndef HARRINGTON1D_CODE_CAPACITY_H
#define HARRINGTON1D_CODE_CAPACITY_H

#include <cstddef>
#include <cstdint>
#include <vector>

std::vector<std::uint8_t> run_harrington1d_code_capacity(
    std::size_t distance,
    int samples,
    int max_steps,
    double p_data,
    int U,
    double fN,
    double fC,
    unsigned int seed
);

#endif
