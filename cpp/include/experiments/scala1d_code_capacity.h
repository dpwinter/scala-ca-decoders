#ifndef SCALA1D_CODE_CAPACITY_H
#define SCALA1D_CODE_CAPACITY_H

#include <cstddef>
#include <cstdint>
#include <vector>

std::vector<std::uint8_t> run_scala1d_code_capacity(
    std::size_t distance,
    int samples,
    int max_steps,
    double p_data,
    unsigned int seed
);

#endif
