#ifndef HARRINGTON1D_PHENOMENOLOGICAL_H
#define HARRINGTON1D_PHENOMENOLOGICAL_H

#include <cstddef>
#include <vector>

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
);

#endif
