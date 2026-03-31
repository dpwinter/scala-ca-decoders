#ifndef SCALA1D_PHENOMENOLOGICAL_H
#define SCALA1D_PHENOMENOLOGICAL_H

#include <cstddef>
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
);

#endif
