#ifndef LIFETIME_UTILS_H
#define LIFETIME_UTILS_H

#include <string>
#include <vector>

double mean_lifetime(const std::vector<int>& lifetimes);

void write_lifetimes_npy(
    const std::vector<int>& lifetimes,
    const std::string& output_path
);

#endif
