#include "experiments/outcome_utils.h"
#include "experiments/scala1d_code_capacity.h"

#include <iostream>

int main()
{
    const std::size_t distance = 15;
    const int samples = 2000;
    const int max_steps = 1000;
    const unsigned int seed = 12345;

    const auto low_outcomes = run_scala1d_code_capacity(
        distance,
        samples,
        max_steps,
        0.10,
        seed
    );

    const auto high_outcomes = run_scala1d_code_capacity(
        distance,
        samples,
        max_steps,
        0.30,
        seed
    );

    const double p_low = logical_error_rate(low_outcomes);
    const double p_high = logical_error_rate(high_outcomes);

    std::cout << "scala1d code-capacity p_low  = " << p_low << "\n";
    std::cout << "scala1d code-capacity p_high = " << p_high << "\n";

    if (!(p_low < p_high)) {
        std::cerr << "Expected logical error rate to increase with p_data\n";
        return 1;
    }

    std::cout << "test_scala1d_code_capacity passed\n";
    return 0;
}
