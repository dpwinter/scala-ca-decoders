#ifndef REPETITION_CODE_H
#define REPETITION_CODE_H

#include <cstddef>
#include <iosfwd>
#include <random>
#include <vector>

class RepetitionCode
{
public:
    explicit RepetitionCode(std::size_t distance);

    // Reset all data qubits to the trivial state.
    void reset();

    std::size_t distance() const noexcept;
    bool qubit(std::size_t idx) const;
    bool qubit_parity(std::size_t idx1, std::size_t idx2) const;

    // Flip a single data qubit.
    void flip_qubit(std::size_t idx);

    int num_errors() const;

    // Apply iid data-qubit bit-flip noise.
    void apply_data_noise(double p, std::mt19937& rng);

    // Return the ideal syndrome.
    std::vector<bool> syndrome() const;

    bool has_logical_error() const;

    void print_qubits() const;

    friend std::ostream& operator<<(std::ostream& out, const RepetitionCode& code);

private:
    std::size_t mod(long long idx) const noexcept;
    static bool sample_bernoulli(double p, std::mt19937& rng);

    std::size_t nqubits_;
    std::vector<bool> qubits_;
};

#endif
