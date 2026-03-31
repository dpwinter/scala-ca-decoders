#include "codes/repetition_code.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

RepetitionCode::RepetitionCode(std::size_t distance)
    : nqubits_(distance), qubits_(distance, false)
{
    if (distance == 0) {
        throw std::invalid_argument("RepetitionCode distance must be positive.");
    }
}

void RepetitionCode::reset()
{
    std::fill(qubits_.begin(), qubits_.end(), false);
}

std::size_t RepetitionCode::distance() const noexcept
{
    return nqubits_;
}

bool RepetitionCode::qubit(std::size_t idx) const
{
    if (idx >= nqubits_) {
        throw std::out_of_range("qubit index out of range");
    }
    return qubits_[idx];
}

bool RepetitionCode::qubit_parity(std::size_t idx1, std::size_t idx2) const
{
    if (idx1 >= nqubits_ || idx2 >= nqubits_) {
        throw std::out_of_range("qubit_parity index out of range");
    }
    return qubits_[idx1] ^ qubits_[idx2];
}

void RepetitionCode::flip_qubit(std::size_t idx)
{
    if (idx >= nqubits_) {
        throw std::out_of_range("flip_qubit index out of range");
    }
    qubits_[idx] = qubits_[idx] ^ true;
}

int RepetitionCode::num_errors() const
{
    return static_cast<int>(std::accumulate(qubits_.begin(), qubits_.end(), 0));
}

void RepetitionCode::apply_data_noise(double p, std::mt19937& rng)
{
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("apply_data_noise: p must be in [0,1].");
    }

    for (std::size_t i = 0; i < nqubits_; ++i) {
        if (sample_bernoulli(p, rng)) {
            flip_qubit(i);
        }
    }
}

std::vector<bool> RepetitionCode::syndrome() const
{
	// Convention: S_i = Q_i xor Q_i+1
	
    std::vector<bool> syn(nqubits_, false);

    for (std::size_t i = 0; i < nqubits_; ++i) {
        const std::size_t right = mod(static_cast<long long>(i) + 1);
        syn[i] = qubit_parity(i, right);
    }

    return syn;
}

bool RepetitionCode::has_logical_error() const
{
    const int nerrors = num_errors();
    return nerrors > static_cast<int>(nqubits_ / 2);
}

void RepetitionCode::print_qubits() const
{
    for (std::size_t i = 0; i < nqubits_; ++i) {
        std::cout << qubits_[i] << ' ';
    }
    std::cout << '\n';
}

std::ostream& operator<<(std::ostream& out, const RepetitionCode& code)
{
    const auto syn = code.syndrome();
    for (std::size_t i = 0; i < code.nqubits_; ++i) {
        out << syn[i] << ' ';
    }
    out << '\n';
    return out;
}

std::size_t RepetitionCode::mod(long long idx) const noexcept
{
    const long long L = static_cast<long long>(nqubits_);
    return static_cast<std::size_t>((idx % L + L) % L);
}

bool RepetitionCode::sample_bernoulli(double p, std::mt19937& rng)
{
    std::bernoulli_distribution dist(p);
    return dist(rng);
}
