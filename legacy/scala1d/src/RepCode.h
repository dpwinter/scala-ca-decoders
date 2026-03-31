#ifndef REPCODE_H
#define REPCODE_H

#include "Random.h"
#include <array>
#include <iostream>

// template class, define functions in header.

template <size_t T> // known at compile time
class RepCode
{
	private:
		size_t nqubits;
		std::array<bool,T> qubits;
	public:
		RepCode() : nqubits{T} { reset(); }
		void reset() { qubits.fill(0); };
		bool getQubitParity(int idx1, int idx2) const { return qubits[idx1] ^ qubits[idx2]; }
		size_t getNQubits() const { return nqubits; }
		void flipQubit(int idx) { qubits[idx] = qubits[idx] ^ 1; }

		int getNErrs() {
			int n_errs = 0;
			for (int i=0; i<nqubits; i++)
			{
				n_errs = n_errs + qubits[i];
			}
			return n_errs;
		}

		void setNoise(double p)
		{
			for (int i=0; i<nqubits; i++)
			{
				if (Random::getError(p)) flipQubit(i);
			}
		}

		std::array<bool,T> getSyndrome(double q=0.0) const
		{
			std::array<bool,T> syndrome;
			for (int i=0; i<nqubits; i++)
			{
				// syndrome[i] = getQubitParity(i, (i+1) % nqubits);
				int i_ = (i==0) ? nqubits-1 : i-1;
				syndrome[i] = getQubitParity(i_, i);

				// noise
				if (q > 0.0)
					if (Random::getError(q)) syndrome[i] = syndrome[i] ^ 1;
			}
			return syndrome;
		}

		bool hasLogErr()
		{
			int nerrors = std::accumulate(qubits.begin(), qubits.end(), 0);
			return nerrors > nqubits / 2;
		}

		void print_qubits()
		{
			for (int i=0; i<nqubits; i++)
			{
				std::cout << qubits[i] << ' ';
			}
			std::cout << '\n';
		}

		friend std::ostream& operator<<(std::ostream& out, const RepCode& code)
		{
			std::array<bool,T> syndrome = code.getSyndrome();
			for (int i=0; i<code.nqubits; i++)
			{
				out << syndrome[i] << ' ';
			}
			out << '\n';
			return out;
		}
};

#endif
