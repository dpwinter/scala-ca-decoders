#ifndef REPCODE_H_
#define REPCODE_H_

#include "Random.h"
#include "Location.h"
#include <vector>
#include <iostream>

class RepCode {
	private:
		int L;
	public:
		std::vector<bool> qubits;
		RepCode(int L);
		void reset();
		void flip(int i, Location loc);
		bool getStab(int i);
		const bool getQubit(int i);
		void setQubit(int i, bool val);
		std::vector<bool> getSyndrome();
		std::vector<bool> getSyndrome(double p);
		void noise(double p);
		bool hasLogErr();
		friend std::ostream& operator<<(std::ostream &out, RepCode const& code);
};

#endif
