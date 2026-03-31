#ifndef MEMORY_H_
#define MEMORY_H_

#include "Location.h"
#include <algorithm>
#include <vector>

struct Memory {

	Location addr;
	int U; // work period (of current level)
	int Q; // colony size (of current level)
	int age; // time step % U^k

	std::vector<bool> countSig;
	std::vector<bool> n_countSig;
	std::vector<bool> flipSig;
	std::vector<bool> n_flipSig;
	std::vector<int> count;

	Memory(Location addr, int U, int Q);
	void reset();

};

#endif
