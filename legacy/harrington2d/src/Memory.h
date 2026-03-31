#ifndef MEMORY_H_
#define MEMORY_H_

#include "Location.h"
#include <vector>
#include <algorithm>

struct Memory {

    Location addr;
    int U; // work period (of this level)
    int Q; // colony size

    int age;
	std::vector<bool> countSig;
	std::vector<bool> n_countSig;
	std::vector<bool> flipSig;
	std::vector<bool> n_flipSig;
	std::vector<int> count;

	Memory(Location addr, int U, int Q);
	void reset();
};

#endif
