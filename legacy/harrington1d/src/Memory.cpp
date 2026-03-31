#include "Memory.h"

Memory::Memory(Location addr, int U, int Q) : addr{addr}, U{U}, Q{Q} {
	age = 0;
	countSig = std::vector<bool>(2,0); // L,R
	n_countSig = std::vector<bool>(2,0);
	flipSig = std::vector<bool>(2,0);
	n_flipSig = std::vector<bool>(2,0);
	count = std::vector<int>(3,0); // L,R,C
}

void Memory::reset() {
	std::fill(countSig.begin(), countSig.end(), 0);
	std::fill(n_countSig.begin(), n_countSig.end(), 0);
	std::fill(flipSig.begin(), flipSig.end(), 0);
	std::fill(n_flipSig.begin(), n_flipSig.end(), 0);
	std::fill(count.begin(), count.end(), 0);
}
