#ifndef CA_H_
#define CA_H_

#include "Cell.h"
#include "Location.h"
#include <vector>
#include <cassert>

class CA {
	
	private:
		int L;
		int Q;
		std::vector<Cell> cells;
	public:
		CA(int L, int U, double fC, double fN);
		// void reset();
		const Cell& getCell(int i);
		std::vector<Location> step(const std::vector<bool>& syndrome);
		// friend std::ostream& operator<<(std::ostream &out, CA const& ca);
};

#endif
