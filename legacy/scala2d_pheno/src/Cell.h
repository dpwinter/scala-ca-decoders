#ifndef CELL_H_
#define CELL_H_

#include "Location.h"
#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include "Random.h"

class Cell
{
	private:
		std::array<Cell*,4> neighbors;
		std::array<bool,4> sigs;
		std::array<bool,4> nsigs;
		std::array<bool,5> defects;

	public:
		Cell();
		void setNeighbors(std::array<Cell*,4>&);
		void setDefect(bool value);
		void reset();

		void acquire();
		void broadcast();
		void propagate();
		void update(double psig=0.0);
		Location rule();
};

#endif
