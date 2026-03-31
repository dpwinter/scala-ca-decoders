#ifndef CELL_H_
#define CELL_H_

#include "Memory.h"
#include "Random.h"
#include "Location.h"
#include <cmath>

class Cell {
	private:
		Location addr;
		int d;
		double fN = 4/10.;
		double fC = 9/10.;

		std::vector<Cell*> neighbors; 
		std::vector<bool> defects; 
		std::vector<Memory> memory;

	public:
		Cell(int row, int col, int Q, int U, int d);
		// void reset();

		void setNeighbors(std::vector<Cell*>&);
		void setDefect(bool value);
		// const Memory& getMemory(int k);

		void acquire();
		void update(double psig=0.0);
		Location rule();

		Location harringtonRule(const Location& addr, const std::vector<bool>& defects);
};

#endif
