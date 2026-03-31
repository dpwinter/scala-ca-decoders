#ifndef CELL_H_
#define CELL_H_

#include "Random.h"
#include "Memory.h"
#include "Location.h"

class Cell {
	private:
		Location addr;
		int d;
		double fN;
		double fC;
		
		std::vector<Cell*> neighbors;
		std::vector<bool> defects;
		std::vector<Memory> memory;

		/* void noise(std::vector<bool>& v, double p); */
	public:
		Cell(int addr, int Q, int U, int d, double fC, double fN);
		// void reset();

		void setNeighbors(std::vector<Cell*>&);
		void setDefect(bool value);
		// const Memory& getMemory(int k);

		void acquire();
		void update();
		Location rule();

		Location harringtonRule(const Location& addr, const std::vector<bool>& defects);
		// friend std::ostream& operator<<(std::ostream &out, Cell const& cell);
};

#endif
