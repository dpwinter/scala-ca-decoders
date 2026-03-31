#include "CA.h"

#include <iostream>

CA::CA(int L, int U, double fC, double fN) : L{L}, Q{3} {

	float df = std::log2(L) / std::log2(Q);
	assert( std::ceil(df) == df || std::floorf(df) == df );
	int d = int( std::ceil(df) );

	// create cells
	for (int i=0; i<L; i++)
		cells.push_back( Cell(i,Q,U,d,fC,fN) );

	// set neighbors
	for (int i=0; i<L; i++) {
		std::vector<Cell*> neighbors;
		int l = i==0 ? L-1 : i-1;
		int r = (i+1) % L;
		neighbors.push_back( &cells[l] );
		neighbors.push_back( &cells[r] );

		cells[i].setNeighbors(neighbors);
	}
}

// std::ostream& operator<<(std::ostream &out, CA const& ca) {
// 	for (int i=0; i<ca.L; i++) {
// 		out << ca.cells[i] << " ";
// 	}
// 	out << '\n';
// 	return out;
// }

// void CA::reset() {
// 	for (int i=0; i<L; i++)
// 		cells[i].reset();
// }

const Cell& CA::getCell(int i) {
	assert( i < L );
	return cells[i];
}

std::vector<Location> CA::step(const std::vector<bool>& syndrome) {

	// 1. Measure syndrome, assign to cells
	for (int i=0; i<L; i++)
		cells[i].setDefect( syndrome[i] );
	
	// 2. Copy neighbor data
	for (int i=0; i<L; i++)
		cells[i].acquire();

	// 3. Synchronous update
	for (int i=0; i<L; i++)
		cells[i].update();

	// 4. Corrections from local rule
	std::vector<Location> cors;
	for (int i=0; i<L; i++)
		cors.push_back( cells[i].rule() );

	return cors;
}


