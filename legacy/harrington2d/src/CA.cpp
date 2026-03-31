#include "CA.h"

#include <iostream>
#include <string>

CA::CA(int L) : L{L} {

	int Q = 3; // hard-coded
	// int U = 10;
	int U = 16;

	float df = std::log2(L) / std::log2(Q); // hierarchy level
	assert( std::ceilf(df) == df || std::floorf(df) == df );
	int d = int( std::ceil(df) ); // int repr of level
										
	// create cells
	for (int i=0; i<L; i++) 
	{
		for (int j=0; j<L; j++) 
		{
			cells.push_back( Cell(i,j,Q,U,d) );
		}
	}

	// set neighbors
	for (int i=0; i<L; i++) 
	{
		for (int j=0; j<L; j++) 
		{
			std::vector<Cell*> neighbors;

            int i_ = i == 0 ? L-1 : i-1;
            int j_ = j == 0 ? L-1 : j-1;
            int ip = (i+1) % L;
            int jp = (j+1) % L;
            															   
            neighbors.push_back( &cells[i_*L+j] ); // N
            neighbors.push_back( &cells[i*L+j_] ); // W
            neighbors.push_back( &cells[i*L+jp] ); // E
            neighbors.push_back( &cells[ip*L+j] ); // S
            neighbors.push_back( &cells[i_*L+j_] ); // NW
            neighbors.push_back( &cells[i_*L+jp] ); // NE
            neighbors.push_back( &cells[ip*L+j_] ); // SW
            neighbors.push_back( &cells[ip*L+jp] ); // SE
            									  
            cells[i*L+j].setNeighbors(neighbors);
		}
	}
}

// void CA::reset() {
// 	for (int i=0; i<L*L; i++) {
// 		cells[i].reset();
// 	}
// }

// const Cell& CA::getCell(int i, int j) {
// 	assert( i*j < L*L );
// 	return cells[i*L+j];
// }

// std::vector<Location> CA::step(const std::vector<bool>& syndrome) {
void CA::step(py::array_t<bool>& xframe, py::array_t<bool>& syndrome, double psig)
{
	
	auto syndrome_buf = syndrome.request(true);
	bool* syndrome_ptr = static_cast<bool*>(syndrome_buf.ptr);

	auto xframe_buf = xframe.request(true);
	bool* xframe_ptr = static_cast<bool*>(xframe_buf.ptr);

	
	// 1. measure syndrome, assign to cells
	for (int i=0; i<L*L; i++) {
		cells[i].setDefect( syndrome_ptr[i] );
	}
	// 2. copy neighbor data
	for (int i=0; i<L*L; i++) {
		cells[i].acquire();
	}
	// 3. synchronous update
	for (int i=0; i<L*L; i++) {
		cells[i].update(psig);
	}
	// 4. corrections from local rule
	for (int i=0; i<L; i++) {
        for (int j=0; j<L; j++) {
            Location cor = cells[i*L+j].rule();
			int idx = 0;
			switch(cor) {
				case Location::N:
					idx = L*L + L*(((i-1)%L+L)%L) + j;
					break;
				case Location::W:
					idx = i*L + j;
					break;
				case Location::E:
					idx = i*L + (j+1)%L;
					break;
				case Location::S:
					idx = L*L + i*L + j;
					break;
				default:
					break;
			}   
			if (cor != Location::None) {
				xframe_ptr[idx] = !xframe_ptr[idx];
			}   
        }   
    }   

}

// std::vector<Location> CA::step(const std::vector<bool>& syndrome, double pfs, double pcs) {
// 	// 1. measure syndrome, assign to cells
// 	for (int i=0; i<L*L; i++) {
// 		cells[i].setDefect( syndrome[i] );
// 	}
// 	// 2. copy neighbor data
// 	for (int i=0; i<L*L; i++) {
// 		cells[i].acquire(pfs, pcs);
// 	}
// 	// 3. synchronous update
// 	for (int i=0; i<L*L; i++) {
// 		cells[i].update();
// 	}
// 	// 4. corrections from local rule
// 	std::vector<Location> cors;
// 	for (int i=0; i<L*L; i++) {
// 		cors.push_back( cells[i].rule() );
// 	}
// 	return cors;
// }
