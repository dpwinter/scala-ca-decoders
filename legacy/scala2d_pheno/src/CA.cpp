#include "CA.h"

CA::CA(int d) : d{d}
{
	// create cells
	for (int i=0; i<d; i++) {
		for (int j=0; j<d; j++) {
			cells.push_back( Cell() );
		}
	}
	// set neighbors
	for (int i=0; i<d; i++) {
		for (int j=0; j<d; j++) {
			std::array<Cell*,4> neighbors;

            int i_ = i == 0 ? d-1 : i-1;
            int j_ = j == 0 ? d-1 : j-1;
            int ip = (i+1) % d;
            int jp = (j+1) % d;

            neighbors[0] = &cells[i_*d+j] ; // N
            neighbors[1] = &cells[i*d+j_] ; // W
            neighbors[2] = &cells[i*d+jp] ; // E
            neighbors[3] = &cells[ip*d+j] ; // S

            cells[i*d+j].setNeighbors(neighbors);
		}
	}
	reset();
}

void CA::reset()
{
	age = 0;
	for (int i=0; i<d*d; i++) {
		cells[i].reset();
	}
}

void CA::step(py::array_t<bool>& xframe, py::array_t<bool>& syndrome, double psig)
{
	// age = (age + 1) % d;
	// // age = (age + 1) % ((d+1)/2);
	// if (age == 0)
	// 	reset();

	// retrieve syndrome
	auto syndrome_buf = syndrome.request(true);
	bool* syndrome_ptr = static_cast<bool*>(syndrome_buf.ptr);

	// 1. set defects
	for (int i=0; i<d*d; i++) {
		cells[i].setDefect( syndrome_ptr[i] );
	}
	// 2. copy neighbor data
	for (int i=0; i<d*d; i++) {
		cells[i].acquire();
	}
	// 3. broadcast
	for (int i=0; i<d*d; i++) {
		cells[i].broadcast();
	}
	// 4. propagte
	for (int i=0; i<d*d; i++) {
		cells[i].propagate();
	}
	// 5. synchronous update
	for (int i=0; i<d*d; i++) {
		cells[i].update(psig);
	}

	auto xframe_buf = xframe.request(true);
	bool* xframe_ptr = static_cast<bool*>(xframe_buf.ptr);
	// 6. corrections
	for (int i=0; i<d; i++) {
		for (int j=0; j<d; j++) {
			Location cor = cells[i*d+j].rule();
			// std::vector<Location> cors = cells[i*d+j].rule();
			// for (auto cor : cors) {
				int idx = 0;
				switch(cor) {
					case Location::N:
						idx = d*d + d*(((i-1)%d+d)%d) + j;
						break;
					case Location::W:
						idx = i*d + j;
						break;
					case Location::E:
						idx = i*d + (j+1)%d;
						break;
					case Location::S:
						idx = d*d + i*d + j;
						break;
					default:
						break;
				}
				if (cor != Location::None) {
					xframe_ptr[idx] = !xframe_ptr[idx];
				}
			// }

		}
	}
}
