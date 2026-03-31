#include "Cell.h"
#include <iostream>

Cell::Cell(int row, int col, int Q, int U, int d) : d{d}
{
	addr = locFromCoords(row % Q,col % Q); // level-0 address

	// for (int k=0; k<d; k++) {
	for (int k=1; k<d; k++) {
		int offset = int( (std::pow(Q,k) - 1) / 2.0 );
		double krow = (row - offset) / std::pow(Q,k);
		double kcol = (col - offset) / std::pow(Q,k);

		// higher-level address only if krow, kcol are ints.
		Location kaddr = (int(krow) == krow && int(kcol) == kcol) ? locFromCoords(int(krow)%Q, int(kcol)%Q) : Location::None;

		// if (k==0) {
		// 	addr = kaddr;
		// } else {
			memory.push_back( Memory{kaddr, int(std::pow(U,k)), int(std::pow(Q,k))} );
		// }
	}
	// reset();
	defects = std::vector<bool>(9,0);
}

void Cell::setNeighbors(std::vector<Cell*>& neighbors_) {
	neighbors = std::move(neighbors_);
}

// void Cell::reset() {
// 	defects = std::vector<bool>(9,0);
// 	for (Memory& mem : memory) {
// 		mem.reset();
// 	}
// }

// const Memory& Cell::getMemory(int k) {
// 	return memory.at(k);
// }

void Cell::setDefect(bool value) {
	defects[Location::C] = value;
}

void Cell::acquire() {

	// copy defects of neighbors
	for (int i=0; i<8; i++) {
		defects[i] = neighbors[i]->defects[Location::C];
	}
	// propagate signals (from opposite neighbor in same direction)
	for (int k=0; k<d-1; k++) {
		for (int i=0; i<8; i++) {
			Cell opposite_neighbor = *neighbors[oppositeLoc(i)];
			memory[k].n_countSig[i] = opposite_neighbor.memory[k].countSig[i];
			if (i<4) {
				memory[k].n_flipSig[i] = opposite_neighbor.memory[k].flipSig[i];
			}
		}
	}
}

void Cell::update(double psig) {
	for (Memory& mem : memory) {

		// signal noise
		for (int i=0; i<8; i++){
			if (Random::get(0.,1.) <= psig)
				mem.n_countSig[i] = mem.n_countSig[i] ^ 1;
		}

		// ONLY CS NOISE
		// for (int i=0; i<4; i++){
		// 	if (Random::get(0.,1.) <= psig)
		// 		mem.n_flipSig[i] = mem.n_flipSig[i] ^ 1;
		// }
		//

		// mem.age = (mem.age + 1) % mem.U;
		mem.age = (mem.age + 1) % (mem.U + mem.Q + 1);
		
		if (mem.addr != Location::None) { // hierarchy representative
										  
			for (int i=0; i<8; i++) {
				mem.countSig[i] = defects[Location::C]; // broadcast
			}

			// update counts
			mem.count[Location::C] += defects[Location::C]; 
			for (int i=0; i<8; i++) {
				mem.count[i] += mem.n_countSig[oppositeLoc(i)];
			}

		} 
		if (mem.addr == Location::None) { // copy signals for non-representatives
			mem.countSig = mem.n_countSig; // copy vector
			mem.flipSig = mem.n_flipSig;
		}
	}

}

Location Cell::rule() {

	// ! d is hierarchy level.
	// for (Memory& mem : memory) {
	for (int k=d-2; k>-1; k--) {

		// std::cout << k << ", " << memory.size() << '\n';
		// if (mem.age == 0 && mem.addr != Location::None) { // representative cell
		// if (memory[k].age == memory[k].U - 1 && memory[k].addr != Location::None) { // repr cell
		if (memory[k].age == memory[k].U && memory[k].addr != Location::None) { // repr cell
														  
			std::vector<bool> k_defects(9,0);

			// std::cout << "Cell " << memory[k].addr << " : ";
			for (int i=0; i<9; i++) {
				double f = (i == Location::C) ? fC : fN;
				k_defects[i] = memory[k].count[i] >= f * memory[k].U;
				// std::cout << memory[k].count[i] << " ";
				memory[k].count[i] = 0; // reset count
			}
			// std::cout << '\n';


			// level-k rule
			Location dir = harringtonRule(memory[k].addr, k_defects); // level-k rule
			if (dir != Location::None) { // emit flip signal
				memory[k].flipSig[dir] = 1;
			}
		} 
		// if (memory[k].age == memory[k].U + memory[k].Q - 1) {
		if (memory[k].age == memory[k].U + memory[k].Q ) {
			std::vector<bool> temp = memory[k].flipSig;
			memory[k].reset();
			for (int i=0; i<4; i++) {
				if (temp[i]) {
					return Location(i);
				}
				// if (memory[k].flipSig[i]) {
				// 	memory[k].flipSig[i] = 0; // reset flip signal
				// 	return Location(i); // correction
				// }
			}
		}
	}
	return harringtonRule(addr, defects); // level-0 rule
	// Location res = harringtonRule(addr, defects); // level-0 rule
	// if (res != Location::None) 
	// 	std::cout << "addr: " << addr << ", cor: " << res << '\n';
	// return res;
}

// NOT USED YET.
// Breuckmann rules
// Location Cell::harringtonRule(const Location& addr, const std::vector<bool>& defects) {

//     // Immediate exit
//     if(addr == Location::C || defects[Location::C] == 0) {
//         return Location::None;
//     }

//     if (addr == Location::NW) {
// 		if (defects[Location::W]) {
//     		return Location::W;
// 		}
// 		else if (defects[Location::N]) {
//     		return Location::None;
// 		}
// 		else if (defects[Location::E]) {
//     		return Location::E;
// 		}
// 		else if (defects[Location::S]) {
//     		return Location::S;
// 		}
// 		else if (defects[Location::NE]) {
//     		return Location::E;
// 		}
// 		else if (defects[Location::SE]) {
//     		return Location::S;
// 		}
// 		else if (defects[Location::SW]) {
//     		return Location::S;
// 		}
// 		else if (defects[Location::NW]) {
//     		return Location::W;
// 		}
// 		else { // default movement
// 			return Location::S;
// 		}
// 	}

//     if (addr == Location::N) {
//     	if (defects[Location::N]) {
//     		return Location::None;
// 		}
// 		else if (defects[Location::E]) {
//     		return Location::None;
// 		}
// 		else if (defects[Location::S]) {
//     		return Location::S;
// 		}
// 		else if (defects[Location::W]) {
//     		return Location::None;
// 		}
// 		else if (defects[Location::NW]) {
//     		return Location::N;
// 		}
// 		else if (defects[Location::NE]) {
//     		return Location::N;
// 		}
// 		else if (defects[Location::SE]) {
//     		return Location::S;
// 		}
// 		else if (defects[Location::SW]) {
//     		return Location::S;
// 		}
// 		else { // default movement
// 			return Location::S;
// 		}
// 	}

//     if (addr == Location::NE) {
//     	if (defects[Location::N]) {
//     		return Location::None;
// 		}
// 		else if (defects[Location::E]) {
//     		return Location::None;
// 		}
// 		else if (defects[Location::S]) {
//     		return Location::S;
// 		}
// 		else if (defects[Location::W]) {
//     		return Location::W;
// 		}
// 		else if (defects[Location::NE]) {
//     		return Location::N;
// 		}
// 		else if (defects[Location::SE]) {
//     		return Location::S;
// 		}
// 		else if (defects[Location::SW]) {
//     		return Location::S;
// 		}
// 		else if (defects[Location::NW]) {
//     		return Location::W;
// 		}
// 		else { // default movement
// 			return Location::S;
// 		}
// 	}

//     if (addr == Location::W) {
// 		if (defects[Location::W]) {
//     		return Location::W;
// 		}
// 		else if (defects[Location::N]) {
//     		return Location::None;
// 		}
// 		else if (defects[Location::E]) {
//     		return Location::E;
// 		}
// 		else if (defects[Location::S]) {
//     		return Location::None;
// 		}
// 		else if (defects[Location::NE]) {
//     		return Location::E;
// 		}
// 		else if (defects[Location::SE]) {
//     		return Location::E;
// 		}
// 		else if (defects[Location::SW]) {
//     		return Location::W;
// 		}
// 		else if (defects[Location::NW]) {
//     		return Location::W;
// 		}
// 		else { // default movement
// 			return Location::E;
// 		}
// 	}

//     if (addr == Location::E) {
//     	if (defects[Location::E]) {
//     		return Location::None;
// 		}
// 		else if (defects[Location::N]) {
//     		return Location::None;
// 		}
// 		else if (defects[Location::S]) {
//     		return Location::None;
// 		}
// 		else if (defects[Location::W]) {
//     		return Location::W;
// 		}
// 		else if (defects[Location::NE]) {
//     		return Location::E;
// 		}
// 		else if (defects[Location::SE]) {
//     		return Location::E;
// 		}
// 		else if (defects[Location::SW]) {
//     		return Location::W;
// 		}
// 		else if (defects[Location::NW]) {
//     		return Location::W;
// 		}
// 		else { // default movement
// 			return Location::W;
// 		}
// 	}

//     if (addr == Location::SW) {
//     	if (defects[Location::W]) {
//     		return Location::W;
// 		}
// 		else if (defects[Location::S]) {
//     		return Location::S;
// 		}
// 		else if (defects[Location::N]) {
//     		return Location::N;
// 		}
// 		else if (defects[Location::E]) {
//     		return Location::E;
// 		}
// 		else if (defects[Location::NW]) {
//     		return Location::N;
// 		}
// 		else if (defects[Location::NE]) {
//     		return Location::N;
// 		}
// 		else if (defects[Location::SE]) {
//     		return Location::E;
// 		}
// 		else if (defects[Location::SW]) {
//     		return Location::W;
// 		}
// 		else { // default movement
// 			return Location::N;
// 		}
// 	}

//     if (addr == Location::S) {
//     	if (defects[Location::S]) {
//     		return Location::S;
// 		}
// 		else if (defects[Location::N]) {
//     		return Location::N;
// 		}
// 		else if (defects[Location::E]) {
//     		return Location::None;
// 		}
// 		else if (defects[Location::W]) {
//     		return Location::None;
// 		}
// 		else if (defects[Location::NW]) { // NEW
//     		return Location::N;
// 		}
// 		else if (defects[Location::NE]) { // NEW
//     		return Location::N;
// 		}
// 		else if (defects[Location::SE]) {
//     		return Location::S;
// 		}
// 		else if (defects[Location::SW]) {
//     		return Location::S;
// 		}
// 		else { // default movement
// 			return Location::N;
// 		}
// 	}

//     if (addr == Location::SE) {
// 		if (defects[Location::S]) {
//     		return Location::S;
// 		}
// 		else if (defects[Location::E]) {
//     		return Location::None;
// 		}
// 		else if (defects[Location::N]) {
//     		return Location::N;
// 		}
// 		else if (defects[Location::W]) {
//     		return Location::W;
// 		}
// 		else if (defects[Location::NW]) { // NEW: prefer inner colony move
//     		return Location::N;
// 		}
// 		else if (defects[Location::NE]) {
//     		return Location::N;
// 		}
// 		else if (defects[Location::SE]) {
//     		return Location::S;
// 		}
// 		else if (defects[Location::SW]) {
//     		return Location::W;
// 		}
// 		else { // default movement
// 			return Location::N;
// 		}
// 	}

//     return Location::None;
// }

// // THIS IS THE ONE WE USE FOR cc_U=16
// // Modified Harrington rule
Location Cell::harringtonRule(const Location& addr, const std::vector<bool>& defects) {

    // Immediate exit
    if(addr == Location::C || defects[Location::C] == 0) {
        return Location::None;
    }

    if (addr == Location::NW) {
    	if (defects[Location::N]) {
    		return Location::None;
		}
		else if (defects[Location::W]) {
    		return Location::W;
		}
		else if (defects[Location::E]) {
    		return Location::E;
		}
		else if (defects[Location::S]) {
    		return Location::S;
		}
		// else if (defects[Location::SE]) {
    		// return Location::E;
		// }
		else if (defects[Location::NW]) {
    		return Location::W;
		}
		else if (defects[Location::NE]) {
    		// return Location::W;
    		return Location::None;
		}
		else if (defects[Location::SW]) {
    		return Location::W;
		}
		else { // default movement
			return Location::E;
		}
	}

    if (addr == Location::N) {
    	if (defects[Location::N]) {
    		return Location::None;
		}
		else if (defects[Location::W]) {
    		return Location::None;
		}
		else if (defects[Location::E]) {
    		return Location::None;
		}
		else if (defects[Location::S]) {
    		return Location::S;
		}
		// else if (defects[Location::SW]) {
    		// return Location::S;
		// }
		// else if (defects[Location::SE]) {
    		// return Location::S;
		// }
		else if (defects[Location::NW]) {
    		// return Location::W;
    		return Location::None;
		}
		else if (defects[Location::NE]) {
    		// return Location::E;
    		return Location::None;
		}
		else { // default movement
			return Location::S;
		}
	}

    if (addr == Location::NE) {
    	if (defects[Location::N]) {
    		return Location::None;
		}
		else if (defects[Location::E]) {
    		return Location::None;
		}
		else if (defects[Location::W]) {
    		return Location::W;
		}
		else if (defects[Location::S]) {
    		return Location::S;
		}
		// else if (defects[Location::SW]) {
    		// return Location::W;
		// }
		else if (defects[Location::NW]) {
    		// return Location::W;
    		return Location::None;
		}
		else if (defects[Location::NE]) {
    		return Location::None;
    		// return Location::E;
		}
		else if (defects[Location::SE]) {
    		return Location::None;
    		// return Location::S;
		}
		else { // default movement
			return Location::W;
		}
	}

    if (addr == Location::W) {
		if (defects[Location::W]) {
    		return Location::W;
		}
		else if (defects[Location::N]) {
    		return Location::None;
		}
		else if (defects[Location::S]) {
    		return Location::None;
		}
		else if (defects[Location::E]) {
    		return Location::E;
		}
		// else if (defects[Location::NE]) {
    		// return Location::E;
		// }
		// else if (defects[Location::SE]) {
    		// return Location::E;
		// }
		else if (defects[Location::NW]) {
    		return Location::W;
		}
		else if (defects[Location::SW]) {
    		return Location::W;
		}
		else { // default movement
			return Location::E;
		}
	}

    if (addr == Location::E) {
    	if (defects[Location::E]) {
    		return Location::None;
		}
		else if (defects[Location::N]) {
    		return Location::None;
		}
		else if (defects[Location::S]) {
    		return Location::None;
		}
		else if (defects[Location::W]) {
    		return Location::W;
		}
		// else if (defects[Location::SW]) {
    		// return Location::W;
		// }
		// else if (defects[Location::NW]) {
    		// return Location::W;
		// }
		else if (defects[Location::NE]) {
    		// return Location::N;
    		return Location::None;
		}
		else if (defects[Location::SE]) {
    		// return Location::S;
    		return Location::None;
		}
		else { // default movement
			return Location::W;
		}
	}

    if (addr == Location::SW) {
    	if (defects[Location::W]) {
    		return Location::W;
		}
		else if (defects[Location::S]) {
    		return Location::S;
		}
		else if (defects[Location::N]) {
    		return Location::N;
		}
		else if (defects[Location::E]) {
    		return Location::E;
		}
		// else if (defects[Location::NE]) {
    		// return Location::E;
		// }
		else if (defects[Location::NW]) {
    		return Location::W;
		}
		else if (defects[Location::SW]) {
    		return Location::S;
		}
		else if (defects[Location::SE]) {
    		return Location::S;
		}
		else { // default movement
			return Location::E;
		}
	}

    if (addr == Location::S) {
    	if (defects[Location::S]) {
    		return Location::S;
		}
		else if (defects[Location::W]) {
    		return Location::None;
		}
		else if (defects[Location::E]) {
    		return Location::None;
		}
		else if (defects[Location::N]) {
    		return Location::N;
		}
		// else if (defects[Location::NE]) { // NEW
    		// return Location::N;
		// }
		// else if (defects[Location::NW]) { // NEW
    		// return Location::N;
		// }
		else if (defects[Location::SE]) {
    		return Location::S;
		}
		else if (defects[Location::SW]) {
    		return Location::S;
		}
		else { // default movement
			return Location::N;
		}
	}

    if (addr == Location::SE) {
    	if (defects[Location::E]) {
    		return Location::None;
		}
		else if (defects[Location::S]) {
    		return Location::S;
		}
		else if (defects[Location::W]) {
    		return Location::W;
		}
		else if (defects[Location::N]) {
    		return Location::N;
		}
		// else if (defects[Location::NW]) { // NEW: prefer inner colony move
    		// return Location::W;
		// }
		else if (defects[Location::NE]) {
    		// return Location::N;
    		return Location::None;
		}
		else if (defects[Location::SE]) {
    		// return Location::S;
    		return Location::None;
		}
		else if (defects[Location::SW]) {
    		return Location::S;
		}
		else { // default movement
			return Location::W;
		}
	}

    return Location::None;
}


// Harrington orig rules
// Location Cell::harringtonRule(const Location& addr, const std::vector<bool>& defects) {

//     // Immediate exit
//     if(addr == Location::C || defects[Location::C] == 0) {
//         return Location::None;
//     }
//     // W border
//     if(addr == Location::NW || addr == Location::W || addr == Location::SW) {
//         if(defects[Location::W] || defects[Location::NW] || defects[Location::SW]) {
//             return Location::W;
//         }
//     }
//     // S border
//     if(addr == Location::S || addr == Location::SW || addr == Location::SE) {
//         if(defects[Location::S] || defects[Location::SW] || defects[Location::SE]) {
//             return Location::S;
//         }
//     }

//     // SW quadrant
//     if(addr == Location::SW) {
//         if(defects[Location::S] || defects[Location::W]) {
//             return Location::None;
//         }
//         else if(defects[Location::N]) {
//             return Location::N;
//         }
//         else if(defects[Location::E]) {
//             return Location::E;
//         }
//         else if(defects[Location::SW]) {
//             return Location::None;
//         }
//         else if(defects[Location::NW]) {
//             return Location::N;
//         }
//         else if(defects[Location::SE]) {
//             return Location::E;
//         }
//         else {
//             return Location::E;
//         }
//     }

//     // W corridor
//     if(addr == Location::W) {
//         if(defects[Location::S] || defects[Location::W] || defects[Location::N]) {
//             return Location::None;
//         }
//         else if(defects[Location::E]) {
//             return Location::E;
//         }
//         else if(defects[Location::SW] || defects[Location::NW]) {
//             return Location::None;
//         }
//         else {
//             return Location::E;
//         }
//     }

//     // NW quadrant
//     if(addr == Location::NW) {
//         if(defects[Location::W] || defects[Location::N]) {
//             return Location::None;
//         }
//         else if(defects[Location::E]) {
//             return Location::E;
//         }
//         else if(defects[Location::S]) {
//             return Location::S;
//         }
//         else if(defects[Location::NW]) {
//             return Location::None;
//         }
//         else if(defects[Location::NE]) {
//             return Location::E;
//         }
//         else if(defects[Location::SW]) {
//             return Location::S;
//         }
//         else {
//             return Location::E;
//         }
//     }

//     // N corridor
//     if(addr == Location::N) {
//         if(defects[Location::W] || defects[Location::N] || defects[Location::E]) {
//             return Location::None;
//         }
//         else if(defects[Location::S]) {
//             return Location::S;
//         }
//         else if(defects[Location::NW] || defects[Location::NE]) {
//             return Location::None;
//         }
//         else {
//             return Location::S;
//         }
//     }

//     // NE quadrant
//     if(addr == Location::NE) {
//         if(defects[Location::N] || defects[Location::E]) {
//             return Location::None;
//         }
//         else if(defects[Location::S]) {
//             return Location::S;
//         }
//         else if(defects[Location::W]) {
//             return Location::W;
//         }
//         else if(defects[Location::NE]) {
//             return Location::None;
//         }
//         else if(defects[Location::SE]) {
//             return Location::S;
//         }
//         else if(defects[Location::NW]) {
//             return Location::W;
//         }
//         else {
//             return Location::W;
//         }
//     }

//     // E corridor
//     if(addr == Location::E) {
//         if(defects[Location::N] || defects[Location::E] || defects[Location::S]) {
//             return Location::None;
//         }
//         else if(defects[Location::W]) {
//             return Location::W;
//         }
//         else if(defects[Location::NE] || defects[Location::SE]) {
//             return Location::None;
//         }
//         else {
//             return Location::W;
//         }
//     }

//     // SE quadrant
//     if(addr == Location::SE) {

//         if(defects[Location::E] || defects[Location::S]) {
//             return Location::None;
//         }
//         else if(defects[Location::W]) {
//             return Location::W;
//         }
//         else if(defects[Location::N]) {
//             return Location::N;
//         }
//         else if(defects[Location::SE]) {
//             return Location::None;
//         }
//         else if(defects[Location::SW]) {
//             return Location::W;
//         }
//         else if(defects[Location::NE]) {
//             return Location::N;
//         }
//         else {
//             return Location::W;
//         }
//     }

//     // S corridor
//     if(addr == Location::S) {
//         if(defects[Location::E] || defects[Location::S] || defects[Location::W]) {
//             return Location::None;
//         }
//         else if(defects[Location::N]) {
//             return Location::N;
//         }
//         else if(defects[Location::SE] || defects[Location::SW]) {
//             return Location::None;
//         }
//         else {
//             return Location::N;
//         }
//     }

//     return Location::None;
// }

