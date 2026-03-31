#include "Cell.h"

#include <iostream> // debug
					
Cell::Cell(int addr_, int Q, int U, int d, double fC, double fN) : d{d}, fC{fC}, fN{fN} {

	addr = locFromCoords(addr_ % Q); // level-0 address

	// std::cout << d << '\n';
	for (int k=1; k<d; k++) { // higher levels
		int offset = int( (std::pow(Q,k) - 1) / 2.0 );
		double addr__ = (addr_ - offset) / std::pow(Q,k);
		Location kaddr = ( int(addr__) == addr__ ) ? locFromCoords( int(addr__) % Q ) : Location::None;
		memory.push_back( Memory{kaddr, int(std::pow(U,k)), int(std::pow(Q,k))} );
	}
	// reset();
	defects = std::vector<bool>(3,0);
}

// std::ostream& operator<<(std::ostream &out, Cell const& cell) {
// 	/* out << cell.defects[Location::C]; */
// 	if (cell.d>1) {
// 		out << "[" << cell.memory[0].flipSig[0] << " " << cell.defects[Location::C] << " " << cell.memory[0].flipSig[1] << "]";
// 	}
// 	return out;
// }

void Cell::setNeighbors(std::vector<Cell*>& neighbors_) {
	neighbors = std::move(neighbors_); // move previously created vector of pointers here
}

// void Cell::reset() {
// 	defects = std::vector<bool>(3,0);
// 	for (Memory& mem : memory) {
// 		mem.reset();
// 	}
// }

// const Memory& Cell::getMemory(int k) {
// 	return memory.at(k);
// }

void Cell::setDefect(bool val) {
	defects[Location::C] = val;
}

void Cell::acquire() {

	// copy defects of neighbors
	defects[Location::L] = neighbors[Location::L]->defects[Location::C];
	defects[Location::R] = neighbors[Location::R]->defects[Location::C];

	// copy signals (from opposite neighbor in same direction)
	for (int k=0; k<d-1; k++) { // all levels, all cells
		memory[k].n_countSig[Location::L] = neighbors[Location::R]->memory[k].countSig[Location::L];
		memory[k].n_countSig[Location::R] = neighbors[Location::L]->memory[k].countSig[Location::R];

		memory[k].n_flipSig[Location::L] = neighbors[Location::R]->memory[k].flipSig[Location::L];
		memory[k].n_flipSig[Location::R] = neighbors[Location::L]->memory[k].flipSig[Location::R];
	}
}

void Cell::update() {
	for (Memory& mem : memory) {
		// mem.age = (mem.age + 1) % mem.U;
		mem.age = (mem.age + 1) % (mem.U + mem.Q);
		// if (mem.addr != Location::None && mem.age <= mem.U) { // hierarchy representative
		if (mem.addr != Location::None) { // hierarchy representative
										  
			// broadcast
			mem.countSig[Location::L] = defects[Location::C];
			mem.countSig[Location::R] = defects[Location::C];

			// update counts
			mem.count[Location::C] += defects[Location::C];
			mem.count[Location::L] += mem.n_countSig[Location::R];
			mem.count[Location::R] += mem.n_countSig[Location::L];

		} 
		if (mem.addr == Location::None) { // all non-representative cells
			// Note: representative cells have their FlipSig value not changed.
			mem.countSig = mem.n_countSig;
			mem.flipSig = mem.n_flipSig;
		}
	}
}

Location Cell::rule() {
	// we want to start with last level of hierarchy for a cell!
	for (int k=d-2; k>-1; k--) { // all levels, all cells

		if (memory[k].age == memory[k].U - 1 && memory[k].addr != Location::None) {

			std::vector<bool> k_defects(3,0);

			k_defects[Location::C] = memory[k].count[Location::C] >= fC * memory[k].U;
			k_defects[Location::L] = memory[k].count[Location::L] >= fN * memory[k].U;
			k_defects[Location::R] = memory[k].count[Location::R] >= fN * memory[k].U;

			Location dir = harringtonRule(memory[k].addr, k_defects);
			if (dir != Location::None) {
				memory[k].flipSig[dir] = 1; // set FlipSig to 1
			}
		} 
		if (memory[k].age == memory[k].U + memory[k].Q - 1) {
			std::vector<bool> temp = memory[k].flipSig; // store before resetting
			memory[k].reset();
			for (int i=0; i<2; i++) {
				if (temp[i]) {
					return Location(i); // correction (highest level first)
				}
			}
		}
	}

	return harringtonRule(addr, defects); // level-0 rule
}

Location Cell::harringtonRule(const Location& addr, const std::vector<bool>& defects) {

	// immediate exit
	if (addr == Location::C || defects[Location::C] == 0) {
		return Location::None;
	}

	if (addr == Location::L) {
		if (defects[Location::L]) {
			return Location::L;
		} else {
			return Location::R;
		}
	}

	if (addr == Location::R) {
		if (defects[Location::R]) {
			return Location::None;
		} else {
			return Location::L;
		}
	}
	return Location::None;
}

