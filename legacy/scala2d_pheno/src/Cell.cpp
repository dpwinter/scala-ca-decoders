#include "Cell.h"

Cell::Cell()
{
	reset();
}

void Cell::reset()
{
	std::fill(defects.begin(), defects.end(), 0);
	std::fill(sigs.begin(), sigs.end(), 0);
	std::fill(nsigs.begin(), nsigs.end(), 0);
}

void Cell::setNeighbors(std::array<Cell*,4>& neighbors_) 
{
	neighbors = std::move(neighbors_);
}

void Cell::setDefect(bool value) 
{
	defects[Location::C] = value;
}

void Cell::acquire() 
{
	for (int i=0; i<4; i++) {
		defects[i] = neighbors[i]->defects[Location::C]; // copy defects of neighbors
	}
}

void Cell::broadcast() 
{
	// Two directions independent SCALA1D.
	if (defects[Location::C]) {
		if (!sigs[Location::W] && !sigs[Location::E]) {
			sigs[Location::W] = 1;
			sigs[Location::E] = 1;
		}
		if (!sigs[Location::N] && !sigs[Location::S]) {
			sigs[Location::N] = 1;
			sigs[Location::S] = 1;
		}
	}
}

void Cell::propagate() 
{
	for (int i=0; i<4; i++) {
		nsigs[3-i] = neighbors[i]->sigs[3-i];
	}
}

void Cell::update(double psig) 
{
	int num_nsigs = int(std::accumulate(nsigs.begin(), nsigs.end(), 0.0));

	if (!defects[Location::C] && num_nsigs > 1) {
		for (int i=0; i<4; i++)
			sigs[i] = nsigs[3-i]; // reflect
	}
	else {
		for (int i=0; i<4; i++) {
			sigs[i] = nsigs[i]; // transmit
		}
	}

	// sig noise
	for (int i=0; i<4; i++)
	{
		if (Random::getError(psig))
			sigs[i] = sigs[i] ^ 1;
	}

}

Location Cell::rule() 
{
	if (defects[Location::C]) {

		int num_sigs = int(std::accumulate(sigs.begin(), sigs.end(), 0.0));
		bool is_isolated_defect = std::accumulate(defects.begin(), defects.end(), 0.0) == 1.0;

		// NN rule (with preference)
		if (defects[Location::E])
			return Location::E;
		else if (defects[Location::N])
			return Location::N ;

		// FF rules
		else if (is_isolated_defect) { // new: isolated defect
			if (num_sigs == 1) {
				if (sigs[Location::W])
					return Location::E;
				else if (sigs[Location::E])
					return Location::W;
				else if (sigs[Location::N])
					return Location::S;
				else if (sigs[Location::S])
					return Location::N;
			}
			else if (num_sigs == 2) {
				if (sigs[Location::W] && sigs[Location::N])
					return Location::S;
				else if (sigs[Location::E] && sigs[Location::N])
					return Location::W;
			}
			else if (num_sigs == 3) {
				if (sigs[Location::W] && sigs[Location::N] && sigs[Location::E])
					return Location::S;
				else if (sigs[Location::S] && sigs[Location::N] && sigs[Location::E])
					return Location::W;
				else if (sigs[Location::S] && sigs[Location::W] && sigs[Location::E])
					return Location::N;
				else if (sigs[Location::S] && sigs[Location::W] && sigs[Location::N])
					return Location::E;
			}
		}
	}
	return Location::None;
}
