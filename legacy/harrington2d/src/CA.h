#ifndef CA_H_
#define CA_H_

#include "Cell.h"
#include "Location.h"
#include <vector>
#include <cassert>
#include <cmath>
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class CA 
{
	private:
		int L;
		std::vector<Cell> cells;
	public:
		CA(int L);
		// ~CA();
		// void reset();
		// const Cell& getCell(int i, int j);
		void step(py::array_t<bool>& xframe, py::array_t<bool>& syndrome, double psig=0.0);
		// std::vector<Location> step(const std::vector<bool>& syndrome);
		// std::vector<Location> step(const std::vector<bool>& syndrome, double pfs, double pcs);
};

#endif
