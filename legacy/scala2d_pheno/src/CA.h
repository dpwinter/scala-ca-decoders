#ifndef CA_H_
#define CA_H_

#include "Cell.h"
#include "Location.h"
#include <vector>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class CA
{
	private:
		int age;
		int d;
		std::vector<Cell> cells;
	public:
		CA(int d_);
		void step(py::array_t<bool>& xframe, py::array_t<bool>& syndrome, double psig=0.0);
		void reset();
};

#endif
