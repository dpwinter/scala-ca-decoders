#include <pybind11/pybind11.h>
#include "CA.h"

namespace py = pybind11;

PYBIND11_MODULE(harrington, m) {
	py::class_<CA>(m, "CA")
		.def(py::init<int>(), py::arg("d"))
		.def("step", &CA::step, py::arg("xframe"), py::arg("syndrome"), py::arg("psig")=0.0);
}
