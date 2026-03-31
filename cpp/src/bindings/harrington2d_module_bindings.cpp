#include <cassert>
#include <pybind11/pybind11.h>

#include "models/harrington2d.h"

namespace py = pybind11;

PYBIND11_MODULE(harrington2d, m)
{
    py::class_<Harrington2D>(m, "CA")
        .def(py::init<int, unsigned int>(), py::arg("d"), py::arg("seed") = 12345)
        .def("step", &Harrington2D::step, py::arg("xframe"), py::arg("syndrome"), py::arg("psig") = 0.0);
}
