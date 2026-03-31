#include <cassert>
#include <pybind11/pybind11.h>

#include "models/scala2d.h"

namespace py = pybind11;

PYBIND11_MODULE(scala2d, m)
{
    py::class_<Scala2D>(m, "CA")
        .def(py::init<int, unsigned int>(), py::arg("d"), py::arg("seed") = 12345)
        .def("reset", &Scala2D::reset)
        .def("step", &Scala2D::step, py::arg("xframe"), py::arg("syndrome"), py::arg("psig") = 0.0);
}
