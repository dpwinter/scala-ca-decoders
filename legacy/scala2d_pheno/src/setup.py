from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [ Pybind11Extension("scala", ["CA.cpp", "Cell.cpp", "ca_decoder_module_bindings.cpp"],), ]
setup(name="my_project", version="0.1", ext_modules=ext_modules, cmdclass={"build_ext": build_ext}, zip_safe=False,)

