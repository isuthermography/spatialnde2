import os
import os.path
import numpy as np
from setuptools import setup,Extension
from Cython.Build import cythonize
import spatialnde2 as snde

ext_modules=cythonize(
    Extension("spatialnde2_example_external_cpp_function.scalar_multiply",
              sources=["spatialnde2_example_external_cpp_function/scalar_multiply.pyx"],
              include_dirs=[os.path.dirname(snde.__file__)],
              
              library_dirs=[os.path.dirname(snde.__file__)],
              extra_compile_args = open(os.path.join(os.path.dirname(snde.__file__),"compile_definitions.txt")).read().strip().split(" "),
              libraries=["spatialnde2"]
              ))

setup(name="spatialnde2_example_external_cpp_function",
            description="Example external c++ function for spatialnde2",
            author="Stephen D. Holland",
            url="http://thermal.cnde.iastate.edu",
            ext_modules=ext_modules,
            packages=["spatialnde2_example_external_cpp_function"])
