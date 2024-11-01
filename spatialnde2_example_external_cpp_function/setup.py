import sys
import os
import os.path
import numpy as np
from setuptools import setup,Extension
from Cython.Build import cythonize
import spatialnde2 as snde

def fixup_cmake_libs(libs):
    ret_libs = []
    ret_explicit = []
    for lib in libs:
        if lib.startswith("-l"):
            ret_libs.append(lib[2:])
            pass
        elif "/" in lib or "\\" in lib:
            ret_explicit.append(lib)
            pass
        else:
            ret_libs.append(lib)
            pass
        pass
    return (ret_explicit,ret_libs)

spatialnde2_compile_definitions = open(os.path.join(os.path.dirname(snde.__file__),"compile_definitions.txt")).read().strip().split(" ")
spatialnde2_compile_include_dirs = open(os.path.join(os.path.dirname(snde.__file__),"compile_include_dirs.txt")).read().strip().split(" ")
spatialnde2_compile_library_dirs = open(os.path.join(os.path.dirname(snde.__file__),"compile_library_dirs.txt")).read().strip().split(" ")
spatialnde2_compile_libraries_cmake = open(os.path.join(os.path.dirname(snde.__file__),"compile_libraries.txt")).read().strip().split(" ")

(spatialnde2_compile_explicit_libraries,spatialnde2_compile_libraries) = fixup_cmake_libs(spatialnde2_compile_libraries_cmake)


ext_modules=cythonize(
    Extension("spatialnde2_example_external_cpp_function.scalar_multiply",
              sources=["spatialnde2_example_external_cpp_function/scalar_multiply.pyx" ],
              include_dirs=[os.path.dirname(snde.__file__)] + spatialnde2_compile_include_dirs,
              
              library_dirs=[os.path.dirname(snde.__file__)] + spatialnde2_compile_library_dirs,
              extra_compile_args = ["-O0"] + spatialnde2_compile_definitions,
              extra_link_args = spatialnde2_compile_explicit_libraries,
              libraries=["spatialnde2"]  + spatialnde2_compile_libraries,
              undef_macros = ["NDEBUG"]
              ),include_path=["spatialnde2_example_external_cpp_function"])

setup(name="spatialnde2_example_external_cpp_function",
            description="Example external c++ function for spatialnde2",
            author="Stephen D. Holland",
            url="http://thermal.cnde.iastate.edu",
            ext_modules=ext_modules,
            packages=["spatialnde2_example_external_cpp_function"])
