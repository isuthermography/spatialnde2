# distutils: language = c++

from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr
from libc.stdint cimport uintptr_t

import spatialnde2 as snde

cdef extern from "snde/recmath.hpp" namespace "snde" nogil:
    cdef cppclass math_function:
        pass
    pass


cdef extern from "scalar_multiply_cpp.hpp" namespace "snde2_fn_ex" nogil:
    cdef shared_ptr[math_function] define_scalar_multiply()
    pass

# Create the Cython shared pointer
cdef shared_ptr[math_function] snde2_example_scalar_multiply_function = define_scalar_multiply()

# Create the swig-wrapped object by wrapping the Cython shared pointer
scalar_multiply_function = snde.math_function.from_raw_shared_ptr(<uintptr_t>&snde2_example_scalar_multiply_function)


