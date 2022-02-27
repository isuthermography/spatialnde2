from libcpp.memory cimport shared_ptr

from spatialnde2.recmath cimport math_function


cdef extern from "scalar_multiply_cpp.hpp" namespace "snde2_fn_ex" nogil:
    cdef shared_ptr[math_function] define_scalar_multiply()
    cdef shared_ptr[math_function] scalar_multiply_function
    pass
