from libcpp.memory cimport shared_ptr

cdef extern from "snde/recmath.hpp" namespace "snde" nogil:
    cdef cppclass math_function:
        pass
    pass


cdef extern from "scalar_multiply_cpp.hpp" namespace "snde2_fn_ex" nogil:
    cdef shared_ptr[math_function] define_scalar_multiply()
    cdef shared_ptr[math_function] scalar_multiply_function
    pass
