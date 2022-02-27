cdef extern from "snde/recstore.hpp" namespace "snde" nogil:
    cdef cppclass recdatabase:
        pass

    cdef cppclass multi_ndarray_recording:
        pass

    cdef cppclass ndarray_recording_ref:
        pass

    cdef cppclass ndtyped_recording_ref[T]:
        pass    
    pass
