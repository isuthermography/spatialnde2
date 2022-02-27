from libcpp.memory cimport shared_ptr

cdef extern from "snde/lock_types.hpp" namespace "snde" nogil:
    cdef cppclass rwlock_token_set_content:
        pass
    ctypedef shared_ptr[rwlock_token_set_content] rwlock_token_set
    pass
