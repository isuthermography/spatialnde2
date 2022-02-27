from libc.stdint cimport uintptr_t,uint64_t

cdef extern from "snde/geometry_types.h" nogil:
    ctypedef float snde_coord;
    ctypedef uint64_t snde_index
   

    cdef struct _snde_coord3:
        snde_coord coord[3]
        pass
    ctypedef _snde_coord3 snde_coord3

    cdef struct _snde_coord4:
        snde_coord coord[4]
        pass
    ctypedef _snde_coord4 snde_coord4
    
    cdef struct _snde_orientation3: 
        snde_coord4 offset
        snde_coord4 quat
        pass
    
    ctypedef _snde_orientation3 snde_orientation3


    cdef struct snde_kdnode:
        snde_index cutting_vertex
        snde_index left_subtree
        snde_index right_subtree
        pass

    pass

