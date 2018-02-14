//%shared_ptr(snde::memallocator);
//%shared_ptr(snde::cmemallocator);

%{
  
#include "geometry_types.h"
%}

%pythonbegin %{
import ctypes
%}


#ifdef USE_OPENCL

typedef cl_double snde_coord;
typedef cl_float snde_imagedata;
typedef cl_ulong snde_index;
typedef cl_uint snde_shortindex;
typedef cl_long snde_ioffset;
typedef cl_char snde_bool;

#else
typedef double snde_coord;
typedef float snde_imagedata;
typedef uint64_t snde_index;
typedef uint32_t snde_shortindex;
typedef int64_t snde_ioffset;
typedef char snde_bool;
#endif /* USE_OPENCL*/

//#define SNDE_INDEX_INVALID (~((snde_index)0))
%pythoncode %{
SNDE_INDEX_INVALID=(long(1)<<64)-1

ct_snde_coord = ctypes.c_double
ct_snde_imagedata=ctypes.c_float
ct_snde_index=ctypes.c_uint64
ct_snde_shortindex=ctypes.c_uint32
ct_snde_ioffset=ctypes.c_int64
ct_snde_bool=ctypes.c_char

%}



