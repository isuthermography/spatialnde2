/* SWIG interface for spatialnde2 */
// swig -c++ -python spatialnde2.i

%module spatialnde2

%pythonbegin %{
import sys
%}


/* warning suppression */
//#pragma SWIG nowarn=509,454,341


// Perform includes
%{
// C includes
#include <assert.h>
#include <string.h>
#include <cstdarg>

// C++ requirements
#include <functional>
%}

//%include "stl.i"
%include "stdint.i"
%include "std_string.i"
%include "std_vector.i"
%include "std_map.i"
%include "std_deque.i"
%include "std_except.i"
%include "std_pair.i"
%include "python/std_unordered_map.i"
%include "std_multimap.i"
%include "std_shared_ptr.i"
 

//numpy
%include "numpy.i"

// define numpy arrays of size_t
 // diagnose size mismatch on compile
%{

  static_assert(sizeof(size_t) == sizeof(unsigned long));
%}

%numpy_typemaps(size_t,NPY_ULONG,size_t);
%numpy_typemaps(cl_event,NPY_UINTP,size_t);

%begin %{
  #include <numpy/npy_common.h>
  #include <numpy/ndarrayobject.h>
%}

%init %{
  import_array();
%}


// exception handling
%include "exception.i"

 
 /*%exception{
	try {
		$action
	}
	catch (const std::exception& e) {
		SWIG_exception(SWIG_RuntimeError, e.what());
	}
	}*/


%{
#include <vector>
#include <map>
#include <condition_variable>
#include <deque>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <tuple>


#include "snde_error.hpp"
%}

%include "geometry_types.i"
%include "memallocator.i"
%include "lock_types.i"
%include "rangetracker.i"
%include "allocator.i"
%include "arraymanager.i"
%include "pywrapper.i"
%include "lockmanager.i"
%include "geometrydata.i"
%include "geometry.i"

#ifdef SNDE_OPENCL
%include "opencl_utils.i"
%include "openclcachemanager.i"
#endif

#ifdef SNDE_X3D
%include "x3d.i"
#endif

%{
//#include "memallocator.hpp"

//#include "geometry_types_h.h"
//#include "testkernel_c.h"

%}



// Instantiate templates for shared ptrs
//%shared_ptr(snde::openclcachemanager);
%template(StringVector) std::vector<std::string>;
