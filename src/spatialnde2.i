/* SWIG interface for spatialnde2 */
// swig -c++ -python spatialnde2.i

%module spatialnde2

%pythonbegin %{
import sys
%}
 

// Workaround for swig to understand what size_t aliases
#ifdef SIZEOF_SIZE_T_IS_8
#ifdef SIZEOF_LONG_IS_8
typedef unsigned long size_t;
#else
typedef unsigned long long size_t;
#endif
#else
/* assume sizeof(size_t)==4) */
#ifdef SIZEOF_LONG_IS_8
typedef unsigned size_t;
#else
typedef unsigned long size_t;
#endif
#endif


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
%include "std_list.i"
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

namespace snde {
  static std::unordered_map<unsigned,PyArray_Descr*> rtn_numpytypemap;
};
 
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
 //%include "infostore_or_component.i"
%include "geometrydata.i"
 //%include "geometry.i"
%include "metadata.i"
%include "recording.i"
%include "recdb_paths.i"
%include "recstore_storage.i"
%include "recstore.i"
%include "recmath_compute_resource.i"
%include "recmath.i"
%include "recmath_cppfunction.i"

#ifdef SNDE_OPENCL
%include "opencl_utils.i"
%include "openclcachemanager.i"
#endif

 //#ifdef SNDE_X3D
 //%include "x3d.i"
 //#endif

%{
//#include "memallocator.hpp"

//#include "geometry_types_h.h"
//#include "testkernel_c.h"

%}



// Instantiate templates for shared ptrs
//%shared_ptr(snde::openclcachemanager);
%template(StringVector) std::vector<std::string>;


%init %{
  import_array();

  PyObject *Globals = PyDict_New(); // for creating numpy dtypes
  PyObject *NumpyModule = PyImport_ImportModule("numpy");
  if (!NumpyModule) {
    throw snde::snde_error("Error importing numpy");
  }
  PyObject *np_dtype = PyObject_GetAttrString(NumpyModule,"dtype");
  PyDict_SetItemString(Globals,"np",NumpyModule);
  PyDict_SetItemString(Globals,"dtype",np_dtype);

  
  snde::rtn_numpytypemap.emplace(SNDE_RTN_FLOAT32,PyArray_DescrFromType(NPY_FLOAT32));
  snde::rtn_numpytypemap.emplace(SNDE_RTN_FLOAT64,PyArray_DescrFromType(NPY_FLOAT64));
  snde::rtn_numpytypemap.emplace(SNDE_RTN_FLOAT16,PyArray_DescrFromType(NPY_FLOAT16));
  snde::rtn_numpytypemap.emplace(SNDE_RTN_UINT64,PyArray_DescrFromType(NPY_UINT64));
  snde::rtn_numpytypemap.emplace(SNDE_RTN_INT64,PyArray_DescrFromType(NPY_INT64));
  snde::rtn_numpytypemap.emplace(SNDE_RTN_UINT32,PyArray_DescrFromType(NPY_UINT32));
  snde::rtn_numpytypemap.emplace(SNDE_RTN_INT32,PyArray_DescrFromType(NPY_INT32));
  snde::rtn_numpytypemap.emplace(SNDE_RTN_UINT16,PyArray_DescrFromType(NPY_UINT16));
  snde::rtn_numpytypemap.emplace(SNDE_RTN_INT16,PyArray_DescrFromType(NPY_INT16));
  snde::rtn_numpytypemap.emplace(SNDE_RTN_UINT8,PyArray_DescrFromType(NPY_UINT8));
  snde::rtn_numpytypemap.emplace(SNDE_RTN_INT8,PyArray_DescrFromType(NPY_INT8));
  // SNDE_RTN_RGBA32
  PyObject *rgba32_dtype = PyRun_String("dtype([('r', np.uint8), ('g', np.uint8), ('b',np.uint8),('a',np.uint8)])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_RGBA32,(PyArray_Descr *)rgba32_dtype);
  
  snde::rtn_numpytypemap.emplace(SNDE_RTN_COMPLEXFLOAT32,PyArray_DescrFromType(NPY_COMPLEX64));
  snde::rtn_numpytypemap.emplace(SNDE_RTN_COMPLEXFLOAT64,PyArray_DescrFromType(NPY_COMPLEX128));
  // SNDE_RTN_COMPLEXFLOAT16
  PyObject *complexfloat16_dtype = PyRun_String("dtype([('real', np.float16), ('imag', np.float16) ])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_COMPLEXFLOAT16,(PyArray_Descr *)complexfloat16_dtype);

  // SNDE_RTN_RGBD64
  PyObject *rgbd64_dtype = PyRun_String("dtype([('r', np.uint8), ('g', np.uint8), ('b',np.uint8),('a',np.uint8),('d',np.float32)])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_RGBD64,(PyArray_Descr *)rgbd64_dtype);

  //  SNDE_RTN_STRING not applicable
  //  SNDE_RTN_RECORDING not applicable

  Py_DECREF(NumpyModule);
  Py_DECREF(np_dtype);
  Py_DECREF(Globals);

%}

