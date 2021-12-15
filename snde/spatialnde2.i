/* SWIG interface for spatialnde2 */
// swig -c++ -python spatialnde2.i

%module spatialnde2

%pythonbegin %{
import sys
%}

%pythoncode %{
snde_error = _spatialnde2_python.snde_error
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


// Exception handling
%exception {
  try {
    $action
  } catch (const snde::snde_error &serr) {
    PyErr_SetString(snde_error_exc,serr.what());
    SWIG_fail;
  }
}



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


#ifdef SIZEOF_SIZE_T_IS_8
#ifdef SIZEOF_LONG_IS_8
%numpy_typemaps(size_t, NPY_ULONG, size_t);
// define numpy arrays of size_t
// diagnose size mismatch on compile
%{
  static_assert(sizeof(size_t) == sizeof(unsigned long), "Mismatch of size_t");
%}

#else
%numpy_typemaps(size_t, NPY_ULONGLONG, size_t);
// define numpy arrays of size_t
// diagnose size mismatch on compile
%{
  static_assert(sizeof(size_t) == sizeof(unsigned long long), "Mismatch of size_t");
%}


#endif
#else
/* assume sizeof(size_t)==4) */
#ifdef SIZEOF_LONG_IS_8
%numpy_typemaps(size_t, NPY_UINT, size_t);
// define numpy arrays of size_t
// diagnose size mismatch on compile
%{
  static_assert(sizeof(size_t) == sizeof(unsigned int), "Mismatch of size_t");
%}
#else
%numpy_typemaps(size_t, NPY_ULONG, size_t);

// define numpy arrays of size_t
// diagnose size mismatch on compile
%{
  static_assert(sizeof(size_t) == sizeof(unsigned long), "Mismatch of size_t");
%}

#endif
#endif

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

%}

%{
  namespace snde {
    static std::shared_ptr<std::string> shared_string(std::string input)
    {
      return std::make_shared<std::string>(input);
    }
  };
%}


namespace snde {
  static std::shared_ptr<std::string> shared_string(std::string input);
  

};

%template(shared_string_vector) std::vector<std::shared_ptr<std::string>>;

%{
#define SWIG_FILE_WITH_INIT
#include "snde/snde_error.hpp"
  

  static PyObject *snde_error_exc;

  namespace snde {
    static std::unordered_map<unsigned,PyArray_Descr*> rtn_numpytypemap;
  };
  
%}


// The macro snde_rawaccessible()  marks a C++ class
// (that must also be marked with %shared_ptr()) as
// accessible through short-term raw integer references to the
// shared_ptr. The snde_rawaccessible() declaration
// adds a method produce_raw_shared_ptr() which returns
// a python long which is the address of a new shared_ptr
// object that needs to be destroyed by
// consume_raw_shared_ptr().
//
// The snde_rawaccessible() declaration also adds a
// classmethod consume_raw_shared_ptr( python long )
// which creates a new Python reference to the
// original object, destroying the meaning of the
// raw shared pointer in the process (deleting the
// shared_ptr object that was created by
// produce_raw_shared_ptr(). 

// In addtion, the snde_rawaccessible() declaration also adds
// another classmethod  from_raw_shared_ptr( python long ) which
// takes an address of a shared_ptr object and creates
// and returns a new Python wrapper with a new shared_ptr
// that is initialized from the (shared pointer the
// Python long points at). It does not affect its input
// shared pointer object, which could have been created
// independently or returned from produce_raw_shared_ptr().
// (but if the latter, you still ned to make sure the
// pointer value gets consumed at some point). 
//
// These methods make the object wrapping interoperable
// with other means of accessing the same underlying
// objects.
//
// For example if you are coding in Cython and have
// created a math_function and want to return a
// SWIG-wrapped version:
//
// from libc.stdint cimport uintptr_t
// cdef shared_ptr[math_function] func
// # (assign func here)
// return spatialnde2.math_function.from_raw_shared_ptr(<uintptr_t>&func)
//
// Likewise if you have a math_function from swig
// and want a Cython cdef:
//
// from cython.operator cimport dereference as deref
// raw_shared_ptr = swigwrapped_math_function.produce_raw_shared_ptr()
// cdef shared_ptr[math_function] func = deref(<shared_ptr[math_function]*>raw_shared_ptr)
// spatialnde2.math_function.consume_raw_shared_ptr(raw_shared_ptr)  # or you could delete (<shared_ptr[math_function]*>raw_shared_ptr) 

%{
template <typename T>
  class uint_raw_shared_ptr {
  public:
    uintptr_t rsp;
  };
%}

%define snde_rawaccessible(rawaccessible_class)

// This typemap takes allows passing an integer which is the address
// of a C++ shared_ptr structure to a raw_shared_ptr parameter.
// It then initialized a C++ shared_ptr object from that shared_ptr
// (we can't just use the typemap matching to make this work
// because if we do that, the swig %shared_ptr() typemap overrides us)
%typemap(in) std::shared_ptr< rawaccessible_class > raw_shared_ptr {
  void *rawptr = PyLong_AsVoidPtr($input);
  if (!rawptr) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_ValueError,"null pointer");
    }
    SWIG_fail; 
  }
  $1 = *((std::shared_ptr<$1_ltype::element_type> *)rawptr);
}

%typemap(in) std::shared_ptr< rawaccessible_class > consumable_raw_shared_ptr {
  void *rawptr = PyLong_AsVoidPtr($input);
  if (!rawptr) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_ValueError,"null pointer");
    }
    SWIG_fail; 
  }
  $1 = *((std::shared_ptr<$1_ltype::element_type> *)rawptr);
  delete ((std::shared_ptr<$1_ltype::element_type> *)rawptr);
}


// This pythonappend does the work for produce_raw_shared_ptr() below,
// extracting the pointer to the shared_ptr object from the
// swig wrapper. 
//%feature("pythonappend") rawaccessible_class::produce_raw_shared_ptr() %{
//  val = self.this.ptr
//%}

// out typemap used by produce_raw_shared_ptr that steals
// tempshared1 or  smartarg1 which should be a std::shared_ptr<T> and a pointer to a std::shared_ptr<T> respectively
 // (This is a bit hacky) 
%typemap(out) uint_raw_shared_ptr< rawaccessible_class > {
  if (tempshared1) {
    $result = PyLong_FromVoidPtr(new std::shared_ptr< rawaccessible_class > (tempshared1));
  } else {
    $result = PyLong_FromVoidPtr(new std::shared_ptr< rawaccessible_class > (*smartarg1));
  } 
}

// This extension provides the from_raw_shared_ptr(), consume_raw_shared_ptr(), and produce_raw_shared_ptr() methods
  %extend rawaccessible_class {
    static std::shared_ptr< rawaccessible_class > from_raw_shared_ptr(std::shared_ptr< rawaccessible_class > raw_shared_ptr)
    {
      return raw_shared_ptr;
    }

    static std::shared_ptr< rawaccessible_class > consume_raw_shared_ptr(std::shared_ptr< rawaccessible_class > consumable_raw_shared_ptr)
    {
      return consumable_raw_shared_ptr;
    }

    uint_raw_shared_ptr< rawaccessible_class > produce_raw_shared_ptr()
    {
      return uint_raw_shared_ptr< rawaccessible_class >{0}; // actual work done by the uint_raw_shared_ptr_t out typemap, above
    }

  };

%enddef


// (Old specifc implementation of the above general implementation)
//%typemap(in) std::shared_ptr<snde::math_function> raw_shared_ptr {
//  void *rawptr = PyLong_AsVoidPtr($input);
//  if (!rawptr) {
//    if (!PyErr_Occurred()) {
//      PyErr_SetString(PyExc_ValueError,"null pointer");
//    }
//    SWIG_fail; 
//  }
//  $1 = *((std::shared_ptr<snde::math_function> *)rawptr);
// }

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
%include "recmath_parameter.i"
%include "recmath_compute_resource.i"
%include "recmath.i"
%include "recmath_cppfunction.i"
%include "notify.i"
%include "recstore_setup.i"

#ifdef SNDE_OPENCL
%include "opencl_utils.i"
%include "recstore_setup_opencl.i"
 //%include "openclcachemanager.i"
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

%template(shared_ptr_string) std::shared_ptr<std::string>;



%init %{
  import_array();

  snde_error_exc = PyErr_NewException("spatialnde2.snde_error",NULL,NULL);
  Py_INCREF(snde_error_exc);
  PyModule_AddObject(m,"snde_error",snde_error_exc);
  
  PyObject *Globals = PyDict_New(); // for creating numpy dtypes
  PyObject *NumpyModule = PyImport_ImportModule("numpy");
  if (!NumpyModule) {
    throw snde::snde_error("Error importing numpy");
  }
  PyObject *np_dtype = PyObject_GetAttrString(NumpyModule,"dtype");
  PyDict_SetItemString(Globals,"np",NumpyModule);
  PyDict_SetItemString(Globals,"dtype",np_dtype);

  // SNDE_RTN_UNASSIGNED: Not valid for Numpy  
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
  //  SNDE_RTN_RECORDING_REF not applicable

  PyObject *coord3_int16_dtype = PyRun_String("dtype([('x', np.int16), ('y', np.int16), ('z',np.int16)])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_COORD3_INT16,(PyArray_Descr *)coord3_int16_dtype);

  // SNDE_RTN_INDEXVEC through SNDE_RTN_ASSEMBLY_RECORDING not applicable

  // ***!!! Still need numpy dtypes for graphics arrays!!!***

  Py_DECREF(NumpyModule);
  Py_DECREF(np_dtype);
  Py_DECREF(Globals);

%}

