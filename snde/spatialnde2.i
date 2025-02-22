/* SWIG interface for spatialnde2 */ 
// swig -c++ -python spatialnde2.i
  
%module(directors="1") spatialnde2

%pythonbegin %{
import sys
import copy
%}

//%pythoncode %{
//snde_error = _spatialnde2_python.snde_error
//%}

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

// Workaround for swig to understand what uint64_t aliases
typedef unsigned long long uint64_t;

/* warning suppression */
//#pragma SWIG nowarn=509,454,341


// Exception handling
// on non-win32 we only catch our own errors
#ifndef WIN32
%exception {
  try {
    $action
  } catch (const snde::snde_indexerror &serr) {
    PyErr_SetString(snde_indexerror_exc,serr.what());
    SWIG_fail;
  }catch (const snde::snde_stopiteration &serr) {
    PyErr_SetString(snde_stopiteration_exc,serr.what());
    SWIG_fail;
  } catch (const snde::snde_error &serr) {
    PyErr_SetString(snde_error_exc,serr.what());
    SWIG_fail;
  }
}
#endif

#ifdef WIN32
// on WIN32, unhandled exceptions can be very mysterious bugs
// to track down because they may just cause it to silently crash.
// On other platforms, we don't want to catch these exceptions
// because it makes it much harder to find the problems.
%exception {
  try {
    $action
  } catch (const snde::snde_indexerror &serr) {
    PyErr_SetString(snde_indexerror_exc,serr.what());
    SWIG_fail;
  } catch (const snde::snde_stopiteration &serr) {
    PyErr_SetString(snde_stopiteration_exc,serr.what());
    SWIG_fail;
  } catch (const snde::snde_error &serr) {
    PyErr_SetString(snde_error_exc,serr.what());
    SWIG_fail;
  } catch (const std::exception &serr) {
    PyErr_SetString(PyExc_RuntimeError,serr.what());
    SWIG_fail;
  }
}
#endif


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
%include "std_set.i"
%include "python/std_unordered_map.i"
%include "std_multimap.i"
%include "std_shared_ptr.i"
%include "attribute.i" 

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

%template(string_set) std::set<std::string>;

%{
#define SWIG_FILE_WITH_INIT
  //#include "snde/snde_error.hpp"
  

  static PyObject *snde_error_exc;
  static PyObject *snde_indexerror_exc;
  static PyObject *snde_stopiteration_exc;

  namespace snde {
    static std::unordered_map<unsigned,PyArray_Descr*> rtn_numpytypemap;
  };
  
%}

  %typemap(out) void * {
    $result = PyLong_FromVoidPtr($1);
  }

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

%define snde_rawaccessible(rawaccessible_class...) // the ... enables it to accept templates that have commas in int

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

    // get raw pointer
    void *raw()
    {
      return self;
    }
    
  };

%enddef


%template(BoolVector) std::vector<bool>;


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

%shared_ptr(std::vector<std::string>);
%template(StringVector) std::vector<std::string>;
//%template(StringVectorPtr) std::shared_ptr<std::vector<std::string>>;

%extend std::vector<std::string> {
  std::string __str__()
  {
    std::string strval="[\n";
    for (auto elem: *self) {
      strval += "\"" + elem+"\",\n";
    }
    strval +="]";
    return strval;
  }
  std::string __repr__()
  {
    std::string strval="[\n";
    for (auto elem: *self) {
      strval += "\""+elem+"\",\n";
    }
    strval +="]";

    return strval;
  }
}



%template(StringPair) std::pair<std::string,std::string>;
%shared_ptr(std::vector<std::pair<std::string,std::string>>);
%template(StringPairVector) std::vector<std::pair<std::string,std::string>>;

%extend std::vector<std::pair<std::string,std::string>> {
  std::string __str__()
  {
    std::string strval="[\n";
    for (auto elem: *self) {
      strval += "( \"" + elem.first + "\", \"" + elem.second + "\" ),\n";
    }
    strval +="]";

    return strval;
  }
  std::string __repr__()
  {
    std::string strval="[\n";
    for (auto elem: *self) {
      strval += "( \"" + elem.first + "\", \"" + elem.second + "\" ),\n";
    }
    strval +="]";

    return strval;
  }
}


// These are really supposed to be uint64_t but at least g++
// considers unsigned long and unsigned long long to be distinct
// types of the same size, so if longs are 64 bit better for us
// ust to use "unsigned long"
#ifdef SIZEOF_LONG_IS_8
%template(StringUnsigned64Pair) std::pair<std::string,unsigned long>;
%shared_ptr(std::vector<std::pair<std::string,unsigned long>>);
%template(StringUnsigned64PairVector) std::vector<std::pair<std::string,unsigned long>>;
%extend std::vector<std::pair<std::string,unsigned long>> {
  std::string __str__()
  {
    std::string strval="[\n";
    for (auto elem: *self) {
      strval += "  \"" + elem.first + "\" " + std::to_string(elem.second) + "\n";
    }
    strval +="]";

    return strval;
  }
  std::string __repr__()
  {
    std::string strval="[\n";
    for (auto elem: *self) {
      strval += "  \"" + elem.first + "\" " + std::to_string(elem.second) + "\n";
    }
    strval +="]";

    return strval;
  }
}


#else
%template(StringUnsigned64Pair) std::pair<std::string,unsigned long long >;
%shared_ptr(std::vector<std::pair<std::string,unsigned long long>>);
%template(StringUnsigned64PairVector) std::vector<std::pair<std::string,unsigned long long>>;
%extend std::vector<std::pair<std::string,unsigned long long>> {
  std::string __str__()
  {
    std::string strval="[\n";
    for (auto elem: *self) {
      strval += "  \"" + elem.first + "\" " + std::to_string(elem.second) + "\n";
    }
    strval +="]";

    return strval;
  }
  std::string __repr__()
  {
    std::string strval="[\n";
    for (auto elem: *self) {
      strval += "  \"" + elem.first + "\" " + std::to_string(elem.second) + "\n";
    }
    strval +="]";

    return strval;
  }
}
#endif


//#ifdef SIZEOF_LONG_IS_8
// regular unsigned are different from Unsigned64
%template(StringUnsignedPair) std::pair<std::string,unsigned>;
//%shared_ptr(std::pair<std::string,unsigned>);
%template(StringUnsignedPairVector) std::vector<std::pair<std::string,unsigned>>;
//#else
//// regular unsigned are the same as Unsigned64
//%pythoncode %{
//  StringUnsignedPair=StringUnsigned64Pair
//  StringUnsignedPairVector=StringUnsigned64PairVector
//%}
//#endif

%template(shared_ptr_string) std::shared_ptr<std::string>;




%include "snde_types.i"
%include "geometry_types.i"
%include "snde_error.i"
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
%include "graphics_recording.i"
%include "graphics_storage.i"
%include "recmath_parameter.i"
%include "recmath_compute_resource.i"
%include "recmath.i"
%include "recmath_cppfunction.i"
%include "cached_recording.i"
%include "notify.i"
%include "arrayposition.i"
%include "normal_calculation.i"
%include "inplanemat_calculation.i"
%include "projinfo_calculation.i"
%include "boxes_calculation.i"
%include "area_calculation.i"
%include "averaging_downsampler.i"
%include "mean.i"
%include "NumPy_BGRtoRGBA.i"
%include "NumPy_BGRtoGray16.i"
%include "batched_live_accumulator.i"
%include "accumulate_once.i"
%include "nd_accumulate_once.i"
%include "project_onto_parameterization.i"
%include "offset_calibration.i"
%include "quaternion.i"
%include "kdtree.i"
%include "orientation_product.i"
%include "rendermode.i"
%include "display_requirements.i"
%include "snde_qt.i"
%include "qt_osg_compositor.i"
%include "qtrecviewer.i"
%include "qtrecviewer_support.i"
%include "recstore_setup.i"
%include "x3d.i"
%include "utils.i"
%include "ande_file.i"
%include "polynomial_transform.i"
%include "geometry_processing.i"
%include "recstore_transaction_manager.i"
%include "dexela2923_image_transform.i"
%include "arithmetic.i"

%include "bad_pixel_correction.i"

#ifdef SNDE_OPENCL
%include "opencl_utils.i"
%include "recstore_setup_opencl.i"
%include "openclcachemanager.i"
%include "recmath_compute_resource_opencl.i"
#endif

 /*
 // additional support needed for VectorOfStringOrientationPairs
 // template below to avoid the problem listed at
 // https://stackoverflow.com/questions/38404806/error-c2039-type-name-is-not-a-member-of-of-swigtraitsbar
%{
  namespace swig {
    template <> struct traits<_snde_orientation3> {
      typedef pointer_category category;
      static const char *type_name() { return "_snde_orientation3"; }
    };
  };
%}
*/
%template(StringOrientationPair) std::pair<std::string,snde_orientation3>;

%template(VectorOfStringOrientationPairs) std::vector<std::pair<std::string,snde_orientation3>>;


// Instantiate templates for shared ptrs
//%shared_ptr(snde::openclcachemanager);



%init %{
  import_array();

  snde_error_exc = PyErr_NewException("spatialnde2.snde_error",NULL,NULL);
  Py_INCREF(snde_error_exc);
  PyModule_AddObject(m,"snde_error",snde_error_exc);

  snde_indexerror_exc = PyErr_NewException("spatialnde2.snde_indexerror",PyExc_IndexError,NULL);
  Py_INCREF(snde_indexerror_exc);
  PyModule_AddObject(m,"snde_indexerror",snde_indexerror_exc);

  snde_stopiteration_exc = PyErr_NewException("spatialnde2.snde_stopiteration",PyExc_StopIteration,NULL);
  Py_INCREF(snde_stopiteration_exc);
  PyModule_AddObject(m,"snde_stopiteration",snde_stopiteration_exc);
  
  PyObject *Globals = PyDict_New(); // for creating numpy dtypes
  PyObject *NumpyModule = PyImport_ImportModule("numpy");
  if (!NumpyModule) {
    throw snde::snde_error("Error importing numpy");
  }
  PyObject *np_dtype = PyObject_GetAttrString(NumpyModule,"dtype");
  PyDict_SetItemString(Globals,"np",NumpyModule);
  PyDict_SetItemString(Globals,"dtype",np_dtype);

  // ***!!!***!!! Need to regularize these dtypes with the ones defined in geometry_types.i that
  // end up as nt_snde_xxxxx in the generated Python.

  // WARNING:
  // Errors in this dtype initialization can trigger mysterious errors
  // when spatialnde2 is imported, e.g.:
  //  SystemError: initialization of _spatialnde2_python raised unreported exception
  
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
  // SNDE_RTN_SNDE_RGBA
  PyObject *rgba32_dtype = PyRun_String("dtype([('r', np.uint8), ('g', np.uint8), ('b',np.uint8),('a',np.uint8)])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_RGBA,(PyArray_Descr *)rgba32_dtype);
  
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

  PyObject *coord3_int16_dtype = PyRun_String("dtype([('coord', np.int16, 3), ])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_COORD3_INT16,(PyArray_Descr *)coord3_int16_dtype);

  // SNDE_RTN_INDEXVEC through SNDE_RTN_ASSEMBLY_RECORDING not applicable

  // ***!!! Still need numpy dtypes for most graphics arrays!!!***
#ifdef SNDE_DOUBLEPREC_COORDS
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_COORD,PyArray_DescrFromType(NPY_FLOAT64));
  
  PyObject *coord3_dtype = PyRun_String("dtype([('coord', np.float64, 3), ])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_COORD3,(PyArray_Descr *)coord3_dtype);
  PyObject *coord4_dtype = PyRun_String("dtype([('coord', np.float64, 4), ])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_COORD4,(PyArray_Descr *)coord4_dtype);
  PyObject *coord2_dtype = PyRun_String("dtype([('coord', np.float64, 2), ])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_COORD2,(PyArray_Descr *)coord2_dtype);
  PyObject *cmat23_dtype = PyRun_String("dtype([('row', dtype([('coord', np.float64, 3), ]) , 2), ])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_CMAT23,(PyArray_Descr *)cmat23_dtype);
  PyObject *orientation3_dtype = PyRun_String("dtype([('quat', np.float64,4),('offset', np.float64, 4),  ])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_ORIENTATION3,(PyArray_Descr *)orientation3_dtype);
#else
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_COORD,PyArray_DescrFromType(NPY_FLOAT32));
  
  PyObject *coord3_dtype = PyRun_String("dtype([('coord', np.float32, 3), ])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_COORD3,(PyArray_Descr *)coord3_dtype);
  PyObject *coord4_dtype = PyRun_String("dtype([('coord', np.float32, 4), ])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_COORD4,(PyArray_Descr *)coord4_dtype);
  PyObject *coord2_dtype = PyRun_String("dtype([('coord', np.float32, 2), ])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_COORD2,(PyArray_Descr *)coord2_dtype);
  PyObject *cmat23_dtype = PyRun_String("dtype([('row', dtype([('coord', np.float32, 3), ]) , 2), ])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_CMAT23,(PyArray_Descr *)cmat23_dtype);
  PyObject *orientation3_dtype = PyRun_String("dtype([('quat', np.float32,4),('offset', np.float32, 4),  ])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_ORIENTATION3,(PyArray_Descr *)orientation3_dtype);

  // !!!!!***** This needs to be adjusted to consider the various possible sizes for snde_index and snde_coord
  PyObject *snde_part_dtype = PyRun_String("dtype([('firstboundary',np.uint64),('numboundaries',np.uint64),('first_topo',np.uint64),('num_topo',np.uint64),('first_topoidx',np.uint64),('num_topoidxs',np.uint64),('firstface',np.uint64),('num_faces',np.uint64),('firsttri',np.uint64),('numtris',np.uint64),('firstedge',np.uint64),('numedges',np.uint64),('firstvertex',np.uint64),('numvertices',np.uint64),('first_vertex_edgelist',np.uint64),('num_vertex_edgelist',np.uint64),('firstbox',np.uint64),('numboxes',np.uint64),('firstboxpoly',np.uint64),('numboxpolys',np.uint64),('firstboxnurbssurface',np.uint64),('numboxnurbssurfaces',np.uint64),('pivot_point',np.float32,3),('length_scale',np.float32),('bounding_box',[('min',np.float32,3),('max',np.float32,3)]),('first_vertex_kdnode',np.uint64),('num_vertex_kdnodes',np.uint64),('first_triarea',np.uint64),('first_vertarea',np.uint64),('reserved',np.uint8,12),('solid',np.bool_),('has_triangledata',np.bool_),('has_curvatures',np.bool_),('pad1',np.uint8),('pad2',np.uint8,4)])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_PART,(PyArray_Descr *)snde_part_dtype);

  PyObject *snde_parameterization_dtype = PyRun_String("dtype([('first_uv_topo',np.uint64),('num_uv_topos',np.uint64),('first_uv_topoidx',np.uint64),('num_uv_topoidxs',np.uint64),('firstuvtri',np.uint64),('numuvtris',np.uint64),('firstuvface',np.uint64),('numuvfaces',np.uint64),('firstuvedge',np.uint64),('numuvedges',np.uint64),('firstuvvertex',np.uint64),('numuvvertices',np.uint64),('first_uv_vertex_edgelist',np.uint64),('num_uv_vertex_edgelist',np.uint64),('firstuvpatch',np.uint64),('numuvpatches',np.uint64),('reserved',np.uint64,16)])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_PARAMETERIZATION,(PyArray_Descr *)snde_parameterization_dtype);

  
#endif // SNDE_DOUBLEPREC_COORDS


  PyObject *kdnode_dtype = PyRun_String("dtype([('cutting_vertex',np.uint64),('left_subtree',np.uint64),('right_subtree',np.uint64)])",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_KDNODE,(PyArray_Descr *)kdnode_dtype);
  

  PyObject *complexfloat32_dtype = PyRun_String("dtype(np.complex64)",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_COMPLEXFLOAT32,(PyArray_Descr *)complexfloat32_dtype);

  PyObject *complexfloat64_dtype = PyRun_String("dtype(np.complex128)",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_COMPLEXFLOAT64,(PyArray_Descr *)complexfloat64_dtype);


    PyObject *snde_imagedata_dtype = PyRun_String("dtype(np.float32)",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_IMAGEDATA,(PyArray_Descr *)snde_imagedata_dtype);
  
  PyObject *snde_compleximagedata_dtype = PyRun_String("dtype(np.complex64)",Py_eval_input,Globals,Globals);
  snde::rtn_numpytypemap.emplace(SNDE_RTN_SNDE_COMPLEXIMAGEDATA,(PyArray_Descr *)snde_compleximagedata_dtype);
  

  Py_DECREF(NumpyModule);
  Py_DECREF(np_dtype);
  Py_DECREF(Globals);

%}

%pythoncode %{
  try:
    import importlib.metadata
    __version__ = importlib.metadata.version("spatialnde2")
    pass
  except ImportError:
    # Python3.7 may not have importlib.metadata
    pass
  
%}

