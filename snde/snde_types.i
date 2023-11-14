
%{

  #include "snde/snde_types.h"
  
%}

%pythonbegin %{
import ctypes
import numpy as np
%}



typedef uint32_t snde_shortindex;
typedef char snde_bool;

  // Don't specify 64 bit integers in terms of int64_t/uint64_t to work around
  // https://github.com/swig/swig/issues/568
  //typedef uint64_t snde_index;
  //typedef int64_t snde_ioffset;
#ifdef SIZEOF_LONG_IS_8
  typedef unsigned long snde_index;
  typedef long snde_ioffset;
#else
  typedef unsigned long long snde_index;
  typedef long long snde_ioffset;
#endif

//%shared_ptr(std::vector<snde_index>);


%template(snde_index_vector) std::vector<snde_index>;

  typedef struct _snde_complexfloat32 {
    snde_float32 real;
    snde_float32 imag;
  } snde_complexfloat32;

#if (!defined(__OPENCL_VERSION__) || defined(SNDE_OCL_HAVE_DOUBLE))
  typedef struct _snde_complexfloat64 {
    snde_float64 real;
    snde_float64 imag;
  } snde_complexfloat64;
#endif // (!defined(__OPENCL_VERSION__) || defined(SNDE_OCL_HAVE_DOUBLE))
  
#ifdef SNDE_HAVE_FLOAT16
  typedef struct _snde_complexfloat16 {
    snde_float16 real;
    snde_float16 imag;
  } snde_complexfloat16;

#endif // SNDE_HAVE_FLOAT16




%typemap(in) snde_bool {
  $1 = (snde_bool)PyObject_IsTrue($input);
}

%typemap(out) snde_bool {
  if ($1) {
    $result = Py_True;
    Py_INCREF($result);
  } else {
    $result = Py_False;
    Py_INCREF($result);
  }
}


%typemap(typecheck,precedence=SWIG_TYPECHECK_INTEGER) snde_index  {
  $1 = PyInt_Check($input);
#if PY_VERSION_HEX < 0x03000000
  if (!$1) {
    $1=PyLong_Check($input);  
  }  
#endif
  if (!$1) {
    PyObject *numbers=NULL;
    PyObject *numbersIntegral;
    numbers = PyImport_ImportModule("numbers");
    numbersIntegral=PyObject_GetAttrString(numbers,"Integral");
    if (PyObject_IsInstance($input,numbersIntegral)==1) {
      $1 = true;
    }
    Py_XDECREF(numbers);
  }
} 

%typemap(in) snde_index (PyObject *builtins_mod=NULL,PyObject *LongTypeObj,PyObject *LongObj=NULL)  {
  if (PyLong_Check($input)) {
    $1=PyLong_AsUnsignedLongLong($input);
  }
#if PY_VERSION_HEX < 0x03000000
  else if (PyInt_Check($input)) {
    $1=PyInt_AsUnsignedLongLongMask($input);
  }
#endif
  else {
#if PY_VERSION_HEX < 0x03000000
    builtins_mod= PyImport_ImportModule("__builtin__");
    LongTypeObj=PyObject_GetAttrString(builtins_mod,"long");
#else
    builtins_mod= PyImport_ImportModule("builtins");
    LongTypeObj=PyObject_GetAttrString(builtins_mod,"int");
#endif
    LongObj=PyObject_CallFunctionObjArgs(LongTypeObj,$input,NULL);
    if (LongObj) {
      if (PyLong_Check(LongObj)) {
        $1=PyLong_AsUnsignedLongLong(LongObj);
      }
#if PY_VERSION_HEX < 0x03000000
      else if (PyInt_Check(LongObj)) {
        $1=PyInt_AsUnsignedLongLongMask(LongObj);
      }
#endif
      else {
        Py_XDECREF(LongObj);
        SWIG_fail;
      }
      Py_XDECREF(LongObj);
    } else {
      SWIG_fail;
    }
    Py_XDECREF(builtins_mod);
  }
} 



%define numpy_rtm_input_typemaps(snde_cpptype,varname,snde_cpptype_string,SNDE_RTN_SNDE_CPPTYPE,check_nan,check_nan_float_type)
 // implements input and  const & input  typemaps for the specified type,
 // which must have an entry in snd::rtn_numpytypemap

 //%feature("novaluewrapper") snde_cpptype; // because valuewrapper screws up our output typemap and it shouldn't be necessary because we only use numpy_rtm_typemaps on plain-old-data types  (but disabling it doesn't seem to work)

%typemap(typecheck,precedence=SWIG_TYPECHECK_FLOAT_ARRAY) snde_cpptype varname {
  std::unordered_map<unsigned,PyArray_Descr*>::iterator numpytypemap_it;
  PyArray_Descr *ArrayDescr;
  PyArrayObject *castedarrayobj;
  numpytypemap_it = snde::rtn_numpytypemap.find( SNDE_RTN_SNDE_CPPTYPE );

  if (numpytypemap_it == snde::rtn_numpytypemap.end()) {
    //throw snde::snde_error("No corresponding numpy datatype found for " snde_cpptype_string );
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			  "$argnum"" No corresponding numpy datatype found for " snde_cpptype_string);
    SWIG_fail;
  }
  ArrayDescr = numpytypemap_it->second;
  Py_IncRef((PyObject *)ArrayDescr); // because PyArray_NewFromDescr steals a reference to its descr parameter

  // Cast to our desired type
  castedarrayobj = (PyArrayObject *)PyArray_CheckFromAny($input,ArrayDescr,0,0,NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_NOTSWAPPED|NPY_ARRAY_ELEMENTSTRIDES,nullptr);
  if (castedarrayobj) {
    Py_DecRef((PyObject *)castedarrayobj);
    $1 = 1;
  } else {
    $1 = 0;
  }
  
}



%typemap(in) snde_cpptype varname (std::unordered_map<unsigned,PyArray_Descr*>::iterator numpytypemap_it, PyArray_Descr *ArrayDescr,PyArrayObject *castedarrayobj) {
  numpytypemap_it = snde::rtn_numpytypemap.find( SNDE_RTN_SNDE_CPPTYPE );

  if (numpytypemap_it == snde::rtn_numpytypemap.end()) {
    //throw snde::snde_error("No corresponding numpy datatype found for " snde_cpptype_string );
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			  "$argnum"" No corresponding numpy datatype found for " snde_cpptype_string);
    SWIG_fail;
  }
  ArrayDescr = numpytypemap_it->second;
  Py_IncRef((PyObject *)ArrayDescr); // because PyArray_NewFromDescr steals a reference to its descr parameter

  // Cast to our desired type
  castedarrayobj = (PyArrayObject *)PyArray_CheckFromAny($input,ArrayDescr,0,0,NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_NOTSWAPPED|NPY_ARRAY_ELEMENTSTRIDES,nullptr);
  if (!castedarrayobj) {
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			  "$argnum"" input typemap: Input data is not compatible with  " snde_cpptype_string);
    SWIG_fail;
  }
  
  if (PyArray_SIZE(castedarrayobj) != 1) {
    //throw snde::snde_error(snde_cpptype_string " input typemap: Only single input orientation is allowed");
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			  "$argnum"" input typemap: Only single input orientation is allowed for " snde_cpptype_string);
    SWIG_fail;
  }
  

  // now we can interpret the data as an snde_orientation3
  %evalif(check_nan,{

    check_nan_float_type *dataptr = (check_nan_float_type *)PyArray_DATA(castedarrayobj);
    unsigned idx;
    for (idx = 0;idx<sizeof(snde_cpptype)/sizeof(check_nan_float_type);idx++){
      if (isnan(dataptr[idx]) || isinf(dataptr[idx])){
	SWIG_exception_fail(SWIG_TypeError,snde::ssprintf( "in method '" "$symname" "', argument "
						     "$argnum"" input typemap: nan value within input structure at index %u " snde_cpptype_string,idx).c_str());
	SWIG_fail;
      }
    }
    })
  $1 = *(snde_cpptype *)PyArray_DATA(castedarrayobj);

  // free castedarrayobj
  Py_DecRef((PyObject *)castedarrayobj);
}

// input typemap for snde_cpptype const references
%typemap(in) const snde_cpptype & varname (std::unordered_map<unsigned,PyArray_Descr*>::iterator numpytypemap_it, PyArray_Descr *ArrayDescr,PyArrayObject *castedarrayobj) {
  numpytypemap_it = snde::rtn_numpytypemap.find(SNDE_RTN_SNDE_CPPTYPE);

  if (numpytypemap_it == snde::rtn_numpytypemap.end()) {
    //throw snde::snde_error("No corresponding numpy datatype found for " snde_cpptype_string );
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			"$argnum"" No corresponding numpy datatype found for " snde_cpptype_string);
    SWIG_fail;
  }
  ArrayDescr = numpytypemap_it->second;
  Py_IncRef((PyObject *)ArrayDescr); // because PyArray_NewFromDescr steals a reference to its descr parameter

  // Cast to our desired type
  castedarrayobj = (PyArrayObject *)PyArray_CheckFromAny($input,ArrayDescr,0,0,NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_NOTSWAPPED|NPY_ARRAY_ELEMENTSTRIDES,nullptr);
  if (!castedarrayobj) {
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			  "$argnum"" input typemap: Input data is not compatible with  " snde_cpptype_string);
    SWIG_fail;
  }

  if (PyArray_SIZE(castedarrayobj) != 1) {
    //throw snde::snde_error(snde_cpptype_string " input typemap: Only single input orientation is allowed");
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			  "$argnum"" input typemap: Only single input orientation is allowed for " snde_cpptype_string);
    SWIG_fail;
  }

  // now we can interpret the data as an snde_orientation3
  
  $1 = (snde_cpptype *)malloc(sizeof(snde_cpptype)); // freed by freearg typemap, below
  *$1 = *(snde_cpptype *)PyArray_DATA(castedarrayobj);

  // free castedarrayobj
  Py_DecRef((PyObject *)castedarrayobj);
}

%typemap(freearg) const snde_cpptype & varname // free orientation from const snde_cpptype & input typemap, above
{
  free($1);
}
%enddef

%define numpy_rtm_output_typemaps(snde_cpptype,snde_cpptype_string,SNDE_RTN_SNDE_CPPTYPE)
 // implements output and argout typemaps for the specified type,
 // which must have an entry in snd::rtn_numpytypemap



%typemap(out) snde_cpptype (std::unordered_map<unsigned,PyArray_Descr*>::iterator numpytypemap_it, PyArray_Descr *ArrayDescr,PyArrayObject *arrayobj,snde_cpptype resvalue) {
  numpytypemap_it = snde::rtn_numpytypemap.find(SNDE_RTN_SNDE_CPPTYPE);
  if (numpytypemap_it == snde::rtn_numpytypemap.end()) {
    //throw snde::snde_error("No corresponding numpy datatype found for " snde_cpptype_string );
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			"$argnum"" No corresponding numpy datatype found for " snde_cpptype_string);
    SWIG_fail;
  }
  ArrayDescr = numpytypemap_it->second;

  Py_IncRef((PyObject *)ArrayDescr); // because PyArray_CheckFromAny steals a reference to its descr parameter

  // create new 0D array 
  arrayobj = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,ArrayDescr,0,nullptr,nullptr,nullptr,0,nullptr);

  assert(PyArray_SIZE(arrayobj) == 1);
  resvalue = (snde_cpptype)($1);
  memcpy(PyArray_DATA(arrayobj),&resvalue,sizeof(resvalue));

  $result = (PyObject *)arrayobj;
}


// this typemap allows the input to not be present for a strict output argument
%typemap(in, numinputs=0) snde_cpptype *OUTPUT(snde_cpptype temp) {
  $1 = &temp;
}

%typemap(argout) snde_cpptype *OUTPUT (std::unordered_map<unsigned,PyArray_Descr*>::iterator numpytypemap_it, PyArray_Descr *ArrayDescr,PyArrayObject *arrayobj) {
  numpytypemap_it = snde::rtn_numpytypemap.find(SNDE_RTN_SNDE_CPPTYPE);
  if (numpytypemap_it == snde::rtn_numpytypemap.end()) {
    //throw snde::snde_error("No corresponding numpy datatype found for " snde_cpptype_string );
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			"$argnum"" No corresponding numpy datatype found for " snde_cpptype_string);
    SWIG_fail;
  }
  ArrayDescr = numpytypemap_it->second;

  Py_IncRef((PyObject *)ArrayDescr); // because PyArray_CheckFromAny steals a reference to its descr parameter

  // create new 0D array 
  arrayobj = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,ArrayDescr,0,nullptr,nullptr,nullptr,0,nullptr);

  assert(PyArray_SIZE(arrayobj) == 1);
  memcpy(PyArray_DATA(arrayobj),$1,sizeof(*$1));

  $result = (PyObject *)arrayobj;
}

// this typemap allows the input to not be present for a strict output argument
%typemap(in, numinputs=0) snde_cpptype *OUTPUT_MATRIX(std::unordered_map<unsigned,PyArray_Descr*>::iterator numpytypemap_it,PyObject *zeroval,PyArray_Descr *ArrayDescr,PyArrayObject *arrayobj,PyObject *first_member,PyObject *shape_attr,PyObject *first_shape_member, Py_ssize_t first_shape_value,snde_cpptype *temp) {
  
  numpytypemap_it = snde::rtn_numpytypemap.find(SNDE_RTN_SNDE_CPPTYPE);
  if (numpytypemap_it == snde::rtn_numpytypemap.end()) {
    //throw snde::snde_error("No corresponding numpy datatype found for " snde_cpptype_string );
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			"$argnum"" No corresponding numpy datatype found for " snde_cpptype_string);
    SWIG_fail;
  }

  zeroval = PyLong_FromLong(0);
  ArrayDescr = numpytypemap_it->second;

  Py_IncRef((PyObject *)ArrayDescr); // because PyArray_CheckFromAny steals a reference to its descr parameter
  // we need to obtain ArrayDescr[0].shape[0] to use that as the number of elements
  first_member = PyObject_GetItem((PyObject*)ArrayDescr,zeroval);
  if (!first_member) {
    PyErr_Print();
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', TEST2 Array descriptor first element not found for " snde_cpptype_string);
    SWIG_fail;
  }

  shape_attr = PyObject_GetAttrString(first_member,"shape");

  if (!shape_attr) {
    PyErr_Print();
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', Array descriptor first element does not have a .shape attribute for " snde_cpptype_string);
    SWIG_fail;
  }

  first_shape_member = PyObject_GetItem(shape_attr,zeroval);
  if (!first_shape_member) {
    PyErr_Print();
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', Array descriptor first element .shape attr element 0 not found for " snde_cpptype_string);
    SWIG_fail;
  }

  first_shape_value = PyNumber_AsSsize_t(first_shape_member,nullptr);

  
  Py_DecRef(first_shape_member);
  Py_DecRef(shape_attr);
  Py_DecRef(first_member);
  Py_DecRef(zeroval);

  temp = (snde_cpptype*)calloc(sizeof(snde_cpptype),first_shape_value);
  $1 = temp;
}

%typemap(freearg) snde_cpptype *OUTPUT_MATRIX // free the temp calloc variable from snde_cpptype *OUTPUT_MATRIX typemap, above
{
  free($1);
}

%typemap(argout) snde_cpptype *OUTPUT_MATRIX (std::unordered_map<unsigned,PyArray_Descr*>::iterator numpytypemap_it,PyObject *zeroval, PyArray_Descr *ArrayDescr,PyArrayObject *arrayobj,npy_intp num_el,PyObject *first_member,PyObject *shape_attr,PyObject *first_shape_member, Py_ssize_t first_shape_value) {
  numpytypemap_it = snde::rtn_numpytypemap.find(SNDE_RTN_SNDE_CPPTYPE);
  if (numpytypemap_it == snde::rtn_numpytypemap.end()) {
    //throw snde::snde_error("No corresponding numpy datatype found for " snde_cpptype_string );
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			"$argnum"" No corresponding numpy datatype found for " snde_cpptype_string);
    SWIG_fail;
  }

  zeroval = PyLong_FromLong(0);
  ArrayDescr = numpytypemap_it->second;

  Py_IncRef((PyObject *)ArrayDescr); // because PyArray_CheckFromAny steals a reference to its descr parameter
  // we need to obtain ArrayDescr[0].shape[0] to use that as the number of elements
  first_member = PyObject_GetItem((PyObject*)ArrayDescr,zeroval);
  if (!first_member) {
    PyErr_Print();
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', Array descriptor first element not found for " snde_cpptype_string);
    SWIG_fail;
  }

  shape_attr = PyObject_GetAttrString(first_member,"shape");

  if (!shape_attr) {
    PyErr_Print();
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', Array descriptor first element does not have a .shape attribute for " snde_cpptype_string);
    SWIG_fail;
  }

  first_shape_member = PyObject_GetItem(shape_attr,zeroval);
  if (!first_shape_member) {
    PyErr_Print();
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', Array descriptor first element .shape attr element 0 not found for " snde_cpptype_string);
    SWIG_fail;
  }

  first_shape_value = PyNumber_AsSsize_t(first_shape_member,nullptr);
  num_el = first_shape_value;

  Py_DecRef(first_shape_member);
  Py_DecRef(shape_attr);
  Py_DecRef(first_member);
  Py_DecRef(zeroval);


  // create new 1D array 
  arrayobj = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,ArrayDescr,1,&num_el,nullptr,nullptr,0,nullptr);

  assert(PyArray_SIZE(arrayobj) == first_shape_value);
  memcpy(PyArray_DATA(arrayobj),$1,sizeof(*$1)*first_shape_value);

  $result = (PyObject *)arrayobj;
}



%enddef


numpy_rtm_input_typemaps(snde_complexfloat32, ,"snde_complexfloat32",SNDE_RTN_COMPLEXFLOAT32,0, );

numpy_rtm_output_typemaps(snde_complexfloat32,"snde_complexfloat32",SNDE_RTN_COMPLEXFLOAT32);

numpy_rtm_input_typemaps(snde_complexfloat64, ,"snde_complexfloat64",SNDE_RTN_COMPLEXFLOAT64,0, );

numpy_rtm_output_typemaps(snde_complexfloat64,"snde_complexfloat64",SNDE_RTN_COMPLEXFLOAT64);


%pythoncode %{
try: 
  SNDE_INDEX_INVALID=(long(1)<<64)-1
  pass
except NameError:
  # python3
  SNDE_INDEX_INVALID=(int(1)<<64)-1
  pass

nt_snde_index = np.dtype(np.uint64)
nt_snde_shortindex = np.dtype(np.uint32)
nt_snde_ioffset = np.dtype(np.int64)
nt_snde_bool = np.dtype(np.int8)

nt_snde_complexfloat32 = np.complex64
nt_snde_complexfloat64 = np.complex128
  
%}
