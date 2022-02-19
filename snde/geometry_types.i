//%shared_ptr(snde::memallocator);
//%shared_ptr(snde::cmemallocator);

%{
  
#include "geometry_types.h"
%}

%shared_ptr(std::vector<snde_index>);

%pythonbegin %{
import ctypes
import numpy as np
%}


typedef double snde_coord;
typedef float snde_imagedata;
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

%template(snde_index_vector) std::vector<snde_index>;


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

//#define SNDE_INDEX_INVALID (~((snde_index)0))

// typecheck typemap for snde_index... This is needed because sometimes
// we get np.uint64's that fail the default swig typecheck

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


%typemap(in) snde_orientation3 (std::unordered_map<unsigned,PyArray_Descr*>::iterator numpytypemap_it, PyArray_Descr *ArrayDescr,PyArrayObject *castedarrayobj) {
  numpytypemap_it = snde::rtn_numpytypemap.find(SNDE_RTN_SNDE_ORIENTATION3);

  if (numpytypemap_it == snde::rtn_numpytypemap.end()) {
    throw snde::snde_error("No corresponding numpy datatype found for snde_orientation3");
  }
  ArrayDescr = numpytypemap_it->second;
  Py_IncRef((PyObject *)ArrayDescr); // because PyArray_NewFromDescr steals a reference to its descr parameter

  // Cast to our desired type
  castedarrayobj = (PyArrayObject *)PyArray_CheckFromAny($input,ArrayDescr,0,0,NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_NOTSWAPPED|NPY_ARRAY_ELEMENTSTRIDES,nullptr);

  if (PyArray_SIZE(castedarrayobj) != 1) {
    throw snde::snde_error("snde_orientation3 input typemap: Only single input orienation is allowed");
  }

  // now we can interpret the data as an snde_orientation3
  
  $1 = *(snde_orientation3 *)PyArray_DATA(castedarrayobj);

  // free castedarrayobj
  Py_DecRef((PyObject *)castedarrayobj);
}

// input typemap for snde_orientation3 const references
%typemap(in) const snde_orientation3 &(std::unordered_map<unsigned,PyArray_Descr*>::iterator numpytypemap_it, PyArray_Descr *ArrayDescr,PyArrayObject *castedarrayobj) {
  numpytypemap_it = snde::rtn_numpytypemap.find(SNDE_RTN_SNDE_ORIENTATION3);

  if (numpytypemap_it == snde::rtn_numpytypemap.end()) {
    throw snde::snde_error("No corresponding numpy datatype found for snde_orientation3");
  }
  ArrayDescr = numpytypemap_it->second;
  Py_IncRef((PyObject *)ArrayDescr); // because PyArray_NewFromDescr steals a reference to its descr parameter

  // Cast to our desired type
  castedarrayobj = (PyArrayObject *)PyArray_CheckFromAny($input,ArrayDescr,0,0,NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_NOTSWAPPED|NPY_ARRAY_ELEMENTSTRIDES,nullptr);

  if (PyArray_SIZE(castedarrayobj) != 1) {
    throw snde::snde_error("snde_orientation3 input typemap: Only single input orienation is allowed");
  }

  // now we can interpret the data as an snde_orientation3
  
  $1 = (snde_orientation3 *)malloc(sizeof(snde_orientation3)); // freed by freearg typemap, below
  *$1 = *(snde_orientation3 *)PyArray_DATA(castedarrayobj);

  // free castedarrayobj
  Py_DecRef((PyObject *)castedarrayobj);
}

%typemap(freearg) const snde_orientation3 &// free orientation from const snde_orientation3 & input typemap, above
{
  free($1);
}



%typemap(out) snde_orientation3 (std::unordered_map<unsigned,PyArray_Descr*>::iterator numpytypemap_it, PyArray_Descr *ArrayDescr,PyArrayObject *arrayobj) {
  numpytypemap_it = snde::rtn_numpytypemap.find(SNDE_RTN_SNDE_ORIENTATION3);
  if (numpytypemap_it == snde::rtn_numpytypemap.end()) {
    throw snde::snde_error("No corresponding numpy datatype found for snde_orientation3");
  }
  ArrayDescr = numpytypemap_it->second;

  Py_IncRef((PyObject *)ArrayDescr); // because PyArray_CheckFromAny steals a reference to its descr parameter

  // create new 0D array 
  arrayobj = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,ArrayDescr,0,nullptr,nullptr,nullptr,0,nullptr);

  assert(PyArray_SIZE(arrayobj) == 1);
  memcpy(PyArray_DATA(arrayobj),&$1,sizeof($1));

  $result = (PyObject *)arrayobj;
}



// ArrayType should be np.ndarray,
PyObject *Pointer_To_Numpy_Array(PyObject *ArrayType, PyObject *DType,PyObject *Base,bool write,size_t n,void **ptraddress,size_t elemsize,size_t startidx);
%{
PyObject *Pointer_To_Numpy_Array(PyObject *ArrayType, PyObject *DType,PyObject *Base,bool write,size_t n,void **ptraddress,size_t elemsize,size_t startidx)
{
  // ArrayType should usually be numpy.ndarray
  npy_intp dims;
  dims=n;
  Py_INCREF(DType); // because NewFromDescr() steals a reference to Descr
  PyObject *NewArray=PyArray_NewFromDescr((PyTypeObject *)ArrayType,(PyArray_Descr *)DType,1,&dims,NULL,((char *)*ptraddress)+elemsize*startidx,write ? NPY_ARRAY_WRITEABLE:0,NULL);
  
  Py_INCREF(Base); // because SetBaseObject() steals a reference 
  PyArray_SetBaseObject((PyArrayObject *)NewArray,Base);
  
  return NewArray;
} 
%}

%pythoncode %{
try: 
  SNDE_INDEX_INVALID=(long(1)<<64)-1
  pass
except NameError:
  # python3
  SNDE_INDEX_INVALID=(int(1)<<64)-1
  pass
ct_snde_coord = ctypes.c_double
ct_snde_imagedata=ctypes.c_float
ct_snde_index=ctypes.c_uint64
ct_snde_shortindex=ctypes.c_uint32
ct_snde_ioffset=ctypes.c_int64
ct_snde_bool=ctypes.c_char

nt_snde_coord = np.dtype(np.double)
nt_snde_imagedata = np.dtype(np.float)
nt_snde_index = np.dtype(np.uint64)
nt_snde_shortindex = np.dtype(np.uint32)
nt_snde_ioffset = np.dtype(np.int64)
nt_snde_bool = np.dtype(np.int8)

nt_snde_orientation3=np.dtype([("offset",nt_snde_coord,4), # fourth coordinate always zero
			       ("quat",nt_snde_coord,4)])
  
nt_snde_coord3=np.dtype((nt_snde_coord,3))
nt_snde_coord2=np.dtype((nt_snde_coord,2))

nt_snde_edge=np.dtype([("vertex",nt_snde_index,2),
	               ("face_a",nt_snde_index),
		       ("face_b",nt_snde_index),
		       ("face_a_prev_edge",nt_snde_index),
		       ("face_a_next_edge",nt_snde_index),
		       ("face_b_prev_edge",nt_snde_index),
		       ("face_b_next_edge",nt_snde_index)])


nt_snde_vertex_edgelist_index=np.dtype([("edgelist_index",nt_snde_index),
	                                ("edgelist_numentries",nt_snde_index)])					

nt_snde_triangle=np.dtype((nt_snde_index,3))
nt_snde_axis32=np.dtype((nt_snde_coord,(2,3)))
nt_snde_mat23=np.dtype((nt_snde_coord,(2,3)))


nt_snde_meshedpart=np.dtype([  # ('orientation', nt_snde_orientation3),
		    ('firsttri', nt_snde_index),
		    ('numtris', nt_snde_index),
		    ('firstedge', nt_snde_index),
		    ('numedges', nt_snde_index),
		    ('firstvertex', nt_snde_index),
		    ('numvertices', nt_snde_index),
		    ('first_vertex_edgelist', nt_snde_index),
		    ('num_vertex_edgelist', nt_snde_index),
  
		    ('firstbox', nt_snde_index),
		    ('numboxes', nt_snde_index),
		    
		    ('firstboxpoly', nt_snde_index),
		    ('numboxpolys', nt_snde_index),
		    ('solid', nt_snde_bool),
		    ('pad1', nt_snde_bool,7)])
		    
def build_geometrystruct_class(arraymgr):  # Don't think this is used anymore!!!
  class snde_geometrystruct(ctypes.Structure):
    manager=arraymgr;
    
    def __init__(self):
      super(snde_geometrystruct,self).__init__()
      pass

    def __repr__(self):
      descr="%s instance at 0x%x\n" % (self.__class__.__name__,ctypes.addressof(self))
      descr+="------------------------------------------------\n"
      for (fieldname,fieldtype) in self._fields_:
        descr+="array %25s @ 0x%x\n" % (fieldname,ctypes.addressof(self)+getattr(self.__class__,fieldname).offset)
        pass
      return descr

    def has_field(self,fieldname):
      return hasattr(self.__class__,fieldname)

    def addr(self,fieldname):
      # unfortunately byref() doesnt work right because struct members when accesed become plain ints
      offset=getattr(self.__class__,fieldname).offset
      return ArrayPtr_fromint(ctypes.addressof(self)+offset)  # return swig-wrapped void **
    
    def field_valid(self,fieldname):
      val=getattr(self,fieldname)
      return val is None or val==0
    
    def allocfield(self,lockholder,fieldname,dtype,allocid,numelem):
      startidx=lockholder.get_alloc(self.addr(fieldname),allocid)
      return self.field(lockholder,fieldname,True,dtype,startidx,numelem)
    
      
    def field(self,lockholder,fieldname,write,dtype,startidx,numelem=SNDE_INDEX_INVALID):
      """Extract a numpy array representing the specified field. 
         This numpy array 
         will only be valid while the lockholder.fieldname locks are held"""

      write=bool(write)
      offset=getattr(self.__class__,fieldname).offset
      Ptr = ArrayPtr_fromint(ctypes.addressof(self)+offset)
      max_n = self.manager.get_total_nelem(Ptr)-startidx
      numpy_numelem = numelem
      if numpy_numelem == SNDE_INDEX_INVALID:
        numpy_numelem=max_n
        pass
      assert(numpy_numelem <= max_n)
      
      elemsize=self.manager.get_elemsize(Ptr)
      assert(dtype.itemsize==elemsize)
      ### Could set the writable flag of the numpy array according to whether
      ### we have at least one write lock
      return Pointer_To_Numpy_Array(np.ndarray,dtype,lockholder.get(self.addr(fieldname),write,startidx,numelem),write,numpy_numelem,Ptr,elemsize,startidx)    
    pass
    
  return snde_geometrystruct


  
%}



