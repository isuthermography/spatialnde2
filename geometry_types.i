//%shared_ptr(snde::memallocator);
//%shared_ptr(snde::cmemallocator);

%{
  
#include "geometry_types.h"
%}

%pythonbegin %{
import ctypes
import numpy as np
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

// ArrayType should be np.ndarray,
PyObject *Pointer_To_Numpy_Array(PyObject *ArrayType, PyObject *DType,PyObject *Base,size_t n,void **ptraddress);
%{
PyObject *Pointer_To_Numpy_Array(PyObject *ArrayType, PyObject *DType,PyObject *Base,size_t n,void **ptraddress)
{
  // ArrayType should usually be numpy.ndarray
  npy_intp dims;
  dims=n;
  Py_INCREF(DType); // because NewFromDescr() steals a reference to Descr
  PyObject *NewArray=PyArray_NewFromDescr((PyTypeObject *)ArrayType,(PyArray_Descr *)DType,1,&dims,NULL,*ptraddress,0,NULL);

  Py_INCREF(Base); // because SetBaseObject() steals a reference 
  PyArray_SetBaseObject((PyArrayObject *)NewArray,Base);
  
  return NewArray;
} 
%}

%pythoncode %{
SNDE_INDEX_INVALID=(long(1)<<64)-1

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

nt_snde_orientation3=np.dtype([("offset",nt_snde_coord,3),
                               ("pad1",nt_snde_coord),
			       ("quat",nt_snde_coord,3),
			       ("pad2",nt_snde_coord),])

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


nt_snde_meshedpart=np.dtype([('orientation', nt_snde_orientation3),
		    ('firsttri', nt_snde_index),
		    ('numtris', nt_snde_index),
		    ('firstedge', nt_snde_index),
		    ('numedges', nt_snde_index),
		    ('firstvertex', nt_snde_index),
		    ('numvertices', nt_snde_index),
		    ('first_vertex_edgelist_entry', nt_snde_index),
		    ('num_vertex_edgelist_entries', nt_snde_index),
  
		    ('firstbox', nt_snde_index),
		    ('numboxes', nt_snde_index),
		    
		    ('firstboxpoly', nt_snde_index),
		    ('numboxpolys', nt_snde_index),
		    ('solid', nt_snde_bool),
		    ('pad1', nt_snde_bool,7)])
		    

class snde_geometrystruct(ctypes.Structure):
  
  def __init__(self):
    super(snde_geometrystruct,self).__init__()
    pass
    
  def field_address(self,fieldname):
    # unfortunately byref() doesnt work right because struct members when accesed become plain ints
    offset=getattr(self.__class__,fieldname).offset
    return ArrayPtr_fromint(ctypes.addressof(self)+offset)

  def field_numpy(self,manager,lockholder,fieldname,dtype):
      """Extract a numpy array representing the specified field. 
         This numpy array 
         will only be valid while the lockholder.fieldname locks are held"""
      
      offset=getattr(self.__class__,fieldname).offset
      Ptr = ArrayPtr_fromint(ctypes.addressof(self)+offset)
      n = manager.get_total_nelem(Ptr)
      assert(dtype.itemsize==manager.get_elemsize(Ptr))
      return Pointer_To_Numpy_Array(np.ndarray,dtype,getattr(lockholder,fieldname),n,Ptr)    
  pass



  
%}



