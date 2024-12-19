//%shared_ptr(snde::memallocator);
//%shared_ptr(snde::cmemallocator);

%{
  
#include "geometry_types.h"
%}



typedef double snde_coord;
typedef float snde_imagedata;



//#define SNDE_INDEX_INVALID (~((snde_index)0))

// typecheck typemap for snde_index... This is needed because sometimes
// we get np.uint64's that fail the default swig typecheck


numpy_rtm_input_typemaps(snde_coord3, ,"snde_coord3",SNDE_RTN_SNDE_COORD3,0, );
numpy_rtm_input_typemaps(snde_coord4, ,"snde_coord4",SNDE_RTN_SNDE_COORD4,0, );
numpy_rtm_input_typemaps(snde_orientation3, ,"snde_orientation3",SNDE_RTN_SNDE_ORIENTATION3,0, );


numpy_rtm_output_typemaps(snde_coord3,"snde_coord3",SNDE_RTN_SNDE_COORD3);
numpy_rtm_output_typemaps(snde_coord4,"snde_coord4",SNDE_RTN_SNDE_COORD4);
numpy_rtm_output_typemaps(snde_orientation3,"snde_orientation3",SNDE_RTN_SNDE_ORIENTATION3);

// Specialized input type maps that enable nan checking
numpy_rtm_input_typemaps(snde_coord4,quat,"snde_coord4",SNDE_RTN_SNDE_COORD4,1,snde_coord); 
numpy_rtm_input_typemaps(snde_coord4,quat1,"snde_coord4",SNDE_RTN_SNDE_COORD4,1,snde_coord); 
numpy_rtm_input_typemaps(snde_coord4,quat2,"snde_coord4",SNDE_RTN_SNDE_COORD4,1,snde_coord);
numpy_rtm_input_typemaps(snde_coord4,vec,"snde_coord4",SNDE_RTN_SNDE_COORD4,1,snde_coord);
numpy_rtm_input_typemaps(snde_orientation3,orient,"snde_orientation3",SNDE_RTN_SNDE_ORIENTATION3,1,snde_coord);
numpy_rtm_input_typemaps(snde_orientation3,left,"snde_orientation3",SNDE_RTN_SNDE_ORIENTATION3,1,snde_coord);
numpy_rtm_input_typemaps(snde_orientation3,right,"snde_orientation3",SNDE_RTN_SNDE_ORIENTATION3,1,snde_coord);



typedef struct _snde_coord3 {
  snde_coord coord[3];
} snde_coord3;

typedef struct _snde_coord4 {
  snde_coord coord[4];
} snde_coord4;

typedef struct _snde_orientation3 {
  /* for point p, orientation represents q p q' + o  */
  snde_coord4 quat; // normalized quaternion ... represented as real (w) component, i (x) component, j (y) component, k (z) component, 
  snde_coord4 offset; // 4th coordinate of offset always one
  
} snde_orientation3;

// give structure a constructor per
// https://stackoverflow.com/questions/33564645/how-to-add-an-alternative-constructor-to-the-target-language-specifically-pytho

%ignore snde_orientation3::snde_orientation3();

%extend _snde_orientation3 {
  _snde_orientation3(const snde_orientation3 &orig)
  {
    snde_orientation3 *neworient = new snde_orientation3();
    neworient->offset=orig.offset;
    neworient->quat=orig.quat;
    return neworient;
  }
};



/* This commented section has old manual implementations of the new numpy_rtm_typemaps() macro

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



%typemap(in, numinputs=0) snde_orientation3 *OUTPUT(snde_orientation3 temp) {
  $1 = &temp;
}

%typemap(argout) snde_orientation3 *OUTPUT (std::unordered_map<unsigned,PyArray_Descr*>::iterator numpytypemap_it, PyArray_Descr *ArrayDescr,PyArrayObject *arrayobj) {
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




%typemap(in, numinputs=0) snde_coord4 *OUTPUT(snde_coord4 temp) {
  $1 = &temp;
}

%typemap(argout) snde_coord4 *OUTPUT (std::unordered_map<unsigned,PyArray_Descr*>::iterator numpytypemap_it, PyArray_Descr *ArrayDescr,PyArrayObject *arrayobj) {
  numpytypemap_it = snde::rtn_numpytypemap.find(SNDE_RTN_SNDE_COORD4);
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


*/

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
ct_snde_coord = ctypes.c_double
ct_snde_imagedata=ctypes.c_float
ct_snde_index=ctypes.c_uint64
ct_snde_shortindex=ctypes.c_uint32
ct_snde_ioffset=ctypes.c_int64
ct_snde_bool=ctypes.c_char

nt_snde_coord = np.dtype(np.float32)
nt_snde_imagedata = np.dtype(np.float32)

nt_snde_orientation3=np.dtype([("quat",nt_snde_coord,4),
				 ("offset",nt_snde_coord,4), # fourth coordinate always one
			       ])
  
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


# WARNING: OBSOLETE!!!***
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



