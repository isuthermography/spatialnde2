%{

  #include "snde/quaternion.h"
  
%}


// input typemap for const snde_coord4 *rotmtx arrays
%typemap(in) const snde_coord4 *rotmtx (std::unordered_map<unsigned,PyArray_Descr*>::iterator numpytypemap_it, PyArray_Descr *ArrayDescr,PyArrayObject *castedarrayobj) {
  numpytypemap_it = snde::rtn_numpytypemap.find(SNDE_RTN_SNDE_COORD);

  if (numpytypemap_it == snde::rtn_numpytypemap.end()) {
    //throw snde::snde_error("No corresponding numpy datatype found for " snde_cpptype_string );
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			"$argnum"" No corresponding numpy datatype found for snde_coord");
    
  }
  ArrayDescr = numpytypemap_it->second;
  Py_IncRef((PyObject *)ArrayDescr); // because PyArray_NewFromDescr steals a reference to its descr parameter

  // Cast to our desired type
  castedarrayobj = (PyArrayObject *)PyArray_CheckFromAny($input,ArrayDescr,0,0,NPY_ARRAY_F_CONTIGUOUS|NPY_ARRAY_NOTSWAPPED|NPY_ARRAY_ELEMENTSTRIDES,nullptr);
  if (!castedarrayobj) {
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			  "$argnum"" input typemap: Input data is not compatible with contiguous snde_coord");
    
  }

  if (PyArray_SIZE(castedarrayobj) != 16) {
    //throw snde::snde_error(snde_cpptype_string " input typemap: Only single input orientation is allowed");
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			  "$argnum"" input typemap: 4x4 (16-element) snde_coord matrix required for rotmtx input.");
  }

  // now we can interpret the data as 16 snde_coords or 4 snde_coord4s
  
  $1 = (snde_coord4 *)malloc(sizeof(snde_coord4)*4); // freed by freearg typemap, below
  for (unsigned cnt=0;cnt < 4;cnt++) { // each iteration does an entire snde_coord4
    ($1)[cnt] = ((snde_coord4 *)PyArray_DATA(castedarrayobj))[cnt];
  }
  // free castedarrayobj
  Py_DecRef((PyObject *)castedarrayobj);
}

%typemap(freearg) const snde_coord4 *rotmtx // free orientation from const snde_cpptype & input typemap, above
{
  free($1);
}





void snde_null_orientation3(snde_orientation3 *OUTPUT);

void snde_invalid_orientation3(snde_orientation3 *OUTPUT);

snde_bool quaternion_equal(const snde_coord4 a, const snde_coord4 b);

snde_bool orientation3_equal(const snde_orientation3 a, const snde_orientation3 b);


void quaternion_normalize(const snde_coord4 unnormalized,snde_coord4 *OUTPUT);
  
void quaternion_product(const snde_coord4 quat1, const snde_coord4 quat2,snde_coord4 *OUTPUT);


void quaternion_product_normalized(const snde_coord4 quat1, const snde_coord4 quat2,snde_coord4 *OUTPUT);

void quaternion_inverse(const snde_coord4 quat, snde_coord4 *OUTPUT);


void quaternion_apply_vector(const snde_coord4 quat,const snde_coord4 vec,snde_coord4 *OUTPUT);


void quaternion_build_rotmtx(const snde_coord4 quat,snde_coord4 *OUTPUT /* (array of 3 or 4 coord4's, interpreted as column-major). Does not write 4th column  */ );

void orientation_build_rotmtx(const snde_orientation3 orient,snde_coord4 *OUTPUT /* (array of 4 coord4's, interpreted as column-major).  */ );

void rotmtx_build_orientation(const snde_coord4 *rotmtx, // array of 4 coord4s, interpreted as column-major homogeneous coordinates 4x4
			      snde_orientation3 *OUTPUT);

void orientation_inverse(const snde_orientation3 orient,snde_orientation3 *OUTPUT);

void orientation_apply_vector(const snde_orientation3 orient,const snde_coord4 vec,snde_coord4 *OUTPUT);

void orientation_apply_position(const snde_orientation3 orient,const snde_coord4 pos,snde_coord4 *OUTPUT);

void orientation_orientation_multiply(const snde_orientation3 left,const snde_orientation3 right,snde_orientation3 *OUTPUT);
