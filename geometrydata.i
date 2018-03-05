//%shared_ptr(snde::memallocator);
//%shared_ptr(snde::cmemallocator);

%{
  
#include "geometrydata.h"
%}

%pythonbegin %{
import ctypes
%}


#ifdef __cplusplus
#include "geometry.hpp"
typedef snde::geometry snde_geometry;
#else
typedef struct snde_geometry snde_geometry;
#endif

#ifdef __cplusplus
extern "C" {
#endif
  // C function definitions for geometry manipulation go here... 
  

#ifdef __cplusplus
}
#endif


%pythoncode %{
  # IMPORTANT: definition in geometrydata.h must be changed in parallel with this.



# We build the geometrydata class dynamically, because
# That seems to be the only way to add data to it
# (such as a reference to our array manager)

# This function is called by the SWIG out typemap for "snde_geometrydata geom" 
# in geometry.i so that when the "geom" member is accessed we get
# this class structure.
#
# The dynamically generated superclass is built by
# build_geometrystruct_class() in geometry_types.i

def build_geometrydata_class(arraymgr):
  snde_geometrystruct = build_geometrystruct_class(arraymgr)
  
  class snde_geometrydata(snde_geometrystruct):
    # ***!!! _fields_ must match EXACTLY the primary specification in geometrydata.h
    _fields_=[("tol",ctypes.c_double),
	     ("meshedparts",ctypes.c_void_p), # POINTER(snde_meshedpart),
	     ("triangles",ctypes.c_void_p),
	     ("refpoints",ctypes.c_void_p),
	     ("maxradius",ctypes.c_void_p),
	     ("normal",ctypes.c_void_p),
	     ("inplanemat",ctypes.c_void_p),
	     ("edges",ctypes.c_void_p),
	     ("vertices",ctypes.c_void_p),
   	     ("principal_curvatures",ctypes.c_void_p),
	     ("curvature_tangent_axes",ctypes.c_void_p),
	     ("vertex_edgelist_indices",ctypes.c_void_p),
	     ("vertex_edgelist",ctypes.c_void_p),
	     # !!!*** Need to add NURBS entries!!!***
	     ]
    pass
  return snde_geometrydata


%}
