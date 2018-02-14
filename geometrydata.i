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
  
class snde_geometrydata(ctypes.Structure):
  _fields_=[("tol",ctypes.c_double),
	   ("meshparts",ctypes.c_void_p), # POINTER(snde_meshedpart)),
	   # ... !!!*** Need to fill this in!!!***
	   ("vertices",ctypes.c_void_p)]

  def field_address(self,fieldname):
    # unfortunately byref() doesnt work right because struct members when accesed become plain ints
    offset=getattr(self.__class__,fieldname).offset
    return ArrayPtr_fromint(ctypes.addressof(self)+offset)
  pass


%}
