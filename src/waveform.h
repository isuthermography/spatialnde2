#ifndef SNDE_WAVEFORM_H
#define SNDE_WAVEFORM_H

#include <stdint.h>

#include "geometry_types.h"

#ifndef __cplusplus
typedef void snde_immutable_metadata; // must treat metadata pointer as opaque from C
#else
#include "metadata.hpp"
typedef snde::immutable_metadata snde_immutable_metadata;
#endif

struct snde_waveform {
  // This structure and pointed data are fully mutable during the INITIALIZING state
  // In METADATAREADY state metadata_valid should be set and metadata storage becomes immutable
  // in READY state the rest of the structure and data pointed to is immutable except in the case of a mutable waveform for the state variable, which could could change to OBSOLETE and the data pointed to, which could change as well once the state becomes OBSOLETE
  // Note that in a threaded environment you can't safely read the state without an assocated lock, or you can read a mirrored atomic state variable, such as in class waveform. 
  
  char *name; // separate malloc(); must match full channelpath
  uint64_t revision;
  int state; // see SNDE_WFMS... defines below

  snde_immutable_metadata *metadata; 
  
  // what has been filled out in this struct so far
  snde_bool metadata_valid;
  snde_bool dims_valid;
  snde_bool data_valid;
  snde_bool deletable;  // whether it is OK to call snde_waveform_delete() on this structure

  // This info must be kept sync'd with class waveform.layout
  snde_index ndim;
  snde_index base_index; // index in elements beyond (*basearray)
  snde_index *dimlen; // pointer often from waveform.layout.dimlen.get()
  snde_index *strides; // pointer often from waveform.layout.strides.get()

  snde_bool owns_dimlen_strides; // if set, dimlen and strides should be free()'d with this data structure.
  snde_bool immutable; // doesn't mean necessarily immutable __now__, just immutable once ready

  unsigned typenum; /// See SNDE_WTN_... below (so far largely matches mutablewfmstore.hpp typenum)
  size_t elementsize; 

  // *** Need Metadata storage ***
  // Use C or C++ library interface???

  
  // physical data storage
  void **basearray; // pointer into low-level store
};
#define SNDE_WFMS_INITIALIZING 0
#define SNDE_WFMS_METADATAREADY 1
#define SNDE_WFMS_READY 2
#define SNDE_WFMS_OBSOLETE 3


// #defines for typenum
#define SNDE_WTN_FLOAT 0
#define SNDE_WTN_DOUBLE 1
#define SNDE_WTN_HALFFLOAT 2
#define SNDE_WTN_UINT64 3
#define SNDE_WTN_INT64 4
#define SNDE_WTN_UINT32 5
#define SNDE_WTN_INT32 6
#define SNDE_WTN_UINT16 7
#define SNDE_WTN_INT16 8
#define SNDE_WTN_UINT8 9
#define SNDE_WTN_INT8 10
#define SNDE_WTN_RGBA32 11 /* R stored in lowest address... Like OpenGL with GL_RGBA and GL_UNSIGNED_BYTE, or snde_rgba type */ 
#define SNDE_WTN_COMPLEXFLOAT 12
#define SNDE_WTN_COMPLEXDOUBLE 13
#define SNDE_WTN_RGBD64 14 /* as address goes from low to high: R (byte) G (byte) B (byte) A (byte) D (float32) */ 


#endif // SNDE_WAVEFORM_H
