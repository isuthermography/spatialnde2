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

struct snde_waveform_base {
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

  
  snde_bool deletable;  // whether it is OK to call snde_waveform_delete() on this structure

  snde_bool immutable; // doesn't mean necessarily immutable __now__, just immutable once ready

  
};
#define SNDE_WFMS_INITIALIZING 0
#define SNDE_WFMS_METADATAREADY 1
#define SNDE_WFMS_READY 2
#define SNDE_WFMS_OBSOLETE 3


struct snde_ndarray_waveform {
  // This structure and pointed data are fully mutable during the INITIALIZING state
  // In METADATAREADY state metadata_valid should be set and metadata storage becomes immutable
  // in READY state the rest of the structure and data pointed to is immutable except in the case of a mutable waveform for the state variable, which could could change to OBSOLETE and the data pointed to, which could change as well once the state becomes OBSOLETE
  // Note that in a threaded environment you can't safely read the state without an assocated lock, or you can read a mirrored atomic state variable, such as in class waveform. 
  struct snde_waveform_base wfm;
  snde_bool dims_valid;
  snde_bool data_valid;

  // This info must be kept sync'd with class waveform.layout
  snde_index ndim;
  snde_index base_index; // index in elements beyond (*basearray)
  snde_index *dimlen; // pointer often from waveform.layout.dimlen.get()
  snde_index *strides; // pointer often from waveform.layout.strides.get()

  snde_bool owns_dimlen_strides; // if set, dimlen and strides should be free()'d with this data structure.

  unsigned typenum; /// See SNDE_WTN_... below (so far largely matches mutablewfmstore.hpp typenum)
  size_t elementsize; 

  // physical data storage
  void **basearray; // double-pointer generally passed around, used for locking, etc. so that storage can be moved around if the waveform is mutable
  
  void *basearray_holder; // This is what basearray is pointed to in most (all?) cases
};



// #defines for typenum
// New type numbers need to be added to both create_typed_waveform() definitions in wfmstore.cpp and to the typemaps in wfmstore.cpp
#define SNDE_WTN_FLOAT32 0
#define SNDE_WTN_FLOAT64 1
#define SNDE_WTN_FLOAT16 2
#define SNDE_WTN_UINT64 3
#define SNDE_WTN_INT64 4
#define SNDE_WTN_UINT32 5
#define SNDE_WTN_INT32 6
#define SNDE_WTN_UINT16 7
#define SNDE_WTN_INT16 8
#define SNDE_WTN_UINT8 9
#define SNDE_WTN_INT8 10
#define SNDE_WTN_RGBA32 11 /* R stored in lowest address... Like OpenGL with GL_RGBA and GL_UNSIGNED_BYTE, or snde_rgba type */ 
#define SNDE_WTN_COMPLEXFLOAT32 12
#define SNDE_WTN_COMPLEXFLOAT64 13
#define SNDE_WTN_COMPLEXFLOAT16 14
#define SNDE_WTN_RGBD64 15 /* as address goes from low to high: R (byte) G (byte) B (byte) A (byte) D (float32) */ 


#endif // SNDE_WAVEFORM_H
