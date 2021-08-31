%{
  #include "recording.h"
%}

struct snde_recording_base {
  // This structure and pointed data are fully mutable during the INITIALIZING state
  // In METADATAREADY state metadata_valid should be set and metadata storage becomes immutable
  // in READY state the rest of the structure and data pointed to is immutable except in the case of a mutable recording for the state variable, which could could change to OBSOLETE and the data pointed to, which could change as well once the state becomes OBSOLETE
  // Note that in a threaded environment you can't safely read the state without an assocated lock, or you can read a mirrored atomic state variable, such as in class recording. 
  
  char *name; // separate malloc(); must match full channelpath
  uint64_t revision;
  int state; // see SNDE_RECS... defines below

  snde_immutable_metadata *metadata; 
  
  // what has been filled out in this struct so far
  snde_bool metadata_valid;

  
  snde_bool deletable;  // whether it is OK to call snde_recording_delete() on this structure

  snde_bool immutable; // doesn't mean necessarily immutable __now__, just immutable once ready

  
};
#define SNDE_RECS_INITIALIZING 0
#define SNDE_RECS_METADATAREADY 1
#define SNDE_RECS_READY 2
#define SNDE_RECS_OBSOLETE 3


struct snde_ndarray_recording {
  // This structure and pointed data are fully mutable during the INITIALIZING state
  // In METADATAREADY state metadata_valid should be set and metadata storage becomes immutable
  // in READY state the rest of the structure and data pointed to is immutable except in the case of a mutable recording for the state variable, which could could change to OBSOLETE and the data pointed to, which could change as well once the state becomes OBSOLETE
  // Note that in a threaded environment you can't safely read the state without an assocated lock, or you can read a mirrored atomic state variable, such as in class recording. 
  struct snde_recording_base rec;
  snde_bool dims_valid;
  snde_bool data_valid;

  // This info must be kept sync'd with class recording.layout
  snde_index ndim;
  snde_index base_index; // index in elements beyond (*basearray)
  snde_index *dimlen; // pointer often from recording.layout.dimlen.get()
  snde_index *strides; // pointer often from recording.layout.strides.get()

  snde_bool owns_dimlen_strides; // if set, dimlen and strides should be free()'d with this data structure.

  unsigned typenum; /// See SNDE_RTN_... below (so far largely matches mutablerecstore.hpp typenum)
  size_t elementsize; 

  // physical data storage
  void **basearray; // double-pointer generally passed around, used for locking, etc. so that storage can be moved around if the recording is mutable. For independently-stored recordings this points at the _baseptr of the recording_storage_simple object. 
  
  //void *basearray_holder; // replaced by _baseptr of recording_storage_simple object 
};



// #defines for typenum
// New type numbers need to be added to both create_typed_recording() definitions in recstore.cpp and to the typemaps in recstore.cpp
#define SNDE_RTN_FLOAT32 0
#define SNDE_RTN_FLOAT64 1
#define SNDE_RTN_FLOAT16 2
#define SNDE_RTN_UINT64 3
#define SNDE_RTN_INT64 4
#define SNDE_RTN_UINT32 5
#define SNDE_RTN_INT32 6
#define SNDE_RTN_UINT16 7
#define SNDE_RTN_INT16 8
#define SNDE_RTN_UINT8 9
#define SNDE_RTN_INT8 10
#define SNDE_RTN_RGBA32 11 /* R stored in lowest address... Like OpenGL with GL_RGBA and GL_UNSIGNED_BYTE, or snde_rgba type */ 
#define SNDE_RTN_COMPLEXFLOAT32 12
#define SNDE_RTN_COMPLEXFLOAT64 13
#define SNDE_RTN_COMPLEXFLOAT16 14
#define SNDE_RTN_RGBD64 15 /* as address goes from low to high: R (byte) G (byte) B (byte) A (byte) D (float32) */ 
#define SNDE_RTN_STRING 16 // not usable for recordings, but used internally for math parameters. 
#define SNDE_RTN_RECORDING 17 // not usable for recordings, but used internally for math parameters. 
