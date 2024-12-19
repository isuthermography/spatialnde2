%{
  #include "recording.h"
%}

struct snde_recording_base {
  // This structure and pointed data are fully mutable during the INITIALIZING state
  // In METADATAREADY state metadata_valid should be set and metadata storage becomes immutable
  // in READY state the rest of the structure and data pointed to is immutable except in the case of a mutable recording for the state variable, which could could change to OBSOLETE and the data pointed to, which could change as well once the state becomes OBSOLETE
  // Note that in a threaded environment you can't safely read the state without an assocated lock, or you can read a mirrored atomic state variable, such as in class recording. 
  
  char *name; // separate malloc(); must match full channelpath
  uint64_t revision; // assigned in _realize_transaction(). If not yet assigned it will have the value SNDE_REVISION_INVALID
  int state; // see SNDE_RECS... defines below

  snde_immutable_metadata *metadata; 
  
  // what has been filled out in this struct so far
  snde_bool metadata_valid;

  
  snde_bool deletable;  // whether it is OK to call snde_recording_delete() on this structure

  snde_bool immutable; // doesn't mean necessarily immutable __now__, just immutable once ready

  
};

#define SNDE_REVISION_INVALID ((uint64_t)(~((uint64_t)0)))

// note distinction between states (SNDE_RECS_XXXX) and flags (SNDE_RECF_XXXX) 
#define SNDE_RECS_INITIALIZING 0
#define SNDE_RECF_STATICMETADATAREADY (1<<0)
#define SNDE_RECF_DYNAMICMETADATAREADY (1<<1)
#define SNDE_RECF_ALLMETADATAREADY (1<<2) // automatically set by _merge_static_and_dynamic_metadata_admin_locked()
#define SNDE_RECF_DATAREADY (1<<3)
#define SNDE_RECS_FULLYREADY (SNDE_RECF_STATICMETADATAREADY|SNDE_RECF_DYNAMICMETADATAREADY|SNDE_RECF_ALLMETADATAREADY|SNDE_RECF_DATAREADY)
#define SNDE_RECF_OBSOLETE (1<<4)

struct snde_ndarray_info {
  snde_index ndim;
  snde_index base_index; // index in elements beyond (*basearray)
  snde_index *dimlen; // pointer often from recording.layouts.at(index).dimlen.get()
  snde_index *strides; // pointer often from recording.layouts.at(index).strides.get()

  snde_bool owns_dimlen_strides; // if set, dimlen and strides should be free()'d with this data structure.

  unsigned typenum; /// See SNDE_RTN_... below (so far largely matches mutablerecstore.hpp typenum)
  size_t elementsize;

  snde_bool requires_locking_read;
  snde_bool requires_locking_write; 

  // physical data storage
  void **basearray; // double-pointer generally passed around, used for locking, etc. so that storage can be moved around if the recording is mutable. For independently-stored recordings this points at the _baseptr of the recording_storage_simple object. 

  void *shiftedarray;
  //void *basearray_holder; // replaced by _baseptr of recording_storage_simple object 

};


struct snde_multi_ndarray_recording {
  // This structure and pointed data are fully mutable during the INITIALIZING state
  // In METADATAREADY state metadata_valid should be set and metadata storage becomes immutable
  // in READY state the rest of the structure and data pointed to is immutable except in the case of a mutable recording for the state variable, which could could change to OBSOLETE and the data pointed to, which could change as well once the state becomes OBSOLETE
  // Note that in a threaded environment you can't safely read the state without an assocated lock, or you can read a mirrored atomic state variable, such as ixon class recording. 
  struct snde_recording_base rec;
  snde_bool dims_valid;
  snde_bool data_valid;

  // This info must be kept sync'd with class recording.layouts
  size_t num_arrays; // usually 1
  struct snde_ndarray_info *arrays; // must be preallocated to the needed size before any ndarray_recording_ref's are created 
  
};



// #defines for typenum
// New type numbers need to be added to
//   * definitions here in recording.h
//   * definitions in recording.i (for SWIG)
//   * reference_ndarray() definition in recstore.cpp 
//   * typemaps near beginning of recstore.cpp
//   * rtn_numpytypemap at end of spatialnde2.i
#define SNDE_RTN_UNASSIGNED 0
#define SNDE_RTN_FLOAT32 1
#define SNDE_RTN_FLOAT64 2
#define SNDE_RTN_FLOAT16 3
#define SNDE_RTN_UINT64 4
#define SNDE_RTN_INT64 5
#define SNDE_RTN_UINT32 6
#define SNDE_RTN_INT32 7
#define SNDE_RTN_UINT16 8
#define SNDE_RTN_INT16 9
#define SNDE_RTN_UINT8 10
#define SNDE_RTN_INT8 11
#define SNDE_RTN_SNDE_RGBA 12 /* R stored in lowest address... Like OpenGL with GL_RGBA and GL_UNSIGNED_BYTE, or snde_rgba type */ 
#define SNDE_RTN_COMPLEXFLOAT32 13
#define SNDE_RTN_COMPLEXFLOAT64 14
#define SNDE_RTN_COMPLEXFLOAT16 15
#define SNDE_RTN_RGBD64 16 /* as address goes from low to high: R (byte) G (byte) B (byte) A (byte) D (float32) */ 
#define SNDE_RTN_STRING 17 // not usable for recordings, but used internally for math parameters. 
#define SNDE_RTN_RECORDING 18 // not usable for recordings, but used internally for math parameters. 
#define SNDE_RTN_RECORDING_REF 19 // not usable for recordings, but used internally for math parameters. 
#define SNDE_RTN_SNDE_COORD3_INT16 20 // x,y,z coordinates, with each being 16 bit signed integer
#define SNDE_RTN_INDEXVEC 21 // std::vector<snde_index>... used for math function params only
#define SNDE_RTN_RECORDING_GROUP 22
//#define SNDE_RTN_POINTCLOUD_RECORDING 23
#define SNDE_RTN_MESHED_PART_RECORDING 24
#define SNDE_RTN_MESHED_VERTEXARRAY_RECORDING 25
#define SNDE_RTN_MESHED_TEXVERTEX_RECORDING 26
#define SNDE_RTN_MESHED_VERTNORMALS_RECORDING 27
#define SNDE_RTN_MESHED_TRINORMALS_RECORDING 28
#define SNDE_RTN_MESHED_PARAMETERIZATION_RECORDING 29
#define SNDE_RTN_TEXTURED_PART_RECORDING 30
#define SNDE_RTN_ASSEMBLY_RECORDING 31

#define SNDE_RTN_SNDE_PART 32
#define SNDE_RTN_SNDE_TOPOLOGICAL 33
#define SNDE_RTN_SNDE_TRIANGLE 34
#define SNDE_RTN_SNDE_COORD3 35
#define SNDE_RTN_SNDE_CMAT23 36
#define SNDE_RTN_SNDE_EDGE 37
#define SNDE_RTN_SNDE_COORD2 38
#define SNDE_RTN_SNDE_AXIS32 39
#define SNDE_RTN_SNDE_VERTEX_EDGELIST_INDEX 40
#define SNDE_RTN_SNDE_BOX3 41
#define SNDE_RTN_SNDE_BOXCOORD3 42
#define SNDE_RTN_SNDE_PARAMETERIZATION 43
#define SNDE_RTN_SNDE_PARAMETERIZATION_PATCH 44
#define SNDE_RTN_SNDE_TRIVERTNORMALS 45
#define SNDE_RTN_SNDE_RENDERCOORD 46


#define SNDE_RTN_SNDE_BOX2 47
#define SNDE_RTN_SNDE_BOXCOORD2 48
#define SNDE_RTN_SNDE_IMAGEDATA 49
#define SNDE_RTN_SNDE_COORD 50
// (note gap)
#define SNDE_RTN_SNDE_COORD4 52
#define SNDE_RTN_SNDE_ORIENTATION2 53
#define SNDE_RTN_SNDE_ORIENTATION3 54
#define SNDE_RTN_SNDE_AXIS3 55
#define SNDE_RTN_SNDE_INDEXRANGE 56
#define SNDE_RTN_SNDE_PARTINSTANCE 57
#define SNDE_RTN_SNDE_IMAGE 58
#define SNDE_RTN_SNDE_KDNODE 59
#define SNDE_RTN_SNDE_BOOL 60
#define SNDE_RTN_SNDE_COMPLEXIMAGEDATA 61

#define SNDE_RTN_CONSTRUCTIBLEMETADATA 62 // not usable for recordings, but used internally for math parameters.
