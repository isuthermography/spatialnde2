#ifndef SNDE_GEOMETRY_TYPES
#define SNDE_GEOMETRY_TYPES

#ifndef __OPENCL_VERSION__
// if this is not an OpenCL kernel

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <math.h>
#include "snde/snde_types.h"

#endif // __OPENCL_VERSION__

#if (defined(_MSC_VER) && !defined(__cplusplus))
#define GEOTYPES_INLINE  __inline
#else
#define GEOTYPES_INLINE  inline
#endif

#ifdef __cplusplus
extern "C" {
#endif

  /* *** Changes to these type mappings must also go into 
     python definitions at bottom of geometry_types.i */
#ifdef __OPENCL_VERSION__
/* if this is an opencl kernel */
#ifdef SNDE_DOUBLEPREC_COORDS
typedef double snde_coord;
#else // SNDE_DOUBLEPREC_COORDS
typedef float snde_coord;
#endif // SNDE_DOUBLEPREC_COORDS

#ifdef SNDE_DOUBLEPREC_COORDS
#define M_PI_SNDE_COORD M_PI  // or CL_M_PI?
#else
#define M_PI_SNDE_COORD M_PI_F // or CL_M_PI_F?
#endif // SNDE_DOUBLEPREC_COORDS

  
  typedef float snde_rendercoord;
  typedef float snde_imagedata;
  typedef snde_complexfloat32 snde_compleximagedata;
  
  typedef union {
    unsigned int intval;
    float floatval;
  } snde_atomicimagedata;
  // OpenCL explicitly supports union-based type aliasing
  
  static GEOTYPES_INLINE void atomicpixel_accumulate(volatile __global snde_atomicimagedata *dest,snde_imagedata toaccum) {
    snde_atomicimagedata current,expected, next;

    current.floatval = dest->floatval;
    do {
      expected.floatval = current.floatval;
      next.floatval = expected.floatval + toaccum;

      current.intval = atomic_cmpxchg((volatile __global unsigned int *)&dest->intval,
				      expected.intval,
				      next.intval);
      
    } while (current.intval != expected.intval);
  }

  // remove atomicpixel_ready in OpenCL because
  // we can't legitimately implement it without
  // attempting a write, which is not really what is intended.
  // in almost all cases, a regular read will be fine
  //static GEOTYPES_INLINE float atomicpixel_read(volatile __global snde_atomicimagedata *src)
  //{
  //  snde_atomicimagedata copy;
  //  copy.intval = atomic_load((volatile __global atomic_uint *)&src->intval);
  //  
  //  return copy.floatval;
  //}
#else // __OPENCL_VERSION__ code that follows is for NOT OPENCL
  //#if 0 && defined(SNDE_OPENCL)

//typedef cl_double snde_coord;
//typedef cl_float snde_imagedata;
//typedef cl_ulong snde_index;
//typedef cl_uint snde_shortindex;
//typedef cl_long snde_ioffset;
//typedef cl_char snde_bool;

//#else
#ifdef SNDE_DOUBLEPREC_COORDS
typedef double snde_coord;
#else // SNDE_DOUBLEPREC_COORDS
typedef float snde_coord;
#endif // SNDE_DOUBLEPREC_COORDS

typedef float snde_rendercoord;
typedef float snde_imagedata;
typedef snde_complexfloat32 snde_compleximagedata;

#if (defined(__STDC_VERSION__) && (__STDC_VERSION__>= 201112L) && !defined(__STDC_NO_ATOMICS__)) //|| (defined(__cplusplus) && defined(__clang__)) 
  // Use C11 atomics when supported under C and also under C++ with clang
#include <stdatomic.h>
  
typedef _Atomic uint32_t snde_atomicimagedata;


static GEOTYPES_INLINE void atomicpixel_accumulate(volatile snde_atomicimagedata *var,float toadd)
{
  // Use of a union like this is legal in C11/C++11, even under the strictest
  // aliasing rules
  union {
    uint32_t intval;
    snde_float32 floatval;
    char workbuf[4];  // having the char[] in here too is what gives us the exception to the strict aliasing rules 
  } oldvalue,newvalue; // ,workvalue;

  //  pthread_mutex_lock(&accumulatemutex);

  
  //oldvalue.floatval=atomicpixel_load(var);
  oldvalue.intval=atomic_load_explicit(var,memory_order_acquire);//,memory_order_consume);
  
  do {
    //memcpy(workvalue.workbuf,&oldvalue.intval,4);
    newvalue.floatval=oldvalue.floatval+toadd;
    //workvalue.floatval+=toadd;
    //memcpy(&newvalue.intval,&workvalue.workbuf,4);
  } while (!atomic_compare_exchange_strong_explicit(var,&oldvalue.intval,newvalue.intval,memory_order_seq_cst,memory_order_acquire)); //,memory_order_consume));


  //  pthread_mutex_unlock(&accumulatemutex);

}


static GEOTYPES_INLINE float atomicpixel_read(volatile  snde_atomicimagedata *src)
{
  // Use of a union like this is legal in C11, even under the strictest
  // aliasing rules
  union {
    uint32_t intval;
    snde_float32 floatval;
    char workbuf[4];
  } value; // ,workvalue;


  value.intval = atomic_load(src);
  return value.floatval;
}

  
#else // code below is for NOT using C11 atomics
#if defined(__GNUC__) || defined(__ATOMIC_ACQUIRE)
  // Gcc has its own atomics extensions that will work under c++
  // This should catch GCC and any other compiler that implements it based on
  // the __ATOMIC_AQUIRE symbol
  
typedef uint32_t snde_atomicimagedata;


static GEOTYPES_INLINE void atomicpixel_accumulate(volatile snde_atomicimagedata *var,float toadd)
{
  // Use of chars vs. other types is legal even under the strictest
  // aliasing rules
  
  union {
    float floatval;
    char workbuf[4];
  } oldfloatvalue,newfloatvalue; // ,workvalue;
  
  union {
    uint32_t intval;
    char workbuf[4];
  } oldvalue,newvalue; // ,workvalue;

  //  pthread_mutex_lock(&accumulatemutex);


  oldvalue.intval=__atomic_load_n(var,__ATOMIC_ACQUIRE);//,memory_order_consume);
  
  do {
    
    memcpy(&oldfloatvalue.workbuf[0],&oldvalue.workbuf[0],sizeof(float));
    newfloatvalue.floatval=oldfloatvalue.floatval+toadd;
    memcpy(&newvalue.workbuf[0],&newfloatvalue.workbuf[0],sizeof(float));

    
  } while (!__atomic_compare_exchange_n(var,&oldvalue.intval,newvalue.intval,0,__ATOMIC_SEQ_CST,__ATOMIC_ACQUIRE)); //,memory_order_consume));


  //  pthread_mutex_unlock(&accumulatemutex);

}

static GEOTYPES_INLINE float atomicpixel_read(volatile snde_atomicimagedata *src)
{
  union {
    uint32_t intval;
    char workbuf[4];
  } intvalue; // ,workvalue;

  union {
    float floatval;
    char workbuf[4];
  } floatvalue; // ,workvalue;

  intvalue.intval=__atomic_load_n(src,__ATOMIC_ACQUIRE);//,memory_order_consume);

  memcpy(&floatvalue.workbuf[0],&intvalue.workbuf[0],sizeof(float));
  
  return floatvalue.floatval;
}

  
  
#else // code below is for NOT using C11 atomics and NOT using Gcc atomics
#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <Windows.h>
typedef LONG snde_atomicimagedata;

static GEOTYPES_INLINE void atomicpixel_accumulate(volatile snde_atomicimagedata *var,float toadd)
{
  // Use of chars vs. other types is legal even under the strictest
  // aliasing rules
  
  union {
    float floatval;
    char workbuf[4];
  } oldfloatvalue,newfloatvalue; 
  
  union {
    LONG intval;
    char workbuf[4];
  } current,expected,next;

  //  pthread_mutex_lock(&accumulatemutex);

  // the Interlocked functions don't seem to have a simple read,
  // so we use compare exchange as a read by setting next and
  // expected to the same value, which is a no-op in all cases
  
  //current.intval = InterlockedCompareExchange(var,0,0);

  // Unecessary per https://stackoverflow.com/questions/779996/reading-interlocked-variables/19982243
  current.intval = *var;
  
  do {

    expected.intval = current.intval;
    memcpy(&oldfloatvalue.workbuf[0],&expected.workbuf[0],sizeof(float));
    newfloatvalue.floatval=oldfloatvalue.floatval+toadd;
    memcpy(&next.workbuf[0],&newfloatvalue.workbuf[0],sizeof(float));

    current.intval = InterlockedCompareExchange(var,next.intval,expected.intval);
    
  } while (current.intval != expected.intval);
  
  //  pthread_mutex_unlock(&accumulatemutex);

}

static GEOTYPES_INLINE float atomicpixel_read(volatile snde_atomicimagedata *src)
{
  union {
    uint32_t intval;
    char workbuf[4];
  } intvalue; // ,workvalue;

  union {
    float floatval;
    char workbuf[4];
  } floatvalue; // ,workvalue;

  //  pthread_mutex_lock(&accumulatemutex);

  // the Interlocked functions don't seem to have a simple read,
  // so we use compare exchange as a read by setting next and
  // expected to the same value, which is a no-op in all cases
  
  //current.intval = InterlockedCompareExchange(var,0,0);

  // Unecessary per https://stackoverflow.com/questions/779996/reading-interlocked-variables/19982243
  intvalue.intval= *src;

  memcpy(&floatvalue.workbuf[0],&intvalue.workbuf[0],sizeof(float));
  
  return floatvalue.floatval;
}
  
#else // NOT using C11 atomics, NOT using Gcc atomics, NOT using Win32 atomics
  
#ifdef __cplusplus
  // worst-case drop down to a single C++11 mutex: Note that this is per compilation unit,
  // so synchronization aross modules is not ensured!
#ifdef _MSC_VER
#pragma message("No atomic support available from C++ compiler; Dropping down to std::mutex (may be very slow)")
#else // _MSC_VER
#warning No atomic support available from C++ compiler; Dropping down to std::mutex (may be very slow)
#endif
#include <mutex>

  
typedef float snde_atomicimagedata;
static GEOTYPES_INLINE void atomicpixel_accumulate(volatile snde_atomicimagedata *var,float toadd)
{
  static std::mutex accumulatormutex;

  std::lock_guard<std::mutex> accumulatorlock;
  
  *var += toadd; 
}

// Note no atomicpixel_read implemented yet
  
#else // NOT using C11 atomics, NOT using Gcc atomics, NOT using Win32 atomics, NOT using C++
#warning No atomic support available from compiler; projection pixel corruption is possible!
typedef float snde_atomicimagedata;
static GEOTYPES_INLINE void atomicpixel_accumulate(volatile snde_atomicimagedata *var,float toadd)
{
  *var += toadd; 
}
  
#endif
#endif
#endif
#endif
  

#define M_PI_SNDE_COORD M_PI 

  
  //#endif /* 0 && SNDE_OPENCL*/
#endif /* __OPENCL_VERSION__ */




#ifdef __OPENCL_VERSION__
GEOTYPES_INLINE __global void *snde_memset(__global void *s, int c, size_t n)
{
  __global char *sc;
  size_t pos;
  sc = (__global char *)s;

  for (pos=0;pos < n;pos++) {
    sc[pos]=(char)c;
  }
  return s; 
}
#else // __OPENCL_VERSION__
#define snde_memset memset
#endif // __OPENCL_VERSION__




#define SNDE_DIRECTION_CCW 0 // counterclockwise
#define SNDE_DIRECTION_CW 1 // clockwise
  
static GEOTYPES_INLINE int snde_direction_flip(int direction)
{
  if (direction==SNDE_DIRECTION_CCW) {
    return SNDE_DIRECTION_CW;
  } else if (direction==SNDE_DIRECTION_CW) {
    return SNDE_DIRECTION_CCW;
  } else {
#ifndef __OPENCL_VERSION__
    assert(0); // bad direction
#endif
  }
  return SNDE_DIRECTION_CCW;
}

typedef struct _snde_coord3_int16 {
  int16_t coord[3];
} snde_coord3_int16;

typedef struct _snde_coord4 {
  snde_coord coord[4];
} snde_coord4;

typedef struct _snde_coord3 {
  snde_coord coord[3];
} snde_coord3;

typedef struct _snde_coord2 {
  snde_coord coord[2];
} snde_coord2;

typedef struct _snde_orientation3 {
  /* for point p, orientation represents q p q' + o  */
  snde_coord4 quat; // normalized quaternion ... represented as real (w) component, i (x) component, j (y) component, k (z) component, 
  snde_coord4 offset; // 4th coordinate of offset always 1.0
 
} snde_orientation3;

typedef struct _snde_rgba {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;
#ifndef __OPENCL_VERSION__
#ifdef __cplusplus
  operator double() const // need operator(double) because we don't (yet) have template code to check for existance of such a cast method. 
  {
    // operator(double) always returns NaN... 
    uint8_t NaNconstLE[4]={ 0x00,0x00,0xc0,0x7f };
    uint8_t NaNconstBE[4]={ 0x7f,0xc0,0x00,0x00 };

    if ((*((uint32_t*)NaNconstBE) & 0xff) == 0x00) {
      // big endian
      return (double)*((float *)NaNconstBE);
    } else {
      // little endian
      return (double)*((float *)NaNconstLE);
    }
    
  }  
#endif // __cplusplus
#endif // __OPENCL_VERSION__
} snde_rgba;


typedef struct _snde_rgbd {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;
  snde_float32 d;
#ifndef __OPENCL_VERSION__
#ifdef __cplusplus
  operator double() const // need operator(double) because we don't (yet) have template code to check for existance of such a cast method. 
  {
    // operator(double) always returns NaN... 
    uint8_t NaNconstLE[4]={ 0x00,0x00,0xc0,0x7f };
    uint8_t NaNconstBE[4]={ 0x7f,0xc0,0x00,0x00 };

    if ((*((uint32_t*)NaNconstBE) & 0xff) == 0x00) {
      // big endian
      return (double)*((float *)NaNconstBE);
    } else {
      // little endian
      return (double)*((float *)NaNconstLE);
    }
    
  }  
#endif // __cplusplus
#endif // __OPENCL_VERSION__
} snde_rgbd;

  
typedef struct _snde_orientation2 {
  // i.e. rotate point coordinates (rhs) by angle,
  // then add offset
  snde_coord angle; // radians
  snde_coord offset[2];

} snde_orientation2;



typedef struct _snde_axis3 {
  snde_coord coord[3];
} snde_axis3;


  /* Note: mesh edges (snde_edge) connect between triangles, 
     see struct snde_faceedge for connection between faces */ 
typedef struct _snde_edge {
  snde_index vertex[2];
  snde_index tri_a,tri_b;
  snde_index tri_a_prev_edge, tri_a_next_edge; /* counter-clockwise ordering */
  snde_index tri_b_prev_edge, tri_b_next_edge; /* counter-clockwise ordering */
} snde_edge;


typedef struct _snde_vertex_edgelist_index {
  snde_index edgelist_index;
  snde_index edgelist_numentries;
} snde_vertex_edgelist_index;

typedef struct _snde_triangle {
  snde_index edges[3];
  snde_index face; // topological face (3D or 2D depending on whether this is part of triangles or uv_triangles) this triangle is part of (index, relative to the part's first_face for a 3D face; relative to the firstuvface of this parameterization for a 2D face) 
} snde_triangle; // NOTE if this triangle does not exist, is invalid, etc, then face should be set to SNDE_INDEX_INVALID
  
typedef struct _snde_indexrange {
  snde_index start;
  snde_index len;
} snde_indexrange;
  
typedef struct _snde_trivertnormals {
  snde_coord3 vertnorms[3]; // vertex follow the order of vertices, counterclockwise as seen from the outside. The first vertex is the vertex from edges[0] that is NOT shared by the next_edge
} snde_trivertnormals;

typedef struct _snde_box3 {
  snde_index subbox[8];
  snde_index boxpolysidx;
  snde_index numboxpolys; 
} snde_box3;

typedef struct _snde_boxcoord3 {
  snde_coord3 min,max;
} snde_boxcoord3;

typedef struct _snde_box2 {
  snde_index subbox[4];
  snde_index boxpolysidx;
  snde_index numboxpolys; 
} snde_box2;

typedef struct _snde_boxcoord2 {
  snde_coord2 min,max;
} snde_boxcoord2;


  
typedef struct _snde_axis32 {
  snde_axis3 axis[2];
} snde_axis32;


  //typedef struct {
  //snde_coord cols[3];
  //} snde_row3;

typedef struct _snde_cmat23 {
  snde_coord3 row[2];
} snde_cmat23;



struct snde_nurbsubssurfaceuv {
  snde_index firstuvcontrolpoint,numuvcontrolpoints; /* control points locations giving reparameterized coordinates in terms of the (u,v) intrinsic parameterization */
  snde_index firstuvweight,numuvweights;
  snde_index firstuknot,numuknots;
  snde_index firstvknot,numvknots;
  snde_index udimension,vdimension;
  snde_index uorder,vorder;
  snde_index firsttrimcurvesegment,numtrimcurvesegments; // ***!!! separate storage for these should be defined... we should also probably define topological relations of the trim curves. subsurface is bounded by these segments (some of which may be pieces of the surface's bounding edges); these need to be in the surface's intrinsic (u,v) space so they are equivalent for multiple subsurfaces that cover the surface. 
  snde_index uv_patch_index; // which patch of uv space for this nurbsuv the control point coordinates correspond to
  
};

  

struct snde_nurbssurfaceuv {
  snde_index nurbssurface; /* surface we are reparameterizing */
  snde_index firstnurbssubsurfaceuv, numnurbssubsurfaceuv;
  snde_bool valid;
};
  
struct snde_nurbsuv {
  snde_index nurbspartnum;
  snde_index firstsurfaceuv,numsurfaceuvs; /* same length as nurbspart->numnurbssurfaces */

  snde_index numuvpatches; /* "patches" are regions in uv space that the vertices are represented in. There can be multiple images pointed to by the different patches.  Indexes in nurbssubsurfaceuv.uv_patch_index go from zero to numuvpatches. Those indexes will need to be added to the firstuvpatch of the snde_partinstance to get the correct patch indexes */ 

  
  snde_index firstuvbox, numuvboxes;
  snde_index firstuvboxpoly,numuvboxpolys;

};

  

  //struct snde_trimcurvesegment {  // replaced with nurbsedge
  //snde_index firstcontrolpointtrim,numcontrolpointstrim;
  //snde_index firstweight,numweights;
  //snde_index firstknot,numknots;
  //snde_index order;
  //};

struct snde_nurbsedge {
  snde_index firstcontrolpoint,numcontrolpoints; // index into 2D or 3D control point array according
  // to context: 3D for edges between surfaces, 2D for trim curves or edges in uv parameterization
  snde_index firstweight,numweights;
  snde_index firstknot,numknots;
  snde_index order;
  snde_bool valid;
};

  struct snde_meshededge { // additional edge structure showing mesh entries that comprise a face edge
  snde_index firstmeshedgeindex; // indices are stored in the topo_indices array, and refer to
                                 // mesh edges... Edges start from the vertex[0] of the
                                 // faceedge and go to the vertex[1] of the faceedge.
                                 // refer either to 3D edges or 2D uv_edges depending
                                 // on context... edges in CCW order!
  snde_index nummeshedgeindices;
  snde_bool valid; 
};
  

struct snde_nurbssurface {
  snde_index firstcontrolpoint,numcontrolpoints; /* NOTE: Control points are in part coordinates, and need to be transformed */
  snde_index firstweight,numweights;
  snde_index firstuknot,numuknots;
  snde_index firstvknot,numvknots;
  snde_index uorder,vorder;
  snde_index firsttrimcurvesegment,numtrimcurvesegments; /* trim curve segments form a closed loop in (u,v) space and are the projection of the edge onto this surface. Should be ordered in parallel with the faceedgeindices of the underlying snde_face, etc. struct snde_nurbsedge, but referring to 2D control point array by context */ 
  snde_bool uclosed,vclosed;
  snde_bool valid;
};

    
struct snde_meshedsurface { /* !!!*** Be careful about CPU <-> GPU structure layout differences ***!!! */
  /* indices into raw geometry */
  /* snde_orientation3 orientation; (orientation now */ /* orientation of this part relative to its environment */
  
  // winged edge triangular mesh
  snde_index firsttri,numtris; /* apply to triangles, refpoints, maxradius, normal, inplanemat... WITHIN the pool specified in the snde_part */

  snde_bool valid; // is this snde_meshedsurface valid and can be used? 
  snde_bool pad1[7];

  // formerly 104 bytes total (pad carefully because precedes nurbssurface in snde_face data structure
};

  struct snde_mesheduvsurface {

  /* !!!*** Possible problem: What if we need to go from a triangle index 
     to identify the uv surface? Answer: use the face index in the struct snde_triangle */

  snde_index firstuvtriindex, numuvtriindices; /* refer to a range of topo_indices referencing triangle indices for the triangles composing this surface. The triangles in mesheduv themselves line up with triangles in object. */

  //snde_coord2 tex_startcorner; /* (x,y) coordinates of one corner of parameterization (texture) space */
  //snde_coord2 tex_endcorner; /* (x,y) coordinates of other corner of parameterization (texture) space */
  
  //snde_index /*firstuvpatch,*/ numuvpatches; /* "patches" are regions in uv space that the vertices are represented in. There can be multiple images pointed to by the different patches.  Indexes in uv_patch_index go from zero to numuvpatches. They will need to be added to the firstuvpatch of the snde_partinstance */ 
  
};

  
struct snde_facevertex {
  snde_index meshedvertex; // could be SNDE_INDEX_INVALID if no meshed representation... either an intex into the vertices array or the uv_vertices array dpeending on context
  union {
    snde_coord3 ThreeD;
    snde_coord2 TwoD;
  } coord; // should match meshedvertex, if meshedvertex is valid
  snde_index firstfaceedgeindex;  // reference to list of edges for this vertex in the topo_indices array. Edges should be CCW ordered. 
  snde_index numfaceedgeindices;  
};
  
struct snde_faceedge {
  snde_index vertex[2]; // indices of facevertex 
  snde_index face_a; // indices of snde_face
  snde_index face_b;
  snde_index face_a_prev_edge, face_a_next_edge; // counter-clockwise ordering
  snde_index face_b_prev_edge, face_b_next_edge; // counter-clockwise ordering
  struct snde_meshededge meshededge;
  struct snde_nurbsedge nurbsedge; 
};
 
struct snde_face {
  snde_index firstfaceedgeindex; // refer to a list of faceedges within the topo_indices array
  snde_index numfaceedgeindices;
  snde_index patchnum; // index of snde_image within which this face is located. should be less than snde_parameterization.numuvpatches
  snde_index boundary_num; // boundary number within the part (NOT relative to first_topological) (valid for 3D faces but not 2D faces)
                           // boundary number of 0 means outer boundary, > 0 means void boundary
  union {
    struct {
      struct snde_meshedsurface meshed; 
      struct snde_nurbssurface nurbs; 
    } ThreeD;

    struct {
      struct snde_mesheduvsurface meshed;
      struct snde_nurbssurfaceuv nurbs;
    } TwoD;
  } surface;
  // 227 bytes total
  
};

  
struct snde_boundary {
  snde_index firstface,numfaces; // relative to the part's first_face
  // Each face contained within the part can have a meshed surface representation,
  // a NURBS surface representation, or both, controlled by the "valid" boolean within the
  // edges referenced in the meshedsurface
  // corresponding surface structure
  // 16 bytes total
};

union snde_topological {
  struct snde_boundary boundary;
  struct snde_face face;
  struct snde_faceedge faceedge;
  struct snde_facevertex facevertex;
  struct snde_nurbssurface nurbssurface;
  struct snde_meshedsurface meshedsurface; 
  //struct snde_trimcurvesegment trimcurvedsegment; 
  struct snde_nurbsedge nurbsedge; 
  struct snde_meshededge meshededge;
  // IDEA: Allow freelist to also be present within
  // the snde_topological array for a part, so allocation
  // and freeing can be performed on-GPU with non-locking data structures
  // (first entry would always have to be free)
};
  
struct snde_part {
  // ***!!! IMPORTANT: We loosen our "immutable" criteria here a little bit
  // because some entries (such as first_box, etc.) are generated by math functions
  // dependent on the part -- but we still want them in the structure -- conceptually
  // we place those entries as part of the math function output, even though
  // they are physically stored here. That means even though the
  // snde_part array is "immutable", those parts of it might actually be changed.
  // This is properly supported by the graphics_storage_manager, and all changes
  // require a write-lock to the underlying array. 

  // ***!!! IMPORTANT: Modify the Python Numpy struct definition in spatialnde2.i
  // also, whenever this gets changed. In addition, modify snde_part_initialize()
  // in geometry_ops.h

  // NOTE: Need strategy to identify presence of optional array data for this part
  // to replace the has_triangledata, etc. boolean
  snde_index firstboundary;  // firstboundary is outer boundary (relative to first_topological)
  snde_index numboundaries;  // all remaining boundaries are boundaries of voids.

  snde_index first_topo;
  snde_index num_topo; 

  snde_index first_topoidx;
  snde_index num_topoidxs; 
  
  snde_index first_face; // faces are an array of struct snde_face with the .ThreeD filled out. Relative to first_topo
  snde_index num_faces; 

  
  snde_index firsttri,numtris; /* apply to triangles, refpoints, maxradius, normal, inplanemat... broken into segments, one per surface. NOTE: any triangles that are not valid should have their .face set to SNDE_INDEX_INVALID */
  
  snde_index firstedge,numedges; /* apply to triangle edges of the mesh -- single pool for entire mesh */
  snde_index firstvertex,numvertices; /* vertex indices of the mesh, if present NOTE: Vertices must be transformed according to instance orientation prior to rendering */ /* These indices also apply to principal_curvatures and principal_tangent_axes, if present */
  snde_index first_vertex_edgelist,num_vertex_edgelist; // vertex edges for a particular vertex are listed in in CCW order
  
  // Boxes for identifying which triangle and face is intersected along a ray
  // NOTE: These are conceptually part of the boxes3d recording (see comment above on mutability!)
  snde_index firstbox,numboxes;  /* also applies to boxcoord */
  snde_index firstboxpoly,numboxpolys; /* NOTE: Boxes are in part coordinates, not world coordinates */

  
  snde_index firstboxnurbssurface,numboxnurbssurfaces; /* nurbs equivalent of boxpolys */

  snde_coord3 pivot_point; // rough (at least) center. Ideally could be center of mass

  snde_coord length_scale; // rough lengthscale around pivot_point

  snde_boxcoord3 bounding_box; 

  // NOTE: These are conceptually part of the vertex_kdtreee recording (see comment above on mutability!)
  snde_index first_vertex_kdnode; // index into vertex_kdtree
  snde_index num_vertex_kdnodes;

  snde_index first_triarea; // size should be the same as numtris. 
  snde_index first_vertarea; // size should be the same as numvertices. 
  
  snde_index reserved[12];
  
  
  snde_bool solid; // needed? 
  snde_bool has_triangledata; // Have we stored/updated refpoints, maxradius, normal, inplanemat  NOTE: Needs updated/removed!
  snde_bool has_curvatures; // Have we stored principal_curvatures/curvature_tangent_axes?  
  uint8_t pad1;
  uint8_t pad2[4];
  // formerly 81 bytes total  
};


struct snde_parameterization_patch {
  // ***!!! IMPORTANT: We loosen our "immutable" criteria here a little bit
  // because some entries (such as first_box, etc.) are generated by math functions
  // dependent on the part -- but we still want them in the structure -- conceptually
  // we place those entries as part of the math function output, even though
  // they are physically stored here. That means even though the
  // snde_part array is "immutable", those parts of it might actually be changed.
  // This is properly supported by the graphics_storage_manager, and all changes
  // require a write-lock to the underlying array. 

  
  snde_boxcoord2 domain; // image projection domain for this patch
  
  // Boxes for identifying which triangle and face is at a particular (u,v)
  // NOTE: These are conceptually part of the boxes2d recording (see comment above on mutability!)
  snde_index firstuvbox, numuvboxes; /* the first numuvpatches boxes correspond to the outer boxes for each patch */
  //snde_index firstuvboxcoord,numuvboxcoords; uv boxcoords allocated with uv boxes
  snde_index firstuvboxpoly,numuvboxpolys;

};

struct snde_parameterization {
  // ***!!! IMPORTANT: We loosen our "immutable" criteria here a little bit
  // because some entries (such as first_box, etc.) are generated by math functions
  // dependent on the part -- but we still want them in the structure -- conceptually
  // we place those entries as part of the math function output, even though
  // they are physically stored here. That means even though the
  // snde_part array is "immutable", those parts of it might actually be changed.
  // This is properly supported by the graphics_storage_manager, and all changes
  // require a write-lock to the underlying array. 

  // ***!!! IMPORTANT: Modify the Python Numpy struct definition in spatialnde2.i
  // also, whenever this gets changed. In addition, modify snde_part_initialize()
  // in geometry_ops.h
  
  // specific to a part;
  snde_index first_uv_topo;
  snde_index num_uv_topos;

  snde_index first_uv_topoidx;
  snde_index num_uv_topoidxs;

  snde_index firstuvtri, numuvtris; // storage region in uv_triangles... triangles must line up with triangles of underlying 3D mesh. Triangles identified topologically via the faces array. 

  
  snde_index firstuvface;  //  uv faces are struct snde_face with the .TwoD filled out. relative to first_uv_topo
  snde_index numuvfaces;  


  // The rest of these fields are the storage where triangle edges, vertices, 
  snde_index firstuvedge, numuvedges; /* edges in mesheduv may not line up with edges in object, 
					 but instead specify connectivity in (u,v) space. */
  snde_index firstuvvertex,numuvvertices; /* vertices in mesheduv may not line up with edges in object */
  snde_index first_uv_vertex_edgelist,num_uv_vertex_edgelist; // vertex edges for a particular vertex are listed in in CCW order

  snde_index firstuvpatch; // index into array of snde_parameterization_patch... number of elements used is numuvpatches
  snde_index numuvpatches; /* "patches" are regions in uv space that the vertices are represented in. There can be multiple images pointed to by the different patches.  Indexes  go from zero to numpatches. They will need to be added to the firstuvpatch of the snde_partinstance. Note that if numuvpatches > 1, the parameterization is not directly renderable and needs a processing step prior to rendering to combine the uv patches into a single parameterization space. NOTE: this parameter (numuvpatches) is not permitted to be changed once created (create an entirely new snde_parameterization) */

  snde_index reserved[16];
};
  



  
  

  //struct snde_assemblyelement {
  /***!!!*** NOTE: Because GPU code generally can't be 
      recursive, we will need to provide a utility routine
      that provides a flattened structure that can be 
      iterated over (should this even be in the database? ... 
      rather should probably dynamically generate flattened
      partinstance structure with CPU prior to bulk computation) !!!***/
  //snde_orientation3 orientation; /* orientation of this part/assembly relative to its parent */
  ///* if assemblynum is set, this element is a sub-assembly */
  //snde_index assemblynum;  
  ///*  if assemblynum is SNDE_INDEX_INVALID, 
  //    then one or more of the following can be set...*/
  
  //snde_index nurbspartnum;
  //snde_index nurbspartnum_reduceddetail;
  //snde_index meshedpartnum;
  //snde_index meshedpartnum_reduceddetail;
  
  //};

/* partinstance table created by walking the assembly structure and choosing level of detail */
struct snde_partinstance {
  /* (this isn't really in the database? Actually generated dynamically from the assembly structures) */
  snde_orientation3 orientation;  // Take a vector in the part's coordinates and gives a vector in world coordinates
  snde_orientation3 orientation_inverse; // Takes a vector in world coordinates and gives a vector in part coordinates
  //snde_index nurbspartnum; /* if nurbspartnum is SNDE_INDEX_INVALID, then there is a meshed representation only */
  snde_index partnum; // was meshedpartnum
  //std::string discrete_parameterization_name; -- really maps to mesheduvnmum /* index of the discrete parameterization */
  snde_index firstuvpatch; /* starting uv_patch # (snde_image) for this instance...  */ 
  snde_index uvnum; /* select which parameterization... can be SNDE_INDEX_INVALID or .idx of the parameterization */
  //snde_index imgbuf_extra_offset; // Additional offset into imgbuf, e.g. to select a particular frame of multiframe image data 
};
  

  
struct snde_image  {
  // ***!!!! Probably needs an update!
  snde_index projectionbufoffset; /* index into projection buffer array where the data for this image starts */
  snde_index weightingbufoffset; /* index into weighting buffer array where the data for this image starts */
  snde_index validitybufoffset; /* index into validity buffer array where the data for this image starts */
  //snde_index rgba_imgbufoffset; /* index into rgba image buffer array (if imgbufoffset is SNDE_INDEX_INVALID */
  
  snde_index nx,ny; // X and Y size (ncols, nrows) ... note Fortran style indexing
  snde_coord2 inival; /* Coordinates of the center of the first texel in image, 
			      in meaningful units (meters). Assume for the moment
			      that steps will always be positive. first coordinate is the 
			      x (column) position; second coordinate
			      is the y (row position). The edge of the first texel 
			      is at (startcorner.coord[0]-step.coord[0]/2,
			             startcorner.coord[1]-step.coord[1]/2) 

			      The coordinates of the endcorner are:
                                    (startcorner.coord[0]+step.coord[0]*(nx-0.5),
				    (startcorner.coord[1]+step.coord[1]*(ny-0.5))
                            */
  snde_coord2 step; /* step size per texel, in meaningful units. For the moment, at least, both should be positive */

  snde_index projection_strides[2]; // usually (1,nx)
  snde_index weighting_strides[2]; // usually (1,nx)
  snde_index validity_strides[2]; // usually (1,nx)

  
};


  struct snde_kdnode {
    snde_index cutting_vertex;
    snde_index left_subtree;
    snde_index right_subtree; 
  };

  
#ifdef __cplusplus
}
#endif


#endif /* SNDE_GEOMETRY_TYPES */
