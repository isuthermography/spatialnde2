

#ifndef SNDE_GEOMETRY_TYPES
#define SNDE_GEOMETRY_TYPES

#ifndef __OPENCL_VERSION__
// if this is not an OpenCL kernel
#include <assert.h>
#include <stdint.h>
#include <string.h>
#endif

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
typedef double snde_coord;
typedef float snde_rendercoord;
typedef float snde_imagedata;
typedef ulong snde_index;
typedef uint snde_shortindex;
typedef long snde_ioffset;
typedef unsigned char snde_bool;
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
 
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
  
#else
  //#if 0 && defined(SNDE_OPENCL)

//typedef cl_double snde_coord;
//typedef cl_float snde_imagedata;
//typedef cl_ulong snde_index;
//typedef cl_uint snde_shortindex;
//typedef cl_long snde_ioffset;
//typedef cl_char snde_bool;

//#else
typedef double snde_coord;
typedef float snde_rendercoord;
typedef float snde_imagedata;

#if (defined(__STDC_VERSION__) && (__STDC_VERSION__>= 201112L) && !defined(__STDC_NO_ATOMICS__)) || (defined(__cplusplus) && defined(__clang__))
  // Use C11 atomics when supported under C and also under C++ with clang
  // (can add other compilers that support C11 atomics under C++ as well)
typedef _Atomic uint32_t snde_atomicimagedata;


static GEOTYPES_INLINE void atomicpixel_accumulate(volatile snde_atomicimagedata *var,float toadd)
{
  // Use of a union like this is legal in C11, even under the strictest
  // aliasing rules
  union {
    uint32_t intval;
    float32_t floatval;
    char workbuf[4];
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
  
#else
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

  
#else
#ifdef WIN32
  #include <winnt.h>
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
  
  current.intval = InterlockedCompareExchange(var,0,0);
  
  do {

    expected.intval = current.intval;
    memcpy(&oldfloatvalue.workbuf[0],&expected.workbuf[0],sizeof(float));
    newfloatvalue.floatval=oldfloatvalue.floatval+toadd;
    memcpy(&next.workbuf[0],&newfloatvalue.workbuf[0],sizeof(float));

    current.intval = InterlockedCompareExchange(var,next.intval,expected.intval);
    
  } while (current.intval != expected.intval);
  
  //  pthread_mutex_unlock(&accumulatemutex);

}

  
#else
  
#ifdef __cplusplus
  // worst-case drop down to a single C++11 mutex: Note that this is per compilation unit,
  // so synchronization aross modules is not ensured!

#warning No atomic support available from C++ compiler; Dropping down to std::mutex (may be very slow)
#include <mutex>

  
typedef float snde_atomicimagedata;
static GEOTYPES_INLINE void atomicpixel_accumulate(volatile snde_atomicimagedata *var,float toadd)
{
  static std::mutex accumulatormutex;

  std::lock_guard<std::mutex> accumulatorlock;
  
  *var += toadd; 
}
  
  
#else
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
  
typedef uint32_t snde_shortindex;
typedef unsigned char snde_bool;

  // Don't specify 64 bit integers in terms of int64_t/uint64_t to work around
  // https://github.com/swig/swig/issues/568
  //typedef uint64_t snde_index;
  //typedef int64_t snde_ioffset;
#ifdef SIZEOF_LONG_IS_8
  typedef unsigned long snde_index;
  typedef long snde_ioffset;
#else
  typedef unsigned long long snde_index;
  typedef long long snde_ioffset;
#endif

#define M_PI_SNDE_COORD M_PI // change to M_PI_F if you change snde_coord to float

  
  //#endif /* 0 && SNDE_OPENCL*/
#endif /* __OPENCL_VERSION__ */

#define SIZEOF_SNDE_INDEX 8 // must be kept consistent with typedefs!
  
#define SNDE_INDEX_INVALID (~((snde_index)0))
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
  
typedef struct {
  snde_coord coord[4];
} snde_coord4;

typedef struct {
  snde_coord coord[3];
} snde_coord3;

typedef struct {
  snde_coord coord[2];
} snde_coord2;

typedef struct {
  /* for point p, orientation represents q p q' + o  */
  snde_coord4 offset; // 4th coordinate of offset always zero
  snde_coord4 quat; // normalized quaternion ... represented as , i (x) component, j (y) component, k (z) component, real (w) component
} snde_orientation3;

typedef struct {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;
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
#endif
} snde_rgba;
  
  
typedef struct {
  // i.e. rotate point coordinates (rhs) by angle,
  // then add offset
  snde_coord offset[2];
  snde_coord angle; // radians
} snde_orientation2;



typedef struct {
  snde_coord coord[3];
} snde_axis3;


  /* Note: mesh edges (snde_edge) connect between triangles, 
     see struct snde_faceedge for connection between faces */ 
typedef struct {
  snde_index vertex[2];
  snde_index tri_a,tri_b;
  snde_index tri_a_prev_edge, tri_a_next_edge; /* counter-clockwise ordering */
  snde_index tri_b_prev_edge, tri_b_next_edge; /* counter-clockwise ordering */
} snde_edge;


typedef struct {
  snde_index edgelist_index;
  snde_index edgelist_numentries;
} snde_vertex_edgelist_index;

typedef struct {
  snde_index edges[3];
  snde_index face; // topological face (3D or 2D depending on whether this is part of triangles or uv_triangles) this triangle is part of (index, relative to first_topo for a 3D face; relative to 0 for a 2D face) 
} snde_triangle; // NOTE if this triangle does not exist, is invalid, etc, then face should be set to SNDE_INDEX_INVALID
  
typedef struct {
  snde_index start;
  snde_index len;
} snde_indexrange;
  
typedef struct {
  snde_coord3 vertnorms[3]; // vertex follow the order of vertices, counterclockwise as seen from the outside. The first vertex is the vertex from edges[0] that is NOT shared by the next_edge
} snde_trivertnormals;

typedef struct {
  snde_index subbox[8];
  snde_index boxpolysidx;
  snde_index numboxpolys; 
} snde_box3;

typedef struct {
  snde_coord3 min,max;
} snde_boxcoord3;

typedef struct {
  snde_index subbox[4];
  snde_index boxpolysidx;
  snde_index numboxpolys; 
} snde_box2;

typedef struct {
  snde_coord2 min,max;
} snde_boxcoord2;


  
typedef struct {
  snde_axis3 axis[2];
} snde_axis32;


  //typedef struct {
  //snde_coord cols[3];
  //} snde_row3;

typedef struct {
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
  bool valid;
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
  snde_index imagenum; // index of snde_image within which this face is located. should be less than snde_parameterization.numuvimages
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
  snde_index firstface,numfaces; // relative to first_topo 
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

  snde_index firstboundary;  // firstboundary is outer boundary (relative to first_topological)
  snde_index numboundaries;  // all remaining boundaries are boundaries of voids.

  snde_index first_topo;
  snde_index num_topo; 

  snde_index first_topoidx;
  snde_index num_topoidxs; 
  
  
  snde_index firsttri,numtris; /* apply to triangles, refpoints, maxradius, normal, inplanemat... broken into segments, one per surface. NOTE: any triangles that are not valid should have their .face set to SNDE_INDEX_INVALID */
  
  snde_index firstedge,numedges; /* apply to triangle edges of the mesh -- single pool for entire mesh */
  snde_index firstvertex,numvertices; /* vertex indices of the mesh, if present NOTE: Vertices must be transformed according to instance orientation prior to rendering */ /* These indices also apply to principal_curvatures and principal_tangent_axes, if present */
  snde_index first_vertex_edgelist,num_vertex_edgelist; // vertex edges for a particular vertex are listed in in CCW order
  
  // Boxes for identifying which triangle and face is intersected along a ray
  snde_index firstbox,numboxes;  /* also applies to boxcoord */
  snde_index firstboxpoly,numboxpolys; /* NOTE: Boxes are in part coordinates, not world coordinates */
  snde_index firstboxnurbssurface,numboxnurbssurfaces; /* nurbs equivalent of boxpolys */
  
  snde_bool solid;
  snde_bool has_triangledata; // Have we stored/updated refpoints, maxradius, normal, inplanemat
  snde_bool has_curvatures; // Have we stored principal_curvatures/curvature_tangent_axes?  
  uint8_t pad1;
  uint8_t pad2[4];
  // formerly 81 bytes total  
};


struct snde_parameterization_patch {
  // Boxes for identifying which triangle and face is at a particular (u,v)
  snde_index firstuvbox, numuvboxes; /* the first numuvpatches boxes correspond to the outer boxes for each patch */
  //snde_index firstuvboxcoord,numuvboxcoords; uv boxcoords allocated with uv boxes
  snde_index firstuvboxpoly,numuvboxpolys;

};

struct snde_parameterization {
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

  snde_index firstuvpatch; // index into array of snde_parameterization_patch... number of elements used is numuvimages
  snde_index numuvimages; /* "images" are regions in uv space that the vertices are represented in. There can be multiple images pointed to by the different patches.  Indexes  go from zero to numuvimage. They will need to be added to the firstuvimage of the snde_partinstance. Note that if numuvimages > 1, the parameterization is not directly renderable and needs a processing step prior to rendering to combine the uv images into a single parameterization space. NOTE: this parameter (numuvimages) is not permitted to be changed once created (create an entirely new snde_parameterization) */
    
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
   snde_orientation3 orientation;
  //snde_index nurbspartnum; /* if nurbspartnum is SNDE_INDEX_INVALID, then there is a meshed representation only */
  snde_index partnum; // was meshedpartnum
  //std::string discrete_parameterization_name; -- really maps to mesheduvnmum /* index of the discrete parameterization */
  snde_index firstuvimage; /* starting uv_patch # (snde_image) for this instance...  */ 
  snde_index uvnum; /* select which parameterization... can be SNDE_INDEX_INVALID or .idx of the parameterization */
  //snde_index imgbuf_extra_offset; // Additional offset into imgbuf, e.g. to select a particular frame of multiframe image data 
};
  

  
struct snde_image  {
  snde_index imgbufoffset; /* index into image buffer array */
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
  //snde_index nextimage; // index of alternate image to be used if data in this image shows as NaN, fully transparent, or is outside the image boundary... set to SNDE_INDEX_INVALID if there should be no nextimage.
};

  
#ifdef __cplusplus
}
#endif


#endif /* SNDE_GEOMETRY_TYPES */
