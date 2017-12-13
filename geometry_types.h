#ifndef SNDE_GEOMETRY_TYPES
#define SNDE_GEOMETRY_TYPES

#ifdef __cplusplus
extern "C" {
#endif
  
#ifdef __OPENCL_VERSION__
/* if this is an opencl kernel */
typedef double snde_coord;
typedef float snde_imagedata;
typedef ulong snde_index;
typedef uint snde_shortindex;
typedef long snde_ioffset;
typedef char snde_bool;
  
#else
#ifdef USE_OPENCL

typedef cl_double snde_coord;
typedef cl_float snde_imagedata;
typedef cl_ulong snde_index;
typedef cl_uint snde_shortindex;
typedef cl_long snde_ioffset;
typedef cl_char snde_bool;

#else
typedef double snde_coord;
typedef float snde_imagedata;
typedef uint64_t snde_index;
typedef uint32_t snde_shortindex;
typedef int64_t snde_ioffset;
typedef char snde_bool;
#endif /* USE_OPENCL*/
#endif /* __OPENCL_VERSION__ */

#define SNDE_INDEX_INVALID (~((snde_index)0))

typedef struct {
  snde_coord offset[3];
  snde_coord quat[3]; // First 3 coordinates of normalized quaternion  
} snde_orientation3;

typedef struct {
  // i.e. rotate point coordinates (rhs) by angle,
  // then add offset
  snde_coord offset[2];
  snde_coord angle; // radians
} snde_orientation2;


typedef struct {
  snde_coord coord[3];
} snde_coord3;

typedef struct {
  snde_coord coord[2];
} snde_coord2;

typedef struct {
  snde_coord coord[3];
} snde_axis3;

typedef struct {
  snde_shortindex vertex[3];
} snde_triangleindices;

typedef struct {
  snde_index subbox[8];
  snde_index boxpolysidx;
} snde_box3;

typedef struct {
  snde_coord3 min,max;
} snde_boxcoord3;

typedef struct {
  snde_index subbox[4];
  snde_index boxpolysidx;
  snde_index numboxpolys; /* why we have this member for 2D boxes but not 3D boxes is unclear... should probably be consistent */
} snde_box2;

typedef struct {
  snde_coord2 min,max;
} snde_boxcoord2;


  
typedef struct {
  snde_axis3 axis[2];
} snde_axis32;


typedef struct {
  snde_coord cols[3];
} snde_row3;

typedef struct {
  snde_row3 row[2];
} snde_mat23;


  /* **** Need (X,Y,Z) nurbs curved explicitly representing the joints, so instead of testing against UV trim, can
     test against these curves, can guarantee no holes */ 

struct snde_nurbssurface {
  snde_index firstcontrolpoint,numcontrolpoints; /* NOTE: Control points are in part coordinates, and need to be transformed */
  snde_index firstweight,numweights;
  snde_index firstuknot,numuknots;
  snde_index firstvknot,numvknots;
  snde_index udimension,vdimension;
  snde_index uorder,vorder;
  snde_index firsttrimcurvesegment,numtrimcurvesegments; /* trim curve segments form a closed loop in (u,v) space */ 
  snde_bool uclosed,vclosed;
};

struct snde_trimcurvesegment {
  snde_index first2dcontrolpoint,num2dcontrolpoints;
  snde_index firstweight,numweights;
  snde_index firstknot,numknots;
  snde_index order;
};

struct snde_nurbsubssurfaceuv {
  snde_index first2dcontrolpoint,num2dcontrolpoints;
  snde_index firstweight,numweights;
  snde_index firstuknot,numuknots;
  snde_index firstvknot,numvknots;
  snde_index udimension,vdimension;
  snde_index uorder,vorder;
  snde_index trimcurvesegment;
  snde_index uv_patch_index; // which patch of uv space for this nurbsuv the control point coordinates correspond to
  
};

struct snde_nurbssurfaceuv {
  snde_index nurbssurface; /* surface we are reparameterizing */
  snde_index firstnurbssubsurfaceuv, numnurbssubsurfaceuv;
};
  
struct snde_nurbsuv {
  snde_index nurbspartnum;
  snde_index firstsurfaceuv,numsurfaceuvs; /* same length as nurbspart->numnurbssurfaces */

  snde_index numuvpatches; /* "patches" are regions in uv space that the vertices are represented in. There can be multiple images pointed to by the different patches.  Indexes in nurbssubsurfaceuv.uv_patch_index go from zero to numuvpatches. Those indexes will need to be added to the firstuvpatch of the snde_partinstance to get the correct patch indexes */ 

  
  snde_index firstuvbox, numuvboxes;
  snde_index firstuvboxpoly,numuvboxpoly;
  snde_index firstuvboxcoord,numuvboxcoords; 

};

  
struct snde_nurbspart {
  snde_orientation3 orientation; /* orientation of this part relative to its environment */
  snde_index firstnurbssurface,numnurbssurfaces;

  snde_index firstbox,numboxes;
  snde_index firstboxnurbssurface,numboxnurbssurfaces;
  snde_index firstboxcoord,numboxcoords; /* NOTE: Boxes are in part coordinates, and need to be transformed */
  
  snde_bool solid;
  
};



  
struct snde_meshedpart { /* !!!*** Be careful about CPU <-> GPU structure layout differences ***!!! */
  /* indices into raw geometry */
  snde_orientation3 orientation; /* orientation of this part relative to its environment */
  snde_index firstvertex,numvertices; /* NOTE: Vertices must be transformed according to orientation prior to rendering */ /* These indices also apply to principal_curvatures and principal_tangent_axes, if present */
  snde_index firsttri,numtris; /* apply to vertexidx, refpoints, maxradius, normal */
  

  /* indices into calculated fields */
  snde_index firstbox,numboxes;
  snde_index firstboxpoly,numboxpoly;
  snde_index firstboxcoord,numboxcoords; /* NOTE: Boxes are in part coordinates, not world coordinates */
  snde_index firstrefpoint,numrefpoints; /* NOTE: Refpoints are in part coordinates, not world coordinates */
  snde_index firstmaxradius,nummaxradius;
  snde_index firstnormal,numnormals; /* NOTE: Normals must be transformed according to orientation prior to rendering */
  // snde_index firstrefsize,numrefsizes;
  snde_index firstinplanemat,numinplanemats; 
  
  
  

  snde_bool solid; // If nonzero, this part is considered solid (fully enclosed), so the back side does not need to be rendered. Otherwise, it may be a bounded surface 
  
};


  
struct snde_mesheduv {
  snde_orientation2 orientation; /* orientation multiplied on right by coordinates of vertices to get output coordinates in parameterization */
  snde_index meshedpartnum;
  snde_index firstuvvertex,numuvvertices;
  snde_index firstuvtri, numuvtris;
  
  snde_index firstuvbox, numuvboxes;
  snde_index firstuvboxpoly,numuvboxpoly;
  snde_index firstuvboxcoord,numuvboxcoords; 
  
  snde_index firstinplane2uvcoords,numinplane2uvcoords;
  snde_index firstuvcoords2inplane,numuvcoords2inplane;

  snde_index firstuvpatch, numuvpatches; /* "patches" are regions in uv space that the vertices are represented in. There can be multiple images pointed to by the different patches.  Indexes in uv_patch_index go from zero to numuvpatches. They will need to be added to the firstuvpatch of the snde_partinstance */ 
  
};

struct snde_assemblyelement {
  /***!!!*** NOTE: Because GPU code generally can't be 
      recursive, we will need to provide a utility routine
      that provides a flattened structure that can be 
      iterated over (should this even be in the database? ... 
      rather should probably dynamically generate flattened
      partinstance structure with CPU prior to bulk computation) !!!***/
  snde_orientation3 orientation; /* orientation of this part/assembly relative to its parent */
  /* if assemblynum is set, this element is a sub-assembly */
  snde_index assemblynum;  
  /*  if assemblynum is SNDE_INDEX_INVALID, 
      then one or more of the following can be set...*/
  
  snde_index nurbspartnum;
  snde_index nurbspartnum_reduceddetail;
  snde_index meshedpartnum;
  snde_index meshedpartnum_reduceddetail;
  
};

/* partinstance table created by walking the assembly structure and choosing level of detail */
struct snde_partinstance {
  /* ***!!! (should this even be in the database? probably generated dynamically; see above) !!!***/
   snde_orientation3 orientation;
  snde_index nurbspartnum; /* if nurbspartnum is SNDE_INDEX_INVALID, then there is a meshed representation only */
  snde_index meshedpartnum;
  snde_index discrete_parameterizationnum; /* index of the discrete parameterization */
  snde_index firstuvpatch; /* starting uv_patch # (snde_image) for this instance... # of patches is an attribute of mesheduv, or nurbsuv */ 
  snde_index mesheduvnum; /* numvertices arrays must match between partinstance and mesheduv */

  /* ***!!! Also need to indicate "texture" image to reference uv parameterization ***/
};
  

  
struct snde_image  {
  snde_index imgbufoffset; /* index into image buffer array */
  snde_index ncols,nrows; 
  snde_coord2 inival; /* Coordinates of first data in image, in meaningful units */
  snde_coord2 step; /* step size per texel, in meaningful units */ 
};

  
#ifdef __cplusplus
}
#endif


#endif /* SNDE_GEOMETRY_TYPES */
