#ifndef SNDE_GEOMETRY_TYPES
#define SNDE_GEOMETRY_TYPES

#ifdef __cplusplus
extern "C" {
#endif

  /* *** Changes to these type mappings must also go into 
     python definitions at bottom of geometry_types.i */
#ifdef __OPENCL_VERSION__
/* if this is an opencl kernel */
typedef double snde_coord;
typedef float snde_imagedata;
typedef ulong snde_index;
typedef uint snde_shortindex;
typedef long snde_ioffset;
typedef unsigned char snde_bool;
typedef unsigned char uint8_t;
  
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
typedef float snde_imagedata;
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

  
  //#endif /* 0 && SNDE_OPENCL*/
#endif /* __OPENCL_VERSION__ */

#define SNDE_INDEX_INVALID (~((snde_index)0))

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
  snde_coord3 offset;
  snde_coord pad1;
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

    if (*((uint32_t*)NaNconstBE) & 0xff == 0x00) {
      // big endian
      return (double)*((float *)NaNconstBE);
    } else {
      // little endian
      return (double)*((float *)NaNconstLE);
    }
    
  }  
#endif
} snde_rgba;
  
static inline snde_orientation3 snde_null_orientation3()
{
  snde_orientation3 null_orientation = { { 0.0, 0.0, 0.0 }, 0.0, {0.0, 0.0, 0.0, 1.0} }; /* unit (null) quaternion */
  return null_orientation;
}
  
typedef struct {
  // i.e. rotate point coordinates (rhs) by angle,
  // then add offset
  snde_coord offset[2];
  snde_coord angle; // radians
} snde_orientation2;



typedef struct {
  snde_coord coord[3];
} snde_axis3;

typedef struct {
  snde_index vertex[2];
  snde_index face_a,face_b;
  snde_index face_a_prev_edge, face_a_next_edge; /* counter-clockwise ordering */
  snde_index face_b_prev_edge, face_b_next_edge; /* counter-clockwise ordering */
} snde_edge;


typedef struct {
  snde_index edgelist_index;
  snde_index edgelist_numentries;
} snde_vertex_edgelist_index;

typedef struct {
  snde_index edges[3];
} snde_triangle;
  
typedef struct {
  snde_index start;
  snde_index len;
} snde_indexrange;
  
typedef struct {
  snde_coord3 vertnorms[3]; // vertex follow the order of vertices, counterclockwise as seen from the outside. The first vertex is the vertex from edges[0] that is NOT shared by the next_edge 
} snde_trinormals;

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
};
  
struct snde_nurbsuv {
  snde_index nurbspartnum;
  snde_index firstsurfaceuv,numsurfaceuvs; /* same length as nurbspart->numnurbssurfaces */

  snde_index numuvpatches; /* "patches" are regions in uv space that the vertices are represented in. There can be multiple images pointed to by the different patches.  Indexes in nurbssubsurfaceuv.uv_patch_index go from zero to numuvpatches. Those indexes will need to be added to the firstuvpatch of the snde_partinstance to get the correct patch indexes */ 

  
  snde_index firstuvbox, numuvboxes;
  snde_index firstuvboxpoly,numuvboxpoly;
  snde_index firstuvboxcoord,numuvboxcoords; 

};

  

struct snde_trimcurvesegment {
  snde_index firstcontrolpointtrim,numcontrolpointstrim;
  snde_index firstweight,numweights;
  snde_index firstknot,numknots;
  snde_index order;
};

struct snde_nurbsedge {
  snde_index firstcontrolpoint,numcontrolpoints;
  snde_index firstweight,numweights;
  snde_index firstknot,numknots;
  snde_index order;
};

struct snde_nurbssurface {
  snde_index firstcontrolpoint,numcontrolpoints; /* NOTE: Control points are in part coordinates, and need to be transformed */
  snde_index firstweight,numweights;
  snde_index firstuknot,numuknots;
  snde_index firstvknot,numvknots;
  snde_index uorder,vorder;
  snde_index firstnurbsedgeindex,numnurbsedgeindices; /* edges form a closed loop in (x,y,z) space. Note that multiple surfaces which share an edge need to refer to the same edge database entry */ 
  snde_index firsttrimcurvesegment,numtrimcurvesegments; /* trim curve segments form a closed loop in (u,v) space and are the projection of the edge onto this surface. Should be ordered in parallel with firstnurbsedgeindex, etc .*/ 
  snde_bool uclosed,vclosed;
};


  
struct snde_nurbspart {
  snde_orientation3 orientation; /* orientation of this part relative to its environment */
  snde_index firstnurbssurface,numnurbssurfaces;

  snde_index firstbox,numboxes;
  snde_index firstboxnurbssurface,numboxnurbssurfaces; /* nurbs equivalent of boxpolys */
  snde_index firstboxcoord,numboxcoords; /* NOTE: Boxes are in part coordinates, and need to be transformed */
  
  snde_bool solid;
  
};



  
struct snde_meshedpart { /* !!!*** Be careful about CPU <-> GPU structure layout differences ***!!! */
  /* indices into raw geometry */
  /* snde_orientation3 orientation; (orientation now */ /* orientation of this part relative to its environment */
  snde_index firsttri,numtris; /* apply to triangles, refpoints, maxradius, normal, inplanemat */
  snde_index firstedge,numedges; /* apply to edges, r */
  snde_index firstvertex,numvertices; /* NOTE: Vertices must be transformed according to orientation prior to rendering */ /* These indices also apply to principal_curvatures and principal_tangent_axes, if present */
  snde_index first_vertex_edgelist,num_vertex_edgelist; 

  /* indices into calculated fields */
  snde_index firstbox,numboxes;  /* also applies to boxcoord */
  snde_index firstboxpoly,numboxpolys; /* NOTE: Boxes are in part coordinates, not world coordinates */
    
 
  snde_bool solid; // If nonzero, this part is considered solid (fully enclosed), so the back side does not need to be rendered. Otherwise, it may be a bounded surface
  snde_bool has_triangledata; // Have we stored/updated refpoints, maxradius, normal, inplanemat
  snde_bool has_curvatures; // Have we stored principal_curvatures/curvature_tangent_axes? 
  snde_bool pad1[5];
};


  
struct snde_mesheduv {
  /* snde_orientation2 orientation; */ /* orientation multiplied on right by coordinates of vertices to get output coordinates in parameterization */
  /* snde_index meshedpartnum; */ /* Do we really need this? */
  snde_index firstuvtri, numuvtris; /* triangles in mesheduv MUST line up with triangles in object. */
  snde_index firstuvedge, numuvedges; /* triangles in mesheduv MUST line up with triangles in object. */
  snde_index firstuvvertex,numuvvertices;
  snde_index first_uv_vertex_edgelist,num_uv_vertex_edgelist; 

  snde_index firstuvbox, numuvboxes; /* the first numuvpatches boxes correspond to the outer boxes for each patch */
  snde_index firstuvboxpoly,numuvboxpoly;
  snde_index firstuvboxcoord,numuvboxcoords; 

  snde_coord2 tex_startcorner; /* (x,y) coordinates of one corner of parameterization (texture) space */
  snde_coord2 tex_endcorner; /* (x,y) coordinates of other corner of parameterization (texture) space */
  
  //snde_index /*firstuvpatch,*/ numuvpatches; /* "patches" are regions in uv space that the vertices are represented in. There can be multiple images pointed to by the different patches.  Indexes in uv_patch_index go from zero to numuvpatches. They will need to be added to the firstuvpatch of the snde_partinstance */ 
  
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
  snde_index nurbspartnum; /* if nurbspartnum is SNDE_INDEX_INVALID, then there is a meshed representation only */
  snde_index meshedpartnum;
  //std::string discrete_parameterization_name; -- really maps to mesheduvnmum /* index of the discrete parameterization */
  snde_index firstuvpatch; /* starting uv_patch # (snde_image) for this instance...  */ 
  snde_index numuvpatches; /* "patches" are regions in uv space that the vertices are represented in. There can be multiple images pointed to by the different patches.  Indexes in uv_patch_index go from zero to numuvpatches. They will need to be added to the firstuvpatch of the snde_partinstance */
  snde_index mesheduvnum; /* numvertices arrays must match between partinstance and mesheduv */
  snde_index imgbuf_extra_offset; // Additional offset into imgbuf, e.g. to select a particular frame of multiframe image data 
};
  

  
struct snde_image  {
  snde_index imgbufoffset; /* index into image buffer array */
  //snde_index rgba_imgbufoffset; /* index into rgba image buffer array (if imgbufoffset is SNDE_INDEX_INVALID */
  
  snde_index nx,ny; // X and Y size (ncols, nrows) ... note Fortran style indexing
  snde_coord2 startcorner; /* Coordinates of the edge of the first texel in image, 
			      in meaningful units (meters). first coordinate is the 
			      x (column) position; second coordinate
			      is the y (row position). The center of the first texel 
			      is at (startcorner.coord[0]+step.coord[0]/2,
			             startcorner.coord[1]+step.coord[1]/2) 

			      The coordinates of the endcorner are:
                                    (startcorner.coord[0]+step.coord[0]*nx,
				    (startcorner.coord[1]+step.coord[1]*ny)
                            */
  snde_coord2 step; /* step size per texel, in meaningful units */ 
};

  
#ifdef __cplusplus
}
#endif


#endif /* SNDE_GEOMETRY_TYPES */
