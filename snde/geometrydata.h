#ifndef SNDE_GEOMETRYDATA_H
#define SNDE_GEOMETRYDATA_H


#ifdef __cplusplus
extern "C" {
#endif
  /*** IMPORTANT: Ctypes definition in geometrydata.i must be changed in parallel with this. Also the calls to add_allocated_array() and add_follower_array() in geometry.hpp... ***/
  struct snde_geometrydata {
    double tol; // tolerance

    //struct snde_assemblyelement *assemblies;
    //allocatorbase  *assemblies_alloc; // really allocator<struct snde_assemblyelement> *
    
    //struct snde_partinstance *instances; (no longer in database) 
    //allocatorbase  *instances_alloc; // really allocator<struct snde_partinstance>*

    /* meshed 3D geometry */
    struct snde_part *parts; /* allocated separately */
    union snde_topological *topos; /* allocated separately */
    snde_index  *topo_indices; /* allocated separately. Pool of indices used by the topos */
    

    /* winged edge mesh format */
    snde_triangle *triangles; // allocated separately
    snde_coord3 *refpoints; // allocated with triangles  NOTE: Refpoints are in part coordinates, not world coordinates !!!*** ARE REFPOINTS STILL NEEDED?
    snde_coord *maxradius; // allocated with triangles !!!*** ARE MAXRADIUS STILL NEEDED?
    snde_coord3 *trinormals; // allocated with triangles, one normal/triangle NOTE: Normals are in part coordinates, not world coordinates.
    snde_cmat23 *inplanemats; // allocated with triangles
    

    snde_edge *edges; // allocated separately

    // polygon (triangle) vertices...
    snde_coord3 *vertices; // allocated separately
    snde_coord3 *vertnormals; // allocated with vertices (in part coordinates)

    snde_coord2 *principal_curvatures; // allocated with vertices
    snde_axis32 *curvature_tangent_axes; // allocated with vertices

    snde_vertex_edgelist_index *vertex_edgelist_indices; // allocated with vertices
    snde_index *vertex_edgelist; // allocated separately; vertex edges are listed in in CCW order 
    

    snde_kdnode *vertex_kdtree; // allocated separately

    
    // polygon vertexidx... representing vertices in a particular polygon. It is an integer array of vertex ids.... Each triangle specifies three vertices
    //snde_triangleindices *vertexidx;
    //allocatorbase  *triangle_alloc; // really allocator<snde_triangleindices> *
    
    //// polygon numvertices... representing number of vertices in a particular polygon. It is an integer array of numbers of vertices
    //snde_index *numvertices;
    

    // polygon vertexidx_indices... representing the first index into vertexidx corresponding to a particular polygon. It is an integer array of indexes into vertexidx
    //snde_index *vertexidx_indices;
    // This needs to be able to hang off of polygon_alloc



    /* NURBS 3D geometry */
    struct snde_nurbspart *nurbsparts;
    //allocatorbase  *nurbsparts_alloc; // really allocator<struct snde_nurbspart>*

    struct snde_nurbssurface *nurbssurfaces;
    //allocatorbase *nurbssurfaces_alloc; // really allocator<struct snde_nurbssurface> *

    // ***!!!*** Need to define edge topology for the object as a whole
    // ***!!!*** Each edge is a curve in 3-space, and it should point to a
    // ***!!!*** trimcurvesegment
    // ***!!!*** for each face that shares that edge. 

    snde_index *nurbsedgeindex; // separate allocator
    struct snde_trimcurvesegment *nurbstrimcurves; // allocated with nurbsedgeindex

    struct snde_nurbsedge *nurbsedge; // separate allocator


    snde_coord3 *controlpoints; /* control points for nurbssurface and nurbsedge; separate allocator */
    snde_coord3 *controlpointstrim; /* control points for snde_trimcurvesegment; separate allocator */
    snde_coord *weights; // separate allocator
    snde_coord *knots; // separate allocator



    /* boxes */
    snde_box3 *boxes;  // allocated by boxes_alloc... NOTE: Boxes are in part coordinates, not world coordinates 
    //allocatorbase *boxes_alloc; // really allocator<snde_box3> * 
    snde_boxcoord3 *boxcoord; // allocated by boxes_alloc
    snde_index *boxpolys; /* separate alocation */
    //allocatorbase *boxpolys_alloc; // really allocator<snde_index> *
    

    // Add polynum_by_vertex and/or adjacent_vertices? 

    // Meshed parameterizations
    snde_parameterization *uvs; /* array of uv parameterizations */
    snde_parameterization_patch *uv_patches;
    
    union snde_topological *uv_topos; /* allocated separately */
    snde_index  *uv_topo_indices; /* allocated separately. Pool of indices used by the uv_topos */

    snde_triangle *uv_triangles; /* allocated separately */
    snde_cmat23 *inplane2uvcoords;  /* allocated with uv_triangles ... multiply this by (x,y,1), where (x,y) are in inplanemat coordinates, to get u,v coordinates */  
    snde_cmat23 *uvcoords2inplane; /* allocated with uv_triangles ...  multiply this by (u,v,1) to get inplanemat coordinates */
    // (inplane2uvcoords and uvcoords2inplane together are sometimes referred to as "projinfo") 
    
    //snde_index *uv_patch_index; // uv_patch_index is indexed by and allocated with uv_triangle, like uv_vertexidx, and indicates which patch of uv space for this mesheduv the triangle vertices correspond to  
    
    snde_edge *uv_edges; /* allocated separately */
    
    // surface parameterization (texture coordinate) vertices...
    snde_coord2 *uv_vertices;
    snde_vertex_edgelist_index *uv_vertex_edgelist_indices; // allocated with uv_vertices

    snde_index *uv_vertex_edgelist; // allocated separately... vertex edges are listed in in CCW order

    

    
    //allocatorbase *uv_vertices_alloc; // really allocator<snde_coord2>

    //// uv polygon numvertices... representing number of vertices in a particular polygon. It is an integer array of numbers of vertices
    //snde_index *uv_numvertices;

    /* Note: compatibility of parameterization with model:
     * The numvertices segment for a part must 
     * match the uv_numvertices segment for a parameterization 
     * of the part */

    // uv vertexidx... representing vertices of a particular 2D triangle. It is an integer array of vertex ids.... 
    //snde_triangle *uv_vertexidx; // !!!*** Needs to be changed to winged edge!!!***
    //allocatorbase *uv_triangle_alloc; // really allocator<snde_triangleindices>

    
    
    // Continuous (NURBS) parameterizations
    struct snde_nurbsuv *nurbsuv;
    //allocatorbase *nurbsuv_alloc; // really allocator<struct snde_nurbsuv>*
    
    struct snde_nurbssurfaceuv *nurbssurfaceuv;
    //allocatorbase *nurbssurfaceuv_alloc; // really allocator<struct snde_nurbssurfaceuv>

    struct snde_nurbssubsurfaceuv *nurbssubsurfaceuv;
    //allocatorbase *nurbssubsurfaceuv_alloc; // really allocator<struct snde_nurbssubsurfaceuv> *

    snde_coord3 *uvcontrolpoints; /* control points for snde_nurbssubsurfaceuv; separate allocator */
    snde_coord *uvweights; // separate allocator
    snde_coord *uvknots; // separate allocator


    // 2D (uv-space) boxes 
    
    snde_box2 *uv_boxes;  // allocated by uv_boxes_alloc... NOTE: Boxes are in patch coordinates, not world coordinates 
    //allocatorbase *uv_boxes_alloc; // really allocator<snde_box2> *
    snde_boxcoord2 *uv_boxcoord; // allocated by uv_boxes_alloc

    snde_index *uv_boxpolys;
    //allocatorbase *uv_boxpolys_alloc; // really allocator<snde_index> *
    

    

    //snde_image *uv_images; /* allocated separately */
    //allocatorbase *uv_images_alloc; // really allocator<snde_image> *    
    
    // Generally can only operate with a single uv_image at a time... Shouldn't
    // be a problem because we can real-time composite the image from
    // multiple data stores using the TRM (which then will also need
    // to do RGBA conversion for rendering)

    // 
    // We can also think about breaking 2dobj into contiguous pieces (?)
    // Also want in some cases a reference to a
    // concrete instance of the parameterization (bitmap)
    // or region thereof. 

    snde_compleximagedata *compleximagebuf;

    snde_imagedata *imagebuf; // image data buffer used for projecting data into. This is also used by geometry_storage for anonymous allocations
    //allocatorbase *imagebuf_alloc; // really allocator<snde_imagedata>*

    snde_float32 *totals; // used for fusion_ndarray_recordings that usually go into imagebuf or compleximagebuf
    
    snde_coord *zbuffer; /* separately allocated */
    snde_coord *weightingbuf; /* separately allocated */
    snde_coord *validitybuf; /* separately allocated */
   
    
    // Suggest uv_patches_override and second buffer array
    // to kernel so that select patches can be overridden
    // on-demand for a particular operation.


    
    // these remaining arrays are used solely for rendering...
    // Do they really need to be here or can we use a regular storage manager for
    // them if they're not needed for anything but rendering?
    //snde_rendercoord *vertex_arrays; /* transformed vertex array for OpenSceneGraph / OpenGL (allocated separately) */
    //snde_rendercoord *texvertex_arrays; /* transformed texture vertex array for OpenSceneGraph / OpenGL (allocated separate) */
    //snde_trivertnormals *vertnormal_arrays; // allocated separately. For triangles, but are per vertex so three normals/triangle NOTE: Normals are in part coordinates, not world coordinates.
    
    snde_rgba *texbuffer; // allocated separately


    snde_coord *trianglearea; // allocated separately
    snde_coord *vertexarea; // allocated separately

  };
#ifdef __cplusplus
}
#endif


//#ifdef __cplusplus
//#include "snde/geometry.hpp"
//typedef snde::geometry snde_geometry;
//#else
//typedef struct snde_geometry snde_geometry;
//#endif

#ifdef __cplusplus
extern "C" {
#endif
  // C function definitions for geometry manipulation go here... 
  

#ifdef __cplusplus
}
#endif

#endif /* SNDE_GEOMETRYDATA_H */
