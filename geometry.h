#ifndef SNDE_GEOMETRY_H
#define SNDE_GEOMETRY_H


#ifdef __cplusplus
extern "C" {
#endif

  struct snde_geometrydata {
    double tol; // tolerance
    
    // polygon (triangle) vertices...
    snde_coord3 *vertices;
    //allocatorbase  *vertex_alloc; // really allocator<snde_coord3> *

    snde_coord2 *principal_curvatures; // allocated by vertex_alloc
    snde_axis32 *curvature_tangent_axes; // allocated by vertex_alloc
    
    
    // polygon vertexidx... representing vertices in a particular polygon. It is an integer array of vertex ids.... Each triangle specifies three vertices
    snde_triangleindices *vertexidx;
    //allocatorbase  *triangle_alloc; // really allocator<snde_triangleindices> *
    
    //// polygon numvertices... representing number of vertices in a particular polygon. It is an integer array of numbers of vertices
    //snde_index *numvertices;
    

    // polygon vertexidx_indices... representing the first index into vertexidx corresponding to a particular polygon. It is an integer array of indexes into vertexidx
    //snde_index *vertexidx_indices;
    // This needs to be able to hang off of polygon_alloc

    snde_coord3 *refpoints; // allocated by triangle_alloc  NOTE: Refpoints are in part coordinates, not world coordinates
    snde_coord *maxradius; // allocated by triangle_alloc
    snde_coord3 *normal; // allocated by triangle_alloc NOTE: Normals are in part coordinates, not world coordinates
    
    snde_mat23 *inplanemat; // allocated by triangle_alloc

    snde_box3 *boxes;  // allocated by boxes_alloc... NOTE: Boxes are in part coordinates, not world coordinates 
    //allocatorbase *boxes_alloc; // really allocator<snde_box3> * 

    snde_boxcoord3 *boxcoord; // allocated by boxes_alloc

    snde_index *boxpolys;
    //allocatorbase *boxpolys_alloc; // really allocator<snde_index> *
    

    // Add polynum_by_vertex and/or adjacent_vertices? 

    
    // surface parameterization (texture coordinate) vertices...
    snde_coord2 *uv_vertices; 
    //allocatorbase *uv_vertices_alloc; // really allocator<snde_coord2>

    //// uv polygon numvertices... representing number of vertices in a particular polygon. It is an integer array of numbers of vertices
    //snde_index *uv_numvertices;

    /* Note: compatibility of parameterization with model:
     * The numvertices segment for a part must 
     * match the uv_numvertices segment for a parameterization 
     * of the part */

    // uv vertexidx... representing vertices of a particular 2D triangle. It is an integer array of vertex ids.... 
    snde_triangleindices *uv_vertexidx;
    //allocatorbase *uv_triangle_alloc; // really allocator<snde_triangleindices>


    snde_box2 *uv_boxes;  // allocated by uv_boxes_alloc... NOTE: Boxes are in part coordinates, not world coordinates 
    //allocatorbase *uv_boxes_alloc; // really allocator<snde_box2> *

    snde_index *uv_boxpolys;
    //allocatorbase *uv_boxpolys_alloc; // really allocator<snde_index> *
    
    snde_boxcoord2 *uv_boxcoord; // allocated by uv_boxes_alloc

    snde_mesheduv *mesheduv; /* array of meshed uv parameterizations */
    //allocatorbase *mesheduv_alloc; // really allocator<struct snde_mesheduv> *

    
    snde_index *uv_patch_index; // uv_patch_index is indexed by triangle, like uv_vertexidx, and indicates which patch of uv space for this mesheduv the triangle vertices correspond to  


    snde_image *uv_patches;
    //allocatorbase *uv_patches_alloc; // really allocator<snde_image> *

    
    snde_mat23 *uv2texcoords;
    //allocatorbase *inplaneuvcoords_alloc; // really allocator<snde_mat23> *
    snde_mat23 *uvcoords2inplane;



    struct snde_nurbssurface *nurbssurfaces;
    //allocatorbase *nurbssurfaces_alloc; // really allocator<struct snde_nurbssurface> *

    struct snde_trimcurvesegment *nurbstrimcurves;
    //allocatorbase *nurbstrimcurves_alloc; // really allocator<struct snde_trimcurvesegment>*

    struct snde_nurbssubsurfaceuv *nurbssubsurfaceuv;
    //allocatorbase *nurbssubsurfaceuv_alloc; // really allocator<struct snde_nurbssubsurfaceuv> *

    struct snde_nurbssurfaceuv *nurbssurfaceuv;
    //allocatorbase *nurbssurfaceuv_alloc; // really allocator<struct snde_nurbssurfaceuv>

    struct snde_nurbsuv *nurbsuv;
    //allocatorbase *nurbsuv_alloc; // really allocator<struct snde_nurbsuv>*

    struct snde_nurbspart *nurbsparts;
    //allocatorbase  *nurbsparts_alloc; // really allocator<struct snde_nurbspart>*
    
    
    // We can also think about breaking 2dobj into contiguous pieces (?)
    // Also want in some cases a reference to a
    // concrete instance of the parameterization (bitmap)
    // or region thereof. 

    struct snde_assemblyelement *assemblies;
    //allocatorbase  *assemblies_alloc; // really allocator<struct snde_assemblyelement> *
    
    struct snde_partinstance *instances;
    //allocatorbase  *instances_alloc; // really allocator<struct snde_partinstance>*
    

    snde_imagedata *imagebuf;
    //allocatorbase *imagebuf_alloc; // really allocator<snde_imagedata>*

    snde_imagedata *zbuffer; /* allocated by imagebuf_alloc */

    
    
    // Suggest uv_patches_override and second buffer array
    // to kernel so that select patches can be overridden
    // on-demand for a particular operation. 
    
  };
#ifdef __cplusplus
}
#endif


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

#endif /* SNDE_GEOMETRY_H */
