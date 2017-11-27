#ifndef SNDE_GEOMETRY
#define SNDE_GEOMETRY


/* Thoughts/updates: 
 * For OpenGL compatibility, drop everything down to triangle meshes
 * OpenGL does not support separate UV index. Per malexander in https://www.opengl.org/discussion_boards/showthread.php/183513-How-to-use-UV-Indices-in-OpenGL-4-4  
 * this can be addressed by passing a "Buffer Texture" object to the
 * geometry shader https://stackoverflow.com/questions/7954927/passing-a-list-of-values-to-fragment-shader,  which can then (use this to determine texture 
 * coordinates on the actual texture?) (Provide the correct texture
 * index on gl_PrimitiveID? and texture id on a custom output?
 * ... also support 32-bit index storage (snde_shortindex) that gets
 * immediately multiplied out because OpenGL doesn't support 
 * ... 64 bit indices yet. 

*/


#include "geometry_types.h"

namespace snde {  
  
  /* *** Where to store landmarks ***/
  /* Where to store frames? ***/
  
  class geometry {
    double tol; // tolerance

    // polygon (triangle) vertices...
    snde_coord3 *vertices;
    allocator<snde_coord3> *vertex_alloc;

    snde_coord2 *principal_curvatures; // allocated by vertex_alloc
    snde_axis32 *curvature_tangent_axes; // allocated by vertex_alloc
    

    // polygon vertexidx... representing vertices in a particular polygon. It is an integer array of vertex ids.... Each triangle specifies three vertices
    snde_triangleindices *vertexidx;
    allocator<snde_triangleindices>  *triangle_alloc;

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
    allocator<snde_box3> *boxes_alloc;

    snde_index *boxpolys;
    allocator<snde_index> *boxpolys_alloc;
    
    snde_boxcoord3 *boxcoord; // allocated by boxes_alloc

    // Add polynum_by_vertex and/or adjacent_vertices? 

    
    // surface parameterization (texture coordinate) vertices...
    snde_coord2 *uv_vertices; 
    allocator<snde_coord2> *uv_vertices_alloc;

    //// uv polygon numvertices... representing number of vertices in a particular polygon. It is an integer array of numbers of vertices
    //snde_index *uv_numvertices;

    /* Note: compatibility of parameterization with model:
     * The numvertices segment for a part must 
     * match the uv_numvertices segment for a parameterization 
     * of the part */

    // uv vertexidx... representing vertices of a particular 2D triangle. It is an integer array of vertex ids.... 
    snde_triangleindices *uv_vertexidx;
    allocator<snde_triangleindices> *uv_triangle_alloc;


    snde_box2 *uv_boxes;  // allocated by uv_boxes_alloc... NOTE: Boxes are in part coordinates, not world coordinates 
    allocator<snde_box2> *uv_boxes_alloc;

    snde_index *uv_boxpolys;
    allocator<snde_index> *uv_boxpolys_alloc;
    
    snde_boxcoord2 *uv_boxcoord; // allocated by uv_boxes_alloc

    struct snde_mesheduv *mesheduv; /* array of meshed uv parameterizations */
    allocator<struct snde_mesheduv> mesheduv_alloc;

    
    snde_index *uv_patch_index; // uv_patch_index is indexed by triangle, like uv_vertexidx, and indicates which patch of uv space for this mesheduv the triangle vertices correspond to  


    snde_image *uv_patches;
    allocator<snde_image> *uv_patches_alloc;

    
    snde_mat23 *uv2texcoords;
    allocator<snde_mat23> *inplaneuvcoords_alloc;
    snde_mat23 *uvcoords2inplane;


    struct snde_nurbspart *nurbsparts;
    allocator<struct snde_nurbspart> *nurbsparts_alloc;

    struct snde_nurbssurface *nurbssurfaces;
    allocator<struct snde_nurbssurface> *nurbssurfaces_alloc;

    struct snde_trimcurvesegment *nurbstrimcurves;
    allocator<struct snde_trimcurvesegment> *nurbstrimcurves_alloc;

    struct snde_nurbssubsurfaceuv *nurbssubsurfaceuv;
    allocator<struct snde_nurbssubsurfaceuv> *nurbssubsurfaceuv_alloc;

    struct snde_nurbssurfaceuv *nurbssurfaceuv;
    allocator<struct snde_nurbssurfaceuv> *nurbssurfaceuv_alloc;

    struct snde_nurbsuv *nurbsuv;
    allocator<struct snde_nurbsuv> nurbsuv_alloc;

    struct snde_nurbspart *nurbsparts;
    allocator<struct snde_nurbspart> nurbsparts_alloc;
    
    
    // We can also think about breaking 2dobj into contiguous pieces (?)
    // Also want in some cases a reference to a
    // concrete instance of the parameterization (bitmap)
    // or region thereof. 

    struct snde_assemblyelement *assemblies;
    allocator<struct snde_assemblyelement>  *assemblies_alloc;
    
    struct snde_partinstance *instances;
    allocator<struct snde_partinstance>  *instances_alloc;
    

    snde_imagedata *imagebuf;
    allocator<snde_imagedata> *imagebuf_alloc;

    snde_imagedata *zbuffer; /* allocated by imagebuf_alloc */

    
    
    // Suggest uv_patches_override and second buffer array
    // to kernel so that select patches can be overridden
    // on-demand for a particular operation. 
    
  };
  
}


#endif /* SNDE_GEOMETRY */
