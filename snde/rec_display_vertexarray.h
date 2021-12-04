#ifndef SNDE_REC_DISPLAY_VERTEXARRAY_H
#define SNDE_REC_DISPLAY_VERTEXARRAY_H

#include "snde/snde_types.h"
#include "snde/geometry_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
  
  // This function extract vertices from the winged edge data structure
  //  defined by meshedpart and the triangles, edges, and vertices arrays,
  // and emits them into the vertex_arrays() location.
  void snde_rec_display_vertexarray_onetri(const struct snde_part *part,
					   const snde_triangle *part_triangles,
					   const snde_edge *part_edges,
					   const snde_coord3 *part_vertices,
					   snde_rendercoord *vertex_arrays,
					   snde_index trianglenum);


#ifdef __cplusplus
};
#endif // __cplusplus

#endif // SNDE_REC_DISPLAY_VERTEXARRAY_H

