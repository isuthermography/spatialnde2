#ifndef SNDE_REC_DISPLAY_TEXVERTEXARRAY_H
#define SNDE_REC_DISPLAY_TEXVERTEXARRAY_H

#include "snde/snde_types.h"
#include "snde/geometry_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


  // This function extract texture vertices from the winged edge data structure
  //  defined by meshedpart and the triangles, edges, and vertices arrays,
  // and emits them into the texvertex_arrays location.
  void snde_rec_display_texvertexarray_onetri(OCL_GLOBAL_ADDR const struct snde_parameterization *uv,
					      OCL_GLOBAL_ADDR const snde_triangle *uv_triangles,
					      OCL_GLOBAL_ADDR const snde_edge *uv_edges,
					      OCL_GLOBAL_ADDR const snde_coord2 *uv_vertices,
					      OCL_GLOBAL_ADDR snde_rendercoord *texvertex_arrays,
					      snde_index trianglenum);
  
#ifdef __cplusplus
};
#endif // __cplusplus

#endif // SNDE_REC_DISPLAY_TEXVERTEXARRAY_H

