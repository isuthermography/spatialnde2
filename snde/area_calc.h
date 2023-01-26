#ifndef SNDE_AREA_CALC_H
#define SNDE_AREA_CALC_H

#include "snde/snde_types.h"
#include "snde/geometry_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

  snde_coord snde_areacalc_trianglearea(OCL_GLOBAL_ADDR const snde_triangle *part_triangles,
					 OCL_GLOBAL_ADDR const snde_edge *part_edges,
					 OCL_GLOBAL_ADDR const snde_coord3 *part_vertices,
					 snde_index trianglenum);


  snde_coord snde_areacalc_vertexarea(OCL_GLOBAL_ADDR const snde_triangle *part_triangles,
				       OCL_GLOBAL_ADDR const snde_edge *part_edges,
				       OCL_GLOBAL_ADDR const snde_coord3 *part_vertices,
				       OCL_GLOBAL_ADDR const snde_vertex_edgelist_index *part_vertex_edgelist_indices,
				       OCL_GLOBAL_ADDR const snde_index *part_vertex_edgelist,
				       OCL_GLOBAL_ADDR const snde_coord *part_trianglearea,
				       
				       snde_index vertexnum);
  
  
#ifdef __cplusplus
}
#endif // __cplusplus


#endif // SNDE_AREA_CALC_H
