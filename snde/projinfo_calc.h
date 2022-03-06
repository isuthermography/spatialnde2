#ifndef SNDE_PROJINFO_CALC_H
#define SNDE_PROJINFO_CALC_H

#include "snde/snde_types.h"
#include "snde/geometry_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

  void snde_projinfo_calc_one(// OCL_GLOBAL_ADDR const struct snde_part *part,
			      // OCL_GLOBAL_ADDR const struct snde_parameterization *param,
			      OCL_GLOBAL_ADDR const snde_triangle *part_triangles,
			      OCL_GLOBAL_ADDR const snde_edge *part_edges,
			      OCL_GLOBAL_ADDR const snde_coord3 *part_vertices,
			      OCL_GLOBAL_ADDR const snde_cmat23 *part_inplanemats,
			      OCL_GLOBAL_ADDR const snde_triangle *param_triangles,
			      OCL_GLOBAL_ADDR const snde_edge *param_edges,
			      OCL_GLOBAL_ADDR const snde_coord2 *param_vertices,
			      OCL_GLOBAL_ADDR snde_cmat23 *inplane2uvcoords,
			      OCL_GLOBAL_ADDR snde_cmat23 *uvcoords2inplane,
			      snde_index trianglenum);
  
#ifdef __cplusplus
};
#endif // __cplusplus


#endif // SNDE_PROJINF_CALC_H
