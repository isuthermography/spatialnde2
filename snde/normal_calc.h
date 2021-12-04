#ifndef SNDE_NORMAL_CALC_H
#define SNDE_NORMAL_CALC_H

#include "snde/snde_types.h"
#include "snde/geometry_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

snde_coord3 snde_normalcalc_triangle(OCL_GLOBAL_ADDR const struct snde_part *part,
				     OCL_GLOBAL_ADDR const snde_triangle *part_triangles,
				     OCL_GLOBAL_ADDR const snde_edge *part_edges,
				     OCL_GLOBAL_ADDR const snde_coord3 *part_vertices,
				     snde_index trianglenum);
  

#ifdef __cplusplus
}
#endif // __cplusplus


#endif // SNDE_NORMAL_CALC_H
