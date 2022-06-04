#ifndef SNDE_KDTREE_KNN_H
#define SNDE_KDTREE_KNN_H

#ifndef __OPENCL_VERSION__
// if this is not an OpenCL kernel
#include <stdint.h>

#include "snde/snde_types.h"
#include "snde/geometry_types.h"

#define KDTREE_GLOBAL
#define KDTREE_LOCAL

#else // __OPENCL_VERSION__

#define KDTREE_GLOBAL __global
#define KDTREE_LOCAL __local

#endif // __OPENCL_VERSION__

#ifdef __cplusplus
extern "C" {
#endif


  snde_index snde_kdtree_knn_one(KDTREE_GLOBAL struct snde_kdnode *tree,
				 KDTREE_GLOBAL uint32_t *nodemask,
				 KDTREE_GLOBAL snde_coord *vertices,
				 KDTREE_LOCAL snde_index *nodestack, // (max_depth+1)*sizeof(snde_index)
				 KDTREE_LOCAL uint8_t *statestack, // (max_depth+1)*sizeof(uint8_t)
				 KDTREE_LOCAL snde_coord *bboxstack, // (max_depth+1)*sizeof(snde_coord)*2
				 KDTREE_GLOBAL snde_coord *to_find,
				 //KDTREE_GLOBAL snde_coord *dist_squared_out,
				 uint32_t ndim,
				 uint32_t max_depth,
				 snde_index find_index);
  
  
#ifdef __cplusplus
}
#endif


#endif // SNDE_KDTREE_KNN_H
