#ifndef __OPENCL_VERSION__
#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/vecops.h"
#include "snde/geometry_ops.h"

#include "snde/normal_calc.h"
#endif

/* implicit include of geometry_types.h */
/* implicit include of vecops.h */

snde_coord3 snde_normalcalc_triangle(OCL_GLOBAL_ADDR const struct snde_part *part,
				     OCL_GLOBAL_ADDR const snde_triangle *part_triangles,
				     OCL_GLOBAL_ADDR const snde_edge *part_edges,
				     OCL_GLOBAL_ADDR const snde_coord3 *part_vertices,
				     snde_index trianglenum)
{
  snde_coord3 triverts[3];
  
  // For the moment, this calculates a normal per triangle and stores it
  // for all vertices of the triangle, and for the triangle as a whole;
  

  /* traverse edges of this triangle and extract vertex coordinates -> triverts*/
  get_we_triverts_3d(part_triangles,trianglenum,part_edges,part_vertices,triverts);

  /* got vertex coordinates in triverts */

  snde_coord3 V,W,N;

  // V = normalized trivert[1]-trivert[0]
  subvecvec3(triverts[1].coord,triverts[0].coord,V.coord);
  normalizevec3(V.coord);
  

  // W = normalized trivert[2]-trivert[0]
  subvecvec3(triverts[2].coord,triverts[0].coord,W.coord);
  normalizevec3(W.coord);
  

  // N = V cross W
  crossvecvec3(V.coord,W.coord,N.coord);

  // If vector from 0th to 1st vertex and vector from 0th to 2nd
  // vertex are too close to parallel, find another vertex
    
  const snde_coord min_cross_product = 1e-3f;
  snde_bool tooparallel=normvec3(N.coord) < min_cross_product;

  if (tooparallel) {
    // replace W with vector from element 2 to element 1
    subvecvec3(triverts[2].coord,triverts[1].coord,W.coord);
    normalizevec3(W.coord);

    if (normvec3(N.coord) < min_cross_product) {
      printf("Normal calculation: Triangle 0x%x edges too parallel\n",(unsigned)trianglenum);
    }
    
  }
  // Normalize normal
  normalizevec3(N.coord);

  return N;
  
}



snde_coord3 snde_normalcalc_vertex(OCL_GLOBAL_ADDR const struct snde_part *part,
				   OCL_GLOBAL_ADDR const snde_triangle *part_triangles,
				   OCL_GLOBAL_ADDR const snde_edge *part_edges,
				   OCL_GLOBAL_ADDR const snde_coord3 *part_vertices,
				   OCL_GLOBAL_ADDR const snde_vertex_edgelist_index *part_vertex_edgelist_indices,
				   OCL_GLOBAL_ADDR const snde_index *part_vertex_edgelist,
				   OCL_GLOBAL_ADDR const snde_coord3 *part_trinormals,
				   
				   snde_index vertexnum)
{

  snde_index edgelist_index_index;

  snde_coord3 accum = { { 0.f,0.f,0.f }  };
  //snde_coord weight = 0.f;

  // If we wanted here we could weight the average according to the area of the
  // triangle close to the edge. That may be advantageous for some algorithms,
  // so don't rely on the current behavior of an unweighted average, as it may change.
  
  for (edgelist_index_index=0;edgelist_index_index < part_vertex_edgelist_indices[vertexnum].edgelist_numentries;edgelist_index_index++) {
    snde_index edge_index = part_vertex_edgelist[part_vertex_edgelist_indices[vertexnum].edgelist_index + edgelist_index_index];
    
    snde_index tri_a = part_edges[edge_index].tri_a;
    snde_index tri_b = part_edges[edge_index].tri_b;

    if (tri_a != SNDE_INDEX_INVALID) {
      accum.coord[0] += part_trinormals[tri_a].coord[0];
      accum.coord[1] += part_trinormals[tri_a].coord[1];
      accum.coord[2] += part_trinormals[tri_a].coord[2];
      //weight += 1; 
    }

    if (tri_b != SNDE_INDEX_INVALID) {
      accum.coord[0] += part_trinormals[tri_b].coord[0];
      accum.coord[1] += part_trinormals[tri_b].coord[1];
      accum.coord[2] += part_trinormals[tri_b].coord[2];
      //weight += 1; 
    }

  }

  normalizecoord3(&accum);
  
  return accum;
  
}



#ifdef __OPENCL_VERSION__
__kernel void snde_normalcalc_trinormals(__global const struct snde_part *part,
			      __global const snde_triangle *part_triangles,
			      __global const snde_edge *part_edges,
			      __global const snde_coord3 *part_vertices,
			      __global snde_coord3 *trinormals)
{
  snde_index trianglenum=get_global_id(0);
  snde_coord3 normal;
  
  normal = snde_normalcalc_triangle(part,
				    part_triangles,
				    part_edges,
				    part_vertices,
				    trianglenum);
  trinormals[trianglenum]=normal;
}


__kernel void snde_normalcalc_vertnormalarray(__global const struct snde_part *part,
					      __global const snde_triangle *part_triangles,
					      __global const snde_edge *part_edges,
					      __global const snde_coord3 *part_vertices,
					      __global snde_trivertnormals *vertnormals)
{
  snde_index trianglenum=get_global_id(0);
  snde_coord3 normal;
  
  normal = snde_normalcalc_triangle(part,
				    part_triangles,
				    part_edges,
				    part_vertices,
				    trianglenum);
  vertnormals[trianglenum].vertnorms[0]=normal;
  vertnormals[trianglenum].vertnorms[1]=normal;
  vertnormals[trianglenum].vertnorms[2]=normal;

}



__kernel void snde_normalcalc_vertnormals(__global const struct snde_part *part,
					  __global const snde_triangle *part_triangles,
					  __global const snde_edge *part_edges,
					  __global const snde_coord3 *part_vertices,
					  __global const snde_vertex_edgelist_index *part_vertex_edgelist_indices,
					  __global const snde_index *part_vertex_edgelist,
					  __global const snde_coord3 *part_trinormals,
					  __global snde_coord3 *vertnormals)
{
  snde_index vertnum=get_global_id(0);
  
  vertnormals[vertnum] = snde_normalcalc_vertex(part,
						part_triangles,
						part_edges,
						part_vertices,
						part_vertex_edgelist_indices,
						part_vertex_edgelist,
						part_trinormals,
						vertnum);

}

#endif // __OPENCL_VERSION__
