#ifndef __OPENCL_VERSION__
#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/vecops.h"
#include "snde/geometry_ops.h"

#include "snde/area_calc.h"
#endif

/* implicit include of geometry_types.h */
/* implicit include of vecops.h */

snde_coord snde_areacalc_trianglearea(OCL_GLOBAL_ADDR const snde_triangle *part_triangles,
				       OCL_GLOBAL_ADDR const snde_edge *part_edges,
				       OCL_GLOBAL_ADDR const snde_coord3 *part_vertices,
				       snde_index trianglenum)
{
  snde_coord3 triverts[3];
  
  // For the moment, this calculates a area per triangle and stores it
  

  /* traverse edges of this triangle and extract vertex coordinates -> triverts*/
  get_we_triverts_3d(part_triangles,trianglenum,part_edges,part_vertices,triverts);

  /* got vertex coordinates in triverts */
  /* Area = (1/2)|(cross product of two edge vectors)| */
  /* Area = (1/2)|(AB cross AC)| */

  snde_coord3 AB,AC,ABcrossAC;

  subcoordcoord3(triverts[1],triverts[0],&AB);
  subcoordcoord3(triverts[2],triverts[0],&AC);

  crosscoordcoord3(AB,AC,&ABcrossAC);

  snde_coord area = 0.5f*sqrt(ABcrossAC.coord[0]*ABcrossAC.coord[0] + ABcrossAC.coord[1]*ABcrossAC.coord[1] + ABcrossAC.coord[2]*ABcrossAC.coord[2]);
  

  return area;
  
}



snde_coord snde_areacalc_vertexarea(OCL_GLOBAL_ADDR const snde_triangle *part_triangles,
				     OCL_GLOBAL_ADDR const snde_edge *part_edges,
				     OCL_GLOBAL_ADDR const snde_coord3 *part_vertices,
				     OCL_GLOBAL_ADDR const snde_vertex_edgelist_index *part_vertex_edgelist_indices,
				     OCL_GLOBAL_ADDR const snde_index *part_vertex_edgelist,
				     OCL_GLOBAL_ADDR const snde_coord *part_trianglearea,
				     
				     snde_index vertexnum)
{

  snde_index edgelist_index_index;

  snde_coord accum = 0.f;
  //snde_coord weight = 0.f;

  // If we wanted here we could weight the average according to the area of the
  // triangle close to the edge. That may be advantageous for some algorithms,
  // so don't rely on the current behavior of an unweighted average, as it may change.
  
  for (edgelist_index_index=0;edgelist_index_index < part_vertex_edgelist_indices[vertexnum].edgelist_numentries;edgelist_index_index++) {
    snde_index edge_index = part_vertex_edgelist[part_vertex_edgelist_indices[vertexnum].edgelist_index + edgelist_index_index];
    
    snde_index tri_a = part_edges[edge_index].tri_a;
    snde_index tri_b = part_edges[edge_index].tri_b;

    // We will hit each triangle twice -- once for the edge on each
    // side. 1/3 the area of each adjacent triangle is assigned to
    // this vertex, so each time we look at a triangle we
    // accumulate 1/6th its area. 
    
    if (tri_a != SNDE_INDEX_INVALID) {
      accum += part_trianglearea[tri_a]/6.0f;
      //weight += 1.0f; 
    }

    if (tri_b != SNDE_INDEX_INVALID) {
      accum += part_trianglearea[tri_b]/6.0f;
      //weight += 1.0f; 
    }

  }

  return accum;
  
}



#ifdef __OPENCL_VERSION__
__kernel void snde_areacalc_triangleareas(__global const snde_triangle *part_triangles,
					  __global const snde_edge *part_edges,
					  __global const snde_coord3 *part_vertices,
					  __global snde_coord *trianglearea)
{
  snde_index trianglenum=get_global_id(0);
  snde_coord area;
  
  area = snde_areacalc_trianglearea(part_triangles,
				    part_edges,
				    part_vertices,
				    trianglenum);
  trianglearea[trianglenum]=area;
}




__kernel void snde_areacalc_vertexareas(__global const snde_triangle *part_triangles,
					__global const snde_edge *part_edges,
					__global const snde_coord3 *part_vertices,
					__global const snde_vertex_edgelist_index *part_vertex_edgelist_indices,
					__global const snde_index *part_vertex_edgelist,
					__global const snde_coord *part_trianglearea,
					__global snde_coord *vertareas)
{
  snde_index vertnum=get_global_id(0);
  
  vertareas[vertnum] = snde_areacalc_vertexarea(part_triangles,
						part_edges,
						part_vertices,
						part_vertex_edgelist_indices,
						part_vertex_edgelist,
						part_trianglearea,
						vertnum);
  
}

#endif // __OPENCL_VERSION__
