/* implicit include of geometry_types.h */
/* implicit include of vecops.h */

__kernel void normalcalc(__global const struct snde_part *part,
			 __global const snde_triangle *part_triangles,
			 __global const snde_edge *part_edges,
			 __global const snde_coord3 *part_vertices,
			 __global snde_trivertnormals *vertnormals,
			 __global snde_coord3 *trinormals)
{
  snde_index trianglenum=get_global_id(0);
  
  snde_coord3 triverts[3];
  
  // For the moment, this calculates a normal per triangle and stores it
  // for all vertices of the triangle, and for the triangle as a whole;
  

  /* traverse edges of this triangle and extract vertex coordinates -> triverts*/
  get_we_triverts(part_triangles,trianglenum,part_edges,part_vertices,triverts);

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
    
  const snde_coord min_cross_product = 1e-3;
  bool tooparallel=normvec3(N.coord) < min_cross_product;

  if (tooparallel) {
    // replace W with vector from element 2 to element 1
    subvecvec3(triverts[2].coord,triverts[1].coord,W.coord);
    normalizevec3(W.coord);

    if (normvec3(N.coord) < min_cross_product) {
      printf("Normal calculation: Triangle %u edges too parallel\n",(unsigned)trianglenum);
    }
    
  }
  // Normalize normal
  normalizevec3(N.coord);

  vertnormals[trianglenum].vertnorms[0]=N;
  vertnormals[trianglenum].vertnorms[1]=N;
  vertnormals[trianglenum].vertnorms[2]=N;
  trinormals[trianglenum]=N;
}

