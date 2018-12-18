/* implicit include of geometry_types.h */
/* implicit include of vecops.h */

__kernel void normalcalc(__global const struct snde_meshedpart *meshedpart,
			 __global const snde_triangle *part_triangles,
			 __global const snde_edge *part_edges,
			 __global const snde_coord3 *part_vertices,
			 __global snde_trinormals *normals)
{
  snde_index trianglenum=get_global_id(0);
  
  snde_index thisedge;
  snde_index nextedge;
  snde_coord3 triverts[3];
  snde_coord3 thisvert;
  int edgecnt;
  
  // For the moment, this calculates a normal per triangle and stores it
  // for all vertices of the triangle. 
  
  thisedge=part_triangles[trianglenum].edges[0];

  /* traverse edges of this triangle and extract vertex coordinates -> triverts*/
  edgecnt=0;
  while (edgecnt < 3) {
    //printf("thisedge=%d\n",(int)thisedge);
    
    if (part_edges[thisedge].face_a==trianglenum) {
      nextedge = part_edges[thisedge].face_a_next_edge;  /* get our next edge from the Winged Edge */
    } else {
      nextedge = part_edges[thisedge].face_b_next_edge;
    }
    //printf("nextedge=%d\n",(int)nextedge);

    /* Extract the vertex of this edge that is NOT shared with the next edge */
    if (part_edges[thisedge].vertex[0] != part_edges[nextedge].vertex[0] &&
	part_edges[thisedge].vertex[0] != part_edges[nextedge].vertex[1]) {
      //printf("vertex_index=%d.0\n",(int)(vert_offs+edges[edge_offs+thisedge].vertex[0]));
      //triverts[edgecnt]=part_vertices[part_edges[thisedge].vertex[0]];
      thisvert=part_vertices[part_edges[thisedge].vertex[0]];
    } else {
      //printf("vertex_index=%d.1\n",(int)(vert_offs+edges[edge_offs+thisedge].vertex[1]));
      //triverts[edgecnt]=part_vertices[part_edges[thisedge].vertex[1]];
      thisvert=part_vertices[part_edges[thisedge].vertex[1]];
    }
    triverts[edgecnt].coord[0]=thisvert.coord[0];
    triverts[edgecnt].coord[1]=thisvert.coord[1];
    triverts[edgecnt].coord[2]=thisvert.coord[2];
    
    //printf("vertex: (%lf,%lf,%lf)\n",(double)triverts[edgecnt].coord[0],(double)triverts[edgecnt].coord[1],(double)triverts[edgecnt].coord[2]);

    thisedge=nextedge;
    edgecnt++;
  }


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

  normals[trianglenum].vertnorms[0]=N;
  normals[trianglenum].vertnorms[1]=N;
  normals[trianglenum].vertnorms[2]=N;
}

