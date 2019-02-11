/* implicit include of geometry_types.h */

void evalarea_onetriangle(__global const struct snde_meshedpart *meshedpart,
			  __global const snde_triangle *part_triangles,
			  __global const snde_edge *part_edges,
			  __global const snde_coord3 *part_vertices,
			  __global snde_coord *area_output,
			  snde_index trianglenum)
{
  snde_index thisedge;
  snde_index nextedge;
  snde_coord3 triverts[3];
  int edgecnt;

  //printf("tri_offs=%d,trianglenum=%d,meshedpartnum=%d\n",(int)tri_offs,(int)trianglenum,(int)meshedpartnum);
  //printf("edge_offs=%d, vert_offs=%d\n",(int)edge_offs,(int)vert_offs);
  thisedge=part_triangles[trianglenum].edges[0];
  
  /* traverse edges of this triangle and extract vertex coordinates -> triverts*/
  edgecnt=0;
  while (edgecnt < 3) {
    //printf("thisedge=%d\n",(int)thisedge);
    
    if (part_edges[thisedge].tri_a==trianglenum) {
      nextedge = part_edges[thisedge].tri_a_next_edge;
    } else {
      nextedge = part_edges[thisedge].tri_b_next_edge;
    }
    //printf("nextedge=%d\n",(int)nextedge);

    /* Extract the vertex of this edge that is NOT shared with the next edge */
    if (part_edges[thisedge].vertex[0] != part_edges[nextedge].vertex[0] &&
	part_edges[thisedge].vertex[0] != part_edges[nextedge].vertex[1]) {
      //printf("vertex_index=%d.0\n",(int)(vert_offs+edges[edge_offs+thisedge].vertex[0]));
      triverts[edgecnt]=part_vertices[part_edges[thisedge].vertex[0]];
    } else {
      //printf("vertex_index=%d.1\n",(int)(vert_offs+edges[edge_offs+thisedge].vertex[1]));
      triverts[edgecnt]=part_vertices[part_edges[thisedge].vertex[1]];
    }
    //printf("vertex: (%lf,%lf,%lf)\n",(double)triverts[edgecnt].coord[0],(double)triverts[edgecnt].coord[1],(double)triverts[edgecnt].coord[2]);

    thisedge=nextedge;
    edgecnt++;
  }
  
  // Calculate area from triverts
  //area_output[trianglenum]=vertices[vert_offs+edges[edge_offs+thisedge].vertex[1]].coord[0];//triverts[0].coord[0]*1.0;
  area_output[trianglenum]=(snde_coord)fabs((snde_coord)((triverts[0].coord[0]*(triverts[1].coord[1]-triverts[2].coord[1]) + triverts[1].coord[0]*(triverts[2].coord[1]-triverts[0].coord[1]) + triverts[2].coord[0]*(triverts[0].coord[1]-triverts[1].coord[1]))/2.0));
  //printf("area_output=%lf\n",(double)area_output[trianglenum]);
  
}

__kernel void testkern_onepart(__global const struct snde_meshedpart *meshedpart,
		       __global const snde_triangle *part_triangles,
		       __global const snde_edge *part_edges,
		       __global const snde_coord3 *part_vertices,
		       __global snde_coord *area_output)
{
  snde_index trianglenum=get_global_id(0);
  evalarea_onetriangle(meshedpart,part_triangles,part_edges,part_vertices,area_output,trianglenum);
  

}


__kernel void testkern(__global const struct snde_meshedpart *meshedparts,
		       __global const snde_triangle *triangles,
		       __global const snde_edge *edges,
		       __global const snde_coord3 *vertices,
		       __global snde_coord *area_output) // area_output presumed to be offsetted like triangles
{
  /* global_id is presumed to be the meshedpartnum, which is kind or ridiculous.... 
     iteration over parts should probably be done on the host, in general. */

  snde_index meshedpartnum=get_global_id(0);
  /* calculate triangle area */

  snde_index tri_offs=meshedparts[meshedpartnum].firsttri;
  snde_index edge_offs=meshedparts[meshedpartnum].firstedge;
  snde_index vert_offs=meshedparts[meshedpartnum].firstvertex;


  testkern_onepart(meshedparts+meshedpartnum,
		   triangles+tri_offs,
		   edges+edge_offs,
		   vertices+vert_offs,
		   area_output+tri_offs);
  
}

