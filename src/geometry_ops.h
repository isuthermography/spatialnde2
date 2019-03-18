#ifndef SNDE_GEOMETRY_OPS_H
#define SNDE_GEOMETRY_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __OPENCL_VERSION__
#define OCL_GLOBAL_CONTEXT __global
#else
#define OCL_GLOBAL_CONTEXT
#endif
  
static inline void get_we_triverts(OCL_GLOBAL_CONTEXT const snde_triangle *part_triangles, snde_index trianglenum,OCL_GLOBAL_CONTEXT const snde_edge *part_edges, OCL_GLOBAL_CONTEXT const snde_coord3 *part_vertices,snde_coord3 *tri_vertices)
{
  snde_index edgecnt;
  snde_index thisedge;
  snde_index nextedge;
  snde_coord3 thisvert;

  /* traverse edges of this triangle and extract vertex coordinates -> tri_vertices*/
  edgecnt=0;

  
  thisedge=part_triangles[trianglenum].edges[0];

  
  while (edgecnt < 3) {
    //printf("thisedge=%d\n",(int)thisedge);
    
    if (part_edges[thisedge].tri_a==trianglenum) {
      nextedge = part_edges[thisedge].tri_a_next_edge;  /* get our next edge from the Winged Edge */
    } else {
      nextedge = part_edges[thisedge].tri_b_next_edge;
    }
    //printf("nextedge=%d\n",(int)nextedge);

    /* Extract the vertex of this edge that is NOT shared with the next edge */
    if (part_edges[thisedge].vertex[0] != part_edges[nextedge].vertex[0] &&
	part_edges[thisedge].vertex[0] != part_edges[nextedge].vertex[1]) {
      //printf("vertex_index=%d.0\n",(int)(vert_offs+edges[edge_offs+thisedge].vertex[0]));
      //tri_vertices[edgecnt]=part_vertices[part_edges[thisedge].vertex[0]];
      thisvert=part_vertices[part_edges[thisedge].vertex[0]];
    } else {
      //printf("vertex_index=%d.1\n",(int)(vert_offs+edges[edge_offs+thisedge].vertex[1]));
      //tri_vertices[edgecnt]=part_vertices[part_edges[thisedge].vertex[1]];
      thisvert=part_vertices[part_edges[thisedge].vertex[1]];
    }
    tri_vertices[edgecnt].coord[0]=thisvert.coord[0];
    tri_vertices[edgecnt].coord[1]=thisvert.coord[1];
    tri_vertices[edgecnt].coord[2]=thisvert.coord[2];
    
    //printf("vertex: (%lf,%lf,%lf)\n",(double)triverts[edgecnt].coord[0],(double)triverts[edgecnt].coord[1],(double)triverts[edgecnt].coord[2]);

    thisedge=nextedge;
    edgecnt++;
  }


}



static inline int point_in_polygon_2d_c(snde_coord2 *vertices_rel_point,snde_index numvertices)
{ 
      
  //# Apply winding number algorithm.
  //# This algorithm is selected -- in its most simple form --
  //# because it is so  simple and robust in the case of the
  //# intersect point being on or near the edge. It may well
  //# be much slower than optimal. It tries to return True
  //# in the edge case. 
    
  //# Should probably implement a faster algorithm then drop
  //# down to this for the special cases.

  //# See Hormann and Agathos, The point in polygon problem
  //# for arbitrary polygons, Computational Geometry 20(3) 131-144 (2001)
  //# http://dx.doi.org/10.1016/S0925-7721(01)00012-8
  //# https://pdfs.semanticscholar.org/e90b/d8865ddb7c7af2b159d413115050d8e5d297.pdf
    
  //# Winding number is sum over segments of
  //# acos((point_to_vertex1 dot point_to_vertex2)/(magn(point_to_vertex1)*magn(point_to_vertex_2))) * sign(det([ point_to_vertex1  point_to_vertex2 ]))
  //# where sign(det) is really: What is the sign of the z
  //# component of (point_to_vertex1 cross point_to_vertex2)
        
  //# Special cases: magn(point_to_vertex1)==0 or
  //#  magn_point_to_vertex2   -> point is on edge
  //# det([ point_to_vertex1  point_to_vertex2 ]) = 0 -> point may be on edge
    
  snde_coord windingnum=0.0f;
  snde_index VertexCnt;
  snde_index NextVertex;
  snde_coord magn1,magn2;
  snde_coord2 vec1,vec2;
  snde_coord det;
  snde_coord cosparam;
  
  for (VertexCnt=0;VertexCnt < numvertices;VertexCnt++) {
    NextVertex=(VertexCnt+1) % numvertices;
    
    // calculate (thisvertex - ourpoint) -> vec1
    //    vec1=vertices_rel_point[VertexCnt,:]
    magn1=normcoord2(vertices_rel_point[VertexCnt]);
    
    
        
    // calculate (nextvertex - ourpoint) -> vec2
    //    vec2=vertices_rel_point[NextVertex,:]
    magn2=normcoord2(vertices_rel_point[NextVertex]);
    
    if (magn1==0.0f || magn2==0.0f){
      // Got it!!!
      return TRUE;
    }
    scalecoord2(1.0f/magn1,vertices_rel_point[VertexCnt],&vec1);
    scalecoord2(1.0f/magn2,vertices_rel_point[NextVertex],&vec2);

    det=vec1.coord[0]*vec2.coord[1]-vec2.coord[0]*vec1.coord[1]; // matrix determinant
    
    cosparam=(vec1.coord[0]*vec2.coord[0]+vec1.coord[1]*vec2.coord[1]); //  /(magn1*magn2);
    
    if (cosparam < -1.0f) {
      // Shouldn't be possible...just in case of weird roundoff
      cosparam=-1.0f;
    }
        
    if (cosparam > 1.0) {
      // Shouldn't be possible...just in case of weird roundoff
      cosparam=1.0f;
    }
    
    if (det > 0) {
      windingnum += acos(cosparam);
    } else if (det < 0) {
      windingnum -= acos(cosparam);
    } else {
      // det==0.0 
      
      // Vectors parallel or anti-parallel 
      
      if (cosparam > 0.9f) {
	// Vectors parallel. We are OUTSIDE. Do Nothing
      }
      else if (cosparam < -0.9f) {
	// Vectors anti-parallel. We are ON EDGE */
	return TRUE;
      }
      else {
	#ifndef __OPENCL_VERSION__
	assert(0); //# Should only be able to get cosparam = +/- 1.0 if abs(det) > 0.0 */
	#else
	printf("point_in_polygon_2d_c(): Error: Invalid value for cosparam\n");
	#endif
      }
    }
  }
  
  
  windingnum=fabs(windingnum)*(1.0f/(2.0f*((float)M_PI))); // divide out radians to number of winds; don't care about clockwise vs. ccw
  if (windingnum > .999f && windingnum < 1.001f) {
    // Almost exactly one loop... got it! 
    return TRUE;
  } else if (windingnum >= .001f) {
#ifndef __OPENCL_VERSION__
    fprintf(stderr,"spatialnde.geometry.point_in_polygon_2d() Got weird winding number of %le; assuming inaccurate calculation on polygon edge\n",(double)windingnum);
#else
    printf("spatialnde.geometry.point_in_polygon_2d() Got weird winding number of %le; assuming inaccurate calculation on polygon edge\n",(double)windingnum);
#endif
    
    // Could also be self intersecting polygon 
    // got it !!! 
    return TRUE;
  }  
  // If we got this far, the search failed 
  
  return FALSE;
}

static inline int point_in_polygon_3d_c(snde_coord3 *vertices, snde_coord2 *vertbuf2d_vert2d_rel_point,snde_coord nvertices,snde_coord3 point, snde_mat23 inplanemat)
{
  snde_coord3 vert3d_rel_point;
  //snde_coord2 vert2d_rel_point;
  snde_index vertexidx;
  
  //vert3d_rel_point=GAOPS_ALLOCA(nvertices*3*sizeof(*vert3d_rel_point));
  //vert2d_rel_point=GAOPS_ALLOCA(nvertices*2*sizeof(*vert2d_rel_point));
  
  for (vertexidx=0;vertexidx < nvertices;vertexidx++) {

    subcoordcoord3(vertices[vertexidx],point,&vert3d_rel_point);
    vertbuf2d_vert2d_rel_point[vertexidx].coord[0]=dotcoordcoord3(vert3d_rel_point,inplanemat.row[0]);
    vertbuf2d_vert2d_rel_point[vertexidx].coord[1]=dotcoordcoord3(vert3d_rel_point,inplanemat.row[1]);
    
  }
  return point_in_polygon_2d_c(vertbuf2d_vert2d_rel_point,nvertices);
}




static inline int vertices_in_box_3d(snde_coord3 *vertices,size_t numvertices,snde_coord3 box_v0,snde_coord3 box_v1)
/* v0 must have lower coordinates than v1 */
/* returns whether all vertices are inside or on the edge of the specified box */
{
  size_t vertexcnt;
  uint32_t vertexidx;
  
  for (vertexcnt=0;vertexcnt < numvertices;vertexcnt++) {

    /* if this vertex is outside the box... */
    if (vertices[vertexcnt].coord[0] < box_v0.coord[0] ||
	vertices[vertexcnt].coord[0] > box_v1.coord[0] ||
	vertices[vertexcnt].coord[1] < box_v0.coord[1] ||
	vertices[vertexcnt].coord[1] > box_v1.coord[1] ||
	vertices[vertexcnt].coord[2] < box_v0.coord[2] ||
	vertices[vertexcnt].coord[2] > box_v1.coord[2]) {
      return FALSE;
    }
    
  }
  return TRUE;
}



static inline int segment_intersects_box_c(snde_coord3 box_v0,snde_coord3 box_v1,snde_coord3 seg_v0, snde_coord3 seg_v1)
{
  snde_coord3 original_center;
  snde_coord3 segvec;
  snde_coord3 box_width;
  snde_coord3 seg_axisdirections;
  int cnt;
  int axis;
  snde_coord3 axisvec;
  snde_coord3 surf_normal;
  snde_coord3 sn_sign;
  snde_coord3 vert0_minus_center;
  snde_coord3 directed_box_width;

  mean2coord3(box_v0,box_v1,&original_center);

  subcoordcoord3(seg_v1,seg_v0,&segvec);
  subcoordcoord3(box_v1,box_v0,&box_width);

  sign_nonzerocoord3(segvec,&seg_axisdirections);


  for (cnt=0;cnt < 3;cnt++) {
    //Surfaces at v0 end of the slide
    if (seg_v0.coord[cnt]*seg_axisdirections.coord[cnt]-box_width.coord[cnt]/2.0 > original_center.coord[cnt]*seg_axisdirections.coord[cnt]) {
      return FALSE;
    }
    
    // Surfaces at v1 end of the slide
    if (seg_v1.coord[cnt]*seg_axisdirections.coord[cnt] + box_width.coord[cnt]/2.0 < original_center.coord[cnt]*seg_axisdirections.coord[cnt]) {
      return FALSE;
    }

  }
  
  // Remaining six faces connect the two ends
  
  for (axis=0;axis < 3; axis++) {
    
    //surf_normal should be normal to axis
    // and normal to segvec
    for (cnt=0;cnt < 3;cnt++) {
      if (cnt==axis) axisvec.coord[cnt]=1;
      else axisvec.coord[cnt]=0;
    }
    
    crosscoordcoord3(axisvec,segvec,&surf_normal);
    sign_nonzerocoord3(surf_normal,&sn_sign);

    subcoordcoord3(seg_v0,original_center,&vert0_minus_center);
    multcoordcoord3(box_width,sn_sign,&directed_box_width);
    if (fabs(dotcoordcoord3(vert0_minus_center,surf_normal)) > 0.5*dotcoordcoord3(directed_box_width,surf_normal)) {
      return FALSE;
    }
    
  }
  return TRUE;
  
}



// box-polygon intersection algorithm similar to 
// discussed in Graphics Gems ch 7-2
 
static inline int polygon_intersects_box_3d_c(snde_coord3 box_v0, snde_coord3 box_v1, snde_coord3 *vertices, snde_coord2 *vertbuf2d, size_t nvertices, snde_mat23 inplanemat, snde_coord3 facetnormal)
// vertbuf2d should be big enough for nvertices 2d coordinates
{
  size_t startvertex,endvertex;
  snde_coord3 diagonalvec;
  snde_coord3 firstdiagonal;
  snde_coord3 normalsigns;
  snde_coord3 firstvertex_rel_corner;
  snde_coord t;
  snde_coord3 intersectioncoords;
  snde_coord3 starting_corner;
  snde_index cnt;
  
  
  for (startvertex=0;startvertex < nvertices;startvertex++) {
    endvertex = (startvertex+1) % nvertices;
    
    if (segment_intersects_box_c(box_v0,box_v1,vertices[startvertex],vertices[endvertex])) {
      return TRUE;
    }
    
  }
  
  subcoordcoord3(box_v1,box_v0,&firstdiagonal);
  
  sign_nonzerocoord3(facetnormal,&normalsigns);
  multcoordcoord3(normalsigns,firstdiagonal,&diagonalvec);

  for (cnt=0;cnt < 3;cnt++) {
    if (normalsigns.coord[cnt] >= 0) {
      starting_corner.coord[cnt]=box_v0.coord[cnt];
    } else {
      starting_corner.coord[cnt]=box_v1.coord[cnt];
      
    }
  }
  

  subcoordcoord3(vertices[0],starting_corner,&firstvertex_rel_corner);

  t=dotcoordcoord3(firstvertex_rel_corner,facetnormal)/dotcoordcoord3(diagonalvec,facetnormal);
  
  if (t > 1.0 || t < 0.0) {
    return FALSE;
  }
  
  addcoordscaledcoord3(starting_corner,t,diagonalvec,&intersectioncoords);
 
  

  return point_in_polygon_3d_c(vertices,vertbuf2d,nvertices,intersectioncoords,inplanemat);
}

#ifdef __cplusplus
}
#endif

#endif // SNDE_GEOMETRY_OPS_H

