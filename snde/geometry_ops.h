#ifndef SNDE_GEOMETRY_OPS_H
#define SNDE_GEOMETRY_OPS_H


#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
#define GEOMOPS_INLINE  __inline
#else
#define GEOMOPS_INLINE  inline
#endif

static GEOMOPS_INLINE void snde_part_initialize(OCL_GLOBAL_ADDR struct snde_part *part)
{
  part->firstboundary = SNDE_INDEX_INVALID;
  part->numboundaries = 0;

  part->first_topo = SNDE_INDEX_INVALID;
  part->num_topo = 0;

  part->first_topoidx = SNDE_INDEX_INVALID;
  part->num_topoidxs = 0;

  part->first_face = SNDE_INDEX_INVALID;
  part->num_faces = 0;

  part->firsttri = SNDE_INDEX_INVALID;
  part->numtris = 0;

  part->firstedge = SNDE_INDEX_INVALID;
  part->numedges = 0;

  part->firstvertex = SNDE_INDEX_INVALID;
  part->numvertices = 0;

  part->first_vertex_edgelist = SNDE_INDEX_INVALID;
  part->num_vertex_edgelist = 0;

  part->firstbox = SNDE_INDEX_INVALID;
  part->numboxes = 0;

  part->firstboxpoly = SNDE_INDEX_INVALID;
  part->numboxpolys = 0;

  part->firstboxnurbssurface = SNDE_INDEX_INVALID;
  part->numboxnurbssurfaces = 0;

  part->pivot_point.coord[0]=snde_infnan(0);
  part->pivot_point.coord[1]=snde_infnan(0);
  part->pivot_point.coord[2]=snde_infnan(0);

  part->length_scale = snde_infnan(0);

  part->bounding_box.min.coord[0]=snde_infnan(0);
  part->bounding_box.min.coord[1]=snde_infnan(0);
  part->bounding_box.min.coord[2]=snde_infnan(0);
  part->bounding_box.max.coord[0]=snde_infnan(0);
  part->bounding_box.max.coord[1]=snde_infnan(0);
  part->bounding_box.max.coord[2]=snde_infnan(0);

  part->first_vertex_kdnode = SNDE_INDEX_INVALID;
  part->num_vertex_kdnodes = 0;
  part->first_triarea = SNDE_INDEX_INVALID;
  part->first_vertarea = SNDE_INDEX_INVALID;
  snde_memset(&part->reserved[0],0xff,sizeof(part->reserved));

  part->solid = FALSE;
  part->has_triangledata = FALSE;
  part->has_curvatures = FALSE;
  part->pad1 = 0;
  snde_memset(&part->pad2[0],0x0,sizeof(part->pad2));
  
}

static GEOMOPS_INLINE void snde_parameterization_initialize(OCL_GLOBAL_ADDR struct snde_parameterization *uv)
{
  uv->first_uv_topo = SNDE_INDEX_INVALID;
  uv->num_uv_topos = 0;
  uv->first_uv_topoidx=SNDE_INDEX_INVALID;
  uv->num_uv_topoidxs=0;
  uv->firstuvtri=SNDE_INDEX_INVALID;
  uv->numuvtris=0;
  uv->firstuvface=SNDE_INDEX_INVALID;
  uv->numuvfaces=0;
  uv->firstuvedge=SNDE_INDEX_INVALID;
  uv->numuvedges=0;
  uv->firstuvvertex=SNDE_INDEX_INVALID;
  uv->numuvvertices=0;
  uv->first_uv_vertex_edgelist=SNDE_INDEX_INVALID;
  uv->num_uv_vertex_edgelist=0;
  uv->firstuvpatch=SNDE_INDEX_INVALID;
  uv->numuvpatches=0;

  snde_memset(&uv->reserved[0],0xff,sizeof(uv->reserved));
  
	//.firstuvbox=SNDE_INDEX_INVALID,
	//.numuvboxes=SNDE_INDEX_INVALID,
	//.firstuvboxpoly=SNDE_INDEX_INVALID,
	//.numuvboxpolys=SNDE_INDEX_INVALID,
	//.firstuvboxcoord=SNDE_INDEX_INVALID,
	//.numuvboxcoords=SNDE_INDEX_INVALID
      //};

}

static GEOMOPS_INLINE void tricentroid3(snde_coord3 *verts, snde_coord3 *centroid_out)
{
  size_t vertcnt,axcnt;

  for (axcnt=0; axcnt < 3; axcnt++) {
    centroid_out->coord[axcnt]=0.0f;
  }
  
  for (vertcnt=0; vertcnt < 3; vertcnt++) {
    for (axcnt=0; axcnt < 3; axcnt++) {
      centroid_out->coord[axcnt] += verts[vertcnt].coord[axcnt];
    }
  }
  for (axcnt=0; axcnt < 3; axcnt++) {
    centroid_out->coord[axcnt] /= 3.0f;
  }
  
}

static GEOMOPS_INLINE void polycentroid3(snde_coord3 *verts, uint32_t numvertices, snde_coord3 *centroid_out)
{
  size_t vertcnt,axcnt;

  for (axcnt=0; axcnt < 3; axcnt++) {
    centroid_out->coord[axcnt]=0.0f;
  }
  
  for (vertcnt=0; vertcnt < numvertices; vertcnt++) {
    for (axcnt=0; axcnt < 3; axcnt++) {
      centroid_out->coord[axcnt] += verts[vertcnt].coord[axcnt];
    }
  }
  for (axcnt=0; axcnt < 3; axcnt++) {
    centroid_out->coord[axcnt] /= numvertices;
  }
  
}


  static GEOMOPS_INLINE snde_coord polymaxradius_sq3(snde_coord3 *verts, uint32_t numvertices,snde_coord3 refpoint)
// evaluate max(radius squared) of radii of verts measured from refpoint
{
  size_t vertcnt,axcnt;

  snde_coord max_sq=0.0f;
  snde_coord val;
  
  for (vertcnt=0; vertcnt < numvertices; vertcnt++) {
    val=0.0f;
    for (axcnt=0; axcnt < 3; axcnt++) {
      val += (snde_coord)pow(verts[vertcnt].coord[axcnt] - refpoint.coord[axcnt],(snde_coord)2.0f);
    }

    if (val > max_sq) {
      max_sq=val;
    }
  }

  return max_sq;
}


  
static GEOMOPS_INLINE void get_we_triverts_3d(OCL_GLOBAL_ADDR const snde_triangle *part_triangles, snde_index trianglenum,OCL_GLOBAL_ADDR const snde_edge *part_edges, OCL_GLOBAL_ADDR const snde_coord3 *part_vertices,snde_coord3 *tri_vertices)
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


  static GEOMOPS_INLINE void get_we_trivert_3d(OCL_GLOBAL_ADDR const snde_triangle *part_triangles, snde_index trianglenum,OCL_GLOBAL_ADDR const snde_edge *part_edges, OCL_GLOBAL_ADDR const snde_coord3 *part_vertices,snde_coord3 *tri_vertex)
  // returns a vertex from the given triangle
{
  snde_index thisedge;

  //edgecnt=0;

  
  thisedge=part_triangles[trianglenum].edges[0];
  *tri_vertex=part_vertices[part_edges[thisedge].vertex[0]];

}


  
static GEOMOPS_INLINE void get_we_tricentroid_3d(OCL_GLOBAL_ADDR const snde_triangle *part_triangles, snde_index trianglenum,OCL_GLOBAL_ADDR const snde_edge *part_edges, OCL_GLOBAL_ADDR const snde_coord3 *part_vertices,snde_coord3 *tricentroid)
{
  snde_coord3 tri_vertices[3];
  
  get_we_triverts_3d(part_triangles,trianglenum,part_edges,part_vertices,tri_vertices);
  tricentroid3(tri_vertices,tricentroid);
  
}

  static GEOMOPS_INLINE void get_we_triverts_2d(OCL_GLOBAL_ADDR const snde_triangle *uv_triangles, snde_index trianglenum,OCL_GLOBAL_ADDR const snde_edge *uv_edges, OCL_GLOBAL_ADDR const snde_coord2 *uv_vertices,snde_coord2 *tri_vertices)
{
  snde_index edgecnt;
  snde_index thisedge;
  snde_index nextedge;
  snde_coord2 thisvert;

  /* traverse edges of this triangle and extract vertex coordinates -> tri_vertices*/
  edgecnt=0;

  
  thisedge=uv_triangles[trianglenum].edges[0];

  
  while (edgecnt < 3) {
    //printf("thisedge=%d\n",(int)thisedge);
    
    if (uv_edges[thisedge].tri_a==trianglenum) {
      nextedge = uv_edges[thisedge].tri_a_next_edge;  /* get our next edge from the Winged Edge */
    } else {
      nextedge = uv_edges[thisedge].tri_b_next_edge;
    }
    //printf("nextedge=%d\n",(int)nextedge);

    /* Extract the vertex of this edge that is NOT shared with the next edge */
    if (uv_edges[thisedge].vertex[0] != uv_edges[nextedge].vertex[0] &&
	uv_edges[thisedge].vertex[0] != uv_edges[nextedge].vertex[1]) {
      //printf("vertex_index=%d.0\n",(int)(vert_offs+edges[edge_offs+thisedge].vertex[0]));
      //tri_vertices[edgecnt]=uv_vertices[uv_edges[thisedge].vertex[0]];
      thisvert=uv_vertices[uv_edges[thisedge].vertex[0]];
    } else {
      //printf("vertex_index=%d.1\n",(int)(vert_offs+edges[edge_offs+thisedge].vertex[1]));
      //tri_vertices[edgecnt]=uv_vertices[uv_edges[thisedge].vertex[1]];
      thisvert=uv_vertices[uv_edges[thisedge].vertex[1]];
    }
    tri_vertices[edgecnt].coord[0]=thisvert.coord[0];
    tri_vertices[edgecnt].coord[1]=thisvert.coord[1];
    
    //printf("vertex: (%lf,%lf,%lf)\n",(double)triverts[edgecnt].coord[0],(double)triverts[edgecnt].coord[1],(double)triverts[edgecnt].coord[2]);

    thisedge=nextedge;
    edgecnt++;
  }


}


  

static GEOMOPS_INLINE int point_in_polygon_2d_c(snde_coord2 *vertices_rel_point,snde_index numvertices)
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
        
    if (cosparam > 1.0f) {
      // Shouldn't be possible...just in case of weird roundoff
      cosparam=1.0f;
    }
    
    if (det > 0) {
      windingnum += (snde_coord)acos(cosparam);
    } else if (det < 0) {
      windingnum -= (snde_coord)acos(cosparam);
    } else {
      // det==0.0f 
      
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
  
  
  windingnum=(snde_coord)fabs(windingnum)*(1.0f/(2.0f*(M_PI_SNDE_COORD))); // divide out radians to number of winds; don't care about clockwise vs. ccw
  if (windingnum > .999f && windingnum < 1.001f) {
    // Almost exactly one loop... got it! 
    return TRUE;
  } else if (windingnum >= .001f) {
#ifndef __OPENCL_VERSION__
    fprintf(stderr,"spatialnde.geometry.point_in_polygon_2d() Got weird winding number of %le; assuming inaccurate calculation on polygon edge\n",(double)windingnum);
#else
    printf("spatialnde.geometry.point_in_polygon_2d() Got weird winding number of %e; assuming inaccurate calculation on polygon edge\n",windingnum);
#endif
    
    // Could also be self intersecting polygon 
    // got it !!! 
    return TRUE;
  }  
  // If we got this far, the search failed 
  
  return FALSE;
}

static GEOMOPS_INLINE int point_in_polygon_3d_c(snde_coord3 *vertices, snde_coord2 *vertbuf2d_vert2d_rel_point,snde_index nvertices,snde_coord3 point, snde_cmat23 inplanemat)
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



static GEOMOPS_INLINE int vertices_in_box_2d(snde_coord2 *vertices,size_t numvertices,snde_coord2 box_v0,snde_coord2 box_v1)
/* v0 must have lower coordinates than v1 */
/* returns whether all vertices are inside or on the edge of the specified box */
{
  size_t vertexcnt;
  
  for (vertexcnt=0;vertexcnt < numvertices;vertexcnt++) {

    /* if this vertex is outside the box... */
    if (vertices[vertexcnt].coord[0] < box_v0.coord[0] ||
	vertices[vertexcnt].coord[0] > box_v1.coord[0] ||
	vertices[vertexcnt].coord[1] < box_v0.coord[1] ||
	vertices[vertexcnt].coord[1] > box_v1.coord[1]) {
      return FALSE;
    }
    
  }
  return TRUE;
}

static GEOMOPS_INLINE int vertices_in_box_3d(snde_coord3 *vertices,size_t numvertices,snde_coord3 box_v0,snde_coord3 box_v1)
/* v0 must have lower coordinates than v1 */
/* returns whether all vertices are inside or on the edge of the specified box */
{
  size_t vertexcnt;
  
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



static GEOMOPS_INLINE int segment_intersects_box_c(snde_coord3 box_v0,snde_coord3 box_v1,snde_coord3 seg_v0, snde_coord3 seg_v1)
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
    if (seg_v0.coord[cnt]*seg_axisdirections.coord[cnt]-box_width.coord[cnt]/2.0f > original_center.coord[cnt]*seg_axisdirections.coord[cnt]) {
      return FALSE;
    }
    
    // Surfaces at v1 end of the slide
    if (seg_v1.coord[cnt]*seg_axisdirections.coord[cnt] + box_width.coord[cnt]/2.0f < original_center.coord[cnt]*seg_axisdirections.coord[cnt]) {
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
    if (fabs(dotcoordcoord3(vert0_minus_center,surf_normal)) > 0.5f*dotcoordcoord3(directed_box_width,surf_normal)) {
      return FALSE;
    }
    
  }
  return TRUE;
  
}



// box-polygon intersection algorithm similar to 
// discussed in Graphics Gems ch 7-2
 
static GEOMOPS_INLINE int polygon_intersects_box_3d_c(snde_coord3 box_v0, snde_coord3 box_v1, snde_coord3 *vertices, snde_coord2 *vertbuf2d, size_t nvertices, snde_cmat23 inplanemat, snde_coord3 facetnormal)
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
  
  if (t > 1.0f || t < 0.0f) {
    return FALSE;
  }
  
  addcoordscaledcoord3(starting_corner,t,diagonalvec,&intersectioncoords);
 
  

  return point_in_polygon_3d_c(vertices,vertbuf2d,nvertices,intersectioncoords,inplanemat);
}



static GEOMOPS_INLINE int CCW(snde_coord2 a,snde_coord2 b,snde_coord2 c)
{
  // 2D in-plane: are a,b,c in CCW order?
  snde_coord2 b_minus_a,c_minus_a;
  subcoordcoord2(b,a,&b_minus_a);
  subcoordcoord2(c,a,&c_minus_a);
  
  return (crosscoordcoord2(b_minus_a,c_minus_a) > 0.0f);
}

static GEOMOPS_INLINE int check_lineseg_intersection(snde_coord2 a,snde_coord2 b,snde_coord2 c,snde_coord2 d)
{
  // Check if the planar (2D) line segments ab and cd intersect
  // Per http://jeffe.cs.illinois.edu/teaching/373/notes/x06-sweepline.pdf
  // The segments intersect if and only if
  //     endpoints a and b are on opposite sides of cd
  //  and endpoints c and d are on opposite sides of ab
  //
  // a and b are on opposite sides of c and d if and only if 
  // exactly one of (a,c,d) and (b,c,d) are CCW
  //
  //  This is true if the counter clock wise ordering test passes
  //  (CCW(a,c,d) != CCW(b,c,d)) and (CCW(a,b,c) != CCW(a,b,d))

  // ... What if (a,b) and (c,d) are colinear? 
  // ... Then the cross product is 0 and CCW returns False
  // and they count as not intersecting
  // ... That is probably reasonable in this application
  
  return (CCW(a,c,d) != CCW(b,c,d)) && (CCW(a,b,c) != CCW(a,b,d));
}


static GEOMOPS_INLINE int polygon_intersects_box_2d_c(snde_coord2 box_v0,snde_coord2 box_v1,snde_coord2 *vertices,size_t numvertices)
{
  snde_coord2 box_v0b;
  snde_coord2 box_v1b;
  size_t vertexcnt,nextvertex;

  box_v0b.coord[0]=box_v0.coord[0];
  box_v0b.coord[1]=box_v1.coord[1];

  box_v1b.coord[0]=box_v1.coord[0];
  box_v1b.coord[1]=box_v0.coord[1];

  for (vertexcnt=0;vertexcnt < numvertices;vertexcnt++) {
    nextvertex=(vertexcnt+1) % numvertices;

    
    // Compare with all of the edges of the box
    if (check_lineseg_intersection(box_v0,box_v1b,vertices[vertexcnt],vertices[nextvertex]) ||
	check_lineseg_intersection(box_v1b,box_v1,vertices[vertexcnt],vertices[nextvertex]) ||
	check_lineseg_intersection(box_v1,box_v0b,vertices[vertexcnt],vertices[nextvertex]) ||
	check_lineseg_intersection(box_v0b,box_v0,vertices[vertexcnt],vertices[nextvertex])) {
      // Found intersection of this polygon with this box
      return TRUE;
    }
  }   
  return FALSE;
  
}




#ifdef __cplusplus
}
#endif

#endif // SNDE_GEOMETRY_OPS_H

