#ifndef SNDE_RAYTRACE_H
#define SNDE_RAYTRACE_H

#ifdef _MSC_VER
#define RAYTRACE_INLINE  __inline
#else
#define RAYTRACE_INLINE  inline
#endif



static RAYTRACE_INLINE void camera_rayvec(snde_cmat23 cam_mtx, snde_coord a,snde_coord b,snde_coord3 *rayveccam)
/// cam_mtx is the first two
/// rows of the camera matrix
/// evaluate the camera ray vector for pixel (a,b) relative to the origin at the camera focus
/// with the camera pointing in the -z direction
/// stores ray vector in camera coordinates in rayveccam
/// NOTE: Camera matrix defined based on OpenGL coordinate system, not OpencL

{
  snde_coord fx,fy,cx,cy,x_over_w,y_over_w,rayvecfactor;

  fx = cam_mtx.row[0].coord[0];
  fy = cam_mtx.row[1].coord[1];
  cx = cam_mtx.row[0].coord[2];
  cy = cam_mtx.row[1].coord[2];
  x_over_w = (a-cx)/fx;
  y_over_w = (b-cy)/fy;

  rayvecfactor = 1.0/sqrt(x_over_w*x_over_w + y_over_w*y_over_w + 1);
  rayveccam->coord[0]=x_over_w*rayvecfactor;
  rayveccam->coord[1]=y_over_w*rayvecfactor;
  rayveccam->coord[2]=rayvecfactor;
}


static RAYTRACE_INLINE void camera_rayvec_wrl(snde_cmat23 cam_mtx, snde_coord a,snde_coord b,snde_orientation3 orient_wrlcoords_over_camcoords,snde_coord4 *rayvecwrlproj)
// evaluate the camera ray vector for pixel (a,b) relative to the world coordinate frame
// with the camera pointing in the -z direction
// outputs rayvec in projective object coordinates (4th entry zero)
  
// cam_mtx is the first two
// rows of the camera matrix
// NOTE: Camera matrix defined based on OpenGL coordinate system, not OpencL
{
  snde_coord4 rayvec_cam;
  camera_rayvec(cam_mtx,a,b,(snde_coord3*)&rayvec_cam);
  rayvec_cam.coord[3]=0.0;  // a vector in projective coordinates has a final coordinate of 0

  orientation_apply_vector(orient_wrlcoords_over_camcoords,rayvec_cam,rayvecwrlproj);
  
}



static RAYTRACE_INLINE void camera_rayvec_deriv_a(snde_cmat23 cam_mtx, snde_coord a,snde_coord b,snde_coord3 *rayvecderivcam)
// cam_mtx is the first two
// rows of the camera matrix
// evaluate derivative of camera ray vector for pixel (a,b) with respect to an increment of a,
// relative to the origin at the camera focus
// with the camera pointing in the -z direction
// NOTE: Camera matrix defined based on OpenGL coordinate system, not OpencL
{
  snde_coord fx,fy,cx,cy,x_over_w,y_over_w,rayvecfactor,rayvecfactor2;

  fx = cam_mtx.row[0].coord[0];
  fy = cam_mtx.row[1].coord[1];
  cx = cam_mtx.row[0].coord[2];
  cy = cam_mtx.row[1].coord[2];
  x_over_w = (a-cx)/fx;
  y_over_w = (b-cy)/fy;

  rayvecfactor2= 1.0/(x_over_w*x_over_w + y_over_w*y_over_w + 1); // rayvecfactor^2
  rayvecfactor=sqrt(rayvecfactor2);
  rayvecderivcam->coord[0]=(rayvecfactor/fx)*(1.0 - x_over_w*rayvecfactor2*(a-cx));
  rayvecderivcam->coord[1]=-(rayvecfactor/fx)*y_over_w*rayvecfactor2*(a-cx);
  rayvecderivcam->coord[2]=-(rayvecfactor/fx)*rayvecfactor2*(a-cx);
  
}


static RAYTRACE_INLINE void camera_rayvec_deriv_b(snde_cmat23 cam_mtx, snde_coord a,snde_coord b,snde_coord3 *rayvecderivcam)
// cam_mtx is the first two
// rows of the camera matrix
// evaluate derivative of camera ray vector for pixel (a,b) with respect to an increment of a,
// relative to the origin at the camera focus
// with the camera pointing in the -z direction
// NOTE: Camera matrix defined based on OpenGL coordinate system, not OpencL
{
  snde_coord fx,fy,cx,cy,x_over_w,y_over_w,rayvecfactor,rayvecfactor2;

  fx = cam_mtx.row[0].coord[0];
  fy = cam_mtx.row[1].coord[1];
  cx = cam_mtx.row[0].coord[2];
  cy = cam_mtx.row[1].coord[2];
  x_over_w = (a-cx)/fx;
  y_over_w = (b-cy)/fy;

  rayvecfactor2= 1.0/(x_over_w*x_over_w + y_over_w*y_over_w + 1); // rayvecfactor^2
  rayvecfactor=sqrt(rayvecfactor2);
  rayvecderivcam->coord[0]=-(rayvecfactor/fy)*(x_over_w*rayvecfactor2*(b-cy));
  rayvecderivcam->coord[1]=(rayvecfactor/fy)*(1.0 - y_over_w*rayvecfactor2*(b-cy));
  rayvecderivcam->coord[2]=-(rayvecfactor/fy)*rayvecfactor2*(b-cy);
  
}

static RAYTRACE_INLINE void camera_rayvec_deriv_a_wrl(snde_cmat23 cam_mtx, snde_coord a,snde_coord b,snde_orientation3 orient_wrlcoords_over_camcoords,snde_coord4 *rayvecderivwrlproj)
// evaluate the camera ray vector derivative w.r.t motion in the a direction,
// for pixel (a,b) relative to world coordinates
// outputs rayvec in projective object coordinates (4th entry zero)
  
// cam_mtx is the first two
// rows of the camera matrix
// NOTE: Camera matrix defined based on OpenGL coordinate system, not OpencL
{
  snde_coord4 rayvecderiv_cam;
  camera_rayvec_deriv_a(cam_mtx,a,b,(snde_coord3*)&rayvecderiv_cam);
  rayvecderiv_cam.coord[3]=0.0;  // a vector in projective coordinates has a final coordinate of 0

  orientation_apply_vector(orient_wrlcoords_over_camcoords,rayvecderiv_cam,rayvecderivwrlproj);
  
}

static RAYTRACE_INLINE void camera_rayvec_deriv_b_wrl(snde_cmat23 cam_mtx, snde_coord a,snde_coord b,snde_orientation3 orient_wrlcoords_over_camcoords,snde_coord4 *rayvecderivwrlproj)
// evaluate the camera ray vector derivative w.r.t motion in the b direction,
// for pixel (a,b) relative to world coordinates
// outputs rayvec in projective object coordinates (4th entry zero)
  
// cam_mtx is the first two
// rows of the camera matrix
// NOTE: Camera matrix defined based on OpenGL coordinate system, not OpencL
{
  snde_coord4 rayvecderiv_cam;
  camera_rayvec_deriv_b(cam_mtx,a,b,(snde_coord3*)&rayvecderiv_cam);
  rayvecderiv_cam.coord[3]=0.0;  // a vector in projective coordinates has a final coordinate of 0

  orientation_apply_vector(orient_wrlcoords_over_camcoords,rayvecderiv_cam,rayvecderivwrlproj);
  
}

static RAYTRACE_INLINE int ray_box_intersection(snde_boxcoord3 boxcoord, snde_coord4 starting_point, snde_coord4 ray_direc)
{
  /* Slab method: Look at distance t along the ray where we first 
     intersect each slab. That's tnear. Look at the distance t 
     along the ray where we last intersect each slab. That's tfar. 
     Look at the largest tnear value. If that's greater than the 
     smallest tfar vluae, the ray misses the box. 
     Also special cases if the ray is parallel to an axis */
  /* See http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm */
  snde_coord tnear,tfar;
  
  tnear = snde_infnan(-ERANGE);
  tfar = snde_infnan(ERANGE);

  snde_coord curnear,curfar,temp;
    
  int index; 
  
  for (index=0;index < 3;index++) {
    /* index indexes starting_point and ray_direc, 
       and finds the min value from box_coords */
    /* Start with X=xmin and X=xmax planes */
    if (ray_direc.coord[index] == 0.0) {
      /* Ray is normal to given axis, parallel to these planes */
      
      /* if origin not between planes, does not intersect */
      if (starting_point.coord[index] < boxcoord.min.coord[index] || starting_point.coord[index] > boxcoord.max.coord[index]) {
	return FALSE;
      }
      
    } else {
      curnear=(boxcoord.min.coord[index]-starting_point.coord[index])/ray_direc.coord[index]; /* distance to reach x value of boxcoord[0] */
      curfar=(boxcoord.max.coord[index]-starting_point.coord[index])/ray_direc.coord[index];
      
      /* did we get curnear and curfar in the correct order? */
      if (curfar < curnear) {
	/* swap */
	temp=curfar;
	curfar=curnear;
	curnear=temp;
      }
      if (curnear > tnear) {
	tnear=curnear; /* largest tnear value */
      }
      if (curfar < tfar) {
	tfar=curfar; /* smallest tfar value */
      }
      if (tnear > tfar) {
	/* missed box */
	return FALSE;
      }
      if (tfar < 0.0) {
	/* box is behind */
	return FALSE;
      }
    }
  }
  return TRUE;
}


static RAYTRACE_INLINE snde_coord ray_to_plane_distance(snde_coord4 starting_point,snde_coord4 ray_direc, snde_coord3 planepoint, snde_coord3 planenormal)
{
  // starting point should be a normalized 4 vector; and planepoint should be a 3-vector  (or normalized 4-vectors)
  // (planepoint - raypoint) dot planenormal / (raydirec dot planenormal)
  // returns -1.0 if ray is parallel to plane
  
  snde_coord3 pointdiff;
  snde_coord denominator;
  
  subvecvec3(&planepoint.coord[0],&starting_point.coord[0],&pointdiff.coord[0]);
  denominator=dotvecvec3(&ray_direc.coord[0],&planenormal.coord[0]);
  if (denominator==0.0) return -1.0;
  return dotcoordcoord3(pointdiff,planenormal)/denominator;
  
}


static RAYTRACE_INLINE int ray_intersects_polygon(snde_coord3 *vertices,
					       uint32_t numvertices,
					       snde_coord3 normal_vector, // unit vector
					       snde_cmat23 inplanemat, // 2x3 matrix: two orthogonal unit vectors normal to normal_vector
					       snde_coord4 starting_point,
					       snde_coord4 ray_direc,
					       snde_coord ray_to_plane_dist,
					       int include_edges,int trace)
{
  snde_coord3 intersectpoint;
  snde_coord3 refpoint;
  
  snde_coord u1,v1,u2,v2;
  snde_coord windingnum=0.0;
  snde_index vertnum,nextvertnum;
  snde_coord3 vec1;
  snde_coord3 vec2;
  snde_coord3 intersectdist;
  snde_coord maxradius2; // maxradius^2
   
  snde_coord magn1,magn2,det,cosparam;
  
  addvecscaledvec3(&starting_point.coord[0],ray_to_plane_dist,&ray_direc.coord[0],&intersectpoint.coord[0]);

  polycentroid3(vertices,numvertices,&refpoint);
  maxradius2=polymaxradius_sq3(vertices,numvertices,refpoint);
  
  subcoordcoord3(intersectpoint,refpoint,&intersectdist);
  
  /* checking whether intersection point within maximum outer
     radius of polygon */

  if (trace) {
    printf("intersectpoint=%g,%g,%g; refpoint=%g,%g,%g\n",intersectpoint.coord[0],intersectpoint.coord[1],intersectpoint.coord[2],refpoint.coord[0],refpoint.coord[1],refpoint.coord[2]);
  
    printf("intersectdist=%g; maxradius^2=%g\n",normcoord3(intersectdist),maxradius2);

    {
      snde_coord intersect_outofplane=dotcoordcoord3(intersectdist,normal_vector);
      fprintf(stderr,"out of plane component = %g\n",intersect_outofplane);
    //if (fabs(intersect_outofplane) > (0.3*normvec3(intersectdist))) {
    //  *((char *)0) = 0;  // force segfault
    // }
    }

  }
  
  if (normsqcoord3(intersectdist) <= maxradius2) {

    if (trace) fprintf(stderr,"within outer radius\n");

  
    /* Within outer radius */
	  
	  
    // for the ray, determine if intersectpoint is inside the polygon

    // Apply winding number algorithm.
    // This algorithm is selected -- in its most simple form --
    // because it is so  simple and robust in the case of the
    // intersect point being on or near the edge. It may well
    // be much slower than optimal.
    //
    // Should probably implement a faster algorithm then drop
    // down to this for the special cases.

    // See Hormann and Agathos, The point in polygon problem
    // for arbitrary polygons, Computational Geometry 20(3) 131-144 (2001)
    // http://dx.doi.org/10.1016/S0925-7721(01)00012-8
    // https://pdfs.semanticscholar.org/e90b/d8865ddb7c7af2b159d413115050d8e5d297.pdf
    
    // Winding number is sum over segments of
    // acos((point_to_vertex1 dot point_to_vertex2)/(magn(point_to_vertex1)*magn(point_to_vertex_2))) * sign(det([ point_to_vertex1  point_to_vertex2 ]))
    // where sign(det) is really: What is the sign of the z
    // component of (point_to_vertex1 cross point_to_vertex2)
    //
    // Special cases: magn(point_to_vertex1)==0 or
    // magn_point_to_vertex2   -> point is on edge
    // det([ point_to_vertex1  point_to_vertex2 ]) = 0 -> point may be on edge
    //

    for (vertnum=0;vertnum < numvertices;vertnum++) {
      nextvertnum=vertnum+1;
      if (nextvertnum >= numvertices) nextvertnum=0;
      
      // calculate (thisvertex - intersectionpoint) -> vec1
      subcoordcoord3(vertices[vertnum],intersectpoint,&vec1);
      
      // calculate (nextvertex - intersectionpoint) -> vec2
      subcoordcoord3(vertices[nextvertnum],intersectpoint,&vec2);

      // Project points into 2-space:
      u1=dotcoordcoord3(vec1,inplanemat.row[0]);
      v1=dotcoordcoord3(vec1,inplanemat.row[1]);

      u2=dotcoordcoord3(vec2,inplanemat.row[0]);
      v2=dotcoordcoord3(vec2,inplanemat.row[1]);

      magn1 = sqrt(u1*u1 + v1*v1);
      magn2 = sqrt(u2*u2 + v2*v2);

      if (magn1==0.0f || magn2==0.0f) {
	return include_edges; 
      }

      /* Normalize vectors */
      u1 /= magn1;
      v1 /= magn1;

      u2 /= magn2;
      v2 /= magn2;
      
      det=(u1*v2-u2*v1); // matrix determinant


      cosparam=(u1*u2 + v1*v2); //  /(magn1*magn2);

      if (cosparam < -1.0f) {
	cosparam=-1.0f; // Shouldn't be possible...just in case of weird roundoff
      }
      if (cosparam > 1.0f) {
	cosparam=1.0f; // Shouldn't be possible...just in case of weird roundoff
      }

      if (det > 0) {	
	windingnum += acos(cosparam);
      } else if (det < 0) {
	windingnum -= acos(cosparam);
      } else {
	/* det==0  */

	/* Vectors parallel or anti-parallel */
	
	if (cosparam > 0.9f) {
	  // Vectors parallel. We are OUTSIDE. Do Nothing */
	}
	else if (cosparam < -0.9f) {
	  // Vectors anti-parallel. We are ON EDGE */	  
	  return include_edges;
	} else {
	  /* Should only be able to get cosparam = +/- 1.0 if det==0.0 */
#ifndef __OPENCL_VERSION__
	  // if this is not an OpenCL kernel
	  assert(0); 
#else
	  printf("ray_intersects_polygon: vector inconsistency in winding number algorithm\n");
#endif
	}
	
      }
      
      
      
    }
    
    windingnum=fabs(windingnum)*(1.0f/(2.0f*M_PI_SNDE_COORD)); // divide out radians to number of winds; don't care about clockwise vs. ccw
    
    if (windingnum > 0.999f && windingnum < 1.001f) {
      //fprintf(stderr,"Winding number ~1.0\n");
      return TRUE;  /* almost exactly one loop */
    }
    if (windingnum < 0.001f) {
      //fprintf(stderr,"Winding number ~0.0\n");
      return FALSE;
      
    }
    printf("raytrace.h/ray_intersects_polygon Got weird winding number of %e; assuming inaccurate calculation on polygon edge\n",windingnum);
    // Could also be self intersecting polygon 
    return include_edges; 
    
  }
  return FALSE;
}


//#define FRIN_STACKSIZE 100 // !!!*** NOTE: Not much point in this being deeper than the maximum number of levels of boxes (which is currently 22)

static RAYTRACE_INLINE int find_ray_first_part_intersection_nonrecursive(snde_index instnum,
									 //snde_index numtris,
									 OCL_GLOBAL_ADDR snde_topological *part_topos,
									 //snde_index *topo_indices,
									 OCL_GLOBAL_ADDR snde_triangle *part_triangles,
									 OCL_GLOBAL_ADDR snde_coord3 *part_trinormals,
									 OCL_GLOBAL_ADDR snde_cmat23 *part_inplanemats,
									 OCL_GLOBAL_ADDR snde_edge *part_edges,
									 OCL_GLOBAL_ADDR snde_coord3 *part_vertices,
									 OCL_GLOBAL_ADDR snde_box3 *part_boxes,
									 OCL_GLOBAL_ADDR snde_boxcoord3 *part_boxcoord,
									 OCL_GLOBAL_ADDR snde_index *part_boxpolys,
									 snde_coord4 raystartproj,
									 snde_coord4 rayvecproj,
									 snde_index frin_stacksize,
									 OCL_LOCAL_ADDR snde_index *boxnum_stack, // size frin_stacksize
									 OCL_GLOBAL_ADDR snde_coord *zdist_out,
									 OCL_GLOBAL_ADDR snde_index *instnum_out, // which instance we found, or SNDE_INDEX_INVALID for no intersection
									 OCL_GLOBAL_ADDR snde_index *boundarynum_out, // which boundary number for the instance's part
									 OCL_GLOBAL_ADDR snde_index *facenum_out, // which topological face? 
									 OCL_GLOBAL_ADDR snde_index *trinum_out, // which triangle number for the instances's part
									 int trace)
// DOES NOT INITIALIZE zdist_out... pre-initialize this to Infinity!
/* returns nonzero if at least one nearer intersection was found */
{
  
  snde_index cnt;
  //snde_index boxnum_stack[FRIN_STACKSIZE]; // ***!!! Should probably move to a OCL_LOCAL_ADDR buffer
  snde_index subbox;
  snde_index boxnum;
  snde_index stackentries=0;
  snde_coord dist;
  snde_index firstidx;
  snde_index polynum;
  snde_coord3 tri_vertices[3];
  
  int retval=FALSE;

  /* Push box surrounding entire part (box #0) onto stack */
  
  boxnum_stack[stackentries]=0;
  stackentries++;

  if (trace) {
    fprintf(stderr,"ray from (%f, %f, %f) in (%f,%f,%f) direction\n",raystartproj.coord[0],raystartproj.coord[1],raystartproj.coord[2],rayvecproj.coord[0],rayvecproj.coord[1],rayvecproj.coord[2]);
  }
  while (stackentries > 0) {
    boxnum=boxnum_stack[stackentries-1];
    if (trace) printf("find_ray_intersections_nonrecursive(boxnum=%d)\n",(int)boxnum);
    
    if (!ray_box_intersection(part_boxcoord[boxnum],raystartproj,rayvecproj)) {
      /* Ray does not pass through our box. We can not possibly have an intersection */
      if (trace) {
	fprintf(stderr,"find_ray_intersections(): Ray does not intersect box %d\n",(int)boxnum);
	fprintf(stderr,"Box from (%f,%f,%f)\n"
		"      to (%f,%f,%f)\n",part_boxcoord[boxnum].min.coord[0],part_boxcoord[boxnum].min.coord[1],part_boxcoord[boxnum].min.coord[2],part_boxcoord[boxnum].max.coord[0],part_boxcoord[boxnum].max.coord[1],part_boxcoord[boxnum].max.coord[2]);
	//box_contains_polygon(surf,boxnum,17281,TRUE);
      }

      // Pop this entry off the stack
      stackentries--;

      // loop back
      continue;
    } else {
      if (trace) {
	fprintf(stderr,"find_ray_intersections(): Ray does intersect box %d\n",(int)boxnum);
	fprintf(stderr,"Box from (%f,%f,%f)\n"
		"      to (%f,%f,%f)\n",part_boxcoord[boxnum].min.coord[0],part_boxcoord[boxnum].min.coord[1],part_boxcoord[boxnum].min.coord[2],part_boxcoord[boxnum].max.coord[0],part_boxcoord[boxnum].max.coord[1],part_boxcoord[boxnum].max.coord[2]);
	//box_contains_polygon(surf,boxnum,17281,TRUE);
      }

    }
    //if (trace) {
    //  box_contains_polygon(surf,boxnum,17281,TRUE);
    //}
    
    
    if (part_boxes[boxnum].boxpolysidx != SNDE_INDEX_INVALID) {
      /* Have index into boxpolys array: This box contains polygons */
      for (cnt=part_boxes[boxnum].boxpolysidx;cnt < part_boxes[boxnum].boxpolysidx + part_boxes[boxnum].numboxpolys;cnt++) {
	/* Got a polygon (actually triangle) */
	polynum=part_boxpolys[cnt];

	get_we_triverts_3d(part_triangles,polynum,part_edges,part_vertices,tri_vertices);
	
	dist = ray_to_plane_distance(raystartproj,rayvecproj, tri_vertices[0], part_trinormals[polynum]);
	
	if (trace) fprintf(stderr,"polynum=%d; zbufdist=%g\n",(int)polynum,dist);
	
	if (dist < *zdist_out) {
	  
	  if (ray_intersects_polygon(tri_vertices,3,part_trinormals[polynum],part_inplanemats[polynum],raystartproj,rayvecproj,dist,TRUE,trace)) {
	    // Will give us a truth value for whether we actually
	    // intersected this polygon.
	    
	    if (trace) fprintf(stderr,"ray_intersects_polygon; zbufdist=%g\n",dist);
	    
	    // If so, record the distance into the z-buffer,
	    // (if we are closest)
	    // the surface count and facet ids into their
	    // proper places, 
	    
	    *zdist_out=dist;
	    *instnum_out=instnum;
	    *facenum_out = part_triangles[polynum].face;	    
	    *boundarynum_out = part_topos[*facenum_out].face.boundary_num;
	    *trinum_out = polynum;
	    
	    retval = retval || TRUE;
	  } else{
	    if (trace) fprintf(stderr,"ray does not intersect polygon\n");
	    
	  }
	  
	}
      }
    }

    /* Pop this box off of the stack */
    stackentries--;

    
    /* This box may be sub-divided into 8 */
    /* Push subboxes onto stack */
    for (subbox=0;subbox < 8; subbox++) {
      if (part_boxes[boxnum].subbox[subbox] != SNDE_INDEX_INVALID) {
	if (stackentries >= frin_stacksize) {
#ifndef __OPENCL_VERSION__
	  // not an OpenCL kernel
	  assert(stackentries < frin_stacksize); /* check for stack overflow */
	
#else
	  printf("raytrace.h: STACK OVERFLOW ERROR; INCREASE FRIN_STACKSIZE\n");
#endif
	} else {
	  boxnum_stack[stackentries]=part_boxes[boxnum].subbox[subbox];	
	  stackentries++;
	}
	
      }
    }
  }
  return retval;
}




static RAYTRACE_INLINE void raytrace_find_first_intersection(snde_coord4 raystartproj,snde_coord4 rayvecproj,
							     OCL_GLOBAL_ADDR snde_partinstance *instances,
							     snde_index num_instances,
							     OCL_GLOBAL_ADDR snde_part *parts,snde_index first_part,
							     OCL_GLOBAL_ADDR snde_topological *topos,snde_index first_topo,
							     //snde_index *topo_indices,
							     OCL_GLOBAL_ADDR snde_triangle *triangles,snde_index first_triangle,
							     OCL_GLOBAL_ADDR snde_coord3 *trinormals,
							     OCL_GLOBAL_ADDR snde_cmat23 *inplanemats,
							     OCL_GLOBAL_ADDR snde_edge *edges,snde_index first_edge,
							     OCL_GLOBAL_ADDR snde_coord3 *vertices,snde_index first_vertex,
							     OCL_GLOBAL_ADDR snde_box3 *boxes,snde_index first_box,
							     OCL_GLOBAL_ADDR snde_boxcoord3 *boxcoord,
							     OCL_GLOBAL_ADDR snde_index *boxpolys, snde_index first_boxpoly,
							     snde_index frin_stacksize,
							     OCL_LOCAL_ADDR snde_index *boxnum_stack, // size frin_stacksize
							     OCL_GLOBAL_ADDR snde_coord *zdist_out,
							     OCL_GLOBAL_ADDR snde_index *instnum_out, // which instance we found, or SNDE_INDEX_INVALID for no intersection
							     OCL_GLOBAL_ADDR snde_index *boundarynum_out, // which boundary number for the instance's part
							     OCL_GLOBAL_ADDR snde_index *facenum_out, // which face number for the instance's part
							     OCL_GLOBAL_ADDR snde_index *trinum_out, // which triangle number for the instances's part
							     int trace)
{
  snde_index instancecnt;
  //snde_orientation3 orient_inv;
  snde_coord zdist;

  *zdist_out=snde_infnan(ERANGE);
  *instnum_out=SNDE_INDEX_INVALID;
  *boundarynum_out=SNDE_INDEX_INVALID;
  *facenum_out=SNDE_INDEX_INVALID;
  *trinum_out=SNDE_INDEX_INVALID;
  

  for (instancecnt=0;instancecnt < num_instances;instancecnt++) {
    snde_coord4 raystartproj_part,rayvecproj_part; // ray start position and vector in projective part coordinates
    snde_index partnum;

    /* ***!!! We could accelerate this a bit if the number of instances is large by building boxes 
       around the instances.... but these boxes would have to be rebuild if the instance list changes.
       Right now we iterate through all instances. 
    */

    partnum=instances[instancecnt].partnum-first_part;
    
    // instances[instancecnt].orientation will take a vector in the part's coordinates and
    // give a vector in world coordinates.

    // we need to do the inverse operation on raystart and rayvec to get them into part coordinates.
    // so we can find triangles.
    //orientation_inverse(instances[instancecnt].orientation,&orient_inv);
    
    if (!orientation_valid(instances[instancecnt].orientation_inverse)) {
      continue;
    }
    // transform ray into part coordinates
    orientation_apply_position(instances[instancecnt].orientation_inverse,raystartproj,&raystartproj_part);
    orientation_apply_vector(instances[instancecnt].orientation_inverse,rayvecproj,&rayvecproj_part);
    
    find_ray_first_part_intersection_nonrecursive(instancecnt,
						  //snde_index numtris,
						  &topos[parts[partnum].first_topo-first_topo],
						  //snde_index *topo_indices,
						  &triangles[parts[partnum].firsttri-first_triangle],
						  &trinormals[parts[partnum].firsttri-first_triangle],
						  &inplanemats[parts[partnum].firsttri-first_triangle],
						  &edges[parts[partnum].firstedge-first_edge],
						  &vertices[parts[partnum].firstvertex-first_vertex],
						  &boxes[parts[partnum].firstbox-first_box],
						  &boxcoord[parts[partnum].firstbox-first_box],
						  &boxpolys[parts[partnum].firstboxpoly-first_boxpoly],
						  raystartproj_part,
						  rayvecproj_part,
						  frin_stacksize,
						  boxnum_stack,
						  zdist_out,
						  instnum_out, 
						  boundarynum_out,
						  facenum_out, // which topological face? 
						  trinum_out, // which triangle number for the instances's part
						  trace);
    
  }
}






static RAYTRACE_INLINE void ray_to_plane_raydirec_shift(snde_coord4 starting_point,snde_coord4 ray_direc, snde_coord3 planepoint, snde_coord3 planenormal,snde_coord4 ray_direc_deriv,snde_coord4 *deriv_out)
  /* Calculate change in ray_to_plane intersection point with respect to ray_direc (change in normalized vector) */ 
{
  // starting point and planepoint should be 3-vectors (or normalized 4-vectors)
  // (planepoint - raypoint) dot planenormal / (raydirec dot planenormal)
  // returns -1.0 if ray is parallel to plane
  
  snde_coord4 pointdiff,firstterm,secterm,diff;
  snde_coord ray_direc_dot_planenormal;

  
  // Regular evaluation of intersection point
  subvecvec3(&planepoint.coord[0],&starting_point.coord[0],&pointdiff.coord[0]);
  pointdiff.coord[3]=0.0; // pointdiff is a vector
  
  ray_direc_dot_planenormal=dotvecvec3(&ray_direc.coord[0],&planenormal.coord[0]);
  //intersectpoint = startingpoint + dotvecvec3(pointdiff,planenormal)/ray_direc_dot_planenormal * ray_direc;

  // We need to calculate dintersectpoint/dray_direc:
  // pointdiff is constant, planenormal is constant
  //intersectpoint = startingpoint + dotvecvec3(pointdiff,planenormal)* (ray_direc/(ray_direc dot planenormal)

  // deriv intersectpoint = dotvecvec3(pointdiff,planenormal)* ( deriv ray_direc * (ray_direc dot planenormal) - ((deriv ray_direc) dot planenormal) * ray_direc)/(ray_direc dot planenormal)^2 

  // deriv intersectpoint = (dotvecvec3(pointdiff,planenormal)/(ray_direc_dot_planenormal)^2) * ( deriv ray_direc * (ray_direc dot planenormal) - ((deriv ray_direc) dot planenormal) * ray_direc)

  scalecoord4(ray_direc_dot_planenormal,ray_direc_deriv,&firstterm);
  scalecoord4(dotvecvec3(&ray_direc_deriv.coord[0],&planenormal.coord[0]),ray_direc,&secterm);
  subcoordcoord4(firstterm,secterm,&diff);
  
  scalecoord4(dotvecvec3(&pointdiff.coord[0],&planenormal.coord[0])/(ray_direc_dot_planenormal*ray_direc_dot_planenormal),diff,deriv_out); // store result in deriv_out

  
  assert(deriv_out->coord[3]==0.0); // deriv_out should be a vector
  

}





static RAYTRACE_INLINE void ray_to_plane_raypos_shift(snde_coord4 starting_point,snde_coord4 ray_direc, snde_coord3 planepoint, snde_coord3 planenormal,snde_coord4 starting_point_direc_deriv,snde_coord4 *deriv_out)
  /* Calculate change in ray_to_plane intersection point with respect to ray_direc (change in normalized vector) */ 
{
  // starting point and planepoint should be 3-vectors (or normalized 4-vectors)
  // (planepoint - raypoint) dot planenormal / (raydirec dot planenormal)
  // returns -1.0 if ray is parallel to plane
  
  snde_coord4 pointdiff;
  snde_coord ray_direc_dot_planenormal;

  
  // Regular evaluation of intersection point
  subvecvec3(&planepoint.coord[0],&starting_point.coord[0],&pointdiff.coord[0]);
  pointdiff.coord[3]=0.0; // a vector
  
  ray_direc_dot_planenormal=dotvecvec3(&ray_direc.coord[0],&planenormal.coord[0]);
  //intersectpoint = startingpoint + dotvecvec3(planepoint-starting_point,planenormal)/ray_direc_dot_planenormal * ray_direc;
  
  // We need to calculate dintersectpoint/dstartingpoint:
  // ray_direc is constant, planenormal is constant
  
  //intersectpoint = startingpoint + dotvecvec3(planepoint-starting_point,planenormal)*(ray_direc/(ray_direc dot planenormal))
  //intersectpoint = startingpoint + dotvecvec3(planepoint,planenormal)*(ray_direc/(ray_direc dot planenormal)) - dotvecvec3(starting_point,planenormal)*(ray_direc/(ray_direc dot planenormal))

  // deriv intersectpoint = deriv startingpoint  - dotvecvec3(deriv starting_point,planenormal)*(ray_direc/(ray_direc dot planenormal))
  
  addcoordscaledcoord4(starting_point_direc_deriv,-dotvecvec3(&starting_point_direc_deriv.coord[0],&planenormal.coord[0])/ray_direc_dot_planenormal,ray_direc,deriv_out);

  assert(deriv_out->coord[3]==0.0); // deriv_out should be a vector

}





static RAYTRACE_INLINE void raytrace_find_intersection_rayvec_derivative(snde_coord4 raystartproj,snde_coord4 rayvecproj,
									 snde_coord4 rayvecproj_deriv,
									 OCL_GLOBAL_ADDR snde_partinstance *instances,
									 OCL_GLOBAL_ADDR snde_part *parts, snde_index firstpart,
									 OCL_GLOBAL_ADDR snde_triangle *triangles, snde_index firsttri,
									 OCL_GLOBAL_ADDR snde_coord3 *trinormals,
									 OCL_GLOBAL_ADDR snde_edge *edges, snde_index firstedge,
									 OCL_GLOBAL_ADDR snde_coord3 *vertices, snde_index firstvertex,
									 snde_coord zdist,
									 snde_index instnum, // which instance we found, or SNDE_INDEX_INVALID for no intersection
									 snde_index trinum, // which triangle number for the instances's part
									 snde_coord4 *deriv_out)
{
  snde_coord3 tri_vertex;
  snde_index partnum;

  partnum=instances[instnum].partnum-firstpart;
  
  get_we_trivert_3d(&triangles[parts[partnum].firsttri-firsttri],trinum,&edges[parts[partnum].firstedge-firstedge],&vertices[parts[partnum].firstvertex-firstvertex],&tri_vertex);
  
  
  ray_to_plane_raydirec_shift(raystartproj,rayvecproj,tri_vertex,trinormals[parts[partnum].firsttri-firsttri],rayvecproj_deriv,deriv_out);
  
}


static RAYTRACE_INLINE void raytrace_find_intersection_raypos_derivative(snde_coord4 raystartproj,snde_coord4 rayvecproj,
									 snde_coord4 raystartproj_deriv,
									 OCL_GLOBAL_ADDR snde_partinstance *instances,
									 OCL_GLOBAL_ADDR snde_part *parts,snde_index first_part,
									 OCL_GLOBAL_ADDR snde_triangle *triangles, snde_index first_tri,
									 OCL_GLOBAL_ADDR snde_coord3 *trinormals,
									 OCL_GLOBAL_ADDR snde_edge *edges, snde_index first_edge,
									 OCL_GLOBAL_ADDR snde_coord3 *vertices, snde_index first_vertex,
									 snde_coord zdist,
									 snde_index instnum, // which instance we found, or SNDE_INDEX_INVALID for no intersection
									 snde_index trinum, // which triangle number for the instances's part
									 snde_coord4 *deriv_out)
{
  snde_coord3 tri_vertex;
  snde_index partnum;

  partnum=instances[instnum].partnum-first_part;
  
  get_we_trivert_3d(&triangles[parts[partnum].firsttri-first_tri],trinum,&edges[parts[partnum].firstedge-first_edge],&vertices[parts[partnum].firstvertex-first_vertex],&tri_vertex);
  
  
  ray_to_plane_raypos_shift(raystartproj,rayvecproj,tri_vertex,trinormals[parts[partnum].firsttri-first_tri],raystartproj_deriv,deriv_out);
  
}

static RAYTRACE_INLINE snde_coord raytrace_get_angle_of_incidence(snde_coord4 raystartproj,snde_coord4 rayvecproj,
								  OCL_GLOBAL_ADDR snde_partinstance *instances,
								  OCL_GLOBAL_ADDR snde_part *parts, snde_index first_part,
								  OCL_GLOBAL_ADDR snde_coord3 *trinormals, snde_index first_tri,
								  snde_index instnum, // which instance we found, or SNDE_INDEX_INVALID for no intersection
								  snde_index trinum) // which triangle number for the instances's part
{
  snde_coord angle_of_incidence;
  snde_index partnum;
  
  partnum=instances[instnum].partnum-first_part;

  snde_coord4 rayvecproj_partcoords;
  orientation_apply_vector(instances[instnum].orientation_inverse,rayvecproj,&rayvecproj_partcoords);
  
  angle_of_incidence = acos(fabs(dotvecvec3(&rayvecproj_partcoords.coord[0],&trinormals[parts[partnum].firsttri + trinum-first_tri].coord[0]))); // rayvecproj and facetnormals should be unit vectors 
  
  return angle_of_incidence;

}


static RAYTRACE_INLINE
void raytrace_find_intersect_uv(snde_coord4 raystartproj,snde_coord4 rayvecproj,
				OCL_GLOBAL_ADDR snde_partinstance *instances,
				OCL_GLOBAL_ADDR snde_part *parts, snde_index first_part,
				OCL_GLOBAL_ADDR snde_triangle *triangles, snde_index first_triangle, // index of first triangle in this array
				OCL_GLOBAL_ADDR snde_cmat23 *inplanemats,
				OCL_GLOBAL_ADDR snde_edge *edges, snde_index first_edge,
				OCL_GLOBAL_ADDR snde_coord3 *vertices, snde_index first_vertex,	
				OCL_GLOBAL_ADDR snde_parameterization *uvs, snde_index first_uv,
				OCL_GLOBAL_ADDR snde_triangle *uv_triangles, snde_index first_uv_tri,
				OCL_GLOBAL_ADDR snde_cmat23 *inplane2uvcoords,
				snde_coord zdist,
				snde_index instnum, // which instance we found, or SNDE_INDEX_INVALID for no intersection
				snde_index trinum, // which triangle number for the instances's part
				snde_coord3 *intersectpoint_uvproj_out, // intersection point in projective uv coordinates
				snde_index *uv_facenum_out)
				 
{
  snde_coord4 intersectpoint_worldcoords;
  snde_coord4 intersectpoint_partcoords;
  snde_coord3 tri_vertices[3];
  snde_coord4 refpoint;

  snde_coord4 polyintersectvec; // vector from polygon center to intersection point
  snde_coord3 polyintersect2; // vector from polygon center to intersection point, in projective 2D in-plane triangle coords
  snde_index partnum,paramnum;
  
  partnum=instances[instnum].partnum-first_part;
  paramnum = instances[instnum].uvnum-first_uv;
  
  // find intersectpoint 
  addcoordscaledcoord4(raystartproj,zdist,rayvecproj,&intersectpoint_worldcoords);

  orientation_apply_position(instances[instnum].orientation_inverse,intersectpoint_worldcoords,&intersectpoint_partcoords);

  // get 3D vertices
  get_we_triverts_3d(&triangles[parts[partnum].firsttri-first_triangle],trinum,&edges[parts[partnum].firstedge-first_edge],&vertices[parts[partnum].firstvertex-first_vertex],tri_vertices);

  // get centroid
  polycentroid3(tri_vertices,3 /*numvertices*/,(snde_coord3 *)&refpoint);
  refpoint.coord[3]=1.0; // refpoint is a point

  // Evaluate 3D intersection point relative to polygon
  subcoordcoord4(intersectpoint_partcoords,refpoint,&polyintersectvec);

  // Evaluate 2D intersection point relative to polygon
  multcmat23vec(&inplanemats[parts[partnum].firsttri + trinum-first_triangle].row[0].coord[0],&polyintersectvec.coord[0],&polyintersect2.coord[0]);

  polyintersect2.coord[2]=1.0; // polyintersect2 is considered a point in 2D in-plane projective space of the triangle

  // Evaluate 2D polygon to (u,v) (transform in-plane 2D coords -> parameterization coords)

  // !!!*** This is an absolute index into the full inplane2uvcoords
  // but the current projection code provides only the relevant segment (!)
  // ***!!!
  multcmat23coord(inplane2uvcoords[uvs[paramnum].firstuvtri+trinum-first_uv_tri],polyintersect2,(snde_coord2*)intersectpoint_uvproj_out);
  intersectpoint_uvproj_out->coord[2]=1.0; // this is a point, so 3rd coordinate is 1
  
  *uv_facenum_out = uv_triangles[uvs[paramnum].firstuvtri+trinum-first_uv_tri].face;
}


static RAYTRACE_INLINE
void raytrace_find_intersect_uv_deriv(snde_coord4 raystartproj,snde_coord4 rayvecproj,
				      snde_coord4 intersectpos_deriv, // from raytrace_find_intersection_raypos_derivative
				      OCL_GLOBAL_ADDR snde_partinstance *instances,
				      OCL_GLOBAL_ADDR snde_part *parts, snde_index first_part,
				      OCL_GLOBAL_ADDR snde_cmat23 *inplanemats, snde_index first_triangle,
				      OCL_GLOBAL_ADDR snde_parameterization *uvs, snde_index first_uv,
				      OCL_GLOBAL_ADDR snde_cmat23 *inplane2uvcoords, snde_index first_uv_tri,
				      snde_coord zdist,
				      snde_index instnum, // which instance we found, or SNDE_INDEX_INVALID for no intersection
				      snde_index trinum, // which triangle number for the instances's part
				      snde_coord3 *intersectderiv_uvproj_out) // intersection derivative in projective uv coordinates
{

  snde_index partnum,paramnum;
  snde_coord3 polyintersect2_deriv;
  
  partnum=instances[instnum].partnum - first_part;
  paramnum = instances[instnum].uvnum - first_uv;

  // calculate in-plane derivative
  multcmat23vec(&inplanemats[parts[partnum].firsttri + trinum - first_triangle].row[0].coord[0],&intersectpos_deriv.coord[0],&polyintersect2_deriv.coord[0]);
  polyintersect2_deriv.coord[2]=0.0; // polyintersect2_deriv is considered a vector in 2D in-plane projective space of the triangle


  multcmat23coord(inplane2uvcoords[uvs[paramnum].firstuvtri+trinum-first_uv_tri],polyintersect2_deriv,(snde_coord2 *)intersectderiv_uvproj_out);
  intersectderiv_uvproj_out->coord[2]=0.0f; // derivative is a vector not a point
  
}


static RAYTRACE_INLINE
void project_to_uv_arrays(snde_imagedata pixelval,snde_imagedata pixelweighting,
			  snde_coord2 uvcoords,snde_coord2 *uvcoords_deriv_a, snde_coord2 *uvcoords_deriv_b,
			  snde_image projectionarray_instanceinfo,
			  volatile OCL_GLOBAL_ADDR snde_atomicimagedata *uvdata_projection_array, // should have projectionbufoffset already added in
			  snde_index uvdata_projection_strides[2], // usually (1,nu)
			  OCL_GLOBAL_ADDR snde_imagedata *uvdata_weighting_array, // should have weightingbufoffset already added in 
			  snde_index uvdata_weighting_strides[2], // usually (1,nu)
			  volatile OCL_GLOBAL_ADDR snde_atomicimagedata *uvdata_validity_array, // should have validitybufoffset already added in
			  snde_index uvdata_validity_strides[2], // usually (1,nu)
			  snde_coord min_radius_uv_pixels,snde_coord min_radius_src_pixels,snde_coord bandwidth_fraction)
{

  snde_index arraywidth,arrayheight;
  int arrayu0,arrayv0;
  snde_coord projecthalfwidth,projecthalfheight;

  snde_coord uvcoords0_pixels=(uvcoords.coord[0]-projectionarray_instanceinfo.inival.coord[0])/(projectionarray_instanceinfo.step.coord[0]);
  snde_coord uvcoords1_pixels=(uvcoords.coord[1]-projectionarray_instanceinfo.inival.coord[1])/(projectionarray_instanceinfo.step.coord[1]);

  //CvMat *jacobian,*jacinv;
  snde_coord jacobian[4],jacinv[4],detinv;
  size_t jacobian_pivots[2];
  
  snde_coord weightingfactor;
    
  size_t xcnt,ycnt;

  snde_coord r2_uv_pixels,r2_src_pixels=0.0f,coeff,cosval,cosparam;  //sincparam
  snde_coord2 pos,pos_frac,srcpos;
  //float64_t angle_of_incidence_factor;

  if (isnan(pixelval) || isnan(uvcoords0_pixels) || isnan(uvcoords1_pixels)) {
    return; /* never project NaN */
  }

  // Ignore anything at extreme angles of incidence
  //if (angle_of_incidence > 3*M_PI/8) return;

  projecthalfwidth=min_radius_uv_pixels;  // texture coordinates are relative to image size, still 
  projecthalfheight=min_radius_uv_pixels;
  
  
  if (uvcoords_deriv_a && uvcoords_deriv_b) {
    snde_coord newwidth,newheight;

    // jacobian stored row-major, eats (a,b) direction for lunch, gives (u,v) direction
    jacobian[0]=uvcoords_deriv_a->coord[0]; 
    jacobian[1]=uvcoords_deriv_b->coord[0];
    jacobian[2]=uvcoords_deriv_a->coord[1]; 
    jacobian[3]=uvcoords_deriv_b->coord[1];
    
    // set up jacinv as identity so we can solve for it
    jacinv[0]=1.0;
    jacinv[1]=0.0;
    jacinv[2]=0.0;
    jacinv[3]=1.0;
    fmatrixsolve(jacobian,jacinv,2,2,jacobian_pivots,0); // ***!!! NOTE: fmatrixsolve destroys jacobian!
    // jacinv stored row-major, eats (u,v) direction for lunch, gives (a,b) direction
  
    newwidth=(uvcoords_deriv_a->coord[0]/projectionarray_instanceinfo.step.coord[0])*min_radius_src_pixels; 
    if (newwidth > projecthalfwidth) projecthalfwidth=newwidth;
    newheight=(uvcoords_deriv_a->coord[1]/projectionarray_instanceinfo.step.coord[1])*min_radius_src_pixels;
    if (newheight > projecthalfheight) projecthalfheight=newheight;
    
    newwidth=(uvcoords_deriv_b->coord[0]/projectionarray_instanceinfo.step.coord[0])*min_radius_src_pixels;
    if (newwidth > projecthalfwidth) projecthalfwidth=newwidth;
    newheight=(uvcoords_deriv_b->coord[1]/projectionarray_instanceinfo.step.coord[1])*min_radius_src_pixels;
    if (newheight > projecthalfheight) projecthalfheight=newheight;

  }
  
  arraywidth = (size_t) (projecthalfwidth*2+1);
  arrayheight= (size_t) (projecthalfheight*2+1);

  // arrayu0, arrayv0 in pixels
  arrayu0 = (int)(uvcoords0_pixels-projecthalfwidth+0.5);
  arrayv0 = (int)(uvcoords1_pixels-projecthalfheight+0.5);
  
  if (arrayu0 < 0) arrayu0=0;
  if (arrayv0 < 0) arrayv0=0;
  if (arrayu0 + arraywidth >= projectionarray_instanceinfo.nx) arraywidth = projectionarray_instanceinfo.nx-arrayu0-1;
  if (arrayv0 + arrayheight >= projectionarray_instanceinfo.ny) arrayheight = projectionarray_instanceinfo.ny-arrayv0-1;
  if (arraywidth < 0) arraywidth=0;
  if (arrayheight < 0) arrayheight=0;

  //pixelcnt=0;
  for (ycnt=0;ycnt < arrayheight;ycnt++) {
    for (xcnt=0;xcnt < arraywidth;xcnt++) {
      // xcnt+arrayx0, ycnt+arrayy0 are the indices into the pixel image
      // xcnt+arrayx0=0 corresponds to the center of the leftmost pixel
      // The left edge of this pixel should map to u=0.0,u_pixels=0.0
      // this left edge corresponds to xcnt+arrayx0=-0.5
      // That gives pos[0] = 0.0 = -0.5 -uvcoords0_pixels = -0.5 -uvcoords[0]*(imgbuf_nx) + 0.5
      //                 pos[0] = 0.0 = -0.5 - 0.0 + 0.5
      //
      // xcnt+arrayx0=imgbuf_nx-1 corresponds to the rightmost pixel
      // the right edge of thix pixel should map to u=1.0
      // This right edge corresponds to xcnt+arrayx0 = imgbuf_nx - 0.5
      // that gives pos[0] = imgbuf_nx-0.5 - uvcoords0_pixels
      //            pos[0] = imgbuf_nx-0.5 - uvcoords[0]*(imgbuf_nx) + 0.5
      //            pos[0] = imgbuf_nx-0.5 - (imgbuf_nx) + 0.5
      //            pos[0] = -0.5  + 0.5  = 0 ... so we are right on target

      
      //pos.coord[0]=projectionarray_instanceinfo.inival.coord[0]+ (xcnt+arrayu0)*projectionarray_instanceinfo.step.coord[0] - uvcoords.coord[0];
      //pos.coord[1]=projectionarray_instanceinfo.inival.coord[1]+ (ycnt+arrayv0)*projectionarray_instanceinfo.step.coord[1] - uvcoords.coord[1];
      
      //r2_uv = pos.coord[0]*pos.coord[0] + pos.coord[1]*pos.coord[1];

      // pos is position, in meaningful units, relative to the intersection point
      pos.coord[0]=projectionarray_instanceinfo.inival.coord[0]+ (xcnt+arrayu0)*projectionarray_instanceinfo.step.coord[0] - uvcoords.coord[0];
      pos.coord[1]=projectionarray_instanceinfo.inival.coord[1]+ (ycnt+arrayv0)*projectionarray_instanceinfo.step.coord[1] - uvcoords.coord[1];

      
      //if (xcnt==0 && ycnt==0) {
      //fprintf(stderr,"Projecting, pos = (%f,%f)\n",pos.coord[0],pos.coord[1]);
      //}

      // convert pos from meaningful units relative to intersection point into pixels relative to intersection point
      pos.coord[0] /= projectionarray_instanceinfo.step.coord[0];
      pos.coord[1] /= projectionarray_instanceinfo.step.coord[1];
      
      r2_uv_pixels = pos.coord[0]*pos.coord[0] + pos.coord[1]*pos.coord[1];

      
      //fprintf(stderr,"r_uv=%g; min_radius_uv = %g\n",sqrt(r2_uv),fabs(min_radius_uv_pixels));
      
      if (uvcoords_deriv_a && uvcoords_deriv_b) {
	// pos is in uv units so far, and so are jacobian/jacimg
	multcmatvec2(jacinv,&pos.coord[0],&srcpos.coord[0]); // srcpos gives position in (a,b) units, which are source pixels
	r2_src_pixels = srcpos.coord[0]*srcpos.coord[0]+srcpos.coord[1]*srcpos.coord[1];
	//fprintf(stderr,"r_src=%g; min_radius_src = %g\n",sqrt(r2_src),fabs(min_radius_src_pixels));
	assert(0); // need to debug here and check units... r2_src probably needs to be scaled from meaningful units into source pixels (?) ***!!! See also cosparam, below !!!***
      }
      if (r2_uv_pixels <= min_radius_uv_pixels*min_radius_uv_pixels ||
	  r2_src_pixels <= min_radius_src_pixels*min_radius_src_pixels) {
	/* Include this point */
	// Forget complicated bandlimited interpolation
	// Instead project 2D generalized circular sinc function
	// from source into UV space

	
	//if (pixelcnt >= max_projection_pixels) {
	//  /* too many pixels */
	//  goto fail; 
	//}

	weightingfactor=1.0;
	if (uvdata_weighting_array) {
	  // does this really need to be atomic??? 
	  //weightingfactor=atomicpixel_load(&weightingbuf[(arrayx0+xcnt)  (arrayy0+ycnt)*projectionarray_instanceinfo.nx]);
	  weightingfactor = uvdata_weighting_array[(arrayu0+xcnt)*uvdata_weighting_strides[0] + (arrayv0+ycnt)*uvdata_weighting_strides[1]];
	}


	snde_coord r2_pixels=r2_src_pixels;

	if (r2_uv_pixels > r2_pixels) {
	  r2_pixels = r2_uv_pixels; // whichever is bigger
	}

	// Replacement -- Just use raised Cosine weigting
	// 1+cos(sqrt(r2_pixels)*bandwidth_fraction)
	
	cosparam=sqrt(r2_pixels)*bandwidth_fraction*M_PI/M_SQRT2; // !!!** Need to check scaling/units of r2_src.... see assert(0) above
	if (cosparam > M_PI_SNDE_COORD) {
	  cosval=0.0;
	} else {
	  cosval=0.5+0.5*cos(cosparam);
	}
	coeff=pixelweighting*weightingfactor*cosval;
	
	//fprintf(stderr,"imgbuf[%d]+=%g\n",framenum*imgbuf_ny*imgbuf_nx + (arrayy0+ycnt)*imgbuf_nx + (arrayx0+xcnt),coeff*pixelval);

	if (uvdata_projection_array) {
	  //if (xcnt==(int)arraywidth/2 && ycnt==(int)arrayheight/2) {
	  //  fprintf(stderr,"Projecting, to pixel = (%d,%d)\n",(int)(arrayu0+xcnt),(int)(arrayv0+ycnt));
	  //}
	  atomicpixel_accumulate(&uvdata_projection_array[(arrayu0+xcnt)*uvdata_projection_strides[0] + (arrayv0+ycnt)*uvdata_projection_strides[1]], coeff*pixelval);
	  
	}
	if (uvdata_validity_array) {
	  atomicpixel_accumulate(&uvdata_validity_array[(arrayu0+xcnt)*uvdata_validity_strides[0] + (arrayv0+ycnt)*uvdata_validity_strides[1] ],coeff);
	}
	      
      }
    }
  }

}

struct rayintersection_properties {
  // these are defined for each pixel of the image being projected

  // outputs of zbuffer calculation function
  snde_coord zdist;  // z distance
  snde_index instnum; // which instance number?
  snde_index boundarynum; // which boundary number of the instance
  snde_index facenum; // which face number of the instance boundary
  snde_index trinum;  // which triangle number of the instance boundary face
  snde_imagedata angleofincidence;
  snde_imagedata angleofincidence_weighting;
  snde_imagedata ray_intersect_deriv_a_cam_z;
  snde_imagedata ray_intersect_deriv_b_cam_z;
  snde_coord2 uvcoords;
  snde_index uvfacenum;
  snde_coord2 uvcoords_deriv_a;
  snde_coord2 uvcoords_deriv_b;


  // inputs of raytrace_camera_projection: above (specifically including uvcoords and uvcoords derivs)
  // plus:
  snde_imagedata pixelweighting;
};



static RAYTRACE_INLINE void raytrace_camera_projection(OCL_GLOBAL_ADDR snde_imagedata *pixelbuf,
						       snde_index src_na, snde_index src_nb,
						       snde_index src_strides[2], // usually (1, src_na)... really used for imagedata_intersectprops layout
						       OCL_GLOBAL_ADDR snde_partinstance *instances,
						       OCL_GLOBAL_ADDR snde_parameterization *uvs, snde_index first_uv,
						       OCL_GLOBAL_ADDR snde_topological *uv_topos, snde_index first_uv_topo,
						       OCL_GLOBAL_ADDR snde_image *projectionarray_info, // projectionarray_info, indexed according to the firstuvimages of the partinstance, defines the layout of uvdata_angleofincidence_weighting and uvdata_angleofincidence_weighting_validity uv imagedata arrays
						       OCL_GLOBAL_ADDR struct rayintersection_properties *imagedata_intersectprops, // Array of structures, one per pixel, laid out same way as pixelbuf (same strides)
						       volatile OCL_GLOBAL_ADDR snde_imagedata *uvdata_projection_arrays,
						       snde_index uvdata_projection_strides[2], // usually (1,nu)
						       OCL_GLOBAL_ADDR snde_imagedata *uvdata_weighting_arrays,
						       snde_index uvdata_weighting_strides[2], // usually (1,nu)
						       volatile OCL_GLOBAL_ADDR snde_imagedata *uvdata_validity_arrays,
						       snde_index uvdata_validity_strides[2], // usually (1,nu)
						       snde_coord min_radius_uv_pixels,snde_coord min_radius_src_pixels,snde_coord bandwidth_fraction)
{
  
  
  {
    int64_t bcnt; // MSVC openmp won't accept unsigneds like snde_index

#ifdef RAYTRACE_USE_OPENMP
#pragma omp parallel default(shared) private(bcnt)
#pragma omp for
#endif
    for (bcnt=0;bcnt < src_nb;bcnt++) {
      
      snde_index acnt;
      
      
      for (acnt=0; acnt < src_na; acnt++) {

  
	//pixelval = pixelbuf[acnt + bcnt*src_na];
	snde_index instnum = imagedata_intersectprops[acnt*src_strides[0] + bcnt*src_strides[1]].instnum;
	
	snde_index paramnum = instances[instnum].uvnum-first_uv;
	snde_index uv_intersection_face_topo_num = uvs[paramnum].first_uv_topo + uvs[paramnum].firstuvface + imagedata_intersectprops[acnt*src_strides[0] + bcnt*src_strides[1]].uvfacenum - first_uv_topo;
	snde_index patchnum = uv_topos[uv_intersection_face_topo_num].face.patchnum;
	
	snde_imagedata pixelval = pixelbuf[acnt*src_strides[0] + bcnt*src_strides[1]];
	
	snde_image thisprojarray_info = projectionarray_info[instances[instnum].firstuvpatch + patchnum];

	snde_index projectionbufoffset = thisprojarray_info.projectionbufoffset;
	snde_index weightingbufoffset = thisprojarray_info.weightingbufoffset;
	snde_index validitybufoffset = thisprojarray_info.validitybufoffset;

	snde_index thispatch = instances[imagedata_intersectprops[acnt*src_strides[0] + bcnt*src_strides[1]].instnum].firstuvpatch + patchnum;
	
	project_to_uv_arrays(pixelval,imagedata_intersectprops[acnt*src_strides[0] + bcnt*src_strides[1]].pixelweighting,
			     imagedata_intersectprops[acnt*src_strides[0] + bcnt*src_strides[1]].uvcoords,
			     &imagedata_intersectprops[acnt*src_strides[0] + bcnt*src_strides[1]].uvcoords_deriv_a,
			     &imagedata_intersectprops[acnt*src_strides[0] + bcnt*src_strides[1]].uvcoords_deriv_b,
			     projectionarray_info[thispatch],
			     (snde_atomicimagedata *)((void *)(uvdata_projection_arrays + projectionbufoffset)),
			     uvdata_projection_strides,
			     uvdata_weighting_arrays ? (uvdata_weighting_arrays + weightingbufoffset):nullptr,
			     uvdata_weighting_strides,
			     (snde_atomicimagedata *)((void *)(uvdata_validity_arrays + validitybufoffset)),
			     uvdata_validity_strides,
			     min_radius_uv_pixels,min_radius_src_pixels,bandwidth_fraction);
      }
    }
  }
}


#ifdef __OPENCL_VERSION__
// For an OpenCL kernel
static RAYTRACE_INLINE void raytrace_camera_projection_opencl(OCL_GLOBAL_ADDR snde_imagedata *pixelbuf,
							      snde_index src_na, snde_index src_nb,
							      snde_index src_strides[2], // usually (1, src_na)
							      OCL_GLOBAL_ADDR snde_partinstance *instances,
							      OCL_GLOBAL_ADDR snde_image *projectionarray_info, // projectionarray_info, indexed according to the firstuvimages of the partinstance, defines the layout of uvdata_angleofincidence_weighting and uvdata_angleofincidence_weighting_validity uv imagedata arrays
							      OCL_GLOBAL_ADDR struct rayintersection_properties *imagedata_intersectprops, // Array of structures, one per pixel
							      volatile OCL_GLOBAL_ADDR snde_imagedata *uvdata_projection_arrays,
							      snde_index uvdata_projection_strides[2], // usually (1,nu)
							      OCL_GLOBAL_ADDR snde_imagedata *uvdata_weighting_arrays,
							      snde_index uvdata_weighting_strides[2], // usually (1,nu)
							      volatile OCL_GLOBAL_ADDR snde_imagedata *uvdata_validity_arrays,
							      snde_index uvdata_validity_strides[2], // usually (1,nu)
							      snde_coord min_radius_uv_pixels,snde_coord min_radius_src_pixels,snde_coord bandwidth_fraction);

  
{
  snde_index aindex = get_global_id(0);
  snde_index bindex = get_global_id(1);
  pixelval = pixelbuf[aindex*src_strides[0] + bindex*src_strides[1]];
  project_to_uv_arrays(pixelval,imagedata_intersectprops[aindex*src_strides[0] + bindex*src_strides[1]].pixelweighting,
		       imagedata_intersectprops[aindex*src_strides[0] + bindex*src_strides[1]].uvcoords,
		       imagedata_intersectprops[aindex*src_strides[0] + bindex*src_strides[1]].uvcoords_deriv_a,
		       imagedata_intersectprops[aindex*src_strides[0] + bindex*src_strides[1]].uvcoords_deriv_b,
		       &projectionarray_info[instances[imagedata_intersectprops[aindex*src_strides[0] + bindex*src_strides[1]]].firstuvimage],
		       uvdata_projection_arrays,
		       uvdata_projection_strides,
		       uvdata_weighting_arrays,
		       uvdata_weighting_strides,
		       uvdata_validity_arrays,
		       uvdata_validity_strides,
		       min_radius_uv_pixels,min_radius_src_pixels,bandwidth_fraction);
}

#endif // __OPENCL_VERSION__


static RAYTRACE_INLINE void raytrace_evaluate_focalpointwrl(snde_orientation3 orient_wrlcoords_over_camcoords,snde_coord4 *focalpointwrl)
{
  snde_coord4 focalpointcam={.coord={0.0,0.0,0.0,1.0}};

  // calculation of focalpointobj: Coordinates of focal point in world coordinates
  orientation_apply_position(orient_wrlcoords_over_camcoords,focalpointcam,focalpointwrl);
  
}

static RAYTRACE_INLINE void raytrace_evaluate_orient_camcoords_over_wrlcoords(snde_orientation3 orient_wrlcoords_over_camcoords,snde_orientation3 *orient_camcoords_over_wrlcoords)
{
  orientation_inverse(orient_wrlcoords_over_camcoords,orient_camcoords_over_wrlcoords);
  
}


static RAYTRACE_INLINE
void raytrace_sensor_evaluate_zdist(
				    snde_orientation3 orient_wrlcoords_over_sensorcoords,
				    snde_orientation3 orient_sensorcoords_over_wrlcoords,
				    snde_coord mindist, // Generally negative to accommodate slight sensor/specimen overlap due to positioning error
				    snde_coord maxdist,
				    OCL_GLOBAL_ADDR snde_partinstance *instances,
				    snde_index num_instances,
				    OCL_GLOBAL_ADDR snde_part *parts, snde_index first_part,
				    OCL_GLOBAL_ADDR snde_topological *topos, snde_index first_topo,
				    OCL_GLOBAL_ADDR snde_triangle *triangles, snde_index first_triangle,
				    OCL_GLOBAL_ADDR snde_coord3 *trinormals,
				    OCL_GLOBAL_ADDR snde_cmat23 *inplanemats,
				    OCL_GLOBAL_ADDR snde_edge *edges, snde_index first_edge,
				    OCL_GLOBAL_ADDR snde_coord3 *vertices, snde_index first_vertex,
				    OCL_GLOBAL_ADDR snde_box3 *boxes, snde_index first_box,
				    OCL_GLOBAL_ADDR snde_boxcoord3 *boxcoord,
				    OCL_GLOBAL_ADDR snde_index *boxpolys, snde_index first_boxpoly,
				    OCL_GLOBAL_ADDR snde_parameterization *uvs, snde_index first_uv,
				    OCL_GLOBAL_ADDR snde_triangle *uv_triangles, snde_index first_uv_tri,
				    OCL_GLOBAL_ADDR snde_cmat23 *inplane2uvcoords,
				    OCL_GLOBAL_ADDR snde_image *projectionarray_info, // projectionarray_info, indexed according to the firstuvimages of the partinstance, defines the layout of uvdata_angleofincidence_weighting and uvdata_angleofincidence_weighting_validity uv imagedata arrays
				    snde_index frin_stacksize,
				    OCL_LOCAL_ADDR snde_index *boxnum_stack, // size frin_stacksize
				    OCL_GLOBAL_ADDR struct rayintersection_properties *imagedata_intersectprops) // JUST the structure for this pixel... we don't index it
{

  snde_float32 NaNval;
  NaNval=snde_infnan(0);

  snde_coord4 rayvecwrl; // rayvec in object coordinates (projective)


  int trace=FALSE;
  //int trace=TRUE;
  snde_index firstidx;

  // find the first surface intersection in the direction of the sensor
  // sensor active area is presumed to be the origin in its frame, and it looks in
  // the -z direction.
  
  // We get the surface ID. This will give us a map
  // of the source image for each pixel, which surface
  // it maps onto.
  
  //camera_rayvec_wrl(cam_mtx,aindex,bindex,orient_wrlcoords_over_camcoords,&rayvecwrl);
  snde_coord4 sensorloc_sensor = { { 0,0,-mindist,1 } }; // Point located at the minimum distance along the sensor axis (z)  in sensor coordinates. Negated because behind the sensor is in the +z direction
  snde_coord4 sensordirec_sensor = { { 0,0,-1,0 } };  // Looks in the -z direction
  
  snde_coord4 sensorloc_wrl;
  snde_coord4 sensordirec_wrl;

  orientation_apply_position(orient_wrlcoords_over_sensorcoords,sensorloc_sensor,&sensorloc_wrl);
  orientation_apply_vector(orient_wrlcoords_over_sensorcoords,sensordirec_sensor,&sensordirec_wrl);
  
  
  raytrace_find_first_intersection(sensorloc_wrl,sensordirec_wrl,
				   instances,
				   num_instances,
				   parts, first_part,
				   topos, first_topo,
				   //snde_index *topo_indices,
				   triangles, first_triangle,
				   trinormals,
				   inplanemats,
				   edges, first_edge,
				   vertices, first_vertex,
				   boxes, first_box,
				   boxcoord,
				   boxpolys, first_boxpoly,
				   frin_stacksize,
				   boxnum_stack,
				   &imagedata_intersectprops->zdist,
				   &imagedata_intersectprops->instnum, // which instance we found, or SNDE_INDEX_INVALID for no intersection
				   &imagedata_intersectprops->boundarynum, // which boundary number for the instance's part
				   &imagedata_intersectprops->facenum, // which face number for the instance's part
				   &imagedata_intersectprops->trinum, // which triangle number for the instances's part
				   trace);
  
  /* Next phase: evaluate derivatives, etc. */
  {
    snde_coord zdist;
    snde_index instnum,trinum,uv_facenum;
    
    
    
    snde_coord angle_of_incidence,angle_of_incidence_factor;
    snde_coord3 intersectpoint_uvcoords;


    
    zdist=imagedata_intersectprops->zdist;
    instnum=imagedata_intersectprops->instnum;
    trinum=imagedata_intersectprops->trinum;
    //paramnum = instances[instnum].uvnum;

    if (zdist > maxdist-mindist) {
      // too far: Set zdist to infinity
      zdist = snde_infnan(ERANGE);
    }
    
    if (!isinf(zdist)) { 

      
      imagedata_intersectprops->ray_intersect_deriv_a_cam_z=NaNval;
      //if (imagedata_ray_intersect_deriv_b_cam_z)
      imagedata_intersectprops->ray_intersect_deriv_b_cam_z=NaNval;

      
      
      //if (imagedata_angleofincidence || imagedata_angleofincidence_weighting || uvdata_angleofincidence_weighting) {
      angle_of_incidence = raytrace_get_angle_of_incidence(sensorloc_wrl,sensordirec_wrl,
							   instances,
							   parts, first_part,
							   trinormals, first_triangle,
							   instnum, // which instance we found, or SNDE_INDEX_INVALID for no intersection
							   trinum); // which triangle number for the instances's part
      
      //if (imagedata_angleofincidence) {
      imagedata_intersectprops->angleofincidence = angle_of_incidence;
      imagedata_intersectprops->angleofincidence_weighting = NaNval;
      //}
      
      
      
      
      //if (imagedata_uvcoords || imagedata_uvfacenum  || imagedata_uvcoords_deriv_a || imagedata_uvcoords_deriv_b || projectionarrayinfo) {
      raytrace_find_intersect_uv(sensorloc_wrl,sensordirec_wrl,
				 instances,
				 parts, first_part,
				 triangles, first_triangle,
				 inplanemats,
				 edges, first_edge,
				 vertices, first_vertex, 
				 uvs, first_uv,
				 uv_triangles, first_uv_tri,
				 inplane2uvcoords, 
				 zdist,
				 instnum,
				 trinum,
				 &intersectpoint_uvcoords,
				 &uv_facenum);
      
      //if (imagedata_uvcoords) {
      imagedata_intersectprops->uvcoords.coord[0]=intersectpoint_uvcoords.coord[0]; // Store u coordinate
      imagedata_intersectprops->uvcoords.coord[1]=intersectpoint_uvcoords.coord[1];  // store v coordinate
      //}
      
      //if (imagedata_uvfacenum) {
      imagedata_intersectprops->uvfacenum=uv_facenum;
      //}
      
      
      imagedata_intersectprops->uvcoords_deriv_a.coord[0] = NaNval;
      imagedata_intersectprops->uvcoords_deriv_a.coord[1] = NaNval;
      
      
      imagedata_intersectprops->uvcoords_deriv_b.coord[0] = NaNval;
      imagedata_intersectprops->uvcoords_deriv_b.coord[1] = NaNval;
      
      //}
      //}
      
      
    } else {
      // zdist == inf, i.e. no intersection. Other outputs should be NaN
      //if (imagedata_ray_intersect_deriv_a_cam_z)
      imagedata_intersectprops->ray_intersect_deriv_a_cam_z=NaNval;
      //if (imagedata_ray_intersect_deriv_b_cam_z)
      imagedata_intersectprops->ray_intersect_deriv_b_cam_z=NaNval;
      //if (imagedata_angleofincidence)
      imagedata_intersectprops->angleofincidence=NaNval;
      //if (imagedata_angleofincidence_weighting)
      imagedata_intersectprops->angleofincidence_weighting=NaNval;
      //if (imagedata_uvcoords) {
      imagedata_intersectprops->uvcoords.coord[0]=NaNval;
      imagedata_intersectprops->uvcoords.coord[1]=NaNval;
      //}
      //if (imagedata_uvfacenum)
      imagedata_intersectprops->uvfacenum=SNDE_INDEX_INVALID;
      
      //if (imagedata_uvcoords_deriv_a) {
      imagedata_intersectprops->uvcoords_deriv_a.coord[0] = NaNval;
      imagedata_intersectprops->uvcoords_deriv_a.coord[1] = NaNval;
      //}
      
      //if (imagedata_uvcoords_deriv_b) {
      imagedata_intersectprops->uvcoords_deriv_b.coord[0] = NaNval;
      imagedata_intersectprops->uvcoords_deriv_b.coord[1] = NaNval;
      //}
    }
    
  }
  
  
}



static RAYTRACE_INLINE
void raytrace_camera_evaluate_zdist(
				    snde_cmat23 cam_mtx,
				    snde_coord4 focalpointwrl,
				    snde_orientation3 orient_wrlcoords_over_camcoords,
				    snde_orientation3 orient_camcoords_over_wrlcoords,
				    snde_index aindex,snde_index bindex, // index of the particular pixel ray we are interested in
				      
				    OCL_GLOBAL_ADDR snde_partinstance *instances,
				    snde_index num_instances,
				    OCL_GLOBAL_ADDR snde_part *parts, snde_index first_part,
				    OCL_GLOBAL_ADDR snde_topological *topos, snde_index first_topo,
				    OCL_GLOBAL_ADDR snde_triangle *triangles, snde_index first_triangle,
				    OCL_GLOBAL_ADDR snde_coord3 *trinormals,
				    OCL_GLOBAL_ADDR snde_cmat23 *inplanemats,
				    OCL_GLOBAL_ADDR snde_edge *edges, snde_index first_edge,
				    OCL_GLOBAL_ADDR snde_coord3 *vertices, snde_index first_vertex,
				    OCL_GLOBAL_ADDR snde_box3 *boxes, snde_index first_box,
				    OCL_GLOBAL_ADDR snde_boxcoord3 *boxcoord,
				    OCL_GLOBAL_ADDR snde_index *boxpolys, snde_index first_boxpoly,
				    OCL_GLOBAL_ADDR snde_parameterization *uvs, snde_index first_uv,
				    OCL_GLOBAL_ADDR snde_topological *uv_topos, snde_index first_uv_topo,
				    OCL_GLOBAL_ADDR snde_triangle *uv_triangles, snde_index first_uv_tri,
				    OCL_GLOBAL_ADDR snde_cmat23 *inplane2uvcoords, 
				    OCL_GLOBAL_ADDR snde_image *projectionarray_info, // projectionarray_info, indexed according to the firstuvimages of the partinstance, defines the layout of uvdata_angleofincidence_weighting and uvdata_angleofincidence_weighting_validity uv imagedata arrays
				    snde_coord min_radius_uv_pixels,snde_coord min_radius_src_pixels,snde_coord bandwidth_fraction,
				    snde_index frin_stacksize,
				    OCL_LOCAL_ADDR snde_index *boxnum_stack, // size frin_stacksize

				    OCL_GLOBAL_ADDR struct rayintersection_properties *imagedata_intersectprops, // JUST the structure for this pixel... we don't index it
				    //OCL_GLOBAL_ADDR snde_coord *imagedata_zbuffer,
				    //OCL_GLOBAL_ADDR snde_index *imagedata_instnum,
				    //OCL_GLOBAL_ADDR snde_index *imagedata_facenum,
				    //OCL_GLOBAL_ADDR snde_index *imagedata_trinum,
				    //OCL_GLOBAL_ADDR snde_imagedata *imagedata_angleofincidence,
				    //OCL_GLOBAL_ADDR snde_imagedata *imagedata_angleofincidence_weighting,
				    //OCL_GLOBAL_ADDR snde_imagedata *imagedata_ray_intersect_deriv_a_cam_z,
				    //OCL_GLOBAL_ADDR snde_imagedata *imagedata_ray_intersect_deriv_b_cam_z,
				    //OCL_GLOBAL_ADDR snde_coord *imagedata_uvcoords,
				    //OCL_GLOBAL_ADDR snde_index *imagedata_uvfacenum,
				    //OCL_GLOBAL_ADDR snde_coord *imagedata_uvcoords_deriv_a,
				    //OCL_GLOBAL_ADDR snde_coord *imagedata_uvcoords_deriv_b,
				    OCL_GLOBAL_ADDR snde_imagedata *uvdata_angleofincidence_weighting,
				    OCL_GLOBAL_ADDR snde_imagedata *uvdata_angleofincidence_weighting_validity)
{

  snde_coord NaNval;
  NaNval=snde_infnan(0);

  snde_coord4 rayvecwrl; // rayvec in object coordinates (projective)


  int trace=FALSE;
  snde_index firstidx;

  // Go through this source image pixel,
  // find the first surface intersection mark the z-buffer
  // and surface ID. This will give us a map
  // of the source image for each pixel, which surface
  // it maps onto.
    
  camera_rayvec_wrl(cam_mtx,aindex,bindex,orient_wrlcoords_over_camcoords,&rayvecwrl);
  
  
  raytrace_find_first_intersection(focalpointwrl,rayvecwrl,
				   instances,
				   num_instances,
				   parts, first_part,
				   topos, first_topo,
				   //snde_index *topo_indices,
				   triangles, first_triangle,
				   trinormals, 
				   inplanemats,
				   edges, first_edge,
				   vertices, first_vertex,
				   boxes, first_box,
				   boxcoord,
				   boxpolys, first_boxpoly,
				   frin_stacksize,
				   boxnum_stack,
				   &imagedata_intersectprops->zdist,
				   &imagedata_intersectprops->instnum, // which instance we found, or SNDE_INDEX_INVALID for no intersection
				   &imagedata_intersectprops->boundarynum, // which boundary number for the instance's part
				   &imagedata_intersectprops->facenum, // which face number for the instance's part
				   &imagedata_intersectprops->trinum, // which triangle number for the instances's part
				   trace);
  
  /* Next phase: evaluate derivatives, etc. */
  {
    snde_coord zdist;
    snde_index instnum,trinum,paramnum,uv_facenum;
    
    
    snde_coord4 rayvecderiv_a;
    snde_coord4 ray_intersect_deriv_a;
    snde_coord4 ray_intersect_deriv_a_cam;
    snde_coord4 rayvecderiv_b;
    snde_coord4 ray_intersect_deriv_b;
    snde_coord4 ray_intersect_deriv_b_cam;
    
    snde_coord angle_of_incidence,angle_of_incidence_factor;
    snde_coord3 intersectpoint_uvcoords;
    snde_coord3 intersectderiv_uvproj_a;
    snde_coord3 intersectderiv_uvproj_b;
    
    zdist=imagedata_intersectprops->zdist;
    instnum=imagedata_intersectprops->instnum;
    trinum=imagedata_intersectprops->trinum;
    paramnum = instances[instnum].uvnum-first_uv;


    // ***!!!! BUG: Does not currently consider the coordinate transformation in the instance!
    
    camera_rayvec_deriv_a_wrl(cam_mtx,aindex,bindex,orient_wrlcoords_over_camcoords,&rayvecderiv_a);
    camera_rayvec_deriv_b_wrl(cam_mtx,aindex,bindex,orient_wrlcoords_over_camcoords,&rayvecderiv_b);

    raytrace_find_intersection_rayvec_derivative(focalpointwrl,rayvecwrl,
						 rayvecderiv_a,
						 instances,
						 parts, first_part,
						 triangles, first_triangle,
						 trinormals, 
						 edges, first_edge,
						 vertices, first_vertex,
						 zdist,
						 instnum, // which instance we found, or SNDE_INDEX_INVALID for no intersection
						 trinum, // which triangle number for the instances's part
						 &ray_intersect_deriv_a);
    
    // ***!!!! BUG: Does not currently consider the coordinate transformation in the instance!
    
    raytrace_find_intersection_rayvec_derivative(focalpointwrl,rayvecwrl,
						 rayvecderiv_b,
						 instances,
						 parts, first_part,
						 triangles, first_triangle,
						 trinormals,
						 edges, first_edge,
						 vertices, first_vertex,
						 zdist,
						 instnum, // which instance we found, or SNDE_INDEX_INVALID for no intersection
						 trinum, // which triangle number for the instances's part
						 &ray_intersect_deriv_b);
    
    if (!isinf(zdist)) { 
      
      //if (imagedata_ray_intersect_deriv_a_cam_z) {
      // derivative with respect to a in camera coordinates
      // ***!!!! BUG: Does not currently consider the coordinate transformation in the instance!
      orientation_apply_vector(orient_camcoords_over_wrlcoords,ray_intersect_deriv_a,&ray_intersect_deriv_a_cam);
      
      // export z-derivative with respect to a in camera coordinates
      imagedata_intersectprops->ray_intersect_deriv_a_cam_z=ray_intersect_deriv_a_cam.coord[2];
      //}
      
      //if (imagedata_ray_intersect_deriv_b_cam_z) {
      // derivative with respect to b in camera coordinates
      orientation_apply_vector(orient_camcoords_over_wrlcoords,ray_intersect_deriv_b,&ray_intersect_deriv_b_cam);
      // export z-derivative with respect to b in camera coordinates
      imagedata_intersectprops->ray_intersect_deriv_b_cam_z=ray_intersect_deriv_b_cam.coord[2];
      //}
      
      
      //if (imagedata_angleofincidence || imagedata_angleofincidence_weighting || uvdata_angleofincidence_weighting) {
      angle_of_incidence = raytrace_get_angle_of_incidence(focalpointwrl,rayvecwrl,
							   instances,
							   parts, first_part,
							   trinormals, first_triangle,
							   instnum, // which instance we found, or SNDE_INDEX_INVALID for no intersection
							   trinum); // which triangle number for the instances's part
      
      //if (imagedata_angleofincidence) {
      imagedata_intersectprops->angleofincidence = angle_of_incidence;
      //}
      
      // Ignore anything at extreme angles of incidence
      if (angle_of_incidence > 3.0f*M_PI_SNDE_COORD/8.0f) {
	angle_of_incidence_factor=0.0f;
      } else {  
	// Define factor by which we de-emphasize data at larger angles of incidence
	angle_of_incidence_factor = cos(angle_of_incidence * (M_PI_SNDE_COORD/2.0f)/(3.0f*M_PI_SNDE_COORD/8.0f));	    
      }
      
      //if (imagedata_angleofincidence_weighting) {
      imagedata_intersectprops->angleofincidence_weighting = angle_of_incidence_factor;
      //  }
      //}
      
      
      
      //if (imagedata_uvcoords || imagedata_uvfacenum  || imagedata_uvcoords_deriv_a || imagedata_uvcoords_deriv_b || projectionarray_info) {
      raytrace_find_intersect_uv(focalpointwrl,rayvecwrl,
				 instances,
				 parts, first_part,
				 triangles, first_triangle,
				 inplanemats, 
				 edges, first_edge, 
				 vertices, first_vertex,
				 uvs, first_uv, 
				 uv_triangles, first_uv_tri,
				 inplane2uvcoords, 
				 zdist,
				 instnum,
				 trinum,
				 &intersectpoint_uvcoords,
				 &uv_facenum);
      
      //if (imagedata_uvcoords) {
      imagedata_intersectprops->uvcoords.coord[0]=intersectpoint_uvcoords.coord[0]; // Store u coordinate
      imagedata_intersectprops->uvcoords.coord[1]=intersectpoint_uvcoords.coord[1];  // store v coordinate
      //}
      
      //if (imagedata_uvfacenum) {
      imagedata_intersectprops->uvfacenum=uv_facenum;
      //}
      
      raytrace_find_intersect_uv_deriv(focalpointwrl,rayvecwrl,
				       ray_intersect_deriv_a, // from raytrace_find_intersection_raypos_derivative
				       instances,
				       parts, first_part,
				       inplanemats, first_triangle,
				       uvs, first_uv,
				       inplane2uvcoords, first_uv_tri,
				       zdist,
				       instnum, // which instance we found, or SNDE_INDEX_INVALID for no intersection
				       trinum, // which triangle number for the instances's part
				       &intersectderiv_uvproj_a); // intersection point in projective uv coords
      
      //if (imagedata_uvcoords_deriv_a) {
      
      imagedata_intersectprops->uvcoords_deriv_a.coord[0] = intersectderiv_uvproj_a.coord[0];
      imagedata_intersectprops->uvcoords_deriv_a.coord[1] = intersectderiv_uvproj_a.coord[1];
      //  }
      
      raytrace_find_intersect_uv_deriv(focalpointwrl,rayvecwrl,
				       ray_intersect_deriv_b, // from raytrace_find_intersection_raypos_derivative
				       instances,
				       parts, first_part, 
				       inplanemats, first_triangle,
				       uvs, first_uv, 
				       inplane2uvcoords, first_uv_tri,
				       zdist,
				       instnum, // which instance we found, or SNDE_INDEX_INVALID for no intersection
				       trinum, // which triangle number for the instances's part
				       &intersectderiv_uvproj_b); // intersection derivative in projective uv coords
      
      
      //if (imagedata_uvcoords_deriv_b) {
      
      
      imagedata_intersectprops->uvcoords_deriv_b.coord[0] = intersectderiv_uvproj_b.coord[0];
      imagedata_intersectprops->uvcoords_deriv_b.coord[1] = intersectderiv_uvproj_b.coord[1];
      
      //}
      //}
      
      
      if (projectionarray_info && uvdata_angleofincidence_weighting && uvdata_angleofincidence_weighting_validity) {
	/* if buffer to store angle_of_incidence_factor in uv coordinate frame is provided... */
	/* use project_to_uv_arrays() to map out projection validity region */
	/* we provide angle_of_incidence_factor as the pixel value
	   so that imgbuf ends up with the sum of projection weighting * angle of incidence factor 
	   whereas validitybuf ends up with the sum of the angle of incidence factors */
	snde_index uv_intersection_face_topo_idx=uvs[paramnum].first_uv_topo+uvs[paramnum].firstuvface+uv_facenum-first_uv_topo;
	
	snde_image thisprojarray_info = projectionarray_info[instances[instnum].firstuvpatch + uv_topos[uv_intersection_face_topo_idx].face.patchnum];
	
	snde_index projectionbufoffset = thisprojarray_info.projectionbufoffset;
	snde_index weightingbufoffset = thisprojarray_info.weightingbufoffset;
	snde_index validitybufoffset = thisprojarray_info.validitybufoffset;

	snde_coord2 intersectpoint_uvcoords_coord2;
	intersectpoint_uvcoords_coord2.coord[0]=intersectpoint_uvcoords.coord[0];
	intersectpoint_uvcoords_coord2.coord[1]=intersectpoint_uvcoords.coord[1];

	snde_coord2 intersectderiv_uvproj_a_coord2;
	intersectderiv_uvproj_a_coord2.coord[0]=intersectderiv_uvproj_a.coord[0];
	intersectderiv_uvproj_a_coord2.coord[1]=intersectderiv_uvproj_a.coord[1];
	snde_coord2 intersectderiv_uvproj_b_coord2;
	intersectderiv_uvproj_b_coord2.coord[0]=intersectderiv_uvproj_b.coord[0];
	intersectderiv_uvproj_b_coord2.coord[1]=intersectderiv_uvproj_b.coord[1];

	
	project_to_uv_arrays(angle_of_incidence_factor,1.0f,
			     intersectpoint_uvcoords_coord2,&intersectderiv_uvproj_a_coord2,&intersectderiv_uvproj_b_coord2,
			     thisprojarray_info,
			     (snde_atomicimagedata *)((void *)(uvdata_angleofincidence_weighting+projectionbufoffset)),
			     thisprojarray_info.projection_strides,
			     NULL,
			     thisprojarray_info.weighting_strides,
			     (snde_atomicimagedata *)((void *)(uvdata_angleofincidence_weighting_validity+validitybufoffset)),
			     thisprojarray_info.validity_strides,
			     min_radius_uv_pixels,min_radius_src_pixels,bandwidth_fraction);
      }
      
    } else {
      // zdist == inf, i.e. no intersection. Other outputs should be NaN
      //if (imagedata_ray_intersect_deriv_a_cam_z)
      imagedata_intersectprops->ray_intersect_deriv_a_cam_z=NaNval;
      //if (imagedata_ray_intersect_deriv_b_cam_z)
      imagedata_intersectprops->ray_intersect_deriv_b_cam_z=NaNval;
      //if (imagedata_angleofincidence)
      imagedata_intersectprops->angleofincidence=NaNval;
      //if (imagedata_angleofincidence_weighting)
      imagedata_intersectprops->angleofincidence_weighting=NaNval;
      //if (imagedata_uvcoords) {
      imagedata_intersectprops->uvcoords.coord[0]=NaNval;
      imagedata_intersectprops->uvcoords.coord[1]=NaNval;
      //}
      //if (imagedata_uvfacenum)
      imagedata_intersectprops->uvfacenum=SNDE_INDEX_INVALID;
      
      //if (imagedata_uvcoords_deriv_a) {
      imagedata_intersectprops->uvcoords_deriv_a.coord[0] = NaNval;
      imagedata_intersectprops->uvcoords_deriv_a.coord[1] = NaNval;
      //}
      
      //if (imagedata_uvcoords_deriv_b) {
      imagedata_intersectprops->uvcoords_deriv_b.coord[0] = NaNval;
      imagedata_intersectprops->uvcoords_deriv_b.coord[1] = NaNval;
      //}
    }
    
  }
  
  
}

static RAYTRACE_INLINE
void raytrace_camera_evaluate_zbuffer(
				      snde_cmat23 cam_mtx,
				      snde_orientation3 orient_wrlcoords_over_camcoords,
				      snde_index src_na, snde_index src_nb, // assume rays start at origin of camcoords, projected by cam_mtx with pixel coordinates 0..na and 0..nb. Use Fortran ordering for image array
				      snde_index src_strides[2], // usually (1, src_na)
				      OCL_GLOBAL_ADDR snde_partinstance *instances,
				      snde_index num_instances,
				      OCL_GLOBAL_ADDR snde_part *parts, snde_index first_part,
				      OCL_GLOBAL_ADDR snde_topological *topos, snde_index first_topo,
				      OCL_GLOBAL_ADDR snde_triangle *triangles, snde_index first_triangle,
				      OCL_GLOBAL_ADDR snde_coord3 *trinormals,
				      OCL_GLOBAL_ADDR snde_cmat23 *inplanemats,
				      OCL_GLOBAL_ADDR snde_edge *edges, snde_index first_edge, 
				      OCL_GLOBAL_ADDR snde_coord3 *vertices, snde_index first_vertex,
				      OCL_GLOBAL_ADDR snde_box3 *boxes, snde_index first_box,
				      OCL_GLOBAL_ADDR snde_boxcoord3 *boxcoord,
				      OCL_GLOBAL_ADDR snde_index *boxpolys, snde_index first_boxpoly,
				      OCL_GLOBAL_ADDR snde_parameterization *uvs, snde_index first_uv,
				      OCL_GLOBAL_ADDR snde_topological *uv_topos, snde_index first_uv_topo,
				      OCL_GLOBAL_ADDR snde_triangle *uv_triangles, snde_index first_uv_tri,
				      OCL_GLOBAL_ADDR snde_cmat23 *inplane2uvcoords,
				      OCL_GLOBAL_ADDR snde_image *projectionarray_info, // projectionarray_info, indexed according to the firstuvimages of the partinstance, defines the layout of uvdata_angleofincidence_weighting and uvdata_angleofincidence_weighting_validity uv imagedata arrays
				      snde_coord min_radius_uv_pixels,snde_coord min_radius_src_pixels,snde_coord bandwidth_fraction,
				      snde_index frin_stacksize, // stack needed for a single box lookup 
				      OCL_GLOBAL_ADDR struct rayintersection_properties *imagedata_intersectprops,
				      OCL_GLOBAL_ADDR snde_imagedata *uvdata_angleofincidence_weighting,
				      OCL_GLOBAL_ADDR snde_imagedata *uvdata_angleofincidence_weighting_validity)
{
  int64_t bcnt;  // must be signed (e.g. not size_t) for MSVC compatibility
  snde_coord4 focalpointwrl;
  snde_orientation3 orient_camcoords_over_wrlcoords;

  snde_coord NaNval;
  NaNval=snde_infnan(0);
    

  raytrace_evaluate_focalpointwrl(orient_wrlcoords_over_camcoords,&focalpointwrl);
  raytrace_evaluate_orient_camcoords_over_wrlcoords(orient_wrlcoords_over_camcoords,&orient_camcoords_over_wrlcoords);
  
  
  
#ifdef RAYTRACE_USE_OPENMP
#pragma omp parallel default(shared) private(bcnt) // ,src_ny,numsurfaces)
#endif
  {
#ifdef RAYTRACE_USE_OPENMP
#pragma omp for
#endif
    for (bcnt=0;bcnt < src_nb;bcnt++) {
      
      snde_index acnt;

      snde_coord4 rayvecwrl; // rayvec in object coordinates (projective)


      int trace=FALSE;
      snde_index firstidx;
      //float64_t rayvecfactor;
      
      //bcnt=instancebcnt % src_nb; 
      //instancecnt=instancebcnt / src_nb; 

      snde_index *boxnum_stack = (snde_index *)malloc(frin_stacksize*sizeof(*boxnum_stack));

      
      for (acnt=0; acnt < src_na; acnt++) {
	// Go through each source image pixel,
	// project it through to surface. If it
	// intersects closer, mark the z-buffer
	// and surface ID. This will give us a map
	// of the source image for each pixel, which surface
	// it maps onto.
	
	// NOTE: These are recalculated later in project_image... formulas should be THE SAME!!!

	

	//camera_rayvec_wrl(cam_mtx,acnt,bcnt,orient_wrlcoords_over_camcoords,&rayvecwrl);


	raytrace_camera_evaluate_zdist(cam_mtx,
				       focalpointwrl,
				       orient_wrlcoords_over_camcoords,
				       orient_camcoords_over_wrlcoords,
				       acnt,bcnt, // index of the particular pixel ray we are interested in
				       instances,
				       num_instances,
				       parts, first_part,
				       topos, first_topo,
				       triangles, first_triangle,
				       trinormals,
				       inplanemats,
				       edges, first_edge,
				       vertices, first_vertex,
				       boxes, first_box,
				       boxcoord,
				       boxpolys, first_boxpoly,
				       uvs, first_uv,
				       uv_topos, first_uv_topo,
				       uv_triangles, first_uv_tri,
				       inplane2uvcoords,
				       projectionarray_info, // projectionarray_info, indexed according to the firstuvimages of the partinstance, defines the layout of uvdata_angleofincidence_weighting and uvdata_angleofincidence_weighting_validity uv imagedata arrays
				       min_radius_uv_pixels, min_radius_src_pixels, bandwidth_fraction,
				       frin_stacksize,
				       boxnum_stack,
				       &imagedata_intersectprops[acnt*src_strides[0] + bcnt*src_strides[1]], // JUST the structure for this pixel... we don't index it
				       uvdata_angleofincidence_weighting,
				       uvdata_angleofincidence_weighting_validity);
      }
      free(boxnum_stack);
    }
  }
}


#ifdef __OPENCL_VERSION__
// For an OpenCL kernel

__kernel void raytrace_camera_evaluate_zbuffer_opencl(
							   snde_cmat23 cam_mtx,
							   snde_orientation3 orient_wrlcoords_over_camcoords,
							   snde_orientation3 orient_camcoords_over_wrlcoords,
							   snde_index src_na, snde_index src_nb, // assume rays start at origin of camcoords, projected by cam_mtx with pixel coordinates 0..na and 0..nb. Use Fortran ordering for image array
							   OCL_GLOBAL_ADDR snde_partinstance *instances,
							   snde_index num_instances,
							   OCL_GLOBAL_ADDR snde_part *parts, snde_index first_part,
							   OCL_GLOBAL_ADDR snde_triangle *triangles, snde_index first_triangle,
							   OCL_GLOBAL_ADDR snde_cmat23 *inplanemats,
							   OCL_GLOBAL_ADDR snde_edge *edges, snde_index first_edge, 
							   OCL_GLOBAL_ADDR snde_coord3 *vertices, snde_index first_vertex,
							   OCL_GLOBAL_ADDR snde_box3 *boxes, snde_index first_box,
							   OCL_GLOBAL_ADDR snde_boxcoord3 *boxcoord, 
							   OCL_GLOBAL_ADDR snde_index *boxpolys, snde_index first_boxpoly,
							   OCL_GLOBAL_ADDR snde_parameterizations *uvs, snde_index first_uv,
							   OCL_GLOBAL_ADDR snde_cmat23 *inplane2uvcoords, snde_index first_uv_tri,
							   OCL_GLOBAL_ADDR snde_image *projectionarray_info, // projectionarray_info, indexed according to the firstuvimages of the partinstance, defines the layout of uvdata_angleofincidence_weighting and uvdata_angleofincidence_weighting_validity uv imagedata arrays
							   snde_index frin_stacksize,
							   OCL_LOCAL_ADDR *boxnum_stack, // size frin_stacksize*num_global_0*num_global_1?
							   OCL_GLOBAL_ADDR struct rayintersection_properties *imagedata_intersectprops,
							   OCL_GLOBAL_ADDR snde_imagedata *uvdata_angleofincidence_weighting,
							   OCL_GLOBAL_ADDR snde_imagedata *uvdata_angleofincidence_weighting_validity)
{

  snde_index aindex = get_global_id(0);
  snde_index bindex = get_global_id(1);
  
  raytrace_camera_evaluate_zdist(cam_mtx,
				 focalpointwrl;
				 orient_wrlcoords_over_camcoords,
				 orient_camcoords_over_wrlcoords,
				 aindex,bindex, // index of the particular pixel ray we are interested in
				 instances,
				 num_instances,
				 parts, first_part,
				 triangles, first_triangle,
				 inplanemats,
				 edges, first_edge, 
				 vertices, first_vertex,
				 boxes, first_box,
				 boxcoord,
				 boxpolys, first_boxpoly,
				 uvs, first_uv,
				 inplane2uvcoords, first_uv_tri,
				 projectionarray_info, // projectionarray_info, indexed according to the firstuvimages of the partinstance, defines the layout of uvdata_angleofincidence_weighting and uvdata_angleofincidence_weighting_validity uv imagedata arrays
				 frin_stacksize,
				 &boxnum_stack[frin_stacksize*xxxxxx],
				 &imagedata_intersectprops[aindex + bindex*src_na], // JUST the structure for this pixel... we don't index it
				 uvdata_angleofincidence_weighting,
				 uvdata_angleofincidence_weighting_validity);
  
}


#endif

#endif // SNDE_RAYTRACE_H


