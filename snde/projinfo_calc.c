#ifndef __OPENCL_VERSION__
#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/vecops.h"
#include "snde/geometry_ops.h"

#include "snde/projinfo_calc.h"
#endif

/* implicit include of geometry_types.h */
/* implicit include of vecops.h */

#define SNDE_PROJINFO_CALC_DEBUG_TRIANGLE (SNDE_INDEX_INVALID) // can replace with index of triangle we want to print

void snde_projinfo_calc_one(// OCL_GLOBAL_ADDR const struct snde_part *part,
			    // OCL_GLOBAL_ADDR const struct snde_parameterization *param,
			    OCL_GLOBAL_ADDR const snde_triangle *part_triangles,
			    OCL_GLOBAL_ADDR const snde_edge *part_edges,
			    OCL_GLOBAL_ADDR const snde_coord3 *part_vertices,
			    OCL_GLOBAL_ADDR const snde_cmat23 *part_inplanemats,
			    OCL_GLOBAL_ADDR const snde_triangle *param_triangles, // parameterization_ i.e. uv
			    OCL_GLOBAL_ADDR const snde_edge *param_edges,
			    OCL_GLOBAL_ADDR const snde_coord2 *param_vertices,
			    OCL_GLOBAL_ADDR snde_cmat23 *inplane2uvcoords,
			    OCL_GLOBAL_ADDR snde_cmat23 *uvcoords2inplane,
			    snde_index trianglenum)
{

  snde_coord3 centroid_3d;
  snde_coord3 triverts_3d[3];
  snde_coord3 coordvals[3];
  snde_coord2 coordvals2d[3];  
  snde_coord2 uvcoordvals[3];
  snde_coord TexXformMtx[36]={0.0f}; // initialze array to 0.0
  snde_coord TexCoordVec[6];
  size_t TexXformMtx_pivots[6];
  size_t Amat_pivots[2];
  snde_coord Amat[4];
  snde_coord Amatinv[4];
  size_t vertnum;
  
  // inplane2uvcoords is a transform, the first two rows of A:
  //    [ A11 A12 A13 ][ x ] = [ u ]
  //    [ A21 A22 A23 ][ y ] = [ v ]
  //    [ 0    0  1   ][ 1 ] = [ 1 ]
  //    or
  //    [ x1  y1  1  0  0  0 ] [ A11 ] = [ u1 ]
  //    [ 0  0  0  x1  y1  1 ] [ A12 ] = [ v1 ]
  //    [ x2  y2  1  0  0  0 ] [ A13 ] = [ u2 ]
  //    [ 0  0  0  x2  y2  1 ] [ A21 ] = [ v2 ]
  //    [ x3  y3  1  0  0  0 ] [ A22 ] = [ u3 ]
  //    [ 0  0  0  x3  y3  1 ] [ A23 ] = [ v3 ]

  // where (x,y) are in the orthogonal 2D frame that is the output of inplanemat
  // for this triangle, and (u,v) are parameterization coordinates
  // basically, A11, A12, A21, A22 are components of a transformation matrix,
  // and A13, A23 are the offset
  //
  // uvcoords2inplane is the inverse: 
  //    [ A11 A12 A13 ][ x ] = [ u ]
  //    [ A21 A22 A23 ][ y ] = [ v ]
  //                   [ 1 ]
  //    A11x + A12y + A13 = u 
  //    A21x + A22y + A23 = v 
  //    [ A11 A12 ][ x ] = [ u-A13 ]
  //    [ A21 A22 ][ y ] = [ v-A23 ]
  //    [ x ] = [ A11 A12 ]^-1  [ u-A13 ]
  //    [ y ] = [ A21 A22 ]     [ v-A23 ]
  //    [ x ] = [ A11 A12 ]^-1  [ u ]   - [ A11 A12 ]^-1 [ A13 ]
  //    [ y ] = [ A21 A22 ]     [ v ]   - [ A21 A22 ]    [ A23 ]
  // For the moment, this calculates a normal per triangle and stores it
  // for all vertices of the triangle, and for the triangle as a whole;
  

  /* traverse edges of this triangle and extract vertex coordinates -> triverts*/
  get_we_triverts_3d(part_triangles,trianglenum,part_edges,part_vertices,triverts_3d);
  get_we_triverts_2d(param_triangles,trianglenum,param_edges,param_vertices,uvcoordvals);

  /* got vertex coordinates in triverts */
  tricentroid3(triverts_3d, &centroid_3d);

  if (trianglenum==SNDE_PROJINFO_CALC_DEBUG_TRIANGLE) {
    printf("triverts_3d[0]=(%f,%f,%f)\n",triverts_3d[0].coord[0],triverts_3d[0].coord[1],triverts_3d[0].coord[2]);
    printf("triverts_3d[1]=(%f,%f,%f)\n",triverts_3d[1].coord[0],triverts_3d[1].coord[1],triverts_3d[1].coord[2]);
    printf("triverts_3[2]=(%f,%f,%f)\n",triverts_3d[2].coord[0],triverts_3d[2].coord[1],triverts_3d[2].coord[2]);

    
    printf("centroid_3d=(%f,%f,%f)\n",centroid_3d.coord[0],centroid_3d.coord[1],centroid_3d.coord[2]);
  }

  // coordvals are 3D coordinates relative to 3D centroid
  subcoordcoord3(triverts_3d[0],centroid_3d,&coordvals[0]);
  subcoordcoord3(triverts_3d[1],centroid_3d,&coordvals[1]);
  subcoordcoord3(triverts_3d[2],centroid_3d,&coordvals[2]);

  if (trianglenum==SNDE_PROJINFO_CALC_DEBUG_TRIANGLE) {
    
    printf("coordvals[0]=(%f,%f,%f)\n",coordvals[0].coord[0],coordvals[0].coord[1],coordvals[0].coord[2]);
    printf("coordvals[1]=(%f,%f,%f)\n",coordvals[1].coord[0],coordvals[1].coord[1],coordvals[1].coord[2]);
    printf("coordvals[2]=(%f,%f,%f)\n",coordvals[2].coord[0],coordvals[2].coord[1],coordvals[2].coord[2]);
  }

  
  // coordvals2d are 2D coordinates in the orthogonal frame for the triangle
  multcmat23coord(part_inplanemats[trianglenum],coordvals[0],&coordvals2d[0]);
  multcmat23coord(part_inplanemats[trianglenum],coordvals[1],&coordvals2d[1]);
  multcmat23coord(part_inplanemats[trianglenum],coordvals[2],&coordvals2d[2]);

  if (trianglenum==SNDE_PROJINFO_CALC_DEBUG_TRIANGLE) {
    printf("coordvals2d[0]=(%f,%f)\n",coordvals2d[0].coord[0],coordvals2d[0].coord[1]);
    printf("coordvals2d[1]=(%f,%f)\n",coordvals2d[1].coord[0],coordvals2d[1].coord[1]);
    printf("coordvals2d[2]=(%f,%f)\n",coordvals2d[2].coord[0],coordvals2d[2].coord[1]);
    printf("uvcoordvals[2]=(%f,%f)\n",uvcoordvals[2].coord[0],uvcoordvals[2].coord[1]);
  }

  // also have uvcoordvals, from above.
  // Now we just have to determine the transform.
  // build a fortran-order matrix that we will invert
  // as described in the comments above.

  // NOTE: TexXformMtx stored fotran order 
  for (vertnum=0;vertnum < 3; vertnum++) {
    TexXformMtx[ vertnum*2 + 0*6 ] = coordvals2d[vertnum].coord[0]; // x1, etc
    TexXformMtx[ vertnum*2 + 1*6 ] = coordvals2d[vertnum].coord[1]; // y1
    TexXformMtx[ vertnum*2 + 2*6 ] = 1.0f;
    TexXformMtx[ vertnum*2+1 + 3*6 ] = coordvals2d[vertnum].coord[0]; // x1
    TexXformMtx[ vertnum*2+1 + 4*6 ] = coordvals2d[vertnum].coord[1]; // y1
    TexXformMtx[ vertnum*2+1 + 5*6 ] = 1.0f;

    TexCoordVec[ vertnum*2 ] = uvcoordvals[vertnum].coord[0]; // u1, etc.
    TexCoordVec[ vertnum*2+1 ] = uvcoordvals[vertnum].coord[1]; // v1, etc.
  }

  if (trianglenum==SNDE_PROJINFO_CALC_DEBUG_TRIANGLE) {
    printf("A=["); // actually print out transpose
    for (unsigned row=0;row < 6;row++) {
      for (unsigned col=0;col < 6; col++) {	
	printf(" %8.8g",TexXformMtx[col+row*6]); // column and row of transpose
	if (col != 5) {
	  printf(",");
	} else {
	  printf(";");
	}
      }
    }
    printf("]';\n");
    printf("b=[");
    for (unsigned row=0;row < 6; row++) {	
      printf(" %8.8g",TexCoordVec[row]);
      if (row != 5) {
	printf(",");
      } 
    }
    printf("]';\n");

  }
  // Solve TexXformMtx*x = TexCoordVec... x will overwrite TexCoordVec
  fmatrixsolve(TexXformMtx,TexCoordVec,6,1,TexXformMtx_pivots,trianglenum==SNDE_PROJINFO_CALC_DEBUG_TRIANGLE);

  inplane2uvcoords[trianglenum].row[0].coord[0]=TexCoordVec[0]; // A11
  inplane2uvcoords[trianglenum].row[0].coord[1]=TexCoordVec[1]; // A12
  inplane2uvcoords[trianglenum].row[0].coord[2]=TexCoordVec[2]; // A13
  inplane2uvcoords[trianglenum].row[1].coord[0]=TexCoordVec[3]; // A21
  inplane2uvcoords[trianglenum].row[1].coord[1]=TexCoordVec[4]; // A22
  inplane2uvcoords[trianglenum].row[1].coord[2]=TexCoordVec[5]; // A23

  if (trianglenum==SNDE_PROJINFO_CALC_DEBUG_TRIANGLE) {
    printf("A11 = %f; A12=%f; A13=%f\nA21=%f; A22=%f; A23=%f\n",TexCoordVec[0],TexCoordVec[1],TexCoordVec[2],TexCoordVec[3],TexCoordVec[4],TexCoordVec[5]);
  }
  

  // need to evaluate uvcoords2inplane, which is the inverse of
  // inplane2uvcoords. From above:
  //    [ x ] = [ A11 A12 ]^-1  [ u ]   - [ A11 A12 ]^-1 [ A13 ]
  //    [ y ] = [ A21 A22 ]     [ v ]   - [ A21 A22 ]    [ A23 ]
  // NOTE: Amat and Amatinv stored fortran order
  Amat[0]=TexCoordVec[0]; // A11
  Amat[1]=TexCoordVec[3]; // A21
  Amat[2]=TexCoordVec[1]; // A12
  Amat[3]=TexCoordVec[4]; // A22

  // solve for inverse by solving for [ 1 0 ; 0 1 ]
  Amatinv[0]=1.0f;
  Amatinv[1]=0.0f;
  Amatinv[2]=0.0f;
  Amatinv[3]=1.0f;
  fmatrixsolve(Amat,Amatinv,2,2,Amat_pivots,trianglenum==SNDE_PROJINFO_CALC_DEBUG_TRIANGLE);

  // extract 2x2 matrix on left of uvcoords2inplane
  uvcoords2inplane[trianglenum].row[0].coord[0]=Amatinv[0]; //Amatinv11
  uvcoords2inplane[trianglenum].row[0].coord[1]=Amatinv[2]; //Amatinv12 -- remember Amatinv stored Fortran style
  uvcoords2inplane[trianglenum].row[1].coord[0]=Amatinv[1]; //Amatinv21 -- remember Amatinv stored Fortran style
  uvcoords2inplane[trianglenum].row[1].coord[1]=Amatinv[3]; //Amatinv22

  // offset vector on right of uvcoords2inplane is
  //   - [ A11 A12 ]^-1 [ A13 ]
  //   - [ A21 A22 ]    [ A23 ]
  //   = - [ Amatinv[0] Amatinv[2] ] [ A13 ]
  //   = - [ Amatinv[1] Amatinv[3] ] [ A23 ]
  uvcoords2inplane[trianglenum].row[0].coord[2] = -Amatinv[0]*TexCoordVec[2] - Amatinv[2]*TexCoordVec[5];
  uvcoords2inplane[trianglenum].row[1].coord[2] = -Amatinv[1]*TexCoordVec[2] - Amatinv[3]*TexCoordVec[5];
  
}


#ifdef __OPENCL_VERSION__

__kernel void snde_projinfo_calc(// OCL_GLOBAL_ADDR const struct snde_part *part,
				 // OCL_GLOBAL_ADDR const struct snde_parameterization *param,
				 OCL_GLOBAL_ADDR const snde_triangle *part_triangles,
				 OCL_GLOBAL_ADDR const snde_edge *part_edges,
				 OCL_GLOBAL_ADDR const snde_coord3 *part_vertices,
				 OCL_GLOBAL_ADDR const snde_cmat23 *part_inplanemats,
				 OCL_GLOBAL_ADDR const snde_triangle *param_triangles,
				 OCL_GLOBAL_ADDR const snde_edge *param_edges,
				 OCL_GLOBAL_ADDR const snde_coord2 *param_vertices,
				 OCL_GLOBAL_ADDR snde_cmat23 *inplane2uvcoords,
				 OCL_GLOBAL_ADDR snde_cmat23 *uvcoords2inplane)
{
  snde_index trianglenum=get_global_id(0);

  snde_projinfo_calc_one(part_triangles,
			 part_edges,
			 part_vertices,
			 part_inplanemats,
			 param_triangles,
			 param_edges,
			 param_vertices,
			 inplane2uvcoords,
			 uvcoords2inplane,
			 trianglenum);
  

}

#endif // __OPENCL_VERSION__
