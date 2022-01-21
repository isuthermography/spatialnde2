#ifndef __OPENCL_VERSION__
#include "snde/snde_types.h"
#include "snde/geometry_types.h"

#include "snde/rec_display_texvertexarray.h"
#endif
/* implicit include of geometry_types.h */

void snde_rec_display_texvertexarray_onetri(OCL_GLOBAL_ADDR const struct snde_parameterization *uv,
					    OCL_GLOBAL_ADDR const snde_triangle *uv_triangles,
					    OCL_GLOBAL_ADDR const snde_edge *uv_edges,
					    OCL_GLOBAL_ADDR const snde_coord2 *uv_vertices,
					    OCL_GLOBAL_ADDR snde_rendercoord *texvertex_arrays,
					    snde_index trianglenum)
// This function extract parameterization (texture) vertices from the winged edge data structure
//  defined by uv and the triangles, edges, and vertices arrays,
// and emits them into the texvertex_arrays() location.
// The triangle index is specified by the global_id
{


  snde_index thisedge;
  snde_index nextedge;
  //snde_coord3 triverts[3];
  snde_coord2 thisvert;
  int edgecnt;
  snde_index vapos=trianglenum*6;

  if (uv_triangles[trianglenum].face==SNDE_INDEX_INVALID) {

    // invalid triangle... set vertices to NaN
    
    uint8_t NaNconstLE[4]={ 0x00,0x00,0xc0,0x7f };
    uint8_t NaNconstBE[4]={ 0x7f,0xc0,0x00,0x00 };
    float NaN;
    
    if ((*((uint32_t*)NaNconstBE) & 0xff) == 0x00) {
      // big endian
      NaN = (float)*((float *)NaNconstBE);
    } else {
      // little endian
      NaN = (float)*((float *)NaNconstLE);
    }

    for (edgecnt=0;edgecnt < 3;edgecnt++) {
      texvertex_arrays[vapos+2*edgecnt]=NaN;
      texvertex_arrays[vapos+2*edgecnt+1]=NaN;
      //texvertex_arrays[vapos+3*edgecnt+2]=NaN;
    }
    return;
  }
  
  //printf("tri_offs=%d,trianglenum=%d,meshedpartnum=%d\n",(int)tri_offs,(int)trianglenum,(int)meshedpartnum);
  //printf("edge_offs=%d, vert_offs=%d\n",(int)edge_offs,(int)vert_offs);
  thisedge=uv_triangles[trianglenum].edges[0];
  
  /* traverse edges of this triangle and extract vertex coordinates -> triverts*/
  edgecnt=0;
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
      //triverts[edgecnt]=part_vertices[part_edges[thisedge].vertex[0]];
      thisvert=uv_vertices[uv_edges[thisedge].vertex[0]];
    } else {
      //printf("vertex_index=%d.1\n",(int)(vert_offs+edges[edge_offs+thisedge].vertex[1]));
      //triverts[edgecnt]=part_vertices[part_edges[thisedge].vertex[1]];
      thisvert=uv_vertices[uv_edges[thisedge].vertex[1]];
    }
    texvertex_arrays[vapos+2*edgecnt]=thisvert.coord[0];
    texvertex_arrays[vapos+2*edgecnt+1]=thisvert.coord[1];
    //texvertex_arrays[vapos+3*edgecnt+2]=thisvert.coord[2];


    //if (trianglenum==47759) {
    //  printf("triangle 47759 vertex: (%lf,%lf)\n",(double)thisvert.coord[0],(double)thisvert.coord[1]); // ,(double)thisvert.coord[2]);
    //}
    
    thisedge=nextedge;
    edgecnt++;
  }
  
  
}


#ifdef __OPENCL_VERSION__

__kernel void rec_display_texvertexarray(__global const struct snde_parameterization *uv,
				 __global const snde_triangle *uv_triangles,
				 __global const snde_edge *uv_edges,
				 __global const snde_coord2 *uv_vertices,
				 __global snde_rendercoord *texvertex_arrays)
// This function extract parameterization (texture) vertices from the winged edge data structure
//  defined by uv and the triangles, edges, and vertices arrays,
// and emits them into the texvertex_arrays() location.
// The triangle index is specified by the global_id
{

  snde_index trianglenum=get_global_id(0);
  snde_rec_display_texvertexarray_onetri(uv,
					 uv_triangles,
					 uv_edges,
					 uv_vertices,
					 texvertex_arrays,
					 trianglenum);

}  
#endif // __OPENCL_VERSION__
