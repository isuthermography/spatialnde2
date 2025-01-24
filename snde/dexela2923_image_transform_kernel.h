#ifndef SNDE_DEXELA2923_IMAGE_TRANSFORM_KERNEL_H
#define SNDE_DEXELA2923_IMAGE_TRANSFORM_KERNEL_H

#ifndef __OPENCL_VERSION__
#include <errno.h>
#include <stddef.h>
#include <math.h>
#include <stdio.h>

#include "snde/snde_types.h"
#endif


#ifdef _MSC_VER
#define DEXELA_INLINE  __inline
#else
#define DEXELA_INLINE  inline
#endif


static DEXELA_INLINE void dexela2923_image_transform_row_strippos(uint16_t *rawimage,uint16_t *result_rec,size_t row, size_t strippos)
{
  size_t srcoffset;
  int stripcnt;
  srcoffset=row*256*6*4 + strippos*6*4;
  for (stripcnt=0; stripcnt < 6; stripcnt++) {

    //# Lower left detector
    result_rec[(3887-row)*3072 + 1535 - stripcnt*256 - strippos ]=	rawimage[srcoffset+stripcnt*4+3];
	
    //# Lower right detector
    result_rec[(3887-row)*3072 + 3071 - stripcnt*256 - strippos ]=rawimage[srcoffset+stripcnt*4+2];
	
    //# Upper right detector
    result_rec[(row)*3072 + 1536 + stripcnt*256 + strippos ]=	rawimage[srcoffset+stripcnt*4+1];

    //# Upper left detector
    result_rec[(row)*3072 + stripcnt*256 + strippos ]=rawimage[srcoffset+stripcnt*4];
	



  }
  
}

static DEXELA_INLINE void dexela2923_image_transform_row_strippos_fliprotate(uint16_t *rawimage,uint16_t *result_rec,size_t row, size_t strippos)
{
  size_t srcoffset;
  int stripcnt;
  srcoffset=row*256*6*4 + strippos*6*4;
  

  for (stripcnt=0; stripcnt < 6; stripcnt++) {
    //# Lower left detector
    result_rec[(stripcnt*256+strippos)*3888 + (1944+row)]=rawimage[srcoffset];
    
    //# Lower right detector
    result_rec[(1536+stripcnt*256+strippos)*3888 + (1944+row)]=rawimage[srcoffset+1];


    //# Upper right detector
    result_rec[(3071-stripcnt*256-strippos)*3888 + (1943-row)]=rawimage[srcoffset+2];

    //# Upper left detector
    result_rec[(1535-stripcnt*256-strippos)*3888 + (1943-row)]=rawimage[srcoffset+3];


 } 
}

#ifdef __OPENCL_VERSION__

__kernel void dexela2923_image_transform_kernel(__global const uint16_t *rawimage,
						__global uint16_t *result_rec)
{
  size_t row=get_global_id(0);
  size_t strippos = get_global_id(1);
  dexela2923_image_transform_row_strippos(rawimage,result_rec, row, strippos);
}


#endif

#endif 
