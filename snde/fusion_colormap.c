
#ifndef __OPENCL_VERSION__

#include <math.h>

#include <snde/snde_types.h>
#include <snde/geometry_types.h>
#include "snde/colormap.h"
#include "snde/fusion_colormap.h" // defines fc_intype as float32

#endif // !__OPENCL_VERSION__


/* implicit include of geometry_types.h */
/* implicit include of colormap.h */
/* implicit typedef of sc_inttype */

// scaled value of 0 -> 0
// scaled value of 1.0 -> 255.0


snde_rgba snde_fusion_colormap_real(const fc_real_intype inval,
				    const snde_float32 total,
				    uint32_t colormap_type,
				    snde_float32 offset,
				    snde_float32 unitsperintensity,
				    snde_float32 maxtotal,
				    uint8_t alpha)
{
  snde_float32 val = (float)((inval/total-offset)/unitsperintensity);
  
  if (isnan(val)) {
    val=0.0f;
  }
  
  snde_rgba res = snde_colormap(colormap_type,val,255);
  
  // override blue compoment based on ratio to maxtotal
  snde_float32 total_ratio = 256*total/maxtotal;
  if (total_ratio > 255.0f) {
    total_ratio=255.0f;
  }
  res.b = (uint8_t)total_ratio;
  
  return res; 
  
  
}

snde_rgba snde_fusion_colormap_complex(const fc_complex_intype inval,
				       const snde_float32 total,
				       uint32_t colormap_type,
				       snde_float32 offset,
				       snde_float32 unitsperintensity,
				       snde_float32 maxtotal,
				       uint8_t alpha)
{

        
  snde_float32 realval = (float)((inval.real/total-offset)/unitsperintensity);
  snde_float32 imagval = (float)((inval.imag/total-offset)/unitsperintensity);
  snde_rgba res;
  
  if (isnan(realval)) {
    realval=0.0f;
  }
  
  if (isnan(imagval)) {
    imagval=0.0f;
  }
  
  snde_float32 scaled_realval = (realval+0.5f)*255.0f;
  if (scaled_realval > 255.0f) {
    scaled_realval = 255.0f;
  } else if (scaled_realval < 0.0f) {
    scaled_realval=0.0f;
  }
  
  snde_float32 scaled_imagval = (imagval+0.5f)*255.0f;
  if (scaled_imagval > 255.0f) {
    scaled_imagval = 255.0f;
  } else if (scaled_imagval < 0.0f) {
    scaled_imagval=0.0f;
  }
  
  snde_float32 total_ratio = 256*total/maxtotal;
  if (total_ratio > 255.0f) {
    total_ratio=255.0f;
  }
  
  res.r = (uint8_t)scaled_realval;
  res.g = (uint8_t)scaled_imagval;
  res.b = (uint8_t)total_ratio;
  res.a = alpha;
  
  return res;
  
  
}



#ifdef __OPENCL_VERSION__

__kernel void fusion_colormap_kern(__global const fc_intype *inarray,
				   __global const snde_float32 *total,
				   __global snde_rgba *outimage,
				   snde_index stride_u,snde_index stride_v,
				   uint32_t colormap_type,
				   snde_float32 offset,
				   snde_float32 unitsperintensity,
				   snde_float32 maxtotal,
				   uint8_t alpha)
{
  snde_index uidx=get_global_id(0);
  snde_index vidx=get_global_id(1);

  outimage[uidx + get_global_size(0)*vidx] = snde_fusion_colormap(inarray[stride_u*uidx + stride_v*vidx],
								  total[stride_u*uidx + stride_v*vidx],
								  colormap_type,
								  offset,
								  unitsperintensity,
								  maxtotal,
								  alpha);
  
}

#endif // __OPENCL_VERSION__



