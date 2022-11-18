/* implicit include of geometry_types.h */
/* implicit include of colormap.h */
/* implicit typedef of sc_inttype */

// scaled value of 0 -> 0
// scaled value of 1.0 -> 255.0



__kernel void scale_colormap(__global const sc_intype *inarray,
			     __global snde_rgba *outimage,
			     snde_index stride_u,snde_index stride_v,snde_index stride_w,
	             snde_index DisplayFrame,
			     snde_float32 offset,
			     uint8_t alpha,
			     uint32_t colormap_type,
			     snde_float32 intensityperunits)
{
  snde_index uidx=get_global_id(0);
  snde_index vidx=get_global_id(1);
  snde_rgba out;
  float scaledval=(inarray[stride_u*uidx + stride_v*vidx + stride_w*DisplayFrame]-offset)*intensityperunits;

  outimage[uidx + get_global_size(0)*vidx] = snde_colormap(colormap_type,scaledval,alpha);
}




__kernel void scale_pointcloud_colormap(__global const snde_coord3 *inarray,
					__global snde_float32 *outimage, // array of vec4's
					snde_index stride,
					snde_float32 offset,
					snde_float32 alpha,
					uint32_t colormap_type,
					snde_float32 intensityperunits)
{
  snde_index idx=get_global_id(0);
  float out[4];
  float scaledval=(inarray[stride*idx].coord[2]-offset)*intensityperunits;  // Currently hardwired to scale acoording to z component coord[2]
  
  
  snde_colormap_float(colormap_type,scaledval,alpha,out);
  outimage[4*idx + 0] = out[0];
  outimage[4*idx + 1] = out[1];
  outimage[4*idx + 2] = out[2];
  outimage[4*idx + 3] = out[3];
}

