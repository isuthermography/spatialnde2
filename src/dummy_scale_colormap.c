/* implicit include of geometry_types.h */
/* implicit include of colormap.h */
/* implicit typedef of sc_intype */

// scaled value of 0 -> 0
// scaled value of 1.0 -> 255.0

//  dummy version for inttype of snde_rgba

__kernel void scale_colormap(__global const sc_intype *inarray,
			     __global snde_rgba *outimage,
			     snde_index input_offset,
			     snde_index stride_x,snde_index stride_y,
			     float offset,
			     uint8_t alpha,
			     snde_index ColorMap,
			     float DivPerUnits)
{
  snde_index xidx=get_global_id(0);
  snde_index yidx=get_global_id(1);

  outimage[xidx + get_global_size(0)*yidx] = inarray[input_offset + stride_x*xidx + stride_y*yidx];
}

