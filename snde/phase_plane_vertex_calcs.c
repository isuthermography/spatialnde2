#ifndef PPVAO_DECL
#define PPVAO_DECL
#endif


PPVAO_DECL
void phase_plane_vertices_alphas_one(OCL_GLOBAL_ADDR ppvao_intype *complex_inputs,
				     OCL_GLOBAL_ADDR snde_coord3 *tri_vertices,
				     OCL_GLOBAL_ADDR snde_float32 *trivert_colors,
				     ppvao_intype previous_coords,
				     snde_index pos, // within these inputs and these outputs,
				     snde_index totalpos, // for historical_fade, with 0 representing the previous_coords for the first call
				     snde_index totallen, // for historical_fade -- generally the number of end points, or 1 more than the total number of calls to this function
				     snde_float32 linewidth_horiz,
				     snde_float32 linewidth_vert,
				     snde_float32 R,
				     snde_float32 G,
				     snde_float32 B,
				     snde_float32 A,
				     snde_bool historical_fade)
{

  ppvao_intype prior_coords;

  if (pos==0) {
    prior_coords.real=previous_coords.real;
    prior_coords.imag=previous_coords.imag;
  } else {
    prior_coords.real = complex_inputs[pos-1].real;
    prior_coords.imag = complex_inputs[pos-1].imag;
  }

  // draw line from prior_coords to complex_inputs[pos] via 2 CCW triangles

  // x to the right, y up; z is pointing at us.
  // want to select width_direction such that length_direction x width_direction = z ; i.e. width is like y. 
  // or equivalently z x length_direction = width_direction

  //                    |   i     j     k   |
  // width_direction =  |   0     0     1   |
  //                    |  lx0   ly0    0   |
  // (where lx0, ly0 presumed normalized)
  // Therefore width_direction = -i*l0y + j*l0x

  ppvao_intype l,l0;
  l.real = complex_inputs[pos].real-prior_coords.real;
  l.imag = complex_inputs[pos].imag-prior_coords.imag;

  ppvao_intype scale={0.f,0.f};

  scale.real = 1.0f/sqrt(l.real*l.real + l.imag*l.imag);
;

  l0.real=l.real*scale.real;
  l0.imag=l.imag*scale.real;

  ppvao_intype width_direction;
  width_direction.real = -l0.imag;
  width_direction.imag = l0.real;
  //#ifdef __OPENCL_VERSION__
  //if (isnan(width_direction.real)) {
  //  printf("phase_plane_vertices_alphas_one() got NaN width_direction.real\n");
  // }
  //#else
  //assert(!isnan(width_direction.real));
  //#endif

  if (isnan(width_direction.real)) {
    // This means either the incoming data is invalid, or
    // the two endpoints are the same. Either way we don't draw anything
    snde_float32 nanval = snde_infnan(0);
    tri_vertices[pos*6].coord[0] = nanval;
    tri_vertices[pos*6].coord[1] = nanval;
    tri_vertices[pos*6].coord[2] = nanval;

    tri_vertices[pos*6+1].coord[0] = nanval;
    tri_vertices[pos*6+1].coord[1] = nanval;
    tri_vertices[pos*6+1].coord[2] = nanval;

    tri_vertices[pos*6+2].coord[0] = nanval;
    tri_vertices[pos*6+2].coord[1] = nanval;
    tri_vertices[pos*6+2].coord[2] = nanval;

    tri_vertices[pos*6+3].coord[0] = nanval;
    tri_vertices[pos*6+3].coord[1] = nanval;
    tri_vertices[pos*6+3].coord[2] = nanval;

    tri_vertices[pos*6+4].coord[0] = nanval;
    tri_vertices[pos*6+4].coord[1] = nanval;
    tri_vertices[pos*6+4].coord[2] = nanval;
    
    tri_vertices[pos*6+5].coord[0] = nanval;
    tri_vertices[pos*6+5].coord[1] = nanval;
    tri_vertices[pos*6+5].coord[2] = nanval;


    trivert_colors[pos*6*4 + 0*4 + 0] = 0.f;
    trivert_colors[pos*6*4 + 0*4 + 1] = 0.f; 
    trivert_colors[pos*6*4 + 0*4 + 2] = 0.f;
    trivert_colors[pos*6*4 + 0*4 + 3] = 0.f; 

    trivert_colors[pos*6*4 + 1*4 + 0] = 0.f;
    trivert_colors[pos*6*4 + 1*4 + 1] = 0.f; 
    trivert_colors[pos*6*4 + 1*4 + 2] = 0.f;
    trivert_colors[pos*6*4 + 1*4 + 3] = 0.f; 

    trivert_colors[pos*6*4 + 2*4 + 0] = 0.f;
    trivert_colors[pos*6*4 + 2*4 + 1] = 0.f; 
    trivert_colors[pos*6*4 + 2*4 + 2] = 0.f;
    trivert_colors[pos*6*4 + 2*4 + 3] = 0.f; 

    trivert_colors[pos*6*4 + 3*4 + 0] = 0.f;
    trivert_colors[pos*6*4 + 3*4 + 1] = 0.f; 
    trivert_colors[pos*6*4 + 3*4 + 2] = 0.f;
    trivert_colors[pos*6*4 + 3*4 + 3] = 0.f; 

    trivert_colors[pos*6*4 + 4*4 + 0] = 0.f;
    trivert_colors[pos*6*4 + 4*4 + 1] = 0.f; 
    trivert_colors[pos*6*4 + 4*4 + 2] = 0.f;
    trivert_colors[pos*6*4 + 4*4 + 3] = 0.f; 

    trivert_colors[pos*6*4 + 5*4 + 0] = 0.f;
    trivert_colors[pos*6*4 + 5*4 + 1] = 0.f; 
    trivert_colors[pos*6*4 + 5*4 + 2] = 0.f;
    trivert_colors[pos*6*4 + 5*4 + 3] = 0.f; 

    return;
  }
  //printf("ppvao: width_direction.real=%f;linewidth_horiz=%f\n",width_direction.real,linewidth_horiz);
  //printf("ppvao: tvout.coord[0]=%f\n",prior_coords.real - linewidth_horiz*width_direction.real/2.0);

  //printf("ppvao: totalpos=%u; totallen=%u\n",(unsigned)totalpos,(unsigned)totallen);
  tri_vertices[pos*6].coord[0] = prior_coords.real - linewidth_horiz*width_direction.real/2.0;
  tri_vertices[pos*6].coord[1] = prior_coords.imag - linewidth_vert*width_direction.imag/2.0;
  tri_vertices[pos*6].coord[2] = 0.0f;

  tri_vertices[pos*6+1].coord[0] = complex_inputs[pos].real - linewidth_horiz*width_direction.real/2.0;
  tri_vertices[pos*6+1].coord[1] = complex_inputs[pos].imag - linewidth_vert*width_direction.imag/2.0;
  tri_vertices[pos*6+1].coord[2] = 0.0f;

  tri_vertices[pos*6+2].coord[0] = prior_coords.real + linewidth_horiz*width_direction.real/2.0;
  tri_vertices[pos*6+2].coord[1] = prior_coords.imag + linewidth_vert*width_direction.imag/2.0;
  tri_vertices[pos*6+2].coord[2] = 0.0f;

  tri_vertices[pos*6+3].coord[0] = prior_coords.real + linewidth_horiz*width_direction.real/2.0;
  tri_vertices[pos*6+3].coord[1] = prior_coords.imag + linewidth_vert*width_direction.imag/2.0;
  tri_vertices[pos*6+3].coord[2] = 0.0f;

  tri_vertices[pos*6+4].coord[0] = complex_inputs[pos].real - linewidth_horiz*width_direction.real/2.0;
  tri_vertices[pos*6+4].coord[1] = complex_inputs[pos].imag - linewidth_vert*width_direction.imag/2.0;
  tri_vertices[pos*6+4].coord[2] = 0.0f;

  tri_vertices[pos*6+5].coord[0] = complex_inputs[pos].real + linewidth_horiz*width_direction.real/2.0;
  tri_vertices[pos*6+5].coord[1] = complex_inputs[pos].imag + linewidth_vert*width_direction.imag/2.0;
  tri_vertices[pos*6+5].coord[2] = 0.0f;

  snde_float32 historical_fade_prior_coords_value=1.0f;
  snde_float32 historical_fade_current_value=1.0f;

  if (historical_fade) {
    // subtract 25 from totallen because it is in terms of our
    // output position and represents the start, whereas we
    // really want it to represent the end, and we have
    // 24 entries. so end-1-24 = end-25
    historical_fade_prior_coords_value = (totalpos*1.0f-25.0f)/(totallen-25.0f);
    historical_fade_current_value = totalpos*1.0f/(totallen-25.0f);

    if (historical_fade_prior_coords_value < 0.0) {
      historical_fade_prior_coords_value=0.0;
    }
    if (historical_fade_prior_coords_value > 1.0) {
      historical_fade_prior_coords_value=1.0;
    }

    if (historical_fade_current_value < 0.0) {
      historical_fade_current_value=0.0;
    }
    if (historical_fade_current_value > 1.0) {
      historical_fade_current_value=1.0;
    }

  }
  

  trivert_colors[pos*6*4 + 0*4 + 0] = R;
  trivert_colors[pos*6*4 + 0*4 + 1] = G; 
  trivert_colors[pos*6*4 + 0*4 + 2] = B;
  trivert_colors[pos*6*4 + 0*4 + 3] = A*historical_fade_prior_coords_value; 

  trivert_colors[pos*6*4 + 1*4 + 0] = R;
  trivert_colors[pos*6*4 + 1*4 + 1] = G; 
  trivert_colors[pos*6*4 + 1*4 + 2] = B;
  trivert_colors[pos*6*4 + 1*4 + 3] = A*historical_fade_current_value; 

  trivert_colors[pos*6*4 + 2*4 + 0] = R;
  trivert_colors[pos*6*4 + 2*4 + 1] = G; 
  trivert_colors[pos*6*4 + 2*4 + 2] = B;
  trivert_colors[pos*6*4 + 2*4 + 3] = A*historical_fade_prior_coords_value; 

  trivert_colors[pos*6*4 + 3*4 + 0] = R;
  trivert_colors[pos*6*4 + 3*4 + 1] = G; 
  trivert_colors[pos*6*4 + 3*4 + 2] = B;
  trivert_colors[pos*6*4 + 3*4 + 3] = A*historical_fade_prior_coords_value; 

  trivert_colors[pos*6*4 + 4*4 + 0] = R;
  trivert_colors[pos*6*4 + 4*4 + 1] = G; 
  trivert_colors[pos*6*4 + 4*4 + 2] = B;
  trivert_colors[pos*6*4 + 4*4 + 3] = A*historical_fade_current_value; 

  trivert_colors[pos*6*4 + 5*4 + 0] = R;
  trivert_colors[pos*6*4 + 5*4 + 1] = G; 
  trivert_colors[pos*6*4 + 5*4 + 2] = B;
  trivert_colors[pos*6*4 + 5*4 + 3] = A*historical_fade_current_value; 

}


#ifdef __OPENCL_VERSION__

__kernel void phase_plane_vertices_alphas(OCL_GLOBAL_ADDR ppvao_intype *complex_inputs,
					  OCL_GLOBAL_ADDR snde_coord3 *tri_vertices,
					  OCL_GLOBAL_ADDR snde_float32 *trivert_colors,
					  ppvao_intype previous_coords,
					  snde_index totalpos, // for historical_fade, with 0 representing the previous_coords for the first call (whih we would never supply)
					  snde_index totallen, // for historical_fade
					  snde_float32 linewidth_horiz,
					  snde_float32 linewidth_vert,
					  snde_float32 R,
					  snde_float32 G,
					  snde_float32 B,
					  snde_float32 A,
					  snde_bool historical_fade)
{
  snde_index pos = get_global_id(0);
  
  phase_plane_vertices_alphas_one(complex_inputs,
				  tri_vertices,
				  trivert_colors,
				  previous_coords,
				  pos,
				  totalpos+pos,
				  totallen,
				  linewidth_horiz,
				  linewidth_vert,
				  R,
				  G,
				  B,
				  A,
				  historical_fade);
}


#endif // __OPENCL_VERSION__
