#ifndef WAVEFORM_DECL
#define WAVEFORM_DECL
#endif


WAVEFORM_DECL
void waveform_vertices_alphas_one(OCL_GLOBAL_ADDR waveform_intype *inputs,
				     OCL_GLOBAL_ADDR snde_coord3 *tri_vertices,
				     OCL_GLOBAL_ADDR snde_float32 *trivert_colors,
				     snde_index pos, // within these inputs and these outputs,
					 double inival,
					 double step,
				     snde_float32 linewidth_horiz,
				     snde_float32 linewidth_vert,
				     snde_float32 R,
				     snde_float32 G,
				     snde_float32 B,
				     snde_float32 A)
{

  waveform_intype priorx, priory, curx, cury;

  priorx = step * (waveform_intype)(pos - 1) + inival;
  priory = inputs[pos - 1];
  curx = step * (waveform_intype)(pos) + inival;
  cury = inputs[pos];

  // draw line from prior_coords to complex_inputs[pos] via 2 CCW triangles

  // x to the right, y up; z is pointing at us.
  // want to select width_direction such that length_direction x width_direction = z ; i.e. width is like y. 
  // or equivalently z x length_direction = width_direction

  //                    |   i     j     k   |
  // width_direction =  |   0     0     1   |
  //                    |  lx0   ly0    0   |
  // (where lx0, ly0 presumed normalized)
  // Therefore width_direction = -i*l0y + j*l0x

  waveform_intype x, y, x0, y0;
  x = inputs[pos]-priorx;
  y = inputs[pos]-priory;

  waveform_intype scale;
  scale = 1.0f/sqrt(x*x + y*y); 

  x0=x*scale;
  y0=y*scale;

  waveform_intype width_directionx, width_directiony;
  width_directionx = -y0;
  width_directiony = x0;
  assert(!isnan(width_directionx));
  
  //printf("ppvao: width_direction.real=%f;linewidth_horiz=%f\n",width_direction.real,linewidth_horiz);
  //printf("ppvao: tvout.coord[0]=%f\n",priorx - linewidth_horiz*width_direction.real/2.0);

  //printf("ppvao: totalpos=%u; totallen=%u\n",(unsigned)totalpos,(unsigned)totallen);
  tri_vertices[pos*6].coord[0] = priorx - linewidth_horiz*width_directionx/2.0;
  tri_vertices[pos*6].coord[1] = priory - linewidth_vert*width_directiony/2.0;
  tri_vertices[pos*6].coord[2] = 0.0f;

  tri_vertices[pos*6+1].coord[0] = curx - linewidth_horiz*width_directionx/2.0;
  tri_vertices[pos*6+1].coord[1] = cury - linewidth_vert*width_directiony/2.0;
  tri_vertices[pos*6+1].coord[2] = 0.0f;

  tri_vertices[pos*6+2].coord[0] = priorx + linewidth_horiz*width_directionx/2.0;
  tri_vertices[pos*6+2].coord[1] = priory + linewidth_vert*width_directiony/2.0;
  tri_vertices[pos*6+2].coord[2] = 0.0f;

  tri_vertices[pos*6+3].coord[0] = priorx + linewidth_horiz*width_directionx/2.0;
  tri_vertices[pos*6+3].coord[1] = priory + linewidth_vert*width_directiony/2.0;
  tri_vertices[pos*6+3].coord[2] = 0.0f;

  tri_vertices[pos*6+4].coord[0] = curx - linewidth_horiz*width_directionx/2.0;
  tri_vertices[pos*6+4].coord[1] = cury - linewidth_vert*width_directiony/2.0;
  tri_vertices[pos*6+4].coord[2] = 0.0f;

  tri_vertices[pos*6+5].coord[0] = curx + linewidth_horiz*width_directionx/2.0;
  tri_vertices[pos*6+5].coord[1] = cury + linewidth_vert*width_directiony/2.0;
  tri_vertices[pos*6+5].coord[2] = 0.0f;

  trivert_colors[pos*6*4 + 0*4 + 0] = R;
  trivert_colors[pos*6*4 + 0*4 + 1] = G; 
  trivert_colors[pos*6*4 + 0*4 + 2] = B;
  trivert_colors[pos*6*4 + 0*4 + 3] = A; 

  trivert_colors[pos*6*4 + 1*4 + 0] = R;
  trivert_colors[pos*6*4 + 1*4 + 1] = G; 
  trivert_colors[pos*6*4 + 1*4 + 2] = B;
  trivert_colors[pos*6*4 + 1*4 + 3] = A; 

  trivert_colors[pos*6*4 + 2*4 + 0] = R;
  trivert_colors[pos*6*4 + 2*4 + 1] = G; 
  trivert_colors[pos*6*4 + 2*4 + 2] = B;
  trivert_colors[pos*6*4 + 2*4 + 3] = A; 

  trivert_colors[pos*6*4 + 3*4 + 0] = R;
  trivert_colors[pos*6*4 + 3*4 + 1] = G; 
  trivert_colors[pos*6*4 + 3*4 + 2] = B;
  trivert_colors[pos*6*4 + 3*4 + 3] = A; 

  trivert_colors[pos*6*4 + 4*4 + 0] = R;
  trivert_colors[pos*6*4 + 4*4 + 1] = G; 
  trivert_colors[pos*6*4 + 4*4 + 2] = B;
  trivert_colors[pos*6*4 + 4*4 + 3] = A; 

  trivert_colors[pos*6*4 + 5*4 + 0] = R;
  trivert_colors[pos*6*4 + 5*4 + 1] = G; 
  trivert_colors[pos*6*4 + 5*4 + 2] = B;
  trivert_colors[pos*6*4 + 5*4 + 3] = A; 

}


#ifdef __OPENCL_VERSION__DISABLED

__kernel void waveform_vertices_alphas(OCL_GLOBAL_ADDR waveform_intype *complex_inputs,
					  OCL_GLOBAL_ADDR snde_coord3 *tri_vertices,
					  OCL_GLOBAL_ADDR snde_float32 *trivert_colors,
					  waveform_intype previous_coords,
					  snde_index totalpos, // for historical_fade, with 0 representing the previous_coords for the first call (whih we would never supply)
					  snde_index totallen, // for historical_fade
					  snde_float32 linewidth_horiz,
					  snde_float32 linewidth_vert,
					  snde_float32 R
					  snde_float32 G,
					  snde_float32 B,
					  snde_float32 A,
					  snde_bool historical_fade)
{
  snde_index pos = get_global_id(0);
  
  waveform_vertices_alphas_one(complex_inputs,
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
