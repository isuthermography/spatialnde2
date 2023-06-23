#ifndef WAVEFORM_DECL
#define WAVEFORM_DECL
#endif



WAVEFORM_DECL
void waveform_as_vertlines(OCL_GLOBAL_ADDR waveform_intype* inputs,
	OCL_GLOBAL_ADDR snde_coord3* tri_vertices,
	OCL_GLOBAL_ADDR snde_float32* trivert_colors,
	snde_index cnt, // within these inputs and these outputs,
	snde_index startidx,
	snde_index endidx,
	float inival,
	float step,
	snde_index pxstep,
	snde_float32 linewidth_horiz,
	snde_float32 linewidth_vert,
	snde_float32 R,
	snde_float32 G,
	snde_float32 B,
	snde_float32 A)
{

	
        snde_index endpt = (startidx + (cnt+1) * pxstep) + 1;
	if (((startidx + (cnt + 2) * pxstep) + 1) > endidx)
	  endpt = endidx;

	waveform_intype minval = -log(0.0f);
	waveform_intype maxval = -minval;
	for (snde_index i = (startidx + cnt*pxstep); i <= endpt; ++i) {
		if (inputs[i] > maxval) {
			maxval = inputs[i];
		}
		if (inputs[i] < minval) {
			minval = inputs[i];
		}
	}

	waveform_intype priorx, priory, curx, cury;

	if ((maxval - minval) < (linewidth_vert))
	{
		minval = minval - linewidth_vert / 2.0f;
		maxval = maxval + linewidth_vert / 2.0f;
	}

	priorx = step * pxstep * (waveform_intype)(cnt)+inival + step * (waveform_intype)(startidx)+0.5f * step;
	curx = priorx;
	priory = minval;	
	cury = maxval;	

	// draw line from prior_coords to complex_inputs[cnt] via 2 CCW triangles

	// x to the right, y up; z is pointing at us.
	// want to select width_direction such that length_direction x width_direction = z ; i.e. width is like y. 
	// or equivalently z x length_direction = width_direction

	//                    |   i     j     k   |
	// width_direction =  |   0     0     1   |
	//                    |  lx0   ly0    0   |
	// (where lx0, ly0 presumed normalized)
	// Therefore width_direction = -i*l0y + j*l0x

	//printf("ppvao: width_direction.real=%f;linewidth_horiz=%f\n",width_direction.real,linewidth_horiz);
	//printf("ppvao: tvout.coord[0]=%f\n",priorx - linewidth_horiz*width_direction.real/2.0f);

	//printf("ppvao: totalcnt=%u; totallen=%u\n",(unsigned)totalcnt,(unsigned)totallen);
	tri_vertices[cnt * 6].coord[0] = priorx - linewidth_horiz / 2.0f;
	tri_vertices[cnt * 6].coord[1] = priory;
	tri_vertices[cnt * 6].coord[2] = 0.0f;

	tri_vertices[cnt * 6 + 1].coord[0] = curx - linewidth_horiz / 2.0f;
	tri_vertices[cnt * 6 + 1].coord[1] = cury;
	tri_vertices[cnt * 6 + 1].coord[2] = 0.0f;

	tri_vertices[cnt * 6 + 2].coord[0] = priorx + linewidth_horiz / 2.0f;
	tri_vertices[cnt * 6 + 2].coord[1] = priory;
	tri_vertices[cnt * 6 + 2].coord[2] = 0.0f;

	tri_vertices[cnt * 6 + 3].coord[0] = priorx + linewidth_horiz / 2.0f;
	tri_vertices[cnt * 6 + 3].coord[1] = priory;
	tri_vertices[cnt * 6 + 3].coord[2] = 0.0f;

	tri_vertices[cnt * 6 + 4].coord[0] = curx - linewidth_horiz / 2.0f;
	tri_vertices[cnt * 6 + 4].coord[1] = cury;
	tri_vertices[cnt * 6 + 4].coord[2] = 0.0f;

	tri_vertices[cnt * 6 + 5].coord[0] = curx + linewidth_horiz / 2.0f;
	tri_vertices[cnt * 6 + 5].coord[1] = cury;
	tri_vertices[cnt * 6 + 5].coord[2] = 0.0f;

	trivert_colors[cnt * 6 * 4 + 0 * 4 + 0] = R;
	trivert_colors[cnt * 6 * 4 + 0 * 4 + 1] = G;
	trivert_colors[cnt * 6 * 4 + 0 * 4 + 2] = B;
	trivert_colors[cnt * 6 * 4 + 0 * 4 + 3] = A;

	trivert_colors[cnt * 6 * 4 + 1 * 4 + 0] = R;
	trivert_colors[cnt * 6 * 4 + 1 * 4 + 1] = G;
	trivert_colors[cnt * 6 * 4 + 1 * 4 + 2] = B;
	trivert_colors[cnt * 6 * 4 + 1 * 4 + 3] = A;

	trivert_colors[cnt * 6 * 4 + 2 * 4 + 0] = R;
	trivert_colors[cnt * 6 * 4 + 2 * 4 + 1] = G;
	trivert_colors[cnt * 6 * 4 + 2 * 4 + 2] = B;
	trivert_colors[cnt * 6 * 4 + 2 * 4 + 3] = A;

	trivert_colors[cnt * 6 * 4 + 3 * 4 + 0] = R;
	trivert_colors[cnt * 6 * 4 + 3 * 4 + 1] = G;
	trivert_colors[cnt * 6 * 4 + 3 * 4 + 2] = B;
	trivert_colors[cnt * 6 * 4 + 3 * 4 + 3] = A;

	trivert_colors[cnt * 6 * 4 + 4 * 4 + 0] = R;
	trivert_colors[cnt * 6 * 4 + 4 * 4 + 1] = G;
	trivert_colors[cnt * 6 * 4 + 4 * 4 + 2] = B;
	trivert_colors[cnt * 6 * 4 + 4 * 4 + 3] = A;

	trivert_colors[cnt * 6 * 4 + 5 * 4 + 0] = R;
	trivert_colors[cnt * 6 * 4 + 5 * 4 + 1] = G;
	trivert_colors[cnt * 6 * 4 + 5 * 4 + 2] = B;
	trivert_colors[cnt * 6 * 4 + 5 * 4 + 3] = A;

}

#ifdef __OPENCL_VERSION__

	__kernel void waveform_vertlines(OCL_GLOBAL_ADDR waveform_intype* inputs,
	  OCL_GLOBAL_ADDR snde_coord3* tri_vertices,
	  OCL_GLOBAL_ADDR snde_float32* trivert_colors,
	  snde_index startidx,
	  snde_index endidx,
	  float inival,
	  float step,
	  snde_index pxstep,
	  snde_float32 linewidth_horiz,
	  snde_float32 linewidth_vert,
	  snde_float32 R,
	  snde_float32 G,
	  snde_float32 B,
	  snde_float32 A)
	{
	  snde_index cnt = get_global_id(0);

	  waveform_as_vertlines(inputs,
	    tri_vertices,
	    trivert_colors,
	    cnt, // within these inputs and these outputs,
	    startidx,
	    endidx,
	    inival,
	    step,
	    pxstep,
	    linewidth_horiz,
	    linewidth_vert,
	    R,
	    G,
	    B,
	    A);
	}


#endif // __OPENCL_VERSION__


WAVEFORM_DECL
void waveform_as_interplines(OCL_GLOBAL_ADDR waveform_intype* inputs,
	OCL_GLOBAL_ADDR snde_coord3* tri_vertices,
	OCL_GLOBAL_ADDR snde_float32* trivert_colors,
	snde_index cnt, // within these inputs and these outputs,
	snde_index pos,
	float inival,
	float step,
	snde_float32 linewidth_horiz,
	snde_float32 linewidth_vert,
	snde_float32 R,
	snde_float32 G,
	snde_float32 B,
	snde_float32 A)
{

	waveform_intype priorx, priory, curx, cury;

	priorx = step * (waveform_intype)(pos+cnt - 1) + inival;
	priory = inputs[(pos+cnt) - 1];
	curx = step * (waveform_intype)(pos+cnt)+inival;
	cury = inputs[pos+cnt];

	// draw line from prior_coords to complex_inputs[cnt] via 2 CCW triangles

	// x to the right, y up; z is pointing at us.
	// want to select width_direction such that length_direction x width_direction = z ; i.e. width is like y. 
	// or equivalently z x length_direction = width_direction

	//                    |   i     j     k   |
	// width_direction =  |   0     0     1   |
	//                    |  lx0   ly0    0   |
	// (where lx0, ly0 presumed normalized)
	// Therefore width_direction = -i*l0y + j*l0x

	waveform_intype x, y, x0, y0;
	x = inputs[pos+cnt] - priorx;
	y = inputs[pos+cnt] - priory;

	waveform_intype scale;
	scale = 1.0f / sqrt(x * x + y * y);

	x0 = x * scale;
	y0 = y * scale;

	waveform_intype width_directionx, width_directiony;
	width_directionx = -y0;
	width_directiony = x0;
	//assert(!isnan(width_directionx));   // Not allowed in OpenCL

	//printf("ppvao: width_direction.real=%f;linewidth_horiz=%f\n",width_direction.real,linewidth_horiz);
	//printf("ppvao: tvout.coord[0]=%f\n",priorx - linewidth_horiz*width_direction.real/2.0);

	//printf("ppvao: totalcnt=%u; totallen=%u\n",(unsigned)totalcnt,(unsigned)totallen);
	tri_vertices[cnt * 6].coord[0] = priorx - linewidth_horiz * width_directionx / 2.0;
	tri_vertices[cnt * 6].coord[1] = priory - linewidth_vert * width_directiony / 2.0;
	tri_vertices[cnt * 6].coord[2] = 0.0f;

	tri_vertices[cnt * 6 + 1].coord[0] = curx - linewidth_horiz * width_directionx / 2.0;
	tri_vertices[cnt * 6 + 1].coord[1] = cury - linewidth_vert * width_directiony / 2.0;
	tri_vertices[cnt * 6 + 1].coord[2] = 0.0f;

	tri_vertices[cnt * 6 + 2].coord[0] = priorx + linewidth_horiz * width_directionx / 2.0;
	tri_vertices[cnt * 6 + 2].coord[1] = priory + linewidth_vert * width_directiony / 2.0;
	tri_vertices[cnt * 6 + 2].coord[2] = 0.0f;

	tri_vertices[cnt * 6 + 3].coord[0] = priorx + linewidth_horiz * width_directionx / 2.0;
	tri_vertices[cnt * 6 + 3].coord[1] = priory + linewidth_vert * width_directiony / 2.0;
	tri_vertices[cnt * 6 + 3].coord[2] = 0.0f;

	tri_vertices[cnt * 6 + 4].coord[0] = curx - linewidth_horiz * width_directionx / 2.0;
	tri_vertices[cnt * 6 + 4].coord[1] = cury - linewidth_vert * width_directiony / 2.0;
	tri_vertices[cnt * 6 + 4].coord[2] = 0.0f;

	tri_vertices[cnt * 6 + 5].coord[0] = curx + linewidth_horiz * width_directionx / 2.0;
	tri_vertices[cnt * 6 + 5].coord[1] = cury + linewidth_vert * width_directiony / 2.0;
	tri_vertices[cnt * 6 + 5].coord[2] = 0.0f;

	trivert_colors[cnt * 6 * 4 + 0 * 4 + 0] = R;
	trivert_colors[cnt * 6 * 4 + 0 * 4 + 1] = G;
	trivert_colors[cnt * 6 * 4 + 0 * 4 + 2] = B;
	trivert_colors[cnt * 6 * 4 + 0 * 4 + 3] = A;

	trivert_colors[cnt * 6 * 4 + 1 * 4 + 0] = R;
	trivert_colors[cnt * 6 * 4 + 1 * 4 + 1] = G;
	trivert_colors[cnt * 6 * 4 + 1 * 4 + 2] = B;
	trivert_colors[cnt * 6 * 4 + 1 * 4 + 3] = A;

	trivert_colors[cnt * 6 * 4 + 2 * 4 + 0] = R;
	trivert_colors[cnt * 6 * 4 + 2 * 4 + 1] = G;
	trivert_colors[cnt * 6 * 4 + 2 * 4 + 2] = B;
	trivert_colors[cnt * 6 * 4 + 2 * 4 + 3] = A;

	trivert_colors[cnt * 6 * 4 + 3 * 4 + 0] = R;
	trivert_colors[cnt * 6 * 4 + 3 * 4 + 1] = G;
	trivert_colors[cnt * 6 * 4 + 3 * 4 + 2] = B;
	trivert_colors[cnt * 6 * 4 + 3 * 4 + 3] = A;

	trivert_colors[cnt * 6 * 4 + 4 * 4 + 0] = R;
	trivert_colors[cnt * 6 * 4 + 4 * 4 + 1] = G;
	trivert_colors[cnt * 6 * 4 + 4 * 4 + 2] = B;
	trivert_colors[cnt * 6 * 4 + 4 * 4 + 3] = A;

	trivert_colors[cnt * 6 * 4 + 5 * 4 + 0] = R;
	trivert_colors[cnt * 6 * 4 + 5 * 4 + 1] = G;
	trivert_colors[cnt * 6 * 4 + 5 * 4 + 2] = B;
	trivert_colors[cnt * 6 * 4 + 5 * 4 + 3] = A;

}


#ifdef __OPENCL_VERSION__

	__kernel void waveform_interplines(OCL_GLOBAL_ADDR waveform_intype* inputs,
	  OCL_GLOBAL_ADDR snde_coord3* tri_vertices,
	  OCL_GLOBAL_ADDR snde_float32* trivert_colors,
	  snde_index pos,
	  float inival,
	  float step,
	  snde_float32 linewidth_horiz,
	  snde_float32 linewidth_vert,
	  snde_float32 R,
	  snde_float32 G,
	  snde_float32 B,
	  snde_float32 A)
	{
	  snde_index cnt = get_global_id(0);

	  waveform_as_interplines(inputs,
	    tri_vertices,
	    trivert_colors,
	    cnt,
	    pos,
	    inival,
	    step,
	    linewidth_horiz,
	    linewidth_vert,
	    R,
	    G,
	    B,
	    A);
	}


#endif // __OPENCL_VERSION__




	WAVEFORM_DECL
	void waveform_as_points(OCL_GLOBAL_ADDR waveform_intype* inputs,
		OCL_GLOBAL_ADDR snde_coord3* tri_vertices,
		OCL_GLOBAL_ADDR snde_float32* trivert_colors,
		snde_index cnt, // within these inputs and these outputs,
		snde_index pos,
		float inival,
		float step,
		snde_float32 R,
		snde_float32 G,
		snde_float32 B,
		snde_float32 A)
	{

		waveform_intype priorx, priory, curx, cury;

		curx = step * (waveform_intype)(pos+cnt)+inival;
		cury = inputs[pos+cnt];

		tri_vertices[cnt].coord[0] = curx;
		tri_vertices[cnt].coord[1] = cury;
		tri_vertices[cnt].coord[2] = 0.0f;

		trivert_colors[cnt * 4 + 0 * 4 + 0] = R;
		trivert_colors[cnt * 4 + 0 * 4 + 1] = G;
		trivert_colors[cnt * 4 + 0 * 4 + 2] = B;
		trivert_colors[cnt * 4 + 0 * 4 + 3] = A;

	}


#ifdef __OPENCL_VERSION__

	__kernel void waveform_points(OCL_GLOBAL_ADDR waveform_intype* inputs,
	  OCL_GLOBAL_ADDR snde_coord3* tri_vertices,
	  OCL_GLOBAL_ADDR snde_float32* trivert_colors,
	  snde_index pos,
	  float inival,
	  float step,
	  snde_float32 R,
	  snde_float32 G,
	  snde_float32 B,
	  snde_float32 A)
	{
	  snde_index cnt = get_global_id(0);

	  waveform_as_points(inputs,
	    tri_vertices,
	    trivert_colors,
	    cnt,
	    pos,
	    inival,
	    step,
	    R,
	    G,
	    B,
	    A);
	}


#endif // __OPENCL_VERSION__
