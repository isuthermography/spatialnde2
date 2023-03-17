#ifndef POLYNOMIAL_TRANSFORM_DECL
#define POLYNOMIAL_TRANSFORM_DECL
#endif


POLYNOMIAL_TRANSFORM_DECL
void polynomial_transform_math(OCL_GLOBAL_ADDR function_outtype* output, // output array
  OCL_GLOBAL_ADDR function_inputtype* data, // input data -- must be same dimensions as output array
  OCL_GLOBAL_ADDR function_polytype* poly, // polynomial -- must be same dimensions as data in all but one axis
  snde_index datandim, // number of dimensions of data array
  OCL_GLOBAL_ADDR snde_index* datadims, // array of data dimensions
  OCL_LOCAL_ADDR snde_index* datacalc, // Empty array of length datandim used during calculation to determine where we are in the data array
  snde_index polyndim, // number of dimensions of data array
  OCL_GLOBAL_ADDR snde_index* polydims, // array of data dimensions
  OCL_LOCAL_ADDR snde_index* polycalc, // Empty array of length datandim used during calculation to determine where we are in the data array
  snde_index polyaxis, // Axis along which polynomial coefficients exist
  snde_index idx)  // index to calculate
{

  if (polyndim == 1) // Single polynomial to be applied everywhere
  {

    // Directly compute result
    output[idx] = 0;
    for (snde_index i = 0; i < polydims[0]; i++) {
      output[idx] += (function_outtype)poly[i] * pow((function_outtype)data[idx], (function_outtype)i);
    }

  }
  else if (polyndim == datandim) // Two dimensions must match and the third contains the polynomial coefficients
  {
    // Compute current position in data array given index and sizes
    // WARNING!  This does not currently support Fortran ordering or unusual stride lengths
    // This will be fixed later
    //printf("%llu, %llu, %llu, %llu, %llu, %llu\n", (unsigned long)datadims[0], (unsigned long)datadims[1], (unsigned long)datadims[2], (unsigned long)datadims[3], (unsigned long)datadims[4], (unsigned long)datadims[5]);
    //printf("%llu -- %llu -- %llu\n", (unsigned long)sizeof(unsigned long), (unsigned long)sizeof(*datadims), (unsigned long)sizeof(snde_index));
    for (snde_index i = 0; i < datandim; i++) {
      snde_index remaining = 1;
      for (snde_index j = i + 1; j < datandim; j++) {
	remaining *= datadims[j];
      }

      snde_index found = 0;
      for (snde_index j = 0; j < i; j++) {

	snde_index mult = 1;
	for (snde_index k = j + 1; k < datandim; k++) {
	  mult *= datadims[k];
	}

	found += datacalc[j] * mult;
      }

      datacalc[i] = (idx - found) / remaining;
    }

    // Calculate Result
    output[idx] = 0;
    for (snde_index i = 0; i < polydims[polyaxis]; i++) {

      // Populate correct position in polynomial array
      for (snde_index k = 0; k < polyndim; k++) {
	if (k == polyaxis) {
	  polycalc[k] = i;
	}
	else {
	  polycalc[k] = datacalc[k];
	}
      }

      // Get current polynomial index
      // Assumes C Ordering and No Unusual Strides -- Will be fixed later
      snde_index curpolyidx = 0;
      for (snde_index j = 0; j < polyndim; j++) {
	snde_index mult = 1;
	for (snde_index k = j + 1; k < polyndim; k++) {
	  mult *= polydims[k];
	}
	curpolyidx += polycalc[j] * mult;
      }

      output[idx] += (function_outtype)poly[curpolyidx] * pow((function_outtype)data[idx], (function_outtype)i);
    }

  }      

}

#ifdef __OPENCL_VERSION__

	__kernel void polynomial_transform_math_ocl(OCL_GLOBAL_ADDR function_outtype* output, // output array
	  OCL_GLOBAL_ADDR function_inputtype* data, // input data -- must be same dimensions as output array
	  OCL_GLOBAL_ADDR function_polytype* poly, // polynomial -- must be same dimensions as data in all but one axis
	  snde_index datandim, // number of dimensions of data array
	  OCL_GLOBAL_ADDR snde_index* datadims, // array of data dimensions
	  OCL_LOCAL_ADDR snde_index* datacalc, // Empty array of length datandim used during calculation to determine where we are in the data array
	  snde_index polyndim, // number of dimensions of data array
	  OCL_GLOBAL_ADDR snde_index* polydims, // array of data dimensions
	  OCL_LOCAL_ADDR snde_index* polycalc, // Empty array of length datandim used during calculation to determine where we are in the data array
	  snde_index polyaxis) // Axis along which polynomial coefficients exist 
	{
	  snde_index idx = get_global_id(0);

	  OCL_LOCAL_ADDR snde_index* localdatacalc = (OCL_LOCAL_ADDR snde_index*)(((OCL_LOCAL_ADDR uint8_t*)datacalc) + get_local_id(0) * datandim * sizeof(snde_index));
	  OCL_LOCAL_ADDR snde_index* localpolycalc = (OCL_LOCAL_ADDR snde_index*)(((OCL_LOCAL_ADDR uint8_t*)polycalc) + get_local_id(0) * polyndim * sizeof(snde_index));

	  //printf("global: %u   local: %u   datacalc: 0x%p   localdatacalc: 0x%p   datandim: %llu   sizeof(snde_index): %llu\n", (unsigned)idx, (unsigned)get_local_id(0), (OCL_LOCAL_ADDR void*)datacalc, (OCL_LOCAL_ADDR void*)localdatacalc, (unsigned long)datandim, (unsigned long)sizeof(snde_index));

	  polynomial_transform_math(output, // output array
	    data, // input data -- must be same dimensions as output array
	    poly, // polynomial -- must be same dimensions as data in all but one axis
	    datandim, // number of dimensions of data array
	    datadims, // array of data dimensions
	    localdatacalc, // Empty array of length datandim used during calculation to determine where we are in the data array
	    polyndim, // number of dimensions of data array
	    polydims, // array of data dimensions
	    localpolycalc, // Empty array of length datandim used during calculation to determine where we are in the data array
	    polyaxis,
	    idx); // Axis along which polynomial coefficients exist);
	}


#endif // __OPENCL_VERSION__


