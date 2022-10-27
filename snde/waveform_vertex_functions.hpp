#ifndef SNDE_WAVEFORM_VERTEX_FUNCTIONS_HPP
#define SNDE_WAVEFORM_VERTEX_FUNCTIONS_HPP

#ifdef SNDE_OPENCL
#include "snde/opencl_utils.hpp"
#endif

namespace snde {

  extern SNDE_OCL_API std::shared_ptr<math_function> waveform_line_triangle_vertices_alphas_function;
  
};


#endif // SNDE_WAVEFORM_VERTEX_FUNCTIONS_HPP
