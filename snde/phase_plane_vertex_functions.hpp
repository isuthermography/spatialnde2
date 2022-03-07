#ifndef SNDE_PHASE_PLANE_VERTEX_FUNCTIONS_HPP
#define SNDE_PHASE_PLANE_VERTEX_FUNCTIONS_HPP

#ifdef SNDE_OPENCL
#include "snde/opencl_utils.hpp"
#endif

namespace snde {

  extern SNDE_OCL_API std::shared_ptr<math_function> phase_plane_line_triangle_vertices_alphas_function;;
  
  extern SNDE_OCL_API std::shared_ptr<math_function> phase_plane_endpoint_octagon_vertices_function;
  
};


#endif // SNDE_PHASE_PLANE_VERTEX_FUNCTIONS_HPP
