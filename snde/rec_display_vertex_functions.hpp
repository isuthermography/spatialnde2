#ifndef SNDE_REC_DISPLAY_VERTEX_FUNCTIONS_HPP
#define SNDE_REC_DISPLAY_VERTEX_FUNCTIONS_HPP

#include <memory>

#include "snde/recmath.hpp"

namespace snde {

  std::shared_ptr<math_function> define_meshedpart_vertexarray_function();
  std::shared_ptr<math_function> define_meshedparameterization_texvertexarray_function();
}

#endif // SNDE_REC_DISPLAY_VERTEX_FUNCTIONS_HPP

