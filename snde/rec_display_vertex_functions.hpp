#ifndef SNDE_REC_DISPLAY_VERTEX_FUNCTIONS_HPP
#define SNDE_REC_DISPLAY_VERTEX_FUNCTIONS_HPP

#include <memory>

#include "snde/recmath.hpp"

namespace snde {
  std::shared_ptr<math_function> define_vertexarray_recording_function();
  std::shared_ptr<math_function> define_texvertexarray_recording_function();
}

#endif // SNDE_REC_DISPLAY_VERTEX_FUNCTIONS_HPP

