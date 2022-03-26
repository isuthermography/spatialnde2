#ifndef SNDE_REC_DISPLAY_COLORMAP_HPP
#define SNDE_REC_DISPLAY_COLORMAP_HPP

#include <memory>

#include "snde/recmath.hpp"

namespace snde {
  std::shared_ptr<math_function> define_colormap_function();

  std::shared_ptr<math_function> define_fusion_colormapping_function();
  extern SNDE_OCL_API std::shared_ptr<math_function> fusion_colormapping_function;

};

#endif // SNDE_REC_DISPLAY_COLORMAP_HPP

