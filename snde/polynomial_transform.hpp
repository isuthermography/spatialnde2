#ifndef SNDE_POLYNOMIAL_TRANSFORM_HPP
#define SNDE_POLYNOMIAL_TRANSFORM_HPP

#ifdef SNDE_OPENCL
#include "snde/opencl_utils.hpp"
#endif

namespace snde {

  std::shared_ptr<math_function> define_polynomial_transform_function_float32();
  std::shared_ptr<math_function> define_polynomial_transform_function_float64();

  extern SNDE_OCL_API std::shared_ptr<math_function> polynomial_transform_function_float32;
  extern SNDE_OCL_API std::shared_ptr<math_function> polynomial_transform_function_float64;
  
};


#endif // SNDE_POLYNOMIAL_TRANSFORM_HPP
