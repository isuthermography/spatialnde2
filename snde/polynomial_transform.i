%{
#include "snde/polynomial_transform.hpp"
%}

namespace snde {

  std::shared_ptr<math_function> define_polynomial_transform_function_float32();
  std::shared_ptr<math_function> define_polynomial_transform_function_float64();
  
  extern /* SNDE_API */ std::shared_ptr<math_function> polynomial_transform_function_float32;
  extern /* SNDE_API */ std::shared_ptr<math_function> polynomial_transform_function_float64;

  %pythoncode %{
polynomial_transform_float32 = cvar.polynomial_transform_function_float32 # make our swig-wrapped math_function accessible as 'spatialnde2.polynomial_transform_function_float32'
polynomial_transform_float64 = cvar.polynomial_transform_function_float64 # make our swig-wrapped math_function accessible as 'spatialnde2.polynomial_transform_function_float64'

  %}
  
};

