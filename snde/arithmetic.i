%{
#include "snde/arithmetic.hpp"
%}

namespace snde {

  std::shared_ptr<math_function> define_addition_function();
  
  extern /* SNDE_OCL_API*/ std::shared_ptr<math_function> addition_function;

  std::shared_ptr<math_function> define_subtraction_function();
  
  extern /* SNDE_OCL_API*/ std::shared_ptr<math_function> subtraction_function;

  %pythoncode %{
addition = cvar.addition_function # make our swig-wrapped math_function accessible as 'spatialnde2.addition'

subtraction = cvar.subtraction_function # make our swig-wrapped math_function accessible as 'spatialnde2.subtraction'

  %}
  
};
