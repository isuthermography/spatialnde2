%{
#include "snde/NumPy_BGRtoGray16.hpp"
%}

namespace snde {

  std::shared_ptr<math_function> define_spatialnde2_numpy_bgrtogray16_function();
  
  extern /* SNDE_API */ std::shared_ptr<math_function> numpy_bgrtogray16_function;

  %pythoncode %{
numpy_bgrtogray16 = cvar.numpy_bgrtogray16_function # make our swig-wrapped math_function accessible as 'spatialnde2.numpy_bgrtogray16

  %}
  
};

