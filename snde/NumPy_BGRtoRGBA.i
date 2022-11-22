%{
#include "snde/NumPy_BGRtoRGBA.hpp"
%}

namespace snde {

  std::shared_ptr<math_function> define_spatialnde2_numpy_bgrtorgba_function();
  
  extern /* SNDE_API */ std::shared_ptr<math_function> numpy_bgrtorgba_function;

  %pythoncode %{
numpy_bgrtorgba = cvar.numpy_bgrtorgba_function # make our swig-wrapped math_function accessible as 'spatialnde2.numpy_bgrtorgba

  %}
  
};

