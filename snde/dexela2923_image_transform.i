%{
#include "snde/dexela2923_image_transform.hpp"
%}

namespace snde {

  std::shared_ptr<math_function> define_dexela2923_image_transform_function();
  
  extern /* SNDE_API */ std::shared_ptr<math_function> dexela2923_image_transform_function;

  %pythoncode %{
dexela2923_image_transform = cvar.dexela2923_image_transform_function # make our swig-wrapped math_function accessible as 'spatialnde2.dexela2923_image_transform'


  %}
  
};

