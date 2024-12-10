%{
#include "snde/bad_pixel_correction.hpp"
%}

namespace snde {

  std::shared_ptr<math_function> define_bad_pixel_correction_function();
  
  extern /* SNDE_API */ std::shared_ptr<math_function> bad_pixel_correction_function;

  %pythoncode %{
bad_pixel_correction = cvar.bad_pixel_correction_function # make our swig-wrapped math_function accessible as 'spatialnde2.bad_pixel_correction


  %}
  
};

