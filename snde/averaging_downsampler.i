%{
#include "snde/averaging_downsampler.hpp"
%}

namespace snde {

  std::shared_ptr<math_function> define_averaging_downsampler_function();
  
  extern /* SNDE_API */ std::shared_ptr<math_function> averaging_downsampler_function;

  %pythoncode %{
averaging_downsampler = cvar.averaging_downsampler_function # make our swig-wrapped math_function accessible as 'spatialnde2.averaging_downsampler

  %}
  
};

