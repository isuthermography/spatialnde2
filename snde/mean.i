%{
#include "snde/mean.hpp"
%}

namespace snde {

  std::shared_ptr<math_function> define_mean_function();
  
  extern /* SNDE_API */ std::shared_ptr<math_function> mean_function;

  %pythoncode %{
  mean = cvar.mean_function # make our swig-wrapped math_function accessible as 'spatialnde2.mean

  %}
  
};

