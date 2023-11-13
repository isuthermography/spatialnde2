%{
#include "snde/nd_accumulate_once.hpp"
%}

namespace snde {

  std::shared_ptr<math_function> define_nd_accumulate_once_function();
  
  extern /* SNDE_API */ std::shared_ptr<math_function> nd_accumulate_once_function;

  %pythoncode %{
nd_accumulate_once = cvar.nd_accumulate_once_function # make our swig-wrapped math_function accessible as 'spatialnde2.nd_accumulate_once


  %}
  
};

