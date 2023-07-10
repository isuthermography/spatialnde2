%{
#include "snde/accumulate_once.hpp"
%}

namespace snde {

  std::shared_ptr<math_function> define_accumulate_once_function();
  
  extern /* SNDE_API */ std::shared_ptr<math_function> accumulate_once_function;

  %pythoncode %{
accumulate_once = cvar.accumulate_once_function # make our swig-wrapped math_function accessible as 'spatialnde2.accumulate_once


  %}
  
};

