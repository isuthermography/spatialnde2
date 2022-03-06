%{
#include "snde/batched_live_accumulator.hpp"
%}

namespace snde {

  std::shared_ptr<math_function> define_batched_live_accumulator_function();
  
  extern /* SNDE_API */ std::shared_ptr<math_function> batched_live_accumulator_function;

  %pythoncode %{
batched_live_accumulator = cvar.batched_live_accumulator_function # make our swig-wrapped math_function accessible as 'spatialnde2.batched_live_accumulator

  %}
  
};

