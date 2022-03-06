%{
#include "snde/projinfo_calculation.hpp"
%}



namespace snde {

  
  std::shared_ptr<math_function> define_spatialnde2_projinfo_calculation_function();
  extern /* SNDE_OCL_API */ std::shared_ptr<math_function> projinfo_calculation_function;

  %pythoncode %{
projinfo_calculation = cvar.projinfo_calculation_function  # make our swig-wrapped math_function accessible as 'spatialnde2.projinfo_calculation'
  %}


};
