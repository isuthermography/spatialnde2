%{
#include "snde/inplanemat_calculation.hpp"
%}



namespace snde {

  
  std::shared_ptr<math_function> define_spatialnde2_inplanemat_calculation_function();
  extern /* SNDE_API */ std::shared_ptr<math_function> inplanemat_calculation_function;

  %pythoncode %{
inplanemat_calculation = cvar.inplanemat_calculation_function  # make our swig-wrapped math_function accessible as 'spatialnde2.inplanemat_calculation'
  %}


};
