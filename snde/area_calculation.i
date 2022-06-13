%{
#include "snde/area_calculation.hpp"
%}


namespace snde {

  std::shared_ptr<math_function> define_spatialnde2_trianglearea_calculation_function();

  extern /* SNDE_OCL_API */ std::shared_ptr<math_function> trianglearea_calculation_function;

  %pythoncode %{
trianglearea_calculation = cvar.trianglearea_calculation_function  # make our swig-wrapped math_function accessible as 'spatialnde2.trianglearea'
  %}


  std::shared_ptr<math_function> define_spatialnde2_vertexarea_calculation_function();

  extern /* SNDE_OCL_API */ std::shared_ptr<math_function> vertexarea_calculation_function;

  %pythoncode %{
vertexarea_calculation = cvar.vertexarea_calculation_function  # make our swig-wrapped math_function accessible as 'spatialnde2.vertexarea'
  %}


};
