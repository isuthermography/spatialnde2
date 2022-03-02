%{
#include "snde/boxes_calculation.hpp"
%}



namespace snde {

  std::shared_ptr<math_function> define_spatialnde2_boxes_calculation_3d_function();
  extern /* SNDE_API */ std::shared_ptr<math_function> boxes_calculation_3d_function;
  
  std::shared_ptr<math_function> define_spatialnde2_boxes_calculation_2d_function();
  extern /* SNDE_API */ std::shared_ptr<math_function> boxes_calculation_2d_function;


  %pythoncode %{
boxes_calculation_3d = cvar.boxes_calculation_3d_function  # make our swig-wrapped math_function accessible as 'spatialnde2.boxes_calculation_3d'
boxes_calculation_2d = cvar.boxes_calculation_2d_function  # make our swig-wrapped math_function accessible as 'spatialnde2.boxes_calculation_2d'
  %}


};
