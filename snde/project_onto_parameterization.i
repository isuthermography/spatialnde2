%{
#include "snde/project_onto_parameterization.hpp"
%}



namespace snde {

  
  std::shared_ptr<math_function> define_spatialnde2_project_point_onto_parameterization_function();

  extern /* SNDE_API */ std::shared_ptr<math_function> project_point_onto_parameterization_function;

  %pythoncode %{
project_point_onto_parameterization = cvar.project_point_onto_parameterization_function # make our swig-wrapped math_function accessible as 'spatialnde2.project_point_onto_parameterization'
project_onto_parameterization_processing_tags = [ "trinormals","inplanemat", "projinfo",  "boxes3d", "boxes2d"]
  %}


};
