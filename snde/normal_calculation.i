%{
#include "snde/normal_calculation.hpp"
%}



namespace snde {

  

  std::shared_ptr<math_function> define_spatialnde2_trinormals_function();
  extern std::shared_ptr<math_function> trinormals_function;

  %pythoncode %{
trinormals = cvar.trinormals_function  # make our swig-wrapped math_function accessible as 'spatialnde2.trinormals'
  %}

  
  std::shared_ptr<math_function> define_vertnormals_recording_function();
  extern std::shared_ptr<math_function> vertnormals_recording_function;

%pythoncode %{
vertnormals_recording = cvar.vertnormals_recording_function  # make our swig-wrapped math_function accessible as 'spatialnde2.vertnormals_recording'
%}

};
