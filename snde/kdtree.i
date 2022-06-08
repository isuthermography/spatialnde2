// %shared_ptr(snde::xxxx)
// snde_rawaccessible(snde::xxxx)

%{
#include "snde/kdtree.hpp"
%}

namespace snde {
  class assigned_compute_resource;
  class rwlock_token_set;
  template<typename T> class ndtyped_recording_ref<T>;
  
  std::shared_ptr<math_function> define_kdtree_calculation_function();
  extern std::shared_ptr<math_function> kdtree_calculation_function;

  %pythoncode %{
kdtree_calculation = cvar.kdtree_calculation_function  # make our swig-wrapped math_function accessible as 'spatialnde2.kdtree_calculation'
  %}
  
  void perform_knn_calculation(std::shared_ptr<assigned_compute_resource> compute_resource,rwlock_token_set locktokens, std::shared_ptr<ndtyped_recording_ref<snde_kdnode>> kdtree, std::shared_ptr<ndtyped_recording_ref<uint32_t>> nodemask,std::shared_ptr<ndtyped_recording_ref<snde_coord3>> kdtree_vertices, std::shared_ptr<ndtyped_recording_ref<snde_index>> search_point_indices,std::shared_ptr<ndtyped_recording_ref<snde_coord3>> search_points,std::shared_ptr<ndtyped_recording_ref<snde_index>> result_ref); // NOTE: nodemask and/or search_point_indices may be nullptr

  std::shared_ptr<math_function> define_knn_calculation_function();
  extern std::shared_ptr<math_function> knn_calculation_function;


  %pythoncode %{
knn_calculation = cvar.knn_calculation_function  # make our swig-wrapped math_function accessible as 'spatialnde2.knn_calculation'
  %}


};
