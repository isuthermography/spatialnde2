#ifndef SNDE_NORMAL_CALCULATION_HPP
#define SNDE_NORMAL_CALCULATION_HPP


namespace snde {

  

  std::shared_ptr<math_function> define_spatialnde2_trinormals_function();
  SNDE_OCL_API extern std::shared_ptr<math_function> trinormals_function;

  void instantiate_trinormals(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags);
  
  std::shared_ptr<math_function> define_vertnormals_recording_function();
  SNDE_OCL_API extern std::shared_ptr<math_function> vertnormals_recording_function;

  void instantiate_vertnormals(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags);

};
#endif // SNDE_NORMAL_CALCULATION_HPP
