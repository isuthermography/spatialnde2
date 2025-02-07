#ifndef SNDE_AREA_CALCULATION_HPP
#define SNDE_AREA_CALCULATION_HPP


namespace snde {

  std::shared_ptr<math_function> define_spatialnde2_trianglearea_calculation_function();

  extern SNDE_OCL_API std::shared_ptr<math_function> trianglearea_calculation_function;

  void instantiate_trianglearea(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags);


  std::shared_ptr<math_function> define_spatialnde2_vertexarea_calculation_function();

  extern SNDE_OCL_API std::shared_ptr<math_function> vertexarea_calculation_function;

  void instantiate_vertexarea(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags);

};
#endif // SNDE_AREA_CALCULATION_HPP
