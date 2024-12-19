#ifndef SNDE_BOXES_CALCULATION_HPP
#define SNDE_BOXES_CALCULATION_HPP


namespace snde {
  

  std::shared_ptr<math_function> define_spatialnde2_boxes_calculation_3d_function();
  extern SNDE_API std::shared_ptr<math_function> boxes_calculation_3d_function;
  
  void instantiate_boxes3d(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags);

  std::shared_ptr<math_function> define_spatialnde2_boxes_calculation_2d_function();
  extern SNDE_API std::shared_ptr<math_function> boxes_calculation_2d_function;

  void instantiate_boxes2d(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags);



};
#endif // SNDE_BOXES_CALCULATION_HPP
