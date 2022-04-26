#ifndef SNDE_BOXES_CALCULATION_HPP
#define SNDE_BOXES_CALCULATION_HPP


namespace snde {
  

  std::shared_ptr<math_function> define_spatialnde2_boxes_calculation_3d_function();
  extern SNDE_API std::shared_ptr<math_function> boxes_calculation_3d_function;
  
  void instantiate_boxes3d(std::shared_ptr<recdatabase> recdb,std::shared_ptr<loaded_part_geometry_recording> loaded_geom);

  std::shared_ptr<math_function> define_spatialnde2_boxes_calculation_2d_function();
  extern SNDE_API std::shared_ptr<math_function> boxes_calculation_2d_function;

  void instantiate_boxes2d(std::shared_ptr<recdatabase> recdb,std::shared_ptr<loaded_part_geometry_recording> loaded_geom);



};
#endif // SNDE_BOXES_CALCULATION_HPP
