#ifndef SNDE_BOXES_CALCULATION_HPP
#define SNDE_BOXES_CALCULATION_HPP


namespace snde {
  

  std::shared_ptr<math_function> define_spatialnde2_boxes_calculation_3d_function();
  extern SNDE_API std::shared_ptr<math_function> boxes_calculation_3d_function;
  
  std::shared_ptr<math_function> define_spatialnde2_boxes_calculation_2d_function();
  extern SNDE_API std::shared_ptr<math_function> boxes_calculation_2d_function;




};
#endif // SNDE_BOXES_CALCULATION_HPP
