#ifndef SNDE_AREA_CALCULATION_HPP
#define SNDE_AREA_CALCULATION_HPP


namespace snde {

  std::shared_ptr<math_function> define_spatialnde2_trianglearea_calculation_function();

  extern SNDE_API std::shared_ptr<math_function> trianglearea_calculation_function;

  void instantiate_trianglearea(std::shared_ptr<recdatabase> recdb,std::shared_ptr<loaded_part_geometry_recording> loaded_geom);

};
#endif // SNDE_AREA_CALCULATION_HPP
