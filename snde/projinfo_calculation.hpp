#ifndef SNDE_PROJINFO_CALCULATION_HPP
#define SNDE_PROJINFO_CALCULATION_HPP

namespace snde {
  
  std::shared_ptr<math_function> define_spatialnde2_projinfo_calculation_function();
  extern SNDE_OCL_API std::shared_ptr<math_function> projinfo_calculation_function;
  
  void instantiate_projinfo(std::shared_ptr<recdatabase> recdb,std::shared_ptr<loaded_part_geometry_recording> loaded_geom);


};



#endif // SNDE_PROJINFO_CALCULATION_HPP
