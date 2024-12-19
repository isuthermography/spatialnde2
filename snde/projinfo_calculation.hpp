#ifndef SNDE_PROJINFO_CALCULATION_HPP
#define SNDE_PROJINFO_CALCULATION_HPP

namespace snde {
  
  std::shared_ptr<math_function> define_spatialnde2_projinfo_calculation_function();
  extern SNDE_OCL_API std::shared_ptr<math_function> projinfo_calculation_function;
  
  void instantiate_projinfo(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags);


};



#endif // SNDE_PROJINFO_CALCULATION_HPP
