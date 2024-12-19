#ifndef SNDE_INPLANEMAT_CALCULATION_HPP
#define SNDE_INPLANEMAT_CALCULATION_HPP


namespace snde {
  
  std::shared_ptr<math_function> define_spatialnde2_inplanemat_calculation_function();

  // NOTE: Change to SNDE_OCL_API if/when we add GPU acceleration support, and
  // (in CMakeLists.txt) make it move into the _ocl.so library)
  extern SNDE_API std::shared_ptr<math_function> inplanemat_calculation_function;

  void instantiate_inplanemat(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags);

};
#endif // SNDE_INPLANEMAT_CALCULATION_HPP
