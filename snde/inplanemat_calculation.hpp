#ifndef SNDE_INPLANEMAT_CALCULATION_HPP
#define SNDE_INPLANEMAT_CALCULATION_HPP


namespace snde {
  
  std::shared_ptr<math_function> define_spatialnde2_inplanemat_calculation_function();

  // NOTE: Change to SNDE_OCL_API if/when we add GPU acceleration support, and
  // (in CMakeLists.txt) make it move into the _ocl.so library)
  extern SNDE_API std::shared_ptr<math_function> inplanemat_calculation_function;

  void instantiate_inplanemat(std::shared_ptr<recdatabase> recdb,std::shared_ptr<loaded_part_geometry_recording> loaded_geom);

};
#endif // SNDE_INPLANEMAT_CALCULATION_HPP
