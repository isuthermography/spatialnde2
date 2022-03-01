// ***!!!! Should modify revision manager to better use common code
// to determine inputs, determine output regions, and perform locking. 



#ifndef SNDE_NORMAL_CALCULATION_HPP
#define SNDE_NORMAL_CALCULATION_HPP


namespace snde {

  

  std::shared_ptr<math_function> define_spatialnde2_trinormals_function();
  SNDE_OCL_API extern std::shared_ptr<math_function> trinormals_function;
  
  std::shared_ptr<math_function> define_vertnormals_recording_function();
  SNDE_OCL_API extern std::shared_ptr<math_function> vertnormals_recording_function;
  
};
#endif // SNDE_NORMAL_CALCULATION_HPP
