// ***!!!! Should modify revision manager to better use common code
// to determine inputs, determine output regions, and perform locking. 



#ifndef SNDE_NORMAL_CALCULATION_HPP
#define SNDE_NORMAL_CALCULATION_HPP


namespace snde {

  
  // extern opencl_program normalcalc_opencl_program;
  // note: we have a normal_calculation_trinormals
  // in normal_calculation.cpp but not currently any
  // way to access it

  std::shared_ptr<math_function> define_vertnormals_recording_function();


};
#endif // SNDE_NORMAL_CALCULATION_HPP
