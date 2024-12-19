#ifndef SNDE_BAD_PIXEL_CORRECTION_HPP
#define SNDE_BAD_PIXEL_CORRECTION_HPP

#define SNDE_ACCUM_CONCATENATE 2
#define SNDE_ACCUM_NEW_AXIS 3
#define SNDE_ACCUM_NEW_NDARRAY 4

namespace snde {


  std::shared_ptr<math_function> define_bad_pixel_correction_function();
  
  extern SNDE_API std::shared_ptr<math_function> bad_pixel_correction_function;

};

#endif // SNDE_BAD_PIXEL_CORRECTION_HPP
