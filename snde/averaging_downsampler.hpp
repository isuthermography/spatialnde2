#ifndef SNDE_AVERAGING_DOWNSAMPLER_HPP
#define SNDE_AVERAGING_DOWNSAMPLER_HPP

namespace snde {

  std::shared_ptr<math_function> define_averaging_downsampler_function();
  
  extern SNDE_API std::shared_ptr<math_function> averaging_downsampler_function;


};

#endif // SNDE_AVERAGING_DOWNSAMPLER_HPP
