#ifndef SNDE_ACCUMULATE_ONCE_HPP
#define SNDE_ACCUMULATE_ONCE_HPP

#define SNDE_ACCUM_CONCATENATE 2
#define SNDE_ACCUM_NEW_AXIS 3
#define SNDE_ACCUM_NEW_NDARRAY 4

namespace snde {


  std::shared_ptr<math_function> define_accumulate_once_function();
  
  extern SNDE_API std::shared_ptr<math_function> accumulate_once_function;

};

#endif // SNDE_ACCUMULATE_ONCE_HPP
