#ifndef SNDE_MEAN_HPP
#define SNDE_MEAN_HPP

namespace snde {

  std::shared_ptr<math_function> define_mean_function();
  
  extern SNDE_API std::shared_ptr<math_function> mean_function;


};

#endif // SNDE_MEAN_HPP
