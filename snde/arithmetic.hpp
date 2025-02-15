#ifndef SNDE_ARITHMETIC_HPP
#define SNDE_ARITHMETIC_HPP


namespace snde {



  std::shared_ptr<math_function> define_addition_function();
  

  extern SNDE_OCL_API std::shared_ptr<math_function> addition_function;

};

#endif // SNDE_ARITHMETIC_HPP
