%{
#include "snde/orientation_product.hpp"
%}


namespace snde {

  std::shared_ptr<math_function> define_spatialnde2_orientation_const_product_function();
  
  extern /* SNDE_API */ std::shared_ptr<math_function> orientation_const_product_function;

  std::shared_ptr<math_function> define_spatialnde2_orientation_rec_product_function();
  
  extern /* SNDE_API */ std::shared_ptr<math_function> orientation_rec_product_function;

  %pythoncode %{
orientation_const_product = cvar.orientation_const_product_function  # make our swig-wrapped math_function accessible as 'spatialnde2.orientation_const_product'
orientation_rec_product = cvar.orientation_rec_product_function  # make our swig-wrapped math_function accessible as 'spatialnde2.orientation_rec_product'
  %}


};

