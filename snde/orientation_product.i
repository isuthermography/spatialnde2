%{
#include "snde/orientation_product.hpp"
%}


namespace snde {

  std::shared_ptr<math_function> define_spatialnde2_const_orientation_product_function();
  
  extern /* SNDE_API */ std::shared_ptr<math_function> const_orientation_product_function;

  std::shared_ptr<math_function> define_spatialnde2_orientation_const_product_function();
  
  extern /* SNDE_API */ std::shared_ptr<math_function> orientation_const_product_function;

  
  std::shared_ptr<math_function> define_spatialnde2_orientation_rec_product_function();
  
  extern /* SNDE_API */ std::shared_ptr<math_function> orientation_rec_product_function;

  std::shared_ptr<math_function> define_spatialnde2_pose_follower_function();
  extern /*SNDE_API*/ std::shared_ptr<math_function> pose_follower_function;

  %pythoncode %{
const_orientation_product = cvar.const_orientation_product_function  # make our swig-wrapped math_function accessible as 'spatialnde2.const_orientation_product'
orientation_const_product = cvar.orientation_const_product_function  # make our swig-wrapped math_function accessible as 'spatialnde2.orientation_const_product'
orientation_rec_product = cvar.orientation_rec_product_function  # make our swig-wrapped math_function accessible as 'spatialnde2.orientation_rec_product'
pose_follower = cvar.pose_follower_function  # make our swig-wrapped math_function accessible as 'spatialnde2.pose_follower'
  %}


};

