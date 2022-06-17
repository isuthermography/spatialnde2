#ifndef SNDE_ORIENTATION_PRODUCT_HPP
#define SNDE_ORIENTATION_PRODUCT_HPP



namespace snde {
  
  std::shared_ptr<math_function> define_spatialnde2_const_orientation_product_function();
  extern SNDE_API std::shared_ptr<math_function> const_orientation_product_function;

  std::shared_ptr<math_function> define_spatialnde2_orientation_const_product_function();
  extern SNDE_API std::shared_ptr<math_function> orientation_const_product_function;


  
  std::shared_ptr<math_function> define_spatialnde2_orientation_rec_product_function();
  extern SNDE_API std::shared_ptr<math_function> orientation_rec_product_function;

  std::shared_ptr<math_function> define_spatialnde2_pose_follower_function();
  extern SNDE_API std::shared_ptr<math_function> pose_follower_function;
  

};

#endif // SNDE_ORIENTATION_PRODUCT_HPP
