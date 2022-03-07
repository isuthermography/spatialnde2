#ifndef SNDE_BATCHED_LIVE_ACCUMULATOR_HPP
#define SNDE_BATCHED_LIVE_ACCUMULATOR_HPP

namespace snde {

  std::shared_ptr<math_function> define_batched_live_accumulator_function();
  
  extern SNDE_API std::shared_ptr<math_function> batched_live_accumulator_function;


  // returns consistent_ndim, consistent_layout_c, consistent_layout_f,layout_dims,layout_length
  std::tuple<bool,bool,bool,std::vector<snde_index>,snde_index> analyze_potentially_batched_multi_ndarray_layout(std::shared_ptr<multi_ndarray_recording> array_rec);


};

#endif // SNDE_BATCHED_LIVE_ACCUMULATOR_HPP
