#ifndef SNDE_BATCHED_LIVE_ACCUMULATOR_HPP
#define SNDE_BATCHED_LIVE_ACCUMULATOR_HPP

namespace snde {

  std::shared_ptr<math_function> define_batched_live_accumulator_function();
  
  extern SNDE_API std::shared_ptr<math_function> batched_live_accumulator_function;


};

#endif // SNDE_BATCHED_LIVE_ACCUMULATOR_HPP
