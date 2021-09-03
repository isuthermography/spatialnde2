// ***!!!! Should modify revision manager to better use common code
// to determine inputs, determine output regions, and perform locking. 

#include "snde/opencl_utils.hpp"


#ifndef SNDE_NORMAL_CALCULATION_HPP
#define SNDE_NORMAL_CALCULATION_HPP


namespace snde {

  class geometry;
  class trm;
  class component;
  
extern opencl_program normalcalc_opencl_program;

// The snde::geometry's object_trees_lock should be held when making this call,
  // and it should be inside a revman transaction

  std::shared_ptr<trm_dependency> normal_calculation(std::shared_ptr<geometry> geom,std::shared_ptr<trm> revman,std::shared_ptr<component> comp,cl_context context,cl_device_id device,cl_command_queue queue);


};
#endif // SNDE_NORMAL_CALCULATION_HPP
