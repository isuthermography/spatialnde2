// ***!!!! Should modify revision manager to better use common code
// to determine inputs, determine output regions, and perform locking. 

#include "opencl_utils.hpp"

#ifndef SNDE_BOXES_CALCULATION_HPP
#define SNDE_BOXES_CALCULATION_HPP


namespace snde {
  class geometry;
  class trm;
  class component;
  class parameterization;
  
  
  //extern opencl_program boxescalc_opencl_program;

// The snde::geometry's object_trees_lock should be held when making this call,
  // and it should be inside a revman transaction
  std::shared_ptr<trm_dependency> boxes_calculation_3d(std::shared_ptr<geometry> geom,std::shared_ptr<trm> revman,std::shared_ptr<component> comp,cl_context context,cl_device_id device,cl_command_queue queue);



  std::shared_ptr<trm_dependency> boxes_calculation_2d(std::shared_ptr<mutablewfmdb> wfmdb,std::string wfmdb_context,std::string wfmname,std::shared_ptr<geometry> geom,std::shared_ptr<trm> revman,std::shared_ptr<parameterization> param,snde_index patchnum,cl_context context,cl_device_id device,cl_command_queue queue);



};
#endif // SNDE_BOXES_CALCULATION_HPP
