// ***!!!! Should modify revision manager to better use common code
// to determine inputs, determine output regions, and perform locking. 

#include "snde/opencl_utils.hpp"


#ifndef SNDE_PROJINFO_CALCULATION_HPP
#define SNDE_PROJINFO_CALCULATION_HPP



// projectioninfo (projinfo) is the inplane2uvcoords and uvcoords2inplane members of the
// geometrydata structure
namespace snde {
  class geometry;
  class trm;
  class component;
  class part;
  class parameterization;
  
  
  //extern opencl_program projinfocalc_opencl_program;

// The snde::geometry's object_trees_lock should be held when making this call,
  // and it should be inside a revman transaction

  std::shared_ptr<trm_dependency> projinfo_calculation(std::shared_ptr<mutablerecdb> recdb,std::string recdb_context,std::string recname,std::shared_ptr<geometry> geom,std::shared_ptr<trm> revman,std::shared_ptr<part> partobj,std::shared_ptr<parameterization> param,cl_context context,cl_device_id device,cl_command_queue queue);


};
#endif // SNDE_PROJINFO_CALCULATION_HPP
