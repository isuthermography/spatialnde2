#ifndef SNDE_RECSTORE_SETUP_OPENCL_HPP
#define SNDE_RECSTORE_SETUP_OPENCL_HPP

#include <memory>

#include "snde/recmath_compute_resource.hpp"
#include "snde/recstore.hpp"


namespace snde {

  std::tuple<cl::Context,std::vector<cl::Device>> setup_opencl(std::shared_ptr<recdatabase> recdb,bool primary_doubleprec, size_t max_parallel, char *primary_platform_prefix_or_null);
  

};

#endif // SNDE_RECSTORE_SETUP_OPENCL_HPP
