%{

  #include "snde/recstore_setup_opencl.hpp"
  
%}
namespace snde {
  class recdatabase;

  std::tuple<cl::Context,std::vector<cl::Device>> setup_opencl(std::shared_ptr<recdatabase> recdb,bool primary_doubleprec, size_t max_parallel, char *primary_platform_prefix_or_null);
  

};
