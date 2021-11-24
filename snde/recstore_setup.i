%{

  #include "snde/recstore_setup.hpp"
  
%}
namespace snde {
  class recdatabase;

  void setup_cpu(std::shared_ptr<recdatabase> recdb,size_t nthreads);
  void setup_storage_manager(std::shared_ptr<recdatabase> recdb);
  

};



