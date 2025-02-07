%{

  #include "snde/recstore_setup.hpp"
  
%}
namespace snde {
  class recdatabase;

  void setup_cpu(std::shared_ptr<recdatabase> recdb,std::set<std::string> tags,size_t nthreads);
  void setup_storage_manager(std::shared_ptr<recdatabase> recdb);
  
  void setup_math_functions(std::shared_ptr<recdatabase> recdb,
			    std::vector<std::pair<std::string,std::shared_ptr<math_function>>> math_functions);

};



