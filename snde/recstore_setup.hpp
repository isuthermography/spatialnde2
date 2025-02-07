#ifndef SNDE_RECSTORE_SETUP_HPP
#define SNDE_RECSTORE_SETUP_HPP

#include <memory>
#include <set>

#include "snde/recmath_compute_resource.hpp"
#include "snde/recstore.hpp"


namespace snde {

  void setup_cpu(std::shared_ptr<recdatabase> recdb,std::set<std::string> tags,size_t nthreads);
  void setup_storage_manager(std::shared_ptr<recdatabase> recdb);
  void setup_math_functions(std::shared_ptr<recdatabase> recdb,
			    std::vector<std::pair<std::string,std::shared_ptr<math_function>>> math_functions);


};

#endif // SNDE_RECSTORE_SETUP_HPP
