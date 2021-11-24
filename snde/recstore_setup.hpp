#ifndef SNDE_RECSTORE_SETUP_HPP
#define SNDE_RECSTORE_SETUP_HPP

#include <memory>

#include "snde/recmath_compute_resource.hpp"
#include "snde/recstore.hpp"


namespace snde {

  void setup_cpu(std::shared_ptr<recdatabase> recdb,size_t nthreads);
  void setup_storage_manager(std::shared_ptr<recdatabase> recdb);


};

#endif // SNDE_RECSTORE_SETUP_HPP
