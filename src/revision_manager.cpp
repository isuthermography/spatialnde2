#include "revision_manager.hpp"

namespace snde {

   // destructor in .cpp file to avoid circular class dependency
  trm_dependency::~trm_dependency()
  {
    std::shared_ptr<trm> revman_strong=revman.lock();
    
    cleanup(metadata_inputs,inputs,outputs);
    
    revman_strong->_erase_dep_from_tree(weak_this,input_dependencies,output_dependencies);
    
   
  }
}
