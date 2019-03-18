#include "revision_manager.hpp"

namespace snde {


  void trm_struct_depend_notifier::trm_notify()
  {
    std::shared_ptr<trm> recipient_strong(recipient);

    if (recipient_strong) {
      recipient_strong->mark_struct_depend_as_modified(key);
    }
  }

  // destructor in .cpp file to avoid circular class dependency
  trm_dependency::~trm_dependency()
  {
    std::shared_ptr<trm> revman_strong=revman.lock();
    
    cleanup(this);

    if (revman_strong) {
      revman_strong->_erase_dep_from_tree(weak_this,input_dependencies,output_dependencies);
    }
   
  }
}
