#include "snde/revision_manager.hpp"

namespace snde {


  void trm_struct_depend_notifier::trm_notify()
  {
    std::shared_ptr<trm> recipient_strong(recipient);

    if (recipient_strong) {
      recipient_strong->mark_struct_depend_as_modified(key);
    }
  }

  bool trm_dependency::update_struct_inputs(const std::vector<trm_struct_depend> &new_struct_inputs)
    {
      // returns true if inputs are updated
      std::shared_ptr<trm> revman_strong=revman.lock();
      if (revman_strong) {
	std::lock_guard<std::recursive_mutex> dep_tbl(revman_strong->dependency_table_lock);
	
	
	if (new_struct_inputs.size() != struct_inputs.size()) {
	  struct_inputs = new_struct_inputs; 
	  return true; 
	}
	for (size_t cnt=0; cnt < new_struct_inputs.size();cnt++) {
	  // we are evaluating (struct_inputs[cnt].first != new_struct_inputs[cnt].first)
	  // but we do this with operator< because that is what is defined
	  // by trm_struct_depend_key 
	  if (struct_inputs[cnt].first < new_struct_inputs[cnt].first || new_struct_inputs[cnt].first < struct_inputs[cnt].first) {
	    struct_inputs = new_struct_inputs; 
	    return true; 
	    
	  }
	}
      }
      return false;
    }

  bool trm_dependency::update_struct_outputs(const std::vector<trm_struct_depend> &new_struct_outputs)
    {
      // returns true if inputs are updated
      std::shared_ptr<trm> revman_strong=revman.lock();
      if (revman_strong) {
	std::lock_guard<std::recursive_mutex> dep_tbl(revman_strong->dependency_table_lock);
	
	
	if (new_struct_outputs.size() != struct_outputs.size()) {
	  struct_outputs = new_struct_outputs; 
	  return true; 
	}
	for (size_t cnt=0; cnt < new_struct_outputs.size();cnt++) {
	  // we are evaluating (struct_outputs[cnt].first != new_struct_outputs[cnt].first)
	  // but we do this with operator< because that is what is defined
	  // by trm_struct_depend_key 
	  if (struct_outputs[cnt].first < new_struct_outputs[cnt].first || new_struct_outputs[cnt].first < struct_outputs[cnt].first) {
	    struct_outputs = new_struct_outputs; 
	    return true; 
	    
	  }
	}
      }
      return false;
    }

  bool trm_dependency::update_inputs(const std::vector<trm_arrayregion> &new_inputs)
    {
      // returns true if inputs are updated
      std::shared_ptr<trm> revman_strong=revman.lock();
      if (revman_strong) {
	std::lock_guard<std::recursive_mutex> dep_tbl(revman_strong->dependency_table_lock);
	if (new_inputs != inputs) {
	  inputs = new_inputs;
	  return true; 
	}
      }
      return false;
    }
    
  bool trm_dependency::update_outputs(const std::vector<trm_arrayregion> &new_outputs)
    {
      // returns true if inputs are updated
      std::shared_ptr<trm> revman_strong=revman.lock();
      if (revman_strong) {
	std::lock_guard<std::recursive_mutex> dep_tbl(revman_strong->dependency_table_lock);
	if (new_outputs != outputs) {
	  outputs = new_outputs;
	  return true; 
	}
      }
      return false;
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
