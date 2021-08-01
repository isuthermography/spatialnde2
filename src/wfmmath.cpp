#include "wfmmath.hpp"
#include "wfmstore.hpp"

namespace snde {
  math_function::math_function(size_t num_results,const std::list<std::tuple<std::string,unsigned>> &param_names_types) :
    num_results(num_results),
    param_names_types(param_names_types)
  {
    new_revision_optional=false;
    pure_optionally_mutable=false;
    mandatory_mutable=false;
    self_dependent=false; 
  }

  instantiated_math_function::instantiated_math_function() :
    disabled(false),
    ondemand(false),
    mdonly(false)
  {
    
  }
  
  std::shared_ptr<instantiated_math_function> instantiated_math_function::clone()
  {
    // This code needs to be repeated in every derived class with the make_shared<> referencing the derived class
    std::shared_ptr<instantiated_math_function> copy = std::make_shared<instantiated_math_function>(*this);

    // see comment at start of definition of class instantiated_math_function
    if (definition) {
      assert(!copy->original_function);
      copy->original_function = shared_from_this();
      copy->definition = nullptr; 
    }
    return copy;
  }

  // rebuild all_dependencies_of_channel hash table. Must be called any time any of the defined_math_functions changes. May only be called for the instantiated_math_database within the main waveform database, and the main waveform database admin lock must be locked when this is called. 
  void instantiated_math_database::rebuild_dependency_map()
  {

    all_dependencies_of_channel.clear();
    all_dependencies_of_function.clear();

    std::unordered_set<std::shared_ptr<instantiated_math_function>> all_fcns; 
    
    for (auto&& channame_math_fcn_ptr: defined_math_functions) {
      std::shared_ptr<instantiated_math_function> fcn_ptr = channame_math_fcn_ptr.second;

      all_fcns.emplace(fcn_ptr);
      for (auto && parameter_ptr: fcn_ptr->parameters) {

	std::set<std::string> prereqs = parameter_ptr->get_prerequisites(fcn_ptr->channel_path_context);
	for (auto && prereq: prereqs) {
	  auto adc_it = all_dependencies_of_channel.find(prereq);
	  if (adc_it == all_dependencies_of_channel.end()) {
	    // need to insert new element
	    adc_it = all_dependencies_of_channel.emplace(std::piecewise_construct,
							 std::forward_as_tuple(prereq),
							 std::forward_as_tuple()).first;
	    
	  }
	  // add this function as dependent on the prerequisite
	  adc_it->second.emplace(fcn_ptr);
	}
      }
    }

    for (auto && fcn_ptr: all_fcns) {
      // going through each math function, look up all result_channels
      auto adf_it = all_dependencies_of_function.find(fcn_ptr);
      assert(adf_it == all_dependencies_of_function.end()); // shouldn't be possible for this to already be populated
      adf_it = all_dependencies_of_function.emplace(std::piecewise_construct,
						    std::forward_as_tuple(fcn_ptr),
						    std::forward_as_tuple()).first;
      
      for (auto && resultpath_ptr : fcn_ptr->result_channel_paths) {
	if (resultpath_ptr) {
	  // for this result channel, look up all dependent functions
	  auto adc_it = all_dependencies_of_channel.find(*resultpath_ptr);
	  if (adc_it != all_dependencies_of_channel.end()) {
	    // iterate over all dependencies
	    for (auto && dep_fcn: adc_it->second) { // dep_fcn is a shared_ptr<instantiated_math_function>
	      // mark dep_fcn as dependent on fcn_ptr
	      adf_it->second.emplace(dep_fcn);
	    }
	  }
	}
      }
    }
    
  }

  math_function_status::math_function_status(bool mdonly,bool is_mutable) :
    mdonly(mdonly),
    is_mutable(is_mutable),
    ready_to_execute(false),
    execution_demanded(false),
    metadataonly_complete(false),
    complete(false)    
  {

  }
  
  math_status::math_status(std::shared_ptr<instantiated_math_database> math_functions,const std::map<std::string,channel_state> & channel_map) :
    math_functions(math_functions)
  {

    
    std::atomic_store(&_external_dependencies_on_function,std::make_shared<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>>>>());

    std::atomic_store(&_external_dependencies_on_channel,std::make_shared<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>>>>());

    
    // put all math functions into function_status and _external_dependencies databases and into pending_functions?
    for (auto && math_function_name_ptr: math_functions->defined_math_functions) {
      if (function_status.find(math_function_name_ptr.second) != function_status.end()) {
	function_status.emplace(std::piecewise_construct, // ***!!! Need to fill out prerequisites somewhere, but not sure constructor is the place!!!*** ... NO. prerequisites are filled out in end_transaction(). 
				std::forward_as_tuple(math_function_name_ptr.second),
				std::forward_as_tuple(math_function_name_ptr.second->mdonly,math_function_name_ptr.second->fcn->mandatory_mutable)); // For now we only set the mutable flag for mandatory_mutable functions. In the future we can also enable it when suitable for pure_optionally_mutable functions. 
	_external_dependencies_on_function->emplace(std::piecewise_construct,
						    std::forward_as_tuple(math_function_name_ptr.second),
						    std::forward_as_tuple());
	
	
	if (math_function_name_ptr.second->mdonly) {
	  mdonly_pending_functions.emplace(math_function_name_ptr.second);
	} else {
	  pending_functions.emplace(math_function_name_ptr.second);	  
	}
	
	
      }
    }

    for (auto && channel_path_channel_state: channel_map) {
      _external_dependencies_on_channel->emplace(std::piecewise_construct,
						 std::forward_as_tuple(channel_path_channel_state.second.config),
						 std::forward_as_tuple());
      
    }
  }
  
  void math_status::notify_math_function_executed(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<waveform_set_state> wfmstate,std::shared_ptr<instantiated_math_function> fcn,bool mdonly)
  {
    std::shared_ptr<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>>>> external_dependencies_on_function;
    
    {
      std::lock_guard<std::mutex> wss_admin(wfmstate->admin);
      math_function_status &our_status = wfmstate->mathstatus.function_status.at(fcn);
      
      // Find this fcn in matstatus [mdonly_]pending_functions and remove it, adding it to the completed block
      std::unordered_set<std::shared_ptr<instantiated_math_function>>::iterator pending_it;
      if (mdonly) {
	pending_it = wfmstate->mathstatus.mdonly_pending_functions.find(fcn);
	if (pending_it==wfmstate->mathstatus.mdonly_pending_functions.end()) {
	  throw snde_error("Math function %s executed mdonly without being in mdonly_pending_functions queue",fcn->definition->definition_command.c_str());
	}
	wfmstate->mathstatus.mdonly_pending_functions.erase(pending_it);
	wfmstate->mathstatus.completed_mdonly_functions.emplace(fcn);
      } else {
	pending_it = wfmstate->mathstatus.pending_functions.find(fcn);
	if (pending_it==wfmstate->mathstatus.pending_functions.end()) {
	  throw snde_error("Math function %s executed without being in pending_functions queue",fcn->definition->definition_command.c_str());
	}
	wfmstate->mathstatus.pending_functions.erase(pending_it);
	wfmstate->mathstatus.completed_functions.emplace(fcn);
      }
      
      our_status.mdonly_executed=true;
      if (mdonly && our_status.mdonly) {
	our_status.complete=true;
      }
      
      // look up anything dependent on this function's execution
      external_dependencies_on_function = wfmstate->mathstatus.external_dependencies_on_function();
    } // release lock

    
    std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>> ready_to_execute;

    // Search for external dependencies on this function; accumulate in ready_to_execute
    
    std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>>>::iterator ext_dep_it = external_dependencies_on_function->find(fcn);    
    assert(ext_dep_it != external_dependencies_on_function->end()); // should always have a vector, even if it's empty


    for (auto && wss_fcn: ext_dep_it->second) {
      std::shared_ptr<waveform_set_state> ext_dep_wss;
      std::shared_ptr<instantiated_math_function> ext_dep_fcn;

      std::tie(ext_dep_wss,ext_dep_fcn) = wss_fcn;
      std::lock_guard<std::mutex> dep_wss_admin(ext_dep_wss->admin);
      math_function_status &ext_dep_status = ext_dep_wss->mathstatus.function_status.at(ext_dep_fcn);

      std::set<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>>::iterator
	dependent_prereq_it = ext_dep_status.missing_external_function_prerequisites.find(std::make_tuple((std::shared_ptr<waveform_set_state>)wfmstate,(std::shared_ptr<instantiated_math_function>)fcn));
      if (dependent_prereq_it != ext_dep_status.missing_external_function_prerequisites.end()) {
	ext_dep_status.missing_external_function_prerequisites.erase(dependent_prereq_it);
      }
      ext_dep_wss->mathstatus.check_dep_fcn_ready(ext_dep_wss,ext_dep_fcn,&ext_dep_status,ready_to_execute);
      
    }

    // Queue computations from dependent functions
    
    for (auto && ready_wss_ready_fcn: ready_to_execute) {
      // Need to queue as a pending_computation
      std::shared_ptr<waveform_set_state> ready_wss;
      std::shared_ptr<instantiated_math_function> ready_fcn;

      std::tie(ready_wss,ready_fcn) = ready_wss_ready_fcn;
      wfmdb->compute_resources.queue_computation(ready_wss,ready_fcn);
    }


    // Go through the function's output channels and issue suitable notifications
    for (auto && result_channel_relpath: fcn->result_channel_paths) {
      std::string result_channel_path = wfmdb_path_join(fcn->channel_path_context,*result_channel_relpath);
      channel_state &chanstate = wfmstate->wfmstatus.channel_map.at(result_channel_path);
      chanstate.issue_math_notifications(wfmdb,wfmstate);
    }

      
    
  }

  void math_status::check_dep_fcn_ready(std::shared_ptr<waveform_set_state> dep_wss,
					std::shared_ptr<instantiated_math_function> dep_fcn,
					math_function_status *mathstatus_ptr,
					std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>> &ready_to_execute_appendvec)
    // assumes dep_wss admin lock is already held
  {

    if (!mathstatus_ptr->missing_prerequisites.size() &&
	!mathstatus_ptr->missing_external_channel_prerequisites.size() &&
	!mathstatus_ptr->missing_external_function_prerequisites.size()) {
      mathstatus_ptr->ready_to_execute=true;
      
      ready_to_execute_appendvec.emplace_back(std::make_tuple(dep_wss,dep_fcn));  // waveform_set_state,  instantiated_math_function
    }
    
  }
				  


};
