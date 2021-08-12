#include "wfmmath_compute_resource.hpp"
#include "wfmstore.hpp"
#include "wfmmath.hpp"

namespace snde {
  compute_resource_option::compute_resource_option(unsigned type, size_t metadata_bytes,size_t data_bytes,std::shared_ptr<void> parameter_block) :
    type(type),
    metadata_bytes(metadata_bytes),
    data_bytes(data_bytes),
    parameter_block(parameter_block)
  {

  }


  compute_resource_option_cpu::compute_resource_option_cpu(unsigned type,
							   size_t metadata_bytes,
							   size_t data_bytes,
							   std::shared_ptr<void> parameter_block,
							   snde_float64 flops,
							   size_t max_effective_cpu_cores,
							   size_t useful_cpu_cores) :
    compute_resource_option(type,metadata_bytes,data_bytes,parameter_block),
    flops(flops),
    max_effective_cpu_cores(max_effective_cpu_cores),
    useful_cpu_cores(useful_cpu_cores)
  {

  }

#ifdef SNDE_OPENCL
  compute_resource_option_opencl::compute_resource_option_opencl(unsigned type,
								 size_t metadata_bytes,
								 size_t data_bytes,
								 std::shared_ptr<void> parameter_block,
								 snde_float64 cpu_flops,
								 snde_float64 gpu_flops,
								 size_t max_effective_cpu_cores,
								 size_t max_useful_cpu_cores) :
    compute_resource_option(type,metadata_bytes,data_bytes,parameter_block),
    cpu_flops(cpu_flops),
    gpu_flops(gpu_flops),
    max_effective_cpu_cores(max_effective_cpu_cores),
    max_useful_cpu_cores(max_useful_cpu_cores)
  {

  }
#endif // SNDE_OPENCL
  
  available_compute_resource_database::available_compute_resource_database() :
    admin(std::make_shared<std::mutex>())
  {

  }

  
  void available_compute_resource_database::_queue_computation_internal(std::shared_ptr<pending_computation> &computation) // NOTE: Sets computation to nullptr once queued
  {

    bool computation_wss_is_globalrev = false; 
    // we cheat a bit in getting the globalrev index: We try dynamically casting ready_wss, and if that fails we use the value from the prerequisite state
    std::shared_ptr<globalrevision> globalrev_ptr = std::dynamic_pointer_cast<globalrevision>(computation->wfmstate);
    if (!globalrev_ptr) {
      // try prerequisite_state
      globalrev_ptr = std::dynamic_pointer_cast<globalrevision>(computation->wfmstate->prerequisite_state());

      if (!globalrev_ptr) {
	throw snde_error("waveform_set_state does not appear to be associated with any global revision");
      }      
    } else {
      computation_wss_is_globalrev=true; 
    }


    
    // get the compute options
    //std::list<std::shared_ptr<compute_resource_option>> compute_options = computation->function_to_execute->get_compute_options();
    std::list<std::shared_ptr<compute_resource_option>> compute_options = computation->function_to_execute->execution_tracker->perform_compute_options();

    
    // we can execute anything immutable, anything that is not part of a globalrev (i.e. ondemand), or once the prior globalrev is fully ready
    // (really we just need to make sure all immutable waveforms in the prior globalrev are ready, but there isn't currently a good
    // way to do that)
    std::shared_ptr<waveform_set_state> prior_globalrev=computation->wfmstate->prerequisite_state(); // only actually prior_globalrev if computation_wss_is_globalrev


    std::lock_guard<std::mutex> acrdb_admin(*admin);

    // ***!!!! Really here we need to see if we want to run it in
    // mutable or immutable mode !!!***
    //if (!computation->function_to_execute->inst->fcn->mandatory_mutable || !computation_wss_is_globalrev || !prior_globalrev || prior_globalrev->ready) {
    
    if (!computation->function_to_execute->is_mutable || !computation_wss_is_globalrev || !prior_globalrev || prior_globalrev->ready) {
      todo_list.emplace(computation);
      
      // This is a really dumb loop that just assigns all matching resources
      std::shared_ptr <available_compute_resource> selected_resource;
      std::shared_ptr <compute_resource_option> selected_option;
      for (auto && compute_resource: compute_resources) { // compute_resource is a shared_ptr<available_compute_resource>
	for (auto && compute_option: compute_options) { // compute_option is a shared_ptr<compute_resource_option>
	  if (compute_option->type == compute_resource->type) {
	    selected_resource = compute_resource;
	    selected_option = compute_option;
	    
	    selected_resource->prioritized_computations.emplace(std::make_pair(globalrev_ptr->globalrev,
									       std::make_tuple(std::weak_ptr<pending_computation>(computation),selected_option)));
	    //compute_resource_lock.unlock();  (???What was this supposed to do???)
	    //selected_resource->computations_added.notify_one();
	    selected_resource->notify_acr_of_changes_to_prioritized_computations();
	  }	  
	}

      }
      
      if (!selected_resource) {
	throw snde_error("No suitable compute resource found for math function %s",computation->function_to_execute->inst->definition->definition_command.c_str());
      }
      
      
    } else {
      // blocked... we have to wait for previous revision to at least
      // complete its mutable waveforms !!!*** NEED PROCESS FOR TRIGGERING REMOVAL FROM BLOCKED LIST
      blocked_list.emplace(globalrev_ptr->globalrev,computation);
    }
    computation=nullptr; // release shared pointer prior to releasing acrdb_admin lock. 
  }


  void available_compute_resource_database::queue_computation(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<waveform_set_state> ready_wss,std::shared_ptr<instantiated_math_function> ready_fcn)
  {

    bool ready_wss_is_globalrev = false; 
    // we cheat a bit in getting the globalrev index: We try dynamically casting ready_wss, and if that fails we use the value from the prerequisite state
    std::shared_ptr<globalrevision> globalrev_ptr = std::dynamic_pointer_cast<globalrevision>(ready_wss);
    if (!globalrev_ptr) {
      // try prerequisite_state
      globalrev_ptr = std::dynamic_pointer_cast<globalrevision>(ready_wss->prerequisite_state());
      
      if (!globalrev_ptr) {
	throw snde_error("waveform_set_state does not appear to be associated with any global revision");
      }      
    } else {
      ready_wss_is_globalrev=true; 
    }

    bool is_mutable=false;
    bool mdonly;
    std::shared_ptr<math_function_execution> execfunc;
    {
      std::unique_lock<std::mutex> ready_wss_admin(ready_wss->admin);
      math_function_status &ready_wss_status = ready_wss->mathstatus.function_status.at(ready_fcn);
      execfunc = ready_wss_status.execfunc;
      if (ready_wss_status.ready_to_execute && execfunc->try_execution_ticket()) {
	// we are taking care of execution
	assert(ready_wss_status.execution_demanded); 
	ready_wss_status.ready_to_execute=false;
	ready_wss_admin.unlock();


	execfunc->execution_tracker = ready_fcn->fcn->initiate_execution(ready_wss,ready_fcn);
	bool actually_execute=true;
	
	// Check new_revision_optional
	if (ready_fcn->fcn->new_revision_optional) {
	  // Need to check if it s OK to execute
	  
	  actually_execute = execfunc->execution_tracker->perform_decide_execution();
	}
	
	if (!actually_execute) {
	  // because new_revision_optional and mdonly are incompatible (see comment in instantiated_math_function
	  // constructor, the prior revision deriving from our self-dependency must be fully ready, and since
	  // we're not executing, so are we. 

	  // grab waveform results from execfunc->execution_tracker->self_dependent_waveforms
	  assert(ready_fcn->result_channel_paths.size()==execfunc->execution_tracker->self_dependent_waveforms.size());

	  // assign waveforms to all referencing wss waveforms (should all still exist)
	  for (auto && referencing_wss_weak: execfunc->referencing_wss) {
	    std::shared_ptr<waveform_set_state> referencing_wss_strong(referencing_wss_weak);
	    
	    for (size_t cnt=0;cnt < ready_fcn->result_channel_paths.size(); cnt++) {
	      assert(!referencing_wss_strong->wfmstatus.channel_map.at(*ready_fcn->result_channel_paths.at(cnt)).wfm());
	      
	      referencing_wss_strong->wfmstatus.channel_map.at(*ready_fcn->result_channel_paths.at(cnt)).end_atomic_wfm_update(execfunc->execution_tracker->self_dependent_waveforms.at(cnt));
	    }
	  }
	  
	  // mark math_status and execfunc as complete
	  execfunc->metadata_executed = true;
	  execfunc->fully_complete = true;
	  {
	    std::lock_guard<std::mutex> ready_wss_admin(ready_wss->admin);
	    math_function_status &ready_wss_status = ready_wss->mathstatus.function_status.at(ready_fcn);
	    
	    ready_wss_status.complete=true;
	  }
	  
	  // clear execfunc->wss to eliminate reference loop once at least metadata execution is complete
	  execfunc->wss = nullptr;
	  execfunc->execution_tracker->wss = nullptr;
	  execfunc->executing = false; // release the execution ticket

	  
	  // issue notifications

	  // ... in all referencing wss's
	  for (auto && referencing_wss_weak: execfunc->referencing_wss) {
	    std::shared_ptr<waveform_set_state> referencing_wss_strong(referencing_wss_weak);
	    
	    for (auto && result_channel_path_ptr: ready_fcn->result_channel_paths) {
	      channel_state &chanstate = referencing_wss_strong->wfmstatus.channel_map.at(*result_channel_path_ptr);
	      //chanstate.issue_math_notifications(wfmdb,ready_wss); // taken care of by notify_math_function_executed(), below
	      chanstate.issue_nonmath_notifications(referencing_wss_strong);
	      
	    }
	    
	    // Issue function completion notification
	    referencing_wss_strong->mathstatus.notify_math_function_executed(wfmdb,referencing_wss_strong,execfunc->inst,false); // note mdonly hardwired to false here
	  }
	  
	  
	  return;
	}
	std::shared_ptr<pending_computation> computation = std::make_shared<pending_computation>(execfunc,ready_wss,globalrev_ptr->globalrev,0);
	
	
	_queue_computation_internal(computation); // this call sets computation to nullptr; execution ticket delegated to the queued computation
	
	
      } else {
	// Somebody else is taking care of computation
	
	// no need to do anything; just return
      }
      // warning: ready_wss_admin lock may or may not be held here 
    }
    
  }

  void available_compute_resource_database::start()
  // start all of the compute_resources
  {
    std::list<std::shared_ptr<available_compute_resource>> compute_resources_copy;
    {
      std::lock_guard<std::mutex> acrd_admin(*admin);
      
      compute_resources_copy = compute_resources;
    }

    for (auto && compute_resource: compute_resources_copy) {
      compute_resource->start();
    }

  }

  
  pending_computation::pending_computation(std::shared_ptr<math_function_execution> function_to_execute,std::shared_ptr<waveform_set_state> wfmstate,uint64_t globalrev,uint64_t priority_reduction) :
    function_to_execute(function_to_execute),
    wfmstate(wfmstate),
    globalrev(globalrev),
    priority_reduction(priority_reduction)
  {

  }


  available_compute_resource::available_compute_resource(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<available_compute_resource_database> acrd,unsigned type) :
    wfmdb(wfmdb),
    acrd_admin(acrd->admin),
    type(type)
  {
    
  }

  available_compute_resource_cpu::available_compute_resource_cpu(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<available_compute_resource_database> acrd,unsigned type,size_t total_cpu_cores_available) :
    available_compute_resource(wfmdb,acrd,type),
    total_cpu_cores_available(total_cpu_cores_available)
  {

  }

  void available_compute_resource_cpu::start()
  {
    
    size_t cnt;

    std::lock_guard<std::mutex> acrd_lock(*acrd_admin); // because threads will start, and we need to lock them out while we fill up the vector data structures
    for (cnt=0; cnt < total_cpu_cores_available;cnt++) {
      functions_using_cores.push_back(nullptr);
      thread_triggers.emplace_back(std::make_shared<std::condition_variable>());
      thread_actions.push_back(std::make_tuple((std::shared_ptr<waveform_set_state>)nullptr,(std::shared_ptr<math_function_execution>)nullptr,(std::shared_ptr<compute_resource_option_cpu>)nullptr));
      available_threads.emplace_back(std::thread([this](size_t n){ pool_code(n); },cnt));

      available_threads.at(cnt).detach(); // we won't be join()ing these threads
      
    }

    // instantiate dispatch thread
    dispatcher_thread = std::thread([this]() { dispatch_code(); });
    dispatcher_thread.detach(); // we won't be join()ing this thread
  }
  
  void available_compute_resource_cpu::notify_acr_of_changes_to_prioritized_computations() // should be called WITH ACRD's admin lock held
  {
    computations_added_or_completed.notify_one();
  }

  size_t available_compute_resource_cpu::_number_of_free_cpus()
  // Must call with ACRD admin lock locked
  {
    size_t number_of_free_cpus=0;

    for (auto && exec_fcn: functions_using_cores) {
      if (!exec_fcn) {
	number_of_free_cpus++;
      }
    }
    return number_of_free_cpus;
  }

  std::shared_ptr<assigned_compute_resource_cpu> available_compute_resource_cpu::_assign_cpus(std::shared_ptr<math_function_execution> function_to_execute,size_t number_of_cpus)
  // called with acrd admin lock held
  {
    size_t cpu_index=0;
    std::vector<size_t> cpu_assignments; 
    for (auto && exec_fcn: functions_using_cores) {
      if (!exec_fcn) {
	// this cpu is available
	cpu_assignments.push_back(cpu_index);
	//function_to_execute->cpu_cores.push_back(cpu_index);
	number_of_cpus--;

	if (!number_of_cpus) {
	  break;
	}
      }
      cpu_index++;
    }
    assert(!number_of_cpus); // should have been able to assign everything
    
    return std::make_shared<assigned_compute_resource_cpu>(cpu_assignments);

  }

  void available_compute_resource_cpu::_dispatch_thread(std::shared_ptr<waveform_set_state> wfmstate,std::shared_ptr<math_function_execution> function_to_execute,std::shared_ptr<compute_resource_option_cpu> compute_option_cpu,size_t first_thread_index)
  // Must be called with acrd_admin lock held
  {
    //printf("_dispatch_thread()!\n");

    
    //std::lock_guard<std::mutex> admin_lock(*acrd_admin); // lock assumed to be already held
      
    // assign ourselves to functions_using_cores;
    for (auto && core_index: (std::dynamic_pointer_cast<assigned_compute_resource_cpu>(function_to_execute->execution_tracker->compute_resource)->assigned_cpu_core_indices)) {
      assert(functions_using_cores.at(core_index)==nullptr);
      functions_using_cores.at(core_index) = function_to_execute; 
    }
    
    std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<math_function_execution>,std::shared_ptr<compute_resource_option_cpu>> & this_thread_action = thread_actions.at(first_thread_index);
    assert(this_thread_action==std::make_tuple((std::shared_ptr<waveform_set_state>)nullptr,(std::shared_ptr<math_function_execution>)nullptr,(std::shared_ptr<compute_resource_option_cpu>)nullptr));
    
    this_thread_action = std::make_tuple(wfmstate,function_to_execute,compute_option_cpu);
    
    
  
    //printf("triggering thread %d\n",first_thread_index);
    thread_triggers.at(first_thread_index)->notify_one();
  }
  
  void available_compute_resource_cpu::dispatch_code()
  // Does this really need to be its own thread? Maybe not...
  {
    bool no_actual_dispatch = false;
    std::shared_ptr<available_compute_resource> acr_keepinmemory=shared_from_this();  // this shared_ptr prevents the available_compute_resource object from getting released, which would make "this" become invalid under us
    while(true) {
      
      std::unique_lock<std::mutex> admin_lock(*acrd_admin);
      if (no_actual_dispatch) {
	computations_added_or_completed.wait(admin_lock);
      }

      //printf("Dispatch!\n");
      no_actual_dispatch = true; 
      if (prioritized_computations.size() > 0) {
	std::multimap<uint64_t,std::tuple<std::weak_ptr<pending_computation>,std::shared_ptr<compute_resource_option>>>::iterator this_computation_it = prioritized_computations.begin();
	std::weak_ptr<pending_computation> this_computation_weak;
	std::shared_ptr<compute_resource_option> compute_option;

	std::tie(this_computation_weak,compute_option) = this_computation_it->second;
	std::shared_ptr<pending_computation> this_computation = this_computation_weak.lock();
	if (!this_computation) {
	  // pointer expired; computation has been handled elsewhere
	  prioritized_computations.erase(this_computation_it); // remove from our list
	  no_actual_dispatch = false; // removing from prioritized_computations counts as an actual dispatch
	} else {
	  // got this_computation and compute_option to possibly try.
	  // Check if we have enough cores available for compute_option
	  std::shared_ptr<compute_resource_option_cpu> compute_option_cpu=std::dynamic_pointer_cast<compute_resource_option_cpu>(compute_option);
	  // this had better be one of our pointers...
	  assert(compute_option_cpu);

	  // For now, just blindly use the useful # of cpu cores
	  size_t free_cores = _number_of_free_cpus();
	  
	  if (compute_option_cpu->useful_cpu_cores <= free_cores || free_cores == total_cpu_cores_available) {	    
	    // we have enough cores available (or all of them)
	    std::shared_ptr<math_function_execution> function_to_execute=this_computation->function_to_execute;
	    std::shared_ptr<waveform_set_state> wfmstate=this_computation->wfmstate;

	    prioritized_computations.erase(this_computation_it); // take charge of this computation
	    this_computation = nullptr; // force pointer to expire so nobody else tries this computation;

	    std::shared_ptr<assigned_compute_resource_cpu> assigned_cpus = _assign_cpus(function_to_execute,std::min(compute_option_cpu->useful_cpu_cores,total_cpu_cores_available));
	    function_to_execute->execution_tracker->compute_resource = assigned_cpus;
	    
	    _dispatch_thread(wfmstate,function_to_execute,compute_option_cpu,assigned_cpus->assigned_cpu_core_indices.at(0));


	    no_actual_dispatch = false; // dispatched execution, so don't wait at next iteration. 
	  }

	  this_computation = nullptr; // remove all references to pending_computation before we release the lock
	}
      }
      
    }
  }

  void available_compute_resource_cpu::pool_code(size_t threadidx)
  {
    std::shared_ptr<available_compute_resource> acr_keepinmemory=shared_from_this();  // this shared_ptr prevents the available_compute_resource object from getting released, which would make "this" become invalid under us
    
    //printf("pool_code_startup!\n");
    while(true) {
      
      std::unique_lock<std::mutex> admin_lock(*acrd_admin);
      thread_triggers.at(threadidx)->wait(admin_lock);
      //printf("pool_code_wakeup!\n");

      std::shared_ptr<waveform_set_state> wfmstate;
      std::shared_ptr<math_function_execution> func;
      std::shared_ptr<compute_resource_option_cpu> compute_option_cpu;
      
      std::tie(wfmstate,func,compute_option_cpu) = thread_actions.at(threadidx);
      if (wfmstate) {
	//printf("Pool code got thread action\n");
	// Remove parameters from thread_actions
	thread_actions.at(threadidx)=std::make_tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<math_function_execution>,std::shared_ptr<compute_resource_option_cpu>>(nullptr,nullptr,nullptr);

	// not worrying about cpu affinity yet.


	// Need also our assigned_compute_resource (func->compute_resource)
	admin_lock.unlock();

	// will eventually need the lock manager from somewhere(!!!***???)
	bool mdonly = func->mdonly;
	//bool is_mutable;
	//bool mdonly_executed;

	//// Get the mdonly and is_mutable flags from the math_function_status 
	//{
	//std::lock_guard<std::mutex> wss_admin(wfmstate->admin);

	  
	  
	//math_function_status &our_status = wfmstate->mathstatus.function_status.at(func->inst);
	//  mdonly = our_status.mdonly;
	//  is_mutable = our_status.is_mutable;
	// mdonly_executed = our_status.mdonly_executed; 
	//
	//}
	
	if (!func->metadata_executed) {
	  func->execution_tracker->perform_define_wfms();

	  // grab generated waveforms and move them from defined_waveforms to instantiated_waveforms list
	  {
	    std::lock_guard<std::mutex> wssadmin(func->wss->admin);
	    for (auto && result_channel_path_ptr: func->inst->result_channel_paths) {
	      if (result_channel_path_ptr) {
		
		channel_state &chanstate = func->wss->wfmstatus.channel_map.at(*result_channel_path_ptr);
		if (chanstate.wfm()) {
		  // have a waveform
		  size_t erased_from_defined_waveforms = func->wss->wfmstatus.defined_waveforms.erase(chanstate.config);
		  
		  assert(erased_from_defined_waveforms==1); // waveform should have been listed in defined_waveforms
		  func->wss->wfmstatus.instantiated_waveforms.emplace(chanstate.config,&chanstate);
		} else {
		  // ***!!! If the math function failed to define
		  // a waveform here, we should probably import
		  // from the previous globalrev !!!***
		}
	      }
	    }
	  }
	  
	  func->execution_tracker->perform_metadata();
	  func->metadata_executed=true;
	}

	if (mdonly) {
	  func->metadataonly_complete = true; 
	  
	  // assign waveforms to all referencing wss waveforms (should all still exist)
	  // (only needed if we're not doing it below)
	  // !!!*** This block should be refactored
	  for (auto && referencing_wss_weak: func->referencing_wss) {
	    std::shared_ptr<waveform_set_state> referencing_wss_strong(referencing_wss_weak);
	    if (referencing_wss_strong == func->wss) {
	      continue; // no need to reassign our source
	    }
	    
	    for (size_t cnt=0;cnt < func->inst->result_channel_paths.size(); cnt++) {
	      // none should have preassigned waveforms --not true because parallel update or end_transaction() could be going on
	      //assert(!referencing_wss->wfmstatus.channel_map.at(&result_channel_paths.at(cnt)).wfm());
	      
	      referencing_wss_strong->wfmstatus.channel_map.at(*func->inst->result_channel_paths.at(cnt)).end_atomic_wfm_update(func->wss->wfmstatus.channel_map.at(*func->inst->result_channel_paths.at(cnt)).wfm());
	    }
	  }
	}

	if (!mdonly) {
	  func->execution_tracker->perform_lock_alloc();
	  func->execution_tracker->perform_exec();

	  func->fully_complete = true;


	  // re-assign waveforms to all referencing wss waveforms (should all still exist) in case we have more followers than before
	  // !!!*** This block should be refactored
	  for (auto && referencing_wss_weak: func->referencing_wss) {
	    std::shared_ptr<waveform_set_state> referencing_wss_strong(referencing_wss_weak);
	    if (referencing_wss_strong == func->wss) {
	      continue; // no need to reassign our source
	    }
	    
	    for (size_t cnt=0;cnt < func->inst->result_channel_paths.size(); cnt++) {
	      // none should have preassigned waveforms --not true because parallel update or end_transaction() could be going on
	      //assert(!referencing_wss->wfmstatus.channel_map.at(&result_channel_paths.at(cnt)).wfm());
	      
	      referencing_wss_strong->wfmstatus.channel_map.at(*func->inst->result_channel_paths.at(cnt)).end_atomic_wfm_update(func->wss->wfmstatus.channel_map.at(*func->inst->result_channel_paths.at(cnt)).wfm());
	    }
	  }
	  
	}
	
	
	// Mark execution as no longer ongoing in the math_function_status 
	{
	  std::unique_lock<std::mutex> wss_admin(wfmstate->admin);
	  std::unique_lock<std::mutex> func_admin(func->admin);
	  
	  math_function_status &our_status = wfmstate->mathstatus.function_status.at(func->inst);
	  if (mdonly && !func->mdonly) {
	    // execution status changed from mdonly to non-mdonly behind our back...
	    mdonly=false;
	    // finish up execution before we mark as finished
	    func_admin.unlock();
	    wss_admin.unlock();

	    func->execution_tracker->perform_lock_alloc();
	    func->execution_tracker->perform_exec();

	    func->fully_complete = true;
	    
	    // re-assign waveforms to all referencing wss waveforms (should all still exist) in case we have more followers than before
	    // !!!*** This block should be refactored
	    for (auto && referencing_wss_weak: func->referencing_wss) {
	      std::shared_ptr<waveform_set_state> referencing_wss_strong(referencing_wss_weak);
	      if (referencing_wss_strong == func->wss) {
		continue; // no need to reassign our source
	      }
	      
	      for (size_t cnt=0;cnt < func->inst->result_channel_paths.size(); cnt++) {
		// none should have preassigned waveforms --not true because parallel update or end_transaction() could be going on
		//assert(!referencing_wss->wfmstatus.channel_map.at(&result_channel_paths.at(cnt)).wfm());
		
		referencing_wss_strong->wfmstatus.channel_map.at(*func->inst->result_channel_paths.at(cnt)).end_atomic_wfm_update(func->wss->wfmstatus.channel_map.at(*func->inst->result_channel_paths.at(cnt)).wfm());
	      }
	    }
	    
	    
	    
	    wss_admin.lock();
	    func_admin.lock();
	  }
	  our_status.complete=true;

	}

	// clear execfunc->wss to eliminate reference loop once at least metadata execution is complete for an mdonly waveform
	// or once all execution is complete for a regular waveform
	if (func->metadataonly_complete || func->fully_complete) {
	  func->wss = nullptr;
	  func->execution_tracker->wss = nullptr;
	}

	func->executing = false; // release the execution ticket

	std::shared_ptr<wfmdatabase> wfmdb_strong = wfmdb.lock();

	// Need to do notifications that the math function finished in all referencing wss's
	for (auto && referencing_wss_weak: func->referencing_wss) {
	  std::shared_ptr<waveform_set_state> referencing_wss_strong(referencing_wss_weak);
	  
	  for (auto && result_channel_path_ptr: func->inst->result_channel_paths) {
	    channel_state &chanstate = referencing_wss_strong->wfmstatus.channel_map.at(*result_channel_path_ptr);
	    //chanstate.issue_math_notifications(wfmdb,ready_wss); // taken care of by notify_math_function_executed(), below
	    chanstate.issue_nonmath_notifications(referencing_wss_strong);
	    
	  }
	    
	  //printf("Pool code completed math function\n");
	  //fflush(stdout);
	  // Issue function completion notification
	  if (wfmdb_strong) {
	    referencing_wss_strong->mathstatus.notify_math_function_executed(wfmdb_strong,referencing_wss_strong,func->inst,mdonly); 
	  }
	}

	//printf("Pool code completed notification\n");
	//fflush(stdout);
	// Completion notification:
	//  * removing ourselves from functions_using_cores and triggering computations_added_or_completed
	admin_lock.lock();
	
	// remove ourselves from functions_using_cores
	for (size_t corenum=0;corenum < total_cpu_cores_available;corenum++) {
	  if (functions_using_cores.at(corenum) == func) {
	    functions_using_cores.at(corenum) = nullptr; 
	  }
	}
	// Notify that we are done
	notify_acr_of_changes_to_prioritized_computations();
      }
      
    }
  }

#ifdef SNDE_OPENCL
  available_compute_resource_opencl::available_compute_resource_opencl(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<available_compute_resource_database> acrd,unsigned type,cl_context opencl_context,cl_device_id *opencl_devices,size_t num_devices,size_t max_parallel) :
    available_compute_resource(wfmdb,acrd,type),
    opencl_context(opencl_context),
    opencl_devices(opencl_devices),
    num_devices(num_devices),
    max_parallel(max_parallel)

  {
    
  }

  void available_compute_resource_opencl::start() // set the compute resource going
  {
    assert(0);
  }
#endif // SNDE_OPENCL

  assigned_compute_resource::assigned_compute_resource(unsigned type) :
    type(type)
  {
    
  }
  
  assigned_compute_resource_cpu::assigned_compute_resource_cpu(const std::vector<size_t> &assigned_cpu_core_indices) :
    assigned_compute_resource(SNDE_CR_CPU),
    assigned_cpu_core_indices(assigned_cpu_core_indices)
  {
    
  }

#ifdef SNDE_OPENCL
  assigned_compute_resource_opencl::assigned_compute_resource_opencl(const std::vector<size_t> &assigned_cpu_core_indices,const std::vector<size_t> &assigned_opencl_job_indices,cl_context opencl_context,cl_device_id opencl_device) :
    assigned_compute_resource(SNDE_CR_OPENCL),
    assigned_cpu_core_indices(assigned_cpu_core_indices),
    assigned_opencl_job_indices(assigned_opencl_job_indices),
    opencl_context(opencl_context),
    opencl_device(opencl_device)
    
  {
    
  }
#endif //SNDE_OPENCL
};
