#include "snde/recmath_compute_resource.hpp"
#include "snde/recstore.hpp"
#include "snde/recmath.hpp"

namespace snde {
  compute_resource_option::compute_resource_option(unsigned type, size_t metadata_bytes,size_t data_bytes) :
    type(type),
    metadata_bytes(metadata_bytes),
    data_bytes(data_bytes)
  {

  }


  compute_resource_option_cpu::compute_resource_option_cpu(size_t metadata_bytes,
							   size_t data_bytes,
							   snde_float64 flops,
							   size_t max_effective_cpu_cores,
							   size_t useful_cpu_cores) :
    compute_resource_option(SNDE_CR_CPU,metadata_bytes,data_bytes),
    flops(flops),
    max_effective_cpu_cores(max_effective_cpu_cores),
    useful_cpu_cores(useful_cpu_cores)
  {

  }

#ifdef SNDE_OPENCL
  compute_resource_option_opencl::compute_resource_option_opencl(size_t metadata_bytes,
								 size_t data_bytes,
								 snde_float64 cpu_flops,
								 snde_float64 gpu_flops,
								 size_t max_effective_cpu_cores,
								 size_t useful_cpu_cores) :
    compute_resource_option(SNDE_CR_OPENCL,metadata_bytes,data_bytes),
    cpu_flops(cpu_flops),
    gpu_flops(gpu_flops),
    max_effective_cpu_cores(max_effective_cpu_cores),
    useful_cpu_cores(useful_cpu_cores)
  {

  }
#endif // SNDE_OPENCL


  _compute_resource_option_cpu_combined::_compute_resource_option_cpu_combined(size_t metadata_bytes,
									       size_t data_bytes,
									       snde_float64 flops,
									       size_t max_effective_cpu_cores,
									       size_t useful_cpu_cores,
									       std::shared_ptr<compute_resource_option> orig,
									       std::shared_ptr<assigned_compute_resource> orig_assignment) :
    compute_resource_option_cpu(metadata_bytes,data_bytes,flops,
				max_effective_cpu_cores,useful_cpu_cores),
    orig(orig),
    orig_assignment(orig_assignment)
  {
    
  }



  _compute_resource_option_cpu_combined_opencl::_compute_resource_option_cpu_combined_opencl(size_t metadata_bytes,
											     size_t data_bytes,
											     snde_float64 flops,
											     size_t max_effective_cpu_cores,
											     size_t useful_cpu_cores,
											     std::shared_ptr<compute_resource_option> orig,
											     std::shared_ptr<assigned_compute_resource> orig_assignment) :
    _compute_resource_option_cpu_combined(metadata_bytes,data_bytes,flops,
					 max_effective_cpu_cores,useful_cpu_cores,
					 orig,orig_assignment)
  {
    
  }

  std::shared_ptr<assigned_compute_resource> _compute_resource_option_cpu_combined_opencl::combine_cpu_assignment(std::shared_ptr<assigned_compute_resource_cpu> assigned_cpus)
  {
    std::shared_ptr<compute_resource_option_opencl> orig_ocl = std::dynamic_pointer_cast<compute_resource_option_opencl>(orig);
    assert(orig_ocl);
    std::shared_ptr<assigned_compute_resource_opencl> orig_assignment_ocl = std::dynamic_pointer_cast<assigned_compute_resource_opencl>(orig_assignment);
    assert(orig_assignment_ocl);

    // combined the assigned_cpus info into the orig_assignment_ocl and return it.
    orig_assignment_ocl->cpu_assignment = assigned_cpus;
    
    return orig_assignment_ocl; 
  }
  
  
  available_compute_resource_database::available_compute_resource_database() :
    admin(std::make_shared<std::mutex>())
  {

  }

  void available_compute_resource_database::add_resource(std::shared_ptr<available_compute_resource> new_resource)
  {
    compute_resources.emplace(new_resource->get_dispatch_priority(),new_resource);
  }

  bool available_compute_resource_database::_queue_computation_into_database_acrdb_locked(uint64_t globalrev,std::shared_ptr<pending_computation> computation,const std::vector<std::shared_ptr<compute_resource_option>> &compute_options)
  // returns true if we successfully queued it into at least one place. 
  {
    bool retval=false;
    
    // This is a really dumb loop that just assigns all matching resources

    
    
    std::shared_ptr <available_compute_resource> selected_resource;
    std::shared_ptr <compute_resource_option> selected_option;
    for (auto && compute_resource: compute_resources) { // compute_resource is a shared_ptr<available_compute_resource>
      for (auto && compute_option: compute_options) { // compute_option is a shared_ptr<compute_resource_option>
	if (compute_option->type == compute_resource.second->type) {
	  selected_resource = compute_resource.second;
	  selected_option = compute_option;
	  
	  selected_resource->prioritized_computations.emplace(std::make_pair(globalrev,
									     std::make_tuple(std::weak_ptr<pending_computation>(computation),selected_option)));
	  //compute_resource_lock.unlock();  (???What was this supposed to do???)
	  //selected_resource->computations_added.notify_one();
	  notify_acrd_of_changes_to_prioritized_computations();
	  retval=true;

	}	  
      }
      
    }

    if (retval) {
      todo_list.emplace(computation);
    }

    return retval; 
  }
  
  void available_compute_resource_database::_queue_computation_internal(std::shared_ptr<pending_computation> &computation) // NOTE: Sets computation to nullptr once queued
  {
    snde_debug(SNDE_DC_RECMATH,"_queue_computation_internal");

    bool computation_wss_is_globalrev = false; 
    // we cheat a bit in getting the globalrev index: We try dynamically casting ready_wss, and if that fails we use the value from the prerequisite state
    std::shared_ptr<globalrevision> globalrev_ptr = std::dynamic_pointer_cast<globalrevision>(computation->recstate);
    if (!globalrev_ptr) {
      // try prerequisite_state
      globalrev_ptr = std::dynamic_pointer_cast<globalrevision>(computation->recstate->prerequisite_state());

      if (!globalrev_ptr) {
	throw snde_error("recording_set_state does not appear to be associated with any global revision");
      }      
    } else {
      computation_wss_is_globalrev=true; 
    }


    
    // get the compute options
    //std::list<std::shared_ptr<compute_resource_option>> compute_options = computation->function_to_execute->get_compute_options();
    std::vector<std::shared_ptr<compute_resource_option>> compute_options = computation->function_to_execute->execution_tracker->perform_compute_options();

    
    // we can execute anything immutable, anything that is not part of a globalrev (i.e. ondemand), or once the prior globalrev is fully ready
    // (really we just need to make sure all immutable recordings in the prior globalrev are ready, but there isn't currently a good
    // way to do that)
    std::shared_ptr<recording_set_state> prior_globalrev=computation->recstate->prerequisite_state(); // only actually prior_globalrev if computation_wss_is_globalrev

    std::shared_ptr<globalrevision> prior_globalrev_globalrev=std::dynamic_pointer_cast<globalrevision>(prior_globalrev);
    
    std::lock_guard<std::mutex> acrdb_admin(*admin);

    // ***!!!! Really here we need to see if we want to run it in
    // mutable or immutable mode !!!***
    //if (!computation->function_to_execute->inst->fcn->mandatory_mutable || !computation_wss_is_globalrev || !prior_globalrev || prior_globalrev->ready) {
    
    if (!computation->function_to_execute->is_mutable || !computation_wss_is_globalrev || !prior_globalrev || (prior_globalrev->ready && (!prior_globalrev_globalrev || !prior_globalrev_globalrev->mutable_recordings_still_needed))) {

      if (!_queue_computation_into_database_acrdb_locked(globalrev_ptr->globalrev,computation,compute_options)) {
      
  	throw snde_error("No suitable compute resource found for math function %s",computation->function_to_execute->inst->definition->definition_command.c_str());
      }
      
      
    } else {
      // blocked... we have to wait for previous revision to at least
      // complete its mutable recordings and then for any consumers
      // of its mutable recordings to be finished. This is identified
      // when the globalrevision's mutable_recordings_need_holder --
      // which is passed out to all the monitor_globalrevs with the
      // inhibit_mutable flag set -- expires, triggering the
      // blocked_computations for that globalrev (stored in the
      // recdatabase compute_resources blocked_list) to be queued
      // in recstore.cpp: recdatabase::globalrev_mutablenotneeded_code()
      // by calling this function again. 

      blocked_list.emplace(globalrev_ptr->globalrev,computation);
    }
    computation=nullptr; // release shared pointer prior to releasing acrdb_admin lock so that it will expire from pending_computation lists when extracted.
  }


  void available_compute_resource_database::queue_computation(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_set_state> ready_wss,std::shared_ptr<instantiated_math_function> ready_fcn)
  // Take an identified function ready_fcn that is ready to be computed
  // (all inputs are complete) for execution in the context of ready_wss 
  // and queue it for actual execution by the worker threads
  {

    bool ready_wss_is_globalrev = false;

    snde_debug(SNDE_DC_RECMATH,"queue_computation");
    
    // we cheat a bit in getting the globalrev index: We try dynamically casting ready_wss, and if that fails we use the value from the prerequisite state
    std::shared_ptr<globalrevision> globalrev_ptr = std::dynamic_pointer_cast<globalrevision>(ready_wss);
    if (!globalrev_ptr) {
      // try prerequisite_state
      globalrev_ptr = std::dynamic_pointer_cast<globalrevision>(ready_wss->prerequisite_state());
      
      if (!globalrev_ptr) {
	throw snde_error("recording_set_state does not appear to be associated with any global revision");
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

      assert(ready_wss_status.execfunc); // execfunc should have been assigned (At the latest) by check_dep_fcn_ready(), which  should have been called before us. 
      
      execfunc = ready_wss_status.execfunc;
      
      snde_debug(SNDE_DC_RECMATH,"execfunc=0x%lx; wss=0x%lx",(unsigned long)execfunc.get(),(unsigned long)ready_wss.get());
      snde_debug(SNDE_DC_RECMATH,"ready_to_execute:%d",(int)ready_wss_status.ready_to_execute);
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
	  // we're not executing, we must be fully ready too. 

	  // grab recording results from execfunc->execution_tracker->self_dependent_recordings
	  assert(ready_fcn->result_channel_paths.size()==execfunc->execution_tracker->self_dependent_recordings.size());

	  // assign recordings to all referencing wss recordings (should all still exist)
	  for (auto && referencing_rss_weak: execfunc->referencing_rss) {
	    std::shared_ptr<recording_set_state> referencing_rss_strong(referencing_rss_weak);
	    std::lock_guard<std::mutex> referencing_rss_admin(referencing_rss_strong->admin);
	    
	    for (size_t cnt=0;cnt < ready_fcn->result_channel_paths.size(); cnt++) {
	      channel_state &referencing_rss_channel_state = referencing_rss_strong->recstatus.channel_map.at(recdb_path_join(ready_fcn->channel_path_context,*ready_fcn->result_channel_paths.at(cnt)));
	      assert(!referencing_rss_channel_state.rec());

	      referencing_rss_strong->recstatus.defined_recordings.erase(referencing_rss_channel_state.config);
	      referencing_rss_strong->recstatus.completed_recordings.emplace(referencing_rss_channel_state.config,&referencing_rss_channel_state);
	      referencing_rss_channel_state.end_atomic_rec_update(execfunc->execution_tracker->self_dependent_recordings.at(cnt));
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
	  for (auto && referencing_rss_weak: execfunc->referencing_rss) {
	    std::shared_ptr<recording_set_state> referencing_rss_strong(referencing_rss_weak);
	    
	    for (auto && result_channel_path_ptr: ready_fcn->result_channel_paths) {
	      channel_state &chanstate = referencing_rss_strong->recstatus.channel_map.at(recdb_path_join(ready_fcn->channel_path_context,*result_channel_path_ptr));
	      //chanstate.issue_math_notifications(recdb,ready_wss); // taken care of by notify_math_function_executed(), below
	      chanstate.issue_nonmath_notifications(referencing_rss_strong);
	      
	    }
	    
	    snde_debug(SNDE_DC_RECMATH,"qc: already finished wss notify %lx",(unsigned long)referencing_rss_strong.get());
	    // Issue function completion notification
	    referencing_rss_strong->mathstatus.notify_math_function_executed(recdb,referencing_rss_strong,execfunc->inst,false); // note mdonly hardwired to false here
	  }
	  
	  snde_debug(SNDE_DC_RECMATH,"qc: already finished");
	  return;
	}
	std::shared_ptr<pending_computation> computation = std::make_shared<pending_computation>(execfunc,ready_wss,globalrev_ptr->globalrev,0);
	
	
	_queue_computation_internal(computation); // this call sets computation to nullptr; execution ticket delegated to the queued computation
	
	
      } else {
	// Somebody else is taking care of computation (or not ready to execute)
	snde_debug(SNDE_DC_RECMATH,"qc: somebody else");
	
	// no need to do anything; just return
      }
      // warning: ready_wss_admin lock may or may not be held here 
    }
    
  }

  void available_compute_resource_database::start()
  // start all of the compute_resources
  {
    std::multimap<int,std::shared_ptr<available_compute_resource>> compute_resources_copy;
    {
      std::lock_guard<std::mutex> acrd_admin(*admin);
      
      compute_resources_copy = compute_resources;
    }

    for (auto && compute_resource: compute_resources_copy) {
      compute_resource.second->start();
    }


    // start our dispatch thread
    // instantiate dispatch thread
    dispatcher_thread = std::thread([this]() { dispatch_code(); });
    dispatcher_thread.detach(); // we won't be join()ing this thread
    
  }


  void available_compute_resource_database::dispatch_code()
  {
      
    bool no_actual_dispatch = false;
    std::shared_ptr<available_compute_resource_database> acrd_keepinmemory=shared_from_this();  // this shared_ptr prevents the available_compute_resource_database object from getting released, which would make "this" become invalid under us. It also creates a memory leak unless there is a way to signal to this thread that it should return. 

    while(true) {
      
      std::unique_lock<std::mutex> admin_lock(*admin);
      if (no_actual_dispatch) {
	computations_added_or_completed.wait(admin_lock);
      }

      no_actual_dispatch = true; 

      for (auto && compute_resource: compute_resources) {
	// Note: If we drop acrd_admin, the compute_resources list
	// might change under us and we would then be required to break
	// out of this loop.

	// try to dispatch via this resource
	if (compute_resource.second->dispatch_code(admin_lock)) {
	  // success!   (dispatched something, or removed an expred computation from queue)
	  no_actual_dispatch=false;
	  break;
	}
	
      }
      
    }
  }


  void available_compute_resource_database::notify_acrd_of_changes_to_prioritized_computations() // should be called WITH ACRD's admin lock held
  {
    computations_added_or_completed.notify_one();
  }
  
  pending_computation::pending_computation(std::shared_ptr<math_function_execution> function_to_execute,std::shared_ptr<recording_set_state> recstate,uint64_t globalrev,uint64_t priority_reduction) :
    function_to_execute(function_to_execute),
    recstate(recstate),
    globalrev(globalrev),
    priority_reduction(priority_reduction)
  {

  }


  available_compute_resource::available_compute_resource(std::shared_ptr<recdatabase> recdb,unsigned type) :
    recdb(recdb),
    acrd_admin(recdb->compute_resources->admin),
    acrd(recdb->compute_resources),
    type(type)
  {
    
  }

  available_compute_resource_cpu::available_compute_resource_cpu(std::shared_ptr<recdatabase> recdb,size_t total_cpu_cores_available) :
    available_compute_resource(recdb,SNDE_CR_CPU),
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
      thread_actions.push_back(std::make_tuple((std::shared_ptr<recording_set_state>)nullptr,(std::shared_ptr<math_function_execution>)nullptr,(std::shared_ptr<assigned_compute_resource_cpu>)nullptr));
      available_threads.emplace_back(std::thread([this](size_t n){ pool_code(n); },cnt));

      available_threads.at(cnt).detach(); // we won't be join()ing these threads
      
    }

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
    
    return std::make_shared<assigned_compute_resource_cpu>(shared_from_this(),cpu_assignments);

  }

  void available_compute_resource_cpu::_dispatch_threads_from_pool(std::shared_ptr<recording_set_state> recstate,std::shared_ptr<math_function_execution> function_to_execute,std::shared_ptr<assigned_compute_resource_cpu> assigned_cpu_resource,size_t first_thread_index)
  // Must be called with acrd_admin lock held
  {
    //printf("_dispatch_thread()!\n");

    
    //std::lock_guard<std::mutex> admin_lock(*acrd_admin); // lock assumed to be already held
      
    // assign ourselves to functions_using_cores;
    for (auto && core_index: assigned_cpu_resource->assigned_cpu_core_indices) {
      assert(functions_using_cores.at(core_index)==nullptr);
      functions_using_cores.at(core_index) = function_to_execute; 
    }
    
    std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>,std::shared_ptr<assigned_compute_resource_cpu>> &this_thread_action = thread_actions.at(first_thread_index);
    assert(this_thread_action==std::make_tuple((std::shared_ptr<recording_set_state>)nullptr,(std::shared_ptr<math_function_execution>)nullptr,(std::shared_ptr<assigned_compute_resource_cpu>)nullptr));
    
    this_thread_action = std::make_tuple(recstate,function_to_execute,assigned_cpu_resource);
    
    
  
    //printf("triggering thread %d\n",first_thread_index);
    thread_triggers.at(first_thread_index)->notify_one();
  }
  
  bool available_compute_resource_cpu::dispatch_code(std::unique_lock<std::mutex> &acrd_admin_lock)
  {
    std::shared_ptr<available_compute_resource_database> acrd_strong=acrd.lock();
    if (!acrd_strong) return false;

    snde_debug(SNDE_DC_COMPUTE_DISPATCH,"CPU Dispatch, %u computations",(unsigned)prioritized_computations.size());
    if (prioritized_computations.size() > 0) {

      // ***!!! Instead of just looking at the top entry, we could loop here,
      // looking deeper at least through the current globalrev until we find something dispatchable. 
      
      std::multimap<uint64_t,std::tuple<std::weak_ptr<pending_computation>,std::shared_ptr<compute_resource_option>>>::iterator this_computation_it = prioritized_computations.begin();
      std::weak_ptr<pending_computation> this_computation_weak;
      std::shared_ptr<compute_resource_option> compute_option;

      std::tie(this_computation_weak,compute_option) = this_computation_it->second;
      std::shared_ptr<pending_computation> this_computation = this_computation_weak.lock();
      if (!this_computation) {
	// pointer expired; computation has been handled elsewhere
	prioritized_computations.erase(this_computation_it); // remove from our list
	snde_debug(SNDE_DC_COMPUTE_DISPATCH,"CPU Dispatched expired computation");
	
	return true; // removing from prioritized_computations counts as an actual dispatch
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
	  // !!!*** Would make sense here to limit the cores going to a single computation to slightly
	  // fewer than the total so as to allow single-core GPU jobs to execute in parallel. 
	  std::shared_ptr<math_function_execution> function_to_execute=this_computation->function_to_execute;
	  std::shared_ptr<recording_set_state> recstate=this_computation->recstate;
	  
	  prioritized_computations.erase(this_computation_it); // take charge of this computation
	  acrd_strong->todo_list.erase(this_computation); // remove from todo list so pointer can expire
	  this_computation = nullptr; // force pointer to expire so nobody else tries this computation;
	  
	  std::shared_ptr<assigned_compute_resource_cpu> assigned_cpus = _assign_cpus(function_to_execute,std::min(compute_option_cpu->useful_cpu_cores,total_cpu_cores_available));

	  std::shared_ptr<_compute_resource_option_cpu_combined> combined_resource = std::dynamic_pointer_cast<_compute_resource_option_cpu_combined>(compute_option);
	  if (combined_resource) {
	    // We are just part of the underlying resource, which is combined_resource->orig
	    function_to_execute->execution_tracker->compute_resource = combined_resource->combine_cpu_assignment(assigned_cpus);
	    function_to_execute->execution_tracker->selected_compute_option = combined_resource->orig;
	  
	  } else {
	    function_to_execute->execution_tracker->compute_resource = assigned_cpus;
	    function_to_execute->execution_tracker->selected_compute_option = compute_option;
	  }
	  _dispatch_threads_from_pool(recstate,function_to_execute,assigned_cpus,assigned_cpus->assigned_cpu_core_indices.at(0));
	  
	  snde_debug(SNDE_DC_COMPUTE_DISPATCH,"CPU Dispatched computation");
	  
	  return true; // dispatched execution, so don't wait at next iteration. 
	}

	this_computation = nullptr; // remove all references to pending_computation before we release the lock
      }
      
      
    }
    snde_debug(SNDE_DC_COMPUTE_DISPATCH,"CPU did not dispatch any computation");

    return false;
  }
  
  int available_compute_resource_cpu::get_dispatch_priority() // Get the dispatch priority of this compute resource. Smaller or more negative numbers are higher priority. See SNDE_ACRP_XXXX, above
  {
    return SNDE_ACRP_CPU;
  }
  
  void available_compute_resource_cpu::pool_code(size_t threadidx)
  {
    std::shared_ptr<available_compute_resource> acr_keepinmemory=shared_from_this();  // this shared_ptr prevents the available_compute_resource object from getting released, which would make "this" become invalid under us.  It also creates a memory leak unless there is a way to signal to this thread that it should return. 

    
    //printf("pool_code_startup!\n");
    while(true) {
      
      std::unique_lock<std::mutex> admin_lock(*acrd_admin);
      thread_triggers.at(threadidx)->wait(admin_lock);
      //printf("pool_code_wakeup!\n");

      std::shared_ptr<recording_set_state> recstate;
      std::shared_ptr<math_function_execution> func;
      std::shared_ptr<assigned_compute_resource_cpu> assigned_compute_cpu;
      
      std::tie(recstate,func,assigned_compute_cpu) = thread_actions.at(threadidx);
      if (recstate) {
	//printf("Pool code got thread action\n");
	// Remove parameters from thread_actions
	thread_actions.at(threadidx)=std::make_tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>,std::shared_ptr<assigned_compute_resource_cpu>>(nullptr,nullptr,nullptr);

	// not worrying about cpu affinity yet.

	// Set the build-time variable SNDE_RCR_DISABLE_EXCEPTION_HANDLING to disable the try {} ... catch{} block in math execution so that you can capture the offending scenario in the debugger
	//#define SNDE_RCR_DISABLE_EXCEPTION_HANDLING
#ifndef SNDE_RCR_DISABLE_EXCEPTION_HANDLING
	try {
#endif
	  // Need also our assigned_compute_resource (func->compute_resource)
	  admin_lock.unlock();
	  
	  // will eventually need the lock manager from somewhere(!!!***???)
	  bool mdonly = func->mdonly;
	  //bool is_mutable;
	  //bool mdonly_executed;
	  
	  //// Get the mdonly and is_mutable flags from the math_function_status 
	  //{
	  //std::lock_guard<std::mutex> wss_admin(recstate->admin);
	  
	  
	  
	  //math_function_status &our_status = recstate->mathstatus.function_status.at(func->inst);
	  //  mdonly = our_status.mdonly;
	  //  is_mutable = our_status.is_mutable;
	  // mdonly_executed = our_status.mdonly_executed; 
	  //
	  //}
	
	  if (!func->metadata_executed) {
	    func->execution_tracker->perform_define_recs();

	    // grab generated recordings and move them from defined_recordings to instantiated_recordings list
	    {
	      std::lock_guard<std::mutex> wssadmin(func->wss->admin);
	      for (auto && result_channel_path_ptr: func->inst->result_channel_paths) {
		if (result_channel_path_ptr) {
		  
		  channel_state &chanstate = func->wss->recstatus.channel_map.at(recdb_path_join(func->inst->channel_path_context,*result_channel_path_ptr));
		  if (chanstate.rec()) {
		    // have a recording
		    size_t erased_from_defined_recordings = func->wss->recstatus.defined_recordings.erase(chanstate.config);
		    
		    assert(erased_from_defined_recordings==1); // recording should have been listed in defined_recordings
		    func->wss->recstatus.instantiated_recordings.emplace(chanstate.config,&chanstate);
		  } else {
		    // ***!!! If the math function failed to define
		    // a recording here, we should probably import
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
	    
	    // assign recordings to all referencing wss recordings (should all still exist)
	    // (only needed if we're not doing it below)
	    // !!!*** This block should be refactored
	    for (auto && referencing_rss_weak: func->referencing_rss) {
	      std::shared_ptr<recording_set_state> referencing_rss_strong(referencing_rss_weak);
	      if (referencing_rss_strong == func->wss) {
		continue; // no need to reassign our source
	      }
	      
	      std::lock_guard<std::mutex> referencing_rss_admin(referencing_rss_strong->admin);
	      
	      for (size_t cnt=0;cnt < func->inst->result_channel_paths.size(); cnt++) {
		// none should have preassigned recordings --not true because parallel update or end_transaction() could be going on
		//assert(!referencing_rss->recstatus.channel_map.at(&result_channel_paths.at(cnt)).rec());
		channel_state &referencing_rss_channel_state=referencing_rss_strong->recstatus.channel_map.at(recdb_path_join(func->inst->channel_path_context,*func->inst->result_channel_paths.at(cnt)));
		
		referencing_rss_strong->recstatus.instantiated_recordings.erase(referencing_rss_channel_state.config);
		referencing_rss_strong->recstatus.metadataonly_recordings.emplace(referencing_rss_channel_state.config,&referencing_rss_channel_state);
		
		referencing_rss_channel_state.end_atomic_rec_update(func->wss->recstatus.channel_map.at(*func->inst->result_channel_paths.at(cnt)).rec());
	      }
	    }
	  }
	  
	  if (!mdonly) {
	    func->execution_tracker->perform_lock_alloc();
	    func->execution_tracker->perform_exec();
	    
	    func->fully_complete = true;
	    
	    
	    // re-assign recordings to all referencing wss recordings (should all still exist) in case we have more followers than before
	    // !!!*** This block should be refactored
	    for (auto && referencing_rss_weak: func->referencing_rss) {
	      std::shared_ptr<recording_set_state> referencing_rss_strong(referencing_rss_weak);
	      if (referencing_rss_strong == func->wss) {
		continue; // no need to reassign our source
	      }
	      
	      std::lock_guard<std::mutex> referencing_rss_admin(referencing_rss_strong->admin);
	      
	      for (size_t cnt=0;cnt < func->inst->result_channel_paths.size(); cnt++) {
		// none should have preassigned recordings --not true because parallel update or end_transaction() could be going on
		//assert(!referencing_rss->recstatus.channel_map.at(&result_channel_paths.at(cnt)).rec());
		channel_state & referencing_rss_channel_state = referencing_rss_strong->recstatus.channel_map.at(recdb_path_join(func->inst->channel_path_context,*func->inst->result_channel_paths.at(cnt)));
		
		referencing_rss_strong->recstatus.instantiated_recordings.erase(referencing_rss_channel_state.config);
		referencing_rss_strong->recstatus.completed_recordings.emplace(referencing_rss_channel_state.config,&referencing_rss_channel_state);
		
		referencing_rss_channel_state.end_atomic_rec_update(func->wss->recstatus.channel_map.at(*func->inst->result_channel_paths.at(cnt)).rec());
	      }
	    }
	  
	  }
	  
	  
	  // Mark execution as no longer ongoing in the math_function_status 
	  {
	    std::unique_lock<std::mutex> wss_admin(recstate->admin);
	    std::unique_lock<std::mutex> func_admin(func->admin);
	    
	    math_function_status &our_status = recstate->mathstatus.function_status.at(func->inst);
	    if (mdonly && !func->mdonly) {
	      // execution status changed from mdonly to non-mdonly behind our back...
	      mdonly=false;
	      // finish up execution before we mark as finished
	      func_admin.unlock();
	      wss_admin.unlock();
	      
	      func->execution_tracker->perform_lock_alloc();
	      func->execution_tracker->perform_exec();
	      
	      func->fully_complete = true;
	      
	      // re-assign recordings to all referencing wss recordings (should all still exist) in case we have more followers than before
	      // !!!*** This block should be refactored
	      for (auto && referencing_rss_weak: func->referencing_rss) {
		std::shared_ptr<recording_set_state> referencing_rss_strong(referencing_rss_weak);
		if (referencing_rss_strong == func->wss) {
		  continue; // no need to reassign our source
		}
		
		std::lock_guard<std::mutex> referencing_rss_admin(referencing_rss_strong->admin);
		
		for (size_t cnt=0;cnt < func->inst->result_channel_paths.size(); cnt++) {
		  // none should have preassigned recordings --not true because parallel update or end_transaction() could be going on
		  //assert(!referencing_rss->recstatus.channel_map.at(&result_channel_paths.at(cnt)).rec());
		  
		  channel_state &referencing_rss_channel_state = referencing_rss_strong->recstatus.channel_map.at(recdb_path_join(func->inst->channel_path_context,*func->inst->result_channel_paths.at(cnt)));
		  
		  referencing_rss_strong->recstatus.instantiated_recordings.erase(referencing_rss_channel_state.config);
		  referencing_rss_strong->recstatus.completed_recordings.emplace(referencing_rss_channel_state.config,&referencing_rss_channel_state);
		
		  referencing_rss_channel_state.end_atomic_rec_update(func->wss->recstatus.channel_map.at(recdb_path_join(func->inst->channel_path_context,*func->inst->result_channel_paths.at(cnt))).rec());
		}
	      }
	    
	      
	    
	      wss_admin.lock();
	      func_admin.lock();
	    }
	    our_status.complete=true;
	    
	  }
	  
	  // clear execfunc->wss to eliminate reference loop once at least metadata execution is complete for an mdonly recording
	  // or once all execution is complete for a regular recording
	  if (func->metadataonly_complete || func->fully_complete) {
	    func->wss = nullptr;
	    func->execution_tracker->wss = nullptr;
	  }
	  
	  func->executing = false; // release the execution ticket
	  
	  std::shared_ptr<recdatabase> recdb_strong = recdb.lock();
	  
	  snde_debug(SNDE_DC_RECMATH,"Pool code completed math function");
	  //fflush(stdout);
	  
	  
	  // Need to do notifications that the math function finished in all referencing wss's
	  for (auto && referencing_rss_weak: func->referencing_rss) {
	    std::shared_ptr<recording_set_state> referencing_rss_strong(referencing_rss_weak);
	    
	    for (auto && result_channel_path_ptr: func->inst->result_channel_paths) {
	      channel_state &chanstate = referencing_rss_strong->recstatus.channel_map.at(recdb_path_join(func->inst->channel_path_context,*result_channel_path_ptr));
	      //chanstate.issue_math_notifications(recdb,ready_wss); // taken care of by notify_math_function_executed(), below
	      chanstate.issue_nonmath_notifications(referencing_rss_strong);
	      
	    }
	    
	    // Issue function completion notification
	    if (recdb_strong) {
	      snde_debug(SNDE_DC_RECMATH|SNDE_DC_NOTIFY,"pool code: wss notify %lx",(unsigned long)referencing_rss_strong.get());
	      referencing_rss_strong->mathstatus.notify_math_function_executed(recdb_strong,referencing_rss_strong,func->inst,mdonly); 
	    }
	  }

	  //printf("Pool code completed notification\n");
	  //fflush(stdout);
	  // Completion notification:
	  //  * removing ourselves from functions_using_cores and triggering computations_added_or_completed
	  admin_lock.lock();

	  // release compute resources
	  func->execution_tracker->compute_resource->release_assigned_resources(admin_lock);
	  
	  // Notify that we are done
	  std::shared_ptr<available_compute_resource_database> acrd_strong=acrd.lock();
	  if (acrd_strong) {
	    acrd_strong->notify_acrd_of_changes_to_prioritized_computations();
	  }
	  
#ifndef SNDE_RCR_DISABLE_EXCEPTION_HANDLING
	} catch(const std::exception &exc) {
	  // Only consider exceptions derived from std::exception because there's no general way to print anything else, so we might as well just crash in that case. 
	  snde_warning("Exception caught in math thread pool: %s",exc.what());
	}
#endif // SNDE_RCR_DISABLE_EXCEPTION_HANDLING
      }
    }
  }

#ifdef SNDE_OPENCL
  available_compute_resource_opencl::available_compute_resource_opencl(std::shared_ptr<recdatabase> recdb,std::shared_ptr<available_compute_resource_cpu> controlling_cpu,cl::Context opencl_context,const std::vector<cl::Device> &opencl_devices,size_t max_parallel) :
    available_compute_resource(recdb,SNDE_CR_OPENCL),
    controlling_cpu(controlling_cpu),
    opencl_context(opencl_context),
    opencl_devices(opencl_devices),
    max_parallel(max_parallel),
    functions_using_devices(max_parallel*opencl_devices.size())

  {
    
  }

  void available_compute_resource_opencl::start() // set the compute resource going
  {
    // Nothing to do as we don't execute ourselves
  }
  
  bool available_compute_resource_opencl::dispatch_code(std::unique_lock<std::mutex> &acrd_admin_lock)
  {
    // *** dispatch our entry, delegating the CPU portion to the controlling_cpu...
    std::shared_ptr<available_compute_resource_database> acrd_strong=acrd.lock();
    assert(acrd_strong); // we are called by the acrd, so it really better not have been destroyed!

    snde_debug(SNDE_DC_COMPUTE_DISPATCH,"OpenCL Dispatch, %u computations",(unsigned)prioritized_computations.size());

    if (prioritized_computations.size() > 0) {
            
      std::multimap<uint64_t,std::tuple<std::weak_ptr<pending_computation>,std::shared_ptr<compute_resource_option>>>::iterator this_computation_it = prioritized_computations.begin();
      std::weak_ptr<pending_computation> this_computation_weak;
      std::shared_ptr<compute_resource_option> compute_option;
      
      uint64_t globalrev = this_computation_it->first; 
      
      std::tie(this_computation_weak,compute_option) = this_computation_it->second;
      std::shared_ptr<pending_computation> this_computation = this_computation_weak.lock();
      if (!this_computation) {
	// pointer expired; computation has been handled elsewhere
	prioritized_computations.erase(this_computation_it); // remove from our list
	snde_debug(SNDE_DC_COMPUTE_DISPATCH,"OpenCL Dispatched expired computation");

	return true; // removing from prioritized_computations counts as an actual dispatch
      } else {
	// got this_computation and compute_option to possibly try.
	// Check if we have enough cores available for compute_option
	std::shared_ptr<compute_resource_option_opencl> compute_option_opencl=std::dynamic_pointer_cast<compute_resource_option_opencl>(compute_option);

	// this had better be one of our pointers...
	assert(compute_option_opencl);

	// For now, just blindly use the useful # of cpu cores
	size_t free_gpus = _number_of_free_gpus();

	if (free_gpus > 0) {
	  std::shared_ptr<math_function_execution> function_to_execute=this_computation->function_to_execute;
	  std::shared_ptr<recording_set_state> recstate=this_computation->recstate;

	  prioritized_computations.erase(this_computation_it); // take charge of this computation
	  acrd_strong->todo_list.erase(this_computation); // remove from todo list so pointer can expire
	  this_computation = nullptr; // force pointer to expire so nobody else tries this computation;
	  
	  std::shared_ptr<assigned_compute_resource_opencl> assigned_gpus = _assign_gpu(function_to_execute);

	  // Create a combined resource we use to delegate the CPU portion 
	  std::shared_ptr<_compute_resource_option_cpu_combined_opencl> combined_resource = std::make_shared<_compute_resource_option_cpu_combined_opencl>(compute_option_opencl->metadata_bytes,compute_option_opencl->data_bytes,compute_option_opencl->cpu_flops,compute_option_opencl->max_effective_cpu_cores,compute_option_opencl->useful_cpu_cores,compute_option,assigned_gpus);
	  
	  std::shared_ptr<pending_computation> combined_computation = std::make_shared<pending_computation>(function_to_execute,recstate,globalrev,0);
	  
	  // Enqueue the CPU portion
	  std::vector<std::shared_ptr<compute_resource_option>> compute_options;
	  compute_options.push_back(combined_resource);
	  
	  if (!acrd_strong->_queue_computation_into_database_acrdb_locked(globalrev,combined_computation,compute_options)) {
	    throw snde_error("No suitable CPU compute resource found for math function %s",combined_computation->function_to_execute->inst->definition->definition_command.c_str());
	    
	  }
	  snde_debug(SNDE_DC_COMPUTE_DISPATCH,"OpenCL Dispatched computation to CPU");

	  return true; 
	}
      }
    }
    snde_debug(SNDE_DC_COMPUTE_DISPATCH,"OpenCL did not dispatch any computation");

    return false;
  }

  int available_compute_resource_opencl::get_dispatch_priority() // Get the dispatch priority of this compute resource. Smaller or more negative numbers are higher priority. See SNDE_ACRP_XXXX, above
  {
    // Check to see if all of the devices are actually CPU
    bool all_devices_actually_cpu=true;
    bool any_devices_actually_cpu=false;
    
    for (auto && device: opencl_devices) {
      cl_device_type gottype = device.getInfo<CL_DEVICE_TYPE>();

      if ((gottype & CL_DEVICE_TYPE_CPU) && !(gottype & CL_DEVICE_TYPE_GPU)) {
	any_devices_actually_cpu=true; 
      } else {
	all_devices_actually_cpu=false;
      }
    }

    if (all_devices_actually_cpu && any_devices_actually_cpu) {
      snde_warning("available_compute_resource_opencl: all OpenCL compute devices are actually CPU type. Treating as low-priority fallback.");
      return SNDE_ACRP_CPU_AS_GPU;
    }

    if (any_devices_actually_cpu) {
      snde_warning("available_compute_resource_opencl: some OpenCL compute devices are actually CPU type.");
      
    }
    
    return SNDE_ACRP_GPU_GENERALAPI;
  }

  size_t available_compute_resource_opencl::_number_of_free_gpus()
  // Must call with ACRD admin lock locked
  {
    size_t number_of_free_gpus=0;

    for (auto && exec_fcn: functions_using_devices) {
      if (!exec_fcn) {
	number_of_free_gpus++;
      }
    }
    return number_of_free_gpus;
  }

  std::shared_ptr<assigned_compute_resource_opencl> available_compute_resource_opencl::_assign_gpu(std::shared_ptr<math_function_execution> function_to_execute)
  // called with acrd admin lock held
  {
    
    size_t job_index=0;
    std::vector<size_t> job_assignments;
    std::vector<cl::Device> device_assignments;

    for (auto && exec_fcn: functions_using_devices) {
      if (!exec_fcn) {
	// this gpu is available
	// assign it...
	job_assignments.push_back(job_index);
	device_assignments.push_back(opencl_devices.at(job_index % max_parallel));
	
	break; 

      }
      job_index++;
    }

    assert(job_assignments.size() > 0); // should have been able to assign everything
    
    return std::make_shared<assigned_compute_resource_opencl>(shared_from_this(),std::vector<size_t>(),job_assignments,opencl_context,device_assignments);

  }

  
#endif // SNDE_OPENCL

  assigned_compute_resource::assigned_compute_resource(unsigned type,std::shared_ptr<available_compute_resource> resource) :
    type(type),
    resource(resource)
  {
    
  }
  
  assigned_compute_resource_cpu::assigned_compute_resource_cpu(std::shared_ptr<available_compute_resource> resource,const std::vector<size_t> &assigned_cpu_core_indices) :
    assigned_compute_resource(SNDE_CR_CPU,resource),
    assigned_cpu_core_indices(assigned_cpu_core_indices)
  {
    
  }

  void assigned_compute_resource_cpu::release_assigned_resources(std::unique_lock<std::mutex> &acrd_admin_holder) // resources referenced below no longer meaningful once this is called. Must be called with acrd admin lock locked
  {
    // remove ourselves from functions_using_cores
    //for (size_t corenum=0;corenum < total_cpu_cores_available;corenum++) {
    //  if (functions_using_cores.at(corenum) == func) {
    //functions_using_cores.at(corenum) = nullptr; 
    //  }
    //}

    std::shared_ptr<available_compute_resource_cpu> cpu_resource = std::dynamic_pointer_cast<available_compute_resource_cpu>(resource);
    assert(cpu_resource); // types should always match
    
    for (auto && coreindex: assigned_cpu_core_indices) {
      cpu_resource->functions_using_cores.at(coreindex) = nullptr; 
    }

  }

  
#ifdef SNDE_OPENCL
  assigned_compute_resource_opencl::assigned_compute_resource_opencl(std::shared_ptr<available_compute_resource> resource,const std::vector<size_t> &assigned_cpu_core_indices,const std::vector<size_t> &assigned_opencl_job_indices,cl::Context opencl_context,const std::vector<cl::Device> &opencl_devices) :
    assigned_compute_resource(SNDE_CR_OPENCL,resource),
    assigned_opencl_job_indices(assigned_opencl_job_indices),
    opencl_context(opencl_context),
    opencl_devices(opencl_devices),
    cpu_assignment(nullptr) // will be filled in later once the CPU module has dispatched
  {
    
  }

  void assigned_compute_resource_opencl::release_assigned_resources(std::unique_lock<std::mutex> &acrd_admin_holder) // resources referenced below no longer meaningful once this is called. Must be called with acrd admin lock locked
  {
    std::shared_ptr<available_compute_resource_opencl> opencl_resource = std::dynamic_pointer_cast<available_compute_resource_opencl>(resource);
    assert(opencl_resource); // types should always match
    
    for (auto && paradevice: assigned_opencl_job_indices) {
      opencl_resource->functions_using_devices.at(paradevice) = nullptr; 
    }

    cpu_assignment->release_assigned_resources(acrd_admin_holder);
    
  }

#endif //SNDE_OPENCL
};
