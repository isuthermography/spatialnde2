#include "wfmmath_compute_resource.hpp"
#include "wfmstore.hpp"
#include "wfmmath.hpp"

namespace snde {
  compute_resource_option::compute_resource_option(unsigned type, size_t metadata_bytes,size_t data_bytes,std::shared_ptr<compute_code> function_code) :
    type(type),
    metadata_bytes(metadata_bytes),
    data_bytes(data_bytes),
    function_code(function_code)
  {

  }


  compute_resource_option_cpu::compute_resource_option_cpu(unsigned type,
							   size_t metadata_bytes,
							   size_t data_bytes,
							   std::shared_ptr<compute_code> function_code,
							   snde_float64 flops,
							   size_t max_effective_cpu_cores,
							   size_t useful_cpu_cores) :
    compute_resource_option(type,metadata_bytes,data_bytes,function_code),
    flops(flops),
    max_effective_cpu_cores(max_effective_cpu_cores),
    useful_cpu_cores(useful_cpu_cores)
  {

  }

#ifdef SNDE_OPENCL
  compute_resource_option_opencl::compute_resource_option_opencl(unsigned type,
								 size_t metadata_bytes,
								 size_t data_bytes,
								 std::shared_ptr<compute_code> function_code,
								 snde_float64 cpu_flops,
								 snde_float64 gpu_flops,
								 size_t max_effective_cpu_cores,
								 size_t max_useful_cpu_cores) :
    compute_resource_option(type,metadata_bytes,data_bytes,function_code),
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
    std::list<std::shared_ptr<compute_resource_option>> compute_options = computation->function_to_execute->fcn->fcn->get_compute_options(computation->function_to_execute);

    
    // we can execute anything immutable, anything that is not part of a globalrev (i.e. ondemand), or once the prior globalrev is fully ready
    // (really we just need to make sure all immutable waveforms in the prior globalrev are ready, but there isn't currently a good
    // way to do that)
    std::shared_ptr<waveform_set_state> prior_globalrev=computation->wfmstate->prerequisite_state(); // only actually prior_globalrev if computation_wss_is_globalrev


    std::lock_guard<std::mutex> acrdb_admin(*admin);

    // ***!!!! Really here we need to see if we want to run it in
    // mutable or immutable mode !!!***
    if (!computation->function_to_execute->fcn->fcn->mandatory_mutable || !computation_wss_is_globalrev || !prior_globalrev || prior_globalrev->ready) {
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
	throw snde_error("No suitable compute resource found for math function %s",computation->function_to_execute->fcn->definition->definition_command.c_str());
      }
      
      
    } else {
      // blocked... we have to wait for previous revision to at least
      // complete its mutable waveforms !!!*** NEED PROCESS FOR TRIGGERING REMOVAL FROM BLOCKED LIST
      blocked_list.emplace(globalrev_ptr->globalrev,computation);
    }
    computation=nullptr; // release shared pointer prior to releasing acrdb_admin lock. 
  }


  void available_compute_resource_database::queue_computation(std::shared_ptr<waveform_set_state> ready_wss,std::shared_ptr<instantiated_math_function> ready_fcn)
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

    std::shared_ptr<executing_math_function> function_to_execute=std::make_shared<executing_math_function>(ready_fcn);
    std::shared_ptr<pending_computation> computation = std::make_shared<pending_computation>(function_to_execute,ready_wss,globalrev_ptr->globalrev,0);

    bool is_mutable=false; 
    {
      std::lock_guard<std::mutex> ready_wss_admin(ready_wss->admin);
      math_function_status &ready_wss_status = ready_wss->mathstatus.function_status.at(ready_fcn);
      is_mutable = ready_wss_status.is_mutable;
      if (ready_wss_status.ready_to_execute) {
	assert(ready_wss_status.execution_demanded); 
	// clear ready_to_execute flag, indicating that we are taking care of execution
	ready_wss_status.ready_to_execute=false;
      }
    }

    _queue_computation_internal(computation); // this call sets computation to nullptr;
    
  }


  pending_computation::pending_computation(std::shared_ptr<executing_math_function> function_to_execute,std::shared_ptr<waveform_set_state> wfmstate,uint64_t globalrev,uint64_t priority_reduction) :
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
    size_t cnt;

    std::lock_guard<std::mutex> acrd_lock(*acrd_admin); // because threads will start, and we need to lock them out while we fill up the vector data structures
    for (cnt=0; cnt < total_cpu_cores_available;cnt++) {
      functions_using_cores.push_back(nullptr);
      thread_triggers.emplace_back(std::make_shared<std::condition_variable>());
      thread_actions.push_back(std::make_tuple((std::shared_ptr<waveform_set_state>)nullptr,(std::shared_ptr<executing_math_function>)nullptr,(std::shared_ptr<compute_resource_option_cpu>)nullptr));
      available_threads.emplace_back(std::thread([this](size_t n){ pool_code(n); },cnt));
    }
    
  }
  void available_compute_resource_cpu::notify_acr_of_new_or_finished_computations() // should be called WITH ACRD's admin lock held
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

  std::shared_ptr<assigned_compute_resource_cpu> available_compute_resource_cpu::_assign_cpus(std::shared_ptr<executing_math_function> function_to_execute,size_t number_of_cpus)
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

  void available_compute_resource_cpu::_dispatch_thread(std::shared_ptr<waveform_set_state> wfmstate,std::shared_ptr<executing_math_function> function_to_execute,std::shared_ptr<compute_resource_option_cpu> compute_option_cpu,size_t first_thread_index)
  {

    {
      std::lock_guard<std::mutex> admin_lock(*acrd_admin);
      
      // assign ourselves to functions_using_cores;
      for (auto && core_index: (std::dynamic_pointer_cast<assigned_compute_resource_cpu>(function_to_execute->compute_resource)->assigned_cpu_core_indices)) {
	assert(functions_using_cores.at(core_index)==nullptr);
	functions_using_cores.at(core_index) = function_to_execute; 
      }
      
      std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<executing_math_function>,std::shared_ptr<compute_resource_option_cpu>> & this_thread_action = thread_actions.at(first_thread_index);
      assert(this_thread_action==std::make_tuple((std::shared_ptr<waveform_set_state>)nullptr,(std::shared_ptr<executing_math_function>)nullptr,(std::shared_ptr<compute_resource_option_cpu>)nullptr));
      
      this_thread_action = std::make_tuple(wfmstate,function_to_execute,compute_option_cpu);
      
    }

    thread_triggers.at(first_thread_index)->notify_one();
  }
  
  void available_compute_resource_cpu::dispatch_code()
  // Does this really need to be its own thread? Maybe not...
  {
    bool no_actual_dispatch = false;
    while(true) {
      
      std::unique_lock<std::mutex> admin_lock(*acrd_admin);
      if (no_actual_dispatch) {
	computations_added_or_completed.wait(admin_lock);
      }

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
	    std::shared_ptr<executing_math_function> function_to_execute=this_computation->function_to_execute;
	    std::shared_ptr<waveform_set_state> wfmstate=this_computation->wfmstate;

	    prioritized_computations.erase(this_computation_it); // take charge of this computation
	    this_computation = nullptr; // force pointer to expire so nobody else tries this computation;

	    std::shared_ptr<assigned_compute_resource_cpu> assigned_cpus = _assign_cpus(function_to_execute,std::min(compute_option_cpu->useful_cpu_cores,total_cpu_cores_available));
	    function_to_execute->compute_resource = assigned_cpus;

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
    
    while(true) {
      
      std::unique_lock<std::mutex> admin_lock(*acrd_admin);
      thread_triggers.at(threadidx)->wait(admin_lock);
      std::shared_ptr<waveform_set_state> wfmstate;
      std::shared_ptr<executing_math_function> func;
      std::shared_ptr<compute_resource_option_cpu> compute_option_cpu;
      
      std::tie(wfmstate,func,compute_option_cpu) = thread_actions.at(threadidx);
      if (wfmstate) {
	// Remove parameters from thread_actions
	thread_actions.at(threadidx)=std::make_tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<executing_math_function>,std::shared_ptr<compute_resource_option_cpu>>(nullptr,nullptr,nullptr);

	// not worrying about cpu affinity yet.


	// Need also our assigned_compute_resource (func->compute_resource)
	admin_lock.unlock();

	// will eventually need the lock manager from somewhere(!!!***???)
	bool mdonly;
	bool is_mutable;
	bool mdonly_executed;

	// Get the mdonly and is_mutable flags from the math_function_status 
	{
	  std::lock_guard<std::mutex> wss_admin(wfmstate->admin);

	  
	  
	  math_function_status &our_status = wfmstate->mathstatus.function_status.at(func->fcn);
	  mdonly = our_status.mdonly;
	  is_mutable = our_status.is_mutable;
	  mdonly_executed = our_status.mdonly_executed; 

	}


	if (is_mutable) {
	  // Locking: Read locks can be implicit, as even if we are
	  // a mutable waveform ourselves, nobody should
	  // be allowed to write to our prerequisites
	  throw snde_error("Mutable execution not yet implemented");
	  //func->fcn->result_channel_paths
	} else {
	  // non-mutable execution
	  // !!!*** Need to accommodate possible decision on whether to even define a new revision or reference the old one ***!!!
	  if (mdonly) {
	    if (mdonly_executed) {
	      throw snde_error("Math function %s reexecuting mdonly (?)",func->fcn->definition->definition_command.c_str());
	    }
	    compute_option_cpu->function_code->do_metadata_only(wfmstate,func,compute_option_cpu);
	  } else if (mdonly_executed) {
	    compute_option_cpu->function_code->do_compute_from_metadata(wfmstate,func,compute_option_cpu);
	    
	  } else {
	    compute_option_cpu->function_code->do_compute(wfmstate,func,compute_option_cpu);
	    
	  }
	}

	std::shared_ptr<wfmdatabase> wfmdb_strong = wfmdb.lock();

	// Need to do notifications that the math function finished.
	if (wfmdb_strong) {
	  wfmstate->mathstatus.notify_math_function_executed(wfmdb_strong,wfmstate,func->fcn,mdonly);
	}
	
	// Completion notification:
	//  * removing ourselves from functions_using_cores and triggering computations_added_or_completed
	admin_lock.lock();

	// remove ourselves from functions_using_cores
	for (size_t corenum;corenum < total_cpu_cores_available;corenum++) {
	  if (functions_using_cores.at(corenum) == func) {
	    functions_using_cores.at(corenum) = nullptr; 
	  }
	}
	// Notify that we are done
	notify_acr_of_new_or_finished_computations();
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
