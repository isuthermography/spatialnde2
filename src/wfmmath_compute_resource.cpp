#include "wfmmath_compute_resource.hpp"

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
							   float64_t flops,
							   size_t max_effective_cpu_cores,
							   size_t max_useful_cpu_cores) :
    compute_resource_option(type,metadata_bytes,data_bytes,function_code),
    flops(flops),
    max_effective_cpu_cores(max_effective_cpu_cores),
    max_useful_cpu_cores(max_useful_cpu_cores)
  {

  }

  compute_resource_option_opencl::compute_resource_option_opencl(unsigned type,
								 size_t metadata_bytes,
								 size_t data_bytes,
								 std::shared_ptr<compute_code> function_code,
								 float64_t cpu_flops,
								 float64_t gpu_flops,
								 size_t max_effective_cpu_cores,
								 size_t max_useful_cpu_cores) :
    compute_resource_option(type,metadata_bytes,data_bytes,function_code),
    cpu_flops(cpu_flops),
    gpu_flops(gpu_flops),
    max_effective_cpu_cores(max_effective_cpu_cores),
    max_useful_cpu_cores(max_useful_cpu_cores)
  {

  }

  void available_compute_resource_database::queue_computation(std::shared_ptr<waveform_set_state> ready_wss,std::shared_ptr<instantiated_math_function> ready_fcn)
  {
    std::shared_ptr<pending_computation> computation;
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
    
    
    computation = std::make_shared<pending_computation>(function_to_execute,ready_wss,globalrev_ptr->globalrev,0);

    bool is_mutable=false; 
    {
      std::lock_guard<std::mutex> ready_wss_admin(ready_wss->admin);
      math_function_status &ready_wss_status = ready_wss->mathstatus.function_status.at(ready_fcn);
      is_mutable = ready_wss_status.is_mutable; 
    }
    
    // get the compute options
    std::list<std::shared_ptr<compute_resource_option>> compute_options = ready_fcn->fcn->get_compute_options(function_to_execute);

    
    // we can execute anything immutable, anything that is not part of a globalrev (i.e. ondemand), or once the prior globalrev is fully ready
    // (really we just need to make sure all immutable waveforms in the prior globalrev are ready, but there isn't currently a good
    // way to do that)
    std::shared_ptr<waveform_set_state> prior_globalrev=ready_wss->prerequisite_state(); // only actually prior_globalrev if ready_wss_is_globalrev
    
    std::lock_guard<std::mutex> acrdb_admin(admin);
    if (!ready_fcn->is_mutable || !ready_wss_is_globalrev || !prior_globalrev || prior_globalrev->ready) {
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
									       std::make_tuple(std::weak_ptr(computation),selected_option)));
	    selected_resource->computations_added.notify_one();
	  }	  
	}

      }
      
      if (!selected_resource) {
	throw snde_error("No suitable compute resource found for math function %s",ready_fcn->definition->definition_command.c_str());
      }
      
      
    } else {
      // blocked... we have to wait for previous revision to at least
      // complete its mutable waveforms !!!*** NEED PROCESS FOR TRIGGERING REMOVAL FROM BLOCKED LIST
      blocked_list.emplace(globalrev_ptr->globalrev,computation);
    }
    computation=nullptr; // release shared pointer prior to releasing acrdb_admin lock. 
  }


  available_compute_resource::available_compute_resource(unsigned type) :
    type(type)
  {
    
  }

  available_compute_resource_cpu::available_compute_resource_cpu(unsigned type,size_t total_cpu_cores_available) :
    available_compute_resource(type),
    total_cpu_cores_available(total_cpu_cores_available)
  {
    
  }

  available_compute_resource_opencl::available_compute_resource_opencl(unsigned type,cl_context opencl_context,cl_device_id *opencl_devices,size_t num_devices,size_t max_parallel) :
    available_compute_resource(type),
    opencl_context(opencl_context),
    opencl_devices(opencl_devices),
    num_devices(num_devices),
    max_parallel(max_parallel)

  {
    
  }


  assigned_compute_resource::assigned_compute_resource(unsigned type) :
    type(type)
  {
    
  }
  
  assigned_compute_resource_cpu::assigned_compute_resource_cpu(unsigned type,const std::vector<size_t> &assigned_cpu_core_indices) :
    assigned_compute_resource(type),
    assigned_cpu_core_indices(assigned_cpu_core_indices)
  {
    
  }

  assigned_compute_resource_opencl::assigned_compute_resource_opencl(unsigned type,const std::vector<size_t> &assigned_cpu_core_indices,const std::vector<size_t> &assigned_opencl_job_indices,cl_context opencl_context,cl_device_id opencl_device) :
    assigned_compute_resource(type)
    assigned_cpu_core_indices(assigned_cpu_core_indices),
    assigned_opencl_job_indices(assigned_opencl_job_indices),
    opencl_context(opencl_context),
    opencl_device(opencl_device)
    
  {
    
  }
};
