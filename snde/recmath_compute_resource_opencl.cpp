#include "snde/recmath_compute_resource_opencl.hpp"
#include "snde/recstore.hpp"
#include "snde/recmath.hpp"

#include "snde/openclcachemanager.hpp"

namespace snde {


  compute_resource_option_opencl::compute_resource_option_opencl(std::set<std::string> execution_tags,
								 size_t metadata_bytes,
								 size_t data_bytes,
								 snde_float64 cpu_flops,
								 snde_float64 gpu_flops,
								 size_t max_effective_cpu_cores,
								 size_t useful_cpu_cores,
								 bool requires_doubleprec) :
    compute_resource_option(SNDE_CR_OPENCL,execution_tags,metadata_bytes,data_bytes),
    cpu_flops(cpu_flops),
    gpu_flops(gpu_flops),
    max_effective_cpu_cores(max_effective_cpu_cores),
    useful_cpu_cores(useful_cpu_cores),
    requires_doubleprec(requires_doubleprec)
  {

  }


  bool compute_resource_option_opencl::compatible_with(std::shared_ptr<available_compute_resource> available)
  {
    if (type==available->type) {
      std::shared_ptr<available_compute_resource_opencl> avail_ocl = std::dynamic_pointer_cast<available_compute_resource_opencl>(available);
      assert(avail_ocl);

      if (requires_doubleprec) {
	bool offers_doubleprec=false;
	size_t devnum;

	for (devnum=0;devnum < avail_ocl->opencl_devices.size();devnum++) {
	  if (avail_ocl->device_supports_double.at(devnum)) {
	    offers_doubleprec=true;
	    break;
	  }
	}

	return offers_doubleprec; 
	
      }
      
      return true;
    }
    return false;

  }

  _compute_resource_option_cpu_combined_opencl::_compute_resource_option_cpu_combined_opencl(std::set<std::string> execution_tags,
											     size_t metadata_bytes,
											     size_t data_bytes,
											     snde_float64 flops,
											     size_t max_effective_cpu_cores,
											     size_t useful_cpu_cores,
											     std::shared_ptr<compute_resource_option> orig,
											     std::shared_ptr<assigned_compute_resource> orig_assignment) :
    _compute_resource_option_cpu_combined(execution_tags,metadata_bytes,data_bytes,flops,
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

  available_compute_resource_opencl::available_compute_resource_opencl(std::shared_ptr<recdatabase> recdb,std::set<std::string> tags,std::shared_ptr<available_compute_resource_cpu> controlling_cpu,cl::Context opencl_context,const std::vector<cl::Device> &opencl_devices,size_t max_parallel,std::shared_ptr<openclcachemanager> oclcache/*=nullptr*/) :
    available_compute_resource(recdb,SNDE_CR_OPENCL,tags),
    controlling_cpu(controlling_cpu),
    opencl_context(opencl_context),
    opencl_devices(opencl_devices),
    max_parallel(max_parallel),
    oclcache( (oclcache) ? (oclcache):std::make_shared<openclcachemanager>()),
    functions_using_devices(max_parallel*opencl_devices.size())
    
  {
    // Figure out device_supports_double

    size_t devnum;

    for (devnum=0; devnum < opencl_devices.size();devnum++) {
      std::string DevExt=opencl_devices.at(devnum).getInfo<CL_DEVICE_EXTENSIONS>();
      
      bool has_doubleprec = (DevExt.find("cl_khr_fp64") != std::string::npos);

      device_supports_double.push_back(has_doubleprec);
    }

    // Create command queues
    size_t paranum;
    for (paranum=0; paranum < max_parallel; paranum++) {
      for (devnum=0; devnum < opencl_devices.size();devnum++) {
	queues.push_back(cl::CommandQueue(opencl_context,opencl_devices[devnum],CL_QUEUE_PROFILING_ENABLE));
      }
    }
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

	size_t free_gpus=0;
	if (compute_option_opencl->requires_doubleprec) {
	  free_gpus = _number_of_free_doubleprec_gpus();
	} else {
	  free_gpus = _number_of_free_gpus();
	}
	
	if (free_gpus > 0) {
	  std::shared_ptr<math_function_execution> function_to_execute=this_computation->function_to_execute;
	  std::shared_ptr<recording_set_state> recstate=this_computation->recstate;

	  prioritized_computations.erase(this_computation_it); // take charge of this computation
	  acrd_strong->todo_list.erase(this_computation); // remove from todo list so pointer can expire
	  this_computation = nullptr; // force pointer to expire so nobody else tries this computation;
	  
	  std::shared_ptr<assigned_compute_resource_opencl> assigned_gpus = _assign_gpu(function_to_execute,compute_option_opencl->requires_doubleprec);

	  // Create a combined resource we use to delegate the CPU portion 
	  std::shared_ptr<_compute_resource_option_cpu_combined_opencl> combined_resource = std::make_shared<_compute_resource_option_cpu_combined_opencl>(std::set<std::string>({"CPU","OpenCL_Dispatch"}),compute_option_opencl->metadata_bytes,compute_option_opencl->data_bytes,compute_option_opencl->cpu_flops,compute_option_opencl->max_effective_cpu_cores,compute_option_opencl->useful_cpu_cores,compute_option,assigned_gpus);
	  
	  std::shared_ptr<pending_computation> combined_computation = std::make_shared<pending_computation>(function_to_execute,recstate,globalrev,SNDE_CR_PRIORITY_NORMAL);
	  
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

  std::tuple<int,bool,std::string> available_compute_resource_opencl::get_dispatch_priority() // Get the dispatch priority of this compute resource. Smaller or more negative numbers are higher priority. See SNDE_ACRP_XXXX, above
  // returns (dispatch_priority,fallback_flag,fallback_message)
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
      //snde_warning("available_compute_resource_opencl: all OpenCL compute devices are actually CPU type. Treating as low-priority fallback.");
      return std::make_tuple(SNDE_ACRP_CPU_AS_GPU,true,"available_compute_resource_opencl: all OpenCL compute devices are actually CPU type. Treating as low-priority fallback.");
    }

    //if (any_devices_actually_cpu) {
    //  snde_warning("available_compute_resource_opencl: some OpenCL compute devices are actually CPU type.");
    //  
    //}
    
    return std::make_tuple(SNDE_ACRP_GPU_GENERALAPI,false,"");
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

  size_t available_compute_resource_opencl::_number_of_free_doubleprec_gpus()
  // Must call with ACRD admin lock locked
  {
    size_t number_of_free_gpus=0;
    size_t job_index=0;
    size_t num_devices = device_supports_double.size();

    for (auto && exec_fcn: functions_using_devices) {
      if (!exec_fcn && device_supports_double.at(job_index % num_devices)) {
	number_of_free_gpus++;
      }
      job_index++;
    }
    return number_of_free_gpus;
  }

  
  std::shared_ptr<assigned_compute_resource_opencl> available_compute_resource_opencl::_assign_gpu(std::shared_ptr<math_function_execution> function_to_execute,bool requires_doubleprec)
  // called with acrd admin lock held
  {
    
    size_t job_index=0;
    size_t num_devices = opencl_devices.size();
    std::vector<size_t> job_assignments;
    std::vector<cl::Device> device_assignments;
    std::vector<bool> device_double_assignments;
    std::vector<cl::CommandQueue> queue_assignments;

    for (auto && exec_fcn: functions_using_devices) {
      if (!exec_fcn && (!requires_doubleprec || ( device_supports_double.at(job_index % num_devices) ))) {
	// this gpu is available
	// assign it...
	job_assignments.push_back(job_index);
	device_assignments.push_back(opencl_devices.at(job_index % num_devices));
	device_double_assignments.push_back(device_supports_double.at(job_index % num_devices));
	queue_assignments.push_back(queues.at(job_index));
	
	break; 

      }
      job_index++;
    }

    assert(job_assignments.size() > 0); // should have been able to assign everything
    
    return std::make_shared<assigned_compute_resource_opencl>(shared_from_this(),std::vector<size_t>(),job_assignments,opencl_context,device_assignments,device_double_assignments,queue_assignments,oclcache);

  }



  assigned_compute_resource_opencl::assigned_compute_resource_opencl(std::shared_ptr<available_compute_resource> resource,const std::vector<size_t> &assigned_cpu_core_indices,const std::vector<size_t> &assigned_opencl_job_indices,cl::Context context,const std::vector<cl::Device> &devices,const std::vector<bool> &device_supports_double,const std::vector<cl::CommandQueue> &queues,std::shared_ptr<openclcachemanager> oclcache) :
    assigned_compute_resource(SNDE_CR_OPENCL,resource),
    assigned_opencl_job_indices(assigned_opencl_job_indices),
    context(context),
    devices(devices),
    device_supports_double(device_supports_double),
    queues(queues),
    oclcache(oclcache),
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



};

