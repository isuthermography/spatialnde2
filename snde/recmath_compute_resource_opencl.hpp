#ifndef RECMATH_COMPUTE_RESOURCE_OPENCL_HPP
#define RECMATH_COMPUTE_RESOURCE_OPENCL_HPP

#include <CL/cl2.hpp>
#include "snde/recmath_compute_resource.hpp"

namespace snde {
  class openclcachemanager; // defined in openclcachemanager.hpp

  class assigned_compute_resource_opencl;


  class compute_resource_option_opencl: public compute_resource_option {
  public:
    compute_resource_option_opencl(std::set<std::string> execution_tags,
				   size_t metadata_bytes,
				   size_t data_bytes,
				   snde_float64 cpu_flops,
				   snde_float64 gpu_flops,
				   // Right now we just assume each function using
				   // a gpu takes up a single gpu slot
				   size_t max_useful_cpu_cores,
				   size_t useful_cpu_cores,
				   bool requires_doubleprec);

    virtual bool compatible_with(std::shared_ptr<available_compute_resource> available);

    snde_float64 cpu_flops; 
    snde_float64 gpu_flops; 
    size_t max_effective_cpu_cores; 
    size_t useful_cpu_cores;
    bool requires_doubleprec;
  };


  
  class _compute_resource_option_cpu_combined_opencl: public _compute_resource_option_cpu_combined {
    // This class is used internally by e.g. compute_resource_option_opencl, where once the OpenCL
    // option has been dispatched, it needs a CPU core to dispatch as well. So it gets one of these
    // structures as a wrapper and placed at the front of the priority list.
  public:
    _compute_resource_option_cpu_combined_opencl(std::set<std::string> execution_tags,
						 size_t metadata_bytes,
						 size_t data_bytes,
						 snde_float64 flops,
						 size_t max_effective_cpu_cores,
						 size_t useful_cpu_cores,
						 std::shared_ptr<compute_resource_option> orig,
						 std::shared_ptr<assigned_compute_resource> orig_assignment);

    _compute_resource_option_cpu_combined_opencl(const _compute_resource_option_cpu_combined_opencl &) = delete;  // CC and CAO are deleted because we don't anticipate needing them. 
    _compute_resource_option_cpu_combined_opencl& operator=(const _compute_resource_option_cpu_combined_opencl &) = delete; 
    virtual ~_compute_resource_option_cpu_combined_opencl()=default;  // virtual destructor required so we can be subclassed

    
    virtual std::shared_ptr<assigned_compute_resource> combine_cpu_assignment(std::shared_ptr<assigned_compute_resource_cpu> assigned_cpus);
    

  };



    class available_compute_resource_opencl: public available_compute_resource {
  public: 
      available_compute_resource_opencl(std::shared_ptr<recdatabase> recdb,std::set<std::string> tags,std::shared_ptr<available_compute_resource_cpu> controlling_cpu,cl::Context opencl_context,const std::vector<cl::Device> &opencl_devices,size_t max_parallel,std::shared_ptr<openclcachemanager> oclcache=nullptr);
    virtual void start(); // set the compute resource going
    virtual bool dispatch_code(std::unique_lock<std::mutex> &acrd_admin_lock);
    virtual std::tuple<int,bool,std::string> get_dispatch_priority(); // Get the dispatch priority of this compute resource. Smaller or more negative numbers are higher priority. See SNDE_ACRP_XXXX, above. Returns (dispatch_priority,fallback_flag,fallback_message)
    virtual size_t _number_of_free_gpus();
    virtual size_t _number_of_free_doubleprec_gpus();
    virtual std::shared_ptr<assigned_compute_resource_opencl> _assign_gpu(std::shared_ptr<math_function_execution> function_to_execute,bool requires_doubleprec);
    

    std::shared_ptr<available_compute_resource_cpu> controlling_cpu;
    cl::Context opencl_context;
    std::vector<cl::Device> opencl_devices;
    std::vector<bool> device_supports_double; // same length as opencl_devices

    size_t max_parallel; // max parallel jobs on a single device
    std::shared_ptr<openclcachemanager> oclcache;
    std::vector<std::shared_ptr<math_function_execution>> functions_using_devices; // length num_devices*max_parallel; indexing order: device0para0, device1para0, ... device0para1, device1para1,...   ... locked by the available_compute_resource_database admin lock. 

    std::vector<cl::CommandQueue> queues; // length num_devices*max_parallel, as with functions_using_devices

  };

  class assigned_compute_resource_opencl : public assigned_compute_resource {
  public:
    assigned_compute_resource_opencl(std::shared_ptr<available_compute_resource> resource,const std::vector<size_t> &assigned_cpu_core_indices,const std::vector<size_t> &assigned_opencl_job_indices,cl::Context context,const std::vector<cl::Device> &devices,const std::vector<bool> &device_supports_double,const std::vector<cl::CommandQueue> &queues,std::shared_ptr<openclcachemanager> oclcache);
    //size_t number_of_cpu_cores;
    std::vector<size_t> assigned_opencl_job_indices;
    
    cl::Context context;
    std::vector<cl::Device> devices; // devices corresponding to above-assigned opencl_job_indices
    std::vector<bool> device_supports_double; // same length as opencl_devices
    std::vector<cl::CommandQueue> queues; // devices corresponding to above-assigned opencl_job_indices
    std::shared_ptr<openclcachemanager> oclcache;
    std::shared_ptr<assigned_compute_resource_cpu> cpu_assignment; // contains assigned_cpu_core_indices

    
    virtual ~assigned_compute_resource_opencl()=default;  // virtual destructor required so we can be subclassed. Some subclasses use destructor to release resources

    virtual void release_assigned_resources(std::unique_lock<std::mutex> &acrd_admin_holder); // resources referenced below no longer meaningful once this is called. Must be called with acrd admin lock locked

  };
  
};

#endif // RECMATH_COMPUTE_RESOURCE_OPENCLHPP
