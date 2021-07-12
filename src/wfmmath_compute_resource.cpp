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
