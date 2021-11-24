#include "CL/opencl.hpp"

#include "snde/opencl_utils.hpp"
#include "snde/recmath_compute_resource_opencl.hpp"
#include "snde/recstore_setup_opencl.hpp"

namespace snde {
  std::tuple<cl::Context,std::vector<cl::Device>> setup_opencl(std::shared_ptr<recdatabase> recdb,bool primary_doubleprec, size_t max_parallel, char *primary_platform_prefix_or_null)
  {
    cl::Context context,context_dbl;
    std::vector<cl::Device> devices,devices_dbl;
    std::string clmsgs,clmsgs_dbl;
    
    // The first parameter to get_opencl_context can be used to match a specific device, e.g. "Intel(R) OpenCL HD Graphics:GPU:Intel(R) Iris(R) Xe Graphics"
    // with the colon-separated fields left blank.
    // Set the second (boolean) parameter to limit to devices that can
    // handle double-precision

    if (!recdb->compute_resources->cpu) {
      throw snde_error("setup_opencl(): CPU compute resource must be setup first (see setup_cpu())");
    }

    std::string ocl_query_string(":GPU:");
    if (primary_platform_prefix_or_null) {
      ocl_query_string = ssprintf("%s:GPU:",primary_platform_prefix_or_null);
    }
    std::tie(context,devices,clmsgs) = get_opencl_context(ocl_query_string,primary_doubleprec,nullptr,nullptr);

    if (!context.get()) {
      snde_warning("setup_opencl(): No matching primary GPU found");
    } else {
      
      // NOTE: If using Intel graphics compiler (IGC) you can enable double
      // precision emulation even on single precision hardware with the
      // environment variable OverrideDefaultFP64Settings=1
      // https://github.com/intel/compute-runtime/blob/master/opencl/doc/FAQ.md#feature-double-precision-emulation-fp64
      fprintf(stderr,"OpenCL Primary:\n%s\n\n",clmsgs.c_str());
      
  
      // Each OpenCL device can impose an alignment requirement...
      add_opencl_alignment_requirements(recdb,devices);
      
      recdb->compute_resources->add_resource(std::make_shared<available_compute_resource_opencl>(recdb,recdb->compute_resources->cpu,context,devices,max_parallel)); // limit to max_parallel parallel jobs per GPU to limit contention
    }
    
    if (!opencl_check_doubleprec(devices)) {
    // fallback context, devices supporting double precision
      std::tie(context_dbl,devices_dbl,clmsgs_dbl) = get_opencl_context("::",true,nullptr,nullptr);

      if (!context_dbl.get()) {
	snde_warning("setup_opencl(): No fallback opencl platform with double precision support found");

      } else {
	fprintf(stderr,"OpenCL Fallback:\n%s\n\n",clmsgs_dbl.c_str());
	
	add_opencl_alignment_requirements(recdb,devices_dbl);
	recdb->compute_resources->add_resource(std::make_shared<available_compute_resource_opencl>(recdb,recdb->compute_resources->cpu,context_dbl,devices_dbl,max_parallel)); // limit to max_parallel parallel jobs per GPU to limit contention
      }
    }
  

    cl::Context context_to_return = context; 
    std::vector<cl::Device> devices_to_return = devices; 

    if (!context.get()) {
      context_to_return = context_dbl;
      devices_to_return = devices_dbl;
    }

    return std::make_tuple(context_to_return,devices_to_return);
  }

  
};
