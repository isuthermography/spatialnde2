#include "CL/cl2.hpp"

#include "snde/opencl_utils.hpp"
#include "snde/recmath_compute_resource_opencl.hpp"
#include "snde/recstore_setup_opencl.hpp"

namespace snde {
  static void add_opencl_device_tags(cl_device_type devtype, std::set<std::string> *tags)
  {
  
    if (devtype & CL_DEVICE_TYPE_GPU) {
      tags->emplace("GPU");
    }
    if (devtype & CL_DEVICE_TYPE_ACCELERATOR) {
      tags->emplace("ACCELERATOR");
    }
    if (devtype & CL_DEVICE_TYPE_CPU) {
      tags->emplace("CPU");
    }
  }
  

  
  std::pair<cl::Context,std::vector<cl::Device>> setup_opencl(std::shared_ptr<recdatabase> recdb,std::set<std::string> tags,bool primary_doubleprec, size_t max_parallel, const char *primary_platform_prefix_or_null)
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

    if (!primary_platform_prefix_or_null) {
      // if platform prefix not specified, allow it to be specified via
      // SNDE_OPENCL_PLATFORM environment variable
      const char *snde_opencl_platform_env = std::getenv("SNDE_OPENCL_PLATFORM");

      if (snde_opencl_platform_env) {
	primary_platform_prefix_or_null = snde_opencl_platform_env;
      }
    }

    std::string ocl_query_string(":GPU:");
    if (primary_platform_prefix_or_null) {
      ocl_query_string = ssprintf("%s:GPU:",primary_platform_prefix_or_null);
    }
    std::tie(context,devices,clmsgs) = get_opencl_context(ocl_query_string,primary_doubleprec,nullptr,nullptr);

    if (!context.get() && primary_platform_prefix_or_null) {
      // remove GPU requirement if the user explicitly specified something and we didn't find it yet.
      ocl_query_string = ssprintf("%s::",primary_platform_prefix_or_null);
      std::tie(context,devices,clmsgs) = get_opencl_context(ocl_query_string,primary_doubleprec,nullptr,nullptr);
    }
    
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
      std::set<std::string> primary_tags(tags);
      primary_tags.emplace("OpenCL_Primary");
      if (devices.size() > 0) {
	add_opencl_device_tags(devices.at(0).getInfo<CL_DEVICE_TYPE>(),&primary_tags);
      }
      
      recdb->compute_resources->add_resource(std::make_shared<available_compute_resource_opencl>(recdb,primary_tags,recdb->compute_resources->cpu,context,devices,max_parallel)); // limit to max_parallel parallel jobs per GPU to limit contention
    }
    
    if (!opencl_check_doubleprec(devices)) {
    // fallback context, devices supporting double precision
      std::tie(context_dbl,devices_dbl,clmsgs_dbl) = get_opencl_context("::",true,nullptr,nullptr);

      if (!context_dbl.get()) {
	snde_warning("setup_opencl(): No fallback opencl platform with double precision support found");

      } else {
	fprintf(stderr,"OpenCL Fallback:\n%s\n\n",clmsgs_dbl.c_str());
	
	add_opencl_alignment_requirements(recdb,devices_dbl);
	std::set<std::string> fallback_tags(tags);
	fallback_tags.emplace("OpenCL_Fallback");
	if (devices.size() > 0) {
	  add_opencl_device_tags(devices.at(0).getInfo<CL_DEVICE_TYPE>(),&fallback_tags);
	}
	recdb->compute_resources->add_resource(std::make_shared<available_compute_resource_opencl>(recdb,fallback_tags,recdb->compute_resources->cpu,context_dbl,devices_dbl,max_parallel)); // limit to max_parallel parallel jobs per GPU to limit contention
      }
    }
  

    cl::Context context_to_return = context; 
    std::vector<cl::Device> devices_to_return = devices; 

    if (!context.get()) {
      context_to_return = context_dbl;
      devices_to_return = devices_dbl;
    }

    return std::make_pair(context_to_return,devices_to_return);
  }

  
};
