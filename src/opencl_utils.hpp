#ifndef SNDE_OPENCL_UTILS_HPP
#define SNDE_OPENCL_UTILS_HPP

#ifdef _MSC_VER
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#define strtok_r strtok_s
#endif

#include <vector>
#include <string>

#include <CL/opencl.h>

#include "allocator.hpp"
#include "snde_error_opencl.hpp"

namespace snde {

  std::tuple<cl_context,cl_device_id,std::string> get_opencl_context(std::string query,bool need_doubleprec,void (*pfn_notify)(const char *errinfo,const void *private_info, size_t cb, void *user_data),void *user_data);

  std::tuple<cl_program, std::string> get_opencl_program(cl_context context, cl_device_id device, std::vector<const char *> program_source);
  std::tuple<cl_program, std::string> get_opencl_program(cl_context context, cl_device_id device, std::vector<std::string> program_source);

  void add_opencl_alignment_requirement(std::shared_ptr<allocator_alignment> alignment,cl_device_id device);


  struct context_device { // used internal to class opencl_program as map key
    context_device(cl_context context,cl_device_id device) :
      context(context),
      device(device)
    {
      
    }
    
    cl_context context;
    cl_device_id device; 
  };
  
  // Need to provide hash and equality implementation for context_device so
  // it can be used as a std::unordered_map key
  
  struct context_device_hash
  {
    size_t operator()(const context_device & x) const
    {
      return
	std::hash<void *>{}((void *)x.context) +
			     std::hash<void *>{}((void *)x.device);
    }
  };
  
  struct context_device_equal {
    bool operator()(const context_device & x, const context_device & y) const
    {
      return x.context==y.context && x.device==y.device;
    }
    
  };
  
  
  class opencl_program {
    std::mutex program_mutex;
    std::string kern_fcn_name;
    std::vector<std::string> program_source;
    
    std::unordered_map<context_device,cl_program,context_device_hash,context_device_equal> program_dict;
  public:
    opencl_program(std::string kern_fcn_name,std::vector<std::string> program_source_strings) :
      kern_fcn_name(kern_fcn_name),
      program_source(program_source_strings)
    {
      
    } 
    //opencl_program(std::string kern_fcn_name,std::vector<const char *> program_source_strings) :
    //  kern_fcn_name(kern_fcn_name)
    //{
    //  for (auto cstr : program_source_strings) {
    //    program_source.push_back(cstr);
    //}  
    //
    cl_kernel get_kernel(cl_context context, cl_device_id device)
    // We create a new kernel every time because kernels aren't thread-safe
    // due to the nature of clSetKernelArg
    // be sure to call clReleaseKernel() on the kernel you get from this when you are done with it!
    {
      std::lock_guard<std::mutex> program_lock(program_mutex);
      cl_int clerror=0;
      cl_program program;
      context_device cd(context,device);
      
      if (!program_dict.count(cd)) {
	
	std::string build_log;
	
	// Create the OpenCL program object from the source code (convenience routine). 
	std::tie(program,build_log) = get_opencl_program(context,device,program_source);

	if (build_log.size() > 0) {
	  fprintf(stderr,"OpenCL build log:\n%s\n",build_log.c_str());
	}
	
	program_dict[cd]=program;
      }
      
      program=program_dict[cd];
      
      cl_kernel kernel;
      
      // Create the OpenCL kernel object
      kernel=clCreateKernel(program,kern_fcn_name.c_str(),&clerror);
      if (!kernel) {
	throw openclerror(clerror,"Error creating OpenCL kernel");
      }
      
      //kern_dict[cd]=kernel;
      
      return kernel; // kern_dict[cd];
      
    }
    

  };

  
}

#endif // SNDE_OPENCL_UTILS_HPP

