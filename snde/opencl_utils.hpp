#ifndef SNDE_OPENCL_UTILS_HPP
#define SNDE_OPENCL_UTILS_HPP

#ifdef _MSC_VER
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#define strtok_r strtok_s
#endif

#include <vector>
#include <string>

#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif

//#include <CL/opencl.h>

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl2.hpp>
#endif



#include "snde/allocator.hpp"
#include "snde/snde_error_opencl.hpp"

namespace snde {

  class recdatabase;

  std::tuple<cl::Context,std::vector<cl::Device>,std::string> get_opencl_context(std::string query,bool need_doubleprec,void (*pfn_notify)(const char *errinfo,const void *private_info, size_t cb, void *user_data),void *user_data);

  bool opencl_check_doubleprec(const std::vector<cl::Device> &devices);


  std::tuple<cl::Program, std::string> get_opencl_program(cl::Context context, cl::Device device, cl::Program::Sources program_source /* This is actual std::vector<std::string> */,bool build_with_doubleprec);


  
  void add_opencl_alignment_requirement(std::shared_ptr<recdatabase> recdb,cl::Device device);
  void add_opencl_alignment_requirements(std::shared_ptr<recdatabase> recdb,const std::vector<cl::Device> &devices);


#if 0 // this stuff is obsolste
  struct _snde_cl_retain_command_queue {
    static void retain(cl_command_queue queue)
    {
      clRetainCommandQueue(queue);
    }
  };
  struct _snde_cl_release_command_queue {
    static void release(cl_command_queue queue)
    {
      clReleaseCommandQueue(queue);
    }
  };
  
  
  template <typename T,typename retaincls, typename releasecls>
  struct clobj_wrapper {
    T clobj;
    clobj_wrapper() :
      clobj(nullptr)
    {
      
    }
    
    clobj_wrapper(T clobj) :
      clobj(clobj)
    {
      /* eats exactly one reference on instantiation. If you 
	 don't own the reference provided, you should Retain() first */
     
    }

    clobj_wrapper(const clobj_wrapper &orig) :
      clobj(orig.clobj)
    {
      if (clobj) {
	retaincls::retain(clobj);
      }
    }
    
    clobj_wrapper & operator=(const clobj_wrapper &orig) 
    {
      if (clobj) {
	releasecls::release(clobj);
      }
      clobj=orig.clobj;

      if (clobj) {
	retaincls::retain(clobj);
      }
    }

    T get_noref()
    {
      return clobj;
    }
    
    T get_ref() // return with a retain reference
    {
      if (clobj) {
	retaincls::retain(clobj);
      }
      return clobj;
    }

  };

  typedef clobj_wrapper<cl_command_queue,_snde_cl_retain_command_queue,_snde_cl_release_command_queue> cl_command_queue_wrapped;
#endif // 0 (obsolete)

  
  struct context_device { // used internal to class opencl_program as map key
    cl::Context context;
    cl::Device device; 

    context_device(cl::Context context,cl::Device device) :
      context(context),
      device(device)
    {
      //clRetainContext(this->context); /* increase refcnt */
      //clRetainDevice(this->device);      
    }
    
    //context_device(const context_device &orig) /* copy constructor */
    //{
    //  context=orig.context;
    //  clRetainContext(context);
    //  device=orig.device;
    //  clRetainDevice(device);
    //}
    
    //context_device & operator=(const context_device &orig) /* copy assignment operator */
    //{
    //  clReleaseContext(context);
    //  context=orig.context;
    //  clRetainContext(context);
    //  device=orig.device;
    //  clRetainDevice(device);
    //  
    //  return *this;
    // }

    // equality operator for std::unordered_map
    bool operator==(const context_device b) const
    {
      return b.context==context&& b.device==device;
    }
    
    //~context_device() {
    //  clReleaseContext(context);
    //  clReleaseDevice(device);
    //}
  };
  
  // Need to provide hash and equality implementation for context_device so
  // it can be used as a std::unordered_map key
  
  struct context_device_hash
  {
    size_t operator()(const context_device & x) const
    {
      return
	std::hash<void *>{}((void *)x.context.get()) +
			     std::hash<void *>{}((void *)x.device.get());
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
    
    std::unordered_map<context_device,cl::Program,context_device_hash,context_device_equal> program_dict;
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
    cl::Kernel get_kernel(cl::Context context, cl::Device device)
    // We create a new kernel every time because kernels aren't thread-safe
    // due to the nature of clSetKernelArg
    // be sure to call clReleaseKernel() on the kernel you get from this when you are done with it!
    {
      std::lock_guard<std::mutex> program_lock(program_mutex);
      cl_int clerror=0;
      cl::Program program;
      context_device cd(context,device);
      
      if (!program_dict.count(cd)) {
	
	std::string build_log;

	// check for double precision support
	std::string DevExt=device.getInfo<CL_DEVICE_EXTENSIONS>();
	bool has_doubleprec = (DevExt.find("cl_khr_fp64") != std::string::npos);

	
	// Create the OpenCL program object from the source code (convenience routine). 
	std::tie(program,build_log) = get_opencl_program(context,device,program_source,has_doubleprec);

	if (build_log.size() > 0) {
	  fprintf(stderr,"OpenCL build log:\n%s\n",build_log.c_str());
	}
	
	program_dict[cd]=program;
      }
      
      program=program_dict[cd];
      
      cl::Kernel kernel;
      
      // Create the OpenCL kernel object
      try {
	kernel=cl::Kernel(program,kern_fcn_name.c_str()); // ,&clerror);
      } catch (cl::Error &e) {
	throw snde_error("OpenCL Error: %s(%s)",e.what(),openclerrorstring.at(e.err()).c_str());
      }
      //if (!kernel) {
      //throw openclerror(clerror,"Error creating OpenCL kernel");
      //}
      
      //kern_dict[cd]=kernel;
      
      return kernel; // kern_dict[cd];
      
    }
    

  };

  
}

#endif // SNDE_OPENCL_UTILS_HPP

