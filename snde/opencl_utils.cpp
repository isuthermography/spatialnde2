#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <unordered_map>
#include <memory>

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/opencl.hpp>
#endif

#include "snde/utils.hpp"
#include "snde/opencl_utils.hpp"

#define MAX_DEVICES 1000
#define MAX_PLATFORMS 1000



namespace snde {


  struct oclu_cldevice_hash {
    size_t operator()(const cl::Device & x) const
    {
      return std::hash<void *>{}((void *)x.get());
    }
  };
  
  struct oclu_cldevice_equal {
    bool operator()(const cl::Device & x,const cl::Device &y) const
    {
      return x.get()==y.get();
    }
  };
  
/* query should be of the same structure used for OpenCV: 
   <Platform or Vendor>:<CPU or GPU or ACCELERATOR>:<Device name or number> */
  // So-far only select one device at a time, but for the future
  // we return a vector of devices
  std::tuple<cl::Context,std::vector<cl::Device>,std::string> get_opencl_context(std::string query,bool need_doubleprec,void (*pfn_notify)(const char *errinfo,const void *private_info, size_t cb, void *user_data),void *user_data)
  {
    
    char *buf=strdup(query.c_str());
    char *SavePtr=NULL;
    char *Platform=NULL,*Type=NULL,*Device=NULL;
    cl::Context context;
    
    /* Rating scheme: 
       Exact match: 2 pts
       Partial match: 1 pts
       Inconsistent: -1 (disabled).
       
       All else equal and unless otherwise specified, a GPU outrates a CPU
    */
    
    std::unordered_map<cl::Device,std::tuple<cl::Platform,int,size_t>,oclu_cldevice_hash,oclu_cldevice_equal> ratings; /* int is rating, size_t is summarypos */
    
  
    Platform=c_tokenize(buf,':',&SavePtr);
    if (Platform) {
      Type=c_tokenize(NULL,':',&SavePtr);
      if (Type) {
	Device=c_tokenize(NULL,':',&SavePtr);      
      }
    }

    std::vector<cl::Platform> platforms;
    cl_uint platformnum,devicenum;
    std::vector<cl::Device> devices;
    int rating=0,platform_rating,type_rating,device_rating,doubleprec_rating;
    int maxrating=-1;
    

    std::string summary;
    
    cl::Platform::get(&platforms);
    
    for (platformnum=0;platformnum < platforms.size();platformnum++) {
      
      platforms.at(platformnum).getDevices(CL_DEVICE_TYPE_ALL,&devices);
      
      for (devicenum=0;devicenum < devices.size();devicenum++) {
	platform_rating=0;
	std::string PlatName=platforms.at(platformnum).getInfo<CL_PLATFORM_NAME>();
	std::string PlatVend=platforms.at(platformnum).getInfo<CL_PLATFORM_VENDOR>();
	if (Platform && strlen(Platform)) {
	  fprintf(stderr,"Platform=\"%s\"\n",Platform);
	  platform_rating=-1;
	  
	  if (!strcmp(Platform,PlatName.c_str())) {
	    platform_rating=2;
	  } else if (!strcmp(Platform,PlatVend.c_str())) {
	    platform_rating=2;
	  } else if (!strncmp(Platform,PlatName.c_str(),strlen(Platform))) {
	    platform_rating=1;
	  } else if (!strncmp(Platform,PlatVend.c_str(),strlen(Platform))) {
	    platform_rating=1;
	  } else {
	    platform_rating=-1;
	  }	
	  
	}
	
	
	type_rating=0;
	
	//clGetDeviceInfo(devices[devicenum],CL_DEVICE_TYPE,sizeof(gottype),&gottype,NULL);
	cl_device_type gottype=devices.at(devicenum).getInfo<CL_DEVICE_TYPE>();
	
	if (Type && strlen(Type)) {
	  type_rating=-1;
	  
	  if (!strcmp(Type,"GPU") && gottype & CL_DEVICE_TYPE_GPU) {
	    type_rating=2;
	  } else if (!strcmp(Type,"CPU") && gottype & CL_DEVICE_TYPE_CPU) {
	    type_rating=2;
	  } else if (!strcmp(Type,"ACCELERATOR") && gottype & CL_DEVICE_TYPE_ACCELERATOR) {
	    type_rating=2;
	  } else {
	    type_rating=-1;
	  }
	  
	} else {
	/* GPU gets a type rating of 1 if not otherwise specified */
	  if (gottype & CL_DEVICE_TYPE_GPU) {
	    type_rating=1;
	  }
	}
	
	
	device_rating=0;
	//std::string DevName=GetCLDeviceString(devices[devicenum],CL_DEVICE_NAME);
	std::string DevName=devices.at(devicenum).getInfo<CL_DEVICE_NAME>();
	if (Device && strlen(Device)) {
	  device_rating=-1;
	  
	  if (!strcmp(Device,DevName.c_str())) {
	    device_rating=2;
	  } else if (strlen(Device)==1 && ((unsigned)(Device[0]-'0'))==devicenum) {
	    device_rating=2;
	  } else if (!strncmp(Device,DevName.c_str(),strlen(Device))) {
	    device_rating=1;
	  }  else {
	    device_rating=-1;
	  }	
	  
	}
	
	/* check for 64 bit floating point support */
	//std::string DevExt=GetCLDeviceString(devices[devicenum],CL_DEVICE_EXTENSIONS);
	std::string DevExt=devices.at(devicenum).getInfo<CL_DEVICE_EXTENSIONS>();
	
	bool has_doubleprec = (DevExt.find("cl_khr_fp64") != std::string::npos);
	
	fprintf(stderr,"Platform: %s (rating %d); Device: %s (rating %d, type_rating %d) has_doubleprec=%d\n",PlatName.c_str(),platform_rating,DevName.c_str(),device_rating,type_rating,(int)has_doubleprec);
	
	
	doubleprec_rating=0;
	if (need_doubleprec && !has_doubleprec) {
	  doubleprec_rating=-1;
	} 
	
	
      
      
	summary.append(PlatName);
	summary.append(":");
	if (gottype & CL_DEVICE_TYPE_GPU) {
	  summary.append("GPU");
      } else if (gottype & CL_DEVICE_TYPE_CPU) {
	  summary.append("CPU");	
	} else if (gottype & CL_DEVICE_TYPE_ACCELERATOR) {
	  summary.append("ACCELERATOR");
	}
	summary.append(":");
	summary.append(DevName);
	summary.append(" (#");
	summary.append(std::to_string(devicenum));
	summary.append(")");
	
	if (has_doubleprec) {
	  summary.append(" (supports double precision)");
	} else {
	  summary.append(" (does not support double precision)");
	}
	
	size_t insertpos=summary.size();
	
	summary.append("\n");
      
	if (platform_rating >= 0 && type_rating >= 0 && device_rating >= 0 && doubleprec_rating >= 0) {
	  rating=platform_rating+type_rating+device_rating;
	  
	  ratings[devices[devicenum]] = std::make_tuple(platforms[platformnum],rating,insertpos);
	
	  if (rating > maxrating) {
	  maxrating=rating; 
	  }
	  
	}
	
	
      }
      
    }
    
    free(buf);
    buf=NULL;
    
    if (!ratings.size()) {
      fprintf(stderr,"No available OpenCL devices matched the given criteria (all rating categories >= 0)\n");
      exit(1);
    }
    
    cl::Device device;
    //cl_platform_id platform;
    cl::Platform platform;
    size_t insertpos;
    std::vector<cl::Device> selected_devices;
  
    for (auto dev_pfrtip : ratings) {
      
      device=dev_pfrtip.first;
      std::tie(platform,rating,insertpos)=dev_pfrtip.second;
      
      if (rating==maxrating) {
	
	summary.insert(insertpos," (SELECTED)");
	
	cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform()), 0, 0};
	cl_int errcode_ret=CL_SUCCESS;
	
	//context=clCreateContext(props,1,&device,pfn_notify,user_data,&errcode_ret);
	selected_devices.push_back(device);
	
	context=cl::Context(selected_devices,props,pfn_notify,user_data);
	
	//if (errcode_ret != CL_SUCCESS) {
	//summary.append("\nclCreateContext() failed (error "+std::to_string(errcode_ret)+")\n");	
	//}
	break;
      }
    }
    if (!device.get()) {
      summary.append("\nFailed to identify OpenCL device satisfiying specified requirements\n");
    }
    
    //free(platforms);
    //free(devices);
  
    return std::make_tuple(context,selected_devices,summary);
  }


  bool opencl_check_doubleprec(const std::vector<cl::Device> &devices)
  {
    bool some_have_doubleprec = false;
    size_t devnum;
    
    // check for double precision support

    for (devnum=0; devnum < devices.size();devnum++) {
      std::string DevExt=devices.at(devnum).getInfo<CL_DEVICE_EXTENSIONS>();
      bool has_doubleprec = (DevExt.find("cl_khr_fp64") != std::string::npos);

      if (has_doubleprec) {
	some_have_doubleprec=true;
      }
    }
    return some_have_doubleprec;
  }


  
  std::tuple<cl::Program, std::string> get_opencl_program(cl::Context context, cl::Device device, cl::Program::Sources program_source /* This is actual std::vector<std::string> */,bool build_with_doubleprec)
  {
    
    cl_int clerror=0;
    
    cl::Program program(context,program_source);
    //clCreateProgramWithSource(context,
    //program_source.size(),
    //&program_source[0],
    //NULL,
    //&clerror);
    //if (!program) {
    //  throw openclerror(clerror,"Error creating OpenCL program");
    //}
    
    //clerror=clBuildProgram(program,1,&device,"",NULL,NULL);
    std::string build_log_str="";
    std::string buildoptions="";
    if (build_with_doubleprec) {
      buildoptions +="-D SNDE_OCL_HAVE_DOUBLE ";
    }
    
    try {
      program.build(device,buildoptions.c_str());
    } catch (const cl::BuildError &e) {
      cl::BuildLogType buildlogs = e.getBuildLog();
      fprintf(stderr,"OpenCL Program build error!: size=%u\n",(unsigned)buildlogs.size());
      for (size_t cnt=0; cnt < buildlogs.size(); cnt++) {
	build_log_str += buildlogs.at(cnt).second;
	build_log_str += "\n\n";
      }
    }
    //size_t build_log_size=0;
    //char *build_log=NULL;
    //clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,0,NULL,&build_log_size);
    
    //build_log=(char *)calloc(1,build_log_size+1);
    //clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,build_log_size,(void *)build_log,NULL);
    
    //std::string build_log_str(build_log);
    if (build_log_str.size()==0) {
      build_log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
    }
      

    if (build_log_str.size() > 0) { // include source if there were errors/warnings
      build_log_str += "Source follows:\n";
      for (size_t pscnt=0;pscnt < program_source.size();pscnt++) {
	build_log_str += program_source[pscnt];
      }
    }
      
      //free(build_log);
    
      //if (clerror != CL_SUCCESS) {
      /* build error */
      //   throw openclerror(clerror,"Error building OpenCL program: %s\n",build_log_str.c_str());
      //}
    
    return std::make_tuple(program,build_log_str);
  }
  /*
  std::tuple<cl::Program, std::string> get_opencl_program(cl::Context context, cl::Device device, std::vector<std::string> program_source)
{
  std::vector<const char *> source_cstr(program_source.size());

  size_t cnt;
  for (cnt=0;cnt < program_source.size();cnt++) {
    source_cstr[cnt]=program_source[cnt].c_str();
  }

  return get_opencl_program(context,device,source_cstr);
}
  */

  
  void add_opencl_alignment_requirement(std::shared_ptr<allocator_alignment> alignment,cl::Device device)
{
  cl_uint align_value=0;
  cl_int err;
  //err=clGetDeviceInfo(device,CL_DEVICE_MEM_BASE_ADDR_ALIGN,sizeof(align_value),&align_value,NULL);
  //if (err != CL_SUCCESS || !align_value) {
    //throw openclerror(err,"Error obtaining OpenCL device alignment requirements");
  //  fprintf(stderr,"WARNING: Error obtaining OpenCL device alignment requirements... assuming 256 bits\n");
  //  align_value=256; 
  //}

  align_value = device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();
  if (align_value % 8) {
    throw openclerror(CL_SUCCESS,"OpenCL device memory alignment is not a multiple of 8 bits");
    
  }
  alignment->add_requirement(align_value/8);
}

  void add_opencl_alignment_requirements(std::shared_ptr<allocator_alignment> alignment,const std::vector<cl::Device> &devices)
  {

    for (auto && device: devices) {
      add_opencl_alignment_requirement(alignment,device);
    }
    
  }
  
}

