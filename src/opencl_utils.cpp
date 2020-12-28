#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <unordered_map>
#include <memory>

#include <CL/opencl.h>

#include "utils.hpp"
#include "opencl_utils.hpp"

#define MAX_DEVICES 1000
#define MAX_PLATFORMS 1000



namespace snde {


static std::string GetCLPlatformString(cl_platform_id platform,cl_platform_info param_name)
{
  size_t numbytes=0;
  char *buf;
  
  clGetPlatformInfo(platform,param_name,0,NULL,&numbytes);

  buf=(char *)calloc(numbytes+1,1);
  clGetPlatformInfo(platform,param_name,numbytes+1,buf,NULL);

  std::string retval(buf);
  free(buf);

  return retval;
}

static std::string GetCLDeviceString(cl_device_id device,cl_device_info param_name)
{
  size_t numbytes=0;
  char *buf;
  
  clGetDeviceInfo(device,param_name,0,NULL,&numbytes);

  buf=(char *)calloc(numbytes+1,1);
  clGetDeviceInfo(device,param_name,numbytes+1,buf,NULL);
  
  std::string retval(buf);
  free(buf);

  return retval;
}

  
/* query should be of the same structure used for OpenCV: 
   <Platform or Vendor>:<CPU or GPU or ACCELERATOR>:<Device name or number> */
std::tuple<cl_context,cl_device_id,std::string> get_opencl_context(std::string query,bool need_doubleprec,void (*pfn_notify)(const char *errinfo,const void *private_info, size_t cb, void *user_data),void *user_data)
{

  char *buf=strdup(query.c_str());
  char *SavePtr=NULL;
  char *Platform=NULL,*Type=NULL,*Device=NULL;
  cl_context context=NULL;

  /* Rating scheme: 
     Exact match: 2 pts
     Partial match: 1 pts
     Inconsistent: -1 (disabled).

     All else equal and unless otherwise specified, a GPU outrates a CPU
 */

  std::unordered_map<cl_device_id,std::tuple<cl_platform_id,int,size_t>> ratings; /* int is rating, size_t is summarypos */
  
  
  Platform=c_tokenize(buf,':',&SavePtr);
  if (Platform) {
    Type=c_tokenize(NULL,':',&SavePtr);
    if (Type) {
      Device=c_tokenize(NULL,':',&SavePtr);      
    }
  }

  cl_platform_id *platforms;
  cl_uint platformnum,devicenum;
  cl_uint num_platforms=0;
  cl_device_id *devices;
  cl_uint num_devices;
  int rating=0,platform_rating,type_rating,device_rating,doubleprec_rating;
  int maxrating=-1;
  
  platforms=(cl_platform_id *)calloc(1,MAX_PLATFORMS*sizeof(cl_platform_id));
 
  devices=(cl_device_id *)calloc(1,MAX_DEVICES*sizeof(cl_device_id));

  std::string summary;

  clGetPlatformIDs(MAX_PLATFORMS,platforms,&num_platforms);

  for (platformnum=0;platformnum < num_platforms;platformnum++) {
    
    num_devices=0;
    clGetDeviceIDs(platforms[platformnum],CL_DEVICE_TYPE_ALL,MAX_DEVICES,devices,&num_devices);

    for (devicenum=0;devicenum < num_devices;devicenum++) {
      platform_rating=0;
      std::string PlatName=GetCLPlatformString(platforms[platformnum],CL_PLATFORM_NAME);
      std::string PlatVend=GetCLPlatformString(platforms[platformnum],CL_PLATFORM_VENDOR);
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
      cl_device_type gottype=CL_DEVICE_TYPE_CPU;
      
      clGetDeviceInfo(devices[devicenum],CL_DEVICE_TYPE,sizeof(gottype),&gottype,NULL);

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
      std::string DevName=GetCLDeviceString(devices[devicenum],CL_DEVICE_NAME);
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
      std::string DevExt=GetCLDeviceString(devices[devicenum],CL_DEVICE_EXTENSIONS);
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
  
  cl_device_id device=nullptr;
  cl_platform_id platform;
  size_t insertpos;
  
  for (auto dev_pfrtip : ratings) {

    device=dev_pfrtip.first;
    std::tie(platform,rating,insertpos)=dev_pfrtip.second;

    if (rating==maxrating) {
      
      summary.insert(insertpos," (SELECTED)");
      
      cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0, 0};
      cl_int errcode_ret=CL_SUCCESS;
      
      context=clCreateContext(props,1,&device,pfn_notify,user_data,&errcode_ret);

      if (errcode_ret != CL_SUCCESS) {
	summary.append("\nclCreateContext() failed (error "+std::to_string(errcode_ret)+")\n");	
      }
      break;
    }
  }
  if (!device) {
    summary.append("\nFailed to identify OpenCL device satisfiying specified requirements\n");
  }
	
  free(platforms);
  free(devices);
  
  return std::make_tuple(context,device,summary);
}


std::tuple<cl_program, std::string> get_opencl_program(cl_context context, cl_device_id device, std::vector<const char *> program_source)
  {
    cl_program program;
    cl_int clerror=0;
    
    program=clCreateProgramWithSource(context,
				      program_source.size(),
				      &program_source[0],
				      NULL,
				      &clerror);
    if (!program) {
      throw openclerror(clerror,"Error creating OpenCL program");
    }
    
    clerror=clBuildProgram(program,1,&device,"",NULL,NULL);
    
    size_t build_log_size=0;
    char *build_log=NULL;
    clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,0,NULL,&build_log_size);
    
    build_log=(char *)calloc(1,build_log_size+1);
    clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,build_log_size,(void *)build_log,NULL);
    
    std::string build_log_str(build_log);


    if (build_log_str.size() > 0) { // include source if there were errors/warnings
      build_log_str += "Source follows:\n";
      for (size_t pscnt=0;pscnt < program_source.size();pscnt++) {
	build_log_str += program_source[pscnt];
      }
    }
      
    free(build_log);
    
    if (clerror != CL_SUCCESS) {
      /* build error */
      throw openclerror(clerror,"Error building OpenCL program: %s\n",build_log_str.c_str());
    }
    
    return std::make_tuple(program,build_log_str);
  }

std::tuple<cl_program, std::string> get_opencl_program(cl_context context, cl_device_id device, std::vector<std::string> program_source)
{
  std::vector<const char *> source_cstr(program_source.size());

  size_t cnt;
  for (cnt=0;cnt < program_source.size();cnt++) {
    source_cstr[cnt]=program_source[cnt].c_str();
  }

  return get_opencl_program(context,device,source_cstr);
}

void add_opencl_alignment_requirement(std::shared_ptr<allocator_alignment> alignment,cl_device_id device)
{
  cl_uint align_value=0;
  cl_int err;
  err=clGetDeviceInfo(device,CL_DEVICE_MEM_BASE_ADDR_ALIGN,sizeof(align_value),&align_value,NULL);
  if (err != CL_SUCCESS || !align_value) {
    throw openclerror(err,"Error obtaining OpenCL device alignment requirements");
  }
  if (align_value % 8) {
    throw openclerror(err,"OpenCL device memory alignment is not a multiple of 8 bits");
    
  }
  alignment->add_requirement(align_value/8);
}
  
}

