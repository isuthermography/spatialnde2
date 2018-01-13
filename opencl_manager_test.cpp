
#include <assert.h>
#include <string.h>
#include <cstdint>
#include <cstdarg>


#include <vector>
#include <map>
#include <condition_variable>
#include <deque>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <tuple>


#include "geometry_types.h"
#include "snde_error.hpp"
#include "memallocator.hpp"
#include "rangetracker.hpp"
#include "allocator.hpp"
#include "lockmanager.hpp"
#include "openclcachemanager.hpp"
#include "geometry.h"
#include "opencl_utils.hpp"

#include "geometry_types_h.h"
#include "testkernel_c.h"

using namespace snde;



int main(int argc, char *argv[])
{

  std::shared_ptr<memallocator> lowlevel_alloc;
  std::shared_ptr<arraymanager> manager;
  std::shared_ptr<geometry> geom;

  snde_index blockstart,blocksize;

  lowlevel_alloc=std::make_shared<cmemallocator>();

  //fprintf(stderr,"build manager...\n");
  manager=std::make_shared<arraymanager>(lowlevel_alloc);
  //fprintf(stderr,"build geom...\n");
  geom=std::make_shared<geometry>(1e-6,manager);


  /*
  // Allocate space for 10000 vertices 
  geom->manager->alloc((void **)&geom->geom.vertices,10000);
  
  // perform lock of all arrays
  rwlock_token_set all_locks=empty_rwlock_token_set();
  
  rwlock_token_set read_lock=geom->manager->locker->get_locks_read_all(all_locks);

  // unlock those arrays
  read_lock->clear();
  all_locks->clear();

  // lock single array
  rwlock_token_set vertices_lock=geom->manager->locker->get_locks_read_array(all_locks,(void **)&geom->geom.vertices);
  
  // lock following array, following locking order and acknowledging current lock ownership
  rwlock_token_set triangle_lock=geom->manager->locker->get_locks_read_array(all_locks,(void **)&geom->geom.vertexidx);

  // not legitimate to lock all arrays right now because this would violate locking order
  //fprintf(stderr,"release locks...\n");

  all_locks->clear();
  vertices_lock->clear();  // order of unlocks doesn't matter
  triangle_lock->clear();  // unlocks also happen automatically when the token_set leaves context. 
  */


  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  std::string clmsgs;
  cl_kernel kernel;
  cl_program program;
  cl_int clerror=0;
  
  std::tie(context,device,clmsgs) = get_opencl_context("::",false,NULL,NULL);

  fprintf(stderr,"%s",clmsgs.c_str());

  queue=clCreateCommandQueue(context,device,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,NULL);

  std::vector<const char *> program_source = { geometry_types_h, testkernel_c };
  
  program=clCreateProgramWithSource(context,
				    program_source.size(),
				    &program_source[0],
				    NULL,
				    &clerror);
  if (!program) {
    throw openclerror(clerror,"Error creating OpenCL program");
  }

  clerror=clBuildProgram(program,1,&device,"",NULL,NULL);
  if (clerror != CL_SUCCESS) {
    size_t build_log_size=0;
    char *build_log=NULL;
    clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,0,NULL,&build_log_size);

    build_log=(char *)calloc(1,build_log_size+1);
    clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,build_log_size,(void *)build_log,NULL);
    
    std::string build_log_str(build_log);
    free(build_log);
      
    throw openclerror(clerror,"Error building OpenCL program:\n"+build_log_str);
  }
  
  kernel=clCreateKernel(program,"testkern",&clerror);
  if (!kernel) {
    throw openclerror(clerror,"Error creating OpenCL kernel");
  }
  
  
  rwlock_token_set all_locks;
  std::shared_ptr<std::vector<rangetracker<markedregion>>> readregions;
  std::shared_ptr<std::vector<rangetracker<markedregion>>> writeregions;
  
  lockingprocess lockprocess(manager->locker);
  
  
  std::tie(all_locks,readregions,writeregions) = lockprocess.finish();

  
  OpenCLBuffers Buffers(context,all_locks,readregions,writeregions);
  
  Buffers.AddBufferAsKernelArg(manager,queue,kernel,0,(void **)&geom->geom.meshedparts,(void **)&geom->geom.meshedparts);
  //Buffers.AddBuffer(manager,queue,(void **)&geom->geom.meshedparts,(void **)&geom->geom.meshedparts);
  
  Buffers.AddBufferAsKernelArg(manager,queue,kernel,1,(void **)&geom->geom.vertices,(void **)&geom->geom.vertices);
  // Buffers.AddBufferAsKernelArg(manager,queue,kernel,2,&geom->geom.vertices,&geom->geom.principal_curvatures);
  // Buffers.AddBufferAsKernelArg(manager,queue,kernel,3,&geom->geom.vertices,&geom->geom.curvature_tangent_axes);

  Buffers.AddBufferAsKernelArg(manager,queue,kernel,2,(void **)&geom->geom.vertexidx,(void **)&geom->geom.vertexidx);
  //Buffers.AddBufferAsKernelArg(manager,queue,kernel,5,&geom->geom.boxes,&geom->geom.boxes);
  //Buffers.AddBufferAsKernelArg(manager,queue,kernel,6,&geom->geom.boxes,&geom->geom.boxcoord);
  
  //Buffers.AddBufferAsKernelArg(manager,queue,kernel,7,&geom->geom.boxpolys,&geom->geom.boxpolys);

  size_t global_work_size=100000;
  cl_event kernel_complete=NULL;

  clEnqueueNDRangeKernel(queue,kernel,1,NULL,&global_work_size,NULL,Buffers.NumFillEvents(),Buffers.FillEvents_untracked(),&kernel_complete);
  
  Buffers.RemBuffers(kernel_complete,kernel_complete,true);
  
  all_locks->clear();
  
  
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  
  return 0;
}
