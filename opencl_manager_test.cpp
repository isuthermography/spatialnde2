
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

  //snde_index blockstart,blocksize;

  lowlevel_alloc=std::make_shared<cmemallocator>();

  //fprintf(stderr,"build manager...\n");
  manager=std::make_shared<arraymanager>(lowlevel_alloc);
  //fprintf(stderr,"build geom...\n");
  geom=std::make_shared<geometry>(1e-6,manager);




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

  std::string build_log;
  
  std::tie(program,build_log) = get_opencl_program(context,device,program_source);
  
  kernel=clCreateKernel(program,"testkern",&clerror);
  if (!kernel) {
    throw openclerror(clerror,"Error creating OpenCL kernel");
  }
  
  
  rwlock_token_set locks;
  std::shared_ptr<std::vector<rangetracker<markedregion>>> readregions;
  std::shared_ptr<std::vector<rangetracker<markedregion>>> writeregions;
  
  lockingprocess lockprocess(manager->locker);
  
  
  std::tie(locks,readregions,writeregions) = lockprocess.finish();

  
  OpenCLBuffers Buffers(context,locks,readregions,writeregions);
  
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
  
  unlock_rwlock_token_set(locks);
  
  
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  
  return 0;
}
