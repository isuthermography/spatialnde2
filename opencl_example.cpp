
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
#include "geometrydata.h"
#include "opencl_utils.hpp"

#include "geometry_types_h.h"
#include "testkernel_c.h"

using namespace snde;



int main(int argc, char *argv[])
{

  std::shared_ptr<memallocator> lowlevel_alloc;
  std::shared_ptr<arraymanager> manager;
  std::shared_ptr<geometry> geom;

  // lowlevel_alloc performs the actual host-side memory allocations
  lowlevel_alloc=std::make_shared<cmemallocator>();

  // the arraymanager handles multiple arrays, including
  //   * Allocating space, reallocating when needed
  //   * Locking (implemented by manager.locker)
  //   * On-demand caching of array data to GPUs 
  manager=std::make_shared<arraymanager>(lowlevel_alloc);

  // geom is a C++ wrapper around a C data structure that
  // contains multiple arrays to be managed by the
  // arraymanager. These arrays are managed in
  // groups. All arrays in a group are presumed
  // to have parallel content, and are allocated,
  // freed, and locked in parallel.

  // Note that this initialization (adding arrays to
  // the arraymanager) is presumed to occur in a single-
  // threaded environment, whereas execution can be
  // freely done from multiple threads (with appropriate
  // locking of resources) 
  geom=std::make_shared<geometry>(1e-6,manager);




  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  std::string clmsgs;
  cl_kernel kernel;
  cl_program program;
  cl_int clerror=0;

  // get_opencl_context() is a convenience routine for obtaining an
  // OpenCL context and device. You pass it a query string of the
  // form <Platform or Vendor>:<CPU or GPU or ACCELERATOR>:<Device name or number>
  // and it will try to find the best match.
  
  std::tie(context,device,clmsgs) = get_opencl_context("::",false,NULL,NULL);

  fprintf(stderr,"%s",clmsgs.c_str());


  // Create a command queue for the specified context and device. This logic
  // tries to obtain one that permits out-of-order execution, if available. 
  queue=clCreateCommandQueue(context,device,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&clerror);
  if (clerror==CL_INVALID_QUEUE_PROPERTIES) {
    queue=clCreateCommandQueue(context,device,0,&clerror);

  }

  // Extract sourcecode for an OpenCL kernel by combining geometry_types.h and testkernel.c
  // which have been preprocessed into strings in header files. 
  std::vector<const char *> program_source = { geometry_types_h, testkernel_c };

  std::string build_log;

  // Create the OpenCL program object from the source code (convenience routine). 
  std::tie(program,build_log) = get_opencl_program(context,device,program_source);

  // Create the OpenCL kernel object
  kernel=clCreateKernel(program,"testkern",&clerror);
  if (!kernel) {
    throw openclerror(clerror,"Error creating OpenCL kernel");
  }
  
  
  rwlock_token_set all_locks;
  std::shared_ptr<std::vector<rangetracker<markedregion>>> readregions;
  std::shared_ptr<std::vector<rangetracker<markedregion>>> writeregions;


  // Begin a locking process. A locking process is a
  // (parallel) set of locking instructions, possibly
  // from multiple sources and in multiple sequences,
  // that is executed so as to follow a specified
  // locking order (thus preventing deadlocks).
  //
  // The general rule is that locking must follow
  // the specified order within a sequence, but if
  // needed additional sequences can be spawned that
  // will execute in parallel under the control
  // of the lock manager. 
  
  // The locking order
  // is the order of array creation in the arraymanager.
  // Within each array you must lock earlier regions
  // first. If you are going to lock the array for
  // both read and write, you must lock for write first.
  //
  // Note that the actual locking granularity implemented
  // currently is based on the whole array, but the
  // region specification is used to control flushing
  // changes after write locks.
  //
  // Also note that locking some regions for read and
  // some for write in a single array is currently
  // unsupported and may
  // to deadlocks. In this case, lock the entire
  // array for both read and write. 
  
  lockingprocess_threaded lockprocess(manager->locker);

  // ***!!! Should do allocation within the locking process, that gets us a write lock
  // ***!!! to the allocated region
  
  // Lock the specified region (in this case, the whole thing)
  // of the "vertices" array
  rwlock_token_set vertices_lock = lockprocess.get_locks_read_array_region((void **)&geom->geom.vertices,0,SNDE_INDEX_INVALID);

  // When the lock process is finished, you
  // get a reference to the full set of locks, and to the read locked and
  // write locked regions
  std::tie(all_locks,readregions,writeregions) = lockprocess.finish();


  // Buffers are OpenCL-accessible memory. You don't need to keep the buffers
  // open and active; the arraymanager maintains a cache, so if you
  // request a buffer a second time it will be there already. 
  OpenCLBuffers Buffers(context,all_locks,readregions,writeregions);

  // specify the arguments to the kernel, by argument number.
  // The third parameter is the array element to be passed
  // (actually comes from the OpenCL cache)
  Buffers.AddBufferAsKernelArg(manager,queue,kernel,0,(void **)&geom->geom.meshedparts);
  Buffers.AddBufferAsKernelArg(manager,queue,kernel,1,(void **)&geom->geom.triangles);
  Buffers.AddBufferAsKernelArg(manager,queue,kernel,2,(void **)&geom->geom.edges);
  Buffers.AddBufferAsKernelArg(manager,queue,kernel,3,(void **)&geom->geom.vertices);

  size_t global_work_size=100000;
  cl_event kernel_complete=NULL;

  // Enqueue the kernel 
  clEnqueueNDRangeKernel(queue,kernel,1,NULL,&global_work_size,NULL,Buffers.NumFillEvents(),Buffers.FillEvents_untracked(),&kernel_complete);
  // a clFlush() here would start the kernel executing, but
  // the kernel will alternatively start implicitly when we wait below. 

  // Queue up post-processing (i.e. cache maintenance) for the kernel
  // In this case we also ask it to wait for completion ("true")
  // Otherwise it could return immediately with those steps merely queued
  // (and we could do other stuff as it finishes in the background) 
  Buffers.RemBuffers(kernel_complete,kernel_complete,true);
  
  clReleaseEvent(kernel_complete); /* very important to release OpenCL resources, 
				      otherwise they may keep buffers in memory unnecessarily */

  // This explicitly unlocks the vertices_lock, so that
  // data should not be accessed afterward
  // Each locked set (or the whole set) can be
  // explicitly unlocked up to once (twice is an error).
  unlock_rwlock_token_set(vertices_lock);

  // In addition any locked sets not explicitly unlocked
  // will be implicitly locked once all references have
  // either been released or have gone out of scope. 
  release_rwlock_token_set(all_locks);
  
  
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  
  return 0;
}
