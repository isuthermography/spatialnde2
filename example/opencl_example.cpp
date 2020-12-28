
#include <cassert>
#include <cstring>
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



  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  std::string clmsgs;
  cl_kernel kernel;
  cl_int clerror=0;

  std::shared_ptr<part> Part;
  
  // get_opencl_context() is a convenience routine for obtaining an
  // OpenCL context and device. You pass it a query string of the
  // form <Platform or Vendor>:<CPU or GPU or ACCELERATOR>:<Device name or number>
  // and it will try to find the best match.
  
  std::tie(context,device,clmsgs) = get_opencl_context("::",true,NULL,NULL);

  fprintf(stderr,"%s",clmsgs.c_str());


  std::shared_ptr<memallocator> lowlevel_alloc;
  std::shared_ptr<allocator_alignment> alignment_requirements;
  std::shared_ptr<arraymanager> manager;
  std::shared_ptr<geometry> geom;
  
  // lowlevel_alloc performs the actual host-side memory allocations
  lowlevel_alloc=std::make_shared<cmemallocator>();


  // alignment requirements specify constraints on allocation
  // block sizes
  alignment_requirements=std::make_shared<allocator_alignment>();
  // Each OpenCL device can impose an alignment requirement...
  add_opencl_alignment_requirement(alignment_requirements,device);
  
  // the arraymanager handles multiple arrays, including
  //   * Allocating space, reallocating when needed
  //   * Locking (implemented by manager.locker)
  //   * On-demand caching of array data to GPUs 
  manager=std::make_shared<arraymanager>(lowlevel_alloc,alignment_requirements);

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






  // Create a command queue for the specified context and device. This logic
  // tries to obtain one that permits out-of-order execution, if available. 
  queue=clCreateCommandQueue(context,device,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&clerror);
  if (clerror==CL_INVALID_QUEUE_PROPERTIES) {
    queue=clCreateCommandQueue(context,device,0,&clerror);

  }

  // Extract sourcecode for an OpenCL kernel by combining geometry_types.h and testkernel.c
  // which have been preprocessed into strings in header files.
  opencl_program area_program("testkern_onepart", { geometry_types_h, testkernel_c });
  

  // Create the OpenCL kernel object
  // NOTE: Kernel may be only used by one thread at a time (OpenCL limitation)
  kernel=area_program.get_kernel(context,device);
  
  

  /* new scope representing set of operations */
  {
    rwlock_token_set all_locks;

    
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
    
    // A lockholder is used to store the locks acquired
    // during the locking process
    std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();  
    std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(manager->locker); // new locking process
    
    
    // Allocate a single entry in the "parts" array
    holder->store_alloc(lockprocess->alloc_array_region(manager,(void **)&geom->geom.parts,1,""));
    // Note: Because the triangles allocator also allocates several other fields (per
    // comments in geometrydata.h and add_follower_array() call in geometry.hpp,
    // the allocation of "triangles" allocates and locks both it and the other arrays.  
    holder->store_alloc(lockprocess->alloc_array_region(manager,(void **)&geom->geom.triangles,1,""));
    holder->store_alloc(lockprocess->alloc_array_region(manager,(void **)&geom->geom.edges,3,""));
    holder->store_alloc(lockprocess->alloc_array_region(manager,(void **)&geom->geom.vertices,3,""));
    holder->store_alloc(lockprocess->alloc_array_region(manager,(void **)&geom->geom.vertex_edgelist,6,""));
    
    // When the lock process is finished, you
    // get a reference to the full set of locks 
    all_locks=lockprocess->finish();
    
    
    // Can list the locks in holder with "holder->as_string()"
    // Can access the allocation index with
    //   holder->get_alloc(&geom->geom.<field_name>,<allocid>)
    // Can access the allocation lock with
    //   holder->get_alloc_lock(&geom->geom.<field_name>),<numelem>,<allocid>)
    // Can access any lock with
    //   holder->get(&geom.geom.<field_name>,<write>,<startidx>,<numelem>)
    
    
    // Build a meshed part with exactly one triangle (three edges, and three vertices) 
    
    // Define an array pointer representing the "meshedparts allocation
    struct snde_part *parts=geom->geom.parts+holder->get_alloc((void**)&geom->geom.parts,"");
    //parts[0].orientation.offset[0]=0;
    //parts[0].orientation.offset[1]=0;
    //parts[0].orientation.offset[2]=0;
    
    //parts[0].orientation.quat[0]=0;
    //parts[0].orientation.quat[1]=0;
    //parts[0].orientation.quat[2]=0;
    //parts[0].orientation.quat[3]=1;
    
    parts[0].firsttri =  holder->get_alloc((void **)&geom->geom.triangles,"");
    parts[0].numtris = 1;
    
    parts[0].firstedge =  holder->get_alloc((void **)&geom->geom.edges,"");
    parts[0].numedges = 3;
    
    parts[0].firstvertex =  holder->get_alloc((void **)&geom->geom.vertices,"");
    parts[0].numvertices = 3;

    parts[0].first_vertex_edgelist = holder->get_alloc((void **)&geom->geom.vertex_edgelist_indices,"");
    parts[0].num_vertex_edgelist = 6;
    parts[0].firstbox = SNDE_INDEX_INVALID;
    parts[0].numboxes = SNDE_INDEX_INVALID;
    parts[0].firstboxpoly = SNDE_INDEX_INVALID;
    parts[0].numboxpolys = SNDE_INDEX_INVALID;
    parts[0].solid = false;
    
    snde_triangle *triangles=geom->geom.triangles+holder->get_alloc((void**)&geom->geom.triangles,"");
    triangles[0].edges[0]=0;
    triangles[0].edges[1]=1;
    triangles[0].edges[2]=2;
    
    
    snde_edge *edges=geom->geom.edges+holder->get_alloc((void**)&geom->geom.edges,"");
    edges[0].vertex[0]=0;
    edges[0].vertex[1]=1;
    edges[0].tri_a=0;
    edges[0].tri_b=SNDE_INDEX_INVALID;
    edges[0].tri_a_next_edge=1;
    edges[0].tri_a_prev_edge=2;
    edges[1].vertex[0]=1;
    edges[1].vertex[1]=2;
    edges[1].tri_a=0;
    edges[1].tri_b=SNDE_INDEX_INVALID;
    edges[1].tri_a_next_edge=2;
    edges[1].tri_a_prev_edge=0;
    edges[2].vertex[0]=2;
    edges[2].vertex[1]=0;
    edges[2].tri_a=0;
    edges[2].tri_b=SNDE_INDEX_INVALID;
    edges[2].tri_a_next_edge=0;
    edges[2].tri_a_prev_edge=1;
    
    
    snde_coord3 *vertices=geom->geom.vertices+holder->get_alloc((void **)&geom->geom.vertices,"");
    
    vertices[0].coord[0]=1;
    vertices[0].coord[1]=0;
    vertices[0].coord[2]=0;
    
    vertices[1].coord[0]=0;
    vertices[1].coord[1]=1;
    vertices[1].coord[2]=0;
    
    vertices[2].coord[0]=0;
    vertices[2].coord[1]=0;
    vertices[2].coord[2]=0;
    
    snde_vertex_edgelist_index *vertex_edgelist_indices=geom->geom.vertex_edgelist_indices+holder->get_alloc((void**)&geom->geom.vertex_edgelist_indices,"");
  
    vertex_edgelist_indices[0].edgelist_index=0;
    vertex_edgelist_indices[0].edgelist_numentries=2;
    vertex_edgelist_indices[1].edgelist_index=2;
    vertex_edgelist_indices[1].edgelist_numentries=2;
    vertex_edgelist_indices[2].edgelist_index=4;
    vertex_edgelist_indices[2].edgelist_numentries=2;
    
    snde_index *vertex_edgelist=geom->geom.vertex_edgelist+holder->get_alloc((void**)&geom->geom.vertex_edgelist,"");
    vertex_edgelist[0]=0;
    vertex_edgelist[1]=2;
    vertex_edgelist[2]=0;
    vertex_edgelist[3]=1;
    vertex_edgelist[4]=1;
    vertex_edgelist[5]=2;
    
    // Mark that we have made changes with the CPU
    manager->dirty_alloc(holder,(void **)&geom->geom.parts,"",1);
    manager->dirty_alloc(holder,(void **)&geom->geom.triangles,"",1);
    manager->dirty_alloc(holder,(void **)&geom->geom.edges,"",3);
    manager->dirty_alloc(holder,(void **)&geom->geom.vertices,"",3);
    manager->dirty_alloc(holder,(void **)&geom->geom.vertex_edgelist_indices,"",3);
    manager->dirty_alloc(holder,(void **)&geom->geom.vertex_edgelist,"",6);
    
    // Store which part address for later use
    // Represent the above triangle as a "meshedpart"
    Part=std::make_shared<part>(geom,holder->get_alloc((void **)&geom->geom.parts,""));
    
    // Unlock now that we have written
    // (if we wanted we could use from the GPU under this same lock
    // now that we have marked the modified regions as dirty)
    unlock_rwlock_token_set(all_locks);
  } // Ending the block makes the various lock and reference variables go out of scope

  
  // Now (perhaps later) we want to use the GPU to calculate the surface area of this part...

  
  /* new scope representing set of operations */
  {
    rwlock_token_set all_locks;
    // Begin a new locking process
    std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();  
    
    std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(manager->locker); // new locking process
    
    obtain_graph_lock(lockprocess,
		      Part,
		      std::vector<std::string>(), // no extra channels
		      std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>>(), // no extra components
		      nullptr,"/",
		      SNDE_COMPONENT_GEOM_ALL,
		      0); // 0 indicates flags for which arrays we want write locks (none in this case)
    all_locks=lockprocess->finish();
    
    // now we can access (read only) the data from the cpu
    struct snde_part *parts=geom->geom.parts+Part->idx();
    
    
    
    //lockprocess.spawn([ &lockprocess, &holder, &geom ]() {
    //    //std::tie(vertices_lock,vertices_index)=lockprocess.alloc_array_region(manager,(void **)&geom->geom.vertices,3);
    //    holder->store_alloc(lockprocess.alloc_array_region(manager,(void **)&geom->geom.vertices,3,""));
    //  });
    
    //rwlock_token_set vertices_lock = lockprocess.get_locks_read_array_region((void **)&geom->geom.vertices,0,SNDE_INDEX_INVALID);
    
    
    
    // Buffers are OpenCL-accessible memory. You don't need to keep the buffers
    // open and active; the arraymanager maintains a cache, so if you
    // request a buffer a second time it will be there already. 
    OpenCLBuffers Buffers(context,device,all_locks);
    
    // specify the arguments to the kernel, by argument number.
    // The third parameter is the array element to be passed
    // (actually comes from the OpenCL cache)
    Buffers.AddSubBufferAsKernelArg(manager,kernel,0,(void **)&geom->geom.parts,Part->idx(),1,false);
    Buffers.AddSubBufferAsKernelArg(manager,kernel,1,(void **)&geom->geom.triangles,parts[0].firsttri,parts[0].numtris,false);
    Buffers.AddSubBufferAsKernelArg(manager,kernel,2,(void **)&geom->geom.edges,parts[0].firstedge,parts[0].numedges,false);
    Buffers.AddSubBufferAsKernelArg(manager,kernel,3,(void **)&geom->geom.vertices,parts[0].firstvertex,parts[0].numvertices,false);
    
    size_t worksize=parts[0].numtris;
    
    snde_coord *result_host=(snde_coord*)calloc(1,worksize*sizeof(*result_host));
    cl_int err=CL_SUCCESS;
    cl_mem result_gpu=clCreateBuffer(context,CL_MEM_COPY_HOST_PTR,worksize*sizeof(*result_host),result_host,&err);
    if (err != CL_SUCCESS) {
      throw openclerror(err,"Error creating OpenCL buffer");
    }
    cl_event kernel_complete=NULL;
    
    err=clSetKernelArg(kernel,4,sizeof(cl_mem),&result_gpu);
    if (err != CL_SUCCESS) {
      throw openclerror(err,"Error setting kernel argument");
    }
    
    // Enqueue the kernel 
    err=clEnqueueNDRangeKernel(queue,kernel,1,NULL,&worksize,NULL,Buffers.NumFillEvents(),Buffers.FillEvents_untracked(),&kernel_complete);
    if (err != CL_SUCCESS) {
      throw openclerror(err,"Error enqueueing kernel");
    }
    
    cl_event xfer_complete=NULL;
    // The only reason we need to explicitly transfer our
    // result is that it didn't go into an array (if it did, the
    // opencl cache manager would manage the transfer) 
    err=clEnqueueReadBuffer(queue,result_gpu,CL_FALSE,0,worksize*sizeof(*result_host),result_host,1,&kernel_complete,&xfer_complete);
    
    if (err != CL_SUCCESS) {
      throw openclerror(err,"Error enqueueing OpenCL read");
    }

    
    // a clFlush() here would start the kernel executing, but
    // the kernel will alternatively start implicitly when we wait below. 
    
    // Queue up post-processing (i.e. cache maintenance) for the kernel
    // In this case we also ask it to wait for completion ("true")
    // Otherwise it could return immediately with those steps merely queued
    // (and we could do other stuff as it finishes in the background) 
    Buffers.RemBuffers(kernel_complete,kernel_complete,true);

    clWaitForEvents(1,&xfer_complete); // wait for result transfer to be complete

    printf("Triangle area=%f\n",(float)result_host[0]);

    
    clReleaseEvent(kernel_complete); /* very important to release OpenCL resources, 
					otherwise they may keep buffers in memory unnecessarily */
    
    
    // In addition any locked sets not explicitly unlocked
    // will be implicitly locked once all references have
    // either been released or have gone out of scope. 
    unlock_rwlock_token_set(all_locks);


    clReleaseMemObject(result_gpu);
    free(result_host);
    Part->free(); // explicitly delete the part

    // Ending the block makes the various lock and reference variables go out of scope
  }
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  clReleaseKernel(kernel);
  
  return 0;
}
