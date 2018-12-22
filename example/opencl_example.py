###!!!! Note: Due to a bug in the pOCL/ICD/etc. stack on fedora 26
### This crashes unless we export LD_PRELOAD=/usr/lib64/libpocl.so.2.0.0

import sys
import numpy as np
import spatialnde2
import pyopencl as cl


# get_opencl_context() is a convenience routine for obtaining an
# OpenCL context and device. You pass it a query string of the
# form <Platform or Vendor>:<CPU or GPU or ACCELERATOR>:<Device name or number>
# and it will try to find the best match.
(context,device,clmsgs)=spatialnde2.get_opencl_context("::",True,None,None);

sys.stderr.write(clmsgs)


# lowlevel_alloc performs the actual host-side memory allocations
lowlevel_alloc=spatialnde2.cmemallocator();

# alignment requirements specify constraints on allocation
# block sizes
alignment_requirements=spatialnde2.allocator_alignment()

# Each OpenCL device can impose an alignment requirement...
spatialnde2.add_opencl_alignment_requirement(alignment_requirements,device);


# the arraymanager handles multiple arrays, including
#   * Allocating space, reallocating when needed
#   * Locking (implemented by manager.locker)
#   * On-demand caching of array data to GPUs 
manager=spatialnde2.arraymanager(lowlevel_alloc,alignment_requirements)

# geometry is a C++ wrapper around a C data structure that
# contains multiple arrays to be managed by the
# arraymanager. These arrays are managed in
# groups. All arrays in a group are presumed
# to have parallel content, and are allocated,
# freed, and locked in parallel.

# Note that this initialization (adding arrays to
# the arraymanager) is presumed to occur in a single-
# threaded environment, whereas execution can be
# freely done from multiple threads (with appropriate
# locking of resources) 
geometry=spatialnde2.geometry(1e-6,manager)

# Can list geometry fields with:
# print(geometry.geom.contents)



# Create a command queue for the specified context and device. This logic
# tries to obtain one that permits out-of-order execution, if available. 
queueprops=0
if device.get_info(cl.device_info.QUEUE_PROPERTIES) & cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE:
    queueprops = cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE
    pass

queue=cl.CommandQueue(context,device,queueprops)

geometry_types_h=open("geometry_types.h").read()
testkernel_c = open("testkernel.c").read()

program_source=[ geometry_types_h, testkernel_c ]

# Extract sourcecode for an OpenCL kernel by combining geometry_types.h and testkernel.c
# which have been loaded from files, and compile the code. 
program = cl.Program(context," ".join(program_source)).build()

# Create the OpenCL kernel object
# NOTE: Kernel may be only used by one thread at a time (OpenCL limitation)
kernel=program.testkern_onepart # NOTE: must only extract kernel attribute once (see pyopencl documentation)


# Begin a locking process. A locking process is a
# (parallel) set of locking instructions, possibly
# from multiple sources and in multiple sequences,
# that is executed so as to follow a specified
# locking order (thus preventing deadlocks).

# The general rule is that locking must follow
# the specified order within a sequence, but if
# needed additional sequences can be spawned that
# will execute in parallel under the control
# of the lock manager. 
  
# The locking order
# is the order of array creation in the arraymanager.
# Within each array you must lock earlier regions
# first. If you are going to lock the array for
# both read and write, you must lock for write first.


# A lockholder is used to store the locks acquired
# during the locking process

holder = spatialnde2.lockholder()
all_locks = spatialnde2.pylockprocess(manager.locker,
                                      lambda proc: [  # Remember to follow locking order
                                          # Allocate pieces of these arrays and return the allocated pieces.
                                          # Note that it will temporarily lock the entire arrays for write
                                          # in order to do the allocation
                                          
                                          # Allocate a single entry in the "meshedparts" array
                                          holder.store_alloc((yield proc.alloc_array_region(manager,geometry.addr("meshedparts"),1,""))),
                                          # Note: Because the triangles allocator also allocates several other fields (per
                                          # comments in geometrydata.h and add_follower_array() call in geometry.hpp,
                                          # the allocation of "triangles" allocates and locks both it and the other arrays.  
                                          holder.store_alloc((yield proc.alloc_array_region(manager,geometry.addr("triangles"),1,""))),                                            
                                          holder.store_alloc((yield proc.alloc_array_region(manager,geometry.addr("edges"),3,""))),                                            
                                          holder.store_alloc((yield proc.alloc_array_region(manager,geometry.addr("vertices"),3,""))),
                                          holder.store_alloc((yield proc.alloc_array_region(manager,geometry.addr("vertex_edgelist"),6,""))),
                                      ])
# Can list the locks in holder with "print(holder)"
# Can access the allocation index with
#  holder.get_alloc(geometry.addr(<field_name>),<allocid>)
# Can access the allocation lock with
#  holder.get_alloc_lock(geometry.addr(<field_name>),<numelem>,<allocid>)
# Can access any lock with
#  holder.get(geometry.addr(<field_name>),<write>,<startidx>,<numelem>)


# Build a meshed part with exactly one triangle (three edges, and three vertices) 

# Create a numpy array representing the "meshedpart" allocation
meshedparts=geometry.allocfield(holder,"meshedparts",spatialnde2.nt_snde_meshedpart,"",1)
#meshedparts[0]["orientation"]["offset"]=np.zeros(3)
#meshedparts[0]["orientation"]["quat"]=(0,0,0,1) # identity quaternion

meshedparts[0]["firsttri"] = holder.get_alloc(geometry.addr("triangles"),"")
meshedparts[0]["numtris"] = 1

meshedparts[0]["firstedge"] = holder.get_alloc(geometry.addr("edges"),"")
meshedparts[0]["numedges"] = 3

meshedparts[0]["firstvertex"] = holder.get_alloc(geometry.addr("vertices"),"")
meshedparts[0]["numvertices"] = 3

meshedparts[0]["first_vertex_edgelist"] = holder.get_alloc(geometry.addr("vertex_edgelist_indices"),"")
meshedparts[0]["num_vertex_edgelist"] = 9
meshedparts[0]["firstbox"] = spatialnde2.SNDE_INDEX_INVALID
meshedparts[0]["numboxes"] = spatialnde2.SNDE_INDEX_INVALID
meshedparts[0]["firstboxpoly"] = spatialnde2.SNDE_INDEX_INVALID
meshedparts[0]["numboxpolys"] = spatialnde2.SNDE_INDEX_INVALID
meshedparts[0]["solid"] = False

triangles=geometry.allocfield(holder,"triangles",spatialnde2.nt_snde_triangle,"",1)
triangles[0][0]=0
triangles[0][1]=1
triangles[0][2]=2

edges=geometry.allocfield(holder,"edges",spatialnde2.nt_snde_edge,"",3)
edges[0]["vertex"]=(0,1)
edges[0]["face_a"]=0
edges[0]["face_b"]=spatialnde2.SNDE_INDEX_INVALID
edges[0]["face_a_next_edge"]=1
edges[0]["face_a_prev_edge"]=2
edges[1]["vertex"]=(1,2)
edges[1]["face_a"]=0
edges[1]["face_b"]=spatialnde2.SNDE_INDEX_INVALID
edges[1]["face_a_next_edge"]=2
edges[1]["face_a_prev_edge"]=0
edges[2]["vertex"]=(2,0)
edges[2]["face_a"]=0
edges[2]["face_b"]=spatialnde2.SNDE_INDEX_INVALID
edges[2]["face_a_next_edge"]=0
edges[2]["face_a_prev_edge"]=1



vertices=geometry.allocfield(holder,"vertices",spatialnde2.nt_snde_coord3,"",3)
vertices[0]=(1,0,0)
vertices[1]=(0,1,0)
vertices[2]=(0,0,0)

vertex_edgelist_indices=geometry.allocfield(holder,"vertex_edgelist_indices",spatialnde2.nt_snde_vertex_edgelist_index,"",3)
vertex_edgelist_indices[0]=(0,2)
vertex_edgelist_indices[1]=(2,2)
vertex_edgelist_indices[2]=(4,2)
vertex_edgelist=geometry.allocfield(holder,"vertex_edgelist",spatialnde2.nt_snde_index,"",6)
vertex_edgelist[0]=0
vertex_edgelist[1]=2
vertex_edgelist[2]=0
vertex_edgelist[3]=1
vertex_edgelist[4]=1
vertex_edgelist[5]=2

# Mark that we have made changes with the CPU
manager.dirty_alloc(holder,geometry.addr("meshedparts"),"",1)
manager.dirty_alloc(holder,geometry.addr("triangles"),"",1)
manager.dirty_alloc(holder,geometry.addr("edges"),"",3)
manager.dirty_alloc(holder,geometry.addr("vertices"),"",3)
manager.dirty_alloc(holder,geometry.addr("vertex_edgelist_indices"),"",3)
manager.dirty_alloc(holder,geometry.addr("vertex_edgelist"),"",6)

# Release the array references prior to unlocking -- a good idea to avoid dangling pointers to potentially invalid memory
del vertices
del edges
del triangles
del meshedparts


# Store which part address for later use
# Represent the above triangle as a "meshedpart"
part=spatialnde2.meshedpart(geometry,"my_triangle",holder.get_alloc(geometry.addr("meshedparts"),""))

# Unlock now that we have written
# (if we wanted we could use from the GPU under this same lock
# now that we have marked the modified regions as dirty)
spatialnde2.unlock_rwlock_token_set(all_locks)

del holder 


# Now (perhaps later) we want to use the GPU to calculate the surface area of this part...
# Begin a new locking process
holder = spatialnde2.lockholder()   # new holder
all_locks = spatialnde2.pylockprocess(manager.locker,
                                      lambda proc: part.obtain_lock_pycpp(proc,holder,spatialnde2.SNDE_COMPONENT_GEOM_ALL,0,0))  # 0 indicates flags for which arrays we want write and resize locks (none in this case)

# now we can access (read only) the data from the cpu
meshedparts=geometry.field(holder,"meshedparts",False,spatialnde2.nt_snde_meshedpart,part.idx,1)





# meshedparts is a numpy object...
# i.e. access meshedparts[0]["orientation"]["offset"], etc. 

# Buffers are OpenCL-accessible memory. You don't need to keep the buffers
# open and active; the arraymanager maintains a cache, so if you
# request a buffer a second time it will be there already. 
Buffers=spatialnde2.OpenCLBuffers(context,all_locks)

# specify the arguments to the kernel, by argument number.
# The third parameter is the array element to be passed
# (actually comes from the OpenCL cache)


#Buffers.AddBufferAsKernelArg(manager,queue,kernel,0,geometry.addr("meshedparts"),False)
Buffers.AddSubBufferAsKernelArg(manager,queue,kernel,0,geometry.addr("meshedparts"),part.idx,1,False)
Buffers.AddSubBufferAsKernelArg(manager,queue,kernel,1,geometry.addr("triangles"),meshedparts[0]["firsttri"],meshedparts[0]["numtris"],False)
Buffers.AddSubBufferAsKernelArg(manager,queue,kernel,2,geometry.addr("edges"),meshedparts[0]["firstedge"],meshedparts[0]["numedges"],False)
Buffers.AddSubBufferAsKernelArg(manager,queue,kernel,3,geometry.addr("vertices"),meshedparts[0]["firstvertex"],meshedparts[0]["numvertices"],False)
#kernel.set_arg(4,spatialnde2.nt_snde_index.type(part.idx));

worksize = meshedparts[0]["numtris"]
result_host=np.empty(worksize,dtype=spatialnde2.nt_snde_coord)

result_gpu = cl.Buffer(context,cl.mem_flags.COPY_HOST_PTR,hostbuf=result_host)

kernel.set_arg(4,result_gpu);

# print(Buffers.fill_events)

# Enqueue the kernel 
kernel_complete = cl.enqueue_nd_range_kernel(queue, kernel, (worksize,),None,wait_for=Buffers.fill_events)

xfer_complete=cl.enqueue_copy(queue,result_host,result_gpu,wait_for=(kernel_complete,))
    
# a clFlush() here would start the kernel executing, but
# the kernel will alternatively start implicitly when we wait below. 

# Queue up post-processing (i.e. cache maintenance) for the kernel
# In this case we also ask it to wait for completion ("True")
# Otherwise it could return immediately with those steps merely queued
# (and we could do other stuff as it finishes in the background) 
Buffers.RemBuffers(kernel_complete,kernel_complete,True);

xfer_complete.wait() # wait for result transfer to be complete
print("Triangle area=%f\n" % result_host[0])

# very important to release OpenCL resources, 
# otherwise they may keep buffers in memory unnecessarily 
del kernel_complete
del Buffers  # we don't need the buffers any more


# delete all references to our array prior to unlocking, lest the memory pointed to become invalid (not necessarily a problem if it's not accessed)

del meshedparts


# You can either (a) call RemBuffers() with the wait flag,
# and explicitly release all locks like this
spatialnde2.unlock_rwlock_token_set(all_locks);
# ...  OR (b) call RemBuffers() without the wait flag
# and implicitly release lock, e.g.
#   spatialnde2.release_rwlock_token_set(all_locks);
# in which case locks may not be unlocked until later,
# once all is finished. In this second approach the only
# way to be sure everything is done is to perform a
# new conflicting (i.e. write) lock on the relevant
# arrays
del holder


part.free()


