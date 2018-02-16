import sys
import numpy as np
import spatialnde2
import pyopencl as cl


# lowlevel_alloc performs the actual host-side memory allocations
lowlevel_alloc=spatialnde2.cmemallocator();

# the arraymanager handles multiple arrays, including
#   * Allocating space, reallocating when needed
#   * Locking (implemented by manager.locker)
#   * On-demand caching of array data to GPUs 
manager=spatialnde2.arraymanager(lowlevel_alloc)

# geom is a C++ wrapper around a C data structure that
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

# get_opencl_context() is a convenience routine for obtaining an
# OpenCL context and device. You pass it a query string of the
# form <Platform or Vendor>:<CPU or GPU or ACCELERATOR>:<Device name or number>
# and it will try to find the best match.
(context,device,clmsgs)=spatialnde2.get_opencl_context("::",False,None,None);

sys.stderr.write(clmsgs)

# Create a command queue for the specified context and device. This logic
# tries to obtain one that permits out-of-order execution, if available. 
queueprops=0
if device.get_info(cl.device_info.QUEUE_PROPERTIES) & cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE:
    queueprops = cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE
    pass

commandqueue=cl.CommandQueue(context,device,queueprops)

geometry_types_h=file("geometry_types.h").read()
testkernel_c = file("testkernel.c").read()

program_source=[ geometry_types_h, testkernel_c ]

# Extract sourcecode for an OpenCL kernel by combining geometry_types.h and testkernel.c
# which have been loaded from files, and compile the code. 
program = cl.Program(context," ".join(program_source)).build()

# Create the OpenCL kernel object
kernel=program.testkern # NOTE: must only extract kernel attribute once (see pyopencl documentation)


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

# Note that the actual locking granularity implemented
# currently is based on the whole array, but the
# region specification is used to control flushing
# changes after write locks.

# Also note that locking some regions for read and
# some for write in a single array is currently
# unsupported and may
# to deadlocks. In this case, lock the entire
# array for both read and write. 
lockholder = spatialnde2.pylockholder()
(all_locks,readregions,writeregions) = spatialnde2.pylockprocess(manager.locker,
                                        lambda proc: [  # Remember to follow locking order
# Lock the specified region (in this case, the whole thing)
# of the "meshedparts" array
                                            # lockholder.store() automatically stores under the given field name
                                            lockholder.store((yield proc.get_locks_read_array_region(geometry.geom.contents,"meshedparts",0,spatialnde2.SNDE_INDEX_INVALID))),
                                            # lockholder.store_name() uses an alternate name
                                            lockholder.store_name("triangles",(yield proc.get_locks_read_array_region(geometry.geom.contents,"triangles",0,spatialnde2.SNDE_INDEX_INVALID))),                                            
                                            lockholder.store((yield proc.get_locks_read_array_region(geometry.geom.contents,"edges",0,spatialnde2.SNDE_INDEX_INVALID))),                                            
                                            lockholder.store((yield proc.get_locks_read_array_region(geometry.geom.contents,"vertices",0,spatialnde2.SNDE_INDEX_INVALID)))
                                            ])

# can now access lockholder.vertices, etc.
meshedparts=geometry.geom.contents.field_numpy(manager,lockholder,"meshedparts",spatialnde2.nt_snde_meshedpart)
# meshedparts is a numpy object...
# i.e. access meshedparts[0]["orientation"]["offset"], etc. 

# Buffers are OpenCL-accessible memory. You don't need to keep the buffers
# open and active; the arraymanager maintains a cache, so if you
# request a buffer a second time it will be there already. 
Buffers=spatialnde2.OpenCLBuffers(context,all_locks,readregions,writeregions)

# specify the arguments to the kernel, by argument number.
# The third parameter is the array element to be passed
# (actually comes from the OpenCL cache)
Buffers.AddBufferAsKernelArg(manager,commandqueue,kernel,0,geometry.geom.contents.field_address("meshedparts"))
Buffers.AddBufferAsKernelArg(manager,commandqueue,kernel,1,geometry.geom.contents.field_address("triangles"))
Buffers.AddBufferAsKernelArg(manager,commandqueue,kernel,2,geometry.geom.contents.field_address("edges"))
Buffers.AddBufferAsKernelArg(manager,commandqueue,kernel,3,geometry.geom.contents.field_address("vertices"))



# Enqueue the kernel 
kernel_complete = cl.enqueue_nd_range_kernel(commandqueue, kernel, (1000,),None,wait_for=Buffers.fill_events)

# a clFlush() here would start the kernel executing, but
# the kernel will alternatively start implicitly when we wait below. 

# Queue up post-processing (i.e. cache maintenance) for the kernel
# In this case we also ask it to wait for completion ("True")
# Otherwise it could return immediately with those steps merely queued
# (and we could do other stuff as it finishes in the background) 
Buffers.RemBuffers(kernel_complete,kernel_complete,True);

# very important to release OpenCL resources, 
# otherwise they may keep buffers in memory unnecessarily 
del kernel_complete
del Buffers  # we don't need the buffers any more

del meshedparts # delete all references to our array prior to unlocking, lest the memory pointed to become invalid (not necessarily a problem if it's not accessed)

# This explicitly unlocks the vertices , so that
# data should not be accessed afterward
# Each locked set (or the whole set) can be
# explicitly unlocked up to once (twice is an error).
spatialnde2.unlock_rwlock_token_set(lockholder.vertices)
spatialnde2.unlock_rwlock_token_set(lockholder.edges)
spatialnde2.unlock_rwlock_token_set(lockholder.triangles)
spatialnde2.unlock_rwlock_token_set(lockholder.meshedparts)

# In addition any locked sets not explicitly unlocked
# will be implicitly locked once all references have
# either been released or have gone out of scope. 
spatialnde2.release_rwlock_token_set(all_locks);
  
