###!!!! Note: Due to a bug in the pOCL/ICD/etc. stack on fedora 26
### This crashes unless we export LD_PRELOAD=/usr/lib64/libpocl.so.2.0.0

import sys
import ctypes
import numpy as np
import spatialnde2
import pyopencl as cl


(context,device,clmsgs)=spatialnde2.get_opencl_context("::",True,None,None);

sys.stderr.write(clmsgs)


lowlevel_alloc=spatialnde2.cmemallocator();

alignment_requirements=spatialnde2.allocator_alignment()
spatialnde2.add_opencl_alignment_requirement(alignment_requirements,device);


manager=spatialnde2.arraymanager(lowlevel_alloc,alignment_requirements)

geom=spatialnde2.geometry(1e-6,manager)

# Build a second, custom geometry structure, managed by the same manager
class customgeometry(spatialnde2.build_geometrystruct_class(None)):
    _fields_=[
        ("areas",ctypes.c_void_p ),
    ]
    manager=None
    def __init__(self,manager,parentgeom):
        super(customgeometry,self).__init__()
        self.manager=manager

        # We could do allocated arrays, but here we will actually want the areas
        # to be a follower array of the regular geometry's triangles
        manager.add_follower_array(parentgeom.addr("triangles"),self.addr("areas"),spatialnde2.nt_snde_coord.itemsize)
        pass

    def __del__(self): # destructor detaches the allocated arrays... otherwise
        # the memory from this geometry object can be overwritten
        # after the object is destroyed if the manager still exists
        # (This is required for custom geometry classes!)
        
        self.manager.cleararrays(self.addr(self._fields_[0][0]),ctypes.sizeof(self))
        pass
    pass

geom2=customgeometry(manager,geom)

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


# Allocate arrays
holder = spatialnde2.lockholder()
all_locks = spatialnde2.pylockprocess(manager.locker,
                                      lambda proc: [  # Remember to follow locking order
                                          holder.store_alloc((yield proc.alloc_array_region(manager,geom.addr("meshedparts"),1,""))),
                                          # We have to spawn the request to lock "triangles", so that it can execute in
                                          # parallel because it now includes the request to lock "areas"
                                          # which is at the end of the locking order
                                          (yield proc.spawn(lambda proc: holder.store_alloc((yield proc.alloc_array_region(manager,geom.addr("triangles"),1,""))))),                                            
                                          holder.store_alloc((yield proc.alloc_array_region(manager,geom.addr("edges"),3,""))),                                            
                                          holder.store_alloc((yield proc.alloc_array_region(manager,geom.addr("vertices"),3,""))),
                                          holder.store_alloc((yield proc.alloc_array_region(manager,geom.addr("vertex_edgelist"),6,""))),
                                      ])

# can now access holder.vertices, etc.
# Build a meshed part with exactly one triangle (three edges, and three vertices) 
meshedparts=geom.allocfield(holder,"meshedparts",spatialnde2.nt_snde_meshedpart,"",1)
#meshedparts[0]["orientation"]["offset"]=np.zeros(3)
#meshedparts[0]["orientation"]["quat"]=(0,0,0,1) # identity quaternion

meshedparts[0]["firsttri"] = holder.get_alloc(geom.addr("triangles"),"")
meshedparts[0]["numtris"] = 1

meshedparts[0]["firstedge"] = holder.get_alloc(geom.addr("edges"),"")
meshedparts[0]["numedges"] = 3

meshedparts[0]["firstvertex"] = holder.get_alloc(geom.addr("vertices"),"")
meshedparts[0]["numvertices"] = 3

meshedparts[0]["first_vertex_edgelist"] = holder.get_alloc(geom.addr("vertex_edgelist_indices"),"")
meshedparts[0]["num_vertex_edgelist"] = 9
meshedparts[0]["firstbox"] = spatialnde2.SNDE_INDEX_INVALID
meshedparts[0]["numboxes"] = spatialnde2.SNDE_INDEX_INVALID
meshedparts[0]["firstboxpoly"] = spatialnde2.SNDE_INDEX_INVALID
meshedparts[0]["numboxpolys"] = spatialnde2.SNDE_INDEX_INVALID
meshedparts[0]["solid"] = False

triangles=geom.allocfield(holder,"triangles",spatialnde2.nt_snde_triangle,"",1)
triangles[0][0]=0
triangles[0][1]=1
triangles[0][2]=2

edges=geom.allocfield(holder,"edges",spatialnde2.nt_snde_edge,"",3)
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



vertices=geom.allocfield(holder,"vertices",spatialnde2.nt_snde_coord3,"",3)
vertices[0]=(1,0,0)
vertices[1]=(0,1,0)
vertices[2]=(0,0,0)

vertex_edgelist_indices=geom.allocfield(holder,"vertex_edgelist_indices",spatialnde2.nt_snde_vertex_edgelist_index,"",3)
vertex_edgelist_indices[0]=(0,2)
vertex_edgelist_indices[1]=(2,2)
vertex_edgelist_indices[2]=(4,2)
vertex_edgelist=geom.allocfield(holder,"vertex_edgelist",spatialnde2.nt_snde_index,"",6)
vertex_edgelist[0]=0
vertex_edgelist[1]=2
vertex_edgelist[2]=0
vertex_edgelist[3]=1
vertex_edgelist[4]=1
vertex_edgelist[5]=2

# Mark that we have made changes with the CPU
manager.dirty_alloc(holder,geom.addr("meshedparts"),"",1)
manager.dirty_alloc(holder,geom.addr("triangles"),"",1)
manager.dirty_alloc(holder,geom.addr("edges"),"",3)
manager.dirty_alloc(holder,geom.addr("vertices"),"",3)

# Store which part address for later use
# Represent the above triangle as a "meshedpart"
part=spatialnde2.meshedpart(geom,"my_part",holder.get_alloc(geom.addr("meshedparts"),""))


# This time we don't unlock. Since we've marked stuff as dirty when we try to access with
# the GPU it will know to update any cache.

# A set of buffers for the GPU
Buffers=spatialnde2.OpenCLBuffers(context,all_locks)

# specify the arguments to the kernel, by argument number.
# The third parameter is the array element to be passed
# (actually comes from the OpenCL cache)
Buffers.AddSubBufferAsKernelArg(manager,queue,kernel,0,geom.addr("meshedparts"),part.idx,1,False)
Buffers.AddSubBufferAsKernelArg(manager,queue,kernel,1,geom.addr("triangles"),meshedparts[0]["firsttri"],meshedparts[0]["numtris"],False)
Buffers.AddSubBufferAsKernelArg(manager,queue,kernel,2,geom.addr("edges"),meshedparts[0]["firstedge"],meshedparts[0]["numedges"],False)
Buffers.AddSubBufferAsKernelArg(manager,queue,kernel,3,geom.addr("vertices"),meshedparts[0]["firstvertex"],meshedparts[0]["numvertices"],False)
# Add our separate "areas" array...
# Since it is allocated with the triangles,
# "firsttri" and "numtries" define the array bounds for this part
Buffers.AddSubBufferAsKernelArg(manager,queue,kernel,4,geom2.addr("areas"),meshedparts[0]["firsttri"],meshedparts[0]["numtris"],True)  # This one is marked as write

worksize = meshedparts[0]["numtris"]

# print(Buffers.fill_events)

# Enqueue the kernel 
kernel_complete = cl.enqueue_nd_range_kernel(queue, kernel, (worksize,),None,wait_for=Buffers.fill_events)

# Mark that the kernel has affected the "areas" sub-buffer. This
# means that the "RemBuffers" operation will make sure those changes
# are propagated to main memory
Buffers.SubBufferDirty(geom2.addr("areas"),meshedparts[0]["firsttri"],meshedparts[0]["numtris"])

# a clFlush() here would start the kernel executing, but
# the kernel will alternatively start implicitly when we wait below. 

# Queue up post-processing.
# The wait="True" means that all modified buffers managed by
# the Buffers object will be fully updated by the time
# RemBuffers() returns.
Buffers.RemBuffers(kernel_complete,kernel_complete,True);
areas=geom2.allocfield(holder,"areas",spatialnde2.nt_snde_coord,"",1)

print("Triangle area=%f\n" % areas[0])

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


