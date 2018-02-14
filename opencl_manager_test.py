import sys
import numpy as np
import spatialnde2

lowlevel_alloc=spatialnde2.cmemallocator();
manager=spatialnde2.arraymanager(lowlevel_alloc)

geometry=spatialnde2.geometry(1e-6,manager)

(context,device,clmsgs)=spatialnde2.get_opencl_context("::",False,None,None);

sys.stderr.write(clmsgs)
(commandqueue,retval)=spatialnde2.clCreateCommandQueue(context,device,spatialnde2.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
if retval==spatialnde2.CL_INVALID_QUEUE_PROPERTIES:
    (commandqueue,retval)=spatialnde2.clCreateCommandQueue(context,device,0)
    pass

geometry_types_h=file("geometry_types.h").read()
testkernel_c = file("testkernel.c").read()

program_source=spatialnde2.StringVector([ geometry_types_h, testkernel_c ])

(program,build_log) = spatialnde2.get_opencl_program(context,device,program_source)

(kernel,clerror)=spatialnde2.clCreateKernel(program,"testkern")


lockholder = spatialnde2.pylockholder()
(all_locks,readregions,writeregions) = spatialnde2.pylockprocess(manager.locker,
                                        lambda proc: lockholder.store("vertices",(yield proc.get_locks_read_array_region(geometry.geom.contents.field_address("vertices"),0,spatialnde2.SNDE_INDEX_INVALID))))

# can now access lockholder.vertices, etc.

Buffers=spatialnde2.OpenCLBuffers(context,all_locks,readregions,writeregions)

Buffers.AddBufferAsKernelArg(manager,commandqueue,kernel,0,geometry.geom.contents.field_address("vertices"),geometry.geom.contents.field_address("vertices"))


global_work_offset=np.array((0,),dtype=np.uint64)
global_work_size=np.array((1000,),dtype=np.uint64)
local_work_size=np.array((1,),dtype=np.uint64)


(out,kernel_complete)=spatialnde2.clEnqueueNDRangeKernelArrays(commandqueue,kernel,global_work_offset,global_work_size,local_work_size,Buffers.FillEvents_untracked(),Buffers.NumFillEvents());


if kernel_complete is None:
    #kc_vector=spatialnde2.opencl_event_vector()
    #Buffers.RemBuffers(None,kc_vector,True);
    Buffers.RemBuffers(None,True);
    
    pass
else: 
    Buffers.RemBuffers(kernel_complete,kernel_complete,True);
    pass

spatialnde2.unlock_rwlock_token_set(lockholder.vertices)

spatialnde2.release_rwlock_token_set(all_locks);
  
