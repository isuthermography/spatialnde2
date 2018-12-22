import sys
import copy
import numpy as np
import spatialnde2


# lowlevel_alloc performs the actual host-side memory allocations
lowlevel_alloc=spatialnde2.cmemallocator();

# alignment requirements specify constraints on allocation
# block sizes
alignment_requirements=spatialnde2.allocator_alignment()

manager=spatialnde2.arraymanager(lowlevel_alloc,alignment_requirements)

geometry=spatialnde2.geometry(1e-6,manager)

meshedparts = spatialnde2.x3d_load_geometry(geometry,sys.argv[1],None,False)


lockholder = spatialnde2.lockholder()
all_locks = spatialnde2.pylockprocess(manager,
                                      lambda proc: [
                                          # remember to follow locking order!
                                          lockholder.store((yield proc.get_locks_read_array_region(geometry.addr("meshedparts"),meshedparts[0].idx,1))),
                                          #lockholder.store((yield proc.get_locks_read_array_region(geometry.addr("vertices"),0,spatialnde2.SNDE_INDEX_INVALID)))
                                          
                                      ])

# This next line extracts the raw data
first_meshedpart_raw=geometry.field(lockholder,"meshedparts",False,spatialnde2.nt_snde_meshedpart,meshedparts[0].idx,numelem=1)

# make a copy
first_meshedpart_raw_copy=copy.copy(first_meshedpart_raw)

# release that lock
spatialnde2.unlock_rwlock_token_set(all_locks)

del lockholder

# Now use the automatic locking to look at this meshedpart 
holder = spatialnde2.lockholder()   # new holder
all_locks = spatialnde2.pylockprocess(manager,
                                      lambda proc: meshedparts[0].obtain_lock_pycpp(proc,holder,spatialnde2.SNDE_COMPONENT_GEOM_ALL,0))  # 0 indicates flags for which arrays we want write locks (none in this case)

# now we can access (read only) the data from the cpu
first_meshedpart_raw2=geometry.field(holder,"meshedparts",False,spatialnde2.nt_snde_meshedpart,meshedparts[0].idx,1)

# ... and all of the subcomponents are locked for read by obtain_lock_pycpp() too!


#del holder
#spatialnde2.unlock_rwlock_token_set(all_locks)
