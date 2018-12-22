import sys
import ctypes
import numpy as np
import spatialnde2

class customgeometry(spatialnde2.build_geometrystruct_class(None)):
    _fields_=[ ("vertexidx",ctypes.c_void_p ),
               ("vertices",ctypes.c_void_p ) ]
    manager=None
    def __init__(self,manager):
        super(customgeometry,self).__init__()
        self.manager=manager

        manager.add_allocated_array(self.addr("vertexidx"),spatialnde2.nt_snde_index.itemsize,0)
        manager.add_allocated_array(self.addr("vertices"),spatialnde2.nt_snde_coord3.itemsize,0)
        pass

    def __del__(self): # destructor detaches the allocated arrays... otherwise
        # the memory from this geometry object can be overwritten
        # after the object is destroyed if the manager still exists
        # (This is required for custom geometry classes!)
        
        self.manager.cleararrays(self.addr(self._fields_[0][0]),ctypes.sizeof(self))
        pass
    pass


lowlevel_alloc=spatialnde2.cmemallocator();
alignment_requirements=spatialnde2.allocator_alignment()

manager=spatialnde2.arraymanager(lowlevel_alloc,alignment_requirements)


geometry=customgeometry(manager)


# Pure-python locking process

lockholder = spatialnde2.lockholder()
all_locks = spatialnde2.pylockprocess(manager.locker,
                                      lambda proc: [
                                          # remember to follow locking order!
                                          lockholder.store_alloc((yield proc.alloc_array_region(manager,geometry.addr("vertexidx"),10,""))),
                                          lockholder.store_alloc((yield proc.alloc_array_region(manager,geometry.addr("vertices"),15,"")))
                                      ])

# can now access lockholder.vertices, etc.
vertexidx=geometry.allocfield(lockholder,"vertexidx",spatialnde2.nt_snde_index,"",10)
vertices=geometry.allocfield(lockholder,"vertices",spatialnde2.nt_snde_coord3,"",15)
# (These are numpy arrays) 

del vertices # release numpy object prior to unlocking
del vertexidx # release numpy object prior to unlocking

spatialnde2.unlock_rwlock_token_set(lockholder.get_alloc_lock(geometry.addr("vertices"),15,""))
spatialnde2.unlock_rwlock_token_set(lockholder.get_alloc_lock(geometry.addr("vertexidx"),10,""))

spatialnde2.release_rwlock_token_set(all_locks);


lockholder = spatialnde2.lockholder()
all_locks = spatialnde2.pylockprocess(manager,
                                      lambda proc: [
                                          # remember to follow locking order!
                                          lockholder.store((yield proc.get_locks_read_array_region(geometry.addr("vertexidx"),0,spatialnde2.SNDE_INDEX_INVALID))),
                                          lockholder.store((yield proc.get_locks_read_array_region(geometry.addr("vertices"),0,spatialnde2.SNDE_INDEX_INVALID)))
                                      ])

# can now access lockholder.vertices, etc.
vertexidx=geometry.field(lockholder,"vertexidx",False,spatialnde2.nt_snde_index,0)
vertices=geometry.field(lockholder,"vertices",False,spatialnde2.nt_snde_coord3,0)
# (These are numpy arrays) 

del vertices # release numpy object prior to unlocking
del vertexidx # release numpy object prior to unlocking

spatialnde2.unlock_rwlock_token_set(lockholder.get(geometry.addr("vertices"),False,0,spatialnde2.SNDE_INDEX_INVALID))
spatialnde2.unlock_rwlock_token_set(lockholder.get(geometry.addr("vertexidx"),False,0,spatialnde2.SNDE_INDEX_INVALID))

spatialnde2.release_rwlock_token_set(all_locks);



