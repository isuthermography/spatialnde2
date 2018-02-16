import sys
import ctypes
import numpy as np
import spatialnde2

class customgeometry(spatialnde2.snde_geometrystruct):
    _fields_=[ ("vertexidx",ctypes.c_void_p ),
               ("vertices",ctypes.c_void_p ) ]
    manager=None
    def __init__(self,manager):
        super(customgeometry,self).__init__()
        self.manager=manager

        manager.add_allocated_array(self.field_address("vertexidx"),spatialnde2.nt_snde_index.itemsize,0)
        manager.add_allocated_array(self.field_address("vertices"),spatialnde2.nt_snde_coord3.itemsize,0)
        pass

    def __del__(self): # destructor detaches the allocated arrays... otherwise
        # the memory from this geometry object can be overwritten
        # after the object is destroyed if the manager still exists
        # (This is required for custom geometry classes!)
        
        self.manager.cleararrays(self.field_address(self._fields_[0][0]),ctypes.sizeof(self))
        pass
    pass


lowlevel_alloc=spatialnde2.cmemallocator();
manager=spatialnde2.arraymanager(lowlevel_alloc)

geometry=customgeometry(manager)


# Pure-python locking process
lockholder = spatialnde2.pylockholder()
(all_locks,readregions,writeregions) = spatialnde2.pylockprocess(manager.locker,
                                        lambda proc: [
                                            # remember to follow locking order!
                                            lockholder.store((yield proc.get_locks_read_array_region(geometry,"vertexidx",0,spatialnde2.SNDE_INDEX_INVALID))),
                                            lockholder.store((yield proc.get_locks_read_array_region(geometry,"vertices",0,spatialnde2.SNDE_INDEX_INVALID)))
                                        ])

# can now access lockholder.vertices, etc.
vertexidx=geometry.field_numpy(manager,lockholder,"vertexidx",spatialnde2.nt_snde_index)
vertices=geometry.field_numpy(manager,lockholder,"vertices",spatialnde2.nt_snde_coord3)
# (These are numpy arrays) 

del vertices # release numpy object prior to unlocking
del vertexidx # release numpy object prior to unlocking

spatialnde2.unlock_rwlock_token_set(lockholder.vertices)
spatialnde2.unlock_rwlock_token_set(lockholder.vertexidx)

spatialnde2.release_rwlock_token_set(all_locks);



