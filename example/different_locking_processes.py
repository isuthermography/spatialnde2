import sys
import ctypes
import numpy as np
import spatialnde2


lowlevel_alloc=spatialnde2.cmemallocator();
alignment_requirements=spatialnde2.allocator_alignment()
manager=spatialnde2.arraymanager(lowlevel_alloc,alignment_requirements)

geometry=spatialnde2.geometry(1e-6,manager)


# Locking process Method 1 (Pure-python; cannot call C++ locking functions)
# This method uses Python generators under the hood

# All locking steps must be done within a lambda or function passed as a parameter
lockholder = spatialnde2.lockholder()
all_locks = spatialnde2.pylockprocess(manager,
                                      lambda proc: [
                                          # remember to follow locking order!
                                          lockholder.store((yield proc.get_locks_read_array_region(geometry.addr("edges"),0,spatialnde2.SNDE_INDEX_INVALID))),
                                          lockholder.store((yield proc.get_locks_read_array_region(geometry.addr("vertices"),0,spatialnde2.SNDE_INDEX_INVALID)))
                                      ])

# Note: Each lock acquired can be "unlocked" exactly once
# Any that are not explicitly unlocked will
# automatically be unlocked when all referencing variables
# either go out of scope or are "released"

spatialnde2.unlock_rwlock_token_set(lockholder.get(geometry.addr("vertices"),False,0,spatialnde2.SNDE_INDEX_INVALID))
spatialnde2.unlock_rwlock_token_set(lockholder.get(geometry.addr("edges"),False,0,spatialnde2.SNDE_INDEX_INVALID))
spatialnde2.release_rwlock_token_set(all_locks);


# Variation on locking process Method 1 where a sub "process"
# is spawned that can perform locking in parallel. Because it is
# in parallel with the parent process it is OK that it requests a later
# lock in the locking order than the subsequent parent request.

# (it's actually not really a subprocess, but the parallelism is
# obtained through multiple Python generators (see yield statement)
lockholder = spatialnde2.lockholder()
all_locks = spatialnde2.pylockprocess(manager,
                                      lambda proc: [
                                          (yield proc.spawn(lambda proc: [ lockholder.store((yield proc.get_locks_read_array_region(geometry.addr("vertices"),0,spatialnde2.SNDE_INDEX_INVALID))) ])),
                                          lockholder.store((yield proc.get_locks_read_array_region(geometry.addr("edges"),0,spatialnde2.SNDE_INDEX_INVALID))),
                                      ])

spatialnde2.unlock_rwlock_token_set(lockholder.get(geometry.addr("vertices"),False,0,spatialnde2.SNDE_INDEX_INVALID))
spatialnde2.unlock_rwlock_token_set(lockholder.get(geometry.addr("edges"),False,0,spatialnde2.SNDE_INDEX_INVALID))
spatialnde2.release_rwlock_token_set(all_locks);


# Locking process method #2: Using the threaded C++ interface
# This process is implemented in C++ under the hood, using multiple
# threads. So a spawn() does indeed create a new thread.
#
# Because it is implemented in C++, it can be used with locking methods
# of C++ geometry objects

# With this method you don't have to put everything into a separate function. 

lockholder = spatialnde2.lockholder()
lockprocess = spatialnde2.lockingprocess_threaded_python(manager)

# API is very similar to C++
# Here we are getting first a write lock, then getting a read lock on the
# same region (locking order is defined as write first, then read)

# ***!!!! NOTE: This is technically a locking order violation because get_locks_write_array_region
# has multiple steps, but it is OK because all we are doing is downgrading
lockholder.vertices_write = lockprocess.get_locks_write_array_region(geometry.geom.contents.addr("vertices"),0,spatialnde2.SNDE_INDEX_INVALID);
lockholder.vertices_read = lockprocess.get_locks_read_array_region(geometry.geom.contents.addr("vertices"),0,spatialnde2.SNDE_INDEX_INVALID);

all_locks = lockprocess.finish()


# We can unlock both items by calling unlock_rwlock_token_set() on all_locks. 
spatialnde2.unlock_rwlock_token_set(all_locks);



# Locking process method #2, example with  spawn()
lockholder = spatialnde2.lockholder()
lockprocess = spatialnde2.lockingprocess_threaded_python(manager)

# lockprocess supports spawning to python lambdas/functions 
# Because the lambda doesn't support assignment, we use lockholder.store_name
# to add the lock into our lockholder.

# NOTE: must use lockholder.store_name() not lockholder.store() because C++ api doesn't pass the name back

lockprocess.spawn(lambda proc: [
    lockholder.store_name("vertices_write",lockprocess.get_locks_write_array_region(geometry.addr("vertices"),0,spatialnde2.SNDE_INDEX_INVALID)),
]) 

lockholder.vertices_read = lockprocess.get_locks_read_array_region(geometry.addr("vertices"),0,spatialnde2.SNDE_INDEX_INVALID);

all_locks = lockprocess.finish()

spatialnde2.unlock_rwlock_token_set(all_locks);


