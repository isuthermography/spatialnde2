
#include <assert.h>
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
#include "allocator.hpp"
#include "lockmanager.hpp"
#include "arraymanager.hpp"
#include "geometrydata.h"

using namespace snde;

int main(int argc, char *argv[])
{

  std::shared_ptr<memallocator> lowlevel_alloc;
  std::shared_ptr<arraymanager> manager;
  std::shared_ptr<allocator_alignment> alignment=std::make_shared<allocator_alignment>();
  std::shared_ptr<geometry> geom;

  //snde_index blockstart,blocksize;

  lowlevel_alloc=std::make_shared<cmemallocator>();

  //fprintf(stderr,"build manager...\n");
  manager=std::make_shared<arraymanager>(lowlevel_alloc,alignment);
  //fprintf(stderr,"build geom...\n");
  geom=std::make_shared<geometry>(1e-6,manager);

  // Allocate space for 10000 vertices 


  /* When we begin a locking process we create a token_set to
     keep track of all the locks acquired during the process */
  rwlock_token_set all_locks=empty_rwlock_token_set();
  
  // perform lock of all arrays
  rwlock_token_set read_lock=geom->manager->locker->get_locks_write_all(all_locks);

  {
    rwlock_token_set vertices_write;
    std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>> arraylocks;
    snde_index vertices_block;

    std::tie(vertices_block,arraylocks)=geom->manager->alloc_arraylocked(all_locks,(void **)&geom->geom.vertices,10000);
    //assert(arraylocks.size()==1);
    assert(arraylocks[0].first->value()==(void **)&geom->geom.vertices);
    vertices_write=arraylocks[0].second;
    
    /* Now that we are done with the locking, we can release that
       record of all locks acquired. This doesn't (on its own) 
       release any of the locks */
    release_rwlock_token_set(all_locks);
    
    
    // unlock those arrays
    unlock_rwlock_token_set(read_lock);
  }
  
  // lock single array. This begins a new locking process, so
  // we need a new, empty all_locks 
  all_locks=empty_rwlock_token_set();
  
  rwlock_token_set triangle_lock=geom->manager->locker->get_locks_read_array(all_locks,(void **)&geom->geom.triangles);
  
  // lock succeeding array, following locking order and acknowledging current lock ownership
  rwlock_token_set vertices_lock=geom->manager->locker->get_locks_read_array(all_locks,(void **)&geom->geom.vertices);

  // clear lock tracker
  release_rwlock_token_set(all_locks);
  
  
  // not legitimate to lock all arrays right now because this would violate locking order
  //fprintf(stderr,"release locks...\n");

  unlock_rwlock_token_set(vertices_lock);  // order of unlocks doesn't matter
  unlock_rwlock_token_set(triangle_lock);  // unlocks also happen automatically when the token_set leaves context. 
  return 0;
}
