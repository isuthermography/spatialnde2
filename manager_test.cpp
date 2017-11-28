
#include <assert.h>
#include <string.h>
#include <cstdint>

#include <vector>
#include <map>
#include <condition_variable>
#include <deque>
#include <algorithm>

#include "geometry_types.h"
#include "memallocator.hpp"
#include "allocator.hpp"
#include "lockmanager.hpp"
#include "arraymanager.hpp"
#include "geometry.h"

using namespace snde;

int main(int argc, char *argv[])
{

  std::shared_ptr<memallocator> lowlevel_alloc;
  std::shared_ptr<arraymanager> manager;
  std::shared_ptr<geometry> geom;

  snde_index blockstart,blocksize;

  lowlevel_alloc=std::make_shared<cmemallocator>();

  //fprintf(stderr,"build manager...\n");
  manager=std::make_shared<simplearraymanager>(lowlevel_alloc);
  //fprintf(stderr,"build geom...\n");
  geom=std::make_shared<geometry>(1e-6,manager);

  // Allocate space for 10000 vertices 
  geom->manager->alloc((void **)&geom->geom.vertices,10000);
  
  // perform lock of all arrays
  rwlock_token_set read_lock=geom->manager->locker->get_locks_read_all();

  // unlock those arrays
  read_lock.reset();

  // lock single array
  rwlock_token_set triangle_lock=geom->manager->locker->get_locks_read_array((void **)&geom->geom.vertexidx);
  
  // lock preceding array, following locking order and acknowledging current lock ownership
  rwlock_token_set vertices_lock=geom->manager->locker->get_locks_read_array(triangle_lock,(void **)&geom->geom.vertices);

  // not legitimate to lock all arrays right now because this would violate locking order
  //fprintf(stderr,"release locks...\n");

  vertices_lock.reset();  // order of unlocks doesn't matter
  triangle_lock.reset();  // unlocks also happen automatically when the token_set leaves context. 
  return 0;
}
