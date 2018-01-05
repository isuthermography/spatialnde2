
#include <assert.h>
#include <string.h>
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
#include "rangetracker.hpp"
#include "allocator.hpp"
#include "lockmanager.hpp"
#include "openclarraymanager.hpp"
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
  manager=std::make_shared<arraymanager>(lowlevel_alloc);
  //fprintf(stderr,"build geom...\n");
  geom=std::make_shared<geometry>(1e-6,manager);


  /*
  // Allocate space for 10000 vertices 
  geom->manager->alloc((void **)&geom->geom.vertices,10000);
  
  // perform lock of all arrays
  rwlock_token_set all_locks=empty_rwlock_token_set();
  
  rwlock_token_set read_lock=geom->manager->locker->get_locks_read_all(all_locks);

  // unlock those arrays
  read_lock->clear();
  all_locks->clear();

  // lock single array
  rwlock_token_set vertices_lock=geom->manager->locker->get_locks_read_array(all_locks,(void **)&geom->geom.vertices);
  
  // lock following array, following locking order and acknowledging current lock ownership
  rwlock_token_set triangle_lock=geom->manager->locker->get_locks_read_array(all_locks,(void **)&geom->geom.vertexidx);

  // not legitimate to lock all arrays right now because this would violate locking order
  //fprintf(stderr,"release locks...\n");

  all_locks->clear();
  vertices_lock->clear();  // order of unlocks doesn't matter
  triangle_lock->clear();  // unlocks also happen automatically when the token_set leaves context. 
  */

  rwlock_token_set all_locks;
  std::shared_ptr<std::vector<rangetracker<markedregion>>> readregions;
  std::shared_ptr<std::vector<rangetracker<markedregion>>> writeregions;
  
  lockprocess=lockingprocess(arraymanager->locker);

  
  std::tie(all_locks,readregions,writeregions) = lockprocess.finish();


  OpenCLBuffers Buffers(context,all_locks,readregions,writeregions);

  Buffers.AddBufferAsKernelArg(manager,queue,kernel,0,&geom->geom.meshedparts,&geom->geom.meshedparts);
  Buffers.AddBufferAsKernelArg(manager,queue,kernel,1,&geom->geom.vertices,&geom->geom.vertices);
  Buffers.AddBufferAsKernelArg(manager,queue,kernel,2,&geom->geom.vertices,&geom->geom.principal_curvatures);
  Buffers.AddBufferAsKernelArg(manager,queue,kernel,3,&geom->geom.vertices,&geom->geom.curvature_tangent_axes);

  Buffers.AddBufferAsKernelArg(manager,queue,kernel,4,&geom->geom.vertexidx,&geom->geom.vertexidx);
  Buffers.AddBufferAsKernelArg(manager,queue,kernel,5,&geom->geom.boxes,&geom->geom.boxes);
  Buffers.AddBufferAsKernelArg(manager,queue,kernel,6,&geom->geom.boxes,&geom->geom.boxcoord);
  
  Buffers.AddBufferAsKernelArg(manager,queue,kernel,7,&geom->geom.boxpolys,&geom->geom.boxpolys);
  
  
  return 0;
}
