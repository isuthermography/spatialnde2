
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

using namespace snde;

int main(int argc, char *argv[])
{

  ;
  
  float *test_array;

  snde_index blockstart,blocksize;

  std::shared_ptr<memallocator> lowlevel_alloc(new cmemallocator());
  
  std::shared_ptr<allocator> test_allocator(new allocator(lowlevel_alloc,NULL,(void **)&test_array,sizeof(*test_array),100000));

  // allocate 7739 element array

  blocksize=7739;
  blockstart=test_allocator->alloc(blocksize);
  if (blockstart==SNDE_INDEX_INVALID) {
    fprintf(stderr,"Allocation failed\n");
    exit(1);
  }
  
  
  test_allocator->free(blockstart,blocksize);

  test_allocator.reset(); // discard allocator

  lowlevel_alloc.reset(); // discard lowlevel allocator
  return 0;
}
