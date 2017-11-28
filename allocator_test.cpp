
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

  memallocator *lowlevel_alloc;
  
  allocator<float> *test_allocator;
  float *test_array;

  snde_index blockstart,blocksize;

  lowlevel_alloc=new cmemallocator();
  
  test_allocator=new allocator<float>(lowlevel_alloc,NULL,&test_array,100000);

  // allocate 7739 element array

  blocksize=7739;
  blockstart=test_allocator->alloc(blocksize);
  if (blockstart==SNDE_INDEX_INVALID) {
    fprintf(stderr,"Allocation failed\n");
    exit(1);
  }
  
  
  test_allocator->free(blockstart,blocksize);

  delete test_allocator;

  delete lowlevel_alloc;
  return 1;
}
