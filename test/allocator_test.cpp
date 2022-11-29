#ifndef _WIN32
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#else
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <Lmcons.h>
#endif

#include <assert.h>
#include <string.h>
#include <cstdint>

#include <vector>
#include <map>
#include <condition_variable>
#include <deque>
#include <algorithm>
#include <unordered_map>
#include <functional>

#include "geometry_types.h"
#include "memallocator.hpp"
#include "allocator.hpp"

#ifndef _WIN32
#include "shared_memory_allocator_posix.hpp"
#else
#include "shared_memory_allocator_win32.hpp"
#endif // !_WIN32

using namespace snde;

int main(int argc, char *argv[])
{

  float *test_array;

  snde_index blockstart,blocksize,newblocksize,newblockstart;

  //std::shared_ptr<memallocator> lowlevel_alloc=std::make_shared<cmemallocator>();
#ifndef _WIN32
  std::shared_ptr<memallocator> lowlevel_alloc = std::make_shared<shared_memory_allocator_posix>();
#else
  std::shared_ptr<memallocator> lowlevel_alloc = std::make_shared<shared_memory_allocator_win32>();
#endif // !_WIN32
  std::shared_ptr<allocator_alignment> alignment=std::make_shared<allocator_alignment>();

  // Allocate 10,000 size data storage
  std::shared_ptr<allocator> test_allocator=std::make_shared<allocator>(lowlevel_alloc,nullptr,"/test_allocator",0,0,0,alignment,(void **)&test_array,sizeof(*test_array),70000,std::set<snde_index>(),0);


  // allocate 7739 element array
  blocksize=7739;
  blockstart=test_allocator->_alloc(blocksize);
  if (blockstart==SNDE_INDEX_INVALID) {
    fprintf(stderr,"Allocation failed\n");
    exit(1);
  }


  // allocate a new 20,000 element array to trigger realloc
  newblocksize = 80000;
  newblockstart = test_allocator->_alloc(newblocksize);
  if (blockstart == SNDE_INDEX_INVALID) {
      fprintf(stderr, "Second Allocation failed\n");
      exit(1);
  }


  test_array[49] = 12345.0;

  // get a reference using obtain_nonmoving_copy_or_reference
  void* curarrayptr = *(*test_allocator->arrays()).at(0).arrayptr;
  std::shared_ptr<nonmoving_copy_or_reference> newarrayptr = lowlevel_alloc->obtain_nonmoving_copy_or_reference("/test_allocator", 0, 0, 0, &curarrayptr, curarrayptr, 0, 50);

  // verify non shifted reference using the shiftedptr 
  float* shiftedptr = (float*)newarrayptr->get_shiftedptr();
  
  if (shiftedptr[49] != test_array[49])
  {
      fprintf(stderr, "Shifted Ptr Is Wrong... (%f != %f)\n", shiftedptr[49], test_array[49]);
      exit(1);
  }

  // Get the page size
  unsigned long page_size;

#ifndef _WIN32
  page_size = sysconf(_SC_PAGE_SIZE);
#else
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  page_size = (unsigned long)sysinfo.dwAllocationGranularity;
#endif // !_WIN32

  int shift = page_size;
  int length = 20;

  // Compute where the shift should occur and set a value to check and make sure it's right
  size_t shiftpages = shift / page_size;
  size_t ptrshift = shift - shiftpages * page_size;
  size_t mmaplength = length + ptrshift;
  size_t offset = shiftpages * page_size;
  test_array[offset-1] = 23456.0;
  //fprintf(stderr, "offset is %llu\n", offset);

  // Get a shifted reference
  newarrayptr = lowlevel_alloc->obtain_nonmoving_copy_or_reference("/test_allocator", 0, 0, 0, &curarrayptr, curarrayptr, shift, length);

  // verify non shifted reference using the shiftedptr 
  shiftedptr = (float*)newarrayptr->get_shiftedptr();

  if (shiftedptr[0] != test_array[offset])
  {
      fprintf(stderr, "Shifted Ptr Is Wrong... (%f != %f)\n", shiftedptr[0], test_array[offset]);
      exit(1);
  }

  
  test_allocator->_free(blockstart,blocksize);
  test_allocator->_free(newblockstart, newblocksize);

  test_allocator.reset(); // discard allocator

  lowlevel_alloc.reset(); // discard lowlevel allocator
  return 0;
}
