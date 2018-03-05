
#include <cstdint>

#include "geometry_types.h"
#include "allocator.h"


extern "C" {

  // BUG... these implementations should probably catch exceptions.... 
  /*
  snde_index snde_allocator_alloc(snde::allocator *alloc,snde_index nelem)
  {
    return alloc->alloc(nelem);
  }
  
  void snde_allocator_free(snde::allocator *alloc,snde_index addr,snde_index nelem)
  {
    alloc->free(addr,nelem);
  }
  */  
};
