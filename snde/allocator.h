#ifndef SNDE_ALLOCATOR_H
#define SNDE_ALLOCATOR_H

#ifdef __cplusplus
#include "snde/allocator.hpp"

#else
  
struct allocator;
typedef struct allocator allocator;

#endif

#ifdef __cplusplus
extern "C" {
#endif

  /*
  snde_index snde_allocator_alloc(snde::allocator *alloc,snde_index nelem);
  void snde_allocator_free(snde::allocator *alloc,snde_index addr,snde_index nelem);
  */
  
#ifdef __cplusplus
}
#endif

#endif /* SNDE_ALLOCATOR_H */
