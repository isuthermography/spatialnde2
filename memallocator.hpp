#ifndef SNDE_MEMALLOCATOR
#define SNDE_MEMALLOCATOR

/* This is an abstraction layer that will permit 
   substitution of a specialized memory allocator...
   i.e. one that allocates space directly in OpenCL 
   memory on the GPU card */

#include <cstdlib>


namespace snde {
  class memallocator {
  public:

    virtual void *malloc(std::size_t nbytes)=0;
    virtual void *calloc(std::size_t nbytes)=0;
    virtual void *realloc(void *ptr,std::size_t newsize)=0;
    virtual void free(void *ptr)=0;
  
  };

  class cmemallocator: public memallocator {
    void *malloc(std::size_t nbytes) {
      return std::malloc(nbytes);
    }
  
    void *calloc(std::size_t nbytes) {
      return std::calloc(nbytes,1);
    }

    void *realloc(void *ptr,std::size_t newsize) {
      return std::realloc(ptr,newsize);
    }

    void free(void *ptr) {
      return std::free(ptr);
    }
  };


}

#endif /* SNDE_MEMALLOCATOR */
