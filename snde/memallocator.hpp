#ifndef SNDE_MEMALLOCATOR
#define SNDE_MEMALLOCATOR

/* This is an abstraction layer that will permit 
   substitution of a specialized memory allocator...
   i.e. one that allocates space directly in OpenCL 
   memory on the GPU card */

#include <memory>
#include <cstdlib>
#include <cstdint>
#include <cstring>


namespace snde {
  typedef size_t memallocator_regionid;

  //  inline memallocator_regionid ma_regionid()
  //{
  //  return (uint64_t)ptr;
  //}

  class nonmoving_copy_or_reference {
    // This class is an abstract interface that manages a pointer to
    // what can either be a copy or nonmoving reference to an allocation.
    // When this object is destroyed, the copy or reference is no longer valid.
    // member variables are immutable once published; get_ptr is thread safe
  public:
    size_t offset;
    size_t length; // bytes

    nonmoving_copy_or_reference(size_t offset,size_t length) :
      offset(offset),
      length(length)
    {

    }
    // Rule of 3
    nonmoving_copy_or_reference(const nonmoving_copy_or_reference &) = delete;
    nonmoving_copy_or_reference& operator=(const nonmoving_copy_or_reference &) = delete; 
    virtual ~nonmoving_copy_or_reference()=default;  // virtual destructor required so we can be subclassed
    
    virtual void *get_baseptr()=0;
    virtual void **get_basearray()=0;
  };

  class nonmoving_copy_or_reference_cmem: public nonmoving_copy_or_reference {
  public:
    void **basearray; // pointer to the maintained pointer for the data
    void *base_ptr; // will never move
    nonmoving_copy_or_reference_cmem(size_t offset,size_t length,void *orig_ptr) :
      nonmoving_copy_or_reference(offset,length),
      basearray(&base_ptr),
      base_ptr(malloc(length))
    {
      memcpy(base_ptr,((char *)orig_ptr)+offset,length);
    }
    // rule of 3
    nonmoving_copy_or_reference_cmem(const nonmoving_copy_or_reference_cmem &) = delete;
    nonmoving_copy_or_reference_cmem& operator=(const nonmoving_copy_or_reference_cmem &) = delete; 
    virtual ~nonmoving_copy_or_reference_cmem()  // virtual destructor required so we can be subclassed
    {
      std::free(base_ptr);
      base_ptr=nullptr; 
    }

    virtual void **get_basearray()
    {
      return basearray;
    }
    virtual void *get_baseptr()
    {
      return base_ptr;
    }
    
  };

  
  class memallocator {
  public:
    memallocator() = default;
    
    // Rule of 3
    memallocator(const memallocator &) = delete;
    memallocator& operator=(const memallocator &) = delete; 
    // virtual destructor required so we can be subclassed
    virtual ~memallocator()=default;

    virtual void *malloc(memallocator_regionid id,std::size_t nbytes)=0;
    virtual void *calloc(memallocator_regionid id,std::size_t nbytes)=0;
    virtual void *realloc(memallocator_regionid id,void *ptr,std::size_t newsize)=0;
    virtual std::shared_ptr<nonmoving_copy_or_reference> obtain_nonmoving_copy_or_reference(memallocator_regionid id, void *ptr, std::size_t offset, std::size_t length)=0;
    virtual void free(memallocator_regionid id,void *ptr)=0;
  };

  class cmemallocator: public memallocator {
  public:
    void *malloc(memallocator_regionid id,std::size_t nbytes) {
      return std::malloc(nbytes);
    }
  
    void *calloc(memallocator_regionid id,std::size_t nbytes) {
      return std::calloc(nbytes,1);
    }

    void *realloc(memallocator_regionid id,void *ptr,std::size_t newsize) {
      return std::realloc(ptr,newsize);
    }
    std::shared_ptr<nonmoving_copy_or_reference> obtain_nonmoving_copy_or_reference(memallocator_regionid id, void *ptr, std::size_t offset, std::size_t nbytes)
    {
      void *copyptr = std::malloc(nbytes);
      memcpy(copyptr,((char *)ptr)+offset,nbytes);
      return std::make_shared<nonmoving_copy_or_reference_cmem>(offset,nbytes,ptr);
    }

    
    void free(memallocator_regionid id,void *ptr) {
      return std::free(ptr);
    }

    ~cmemallocator() {

    }
  };


}

#endif /* SNDE_MEMALLOCATOR */
