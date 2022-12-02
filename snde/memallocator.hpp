#ifndef SNDE_MEMALLOCATOR
#define SNDE_MEMALLOCATOR

/* This is an abstraction layer that will permit 
   substitution of a specialized memory allocator...
   i.e. one that allocates space directly in OpenCL 
   memory on the GPU card */

#include <memory>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <cstring>


namespace snde {
  // NOTE: for recording_storage_manager_simple, the index into the multiarray and the memallocator
  // regionid are the same. But this is NOT true for the graphics_storage_manager, which uses index
  // of definition of array in the graphics_storage constructor as the memallocator_regionid
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
    size_t shift;
    size_t length; // bytes
    bool requires_locking_read;
    bool requires_locking_write;
    
    nonmoving_copy_or_reference(size_t shift,size_t length,bool requires_locking_read,bool requires_locking_write) :
      shift(shift),
      length(length),
      requires_locking_read(requires_locking_read),
      requires_locking_write(requires_locking_write)
    {

    }
    // Rule of 3
    nonmoving_copy_or_reference(const nonmoving_copy_or_reference &) = delete;
    nonmoving_copy_or_reference& operator=(const nonmoving_copy_or_reference &) = delete; 
    virtual ~nonmoving_copy_or_reference()=default;  // virtual destructor required so we can be subclassed
    
    virtual void *get_shiftedptr()=0;
    virtual void **get_basearray()=0;
  };

  class nonmoving_copy_or_reference_cmem: public nonmoving_copy_or_reference {
  public:
    void **shiftedarray; // pointer to the maintained pointer for the data
    void *shifted_ptr; // will never move
    void **basearray; // for locking, etc. 
    nonmoving_copy_or_reference_cmem(size_t shift,size_t length,void **basearray,void *orig_ptr) :
      nonmoving_copy_or_reference(shift,length,false,false),
      shiftedarray(&shifted_ptr),
      shifted_ptr(malloc(length)),
      basearray(basearray)
    {
      memcpy(shifted_ptr,((char *)orig_ptr)+shift,length);
    }
    // rule of 3
    nonmoving_copy_or_reference_cmem(const nonmoving_copy_or_reference_cmem &) = delete;
    nonmoving_copy_or_reference_cmem& operator=(const nonmoving_copy_or_reference_cmem &) = delete; 
    virtual ~nonmoving_copy_or_reference_cmem()  // virtual destructor required so we can be subclassed
    {
      std::free(shifted_ptr);
      shifted_ptr=nullptr; 
    }

    virtual void **get_basearray()
    {
      return basearray;
    }
    virtual void *get_shiftedptr()
    {
      return shifted_ptr;
    }
    
  };

  
  class memallocator {
  public:
    bool requires_locking_read; // if set, this memallocator hooks in with the lockmanager and requires that a memory block be locked prior to reading. This offers the potential to support VRAM buffers that are mapped to the CPU or host-memory GPU buffers. Note that the infrastructure in the lockmanager to call back the memallocator to make the memory available to CPU is not in place as of this writing
    bool requires_locking_write; // if set, this memallocator hooks in with the lockmanager and requires that a memory block be locked prior to reading. This offers the potential to support VRAM buffers that are mapped to the CPU or host-memory GPU buffers
    
    memallocator(bool requires_locking_read,bool requires_locking_write) :
      requires_locking_read(requires_locking_read),
      requires_locking_write(requires_locking_write)
    {

    }
    
    // Rule of 3
    memallocator(const memallocator &) = delete;
    memallocator& operator=(const memallocator &) = delete; 
    // virtual destructor required so we can be subclassed
    virtual ~memallocator()=default;

    virtual void *malloc(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,std::size_t membytes,std::size_t addressbytes)=0;
    virtual void *calloc(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,std::size_t membytes,std::size_t addressbytes)=0;
    virtual void *realloc(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,void *ptr,std::size_t newsize)=0;
    virtual bool supports_nonmoving_reference()=0; // returns true if this allocator can return a nonmoving reference rather than a copy. The nonmoving reference will stay coherent with the original. 
    virtual std::shared_ptr<nonmoving_copy_or_reference> obtain_nonmoving_copy_or_reference(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id, void **basearray,void *ptr, std::size_t offset, std::size_t length)=0; // Note: May (generally should be) called with the underlying data array locked for read to prevent ptr from being modified under us
    virtual void free(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,void *ptr)=0;
  };

  class cmemallocator: public memallocator {
  public:


    cmemallocator() :
      memallocator(false,false)
    {

    }
    
    // Rule of 3
    cmemallocator(const cmemallocator &) = delete;
    cmemallocator& operator=(const cmemallocator &) = delete; 
    // virtual destructor required so we can be subclassed
    ~cmemallocator() {

    }

    void *malloc(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,std::size_t membytes,std::size_t addressbytes) {
      return std::malloc(membytes);
    }
  
    void *calloc(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,std::size_t membytes,std::size_t addressbytes) {
      return std::calloc(membytes,1);
    }

    void *realloc(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,void *ptr,std::size_t newsize) {
      return std::realloc(ptr,newsize);
    }
    bool supports_nonmoving_reference() // returns true if this allocator can return a nonmoving reference rather than a copy. The nonmoving reference will stay coherent with the original.
    {
      return false;
    }
    
    std::shared_ptr<nonmoving_copy_or_reference> obtain_nonmoving_copy_or_reference(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id, void **basearray,void *ptr, std::size_t offset, std::size_t nbytes)
    {
      //void *copyptr = std::malloc(nbytes);
      //memcpy(copyptr,((char *)ptr)+offset,nbytes);
      return std::make_shared<nonmoving_copy_or_reference_cmem>(offset,nbytes,basearray,ptr);
    }

    
    void free(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,void *ptr) {
      return std::free(ptr);
    }

  };


}

#endif /* SNDE_MEMALLOCATOR */
