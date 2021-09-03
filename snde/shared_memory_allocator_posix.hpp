#include <mutex>
#include <memory>
#include <unordered_map>

#include <cstdlib>

#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/memallocator.hpp"

namespace snde {

  class nonmoving_copy_or_reference_posix: public nonmoving_copy_or_reference {
  public:
    // immutable once published
    void *baseptr;
    void *mmapaddr;
    size_t mmaplength;
    size_t ptroffset;
    
    nonmoving_copy_or_reference_posix(snde_index offset,snde_index length,void *mmapaddr, size_t mmaplength, size_t ptroffset);
    
    // rule of 3
    nonmoving_copy_or_reference_posix(const nonmoving_copy_or_reference_posix &) = delete;
    nonmoving_copy_or_reference_posix& operator=(const nonmoving_copy_or_reference_posix &) = delete; 
    virtual ~nonmoving_copy_or_reference_posix();  // virtual destructor required so we can be subclassed
    
    virtual void **get_basearray();
    virtual void set_basearray();
    virtual void *get_baseptr();
  };

  class shared_memory_info_posix {
  public:
    memallocator_regionid id;
    std::string shm_name;
    int fd;
    void *addr;
    size_t nbytes; 

    shared_memory_info_posix(memallocator_regionid id,
			     std::string shm_name,
			     int fd,
			     void *addr,
			     size_t nbytes);
  };
  
  class shared_memory_allocator_posix: public memallocator {
  public:
    // recname, recrevision, and base_shm_name are all immutable
    // once created
    std::string recpath;
    uint64_t recrevision;
    std::string base_shm_name; // not including _{id}.dat suffix

    std::mutex _admin; // final lock in locking order; used internally only
    // _shm_info is locked by the admin mutex
    std::unordered_map<memallocator_regionid,shared_memory_info_posix> _shm_info;

    
    shared_memory_allocator_posix(std::string recpath,uint64_t recrevision);
    
    virtual void *malloc(memallocator_regionid id,std::size_t nbytes);
    virtual void *calloc(memallocator_regionid id,std::size_t nbytes);
    virtual void *realloc(memallocator_regionid id,void *ptr,std::size_t newsize);
    virtual std::shared_ptr<nonmoving_copy_or_reference> obtain_nonmoving_copy_or_reference(memallocator_regionid id, void *ptr, std::size_t offset, std::size_t length);
    virtual void free(memallocator_regionid id,void *ptr);

    virtual ~shared_memory_allocator_posix();
    
  };
  
  
}