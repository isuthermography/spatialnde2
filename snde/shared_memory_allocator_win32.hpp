#ifndef SNDE_SHARED_MEMORY_ALLOCATOR_WIN32_HPP
#define SNDE_SHARED_MEMORY_ALLOCATOR_WIN32_HPP
#include <mutex>
#include <memory>
#include <unordered_map>

#include <cstdlib>

#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/memallocator.hpp"

#include "snde/shared_memory_allocator_common.hpp" // for memkey_hash

namespace snde {

  class nonmoving_copy_or_reference_win32: public nonmoving_copy_or_reference {
  public:
    // immutable once published
    void **basearray;
    void *shiftedptr;
    void *mmapaddr;
    size_t mmaplength;
    size_t ptrshift;
    
    nonmoving_copy_or_reference_win32(void **basearray,snde_index shift,snde_index length,void *mmapaddr, size_t mmaplength, size_t ptrshift);
    
    // rule of 3
    nonmoving_copy_or_reference_win32(const nonmoving_copy_or_reference_win32 &) = delete;
    nonmoving_copy_or_reference_win32& operator=(const nonmoving_copy_or_reference_win32 &) = delete; 
    virtual ~nonmoving_copy_or_reference_win32();  // virtual destructor required so we can be subclassed
    
    virtual void **get_shiftedarray();
    virtual void **get_basearray();
    virtual void set_shiftedarray();
    virtual void *get_shiftedptr();
  };

  class shared_memory_info_win32 {
  public:
    memallocator_regionid id;
    std::string shm_name;
    HANDLE hFile;
    void *addr;
    size_t membytes; 
    size_t addressbytes;

    shared_memory_info_win32(memallocator_regionid id,
			     std::string shm_name,
			     HANDLE hFile,
			     LPVOID addr,
			     size_t membytes,
                 size_t addressbytes);
  };

  //memkey_equal
  
  class shared_memory_allocator_win32: public memallocator {
  public:
    // recname, recrevision, and base_shm_name are all immutable
    // once created
    //std::string recpath;
    //uint64_t recrevision;
    //uint64_t originating_rss_unique_id
    //std::string base_shm_name; // not including _{id}.dat suffix

    std::mutex _admin; // final lock in locking order; used internally only
    // _shm_info is locked by the admin mutex
    std::unordered_map<std::tuple<std::string,uint64_t,uint64_t,memallocator_regionid>,shared_memory_info_win32,memkey_hash/*,memkey_equal*/> _shm_info;

    
    shared_memory_allocator_win32();

    std::string base_shm_name(std::string recpath, uint64_t recrevision,uint64_t originating_rss_unique_id);

    virtual void *malloc(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,std::size_t membytes,std::size_t addressbytes);
    virtual void *calloc(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,std::size_t membytes,std::size_t addressbytes);
    virtual void *realloc(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,void *ptr,std::size_t newsize);
    virtual bool supports_nonmoving_reference(); // returns true if this allocator can return a nonmoving reference rather than a copy. The nonmoving reference will stay coherent with the original.
    
    virtual std::shared_ptr<nonmoving_copy_or_reference> obtain_nonmoving_copy_or_reference(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id, void **basearray,void *ptr, std::size_t shift, std::size_t length);
    virtual void free(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,void *ptr);

    virtual ~shared_memory_allocator_win32();

    
  };
  
  
}
#endif // SNDE_SHARED_MEMORY_ALLOCATOR_WIN32_HPP
