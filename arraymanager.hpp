#ifndef SNDE_ARRAYMANAGER_HPP
#define SNDE_ARRAYMANAGER_HPP


#include <cstring>

#include "snde_error.hpp"
#include "lockmanager.hpp"
#include "allocator.hpp"


namespace snde {

  class cachemanager { /* abstract base class for cache managers */
  public:
    virtual ~cachemanager() {};
    
  };

  class allocationinfo { 
  public:
    std::shared_ptr<allocator> alloc;
    size_t allocindex; // index into alloc->arrays
  };

  
  //class arraymanager {
  //public:
  //  std::shared_ptr<lockmanager> locker;
  //  virtual void add_allocated_array(void **arrayptr,size_t elemsize,snde_index totalnelem)=0;
  //  virtual void add_follower_array(void **allocatedptr,void **arrayptr,size_t elemsize)=0;
  //  virtual snde_index alloc(void **allocatedptr,snde_index nelem)=0;
  //  virtual void free(void **allocatedptr,snde_index addr,snde_index nelem)=0;
  //  virtual void clear()=0; /* clear out all references to allocated and followed arrays */
  //  
  //  
  //  virtual ~arraymanager() {};
  //};
  
  class arraymanager  {
  public: 
    /* Look up allocator by __allocated_array_pointer__ only */
    /* mapping index is arrayptr, returns allocator */
    /* in this simple manager, allocators _memalloc and locker are presumed to be fixed
       after single-threaded startup, so we don't worry about thread safety */
    //std::unordered_map<void **,std::shared_ptr<allocator>> allocators;
    std::unordered_map<void **,allocationinfo> allocators;
    std::shared_ptr<memallocator> _memalloc;
    std::shared_ptr<lockmanager> locker;

    std::mutex admin; /* serializes access to caches */
    std::unordered_map<std::string,std::shared_ptr<cachemanager>> _caches;


    arraymanager(std::shared_ptr<memallocator> memalloc) {
      _memalloc=memalloc;
      locker = std::make_shared<lockmanager>();
    }

    arraymanager(const arraymanager &)=delete; /* copy constructor disabled */
    arraymanager& operator=(const arraymanager &)=delete; /* assignment disabled */

    virtual void add_allocated_array(void **arrayptr,size_t elemsize,snde_index totalnelem)
    {
      //std::lock_guard<std::mutex> adminlock(admin);

      // Make sure arrayptr not already managed
      assert(allocators.find(arrayptr)==allocators.end());

      //allocators[arrayptr]=std::make_shared<allocator>(_memalloc,locker,arrayptr,elemsize,totalnelem);
      allocators[arrayptr]=allocationinfo{std::make_shared<allocator>(_memalloc,locker,arrayptr,elemsize,totalnelem),0};
      locker->addarray(arrayptr);
    
    
    }
  
    virtual void add_follower_array(void **allocatedptr,void **arrayptr,size_t elemsize)
    {
      //std::lock_guard<std::mutex> adminlock(admin);
      std::shared_ptr<allocator> alloc=allocators[allocatedptr].alloc;
      // Make sure arrayptr not already managed
      assert(allocators.find(arrayptr)==allocators.end());

      //alloc->add_other_array(arrayptr,elemsize);
      allocators[arrayptr]=allocationinfo{alloc,alloc->add_other_array(arrayptr,elemsize)};

    }
    virtual snde_index alloc(void **allocatedptr,snde_index nelem)
    {
      //std::lock_guard<std::mutex> adminlock(admin);
      std::shared_ptr<allocator> alloc=allocators[allocatedptr].alloc;
      return alloc->alloc(nelem);    
    }

    virtual void free(void **allocatedptr,snde_index addr,snde_index nelem)
    {
      //std::lock_guard<std::mutex> adminlock(admin);
      std::shared_ptr<allocator> alloc=allocators[allocatedptr].alloc;
      alloc->free(addr,nelem);    
    }

    virtual void clear() /* clear out all references to allocated and followed arrays */
    {
      //std::lock_guard<std::mutex> adminlock(admin);
      allocators.clear();
    }
  
    virtual std::shared_ptr<cachemanager> get_cache(std::string name)
    {
      std::lock_guard<std::mutex> lock(admin);
      return _caches[name];
    }

    virtual bool has_cache(std::string name)
    {
      std::lock_guard<std::mutex> lock(admin);

      if (_caches.find(name)==_caches.end()) return false;
      return true;
    }

    virtual void set_cache(std::string name,std::shared_ptr<cachemanager> cache)
    {
      std::lock_guard<std::mutex> lock(admin);
      _caches[name]=cache;
    }

    virtual ~arraymanager() {
      // allocators.clear(); (this is implicit)
    }
  };



  

  
};
#endif /* SNDE_ARRAYMANAGER_HPP */
