#ifndef SNDE_ARRAYMANAGER_HPP
#define SNDE_ARRAYMANAGER_HPP


#include <cstring>
#include <cstdio>

#if defined(_MSC_VER) && _MSC_VER < 1900
#define snprintf _snprintf
#endif

#include "snde_error.hpp"
#include "lockmanager.hpp"
#include "allocator.hpp"


namespace snde {
  typedef void **ArrayPtr;
  static inline ArrayPtr ArrayPtr_fromint(unsigned long long intval) {return (void **)intval; } 

  class cachemanager: public std::enable_shared_from_this<cachemanager> { /* abstract base class for cache managers */
  public:
    virtual void mark_as_dirty(void **arrayptr,snde_index pos,snde_index numelem) {};
    virtual ~cachemanager() {};
    
  };

  static inline std::string AddrStr(void **ArrayPtr)
  {
    char buf[1000];
    snprintf(buf,999,"0x%lx",(unsigned long)ArrayPtr);
    return std::string(buf);
  }
  
  class allocationinfo { 
  public:
    std::shared_ptr<allocator> alloc;
    size_t allocindex; // index into alloc->arrays
    /* !!!***need something here to indicate which cache (or none) most recently updated the 
       data... perhaps the whole array or perhaps per specific piece 
       Need also to consider the locking semantics of the something. 
    */ 
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
    std::shared_ptr<allocator_alignment> alignment_requirements;
    std::unordered_map<void **,void **> allocation_arrays; // look up the arrays used for allocation
    std::multimap<void **,void **> arrays_managed_by_allocator; // look up the managed arrays by the allocator array... ordering is as the arrays are created, which follows the locking order

    std::mutex admin; /* serializes access to caches */
    std::unordered_map<std::string,std::shared_ptr<cachemanager>> _caches;

    

    arraymanager(std::shared_ptr<memallocator> memalloc,std::shared_ptr<allocator_alignment> alignment_requirements)
    {
      _memalloc=memalloc;
      this->alignment_requirements=alignment_requirements;
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
      allocators[arrayptr]=allocationinfo{std::make_shared<allocator>(_memalloc,locker,alignment_requirements,arrayptr,elemsize,totalnelem),0};
      locker->addarray(arrayptr);
    
      allocation_arrays[arrayptr]=arrayptr;
      arrays_managed_by_allocator.emplace(std::make_pair(arrayptr,arrayptr));
      
    
    }
  
    virtual void add_follower_array(void **allocatedptr,void **arrayptr,size_t elemsize)
    {
      //std::lock_guard<std::mutex> adminlock(admin);
      std::shared_ptr<allocator> alloc=allocators[allocatedptr].alloc;
      // Make sure arrayptr not already managed
      assert(allocators.find(arrayptr)==allocators.end());

      //alloc->add_other_array(arrayptr,elemsize);
      allocators[arrayptr]=allocationinfo{alloc,alloc->add_other_array(arrayptr,elemsize)};

      allocation_arrays[arrayptr]=allocatedptr;

      arrays_managed_by_allocator.emplace(std::make_pair(allocatedptr,arrayptr));
      locker->addarray(arrayptr);

    }

    virtual void add_unmanaged_array(void **allocatedptr,void **arrayptr)
    {
      allocators[arrayptr]=allocationinfo{nullptr,0};
      allocation_arrays[arrayptr]=nullptr;
      locker->addarray(arrayptr);
    }
    
    virtual void mark_as_dirty(cachemanager *owning_cache_or_null,void **arrayptr,snde_index pos,snde_index len)
    {
      std::unique_lock<std::mutex> lock(admin);
      size_t cnt=0;

      /* mark as region as dirty under all caches except for the owning cache (if specified) */

      /* copy first to avoid deadlock */
      std::vector<std::shared_ptr<cachemanager>> caches(_caches.size());
      for (auto & cache: _caches) {
	caches[cnt]=cache.second;
	cnt++;
      }
      lock.unlock();

      for (auto & cache: caches) {
	if (cache.get() != owning_cache_or_null) {
	  cache->mark_as_dirty(arrayptr,pos,len);
	}
      }

      
    }
    virtual void dirty_alloc(std::shared_ptr<lockholder> holder,void **arrayptr,std::string allocid, snde_index numelem)
    {
      snde_index startidx=holder->get_alloc(arrayptr,allocid);
      mark_as_dirty(NULL,arrayptr,startidx,numelem);
    }
    
    virtual snde_index get_elemsize(void **arrayptr)
    {
      struct allocationinfo & alloc = allocators[arrayptr];

      return alloc.alloc->arrays[alloc.allocindex].elemsize;
    }
    virtual snde_index get_total_nelem(void **arrayptr)
    {
      struct allocationinfo & alloc = allocators[arrayptr];

      //alloc.alloc->arrays[alloc.allocindex].elemsize
      return alloc.alloc->_totalnchunks*alloc.alloc->_allocchunksize;
    }

    virtual void realloc_down(void **allocatedptr,snde_index addr,snde_index orignelem, snde_index newnelem)
    {
      /* Shrink an allocation. Can ONLY be called if you have a write lock to this allocation */
      std::shared_ptr<allocator> alloc=allocators[allocatedptr].alloc;
      alloc->realloc_down(addr,orignelem,newnelem);
    }

    // This next one gives SWIG trouble because of confusion over whether snde_index is an unsigned long or an unsigned long long
    virtual std::pair<snde_index,std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>>> alloc_arraylocked(rwlock_token_set all_locks,void **allocatedptr,snde_index nelem)
    {
      //std::lock_guard<std::mutex> adminlock(admin);
      std::shared_ptr<allocator> alloc=allocators[allocatedptr].alloc;
      return alloc->alloc_arraylocked(all_locks,nelem);    
    }
    
    virtual std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>> alloc_arraylocked_swigworkaround(snde::rwlock_token_set all_locks,void **allocatedptr,snde_index nelem,snde_index *OUTPUT)
    {
      std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>> retvec;
      
      std::tie(*OUTPUT,retvec)=alloc_arraylocked(all_locks,allocatedptr,nelem);
      return retvec;
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

    virtual void cleararrays(void *structaddr, size_t structlen)
    {
      /* clear all arrays within the specified structure */
      size_t pos;
      char *thisaddr;
      
      /* for each address in the structure... */
      for (pos=0; pos < structlen; pos++) {
	thisaddr = ((char *)structaddr)+pos;

	/* find any allocator pointed at this addr */
	std::unordered_map<void **,allocationinfo>::iterator this_arrayptr_allocationinfo;
	for (std::unordered_map<void **,allocationinfo>::iterator next_arrayptr_allocationinfo=allocators.begin();next_arrayptr_allocationinfo != allocators.end();) {
	  this_arrayptr_allocationinfo=next_arrayptr_allocationinfo;
	  next_arrayptr_allocationinfo++;
	  
	  if ((char *)this_arrayptr_allocationinfo->first == thisaddr) {
	    /* match! */

	    this_arrayptr_allocationinfo->second.alloc->remove_array((void **)thisaddr);
	    
	    if (this_arrayptr_allocationinfo->second.alloc->num_arrays()==0 && this_arrayptr_allocationinfo->second.alloc.use_count() > 1) {
	      throw(std::runtime_error("Residual references to array allocation during structure deletion")); /* This error indicates that excess std::shared_ptr<allocator> references are alive during cleanup */
	    }
	    
	    allocators.erase(this_arrayptr_allocationinfo);
	  }
	
	}
      }
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
