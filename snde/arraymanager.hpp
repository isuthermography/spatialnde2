#ifndef SNDE_ARRAYMANAGER_HPP
#define SNDE_ARRAYMANAGER_HPP


#include <cstring>
#include <cstdio>

#if defined(_MSC_VER) && _MSC_VER < 1900
#define snprintf _snprintf
#endif

#include "snde/snde_error.hpp"
#include "snde/lockmanager.hpp"
#include "snde/allocator.hpp"


namespace snde {
  typedef void **ArrayPtr;
  static inline ArrayPtr ArrayPtr_fromint(unsigned long long intval) {return (void **)intval; } 

  class cachemanager: public std::enable_shared_from_this<cachemanager> { /* abstract base class for cache managers */
  public:
    virtual void mark_as_dirty(std::shared_ptr<arraymanager> manager,void **arrayptr,snde_index pos,snde_index numelem) {};
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
    size_t elemsize;
    size_t _totalnelem; // ***!!! Only used when alloc==nullptr; so far there is no way to update this (!)... whene there is, see openclccachemanager pool_realloc_callback
    size_t allocindex; // index into alloc->arrays, if alloc is not nullptr
    /* !!!***need something here to indicate which cache (or none) most recently updated the 
       data... perhaps the whole array or perhaps per specific piece 
       Need also to consider the locking semantics of the something. 
    */
    size_t totalnelem()
    {
      if (!alloc) return _totalnelem;
      else {
	return alloc->total_nelem();
      }
    }
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
  
  class arraymanager : public std::enable_shared_from_this<arraymanager> {
  public: 
    /* Look up allocator by __allocated_array_pointer__ only */
    /* mapping index is arrayptr, returns allocator */
    /* in this simple manager, allocators _memalloc and locker are presumed to be fixed
       after single-threaded startup, so we don't worry about thread safety */
    //std::unordered_map<void **,std::shared_ptr<allocator>> allocators;
    std::shared_ptr<memallocator> _memalloc; /* must be fixed and unchanged after initialization */
    std::shared_ptr<lockmanager> locker; /* must be fixed and unchanged after initialization */
    std::shared_ptr<allocator_alignment> alignment_requirements; /* must be fixed and unchanged after initialization */

    std::mutex admin; /* serializes  write access (but not read 
			 access) to _allocators, _allocation_arrays, 
			 arrays_managed_by_allocator and _caches, below... 
			 late in all locking orders... the allocatormutex's in the 
			 allocators may be locked while this is locked. */
    

    
    /* Synchronization model for _allocators, _allocation_arrays, 
       arrays_managed_by_allocator, and _caches: Atomic shared pointer for 
       the content for reading. To change the content, lock the 
       admin mutex, make a complete copy, then 
       switch the atomic pointer. 

       non-atomic shared pointer copy retrieved by the allocators(), 
       allocation_arrays(), arrays_managed_by_allocator(), and _caches() methods
    */

    std::shared_ptr<std::unordered_map<void **,allocationinfo>> _allocators; // C++11 atomic shared_ptr
    std::shared_ptr<std::unordered_map<void **,void **>> _allocation_arrays; // C++11 atomic shared_ptr: look up the arrays used for allocation
    std::shared_ptr<std::multimap<void **,void **>> _arrays_managed_by_allocator; // C++11 atomic shared_ptr: look up the managed arrays by the allocator array... ordering is as the arrays are created, which follows the locking order
    std::shared_ptr<std::unordered_map<std::string,std::shared_ptr<cachemanager>>> __caches; // C++11 atomic shared_ptr: Look up a particular cachemanager by name



    

    

    arraymanager(std::shared_ptr<memallocator> memalloc,std::shared_ptr<allocator_alignment> alignment_requirements,std::shared_ptr<lockmanager> locker=nullptr)
    {
      std::atomic_store(&_allocators,std::make_shared<std::unordered_map<void **,allocationinfo>>());
      std::atomic_store(&_allocation_arrays,std::make_shared<std::unordered_map<void **,void **>>());
      std::atomic_store(&_arrays_managed_by_allocator,std::make_shared<std::multimap<void **,void **>>());
      std::atomic_store(&__caches,std::make_shared<std::unordered_map<std::string,std::shared_ptr<cachemanager>>>());
      _memalloc=memalloc;
      
      this->alignment_requirements=alignment_requirements;
      if (!locker) {
	locker = std::make_shared<lockmanager>();
      }
      this->locker=locker;
    }

    arraymanager(const arraymanager &)=delete; /* copy constructor disabled */
    arraymanager& operator=(const arraymanager &)=delete; /* assignment disabled */

    virtual std::shared_ptr<std::unordered_map<void **,allocationinfo>> allocators()
    {
      return std::atomic_load(&_allocators);
    }
    virtual std::shared_ptr<std::unordered_map<void **,void **>> allocation_arrays()
    {
      // look up the arrays used for allocation
      return std::atomic_load(&_allocation_arrays);
    }
    
    virtual std::shared_ptr<std::multimap<void **,void **>> arrays_managed_by_allocator()
    {

      // look up the managed arrays by the allocator array... ordering is as the arrays are created, which follows the locking order
      return std::atomic_load(&_arrays_managed_by_allocator);
    }
    virtual std::shared_ptr<std::unordered_map<std::string,std::shared_ptr<cachemanager>>> _caches()
    {

      // look up the managed arrays by the allocator array... ordering is as the arrays are created, which follows the locking order
      return std::atomic_load(&__caches);
    }

    virtual std::tuple<std::shared_ptr<std::unordered_map<void **,allocationinfo>>,
		       std::shared_ptr<std::unordered_map<void **,void **>>,
		       std::shared_ptr<std::multimap<void **,void **>>> _begin_atomic_update()
    // adminlock must be locked when calling this function...
    // it returns new copies of the atomically-guarded data
    {

      // Make copies of atomically-guarded data 
      std::shared_ptr<std::unordered_map<void **,allocationinfo>> new_allocators=std::make_shared<std::unordered_map<void **,allocationinfo>>(*allocators());
      std::shared_ptr<std::unordered_map<void **,void **>> new_allocation_arrays=std::make_shared<std::unordered_map<void **,void **>>(*allocation_arrays());
      std::shared_ptr<std::multimap<void **,void **>> new_arrays_managed_by_allocator=std::make_shared<std::multimap<void **,void **>>(*arrays_managed_by_allocator());      


	return std::make_tuple(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator);
    }

    virtual void _end_atomic_update(std::shared_ptr<std::unordered_map<void **,allocationinfo>> new_allocators,				  
				    std::shared_ptr<std::unordered_map<void **,void **>> new_allocation_arrays,
				    std::shared_ptr<std::multimap<void **,void **>> new_arrays_managed_by_allocator)
    // adminlock must be locked when calling this function...
    {

      // replace old with new

      std::atomic_store(&_allocators,new_allocators);
      std::atomic_store(&_allocation_arrays,new_allocation_arrays);
      std::atomic_store(&_arrays_managed_by_allocator,new_arrays_managed_by_allocator);
    }


        virtual std::tuple<std::shared_ptr<std::unordered_map<std::string,std::shared_ptr<cachemanager>>>> _begin_caches_atomic_update()
    // adminlock must be locked when calling this function...
    // it returns new copies of the atomically-guarded data
    {

      // Make copies of atomically-guarded data 
      std::shared_ptr<std::unordered_map<std::string,std::shared_ptr<cachemanager>>> new__caches=std::make_shared<std::unordered_map<std::string,std::shared_ptr<cachemanager>>>(*_caches());      

	return std::make_tuple(new__caches);
    }

    virtual void _end_caches_atomic_update(std::shared_ptr<std::unordered_map<std::string,std::shared_ptr<cachemanager>>> new__caches)
    // adminlock must be locked when calling this function...
    {
      
      // replace old with new

      std::atomic_store(&__caches,new__caches);
    }

    
    virtual void add_allocated_array(std::string recording_path,uint64_t recrevision,memallocator_regionid id,void **arrayptr,size_t elemsize,snde_index totalnelem,const std::set<snde_index>& follower_elemsizes = std::set<snde_index>())
    {
      //std::lock_guard<std::mutex> adminlock(admin);

      {
	std::lock_guard<std::mutex> adminlock(admin); // required because we are updating atomically-guarded data
	// Make copies of atomically-guarded data 
	std::shared_ptr<std::unordered_map<void **,allocationinfo>> new_allocators;
	std::shared_ptr<std::unordered_map<void **,void **>> new_allocation_arrays;
	std::shared_ptr<std::multimap<void **,void **>> new_arrays_managed_by_allocator;
	
	std::tie(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator)
	  = _begin_atomic_update();
	
	// Make sure arrayptr not already managed
	assert(new_allocators->find(arrayptr)==new_allocators->end());
	
	//allocators[arrayptr]=std::make_shared<allocator>(_memalloc,locker,arrayptr,elemsize,totalnelem);
	
	
	(*new_allocators)[arrayptr]=allocationinfo{std::make_shared<allocator>(_memalloc,locker,recording_path,recrevision,id,alignment_requirements,arrayptr,elemsize,0,follower_elemsizes),elemsize,totalnelem,0};
	
	(*new_allocation_arrays)[arrayptr]=arrayptr;
	new_arrays_managed_by_allocator->emplace(std::make_pair(arrayptr,arrayptr));
	
	
	// replace old with new
	_end_atomic_update(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator);
      }
      locker->addarray(arrayptr);
    
    }
  
    virtual void add_follower_array(memallocator_regionid id,void **allocatedptr,void **arrayptr,size_t elemsize)
    {

      {
	std::lock_guard<std::mutex> adminlock(admin); // required because we are updating atomically-guarded data
	
	// Make copies of atomically-guarded data 
	std::shared_ptr<std::unordered_map<void **,allocationinfo>> new_allocators;
	std::shared_ptr<std::unordered_map<void **,void **>> new_allocation_arrays;
	std::shared_ptr<std::multimap<void **,void **>> new_arrays_managed_by_allocator;
	
	std::tie(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator)
	  = _begin_atomic_update();
	
	std::shared_ptr<allocator> alloc=(*new_allocators).at(allocatedptr).alloc;
	// Make sure arrayptr not already managed
	assert(new_allocators->find(arrayptr)==new_allocators->end());

	// Make sure alignment requirements were previously registered when we did add_allocated_array()
	//assert(std::find(std::begin(alloc->our_alignment.address_alignment),std::end(alloc->our_alignment.address_alignment),elemsize) != std::end(alloc->our_alignment.address_alignment));
	//assert(alloc->_allocchunksize % elemsize == 0);

	
	//alloc->add_other_array(arrayptr,elemsize);
	new_allocators->at(arrayptr)=allocationinfo{alloc,elemsize,0,alloc->add_other_array(id,arrayptr,elemsize)};
	
	(*new_allocation_arrays)[arrayptr]=allocatedptr;
	
	new_arrays_managed_by_allocator->emplace(std::make_pair(allocatedptr,arrayptr));

	// replace old with new
	_end_atomic_update(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator);
      }
      locker->addarray(arrayptr);

    }

    virtual void add_unmanaged_array(void **arrayptr,size_t elemsize,size_t totalnelem)
    // ***!!! NOTE: Currently not possible to resize unmanaged array -- need function
    // to notify arraymanager of new size... size also used by openclcachemanager -- see pool_realloc_callback
    {
      {
	std::lock_guard<std::mutex> adminlock(admin); // required because we are updating atomically-guarded data
	
	// Make copies of atomically-guarded data 
	std::shared_ptr<std::unordered_map<void **,allocationinfo>> new_allocators;
	std::shared_ptr<std::unordered_map<void **,void **>> new_allocation_arrays;
	std::shared_ptr<std::multimap<void **,void **>> new_arrays_managed_by_allocator;
	
	std::tie(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator)
	  = _begin_atomic_update();
	
	(*new_allocators)[arrayptr]=allocationinfo{nullptr,elemsize,totalnelem,0};
	(*new_allocation_arrays)[arrayptr]=nullptr;
	// replace old with new
	_end_atomic_update(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator);
      }
      
      locker->addarray(arrayptr);
    }
    
    virtual void mark_as_dirty(cachemanager *owning_cache_or_null,void **arrayptr,snde_index pos,snde_index len)
    {
      //std::unique_lock<std::mutex> lock(admin);
      size_t cnt=0;

      /* mark as region as dirty under all caches except for the owning cache (if specified) */

      /* Obtain immutable map copy */
      std::shared_ptr<std::unordered_map<std::string,std::shared_ptr<cachemanager>>> caches=_caches();
      
      //std::vector<std::shared_ptr<cachemanager>> caches(_caches.size());
      //for (auto & cache: _caches) {
      //caches[cnt]=cache.second;
      //cnt++;
      //}
      //lock.unlock();

      for (auto & cache: (*caches)) {
	if (cache.second.get() != owning_cache_or_null) {
	  cache.second->mark_as_dirty(shared_from_this(),arrayptr,pos,len);
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
      struct allocationinfo alloc = (*allocators()).at(arrayptr);

      //return (*alloc.alloc->arrays())[alloc.allocindex].elemsize;
      return alloc.elemsize;
    }
    virtual snde_index get_total_nelem(void **arrayptr)
    {
      struct allocationinfo alloc = (*allocators()).at(arrayptr);
      
      //alloc.alloc->arrays[alloc.allocindex].elemsize
      //if (alloc.alloc) {
      //return alloc.alloc->_totalnchunks*alloc.alloc->_allocchunksize;
      //} else {
	return alloc.totalnelem();
	//}
    }

    virtual void realloc_down(void **allocatedptr,snde_index addr,snde_index orignelem, snde_index newnelem)
    {
      /* Shrink an allocation. Can ONLY be called if you have a write lock to this allocation */
      std::shared_ptr<allocator> alloc=(*allocators()).at(allocatedptr).alloc;
      alloc->realloc_down(addr,orignelem,newnelem);
    }

    // This next one gives SWIG trouble because of confusion over whether snde_index is an unsigned long or an unsigned long long
    virtual std::pair<snde_index,std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>>> alloc_arraylocked(rwlock_token_set all_locks,void **allocatedptr,snde_index nelem)
    {
      //std::lock_guard<std::mutex> adminlock(admin);
      std::shared_ptr<allocator> alloc=(*allocators()).at(allocatedptr).alloc;
      return alloc->alloc_arraylocked(all_locks,nelem);    
    }
    
    virtual std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>> alloc_arraylocked_swigworkaround(snde::rwlock_token_set all_locks,void **allocatedptr,snde_index nelem,snde_index *OUTPUT)
    {
      std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>> retvec;
      
      std::tie(*OUTPUT,retvec)=alloc_arraylocked(all_locks,allocatedptr,nelem);
      return retvec;
    }

    virtual snde_index get_length(void **allocatedptr,snde_index addr)
    {
      //std::lock_guard<std::mutex> adminlock(admin);
      std::shared_ptr<allocator> alloc=(*allocators()).at(allocatedptr).alloc;
      return alloc->get_length(addr);    
    }

    virtual void free(void **allocatedptr,snde_index addr)
    {
      //std::lock_guard<std::mutex> adminlock(admin);
      std::shared_ptr<allocator> alloc=(*allocators()).at(allocatedptr).alloc;
      alloc->free(addr);    
    }

    virtual void clear() /* clear out all references to allocated and followed arrays */
    {
      {
	std::lock_guard<std::mutex> adminlock(admin); // required because we are updating atomically-guarded data
	// Make copies of atomically-guarded data 
	std::shared_ptr<std::unordered_map<void **,allocationinfo>> new_allocators;
	std::shared_ptr<std::unordered_map<void **,void **>> new_allocation_arrays;
	std::shared_ptr<std::multimap<void **,void **>> new_arrays_managed_by_allocator;
	
	std::tie(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator)
	  = _begin_atomic_update();
	new_allocators->clear();
	new_allocation_arrays->clear();
	new_arrays_managed_by_allocator->clear();
	
	// replace old with new
	_end_atomic_update(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator);
      }

      // * Should we remove these arrays from locker (?) 
    }

    virtual void remove_unmanaged_array(void **basearray)
    {
            /* clear all arrays within the specified structure */
      {
	std::lock_guard<std::mutex> adminlock(admin); // required because we are updating atomically-guarded data
	// Make copies of atomically-guarded data 
	std::shared_ptr<std::unordered_map<void **,allocationinfo>> new_allocators;
	std::shared_ptr<std::unordered_map<void **,void **>> new_allocation_arrays;
	std::shared_ptr<std::multimap<void **,void **>> new_arrays_managed_by_allocator;
	
	std::tie(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator)
	  = _begin_atomic_update();

	// remove all references from our data structures
	auto && alloc_iter = new_allocators->find(basearray);
	if (alloc_iter != new_allocators->end()) {
	  new_allocators->erase(alloc_iter);
	}

	auto && allocarray_iter = new_allocation_arrays->find(basearray);
	if (allocarray_iter != new_allocation_arrays->end()) {
	  new_allocation_arrays->erase(allocarray_iter);
	}
	
	std::multimap<void **,void **>::iterator rangepos,rangenext,rangeend;
	std::tie(rangepos,rangeend) = new_arrays_managed_by_allocator->equal_range(basearray);
	for (;rangepos != rangeend;rangepos=rangenext) {
	  rangenext=rangepos;
	  rangenext++;
	  new_arrays_managed_by_allocator->erase(rangepos);
	}

	
	
	// replace old with new
	_end_atomic_update(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator);
      }

    }
    
    virtual void cleararrays(void *structaddr, size_t structlen)
    {
      /* clear all arrays within the specified structure */
      {
	std::lock_guard<std::mutex> adminlock(admin); // required because we are updating atomically-guarded data
	// Make copies of atomically-guarded data 
	std::shared_ptr<std::unordered_map<void **,allocationinfo>> new_allocators;
	std::shared_ptr<std::unordered_map<void **,void **>> new_allocation_arrays;
	std::shared_ptr<std::multimap<void **,void **>> new_arrays_managed_by_allocator;
	
	std::tie(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator)
	  = _begin_atomic_update();
	
	size_t pos;
	char *thisaddr;
      
	/* for each address in the structure... */
	for (pos=0; pos < structlen; pos++) {
	  thisaddr = ((char *)structaddr)+pos;
	  
	  /* find any allocator pointed at this addr */
	  std::unordered_map<void **,allocationinfo>::iterator this_arrayptr_allocationinfo;
	  for (std::unordered_map<void **,allocationinfo>::iterator next_arrayptr_allocationinfo=new_allocators->begin();next_arrayptr_allocationinfo != new_allocators->end();) {
	    this_arrayptr_allocationinfo=next_arrayptr_allocationinfo;
	    next_arrayptr_allocationinfo++;
	    
	    if ((char *)this_arrayptr_allocationinfo->first == thisaddr) {
	      /* match! */
	      
	      this_arrayptr_allocationinfo->second.alloc->remove_array((void **)thisaddr);
	    
	      if (this_arrayptr_allocationinfo->second.alloc->num_arrays()==0 && this_arrayptr_allocationinfo->second.alloc.use_count() > 2) {
		throw(std::runtime_error("Residual references to array allocation during structure deletion")); /* This error indicates that excess std::shared_ptr<allocator> references are alive during cleanup */
	      }
	    
	      while (new_arrays_managed_by_allocator->find(this_arrayptr_allocationinfo->first) != new_arrays_managed_by_allocator->end()) {
		new_arrays_managed_by_allocator->erase(new_arrays_managed_by_allocator->find(this_arrayptr_allocationinfo->first));
	      }
	      new_allocation_arrays->erase(new_allocation_arrays->find(this_arrayptr_allocationinfo->first));
	      new_allocators->erase(this_arrayptr_allocationinfo);
	    }
	    
	  }
	}
	// replace old with new
	_end_atomic_update(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator);
      }
    }
    
    virtual std::shared_ptr<cachemanager> get_cache(std::string name)
    {
      //std::lock_guard<std::mutex> lock(admin);
      return (*_caches()).at(name);
    }

    virtual bool has_cache(std::string name)
    {
      //std::lock_guard<std::mutex> lock(admin);
      /* Obtain immutable map copy */
      std::shared_ptr<std::unordered_map<std::string,std::shared_ptr<cachemanager>>> caches=_caches();

      if (caches->find(name)==caches->end()) return false;
      return true;
    }

    virtual void set_undefined_cache(std::string name,std::shared_ptr<cachemanager> cache)
    /* set a cache according to name, if it is undefined. 
       If a corresponding cache already exists, this does nothing (does NOT replace 
       the cache) */
    {
      std::lock_guard<std::mutex> lock(admin);
      std::shared_ptr<std::unordered_map<std::string,std::shared_ptr<cachemanager>>> new__caches;
      std::tie(new__caches) =  _begin_caches_atomic_update();

      if (new__caches->find(name)==new__caches->end()) {
	(*new__caches)[name]=cache;
      }
      
      _end_caches_atomic_update(new__caches);
    }


    virtual ~arraymanager() {
      // allocators.clear(); (this is implicit)
    }
  };



  

  
};
#endif /* SNDE_ARRAYMANAGER_HPP */
