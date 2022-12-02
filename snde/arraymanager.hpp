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


    allocationinfo():
      alloc(nullptr),
      elemsize(0),
      _totalnelem(0),
      allocindex(0)
    {
      
    }

    allocationinfo(std::shared_ptr<allocator> alloc,size_t elemsize,size_t _totalnelem,size_t allocindex):
      alloc(alloc),
      elemsize(elemsize),
      _totalnelem(_totalnelem),
      allocindex(allocindex)
    {

    }
    
    allocationinfo(const allocationinfo &orig):
      alloc(orig.alloc),
      elemsize(orig.elemsize),
      _totalnelem(orig._totalnelem),
      allocindex(orig.allocindex)
    {

    }

    allocationinfo& operator=(const allocationinfo &orig)
    {
      alloc=orig.alloc;
      elemsize=orig.elemsize;
      _totalnelem=orig._totalnelem;
      allocindex=orig.allocindex;

      return *this;
    }
    ~allocationinfo() = default;
    
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

  class pending_modified_tracker {
  public:
    std::mutex admin;
    rangetracker<markedregion> pending_modified;

    void mark_as_pending_modified(snde_index base_index,snde_index len)
    {
      std::lock_guard<std::mutex> adminlock(admin);
      pending_modified.mark_region(base_index,len);
    }

    void clear_pending_modified(snde_index base_index,snde_index len)
    {
      std::lock_guard<std::mutex> adminlock(admin);
      pending_modified.clear_region(base_index,len);
    }


    rangetracker<markedregion> find_pending_modified(snde_index base_index,snde_index len)
    {
      std::lock_guard<std::mutex> adminlock(admin);
      return pending_modified.iterate_over_marked_portions(base_index,len);
    }
  };
  
  class arraymanager : public std::enable_shared_from_this<arraymanager> {
  public: 
    /* Look up allocator by __allocated_array_pointer__ only */
    /* mapping index is arrayptr, returns allocator */
    /* in this simple manager, allocators _memalloc and locker are presumed to be fixed
       after single-threaded startup, so we don't worry about thread safety */
    //std::unordered_map<void **,std::shared_ptr<allocator>> allocators;
    std::shared_ptr<memallocator> _memalloc; /* must be fixed and unchanged after initialization (reason for leading underscore is unclear (?))*/
    std::shared_ptr<lockmanager> locker; /* must be fixed and unchanged after initialization */
    std::shared_ptr<allocator_alignment> alignment_requirements; /* must be fixed and unchanged after initialization */
    std::size_t maxaddressbytes;

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
       allocation_arrays(), arrays_managed_by_allocator(), and _pending_modified_trackers() methods
    */

    std::shared_ptr<std::unordered_map<void **,allocationinfo>> _allocators; // C++11 atomic shared_ptr
    std::shared_ptr<std::unordered_map<void **,void **>> _allocation_arrays; // C++11 atomic shared_ptr: look up the arrays used for allocation
    std::shared_ptr<std::multimap<void **,void **>> _arrays_managed_by_allocator; // C++11 atomic shared_ptr: look up the managed arrays by the allocator array... ordering is as the arrays are created, which follows the locking order
    //std::shared_ptr<std::unordered_map<std::string,std::shared_ptr<cachemanager>>> __caches; // C++11 atomic shared_ptr: Look up a particular cachemanager by name

    // pending_modified_trackers keep track of array ranges that have been allocated (and thus a modification is pending) but for which the modification is not complete. They are marked on allocation, and then should be cleared either when the data is explicitly passed in (via mark_as_modified()) or when the data implicitly must be complete (via mark_as_ready()). If cleared via mark_as_ready() then all caches should be invalidated for this region, as data was implicitly written. Also cleared on free. 
    std::shared_ptr<std::unordered_map<void **,std::shared_ptr<pending_modified_tracker>>> _pending_modified_trackers;
    

    

    arraymanager(std::shared_ptr<memallocator> memalloc,std::shared_ptr<allocator_alignment> alignment_requirements,size_t maxaddressbytes,std::shared_ptr<lockmanager> locker=nullptr) :
        maxaddressbytes(maxaddressbytes)
    {
      std::atomic_store(&_allocators,std::make_shared<std::unordered_map<void **,allocationinfo>>());
      std::atomic_store(&_allocation_arrays,std::make_shared<std::unordered_map<void **,void **>>());
      std::atomic_store(&_arrays_managed_by_allocator,std::make_shared<std::multimap<void **,void **>>());
      std::atomic_store(&_pending_modified_trackers,std::make_shared<std::unordered_map<void **,std::shared_ptr<pending_modified_tracker>>>());
      //std::atomic_store(&__caches,std::make_shared<std::unordered_map<std::string,std::shared_ptr<cachemanager>>>());
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
    //virtual std::shared_ptr<std::unordered_map<std::string,std::shared_ptr<cachemanager>>> _caches()
    //{

      // look up the managed arrays by the allocator array... ordering is as the arrays are created, which follows the locking order
    //return std::atomic_load(&__caches);
    //}


    virtual std::shared_ptr<std::unordered_map<void **,std::shared_ptr<pending_modified_tracker>>> pending_modified_trackers()
    {
      return std::atomic_load(&_pending_modified_trackers);
    }


    virtual std::tuple<std::shared_ptr<std::unordered_map<void **,allocationinfo>>,
		       std::shared_ptr<std::unordered_map<void **,void **>>,
		       std::shared_ptr<std::multimap<void **,void **>>,
		       std::shared_ptr<std::unordered_map<void **,std::shared_ptr<pending_modified_tracker>>>> _begin_atomic_update()
    // adminlock must be locked when calling this function...
    // it returns new copies of the atomically-guarded data
    {

      // Make copies of atomically-guarded data 
      std::shared_ptr<std::unordered_map<void **,allocationinfo>> new_allocators=std::make_shared<std::unordered_map<void **,allocationinfo>>(*allocators());
      std::shared_ptr<std::unordered_map<void **,void **>> new_allocation_arrays=std::make_shared<std::unordered_map<void **,void **>>(*allocation_arrays());
      std::shared_ptr<std::multimap<void **,void **>> new_arrays_managed_by_allocator=std::make_shared<std::multimap<void **,void **>>(*arrays_managed_by_allocator());      
      std::shared_ptr<std::unordered_map<void **,std::shared_ptr<pending_modified_tracker>>> new_pending_modified_trackers=std::make_shared<std::unordered_map<void **,std::shared_ptr<pending_modified_tracker>>>(*pending_modified_trackers());
      

      return std::make_tuple(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator,new_pending_modified_trackers);
    }

    virtual void _end_atomic_update(std::shared_ptr<std::unordered_map<void **,allocationinfo>> new_allocators,				  
				    std::shared_ptr<std::unordered_map<void **,void **>> new_allocation_arrays,
				    std::shared_ptr<std::multimap<void **,void **>> new_arrays_managed_by_allocator,
				    std::shared_ptr<std::unordered_map<void **,std::shared_ptr<pending_modified_tracker>>> new_pending_modified_trackers)
    // adminlock must be locked when calling this function...
    {

      // replace old with new

      std::atomic_store(&_allocators,new_allocators);
      std::atomic_store(&_allocation_arrays,new_allocation_arrays);
      std::atomic_store(&_arrays_managed_by_allocator,new_arrays_managed_by_allocator);
      std::atomic_store(&_pending_modified_trackers,new_pending_modified_trackers);
    }
    

    //virtual std::tuple<std::shared_ptr<std::unordered_map<std::string,std::shared_ptr<cachemanager>>>> _begin_caches_atomic_update()
    //// adminlock must be locked when calling this function...
    //// it returns new copies of the atomically-guarded data
    //{

      // Make copies of atomically-guarded data 
    // std::shared_ptr<std::unordered_map<std::string,std::shared_ptr<cachemanager>>> new__caches=std::make_shared<std::unordered_map<std::string,std::shared_ptr<cachemanager>>>(*_caches());      

    //	return std::make_tuple(new__caches);
    //}

    //virtual void _end_caches_atomic_update(std::shared_ptr<std::unordered_map<std::string,std::shared_ptr<cachemanager>>> new__caches)
    // adminlock must be locked when calling this function...
    //{
      
      // replace old with new

      //std::atomic_store(&__caches,new__caches);
    //}

    
    virtual void add_allocated_array(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,void **arrayptr,size_t elemsize,snde_index totalnelem,const std::set<snde_index>& follower_elemsizes = std::set<snde_index>())
    {
      //std::lock_guard<std::mutex> adminlock(admin);

      {
	std::lock_guard<std::mutex> adminlock(admin); // required because we are updating atomically-guarded data
	// Make copies of atomically-guarded data 
	std::shared_ptr<std::unordered_map<void **,allocationinfo>> new_allocators;
	std::shared_ptr<std::unordered_map<void **,void **>> new_allocation_arrays;
	std::shared_ptr<std::multimap<void **,void **>> new_arrays_managed_by_allocator;
	std::shared_ptr<std::unordered_map<void **,std::shared_ptr<pending_modified_tracker>>> new_pending_modified_trackers;
	
	std::tie(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator,new_pending_modified_trackers)
	  = _begin_atomic_update();
	
	// Make sure arrayptr not already managed
	assert(new_allocators->find(arrayptr)==new_allocators->end());
	
	//allocators[arrayptr]=std::make_shared<allocator>(_memalloc,locker,arrayptr,elemsize,totalnelem);
	
	
	(*new_allocators)[arrayptr]=allocationinfo(std::make_shared<allocator>(_memalloc,locker,recording_path,recrevision,originating_rss_unique_id,id,alignment_requirements,arrayptr,elemsize,0,follower_elemsizes,maxaddressbytes),elemsize,totalnelem,0);
	
	(*new_allocation_arrays)[arrayptr]=arrayptr;
	new_arrays_managed_by_allocator->emplace(std::make_pair(arrayptr,arrayptr));
	new_pending_modified_trackers->emplace(arrayptr,std::make_shared<pending_modified_tracker>());
	
	// replace old with new
	_end_atomic_update(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator,new_pending_modified_trackers);
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
	std::shared_ptr<std::unordered_map<void **,std::shared_ptr<pending_modified_tracker>>> new_pending_modified_trackers;
	
	std::tie(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator,new_pending_modified_trackers)
	  = _begin_atomic_update();
	
	std::shared_ptr<allocator> alloc=(*new_allocators).at(allocatedptr).alloc;
	// Make sure arrayptr not already managed
	assert(new_allocators->find(arrayptr)==new_allocators->end());

	// Make sure alignment requirements were previously registered when we did add_allocated_array()
	//assert(std::find(std::begin(alloc->our_alignment.address_alignment),std::end(alloc->our_alignment.address_alignment),elemsize) != std::end(alloc->our_alignment.address_alignment));
	//assert(alloc->_allocchunksize % elemsize == 0);

	
	//alloc->add_other_array(arrayptr,elemsize);
	(*new_allocators)[arrayptr]=allocationinfo(alloc,elemsize,0,alloc->add_other_array(id,arrayptr,elemsize));
	
	(*new_allocation_arrays)[arrayptr]=allocatedptr;
	
	new_arrays_managed_by_allocator->emplace(std::make_pair(allocatedptr,arrayptr));
	new_pending_modified_trackers->emplace(arrayptr,std::make_shared<pending_modified_tracker>());

	// replace old with new
	_end_atomic_update(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator,new_pending_modified_trackers);
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
	
	std::shared_ptr<std::unordered_map<void **,std::shared_ptr<pending_modified_tracker>>> new_pending_modified_trackers;

	
	std::tie(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator,new_pending_modified_trackers)
	  = _begin_atomic_update();
	
	(*new_allocators)[arrayptr]=allocationinfo(nullptr,elemsize,totalnelem,0);
	(*new_allocation_arrays)[arrayptr]=nullptr;

	new_pending_modified_trackers->emplace(arrayptr,std::make_shared<pending_modified_tracker>());
	
	// replace old with new
	_end_atomic_update(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator,new_pending_modified_trackers);
      }
      
      locker->addarray(arrayptr);
    }
    
    //virtual void mark_as_dirty(cachemanager *owning_cache_or_null,void **arrayptr,snde_index pos,snde_index len)
    //{
      //std::unique_lock<std::mutex> lock(admin);
    //  size_t cnt=0;

      /* mark as region as dirty under all caches except for the owning cache (if specified) */

      /* Obtain immutable map copy */
    // std::shared_ptr<std::unordered_map<std::string,std::shared_ptr<cachemanager>>> caches=_caches();
      
      //std::vector<std::shared_ptr<cachemanager>> caches(_caches.size());
      //for (auto & cache: _caches) {
      //caches[cnt]=cache.second;
      //cnt++;
      //}
      //lock.unlock();

      //for (auto & cache: (*caches)) {
    //	if (cache.second.get() != owning_cache_or_null) {
    //	  cache.second->mark_as_dirty(shared_from_this(),arrayptr,pos,len);
    //	}
    //}
      
      
    // }
    //virtual void dirty_alloc(std::shared_ptr<lockholder> holder,void **arrayptr,std::string allocid, snde_index numelem)
    //{
    //  snde_index startidx=holder->get_alloc(arrayptr,allocid);
    //  mark_as_dirty(NULL,arrayptr,startidx,numelem);
    //}
    
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

    virtual void mark_as_pending_modified(void **arrayptr,snde_index base_index,snde_index len)
    {
      pending_modified_trackers()->at(arrayptr)->mark_as_pending_modified(base_index,len);
    }

    virtual void clear_pending_modified(void **arrayptr,snde_index base_index,snde_index len)
    {
      pending_modified_trackers()->at(arrayptr)->clear_pending_modified(base_index,len);
    }

    virtual rangetracker<markedregion> find_pending_modified(void **arrayptr,snde_index base_index,snde_index len)
    {
      return pending_modified_trackers()->at(arrayptr)->find_pending_modified(base_index,len);
    }

    virtual void realloc_down(void **allocatedptr,snde_index addr,snde_index orignelem, snde_index newnelem)
    {
      /* Shrink an allocation. Can ONLY be called if you have a write lock to this allocation */
      std::shared_ptr<allocator> alloc=(*allocators()).at(allocatedptr).alloc;

      // clear pending_modified for region no longer allocated in this array and all followers

      std::shared_ptr<std::multimap<void **,void **>> managed_arrays = arrays_managed_by_allocator();
      std::multimap<void **,void **>::iterator managed_array_begin,managed_array_end;
      std::tie(managed_array_begin,managed_array_end)=managed_arrays->equal_range(allocatedptr);
      for (auto managed_array_it=managed_array_begin;managed_array_it != managed_array_end;++managed_array_it) {
	pending_modified_trackers()->at(managed_array_it->second)->clear_pending_modified(addr+newnelem,orignelem-newnelem);
      }

      // perform realloc_down
      alloc->realloc_down(addr,orignelem,newnelem);
    }

    // This next one gives SWIG trouble because of confusion over whether snde_index is an unsigned long or an unsigned long long
    virtual std::pair<snde_index,std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>>> alloc_arraylocked(rwlock_token_set all_locks,void **allocatedptr,snde_index nelem)
    {
      //std::lock_guard<std::mutex> adminlock(admin);
      std::shared_ptr<allocator> alloc=(*allocators()).at(allocatedptr).alloc;

      // perform allocation
      std::pair<snde_index,std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>>> retval = alloc->alloc_arraylocked(all_locks,nelem);

      // Mark allocated region in leader and follower arrays
      // as pending_modified (should be cleared when recording is marked as ready

      snde_index addr = retval.first;
      std::shared_ptr<std::multimap<void **,void **>> managed_arrays = arrays_managed_by_allocator();
      std::multimap<void **,void **>::iterator managed_array_begin,managed_array_end;
      std::tie(managed_array_begin,managed_array_end)=managed_arrays->equal_range(allocatedptr);
      for (auto managed_array_it=managed_array_begin;managed_array_it != managed_array_end;++managed_array_it) {
	pending_modified_trackers()->at(managed_array_it->second)->mark_as_pending_modified(addr,nelem);
      }

      
      return retval;
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

      //snde_warning("arraymanager: free(0x%llx,%d)",(unsigned long long)((uintptr_t)allocatedptr),addr);
      snde_index alloclen = alloc->get_length(addr);

      // remove any pending_modified markers for arrays managed
      // through this allocator
      std::shared_ptr<std::multimap<void **,void **>> managed_arrays = arrays_managed_by_allocator();
      std::multimap<void **,void **>::iterator managed_array_begin,managed_array_end;
      std::tie(managed_array_begin,managed_array_end)=managed_arrays->equal_range(allocatedptr);
      for (auto managed_array_it=managed_array_begin;managed_array_it != managed_array_end;++managed_array_it) {
	clear_pending_modified(managed_array_it->second,addr,alloclen);
      }
      
      
      //clear_pending_modified(allocatedptr,addr,alloclen); (included in above iteration)

      // perform free
      alloc->free(addr);    
    }

    virtual void clear() /* clear out all references to allocated and followed arrays */
    {
      std::set<void **> to_remove_from_locker;
      {
	std::lock_guard<std::mutex> adminlock(admin); // required because we are updating atomically-guarded data
	// Make copies of atomically-guarded data 
	std::shared_ptr<std::unordered_map<void **,allocationinfo>> new_allocators;
	std::shared_ptr<std::unordered_map<void **,void **>> new_allocation_arrays;
	std::shared_ptr<std::multimap<void **,void **>> new_arrays_managed_by_allocator;
	std::shared_ptr<std::unordered_map<void **,std::shared_ptr<pending_modified_tracker>>> new_pending_modified_trackers;
	
	std::tie(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator,new_pending_modified_trackers)
	  = _begin_atomic_update();

	for (auto && arrayptr_allocinfo: *new_allocators) {
	  to_remove_from_locker.emplace(arrayptr_allocinfo.first);
	}
	
	new_allocators->clear();
	new_allocation_arrays->clear();
	new_arrays_managed_by_allocator->clear();
	new_pending_modified_trackers->clear();
	
	// replace old with new
	_end_atomic_update(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator,new_pending_modified_trackers);
      }

      // * Should we remove these arrays from locker 
      for (auto && arrayaddr: to_remove_from_locker) {
	locker->remarray(arrayaddr);
      }
      
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
	std::shared_ptr<std::unordered_map<void **,std::shared_ptr<pending_modified_tracker>>> new_pending_modified_trackers;
	
	std::tie(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator,new_pending_modified_trackers)
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
	  new_pending_modified_trackers->erase(rangepos->second);
	  new_arrays_managed_by_allocator->erase(rangepos);	  
	}

	
	
	// replace old with new
	_end_atomic_update(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator,new_pending_modified_trackers);
      }
      locker->remarray(basearray);
    }
    
    virtual void cleararrays(void *structaddr, size_t structlen)
    {
      std::set<void **> matchedaddrs;
      
      /* clear all arrays within the specified structure */
      {
	std::lock_guard<std::mutex> adminlock(admin); // required because we are updating atomically-guarded data
	// Make copies of atomically-guarded data 
	std::shared_ptr<std::unordered_map<void **,allocationinfo>> new_allocators;
	std::shared_ptr<std::unordered_map<void **,void **>> new_allocation_arrays;
	std::shared_ptr<std::multimap<void **,void **>> new_arrays_managed_by_allocator;
	std::shared_ptr<std::unordered_map<void **,std::shared_ptr<pending_modified_tracker>>> new_pending_modified_trackers;
	
	std::tie(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator,new_pending_modified_trackers)
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

	      matchedaddrs.emplace((void **)thisaddr);
	      
	      this_arrayptr_allocationinfo->second.alloc->remove_array((void **)thisaddr);
	    
	      //if (this_arrayptr_allocationinfo->second.alloc->num_arrays()==0 && this_arrayptr_allocationinfo->second.alloc.use_count() > 3) {
	      //	throw(snde_error("Residual references to array allocation during structure deletion (addr 0x%llx (pos=%d)",(unsigned long long)thisaddr,pos)); /* This error indicates that excess std::shared_ptr<allocator> references are alive during cleanup. I think it was important back when the free array space was used for the free list. */
	      //}
	    
	      while (new_arrays_managed_by_allocator->find(this_arrayptr_allocationinfo->first) != new_arrays_managed_by_allocator->end()) {
		auto amba_it = new_arrays_managed_by_allocator->find(this_arrayptr_allocationinfo->first);
		new_pending_modified_trackers->erase(amba_it->second);
		new_arrays_managed_by_allocator->erase(amba_it);
	      }
	      new_allocation_arrays->erase(new_allocation_arrays->find(this_arrayptr_allocationinfo->first));
	      new_allocators->erase(this_arrayptr_allocationinfo);
	    }
	    
	  }
	}
	// replace old with new
	_end_atomic_update(new_allocators,new_allocation_arrays,new_arrays_managed_by_allocator,new_pending_modified_trackers);
      }

      for (auto && arrayaddr: matchedaddrs) {
	locker->remarray(arrayaddr);
      }
    }
    
    //virtual std::shared_ptr<cachemanager> get_cache(std::string name)
    //{
      //std::lock_guard<std::mutex> lock(admin);
    //  return (*_caches()).at(name);
    //}

    //    virtual bool has_cache(std::string name)
    //{
      //std::lock_guard<std::mutex> lock(admin);
      /* Obtain immutable map copy */
    //      std::shared_ptr<std::unordered_map<std::string,std::shared_ptr<cachemanager>>> caches=_caches();

    //if (caches->find(name)==caches->end()) return false;
    // return true;
    // }

    //    virtual void set_undefined_cache(std::string name,std::shared_ptr<cachemanager> cache)
    /* set a cache according to name, if it is undefined. 
       If a corresponding cache already exists, this does nothing (does NOT replace 
       the cache) */
    //{
    //  std::lock_guard<std::mutex> lock(admin);
    //  std::shared_ptr<std::unordered_map<std::string,std::shared_ptr<cachemanager>>> new__caches;
    //  std::tie(new__caches) =  _begin_caches_atomic_update();

    //if (new__caches->find(name)==new__caches->end()) {
    //(*new__caches)[name]=cache;
    // }
    //  
    //_end_caches_atomic_update(new__caches);
    //}


    virtual ~arraymanager() {
      // allocators.clear(); (this is implicit)
      clear();
    }
  };



  

  
};
#endif /* SNDE_ARRAYMANAGER_HPP */
