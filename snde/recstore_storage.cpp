#include "snde/recstore_storage.hpp"
#include "snde/allocator.hpp"

#ifdef _WIN32
#include "snde/shared_memory_allocator_win32.hpp"
#else
#include "snde/shared_memory_allocator_posix.hpp"
#endif

namespace snde {

  recording_storage::recording_storage(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,void **basearray,size_t elementsize,snde_index base_index,unsigned typenum,snde_index nelem,std::shared_ptr<lockmanager> lockmgr,bool requires_locking_read,bool requires_locking_write,bool requires_locking_read_gpu,bool requires_locking_write_gpu,bool finalized) :
    recording_path(recording_path),
    recrevision(recrevision),
    originating_rss_unique_id(originating_rss_unique_id),
    id(id),
    _basearray(basearray),
    shiftedarray(nullptr),
    elementsize(elementsize),
    base_index(base_index),
    typenum(typenum),
    nelem(nelem),
    lockmgr(lockmgr),
    requires_locking_read(requires_locking_read),
    requires_locking_write(requires_locking_write),
    requires_locking_read_gpu(requires_locking_read_gpu),
    requires_locking_write_gpu(requires_locking_write_gpu),
    finalized(finalized)
    // Note: Called with array locks held

  {

  }

  std::shared_ptr<recording_storage> recording_storage::get_original_storage()
  {
    return shared_from_this();
  }

  
  recording_storage_simple::recording_storage_simple(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,size_t elementsize,unsigned typenum,snde_index nelem,std::shared_ptr<lockmanager> lockmgr,bool requires_locking_read,bool requires_locking_write,bool finalized,std::shared_ptr<memallocator> lowlevel_alloc,void *baseptr,void *allocated_baseptr) :
    recording_storage(recording_path,recrevision,originating_rss_unique_id,id,nullptr,elementsize,0,typenum,nelem,lockmgr,requires_locking_read,requires_locking_write,requires_locking_read,requires_locking_write,finalized),
    lowlevel_alloc(lowlevel_alloc),
    _baseptr(baseptr),
    _allocated_baseptr(allocated_baseptr)
  {
    _basearray = &_baseptr;
  }

  recording_storage_simple::~recording_storage_simple()
  {
    {
      //std::lock_guard<std::mutex> cmgr_lock(follower_cachemanagers_lock);
      // Don't need the lock because we are expiring!!!
      for (auto && cachemgr: follower_cachemanagers) {
	std::shared_ptr<cachemanager> cmgr_strong=cachemgr.lock();
	if (cmgr_strong) {
	  cmgr_strong->notify_storage_expiration(lockableaddr(),base_index,nelem);
	}
      }
    }
    
    // free allocation from recording_storage_manager_simple::allocate_recording()
    lowlevel_alloc->free(recording_path,recrevision,originating_rss_unique_id,id,_allocated_baseptr);
  }

  void *recording_storage_simple::dataaddr_or_null()
  {
    return shiftedarray;
  }

  void *recording_storage_simple::cur_dataaddr()
  {
    if (shiftedarray) {
      return shiftedarray;
    }
    return (void *)(((char *)(*_basearray)) + elementsize*base_index);
  }

  void **recording_storage_simple::lockableaddr()
  {
    return _basearray;
  }

  snde_index recording_storage_simple::lockablenelem()
  {
    return nelem;
  }

  std::shared_ptr<recording_storage> recording_storage_simple::obtain_nonmoving_copy_or_reference()
  {

    std::lock_guard<std::mutex> cache_holder(cache_lock);
    std::shared_ptr<recording_storage_reference> reference = weak_nonmoving_copy_or_reference.lock();

    if (!reference) {
      std::shared_ptr<nonmoving_copy_or_reference> ref = lowlevel_alloc->obtain_nonmoving_copy_or_reference(recording_path,recrevision,originating_rss_unique_id,id,_basearray,_baseptr,base_index*elementsize,nelem*elementsize);
      reference = std::make_shared<recording_storage_reference>(recording_path,recrevision,originating_rss_unique_id,id,nelem,shared_from_this(),ref,finalized);
      weak_nonmoving_copy_or_reference = reference; 
      if (finalized) {
	reference->finalized=true; // work around potential race conditions
      }
    }
    return reference;
  }

  void recording_storage_simple::mark_as_modified(std::shared_ptr<cachemanager> already_knows,snde_index pos, snde_index numelem,bool override_finalized_check /*=false*/)
  // pos and numelem are relative to __this_recording__
  {
    if (!override_finalized_check) {
      assert(!finalized);
    }
    
    std::set<std::weak_ptr<cachemanager>,std::owner_less<std::weak_ptr<cachemanager>>> fc_copy;
    {
      std::lock_guard<std::mutex> cmgr_lock(follower_cachemanagers_lock);
      fc_copy=follower_cachemanagers;
    }
    for (auto && cmgr: fc_copy) {
      std::shared_ptr<cachemanager> cmgr_strong=cmgr.lock();
      if (cmgr_strong) {
	if (cmgr_strong != already_knows) {
	  cmgr_strong->mark_as_invalid(lockableaddr(),base_index,pos,nelem);
	}
      }
    }
  }

  void recording_storage_simple::ready_notification()
  {
    // no-op because if you are overwriting preexisting data you should be calling mark_as_modified()
    // and you should write data in with the CPU BEFORE caching (why wouldn't you???)
  }

  void recording_storage_simple::mark_as_finalized()
  {
    std::lock_guard<std::mutex> cache_holder(cache_lock);
    finalized=true;
    
    std::shared_ptr<recording_storage_reference> reference = weak_nonmoving_copy_or_reference.lock();
    if (reference) {
      reference->finalized=true;
    }
  }

  void recording_storage_simple::add_follower_cachemanager(std::shared_ptr<cachemanager> cachemgr)
  {
    std::lock_guard<std::mutex> cmgr_lock(follower_cachemanagers_lock);
    follower_cachemanagers.emplace(cachemgr);
  }


  recording_storage_reference::recording_storage_reference(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,snde_index nelem,std::shared_ptr<recording_storage> orig,std::shared_ptr<nonmoving_copy_or_reference> ref,bool finalized) :
    recording_storage(recording_path,
		      recrevision,
		      originating_rss_unique_id,
		      id,nullptr,
		      orig->elementsize,
		      orig->base_index,orig->typenum,nelem,
		      orig->lockmgr,
		      ref->requires_locking_read, //  requires_locking_read: Derives from reference because it is the memallocator that knows if locking is required to access the underlying reference
		      ref->requires_locking_write, // requires_locking_write: : Derives from reference because it is the memallocator that knows if locking is required to write the underlying reference. Note that the underlying reference is NOT necessarily immutable: If the memallocator supports_nonmoving_reference() returns true, then the the reference may be created BEFORE the data is finalized
		      ref->requires_locking_read, // these are the _gpu values which for now aren't actually used because the OpenCL code always uses the original not the reference anyway (via get_original_storage())
		      ref->requires_locking_write,		      
		      finalized), // always finalized because it is immutable
    orig(orig),
    ref(ref)
  {
    
  }

  void *recording_storage_reference::dataaddr_or_null()
  {
    return ref->get_shiftedptr();
  }
  void *recording_storage_reference::cur_dataaddr()
  {
    return ref->get_shiftedptr();
  }

  void **recording_storage_reference::lockableaddr()
  {
    return orig->_basearray; // always lock original if needed
  }
  snde_index recording_storage_reference::lockablenelem()
  {
    return orig->nelem;
  }

  std::shared_ptr<recording_storage> recording_storage_reference::obtain_nonmoving_copy_or_reference()
  {
    // delegate to original storage, adding in our own offset
    //assert(ref->shift % elementsize == 0);
    //return orig->obtain_nonmoving_copy_or_reference(/*ref->offset/elementsize + offset_elements,*/);
    return shared_from_this();
  }

  void recording_storage_reference::mark_as_modified(std::shared_ptr<cachemanager> already_knows,snde_index pos, snde_index numelem, bool override_finalized_check /*=false */)
  {
    orig->mark_as_modified(already_knows,pos,numelem,override_finalized_check);
  }
  
  void recording_storage_reference::ready_notification()
  {
    orig->ready_notification();
  }

  void recording_storage_reference::mark_as_finalized()
  {
    orig->mark_as_finalized(); // delegate to original, which should mark us as finalized too
    
  }

  std::shared_ptr<recording_storage> recording_storage_reference::get_original_storage()
  {
    return orig;
  }

  
  void recording_storage_reference::add_follower_cachemanager(std::shared_ptr<cachemanager> cachemgr)
  {
    orig->add_follower_cachemanager(cachemgr);
  }


  recording_storage_manager_simple::recording_storage_manager_simple(std::shared_ptr<memallocator> lowlevel_alloc,std::shared_ptr<lockmanager> lockmgr,std::shared_ptr<allocator_alignment> alignment_requirements) :
    lowlevel_alloc(lowlevel_alloc),
    lockmgr(lockmgr),
    alignment_requirements(alignment_requirements)
  {

  }

  
  std::shared_ptr<recording_storage>
  recording_storage_manager_simple::allocate_recording(std::string recording_path, std::string array_name, // use "" for default behavior -- which is all that is supported anyway
						       uint64_t recrevision,
						       uint64_t originating_rss_unique_id,
						       size_t multiarray_index,
						       size_t elementsize,
						       unsigned typenum, // MET_...
						       snde_index nelem,
						       bool is_mutable) // returns (storage pointer,base_index); note that the recording_storage nelem may be different from what was requested.
  // must be thread-safe
  // NOTE: for recording_storage_manager_simple, the index into the multiarray and the memallocator
  // regionid are the same. But this is NOT true for the graphics_storage_manager
  // in recording_storage_manager_simple we freely cast back and forth because they are
  // both size_t under the hood
  {


    
    size_t alignment_extra=0;
    if (alignment_requirements) {
      alignment_extra=alignment_requirements->get_alignment();
    }
    
    void *allocated_baseptr = lowlevel_alloc->calloc(recording_path,recrevision,originating_rss_unique_id,(memallocator_regionid)multiarray_index,nelem*elementsize+alignment_extra,0);  // freed in destructor for recording_storage_simple
    // enforce alignment requirements
    void *baseptr = allocator_alignment::alignment_shift(allocated_baseptr,alignment_extra);

    if ((is_mutable || lowlevel_alloc->requires_locking_read || lowlevel_alloc->requires_locking_write) && !lockmgr) {
      throw snde_error("recording_storage_manager_simple::allocate_recording(): Mutable recordings or those using allocators that require locking require a lock manager");
    }

    std::shared_ptr<recording_storage_simple> retval = std::make_shared<recording_storage_simple>(recording_path,recrevision,originating_rss_unique_id,(memallocator_regionid)multiarray_index,elementsize,typenum,nelem,lockmgr,is_mutable || lowlevel_alloc->requires_locking_read,is_mutable || lowlevel_alloc->requires_locking_write,false,lowlevel_alloc,baseptr,allocated_baseptr);


    return retval;
  }
    

};
