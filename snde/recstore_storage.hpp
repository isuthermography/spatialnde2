#ifndef SNDE_RECSTORE_STORAGE_HPP
#define SNDE_RECSTORE_STORAGE_HPP

#include <memory>
#include <tuple>
#include <string>
#include <atomic>
#include <set>

#include "snde/snde_types.h"
#include "snde/geometry_types.h"

#include "snde/memallocator.hpp"
#include "snde/cached_recording.hpp" // for cachemanager

namespace snde {

  class cached_recording; // from cached_recording.hpp
  class cachemanager; // cached_recording.hpp
  class allocator_alignment; // from allocator.hpp
  class lockmanager;  // lockmanager.hpp
  class recording_storage_reference;
  
  class recording_storage: public std::enable_shared_from_this<recording_storage> {
    // recording storage locked through lockmanager, except
    // many parameters are immutable: 

    
    // elements and typenum, are immutable once created
    // nelem can only be changed when *_basearray is locked for write
    // finalized is immutable once published
    // _basearray is immutable once published, but *_basearray is
    // mutable for a mutable recording (must use locking following locking order)

    // interpreting this also requires the arraylayout, which is stored
    // in multi_ndarray_recording->layouts.
    // The relevant data is shadowed in
    // ((struct multi_ndarray_recording *)recording->info)->arrays[index]
  public:
    std::string recording_path; // immutable once constructed
    uint64_t recrevision;  // immutable once constructed
    uint64_t originating_rss_unique_id;
    memallocator_regionid id;  //immutable once constructed
    
    void **_basearray; // pointer to lockable address for recording array (lockable if recording is mutable or requires_locking_read or requires_locking_write). Don't use directly, get via lockableaddr() method
    std::atomic<void *> shiftedarray; // if not NULL, overrides *basearray. Includes base_index already added in. But basearray is still used for access when needed in the same address space as other data 
    size_t elementsize; // immutable once constructed
    snde_index base_index; //immutable once constructed 
    unsigned typenum; // MET_...  // immutable once constructed
    std::atomic<snde_index> nelem;  // immutable once constructed for immutable arrays. Might change for mutable arrays. 


    std::mutex cache_lock; // locks cache and possibly other items delegated by the implementation class such as nonmoving_pointer_or_reference.
    std::unordered_map<std::string,std::shared_ptr<cached_recording>> cache; // cache map, indexed by name of module doing caching (unique by get_cache_name()), to an abstract base class. Access to the unordered_map protected by cache_lock.


    
    std::shared_ptr<lockmanager> lockmgr; // may be nullptr if requires_locking_read and requires_locking_write are both false permanently. 
    std::atomic<bool> requires_locking_read;
    std::atomic<bool> requires_locking_write;

    // GPU semantics for locking may be different. This is motivated by OpenCL
    // and the graphics storage manager, where the specification says
    //  "Concurrent reading from, writing to and copying between both a buffer
    //   object and its sub-buffer object(s) is undefined. Concurrent reading
    //   from, writing to and copying between overlapping sub-buffer objects
    //   created with the same buffer object is undefined. Only reading from
    //   both a buffer object and its sub-buffer objects or reading from
    //   multiple overlapping sub-buffer objects is defined."
    // Thus we may not modify a parent buffer as a parent buffer ANYWHERE
    // if something is reading a sub-buffer that parent buffer. i.e.
    // we have to prevent writing to the parent buffer (a different non-
    // overlapping sub-buffer is probably OK) while the sub-buffer is in
    // use. The way we do this is by imposing a read lock requirement
    // on the sub-buffer which will prevent a write lock from being obtained
    // on the parent buffer until the reader is done with the sub-buffer. 
    std::atomic<bool> requires_locking_read_gpu;
    std::atomic<bool> requires_locking_write_gpu;
    
    std::atomic<bool> finalized; // if set, this is an immutable recording and its values have been set. Does NOT mean the data is valid indefinitely, as this could be a reference that loses validity at some point.
    
    // constructor
    recording_storage(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,void **basearray,size_t elementsize,snde_index base_index,unsigned typenum,snde_index nelem,std::shared_ptr<lockmanager> lockmgr,bool requires_locking_read,bool requires_locking_write,bool requires_locking_read_gpu,bool requires_locking_write_gpu,bool finalized);
    
    // Rule of 3
    recording_storage(const recording_storage &) = delete;  // CC and CAO are deleted because we don't anticipate needing them. 
    recording_storage& operator=(const recording_storage &) = delete; 
    virtual ~recording_storage()=default; // virtual destructor so we can subclass

    virtual void *dataaddr_or_null()=0; // return pointer to recording base address pointer for memory access or nullptr if it should be accessed via lockableaddr() because it might yet move in the future. Has base_index already added in
    virtual void *cur_dataaddr()=0; // return pointer with shift built-in.
    virtual void **lockableaddr()=0; // return pointer to recording base address pointer for locking
    virtual snde_index lockablenelem()=0;

    
    virtual std::shared_ptr<recording_storage> obtain_nonmoving_copy_or_reference()=0; // NOTE: The returned storage can only be trusted if (a) the originating recording is immutable, or (b) the originating recording is mutable but has not been changed since obtain_nonmoving_copy_or_reference() was called. i.e. can only be used as long as the originating recording is unchanged. Note that this is used only for getting a direct reference within a larger (perhaps mutable) allocation, such as space for a texture or mesh geometry. If you are just referencing a range of elements of a finalized waveform you can just reference the recording_storage shared pointer with a suitable base_index, stride array, and dimlen array. 


    // ***!!! Need a way to register and notify caches that the data in a mutable array has changed. ***!!!
    // ***!!! It would be nice to be able to mark a "rectangle" as invalid too !!!***
    virtual void mark_as_modified(std::shared_ptr<cachemanager> already_knows,snde_index pos, snde_index numelem, bool override_finalized_check=false)=0; // pos and numelem are relative to __this_recording__. override_finalized_check should only be set in extreme circumstances where a recording that is nominally finalized actually needs to be changed. This would be true e.g. for the snde_meshedpart structure when other recordings that need to be referenced in the struct snde_meshedpart become ready.  Conceptually those fields are actually part of the other recordings but they are stored as part of the meshedpart
    virtual void ready_notification()=0; // Sent by the recording when it is marked as ready. Used by some recording_storage_managers (e.g. graphics_storage) as an indicator that pending_modified data has probably been modified by the CPU and thus needs to be flushed out to cache managers. Note that multiple ready_ontifications on the same recording_storage are possible (but should be rare) if the storage is reused such as for a mutable recording or a later version with no data change uses the same underlying data store
    virtual void mark_as_finalized()=0;
    virtual std::shared_ptr<recording_storage> get_original_storage();
    
    virtual void add_follower_cachemanager(std::shared_ptr<cachemanager> cachemgr)=0; // gets mark_as_modified() and notify_storage_expiration() notifications
  };

  class recording_storage_simple: public recording_storage {
    // recording_storage_simple represents
    // the simple case of a single space used for the entire recording
    // This is as opposed to a reference into a shared space (e.g. in
    // a memory space used for 3D graphics)
    // that might be reallocated or similar.
    
  public:
    // lowlevel_alloc is thread safe
    // _baseptr is immutable once published
    
    std::shared_ptr<memallocator> lowlevel_alloc; // low-level allocator
    void *_baseptr; // this is what _basearray points at; access through superclass addr() method
    void *_allocated_baseptr; // this is what we allocated and what must be passed to lowlevel_alloc->free() May be different from _baseptr if the value returned from the allocator did not satisfy alignment requirements. 


    
    std::weak_ptr<recording_storage_reference> weak_nonmoving_copy_or_reference; // locked using recording_storage's cache lock

    std::mutex follower_cachemanagers_lock; // locks follower_cachemanagers
    std::set<std::weak_ptr<cachemanager>,std::owner_less<std::weak_ptr<cachemanager>>> follower_cachemanagers;

    
    // don't create this yourself, get it from recording_storage_manager_simple
    recording_storage_simple(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,size_t elementsize,unsigned typenum,snde_index nelem,std::shared_ptr<lockmanager> lockmgr,bool requires_locking_read,bool requires_locking_write,bool finalized,std::shared_ptr<memallocator> lowlevel_alloc,void *baseptr,void *allocated_baseptr);
    recording_storage_simple(const recording_storage_simple &) = delete;  // CC and CAO are deleted because we don't anticipate needing them. 
    recording_storage_simple& operator=(const recording_storage_simple &) = delete; 
    virtual ~recording_storage_simple(); // frees  _baseptr, notifies followers 
    virtual void *dataaddr_or_null();
    virtual void *cur_dataaddr();
    virtual void **lockableaddr();
    virtual snde_index lockablenelem();
    virtual std::shared_ptr<recording_storage> obtain_nonmoving_copy_or_reference();

    virtual void mark_as_modified(std::shared_ptr<cachemanager> already_knows,snde_index pos, snde_index numelem,bool override_finalized_check=false); // pos and numelem are relative to __this_recording__
    virtual void ready_notification();
    virtual void mark_as_finalized();
    
    virtual void add_follower_cachemanager(std::shared_ptr<cachemanager> cachemgr);
    
  };

  class recording_storage_reference: public recording_storage {
    // warning: referenced recordings are often, but not always, immutable.
    // if the memallocator supports_nomoving_reference() is true, then we can have nonmoving
    // references that refer to the same underlying data (implemented via the MMU so it is the
    // same memory pages mapped twice into the address space). These do not require immutability
    // orig shared_ptr is immutable once published; ref shared_ptr is immutable once published
  public:
    std::shared_ptr<recording_storage> orig; // low-level allocator
    std::shared_ptr<nonmoving_copy_or_reference> ref; 

    recording_storage_reference(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,snde_index nelem,std::shared_ptr<recording_storage> orig,std::shared_ptr<nonmoving_copy_or_reference> ref,bool finalized);
    virtual ~recording_storage_reference() = default; 
    virtual void *dataaddr_or_null();
    virtual void *cur_dataaddr();
    virtual void **lockableaddr();
    virtual snde_index lockablenelem();
    virtual std::shared_ptr<recording_storage> obtain_nonmoving_copy_or_reference();
    virtual void mark_as_modified(std::shared_ptr<cachemanager> already_knows,snde_index pos, snde_index numelem,bool override_finalized_check=false); // pos and numelem are relative to __this_recording__
    virtual void ready_notification();
    virtual void mark_as_finalized();
    virtual std::shared_ptr<recording_storage> get_original_storage();
    virtual void add_follower_cachemanager(std::shared_ptr<cachemanager> cachemgr);

  };

  class recording_storage_manager : public std::enable_shared_from_this<recording_storage_manager> {
  public:
    // allocate_recording method should be thread-safe
    
    recording_storage_manager() = default;
    
    // Rule of 3
    recording_storage_manager(const recording_storage_manager &) = delete;  // CC and CAO are deleted because we don't anticipate needing them. 
    recording_storage_manager& operator=(const recording_storage_manager &) = delete; 
    virtual ~recording_storage_manager() = default; // virtual destructor so we can subclass
    
    virtual std::shared_ptr<recording_storage>  allocate_recording(std::string recording_path,std::string array_name, // use "" for default array
								   uint64_t recrevision,
								   uint64_t originating_rss_unique_id,
								   size_t multiarray_index,
								   size_t elementsize,
								   unsigned typenum, // MET_...
								   snde_index nelem,
								   bool is_mutable)=0; // returns (storage pointer,base_index); note that the recording_storage nelem may be different from what was requested.
    
  };


  class recording_storage_manager_simple: public recording_storage_manager {
    // allocate_recording method should be thread-safe
  public:
    std::shared_ptr<memallocator> lowlevel_alloc;
    std::shared_ptr<lockmanager> lockmgr; 
    std::shared_ptr<allocator_alignment> alignment_requirements;
    
    recording_storage_manager_simple(std::shared_ptr<memallocator> lowlevel_alloc,std::shared_ptr<lockmanager> lockmgr,std::shared_ptr<allocator_alignment> alignment_requirements);
    virtual ~recording_storage_manager_simple() = default; 
    virtual std::shared_ptr<recording_storage> allocate_recording(std::string recording_path,std::string array_name, // use "" for default array within recording
								  uint64_t recrevision,
								  uint64_t originating_rss_unique_id,
								  size_t multiarray_index,
								  size_t elementsize,
								  unsigned typenum, // MET_...
								  snde_index nelem,
								  bool is_mutable); // returns (storage pointer,base_index); note that the recording_storage nelem may be different from what was requested.
    
    
  };

  


  
};

#endif // SNDE_RECSTORE_STORAGE_HPP
