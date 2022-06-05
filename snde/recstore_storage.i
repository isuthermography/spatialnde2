%shared_ptr(snde::recording_storage)
%shared_ptr(snde::recording_storage_simple)
  %shared_ptr(snde::recording_storage_reference)
  %shared_ptr(snde::recording_storage_manager)
  %shared_ptr(snde::recording_storage_manager_simple)


%{

#include "recstore_storage.hpp"
  %}
namespace snde {
  class cached_recording; // from cached_recording.hpp
  class cachemanager; // cached_recording.hpp
  class allocator_alignment; // from allocator.hpp
  class lockmanager;  // lockmanager.hpp

  
  class recording_storage /*: public std::enable_shared_from_this<recording_storage> */ {
    // recording storage locked through lockmanager, except
    // many parameters are immutable: 

    
    // elements and typenum, are immutable once created
    // nelem can only be changed when *_basearray is locked for write
    // finalized is immutable once published
    // _basearray is immutable once published, but *_basearray is
    // mutable for a mutable recording (must use locking following locking order)
  public:
    std::string recording_path;
    uint64_t recrevision;
    uint64_t originating_rss_unique_id;
    memallocator_regionid id;

    void **_basearray; // pointer to lockable address for recording array (lockability if recording is mutable)
    void *shiftedarray; // if not NULL, overrides *basearray. Includes base_index already added in.
    size_t elementsize;
    snde_index base_index;
    unsigned typenum; // MET_...
    snde_index nelem;

    std::shared_ptr<lockmanager> lockmgr; // may be nullptr if requires_locking_read and requires_locking_write are both false. 
    snde_bool requires_locking_read;
    snde_bool requires_locking_write;
    bool finalized; // if set, this is an immutable recording and its values have been set. Does NOT mean the data is valid indefinitely, as this could be a reference that loses validity at some point. 
    
    // constructor
    recording_storage(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,void **basearray,size_t elementsize,snde_index base_index,unsigned typenum,snde_index nelem,std::shared_ptr<lockmanager> lockmgr,bool requires_locking_read,bool requires_locking_write,bool finalized);
    
    // Rule of 3
    recording_storage(const recording_storage &) = delete;  // CC and CAO are deleted because we don't anticipate needing them. 
    recording_storage& operator=(const recording_storage &) = delete; 
    virtual ~recording_storage()=default; // virtual destructor so we can subclass

    virtual void *dataaddr_or_null()=0; // return pointer to recording base address pointer for memory access or nullptr if it should be accessed via lockableaddr() because it might yet move in the future. Has base_index already added in
    virtual void *cur_dataaddr()=0; // return pointer with shift built-in.
    virtual void **lockableaddr()=0; // return pointer to recording base address pointer for locking
    virtual snde_index lockablenelem()=0;

    virtual std::shared_ptr<recording_storage> obtain_nonmoving_copy_or_reference()=0; // NOTE: The returned storage can only be trusted if (a) the originating recording is immutable, or (b) the originating recording is mutable but has not been changed since obtain_nonmoving_copy_or_reference() was called. i.e. can only be used as long as the originating recording is unchanged. Note that this is used only for getting a direct reference within a larger (perhaps mutable) allocation, such as space for a texture or mesh geometry. If you are just referencing a range of elements of a finalized waveofrm you can just reference the recording_storage shared pointer with a suitable base_index, stride array, and dimlen array. 

    virtual void mark_as_modified(std::shared_ptr<cachemanager> already_knows,snde_index pos, snde_index numelem,bool override_finalized_check=false)=0; // pos and numelem are relative to __this_recording__
    
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

    // don't create this yourself, get it from recording_storage_manager_simple
    recording_storage_simple(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,size_t elementsize,unsigned typenum,snde_index nelem,std::shared_ptr<lockmanager> lockmgr,bool requires_locking_read,bool requires_locking_write,bool finalized,std::shared_ptr<memallocator> lowlevel_alloc,void *baseptr,void *allocated_baseptr);
    recording_storage_simple(const recording_storage_simple &) = delete;  // CC and CAO are deleted because we don't anticipate needing them. 
    recording_storage_simple& operator=(const recording_storage_simple &) = delete; 
    virtual ~recording_storage_simple(); // frees  _baseptr 
    virtual void *dataaddr_or_null();
    virtual void *cur_dataaddr();
    virtual void **lockableaddr();
    virtual snde_index lockablenelem();
    virtual std::shared_ptr<recording_storage> obtain_nonmoving_copy_or_reference();
    virtual void mark_as_modified(std::shared_ptr<cachemanager> already_knows,snde_index pos, snde_index numelem,bool override_finalized_check=false); // pos and numelem are relative to __this_recording__

    //virtual void mark_as_invalid(std::shared_ptr<cachemanager> already_knows,snde_index pos, snde_index numelem); // pos and numelem are relative to __this_recording__
    //virtual void add_follower_cachemanager(std::shared_ptr<cachemanager> cachemgr);

  };

  class recording_storage_reference: public recording_storage {
    // warning: referenced recordings are always immutable and therefore cannot be explicitly locked.
    // orig shared_ptr is immutable once published; ref shared_ptr is immutable once published
  public:
    std::shared_ptr<recording_storage> orig; // low-level allocator
    std::shared_ptr<nonmoving_copy_or_reference> ref; 

    recording_storage_reference(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,snde_index nelem,std::shared_ptr<recording_storage> orig,std::shared_ptr<nonmoving_copy_or_reference> ref);
    virtual ~recording_storage_reference() = default; 
    virtual void *dataaddr_or_null();
    virtual void *cur_dataaddr();
    virtual void **lockableaddr();
    virtual std::shared_ptr<recording_storage> obtain_nonmoving_copy_or_reference();
    virtual void mark_as_modified(std::shared_ptr<cachemanager> already_knows,snde_index pos, snde_index numelem,bool override_finalized_check=false); // pos and numelem are relative to __this_recording__
  };

  class recording_storage_manager {
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

