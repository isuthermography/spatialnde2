namespace snde {
  // ***!!! Need to address thread safety of waveform storage
  
  class waveform_storage: public std::enable_shared_from_this<waveform_storage> {
    // elementsize, typenum, and nelem are immutable once created
    // finalized is immutable once published
    // _basearray is immutable once published, but *_basearray is
    // mutable for a mutable waveform (must use locking following locking order)
    
    void **_basearray; // pointer to lockable address for waveform array (lockability if waveform is mutable)
    size_t elementsize;
    unsigned typenum; // MET_...
    snde_index nelem;
    bool finalized; // if set, this is an immutable waveform and its values have been set. Does NOT mean the data is valid indefinitely, as this could be a reference that loses validity at some point. 
    
    // constructor
    waveform_storage(void **basearray,size_t elementsize,unsigned typenum,snde_index nelem,bool finalized);
    
    // Rule of 3
    waveform_storage(const waveform_storage &) = delete;  // CC and CAO are deleted because we don't anticipate needing them. 
    waveform_storage& operator=(const waveform_storage &) = delete; 
    virtual ~waveform_storage()=default; // virtual destructor so we can subclass

    virtual void *addr()=0; // return waveform base address
    virtual std::shared_ptr<waveform_storage> obtain_nonmoving_copy_or_reference(size_t offset, size_t length)=0; // NOTE: The returned storage can only be trusted if (a) the originating waveform is immutable, or (b) the originating waveform is mutable but has not been changed since obtain_nonmoving_copy_or_reference() was called. i.e. can only be used as long as the originating waveform is unchanged. 
    
  };

  class waveform_storage_simple: public waveform_storage {
  public:
    // lowlevel_alloc is thread safe
    // _baseptr is immutable once published
    
    std::shared_ptr<memallocator> lowlevel_alloc; // low-level allocator
    void *_baseptr; // this is what _basearray points at; access through superclass addr() method

    waveform_storage_simple(size_t elementsize,unsigned typenum,snde_index nelem,bool finalized,std::shared_ptr<memallocator> lowlevel_alloc,void *baseptr);
    virtual ~waveform_storage_simple() = default; // _baseptr contents freed when all references to lowlevel_alloc go away
    virtual void *addr();
    virtual std::shared_ptr<waveform_storage> obtain_nonmoving_copy_or_reference(size_t offset, size_t length);
  };

  class waveform_storage_reference: public waveform_storage {
    // warning: referenced waveforms are always immutable and therefore cannot be explicitly locked.
    // orig shared_ptr is immutable once published; ref shared_ptr is immutable once published
  public:
    std::shared_ptr<waveform_storage> orig; // low-level allocator
    std::shared_ptr<nonmoving_copy_or_reference> ref; 

    waveform_storage_reference(snde_index nelem,std::shared_ptr<waveform_storage> orig,std::shared_ptr<nonmoving_copy_or_reference> ref);
    virtual ~waveform_storage_reference() = default; 
    virtual void *addr();
    virtual std::shared_ptr<waveform_storage> obtain_nonmoving_copy_or_reference(size_t offset, size_t length);
  };

  class waveform_storage_manager {
  public:
    // allocate_waveform method should be thread-safe
    
    waveform_storage_manager() = default;
    
    // Rule of 3
    waveform_storage_manager(const waveform_storage_manager &) = delete;  // CC and CAO are deleted because we don't anticipate needing them. 
    waveform_storage_manager& operator=(const waveform_storage_manager &) = delete; 
    virtual ~waveform_storage_manager() = default; // virtual destructor so we can subclass
    
    virtual std::tuple<std::shared_ptr<waveform_storage>,snde_index> allocate_waveform(std::string waveform_path,
										       uint64_t wfmrevision,
										       size_t elementsize,
										       unsigned typenum, // MET_...
										       snde_index nelem)=0; // returns (storage pointer,base_index); note that the waveform_storage nelem may be different from what was requested.
    
  };


  class waveform_storage_manager_shmem: public waveform_storage_manager {
    // allocate_waveform method should be thread-safe

    waveform_storage_manager_shmem() = default;
    virtual ~waveform_storage_manager_shmem() = default; 
    virtual std::tuple<std::shared_ptr<waveform_storage>,snde_index> allocate_waveform(std::string waveform_path,
										       uint64_t wfmrevision,
										       size_t elementsize,
										       unsigned typenum, // MET_...
										       snde_index nelem); // returns (storage pointer,base_index); note that the waveform_storage nelem may be different from what was requested.
    
    
  };

  


  
};
