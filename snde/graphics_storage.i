%shared_ptr(snde::graphics_storage);
snde_rawaccessible(snde::graphics_storage);
%shared_ptr(snde::graphics_storage_manager);
snde_rawaccessible(snde::graphics_storage_manager);


%{
  #include "snde/graphics_storage.hpp"
%}



namespace snde {

  class graphics_storage_manager; // forward reference

  bool snde_doubleprec_coords(); // returns true if SNDE_DOUBLEPREC_COORDS is defined
  
  class graphics_storage: public recording_storage {
  public:
    // inherited from class recording_storage:
    //std::string recording_path;
    //uint64_t recrevision;
    //memallocator_regionid id;
    
    //void **_basearray; // pointer to lockable address for recording array (lockability if recording is mutable)
    //size_t elementsize;
    //snde_index base_index;
    //unsigned typenum; // MET_...
    //snde_index nelem;

    //snde_bool requires_locking_read;
    //snde_bool requires_locking_write;

    
    
    //bool finalized; // if set, this is an immutable recording and its values hav

    std::shared_ptr<arraymanager> manager; // array manager for the graphics arrays within. Pointer is immutable once constructed
    std::shared_ptr<memallocator> memalloc; // pointer is immutable once constructed
    
    std::shared_ptr<nonmoving_copy_or_reference> _ref; // atomic shared pointer. Access with ref(). once ref is assigned we return the pointers from the reference instead of the main array. Immutable once assigned. This is used for immutable data arrays within the graphics storage, where once they are fully written we create a nonmoving copy or reference (by a separate mapping of the shared memory object) that will survive and keep its location even if the main array is reallocated and gets a new address. Theoretically could be used for mutable data as well, except for the potential for mutable data to change size.

    std::weak_ptr<recording_storage_reference> weak_nonmoving_copy_or_reference; // really points indirectly at _ref but we need the separate structure ponter here so we can mark it as finalized when appropriate. Locked by the recording_storage base class's cache_lock


    std::weak_ptr<graphics_storage_manager> graphman; 

    std::shared_ptr<graphics_storage> leader_storage; // If this object (current object, not the leader_storage object) is a follower array, then the leader_storage pointer is non-null and points to the graphics_storage of the corresponding leader array. We keep the shared pointer to prevent the leader from expiring and us losing our memory. If this object (current object, not the leader_storage object) is a leader array, then the leader_storage pointer is null and we own the storage and therefore will free it via the arraymanager in our destructor
    
    // Note: follower_cachemanagers have nothing to do with leader/follower arrays. Rather "follower" here
    // just refers to the cachemanager having this graphics_storage in its cache.

    // NOTE: We don't have our own follower_cachemanagers -- we just delegate to the graphics_storage_manager 
    //std::mutex follower_cachemanagers_lock; 
    //std::set<std::weak_ptr<cachemanager>,std::owner_less<std::weak_ptr<cachemanager>>> follower_cachemanagers;
    

    // ***!!!! Conceptually creating a graphics_storage should pass ownership
    // of the particular array zone, so it is automatically free'd.
    // ***!!! But what about follower arrays?
    graphics_storage(std::shared_ptr<graphics_storage_manager> graphman,std::shared_ptr<arraymanager> manager,std::shared_ptr<memallocator> memalloc,std::shared_ptr<graphics_storage> leader_storage,std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,void **basearray,size_t elementsize,snde_index base_index,unsigned typenum,snde_index nelem,bool requires_locking_read,bool requires_locking_write,bool finalized); // if leader_storage is nullptr, then the storage onwership is passed to the new object, which will free it. 
    graphics_storage(const graphics_storage &) = delete;  // CC and CAO are deleted because we don't anticipate needing them. 
    graphics_storage& operator=(const graphics_storage &) = delete; 
    virtual ~graphics_storage(); // virtual destructor so we can subclass. Notifies follower cachemanagers, even though openclcachemanager doesn't actuall care

    virtual void *dataaddr_or_null(); // return pointer to recording base address pointer for memory access or nullptr if it should be accessed via lockableaddr() because it might yet move in the future. Has base_index already added in
    virtual void *cur_dataaddr(); // return pointer with shift built-in.
    virtual void **lockableaddr(); // return pointer to recording base address pointer for locking
    virtual snde_index lockablenelem(); // return pointer to recording base address pointer for locking

    virtual std::shared_ptr<recording_storage> obtain_nonmoving_copy_or_reference(); // NOTE: The returned storage can only be trusted if (a) the originating recording is immutable, or (b) the originating recording is mutable but has not been changed since obtain_nonmoving_copy_or_reference() was called. i.e. can only be used as long as the originating recording is unchanged. Note that this is used only for getting a direct reference within a larger (perhaps mutable) allocation, such as space for a texture or mesh geometry. If you are just referencing a range of elements of a finalized waveofrm you can just reference the recording_storage shared pointer with a suitable base_index, stride array, and dimlen array.

    virtual void mark_as_modified(std::shared_ptr<cachemanager> already_knows,snde_index pos, snde_index numelem,bool override_finalized_check=false); // pos and numelem are relative to __this_recording__
    virtual void ready_notification();
    virtual void mark_as_finalized();
    virtual void add_follower_cachemanager(std::shared_ptr<cachemanager> cachemgr);
    

    
    virtual std::shared_ptr<nonmoving_copy_or_reference> ref();
    virtual void assign_ref(std::shared_ptr<nonmoving_copy_or_reference> ref);

  };
  
  
  class graphics_storage_manager: public recording_storage_manager {
public:
    std::shared_ptr<arraymanager> manager; // array manager for the graphics arrays within. Immutable once constructed
    snde_geometrydata geom; // actual graphics storage. Entries locked by their array locks
    std::string graphics_recgroup_path; // path of the graphics recording group (channel group) this storage manager is managing 
    std::unordered_map<std::string,void **> arrayaddr_from_name; // immutable once constructed
    std::unordered_map<void **,std::string> name_from_arrayaddr; // immutable once constructed
    std::unordered_map<std::string,size_t> elemsize_from_name; // immutable once constructed
    std::unordered_map<std::string,unsigned> typenum_from_name; // immutable once constructed
    
    std::unordered_map<std::string,memallocator_regionid> arrayid_from_name; // immutable once constructed

    std::unordered_map<std::string,std::shared_ptr<std::function<void(snde_index)>>> pool_realloc_callbacks; // indexed by names corresponding to leader arrays
    
    //std::mutex follower_cachemanagers_lock; // locks updates to _follower_cachemanagers
    //std::shared_ptr<std::set<std::weak_ptr<cachemanager>,std::owner_less<std::weak_ptr<cachemanager>>>> _follower_cachemanagers; // atomic shared pointer: Access with follower_cachemanagers()

    uint64_t base_rss_unique_id; // initialized from rss_get_unique() in constructor; helps prevent collsions in shared memory object naming

    std::size_t maxaddressbytes;
    
    graphics_storage_manager(const std::string &graphics_recgroup_path,std::shared_ptr<memallocator> memalloc,std::shared_ptr<allocator_alignment> alignment_requirements,std::shared_ptr<lockmanager> lockmngr,double tol,std::size_t maxaddressbytes/*=2000000000*/);    
    graphics_storage_manager(const std::string &graphics_recgroup_path,std::shared_ptr<memallocator> memalloc,std::shared_ptr<allocator_alignment> alignment_requirements,std::shared_ptr<lockmanager> lockmngr,double tol/*,std::size_t maxaddressbytes=2000000000*/);
    
    // Rule of 3
    graphics_storage_manager(const graphics_storage_manager &) = delete;  // CC and CAO are deleted because we don't anticipate needing them. 
    graphics_storage_manager& operator=(const graphics_storage_manager &) = delete; 
    virtual ~graphics_storage_manager(); // virtual destructor so we can subclass

    std::shared_ptr<graphics_storage> storage_from_allocation(std::string recording_path,
							      std::shared_ptr<graphics_storage> leader_storage, // nullptr if we are creating a leader
							      std::string array_name,
							      uint64_t recrevision,
							      uint64_t originating_rss_unique_id,
							      snde_index base_index, // as allocated for leader_array
							      size_t elementsize,
							      unsigned typenum, // MET_...
							      snde_index nelem,
							      bool is_mutable=false);

    //  math funcs use this to  grab space in a follower array
    // NOTE: This doesn't currently prevent two math functions from
    // grabbing the same follower array -- which could lead to corruption
    virtual std::shared_ptr<recording_storage> get_follower_storage(std::string recording_path,
								    std::shared_ptr<graphics_storage> leader_storage,
								    std::string follower_array_name,
								    uint64_t recrevision,
								    uint64_t originating_rss_unique_id, // graphics_storage does not ultimately use this for anything but it passes it around
								    snde_index base_index, // as allocated for leader_array
								    size_t elementsize,
								    unsigned typenum, // MET_...
								    snde_index nelem,
								    bool is_mutable);
    
    
    virtual std::shared_ptr<recording_storage> allocate_recording_lockprocess(std::string recording_path,std::string array_name, // use "" for default array
									      uint64_t recrevision,
									      uint64_t originating_rss_unique_id,
									      size_t elementsize,
									      unsigned typenum, // MET_...
									      snde_index nelem,
									      bool is_mutable,
									      std::shared_ptr<lockingprocess> lockprocess,
									      std::shared_ptr<lockholder> holder); // returns (storage pointer,base_index); note that the recording_storage nelem may be different from what was requested.

    
    virtual std::shared_ptr<recording_storage> allocate_recording(std::string recording_path,std::string array_name, // use "" for default array
								  uint64_t recrevision,
								  uint64_t originating_rss_unique_id,
								  size_t multiarray_index,
								  size_t elementsize,
								  unsigned typenum, // MET_...
								  snde_index nelem,
								  bool is_mutable); // returns (storage pointer,base_index); note that the recording_storage nelem may be different from what was requested.

    inline std::shared_ptr<std::set<std::weak_ptr<cachemanager>,std::owner_less<std::weak_ptr<cachemanager>>>> follower_cachemanagers();
    
    
    virtual void add_follower_cachemanager(std::shared_ptr<cachemanager> cachemgr);
    
    virtual void mark_as_modified(std::shared_ptr<cachemanager> already_knows,void **arrayptr,snde_index pos,snde_index numelem);

    // internal use only; defined at the top of graphics_storage.cpp
  private:
    template <typename L,typename T>
    void add_arrays_given_sizes(memallocator_regionid *nextid,const std::set<snde_index> &elemsizes,L **leaderptr,T **arrayptr,const std::string &arrayname);
    template <typename L,typename T,typename... Args>
    void add_arrays_given_sizes(memallocator_regionid *nextid,const std::set<snde_index> &elemsizes,L **leaderptr,T **arrayptr,const std::string &arrayname,Args... args);

    template <typename L,typename... Args>
    void add_grouped_arrays(memallocator_regionid *nextid,L **leaderptr,const std::string &leadername,Args... args);
    
    
  };
  
};
