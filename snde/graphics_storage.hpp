#ifndef SNDE_GRAPHICS_STORAGE_HPP
#define SNDE_GRAPHICS_STORAGE_HPP

#include "snde/arraymanager.hpp"
#include "snde/recstore_storage.hpp"
#include "snde/geometrydata.h"


namespace snde {

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


    std::shared_ptr<nonmoving_copy_or_reference> ref; // once ref is assigned we return the pointers from the reference instead of the main array.
    
    
    graphics_storage(std::string recording_path,uint64_t recrevision,memallocator_regionid id,void **basearray,size_t elementsize,snde_index base_index,unsigned typenum,snde_index nelem,bool requires_locking_read,bool requires_locking_write,bool finalized);
    graphics_storage(const graphics_storage &) = delete;  // CC and CAO are deleted because we don't anticipate needing them. 
    graphics_storage& operator=(const graphics_storage &) = delete; 
    virtual ~graphics_storage()=default; // virtual destructor so we can subclass

    virtual void *dataaddr_or_null(); // return pointer to recording base address pointer for memory access or nullptr if it should be accessed via lockableaddr() because it might yet move in the future. Has base_index already added in
    virtual void *cur_dataaddr(); // return pointer with shift built-in.
    virtual void **lockableaddr(); // return pointer to recording base address pointer for locking

    virtual std::shared_ptr<recording_storage> obtain_nonmoving_copy_or_reference(); // NOTE: The returned storage can only be trusted if (a) the originating recording is immutable, or (b) the originating recording is mutable but has not been changed since obtain_nonmoving_copy_or_reference() was called. i.e. can only be used as long as the originating recording is unchanged. Note that this is used only for getting a direct reference within a larger (perhaps mutable) allocation, such as space for a texture or mesh geometry. If you are just referencing a range of elements of a finalized waveofrm you can just reference the recording_storage shared pointer with a suitable base_index, stride array, and dimlen array. 

  };
  
  
  class graphics_storage_manager: public recording_storage_manager {
public:
    std::shared_ptr<arraymanager> manager; // array manager for the graphics arrays within
    snde_geometrydata geom; // actual graphics storage
    std::string graphics_recgroup_path; // path of the graphics recording group (channel group) this storage manager is managing 
    std::unordered_map<std::string,void **> arrayaddr_from_name;
    std::unordered_map<std::string,size_t> elemsize_from_name;
    
    std::unordered_map<std::string,memallocator_regionid> arrayid_from_name;
    
    
    graphics_storage_manager(const std::string &graphics_recgroup_path,std::shared_ptr<memallocator> memalloc,std::shared_ptr<allocator_alignment> alignment_requirements,std::shared_ptr<lockmanager> lockmngr,double tol);
    // Rule of 3
    graphics_storage_manager(const graphics_storage_manager &) = delete;  // CC and CAO are deleted because we don't anticipate needing them. 
    graphics_storage_manager& operator=(const graphics_storage_manager &) = delete; 
    virtual ~graphics_storage_manager(); // virtual destructor so we can subclass
    
    virtual std::shared_ptr<recording_storage> allocate_recording(std::string recording_path,std::string array_name, // use "" for default array
								  uint64_t recrevision,
								  size_t elementsize,
								  unsigned typenum, // MET_...
								  snde_index nelem,
								  bool is_mutable); // returns (storage pointer,base_index); note that the recording_storage nelem may be different from what was requested.
    
    

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
  
#endif // SNDE_GRAPHICS_STORAGE_HPP
