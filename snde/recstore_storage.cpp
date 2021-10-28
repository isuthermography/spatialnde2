#include "snde/recstore_storage.hpp"

#ifdef _WIN32
// No win32 shared_memory_allocator yet
#else
#include "snde/shared_memory_allocator_posix.hpp"
#endif

namespace snde {

  recording_storage::recording_storage(std::string recording_path,uint64_t recrevision,memallocator_regionid id,void **basearray,size_t elementsize,unsigned typenum,snde_index nelem,bool finalized) :
    recording_path(recording_path),
    recrevision(recrevision),
    id(id),
    _basearray(basearray),
    elementsize(elementsize),
    typenum(typenum),
    nelem(nelem),
    finalized(finalized)
  {

  }

  recording_storage_simple::recording_storage_simple(std::string recording_path,uint64_t recrevision,memallocator_regionid id,size_t elementsize,unsigned typenum,snde_index nelem,bool finalized,std::shared_ptr<memallocator> lowlevel_alloc,void *baseptr) :
    recording_storage(recording_path,recrevision,id,nullptr,elementsize,typenum,nelem,finalized),
    lowlevel_alloc(lowlevel_alloc),
    _baseptr(baseptr)
  {
    _basearray = &_baseptr;
  }

  void **recording_storage_simple::addr()
  {
    return _basearray;
  }

  std::shared_ptr<recording_storage> recording_storage_simple::obtain_nonmoving_copy_or_reference(snde_index offset_elements, snde_index length_elements)
  {
    //assert(orig_id==0);  // recording_storage_simple only has id's of 0
    std::shared_ptr<nonmoving_copy_or_reference> ref = lowlevel_alloc->obtain_nonmoving_copy_or_reference(recording_path,recrevision,id,_baseptr,offset_elements*elementsize,length_elements*elementsize);
    std::shared_ptr<recording_storage_reference> reference = std::make_shared<recording_storage_reference>(recording_path,recrevision,id,length_elements,shared_from_this(),ref);

    return reference;
  }

  recording_storage_reference::recording_storage_reference(std::string recording_path,uint64_t recrevision,memallocator_regionid id,snde_index nelem,std::shared_ptr<recording_storage> orig,std::shared_ptr<nonmoving_copy_or_reference> ref) :
    recording_storage(recording_path,recrevision,id,nullptr,orig->elementsize,orig->typenum,nelem,true), // always finalized because it is immutable
    orig(orig),
    ref(ref)
  {
    
  }

  void **recording_storage_reference::addr()
  {
    return ref->get_basearray();
  }
  
  std::shared_ptr<recording_storage> recording_storage_reference::obtain_nonmoving_copy_or_reference(snde_index offset_elements, snde_index length_elements)
  {
    // delegate to original storage, adding in our own offset
    assert(ref->offset % elementsize == 0);
    return orig->obtain_nonmoving_copy_or_reference(ref->offset/elementsize + offset_elements,length_elements);
  }


  recording_storage_manager_simple::recording_storage_manager_simple(std::shared_ptr<memallocator> lowlevel_alloc) :
    lowlevel_alloc(lowlevel_alloc)
  {

  }

  
  std::tuple<std::shared_ptr<recording_storage>,snde_index>
  recording_storage_manager_simple::allocate_recording(std::string recording_path, std::string array_name, // use "" for default behavior -- which is all that is supported anyway
							   uint64_t recrevision,
							   size_t elementsize,
							   unsigned typenum, // MET_...
							   snde_index nelem) // returns (storage pointer,base_index); note that the recording_storage nelem may be different from what was requested.
  // must be thread-safe
  {
    
    void *baseptr = lowlevel_alloc->calloc(recording_path,recrevision,0,nelem*elementsize);

    std::shared_ptr<recording_storage_simple> retval = std::make_shared<recording_storage_simple>(recording_path,recrevision,0,elementsize,typenum,nelem,false,lowlevel_alloc,baseptr);

    return std::make_tuple<std::shared_ptr<recording_storage>,snde_index>(retval,0);
  }
    

};
