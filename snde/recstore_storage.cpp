#include "snde/recstore_storage.hpp"
#include "snde/allocator.hpp"

#ifdef _WIN32
// No win32 shared_memory_allocator yet
#else
#include "snde/shared_memory_allocator_posix.hpp"
#endif

namespace snde {

  recording_storage::recording_storage(std::string recording_path,uint64_t recrevision,memallocator_regionid id,void **basearray,size_t elementsize,snde_index base_index,unsigned typenum,snde_index nelem,bool requires_locking_read,bool requires_locking_write,bool finalized) :
    recording_path(recording_path),
    recrevision(recrevision),
    id(id),
    _basearray(basearray),
    shiftedarray(nullptr),
    elementsize(elementsize),
    base_index(base_index),
    typenum(typenum),
    nelem(nelem),
    requires_locking_read(requires_locking_read),
    requires_locking_write(requires_locking_write),
    finalized(finalized)
    // Note: Called with array locks held

  {

  }

  recording_storage_simple::recording_storage_simple(std::string recording_path,uint64_t recrevision,memallocator_regionid id,size_t elementsize,unsigned typenum,snde_index nelem,bool requires_locking_read,bool requires_locking_write,bool finalized,std::shared_ptr<memallocator> lowlevel_alloc,void *baseptr) :
    recording_storage(recording_path,recrevision,id,nullptr,elementsize,0,typenum,nelem,requires_locking_read,requires_locking_write,finalized),
    lowlevel_alloc(lowlevel_alloc),
    _baseptr(baseptr)
  {
    _basearray = &_baseptr;
  }

  recording_storage_simple::~recording_storage_simple()
  {
    // free allocation from recording_storage_manager_simple::allocate_recording()
    lowlevel_alloc->free(recording_path,recrevision,id,_baseptr);
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
    return (void *)(((char *)_basearray) + elementsize*base_index);
  }

  void **recording_storage_simple::lockableaddr()
  {
    return _basearray;
  }

  std::shared_ptr<recording_storage> recording_storage_simple::obtain_nonmoving_copy_or_reference()
  {
    //assert(orig_id==0);  // recording_storage_simple only has id's of 0
    std::shared_ptr<nonmoving_copy_or_reference> ref = lowlevel_alloc->obtain_nonmoving_copy_or_reference(recording_path,recrevision,id,_basearray,_baseptr,base_index*elementsize,nelem*elementsize);
    std::shared_ptr<recording_storage_reference> reference = std::make_shared<recording_storage_reference>(recording_path,recrevision,id,nelem,shared_from_this(),ref);

    return reference;
  }

  recording_storage_reference::recording_storage_reference(std::string recording_path,uint64_t recrevision,memallocator_regionid id,snde_index nelem,std::shared_ptr<recording_storage> orig,std::shared_ptr<nonmoving_copy_or_reference> ref) :
    recording_storage(recording_path,recrevision,id,nullptr,orig->elementsize,orig->base_index,orig->typenum,nelem,orig->requires_locking_read,orig->requires_locking_write,true), // always finalized because it is immutable
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

  std::shared_ptr<recording_storage> recording_storage_reference::obtain_nonmoving_copy_or_reference()
  {
    // delegate to original storage, adding in our own offset
    assert(ref->shift % elementsize == 0);
    return orig->obtain_nonmoving_copy_or_reference(/*ref->offset/elementsize + offset_elements,*/);
  }


  recording_storage_manager_simple::recording_storage_manager_simple(std::shared_ptr<memallocator> lowlevel_alloc,std::shared_ptr<allocator_alignment> alignment_requirements) :
    lowlevel_alloc(lowlevel_alloc),
    alignment_requirements(alignment_requirements)
  {

  }

  
  std::shared_ptr<recording_storage>
  recording_storage_manager_simple::allocate_recording(std::string recording_path, std::string array_name, // use "" for default behavior -- which is all that is supported anyway
						       uint64_t recrevision,
						       size_t elementsize,
						       unsigned typenum, // MET_...
						       snde_index nelem,
						       bool is_mutable) // returns (storage pointer,base_index); note that the recording_storage nelem may be different from what was requested.
  // must be thread-safe
  {
    
    size_t alignment_extra=0;
    if (alignment_requirements) {
      alignment_extra=alignment_requirements->get_alignment();
    }
    
    void *baseptr = lowlevel_alloc->calloc(recording_path,recrevision,0,nelem*elementsize+alignment_extra);  // freed in destructor for recording_storage_simple
    // enforce alignment requirements
    baseptr = allocator_alignment::alignment_shift(baseptr,alignment_extra);

    std::shared_ptr<recording_storage_simple> retval = std::make_shared<recording_storage_simple>(recording_path,recrevision,0,elementsize,typenum,nelem,is_mutable || lowlevel_alloc->requires_locking_read,is_mutable || lowlevel_alloc->requires_locking_write,false,lowlevel_alloc,baseptr);


    return retval;
  }
    

};
