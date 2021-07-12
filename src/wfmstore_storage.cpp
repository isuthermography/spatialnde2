#include "wfmstore_storage.hpp"

namespace snde {

  waveform_storage::waveform_storage(void **basearray,size_t elementsize,unsigned typenum,snde_index nelem,bool finalized) :
    _basearray(basearray),
    elementsize(elementsize),
    typenum(typenum),
    nelem(nelem),
    finalized(finalized)
  {

  }

  waveform_storage_simple::waveform_storage_simple(size_t elementsize,unsigned typenum,snde_index nelem,bool finalized,std::shared_ptr<memallocator> lowlevel_alloc,void *baseptr) :
    waveform_storage(nullptr,elementsize,typenum,nelem,finalized),
    lowlevel_alloc(lowlevel_alloc),
    _baseptr(baseptr)
  {
    basearray = &_baseptr;
  }

  virtual void *waveform_storage_simple::addr()
  {
    return basearray;
  }

  virtual std::shared_ptr<waveform_storage> waveform_storage_simple::obtain_nonmoving_copy_or_reference(snde_index offset_elements, snde_index length_elements)
  {
    std::shared_ptr<nonmoving_copy_or_reference> ref = lowlevel_alloc->obtain_nonmoving_copy_or_reference(0,_baseptr,offset_elements*elementsize,length_elements*elementsize);
    std::shared_ptr<waveform_storage_reference> reference = std::make_shared<waveform_storage_reference>(length_elements,shared_from_this(),ref);

    return reference;
  }

  waveform_storage_reference::waveform_storage_reference(snde_index nelem,std::shared_ptr<waveform_storage> orig,std::shared_ptr<nonmoving_copy_or_reference> ref) :
    waveform_storage(nullptr,orig->elementsize,orig->typenum,nelem,true), // always finalized because it is immutable
    orig(orig),
    ref(ref)
  {
    
  }

  virtual void *waveform_storage_reference::addr()
  {
    return ref->get_ptr();
  }
  
  virtual std::shared_ptr<waveform_storage> waveform_storage_reference::obtain_nonmoving_copy_or_reference(snde_index offset_elements, snde_index length_elements)
  {
    // delegate to original storage, adding in our own offset
    assert(ref->offset % elementsize == 0);
    return orig->obtain_nonmoving_copy_or_reference(ref->offset/elementsize + offset_elements,length_elements);
  }
  
  
  virtual std::tuple<std::shared_ptr<waveform_storage>,snde_index>
  waveform_storage_manager_simple_shmem::allocate_waveform(std::string waveform_path,
							   uint64_t wfmrevision,
							   size_t elementsize,
							   unsigned typenum, // MET_...
							   snde_index nelem) // returns (storage pointer,base_index); note that the waveform_storage nelem may be different from what was requested.
  // must be thread-safe
  {
#ifdef _WIN32
#warning No shared memory allocator available for Win32 yet. Using regular memory instead
    std::shared_ptr<memallocator> lowlevel_alloc=std::make_shared<cmemallocator>();
    
#else
    std::shared_ptr<memallocator> lowlevel_alloc=std::make_shared<shared_memory_allocator_posix>(waveform_path,wfmrevision);
#endif
    
    void *baseptr = shm_allocator->calloc(0,nelem*elementsize);

    std::shared_ptr<waveform_storage_simple> retval = std::make_shared<waveform_storage_simple>(elementsize,typenum,nelem,lowlevel_alloc,baseptr);

    return std::make_tuple<std::shared_ptr<waveform_storage>,snde_index>(retval,0);
  }
    

};
