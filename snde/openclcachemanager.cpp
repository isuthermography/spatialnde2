
#include "snde/openclcachemanager.hpp"
#include "snde/snde_error.hpp"

extern "C" void snde_opencl_callback(cl_event c_event, cl_int event_command_exec_status, void *user_data)
{
  // NOTE: Should pass the cl::Event (object, NOT a reference) as a lambda capture to ensure it 
  // stays valid through the call. The cl::Event object created below will hold it in place
  // until this function returns even after the delete below. 
  std::function<void(cl::Event,cl_int)> *function_ptr=(std::function<void(cl::Event,cl_int)> *)user_data;

  cl::Event CppEvent;
  try {
      CppEvent=cl::Event(c_event, true);
  } catch (const cl::Error &e) {
      snde::snde_warning("OpenCL error: %s (%d)", e.what(),e.err());
      assert(0);
  }
  (*function_ptr)(CppEvent,event_command_exec_status);
  
  delete function_ptr;
}

namespace snde {

  opencldirtyregion::opencldirtyregion(snde_index regionstart,snde_index regionend,std::shared_ptr<recording_storage> owning_storage) :
    regionstart(regionstart),
    regionend(regionend),
    owning_storage(owning_storage),
    FlushDoneEvent(cl::Event()), // initialized to null
    FlushDoneEventComplete(false)
  {
    //fprintf(stderr,"Create opencldirtyregion(%d,%d)\n",regionstart,regionend);
  }
  
  bool opencldirtyregion::attempt_merge(opencldirtyregion &later)
  {
    return false; // ***!!! Should probably implement this
  }

  std::shared_ptr<opencldirtyregion> opencldirtyregion::sp_breakup(snde_index breakpoint,std::shared_ptr<recording_storage> storage)
  /* breakup method ends this region at breakpoint and returns
     a new region starting at from breakpoint to the prior end */
  {
    // note that it will be an error to do a breakup while there
    // are flushdoneevents pending
    std::shared_ptr<opencldirtyregion> newregion;
    
    //fprintf(stderr,"Create opencldirtyregion.sp_breakup(%d,%d)->(%d,%d)\n",regionstart,regionend,regionstart,breakpoint);
    assert(!FlushDoneEvent.get());
    assert(breakpoint > regionstart && breakpoint < regionend);
    
    newregion=std::make_shared<opencldirtyregion>(breakpoint,regionend,storage);
    this->regionend=breakpoint;
    
    return newregion;
  }



  openclregion::openclregion(snde_index regionstart,snde_index regionend)
  {
    this->regionstart=regionstart;
    this->regionend=regionend;

    
    fill_event=cl::Event();  // NULL;
  }

  bool openclregion::attempt_merge(openclregion &later)
  {
    assert(later.regionstart==regionend);
    
    if (!fill_event.get() && !later.fill_event.get()) {      
      regionend=later.regionend;
      return true;
    }
    return false;
  }


  std::shared_ptr<openclregion> openclregion::sp_breakup(snde_index breakpoint)
  /* breakup method ends this region at breakpoint and returns
     a new region starting at from breakpoint to the prior end */
  {
    std::shared_ptr<openclregion> newregion(new openclregion(breakpoint,regionend));
    regionend=breakpoint;

    
    
    if (fill_event.get()) {
      newregion->fill_event=fill_event;
      //clRetainEvent(newregion->fill_event);
    }
    
    
    return newregion;
  }

  openclarrayinfo::openclarrayinfo(cl::Context context, void **arrayptr,size_t numelem) :
    context(context),
    //device(device),
    arrayptr(arrayptr),
    numelem(numelem)
  {
    //clRetainContext(this->context); /* increase refcnt */
    //clRetainDevice(this->device);
  }

  // equality operator for std::unordered_map
  bool openclarrayinfo::operator==(const openclarrayinfo &b) const
  {
    return b.context==context && b.arrayptr==arrayptr && b.numelem==numelem; // && b.device==device;
  }

  size_t openclarrayinfo_hash::operator()(const snde::openclarrayinfo & x) const
  {
    return std::hash<void *>{}((void *)x.context.get()) ^ std::hash<size_t>{}(x.numelem) /* + std::hash<void *>{}((void *)x.device.get())*/ ^ std::hash<void *>{}((void *)x.arrayptr);
  }


  openclcacheentry::openclcacheentry(cl::Context context,
			       std::shared_ptr<allocator> alloc, // nullptr OK
			       snde_index total_nelem,
			       size_t elemsize,void **arrayptr,
			       std::mutex *cachemanageradminmutex) :
    alloc(alloc),
    numelem(total_nelem),
    elemsize(elemsize),
    arrayptr(arrayptr)
  {
    /* initialize buffer in context with specified size */
    /* caller should hold array manager's admin lock; 
       should also hold at least read lock on this buffer */
    snde_index nelem;
    cl_int errcode_ret=CL_SUCCESS;
    
    this->elemsize=elemsize;
    this->arrayptr=arrayptr;
    this->alloc=alloc;
    
    nelem=total_nelem;
    
    //buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,nelem*elemsize, NULL /* *arrayptr */,&errcode_ret);
    buffer=cl::Buffer(context,CL_MEM_READ_WRITE,nelem*elemsize);
    
    //if (errcode_ret != CL_SUCCESS) {
    //throw openclerror(errcode_ret,(std::string)"Error creating buffer of size %d",(long)(nelem*elemsize));
    //}
    
    invalidity.mark_all(nelem);
    
    /* Need to register for notification of realloc
       so we can re-allocate the buffer! */
    if (alloc) {
      pool_realloc_callback=std::make_shared<std::function<void(snde_index)>>([this,context,cachemanageradminmutex](snde_index total_nelem) {
	/* Note: we're not managing context, but presumably the openclarrayinfo that is our key in buffer_map will prevent the context from being freed */
	// Whatever triggered the realloc had better have a write lock (or equivalent) on the array in which case nothing else should be reading 
	// or writing so this stuff should be safe. 
	
	std::lock_guard<std::mutex> lock(*cachemanageradminmutex); // protects invalidity field
	cl_int lambdaerrcode_ret=CL_SUCCESS;
	//clReleaseMemObject(buffer); /* free old buffer */
	
	/* allocate new buffer */
	//buffer=cl::Buffer(context,CL_MEM_READ_WRITE,numelem*this->elemsize,nullptr); // /* *this->arrayptr */, &lambdaerrcode_ret);
	buffer=cl::Buffer(); // just leave an empty buffer
	numelem=total_nelem; /* determine new # of elements */
	
	
	// if (lambdaerrcode_ret != CL_SUCCESS) {
	//  throw openclerror(lambdaerrcode_ret,"Error expanding buffer to size %d",(long)(numelem*this->elemsize));
	//}
	
	/* Mark entire buffer as invalid */
	invalidity.mark_all(numelem);
	
      });
      alloc->register_pool_realloc_callback(pool_realloc_callback);
    }
    
  }
  
  openclcacheentry::~openclcacheentry()
  {
    std::shared_ptr<allocator> alloc_strong = alloc.lock();
    if (alloc_strong) { /* if allocator already freed, it would take care of its reference to the callback */ 
      alloc_strong->unregister_pool_realloc_callback(pool_realloc_callback);
    }
    
    //clReleaseMemObject(buffer);
    //buffer=NULL;
  }

  void openclcacheentry::mark_as_gpu_modified(std::shared_ptr<recording_storage> storage,snde_index pos,snde_index nelem)
  {
    /* ***!!!!! Should really check here to make sure that we're not modifying
       any dirtyregions elements that are marked as done or have a FlushDoneEvent */
    _dirtyregions.mark_region(pos,nelem,storage);
    invalidity.clear_region(pos,nelem);  // if the GPU wrote it, then the GPU copy is no longer invalid
  }

  openclcachemanager::openclcachemanager() 
  {
    name = get_cache_name("OpenCL CMGR");
    //_memalloc=memalloc;
    //locker = std::make_shared<lockmanager>();
  }

  cl::CommandQueue openclcachemanager::_get_queue(cl::Context context,cl::Device device)
  // internal version for when adminlock is alread held 
  {
    
    auto c_d = context_device(context,device);
    auto iter = queue_map.find(c_d);
    if (iter==queue_map.end()) {
      // no entry... create one
      cl_int clerror=0;
      cl::CommandQueue newqueue=cl::CommandQueue(context,device); //clCreateCommandQueue(context,device,0,&clerror);
      queue_map.emplace(c_d,newqueue);
      
    }
    return queue_map.at(c_d); // .get_noref();
  }

  cl::CommandQueue openclcachemanager::get_queue(cl::Context context,cl::Device device)
  {
    std::lock_guard<std::mutex> adminlock(admin);
    return _get_queue(context,device);
    
  }

  
  void openclcachemanager::mark_as_invalid_except_buffer(std::shared_ptr<openclcacheentry> exceptbuffer,void **arrayptr,snde_index pos,snde_index numelem)
  /* marks an array region (with exception of particular buffer) as needing to be updated from CPU copy */
  /* This is typically used after our CPU copy has been updated from exceptbuffer, to push updates out to all of the other buffers */
  {
    std::unique_lock<std::mutex> adminlock(admin);
    
    std::unordered_map<openclarrayinfo,std::weak_ptr<openclcacheentry>,openclarrayinfo_hash/*,openclarrayinfo_equal*/>::iterator buffer;
    
    /* Mark all of our buffers with this region as invalid */
    std::unordered_map<void **,std::vector<openclarrayinfo>>::iterator buffers_by_array_it = buffers_by_array.find(arrayptr);

    /*if (1||numelem==1640960) {
      snde_warning("Searching for buffer for array 0x%llx for invalidity marking",(unsigned long long)arrayptr);
      }*/
    
    if (buffers_by_array_it != buffers_by_array.end()) {
      for (auto & arrayinfo : buffers_by_array_it->second) {
      
	buffer=buffer_map.find(arrayinfo);
	assert(buffer != buffer_map.end()); // If this fails then we're not cleaning up buffers_by_array properly
	std::shared_ptr<openclcacheentry> buffer_strong = buffer->second.lock();
	
	
	if (buffer_strong) {
	  /*if (1||numelem==1640960) {
	    snde_warning("Found buffer for array 0x%llx... invalidity.size=%u",(unsigned long long)arrayptr,(unsigned)buffer_strong->invalidity.size());
	    }*/


	  if (exceptbuffer == nullptr || exceptbuffer.get() != buffer_strong.get()) {
	    
	    if (numelem==SNDE_INDEX_INVALID) {
	      buffer_strong->invalidity.mark_region(pos,SNDE_INDEX_INVALID);
	      
	    } else {
	      buffer_strong->invalidity.mark_region(pos,numelem);
	    }
	  }
	}
      }
    }
    
    // buffer.second is a shared_ptr to an openclcacheentry
    
  }
  
  void openclcachemanager::mark_as_gpu_modified(cl::Context context, std::shared_ptr<recording_storage> storage)
  {
    void **arrayptr = storage->lockableaddr();
    openclarrayinfo arrayinfo=openclarrayinfo(context,arrayptr,storage->lockablenelem());
    
    std::unique_lock<std::mutex> adminlock(admin);
    
    std::weak_ptr<openclcacheentry> buffer=buffer_map.at(arrayinfo);

    std::shared_ptr<openclcacheentry> buffer_strong=buffer.lock();

    assert(buffer_strong); // if we're marking as GPU modified, the buffer had better not have expired!
    buffer_strong->mark_as_gpu_modified(storage,storage->base_index/*+pos*/,storage->nelem/*len*/);
    
  }


  /* marks an array region as needing to be updated from CPU copy */
  void openclcachemanager::mark_as_invalid(void **arrayptr,snde_index base_index,snde_index pos,snde_index numelem)
  {

    //void **arrayptr=storage->lockableaddr();
    //snde_index nelem=storage->lockablenelem();

    //assert(numelem <= storage->nelem);
    /* This is typically used if the CPU copy is updated directly */
    mark_as_invalid_except_buffer(nullptr,arrayptr,base_index + pos,numelem);
  }
  
  void openclcachemanager::notify_storage_expiration(void **arrayptr, snde_index base_index, snde_index nelem)
  {
    //void **arrayptr=storage->lockableaddr();
    //snde_index nelem=storage->lockablenelem();

    std::lock_guard<std::mutex> adminlock(admin);

    if (base_index==0) {
      // We only manage entire arrays here
      std::unordered_map<void **,std::vector<openclarrayinfo>>::iterator bba_it = buffers_by_array.find(arrayptr);
      
      if (bba_it != buffers_by_array.end()) {

	for (auto && arrayinfo: bba_it->second) {

	  // arrayinfo is an openclarrayinfo
	  std::unordered_map<openclarrayinfo,std::weak_ptr<openclcacheentry>,openclarrayinfo_hash/* ,openclarrayinfo_equal*/>::iterator buffer_it = buffer_map.find(arrayinfo);
	  if (buffer_it != buffer_map.end()) {
	    buffer_map.erase(buffer_it); // remove expiring/expired pointer -- note that the buffer might actually stay in memory if there are other references to it 
	  }
	}
	buffers_by_array.erase(bba_it);
      }
      
    }
  }


  void openclcachemanager::_TransferInvalidRegions(cl::Context context, cl::Device device,std::shared_ptr<openclcacheentry> oclbuffer,void **arrayptr,snde_index firstelem,snde_index numelem,std::vector<cl::Event> &ev)
    // internal use only... initiates transfers of invalid regions prior to setting up a read buffer
    // WARNING: operates in-place on prerequisite event vector ev
    // assumes admin lock is held
  {
    
    rangetracker<openclregion>::iterator invalidregion;
    
    if (numelem==SNDE_INDEX_INVALID) {
      numelem=oclbuffer->numelem-firstelem;
    }
    
    /* transfer any relevant areas that are invalid and not currently pending */
    
    rangetracker<openclregion> invalid_regions=oclbuffer->invalidity.iterate_over_marked_portions(firstelem,numelem);

    /*if (1||numelem==1640960) {
      snde_warning("Array 0x%llx... invalidity: _TransferInvalidRegions() invalidity.size()=%u",(unsigned long long)arrayptr,(unsigned)oclbuffer->invalidity.size());
      }*/

    
    for (auto & invalidregion: invalid_regions) {

      /*if (1||numelem==1640960) {
	snde_warning("Array 0x%llx... _TransferInvalidRegions() invalid region @ %llu, fill event = 0x%llx",(unsigned long long)arrayptr,(unsigned long long)invalidregion.first,(unsigned long long)invalidregion.second->fill_event.get());
	}*/

      /* Perform operation to transfer data to buffer object */
      
      /* wait for all other events as it is required for a 
	 partial write to be meaningful */

      
      if (!invalidregion.second->fill_event.get()) {
	snde_index offset=invalidregion.second->regionstart*oclbuffer->elemsize;
	cl::Event newevent;
	
	//cl::Event *evptr=NULL;
	//if (ev.size() > 0) {
	//  evptr=&ev[0];
	//}
	
	snde_debug(SNDE_DC_OPENCL,"enqueueWriteBuffer(queue %llx, buffer %llx, offset=%llu (%llu elem) nbytes=%llu (%llu elem), array %llx)",(unsigned long long)_get_queue(context,device).get(),(unsigned long long)oclbuffer->buffer.get(),(unsigned long long)(offset),(unsigned long long)(offset/oclbuffer->elemsize),(unsigned long long)((invalidregion.second->regionend-invalidregion.second->regionstart)*oclbuffer->elemsize),(unsigned long long)(invalidregion.second->regionend-invalidregion.second->regionstart),(unsigned long long)arrayptr);

	// debugging...
	/*if (1 || numelem==1640960) {
	  snde_warning("Enqueuing write of %u elements, arrayptr=0x%llx.",(unsigned)numelem,(unsigned long long)arrayptr);
	  }*/
	  
	_get_queue(context,device).enqueueWriteBuffer(oclbuffer->buffer,CL_FALSE,offset,(invalidregion.second->regionend-invalidregion.second->regionstart)*oclbuffer->elemsize,(char *)*arrayptr + offset,&ev,&newevent);
	
	/* now that it is enqueued we can replace our event list 
	     with this newevent */
	
	//for (auto & oldevent : ev) {
	//  clReleaseEvent(oldevent);
	//}
	ev.clear();

	//newevent.wait();	// Debugging !!!! TEMPORARY

	
	ev.emplace_back(newevent); /* add new event to our set (this eats our ownership) */
	
	invalidregion.second->fill_event=newevent;
      }
      
    }
    
    //clFlush(_get_queue(context,device));
    _get_queue(context,device).flush();
  }
  
  std::shared_ptr<openclcacheentry> openclcachemanager::_GetBufferObject(std::shared_ptr<recording_storage> storage,cl::Context context, cl::Device device,snde_index nelem,snde_index elemsize,void **arrayptr)
  // manager may be nullptr if this array doesn't have allocations within it
  {
    // internal use only; assumes admin lock is held;
    // ***!!! NOTE: Before calling this and before locking the admin lock, you should call     storage->add_follower_cachemanager(shared_from_this());  to ensure the cachemanager is registered

    
    //fprintf(stderr,"_GetBufferObject(0x%lx,0x%lx,0x%lx)... buffer_map.size()=%u, tid=0x%lx admin->__owner=0x%lx\n",(unsigned long)context,(unsigned long)device,(unsigned long)arrayptr,(unsigned)buffer_map.size(),(unsigned long)((pid_t)syscall(SYS_gettid)),(unsigned long) admin._M_mutex.__data.__owner);
    
    std::shared_ptr<openclcacheentry> oclbuffer;

    std::shared_ptr<allocator> alloc;
    openclarrayinfo arrayinfo=openclarrayinfo(context,storage->lockableaddr(),storage->lockablenelem());
    //allocationinfo thisalloc = (*manager->allocators()).at(arrayptr);
    //std::shared_ptr<allocator> alloc=thisalloc.alloc;
    
    std::unordered_map<openclarrayinfo,std::weak_ptr<openclcacheentry>,openclarrayinfo_hash/*,openclarrayinfo_equal*/>::iterator buffer;
    buffer=buffer_map.find(arrayinfo);
    if (buffer == buffer_map.end() || (!buffer->second.lock())) {
      /* need to create buffer */
      oclbuffer=std::make_shared<openclcacheentry>(context,alloc,nelem,elemsize,arrayptr,&admin);
      if (buffer != buffer_map.end()) {
	// entry in buffer_map already exists but must be expired; replace it
	buffer->second = oclbuffer;
      } else {
	// new entry in buffer_map
	buffer_map.emplace(arrayinfo,oclbuffer);
      }
      
      //fprintf(stderr,"_GetBufferObject(0x%lx,0x%lx,0x%lx) created buffer_map entry. buffer_map.size()=%u, tid=0x%lx admin->__owner=0x%lx\n",(unsigned long)context,(unsigned long)device,(unsigned long)arrayptr,(unsigned)buffer_map.size(),(unsigned long)((pid_t)syscall(SYS_gettid)),(unsigned long) admin._M_mutex.__data.__owner);
      
      std::vector<openclarrayinfo> & buffers_for_this_array=buffers_by_array[arrayinfo.arrayptr];
      buffers_for_this_array.push_back(arrayinfo);
      
    } else {
      oclbuffer=buffer->second.lock();//buffer_map.at(arrayinfo);
    }


    {
      std::lock_guard<std::mutex> cache_lock_holder(storage->cache_lock);
      // putting a shared_ptr to the oclbuffer on the recording_storage cache
      // ensures that it will stick around as long as the recording
      // stays in memory (unless/until we implement a cache cleaner
      // to go out and clean it up)
      storage->cache.emplace(name,oclbuffer);
    }
    
    return oclbuffer;
  }


  std::tuple<rwlock_token_set,cl::Buffer,std::vector<cl::Event>,std::shared_ptr<openclcacheentry>> openclcachemanager::_GetOpenCLSubBuffer(std::shared_ptr<recording_storage> storage,rwlock_token_set alllocks,cl::Context context, cl::Device device,snde_index substartelem,snde_index subnumelem,bool write,bool write_only/*=false*/)
  // It is assumed that the caller has the data adequately locked, if needed. 
  {

    snde_index elemsize=storage->elementsize; 
    void **arrayptr=storage->lockableaddr();
    snde_index nelem=storage->lockablenelem();

    storage->add_follower_cachemanager(shared_from_this());
    
    std::unique_lock<std::mutex> adminlock(admin);
      
    openclarrayinfo arrayinfo=openclarrayinfo(context,arrayptr,nelem);
    
    std::vector<cl::Event> ev;
    
    std::shared_ptr<openclcacheentry> oclbuffer=_GetBufferObject(storage,context,device,nelem,elemsize,arrayptr);
    
    rangetracker<openclregion>::iterator invalidregion;
    
    /*if (1||subnumelem==1640960) {
      snde_warning("Looking up invalidity for array 0x%llx... invalidity.size=%u",(unsigned long long)arrayptr,(unsigned)oclbuffer->invalidity.size());
      }*/
    
    /* make sure we will wait for any currently pending transfers overlapping with this subbuffer*/
    for (invalidregion=oclbuffer->invalidity.begin();invalidregion != oclbuffer->invalidity.end();invalidregion++) {
      if (invalidregion->second->fill_event.get() && region_overlaps(*invalidregion->second,substartelem,substartelem+subnumelem)) {
	ev.emplace_back(invalidregion->second->fill_event); 
	/*if (1||subnumelem==1640960) {
	  snde_warning("Array 0x%llx... invalidity: fill event pending",(unsigned long long)arrayptr);
	  }*/
	
      }
    }
    
    rwlock_token_set regionlocks;

    if (alllocks) {
      if (write && storage->requires_locking_write_gpu) {
	regionlocks=storage->lockmgr->get_preexisting_locks_write_array_region(alllocks,arrayptr,substartelem,subnumelem);
      } else if (!write && storage->requires_locking_read_gpu) {
	regionlocks=storage->lockmgr->get_preexisting_locks_read_array_region(alllocks,arrayptr,substartelem,subnumelem);	
      }
    }
    
    if (!write_only) {
      /* No need to enqueue transfers if kernel is strictly write */
      
      /*if (1||subnumelem==1640960) {
	snde_warning("Array 0x%llx... invalidity: Calling _TransferInvalidRegions() ",(unsigned long long)arrayptr);
	}*/
      _TransferInvalidRegions(context,device,oclbuffer,arrayptr,substartelem,subnumelem,ev);
      
      
    }
    
    cl_mem_flags flags;

    if (write && !write_only) {
      flags=CL_MEM_READ_WRITE;
    } else if (write_only) {
      assert(write);
      flags=CL_MEM_WRITE_ONLY;	
    } else  {
      assert(!write);
      flags=CL_MEM_READ_ONLY;
    }
    cl_int errcode=CL_SUCCESS;
    cl_buffer_region region = { substartelem*elemsize, subnumelem*elemsize };
    cl::Buffer subbuffer = oclbuffer->buffer.createSubBuffer(flags,CL_BUFFER_CREATE_TYPE_REGION,&region);
    //if (errcode != CL_SUCCESS) {
    //throw openclerror(errcode,"Error creating subbuffer");
    //}


    return std::make_tuple(regionlocks,subbuffer,ev,oclbuffer);
  }



  std::tuple<rwlock_token_set,cl::Buffer,std::vector<cl::Event>,std::shared_ptr<openclcacheentry>> openclcachemanager::_GetOpenCLBuffer(std::shared_ptr<recording_storage> storage,rwlock_token_set alllocks,cl::Context context, cl::Device device, snde_index substartelem, snde_index subnumelem, bool write,bool write_only/*=false*/) /* indexed by arrayidx */
/* cl_mem_flags flags,snde_index firstelem,snde_index numelem */ /* numelems may be SNDE_INDEX_INVALID to indicate all the way to the end */


    /* (OBSOLETE) note cl_mem_flags does NOT determine what type of OpenCL buffer we get, but rather what
       our kernel is going to do with it, i.e. CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY or CL_MEM_READ_WRITE */
    /* returns new rwlock_token_set representing readlocks ... let this fall out of scope to release it. */
    /* returns new rwlock_token_set representing writelocks ... let this fall out of scope to release it. */
    /* returns cl_mem... will need to call clReleaseMemObject() on this */
    /* returns snde_index representing the offset, in units of elemsize into the cl_mem of the first element of the cl_mem... (always 0 for now, but may not be zero if we use sub-buffers in the future)  */
    /* returns cl::Events... will need to call clReleaseEvent() on each */

      
    /* old comments below... */
    /* Must have a read lock on allocatedptr to get CL_MEM_READ_ONLY. Must have a write lock to get
       CL_MEM_READ_WRITE or CL_MEM_WRITE_ONLY 

       ... but having a write lock typically implies that the CPU will do the writing... normally 
       when the write lock is released we would flush changes back out to the GPU, but that 
       is exactly backwards in this case. We don't want this function to acquire the write lock
       because one write lock (on allocatedptr) can cover multiple arrayptrs. 

       ... So we'll want something where ownership of the lock is transferred to the GPU and the 
       calling CPU code never unlocks it ... so when the GPU is done, the lock is reinterpreted 
       as the need to (perhaps lazily)  transfer the GPU output back to host memory. 
       
       All of this is complicated by the queue-based nature of the GPU where we may queue up multiple
       operations...
*/
    /* General assumption... prior to this call, the CPU buffer is valid and there may be a valid
       cached copy on the GPU. 

       In case of (device) read: The CPU buffer is locked for read... This 
       causes a wait until other threads have finished writing. One or more clEnqueueWriteBuffer calls 
       may be issued to transfer the changed range of the buffer to the GPU (if necessary). GetOpenCLBuffer will 
       return the OpenCL buffer and the events to wait for (or NULL) to ensure the 
       buffer is transferred. 

       In case of (device) write: The CPU buffer is locked for write... This 
       causes a wait until other threads have finished reading. The write credentials 
       end up embedded with the buffer returned by this call. The caller should tie 
       the event from clEnqueueNDRangeKernel to the buffer returned by this call so that
       when complete clEnqueueReadBuffer can be called over the locked address range
       to transfer changed data back and the write credentials may then be released


       Normally the read locking would trigger a transfer from the device (if modified)
       and write locking would trigger a transfer to the device. So somehow 
       we need to instrument the lock functions to get these notifications but be 
       able to swap them around as needed (?)

       Remember there may be multiple devices. So a write to a (region) on any device needs 
       to invalidate all the others and (potentially) trigger a cascade of transfers. We'll 
       also need a better data structure that can readily find all of the instances of a particular 
       array across multiple contexts so as to trigger the transfers and/or perform invalidation
       on write. 

       So the data structure will (roughly) need to 
         (a) have all of the device buffers for an array, each with some sort of status
  	   * Status includes any pending transfers and how to wait for them 
         (b) have the CPU buffer for the array, also with some sort of status

       The main use-case is acquiring access to many arrays... and perhaps custom arrays as well... 
       then running a kernel on those arrays. ... So plan on a varargs function that accepts pointers
       to the various arrays, accepts custom buffers and the worksize and then enqueues the  the kernel. 
       What if we immediately want to enqueue another kernel? Should be able to reference those same 
       arrays/buffers without retransferring. On write-unlock should initiate (and wait for?) 
       transfer of anything that 
       has changed because future read- or write-locks will require that transfer to be complete. 

       Q: in order to acquire a write lock for a device, do we need to synchronize the array TO that device
       first? 
       A: Yes, at least the range of data being written, because it might not write everything. But it might in fact write everything
       
       * What about giving the kernel read access to the whole array,
         but only write a portion (?)
         * Locking API won't support this in forseeable future. Lock the whole
           array for write. In future it might be possible to downgrade part
    */
    {


      snde_index elemsize=storage->elementsize; 
      void **arrayptr=storage->lockableaddr();
      snde_index nelem=storage->lockablenelem();

      //snde_index substartelem=storage->base_index;
      //snde_index subnumelem=storage->nelem;

      if (substartelem != 0 || subnumelem != nelem) {
	// need a sub-buffer instead
	return _GetOpenCLSubBuffer(storage,alllocks,context,device,substartelem,subnumelem,write,write_only);
	
      }

      
      storage->add_follower_cachemanager(shared_from_this());
      
      std::unique_lock<std::mutex> adminlock(admin);
      

      //std::shared_ptr<arraymanager> manager_strong(manager); /* as manager references us, it should always exist while we do. This will throw an exception if it doesn't */

      //std::shared_ptr<allocator> alloc=(*manager_strong->allocators())[arrayptr].alloc;
      std::shared_ptr<openclcacheentry> oclbuffer;
      std::vector<cl::Event> ev;
      rangetracker<openclregion>::iterator invalidregion;
      //size_t arrayidx=manager_strong->locker->get_array_idx(arrayptr);

    
      oclbuffer=_GetBufferObject(storage,context,device,nelem,elemsize,arrayptr);
      /* Need to enqueue transfer (if necessary) and return an event set that can be waited for and that the
	 clEnqueueKernel can be dependent on */
      
      /* check for pending events and accumulate them into wait list */
      /* IS THIS REALLY NECESSARY so long as we wait for 
	 the events within our region? (Yes; because OpenCL
	 does not permit a buffer to be updated in one place 
	 while it is being used in another place.
	 
	 NOTE THAT THIS PRACTICALLY PREVENTS region-granular
	 locking unless we switch to using non-overlapping OpenCL sub-buffers 
         (which we have done)

	 ... We would have to ensure that all lockable regions (presumably 
	 based on allocation) are aligned per CL_DEVICE_MEM_BASE_ADDR_ALIGN for all
         relevant devices)

	 ... so this means that the region-granular locking could only
	 be at the level of individual allocations (not unreasonable)...

	 ... since allocations cannot be overlapping and may not be adjacent, 
	 this means that one lock per allocation is the worst case, and we 
	 don't have to worry about overlapping lockable regions.

	 ... So in that case we should create a separate rwlock for each allocation
         (in addition to a giant one for the whole thing?  ... or do we just iterate
         them to lock the whole thing?... probably the latter. ) 

	 if locking the whole thing we would use the main cl_mem buffer 
	 if a single allocation it would be a sub_buffer. 

      */

      
      
      /* make sure we will wait for any currently pending transfers */
      for (invalidregion=oclbuffer->invalidity.begin();invalidregion != oclbuffer->invalidity.end();invalidregion++) {
	if (invalidregion->second->fill_event.get()) {
	  //clRetainEvent(invalidregion->second->fill_event);
	  ev.emplace_back(invalidregion->second->fill_event); 
	  
	}
      }
      
      
      
      /* obtain lock on this array (release adminlock because this might wait and hence deadlock) */
      //adminlock.unlock(); /* no-longer needed because we are using preexising locks now */


      rwlock_token_set readlocks=empty_rwlock_token_set();

      //for (auto & markedregion: (*arrayreadregions)[arrayidx]) {
      //snde_index numelems;
      //	
      //if (markedregion.second->regionend==SNDE_INDEX_INVALID) {
      //numelems=SNDE_INDEX_INVALID;
      //} else {
      //numelems=markedregion.second->regionend-markedregion.second->regionstart;
      //}
      //
      ///
      //merge_into_rwlock_token_set(readlocks,regionlocks);
      //}

      
      rwlock_token_set regionlocks;
      if (alllocks) {
	if (write && storage->requires_locking_write_gpu) {
	  regionlocks=storage->lockmgr->get_preexisting_locks_read_array_region(alllocks,arrayptr,0,SNDE_INDEX_INVALID);
	  
	} else if (!write && storage->requires_locking_read_gpu) {	  
	  regionlocks=storage->lockmgr->get_preexisting_locks_read_array_region(alllocks,arrayptr,0,SNDE_INDEX_INVALID);
	}
      }
      //rwlock_token_set writelocks=empty_rwlock_token_set();

      //for (auto & markedregion: (*arraywriteregions)[arrayidx]) {
      //snde_index numelems;
	
      //if (markedregion.second->regionend==SNDE_INDEX_INVALID) {
      //numelems=SNDE_INDEX_INVALID;
      //} else {
      //numelems=markedregion.second->regionend-markedregion.second->regionstart;
      //}
      //
      //rwlock_token_set regionlocks=manager_strong->locker->get_preexisting_locks_write_array_region(alllocks,allocatedptr,markedregion.second->regionstart,numelems);
      //
      //merge_into_rwlock_token_set(writelocks,regionlocks);
      //}
	

      
      /* reaquire adminlock */
      //adminlock.lock();

      //rangetracker<markedregion> regions = range_union((*arrayreadregions)[arrayidx],(*arraywriteregions)[arrayidx]);
      
      
      if (!write_only) { /* No need to enqueue transfers if kernel is strictly write */
	//for (auto & markedregion: regions) {
	  
	snde_index firstelem=0;// markedregion.second->regionstart;
	snde_index numelem=SNDE_INDEX_INVALID;


	_TransferInvalidRegions(context,device,oclbuffer,arrayptr,firstelem,numelem,ev);
	
	
      }
      
      //clRetainMemObject(oclbuffer->buffer); /* ref count for returned cl_mem pointer */
      
      return std::make_tuple(regionlocks,oclbuffer->buffer,ev,oclbuffer);
    }

  std::tuple<rwlock_token_set,cl::Buffer,std::vector<cl::Event>> openclcachemanager::GetOpenCLBuffer(std::shared_ptr<recording_storage> storage,rwlock_token_set alllocks,cl::Context context, cl::Device device, snde_index substartelem, snde_index subnumelem,bool write,bool write_only/*=false*/)
  {
    rwlock_token_set retlocks;
    cl::Buffer buf;
    std::vector<cl::Event> new_fill_events;
    std::shared_ptr<openclcacheentry> cacheentry;

    storage = storage->get_original_storage(); // operate only on the original storage, not a non-moving reference so that we don't have duplicate cache entries

    
    
    
    std::tie(retlocks,buf,new_fill_events,cacheentry) = _GetOpenCLBuffer(storage,alllocks,context,device,substartelem,subnumelem,write,write_only);

    return std::make_tuple(retlocks,buf,new_fill_events);
  }

  std::pair<std::vector<cl::Event>,std::vector<cl::Event>> openclcachemanager::FlushWrittenOpenCLBuffer(cl::Context context,cl::Device device,std::shared_ptr<recording_storage> storage,std::vector<cl::Event> explicit_prerequisites)
  // Flush changes made by an opencl kernel to our in-CPU-memory copy; then notify any other caches about the change. 
    {
      // Gather in implicit prerequisites
      // (any pending transfers to this buffer)
      /* capture our admin lock */

      // Note that if there were any actual transfers initiated,
      // the result will be length-1. Otherwise no transfer was needed.
      std::vector<cl::Event> wait_events;
      std::vector<cl::Event> result_events;

      std::vector<std::pair<cl::Event,std::function<void(cl::Event,cl_int)> *>> callback_requests; 
      rangetracker<openclregion>::iterator invalidregion;

      void **arrayptr = storage->lockableaddr();
      snde_index nelem = storage->lockablenelem();

      //size_t arrayidx=manager_strong->locker->get_array_idx(arrayptr);

      //std::shared_ptr<allocator> alloc=(*manager_strong->allocators())[arrayptr].alloc;

      /* create arrayinfo key */
      openclarrayinfo arrayinfo=openclarrayinfo(context,arrayptr,nelem);
     
      std::unique_lock<std::mutex> adminlock(admin);

      std::shared_ptr<openclcacheentry> oclbuffer;

      oclbuffer=buffer_map.at(arrayinfo).lock(); /* buffer should exist because should have been created in GetOpenCLBuffer() */
      assert(oclbuffer); // Should be present because if we've just been writing to it, then it should be locked in memory!

      /* check for pending events and accumulate them into wait list */
      /* ... all transfers in to this buffer should be complete before we allow a transfer out */
      for (invalidregion=oclbuffer->invalidity.begin();invalidregion != oclbuffer->invalidity.end();invalidregion++) {
	if (invalidregion->second->fill_event.get()) {
	  //clRetainEvent(invalidregion->second->fill_event);
	  wait_events.emplace_back(invalidregion->second->fill_event); 
	  
	}
      }

      for (auto & ep_event : explicit_prerequisites) {
	//clRetainEvent(ep_event);
	wait_events.emplace_back(ep_event);
      }

      /* Go through every write region, flushing out any dirty portions */
      
      //for (auto & writeregion: (*arraywriteregions)[arrayidx]) {
      for (auto & dirtyregion : oclbuffer->_dirtyregions) {
	snde_index numelem;
	
	if (dirtyregion.second->regionend==SNDE_INDEX_INVALID) {
	  numelem=storage->lockablenelem()-dirtyregion.second->regionstart;
	} else {
	  numelem=dirtyregion.second->regionend-dirtyregion.second->regionstart;
	}
	
	snde_index offset=dirtyregion.second->regionstart;  //*oclbuffer->elemsize;
	
	if (dirtyregion.second->FlushDoneEvent.get() && !dirtyregion.second->FlushDoneEventComplete) {
	  //clRetainEvent(dirtyregion.second->FlushDoneEvent);
	  result_events.emplace_back(dirtyregion.second->FlushDoneEvent);	  
	} else if (!dirtyregion.second->FlushDoneEvent.get() && !dirtyregion.second->FlushDoneEventComplete) {
	  /* Need to perform flush */
	  // Queue transfer
	
	  
	  cl::Event newevent; //=NULL;

	  snde_debug(SNDE_DC_OPENCL,"enqueueReadBuffer(queue %llx, buffer %llx, offset=%llu (%llu elem) nbytes=%llu (%llu elem), array %llx)",(unsigned long long)_get_queue(context,device).get(),(unsigned long long)oclbuffer->buffer.get(),(unsigned long long)(offset*oclbuffer->elemsize),(unsigned long long)offset,(unsigned long long)(numelem*oclbuffer->elemsize),(unsigned long long)numelem,(unsigned long long)arrayptr);
	  
	  _get_queue(context,device).enqueueReadBuffer(oclbuffer->buffer,CL_FALSE,offset*oclbuffer->elemsize,(numelem)*oclbuffer->elemsize,((char *)(*arrayptr)) + (offset*oclbuffer->elemsize),&wait_events,&newevent);
	  dirtyregion.second->FlushDoneEvent=newevent;
	  //fprintf(stderr,"Setting FlushDoneEvent...\n");

	  //clRetainEvent(newevent);/* dirtyregion retains a reference to newevent */

	  
	  /**** trigger marking of other caches as invalid */
	  
	  //cleanedregions.mark_region(regionstart,regionend-regionstart,this);
	  /* now that it is enqueued we can replace our wait list 
	     with this newevent */
	  //for (auto & oldevent : wait_events) {
	  //  clReleaseEvent(oldevent);
	  //}
	  wait_events.clear();
	  
	  //dirtyregion.second->FlushDoneEvent=newevent;
	  //clRetainEvent(newevent); 

	  /* queue up our callback request (queue it rather than do it now, so as to avoid a deadlock
	     if it is processed inline) */

	  std::shared_ptr<opencldirtyregion> dirtyregionptr=dirtyregion.second;
	  std::shared_ptr<recording_storage> dirtystorage = dirtyregion.second->owning_storage.lock();

	  if (dirtystorage) {
	    std::shared_ptr<openclcachemanager> shared_this=std::dynamic_pointer_cast<openclcachemanager>(shared_from_this());
	    
	    callback_requests.emplace_back(std::make_pair(newevent,new std::function<void(cl::Event,cl_int)>([ shared_this, dirtystorage, arrayptr, dirtyregionptr, oclbuffer, newevent ](cl::Event event, cl_int event_command_exec_status) { // matching delete is in snde_opencl_callback()
	      /* NOTE: This callback may occur in a separate thread */
	      /* it indicates that the data transfer is complete */
	      if (event_command_exec_status != CL_COMPLETE) {
		throw openclerror(event_command_exec_status,"Error in waited event (from executing clEnqueueReadBuffer()) ");
		
	      }
	      
	      std::unique_lock<std::mutex> adminlock(shared_this->admin);
	      /* copy the info out of dirtyregionptr while we hold the admin lock */
	      //opencldirtyregion dirtyregion=*dirtyregionptr;
	      snde_index dirtyregionstart=dirtyregionptr->regionstart;
	      snde_index dirtyregionend=dirtyregionptr->regionend;
	      
	      adminlock.unlock();
	      /* We must now notify others that this has been modified */
	      
	      /* Others include other buffers (e.g. other contexts) of our own cache manager... */
	      
	      shared_this->mark_as_invalid_except_buffer(oclbuffer,arrayptr,dirtyregionstart,dirtyregionend-dirtyregionstart);

	      assert(dirtyregionstart >= dirtystorage->base_index);
	      assert(dirtyregionend <= dirtystorage->base_index+dirtystorage->nelem);
	      dirtystorage->mark_as_modified(shared_this,dirtyregionstart-dirtystorage->base_index,dirtyregionend-dirtyregionstart);
	      
	      
	      /* We must now mark this region as modified... i.e. that notifications to others have been completed */
	      adminlock.lock();
	      
	      dirtyregionptr->FlushDoneEventComplete=true;
	      dirtyregionptr->complete_condition.notify_all();
	      //clReleaseEvent(dirtyregionptr->FlushDoneEvent);
	      dirtyregionptr->FlushDoneEvent=cl::Event(); // NULL;
	      //fprintf(stderr,"FlushDoneEventComplete\n");
	    
	      
	    })));
	    
	    
	    result_events.emplace_back(newevent); /* add new event to our set, eating our referencee */
	  }
	}
      }
	
      //_get_queue_no_ref(context,device).flush();
      _get_queue(context,device).flush();
      
      /* Trigger notifications to others once transfer is complete and we can release our admin lock */
      adminlock.unlock();
      for (auto & event_func : callback_requests) {
	event_func.first.setCallback(CL_COMPLETE,snde_opencl_callback,(void *)event_func.second );
	
      }
      
      

	//std::unique_lock<std::mutex> arrayadminlock(manager->locker->_locks[arrayidx].admin);

	//markedregion startpos(offset,SNDE_INDEX_INVALID);
	//std::map<markedregion,rwlock>::iterator iter=manager->locker->_locks[arrayidx].subregions.lower_bound(startpos);
	//if (startpos < iter.first.regionstart) { /* probably won't happen due to array layout process, but just in case */
	//  assert(iter != manager->locker->_locks[arrayidx].subregions.begin());
	//  iter--;
	//}
	
	//// iterate over the subregions of this arraylock
	//for (;iter != manager->locker->_locks[arrayidx].subregions.end() && iter->first.regionstart < writeregion.second->regionend;iter++) {
	//  // iterate over the dirty bits of this subregion
	//  
	//  rangetracker<dirtyregion> cleanedregions;
	//  
	//  for (auto dirtyregion &: iter->second._dirtyregions.trackedregions) {
	// } }

	
	
	///if (dirtyregion.cache_with_valid_data==this) {
      ///* removed cleanedregions from dirtyregions */
      //for (auto cleanedregion &: cleanedregions) {
      //iter->second._dirtyregions.clear_region(cleanedregion.regionstart,cleanedregion.regionend-cleanedregion.regionstart,this);
      //}
      //}
	
	
	
	     	
	
      return std::make_pair(wait_events,result_events);
    }



  void openclcachemanager::ForgetOpenCLBuffer(rwlock_token_set locks,cl::Context context, cl::Device device, cl::Buffer mem,std::shared_ptr<recording_storage> storage, cl::Event data_not_needed)
  {
    /* Call this when you are done using the buffer. It will forget any transfers 
       and allow the cache entry to expire (freeing the underlying buffers) after
       the data_not_needed event. 
       
       Note that all copies of buffertoks will 
       no longer represent anything after this call (whatever that means)
       
    */ 
    
    void **arrayptr = storage->lockableaddr();
    snde_index nelem = storage->lockablenelem();
    
    
    /* create arrayinfo key */
    openclarrayinfo arrayinfo=openclarrayinfo(context,arrayptr,nelem);
    
    std::shared_ptr<openclcacheentry> cacheentry;
    {
      std::lock_guard<std::mutex> adminlock(admin);      
      auto buffer_map_it = buffer_map.find(arrayinfo);
      assert(buffer_map_it != buffer_map.end()); /* buffer should exist because should have been created in GetOpenCLBuffer()... Otherwise we shouldn't be calling this! */
      
      cacheentry=buffer_map_it->second.lock(); /* buffer should exist because should have been created in GetOpenCLBuffer()... Otherwise we shouldn't be calling this! */
    }
    assert(cacheentry);
    
    if (data_not_needed.get()) {
      
      // need to hold cacheentry alive until the data is no longer needed. So we pass it
      // as a lamdba parameter. This holds it as part of the std::function<> object until after the callback is finished
      // and the std::function is deleted inside snde_opencl_callback().
      
      data_not_needed.setCallback(CL_COMPLETE,snde_opencl_callback,(void *)new std::function<void(cl::Event,cl_int)>([ data_not_needed, cacheentry ](cl::Event event, cl_int event_command_exec_status) { // matching delete is in snde_opencl_callback()
	/* NOTE: This callback may occur in a separate thread */
	/* it indicates that the input data is no longer needed */
	if (event_command_exec_status != CL_COMPLETE) {
	  throw openclerror(event_command_exec_status,"Error from data_not_needed prerequisite ");
	  
	}	      

	// Note: buffer_map entry will go away (or may have already gone away) when the recording_storage expires. 
	
      } ));
    }
    
  }
  
  
  
  
  std::pair<std::vector<cl::Event>,std::shared_ptr<std::thread>> openclcachemanager::ReleaseOpenCLBuffer(rwlock_token_set locks,cl::Context context, cl::Device device, cl::Buffer mem,std::shared_ptr<recording_storage> storage, cl::Event input_data_not_needed,const std::vector<cl::Event> &output_data_complete)
    {
      /* Call this when you are done using the buffer. If you had 
	 a write lock it will queue a transfer that will 
	 update the CPU memory from the buffer before the 
	 locks actually get released 

	 Note that all copies of buffertoks will 
	 no longer represent anything after this call

      */ 
      /* Does not reduce refcount of mem or passed events */
      /* returns vector of events you can  wait for (one reference each) to ensure all is done */
      
      rangetracker<openclregion>::iterator invalidregion;

      std::vector<cl::Event> all_finished;
      std::shared_ptr<std::thread> unlock_thread;

      void **arrayptr = storage->lockableaddr();
      snde_index nelem = storage->lockablenelem();
      
      /* make copy of locks to delegate to threads... create pointers so it is definitely safe to delegate */
      rwlock_token_set *locks_copy1;
      rwlock_token_set *locks_copy2;

      if (locks) {
	locks_copy1 = new rwlock_token_set(clone_rwlock_token_set(locks));
	locks_copy2 = new rwlock_token_set(clone_rwlock_token_set(locks));
	
	release_rwlock_token_set(locks); /* release our reference to original */
      } else {
	locks_copy1 = new rwlock_token_set(empty_rwlock_token_set());
	locks_copy2 = new rwlock_token_set(empty_rwlock_token_set());
      }
      
      //std::shared_ptr<arraymanager> manager_strong(manager); /* as manager references us, it should always exist while we do. This will throw an exception if it doesn't */
      //size_t arrayidx=manager_strong->locker->get_array_idx(arrayptr);

      
      
      /* don't worry about invalidity here because presumably 
	 that was taken care of before we started writing
	 (validity is a concept that applies to the GPU buffer, 
	 not the memory buffer .... memory buffer is invalid
	 but noone will see that because it will be valid before
	 we release the write lock */
      
      
      std::vector<cl::Event> prerequisite_events(output_data_complete); /* transfer prequisites in */ 
      //output_data_complete.clear();
      if (input_data_not_needed.get()) {
	prerequisite_events.emplace_back(input_data_not_needed);
      }
      std::vector<cl::Event> wait_events,flush_events;

      // Iterate over dirty regions and add event callbacks instead to their flushdoneevents
      // such that once a particular FlushDoneEvent is complete, we do the dirty notification
      // to other caches. Once all FlushDoneEvents are complete, perform the unlocking.

      // (does nothing of substance if there is nothing dirty)
      std::tie(wait_events,flush_events)=FlushWrittenOpenCLBuffer(context,device,storage,prerequisite_events);
      /* Note now that wait_events and flush_events we have ownership of, whereas all of the prerequisite_events we didn't */
	
      // FlushWrittenOpenCLBuffer should have triggered the FlushDoneEvents and done the
      // dirty notification.... we have to set up something that will wait until everything
      // is complete before we unlock 
      
      std::unique_lock<std::mutex> adminlock(admin);
      
      std::weak_ptr<openclcacheentry> oclbuffer;
      std::shared_ptr<openclcacheentry> oclbuffer_strong;
      
      /* create arrayinfo key */
      openclarrayinfo arrayinfo=openclarrayinfo(context,arrayptr,nelem);
      
      oclbuffer=buffer_map.at(arrayinfo); /* buffer should exist because should have been created in GetOpenCLBuffer() */

      //std::vector<std::shared_ptr<opencldirtyregion>> *dirtyregions_copy=new std::vector<std::shared_ptr<opencldirtyregion>>();

      //for (auto & dirtyregion : oclbuffer->_dirtyregions) {
      //  /* Copy stuff out of dirtyregions, and wait for all of the condition variables
      //   so that we have waited for all updates to have occured and can release our reference to the lock */
      //  if (!dirtyregion.second->FlushDoneEventComplete && dirtyregion.second->FlushDoneEvent) {
      //    dirtyregions_copy->emplace_back(dirtyregion.second);
      //  }
      //}


      oclbuffer_strong=oclbuffer.lock();
      bool dirtyflag=false;

      if (oclbuffer_strong) {
	// .. if there is anything dirty...
	rangetracker<opencldirtyregion>::iterator dirtyregion;
	for (dirtyregion=oclbuffer_strong->_dirtyregions.begin();dirtyregion != oclbuffer_strong->_dirtyregions.end();dirtyregion++) {
	  if (!dirtyregion->second->FlushDoneEventComplete) {
	    dirtyflag=true;
	  }
	  
	}
      }
      // release adminlock so we can start the thread on a copy (don't otherwise need adminlock from here on) 
      adminlock.unlock();

      if (dirtyflag) {
	
	/* start thread that will hold our locks until all dirty regions are no longer dirty */
	std::shared_ptr<openclcachemanager> shared_this=std::dynamic_pointer_cast<openclcachemanager>(shared_from_this());
	unlock_thread=std::make_shared<std::thread>( [ shared_this,oclbuffer_strong,locks_copy1 ]() {

	  set_thread_name(nullptr,"snde2 ocl cleanup");
	  //std::vector<std::shared_ptr<opencldirtyregion>>::iterator dirtyregion;
	  rangetracker<opencldirtyregion>::iterator dirtyregion;
	  std::unique_lock<std::mutex> lock(shared_this->admin);
	  
	  // Keep on waiting on the first dirty region, removing it once it is complete,
	  // until there is nothing left
	  for (dirtyregion=oclbuffer_strong->_dirtyregions.begin();dirtyregion != oclbuffer_strong->_dirtyregions.end();dirtyregion=oclbuffer_strong->_dirtyregions.begin()) {
	    if (dirtyregion->second->FlushDoneEventComplete) {
	      oclbuffer_strong->_dirtyregions.erase(dirtyregion);
	    } else {
	      dirtyregion->second->complete_condition.wait(lock);
	    }
	    
	  }
	  
	  /* call verify_rwlock_token_set() or similar here, 
	     so that if somehow the set was unlocked prior, we can diagnose the error */
	  
	  if (!check_rwlock_token_set(*locks_copy1)) {
	    throw std::runtime_error("Opencl buffer locks released prematurely");
	  }
	  
	  release_rwlock_token_set(*locks_copy1); /* release write lock */
	  
	  delete locks_copy1;
	  //delete dirtyregions_copy;  
	    
	});
      } else {
	/* no longer need locks_copy1 */
	delete locks_copy1;
      }
      //unlock_thread.detach(); /* thread will exit on its own. It keeps "this" in memory via shared_this */
      
      /* move resulting events to our result array */
      for (auto & ev : flush_events) {
	all_finished.emplace_back(ev); /* move final event(s) to our return list */
      }
      
      flush_events.clear();
      
      for (auto & ev : wait_events) {
	all_finished.emplace_back(ev); /* move final event(s) to our return list */
      }
      
      wait_events.clear();

      
    
      //} else {
      ///* nowhere to delegate writelocks_copy  */
	//delete writelocks_copy;
      //
      ///* Move event prerequisites from output_data_complete to all_finished (reference ownership passes to all_finished) */
	//for (auto & ev : output_data_complete) {
	//  all_finished.emplace_back(ev); /* add prerequisite to return */
      // clRetainEvent(ev); /* must reference-count it since we are returning it */
      //
      //}
      //output_data_complete.clear();
      //
      //}

      // if input_data_not_needed was supplied, also need
      // to retain locks until that event has occurred. 
      if (input_data_not_needed.get()) {
	
	/* in this case locks_copy2 delegated on to callback */	
	input_data_not_needed.setCallback(CL_COMPLETE,snde_opencl_callback,(void *)new std::function<void(cl::Event,cl_int)>([ locks_copy2, input_data_not_needed ](cl::Event event, cl_int event_command_exec_status) { // matching delete is in snde_opencl_callback()
	      /* NOTE: This callback may occur in a separate thread */
	      /* it indicates that the input data is no longer needed */
	  if (event_command_exec_status != CL_COMPLETE) {
	    throw openclerror(event_command_exec_status,"Error from input_data_not_needed prerequisite ");
	    
	  }
	      
	  /* Should call verify_rwlock_token_set() or similar here, 
	     so that if somehow the set was unlocked prior, we can diagnose the error */
	  release_rwlock_token_set(*locks_copy2); /* release read lock */
	  
	  delete locks_copy2;
	      
	      
	} ));
	
	
      } else  {
	/* no longer need locks_copy2  */
	delete locks_copy2;

      }
      
      return std::make_pair(all_finished,unlock_thread);
    }



  OpenCLBuffer_info::OpenCLBuffer_info(//std::shared_ptr<arraymanager> manager,
				       //cl::CommandQueue transferqueue,  /* adds new reference */
				       cl::Buffer mem, /* adds new reference */
				       //void **arrayptr,
				       std::shared_ptr<recording_storage> storage,
				       //rwlock_token_set readlocks,
				       rwlock_token_set locks,
				       std::shared_ptr<openclcacheentry> cacheentry)
    {
      // this->manager=manager;
      //this->cachemanager=get_opencl_cache_manager(manager);
      //clRetainCommandQueue(transferqueue);
      //this->transferqueue=transferqueue;
      //clRetainMemObject(mem);
      this->mem=mem;
      //this->arrayptr=arrayptr;
      this->storage=storage;
      //this->readlocks=readlocks;
      this->locks=locks;
      this->cacheentry=cacheentry;
    }


  OpenCLBufferKey::OpenCLBufferKey(void **_array,snde_index _firstelem,snde_index _numelem) :
    array(_array), firstelem(_firstelem), numelem(_numelem)
  {
    
  }
  
  // equality operator for std::unordered_map
  bool OpenCLBufferKey::operator==(const OpenCLBufferKey b) const
  {
    return b.array==array && b.firstelem==firstelem && b.numelem==numelem;
  }


  OpenCLBuffers::OpenCLBuffers(std::shared_ptr<openclcachemanager> cachemgr,cl::Context context,cl::Device device,rwlock_token_set all_locks) :
    cachemgr(cachemgr),
    context(context),
    device(device),
    all_locks(all_locks),
    empty_invalid(false)
  {
    
  }

  OpenCLBuffers::OpenCLBuffers() :
    empty_invalid(true)
  {
    
  }
  
  // move assignment operator... so we can assign into a default-initialized variable
  OpenCLBuffers & OpenCLBuffers::operator=(OpenCLBuffers &&orig) noexcept
  {
    cachemgr=orig.cachemgr;
    orig.cachemgr=nullptr;
    context=orig.context;
    device=orig.device;
    all_locks=orig.all_locks;
    orig.all_locks=nullptr;
    buffers=orig.buffers;
    orig.buffers.clear();
    fill_events=orig.fill_events;
    orig.fill_events.clear();
    empty_invalid=false;
    orig.empty_invalid=true;

    return *this;
  }
  
  OpenCLBuffers::~OpenCLBuffers()
  {
    std::unordered_map<OpenCLBufferKey,OpenCLBuffer_info,OpenCLBufferKeyHash>::iterator nextbuffer = buffers.begin();
    if (nextbuffer != buffers.end()) {
      fprintf(stderr,"OpenCL Cachemanager Warning: OpenCLBuffers destructor called with residual active buffers\n");
      RemBuffers(cl::Event(),false); // cl::Event() is like a null pointer
    }
    
  }


  cl::Buffer OpenCLBuffers::Mem(void **arrayptr,snde_index firstelem,snde_index numelem)
  {
    cl::Buffer mem=buffers.at(OpenCLBufferKey(arrayptr,firstelem,numelem)).mem;
    //clRetainMemObject(mem);
    return mem;
  }

  std::vector<cl::Event> OpenCLBuffers::FillEvents(void)
  {
    return fill_events;
  }

  
  cl_uint OpenCLBuffers::NumFillEvents(void)
  {
    return (cl_uint)fill_events.size();
  }


  cl_int OpenCLBuffers::SetBufferAsKernelArg(cl::Kernel kernel, cl_uint arg_index, void **arrayptr,snde_index firstelem,snde_index numelem)
  {
    cl::Buffer mem;

    mem=Mem(arrayptr,firstelem,numelem);
    snde_debug(SNDE_DC_OPENCL,"SetBufferAsKernelArg(kernel %llx,arg %u, array %llx,memobj %llx,firstelem %llu, numelem %llu",(unsigned long long)kernel.get(),(unsigned)arg_index,(unsigned long long)(uintptr_t)arrayptr,(unsigned long long)mem.get(),(unsigned long long)firstelem,(unsigned long long)numelem);
    return kernel.setArg(arg_index,sizeof(mem),&mem);
    
  }



  void OpenCLBuffers::AddBufferPortion(std::shared_ptr<recording_storage> storage,snde_index start_elem, snde_index length,bool write,bool write_only/*=false*/)
  // (works on sub-buffers or full buffers)
  // start_elem relative to storage, which may itself be a sub-portion of the OpenCL array 
    {

      //// accumulate preexisting locks + locks in all buffers together
      //for (auto & arrayptr_buf : buffers) {
      //  merge_into_rwlock_token_set(locks,arrayptr_buf.second.readlocks);
      //  merge_into_rwlock_token_set(locks,arrayptr_buf.second.writelocks);
      //}

      rwlock_token_set locks;
      cl::Buffer mem;
      std::vector<cl::Event> new_fill_events;
      std::shared_ptr<openclcacheentry> cacheentry;
      //void **allocatedptr;

      //allocatedptr=manager->allocation_arrays.at(arrayptr);
      storage = storage->get_original_storage(); // operate only on the original storage, not a non-moving reference so that we don't have duplicate cache entries


      assert(start_elem + length <= storage->nelem);

      // sizes relative to the base storage
      snde_index substartelem=storage->base_index + start_elem;
      snde_index subnumelem=length;

      std::tie(locks,mem,new_fill_events,cacheentry) = cachemgr->_GetOpenCLBuffer(storage,all_locks,context,device,substartelem,subnumelem,write,write_only); // (alternatively gets a sub-buffer if need be)
      
      /* move fill events into our master list */
      fill_events.insert(fill_events.end(),new_fill_events.begin(),new_fill_events.end());
      
      //buffers.emplace(std::make_pair(OpenCLBufferKey(arrayptr,indexstart,numelem),
      //			     OpenCLBuffer_info(manager,	        
      //					       mem,
      //					       arrayptr,
      //					       locks)));
      void **arrayptr = storage->lockableaddr();
      buffers.emplace(std::piecewise_construct,
		      std::forward_as_tuple(arrayptr,substartelem,subnumelem),
		      std::forward_as_tuple(mem,
					    storage,
					    locks,cacheentry));
      //clReleaseMemObject(mem); /* remove extra reference */
  
      
      // add this lock to our database of preexisting locks 
      //locks.push_back(buffers[name][1]); 
    }

  //void OpenCLBuffers::AddBuffer(std::shared_ptr<recording_storage> storage, bool write,bool write_only/*=false*/)
  //{
  //  AddSubBuffer(storage,0,SNDE_INDEX_INVALID,write,write_only);
  //}

  
  void OpenCLBuffers::AddBuffer(std::shared_ptr<recording_storage> storage,bool write,bool write_only/*=false*/)
  // (works on sub-buffers or full buffers)
  {
    AddBufferPortion(storage,0,storage->nelem,write,write_only);
    

  }

  cl_int OpenCLBuffers::AddBufferPortionAsKernelArg(std::shared_ptr<recording_storage> storage,snde_index start_elem, snde_index length,cl::Kernel kernel,cl_uint arg_index,bool write,bool write_only/*=false*/)
  // start_elem relative to storage, which may itself be a sub-portion of the OpenCL array 
  {
    AddBufferPortion(storage,start_elem,length,write,write_only);
    void **arrayptr = storage->lockableaddr();

    assert(start_elem+length <= storage->nelem);
    snde_index indexstart = storage->base_index + start_elem;
    snde_index numelem = length;
    return SetBufferAsKernelArg(kernel,arg_index,arrayptr,indexstart,numelem);
  }
  
  cl_int OpenCLBuffers::AddBufferPortionAsKernelArg(std::shared_ptr<ndarray_recording_ref> ref,snde_index portion_start,snde_index portion_len,cl::Kernel kernel,cl_uint arg_index,bool write,bool write_only)
  {
    snde_index start_elem = ref->ndinfo()->base_index - ref->storage->base_index + portion_start;
    snde_index size = portion_len; // note flattened_size() vs flattened_length()
    
    return AddBufferPortionAsKernelArg(ref->storage,start_elem,size,kernel,arg_index,write,write_only);
  }

  cl_int OpenCLBuffers::AddBufferPortionAsKernelArg(std::shared_ptr<multi_ndarray_recording> rec,size_t arraynum,snde_index portion_start,snde_index portion_len,cl::Kernel kernel,cl_uint arg_index,bool write,bool write_only)
  {
    snde_index start_elem=rec->ndinfo(arraynum)->base_index - rec->storage.at(arraynum)->base_index + portion_start;
    snde_index size = portion_len;
    return AddBufferPortionAsKernelArg(rec->storage.at(arraynum),start_elem,size,kernel,arg_index,write,write_only); // note flattened_size() vs flattened_length()
  }

  cl_int OpenCLBuffers::AddBufferPortionAsKernelArg(std::shared_ptr<multi_ndarray_recording> rec,std::string arrayname,snde_index portion_start,snde_index portion_len,cl::Kernel kernel,cl_uint arg_index,bool write,bool write_only)
  {
    size_t arraynum = rec->name_mapping.at(arrayname);
    
    snde_index start_elem=rec->ndinfo(arraynum)->base_index - rec->storage.at(arraynum)->base_index + portion_start;
    snde_index size = portion_len;
    return AddBufferPortionAsKernelArg(rec->storage.at(arraynum),start_elem,size,kernel,arg_index,write,write_only); // note flattened_size() vs flattened_length()
  }



  cl_int OpenCLBuffers::AddBufferAsKernelArg(std::shared_ptr<ndarray_recording_ref> ref,cl::Kernel kernel,cl_uint arg_index,bool write,bool write_only)
  {
    snde_index start_elem = ref->ndinfo()->base_index - ref->storage->base_index;
    snde_index size = ref->layout.flattened_size(); // note flattened_size() vs flattened_length()
    
    return AddBufferPortionAsKernelArg(ref->storage,start_elem,size,kernel,arg_index,write,write_only);
  }

  cl_int OpenCLBuffers::AddBufferAsKernelArg(std::shared_ptr<multi_ndarray_recording> rec,size_t arraynum,cl::Kernel kernel,cl_uint arg_index,bool write,bool write_only)
  {
    snde_index start_elem=rec->ndinfo(arraynum)->base_index - rec->storage.at(arraynum)->base_index;
    snde_index size = rec->layouts.at(arraynum).flattened_size();
    return AddBufferPortionAsKernelArg(rec->storage.at(arraynum),start_elem,size,kernel,arg_index,write,write_only); // note flattened_size() vs flattened_length()
  }

  cl_int OpenCLBuffers::AddBufferAsKernelArg(std::shared_ptr<multi_ndarray_recording> rec,std::string arrayname,cl::Kernel kernel,cl_uint arg_index,bool write,bool write_only)
  {
    size_t arraynum = rec->name_mapping.at(arrayname);
    
    snde_index start_elem=rec->ndinfo(arraynum)->base_index - rec->storage.at(arraynum)->base_index;
    snde_index size = rec->layouts.at(arraynum).flattened_size();
    return AddBufferPortionAsKernelArg(rec->storage.at(arraynum),start_elem,size,kernel,arg_index,write,write_only); // note flattened_size() vs flattened_length()
  }

  

  //cl_int OpenCLBuffers::AddBufferAsKernelArg(std::shared_ptr<arraymanager> manager,cl::Kernel kernel,cl_uint arg_index,void **arrayptr,bool write,bool write_only/*=false*/)
  //{
  //  AddBuffer(manager,arrayptr,write,write_only);
  //  return SetBufferAsKernelArg(kernel,arg_index,arrayptr,0,SNDE_INDEX_INVALID);
  //}
  

  /* This indicates that the array has been written to by an OpenCL kernel, 
     and that therefore it needs to be copied back into CPU memory */
  void OpenCLBuffers::BufferPortionDirty(std::shared_ptr<recording_storage> storage,snde_index start_elem, snde_index length)
  // Works on buffers and subbuffers
  {

    void **arrayptr = storage->lockableaddr();

    assert(start_elem + length <= storage->nelem);

    //snde_index indexstart = storage->base_index;
    //snde_index numelem = storage->nelem;

    snde_index indexstart = storage->base_index+start_elem;
    snde_index numelem = length;
    
    //snde_index total_nelem = storage->lockablenelem();

    
    OpenCLBuffer_info &info=buffers.at(OpenCLBufferKey(arrayptr,storage->base_index+start_elem,numelem /*storage->nelem*/));
    
    
    //cachemgr->mark_as_gpu_modified(context,storage);
    std::lock_guard<std::mutex> cachemgr_admin(cachemgr->admin);
    
    info.cacheentry->mark_as_gpu_modified(storage,storage->base_index + start_elem,numelem);
    
  }


  void OpenCLBuffers::BufferDirty(std::shared_ptr<recording_storage> storage)
  // Works on buffers and subbuffers
  {
    BufferPortionDirty(storage,0,storage->nelem);
  }

  void OpenCLBuffers::BufferPortionDirty(std::shared_ptr<ndarray_recording_ref> ref,snde_index portion_start, snde_index portion_len)
  {

    snde_index start_elem = ref->ndinfo()->base_index - ref->storage->base_index + portion_start;
    snde_index size = portion_len;

    BufferPortionDirty(ref->storage,start_elem,size);
  }
  
  void OpenCLBuffers::BufferPortionDirty(std::shared_ptr<multi_ndarray_recording> rec,size_t arraynum,snde_index portion_start, snde_index portion_len)
  {

    snde_index start_elem=rec->ndinfo(arraynum)->base_index - rec->storage.at(arraynum)->base_index + portion_start;
    snde_index size = portion_len;

    BufferPortionDirty(rec->storage.at(arraynum),start_elem,size);
  }

  void OpenCLBuffers::BufferPortionDirty(std::shared_ptr<multi_ndarray_recording> rec,std::string arrayname,snde_index portion_start, snde_index portion_len)
  {
    size_t arraynum = rec->name_mapping.at(arrayname);


    snde_index start_elem=rec->ndinfo(arraynum)->base_index - rec->storage.at(arraynum)->base_index + portion_start;
    snde_index size = portion_len;

    BufferPortionDirty(rec->storage.at(arraynum),start_elem,size);
  }

  
  void OpenCLBuffers::BufferDirty(std::shared_ptr<ndarray_recording_ref> ref)
  {
    snde_index start_elem = ref->ndinfo()->base_index - ref->storage->base_index;
    snde_index size = ref->layout.flattened_size(); // note flattened_size() vs flattened_length()
    
    BufferPortionDirty(ref->storage,start_elem,size);
  }

  void OpenCLBuffers::BufferDirty(std::shared_ptr<multi_ndarray_recording> rec,size_t arraynum)
  {
    snde_index start_elem=rec->ndinfo(arraynum)->base_index - rec->storage.at(arraynum)->base_index;
    snde_index size = rec->layouts.at(arraynum).flattened_size();

    BufferPortionDirty(rec->storage.at(arraynum),start_elem,size);
  }

  void OpenCLBuffers::BufferDirty(std::shared_ptr<multi_ndarray_recording> rec,std::string arrayname)
  {
    size_t arraynum = rec->name_mapping.at(arrayname);
    snde_index start_elem=rec->ndinfo(arraynum)->base_index - rec->storage.at(arraynum)->base_index;
    snde_index size = rec->layouts.at(arraynum).flattened_size();
    
    BufferPortionDirty(rec->storage.at(arraynum),start_elem,size);
  }

  /* This indicates that part the array has been written to by an OpenCL kernel, 
     and that therefore it needs to be copied back into CPU memory */
  /*
  void OpenCLBuffers::BufferDirty(std::shared_ptr<recording_storage> storage,snde_index pos, snde_index len)
  // Works on buffers and subbuffers
  {

    void **arrayptr = storage->lockableaddr();
    snde_index indexstart = storage->base_index;
    snde_index numelem = storage->nelem;
    //snde_index total_nelem = storage->lockablenelem();
    
    OpenCLBuffer_info &info=buffers.at(OpenCLBufferKey(arrayptr,storage->base_index,storage->nelem));
    
    assert(len <= storage->nelem);
    
    cachemgr->mark_as_gpu_modified(context,arrayptr,storage->base_index+pos,len);
    
    }*/


  //void OpenCLBuffers::SubBufferDirty(void **arrayptr,snde_index sb_pos,snde_index sb_len)
  ///* This indicates that the array region has been written to by an OpenCL kernel, 
  //   and that therefore it needs to be copied back into CPU memory */
  // {
  //  OpenCLBuffer_info &info=buffers.at(OpenCLBufferKey(arrayptr,sb_pos,sb_len));
  // 
  //  info.cachemanager->mark_as_gpu_modified(context,device,arrayptr,sb_pos,sb_len);
  // 
  //}
  //
  //void OpenCLBuffers::SubBufferDirty(void **arrayptr,snde_index sb_pos,snde_index sb_len,snde_index dirtypos,snde_index dirtylen)
  /* This indicates that the array region has been written to by an OpenCL kernel, 
     and that therefore it needs to be copied back into CPU memory */
  //{
  //  OpenCLBuffer_info &info=buffers.at(OpenCLBufferKey(arrayptr,sb_pos,sb_len));
  //  
  //  info.cachemanager->mark_as_gpu_modified(context,device,arrayptr,sb_pos+dirtypos,dirtylen);
  //}



  std::pair<std::vector<cl::Event>,std::vector<cl::Event>> OpenCLBuffers::FlushBuffer(std::shared_ptr<recording_storage> storage,std::vector<cl::Event> explicit_prerequisites)
  {
    void **arrayptr = storage->lockableaddr();
    OpenCLBuffer_info &info=buffers.at(OpenCLBufferKey(arrayptr,storage->base_index,storage->nelem));
    
    return cachemgr->FlushWrittenOpenCLBuffer(context,device,storage,explicit_prerequisites);
  }



  void OpenCLBuffers::ForgetBuffer(std::shared_ptr<recording_storage> storage,cl::Event data_not_needed)
  // Forget (and queue the deletion of) a buffer so that we don't waste our time
  // transferring its contents anymore -- presumably because our caller has
  // determined that it will never be needed again.
  // This is usually used for temporary anonymous recordings
  // that don't need to be kept in the cache. 
  {
    /* Remove and unlock buffer */
    
    
    
    OpenCLBufferKey Key(storage->lockableaddr(),storage->base_index,storage->nelem);
    auto buffers_it = buffers.find(Key);

    assert(buffers_it != buffers.end()); // Shouldn't be calling this unless you have already gotten access to the buffer. 


    OpenCLBuffer_info &info=buffers_it->second;
    
    // ForgetOpenCLBuffer allows a buffer to be freed once data_not_needed has passed if all shared_pointers expire 
    cachemgr->ForgetOpenCLBuffer(info.locks,context,device,info.mem,storage,data_not_needed);
          
    /* remove from hash table so as to expire the shared pointer kept in this OpenCLBuffers object*/
    buffers.erase(Key);

    
    // Once this is done the buffer will disappear once data_not_needed fires
    // (unless it is referenced elsewhere)
    
  }



  
  std::vector<cl::Event> OpenCLBuffers::RemBuffer(std::shared_ptr<recording_storage> storage,snde_index firstelem, snde_index numelem,cl::Event input_data_not_needed,const std::vector<cl::Event> &output_data_complete,bool wait)
    /* Either specify wait=true, then you can explicitly unlock_rwlock_token_set() your locks because you know they're done, 
       or specify wait=false in which case things may finish later. The only way to make sure they are finished is 
       to obtain a new lock on the same items */
      
    {
      /* Remove and unlock buffer */

      
      
      //OpenCLBufferKey Key(storage->lockableaddr(),storage->base_index,storage->nelem);
      OpenCLBufferKey Key(storage->lockableaddr(),firstelem,numelem);
      OpenCLBuffer_info &info=buffers.at(Key);
      
      std::vector<cl::Event> all_finished;
      std::shared_ptr<std::thread> wrapupthread;
      std::tie(all_finished,wrapupthread)=cachemgr->ReleaseOpenCLBuffer(info.locks,context,device,info.mem,storage,input_data_not_needed,output_data_complete);
      
      if (wait) {
	cl::Event::waitForEvents(all_finished);
	if (wrapupthread && wrapupthread->joinable()) {
	  wrapupthread->join();
	}
      } else {
	if (wrapupthread && wrapupthread->joinable()) {
	  wrapupthread->detach();
	}
      }

      //for (auto & ev: all_finished) {
      //clReleaseEvent(ev);
      //}
      
      /* remove from hash table */
      buffers.erase(Key);

      return all_finished;
    }


  void OpenCLBuffers::ForgetBuffer(std::shared_ptr<ndarray_recording_ref> ref,cl::Event data_not_needed)
  {
    ForgetBuffer(ref->storage,data_not_needed);
  }

  
  std::vector<cl::Event> OpenCLBuffers::RemBuffer(std::shared_ptr<ndarray_recording_ref> ref,cl::Event input_data_not_needed,const std::vector<cl::Event> &output_data_complete,bool wait)
  {
    return RemBuffer(ref->storage,ref->storage->base_index,ref->storage->nelem,input_data_not_needed,output_data_complete,wait);
  }

  void OpenCLBuffers::ForgetBuffer(std::shared_ptr<multi_ndarray_recording> rec,std::string arrayname,cl::Event data_not_needed)
  {
    ForgetBuffer(rec->storage.at(rec->name_mapping.at(arrayname)),data_not_needed);

  }

  std::vector<cl::Event> OpenCLBuffers::RemBuffer(std::shared_ptr<multi_ndarray_recording> rec,std::string arrayname,cl::Event input_data_not_needed,const std::vector<cl::Event> &output_data_complete,bool wait)
  {
    std::shared_ptr<recording_storage> stor = rec->storage.at(rec->name_mapping.at(arrayname));
    return RemBuffer(stor,stor->base_index,stor->nelem,input_data_not_needed,output_data_complete,wait);

  }

  //void OpenCLBuffers::RemBuffer(void **arrayptr,cl::Event input_data_not_needed,std::vector<cl::Event> output_data_complete,bool wait)
  //{
  //  RemSubBuffer(arrayptr,0,SNDE_INDEX_INVALID,input_data_not_needed,output_data_complete,wait);
  //}
  
  std::vector<cl::Event> OpenCLBuffers::RemBuffers(cl::Event input_data_not_needed,std::vector<cl::Event> output_data_complete,bool wait)
  {
    std::vector<cl::Event> all_finished,all_finished_buf;
    
    for (std::unordered_map<OpenCLBufferKey,OpenCLBuffer_info>::iterator nextbuffer = buffers.begin();
	 nextbuffer != buffers.end();) {
      std::unordered_map<OpenCLBufferKey,OpenCLBuffer_info>::iterator thisbuffer=nextbuffer;
      nextbuffer++;

      assert(thisbuffer->first.array == thisbuffer->second.storage->lockableaddr());
      all_finished_buf = RemBuffer(thisbuffer->second.storage,thisbuffer->first.firstelem,thisbuffer->first.numelem,input_data_not_needed,output_data_complete,wait);

      // Move entries from all_finished_buf onto all_finished
      all_finished.insert(all_finished.end(),all_finished_buf.begin(),all_finished_buf.end());
      
    }
    return all_finished;
  }


  std::vector<cl::Event> OpenCLBuffers::RemBuffers(cl::Event input_data_not_needed,cl::Event output_data_complete,bool wait)
  {
    std::vector<cl::Event> output_data_complete_vector{output_data_complete};
    return RemBuffers(input_data_not_needed,output_data_complete_vector,wait);
  }

  std::vector<cl::Event> OpenCLBuffers::RemBuffers(cl::Event input_data_not_needed,bool wait)
  {
    std::vector<cl::Event> output_data_complete_vector{};
    return RemBuffers(input_data_not_needed,output_data_complete_vector,wait);
  }
  
  
};



