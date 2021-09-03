#ifndef SNDE_OPENCLCACHEMANAGER_HPP
#define SNDE_OPENCLCACHEMANAGER_HPP

#include <cstring>

#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#include <CL/opencl.h>

#include "snde/arraymanager.hpp"
#include "snde/rangetracker.hpp"
#include "snde/snde_error_opencl.hpp"
#include "snde/opencl_utils.hpp"

extern "C" void snde_opencl_callback(cl_event event, cl_int event_command_exec_status, void *user_data);


/* Distinction between "dirty" and "modified"

 * "Dirty" can mean somebody else has modified data, so we can't use it until we recopy it
 * "Dirty" can mean our cache has been modified, and needs to be transfered back to the master copy
 * "Modified" means that the master copy has been changed, and that notifications to others 
   that their copies are dirty have been sent out
 
 * "Flush" operation triggers "Dirty->Modified". State is "Modified" once "FlushDoneEventComplete" flag is true
 * Everything must be "modified" prior to release
*/

namespace snde {

  class opencldirtyregion {
  public:
    /* contents locked by cachemanager's admin mutex */
    snde_index regionstart;
    snde_index regionend;
    cl_event FlushDoneEvent;
    bool FlushDoneEventComplete;
    std::condition_variable complete_condition; // associated with the cachemanager's admin mutex
    
    opencldirtyregion& operator=(const opencldirtyregion &)=delete; /* copy assignment disabled */
    
    opencldirtyregion(snde_index regionstart,snde_index regionend)
    {
      this->regionstart=regionstart;
      this->regionend=regionend;
      FlushDoneEvent=NULL;
      FlushDoneEventComplete=false;
      //fprintf(stderr,"Create opencldirtyregion(%d,%d)\n",regionstart,regionend);
    }

    opencldirtyregion(const opencldirtyregion &orig)
    {
      regionstart=orig.regionstart;
      regionend=orig.regionend;
      FlushDoneEvent=orig.FlushDoneEvent;
      if (FlushDoneEvent) {
	clRetainEvent(FlushDoneEvent);       
      }
      FlushDoneEventComplete=orig.FlushDoneEventComplete;
    }
    
    bool attempt_merge(opencldirtyregion &later)
    {
      return false; // ***!!! Should probably implement this
    }
    
    std::shared_ptr<opencldirtyregion> sp_breakup(snde_index breakpoint)
    /* breakup method ends this region at breakpoint and returns
       a new region starting at from breakpoint to the prior end */
    {
      // note that it will be an error to do a breakup while there
      // are flushdoneevents pending
      std::shared_ptr<opencldirtyregion> newregion;

      //fprintf(stderr,"Create opencldirtyregion.sp_breakup(%d,%d)->(%d,%d)\n",regionstart,regionend,regionstart,breakpoint);
      assert(!FlushDoneEvent);
      assert(breakpoint > regionstart && breakpoint < regionend);

      newregion=std::make_shared<opencldirtyregion>(breakpoint,regionend);
      this->regionend=breakpoint;
      
      return newregion;
    }
      ~opencldirtyregion()
    {
      if (FlushDoneEvent) {
	clReleaseEvent(FlushDoneEvent);
	FlushDoneEvent=NULL;
      }
    }
    

  };
  

  class openclregion {
  public:
    snde_index regionstart;
    snde_index regionend;
    cl_event fill_event; // if not NULL, indicates that there is a pending operation to copy data into this region...
    // if you hold a write lock, it should not be possible for there to
    // be a fill_event except at your request, because a fill_event
    // requires at least a read lock and in order to get your write
    // lock you would have waited for all (other) read locks to be released

    openclregion(const openclregion &)=delete; /* copy constructor disabled */
    openclregion& operator=(const openclregion &)=delete; /* copy assignment disabled */
    
    openclregion(snde_index regionstart,snde_index regionend)
    {
      this->regionstart=regionstart;
      this->regionend=regionend;
      fill_event=NULL;
    }

    ~openclregion()
    {
      if (fill_event) {
	clReleaseEvent(fill_event);
	fill_event=NULL;
      }
    }
    
    bool attempt_merge(openclregion &later)
    {
      assert(later.regionstart==regionend);

      if (!fill_event && !later.fill_event) {      
	regionend=later.regionend;
	return true;
      }
      return false;
    }

    std::shared_ptr<openclregion> sp_breakup(snde_index breakpoint)
    /* breakup method ends this region at breakpoint and returns
       a new region starting at from breakpoint to the prior end */
    {
      std::shared_ptr<openclregion> newregion(new openclregion(breakpoint,regionend));
      regionend=breakpoint;

      if (fill_event) {
	newregion->fill_event=fill_event;
	clRetainEvent(newregion->fill_event);
      }
      

      return newregion;
    }

  };
  


  /* openclcachemanager manages access to OpenCL buffers
     of type CL_MEM_READ_WRITE, and helps manage
     the copying back-and-forth of such buffers
     to main memory */

  /* openclarrayinfo is used as a key for a std::unordered_map
     to look up opencl buffers. It also reference counts the 
     context so that it doesn't disappear on us */
  class openclarrayinfo {
  public:
    cl_context context; /* when on openclcachemanager's arrayinfomap, held in memory by clRetainContext() */
    cl_device_id device;
    /* could insert some flag to indicate use of zero-copy memory */
    void **arrayptr;
    
    openclarrayinfo(cl_context context, cl_device_id device,void **arrayptr)
    {
      this->context=context;
      this->device=device;
      clRetainContext(this->context); /* increase refcnt */
      clRetainDevice(this->device);
      this->arrayptr=arrayptr;
    }
    openclarrayinfo(const openclarrayinfo &orig) /* copy constructor */
    {
      context=orig.context;
      clRetainContext(context);
      device=orig.device;
      clRetainDevice(device);
      arrayptr=orig.arrayptr;
    }
    
    openclarrayinfo& operator=(const openclarrayinfo &orig) /* copy assignment operator */
    {
      clReleaseContext(context);
      context=orig.context;
      clRetainContext(context);
      arrayptr=orig.arrayptr;
      device=orig.device;
      clRetainDevice(device);

      return *this;
    }

    // equality operator for std::unordered_map
    bool operator==(const openclarrayinfo b) const
    {
      return b.context==context && b.arrayptr==arrayptr && b.device==device;
    }

    ~openclarrayinfo() {
      clReleaseContext(context);
      clReleaseDevice(device);
    }
  };



  // Need to provide hash implementation for openclarrayinfo so
  // it can be used as a std::unordered_map key
  
  struct openclarrayinfo_hash {
    size_t operator()(const snde::openclarrayinfo & x) const
    {
      return std::hash<void *>{}((void *)x.context) + std::hash<void *>{}((void *)x.device) + std::hash<void *>{}((void *)x.arrayptr);
    }
  };
  struct openclarrayinfo_equal {
    bool operator()(const snde::openclarrayinfo & x,const snde::openclarrayinfo &y) const
    {
      return x.context==y.context && x.device==y.device && x.arrayptr==y.arrayptr;
    }
  };
  
  
  
  
  class _openclbuffer {
    // openclbuffer is our lowlevel wrapper used by openclcachemanager
    // for its internal buffer table
    
    // access serialization managed by our parent openclcachemanager's
    // admin mutex, which should be locked when these methods
    // (except realloc callback) are called.
  public:
    cl_mem buffer; /* cl reference count owned by this object */
    size_t elemsize;
    rangetracker<openclregion> invalidity;
    void **arrayptr;
    std::shared_ptr<std::function<void(snde_index)>> pool_realloc_callback;
    std::weak_ptr<allocator> alloc; /* weak pointer to the allocator because we don't want to inhibit freeing of this (all we need it for is our reallocation callback) */
    rangetracker<opencldirtyregion> _dirtyregions; /* track dirty ranges during a write (all regions that are locked for write... Note that persistent iterators (now pointers) may be present if the FlushDoneEvent exists but the FlushDoneEventComplete is false  */

    
    _openclbuffer(const _openclbuffer &)=delete; /* copy constructor disabled */
    
    _openclbuffer(cl_context context,std::shared_ptr<allocator> alloc,snde_index total_nelem,size_t elemsize,void **arrayptr, std::mutex *arraymanageradminmutex)
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
      
      buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,nelem*elemsize, NULL /* *arrayptr */,&errcode_ret);

      if (errcode_ret != CL_SUCCESS) {
	throw openclerror(errcode_ret,(std::string)"Error creating buffer of size %d",(long)(nelem*elemsize));
      }
      
      invalidity.mark_all(nelem);

      /* Need to register for notification of realloc
	 so we can re-allocate the buffer! */
      if (alloc) {
	pool_realloc_callback=std::make_shared<std::function<void(snde_index)>>([this,context,arraymanageradminmutex](snde_index total_nelem) {
	  /* Note: we're not managing context, but presumably the openclarrayinfo that is our key in buffer_map will prevent the context from being freed */
	  
	  std::lock_guard<std::mutex> lock(*arraymanageradminmutex);
	  snde_index numelem;
	  cl_int lambdaerrcode_ret=CL_SUCCESS;
	  numelem=total_nelem; /* determine new # of elements */
	  clReleaseMemObject(buffer); /* free old buffer */

	  /* allocate new buffer */
	  buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,numelem*this->elemsize,nullptr /* *this->arrayptr */, &lambdaerrcode_ret);

	  if (lambdaerrcode_ret != CL_SUCCESS) {
	    throw openclerror(lambdaerrcode_ret,"Error expanding buffer to size %d",(long)(numelem*this->elemsize));
	  }

	  /* Mark entire buffer as invalid */
	  invalidity.mark_all(numelem);
	  
	});
	alloc->register_pool_realloc_callback(pool_realloc_callback);
      }
      
    }

    void mark_as_gpu_modified(snde_index pos,snde_index nelem)
    {
      /* ***!!!!! Should really check here to make sure that we're not modifying
	 any dirtyregions elements that are marked as done or have a FlushDoneEvent */
      _dirtyregions.mark_region(pos,nelem);
    }

    ~_openclbuffer()
    {
      std::shared_ptr<allocator> alloc_strong = alloc.lock();
      if (alloc_strong) { /* if allocator already freed, it would take care of its reference to the callback */ 
	alloc_strong->unregister_pool_realloc_callback(pool_realloc_callback);
      }
      
      clReleaseMemObject(buffer);
      buffer=NULL;
    }
  };


  
  
  /* openclcachemanager manages opencl caches for arrays 
     (for now) managed by a single arraymanager/lockmanager  */ 
  class openclcachemanager : public cachemanager {
  public: 
    //std::shared_ptr<memallocator> _memalloc;
    // locker defined by arraymanager base class
    std::mutex admin; /* lock our data structures, including buffer_map and descendents. We are allowed to call allocators/lock managers while holding 
			 this mutex, but they are not allowed to call us and only lock their own data structures, 
			 so a locking order is ensured and no deadlock is possible */
    std::unordered_map<context_device,cl_command_queue_wrapped,context_device_hash,context_device_equal> queue_map;
    
    std::unordered_map<openclarrayinfo,std::shared_ptr<_openclbuffer>,openclarrayinfo_hash,openclarrayinfo_equal> buffer_map;

    std::weak_ptr<arraymanager> manager; /* Used for looking up allocators and accessing lock manager (weak to avoid creating a circular reference) */ 

    std::unordered_map<void **,std::vector<openclarrayinfo>> buffers_by_array;
    
    
    openclcachemanager(std::shared_ptr<arraymanager> manager) {
      //_memalloc=memalloc;
      //locker = std::make_shared<lockmanager>();
      this->manager=manager;
    }
    openclcachemanager(const openclcachemanager &)=delete; /* copy constructor disabled */
    openclcachemanager& operator=(const openclcachemanager &)=delete; /* assignment disabled */


    cl_command_queue _get_queue_no_ref(cl_context context,cl_device_id device)
    // internal version for when adminlock is alread held 
    {
    
      auto c_d = context_device(context,device);
      auto iter = queue_map.find(c_d);
      if (iter==queue_map.end()) {
	// no entry... create one
	cl_int clerror=0;
	cl_command_queue newqueue=clCreateCommandQueue(context,device,0,&clerror);
	queue_map.emplace(c_d,cl_command_queue_wrapped(newqueue));
	
      }
      return queue_map[c_d].get_noref();
    }
    
    cl_command_queue get_queue_no_ref(cl_context context,cl_device_id device)
    {
      std::lock_guard<std::mutex> adminlock(admin);
      return _get_queue_no_ref(context,device);
      
    }

    cl_command_queue get_queue_ref(cl_context context,cl_device_id device)
    {
      std::lock_guard<std::mutex> adminlock(admin);
      auto c_d = context_device(context,device);
      auto iter = queue_map.find(c_d);
      if (iter==queue_map.end()) {
	// no entry... create one
	cl_int clerror=0;
	cl_command_queue newqueue=clCreateCommandQueue(context,device,0,&clerror);
	queue_map.emplace(c_d,cl_command_queue_wrapped(newqueue));
	
      }
      return queue_map[c_d].get_ref();
    }

    
    cl_command_queue_wrapped get_queue(cl_context context,cl_device_id device)
    {
      std::lock_guard<std::mutex> adminlock(admin);
      auto c_d = context_device(context,device);
      auto iter = queue_map.find(c_d);
      if (iter==queue_map.end()) {
	// no entry... create one
	cl_int clerror=0;
	cl_command_queue newqueue=clCreateCommandQueue(context,device,0,&clerror);
	queue_map.emplace(c_d,cl_command_queue_wrapped(newqueue));
	
      }
      return queue_map[c_d];
    }

    void mark_as_dirty_except_buffer(std::shared_ptr<_openclbuffer> exceptbuffer,void **arrayptr,snde_index pos,snde_index numelem)
    /* marks an array region (with exception of particular buffer) as needing to be updated from CPU copy */
    /* This is typically used after our CPU copy has been updated from exceptbuffer, to push updates out to all of the other buffers */
    {
      std::unique_lock<std::mutex> adminlock(admin);

      std::unordered_map<openclarrayinfo,std::shared_ptr<_openclbuffer>,openclarrayinfo_hash,openclarrayinfo_equal>::iterator buffer;

      /* Mark all of our buffers with this region as invalid */
      for (auto & arrayinfo : buffers_by_array[arrayptr]) {

	buffer=buffer_map.find(arrayinfo);

	if (exceptbuffer == nullptr || exceptbuffer.get() != buffer->second.get()) {
	  
	  buffer->second->invalidity.mark_region(pos,pos+numelem);
	}
      }
      

      // buffer.second is a shared_ptr to an _openclbuffer
      
    }
      
    virtual void mark_as_gpu_modified(cl_context context, cl_device_id device, void **arrayptr,snde_index pos,snde_index len)
    {
      openclarrayinfo arrayinfo=openclarrayinfo(context,device,arrayptr);
      
      std::unique_lock<std::mutex> adminlock(admin);

      std::shared_ptr<_openclbuffer> buffer=buffer_map.at(arrayinfo);
      buffer->mark_as_gpu_modified(pos,len);

    }
    
    
    virtual void mark_as_dirty(std::shared_ptr<arraymanager> specified_manager,void **arrayptr,snde_index pos,snde_index numelem)
    {
      /* marks an array region as needing to be updated from CPU copy */
      /* This is typically used if the CPU copy is updated directly */
      std::shared_ptr<arraymanager> manager_strong(manager);
      assert(specified_manager==manager_strong);
      mark_as_dirty_except_buffer(nullptr,arrayptr,pos,numelem);
    }
  
    void _TransferInvalidRegions(cl_context context, cl_device_id device,std::shared_ptr<_openclbuffer> oclbuffer,void **arrayptr,snde_index firstelem,snde_index numelem,std::vector<cl_event> &ev)
    // internal use only... initiates transfers of invalid regions prior to setting up a read buffer
    // WARNING: operates in-place on prerequisite event vector ev
    // assumes admin lock is held
    {

      std::shared_ptr<arraymanager> manager_strong(manager); /* as manager references us, it should always exist while we do. This will throw an exception if it doesn't */
      rangetracker<openclregion>::iterator invalidregion;
      
      if (numelem==SNDE_INDEX_INVALID) {
	numelem=(*manager_strong->allocators())[arrayptr].alloc->total_nelem()-firstelem;
      }

      /* transfer any relevant areas that are invalid and not currently pending */
	
      rangetracker<openclregion> invalid_regions=oclbuffer->invalidity.iterate_over_marked_portions(firstelem,numelem);
	
      for (auto & invalidregion: invalid_regions) {
	
	/* Perform operation to transfer data to buffer object */
	
	/* wait for all other events as it is required for a 
	   partial write to be meaningful */
	
	if (!invalidregion.second->fill_event) {
	  snde_index offset=invalidregion.second->regionstart*oclbuffer->elemsize;
	  cl_event newevent=NULL;

	  cl_event *evptr=NULL;
	  if (ev.size() > 0) {
	    evptr=&ev[0];
	  }
	  
	  clEnqueueWriteBuffer(_get_queue_no_ref(context,device),oclbuffer->buffer,CL_FALSE,offset,(invalidregion.second->regionend-invalidregion.second->regionstart)*oclbuffer->elemsize,(char *)*arrayptr + offset,ev.size(),evptr,&newevent);
	  
	  /* now that it is enqueued we can replace our event list 
	     with this newevent */
	  
	  for (auto & oldevent : ev) {
	    clReleaseEvent(oldevent);
	  }
	  ev.clear();
	  
	  ev.emplace_back(newevent); /* add new event to our set (this eats our ownership) */
	  
	  clRetainEvent(newevent); /* increment reference count for fill_event pointer */
	  invalidregion.second->fill_event=newevent;
	}
	
      }

      clFlush(_get_queue_no_ref(context,device));
    }

    std::shared_ptr<_openclbuffer> _GetBufferObject(cl_context context, cl_device_id device,void **arrayptr)
    {
      // internal use only; assumes admin lock is held;

      //fprintf(stderr,"_GetBufferObject(0x%lx,0x%lx,0x%lx)... buffer_map.size()=%u, tid=0x%lx admin->__owner=0x%lx\n",(unsigned long)context,(unsigned long)device,(unsigned long)arrayptr,(unsigned)buffer_map.size(),(unsigned long)((pid_t)syscall(SYS_gettid)),(unsigned long) admin._M_mutex.__data.__owner);

      std::shared_ptr<_openclbuffer> oclbuffer;
      openclarrayinfo arrayinfo=openclarrayinfo(context,device,arrayptr);
      std::shared_ptr<arraymanager> manager_strong(manager); /* as manager references us, it should always exist while we do. This will throw an exception if it doesn't */
      allocationinfo thisalloc = (*manager_strong->allocators()).at(arrayptr);
      //std::shared_ptr<allocator> alloc=thisalloc.alloc;
	    
      std::unordered_map<openclarrayinfo,std::shared_ptr<_openclbuffer>,openclarrayinfo_hash,openclarrayinfo_equal>::iterator buffer;
      buffer=buffer_map.find(arrayinfo);
      if (buffer == buffer_map.end()) {
	/* need to create buffer */
	oclbuffer=std::make_shared<_openclbuffer>(context,thisalloc.alloc,thisalloc.totalnelem(),thisalloc.elemsize,arrayptr,&admin);
	buffer_map[arrayinfo]=oclbuffer;
	
	//fprintf(stderr,"_GetBufferObject(0x%lx,0x%lx,0x%lx) created buffer_map entry. buffer_map.size()=%u, tid=0x%lx admin->__owner=0x%lx\n",(unsigned long)context,(unsigned long)device,(unsigned long)arrayptr,(unsigned)buffer_map.size(),(unsigned long)((pid_t)syscall(SYS_gettid)),(unsigned long) admin._M_mutex.__data.__owner);

	std::vector<openclarrayinfo> & buffers_for_this_array=buffers_by_array[arrayinfo.arrayptr];
	buffers_for_this_array.push_back(arrayinfo);
	
      } else {
	oclbuffer=buffer_map.at(arrayinfo);
      }

      return oclbuffer;
    }
  
      
    std::tuple<rwlock_token_set,cl_mem,std::vector<cl_event>> GetOpenCLSubBuffer(rwlock_token_set alllocks,cl_context context, cl_device_id device, void **arrayptr,snde_index startelem,snde_index numelem,bool write,bool write_only=false)
    {
      std::unique_lock<std::mutex> adminlock(admin);
      
      openclarrayinfo arrayinfo=openclarrayinfo(context,device,arrayptr);
      
      std::shared_ptr<arraymanager> manager_strong(manager); /* as manager references us, it should always exist while we do. This will throw an exception if it doesn't */
      std::shared_ptr<allocator> alloc=(*manager_strong->allocators())[arrayptr].alloc;
      std::vector<cl_event> ev;

      std::shared_ptr<_openclbuffer> oclbuffer=_GetBufferObject(context,device,arrayptr);

      rangetracker<openclregion>::iterator invalidregion;

            
      /* make sure we will wait for any currently pending transfers overlapping with this subbuffer*/
      for (invalidregion=oclbuffer->invalidity.begin();invalidregion != oclbuffer->invalidity.end();invalidregion++) {
	if (invalidregion->second->fill_event && region_overlaps(*invalidregion->second,startelem,startelem+numelem)) {
	  clRetainEvent(invalidregion->second->fill_event);
	  ev.emplace_back(invalidregion->second->fill_event); 
	  
	}
      }

      rwlock_token_set regionlocks;
      if (write) {
	regionlocks=manager_strong->locker->get_preexisting_locks_write_array_region(alllocks,arrayptr,startelem,numelem);
      } else {
	regionlocks=manager_strong->locker->get_preexisting_locks_read_array_region(alllocks,arrayptr,startelem,numelem);	
      }


      if (!write_only) {
	/* No need to enqueue transfers if kernel is strictly write */

	_TransferInvalidRegions(context,device,oclbuffer,arrayptr,startelem,numelem,ev);
	

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
      cl_buffer_region region = { startelem*oclbuffer->elemsize, numelem*oclbuffer->elemsize };
      cl_mem subbuffer = clCreateSubBuffer(oclbuffer->buffer,flags,CL_BUFFER_CREATE_TYPE_REGION,&region,&errcode);
      if (errcode != CL_SUCCESS) {
	throw openclerror(errcode,"Error creating subbuffer");
      }
      
      return std::make_tuple(regionlocks,subbuffer,ev);
    }
      
    /** Will need a function ReleaseOpenCLBuffer that takes an event
        list indicating dependencies. This function queues any necessary
	transfers (should it clFlush()?) to transfer data from the 
	buffer back into host memory. This function will take an 
    rwlock_token_set (that will be released once said transfer
    is complete) and some form of 
    pointer to the openclbuffer. It will need to queue the transfer
    of any writes, and also clReleaseEvent the fill_event fields 
    of the invalidregions, and remove the markings as invalid */

    std::tuple<rwlock_token_set,cl_mem,std::vector<cl_event>> GetOpenCLBuffer(rwlock_token_set alllocks,cl_context context, cl_device_id device, void **arrayptr,bool write,bool write_only=false) /* indexed by arrayidx */
/* cl_mem_flags flags,snde_index firstelem,snde_index numelem */ /* numelems may be SNDE_INDEX_INVALID to indicate all the way to the end */


    /* (OBSOLETE) note cl_mem_flags does NOT determine what type of OpenCL buffer we get, but rather what
       our kernel is going to do with it, i.e. CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY or CL_MEM_READ_WRITE */
    /* returns new rwlock_token_set representing readlocks ... let this fall out of scope to release it. */
    /* returns new rwlock_token_set representing writelocks ... let this fall out of scope to release it. */
    /* returns cl_mem... will need to call clReleaseMemObject() on this */
    /* returns snde_index representing the offset, in units of elemsize into the cl_mem of the first element of the cl_mem... (always 0 for now, but may not be zero if we use sub-buffers in the future)  */
    /* returns cl_events... will need to call clReleaseEvent() on each */

      
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

      std::unique_lock<std::mutex> adminlock(admin);
      

      std::shared_ptr<arraymanager> manager_strong(manager); /* as manager references us, it should always exist while we do. This will throw an exception if it doesn't */

      std::shared_ptr<allocator> alloc=(*manager_strong->allocators())[arrayptr].alloc;
      std::shared_ptr<_openclbuffer> oclbuffer;
      std::vector<cl_event> ev;
      rangetracker<openclregion>::iterator invalidregion;
      //size_t arrayidx=manager_strong->locker->get_array_idx(arrayptr);

    
      oclbuffer=_GetBufferObject(context,device,arrayptr);
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
	if (invalidregion->second->fill_event) {
	  clRetainEvent(invalidregion->second->fill_event);
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
      if (write) {
	regionlocks=manager_strong->locker->get_preexisting_locks_read_array_region(alllocks,arrayptr,0,SNDE_INDEX_INVALID);

      } else {
	regionlocks=manager_strong->locker->get_preexisting_locks_read_array_region(alllocks,arrayptr,0,SNDE_INDEX_INVALID);
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
      
      clRetainMemObject(oclbuffer->buffer); /* ref count for returned cl_mem pointer */
      
      return std::make_tuple(regionlocks,oclbuffer->buffer,ev);
    }

    std::pair<std::vector<cl_event>,std::vector<cl_event>> FlushWrittenOpenCLBuffer(cl_context context,cl_device_id device,void **arrayptr,std::vector<cl_event> explicit_prerequisites)
    {
      // Gather in implicit prerequisites
      // (any pending transfers to this buffer)
      /* capture our admin lock */

      // Note that if there were any actual transfers initiated,
      // the result will be length-1. Otherwise no transfer was needed.
      std::vector<cl_event> wait_events;
      std::vector<cl_event> result_events;

      std::vector<std::pair<cl_event,void *>> callback_requests; 
      rangetracker<openclregion>::iterator invalidregion;
      
      std::shared_ptr<arraymanager> manager_strong(manager); /* as manager references us, it should always exist while we do. This will throw an exception if it doesn't */

      //size_t arrayidx=manager_strong->locker->get_array_idx(arrayptr);

      //std::shared_ptr<allocator> alloc=(*manager_strong->allocators())[arrayptr].alloc;

      /* create arrayinfo key */
      openclarrayinfo arrayinfo=openclarrayinfo(context,device,arrayptr);
     
      std::unique_lock<std::mutex> adminlock(admin);

      std::shared_ptr<_openclbuffer> oclbuffer;

      oclbuffer=buffer_map.at(arrayinfo); /* buffer should exist because should have been created in GetOpenCLBuffer() */

      /* check for pending events and accumulate them into wait list */
      /* ... all transfers in to this buffer should be complete before we allow a transfer out */
      for (invalidregion=oclbuffer->invalidity.begin();invalidregion != oclbuffer->invalidity.end();invalidregion++) {
	if (invalidregion->second->fill_event) {
	  clRetainEvent(invalidregion->second->fill_event);
	  wait_events.emplace_back(invalidregion->second->fill_event); 
	  
	}
      }

      for (auto & ep_event : explicit_prerequisites) {
	clRetainEvent(ep_event);
	wait_events.emplace_back(ep_event);
      }

      /* Go through every write region, flushing out any dirty portions */
      
      //for (auto & writeregion: (*arraywriteregions)[arrayidx]) {
      for (auto & dirtyregion : oclbuffer->_dirtyregions) {
	snde_index numelem;
	
	if (dirtyregion.second->regionend==SNDE_INDEX_INVALID) {
	  numelem=(*manager_strong->allocators())[arrayptr].alloc->total_nelem()-dirtyregion.second->regionstart;
	} else {
	  numelem=dirtyregion.second->regionend-dirtyregion.second->regionstart;
	}

	snde_index offset=dirtyregion.second->regionstart;  //*oclbuffer->elemsize;

	if (dirtyregion.second->FlushDoneEvent && !dirtyregion.second->FlushDoneEventComplete) {
	  clRetainEvent(dirtyregion.second->FlushDoneEvent);
	  result_events.emplace_back(dirtyregion.second->FlushDoneEvent);	  
	} else if (!dirtyregion.second->FlushDoneEvent && !dirtyregion.second->FlushDoneEventComplete) {
	  /* Need to perform flush */
	  // Queue transfer
	
	  
	  cl_event newevent=NULL;
	  
	  clEnqueueReadBuffer(_get_queue_no_ref(context,device),oclbuffer->buffer,CL_FALSE,offset*oclbuffer->elemsize,(numelem)*oclbuffer->elemsize,((char *)(*arrayptr)) + (offset*oclbuffer->elemsize),wait_events.size(),&wait_events[0],&newevent);
	  dirtyregion.second->FlushDoneEvent=newevent;
	  //fprintf(stderr,"Setting FlushDoneEvent...\n");

	  clRetainEvent(newevent);/* dirtyregion retains a reference to newevent */

	  
	  /**** trigger marking of other caches as invalid */
	  
	  //cleanedregions.mark_region(regionstart,regionend-regionstart,this);
	  /* now that it is enqueued we can replace our wait list 
	     with this newevent */
	  for (auto & oldevent : wait_events) {
	    clReleaseEvent(oldevent);
	  }
	  wait_events.clear();
	  
	  //dirtyregion.second->FlushDoneEvent=newevent;
	  //clRetainEvent(newevent); 

	  /* queue up our callback request (queue it rather than do it now, so as to avoid a deadlock
	     if it is processed inline) */

	  std::shared_ptr<opencldirtyregion> dirtyregionptr=dirtyregion.second;
	  std::shared_ptr<openclcachemanager> shared_this=std::dynamic_pointer_cast<openclcachemanager>(shared_from_this());
	  
	  callback_requests.emplace_back(std::make_pair(newevent,(void *)new std::function<void(cl_event,cl_int)>([ shared_this, manager_strong, arrayptr, dirtyregionptr, oclbuffer ](cl_event event, cl_int event_command_exec_status) {
		/* NOTE: This callback may occur in a separate thread */
		/* it indicates that the data transfer is complete */
		
		if (event_command_exec_status != CL_COMPLETE) {
		  throw openclerror(event_command_exec_status,"Error in waited event (from executing clEnqueueReadBuffer()) ");

		}

		std::unique_lock<std::mutex> adminlock(shared_this->admin);
		/* copy the info out of dirtyregionptr while we hold the admin lock */
		opencldirtyregion dirtyregion=*dirtyregionptr;

		adminlock.unlock();
		/* We must now notify others that this has been modified */
		
		/* Others include other buffers of our own cache manager... */
		
		shared_this->mark_as_dirty_except_buffer(oclbuffer,arrayptr,dirtyregion.regionstart,dirtyregion.regionend-dirtyregion.regionstart);
		
		/* and other cache managers... */
		manager_strong->mark_as_dirty(shared_this.get(),arrayptr,dirtyregionptr->regionstart,dirtyregionptr->regionend-dirtyregionptr->regionstart);

		
		/* We must now mark this region as modified... i.e. that notifications to others have been completed */
		adminlock.lock();
		  
		dirtyregionptr->FlushDoneEventComplete=true;
		dirtyregionptr->complete_condition.notify_all();
		clReleaseEvent(dirtyregionptr->FlushDoneEvent);
		dirtyregionptr->FlushDoneEvent=NULL;
		//fprintf(stderr,"FlushDoneEventComplete\n");

		
		})));
	  
	  
	  result_events.emplace_back(newevent); /* add new event to our set, eating our referencee */
	  
	}
      }
	
      clFlush(_get_queue_no_ref(context,device));
      
      /* Trigger notifications to others once transfer is complete and we can release our admin lock */
      adminlock.unlock();
      for (auto & event_func : callback_requests) {
	clSetEventCallback(event_func.first,CL_COMPLETE,snde_opencl_callback,event_func.second );
	
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
      
    std::pair<std::vector<cl_event>,std::shared_ptr<std::thread>> ReleaseOpenCLBuffer(rwlock_token_set locks,cl_context context, cl_device_id device, cl_mem mem, void **arrayptr, cl_event input_data_not_needed,std::vector<cl_event> output_data_complete)
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

      std::vector<cl_event> all_finished;
      std::shared_ptr<std::thread> unlock_thread;

      /* make copy of locks to delegate to threads... create pointers so it is definitely safe to delegate */
      rwlock_token_set *locks_copy1 = new rwlock_token_set(clone_rwlock_token_set(locks));
      rwlock_token_set *locks_copy2 = new rwlock_token_set(clone_rwlock_token_set(locks));

      release_rwlock_token_set(locks); /* release our reference to original */

      
      std::shared_ptr<arraymanager> manager_strong(manager); /* as manager references us, it should always exist while we do. This will throw an exception if it doesn't */
      //size_t arrayidx=manager_strong->locker->get_array_idx(arrayptr);

      
      
      /* don't worry about invalidity here because presumably 
	 that was taken care of before we started writing
	 (validity is a concept that applies to the GPU buffer, 
	 not the memory buffer .... memory buffer is invalid
	 but noone will see that because it will be valid before
	 we release the write lock */
      
      
      std::vector<cl_event> prerequisite_events(output_data_complete); /* transfer prequisites in */
      output_data_complete.clear();
      prerequisite_events.emplace_back(input_data_not_needed);

      std::vector<cl_event> wait_events,flush_events;

      // Iterate over dirty regions and add event callbacks instead to their flushdoneevents
      // such that once a particular FlushDoneEvent is complete, we do the dirty notification
      // to other caches. Once all FlushDoneEvents are complete, perform the unlocking.

      // (does nothing of substance if there is nothing dirty)
      std::tie(wait_events,flush_events)=FlushWrittenOpenCLBuffer(context,device,arrayptr,prerequisite_events);
      /* Note now that wait_events and flush_events we have ownership of, whereas all of the prerequisite_events we didn't */
	
      // FlushWrittenOpenCLBuffer should have triggered the FlushDoneEvents and done the
      // dirty notification.... we have to set up something that will wait until everything
      // is complete before we unlock 
      
      std::unique_lock<std::mutex> adminlock(admin);
      
      std::shared_ptr<_openclbuffer> oclbuffer;
      
      /* create arrayinfo key */
      openclarrayinfo arrayinfo=openclarrayinfo(context,device,arrayptr);
      
      oclbuffer=buffer_map.at(arrayinfo); /* buffer should exist because should have been created in GetOpenCLBuffer() */

      //std::vector<std::shared_ptr<opencldirtyregion>> *dirtyregions_copy=new std::vector<std::shared_ptr<opencldirtyregion>>();

      //for (auto & dirtyregion : oclbuffer->_dirtyregions) {
      //  /* Copy stuff out of dirtyregions, and wait for all of the condition variables
      //   so that we have waited for all updates to have occured and can release our reference to the lock */
      //  if (!dirtyregion.second->FlushDoneEventComplete && dirtyregion.second->FlushDoneEvent) {
      //    dirtyregions_copy->emplace_back(dirtyregion.second);
      //  }
      //}


      // .. if there is anything dirty...
      bool dirtyflag=false;
      rangetracker<opencldirtyregion>::iterator dirtyregion;
      for (dirtyregion=oclbuffer->_dirtyregions.begin();dirtyregion != oclbuffer->_dirtyregions.end();dirtyregion++) {
	if (!dirtyregion->second->FlushDoneEventComplete) {
	  dirtyflag=true;
	}

      }
      // release adminlock so we can start the thread on a copy (don't otherwise need adminlock from here on) 
      adminlock.unlock();

      if (dirtyflag) {

	/* start thread that will hold our locks until all dirty regions are no longer dirty */
	std::shared_ptr<openclcachemanager> shared_this=std::dynamic_pointer_cast<openclcachemanager>(shared_from_this());
	unlock_thread=std::make_shared<std::thread>( [ shared_this,oclbuffer,locks_copy1 ]() {
	    
	    //std::vector<std::shared_ptr<opencldirtyregion>>::iterator dirtyregion;
	    rangetracker<opencldirtyregion>::iterator dirtyregion;
	    std::unique_lock<std::mutex> lock(shared_this->admin);
	    
	    // Keep on waiting on the first dirty region, removing it once it is complete,
	    // until there is nothing left
	    for (dirtyregion=oclbuffer->_dirtyregions.begin();dirtyregion != oclbuffer->_dirtyregions.end();dirtyregion=oclbuffer->_dirtyregions.begin()) {
	      if (dirtyregion->second->FlushDoneEventComplete) {
		oclbuffer->_dirtyregions.erase(dirtyregion);
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
      if (input_data_not_needed) {
	
	/* in this case locks_copy2 delegated on to callback */	
	clSetEventCallback(input_data_not_needed,CL_COMPLETE,snde_opencl_callback,(void *)new std::function<void(cl_event,cl_int)>([ locks_copy2 ](cl_event event, cl_int event_command_exec_status) {
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
    
    /* ***!!! Need a method to throw away all cached buffers with a particular context !!!*** */
    
    //virtual void add_allocated_array(void **arrayptr,size_t elemsize,snde_index totalnelem)
    //{
    //  std::lock_guard<std::mutex> adminlock(admin);
    //
    //  // Make sure arrayptr not already managed
    //  assert(allocators.find(arrayptr)==allocators.end());
    //
    //  allocators[arrayptr]=allocationinfo{std::make_shared<allocator>(_memalloc,locker,arrayptr,elemsize,totalnelem),0};
    //  locker->addarray(arrayptr);
    //  
    //
    //}
  
    //virtual void add_follower_array(void **allocatedptr,void **arrayptr,size_t elemsize)
    //{
    //  std::lock_guard<std::mutex> adminlock(admin);
    //
    //  std::shared_ptr<allocator> alloc=allocators[allocatedptr].alloc;
    //  // Make sure arrayptr not already managed
    //  assert(allocators.find(arrayptr)==allocators.end()); 
    //
    //  allocators[arrayptr]=allocationinfo{alloc,alloc->add_other_array(arrayptr,elemsize)};
    //  
    //}
    //virtual snde_index alloc(void **allocatedptr,snde_index nelem)
    //{
    //  std::lock_guard<std::mutex> adminlock(admin);
    //  std::shared_ptr<allocator> alloc=allocators[allocatedptr].alloc;
    //  return alloc->alloc(nelem);    
    //}

    //virtual void free(void **allocatedptr,snde_index addr,snde_index nelem)
    //{
    //  std::lock_guard<std::mutex> adminlock(admin);
    //  std::shared_ptr<allocator> alloc=allocators[allocatedptr].alloc;
    //  alloc->free(addr,nelem);    
    //}

    //virtual void clear() /* clear out all references to allocated and followed arrays */
    //{
    //  std::lock_guard<std::mutex> adminlock(admin);
    //  allocators.clear();
    //}


    virtual ~openclcachemanager() {
      
    }
  };


  static inline std::shared_ptr<openclcachemanager> get_opencl_cache_manager(std::shared_ptr<arraymanager> manager)
  {
    
    
    if (!manager->has_cache("OpenCL")) {
      /* create the cache manager since it may not exist (... or another thread could be creating int in parallel. set_undefined_cache will ignore us if it is already set */
      manager->set_undefined_cache("OpenCL",std::make_shared<openclcachemanager>(manager));
    } 
    return std::dynamic_pointer_cast<openclcachemanager>(manager->get_cache("OpenCL"));
  }
    
  class OpenCLBuffer_info {
  public:
    std::shared_ptr<arraymanager> manager;
    std::shared_ptr<openclcachemanager> cachemanager;
    //cl_command_queue transferqueue;  /* counted by clRetainCommandQueue */
    cl_mem mem; /* counted by clRetainMemObject */
    void **arrayptr;
    //rwlock_token_set readlocks;
    rwlock_token_set locks;
    
    OpenCLBuffer_info(std::shared_ptr<arraymanager> manager,
		      //cl_command_queue transferqueue,  /* adds new reference */
		      cl_mem mem, /* adds new reference */
		      void **arrayptr,
		      //rwlock_token_set readlocks,
		      rwlock_token_set locks) 
    {
      this->manager=manager;
      this->cachemanager=get_opencl_cache_manager(manager);
      //clRetainCommandQueue(transferqueue);
      //this->transferqueue=transferqueue;
      clRetainMemObject(mem);
      this->mem=mem;
      this->arrayptr=arrayptr;
      //this->readlocks=readlocks;
      this->locks=locks;
    }
    OpenCLBuffer_info(const OpenCLBuffer_info &orig)
    {
      this->manager=orig.manager;
      this->cachemanager=orig.cachemanager;
      //clRetainCommandQueue(orig.transferqueue);
      //this->transferqueue=orig.transferqueue;
      clRetainMemObject(orig.mem);
      this->mem=orig.mem;
      this->arrayptr=orig.arrayptr;
      //this->readlocks=orig.readlocks;
      this->locks=orig.locks;
      
      //for (auto & event: this->fill_events) {
      //clRetainEvent(event);
      //}
    }
    
    OpenCLBuffer_info& operator=(const OpenCLBuffer_info &)=delete; /* copy assignment disabled (for now) */
    ~OpenCLBuffer_info()
    {
      //clReleaseCommandQueue(transferqueue);
      clReleaseMemObject(mem);

    }

    
  };

  class OpenCLBufferKey {
  public:
    void **array;
    snde_index firstelem;
    snde_index numelem; // Is this really necessary?

    OpenCLBufferKey(void **_array,snde_index _firstelem,snde_index _numelem) :
      array(_array), firstelem(_firstelem), numelem(_numelem)
    {

    }
    // equality operator for std::unordered_map
    bool operator==(const OpenCLBufferKey b) const
    {
      return b.array==array && b.firstelem==firstelem && b.numelem==numelem;
    }

    
  };

}
// Need to provide hash implementation for OpenCLBufferKey so
// it can be used as a std::unordered_map key
namespace std {
  template <> struct hash<snde::OpenCLBufferKey>
  {
    size_t operator()(const snde::OpenCLBufferKey & x) const
    {
      return std::hash<void *>{}((void *)x.array) + std::hash<snde_index>{}(x.firstelem) +std::hash<snde_index>{}(x.numelem);
    }
  };
}

namespace snde {
  

  
  class OpenCLBuffers {
    // Class for managing array of opencl buffers returned by the
    // opencl array manager... SHOULD ONLY BE USED BY ONE THREAD.
    
  public:
    cl_context context;  /* counted by clRetainContext() */
    cl_device_id device;  /* counted by clRetainDevice() */
    rwlock_token_set all_locks;

    std::unordered_map<OpenCLBufferKey,OpenCLBuffer_info> buffers; /* indexed by arrayidx */
    
    std::vector<cl_event> fill_events; /* each counted by clRetainEvent() */
     
    OpenCLBuffers(cl_context context,cl_device_id device,rwlock_token_set all_locks)
    {
      clRetainContext(context);
      this->context=context;
      clRetainDevice(device);
      this->device=device;
      this->all_locks=all_locks;
    }

    /* no copying */
    OpenCLBuffers(const OpenCLBuffers &) = delete;
    OpenCLBuffers & operator=(const OpenCLBuffers &) = delete;
  
    ~OpenCLBuffers() {
      std::unordered_map<OpenCLBufferKey,OpenCLBuffer_info>::iterator nextbuffer = buffers.begin();
      if (nextbuffer != buffers.end()) {
	fprintf(stderr,"OpenCL Cachemanager Warning: OpenCLBuffers destructor called with residual active buffers\n");
	RemBuffers(NULL,false);
      }
      
      /* release each cl_event */
      for (auto & ev: fill_events) {
	clReleaseEvent(ev);
      }
      clReleaseContext(context);
      clReleaseDevice(device);

    }
    
    cl_mem Mem(void **arrayptr,snde_index firstelem,snde_index numelem)
    /* Returns protected pointer (ref count increased */
    {
      cl_mem mem=buffers.at(OpenCLBufferKey(arrayptr,firstelem,numelem)).mem;
      clRetainMemObject(mem);
      return mem;
    }

    cl_mem Mem_untracked(void **arrayptr,snde_index firstelem,snde_index numelem)
    /* Returns unprotected pointer (ref count not increased */
    {
      return buffers.at(OpenCLBufferKey(arrayptr,firstelem,numelem)).mem;
    }


    cl_event *FillEvents_untracked(void)
    /* Returns vector of unprotected pointers (ref count not increased */
    {
      return &fill_events[0];
    }

    cl_uint NumFillEvents(void)
    {
      return (cl_uint)fill_events.size();
    }
    

    
    cl_int SetBufferAsKernelArg(cl_kernel kernel, cl_uint arg_index, void **arrayptr,snde_index firstelem,snde_index numelem)
    {
      cl_mem mem;

      mem=Mem_untracked(arrayptr,firstelem,numelem);
      return clSetKernelArg(kernel,arg_index,sizeof(mem),&mem);
      
    }
  
    void AddSubBuffer(std::shared_ptr<arraymanager> manager, void **arrayptr,snde_index indexstart,snde_index numelem,bool write,bool write_only=false)
    {

      //// accumulate preexisting locks + locks in all buffers together
      //for (auto & arrayptr_buf : buffers) {
      //  merge_into_rwlock_token_set(locks,arrayptr_buf.second.readlocks);
      //  merge_into_rwlock_token_set(locks,arrayptr_buf.second.writelocks);
      //}

      rwlock_token_set locks;
      cl_mem mem;
      std::vector<cl_event> new_fill_events;
      std::shared_ptr<openclcachemanager> cachemanager=get_opencl_cache_manager(manager);
      //void **allocatedptr;

      //allocatedptr=manager->allocation_arrays.at(arrayptr);

      if (indexstart==0 && numelem==SNDE_INDEX_INVALID) {
	std::tie(locks,mem,new_fill_events) = cachemanager->GetOpenCLBuffer(all_locks,context,device,arrayptr,write,write_only);
      } else {
	std::tie(locks,mem,new_fill_events) = cachemanager->GetOpenCLSubBuffer(all_locks,context,device,arrayptr,indexstart,numelem,write,write_only);

      }
      
      /* move fill events into our master list */
      fill_events.insert(fill_events.end(),new_fill_events.begin(),new_fill_events.end());
      
      buffers.emplace(std::make_pair(OpenCLBufferKey(arrayptr,indexstart,numelem),
				     OpenCLBuffer_info(manager,	        
						       mem,
						       arrayptr,
						       locks)));
      clReleaseMemObject(mem); /* remove extra reference */
      
      
      // add this lock to our database of preexisting locks 
      //locks.push_back(buffers[name][1]); 
    }

    void AddBuffer(std::shared_ptr<arraymanager> manager, void **arrayptr,bool write,bool write_only=false)
    {
      AddSubBuffer(manager,arrayptr,0,SNDE_INDEX_INVALID,write,write_only);
    }
    
    cl_int AddSubBufferAsKernelArg(std::shared_ptr<arraymanager> manager,cl_kernel kernel,cl_uint arg_index,void **arrayptr,snde_index indexstart,snde_index numelem,bool write,bool write_only=false)
    {
      AddSubBuffer(manager,arrayptr,indexstart,numelem,write,write_only);
      return SetBufferAsKernelArg(kernel,arg_index,arrayptr,indexstart,numelem);
    }

    cl_int AddBufferAsKernelArg(std::shared_ptr<arraymanager> manager,cl_kernel kernel,cl_uint arg_index,void **arrayptr,bool write,bool write_only=false)
    {
      AddBuffer(manager,arrayptr,write,write_only);
      return SetBufferAsKernelArg(kernel,arg_index,arrayptr,0,SNDE_INDEX_INVALID);
    }


    void BufferDirty(void **arrayptr)
    /* This indicates that the array has been written to by an OpenCL kernel, 
       and that therefore it needs to be copied back into CPU memory */
    {
      OpenCLBuffer_info &info=buffers.at(OpenCLBufferKey(arrayptr,0,SNDE_INDEX_INVALID));

      
      snde_index numelem=(*info.manager->allocators())[arrayptr].alloc->total_nelem();
      info.cachemanager->mark_as_gpu_modified(context,device,arrayptr,0,numelem);
      

    }

    void BufferDirty(void **arrayptr,snde_index pos,snde_index len)
    /* This indicates that the array region has been written to by an OpenCL kernel, 
       and that therefore it needs to be copied back into CPU memory */
    {
      OpenCLBuffer_info &info=buffers.at(OpenCLBufferKey(arrayptr,0,SNDE_INDEX_INVALID));
      
      info.cachemanager->mark_as_gpu_modified(context,device,arrayptr,pos,len);
      
    }
    void SubBufferDirty(void **arrayptr,snde_index sb_pos,snde_index sb_len)
    /* This indicates that the array region has been written to by an OpenCL kernel, 
       and that therefore it needs to be copied back into CPU memory */
    {
      OpenCLBuffer_info &info=buffers.at(OpenCLBufferKey(arrayptr,sb_pos,sb_len));

      info.cachemanager->mark_as_gpu_modified(context,device,arrayptr,sb_pos,sb_len);
      
    }

    void SubBufferDirty(void **arrayptr,snde_index sb_pos,snde_index sb_len,snde_index dirtypos,snde_index dirtylen)
    /* This indicates that the array region has been written to by an OpenCL kernel, 
       and that therefore it needs to be copied back into CPU memory */
    {
      OpenCLBuffer_info &info=buffers.at(OpenCLBufferKey(arrayptr,sb_pos,sb_len));

      info.cachemanager->mark_as_gpu_modified(context,device,arrayptr,sb_pos+dirtypos,dirtylen);
    }


    std::pair<std::vector<cl_event>,std::vector<cl_event>> FlushBuffer(void **arrayptr,snde_index sb_pos,snde_index sb_len,std::vector<cl_event> explicit_prerequisites)
    {
      OpenCLBuffer_info &info=buffers.at(OpenCLBufferKey(arrayptr,sb_pos,sb_len));
      
      return info.cachemanager->FlushWrittenOpenCLBuffer(context,device,arrayptr,explicit_prerequisites);
    }

    
    
    void RemSubBuffer(void **arrayptr,snde_index startidx,snde_index numelem,cl_event input_data_not_needed,std::vector<cl_event> output_data_complete,bool wait)
    /* Either specify wait=true, then you can explicitly unlock_rwlock_token_set() your locks because you know they're done, 
       or specify wait=false in which case things may finish later. The only way to make sure they are finished is 
       to obtain a new lock on the same items */
      
    /* Does not decrement refcount of waitevents */
    {
      /* Remove and unlock buffer */
      OpenCLBufferKey Key(arrayptr,startidx,numelem);
      OpenCLBuffer_info &info=buffers.at(Key);
      
      std::vector<cl_event> all_finished;
      std::shared_ptr<std::thread> wrapupthread;
      std::tie(all_finished,wrapupthread)=info.cachemanager->ReleaseOpenCLBuffer(info.locks,context,device,info.mem,arrayptr,input_data_not_needed,output_data_complete);

      if (wait) {
	clWaitForEvents(all_finished.size(),&all_finished[0]);
	if (wrapupthread && wrapupthread->joinable()) {
	  wrapupthread->join();
	}
      } else {
	if (wrapupthread && wrapupthread->joinable()) {
	  wrapupthread->detach();
	}
      }

      for (auto & ev: all_finished) {
	clReleaseEvent(ev);
      }
      
      /* remove from hash table */
      buffers.erase(Key);
    
    }

    void RemBuffer(void **arrayptr,cl_event input_data_not_needed,std::vector<cl_event> output_data_complete,bool wait)
    {
      RemSubBuffer(arrayptr,0,SNDE_INDEX_INVALID,input_data_not_needed,output_data_complete,wait);
    }
    
    void RemBuffers(cl_event input_data_not_needed,std::vector<cl_event> output_data_complete,bool wait)
    {
      for (std::unordered_map<OpenCLBufferKey,OpenCLBuffer_info>::iterator nextbuffer = buffers.begin();
	   nextbuffer != buffers.end();) {
	std::unordered_map<OpenCLBufferKey,OpenCLBuffer_info>::iterator thisbuffer=nextbuffer;
	nextbuffer++;

	RemSubBuffer(thisbuffer->first.array,thisbuffer->first.firstelem,thisbuffer->first.numelem,input_data_not_needed,output_data_complete,wait);
	
	
      }
    }
    
    void RemBuffers(cl_event input_data_not_needed,cl_event output_data_complete,bool wait)
    {
      std::vector<cl_event> output_data_complete_vector{output_data_complete};
      RemBuffers(input_data_not_needed,output_data_complete_vector,wait);
    }

    void RemBuffers(cl_event input_data_not_needed,bool wait)
    {
      std::vector<cl_event> output_data_complete_vector{};
      RemBuffers(input_data_not_needed,output_data_complete_vector,wait);
    }
  };
  
};
#endif /* SNDE_OPENCLCACHEMANAGER_HPP */
