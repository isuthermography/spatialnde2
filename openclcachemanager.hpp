#ifndef SNDE_OPENCLCACHEMANAGER_HPP
#define SNDE_OPENCLCACHEMANAGER_HPP


#include <cstring>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/opencl.h>

#include "arraymanager.hpp"
#include "rangetracker.hpp"
#include "snde_error_opencl.hpp"


extern "C" void snde_opencl_callback(cl_event event, cl_int event_command_exec_status, void *user_data);



namespace snde {


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

    std::shared_ptr<openclregion> breakup(snde_index breakpoint)
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
    /* could insert some flag to indicate use of zero-copy memory */
    void **arrayptr;
    
    openclarrayinfo(cl_context context, void **arrayptr)
    {
      this->context=context;
      clRetainContext(this->context); /* increase refcnt */
      this->arrayptr=arrayptr;
    }
    openclarrayinfo(const openclarrayinfo &orig) /* copy constructor */
    {
      context=orig.context;
      clRetainContext(context);
      arrayptr=orig.arrayptr;
    }
    
    openclarrayinfo& operator=(const openclarrayinfo &orig) /* copy assignment operator */
    {
      clReleaseContext(context);
      context=orig.context;
      clRetainContext(context);
      arrayptr=orig.arrayptr;

      return *this;
    }

    // equality operator for std::unordered_map
    bool operator==(const openclarrayinfo b) const
    {
      return b.context==context && b.arrayptr==arrayptr;
    }

    ~openclarrayinfo() {
      clReleaseContext(context);
    }
  };

} 

// Need to provide hash implementation for openclarrayinfo so
// it can be used as a std::unordered_map key
namespace std {
  template <> struct hash<snde::openclarrayinfo>
  {
    size_t operator()(const snde::openclarrayinfo & x) const
    {
      return std::hash<void *>{}((void *)x.context) + std::hash<void *>{}((void *)x.arrayptr);
    }
  };
}

namespace snde { 
  
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
    std::shared_ptr<std::function<void(snde_index)>> realloc_callback;
    std::weak_ptr<allocator> alloc; /* weak pointer to the allocator because we don't want to inhibit freeing of this (all we need it for is our reallocation callback) */
    
    _openclbuffer(const _openclbuffer &)=delete; /* copy constructor disabled */
    
    _openclbuffer(cl_context context,std::shared_ptr<allocator> alloc,size_t elemsize,void **arrayptr, std::mutex *arraymanageradminmutex)
    {
      /* initialize buffer in context with specified size */
      /* caller should hold array manager's admin lock; 
	 should also hold at least read lock on this buffer */
      snde_index nelem;
      cl_int errcode_ret=CL_SUCCESS;
      
      this->elemsize=elemsize;
      this->arrayptr=arrayptr;
      this->alloc=alloc;
      
      nelem=alloc->total_nelem();
      
      buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,nelem*elemsize, NULL /* *arrayptr */,&errcode_ret);

      if (errcode_ret != CL_SUCCESS) {
	throw openclerror(errcode_ret,(std::string)"Error creating buffer of size %d",(long)(nelem*elemsize));
      }
      
      invalidity.mark_all(nelem);

      /* Need to register for notification of realloc
	 so we can re-allocate the buffer! */
      realloc_callback=std::make_shared<std::function<void(snde_index)>>([this,context,arraymanageradminmutex](snde_index total_nelem) {
	  /* Note: we're not managing context, but presumably the openclarrayinfo that is our key in buffer_map will prevent the context from being freed */
	  
	  std::lock_guard<std::mutex> lock(*arraymanageradminmutex);
	  snde_index numelem;
	  cl_int lambdaerrcode_ret=CL_SUCCESS;
	  numelem=total_nelem; /* determine new # of elements */
	  clReleaseMemObject(buffer); /* free old buffer */

	  /* allocate new buffer */
	  buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,numelem*this->elemsize,*this->arrayptr,&lambdaerrcode_ret);

	  if (lambdaerrcode_ret != CL_SUCCESS) {
	    throw openclerror(lambdaerrcode_ret,"Error expanding buffer to size %d",(long)(numelem*this->elemsize));
	  }

	  /* Mark entire buffer as invalid */
	  invalidity.mark_all(numelem);
	  
	});
      
      alloc->register_realloc_callback(realloc_callback);
      
    }

    ~_openclbuffer()
    {
      std::shared_ptr<allocator> alloc_strong = alloc.lock();
      if (alloc_strong) { /* if allocator already freed, it would take care of its reference to the callback */ 
	alloc_strong->unregister_realloc_callback(realloc_callback);
      }
      
      clReleaseMemObject(buffer);
      buffer=NULL;
    }
  };


  
  
  /* openclcachemanager manages opencl caches for arrays 
     (for now) managed by a single arraymanager/lockmanager  */ 
  class openclcachemanager : public cachemanager {
  public: 
    /* Look up allocator by __allocated_array_pointer__ only */
    /* mapping index is arrayptr, returns allocator */
    //std::unordered_map<void **,allocationinfo> allocators;
    //std::shared_ptr<memallocator> _memalloc;
    // locker defined by arraymanager base class
    std::mutex admin; /* lock our data structures, including buffer_map and descendents. We are allowed to call allocators/lock managers while holding 
			 this mutex, but they are not allowed to call us and only lock their own data structures, 
			 so a locking order is ensured and no deadlock is possible */

    std::unordered_map<openclarrayinfo,std::shared_ptr<_openclbuffer>> buffer_map;

    std::weak_ptr<arraymanager> manager; /* Used for looking up allocators and accessing lock manager (weak to avoid creating a circular reference) */ 
    
    
    openclcachemanager(std::shared_ptr<arraymanager> manager) {
      //_memalloc=memalloc;
      //locker = std::make_shared<lockmanager>();
      this->manager=manager;
    }
    openclcachemanager(const openclcachemanager &)=delete; /* copy constructor disabled */
    openclcachemanager& operator=(const openclcachemanager &)=delete; /* assignment disabled */


    /** Will need a function ReleaseOpenCLBuffer that takes an event
        list indicating dependencies. This function queues any necessary
	transfers (should it clFlush()?) to transfer data from the 
	buffer back into host memory. This function will take an 
    rwlock_token_set (that will be released once said transfer
    is complete) and some form of 
    pointer to the openclbuffer. It will need to queue the transfer
    of any writes, and also clReleaseEvent the fill_event fields 
    of the invalidregions, and remove the markings as invalid */

    std::tuple<rwlock_token_set,rwlock_token_set,cl_mem,snde_index,std::vector<cl_event>> GetOpenCLBuffer(rwlock_token_set alllocks,cl_context context, cl_command_queue queue, void **allocatedptr, void **arrayptr,std::shared_ptr<std::vector<rangetracker<markedregion>>> arrayreadregions, std::shared_ptr<std::vector<rangetracker<markedregion>>> arraywriteregions,bool write_only=false) /* indexed by arrayidx */
/* cl_mem_flags flags,snde_index firstelem,snde_index numelems */ /* numelems may be SNDE_INDEX_INVALID to indicate all the way to the end */


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
      
      openclarrayinfo arrayinfo=openclarrayinfo(context,arrayptr);

      std::shared_ptr<arraymanager> manager_strong(manager); /* as manager references us, it should always exist while we do. This will throw an exception if it doesn't */

      std::unordered_map<openclarrayinfo,std::shared_ptr<_openclbuffer>>::iterator buffer;
      std::shared_ptr<allocator> alloc=manager_strong->allocators[arrayptr].alloc;
      std::shared_ptr<_openclbuffer> oclbuffer;
      std::vector<cl_event> ev;
      rangetracker<openclregion>::iterator invalidregion;
      size_t arrayidx=manager_strong->locker->get_array_idx(arrayptr);

      buffer=buffer_map.find(arrayinfo);
      if (buffer == buffer_map.end()) {
	/* need to create buffer */
	oclbuffer=std::make_shared<_openclbuffer>(context,alloc,alloc->arrays[manager_strong->allocators[arrayptr].allocindex].elemsize,arrayptr,&admin);
	buffer_map[arrayinfo]=oclbuffer;
	
      } else {
	oclbuffer=buffer_map[arrayinfo];
      }
      /* Need to enqueue transfer (if necessary) and return an event set that can be waited for and that the
	 clEnqueueKernel can be dependent on */
      
      /* check for pending events and accumulate them into wait list */
      /* IS THIS REALLY NECESSARY so long as we wait for 
	 the events within our region? (Yes; because OpenCL
	 does not permit a buffer to be updated in one place 
	 while it is being used in another place.
	 
	 NOTE THAT THIS PRACTICALLY PREVENTS region-granular
	 locking unless we switch to using non-overlapping OpenCL sub-buffers */

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

      for (auto & markedregion: (*arrayreadregions)[arrayidx]) {
	snde_index numelems;
	
	if (markedregion.second->regionend==SNDE_INDEX_INVALID) {
	  numelems=SNDE_INDEX_INVALID;
	} else {
	  numelems=markedregion.second->regionend-markedregion.second->regionstart;
	}

	rwlock_token_set regionlocks=manager_strong->locker->get_preexisting_locks_read_array_region(alllocks,allocatedptr,markedregion.second->regionstart,numelems);

	merge_into_rwlock_token_set(readlocks,regionlocks);
      }

      
      rwlock_token_set writelocks=empty_rwlock_token_set();

      for (auto & markedregion: (*arraywriteregions)[arrayidx]) {
	snde_index numelems;
	
	if (markedregion.second->regionend==SNDE_INDEX_INVALID) {
	  numelems=SNDE_INDEX_INVALID;
	} else {
	  numelems=markedregion.second->regionend-markedregion.second->regionstart;
	}

	rwlock_token_set regionlocks=manager_strong->locker->get_preexisting_locks_write_array_region(alllocks,allocatedptr,markedregion.second->regionstart,numelems);

	merge_into_rwlock_token_set(writelocks,regionlocks);
      }
	

      
      /* reaquire adminlock */
      //adminlock.lock();

      rangetracker<markedregion> regions = range_union((*arrayreadregions)[arrayidx],(*arraywriteregions)[arrayidx]);
      
      
      if (!write_only) { /* No need to enqueue transfers if kernel is strictly write */
	for (auto & markedregion: regions) {
	  
	  snde_index firstelem=markedregion.second->regionstart;
	  snde_index numelems;
	  
	  if (markedregion.second->regionend==SNDE_INDEX_INVALID) {
	    numelems=manager_strong->allocators[arrayptr].alloc->total_nelem()-firstelem;
	  } else {
	    numelems=markedregion.second->regionend-markedregion.second->regionstart;
	  }
	  
	  /* transfer any relevant areas that are invalid and not currently pending */

	  rangetracker<openclregion> invalid_regions=oclbuffer->invalidity.get_regions(firstelem,numelems);
	  
	  for (auto & invalidregion: invalid_regions) {
	    
	    /* Perform operation to transfer data to buffer object */
	    
	    /* wait for all other events as it is required for a 
	       partial write to be meaningful */
	    
	    if (!invalidregion.second->fill_event) {
	      snde_index offset=invalidregion.second->regionstart*oclbuffer->elemsize;
	      cl_event newevent=NULL;
	      
	      clEnqueueWriteBuffer(queue,oclbuffer->buffer,CL_FALSE,offset,(invalidregion.second->regionend-invalidregion.second->regionstart)*oclbuffer->elemsize,(char *)*arrayptr + offset,ev.size(),&ev[0],&newevent);

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
	
	}
      }
      clRetainMemObject(oclbuffer->buffer); /* ref count for returned cl_mem pointer */
      
      return std::make_tuple(readlocks,writelocks,oclbuffer->buffer,(snde_index)0,ev);
    }
    
    
    std::vector<cl_event> ReleaseOpenCLBuffer(rwlock_token_set readlocks,rwlock_token_set writelocks,cl_context context, cl_command_queue queue, cl_mem mem, void **allocatedptr, void **arrayptr, std::shared_ptr<std::vector<rangetracker<markedregion>>> arraywriteregions, cl_event input_data_not_needed,std::vector<cl_event> output_data_complete)
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
      
      std::shared_ptr<_openclbuffer> oclbuffer;
      rangetracker<openclregion>::iterator invalidregion;

      std::vector<cl_event> all_finished;

      /* make copy of locks to delegate to thread... create pointer so it is definitely safe to delegate */
      rwlock_token_set *readlocks_copy = new rwlock_token_set(clone_rwlock_token_set(readlocks));
      rwlock_token_set *writelocks_copy = new rwlock_token_set(clone_rwlock_token_set(writelocks));

      release_rwlock_token_set(readlocks); /* release our reference to original */
      release_rwlock_token_set(writelocks); /* release our reference to original */

      /* declare an ownership of each cl_event in parameters */
      for (auto & odcomplete_event : output_data_complete) {
	clRetainEvent(odcomplete_event);
      }

      if (input_data_not_needed) {
	clRetainEvent(input_data_not_needed);
      }
      
      /* capture our admin lock */
      std::unique_lock<std::mutex> adminlock(admin);

      std::shared_ptr<arraymanager> manager_strong(manager); /* as manager references us, it should always exist while we do. This will throw an exception if it doesn't */

      size_t arrayidx=manager_strong->locker->get_array_idx(arrayptr);

      //std::shared_ptr<allocator> alloc=manager_strong->allocators[arrayptr].alloc;

      /* create arrayinfo key */
      openclarrayinfo arrayinfo=openclarrayinfo(context,arrayptr);
      

      oclbuffer=buffer_map[arrayinfo]; /* buffer should exist because should have been created in GetOpenCLBuffer() */

      /* check for pending events and accumulate them into wait list */
      for (invalidregion=oclbuffer->invalidity.begin();invalidregion != oclbuffer->invalidity.end();invalidregion++) {
	if (invalidregion->second->fill_event) {
	  clRetainEvent(invalidregion->second->fill_event);
	  output_data_complete.emplace_back(invalidregion->second->fill_event); 
	  
	}
      }

      if ((*writelocks_copy)->size() > 0) {
	assert((*arraywriteregions)[arrayidx].size() > 0);
	/* in this case writelocks_copy delegated on to callback */
	
	/* don't worry about invalidity here because presumably 
	   that was taken care of before we started writing
	   (validity is a concept that applies to the GPU buffer, 
	   not the memory buffer .... memory buffer is invalid
	   but noone will see that because it will be valid before
	   we release the write lock */

	/* ***!!! Somehow we need to notify any OTHER gpu buffers
	   pointing at this same array that they need to be 
	   marked invalid... before it gets unlocked */

	std::vector<cl_event> prerequisite_events(output_data_complete); /* transfer prequisites in */
	output_data_complete.clear(); 


	
	cl_event newevent=NULL;
	/* Perform operation to transfer data from buffer object
	   to memory */
	/* wait for all other events as it is required for a 
	   partial write to be meaningful */


	for (auto & writeregion: (*arraywriteregions)[arrayidx]) {
	  snde_index numelems;
	
	  if (writeregion.second->regionend==SNDE_INDEX_INVALID) {
	    numelems=manager_strong->allocators[arrayptr].alloc->total_nelem()-writeregion.second->regionstart;
	  } else {
	    numelems=writeregion.second->regionend-writeregion.second->regionstart;
	  }
	
	  snde_index offset=writeregion.second->regionstart*oclbuffer->elemsize;
	  clEnqueueReadBuffer(queue,oclbuffer->buffer,CL_FALSE,offset,(numelems)*oclbuffer->elemsize,(char *)*arrayptr + offset,prerequisite_events.size(),&prerequisite_events[0],&newevent);

	  /* now that it is enqueued we can replace our prerequisite list 
	     with this newevent */
	  for (auto & oldevent : prerequisite_events) {
	    clReleaseEvent(oldevent);
	  }
	  prerequisite_events.clear();

	  prerequisite_events.emplace_back(newevent); /* add new event to our set, eating our referencee */
	}
	
	assert(prerequisite_events.size()==1);
	
	adminlock.unlock(); /* release adminlock in case our callback happens immediately */
	
	clSetEventCallback(prerequisite_events[0],CL_COMPLETE,snde_opencl_callback,(void *)new std::function<void(cl_event,cl_int)>([ writelocks_copy ](cl_event event, cl_int event_command_exec_status) {
	      /* NOTE: This callback may occur in a separate thread */
	      /* it indicates that the data transfer is complete */

	      if (event_command_exec_status != CL_COMPLETE) {
		throw openclerror(event_command_exec_status,"Error executing clEnqueueReadBuffer() ");

	      }

	      release_rwlock_token_set(*writelocks_copy); /* release write lock */

	      delete writelocks_copy;
	      
	      
	    } ));
	adminlock.lock();

	all_finished.emplace_back(prerequisite_events[0]); /* add final event to our return list */
	prerequisite_events.clear();
	
      } else {
	/* nowhere to delegate writelocks_copy  */
	delete writelocks_copy;

	/* Move event prerequisites from output_data_complete to all_finished (reference ownership passes to all_finished) */
	all_finished.insert(all_finished.end(),output_data_complete.begin(),output_data_complete.end());
	
	output_data_complete.clear();
	
      }

      if ((*readlocks_copy)->size() > 0 && input_data_not_needed) {

	/* in this case readlocks_copy delegated on to callback */
	adminlock.unlock(); /* release adminlock in case our callback happens immediately */
	
	clSetEventCallback(input_data_not_needed,CL_COMPLETE,snde_opencl_callback,(void *)new std::function<void(cl_event,cl_int)>([ readlocks_copy ](cl_event event, cl_int event_command_exec_status) {
	      /* NOTE: This callback may occur in a separate thread */
	      /* it indicates that the input data is no longer needed */
	      
	      if (event_command_exec_status != CL_COMPLETE) {
		throw openclerror(event_command_exec_status,"Error from input_data_not_needed prerequisite ");
		
	      }
	      
	      release_rwlock_token_set(*readlocks_copy); /* release read lock */

	      delete readlocks_copy;
	      
	      
	    } ));
	adminlock.lock();

	
      } else  {
	/* nowhere to delegate readlocks_copy  */
	delete readlocks_copy;

	/* Move event prerequisites from input_data_not_needed to all_finished (reference ownership passes to all_finished) */
	if (input_data_not_needed) all_finished.emplace_back(input_data_not_needed);
	input_data_not_needed=NULL;

      }
      
      return all_finished;
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
      /* create the cache manager since it doesn't exist */
      manager->set_cache("OpenCL",std::make_shared<openclcachemanager>(manager));
    } 
    return std::dynamic_pointer_cast<openclcachemanager>(manager->get_cache("OpenCL"));
  }
    
  class OpenCLBuffer_info {
  public:
    std::shared_ptr<arraymanager> manager;
    std::shared_ptr<openclcachemanager> cachemanager;
    cl_command_queue queue;  /* counted by clRetainCommandQueue */
    cl_mem mem; /* counted by clRetainMemObject */
    void **allocatedptr;
    void **arrayptr;
    rwlock_token_set readlocks;
    rwlock_token_set writelocks;
    std::shared_ptr<std::vector<rangetracker<markedregion>>> arraywriteregions;
    
    OpenCLBuffer_info(std::shared_ptr<arraymanager> manager,
		      cl_command_queue queue,  /* adds new reference */
		      cl_mem mem, /* adds new reference */
		      void **allocatedptr,
		      void **arrayptr,
		      rwlock_token_set readlocks,
		      rwlock_token_set writelocks,
		      std::shared_ptr<std::vector<rangetracker<markedregion>>> arraywriteregions) 
    {
      this->manager=manager;
      this->cachemanager=get_opencl_cache_manager(manager);
      clRetainCommandQueue(queue);
      this->queue=queue;
      clRetainMemObject(mem);
      this->mem=mem;
      this->allocatedptr=allocatedptr;
      this->arrayptr=arrayptr;
      this->readlocks=readlocks;
      this->writelocks=writelocks;
      this->arraywriteregions=arraywriteregions;
    }
    OpenCLBuffer_info(const OpenCLBuffer_info &orig)
    {
      this->manager=orig.manager;
      this->cachemanager=orig.cachemanager;
      clRetainCommandQueue(orig.queue);
      this->queue=orig.queue;
      clRetainMemObject(orig.mem);
      this->mem=orig.mem;
      this->allocatedptr=orig.allocatedptr;
      this->arrayptr=orig.arrayptr;
      this->readlocks=orig.readlocks;
      this->writelocks=orig.writelocks;
      this->arraywriteregions=orig.arraywriteregions;
      
      //for (auto & event: this->fill_events) {
      //clRetainEvent(event);
      //}
    }
    
    OpenCLBuffer_info& operator=(const OpenCLBuffer_info &)=delete; /* copy assignment disabled (for now) */
    ~OpenCLBuffer_info()
    {
      clReleaseCommandQueue(queue);
      clReleaseMemObject(mem);

    }

    
  };
  
  class OpenCLBuffers {
    // Class for managing array of opencl buffers returned by the
    // opencl array manager
  public:
    cl_context context;  /* counted by clRetainContext() */
    rwlock_token_set locks;
    std::shared_ptr<std::vector<rangetracker<markedregion>>> arrayreadregions;
    std::shared_ptr<std::vector<rangetracker<markedregion>>> arraywriteregions;

    std::unordered_map<void **,OpenCLBuffer_info> buffers; /* indexed by arrayidx */

    std::vector<cl_event> fill_events; /* each counted by clRetainEvent() */
     
    OpenCLBuffers(cl_context context,rwlock_token_set locks,std::shared_ptr<std::vector<rangetracker<markedregion>>> arrayreadregions,std::shared_ptr<std::vector<rangetracker<markedregion>>> arraywriteregions)
    {
      clRetainContext(context);
      this->context=context;
      this->locks=locks;
      this->arrayreadregions=arrayreadregions;
      this->arraywriteregions=arraywriteregions;
    }

    /* no copying */
    OpenCLBuffers(const OpenCLBuffers &) = delete;
    OpenCLBuffers & operator=(const OpenCLBuffers &) = delete;
  
    ~OpenCLBuffers() {
      clReleaseContext(context);
      
      /* release each cl_event */
      for (auto & ev: fill_events) {
	clReleaseEvent(ev);
      }


    }
    
    cl_mem Mem_untracked(void **arrayptr)
    /* Returns unprotected pointer (ref count not increased */
    {
      return buffers.at(arrayptr).mem;
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
    

    
    cl_int SetBufferAsKernelArg(cl_kernel kernel, cl_uint arg_index, void **arrayptr)
    {
      cl_mem mem;

      mem=Mem_untracked(arrayptr);
      return clSetKernelArg(kernel,arg_index,sizeof(mem),&mem);
      
    }
  
    void AddBuffer(std::shared_ptr<arraymanager> manager,cl_command_queue queue, void **allocatedptr, void **arrayptr,bool write_only=false)
    {

      //// accumulate preexisting locks + locks in all buffers together
      //for (auto & arrayptr_buf : buffers) {
      //  merge_into_rwlock_token_set(locks,arrayptr_buf.second.readlocks);
      //  merge_into_rwlock_token_set(locks,arrayptr_buf.second.writelocks);
      //}

      rwlock_token_set readlocks,writelocks;
      cl_mem mem;
      snde_index offset;
      std::vector<cl_event> new_fill_events;
      std::shared_ptr<openclcachemanager> cachemanager=get_opencl_cache_manager(manager);
      
      std::tie(readlocks,writelocks,mem,offset,new_fill_events) = cachemanager->GetOpenCLBuffer(locks,context,queue,allocatedptr,arrayptr,arrayreadregions,arraywriteregions,write_only);

      /* move fill events into our master list */
      fill_events.insert(fill_events.end(),new_fill_events.begin(),new_fill_events.end());
      
            
      buffers.emplace(std::make_pair(arrayptr,OpenCLBuffer_info(manager,
								queue, 
								mem,
								allocatedptr,
							        arrayptr,
								readlocks,
								writelocks,
								arraywriteregions)));
      clReleaseMemObject(mem); /* remove extra reference */
      
      
      // add this lock to our database of preexisting locks 
      //locks.push_back(buffers[name][1]); 
    }

    cl_int AddBufferAsKernelArg(std::shared_ptr<arraymanager> manager,cl_command_queue queue,cl_kernel kernel,cl_uint arg_index,void **allocatedptr, void **arrayptr)
    {
      AddBuffer(manager,queue,allocatedptr,arrayptr);
      return SetBufferAsKernelArg(kernel,arg_index,arrayptr);
    }

    
    void RemBuffer(void **arrayptr,cl_event input_data_not_needed,std::vector<cl_event> output_data_complete,bool wait)
    /* Does not decrement refcount of waitevents */
    {
      /* Remove and unlock buffer */
      OpenCLBuffer_info &info=buffers.at(arrayptr);
      
      std::vector<cl_event> all_finished=info.cachemanager->ReleaseOpenCLBuffer(info.readlocks,info.writelocks,context,info.queue,info.mem,info.allocatedptr,arrayptr,info.arraywriteregions,input_data_not_needed,output_data_complete);

      if (wait) {
	clWaitForEvents(all_finished.size(),&all_finished[0]);
      }

      for (auto & ev: all_finished) {
	clReleaseEvent(ev);
      }
      
      /* remove from hash table */
      buffers.erase(arrayptr);
    
    }

    void RemBuffers(cl_event input_data_not_needed,std::vector<cl_event> output_data_complete,bool wait)
    {
      for (std::unordered_map<void **,OpenCLBuffer_info>::iterator nextbuffer = buffers.begin();
	   nextbuffer != buffers.end();) {
	std::unordered_map<void **,OpenCLBuffer_info>::iterator thisbuffer=nextbuffer;
	nextbuffer++;

	RemBuffer(thisbuffer->second.arrayptr,input_data_not_needed,output_data_complete,wait);
	
	
      }
    }
    
    void RemBuffers(cl_event input_data_not_needed,cl_event output_data_complete,bool wait)
    {
      std::vector<cl_event> output_data_complete_vector{output_data_complete};
      RemBuffers(input_data_not_needed,output_data_complete_vector,wait);
    }
  };
  
};
#endif /* SNDE_OPENCLCACHEMANAGER_HPP */
