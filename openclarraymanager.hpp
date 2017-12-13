#ifndef SNDE_OPENCLARRAYMANAGER_HPP
#define SNDE_OPENCLARRAYMANAGER_HPP


#include <cstring>
#include <CL/opencl.h>

#include "arraymanager.hpp"
#include "validitytracker.hpp"
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

  };
  


  /* openclarraymanager manages access to OpenCL buffers
     of type CL_MEM_READ_WRITE, and helps manage
     the copying back-and-forth of such buffers
     to main memory */

  /* openclarrayinfo is used as a key for a std::unordered_map
     to look up opencl buffers. It also reference counts the 
     context so that it doesn't disappear on us */
  class openclarrayinfo {
  public:
    cl_context context; /* when on openclarraymanager's arrayinfomap, held in memory by clRetainContext() */
    /* could insert some flag to indicate use of zero-copy memory */
    void **arrayptr;
    
    openclarrayinfo(cl_context context, void **arrayptr)
    {
      clRetainContext(this->context); /* increase refcnt */
      this->context=context;
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
    // openclbuffer is our lowlevel wrapper used by openclarraymanager
    // for its internal buffer table
    
    // access serialization managed by our parent openclarraymanager's
    // admin mutex, which should be locked when these methods
    // (except realloc callback) are called.
  public:
    cl_mem buffer; /* cl reference count owned by this object */
    size_t elemsize;
    validitytracker<openclregion> invalidity;
    void **arrayptr;
    std::shared_ptr<std::function<void()>> realloc_callback;
    std::shared_ptr<allocator> alloc;
    
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
      
      buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,nelem*elemsize,*arrayptr,&errcode_ret);

      if (errcode_ret != CL_SUCCESS) {
	throw openclerror(errcode_ret,"Error creating buffer of size %d",(long)(nelem*elemsize));
      }
      
      invalidity.mark_all(nelem);

      /* Need to register for notification of realloc
	 so we can re-allocate the buffer! */
      realloc_callback=std::make_shared<std::function<void()>>([this,context,arraymanageradminmutex]() {
	  /* Note: we're not managing context, but presumably the openclarrayinfo that is our key in buffer_map will prevent the context from being freed */
	  
	  std::lock_guard<std::mutex> lock(*arraymanageradminmutex);
	  snde_index numelem;
	  cl_int lambdaerrcode_ret=CL_SUCCESS;
	  numelem=this->alloc->total_nelem(); /* determine new # of elements */
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
      alloc->unregister_realloc_callback(realloc_callback);
      
      clReleaseMemObject(buffer);
      buffer=NULL;
    }
  };


  
  

  class openclarraymanager : public arraymanager {
  public: 
    /* Look up allocator by __allocated_array_pointer__ only */
    /* mapping index is arrayptr, returns allocator */
    std::unordered_map<void **,allocationinfo> allocators;
    std::shared_ptr<memallocator> _memalloc;
    // locker defined by arraymanager base class
    std::mutex admin; /* lock our data structures, including buffer_map and descendents. We are allowed to call allocators/lock managers while holding 
			 this mutex, but they are not allowed to call us and only lock their own data structures, 
			 so a locking order is ensured and no deadlock is possible */

    std::unordered_map<openclarrayinfo,std::shared_ptr<_openclbuffer>> buffer_map;
    
    
    openclarraymanager(std::shared_ptr<memallocator> memalloc) {
      _memalloc=memalloc;
      locker = std::make_shared<lockmanager>();
    }
    openclarraymanager(const openclarraymanager &)=delete; /* copy constructor disabled */
    openclarraymanager& operator=(const openclarraymanager &)=delete; /* assignment disabled */


    /** Will need a function ReleaseOpenCLBuffer that takes an event
        list indicating dependencies. This function queues any necessary
	transfers (should it clFlush()?) to transfer data from the 
	buffer back into host memory. This function will take an 
    rwlock_token_set (that will be released once said transfer
    is complete) and some form of 
    pointer to the openclbuffer. It will need to queue the transfer
    of any writes, and also clReleaseEvent the fill_event fields 
    of the invalidregions, and remove the markings as invalid */

    std::tuple<rwlock_token_set,cl_mem,snde_index,std::vector<cl_event>> GetOpenCLBuffer(std::vector<rwlock_token_set> ownedlocks,cl_context context, cl_command_queue queue, void **allocatedptr, void **arrayptr,cl_mem_flags flags,snde_index firstelem,snde_index numelems) /* numelems may be SNDE_INDEX_INVALID to indicate all the way to the end */
    /* note cl_mem_flags does NOT determine what type of OpenCL buffer we get, but rather what
       our kernel is going to do with it, i.e. CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY or CL_MEM_READ_WRITE */
    /* returns new rwlock_token_set representing read or write lock (per flags) ... let this fall out of scope to release it. */
    /* returns cl_mem... will need to call clReleaseMemObject() on this */
    /* returns snde_index representing the offset, in units of elemsize into the cl_mem of the 'firstelem'  */
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

      std::lock_guard<std::mutex> adminlock(admin);
      
      openclarrayinfo arrayinfo=openclarrayinfo(context,arrayptr);
      std::unordered_map<openclarrayinfo,std::shared_ptr<_openclbuffer>>::iterator buffer;
      std::shared_ptr<allocator> alloc=allocators[arrayptr].alloc;
      std::shared_ptr<_openclbuffer> oclbuffer;
      std::vector<cl_event> ev;
      validitytracker<openclregion>::iterator invalidregion;

      if (numelems==SNDE_INDEX_INVALID) {
	numelems=alloc->total_nelem()-firstelem;
      }
      
      /* obtain read lock on this array */
      rwlock_token_set lock;
      if (flags==CL_MEM_READ_ONLY) {
	lock=locker->get_locks_read_array_region(ownedlocks,allocatedptr,firstelem,numelems);
      } else {
	lock=locker->get_locks_write_array_region(ownedlocks,allocatedptr,firstelem,numelems);
      }
      
      buffer=buffer_map.find(arrayinfo);
      if (buffer == buffer_map.end()) {
	/* need to create buffer */
	oclbuffer=std::make_shared<_openclbuffer>(context,alloc,alloc->arrays[allocators[arrayptr].allocindex].elemsize,arrayptr,&admin);
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
      for (invalidregion=oclbuffer->invalidity.begin();invalidregion != oclbuffer->invalidity.end();invalidregion++) {
	if (invalidregion->second->fill_event) {
	  clRetainEvent(invalidregion->second->fill_event);
	  ev.emplace_back(invalidregion->second->fill_event); 
	  
	}
      }
      
      
      if (flags != CL_MEM_WRITE_ONLY) { /* No need to enqueue transfer if kernel is strictly write */


	validitytracker<openclregion> invalid_regions=oclbuffer->invalidity.get_regions(firstelem,numelems);

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
      
      clRetainMemObject(oclbuffer->buffer); /* ref count for returned cl_mem pointer */
      
      return std::make_tuple(lock,oclbuffer->buffer,(snde_index)firstelem,ev);
    }


    std::vector<cl_event> ReleaseOpenCLBuffer(rwlock_token_set buffertoks,cl_context context, cl_command_queue queue, cl_mem mem, void **allocatedptr, void **arrayptr, cl_mem_flags flags, snde_index firstelem, snde_index numelems, std::vector<cl_event> waitevents,bool wait)
    {
      /* Call this when you are done using the buffer. If you had 
	 a write lock it will queue a transfer that will 
	 update the CPU memory from the buffer before the 
	 locks actually get released 

	 Note that all copies of buffertoks will 
	 no longer represent anything after this call

      */ 
      /* Does not reduce refcount of mem or waitevents */

      std::shared_ptr<_openclbuffer> oclbuffer;
      validitytracker<openclregion>::iterator invalidregion;
      std::vector<cl_event> ev(waitevents);

      /* make copy of buffertoks to delegate to thread... create pointer so it is definitely safe to delegate */
      rwlock_token_set *buffertoks_copy = new rwlock_token_set(clone_rwlock_token_set(buffertoks));

      release_rwlock_token_set(buffertoks); /* release our copy */

      /* capture our admin lock */
      std::unique_lock<std::mutex> adminlock(admin);

      std::shared_ptr<allocator> alloc=allocators[arrayptr].alloc;

      /* create arrayinfo key */
      openclarrayinfo arrayinfo=openclarrayinfo(context,arrayptr);
      
      if (numelems==SNDE_INDEX_INVALID) {
	numelems=alloc->total_nelem()-firstelem;
      }


      oclbuffer=buffer_map[arrayinfo]; /* buffer should exist because should have been created in GetOpenCLBuffer() */

      /* check for pending events and accumulate them into wait list */
      for (invalidregion=oclbuffer->invalidity.begin();invalidregion != oclbuffer->invalidity.end();invalidregion++) {
	if (invalidregion->second->fill_event) {
	  clRetainEvent(invalidregion->second->fill_event);
	  ev.emplace_back(invalidregion->second->fill_event); 
	  
	}
      }
      
      if (flags != CL_MEM_READ_ONLY) { /* No need to enqueue transfer if kernel was not strictly read */
	/* in this case buffertoks_copy delegated on to callback */
	
	/* don't worry about invalidity here because presumably 
	   that was taken care of before we started writing
	   (validity is a concept that applies to the GPU buffer, 
	   not the memory buffer .... memory buffer is invalid
	   but noone will see that because it will be valid before
	   we release the write lock */

	/* ***!!! Somehow we need to notify any OTHER gpu buffers
	   pointing at this same array that they need to be 
	   marked invalid... before it gets unlocked */

	
	cl_event newevent=NULL;
	/* Perform operation to transfer data from buffer object
	   to memory */
	/* wait for all other events as it is required for a 
	   partial write to be meaningful */
	snde_index offset=firstelem*oclbuffer->elemsize;
	clEnqueueReadBuffer(queue,oclbuffer->buffer,CL_FALSE,offset,(numelems)*oclbuffer->elemsize,(char *)*arrayptr + offset,ev.size(),&ev[0],&newevent);

	/* now that it is enqueued we can replace our event list 
	   with this newevent */

	for (auto & oldevent : ev) {
	  clReleaseEvent(oldevent);
	}
	ev.clear();
	
	ev.emplace_back(newevent); /* add new event to our set, eating our referencee */

	adminlock.unlock(); /* release adminlock in case our callback happens immediately */
	
	clSetEventCallback(newevent,CL_COMPLETE,snde_opencl_callback,(void *)new std::function<void(cl_event,cl_int)>([ buffertoks_copy ](cl_event event, cl_int event_command_exec_status) {
	      /* NOTE: This callback may occur in a separate thread */
	      /* it indicates that the data transfer is complete */

	      if (event_command_exec_status != CL_COMPLETE) {
		throw openclerror(event_command_exec_status,"Error executing clEnqueueReadBuffer() ");

	      }

	      release_rwlock_token_set(*buffertoks_copy); /* release write lock */

	      delete buffertoks_copy;
	      
	      
	    } ));
	
	
      } else {
	/* nowhere to delegate buffertoks_copy  */
	delete buffertoks_copy;
	
      }

      return ev;
    }
    
    /* ***!!! Need a method to throw away all cached buffers with a particular context !!!*** */
    
    virtual void add_allocated_array(void **arrayptr,size_t elemsize,snde_index totalnelem)
    {
      std::lock_guard<std::mutex> adminlock(admin);

      // Make sure arrayptr not already managed
      assert(allocators.find(arrayptr)==allocators.end());

      allocators[arrayptr]=allocationinfo{std::make_shared<allocator>(_memalloc,locker,arrayptr,elemsize,totalnelem),0};
      locker->addarray(arrayptr);
      
    
    }
  
    virtual void add_follower_array(void **allocatedptr,void **arrayptr,size_t elemsize)
    {
      std::lock_guard<std::mutex> adminlock(admin);

      std::shared_ptr<allocator> alloc=allocators[allocatedptr].alloc;
      // Make sure arrayptr not already managed
      assert(allocators.find(arrayptr)==allocators.end());
      

      allocators[arrayptr]=allocationinfo{alloc,alloc->add_other_array(arrayptr,elemsize)};
      
    }
    virtual snde_index alloc(void **allocatedptr,snde_index nelem)
    {
      std::lock_guard<std::mutex> adminlock(admin);
      std::shared_ptr<allocator> alloc=allocators[allocatedptr].alloc;
      return alloc->alloc(nelem);    
    }

    virtual void free(void **allocatedptr,snde_index addr,snde_index nelem)
    {
      std::lock_guard<std::mutex> adminlock(admin);
      std::shared_ptr<allocator> alloc=allocators[allocatedptr].alloc;
      alloc->free(addr,nelem);    
    }

    virtual void clear() /* clear out all references to allocated and followed arrays */
    {
      std::lock_guard<std::mutex> adminlock(admin);
      allocators.clear();
    }


    ~openclarraymanager() {
      // allocators.clear(); (this is implicit)
    }
  };

  struct OpenCLBuffer_info {
    std::shared_ptr<openclarraymanager> manager;
    cl_context context;  /* counted by clRetainContext() */
    cl_command_queue queue;  /* counted by clRetainCommandQueue */
    rwlock_token_set tokens;
    cl_mem mem; /* counted by clRetainMemObject */
    snde_index index_of_firstelem;
    void **allocatedptr;
    void **arrayptr;
    cl_mem_flags flags;
    snde_index firstelem;
    snde_index numelems;
    std::vector<cl_event> events; /* each counted by clRetainEvent() */
  };
  
  class OpenCLBuffers {
    // Class for managing array of opencl buffers returned by the
    // opencl array manager
  public:
    std::vector<rwlock_token_set> preexisting_locks;

    std::unordered_map<std::string,struct OpenCLBuffer_info> buffers;

    
    OpenCLBuffers(std::vector<rwlock_token_set> preexisting_locks=std::vector<rwlock_token_set>())
    {
      this->preexisting_locks=preexisting_locks;
    };

    /* no copying */
    OpenCLBuffers(const OpenCLBuffers &) = delete;
    OpenCLBuffers & operator=(const OpenCLBuffers &) = delete;

    ~OpenCLBuffers() {
      for (auto & name_buf : buffers) {
	/* release the cl_mem */
	clReleaseMemObject(name_buf.second.mem);

	/* release each cl_event */
	for (auto & ev: name_buf.second.events) {
	  clReleaseEvent(ev);
	}

	/* release queue and context */
	clReleaseCommandQueue(name_buf.second.queue);
	clReleaseContext(name_buf.second.context);

      }
    }

    cl_mem Mem_untracked(std::string name)
    /* Returns unprotected pointer (ref count not increased */
    {
      return buffers[name].mem;
    }

    std::vector<cl_event> Events_untracked(std::string name)
    /* Returns vector of unprotected pointers (ref count not increased */
    {
      return buffers[name].events;
    }

    
    void AddBuffer(std::string name,std::shared_ptr<openclarraymanager> manager,cl_context context,cl_command_queue queue, void **allocatedptr, void **arrayptr, cl_mem_flags flags, snde_index firstelem,snde_index numelems)
    {

      // accumulate preexisting locks + locks in all buffers together
      std::vector<rwlock_token_set> all_locks(preexisting_locks);
      for (auto & name_buf : buffers) {
	all_locks.push_back(name_buf.second.tokens); 
      }

      std::tuple<rwlock_token_set,cl_mem,snde_index,std::vector<cl_event>> token_mem_offset_events=manager->GetOpenCLBuffer(all_locks,context,queue,allocatedptr,arrayptr,flags,firstelem,numelems);
      
      struct OpenCLBuffer_info buf = { manager,
				       context,
				       queue,
				       std::get<0>(token_mem_offset_events), /* token */
				       std::get<1>(token_mem_offset_events), /* mem... We take responsiblity for the ownership of the cl_mem returned by GetOpenCLBuffer() */
				       std::get<2>(token_mem_offset_events), /* index_of_firstelem */
				       allocatedptr,
				       arrayptr,
				       flags,
				       firstelem,
				       numelems,
				       std::get<3>(token_mem_offset_events), /* we take responsibility for the ownership of the events returned by GetOpenCLBuffer() */
      };
      clRetainContext(buf.context);
      clRetainCommandQueue(buf.queue);
      
      buffers[name]=buf;
      
      // add this lock to our database of preexisting locks 
      //all_locks.push_back(buffers[name][1]); 
    }

    void RemBuffer(std::string name,std::vector<cl_event> waitevents,bool wait)
    /* Does not decrement refcount of waitevents */
    {
      /* Remove and unlock buffer */
      buffers[name].manager->ReleaseOpenCLBuffer(buffers[name].tokens,buffers[name].context,buffers[name].queue,buffers[name].mem,buffers[name].allocatedptr,buffers[name].arrayptr,buffers[name].flags,buffers[name].firstelem,buffers[name].numelems,waitevents,wait);
      
      /* release the cl_mem */
      clReleaseMemObject(buffers[name].mem);
      
      /* release each cl_event in the original events 
	 from when this buffer was opened. */
      for (auto & ev: buffers[name].events) {
	clReleaseEvent(ev);
      }
      clReleaseCommandQueue(buffers[name].queue);
      clReleaseContext(buffers[name].context);
      
      buffers.erase(name);
    }
  };

  
};
#endif /* SNDE_OPENCLARRAYMANAGER_HPP */
