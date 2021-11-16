#ifndef SNDE_OPENCLCACHEMANAGER_HPP
#define SNDE_OPENCLCACHEMANAGER_HPP

#include <cstring>

#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/opencl.hpp>
#endif

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
    cl::Event FlushDoneEvent;
    bool FlushDoneEventComplete;
    std::condition_variable complete_condition; // associated with the cachemanager's admin mutex
    
    opencldirtyregion(snde_index regionstart,snde_index regionend);
    opencldirtyregion& operator=(const opencldirtyregion &)=delete; /* copy assignment disabled */
    opencldirtyregion(const opencldirtyregion &orig) = default;
    
    bool attempt_merge(opencldirtyregion &later); // for now always returns false    

    /* breakup method ends this region at breakpoint and returns
       a new region starting at from breakpoint to the prior end */
    std::shared_ptr<opencldirtyregion> sp_breakup(snde_index breakpoint);
    
  };
  

  class openclregion {
  public:
    snde_index regionstart;
    snde_index regionend;
    cl::Event fill_event; // if not NULL, indicates that there is a pending operation to copy data into this region...
    // if you hold a write lock, it should not be possible for there to
    // be a fill_event except at your request, because a fill_event
    // requires at least a read lock and in order to get your write
    // lock you would have waited for all (other) read locks to be released
    
    openclregion(snde_index regionstart,snde_index regionend);

    openclregion(const openclregion &)=delete; /* copy constructor disabled */
    openclregion& operator=(const openclregion &)=delete; /* copy assignment disabled */
    ~openclregion() = default;
    
    bool attempt_merge(openclregion &later);

    /* breakup method ends this region at breakpoint and returns
       a new region starting at from breakpoint to the prior end */
    std::shared_ptr<openclregion> sp_breakup(snde_index breakpoint);

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
    cl::Context context; /* when on openclcachemanager's arrayinfomap, held in memory by clRetainContext() */
    cl::Device device;
    /* could insert some flag to indicate use of zero-copy memory */
    void **arrayptr;
    
    openclarrayinfo(cl::Context context, cl::Device device,void **arrayptr);
    
    openclarrayinfo(const openclarrayinfo &orig)=default; /* copy constructor */    
    openclarrayinfo& operator=(const openclarrayinfo &orig)=default; /* copy assignment operator */
    ~openclarrayinfo()=default;

    // equality operator for std::unordered_map
    bool operator==(const openclarrayinfo b) const;

  };

  // Need to provide hash implementation for openclarrayinfo so
  // it can be used as a std::unordered_map key
  
  struct openclarrayinfo_hash {
    size_t operator()(const snde::openclarrayinfo & x) const;
  };
  //struct openclarrayinfo_equal {
  //  bool operator()(const snde::openclarrayinfo & x,const snde::openclarrayinfo &y) const
  //  {
  //    return x.context==y.context && x.device==y.device && x.arrayptr==y.arrayptr;
  //  }
  //};
  
  
  
  
  class _openclbuffer {
    // openclbuffer is our lowlevel wrapper used by openclcachemanager
    // for its internal buffer table
    
    // access serialization managed by our parent openclcachemanager's
    // admin mutex, which should be locked when these methods
    // (except realloc callback) are called.
  public:
    cl::Buffer buffer; /* cl reference count owned by this object */
    size_t elemsize;
    rangetracker<openclregion> invalidity; // invalidity is where the GPU copy needs to be updated
    void **arrayptr;
    std::shared_ptr<std::function<void(snde_index)>> pool_realloc_callback;
    std::weak_ptr<allocator> alloc; /* weak pointer to the allocator because we don't want to inhibit freeing of this (all we need it for is our reallocation callback) */

    // dirtyregions are where the CPU copy needs to be updated from the GPU copy
    rangetracker<opencldirtyregion> _dirtyregions; /* track dirty ranges during a write (all regions that are locked for write... Note that persistent iterators (now pointers) may be present if the FlushDoneEvent exists but the FlushDoneEventComplete is false  */

    _openclbuffer(cl::Context context,std::shared_ptr<allocator> alloc,snde_index total_nelem,size_t elemsize,void **arrayptr, std::mutex *arraymanageradminmutex);
    
    _openclbuffer(const _openclbuffer &)=delete; /* copy constructor disabled */
    _openclbuffer& operator=(const _openclbuffer &)=delete; /* copy assignment disabled */
    ~_openclbuffer();
     

    void mark_as_gpu_modified(snde_index pos,snde_index nelem);

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
    std::unordered_map<context_device,cl::CommandQueue,context_device_hash,context_device_equal> queue_map;    
    std::unordered_map<openclarrayinfo,std::shared_ptr<_openclbuffer>,openclarrayinfo_hash/* ,openclarrayinfo_equal*/> buffer_map;
    std::weak_ptr<arraymanager> manager; /* Used for looking up allocators and accessing lock manager (weak to avoid creating a circular reference) */ 
    std::unordered_map<void **,std::vector<openclarrayinfo>> buffers_by_array;
    
    
    openclcachemanager(std::shared_ptr<arraymanager> manager);
    openclcachemanager(const openclcachemanager &)=delete; /* copy constructor disabled */
    openclcachemanager& operator=(const openclcachemanager &)=delete; /* assignment disabled */
    virtual ~openclcachemanager()=default;


    cl::CommandQueue _get_queue(cl::Context context,cl::Device device);    // internal version for when adminlock is alread held     
    cl::CommandQueue get_queue(cl::Context context,cl::Device device);


    /* marks an array region (with exception of particular buffer) as needing to be updated from CPU copy */
    /* This is typically used after our CPU copy has been updated from exceptbuffer, to push updates out to all of the other buffers */
    void mark_as_dirty_except_buffer(std::shared_ptr<_openclbuffer> exceptbuffer,void **arrayptr,snde_index pos,snde_index numelem);
    virtual void mark_as_gpu_modified(cl::Context context, cl::Device device, void **arrayptr,snde_index pos,snde_index len);
    
    
    /* marks an array region as needing to be updated from CPU copy */
    /* This is typically used if the CPU copy is updated directly */
    virtual void mark_as_dirty(std::shared_ptr<arraymanager> specified_manager,void **arrayptr,snde_index pos,snde_index numelem);


    
    // internal use only... initiates transfers of invalid regions prior to setting up a read buffer
    // WARNING: operates in-place on prerequisite event vector ev
    // assumes admin lock is held
    void _TransferInvalidRegions(cl::Context context, cl::Device device,std::shared_ptr<_openclbuffer> oclbuffer,void **arrayptr,snde_index firstelem,snde_index numelem,std::vector<cl::Event> &ev);
    
    std::shared_ptr<_openclbuffer> _GetBufferObject(cl::Context context, cl::Device device,void **arrayptr);       // internal use only; assumes admin lock is held;
      
    std::tuple<rwlock_token_set,cl::Buffer,std::vector<cl::Event>> GetOpenCLSubBuffer(rwlock_token_set alllocks,cl::Context context, cl::Device device, void **arrayptr,snde_index startelem,snde_index numelem,bool write,bool write_only=false);
      
    /** Will need a function ReleaseOpenCLBuffer that takes an event
        list indicating dependencies. This function queues any necessary
	transfers (should it clFlush()?) to transfer data from the 
	buffer back into host memory. This function will take an 
    rwlock_token_set (that will be released once said transfer
    is complete) and some form of 
    pointer to the openclbuffer. It will need to queue the transfer
    of any writes, and also clReleaseEvent the fill_event fields 
    of the invalidregions, and remove the markings as invalid */

    std::tuple<rwlock_token_set,cl::Buffer,std::vector<cl::Event>> GetOpenCLBuffer(rwlock_token_set alllocks,cl::Context context, cl::Device device, void **arrayptr,bool write,bool write_only=false);

    std::pair<std::vector<cl::Event>,std::vector<cl::Event>> FlushWrittenOpenCLBuffer(cl::Context context,cl::Device device,void **arrayptr,std::vector<cl::Event> explicit_prerequisites);
      
    std::pair<std::vector<cl::Event>,std::shared_ptr<std::thread>> ReleaseOpenCLBuffer(rwlock_token_set locks,cl::Context context, cl::Device device, cl::Buffer mem, void **arrayptr, cl::Event input_data_not_needed,std::vector<cl::Event> output_data_complete);
    
    /* ***!!! Need a method to throw away all cached buffers with a particular context !!!*** */
    
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
    //cl::CommandQueue transferqueue;  /* counted by clRetainCommandQueue */
    cl::Buffer mem; /* counted by clRetainMemObject */
    void **arrayptr;
    //rwlock_token_set readlocks;
    rwlock_token_set locks;
    
    OpenCLBuffer_info(std::shared_ptr<arraymanager> manager,
		      //cl::CommandQueue transferqueue,  /* adds new reference */
		      cl::Buffer mem, /* adds new reference */
		      void **arrayptr,
		      //rwlock_token_set readlocks,
		      rwlock_token_set locks);
    
    OpenCLBuffer_info(const OpenCLBuffer_info &orig)=delete;    
    OpenCLBuffer_info& operator=(const OpenCLBuffer_info &)=delete; /* copy assignment disabled (for now) */
    ~OpenCLBuffer_info()=default;

    
  };

  class OpenCLBufferKey {
  public:
    void **array;
    snde_index firstelem;
    snde_index numelem; // Is this really necessary?

    OpenCLBufferKey(void **_array,snde_index _firstelem,snde_index _numelem);
    
    // equality operator for std::unordered_map
    bool operator==(const OpenCLBufferKey b) const;

    
  };


  // Need to provide hash implementation for OpenCLBufferKey so
  // it can be used as a std::unordered_map key
  //
  //namespace std {
  struct OpenCLBufferKeyHash {
    size_t operator()(const OpenCLBufferKey & x) const
    {
      return std::hash<void *>{}((void *)x.array) + std::hash<snde_index>{}(x.firstelem) +std::hash<snde_index>{}(x.numelem);
    }

  };


  
  class OpenCLBuffers {
    // Class for managing array of opencl buffers returned by the
    // opencl array manager... SHOULD ONLY BE USED BY ONE THREAD.
    
  public:
    cl::Context context;  /* counted by clRetainContext() */
    cl::Device device;  /* counted by clRetainDevice() */
    rwlock_token_set all_locks;

    std::unordered_map<OpenCLBufferKey,OpenCLBuffer_info,OpenCLBufferKeyHash> buffers; /* indexed by arrayidx */
    
    std::vector<cl::Event> fill_events; /* each counted by clRetainEvent() */
     
    OpenCLBuffers(cl::Context context,cl::Device device,rwlock_token_set all_locks);

    /* no copying */
    OpenCLBuffers(const OpenCLBuffers &) = delete;
    OpenCLBuffers & operator=(const OpenCLBuffers &) = delete;
  
    ~OpenCLBuffers();
    
    cl::Buffer Mem(void **arrayptr,snde_index firstelem,snde_index numelem);


    std::vector<cl::Event> FillEvents(void);

    cl_uint NumFillEvents(void);
    

    
    cl_int SetBufferAsKernelArg(cl::Kernel kernel, cl_uint arg_index, void **arrayptr,snde_index firstelem,snde_index numelem);
  
    void AddSubBuffer(std::shared_ptr<arraymanager> manager, void **arrayptr,snde_index indexstart,snde_index numelem,bool write,bool write_only=false);

    void AddBuffer(std::shared_ptr<arraymanager> manager, void **arrayptr,bool write,bool write_only=false);
    
    cl_int AddSubBufferAsKernelArg(std::shared_ptr<arraymanager> manager,cl::Kernel kernel,cl_uint arg_index,void **arrayptr,snde_index indexstart,snde_index numelem,bool write,bool write_only=false);

    cl_int AddBufferAsKernelArg(std::shared_ptr<arraymanager> manager,cl::Kernel kernel,cl_uint arg_index,void **arrayptr,bool write,bool write_only=false);
    

    /* This indicates that the array has been written to by an OpenCL kernel, 
       and that therefore it needs to be copied back into CPU memory */
    void BufferDirty(void **arrayptr);

    /* This indicates that the array region has been written to by an OpenCL kernel, 
       and that therefore it needs to be copied back into CPU memory */
    void BufferDirty(void **arrayptr,snde_index pos,snde_index len);

      
    /* This indicates that the array region has been written to by an OpenCL kernel, 
       and that therefore it needs to be copied back into CPU memory */
    void SubBufferDirty(void **arrayptr,snde_index sb_pos,snde_index sb_len);

    /* This indicates that the array region has been written to by an OpenCL kernel, 
       and that therefore it needs to be copied back into CPU memory */
    void SubBufferDirty(void **arrayptr,snde_index sb_pos,snde_index sb_len,snde_index dirtypos,snde_index dirtylen);


    std::pair<std::vector<cl::Event>,std::vector<cl::Event>> FlushBuffer(void **arrayptr,snde_index sb_pos,snde_index sb_len,std::vector<cl::Event> explicit_prerequisites);
    
    
    
    /* Either specify wait=true, then you can explicitly unlock_rwlock_token_set() your locks because you know they're done, 
       or specify wait=false in which case things may finish later. The only way to make sure they are finished is 
       to obtain a new lock on the same items */
    void RemSubBuffer(void **arrayptr,snde_index startidx,snde_index numelem,cl::Event input_data_not_needed,std::vector<cl::Event> output_data_complete,bool wait);
    void RemBuffer(void **arrayptr,cl::Event input_data_not_needed,std::vector<cl::Event> output_data_complete,bool wait);
    
    void RemBuffers(cl::Event input_data_not_needed,std::vector<cl::Event> output_data_complete,bool wait);
    void RemBuffers(cl::Event input_data_not_needed,cl::Event output_data_complete,bool wait);
    void RemBuffers(cl::Event input_data_not_needed,bool wait);
      
  };
  
};
#endif /* SNDE_OPENCLCACHEMANAGER_HPP */
