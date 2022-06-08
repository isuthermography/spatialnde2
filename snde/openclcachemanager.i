%shared_ptr(snde::openclcachemanager);
snde_rawaccessible(snde::openclcachemanager);
%shared_ptr(snde::openclcacheentry);
snde_rawaccessible(snde::openclcacheentry);

%shared_ptr(snde::OpenCLBuffer_info);
snde_rawaccessible(snde::OpenCLBuffer_info);
%shared_ptr(snde::OpenCLBuffers);
snde_rawaccessible(snde::OpenCLBuffers);
//%shared_ptr(snde::opencldirtyregion);
//snde_rawaccessible(snde::opencldirtyregion);
%shared_ptr(snde::openclregion);
snde_rawaccessible(snde::openclregion);

//%shared_ptr(snde::cmemallocator);

%{
  
#include "openclcachemanager.hpp"
%}

/*
%typemap(out) std::tuple<rwlock_token_set,rwlock_token_set,cl_mem,snde_index,std::vector<cl_event>> (PyObject *pyopencl=NULL,PyObject *clEvent=NULL,PyObject *clEvent_from_int_ptr=NULL,PyObject *clMem=NULL,PyObject *clMem_from_int_ptr=NULL,PyObject *EventTuple,std::vector<cl_event> &eventvec) {
  pyopencl = PyImport_ImportModule("pyopencl");
  if (!pyopencl) SWIG_fail; // raise exception up 
 
  clEvent=PyObject_GetAttrString(pyopencl,"Event");
  clEvent_from_int_ptr=PyObject_GetAttrString(clEvent,"from_int_ptr");
  clMem=PyObject_GetAttrString(pyopencl,"MemoryObject");
  clMem_from_int_ptr=PyObject_GetAttrString(clMem,"from_int_ptr");

  
  $result = PyTuple_New(5);
  // Substituted code for converting cl_context here came
  // from a typemap substitution "$typemap(out,cl_context)"
  std::shared_ptr<snde::rwlock_token_set_content> result0 = std::get<0>(*&$1);
  std::shared_ptr<snde::rwlock_token_set_content> *smartresult0 = bool(result0) ? new std::shared_ptr<snde::rwlock_token_set_content>(result0 SWIG_NO_NULL_DELETER_SWIG_POINTER_NEW) :0;
  
  
  PyTuple_SetItem($result,0,SWIG_NewPointerObj(%as_voidptr(smartresult0), $descriptor(std::shared_ptr< snde::rwlock_token_set_content > *), SWIG_POINTER_OWN));

  std::shared_ptr<snde::rwlock_token_set_content> result1 = std::get<1>(*&$1);
  std::shared_ptr<snde::rwlock_token_set_content> *smartresult1 = bool(result1) ? new std::shared_ptr<snde::rwlock_token_set_content>(result1 SWIG_NO_NULL_DELETER_SWIG_POINTER_NEW) :0;
  
  
  PyTuple_SetItem($result,1,SWIG_NewPointerObj(%as_voidptr(smartresult1), $descriptor(std::shared_ptr< snde::rwlock_token_set_content > *), SWIG_POINTER_OWN));
  PyTuple_SetItem($result,2,PyObject_CallFunction(clMem_from_int_ptr,(char *)"KO",(unsigned long long)((uintptr_t)std::get<2>(*&$1)),Py_False));
  
  PyTuple_SetItem($result,3,SWIG_From_unsigned_long_SS_long_SS_long(static_cast<unsigned long long>std::get<3>(*&$1)));
  
  eventvec=std::get<4>(*&$1);
  EventTuple=PyTuple_New(eventvec.size());
  for (cnt=0;cnt < eventvec.size();cnt++) {
    PyTuple_SetItem(EventTuple,cnt,PyObject_CallFunction(clEvent_from_int_ptr,(char *)"KO",(unsigned long long)((uintptr_t)eventvec[cnt]),Py_False));
  }

  PyTuple_SetItem($result,4,EventTuple);
  

  Py_XDECREF(clMem_from_int_ptr);
  Py_XDECREF(clMem);
  Py_XDECREF(clEvent_from_int_ptr);
  Py_XDECREF(clEvent);
  Py_XDECREF(pyopencl);
}

*/

extern "C" void snde_opencl_callback(cl_event event, cl_int event_command_exec_status, void *user_data);

namespace snde {

  class ndarray_recording_ref;
  class multi_ndarray_recording;
  class recording_storage;
  
  
  /*
  class opencldirtyregion {
  public:
    // contents locked by cachemanager's admin mutex 
    snde_index regionstart;
    snde_index regionend;
    //std::weak_ptr<recording_storage> owning_storage;
    cl::Event FlushDoneEvent;
    bool FlushDoneEventComplete;
    
    //std::condition_variable complete_condition; // associated with the cachemanager's admin mutex
    
    opencldirtyregion(snde_index regionstart,snde_index regionend,std::shared_ptr<recording_storage> owning_storage);
    opencldirtyregion& operator=(const opencldirtyregion &)=delete; // copy assignment disabled 
    opencldirtyregion(const opencldirtyregion &orig) = default;
    
    bool attempt_merge(opencldirtyregion &later); // for now always returns false    

    // breakup method ends this region at breakpoint and returns
       a new region starting at from breakpoint to the prior end 
    std::shared_ptr<opencldirtyregion> sp_breakup(snde_index breakpoint,std::shared_ptr<recording_storage> storage);
    
  };
  */

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
    //cl::Device device;
    /* could insert some flag to indicate use of zero-copy memory (maybe not; this is the map key) */
    void **arrayptr;
    size_t numelem;
    
    openclarrayinfo(cl::Context context, void **arrayptr,size_t numelem);
    
    openclarrayinfo(const openclarrayinfo &orig)=default; /* copy constructor */    
    //openclarrayinfo& operator=(const openclarrayinfo &orig)=default; /* copy assignment operator */
    ~openclarrayinfo()=default;

    // equality operator for std::unordered_map
    bool operator==(const openclarrayinfo &b) const;

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
  
  
  
  
  class openclcacheentry : public cached_recording {
    // openclbuffer is our lowlevel wrapper used by openclcachemanager
    // for its internal buffer table
    
    // access serialization managed by our parent openclcachemanager's
    // admin mutex, which should be locked when these methods
    // (except realloc callback) are called.

    // Note that we can briefly have two openclcacheentries for the same
    // memory, as the index includes numelem and numelem may increase if
    // the storage is reallocated. This is not a problem because
    // they will have separate invalidity and dirtyregions. The older
    // cache entry will expire once it is no longer referenced. 
  public:
    cl::Buffer buffer; /* cl reference count owned by this object */
    //std::weak_ptr<allocator> alloc; /* weak pointer to the allocator (if any) because we don't want to inhibit freeing of this (all we need it for is our reallocation callback) */
    size_t numelem;
    size_t elemsize;
    void **arrayptr;
    //std::shared_ptr<std::function<void(snde_index)>> pool_realloc_callback;

    cl::Device nominal_device; // most recent device we have used for transfers, may be null...
    
    //rangetracker<openclregion> invalidity; // invalidity is where the GPU copy needs to be updated

    // dirtyregions are where the CPU copy needs to be updated from the GPU copy
    //rangetracker<opencldirtyregion> _dirtyregions; /* track dirty ranges during a write (all regions that are locked for write... Note that persistent iterators (now pointers) may be present if the FlushDoneEvent exists but the FlushDoneEventComplete is false  */

    openclcacheentry(cl::Context context,std::shared_ptr<snde::allocator> alloc,snde_index total_nelem,size_t elemsize,void **arrayptr, std::mutex *cachemanageradminmutex);
    
    openclcacheentry(const openclcacheentry &)=delete; /* copy constructor disabled */
    openclcacheentry& operator=(const openclcacheentry &)=delete; /* copy assignment disabled */
    ~openclcacheentry();
     
    
    void mark_as_gpu_modified(std::shared_ptr<recording_storage> storage,snde_index pos,snde_index nelem);

  };


  
  
  /* openclcachemanager manages opencl caches for arrays 
     (for now) managed by a single arraymanager/lockmanager  */ 
  class openclcachemanager : public cachemanager {
  public:
    std::string name; // guaranteed unique. created by get_cache_name() function
    
    //std::shared_ptr<memallocator> _memalloc;
    // locker defined by arraymanager base class
    //std::mutex admin; // lock our data structures, including buffer_map and descendents. We are allowed to call allocators/lock managers while holding 
    // this mutex, but they are not allowed to call us and only lock their own data structures, 
    //			 so a locking order is ensured and no deadlock is possible */
    //std::unordered_map<context_device,cl::CommandQueue,context_device_hash,context_device_equal> queue_map;    
    //std::unordered_map<openclarrayinfo,std::weak_ptr<openclcacheentry>,openclarrayinfo_hash/* ,openclarrayinfo_equal*/> buffer_map;
    //std::unordered_map<void **,std::vector<openclarrayinfo>> buffers_by_array;
    
    
    openclcachemanager();
    openclcachemanager(const openclcachemanager &)=delete; /* copy constructor disabled */
    openclcachemanager& operator=(const openclcachemanager &)=delete; /* assignment disabled */
    virtual ~openclcachemanager()=default;


    cl::CommandQueue _get_queue(cl::Context context,cl::Device device);    // internal version for when adminlock is alread held     
    cl::CommandQueue get_queue(cl::Context context,cl::Device device);


    /* marks an array region (with exception of particular buffer) as needing to be updated from CPU copy */
    /* This is typically used after our CPU copy has been updated from exceptbuffer, to push updates out to all of the other buffers */
    virtual void mark_as_invalid_except_buffer(std::shared_ptr<openclcacheentry> exceptbuffer,void **arrayptr,snde_index pos,snde_index numelem);
    virtual void mark_as_gpu_modified(cl::Context context, std::shared_ptr<recording_storage> storage);
    
    
    /* marks an array region as needing to be updated from CPU copy */
    /* This is typically used if the CPU copy is updated directly */
    virtual void mark_as_invalid(void **arrayptr,snde_index base_index,snde_index pos,snde_index numelem); // pos and numelem relative to this particular storage
    virtual void notify_storage_expiration(void **arrayptr,snde_index base_index,snde_index nelem);

    
    // internal use only... initiates transfers of invalid regions prior to setting up a read buffer
    // WARNING: operates in-place on prerequisite event vector ev
    // assumes admin lock is held
    void _TransferInvalidRegions(cl::Context context, cl::Device device,std::shared_ptr<openclcacheentry> oclbuffer,void **arrayptr,snde_index firstelem,snde_index numelem,std::vector<cl::Event> &ev);
    
    //std::shared_ptr<openclcacheentry> _GetBufferObject(std::shared_ptr<recording_storage> storage,cl::Context context, cl::Device device,snde_index nelem,snde_index elemsize,void **arrayptr);     // internal use only; assumes admin lock is held;
      

    std::tuple<rwlock_token_set,cl::Buffer,std::vector<cl::Event>,std::shared_ptr<openclcacheentry>> _GetOpenCLSubBuffer(std::shared_ptr<recording_storage> storage,rwlock_token_set alllocks,cl::Context context, cl::Device device,snde_index substartelem,snde_index subnumelem,bool write,bool write_only=false);     // It is assumed that the caller has the data adequately locked, if needed.
      
    /** Will need a function ReleaseOpenCLBuffer that takes an event
        list indicating dependencies. This function queues any necessary
	transfers (should it clFlush()?) to transfer data from the 
	buffer back into host memory. This function will take an 
    rwlock_token_set (that will be released once said transfer
    is complete) and some form of 
    pointer to the openclbuffer. It will need to queue the transfer
    of any writes, and also clReleaseEvent the fill_event fields 
    of the invalidregions, and remove the markings as invalid */
    

    //std::tuple<rwlock_token_set,cl::Buffer,std::vector<cl::Event>,std::shared_ptr<openclcacheentry>> _GetOpenCLBuffer(std::shared_ptr<recording_storage> storage,rwlock_token_set alllocks,cl::Context context, cl::Device device, bool write,bool write_only);
    //std::tuple<rwlock_token_set,cl::Buffer,std::vector<cl::Event>> GetOpenCLBuffer(std::shared_ptr<recording_storage> storage,rwlock_token_set alllocks,cl::Context context, cl::Device device, bool write,bool write_only=false);
    //std::pair<std::vector<cl::Event>,std::vector<cl::Event>> FlushWrittenOpenCLBuffer(cl::Context context,cl::Device device,std::shared_ptr<recording_storage> storage,std::vector<cl::Event> explicit_prerequisites);
      
    void ForgetOpenCLBuffer(rwlock_token_set locks,cl::Context context, cl::Device device, cl::Buffer mem,std::shared_ptr<recording_storage> storage, cl::Event data_not_needed);
    
    //std::pair<std::vector<cl::Event>,std::shared_ptr<std::thread>> ReleaseOpenCLBuffer(rwlock_token_set locks,cl::Context context, cl::Device device, cl::Buffer mem, std::shared_ptr<recording_storage> storage, cl::Event input_data_not_needed,const std::vector<cl::Event> &output_data_complete);
    
    /* ***!!! Need a method to throw away all cached buffers with a particular context !!!*** */
    
  };


    
  class OpenCLBuffer_info {
  public:
    //std::shared_ptr<openclcachemanager> cachemanager;
    //cl::CommandQueue transferqueue;  /* counted by clRetainCommandQueue */
    cl::Buffer mem; /* counted by clRetainMemObject */
    std::shared_ptr<recording_storage> storage;
    //rwlock_token_set readlocks;
    rwlock_token_set locks;
    std::shared_ptr<openclcacheentry> cacheentry;
    
    OpenCLBuffer_info(//std::shared_ptr<arraymanager> manager,
		      //cl::CommandQueue transferqueue,  /* adds new reference */
		      cl::Buffer mem, /* adds new reference */
		      std::shared_ptr<recording_storage> storage,
		      //rwlock_token_set readlocks,
		      rwlock_token_set locks,
		      std::shared_ptr<openclcacheentry> cacheentry);

    // Copy constructor and copy assignment should be OK
    // because we are using default destructor
    OpenCLBuffer_info(const OpenCLBuffer_info &orig)=default;    
    //OpenCLBuffer_info& operator=(const OpenCLBuffer_info &)=default; 
    
    //OpenCLBuffer_info(const OpenCLBuffer_info &orig)=delete;    
    //OpenCLBuffer_info& operator=(const OpenCLBuffer_info &)=delete; /* copy assignment disabled (for now) */
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
    std::shared_ptr<openclcachemanager> cachemgr;
    
    cl::Context context;  /* counted by clRetainContext() */
    cl::Device device;  /* counted by clRetainDevice() */
    rwlock_token_set all_locks;

    
    //std::unordered_map<OpenCLBufferKey,OpenCLBuffer_info,OpenCLBufferKeyHash> buffers; /* indexed by arrayidx */
    
    std::vector<cl::Event> fill_events; /* each counted by clRetainEvent() */

    bool empty_invalid; // set for default constructed object. We allow move assignment into an empty_invalid object but nothing else
    
    OpenCLBuffers(std::shared_ptr<openclcachemanager> cachemgr,cl::Context context,cl::Device device,rwlock_token_set all_locks);
    OpenCLBuffers();

    /* no copying */
    OpenCLBuffers(const OpenCLBuffers &) = delete;
    OpenCLBuffers & operator=(const OpenCLBuffers &) = delete;
    // We have move assignment so that you can initialize into a default-constructed object.
    //OpenCLBuffers & operator=(OpenCLBuffers &&orig) noexcept;
    OpenCLBuffers(OpenCLBuffers &&) noexcept = delete; // no move constructor
  
    ~OpenCLBuffers();
    
    cl::Buffer Mem(void **arrayptr,snde_index firstelem,snde_index numelem);


    std::vector<cl::Event> FillEvents(void);

    cl_uint NumFillEvents(void);
    

    
    
    cl_int SetBufferAsKernelArg(cl::Kernel kernel, cl_uint arg_index, void **arrayptr,snde_index firstelem,snde_index numelem);
  

    void AddBufferPortion(std::shared_ptr<recording_storage> storage,snde_index start_elem, snde_index length,bool write,bool write_only=false);

    void AddBuffer(std::shared_ptr<recording_storage> storage,bool write,bool write_only=false);
    
    cl_int AddBufferPortionAsKernelArg(std::shared_ptr<recording_storage> storage,snde_index start_elem, snde_index length,cl::Kernel kernel,cl_uint arg_index,bool write,bool write_only=false);

    cl_int AddBufferPortionAsKernelArg(std::shared_ptr<ndarray_recording_ref> ref,snde_index portion_start,snde_index portion_len,cl::Kernel kernel,cl_uint arg_index,bool write,bool write_only);

    cl_int AddBufferPortionAsKernelArg(std::shared_ptr<multi_ndarray_recording> rec,size_t arraynum,snde_index portion_start,snde_index portion_len,cl::Kernel kernel,cl_uint arg_index,bool write,bool write_only);
    
    cl_int AddBufferPortionAsKernelArg(std::shared_ptr<multi_ndarray_recording> rec,std::string arrayname,snde_index portion_start,snde_index portion_len,cl::Kernel kernel,cl_uint arg_index,bool write,bool write_only);

    
    //cl_int AddBufferAsKernelArg(std::shared_ptr<recording_storage> storage,cl::Kernel kernel,cl_uint arg_index,bool write,bool write_only=false);

    cl_int AddBufferAsKernelArg(std::shared_ptr<ndarray_recording_ref> ref,cl::Kernel kernel,cl_uint arg_index,bool write,bool write_only=false);

    cl_int AddBufferAsKernelArg(std::shared_ptr<multi_ndarray_recording> rec,size_t arraynum,cl::Kernel kernel,cl_uint arg_index,bool write,bool write_only);
    cl_int AddBufferAsKernelArg(std::shared_ptr<multi_ndarray_recording> rec,std::string arrayname,cl::Kernel kernel,cl_uint arg_index,bool write,bool write_only);

    /* This indicates that the array has been written to by an OpenCL kernel, 
       and that therefore it needs to be copied back into CPU memory */
    void BufferPortionDirty(std::shared_ptr<recording_storage> storage,snde_index start_elem, snde_index length);
    void BufferDirty(std::shared_ptr<recording_storage> storage);
    void BufferPortionDirty(std::shared_ptr<ndarray_recording_ref> ref,snde_index portion_start, snde_index portion_len);
    void BufferPortionDirty(std::shared_ptr<multi_ndarray_recording> rec,size_t arraynum,snde_index portion_start, snde_index portion_len);
    void BufferPortionDirty(std::shared_ptr<multi_ndarray_recording> rec,std::string arrayname,snde_index portion_start, snde_index portion_len);
    void BufferDirty(std::shared_ptr<ndarray_recording_ref> ref);
    void BufferDirty(std::shared_ptr<multi_ndarray_recording> rec,size_t arraynum);
    void BufferDirty(std::shared_ptr<multi_ndarray_recording> rec,std::string arrayname);

    std::pair<std::vector<cl::Event>,std::vector<cl::Event>> FlushBuffer(std::shared_ptr<recording_storage> storage,std::vector<cl::Event> explicit_prerequisites);
    
    
    
    /* Either specify wait=true, then you can explicitly unlock_rwlock_token_set() your locks because you know they're done, 
       or specify wait=false in which case things may finish later. The only way to make sure they are finished is 
       to obtain a new lock on the same items */
    //void RemSubBuffer(void **arrayptr,snde_index startidx,snde_index numelem,cl::Event input_data_not_needed,std::vector<cl::Event> output_data_complete,bool wait);
    //void RemBuffer(void **arrayptr,cl::Event input_data_not_needed,std::vector<cl::Event> output_data_complete,bool wait);
    void ForgetBuffer(std::shared_ptr<recording_storage> storage,cl::Event data_not_needed);
    std::vector<cl::Event> RemBuffer(std::shared_ptr<recording_storage> storage,snde_index firstelem, snde_index numelem,cl::Event input_data_not_needed,const std::vector<cl::Event> &output_data_complete,bool wait);
    void ForgetBuffer(std::shared_ptr<ndarray_recording_ref> ref,cl::Event data_not_needed);
    std::vector<cl::Event> RemBuffer(std::shared_ptr<ndarray_recording_ref> ref,cl::Event input_data_not_needed,const std::vector<cl::Event> &output_data_complete,bool wait);
    void ForgetBuffer(std::shared_ptr<multi_ndarray_recording> rec,std::string arrayname,cl::Event data_not_needed);

    std::vector<cl::Event> RemBuffer(std::shared_ptr<multi_ndarray_recording> rec,std::string arrayname,cl::Event input_data_not_needed,const std::vector<cl::Event> &output_data_complete,bool wait);

    std::vector<cl::Event> RemBuffers(cl::Event input_data_not_needed,std::vector<cl::Event> output_data_complete,bool wait);
    std::vector<cl::Event> RemBuffers(cl::Event input_data_not_needed,cl::Event output_data_complete,bool wait);
    std::vector<cl::Event> RemBuffers(cl::Event input_data_not_needed,bool wait);
      
  };
  
  
}

