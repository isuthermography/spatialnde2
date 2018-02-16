%shared_ptr(snde::openclcachemanager);
%shared_ptr(snde::_openclbuffer);
%shared_ptr(snde::OpenCLBuffer_info);
%shared_ptr(snde::OpenCLBuffers);

//%shared_ptr(snde::cmemallocator);

%{
  
#include "openclcachemanager.hpp"
%}


%typemap(out) std::tuple<rwlock_token_set,rwlock_token_set,cl_mem,snde_index,std::vector<cl_event>> (PyObject *pyopencl=NULL,PyObject *clEvent=NULL,PyObject *clEvent_from_int_ptr=NULL,PyObject *clMem=NULL,PyObject *clMem_from_int_ptr=NULL,PyObject *EventTuple,std::vector<cl_event> &eventvec) {
  pyopencl = PyImport_ImportModule("pyopencl");
  if (!pyopencl) SWIG_fail; /* raise exception up */
  
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
    
    openclregion(snde_index regionstart,snde_index regionend);

    ~openclregion();
    
    bool attempt_merge(openclregion &later);

    std::shared_ptr<openclregion> breakup(snde_index breakpoint);

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
    
    openclarrayinfo(cl_context context, void **arrayptr);
    openclarrayinfo(const openclarrayinfo &orig); /* copy constructor */
    
    //openclarrayinfo& operator=(const openclarrayinfo &orig); /* copy assignment operator */

    // equality operator for std::unordered_map
    bool operator==(const openclarrayinfo b) const;

    ~openclarrayinfo();
    
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
    std::shared_ptr<std::function<void(snde_index)>> realloc_callback;
    std::weak_ptr<allocator> alloc; /* weak pointer to the allocator because we don't want to inhibit freeing of this (all we need it for is our reallocation callback) */
    
    _openclbuffer(const _openclbuffer &)=delete; /* copy constructor disabled */
    
    _openclbuffer(cl_context context,std::shared_ptr<snde::allocator> alloc,size_t elemsize,void **arrayptr, std::mutex *arraymanageradminmutex);

    ~_openclbuffer();
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
    //std::mutex admin;
    /* lock our data structures, including buffer_map and descendents. We are allowed to call allocators/lock managers while holding 
			 this mutex, but they are not allowed to call us and only lock their own data structures, 
			 so a locking order is ensured and no deadlock is possible */
    
    //std::unordered_map<openclarrayinfo,std::shared_ptr<_openclbuffer>> buffer_map;

    std::weak_ptr<arraymanager> manager; /* Used for looking up allocators and accessing lock manager (weak to avoid creating a circular reference) */ 
    
    
    openclcachemanager(std::shared_ptr<arraymanager> manager);
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

    std::tuple<rwlock_token_set,rwlock_token_set,cl_mem,snde_index,std::vector<cl_event>> GetOpenCLBuffer(rwlock_token_set alllocks,cl_context context, cl_command_queue queue, void **allocatedptr, void **arrayptr,std::shared_ptr<std::vector<rangetracker<markedregion>>> arrayreadregions, std::shared_ptr<std::vector<rangetracker<markedregion>>> arraywriteregions,bool write_only=false);    
    std::vector<cl_event> ReleaseOpenCLBuffer(rwlock_token_set readlocks,rwlock_token_set writelocks,cl_context context, cl_command_queue queue, cl_mem mem, void **allocatedptr, void **arrayptr, std::shared_ptr<std::vector<rangetracker<markedregion>>> arraywriteregions, cl_event input_data_not_needed,std::vector<cl_event> output_data_complete);

    virtual ~openclcachemanager();
  };


  static inline std::shared_ptr<openclcachemanager> get_opencl_cache_manager(std::shared_ptr<arraymanager> manager);
  
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
		      std::shared_ptr<std::vector<rangetracker<markedregion>>> arraywriteregions);
    OpenCLBuffer_info(const OpenCLBuffer_info &orig);
    
    OpenCLBuffer_info& operator=(const OpenCLBuffer_info &)=delete; /* copy assignment disabled (for now) */
    ~OpenCLBuffer_info();
    
  };
  
  class OpenCLBuffers {
    // Class for managing array of opencl buffers returned by the
    // opencl array manager
  public:
    cl_context context;  /* counted by clRetainContext() */
    rwlock_token_set locks;
    std::shared_ptr<std::vector<rangetracker<markedregion>>> arrayreadregions;
    std::shared_ptr<std::vector<rangetracker<markedregion>>> arraywriteregions;

    //std::unordered_map<void **,OpenCLBuffer_info> buffers; /* indexed by arrayidx */

    std::vector<cl_event> fill_events; /* each counted by clRetainEvent() */
     
    OpenCLBuffers(cl_context context,rwlock_token_set locks,std::shared_ptr<std::vector<rangetracker<markedregion>>> arrayreadregions,std::shared_ptr<std::vector<rangetracker<markedregion>>> arraywriteregions);

    /* no copying */
    OpenCLBuffers(const OpenCLBuffers &) = delete;
    OpenCLBuffers & operator=(const OpenCLBuffers &) = delete;
  
    ~OpenCLBuffers();
    
    //cl_mem Mem_untracked(void **arrayptr); // disallow unprotected pointer into Python
    cl_mem Mem(void **arrayptr);

    //cl_event *FillEvents_untracked(void); (access fill_events attribute instead)

    cl_uint NumFillEvents(void);

    
    cl_int SetBufferAsKernelArg(cl_kernel kernel, cl_uint arg_index, void **arrayptr);
  
    void AddBuffer(std::shared_ptr<arraymanager> manager,cl_command_queue queue, void **arrayptr,bool write_only=false);

    cl_int AddBufferAsKernelArg(std::shared_ptr<arraymanager> manager,cl_command_queue queue,cl_kernel kernel,cl_uint arg_index,void **arrayptr);
    
    void RemBuffer(void **arrayptr,cl_event input_data_not_needed,std::vector<cl_event> output_data_complete,bool wait);

    void RemBuffers(cl_event input_data_not_needed,std::vector<cl_event> output_data_complete,bool wait);
    
    void RemBuffers(cl_event input_data_not_needed,cl_event output_data_complete,bool wait);
    void RemBuffers(cl_event input_data_not_needed,bool wait);

  };
  
}

