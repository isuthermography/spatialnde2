%shared_ptr(snde::assigned_compute_resource_opencl)
snde_rawaccessible(snde::assigned_compute_resource_opencl);

// Very incomplete so far

%{
  #include "recmath_compute_resource_opencl.hpp"  
%}



%typemap(in) const std::vector<cl::Device> & ( PyObject *clDevice, PyObject *clDevice_int_ptr, Py_ssize_t elemnum) {
  $1 = nullptr;
  
  if (!PySequence_Check($input)) {
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			"$argnum"" is not a sequence.");
  }
  Py_ssize_t num_elem = PySequence_Size($input);
  $1 = new std::vector<cl::Device>();
  $1->reserve(num_elem);

  for (elemnum=0;elemnum < num_elem; elemnum++) {
    clDevice = PySequence_GetItem($input,elemnum);
    if (!clDevice) {
      SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			  "$argnum"" Cannot get sequence item.");

    }
    clDevice_int_ptr = PyObject_GetAttrString(clDevice,"int_ptr");
    if (!clDevice_int_ptr) {
      SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			  "$argnum"" element does not have an int_ptr attribute.");

    }
    $1->push_back(cl::Device((cl_device_id)((uintptr_t)PyLong_AsUnsignedLongLongMask(clDevice_int_ptr)),true));
    
    Py_DECREF(clDevice_int_ptr);
    Py_DECREF(clDevice);
  }
  
}

%typemap(freearg) const std::vector<cl::Device> & // free argument from above input typemap
{
  if ($1) {
    delete $1;
  }
}



%typemap(in) const std::vector<cl::CommandQueue> & ( PyObject *clQueue, PyObject *clQueue_int_ptr, Py_ssize_t elemnum, Py_ssize_t num_elem) {
  $1 = nullptr;
  
  if (!PySequence_Check($input)) {
    SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			"$argnum"" is not a sequence.");
  }
  num_elem = PySequence_Size($input);
  $1 = new std::vector<cl::CommandQueue>();
  $1->reserve(num_elem);
  
  for (elemnum=0;elemnum < num_elem; elemnum++) {
    clQueue = PySequence_GetItem($input,elemnum);
    if (!clQueue) {
      SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			  "$argnum"" Cannot get sequence item.");
      
    }
    clQueue_int_ptr = PyObject_GetAttrString(clQueue,"int_ptr");
    if (!clQueue_int_ptr) {
      SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			  "$argnum"" element does not have an int_ptr attribute.");
      
    }
    $1->push_back(cl::CommandQueue((cl_command_queue)((uintptr_t)PyLong_AsUnsignedLongLongMask(clQueue_int_ptr)),true));
    
    Py_DECREF(clQueue_int_ptr);
    Py_DECREF(clQueue);
  }
  
}

%typemap(freearg) const std::vector<cl::CommandQueue> & // free argument from above input typemap
{
  if ($1) {
    delete $1;
  }
}




namespace snde {

  class assigned_compute_resource_opencl : public assigned_compute_resource {
  public:
    assigned_compute_resource_opencl(std::shared_ptr<available_compute_resource> resource,const std::vector<size_t> &assigned_cpu_core_indices,const std::vector<size_t> &assigned_opencl_job_indices,cl::Context context,const std::vector<cl::Device> &devices,const std::vector<bool> &device_supports_double,const std::vector<cl::CommandQueue> &queues,std::shared_ptr<openclcachemanager> oclcache);
    //size_t number_of_cpu_cores;
    std::vector<size_t> assigned_opencl_job_indices;
    
    cl::Context context;
    std::vector<cl::Device> devices; // devices corresponding to above-assigned opencl_job_indices
    std::vector<bool> device_supports_double; // same length as opencl_devices
    std::vector<cl::CommandQueue> queues; // devices corresponding to above-assigned opencl_job_indices
    std::shared_ptr<openclcachemanager> oclcache;
    std::shared_ptr<assigned_compute_resource_cpu> cpu_assignment; // contains assigned_cpu_core_indices

    
    virtual ~assigned_compute_resource_opencl()=default;  // virtual destructor required so we can be subclassed. Some subclasses use destructor to release resources

    virtual void release_assigned_resources(std::unique_lock<std::mutex> &acrd_admin_holder); // resources referenced below no longer meaningful once this is called. Must be called with acrd admin lock locked

  };
  


};
