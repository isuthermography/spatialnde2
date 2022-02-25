//%shared_ptr(snde::memallocator);
//%shared_ptr(snde::cmemallocator);



%{
  
#include "opencl_utils.hpp"
%}

typedef int32_t    cl_int;
typedef uint32_t        cl_uint;

typedef struct _cl_command_queue *  cl_command_queue;
typedef struct _cl_kernel *  cl_kernel;
typedef struct _cl_event *          cl_event;
typedef struct _cl_context *          cl_context;
typedef struct _cl_device_id *          cl_device_id;
typedef struct _cl_mem *          cl_mem;

%typemap(typecheck,precedence=SWIG_TYPECHECK_POINTER) cl::Buffer {
  PyObject *pyopencl=NULL;
  PyObject *clMemoryObject=NULL;
  pyopencl = PyImport_ImportModule("pyopencl");
  if (!pyopencl) SWIG_fail; /* raise exception up */
  $1 = PyObject_HasAttrString($input,"int_ptr");
  if ($1) {
    clMemoryObject=PyObject_GetAttrString(pyopencl,"MemoryObject");
    $1 = PyObject_IsInstance($input,clMemoryObject);
  }
  Py_XDECREF(clMemoryObject);
  Py_XDECREF(pyopencl);
}

%typemap(in) cl::Buffer (PyObject *clMem_int_ptr=NULL) {
  
  clMem_int_ptr=PyObject_GetAttrString($input,"int_ptr");
  
#if PY_VERSION_HEX < 0x03000000
  $1=cl::Buffer((cl_mem)((uintptr_t)PyInt_AsUnsignedLongLongMask(clMem_int_ptr)),true);
#else
  $1=cl::Buffer((cl_mem)((uintptr_t)PyLong_AsUnsignedLongLongMask(clMem_int_ptr)),true);
#endif

  Py_XDECREF(clMem_int_ptr);
}

%typemap(typecheck,precedence=SWIG_TYPECHECK_POINTER) cl::Kernel {
  PyObject *pyopencl=NULL;
  PyObject *clKernel=NULL;
  pyopencl = PyImport_ImportModule("pyopencl");
  if (!pyopencl) SWIG_fail; /* raise exception up */
  $1 = PyObject_HasAttrString($input,"int_ptr");
  if ($1) {
    clKernel=PyObject_GetAttrString(pyopencl,"Kernel");
    $1 = PyObject_IsInstance($input,clKernel);
  }
  Py_XDECREF(clKernel);
  Py_XDECREF(pyopencl);
}

%typemap(in) cl::Kernel (PyObject *clKernel_int_ptr=NULL) {
  
  clKernel_int_ptr=PyObject_GetAttrString($input,"int_ptr");
  
#if PY_VERSION_HEX < 0x03000000
  $1=cl::Kernel((cl_kernel)((uintptr_t)PyInt_AsUnsignedLongLongMask(clKernel_int_ptr)),true);
#else
  $1=cl::Kernel((cl_kernel)((uintptr_t)PyLong_AsUnsignedLongLongMask(clKernel_int_ptr)),true);
#endif
  
  Py_XDECREF(clKernel_int_ptr);
}

%typemap(typecheck,precedence=SWIG_TYPECHECK_POINTER) cl::CommandQueue {
  PyObject *pyopencl=NULL;
  PyObject *clCommandQueue=NULL;
  pyopencl = PyImport_ImportModule("pyopencl");
  if (!pyopencl) SWIG_fail; /* raise exception up */
  $1 = PyObject_HasAttrString($input,"int_ptr");
  if ($1) {
    clCommandQueue=PyObject_GetAttrString(pyopencl,"CommandQueue");
    $1 = PyObject_IsInstance($input,clCommandQueue);
  }
  Py_XDECREF(clCommandQueue);
  Py_XDECREF(pyopencl);
}

%typemap(in) cl::CommandQueue (PyObject *clCommandQueue_int_ptr=NULL) {
  
  clCommandQueue_int_ptr=PyObject_GetAttrString($input,"int_ptr");
  
#if PY_VERSION_HEX < 0x03000000
  $1=cl::CommandQueue((cl_command_queue)((uintptr_t)PyInt_AsUnsignedLongLongMask(clCommandQueue_int_ptr)),true);
#else
  $1=cl::CommandQueue((cl_command_queue)((uintptr_t)PyLong_AsUnsignedLongLongMask(clCommandQueue_int_ptr)),true);
#endif
  

  Py_XDECREF(clCommandQueue_int_ptr);
}


%typemap(in) cl::Event (PyObject *clEvent_int_ptr=NULL) {
  
  clEvent_int_ptr=PyObject_GetAttrString($input,"int_ptr");
  
#if PY_VERSION_HEX < 0x03000000
  $1=cl::Event((cl_event)((uintptr_t)PyInt_AsUnsignedLongLongMask(clEvent_int_ptr)),true);
#else
  $1=cl::Event((cl_event)((uintptr_t)PyLong_AsUnsignedLongLongMask(clEvent_int_ptr)),true);

#endif

  Py_XDECREF(clEvent_int_ptr);
}

// need typecheck typemap for cl_event to support overloaded
// RemBuffers() method
%typemap(typecheck,precedence=SWIG_TYPECHECK_POINTER) cl::Event  {
  PyObject *pyopencl=NULL;
  PyObject *clEvent=NULL;
  pyopencl = PyImport_ImportModule("pyopencl");
  if (!pyopencl) SWIG_fail; /* raise exception up */
  $1 = PyObject_HasAttrString($input,"int_ptr");
  if ($1) {
    clEvent=PyObject_GetAttrString(pyopencl,"Event");
    $1 = PyObject_IsInstance($input,clEvent);
  }
  Py_XDECREF(clEvent);
  Py_XDECREF(pyopencl);

}

%typemap(typecheck,precedence=SWIG_TYPECHECK_POINTER) cl::Context {
  $1=false;
  { // NOTE: Not a rigorous check
    PyObject *clContext_int_ptr=NULL;
    clContext_int_ptr=PyObject_GetAttrString($input,"int_ptr");
    if (clContext_int_ptr) {
      Py_XDECREF(clContext_int_ptr);
      $1=true;
    }
  }
  
 }

%typemap(in) cl::Context (PyObject *clContext_int_ptr=NULL) {
  
  clContext_int_ptr=PyObject_GetAttrString($input,"int_ptr");
  
#if PY_VERSION_HEX < 0x03000000
  $1=cl::Context((cl_context)((uintptr_t)PyInt_AsUnsignedLongLongMask(clContext_int_ptr)),true);
#else
  $1=cl::Context((cl_context)((uintptr_t)PyLong_AsUnsignedLongLongMask(clContext_int_ptr)),true);
#endif

  Py_XDECREF(clContext_int_ptr);
}


%typemap(typecheck,precedence=SWIG_TYPECHECK_POINTER) cl::Device {
  $1=false;
  { // NOTE: Not a rigorous check
    PyObject *clDevice_int_ptr=NULL;
    clDevice_int_ptr=PyObject_GetAttrString($input,"int_ptr");
    if (clDevice_int_ptr) {
      Py_XDECREF(clDevice_int_ptr);
      $1=true;
    }
  }
  
 }

%typemap(in) cl::Device (PyObject *clDevice_int_ptr=NULL) {
  
  clDevice_int_ptr=PyObject_GetAttrString($input,"int_ptr");
  
#if PY_VERSION_HEX < 0x03000000
  $1=cl::Device((cl_device_id)((uintptr_t)PyInt_AsUnsignedLongLongMask(clDevice_int_ptr)),true);
#else
  $1=cl::Device((cl_device_id)((uintptr_t)PyLong_AsUnsignedLongLongMask(clDevice_int_ptr)),true);
#endif  

  Py_XDECREF(clDevice_int_ptr);
}

%typemap(out) cl::Buffer (PyObject *pyopencl=NULL,PyObject *clMem=NULL,PyObject *clMem_from_int_ptr=NULL) {
  pyopencl = PyImport_ImportModule("pyopencl");
  if (!pyopencl) SWIG_fail; /* raise exception up */
  clMem=PyObject_GetAttrString(pyopencl,"MemoryObject");
  clMem_from_int_ptr=PyObject_GetAttrString(clMem,"from_int_ptr");
  
  
  $result=PyObject_CallFunction(clMem_from_int_ptr,(char *)"KO",(unsigned long long)((uintptr_t)($1.get())),Py_True);

  Py_XDECREF(clMem_from_int_ptr);
  Py_XDECREF(clMem);
  Py_XDECREF(pyopencl);
}

%typemap(out) cl::Event (PyObject *pyopencl=NULL,PyObject *clEvent=NULL,PyObject *clEvent_from_int_ptr=NULL) {
  pyopencl = PyImport_ImportModule("pyopencl");
  if (!pyopencl) SWIG_fail; /* raise exception up */
  clEvent=PyObject_GetAttrString(pyopencl,"Event");
  clEvent_from_int_ptr=PyObject_GetAttrString(clEvent,"from_int_ptr");
  
  
  $result=PyObject_CallFunction(clEvent_from_int_ptr,(char *)"KO",(unsigned long long)((uintptr_t)($1.get())),Py_True);

  Py_XDECREF(clEvent_from_int_ptr);
  Py_XDECREF(clEvent);
  Py_XDECREF(pyopencl);
}


%typemap(out) cl::CommandQueue (PyObject *pyopencl=NULL,PyObject *clQueue=NULL,PyObject *clQueue_from_int_ptr=NULL) {
  // eats the returned reference
  pyopencl = PyImport_ImportModule("pyopencl");
  if (!pyopencl) SWIG_fail; /* raise exception up */
  clQueue=PyObject_GetAttrString(pyopencl,"CommandQueue");
  clQueue_from_int_ptr=PyObject_GetAttrString(clQueue,"from_int_ptr");
  
  
  $result=PyObject_CallFunction(clQueue_from_int_ptr,(char *)"KO",(unsigned long long)((uintptr_t)($1.get())),Py_True);

  Py_XDECREF(clQueue_from_int_ptr);
  Py_XDECREF(clQueue);
  Py_XDECREF(pyopencl);
}



// %typemap(out) std::vector<cl_event> is for functions returning such a vector.
%typemap(out) std::vector<cl::Event> (PyObject *pyopencl=NULL,PyObject *clEvent=NULL,PyObject *clEvent_from_int_ptr=NULL,size_t cnt) {
  pyopencl = PyImport_ImportModule("pyopencl");
  if (!pyopencl) SWIG_fail; /* raise exception up */
  clEvent=PyObject_GetAttrString(pyopencl,"Event");
  clEvent_from_int_ptr=PyObject_GetAttrString(clEvent,"from_int_ptr");
  
  $result = PyTuple_New($1.size());

  for (cnt=0;cnt < $1.size();cnt++) {
    PyTuple_SetItem($result,cnt,PyObject_CallFunction(clEvent_from_int_ptr,(char *)"KO",(unsigned long long)((uintptr_t)($1.operator[](cnt).get())),Py_True));
    
  }

  Py_XDECREF(clEvent_from_int_ptr);
  Py_XDECREF(clEvent);
  Py_XDECREF(pyopencl);
}

%typemap(out) std::pair<std::vector<cl::Event>,std::vector<cl::Event>> (PyObject *pyopencl=NULL,PyObject *clEvent=NULL,PyObject *clEvent_from_int_ptr=NULL,size_t cnt,PyObject *FirstVector=NULL,PyObject *SecondVector=NULL) {
  pyopencl = PyImport_ImportModule("pyopencl");
  if (!pyopencl) SWIG_fail; /* raise exception up */
  clEvent=PyObject_GetAttrString(pyopencl,"Event");
  clEvent_from_int_ptr=PyObject_GetAttrString(clEvent,"from_int_ptr");
  
  $result = PyTuple_New(2);
  FirstVector=PyTuple_New($1.first.size());
  PyTuple_SetItem($result,0,FirstVector);

  SecondVector=PyTuple_New($1.second.size());
  PyTuple_SetItem($result,1,SecondVector);

  for (cnt=0;cnt < $1.first.size();cnt++) {
    PyTuple_SetItem(FirstVector,cnt,PyObject_CallFunction(clEvent_from_int_ptr,(char *)"KO",(unsigned long long)((uintptr_t)($1.first.operator[](cnt).get())),Py_True));
    
  }

  for (cnt=0;cnt < $1.second.size();cnt++) {
    PyTuple_SetItem(SecondVector,cnt,PyObject_CallFunction(clEvent_from_int_ptr,(char *)"KO",(unsigned long long)((uintptr_t)($1.second.operator[](cnt).get())),Py_True));    
  }

  
  Py_XDECREF(clEvent_from_int_ptr);
  Py_XDECREF(clEvent);
  Py_XDECREF(pyopencl);
}



// %typemap(out) std::vector<cl_event> * is for member access such a vector.... we increment the reference count of such events
%typemap(out) std::vector<cl::Event> * (PyObject *pyopencl=NULL,PyObject *clEvent=NULL,PyObject *clEvent_from_int_ptr=NULL,size_t cnt) {
  pyopencl = PyImport_ImportModule("pyopencl");
  if (!pyopencl) SWIG_fail; /* raise exception up */
  clEvent=PyObject_GetAttrString(pyopencl,"Event");
  clEvent_from_int_ptr=PyObject_GetAttrString(clEvent,"from_int_ptr");
  
  $result = PyTuple_New($1->size());

  for (cnt=0;cnt < $1->size();cnt++) {
    PyTuple_SetItem($result,cnt,PyObject_CallFunction(clEvent_from_int_ptr,(char *)"KO",(unsigned long long)((uintptr_t)($1->operator[](cnt).get())),Py_True));
    
  }

  Py_XDECREF(clEvent_from_int_ptr);
  Py_XDECREF(clEvent);
  Py_XDECREF(pyopencl);
}


%typemap(out) cl::Context (PyObject *pyopencl=NULL,PyObject *clContext=NULL,PyObject *clContext_from_int_ptr=NULL) {
  pyopencl = PyImport_ImportModule("pyopencl");
  if (!pyopencl) SWIG_fail; /* raise exception up */
  clContext=PyObject_GetAttrString(pyopencl,"Context");
  clContext_from_int_ptr=PyObject_GetAttrString(clContext,"from_int_ptr");
  
  
  $result=PyObject_CallFunction(clContext_from_int_ptr,(char *)"KO",(unsigned long long)((uintptr_t)($1.get())),Py_True);

  Py_XDECREF(clContext_from_int_ptr);
  Py_XDECREF(clContext);
  Py_XDECREF(pyopencl);
}


%typemap(out) std::tuple< cl::Context,cl::Device,std::string > (PyObject *pyopencl=NULL,PyObject *clContext=NULL,PyObject *clContext_from_int_ptr=NULL,PyObject *clDevice=NULL,PyObject *clDevice_from_int_ptr=NULL) {
  pyopencl = PyImport_ImportModule("pyopencl");
  if (!pyopencl) SWIG_fail; /* raise exception up */
  clContext=PyObject_GetAttrString(pyopencl,"Context");
  clContext_from_int_ptr=PyObject_GetAttrString(clContext,"from_int_ptr");
  clDevice=PyObject_GetAttrString(pyopencl,"Device");
  clDevice_from_int_ptr=PyObject_GetAttrString(clDevice,"from_int_ptr");
  
  
  $result = PyTuple_New(3);
  // Substituted code for converting cl_context here came
  PyTuple_SetItem($result,0,PyObject_CallFunction(clContext_from_int_ptr,(char *)"KO",(unsigned long long)((uintptr_t)(std::get<0>(*&$1).get())),Py_True));
  PyTuple_SetItem($result,1,PyObject_CallFunction(clDevice_from_int_ptr,(char *)"KO",(unsigned long long)((uintptr_t)(std::get<1>(*&$1).get())),Py_True));
  
  PyTuple_SetItem($result,2,SWIG_From_std_string(static_cast< std::string >(std::get<2>(*&$1))));

  Py_XDECREF(clDevice_from_int_ptr);
  Py_XDECREF(clDevice);
  Py_XDECREF(clContext_from_int_ptr);
  Py_XDECREF(clContext);
  Py_XDECREF(pyopencl);
}

%template(CommandQueueVector) std::vector<cl::CommandQueue>;

namespace snde {

  std::tuple<cl::Context,std::vector<cl::Device>,std::string> get_opencl_context(std::string query,bool need_doubleprec,void (*pfn_notify)(const char *errinfo,const void *private_info, size_t cb, void *user_data),void *user_data);

  void add_opencl_alignment_requirement(std::shared_ptr<recdatabase> recdb,cl::Device device);
  void add_opencl_alignment_requirements(std::shared_ptr<recdatabase> recdb,const std::vector<cl::Device> &devices);

};
// Old (nonpyopencl) version below... 
/* 

typedef int8_t        cl_char;
typedef uint8_t         cl_uchar;
typedef int16_t       cl_short;
typedef uint16_t       cl_ushort;
typedef int32_t    cl_int;
typedef uint32_t        cl_uint;
typedef int64_t      cl_long;
typedef uint64_t     cl_ulong;

//typedef unsigned __int16        cl_half;
typedef float                   cl_float;
typedef double                  cl_double;


#define CL_SUCCESS                                  0
#define CL_DEVICE_NOT_FOUND                         -1
#define CL_DEVICE_NOT_AVAILABLE                     -2
#define CL_COMPILER_NOT_AVAILABLE                   -3
#define CL_MEM_OBJECT_ALLOCATION_FAILURE            -4
#define CL_OUT_OF_RESOURCES                         -5
#define CL_OUT_OF_HOST_MEMORY                       -6
#define CL_PROFILING_INFO_NOT_AVAILABLE             -7
#define CL_MEM_COPY_OVERLAP                         -8
#define CL_IMAGE_FORMAT_MISMATCH                    -9
#define CL_IMAGE_FORMAT_NOT_SUPPORTED               -10
#define CL_BUILD_PROGRAM_FAILURE                    -11
#define CL_MAP_FAILURE                              -12
#define CL_MISALIGNED_SUB_BUFFER_OFFSET             -13
#define CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST -14
#define CL_COMPILE_PROGRAM_FAILURE                  -15
#define CL_LINKER_NOT_AVAILABLE                     -16
#define CL_LINK_PROGRAM_FAILURE                     -17
#define CL_DEVICE_PARTITION_FAILED                  -18
#define CL_KERNEL_ARG_INFO_NOT_AVAILABLE            -19

#define CL_INVALID_VALUE                            -30
#define CL_INVALID_DEVICE_TYPE                      -31
#define CL_INVALID_PLATFORM                         -32
#define CL_INVALID_DEVICE                           -33
#define CL_INVALID_CONTEXT                          -34
#define CL_INVALID_QUEUE_PROPERTIES                 -35
#define CL_INVALID_COMMAND_QUEUE                    -36
#define CL_INVALID_HOST_PTR                         -37
#define CL_INVALID_MEM_OBJECT                       -38
#define CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          -39
#define CL_INVALID_IMAGE_SIZE                       -40
#define CL_INVALID_SAMPLER                          -41
#define CL_INVALID_BINARY                           -42
#define CL_INVALID_BUILD_OPTIONS                    -43
#define CL_INVALID_PROGRAM                          -44
#define CL_INVALID_PROGRAM_EXECUTABLE               -45
#define CL_INVALID_KERNEL_NAME                      -46
#define CL_INVALID_KERNEL_DEFINITION                -47
#define CL_INVALID_KERNEL                           -48
#define CL_INVALID_ARG_INDEX                        -49
#define CL_INVALID_ARG_VALUE                        -50
#define CL_INVALID_ARG_SIZE                         -51
#define CL_INVALID_KERNEL_ARGS                      -52
#define CL_INVALID_WORK_DIMENSION                   -53
#define CL_INVALID_WORK_GROUP_SIZE                  -54
#define CL_INVALID_WORK_ITEM_SIZE                   -55
#define CL_INVALID_GLOBAL_OFFSET                    -56
#define CL_INVALID_EVENT_WAIT_LIST                  -57
#define CL_INVALID_EVENT                            -58
#define CL_INVALID_OPERATION                        -59
#define CL_INVALID_GL_OBJECT                        -60
#define CL_INVALID_BUFFER_SIZE                      -61
#define CL_INVALID_MIP_LEVEL                        -62
#define CL_INVALID_GLOBAL_WORK_SIZE                 -63
#define CL_INVALID_PROPERTY                         -64
#define CL_INVALID_IMAGE_DESCRIPTOR                 -65
#define CL_INVALID_COMPILER_OPTIONS                 -66
#define CL_INVALID_LINKER_OPTIONS                   -67
#define CL_INVALID_DEVICE_PARTITION_COUNT           -68
#define CL_INVALID_PIPE_SIZE                        -69
#define CL_INVALID_DEVICE_QUEUE                     -70


typedef int32_t cl_int;
typedef uint64_t cl_ulong;
typedef cl_ulong cl_command_queue_properties;




struct _cl_command_queue;

//typedef struct _cl_command_queue *  cl_command_queue;

// destructors for opencl types
class cl_command_queue {
  struct _cl_command_queue *ptr;
};

%extend cl_command_queue {
  ~cl_command_queue() {
    if (*self) clReleaseCommandQueue(*self);
  }
};


class cl_context {
  struct _cl_context *ptr;
};

%extend cl_context {
  ~cl_context() {
    if (*self) clReleaseContext(*self);
  }
};

class cl_device_id {
  struct _cl_device_id *ptr;
};

%extend cl_device_id {
  ~cl_device_id() {
    if (*self) clReleaseDevice(*self);
  }
};


class cl_program {
  struct _cl_program *ptr;
};

%extend cl_program {
  ~cl_program() {
    if (*self) clReleaseProgram(*self);
  }
};

class cl_kernel {
  struct _cl_kernel *ptr;
};

%extend cl_kernel {
  ~cl_kernel() {
    if (*self) clReleaseKernel(*self);
  }
};
typedef struct _cl_event *          cl_event;
//class cl_event {
//  struct _cl_event *ptr;
//};

//%extend cl_event {
//  ~cl_event() {
//    if (*self) clReleaseEvent(*self);
//  }
//};


cl_command_queue clCreateCommandQueue(cl_context,
				      cl_device_id,
				      cl_command_queue_properties,
				      cl_int *OUTPUT);
#define CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE      (1 << 0)
#define CL_QUEUE_PROFILING_ENABLE                   (1 << 1)
#define CL_QUEUE_ON_DEVICE                          (1 << 2)
#define CL_QUEUE_ON_DEVICE_DEFAULT                  (1 << 3)


cl_kernel clCreateKernel(cl_program,      // program,
			 const char *,    // kernel_name 
			 cl_int *OUTPUT);

%typemap(in,numinputs=0) cl_event *OUTPUT (cl_event outevent) {
  outevent=NULL;
  $1=&outevent;
} 
%typemap(argout) cl_event *OUTPUT {
  %append_output(SWIG_NewPointerObj(%as_voidptr(*$1), $*1_descriptor, SWIG_POINTER_OWN));
} 

cl_int clEnqueueNDRangeKernelArrays(cl_command_queue queue,
				    cl_kernel kern,
				    size_t *IN_ARRAY1,size_t DIM1,
				    size_t *IN_ARRAY1,size_t DIM1,
				    size_t *IN_ARRAY1,size_t DIM1,
				    cl_event *waitevents,size_t numwaitevents,
				    cl_event *OUTPUT);
%{
  cl_int clEnqueueNDRangeKernelArrays(cl_command_queue queue,
				      cl_kernel kern,
				      const size_t *IN_ARRAY1,size_t DIM1,
				      const size_t *IN_ARRAY2,size_t DIM2,
				      const size_t *IN_ARRAY3,size_t DIM3,
				      const cl_event *IN_ARRAY4,size_t DIM4,
				      cl_event *OUTPUT)

  {
    assert(IN_ARRAY2);
    if (IN_ARRAY1 && DIM1 != 0) assert(DIM1==DIM2);
    if (IN_ARRAY3 && DIM3 != 0) assert(DIM3==DIM2);
    if (!DIM1) IN_ARRAY1=NULL;
    if (!DIM3) IN_ARRAY3=NULL;
    if (!DIM4) IN_ARRAY4=NULL;
    
    return clEnqueueNDRangeKernel(queue,
				  kern,
				  DIM2,
				  IN_ARRAY1,
				  IN_ARRAY2,
				  IN_ARRAY3,
				  DIM4,
				  IN_ARRAY4,
				  OUTPUT);
  }
%}
cl_int clEnqueueNDRangeKernel(cl_command_queue, // command_queue ,
                       cl_kernel, //kernel ,
                       cl_uint,          // work_dim ,
                       const size_t *,   // global_work_offset ,
                       const size_t *,   // global_work_size ,
                       const size_t *,   // local_work_size ,
                       cl_uint,          // num_events_in_wait_list ,
                       const cl_event *, // event_wait_list ,
		       cl_event *); //        event 



%template(opencl_event_vector) std::vector<cl_event>;


namespace snde {

%typemap(out) std::tuple< cl_context,cl_device_id,std::string > {
    $result = PyTuple_New(3);
    // Substituted code for converting cl_context here came
    // from a typemap substitution "$typemap(out,cl_context)" 
    PyTuple_SetItem($result,0,SWIG_NewPointerObj((new cl_context(static_cast< const cl_context& >(std::get<0>(*&$1)))), $descriptor(cl_context *), SWIG_POINTER_OWN |  0 ));
    PyTuple_SetItem($result,1,SWIG_NewPointerObj((new cl_device_id(static_cast< const cl_device_id & >(std::get<1>(*&$1)))), $descriptor(cl_device_id *), SWIG_POINTER_OWN |  0 ));
PyTuple_SetItem($result,2,SWIG_From_std_string(static_cast< std::string >(std::get<2>(*&$1))));

  }

  std::tuple<cl_context,cl_device_id,std::string> get_opencl_context(std::string query,bool need_doubleprec,void (*pfn_notify)(const char *errinfo,const void *private_info, size_t cb, void *user_data),void *user_data);


%typemap(out) std::tuple< cl_program,std::string > {
    $result = PyTuple_New(2);
    // Substituted code for converting cl_context here came
    // from a typemap substitution "$typemap(out,cl_context)" 
    PyTuple_SetItem($result,0,SWIG_NewPointerObj((new cl_program(static_cast< const cl_program& >(std::get<0>(*&$1)))), $descriptor(cl_program *), SWIG_POINTER_OWN |  0 ));
    PyTuple_SetItem($result,1,SWIG_From_std_string(static_cast< std::string >(std::get<1>(*&$1))));

  }

  std::tuple<cl_program, std::string> get_opencl_program(cl_context context, cl_device_id device, std::vector<std::string> program_source);

  
}
*/
