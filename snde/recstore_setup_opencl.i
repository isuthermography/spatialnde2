%{

  #include "snde/recstore_setup_opencl.hpp"
  
%}

//%shared_ptr(std::pair<cl::Context,std::vector<cl::Device>>);
//%template(shared_ocl_context_devices) std::pair<cl::Context,std::vector<cl::Device>>;

%typemap(out) std::pair<cl::Context,std::vector<cl::Device>> (PyObject *pyopencl=NULL,PyObject *clContext=NULL,PyObject *clContext_from_int_ptr=NULL,PyObject *clDevice=NULL,PyObject *clDevice_from_int_ptr=NULL,size_t cnt,PyObject *Context=NULL,PyObject *Vector=NULL) {
  pyopencl = PyImport_ImportModule("pyopencl");
  if (!pyopencl) SWIG_fail; /* raise exception up */
  clContext=PyObject_GetAttrString(pyopencl,"Context");
  clContext_from_int_ptr=PyObject_GetAttrString(clContext,"from_int_ptr");

  clDevice=PyObject_GetAttrString(pyopencl,"Device");
  clDevice_from_int_ptr=PyObject_GetAttrString(clDevice,"from_int_ptr");

  $result = PyTuple_New(2);
  Context=PyObject_CallFunction(clContext_from_int_ptr,(char *)"KO",(unsigned long long)((uintptr_t)($1.first.get())),Py_True);
  PyTuple_SetItem($result,0,Context);

  Vector=PyTuple_New($1.second.size());
  PyTuple_SetItem($result,1,Vector);

  for (cnt=0;cnt < $1.second.size();cnt++) {
    PyTuple_SetItem(Vector,cnt,PyObject_CallFunction(clDevice_from_int_ptr,(char *)"KO",(unsigned long long)((uintptr_t)($1.second.operator[](cnt).get())),Py_True));    
  }
  
  
  Py_XDECREF(clContext_from_int_ptr);
  Py_XDECREF(clContext);
  Py_XDECREF(clDevice_from_int_ptr);
  Py_XDECREF(clDevice);
  Py_XDECREF(pyopencl);
}



namespace snde {
  class recdatabase;

  std::pair<cl::Context,std::vector<cl::Device>> setup_opencl(std::shared_ptr<recdatabase> recdb,bool primary_doubleprec, size_t max_parallel, char *primary_platform_prefix_or_null);
  

};
