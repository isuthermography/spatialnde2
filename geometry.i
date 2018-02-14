%shared_ptr(snde::geometry);
%shared_ptr(snde::mesheduv);
%shared_ptr(snde::component);
%shared_ptr(snde::assembly);
%shared_ptr(snde::nurbspart);
%shared_ptr(snde::meshedpart);

%{

#include "geometry_types.h"
#include "geometrydata.h"
#include "geometry.hpp"
%}

%pythonbegin %{
import ctypes
%}

%{
  void *geometry_return_pointer(void*ptr) { return ptr; }
    

%}

// access to .geom snde_geometrydata element returns ctypes-wrapped snde_geometrydata
%typemap(out) snde_geometrydata geom (PyObject *__module__=NULL,PyObject *module=NULL,PyObject *ctypes=NULL, PyObject *snde_geometrydata_class=NULL, PyObject *POINTER=NULL, PyObject *snde_geometrydata_class_p=NULL, PyObject *c_void_p=NULL, PyObject *CFUNCTYPE=NULL, PyObject *CMPFUNC=NULL, PyObject *CMPFUNC_INSTANCE=NULL, PyObject *reswrapper=NULL,snde_geometrydata *geom_ptr=NULL) %{
  geom_ptr=&arg1->geom;
  
  __module__=PyObject_GetAttrString($self,"__module__");
  module = PyImport_Import(__module__);
  ctypes = PyImport_ImportModule("ctypes");

  snde_geometrydata_class = PyDict_GetItemString(PyModule_GetDict(module),"snde_geometrydata");

  POINTER=PyDict_GetItemString(PyModule_GetDict(ctypes),"POINTER");

  //ManagerObj=SWIG_NewPointerObj( SWIG_as_voidptr(new std::shared_ptr<snde::arraymanager>(arg1->manager)), $descriptor(std::shared_ptr<arraymanager> *), SWIG_POINTER_NEW|SWIG_POINTER_OWN);
  
  snde_geometrydata_class_p=PyObject_CallFunctionObjArgs(POINTER,snde_geometrydata_class,NULL);
  
  // define function  geometry_return_pointer()
  // taking $1 pointer as argument
  // and returning a Python ctypes snde_geometrydata

  
  c_void_p = PyDict_GetItemString(PyModule_GetDict(ctypes),"c_void_p");

  CFUNCTYPE=PyDict_GetItemString(PyModule_GetDict(ctypes),"CFUNCTYPE");
  // declare CMPFUNC as returning pointer to geometrydata given a void pointer
  CMPFUNC=PyObject_CallFunctionObjArgs(CFUNCTYPE,snde_geometrydata_class_p,c_void_p,NULL);

  // instantiate CMPFUNC from geometry_return_pointer
  CMPFUNC_INSTANCE=PyObject_CallFunction(CMPFUNC,(char *)"K",(unsigned long long)((uintptr_t)&geometry_return_pointer));

  // create a void pointer from arg
  reswrapper=PyObject_CallFunction(c_void_p,(char *)"K",(unsigned long long)((uintptr_t)geom_ptr));
  
  // call CMPFUNC_INSTANCE on (void *)$1 to get a ctypes pointer to snde_geometrydata
  
  $result = PyObject_CallFunctionObjArgs(CMPFUNC_INSTANCE,reswrapper,NULL);

  //// Assign .manager attribute (doesn't work because .contents is generated dynamically
  //contentsobj = PyObject_GetAttrString($result,"contents");
  //PyObject_SetAttrString(contentsobj,"manager",ManagerObj);
  
  //Py_XDECREF(contentsobj);
  Py_XDECREF(reswrapper);
  Py_XDECREF(CMPFUNC_INSTANCE);
  Py_XDECREF(CMPFUNC);
  Py_XDECREF(snde_geometrydata_class_p);
  //Py_XDECREF(ManagerObj);
  Py_XDECREF(ctypes);
  
  Py_XDECREF(module);
  Py_XDECREF(__module__);
%}

namespace snde {
  class geometry {
  public:
    struct snde_geometrydata geom;

    std::shared_ptr<arraymanager> manager;
    /* All arrays allocated by a particular allocator are locked together by manager->locker */
    
    
    geometry(double tol,std::shared_ptr<arraymanager> manager);
    
    ~geometry();
  };


  class mesheduv {
  public:

    std::shared_ptr<geometry> geom;
    std::string name;
    snde_index mesheduvnum;

    mesheduv(std::shared_ptr<geometry> geom, std::string name,snde_index mesheduvnum);

    ~mesheduv();
    
  };
  
#define SNDE_COMPONENT_GEOMWRITE_MESHEDPARTS (1u<<0)
#define SNDE_COMPONENT_GEOMWRITE_VERTICES (1u<<1)
#define SNDE_COMPONENT_GEOMWRITE_PRINCIPAL_CURVATURES (1u<<2)
#define SNDE_COMPONENT_GEOMWRITE_CURVATURE_TANGENT_AXES (1u<<3)
#define SNDE_COMPONENT_GEOMWRITE_TRIS (1u<<4)
#define SNDE_COMPONENT_GEOMWRITE_REFPOINTS (1u<<5)
#define SNDE_COMPONENT_GEOMWRITE_MAXRADIUS (1u<<6)
#define SNDE_COMPONENT_GEOMWRITE_NORMAL (1u<<7)
#define SNDE_COMPONENT_GEOMWRITE_INPLANEMAT (1u<<8)
#define SNDE_COMPONENT_GEOMWRITE_BOXES (1u<<9)
#define SNDE_COMPONENT_GEOMWRITE_BOXCOORDS (1u<<10)
#define SNDE_COMPONENT_GEOMWRITE_BOXPOLYS (1u<<11)

  class component { /* abstract base class for geometric components (assemblies, nurbspart, meshedpart) */
  public:
    typedef enum {
      subassembly=0,
      nurbs=1,
      meshed=2,
    } TYPE;

    TYPE type;

    virtual snde_orientation3 orientation()=0;

    virtual void obtain_lock(std::shared_ptr<lockingprocess> process, unsigned writemask=0)=0; /* writemask contains OR'd SNDE_COMPONENT_GEOMWRITE_xxx bits */

    virtual ~component()=0;
  };
  

  class assembly : public component {
    /* NOTE: Unlike other types of component, assemblies ARE copyable/assignable */
    /* (this is because they don't have a representation in the underlying
       geometry database) */
  public:    
    std::deque<std::shared_ptr<component>> pieces;
    snde_orientation3 _orientation; /* orientation of this part/assembly relative to its parent */

    assembly(snde_orientation3 orientation);

    virtual snde_orientation3 orientation(void);

    virtual void obtain_lock(std::shared_ptr<lockingprocess> process, unsigned writemask=0);
    
    virtual ~assembly();
  };

  


  /* NOTE: Could have additional abstraction layer to accommodate 
     multi-resolution approximations */
  class nurbspart : public component {
    nurbspart(const nurbspart &)=delete; /* copy constructor disabled */
    nurbspart& operator=(const nurbspart &)=delete; /* copy assignment disabled */
  public:
    snde_index nurbspartnum;
    std::shared_ptr<geometry> geom;

    nurbspart(std::shared_ptr<geometry> geom,snde_index nurbspartnum);
    
    virtual snde_orientation3 orientation(void);
    virtual void obtain_lock(std::shared_ptr<lockingprocess> process, unsigned writemask=0);
    virtual ~nurbspart();
    
  };

  class meshedpart : public component {
    meshedpart(const meshedpart &)=delete; /* copy constructor disabled */
    meshedpart& operator=(const meshedpart &)=delete; /* copy assignment disabled */
    
  public:
    std::shared_ptr<geometry> geom;
    snde_index meshedpartnum;
    std::map<std::string,std::shared_ptr<mesheduv>> parameterizations;

    
    meshedpart(std::shared_ptr<geometry> geom,snde_index meshedpartnum);

    void addparameterization(std::shared_ptr<mesheduv> parameterization);
    
    virtual snde_orientation3 orientation(void);

    virtual void obtain_lock(std::shared_ptr<lockingprocess> process, unsigned writemask=0);


    ~meshedpart();
  };

}

