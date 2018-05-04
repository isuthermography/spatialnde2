%shared_ptr(snde::geometry);
%shared_ptr(snde::mesheduv);
%shared_ptr(snde::component);
%shared_ptr(snde::assembly);
%shared_ptr(snde::nurbspart);
%shared_ptr(snde::meshedpart);
%shared_ptr(std::vector<std::shared_ptr<snde::meshedpart>>);

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
// The snde_geometrydata Python class is built by a wrapper function in geometrydata.i

%typemap(out) snde_geometrydata geom (PyObject *__module__=NULL,PyObject *module=NULL,PyObject *ctypes=NULL, PyObject *snde_geometrydata_class_buildfunc,PyObject *snde_geometrydata_class=NULL, PyObject *POINTER=NULL, PyObject *ManagerObj=NULL,PyObject *snde_geometrydata_class_p=NULL, PyObject *c_void_p=NULL, PyObject *CFUNCTYPE=NULL, PyObject *CMPFUNC=NULL, PyObject *CMPFUNC_INSTANCE=NULL, PyObject *reswrapper=NULL,snde_geometrydata *geom_ptr=NULL) %{
  geom_ptr=&arg1->geom;
  
  __module__=PyObject_GetAttrString($self,"__module__");
  module = PyImport_Import(__module__);
  ctypes = PyImport_ImportModule("ctypes");

  ManagerObj=SWIG_NewPointerObj( SWIG_as_voidptr(new std::shared_ptr<snde::arraymanager>(arg1->manager)), $descriptor(std::shared_ptr<arraymanager> *), SWIG_POINTER_OWN);

  snde_geometrydata_class_buildfunc = PyDict_GetItemString(PyModule_GetDict(module),"build_geometrydata_class");

  snde_geometrydata_class = PyObject_CallFunctionObjArgs(snde_geometrydata_class_buildfunc,ManagerObj,NULL);

  POINTER=PyDict_GetItemString(PyModule_GetDict(ctypes),"POINTER");

  
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
  Py_XDECREF(ManagerObj);
  Py_XDECREF(snde_geometrydata_class);
  Py_XDECREF(snde_geometrydata_class_p);
  //Py_XDECREF(ManagerObj);
  Py_XDECREF(ctypes);
  
  Py_XDECREF(module);
%}

namespace snde {

  static inline snde_coord4 quaternion_normalize(snde_coord4 unnormalized);
  static inline snde_coord4 quaternion_product(snde_coord4 quat1, snde_coord4 quat2);
  static inline snde_coord3 vector3_plus_vector3(snde_coord3 a,snde_coord3 b);
  static inline snde_coord3 quaternion_times_vector(snde_coord4 quat,snde_coord3 vec);
  static inline snde_orientation3 orientation_orientation_multiply(snde_orientation3 left,snde_orientation3 right);



  class geometry {
  public:
    struct snde_geometrydata geom;

    std::shared_ptr<arraymanager> manager;
    /* All arrays allocated by a particular allocator are locked together by manager->locker */
    
    
    geometry(double tol,std::shared_ptr<arraymanager> manager);

    // ***!!! NOTE: "addr()" method delegated to geom.contents by "bit of magic" below
    ~geometry();
  };


  class mesheduv {
  public:

    std::shared_ptr<geometry> geom;
    std::string name;
    snde_index idx;

    mesheduv(std::shared_ptr<geometry> geom, std::string name,snde_index mesheduvnum);

    ~mesheduv();
    
  };


// A little bit of magic that makes fields of the underlying geometry
// data structure directly accessible, significantly simplifying notation
// We do this by rewriting swig's __getattr__ and if the attribute does not exist,
// we catch it and delegate
//
%extend geometry {
  %pythoncode %{
    def __getattr__(self,name):
      try:
        return _swig_getattr(self,lockholder,name)
      except AttributeError:
        return getattr(self.geom.contents,name)
        pass
  %}

}

#define SNDE_PDET_INVALID 0
#define SNDE_PDET_INDEX 1
#define SNDE_PDET_DOUBLE 2
#define SNDE_PDET_STRING 3
 class paramdictentry {
  public:
    int type; /* see SNDE_PDET_... below */
    snde_index indexval;
    double doubleval;
    std::string stringval;

    paramdictentry();
    paramdictentry(snde_index _indexval);
    paramdictentry(double _doubleval);
    paramdictentry(std::string _stringval);
    snde_index idx();
    double dbl();
    std::string str();
  };



/* ***!!! Must keep sync'd with geometry.hpp */
#define SNDE_COMPONENT_GEOMWRITE_MESHEDPARTS (1u<<0)
#define SNDE_COMPONENT_GEOMWRITE_TRIS (1u<<1)
#define SNDE_COMPONENT_GEOMWRITE_REFPOINTS (1u<<2)
#define SNDE_COMPONENT_GEOMWRITE_MAXRADIUS (1u<<3)
#define SNDE_COMPONENT_GEOMWRITE_NORMAL (1u<<4)
#define SNDE_COMPONENT_GEOMWRITE_INPLANEMAT (1u<<5)
#define SNDE_COMPONENT_GEOMWRITE_EDGES (1u<<6)
#define SNDE_COMPONENT_GEOMWRITE_VERTICES (1u<<7)
#define SNDE_COMPONENT_GEOMWRITE_PRINCIPAL_CURVATURES (1u<<8)
#define SNDE_COMPONENT_GEOMWRITE_CURVATURE_TANGENT_AXES (1u<<9)
#define SNDE_COMPONENT_GEOMWRITE_VERTEX_EDGELIST_INDICES (1u<<10)
#define SNDE_COMPONENT_GEOMWRITE_VERTEX_EDGELIST (1u<<11)
#define SNDE_COMPONENT_GEOMWRITE_BOXES (1u<<12)
#define SNDE_COMPONENT_GEOMWRITE_BOXCOORDS (1u<<13)
#define SNDE_COMPONENT_GEOMWRITE_BOXPOLYS (1u<<14)

  class component { /* abstract base class for geometric components (assemblies, nurbspart, meshedpart) */
  public:
    typedef enum {
      subassembly=0,
      nurbs=1,
      meshed=2,
    } TYPE;

    TYPE type;

    virtual std::vector<snde_partinstance> get_instances(snde_orientation3 orientation,std::unordered_map<std::string,paramdictentry> paramdict)=0;

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

    virtual std::vector<snde_partinstance> get_instances(snde_orientation3 orientation,std::unordered_map<std::string,paramdictentry> paramdict);

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
    
    virtual void obtain_lock(std::shared_ptr<lockingprocess> process, unsigned writemask=0);
    virtual ~nurbspart();
    
  };

  %extend meshedpart {
    %pythoncode %{
      def obtain_lock_pycpp(self,process,holder,writemask):
        # NOTE: Parallel C++ implementation obtain_lock_pycpp 
        #  must be maintained in geometry.hpp
        #        holder=pylockholder()
        	
        if self.idx != SNDE_INDEX_INVALID:
          holder.store((yield process.get_locks_array_region(self.geom.addr("meshedparts"),writemask & SNDE_COMPONENT_GEOMWRITE_MESHEDPARTS,self.idx,1)))
            
          meshedparts=self.geom.field(holder,"meshedparts",writemask & SNDE_COMPONENT_GEOMWRITE_MESHEDPARTS,nt_snde_meshedpart,self.idx,1)
          if meshedparts[0]["firsttri"] != SNDE_INDEX_INVALID:
            holder.store((yield process.get_locks_array_region(self.geom.addr("triangles"),writemask & SNDE_COMPONENT_GEOMWRITE_TRIS,meshedparts[0]["firsttri"],meshedparts[0]["numtris"])))
            if self.geom.field_valid("refpoints"):
              holder.store((yield process.get_locks_array_region(self.geom.addr("refpoints"),writemask & SNDE_COMPONENT_GEOMWRITE_REFPOINTS,meshedparts[0]["firsttri"],meshedparts[0]["numtris"])))
              pass
            if self.geom.field_valid("maxradius"):
              holder.store((yield process.get_locks_array_region(self.geom.addr("maxradius"),writemask & SNDE_COMPONENT_GEOMWRITE_MAXRADIUS,meshedparts[0]["firsttri"],meshedparts[0]["numtris"])))
              pass
            if self.geom.field_valid("normal"):
              holder.store((yield process.get_locks_array_region(self.geom.addr("normal"),writemask & SNDE_COMPONENT_GEOMWRITE_NORMAL,meshedparts[0]["firsttri"],meshedparts[0]["numtris"])))
              pass
            if self.geom.field_valid("inplanemat"):
              holder.store((yield process.get_locks_array_region(self.geom.addr("inplanemat"),writemask & SNDE_COMPONENT_GEOMWRITE_INPLANEMAT,meshedparts[0]["firsttri"],meshedparts[0]["numtris"])))
              pass
            pass
          
          if meshedparts[0]["firstedge"] != SNDE_INDEX_INVALID:
            holder.store((yield process.get_locks_array_region(self.geom.addr("edges"),writemask & SNDE_COMPONENT_GEOMWRITE_EDGES,meshedparts[0]["firstedge"],meshedparts[0]["numedges"])))
            pass
          
          if meshedparts[0]["firstvertex"] != SNDE_INDEX_INVALID:
            holder.store((yield process.get_locks_array_region(self.geom.addr("vertices"),writemask & SNDE_COMPONENT_GEOMWRITE_VERTICES,meshedparts[0]["firstvertex"],meshedparts[0]["numvertices"])))
            if self.geom.field_valid("principal_curvatures"):
              holder.store((yield process.get_locks_array_region(self.geom.addr("principal_curvatures"),writemask & SNDE_COMPONENT_GEOMWRITE_PRINCIPAL_CURVATURES,meshedparts[0]["firstvertex"],meshedparts[0]["numvertices"])))
              pass
            if self.geom.field_valid("curvature_tangent_axes"):
              holder.store((yield process.get_locks_array_region(self.geom.addr("curvature_tangent_axes"),writemask & SNDE_COMPONENT_GEOMWRITE_CURVATURE_TANGENT_AXES,meshedparts[0]["firstvertex"],meshedparts[0]["numvertices"])))
              pass
            if self.geom.field_valid("vertex_edgelist_indices"):
              holder.store((yield process.get_locks_array_region(self.geom.addr("vertex_edgelist_indices"),writemask & SNDE_COMPONENT_VERTEX_EDGELIST_INDICES,meshedparts[0]["firstvertex"],meshedparts[0]["numvertices"])))
              pass		  
            pass
          	    
          if meshedparts[0]["first_vertex_edgelist"] != SNDE_INDEX_INVALID:
            holder.store((yield process.get_locks_array_region(self.geom.addr("vertex_edgelist"),writemask & SNDE_COMPONENT_GEOMWRITE_VERTEX_EDGELIST,meshedparts[0]["first_vertex_edgelist"],meshedparts[0]["num_vertex_edgelist"])))
            pass
          	    
          if meshedparts[0]["firstbox"] != SNDE_INDEX_INVALID:
            holder.store((yield process.get_locks_array_region(self.geom.addr("boxes"),writemask & SNDE_COMPONENT_GEOMWRITE_BOXES,meshedparts[0]["firstbox"],meshedparts[0]["numboxes"])))
            if self.geom.field_valid("boxcoord"):
              holder.store((yield process.get_locks_array_region(self.geom.addr("boxcoord"),writemask & SNDE_COMPONENT_GEOMWRITE_BOXCOORDS,meshedparts[0]["firstbox"],meshedparts[0]["numboxes"])))
              pass
            pass
          
          if meshedparts[0]["firstboxpoly"] != SNDE_INDEX_INVALID:
            holder.store((yield process.get_locks_array_region(self.geom.addr("boxpolys"),writemask & SNDE_COMPONENT_GEOMWRITE_BOXPOLYS,meshedparts[0]["firstboxpoly"],meshedparts[0]["numboxpolys"])))
            pass
          del meshedparts  # numpy array is temporary; good practice to explicitly delete
          pass
        pass  
      
%}
  }
  class meshedpart : public component {
    meshedpart(const meshedpart &)=delete; /* copy constructor disabled */
    meshedpart& operator=(const meshedpart &)=delete; /* copy assignment disabled */
    
  public:
    std::shared_ptr<geometry> geom;
    snde_index idx;
    std::map<std::string,std::shared_ptr<mesheduv>> parameterizations;
    bool destroyed;
 
    
    meshedpart(std::shared_ptr<geometry> geom,snde_index idx);

    void addparameterization(std::shared_ptr<mesheduv> parameterization);
    
    virtual std::vector<struct snde_partinstance> get_instances(snde_orientation3 orientation,std::unordered_map<std::string,paramdictentry> paramdict);
    

    virtual void obtain_lock(std::shared_ptr<lockingprocess> process, unsigned writemask=0);

    void free(); /* You must be certain that nothing could be using this part's database entries prior to free() */

    ~meshedpart();
  };

}

%template(meshedpart_vector) std::vector<std::shared_ptr<snde::meshedpart>>;  // used for return of x3d_load_geometry 
