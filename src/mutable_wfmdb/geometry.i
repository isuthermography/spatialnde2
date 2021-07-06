%shared_ptr(snde::geometry);
%shared_ptr(snde::parameterization);
%shared_ptr(snde::component);
%shared_ptr(snde::assembly);
%shared_ptr(snde::part);
%shared_ptr(std::vector<std::shared_ptr<snde::part>>);
%shared_ptr(snde::immutable_metadata);
%shared_ptr(snde::image_data);

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

  class component; // Forward declaration
  class assembly; // Forward declaration
  class part; // Forward declaration
  class geometry_function; // Forward declaration
  // class nurbspart; // Forward declaration
  class mutableinfostore;
  class mutabledatastore;
  class mutableparameterizationstore;
  class mutablewfmdb;

  class immutable_metadata; // forward reference
  class image_data;


  class geometry {
  public:
    struct snde_geometrydata geom;

    std::shared_ptr<arraymanager> manager;
    /* All arrays allocated by a particular allocator are locked together by manager->locker */
    
    
    geometry(double tol,std::shared_ptr<arraymanager> manager);

    // ***!!! NOTE: "addr()" method delegated to geom.contents by "bit of magic" below
    ~geometry();
  };
  class uv_images {
  public:
    /* a collection of uv images represents the uv-data for a meshedpart or nurbspart, as references to images. 
       each patch has a corresponding image and set of meaningful coordinates. The collection is named, so that 
       we can map a different collection onto our part by changing the name. */
    std::shared_ptr<geometry> geom;
    std::string parameterization_name; 
    snde_index firstuvimage,numuvimages; /*must match the numbers to be put into the snde_partinstance/snde_parameterization */

    std::vector<snde_image *> images; // These pointers shouldn't be shared (otherwise they would be shared pointers) because we are responsible for destroying them
    
    bool destroyed;

    uv_images(const uv_images &)=delete; /* copy constructor disabled */
    uv_images& operator=(const uv_images &)=delete; /* copy assignment disabled */

    
    uv_images(std::shared_ptr<geometry> geom, std::string parameterization_name, snde_index firstuvimage, snde_index numuvimages);

    void free();
    
    ~uv_images();

    
  };


  class parameterization {
  public:

    std::shared_ptr<geometry> geom;
    //std::string name;
    snde_index idx;

    parameterization(std::shared_ptr<geometry> geom, snde_index idx,snde_index numuvimages);

    virtual void obtain_uv_lock(std::shared_ptr<lockingprocess> process, std::shared_ptr<iterablewfmrefs> wfmdb_wfmlist=nullptr,std::string wfmdb_context="/",snde_infostore_lock_mask_t readmask=SNDE_UV_GEOM_ALL, snde_infostore_lock_mask_t writemask=0, snde_infostore_lock_mask_t resizemask=0);
    void free();


    ~parameterization();
    
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




%nodefaultctor component; // component is abstract so it shouldn't have default constructor created
  class component : public lockable_infostore_or_component { /* abstract base class for geometric components (assemblies, nurbspart, part) */
  public:
   //typedef enum {
   //   subassembly=0,
   //   nurbs=1,
   //   meshed=2,
   //} TYPE;
   //
   // TYPE type;
   
   // virtual std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>> get_instances(std::shared_ptr<mutablewfmdb> wfmdb,std::string wfmdb_context,snde_orientation3 orientation, std::shared_ptr<immutable_metadata> metadata, std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>(std::shared_ptr<part> partdata,std::vector<std::string> parameterization_data_names)> get_param_data)=0;
    
   //virtual void obtain_geom_lock(std::shared_ptr<lockingprocess> process, snde_infostore_lock_mask_t readmask=SNDE_COMPONENT_GEOM_ALL,snde_infostore_lock_mask_t writemask=0,snde_infostore_lock_mask_t resizemask=0)=0; /* writemask contains OR'd SNDE_COMPONENT_GEOM_xxx bits */

   //virtual void _explore_component(std::set<std::shared_ptr<component>,std::owner_less<std::shared_ptr<component>>> &component_set)=0; /* readmask/writemask contains OR'd SNDE_INFOSTORE_xxx bits */

   // virtual void obtain_lock(std::shared_ptr<lockingprocess> process,snde_infostore_lock_mask_t readmask=SNDE_INFOSTORE_COMPONENTS,snde_infostore_lock_mask_t writemask=0)=0;

    virtual ~component();
  };


 
 

  %extend part {
    %pythoncode %{
      def obtain_lock_pycpp(self,process,holder,readmask,writemask,resizemask):
        # NOTE: Parallel C++ implementation obtain_lock_pycpp 
        #  must be maintained in geometry.hpp
        #        holder=pylockholder()
        	
        if self.idx != SNDE_INDEX_INVALID:
          assert(readmask & SNDE_COMPONENT_GEOM_PARTS)
          holder.store((yield process.get_locks_array_mask(self.geom.addr("parts"),SNDE_COMPONENT_GEOM_PARTS,SNDE_COMPONENT_GEOM_PARTS_RESIZE,readmask,writemask,resizemask,self.idx,1)))
            
          parts=self.geom.field(holder,"parts",writemask & SNDE_COMPONENT_GEOM_PARTS,nt_snde_part,self.idx,1)
          if parts[0]["firsttri"] != SNDE_INDEX_INVALID:
            holder.store((yield process.get_locks_array_mask(self.geom.addr("triangles"),SNDE_COMPONENT_GEOM_TRIS,SNDE_COMPONENT_GEOM_TRIS_RESIZE,readmask,writemask,resizemask,parts[0]["firsttri"],parts[0]["numtris"])))
            pass      
            if self.geom.field_valid("refpoints"):
              holder.store((yield process.get_locks_array_mask(self.geom.addr("refpoints"),SNDE_COMPONENT_GEOM_REFPOINTS,SNDE_COMPONENT_GEOM_TRIS_RESIZE,readmask,writemask,resizemask,parts[0]["firsttri"],parts[0]["numtris"])))
              pass
            if self.geom.field_valid("maxradius"):
              holder.store((yield process.get_locks_array_mask(self.geom.addr("maxradius"),SNDE_COMPONENT_GEOM_MAXRADIUS,SNDE_COMPONENT_GEOM_TRIS_RESIZE,readmask,writemask,resizemask,parts[0]["firsttri"],parts[0]["numtris"])))
              pass
            if self.geom.field_valid("normal"):
              holder.store((yield process.get_locks_array_mask(self.geom.addr("normal"),SNDE_COMPONENT_GEOM_NORMAL,SNDE_COMPONENT_GEOM_TRIS_RESIZE,readmask,writemask,resizemask,parts[0]["firsttri"],parts[0]["numtris"])))
              pass
            if self.geom.field_valid("inplanemat"):
              holder.store((yield process.get_locks_array_mask(self.geom.addr("inplanemat"),SNDE_COMPONENT_GEOM_INPLANEMAT,SNDE_COMPONENT_GEOM_TRIS_RESIZE,readmask,writemask,resizemask,parts[0]["firsttri"],parts[0]["numtris"])))
              pass
            pass
          
          if parts[0]["firstedge"] != SNDE_INDEX_INVALID:
            holder.store((yield process.get_locks_array_mask(self.geom.addr("edges"),SNDE_COMPONENT_GEOM_EDGES,SNDE_COMPONENT_GEOM_EDGES_RESIZE,readmask,writemask,resizemask,parts[0]["firstedge"],parts[0]["numedges"])))
            pass
          
          if parts[0]["firstvertex"] != SNDE_INDEX_INVALID:
            holder.store((yield process.get_locks_array_mask(self.geom.addr("vertices"),SNDE_COMPONENT_GEOM_VERTICES,SNDE_COMPONENT_GEOM_VERTICES_RESIZE,readmask,writemask,resizemask,parts[0]["firstvertex"],parts[0]["numvertices"])))
            if self.geom.field_valid("principal_curvatures"):
              holder.store((yield process.get_locks_array_mask(self.geom.addr("principal_curvatures"),SNDE_COMPONENT_GEOM_PRINCIPAL_CURVATURES,SNDE_COMPONENT_GEOM_VERTICES_RESIZE,readmask,writemask,resizemask,parts[0]["firstvertex"],parts[0]["numvertices"])))
              pass
            if self.geom.field_valid("curvature_tangent_axes"):
              holder.store((yield process.get_locks_array_mask(self.geom.addr("curvature_tangent_axes"),SNDE_COMPONENT_GEOM_CURVATURE_TANGENT_AXES,SNDE_COMPONENT_GEOM_VERTICES_RESIZE,readmask,writemask,resizemask,parts[0]["firstvertex"],parts[0]["numvertices"])))
              pass
            if self.geom.field_valid("vertex_edgelist_indices"):
              holder.store((yield process.get_locks_array_mask(self.geom.addr("vertex_edgelist_indices"),SNDE_COMPONENT_GEOM_VERTEX_EDGELIST_INDICES,SNDE_COMPONENT_GEOM_VERTICES_RESIZE,readmask,writemask,resizemask,parts[0]["firstvertex"],parts[0]["numvertices"])))
              pass		  
            pass
          	    
          if parts[0]["first_vertex_edgelist"] != SNDE_INDEX_INVALID:
            holder.store((yield process.get_locks_array_mask(self.geom.addr("vertex_edgelist"),SNDE_COMPONENT_GEOM_VERTEX_EDGELIST,SNDE_COMPONENT_GEOM_VERTEX_EDGELIST_RESIZE,readmask,writemask,resizemask,parts[0]["first_vertex_edgelist"],parts[0]["num_vertex_edgelist"])))
            pass
          	    
          if parts[0]["firstbox"] != SNDE_INDEX_INVALID:
            holder.store((yield process.get_locks_array_mask(self.geom.addr("boxes"),SNDE_COMPONENT_GEOM_BOXES,SNDE_COMPONENT_GEOM_BOXES_RESIZE,readmask,writemask,resizemask,parts[0]["firstbox"],parts[0]["numboxes"])))
            if self.geom.field_valid("boxcoord"):
              holder.store((yield process.get_locks_array_mask(self.geom.addr("boxcoord"),SNDE_COMPONENT_GEOM_BOXCOORD,SNDE_COMPONENT_GEOM_BOXES_RESIZE,readmask,writemask,resizemask,parts[0]["firstbox"],parts[0]["numboxes"])))
              pass
            pass
          
          if parts[0]["firstboxpoly"] != SNDE_INDEX_INVALID:
            holder.store((yield process.get_locks_array_mask(self.geom.addr("boxpolys"),SNDE_COMPONENT_GEOM_BOXPOLYS,SNDE_COMPONENT_GEOM_BOXPOLYS_RESIZE,readmask,writemask,resizemask,parts[0]["firstboxpoly"],parts[0]["numboxpolys"])))
            pass
          del parts  # numpy array is temporary; good practice to explicitly delete
          pass
        pass  
      
%}
 }
  class part : public component {
    part(const part &)=delete; /* copy constructor disabled */
    part& operator=(const part &)=delete; /* copy assignment disabled */
    
  public:
    std::shared_ptr<geometry> geom;
    snde_index idx();
    bool destroyed;
 
    
    part(std::shared_ptr<geometry> geom,snde_index idx);

    std::shared_ptr<const std::set<std::string>> parameterizations();

    //virtual std::shared_ptr<mutableparameterizationstore> addparameterization(std::shared_ptr<mutablewfmdb> wfmdb,std::string wfmdb_context,std::shared_ptr<snde::parameterization> parameterization,std::string name,const wfmmetadata &metadata);
    
    void free(); /* You must be certain that nothing could be using this part's database entries prior to free() */

    ~part();
  };



    class assembly : public component {
    /* NOTE: Unlike other types of component, assemblies ARE copyable/assignable */
    /* (this is because they don't have a representation in the underlying
       geometry database) */
  public:    
      //std::map<std::string,std::tuple<snde_orientation3,std::shared_ptr<component>>> pieces;

    assembly();


    //virtual void obtain_lock(std::shared_ptr<lockingprocess> process,snde_infostore_lock_mask_t readmask=SNDE_INFOSTORE_COMPONENTS,snde_infostore_lock_mask_t writemask=0); /* readmask/writemask contains OR'd SNDE_INFOSTORE_xxx bits */
    //virtual std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>> get_instances(std::shared_ptr<mutablewfmdb> wfmdb,std::string wfmdb_context,snde_orientation3 orientation, std::shared_ptr<immutable_metadata> metadata, std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>(std::shared_ptr<part> partdata,std::vector<std::string> parameterization_data_names)> get_param_data);
    static std::shared_ptr<assembly> from_partlist(std::shared_ptr<mutablewfmdb> wfmdb,std::string wfmdb_context,std::shared_ptr<std::vector<std::string>> partnames);

    
    virtual ~assembly();

    //static std::tuple<std::shared_ptr<assembly>,std::unordered_map<std::string,metadatum>> from_partlist(std::string name,std::shared_ptr<std::vector<std::pair<std::shared_ptr<snde::part>,std::unordered_map<std::string,metadatum>>>> parts);

  };

  


}

%template(part_vector) std::vector<std::shared_ptr<snde::part>>;  // used for return of x3d_load_geometry 
