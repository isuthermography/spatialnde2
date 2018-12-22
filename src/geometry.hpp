#ifndef SNDE_GEOMETRY_HPP
#define SNDE_GEOMETRY_HPP

/* Thoughts/updates: 
 * For OpenGL compatibility, drop everything down to triangle meshes
 * OpenGL does not support separate UV index. Per malexander in https://www.opengl.org/discussion_boards/showthread.php/183513-How-to-use-UV-Indices-in-OpenGL-4-4  
 * this can be addressed by passing a "Buffer Texture" object to the
 * geometry shader https://stackoverflow.com/questions/7954927/passing-a-list-of-values-to-fragment-shader,  which can then (use this to determine texture 
 * coordinates on the actual texture?) (Provide the correct texture
 * index on gl_PrimitiveID? and texture id on a custom output?
 * ... also support 32-bit index storage (snde_shortindex) that gets
 * immediately multiplied out because OpenGL doesn't support 
 * ... 64 bit indices yet. 

*/


//#include "geometry_types.h"

#include "arraymanager.hpp"


#include <stdexcept>
#include <cstring>




namespace snde {
  
  /* *** Where to store landmarks ***/
  /* Where to store frames? ***/

  static inline snde_coord4 quaternion_normalize(snde_coord4 unnormalized)
  /* returns the components of a normalized quaternion */
  {
    double norm;
    snde_coord4 ret;
    
    norm=sqrt(pow(unnormalized.coord[0],2) + pow(unnormalized.coord[1],2) + pow(unnormalized.coord[2],2)+pow(unnormalized.coord[3],2));

    ret.coord[0]=unnormalized.coord[0]/norm;
    ret.coord[1]=unnormalized.coord[1]/norm;
    ret.coord[2]=unnormalized.coord[2]/norm;
    ret.coord[3]=unnormalized.coord[3]/norm;

    return ret;
  }
  
  static inline snde_coord4 quaternion_product(snde_coord4 quat1, snde_coord4 quat2)
  {
    /* quaternion coordinates are i, j, k, real part */
    snde_coord4 unnormalized;
    unnormalized.coord[0]=quat1.coord[3]*quat2.coord[0] + quat1.coord[0]*quat2.coord[3] + quat1.coord[1]*quat2.coord[2] - quat1.coord[2]*quat2.coord[1];
    unnormalized.coord[1]=quat1.coord[3]*quat2.coord[1] + quat1.coord[1]*quat2.coord[3] - quat1.coord[0]*quat2.coord[2] + quat1.coord[2]*quat2.coord[0];
    unnormalized.coord[2]=quat1.coord[3]*quat2.coord[2] + quat1.coord[2]*quat2.coord[3] + quat1.coord[0]*quat2.coord[1] - quat1.coord[1]*quat2.coord[0];
    unnormalized.coord[3]=quat1.coord[3]*quat2.coord[3] - quat1.coord[0]*quat2.coord[0] - quat1.coord[1]*quat2.coord[1] - quat1.coord[2]*quat2.coord[2];
    
    return quaternion_normalize(unnormalized);
  }

  static inline snde_coord3 vector3_plus_vector3(snde_coord3 a,snde_coord3 b)
  {
    snde_coord3 retval;

    retval.coord[0]=a.coord[0]+b.coord[0];
    retval.coord[1]=a.coord[1]+b.coord[1];
    retval.coord[2]=a.coord[2]+b.coord[2];

    return retval;
  }

  static inline snde_coord3 quaternion_times_vector(snde_coord4 quat,snde_coord3 vec)
  /* assumes quat is normalized, stored as 'i,j,k,w' components */
  {
    snde_coord matrix[9];

    /* first row */
    matrix[0]=pow(quat.coord[0],2)-pow(quat.coord[1],2)-pow(quat.coord[2],2)+pow(quat.coord[3],2);
    matrix[1]=2.0*(quat.coord[0]*quat.coord[1] - quat.coord[3]*quat.coord[2]);
    matrix[2]=2.0*(quat.coord[0]*quat.coord[2] + quat.coord[3]*quat.coord[1]);
    /* second row */
    matrix[3]=2.0*(quat.coord[0]*quat.coord[1] + quat.coord[3]*quat.coord[2]);
    matrix[4]=-pow(quat.coord[0],2) + pow(quat.coord[1],2) - pow(quat.coord[2],2) + pow(quat.coord[3],2);
    matrix[5]=2.0*(quat.coord[1]*quat.coord[2] - quat.coord[3]*quat.coord[0]);
    /* third row */
    matrix[6]=2.0*(quat.coord[0]*quat.coord[2] - quat.coord[3]*quat.coord[1]);
    matrix[7]=2.0*(quat.coord[1]*quat.coord[2] + quat.coord[3]*quat.coord[0]);
    matrix[8]=-pow(quat.coord[0],2) - pow(quat.coord[1],2) + pow(quat.coord[2],2) + pow(quat.coord[3],2);

    snde_coord3 retval;
    unsigned rowcnt,colcnt;
    
    for (rowcnt=0;rowcnt < 3; rowcnt++) {
      retval.coord[rowcnt]=0;
      for (colcnt=0;colcnt < 3; colcnt++) {
	retval.coord[rowcnt] += matrix[rowcnt*3 + colcnt] * vec.coord[colcnt];
      }
    }
    return retval;
  }

  static inline snde_orientation3 orientation_orientation_multiply(snde_orientation3 left,snde_orientation3 right)
  {
      /* orientation_orientation_multiply must consider both quaternion and offset **/
      /* for vector v, quat rotation is q1vq1' */
      /* for point p, q1pq1' + o1  */
      /* for vector v double rotation is q2q1vq1'q2' ... where q2=left, q1=right */
      /* for point p  q2(q1pq1' + o1)q2' + o2 */
      /*             = q2q1pq1'q2' + q2o1q2' + o2 */
      /* so given q2, q1,   and o2, o1
	 product quaternion is q2q1
         product offset is q2o1q2' + o2 */

    snde_orientation3 retval; 

    retval.quat=quaternion_product(left.quat,right.quat);
    retval.offset = vector3_plus_vector3(quaternion_times_vector(left.quat,right.offset),left.offset);
    
    return retval;
  }

  class component; // Forward declaration
  class assembly; // Forward declaration
  class meshedpart; // Forward declaration
  class nurbspart; // Forward declaration

  
  class geometry {
  public:
    struct snde_geometrydata geom;

    std::shared_ptr<arraymanager> manager;
    /* All arrays allocated by a particular allocator are locked together by manager->locker */


    // object_trees is a set of named scene graphs (actually scene trees)
    // They can be accessed by unique names, with any default given a
    // name of "", which (due to the string sort order) is also
    // accessible as object_trees.begin()
    std::map<std::string,std::shared_ptr<component>> object_trees;
    std::mutex object_trees_lock; // Any access to object_trees or contents or rendering engines set up to handle object_trees. PRECEDES any locking of
    // geometry data in the locking order, and PRECEDES any locking of the
    // transactional revision manager (if in use for this geometry).
    
    geometry(double tol,std::shared_ptr<arraymanager> manager) {
      memset(&geom,0,sizeof(geom)); // reset everything to NULL
      this->manager=manager;
      geom.tol=tol;

      //manager->add_allocated_array((void **)&geom.assemblies,sizeof(*geom.assemblies),0);
      manager->add_allocated_array((void **)&geom.meshedparts,sizeof(*geom.meshedparts),0);

      


      manager->add_allocated_array((void **)&geom.triangles,sizeof(*geom.triangles),0);
      manager->add_follower_array((void **)&geom.triangles,(void **)&geom.refpoints,sizeof(*geom.refpoints));
      manager->add_follower_array((void **)&geom.triangles,(void **)&geom.maxradius,sizeof(*geom.maxradius));
      manager->add_follower_array((void **)&geom.triangles,(void **)&geom.normals,sizeof(*geom.normals));
      manager->add_follower_array((void **)&geom.triangles,(void **)&geom.inplanemat,sizeof(*geom.inplanemat));

      manager->add_allocated_array((void **)&geom.edges,sizeof(*geom.edges),0);


      manager->add_allocated_array((void **)&geom.vertices,sizeof(*geom.vertices),0);
      manager->add_follower_array((void **)&geom.vertices,(void **)&geom.principal_curvatures,sizeof(*geom.principal_curvatures));
      manager->add_follower_array((void **)&geom.vertices,(void **)&geom.curvature_tangent_axes,sizeof(*geom.curvature_tangent_axes));

      manager->add_follower_array((void **)&geom.vertices,(void **)&geom.vertex_edgelist_indices,sizeof(*geom.vertex_edgelist_indices));
      manager->add_allocated_array((void **)&geom.vertex_edgelist,sizeof(*geom.vertex_edgelist),0);
      

      manager->add_allocated_array((void **)&geom.boxes,sizeof(*geom.boxes),0);
      manager->add_follower_array((void **)&geom.boxes,(void **)&geom.boxcoord,sizeof(*geom.boxcoord));
      
      manager->add_allocated_array((void **)&geom.boxpolys,sizeof(*geom.boxpolys),0);


      
      /* parameterization */
      manager->add_allocated_array((void **)&geom.mesheduv,sizeof(*geom.mesheduv),0);
      manager->add_allocated_array((void **)&geom.uv_triangles,sizeof(*geom.uv_triangles),0);
      manager->add_follower_array((void **)&geom.uv_triangles,(void **)&geom.inplane2uvcoords,sizeof(*geom.inplane2uvcoords));
      manager->add_follower_array((void **)&geom.uv_triangles,(void **)&geom.uvcoords2inplane,sizeof(*geom.uvcoords2inplane));
      manager->add_follower_array((void **)&geom.uv_triangles,(void **)&geom.uv_patch_index,sizeof(*geom.uv_patch_index));

      manager->add_allocated_array((void **)&geom.uv_edges,sizeof(*geom.uv_edges),0);


      manager->add_allocated_array((void **)&geom.uv_vertices,sizeof(*geom.uv_vertices),0);
      manager->add_follower_array((void **)&geom.uv_vertices,(void **)&geom.uv_vertex_edgelist_indices,sizeof(*geom.uv_vertex_edgelist_indices));

      manager->add_allocated_array((void **)&geom.uv_vertex_edgelist,sizeof(*geom.uv_vertex_edgelist),0);


      // ***!!! insert NURBS here !!!***
      
      manager->add_allocated_array((void **)&geom.uv_boxes,sizeof(*geom.uv_boxes),0);
      manager->add_follower_array((void **)&geom.uv_boxes,(void **)&geom.uv_boxcoord,sizeof(*geom.boxcoord));
      
      manager->add_allocated_array((void **)&geom.uv_boxpolys,sizeof(*geom.uv_boxpolys),0);


      /***!!! Insert uv patches and images here ***!!! */
      
      manager->add_allocated_array((void **)&geom.vertex_arrays,sizeof(*geom.vertex_arrays),0);

      manager->add_allocated_array((void **)&geom.texvertex_arrays,sizeof(*geom.texvertex_arrays),0);

      
      // ... need to initialize rest of struct...
      // Probably want an array manager class to handle all of this
      // initialization,
      // also creation and caching of OpenCL buffers and OpenGL buffers. 
      
    }

    // ***!!! NOTE: "addr()" Python method delegated to geom.contents by "bit of magic" in geometry.i
    
    ~geometry()
    {
      // Destructor needs to wipe out manager's array pointers because they point into this geometry object, that
      // is being destroyed
      manager->cleararrays((void *)&geom,sizeof(geom));
      
    }
  };


  class uv_patches {
  public:
    /* a collection of uv patches represents the uv-data for a meshedpart or nurbspart, as references to images. 
       each patch has a corresponding image and set of meaningful coordinates. The collection is named, so that 
       we can map a different collection onto our part by changing the name. */
    std::string name; /* is name the proper way to index this? probably not. Will need to change once we understand better */
    std::shared_ptr<geometry> geom;
    snde_index firstuvpatch,numuvpatches; /*must match the numbers to be put into the snde_partinstance */
    bool destroyed;

    uv_patches(const uv_patches &)=delete; /* copy constructor disabled */
    uv_patches& operator=(const uv_patches &)=delete; /* copy assignment disabled */

    uv_patches(std::shared_ptr<geometry> geom, std::string name, snde_index firstuvpatch, snde_index numuvpatches)
    {
      this->geom=geom;
      this->name=name;
      this->firstuvpatch=firstuvpatch;
      this->numuvpatches=numuvpatches;
      destroyed=false;
    }

    void free()
    {
      if (firstuvpatch != SNDE_INDEX_INVALID) {
	geom->manager->free((void **)&geom->geom.uv_patches,firstuvpatch); //,numuvpatches);
	firstuvpatch=SNDE_INDEX_INVALID;	
      }
      destroyed=true;
    }
    
    ~uv_patches()
#if !defined(_MSC_VER) || _MSC_VER > 1800 // except for MSVC2013 and earlier
    noexcept(false)
#endif
    {
      if (!destroyed) {
	throw std::runtime_error("Should call free() method of uv_patches object before it goes out of scope and the destructor is called");
      }
    }

    
  };
  

  class mesheduv {
  public:
    std::string name;
    std::shared_ptr<geometry> geom;
    snde_index idx; /* index of the mesheduv in the geometry database */
    std::map<std::string,std::shared_ptr<uv_patches>> patches;
    bool destroyed;
    
    /* Should the mesheduv manage the snde_image data for the various uv patches? probably... */

    mesheduv(std::shared_ptr<geometry> geom, std::string name,snde_index idx)
    /* WARNING: This constructor takes ownership of the mesheduv and 
       subcomponents from the geometry database and frees them when 
       it is destroyed */
    {
      this->geom=geom;
      this->name=name;
      this->idx=idx;
      destroyed=false;
    }

    std::shared_ptr<uv_patches> find_patches(std::string name)
    {
      return patches.at(name);
    }

    void addpatches(std::shared_ptr<uv_patches> to_add)
    {
      //patches.emplace(std::make_pair<std::string,std::shared_ptr<uv_patches>>(to_add->name,to_add));
      patches.emplace(std::pair<std::string,std::shared_ptr<uv_patches>>(to_add->name,to_add));
    }

    
    void free()
    {
      /* Free our entries in the geometry database */

      if (idx != SNDE_INDEX_INVALID) {
	if (geom->geom.mesheduv->firstuvtri != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.uv_vertices,geom->geom.mesheduv->firstuvtri); //,geom->geom.mesheduv->numuvtris);
	  geom->geom.mesheduv->firstuvtri = SNDE_INDEX_INVALID;	    
	}

	if (geom->geom.mesheduv->firstuvedge != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.uv_vertices,geom->geom.mesheduv->firstuvedge);//,geom->geom.mesheduv->numuvedges);
	  geom->geom.mesheduv->firstuvedge = SNDE_INDEX_INVALID;	    
	}

	if (geom->geom.mesheduv->firstuvvertex != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.uv_vertices,geom->geom.mesheduv->firstuvvertex); //,geom->geom.mesheduv->numuvvertices);
	  
	  geom->geom.mesheduv->firstuvvertex = SNDE_INDEX_INVALID;
	  
	}

	if (geom->geom.mesheduv->first_uv_vertex_edgelist != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.uv_vertices,geom->geom.mesheduv->first_uv_vertex_edgelist); //,geom->geom.mesheduv->num_uv_vertex_edgelist);
	  geom->geom.mesheduv->first_uv_vertex_edgelist = SNDE_INDEX_INVALID;
	  
	}

	if (geom->geom.mesheduv->firstuvbox != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.uv_boxes,geom->geom.mesheduv->firstuvbox); //,geom->geom.mesheduv->numuvboxes);
	  geom->geom.mesheduv->firstuvbox = SNDE_INDEX_INVALID; 
	}

	if (geom->geom.mesheduv->firstuvboxpoly != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.uv_boxpolys,geom->geom.mesheduv->firstuvboxpoly); //,geom->geom.mesheduv->numuvboxpoly);
	  geom->geom.mesheduv->firstuvboxpoly = SNDE_INDEX_INVALID;	    
	}


	if (geom->geom.mesheduv->firstuvboxcoord != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.uv_boxcoord,geom->geom.mesheduv->firstuvboxcoord); //,geom->geom.mesheduv->numuvboxcoords);
	  geom->geom.mesheduv->firstuvboxcoord = SNDE_INDEX_INVALID;	    
	}
	

	geom->manager->free((void **)&geom->geom.mesheduv,idx);// ,1);
	idx=SNDE_INDEX_INVALID;

	
      }

      for (auto & name_patches : patches) {
	name_patches.second->free();
      }
      destroyed=true;

    }

    ~mesheduv()
#if !defined(_MSC_VER) || _MSC_VER > 1800 // except for MSVC2013 and earlier
    noexcept(false)
#endif
    {
      if (!destroyed) {
	throw std::runtime_error("Should call free() method of mesheduv object before it goes out of scope and the destructor is called");
      }
    }
    
  };


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

   paramdictentry()
   {
     type=SNDE_PDET_INVALID;
   }
   
    paramdictentry(snde_index _indexval):  indexval(_indexval)
    {
      type=SNDE_PDET_INDEX;
    }
    paramdictentry(double _doubleval):  doubleval(_doubleval)
    {
      type=SNDE_PDET_DOUBLE;
    }
    paramdictentry(std::string _stringval): stringval(_stringval)
    {
      type=SNDE_PDET_STRING;
    }

    snde_index idx()
    {
      if (type!=SNDE_PDET_INDEX) {
	throw std::runtime_error(std::string("Attempt to extract paramdict entry of type ")+std::to_string(type)+" as type index");
      }
      return indexval;
    }
    double dbl()
    {
      if (type!=SNDE_PDET_DOUBLE) {
	throw std::runtime_error(std::string("Attempt to extract paramdict entry of type ")+std::to_string(type)+" as type double");
      }
      return doubleval;
    }
    std::string str()
    {
      if (type!=SNDE_PDET_STRING) {
	throw std::runtime_error(std::string("Attempt to extract paramdict entry of type ")+std::to_string(type)+" as type string");
      }
      return stringval;
    }
  };


  
  /* ***!!! Must keep sync'd with geometry.i */
#define SNDE_COMPONENT_GEOM_MESHEDPARTS (1ull<<0)
#define SNDE_COMPONENT_GEOM_TRIS (1ull<<1)
#define SNDE_COMPONENT_GEOM_REFPOINTS (1ull<<2)
#define SNDE_COMPONENT_GEOM_MAXRADIUS (1ull<<3)
#define SNDE_COMPONENT_GEOM_NORMALS (1ull<<4)
#define SNDE_COMPONENT_GEOM_INPLANEMAT (1ull<<5)
#define SNDE_COMPONENT_GEOM_EDGES (1ull<<6)
#define SNDE_COMPONENT_GEOM_VERTICES (1ull<<7)
#define SNDE_COMPONENT_GEOM_PRINCIPAL_CURVATURES (1ull<<8)
#define SNDE_COMPONENT_GEOM_CURVATURE_TANGENT_AXES (1ull<<9)
#define SNDE_COMPONENT_GEOM_VERTEX_EDGELIST_INDICES (1ull<<10)
#define SNDE_COMPONENT_GEOM_VERTEX_EDGELIST (1ull<<11)
#define SNDE_COMPONENT_GEOM_BOXES (1ull<<12)
#define SNDE_COMPONENT_GEOM_BOXCOORD (1ull<<13)
#define SNDE_COMPONENT_GEOM_BOXPOLYS (1ull<<14)

#define SNDE_COMPONENT_GEOM_ALL ((1ull<<15)-1)

// Resizing masks -- mark those arrays that resize together
#define SNDE_COMPONENT_GEOM_MESHEDPARTS_RESIZE (SNDE_COMPONENT_GEOM_MESHEDPARTS)
#define SNDE_COMPONENT_GEOM_TRIS_RESIZE (SNDE_COMPONENT_GEOM_TRIS|SNDE_COMPONENT_GEOM_REFPOINTS|SNDE_COMPONENT_GEOM_MAXRADIUS|SNDE_COMPONENT_GEOM_NORMALS|SNDE_COMPONENT_GEOM_INPLANEMAT)
#define SNDE_COMPONENT_GEOM_EDGES_RESIZE (SNDE_COMPONENT_GEOM_EDGES)
#define SNDE_COMPONENT_GEOM_VERTICES_RESIZE (SNDE_COMPONENT_GEOM_VERTICES|SNDE_COMPONENT_GEOM_PRINCIPAL_CURVATURES|SNDE_COMPONENT_GEOM_CURVATURE_TANGENT_AXES|SNDE_COMPONENT_GEOM_VERTEX_EDGELIST_INDICES)
#define SNDE_COMPONENT_GEOM_VERTEX_EDGELIST_RESIZE (SNDE_COMPONENT_GEOM_VERTEX_EDGELIST)
#define SNDE_COMPONENT_GEOM_BOXES_RESIZE (SNDE_COMPONENT_GEOM_BOXES|SNDE_COMPONENT_GEOM_BOXCOORD)
#define SNDE_COMPONENT_GEOM_BOXPOLYS_RESIZE (SNDE_COMPONENT_GEOM_BOXPOLYS)


  
typedef uint64_t snde_component_geom_mask_t;

  
  class component : public std::enable_shared_from_this<component> { /* abstract base class for geometric components (assemblies, nurbspart, meshedpart) */

    // orientation model:
    // Each assembly has orientation
    // orientations of nested assemblys multiply
    // ... apply that product of quaternions to a vector in the part space  ...q1 q2 v q2^-1 q1^-1 for
    // inner assembly orientation q2, to get a vector in the world space.
    // ... apply element by element from inside out, the quaternion, then an offset, to a point in the part space
    // to get a point in the world space
  public:
    std::string name; // used for parameter paths ... form: assemblyname.assemblyname.partname.parameter as paramdict key
    typedef enum {
      subassembly=0,
      nurbs=1,
      meshed=2,
    } TYPE;

    TYPE type;

    //   component();// {}
    
    virtual std::vector<std::pair<snde_partinstance,std::shared_ptr<component>>> get_instances(snde_orientation3 orientation,std::shared_ptr<std::unordered_map<std::string,paramdictentry>> paramdict)=0;

    virtual void obtain_lock(std::shared_ptr<lockingprocess> process, snde_component_geom_mask_t readmask=SNDE_COMPONENT_GEOM_ALL,snde_component_geom_mask_t writemask=0,snde_component_geom_mask_t resizemask=0)=0; /* writemask contains OR'd SNDE_COMPONENT_GEOM_xxx bits */

    virtual ~component()
#if !defined(_MSC_VER) || _MSC_VER > 1800 // except for MSVC2013 and earlier
    noexcept(false)
#endif
    {};
  };
  

  class assembly : public component {
    /* NOTE: Unlike other types of component, assemblies ARE copyable/assignable */
    /* (this is because they don't have a representation in the underlying
       geometry database) */
  public:    
    std::deque<std::shared_ptr<component>> pieces;
    snde_orientation3 _orientation; /* orientation of this part/assembly relative to its parent */

    /* NOTE: May want to add cache of 
       openscenegraph group nodes representing 
       this assembly  */ 

    assembly(std::string name,snde_orientation3 orientation)
    {
      this->name=name;
      this->type=subassembly;
      this->_orientation=orientation;
      
    }

    virtual snde_orientation3 orientation(void)
    {
      return _orientation;
    }


    virtual std::vector<std::pair<snde_partinstance,std::shared_ptr<component>>> get_instances(snde_orientation3 orientation,std::shared_ptr<std::unordered_map<std::string,paramdictentry>> paramdict)
    {
      std::vector<std::pair<snde_partinstance,std::shared_ptr<component>>> instances;
      std::shared_ptr<std::unordered_map<std::string,paramdictentry>> reducedparamdict=std::make_shared<std::unordered_map<std::string,paramdictentry>>();

      std::string name_with_dot=name+".";
      
      for (auto & pdename_pde : *paramdict) {
	if (!strncmp(pdename_pde.first.c_str(),name_with_dot.c_str(),name_with_dot.size())) {
	  /* this paramdict entry name stats with this assembly name  + '.' */

	  /* give same entry to reducedparamdict, but with assembly name and dot stripped */
	  (*reducedparamdict)[std::string(pdename_pde.first.c_str()+name_with_dot.size())]=pdename_pde.second;
	}
      }

      
      snde_orientation3 neworientation=orientation_orientation_multiply(orientation,_orientation);
      for (auto & piece : pieces) {
	std::vector<std::pair<snde_partinstance,std::shared_ptr<component>>> newpieces=piece->get_instances(neworientation,reducedparamdict);
	instances.insert(instances.end(),newpieces.begin(),newpieces.end());
      }
      return instances;
    }
    
    virtual void obtain_lock(std::shared_ptr<lockingprocess> process, snde_component_geom_mask_t readmask=SNDE_COMPONENT_GEOM_ALL,snde_component_geom_mask_t writemask=0,snde_component_geom_mask_t resizemask=0)
    {
      /* readmask and writemask contain OR'd SNDE_COMPONENT_GEOM_xxx bits */

      /* 
	 obtain locks from all our components... 
	 These have to be spawned so they can all obtain in parallel, 
	 following the locking order. 
      */
      for (auto piece=pieces.begin();piece != pieces.end(); piece++) {
	std::shared_ptr<component> pieceptr=*piece;
	process->spawn([ pieceptr,process,readmask,writemask,resizemask ]() { pieceptr->obtain_lock(process,readmask,writemask,resizemask); } );
	
      }
      
    }
    virtual ~assembly()
    {
      
    }


    static std::shared_ptr<assembly> from_partlist(std::string name,std::shared_ptr<std::vector<std::shared_ptr<meshedpart>>> parts)
    {
      
      std::shared_ptr<assembly> assem=std::make_shared<assembly>(name,snde_null_orientation3());

      for (size_t cnt=0; cnt < parts->size();cnt++) {
	assem->pieces.push_back(std::static_pointer_cast<component>((*parts)[cnt]));
      }
      
      
      return assem;
    }
    
  };


  /* NOTE: Could have additional abstraction layer to accommodate 
     multi-resolution approximations */
  class nurbspart : public component {
    nurbspart(const nurbspart &)=delete; /* copy constructor disabled */
    nurbspart& operator=(const nurbspart &)=delete; /* copy assignment disabled */
  public:
    snde_index nurbspartnum;
    std::shared_ptr<geometry> geom;

    nurbspart(std::shared_ptr<geometry> geom,std::string name,snde_index nurbspartnum)
    /* WARNING: This constructor takes ownership of the part and 
       subcomponents from the geometry database and (should) free them when 
       it is destroyed */
    {
      this->type=nurbs;
      this->geom=geom;
      this->name=name;
      //this->orientation=geom->geom.nurbsparts[nurbspartnum].orientation;
      this->nurbspartnum=nurbspartnum;
    }
    
    virtual void obtain_lock(std::shared_ptr<lockingprocess> process, snde_component_geom_mask_t readmask=SNDE_COMPONENT_GEOM_ALL,snde_component_geom_mask_t writemask=0,snde_component_geom_mask_t resizemask=0)
    {
      /* writemask contains OR'd SNDE_COMPONENT_GEOM_xxx bits */

      assert(0); /* not yet implemented */
      
    } 
    virtual ~nurbspart()
    {
      assert(0); /* not yet implemented */
    }
    
  };

  class meshedpart : public component {
    meshedpart(const meshedpart &)=delete; /* copy constructor disabled */
    meshedpart& operator=(const meshedpart &)=delete; /* copy assignment disabled */
    
  public:
    std::shared_ptr<geometry> geom;
    snde_index idx; // index in the meshedparts geometrydata array
    std::map<std::string,std::shared_ptr<mesheduv>> parameterizations; /* NOTE: is a string (URI?) really the proper way to index parameterizations? ... may want to change this */
    bool need_normals; // set if this meshedpart was loaded/created without normals being assigned, and therefore still needs normals
    bool destroyed;

    /* NOTE: May want to add cache of 
       openscenegraph geodes or drawables representing 
       this part */ 
    
    meshedpart(std::shared_ptr<geometry> geom,std::string name,snde_index idx)
    /* WARNING: This constructor takes ownership of the part (if given) and 
       subcomponents from the geometry database and (should) free them when 
       free() is called */
    {
      this->type=meshed;
      this->geom=geom;
      this->name=name;
      this->idx=idx;
      this->destroyed=false;
    }

    void addparameterization(std::shared_ptr<mesheduv> parameterization)
    {
      parameterizations[parameterization->name]=parameterization;
    }
    
    virtual std::vector<std::pair<snde_partinstance,std::shared_ptr<component>>> get_instances(snde_orientation3 orientation,std::shared_ptr<std::unordered_map<std::string,paramdictentry>> paramdict)
    {
      struct snde_partinstance ret;
      std::shared_ptr<component> ret_ptr;

      ret_ptr = shared_from_this();

      std::vector<std::pair<snde_partinstance,std::shared_ptr<component>>> ret_vec;

      ret.orientation=orientation;
      ret.nurbspartnum=SNDE_INDEX_INVALID;
      ret.meshedpartnum=idx;

      
      ret.firstuvpatch=SNDE_INDEX_INVALID;
      ret.numuvpatches=SNDE_INDEX_INVALID;
      ret.mesheduvnum=SNDE_INDEX_INVALID;
      ret.imgbuf_extra_offset=0;

      {
	std::string parameterization_name="";
	
	if (parameterizations.size() > 0) {
	  parameterization_name=parameterizations.begin()->first;
	}
      
	auto pname_entry=paramdict->find(name+"."+"parameterization_name");
	if (pname_entry != paramdict->end()) {
	  parameterization_name=pname_entry->second.str();
	}
	
	auto pname_mesheduv=parameterizations.find(parameterization_name);
	if (pname_mesheduv==parameterizations.end()) {
	  ret.mesheduvnum=SNDE_INDEX_INVALID;
	} else {
	  ret.mesheduvnum=pname_mesheduv->second->idx;


	  auto pname_entry=paramdict->find(name+"."+"patches");
	  if (pname_entry != paramdict->end()) {
	    std::string patchesname = pname_entry->second.str();
	    
	    std::shared_ptr<uv_patches> patches = pname_mesheduv->second->find_patches(patchesname);

	    if (!patches) {
	      throw std::runtime_error("meshedpart::get_instances():  Unknown UV patch name: "+patchesname);
	    }
	    ret.firstuvpatch=patches->firstuvpatch;
	    ret.numuvpatches=patches->numuvpatches;
	      
	  }
	  
	}
      }
      
      //return std::vector<struct snde_partinstance>(1,ret);
      ret_vec.push_back(std::make_pair(ret,ret_ptr));
      return ret_vec;
    }
    

    virtual void obtain_lock(std::shared_ptr<lockingprocess> process, snde_component_geom_mask_t readmask=SNDE_COMPONENT_GEOM_ALL, snde_component_geom_mask_t writemask=0, snde_component_geom_mask_t resizemask=0)
    {
      /* writemask contains OR'd SNDE_COMPONENT_GEOM_xxx bits */

      /* 
	 obtain locks from all our components... 
	 These have to be spawned so they can all obtain in parallel, 
	 following the locking order. 
      */

      /* NOTE: Locking order here must follow order in geometry constructor (above) */
      /* NOTE: Parallel Python implementation obtain_lock_pycpp 
	 must be maintained in geometry.i */

      assert(readmask & SNDE_COMPONENT_GEOM_MESHEDPARTS); // Cannot do remainder of locking without read access to meshedpart

      if (idx != SNDE_INDEX_INVALID) {
	process->get_locks_array_mask((void **)&geom->geom.meshedparts,SNDE_COMPONENT_GEOM_MESHEDPARTS,SNDE_COMPONENT_GEOM_MESHEDPARTS_RESIZE,readmask,writemask,resizemask,idx,1);
      
	
	if (geom->geom.meshedparts[idx].firsttri != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.triangles,SNDE_COMPONENT_GEOM_TRIS,SNDE_COMPONENT_GEOM_TRIS_RESIZE,readmask,writemask,resizemask,geom->geom.meshedparts[idx].firsttri,geom->geom.meshedparts[idx].numtris);
	  if (geom->geom.refpoints) {
	    process->get_locks_array_mask((void **)&geom->geom.refpoints,SNDE_COMPONENT_GEOM_REFPOINTS,SNDE_COMPONENT_GEOM_TRIS_RESIZE,readmask,writemask,resizemask,geom->geom.meshedparts[idx].firsttri,geom->geom.meshedparts[idx].numtris);
	  }
	  
	  if (geom->geom.maxradius) {
	    process->get_locks_array_mask((void **)&geom->geom.maxradius,SNDE_COMPONENT_GEOM_MAXRADIUS,SNDE_COMPONENT_GEOM_TRIS_RESIZE,readmask,writemask,resizemask,geom->geom.meshedparts[idx].firsttri,geom->geom.meshedparts[idx].numtris);
	  }
	  
	  if (geom->geom.normals) {
	    process->get_locks_array_mask((void **)&geom->geom.normals,SNDE_COMPONENT_GEOM_NORMALS,SNDE_COMPONENT_GEOM_TRIS_RESIZE,readmask,writemask,resizemask,geom->geom.meshedparts[idx].firsttri,geom->geom.meshedparts[idx].numtris);
	  }
	  
	  if (geom->geom.inplanemat) {
	    process->get_locks_array_mask((void **)&geom->geom.inplanemat,SNDE_COMPONENT_GEOM_INPLANEMAT,SNDE_COMPONENT_GEOM_TRIS_RESIZE,readmask,writemask,resizemask,geom->geom.meshedparts[idx].firsttri,geom->geom.meshedparts[idx].numtris);
	  }
	}
      
	if (geom->geom.meshedparts[idx].firstedge != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.edges,SNDE_COMPONENT_GEOM_EDGES,SNDE_COMPONENT_GEOM_EDGES_RESIZE,readmask,writemask,resizemask,geom->geom.meshedparts[idx].firstedge,geom->geom.meshedparts[idx].numedges);
	}      
	if (geom->geom.meshedparts[idx].firstvertex != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.vertices,SNDE_COMPONENT_GEOM_VERTICES,SNDE_COMPONENT_GEOM_VERTICES_RESIZE,readmask,writemask,resizemask,geom->geom.meshedparts[idx].firstvertex,geom->geom.meshedparts[idx].numvertices);
	}

	if (geom->geom.principal_curvatures) {
	  process->get_locks_array_mask((void **)&geom->geom.principal_curvatures,SNDE_COMPONENT_GEOM_PRINCIPAL_CURVATURES,SNDE_COMPONENT_GEOM_VERTICES_RESIZE,readmask,writemask,resizemask,geom->geom.meshedparts[idx].firstvertex,geom->geom.meshedparts[idx].numvertices);
	    
	}
	
	if (geom->geom.curvature_tangent_axes) {
	  process->get_locks_array_mask((void **)&geom->geom.curvature_tangent_axes,SNDE_COMPONENT_GEOM_CURVATURE_TANGENT_AXES,SNDE_COMPONENT_GEOM_VERTICES_RESIZE,readmask,writemask,resizemask,geom->geom.meshedparts[idx].firstvertex,geom->geom.meshedparts[idx].numvertices);
	  
	}

	if (geom->geom.vertex_edgelist_indices) {
	  process->get_locks_array_mask((void **)&geom->geom.vertex_edgelist_indices,SNDE_COMPONENT_GEOM_VERTEX_EDGELIST_INDICES,SNDE_COMPONENT_GEOM_VERTICES_RESIZE,readmask,writemask,resizemask,geom->geom.meshedparts[idx].firstvertex,geom->geom.meshedparts[idx].numvertices);
	}
	
      }
      
      
      if (geom->geom.meshedparts[idx].first_vertex_edgelist != SNDE_INDEX_INVALID) {
	process->get_locks_array_mask((void **)&geom->geom.vertex_edgelist,SNDE_COMPONENT_GEOM_VERTEX_EDGELIST,SNDE_COMPONENT_GEOM_VERTEX_EDGELIST_RESIZE,readmask,writemask,resizemask,geom->geom.meshedparts[idx].first_vertex_edgelist,geom->geom.meshedparts[idx].num_vertex_edgelist);	
      }      
      
      
      if (geom->geom.meshedparts[idx].firstbox != SNDE_INDEX_INVALID) {
	if (geom->geom.boxes) {
	  process->get_locks_array_mask((void **)&geom->geom.boxes,SNDE_COMPONENT_GEOM_BOXES,SNDE_COMPONENT_GEOM_BOXES_RESIZE,readmask,writemask,resizemask,geom->geom.meshedparts[idx].firstbox,geom->geom.meshedparts[idx].numboxes);
	  
	}
	if (geom->geom.boxcoord) {
	  process->get_locks_array_mask((void **)&geom->geom.boxcoord,SNDE_COMPONENT_GEOM_BOXCOORD,SNDE_COMPONENT_GEOM_BOXES_RESIZE,readmask,writemask,resizemask,geom->geom.meshedparts[idx].firstbox,geom->geom.meshedparts[idx].numboxes);
	}
      }
            
      if (geom->geom.boxpolys && geom->geom.meshedparts[idx].firstboxpoly != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.boxpolys,SNDE_COMPONENT_GEOM_BOXPOLYS,SNDE_COMPONENT_GEOM_BOXPOLYS_RESIZE,readmask,writemask,resizemask,geom->geom.meshedparts[idx].firstboxpoly,geom->geom.meshedparts[idx].numboxpolys);
      } 
      
      
      
    }



    

    void free() /* You must be certain that nothing could be using this part's database entries prior to free() */
    {
      /* Free our entries in the geometry database */
      if (idx != SNDE_INDEX_INVALID) {
	if (geom->geom.meshedparts[idx].firstboxpoly != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.boxpolys,geom->geom.meshedparts[idx].firstboxpoly); // ,geom->geom.meshedparts[idx].numboxpolys);
	  geom->geom.meshedparts[idx].firstboxpoly = SNDE_INDEX_INVALID;
	}

	
	if (geom->geom.meshedparts[idx].firstbox != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.boxes,geom->geom.meshedparts[idx].firstbox); //,geom->geom.meshedparts[idx].numboxes);
	  geom->geom.meshedparts[idx].firstbox = SNDE_INDEX_INVALID;
	}
	
	if (geom->geom.meshedparts[idx].first_vertex_edgelist != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.vertex_edgelist,geom->geom.meshedparts[idx].first_vertex_edgelist); //,geom->geom.meshedparts[idx].num_vertex_edgelist);
	  geom->geom.meshedparts[idx].first_vertex_edgelist = SNDE_INDEX_INVALID;
	}

	
	if (geom->geom.meshedparts[idx].firstvertex != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.vertices,geom->geom.meshedparts[idx].firstvertex); //,geom->geom.meshedparts[idx].numvertices);
	  geom->geom.meshedparts[idx].firstvertex = SNDE_INDEX_INVALID;
	}

	if (geom->geom.meshedparts[idx].firstedge != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.edges,geom->geom.meshedparts[idx].firstedge); //,geom->geom.meshedparts[idx].numedges);
	  geom->geom.meshedparts[idx].firstedge = SNDE_INDEX_INVALID;
	}

	
	if (geom->geom.meshedparts[idx].firsttri != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.triangles,geom->geom.meshedparts[idx].firsttri); //,geom->geom.meshedparts[idx].numtris);
	  geom->geom.meshedparts[idx].firsttri = SNDE_INDEX_INVALID;
	}
	

	geom->manager->free((void **)&geom->geom.meshedparts,idx); //,1);
	idx=SNDE_INDEX_INVALID;
      }
      destroyed=true;
    }
    
    ~meshedpart()
#if !defined(_MSC_VER) || _MSC_VER > 1800 // except for MSVC2013 and earlier
    noexcept(false)
#endif
    {
      if (!destroyed) {
	throw std::runtime_error("Should call free() method of meshedpart object before it goes out of scope and the destructor is called");
      }
    }

  };

  
  
}


#endif /* SNDE_GEOMETRY_HPP */
