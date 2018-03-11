#ifndef SNDE_GEOMETRY_HPP
#define SNDE_GEOMETRY_HPP

#include <cstring>

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

  
  class geometry {
  public:
    struct snde_geometrydata geom;

    std::shared_ptr<arraymanager> manager;
    /* All arrays allocated by a particular allocator are locked together by manager->locker */
    
    
    geometry(double tol,std::shared_ptr<arraymanager> manager) {
      memset(&geom,0,sizeof(geom)); // reset everything to NULL
      this->manager=manager;
      geom.tol=tol;

      //manager->add_allocated_array((void **)&geom.assemblies,sizeof(*geom.assemblies),0);
      manager->add_allocated_array((void **)&geom.meshedparts,sizeof(*geom.meshedparts),0);

      


      manager->add_allocated_array((void **)&geom.triangles,sizeof(*geom.triangles),0);
      manager->add_follower_array((void **)&geom.triangles,(void **)&geom.refpoints,sizeof(*geom.refpoints));
      manager->add_follower_array((void **)&geom.triangles,(void **)&geom.maxradius,sizeof(*geom.maxradius));
      manager->add_follower_array((void **)&geom.triangles,(void **)&geom.normal,sizeof(*geom.normal));
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


  class mesheduv {
  public:
    std::string name;
    std::shared_ptr<geometry> geom;
    snde_index idx; /* index of the mesheduv in the geometry database */

    mesheduv(std::shared_ptr<geometry> geom, std::string name,snde_index idx)
    /* WARNING: This constructor takes ownership of the part and 
       subcomponents from the geometry database and frees them when 
       it is destroyed */
    {
      this->geom=geom;
      this->name=name;
      this->idx=idx;
    }

    ~mesheduv()
    {
      /* Free our entries in the geometry database */
      assert(0); /* not yet implemented */
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
    
    virtual std::vector<snde_partinstance> get_instances(snde_orientation3 orientation,std::unordered_map<std::string,paramdictentry> paramdict)=0;

    virtual void obtain_lock(std::shared_ptr<lockingprocess> process, unsigned writemask=0)=0; /* writemask contains OR'd SNDE_COMPONENT_GEOMWRITE_xxx bits */

    virtual ~component() noexcept(false) {};
  };
  

  class assembly : public component {
    /* NOTE: Unlike other types of component, assemblies ARE copyable/assignable */
    /* (this is because they don't have a representation in the underlying
       geometry database) */
  public:    
    std::deque<std::shared_ptr<component>> pieces;
    snde_orientation3 _orientation; /* orientation of this part/assembly relative to its parent */

    assembly(snde_orientation3 orientation)
    {
      this->type=subassembly;
    }

    virtual snde_orientation3 orientation(void)
    {
      return _orientation;
    }


    virtual std::vector<snde_partinstance> get_instances(snde_orientation3 orientation,std::unordered_map<std::string,paramdictentry> paramdict)
    {
      std::vector<snde_partinstance> instances;
      std::unordered_map<std::string,paramdictentry> reducedparamdict;

      std::string name_with_dot=name+".";
      
      for (auto & pdename_pde : paramdict) {
	if (!strncmp(pdename_pde.first.c_str(),name_with_dot.c_str(),name_with_dot.size())) {
	  /* this paramdict entry name stats with this assembly name  + '.' */

	  /* give same entry to reducedparamdict, but with assembly name and dot stripped */
	  reducedparamdict[std::string(pdename_pde.first.c_str()+name_with_dot.size())]=pdename_pde.second;
	}
      }

      
      snde_orientation3 neworientation=orientation_orientation_multiply(orientation,_orientation);
      for (auto & piece : pieces) {
	std::vector<snde_partinstance> newpieces=piece->get_instances(neworientation,reducedparamdict);
	instances.insert(instances.end(),newpieces.begin(),newpieces.end());
      }
      return instances;
    }
    
    virtual void obtain_lock(std::shared_ptr<lockingprocess> process, unsigned writemask=0)
    {
      /* writemask contains OR'd SNDE_COMPONENT_GEOMWRITE_xxx bits */

      /* 
	 obtain locks from all our components... 
	 These have to be spawned so they can all obtain in parallel, 
	 following the locking order. 
      */
      for (auto piece=pieces.begin();piece != pieces.end(); piece++) {
	std::shared_ptr<component> pieceptr=*piece;
	process->spawn([ pieceptr,process,writemask ]() { pieceptr->obtain_lock(process,writemask); } );
	
      }
      
    }
    virtual ~assembly()
    {
      
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

    nurbspart(std::shared_ptr<geometry> geom,snde_index nurbspartnum)
    /* WARNING: This constructor takes ownership of the part and 
       subcomponents from the geometry database and (should) free them when 
       it is destroyed */
    {
      this->type=nurbs;
      this->geom=geom;
      //this->orientation=geom->geom.nurbsparts[nurbspartnum].orientation;
      this->nurbspartnum=nurbspartnum;
    }
    
    virtual void obtain_lock(std::shared_ptr<lockingprocess> process, unsigned writemask=0)
    {
      /* writemask contains OR'd SNDE_COMPONENT_GEOMWRITE_xxx bits */

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
    snde_index idx; // index in the meshedparts array
    std::map<std::string,std::shared_ptr<mesheduv>> parameterizations;
    bool destroyed;
    
    meshedpart(std::shared_ptr<geometry> geom,snde_index idx)
    /* WARNING: This constructor takes ownership of the part (if given) and 
       subcomponents from the geometry database and (should) free them when 
       free() is called */
    {
      this->type=meshed;
      this->geom=geom;
      this->idx=idx;
      this->destroyed=false;
    }

    void addparameterization(std::shared_ptr<mesheduv> parameterization)
    {
      parameterizations[parameterization->name]=parameterization;
    }
    
    virtual std::vector<struct snde_partinstance> get_instances(snde_orientation3 orientation,std::unordered_map<std::string,paramdictentry> paramdict)
    {
      struct snde_partinstance ret;

      ret.orientation=orientation;
      ret.nurbspartnum=SNDE_INDEX_INVALID;
      ret.meshedpartnum=idx;

      
      ret.firstuvpatch=0;
      {
	auto pname_entry=paramdict.find(name+"."+"firstuvpatch");
	if (pname_entry != paramdict.end()) {
	  ret.firstuvpatch=pname_entry->second.idx();
	}
      }

      {
	std::string parameterization_name="";
	
	if (parameterizations.size() > 0) {
	  parameterization_name=parameterizations.begin()->first;
	}
      
	auto pname_entry=paramdict.find(name+"."+"parameterization_name");
	if (pname_entry != paramdict.end()) {
	  parameterization_name=pname_entry->second.str();
	}
	
	auto pname_mesheduv=parameterizations.find(parameterization_name);
	if (pname_mesheduv==parameterizations.end()) {
	  ret.mesheduvnum=SNDE_INDEX_INVALID;
	} else {
	  ret.mesheduvnum=pname_mesheduv->second->idx;
	}
      }
      return std::vector<struct snde_partinstance>(1,ret);
    }
    

    virtual void obtain_lock(std::shared_ptr<lockingprocess> process, unsigned writemask=0)
    {
      /* writemask contains OR'd SNDE_COMPONENT_GEOMWRITE_xxx bits */

      /* 
	 obtain locks from all our components... 
	 These have to be spawned so they can all obtain in parallel, 
	 following the locking order. 
      */

      /* NOTE: Locking order here must follow order in geometry constructor (above) */
      /* NOTE: Parallel Python implementation obtain_lock_pycpp 
	 must be maintained in geometry.i */

      if (idx != SNDE_INDEX_INVALID) {
	process->get_locks_array_region((void **)&geom->geom.meshedparts,writemask & SNDE_COMPONENT_GEOMWRITE_MESHEDPARTS,idx,1);
      
	
	if (geom->geom.meshedparts[idx].firsttri != SNDE_INDEX_INVALID) {
	  process->get_locks_array_region((void **)&geom->geom.triangles,writemask & SNDE_COMPONENT_GEOMWRITE_TRIS,geom->geom.meshedparts[idx].firsttri,geom->geom.meshedparts[idx].numtris);
	  if (geom->geom.refpoints) {
	    process->get_locks_array_region((void **)&geom->geom.refpoints,writemask & SNDE_COMPONENT_GEOMWRITE_REFPOINTS,geom->geom.meshedparts[idx].firsttri,geom->geom.meshedparts[idx].numtris);
	  }
	  
	  if (geom->geom.maxradius) {
	    process->get_locks_array_region((void **)&geom->geom.maxradius,writemask & SNDE_COMPONENT_GEOMWRITE_MAXRADIUS,geom->geom.meshedparts[idx].firsttri,geom->geom.meshedparts[idx].numtris);
	  }
	  
	  if (geom->geom.normal) {
	    process->get_locks_array_region((void **)&geom->geom.normal,writemask & SNDE_COMPONENT_GEOMWRITE_NORMAL,geom->geom.meshedparts[idx].firsttri,geom->geom.meshedparts[idx].numtris);
	  }
	  
	  if (geom->geom.inplanemat) {
	    process->get_locks_array_region((void **)&geom->geom.inplanemat,writemask & SNDE_COMPONENT_GEOMWRITE_INPLANEMAT,geom->geom.meshedparts[idx].firsttri,geom->geom.meshedparts[idx].numtris);
	  }
	}
      }
      if (geom->geom.meshedparts[idx].firstedge != SNDE_INDEX_INVALID) {
	process->get_locks_array_region((void **)&geom->geom.edges,writemask & SNDE_COMPONENT_GEOMWRITE_EDGES,geom->geom.meshedparts[idx].firstedge,geom->geom.meshedparts[idx].numedges);
	
      }      
      if (geom->geom.meshedparts[idx].firstvertex != SNDE_INDEX_INVALID) {
	process->get_locks_array_region((void **)&geom->geom.vertices,writemask & SNDE_COMPONENT_GEOMWRITE_VERTICES,geom->geom.meshedparts[idx].firstvertex,geom->geom.meshedparts[idx].numvertices);

	if (geom->geom.principal_curvatures) {
	  process->get_locks_array_region((void **)&geom->geom.principal_curvatures,writemask & SNDE_COMPONENT_GEOMWRITE_PRINCIPAL_CURVATURES,geom->geom.meshedparts[idx].firstvertex,geom->geom.meshedparts[idx].numvertices);
	}

	if (geom->geom.curvature_tangent_axes) {
	  process->get_locks_array_region((void **)&geom->geom.curvature_tangent_axes,writemask & SNDE_COMPONENT_GEOMWRITE_CURVATURE_TANGENT_AXES,geom->geom.meshedparts[idx].firstvertex,geom->geom.meshedparts[idx].numvertices);
	}

	if (geom->geom.vertex_edgelist_indices) {
	  process->get_locks_array_region((void **)&geom->geom.vertex_edgelist_indices,writemask & SNDE_COMPONENT_GEOMWRITE_VERTEX_EDGELIST_INDICES,geom->geom.meshedparts[idx].firstvertex,geom->geom.meshedparts[idx].numvertices);
	}
      }
      
      
      if (geom->geom.meshedparts[idx].first_vertex_edgelist_index != SNDE_INDEX_INVALID) {
	process->get_locks_array_region((void **)&geom->geom.vertex_edgelist,writemask & SNDE_COMPONENT_GEOMWRITE_VERTEX_EDGELIST,geom->geom.meshedparts[idx].first_vertex_edgelist_index,geom->geom.meshedparts[idx].num_vertex_edgelist_indices);
	
      }      

	
      if (geom->geom.meshedparts[idx].firstbox != SNDE_INDEX_INVALID) {
	if (geom->geom.boxes) {
	    process->get_locks_array_region((void **)&geom->geom.boxes,writemask & SNDE_COMPONENT_GEOMWRITE_BOXES,geom->geom.meshedparts[idx].firstbox,geom->geom.meshedparts[idx].numboxes);
	}
	if (geom->geom.boxcoord) {
	  process->get_locks_array_region((void **)&geom->geom.boxcoord,writemask & SNDE_COMPONENT_GEOMWRITE_BOXCOORDS,geom->geom.meshedparts[idx].firstbox,geom->geom.meshedparts[idx].numboxes);
	}
      }
      
      
      if (geom->geom.boxpolys && geom->geom.meshedparts[idx].firstboxpoly != SNDE_INDEX_INVALID) {
	  process->get_locks_array_region((void **)&geom->geom.boxpolys,writemask & SNDE_COMPONENT_GEOMWRITE_BOXPOLYS,geom->geom.meshedparts[idx].firstboxpoly,geom->geom.meshedparts[idx].numboxpolys);
      }


      
      
    }



    

    void free() /* You must be certain that nothing could be using this part's database entries prior to free() */
    {
      /* Free our entries in the geometry database */
      if (idx != SNDE_INDEX_INVALID) {
	if (geom->geom.meshedparts[idx].firstboxpoly != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.boxpolys,geom->geom.meshedparts[idx].firstboxpoly,geom->geom.meshedparts[idx].numboxpolys);
	  geom->geom.meshedparts[idx].firstboxpoly = SNDE_INDEX_INVALID;
	}

	
	if (geom->geom.meshedparts[idx].firstbox != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.boxes,geom->geom.meshedparts[idx].firstbox,geom->geom.meshedparts[idx].numboxes);
	  geom->geom.meshedparts[idx].firstbox = SNDE_INDEX_INVALID;
	}
	
	if (geom->geom.meshedparts[idx].first_vertex_edgelist_index != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.vertex_edgelist,geom->geom.meshedparts[idx].first_vertex_edgelist_index,geom->geom.meshedparts[idx].num_vertex_edgelist_indices);
	  geom->geom.meshedparts[idx].first_vertex_edgelist_index = SNDE_INDEX_INVALID;
	}

	
	if (geom->geom.meshedparts[idx].firstvertex != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.vertices,geom->geom.meshedparts[idx].firstvertex,geom->geom.meshedparts[idx].numvertices);
	  geom->geom.meshedparts[idx].firstvertex = SNDE_INDEX_INVALID;
	}

	if (geom->geom.meshedparts[idx].firstedge != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.edges,geom->geom.meshedparts[idx].firstedge,geom->geom.meshedparts[idx].numedges);
	  geom->geom.meshedparts[idx].firstedge = SNDE_INDEX_INVALID;
	}

	
	if (geom->geom.meshedparts[idx].firsttri != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.triangles,geom->geom.meshedparts[idx].firsttri,geom->geom.meshedparts[idx].numtris);
	  geom->geom.meshedparts[idx].firsttri = SNDE_INDEX_INVALID;
	}
	

	geom->manager->free((void **)&geom->geom.meshedparts,idx,1);
	idx=SNDE_INDEX_INVALID;
      }
      destroyed=true;
    }
    
    ~meshedpart() noexcept(false)
    {
      if (!destroyed) {
	throw std::runtime_error("Should call free() method of meshedpart object before it goes out of scope and the destructor is called");
      }
    }

  };

  
  
}


#endif /* SNDE_GEOMETRY_HPP */
