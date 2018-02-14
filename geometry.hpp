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

      


      manager->add_allocated_array((void **)&geom.vertexidx,sizeof(*geom.vertexidx),0);
      manager->add_follower_array((void **)&geom.vertexidx,(void **)&geom.refpoints,sizeof(*geom.refpoints));
      manager->add_follower_array((void **)&geom.vertexidx,(void **)&geom.maxradius,sizeof(*geom.maxradius));
      manager->add_follower_array((void **)&geom.vertexidx,(void **)&geom.normal,sizeof(*geom.normal));
      manager->add_follower_array((void **)&geom.vertexidx,(void **)&geom.inplanemat,sizeof(*geom.inplanemat));
      

      manager->add_allocated_array((void **)&geom.vertices,sizeof(*geom.vertices),0);
      manager->add_follower_array((void **)&geom.vertices,(void **)&geom.principal_curvatures,sizeof(*geom.principal_curvatures));
      manager->add_follower_array((void **)&geom.vertices,(void **)&geom.curvature_tangent_axes,sizeof(*geom.curvature_tangent_axes));



      manager->add_allocated_array((void **)&geom.boxes,sizeof(*geom.boxes),0);
      manager->add_follower_array((void **)&geom.boxes,(void **)&geom.boxcoord,sizeof(*geom.boxcoord));
      
      manager->add_allocated_array((void **)&geom.boxpolys,sizeof(*geom.boxpolys),0);



      // ... need to initialize rest of struct...
      // Probably want an array manager class to handle all of this
      // initialization,
      // also creation and caching of OpenCL buffers and OpenGL buffers. 
      
    }
    
    ~geometry()
    {
      // Destructor needs to wipe out manager's array pointers because they point into this geometry object, that
      // is being destroyed
      manager->cleararrays((void *)&geom,sizeof(geom));
      
    }
  };


  class mesheduv {
  public:

    std::shared_ptr<geometry> geom;
    std::string name;
    snde_index mesheduvnum;

    mesheduv(std::shared_ptr<geometry> geom, std::string name,snde_index mesheduvnum)
    /* WARNING: This constructor takes ownership of the part and 
       subcomponents from the geometry database and frees them when 
       it is destroyed */
    {
      this->geom=geom;
      this->name=name;
      this->mesheduvnum=mesheduvnum;
    }

    ~mesheduv()
    {
      /* Free our entries in the geometry database */
      assert(0); /* not yet implemented */
    }
    
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

    //   component();// {}
    
    virtual snde_orientation3 orientation()=0;

    virtual void obtain_lock(std::shared_ptr<lockingprocess> process, unsigned writemask=0)=0; /* writemask contains OR'd SNDE_COMPONENT_GEOMWRITE_xxx bits */

    virtual ~component() {};
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
    
    virtual snde_orientation3 orientation(void)
    {
      if (nurbspartnum==SNDE_INDEX_INVALID) {
	throw std::invalid_argument("invalid NURBS part number");
      }
      return geom->geom.nurbsparts[nurbspartnum].orientation;
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
    snde_index meshedpartnum;
    std::map<std::string,std::shared_ptr<mesheduv>> parameterizations;

    
    meshedpart(std::shared_ptr<geometry> geom,snde_index meshedpartnum)
    /* WARNING: This constructor takes ownership of the part (if given) and 
       subcomponents from the geometry database and (should) free them when 
       it is destroyed */
    {
      this->type=meshed;
      this->geom=geom;
      this->meshedpartnum=meshedpartnum;
    }

    void addparameterization(std::shared_ptr<mesheduv> parameterization)
    {
      parameterizations[parameterization->name]=parameterization;
    }
    
    virtual snde_orientation3 orientation(void)
    {
      if (meshedpartnum==SNDE_INDEX_INVALID) {
	throw std::invalid_argument("invalid meshed part number");
      }
      return geom->geom.meshedparts[meshedpartnum].orientation;
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

      if (meshedpartnum != SNDE_INDEX_INVALID) {
	if (writemask & SNDE_COMPONENT_GEOMWRITE_MESHEDPARTS) {
	  process->get_locks_write_array_region((void **)&geom->geom.meshedparts,meshedpartnum,1);
	} else {
	  process->get_locks_read_array_region((void **)&geom->geom.meshedparts,meshedpartnum,1);
	}
      }

      if (geom->geom.meshedparts[meshedpartnum].firstvertex != SNDE_INDEX_INVALID) {
	if (writemask & SNDE_COMPONENT_GEOMWRITE_VERTICES) {
	  process->get_locks_write_array_region((void **)&geom->geom.vertices,geom->geom.meshedparts[meshedpartnum].firstvertex,geom->geom.meshedparts[meshedpartnum].numvertices);
	} else {
	  process->get_locks_read_array_region((void **)&geom->geom.vertices,geom->geom.meshedparts[meshedpartnum].firstvertex,geom->geom.meshedparts[meshedpartnum].numvertices);
	}

	if (geom->geom.principal_curvatures) {
	  if (writemask & SNDE_COMPONENT_GEOMWRITE_PRINCIPAL_CURVATURES) {
	    process->get_locks_write_array_region((void **)&geom->geom.principal_curvatures,geom->geom.meshedparts[meshedpartnum].firstvertex,geom->geom.meshedparts[meshedpartnum].numvertices);
	  } else {
	    process->get_locks_read_array_region((void **)&geom->geom.principal_curvatures,geom->geom.meshedparts[meshedpartnum].firstvertex,geom->geom.meshedparts[meshedpartnum].numvertices);
	  }
	}

	if (geom->geom.curvature_tangent_axes) {
	  if (writemask & SNDE_COMPONENT_GEOMWRITE_CURVATURE_TANGENT_AXES) {
	    process->get_locks_write_array_region((void **)&geom->geom.curvature_tangent_axes,geom->geom.meshedparts[meshedpartnum].firstvertex,geom->geom.meshedparts[meshedpartnum].numvertices);
	  } else {
	    process->get_locks_read_array_region((void **)&geom->geom.curvature_tangent_axes,geom->geom.meshedparts[meshedpartnum].firstvertex,geom->geom.meshedparts[meshedpartnum].numvertices);
	  }
	}
      }
      

      if (geom->geom.meshedparts[meshedpartnum].firsttri != SNDE_INDEX_INVALID) {
	if (writemask & SNDE_COMPONENT_GEOMWRITE_TRIS) {
	  process->get_locks_write_array_region((void **)&geom->geom.vertexidx,geom->geom.meshedparts[meshedpartnum].firsttri,geom->geom.meshedparts[meshedpartnum].numtris);
	} else {
	  process->get_locks_read_array_region((void **)&geom->geom.vertexidx,geom->geom.meshedparts[meshedpartnum].firsttri,geom->geom.meshedparts[meshedpartnum].numtris);
	}
	
	if (geom->geom.refpoints) {
	  if (writemask & SNDE_COMPONENT_GEOMWRITE_REFPOINTS) {
	    process->get_locks_write_array_region((void **)&geom->geom.refpoints,geom->geom.meshedparts[meshedpartnum].firsttri,geom->geom.meshedparts[meshedpartnum].numtris);
	  } else {
	    process->get_locks_read_array_region((void **)&geom->geom.refpoints,geom->geom.meshedparts[meshedpartnum].firsttri,geom->geom.meshedparts[meshedpartnum].numtris);
	  }	
	}
	
	if (geom->geom.maxradius) {
	  if (writemask & SNDE_COMPONENT_GEOMWRITE_MAXRADIUS) {
	    process->get_locks_write_array_region((void **)&geom->geom.maxradius,geom->geom.meshedparts[meshedpartnum].firsttri,geom->geom.meshedparts[meshedpartnum].numtris);
	  } else {
	    process->get_locks_read_array_region((void **)&geom->geom.maxradius,geom->geom.meshedparts[meshedpartnum].firsttri,geom->geom.meshedparts[meshedpartnum].numtris);
	  }	
	}
	
	if (geom->geom.normal) {
	  if (writemask & SNDE_COMPONENT_GEOMWRITE_NORMAL) {
	    process->get_locks_write_array_region((void **)&geom->geom.normal,geom->geom.meshedparts[meshedpartnum].firsttri,geom->geom.meshedparts[meshedpartnum].numtris);
	  } else {
	    process->get_locks_read_array_region((void **)&geom->geom.normal,geom->geom.meshedparts[meshedpartnum].firsttri,geom->geom.meshedparts[meshedpartnum].numtris);
	  }	
	}
	
	if (geom->geom.inplanemat) {
	  if (writemask & SNDE_COMPONENT_GEOMWRITE_INPLANEMAT) {
	    process->get_locks_write_array_region((void **)&geom->geom.inplanemat,geom->geom.meshedparts[meshedpartnum].firsttri,geom->geom.meshedparts[meshedpartnum].numtris);
	  } else {
	    process->get_locks_read_array_region((void **)&geom->geom.inplanemat,geom->geom.meshedparts[meshedpartnum].firsttri,geom->geom.meshedparts[meshedpartnum].numtris);
	  }	
	}
      }
	
      if (geom->geom.meshedparts[meshedpartnum].firstbox != SNDE_INDEX_INVALID) {
	if (geom->geom.boxes) {
	  if (writemask & SNDE_COMPONENT_GEOMWRITE_BOXES) {
	    process->get_locks_write_array_region((void **)&geom->geom.boxes,geom->geom.meshedparts[meshedpartnum].firstbox,geom->geom.meshedparts[meshedpartnum].numboxes);
	  } else {
	    process->get_locks_read_array_region((void **)&geom->geom.boxes,geom->geom.meshedparts[meshedpartnum].firstbox,geom->geom.meshedparts[meshedpartnum].numboxes);
	  }	
	}

	if (geom->geom.boxcoord) {
	  if (writemask & SNDE_COMPONENT_GEOMWRITE_BOXCOORDS) {
	    process->get_locks_write_array_region((void **)&geom->geom.boxcoord,geom->geom.meshedparts[meshedpartnum].firstbox,geom->geom.meshedparts[meshedpartnum].numboxes);
	  } else {
	    process->get_locks_read_array_region((void **)&geom->geom.boxcoord,geom->geom.meshedparts[meshedpartnum].firstbox,geom->geom.meshedparts[meshedpartnum].numboxes);
	  }	
	}
      }


      if (geom->geom.boxpolys && geom->geom.meshedparts[meshedpartnum].firstboxpoly != SNDE_INDEX_INVALID) {
	if (writemask & SNDE_COMPONENT_GEOMWRITE_BOXPOLYS) {
	  process->get_locks_write_array_region((void **)&geom->geom.boxpolys,geom->geom.meshedparts[meshedpartnum].firstboxpoly,geom->geom.meshedparts[meshedpartnum].numboxpoly);
	} else {
	  process->get_locks_read_array_region((void **)&geom->geom.boxpolys,geom->geom.meshedparts[meshedpartnum].firstboxpoly,geom->geom.meshedparts[meshedpartnum].numboxpoly);
	}	
      }


      
      
    }


    ~meshedpart()
    {
      /* Free our entries in the geometry database */
      if (meshedpartnum != SNDE_INDEX_INVALID) {
	if (geom->geom.meshedparts[meshedpartnum].firstboxpoly != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.boxpolys,geom->geom.meshedparts[meshedpartnum].firstboxpoly,geom->geom.meshedparts[meshedpartnum].numboxpoly);
	  geom->geom.meshedparts[meshedpartnum].firstboxpoly = SNDE_INDEX_INVALID;
	}

	
	if (geom->geom.meshedparts[meshedpartnum].firstbox != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.boxes,geom->geom.meshedparts[meshedpartnum].firstbox,geom->geom.meshedparts[meshedpartnum].numboxes);
	  geom->geom.meshedparts[meshedpartnum].firstbox = SNDE_INDEX_INVALID;
	}
	
	if (geom->geom.meshedparts[meshedpartnum].firsttri != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.vertexidx,geom->geom.meshedparts[meshedpartnum].firsttri,geom->geom.meshedparts[meshedpartnum].numtris);
	  geom->geom.meshedparts[meshedpartnum].firsttri = SNDE_INDEX_INVALID;
	}

	if (geom->geom.meshedparts[meshedpartnum].firstvertex != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.vertices,geom->geom.meshedparts[meshedpartnum].firstvertex,geom->geom.meshedparts[meshedpartnum].numvertices);
	  geom->geom.meshedparts[meshedpartnum].firstvertex = SNDE_INDEX_INVALID;
	}

	geom->manager->free((void **)&geom->geom.meshedparts,meshedpartnum,1);
	meshedpartnum=SNDE_INDEX_INVALID;
      }
    }
    
  };

  
  
}


#endif /* SNDE_GEOMETRY_HPP */
