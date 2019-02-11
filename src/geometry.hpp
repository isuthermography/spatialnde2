
#include "geometry_types.h"
#include "geometrydata.h"

#include "arraymanager.hpp"

#include "metadata.hpp"

#include <stdexcept>
#include <cstring>


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
  class part; // Forward declaration
  // class nurbspart; // Forward declaration

  
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
      manager->add_allocated_array((void **)&geom.parts,sizeof(*geom.parts),0);

      manager->add_allocated_array((void **)&geom.topos,sizeof(*geom.topos),0);
      manager->add_allocated_array((void **)&geom.topo_indices,sizeof(*geom.topo_indices),0);

      


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
      manager->add_allocated_array((void **)&geom.uvs,sizeof(*geom.uvs),0);
      manager->add_allocated_array((void **)&geom.uv_topos,sizeof(*geom.uv_topos),0);
      manager->add_allocated_array((void **)&geom.uv_topo_indices,sizeof(*geom.uv_topo_indices),0);
      
      manager->add_allocated_array((void **)&geom.uv_triangles,sizeof(*geom.uv_triangles),0);
      manager->add_follower_array((void **)&geom.uv_triangles,(void **)&geom.inplane2uvcoords,sizeof(*geom.inplane2uvcoords));
      manager->add_follower_array((void **)&geom.uv_triangles,(void **)&geom.uvcoords2inplane,sizeof(*geom.uvcoords2inplane));
      //manager->add_follower_array((void **)&geom.uv_triangles,(void **)&geom.uv_patch_index,sizeof(*geom.uv_patch_index));

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

      manager->add_allocated_array((void **)&geom.texbuffer,sizeof(*geom.texbuffer),0);

      
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

  class uv_images; // forward declaration
  class uv_image {
  public:
    // represents single snde_image
    // intended as a base class
    std::shared_ptr<geometry> geom;
    snde_index firstuvimage;
    snde_index imageidx;

    uv_image(std::shared_ptr<uv_images> images,
	     snde_index imageidx); // implementation in geometry.cpp

    uv_image(const uv_image &)=delete; /* copy constructor disabled */
    uv_image& operator=(const uv_image &)=delete; /* copy assignment disabled */

    virtual snde_image &get_ref_arraylocked()
    {
      return geom->geom.uv_images[firstuvimage+imageidx];
      
    }

    virtual ~uv_image()
    {
      
    }
  };

  class uv_images {
  public:
    /* a collection of uv images represents the uv-data for a meshedpart or nurbspart, as references to images. 
       each patch has a corresponding image and set of meaningful coordinates. The collection is named, so that 
       we can map a different collection onto our part by changing the name. */
    std::shared_ptr<geometry> geom;
    std::string parameterization_name; 
    snde_index firstuvimage,numuvimages; /*must match the numbers to be put into the snde_partinstance/snde_parameterization */

    std::vector<uv_image *> images; // These pointers shouldn't be shared (otherwise they would be shared pointers) because we are responsible for destroying them
    
    bool destroyed;

    uv_images(const uv_images &)=delete; /* copy constructor disabled */
    uv_images& operator=(const uv_images &)=delete; /* copy assignment disabled */

    
    uv_images(std::shared_ptr<geometry> geom, std::string parameterization_name, snde_index firstuvimage, snde_index numuvimages) :
      geom(geom),
      parameterization_name(parameterization_name),
      firstuvimage(firstuvimage),
      numuvimages(numuvimages),
      images(numuvimages,nullptr)
            
    // takes ownership of the specifed range of the images array in geom.geom
    {
      
      destroyed=false;


    }

    void set_image(uv_image *image)
    // takes ownership of image
    {
      snde_index index=image->imageidx;
      assert(index < numuvimages);
      images[index] = image;
    }

    void free()
    {
      assert(images.size()==numuvimages);
      for (snde_index cnt=0; cnt < numuvimages;cnt++) {
	delete images[cnt];
	images[cnt]=nullptr;
      }
      if (firstuvimage != SNDE_INDEX_INVALID) {
	geom->manager->free((void **)&geom->geom.uv_images,firstuvimage); //,numuvpatches);
	firstuvimage=SNDE_INDEX_INVALID;	
      }
      destroyed=true;
    }
    
    ~uv_images()
#if !defined(_MSC_VER) || _MSC_VER > 1800 // except for MSVC2013 and earlier
    noexcept(false)
#endif
    {
      if (!destroyed) {
	throw std::runtime_error("Should call free() method of uv_patches object before it goes out of scope and the destructor is called");
      }
    }

    
  };

  

  /* ***!!! Must keep sync'd with geometry.i */
#define SNDE_UV_GEOM_UVS (1ull<<0)
#define SNDE_UV_GEOM_UV_TOPOS (1ull<<1)
#define SNDE_UV_GEOM_UV_TOPO_INDICES (1ull<<2)
#define SNDE_UV_GEOM_UV_TRIANGLES (1ull<<3)
#define SNDE_UV_GEOM_INPLANE2UVCOORDS (1ull<<4)
#define SNDE_UV_GEOM_UVCOORDS2INPLANE (1ull<<5)
#define SNDE_UV_GEOM_UV_EDGES (1ull<<6)
#define SNDE_UV_GEOM_UV_VERTICES (1ull<<7)
#define SNDE_UV_GEOM_UV_VERTEX_EDGELIST_INDICES (1ull<<8)
#define SNDE_UV_GEOM_UV_VERTEX_EDGELIST (1ull<<9)
#define SNDE_UV_GEOM_UV_BOXES (1ull<<10)
#define SNDE_UV_GEOM_UV_BOXCOORD (1ull<<11)
#define SNDE_UV_GEOM_UV_BOXPOLYS (1ull<<12)

#define SNDE_UV_GEOM_ALL ((1ull<<13)-1)

// Resizing masks -- mark those arrays that resize together
#define SNDE_UV_GEOM_UVS_RESIZE (SNDE_UV_GEOM_UVS)
#define SNDE_UV_GEOM_UV_TOPOS_RESIZE (SNDE_UV_GEOM_UV_TOPOS)
#define SNDE_UV_GEOM_UV_TOPO_INDICES_RESIZE (SNDE_UV_GEOM_UV_TOPO_INDICES)
#define SNDE_UV_GEOM_UV_TRIANGLES_RESIZE (SNDE_UV_GEOM_UV_TRIANGLES|SNDE_UV_GEOM_INPLANE2UVCOORDS|SNDE_UV_GEOM_UVCOORDS2INPLANE)
#define SNDE_UV_GEOM_UV_EDGES_RESIZE (SNDE_UV_GEOM_UV_EDGES)
#define SNDE_UV_GEOM_UV_VERTICES_RESIZE (SNDE_UV_GEOM_UV_VERTICES|SNDE_UV_GEOM_UV_VERTEX_EDGELIST_INDICES)
#define SNDE_UV_GEOM_UV_VERTEX_EDGELIST_RESIZE (SNDE_UV_GEOM_UV_VERTEX_EDGELIST)
#define SNDE_UV_GEOM_UV_BOXES_RESIZE (SNDE_UV_GEOM_UV_BOXES|SNDE_UV_GEOM_UV_BOXCOORD)
#define SNDE_UV_GEOM_UV_BOXPOLYS_RESIZE (SNDE_UV_GEOM_UV_BOXPOLYS)


  
typedef uint64_t snde_uv_geom_mask_t;

  class parameterization {
  public:
    std::string name;
    std::shared_ptr<geometry> geom;
    snde_index idx; /* index of the parameterization in the geometry uv database */
    //std::map<std::string,std::shared_ptr<uv_patches>> patches;
    bool destroyed;
    
    /* Should the mesheduv manage the snde_image data for the various uv patches? probably... */

    parameterization(std::shared_ptr<geometry> geom, std::string name,snde_index idx)
    /* WARNING: This constructor takes ownership of the mesheduv and 
       subcomponents from the geometry database and frees them when 
       it is destroyed */
    {
      this->geom=geom;
      this->name=name;
      this->idx=idx;
      destroyed=false;
    }

    //std::shared_ptr<uv_patches> find_patches(std::string name)
    //{
    //  return patches.at(name);
    //}

    //void addpatches(std::shared_ptr<uv_patches> to_add)
    //{
    //  //patches.emplace(std::make_pair<std::string,std::shared_ptr<uv_patches>>(to_add->name,to_add));
    //  patches.emplace(std::pair<std::string,std::shared_ptr<uv_patches>>(to_add->name,to_add));
    //}



    virtual void obtain_lock(std::shared_ptr<lockingprocess> process, snde_uv_geom_mask_t readmask=SNDE_UV_GEOM_ALL, snde_uv_geom_mask_t writemask=0, snde_uv_geom_mask_t resizemask=0)
    {
      /* writemask contains OR'd SNDE_UV_GEOM_xxx bits */
      /* 
	 obtain locks from all our components... 
	 These have to be spawned so they can all obtain in parallel, 
	 following the locking order. 
      */

      /* NOTE: Locking order here must follow order in geometry constructor (above) */
      assert(readmask & SNDE_UV_GEOM_UVS); // Cannot do remainder of locking with out read access to uvs
      
      if (idx != SNDE_INDEX_INVALID) {
	process->get_locks_array_mask((void**)&geom->geom.uvs,SNDE_UV_GEOM_UVS,SNDE_UV_GEOM_UVS_RESIZE,readmask,writemask,resizemask,idx,1);
	
	if (geom->geom.uvs[idx].first_uv_topo != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.uv_topos,SNDE_UV_GEOM_UV_TOPOS,SNDE_UV_GEOM_UV_TOPOS_RESIZE,readmask,writemask,resizemask,geom->geom.uvs[idx].first_uv_topo,geom->geom.uvs[idx].num_uv_topos);
	}

	if (geom->geom.uvs[idx].first_uv_topoidx != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.uv_topo_indices,SNDE_UV_GEOM_UV_TOPO_INDICES,SNDE_UV_GEOM_UV_TOPO_INDICES_RESIZE,readmask,writemask,resizemask,geom->geom.uvs[idx].first_uv_topoidx,geom->geom.uvs[idx].num_uv_topoidxs);
	}

	if (geom->geom.uvs[idx].firstuvtri != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.uv_triangles,SNDE_UV_GEOM_UV_TRIANGLES,SNDE_UV_GEOM_UV_TRIANGLES_RESIZE,readmask,writemask,resizemask,geom->geom.uvs[idx].firstuvtri,geom->geom.uvs[idx].numuvtris);
	  
	  process->get_locks_array_mask((void **)&geom->geom.inplane2uvcoords,SNDE_UV_GEOM_INPLANE2UVCOORDS,SNDE_UV_GEOM_UV_TRIANGLES_RESIZE,readmask,writemask,resizemask,geom->geom.uvs[idx].firstuvtri,geom->geom.uvs[idx].numuvtris);

	  process->get_locks_array_mask((void **)&geom->geom.uvcoords2inplane,SNDE_UV_GEOM_UVCOORDS2INPLANE,SNDE_UV_GEOM_UV_TRIANGLES_RESIZE,readmask,writemask,resizemask,geom->geom.uvs[idx].firstuvtri,geom->geom.uvs[idx].numuvtris);

	  // uv_patch_index
	  
	}
	if (geom->geom.uvs[idx].firstuvedge != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.uv_edges,SNDE_UV_GEOM_UV_EDGES,SNDE_UV_GEOM_UV_EDGES_RESIZE,readmask,writemask,resizemask,geom->geom.uvs[idx].firstuvedge,geom->geom.uvs[idx].numuvedges);
	}

	if (geom->geom.uvs[idx].firstuvvertex != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.uv_vertices,SNDE_UV_GEOM_UV_VERTICES,SNDE_UV_GEOM_UV_VERTICES_RESIZE,readmask,writemask,resizemask,geom->geom.uvs[idx].firstuvvertex,geom->geom.uvs[idx].numuvvertices);
	  process->get_locks_array_mask((void **)&geom->geom.uv_vertex_edgelist_indices,SNDE_UV_GEOM_UV_VERTEX_EDGELIST_INDICES,SNDE_UV_GEOM_UV_VERTICES_RESIZE,readmask,writemask,resizemask,geom->geom.uvs[idx].firstuvvertex,geom->geom.uvs[idx].numuvvertices);
	}

	if (geom->geom.uvs[idx].first_uv_vertex_edgelist != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.uv_vertex_edgelist,SNDE_UV_GEOM_UV_VERTEX_EDGELIST,SNDE_UV_GEOM_UV_VERTEX_EDGELIST_RESIZE,readmask,writemask,resizemask,geom->geom.uvs[idx].first_uv_vertex_edgelist,geom->geom.uvs[idx].num_uv_vertex_edgelist);
	}

	// UV boxes 
	if (geom->geom.uvs[idx].firstuvbox != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.uv_boxes,SNDE_UV_GEOM_UV_BOXES,SNDE_UV_GEOM_UV_BOXES_RESIZE,readmask,writemask,resizemask,geom->geom.uvs[idx].firstuvbox,geom->geom.uvs[idx].numuvboxes);	  
	  process->get_locks_array_mask((void **)&geom->geom.uv_boxcoord,SNDE_UV_GEOM_UV_BOXCOORD,SNDE_UV_GEOM_UV_BOXES_RESIZE,readmask,writemask,resizemask,geom->geom.uvs[idx].firstuvbox,geom->geom.uvs[idx].numuvboxes);	  
	}
	if (geom->geom.uvs[idx].firstuvboxpoly != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.uv_boxpolys,SNDE_UV_GEOM_UV_BOXPOLYS,SNDE_UV_GEOM_UV_BOXPOLYS_RESIZE,readmask,writemask,resizemask,geom->geom.uvs[idx].firstuvboxpoly,geom->geom.uvs[idx].numuvboxpoly);	  
	}
      }
    }
    
    void free()
    {
      /* Free our entries in the geometry database */

      if (idx != SNDE_INDEX_INVALID) {
	if (geom->geom.uvs[idx].first_uv_topo != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.uv_topos,geom->geom.uvs[idx].first_uv_topo); //,geom->geom.mesheduv->numuvtris);
	  geom->geom.uvs[idx].first_uv_topo = SNDE_INDEX_INVALID;	    
	}

	if (geom->geom.uvs[idx].first_uv_topoidx != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.uv_topo_indices,geom->geom.uvs[idx].first_uv_topoidx); //,geom->geom.mesheduv->numuvtris);
	  geom->geom.uvs[idx].first_uv_topoidx = SNDE_INDEX_INVALID;	    
	}

	
	if (geom->geom.uvs[idx].firstuvtri != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.uv_triangles,geom->geom.uvs[idx].firstuvtri); //,geom->geom.mesheduv->numuvtris);
	  geom->geom.uvs[idx].firstuvtri = SNDE_INDEX_INVALID;	    
	}
	
	if (geom->geom.uvs[idx].firstuvedge != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.uv_edges,geom->geom.uvs[idx].firstuvedge);//,geom->geom.mesheduv->numuvedges);
	  geom->geom.uvs[idx].firstuvedge = SNDE_INDEX_INVALID;	    
	}

	if (geom->geom.uvs[idx].firstuvvertex != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.uv_vertices,geom->geom.uvs[idx].firstuvvertex); //,geom->geom.mesheduv->numuvvertices);
	  
	  geom->geom.uvs[idx].firstuvvertex = SNDE_INDEX_INVALID;
	  
	}

	if (geom->geom.uvs[idx].first_uv_vertex_edgelist != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.uv_vertex_edgelist,geom->geom.uvs[idx].first_uv_vertex_edgelist); //,geom->geom.mesheduv->num_uv_vertex_edgelist);
	  geom->geom.uvs[idx].first_uv_vertex_edgelist = SNDE_INDEX_INVALID;
	  
	}

	if (geom->geom.uvs[idx].firstuvbox != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.uv_boxes,geom->geom.uvs[idx].firstuvbox); //,geom->geom.mesheduv->numuvboxes);
	  geom->geom.uvs[idx].firstuvbox = SNDE_INDEX_INVALID; 
	}

	if (geom->geom.uvs[idx].firstuvboxpoly != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.uv_boxpolys,geom->geom.uvs[idx].firstuvboxpoly); //,geom->geom.mesheduv->numuvboxpoly);
	  geom->geom.uvs[idx].firstuvboxpoly = SNDE_INDEX_INVALID;	    
	}


	if (geom->geom.uvs[idx].firstuvboxcoord != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.uv_boxcoord,geom->geom.uvs[idx].firstuvboxcoord); //,geom->geom.mesheduv->numuvboxcoords);
	  geom->geom.uvs[idx].firstuvboxcoord = SNDE_INDEX_INVALID;	    
	}
	
	
	geom->manager->free((void **)&geom->geom.uvs,idx);// ,1);
	idx=SNDE_INDEX_INVALID;
	
	
      }

      //for (auto & name_patches : patches) {
      //name_patches.second->free();
      //}
      destroyed=true;

    }

    ~parameterization()
#if !defined(_MSC_VER) || _MSC_VER > 1800 // except for MSVC2013 and earlier
    noexcept(false)
#endif
    {
      if (!destroyed) {
	throw std::runtime_error("Should call free() method of mesheduv object before it goes out of scope and the destructor is called");
      }
    }
    
  };

  //
  //#define SNDE_PDET_INVALID 0
  //#define SNDE_PDET_INDEX 1
  //#define SNDE_PDET_DOUBLE 2
  //#define SNDE_PDET_STRING 3
  //class paramdictentry {
  //public:
  //  int type; /* see SNDE_PDET_... below */
  //  snde_index indexval;
  //  double doubleval;
  //  std::string stringval;
  //
  // paramdictentry()
  // {
  //   type=SNDE_PDET_INVALID;
  // }
  // 
  //  paramdictentry(snde_index _indexval):  indexval(_indexval)
  //  {
  //    type=SNDE_PDET_INDEX;
  //  }
  //  paramdictentry(double _doubleval):  doubleval(_doubleval)
  //  {
  //    type=SNDE_PDET_DOUBLE;
  //  }
  //  paramdictentry(std::string _stringval): stringval(_stringval)
  //  {
  //    type=SNDE_PDET_STRING;
  //  }
  //
  //  snde_index idx()
  //  {
  //    if (type!=SNDE_PDET_INDEX) {
  //	throw std::runtime_error(std::string("Attempt to extract paramdict entry of type ")+std::to_string(type)+" as type index");
  //    }
  //    return indexval;
  //  }
  //  double dbl()
  //  {
  //    if (type!=SNDE_PDET_DOUBLE) {
  //	throw std::runtime_error(std::string("Attempt to extract paramdict entry of type ")+std::to_string(type)+" as type double");
  //    }
  //    return doubleval;
  //  }
  //  std::string str()
  //  {
  //    if (type!=SNDE_PDET_STRING) {
  //	throw std::runtime_error(std::string("Attempt to extract paramdict entry of type ")+std::to_string(type)+" as type string");
  //    }
  //    return stringval;
  //  }
  //};


  
  /* ***!!! Must keep sync'd with geometry.i */
#define SNDE_COMPONENT_GEOM_PARTS (1ull<<0)
#define SNDE_COMPONENT_GEOM_TOPOS (1ull<<1)
#define SNDE_COMPONENT_GEOM_TOPO_INDICES (1ull<<2)
#define SNDE_COMPONENT_GEOM_TRIS (1ull<<3)
#define SNDE_COMPONENT_GEOM_REFPOINTS (1ull<<4)
#define SNDE_COMPONENT_GEOM_MAXRADIUS (1ull<<5)
#define SNDE_COMPONENT_GEOM_NORMALS (1ull<<6)
#define SNDE_COMPONENT_GEOM_INPLANEMAT (1ull<<7)
#define SNDE_COMPONENT_GEOM_EDGES (1ull<<8)
#define SNDE_COMPONENT_GEOM_VERTICES (1ull<<9)
#define SNDE_COMPONENT_GEOM_PRINCIPAL_CURVATURES (1ull<<10)
#define SNDE_COMPONENT_GEOM_CURVATURE_TANGENT_AXES (1ull<<11)
#define SNDE_COMPONENT_GEOM_VERTEX_EDGELIST_INDICES (1ull<<12)
#define SNDE_COMPONENT_GEOM_VERTEX_EDGELIST (1ull<<13)
#define SNDE_COMPONENT_GEOM_BOXES (1ull<<14)
#define SNDE_COMPONENT_GEOM_BOXCOORD (1ull<<15)
#define SNDE_COMPONENT_GEOM_BOXPOLYS (1ull<<16)

#define SNDE_COMPONENT_GEOM_ALL ((1ull<<17)-1)

// Resizing masks -- mark those arrays that resize together
#define SNDE_COMPONENT_GEOM_PARTS_RESIZE (SNDE_COMPONENT_GEOM_PARTS)
#define SNDE_COMPONENT_GEOM_TOPOS_RESIZE (SNDE_COMPONENT_GEOM_TOPOS)
#define SNDE_COMPONENT_GEOM_TOPO_INDICES_RESIZE (SNDE_COMPONENT_GEOM_TOPO_INDICES)
#define SNDE_COMPONENT_GEOM_TRIS_RESIZE (SNDE_COMPONENT_GEOM_TRIS|SNDE_COMPONENT_GEOM_REFPOINTS|SNDE_COMPONENT_GEOM_MAXRADIUS|SNDE_COMPONENT_GEOM_NORMALS|SNDE_COMPONENT_GEOM_INPLANEMAT)
#define SNDE_COMPONENT_GEOM_EDGES_RESIZE (SNDE_COMPONENT_GEOM_EDGES)
#define SNDE_COMPONENT_GEOM_VERTICES_RESIZE (SNDE_COMPONENT_GEOM_VERTICES|SNDE_COMPONENT_GEOM_PRINCIPAL_CURVATURES|SNDE_COMPONENT_GEOM_CURVATURE_TANGENT_AXES|SNDE_COMPONENT_GEOM_VERTEX_EDGELIST_INDICES)
#define SNDE_COMPONENT_GEOM_VERTEX_EDGELIST_RESIZE (SNDE_COMPONENT_GEOM_VERTEX_EDGELIST)
#define SNDE_COMPONENT_GEOM_BOXES_RESIZE (SNDE_COMPONENT_GEOM_BOXES|SNDE_COMPONENT_GEOM_BOXCOORD)
#define SNDE_COMPONENT_GEOM_BOXPOLYS_RESIZE (SNDE_COMPONENT_GEOM_BOXPOLYS)


  
typedef uint64_t snde_component_geom_mask_t;

  
  class component : public std::enable_shared_from_this<component> { /* abstract base class for geometric components (assemblies, part) */

    // orientation model:
    // Each assembly has orientation
    // orientations of nested assemblys multiply
    // ... apply that product of quaternions to a vector in the part space  ...q1 q2 v q2^-1 q1^-1 for
    // inner assembly orientation q2, to get a vector in the world space.
    // ... apply element by element from inside out, the quaternion, then an offset, to a point in the part space
    // to get a point in the world space
  public:
    std::string name; // used for parameter paths ... form: assemblyname.assemblyname.partname.parameter as paramdict key
    //typedef enum {
    //  subassembly=0,
    //  nurbs=1,
    //  meshed=2,
    //} TYPE;

    //TYPE type;

    //   component();// {}
    
    virtual std::vector<std::pair<snde_partinstance,std::shared_ptr<component>>> get_instances(snde_orientation3 orientation, const std::unordered_map<std::string,std::shared_ptr<uv_images>> & parameterization_data)=0;//,std::shared_ptr<std::unordered_map<std::string,metadatum>> metadata)=0;

    virtual void obtain_lock(std::shared_ptr<lockingprocess> process, snde_component_geom_mask_t readmask=SNDE_COMPONENT_GEOM_ALL,snde_component_geom_mask_t writemask=0,snde_component_geom_mask_t resizemask=0)=0; /* writemask contains OR'd SNDE_COMPONENT_GEOM_xxx bits */

    virtual ~component()
#if !defined(_MSC_VER) || _MSC_VER > 1800 // except for MSVC2013 and earlier
    noexcept(false)
#endif
      ;
  };
  class part : public component {
    part(const part &)=delete; /* copy constructor disabled */
    part& operator=(const part &)=delete; /* copy assignment disabled */
    
  public:
    std::shared_ptr<geometry> geom;
    snde_index idx; // index in the parts geometrydata array
    std::map<std::string,std::shared_ptr<parameterization>> parameterizations; /* NOTE: is a string (URI?) really the proper way to index parameterizations? ... may want to change this */
    bool need_normals; // set if this part was loaded/created without normals being assigned, and therefore still needs normals
    bool destroyed;
    
    /* NOTE: May want to add cache of 
       openscenegraph geodes or drawables representing 
       this part */ 
    
    part(std::shared_ptr<geometry> geom,std::string name,snde_index idx)
    /* WARNING: This constructor takes ownership of the part (if given) and 
       subcomponents from the geometry database and (should) free them when 
       free() is called */
    {
      //this->type=meshed;
      this->geom=geom;
      this->name=name;
      this->idx=idx;
      this->destroyed=false;
    }

    void addparameterization(std::shared_ptr<parameterization> parameterization)
    {
      parameterizations[parameterization->name]=parameterization;
    }
    
    virtual std::vector<std::pair<snde_partinstance,std::shared_ptr<component>>> get_instances(snde_orientation3 orientation, const std::unordered_map<std::string,std::shared_ptr<uv_images>> & parameterization_data) //,std::shared_ptr<std::unordered_map<std::string,metadatum>> metadata)
    {
      struct snde_partinstance ret;
      std::shared_ptr<component> ret_ptr;

      ret_ptr = shared_from_this();

      std::vector<std::pair<snde_partinstance,std::shared_ptr<component>>> ret_vec;

      ret.orientation=orientation;
      //ret.nurbspartnum=SNDE_INDEX_INVALID;
      ret.partnum=idx;

      
      ret.firstuvimage=SNDE_INDEX_INVALID;
      //ret.numuvimages;SNDE_INDEX_INVALID;
      ret.uvnum=SNDE_INDEX_INVALID;
      ret.imgbuf_extra_offset=0;

      if (parameterization_data.find(name) != parameterization_data.end()) {
	std::shared_ptr<uv_images> param_data=parameterization_data.at(name);
	ret.uvnum = parameterizations.at(param_data->parameterization_name)->idx;

	ret.firstuvimage = param_data->firstuvimage;
      }
      /*{
	std::string parameterization_name="";
	
	if (parameterizations.size() > 0) {
	  parameterization_name=parameterizations.begin()->first;
	}
      
	auto pname_entry=metadata->find(name+"."+"parameterization_name");
	if (pname_entry != metadata->end()) {
	  parameterization_name=pname_entry->second.Str(parameterization_name);
	}
	
	auto pname_mesheduv=parameterizations.find(parameterization_name);
	if (pname_mesheduv==parameterizations.end()) {
	  ret.uvnum=SNDE_INDEX_INVALID;
	} else {
	  ret.uvnum=pname_mesheduv->second->idx;

	  // ***!!!!! NEED TO RETHINK HOW THIS WORKS.
	  // REALLY WANT TO IDENTIFY THE TEXTURE DATA FROM
	  // A CHANNEL NAME IN METADATA, THE PARAMETERIZATION
	  // FROM METADATA, AND HAVE SOME KIND OF
	  // CACHE OF TEXTURE DATA -> IMAGE TRANSFORMS
	  auto iname_entry=metadata->find(name+"."+"images");
	  if (iname_entry != metadata->end()) {
	    std::string imagesname = pname_entry->second.Str("");
	    
	    std::shared_ptr<uv_images> images = pname_mesheduv->second->find_images(imagesname);
	    
	    if (!images) {
	      throw std::runtime_error("part::get_instances():  Unknown UV images name: "+patchesname);
	    }
	    ret.firstuvimage=patches->firstuvpatch;
	    //ret.numuvimages=patches->numuvpatches;
	      
	  }
	  
	}
      }*/
      
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

      assert(readmask & SNDE_COMPONENT_GEOM_PARTS); // Cannot do remainder of locking without read access to part

      if (idx != SNDE_INDEX_INVALID) {
	process->get_locks_array_mask((void **)&geom->geom.parts,SNDE_COMPONENT_GEOM_PARTS,SNDE_COMPONENT_GEOM_PARTS_RESIZE,readmask,writemask,resizemask,idx,1);

	if (geom->geom.parts[idx].first_topo != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.topos,SNDE_COMPONENT_GEOM_TOPOS,SNDE_COMPONENT_GEOM_TOPOS_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].first_topo,geom->geom.parts[idx].num_topo);
	}

	if (geom->geom.parts[idx].first_topoidx != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.topo_indices,SNDE_COMPONENT_GEOM_TOPO_INDICES,SNDE_COMPONENT_GEOM_TOPO_INDICES_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].first_topoidx,geom->geom.parts[idx].num_topoidxs);
	}
	
	if (geom->geom.parts[idx].firsttri != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.triangles,SNDE_COMPONENT_GEOM_TRIS,SNDE_COMPONENT_GEOM_TRIS_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].firsttri,geom->geom.parts[idx].numtris);
	  if (geom->geom.refpoints) {
	    process->get_locks_array_mask((void **)&geom->geom.refpoints,SNDE_COMPONENT_GEOM_REFPOINTS,SNDE_COMPONENT_GEOM_TRIS_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].firsttri,geom->geom.parts[idx].numtris);
	  }
	  
	  if (geom->geom.maxradius) {
	    process->get_locks_array_mask((void **)&geom->geom.maxradius,SNDE_COMPONENT_GEOM_MAXRADIUS,SNDE_COMPONENT_GEOM_TRIS_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].firsttri,geom->geom.parts[idx].numtris);
	  }
	  
	  if (geom->geom.normals) {
	    process->get_locks_array_mask((void **)&geom->geom.normals,SNDE_COMPONENT_GEOM_NORMALS,SNDE_COMPONENT_GEOM_TRIS_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].firsttri,geom->geom.parts[idx].numtris);
	  }
	  
	  if (geom->geom.inplanemat) {
	    process->get_locks_array_mask((void **)&geom->geom.inplanemat,SNDE_COMPONENT_GEOM_INPLANEMAT,SNDE_COMPONENT_GEOM_TRIS_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].firsttri,geom->geom.parts[idx].numtris);
	  }
	}
      
	if (geom->geom.parts[idx].firstedge != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.edges,SNDE_COMPONENT_GEOM_EDGES,SNDE_COMPONENT_GEOM_EDGES_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].firstedge,geom->geom.parts[idx].numedges);
	}      
	if (geom->geom.parts[idx].firstvertex != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.vertices,SNDE_COMPONENT_GEOM_VERTICES,SNDE_COMPONENT_GEOM_VERTICES_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].firstvertex,geom->geom.parts[idx].numvertices);
	}

	if (geom->geom.principal_curvatures) {
	  process->get_locks_array_mask((void **)&geom->geom.principal_curvatures,SNDE_COMPONENT_GEOM_PRINCIPAL_CURVATURES,SNDE_COMPONENT_GEOM_VERTICES_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].firstvertex,geom->geom.parts[idx].numvertices);
	    
	}
	
	if (geom->geom.curvature_tangent_axes) {
	  process->get_locks_array_mask((void **)&geom->geom.curvature_tangent_axes,SNDE_COMPONENT_GEOM_CURVATURE_TANGENT_AXES,SNDE_COMPONENT_GEOM_VERTICES_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].firstvertex,geom->geom.parts[idx].numvertices);
	  
	}

	if (geom->geom.vertex_edgelist_indices) {
	  process->get_locks_array_mask((void **)&geom->geom.vertex_edgelist_indices,SNDE_COMPONENT_GEOM_VERTEX_EDGELIST_INDICES,SNDE_COMPONENT_GEOM_VERTICES_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].firstvertex,geom->geom.parts[idx].numvertices);
	}
	
      }
      
      
      if (geom->geom.parts[idx].first_vertex_edgelist != SNDE_INDEX_INVALID) {
	process->get_locks_array_mask((void **)&geom->geom.vertex_edgelist,SNDE_COMPONENT_GEOM_VERTEX_EDGELIST,SNDE_COMPONENT_GEOM_VERTEX_EDGELIST_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].first_vertex_edgelist,geom->geom.parts[idx].num_vertex_edgelist);	
      }      
      
      
      if (geom->geom.parts[idx].firstbox != SNDE_INDEX_INVALID) {
	if (geom->geom.boxes) {
	  process->get_locks_array_mask((void **)&geom->geom.boxes,SNDE_COMPONENT_GEOM_BOXES,SNDE_COMPONENT_GEOM_BOXES_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].firstbox,geom->geom.parts[idx].numboxes);
	  
	}
	if (geom->geom.boxcoord) {
	  process->get_locks_array_mask((void **)&geom->geom.boxcoord,SNDE_COMPONENT_GEOM_BOXCOORD,SNDE_COMPONENT_GEOM_BOXES_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].firstbox,geom->geom.parts[idx].numboxes);
	}
      }
            
      if (geom->geom.boxpolys && geom->geom.parts[idx].firstboxpoly != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.boxpolys,SNDE_COMPONENT_GEOM_BOXPOLYS,SNDE_COMPONENT_GEOM_BOXPOLYS_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].firstboxpoly,geom->geom.parts[idx].numboxpolys);
      } 
      
      
      
    }



    

    void free() /* You must be certain that nothing could be using this part's database entries prior to free() */
    {
      /* Free our entries in the geometry database */
      if (idx != SNDE_INDEX_INVALID) {
	if (geom->geom.parts[idx].firstboxpoly != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.boxpolys,geom->geom.parts[idx].firstboxpoly); // ,geom->geom.parts[idx].numboxpolys);
	  geom->geom.parts[idx].firstboxpoly = SNDE_INDEX_INVALID;
	}

	
	if (geom->geom.parts[idx].firstbox != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.boxes,geom->geom.parts[idx].firstbox); //,geom->geom.parts[idx].numboxes);
	  geom->geom.parts[idx].firstbox = SNDE_INDEX_INVALID;
	}
	
	if (geom->geom.parts[idx].first_vertex_edgelist != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.vertex_edgelist,geom->geom.parts[idx].first_vertex_edgelist); //,geom->geom.parts[idx].num_vertex_edgelist);
	  geom->geom.parts[idx].first_vertex_edgelist = SNDE_INDEX_INVALID;
	}

	
	if (geom->geom.parts[idx].firstvertex != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.vertices,geom->geom.parts[idx].firstvertex); //,geom->geom.parts[idx].numvertices);
	  geom->geom.parts[idx].firstvertex = SNDE_INDEX_INVALID;
	}

	if (geom->geom.parts[idx].firstedge != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.edges,geom->geom.parts[idx].firstedge); //,geom->geom.parts[idx].numedges);
	  geom->geom.parts[idx].firstedge = SNDE_INDEX_INVALID;
	}

	
	if (geom->geom.parts[idx].firsttri != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.triangles,geom->geom.parts[idx].firsttri); //,geom->geom.parts[idx].numtris);
	  geom->geom.parts[idx].firsttri = SNDE_INDEX_INVALID;
	}
	
	if (geom->geom.parts[idx].first_topoidx != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.topo_indices,geom->geom.parts[idx].first_topoidx); //,geom->geom.parts[idx].num_topoidxs);
	  geom->geom.parts[idx].first_topoidx = SNDE_INDEX_INVALID;
	  
	}

	if (geom->geom.parts[idx].first_topo != SNDE_INDEX_INVALID) {
	  geom->manager->free((void **)&geom->geom.topos,geom->geom.parts[idx].first_topo); //,geom->geom.parts[idx].num_topos);
	  geom->geom.parts[idx].first_topo = SNDE_INDEX_INVALID;
	  
	}

	geom->manager->free((void **)&geom->geom.parts,idx); //,1);
	idx=SNDE_INDEX_INVALID;
      }
      destroyed=true;
    }
    
    ~part()
#if !defined(_MSC_VER) || _MSC_VER > 1800 // except for MSVC2013 and earlier
    noexcept(false)
#endif
    {
      if (!destroyed) {
	throw std::runtime_error("Should call free() method of part object before it goes out of scope and the destructor is called");
      }
    }

  };
  

  class assembly : public component {
    /* NOTE: Unlike other types of component, assemblies ARE copyable/assignable */
    /* (this is because they don't have a representation in the underlying
       geometry database) */
  public:
    // NOTE: Name of a part/subassembly inside the assembly
    // may not match global name. This is so an assembly
    // can include multiple copies of a part, but they
    // can still have unique names
    std::map<std::string,std::shared_ptr<component>> pieces;
    snde_orientation3 _orientation; /* orientation of this part/assembly relative to its parent */

    /* NOTE: May want to add cache of 
       openscenegraph group nodes representing 
       this assembly  */ 

    assembly(std::string name,snde_orientation3 orientation)
    {
      this->name=name;
      //this->type=subassembly;
      this->_orientation=orientation;
      
    }

    virtual snde_orientation3 orientation(void)
    {
      return _orientation;
    }


    virtual std::vector<std::pair<snde_partinstance,std::shared_ptr<component>>> get_instances(snde_orientation3 orientation, const std::unordered_map<std::string,std::shared_ptr<uv_images>> & parameterization_data)  //,std::shared_ptr<std::unordered_map<std::string,metadatum>> metadata)
    {
      std::vector<std::pair<snde_partinstance,std::shared_ptr<component>>> instances;
      std::unordered_map<std::string,std::shared_ptr<uv_images>> reducedparamdata;

      std::string name_with_dot=name+".";
      
      for (auto & name_images : parameterization_data) {
	if (!strncmp(name_images.first.c_str(),name_with_dot.c_str(),name_with_dot.size())) {
	  /* this paramdict entry name stats with this assembly name  + '.' */
	  
	  /* give same entry to reducedparamdict, but with assembly name and dot stripped */
	  reducedparamdata[std::string(name_images.first.c_str()+name_with_dot.size())]=name_images.second;
	}
      }
      
      
      snde_orientation3 neworientation=orientation_orientation_multiply(orientation,_orientation);
      for (auto & piece : pieces) {
	std::vector<std::pair<snde_partinstance,std::shared_ptr<component>>> newpieces=piece.second->get_instances(neworientation,reducedparamdata);
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
	std::shared_ptr<component> pieceptr=piece->second;
	process->spawn([ pieceptr,process,readmask,writemask,resizemask ]() { pieceptr->obtain_lock(process,readmask,writemask,resizemask); } );
	
      }
      
    }
    virtual ~assembly()
    {
      
    }


    static std::tuple<std::shared_ptr<assembly>,std::unordered_map<std::string,metadatum>> from_partlist(std::string name,std::shared_ptr<std::vector<std::pair<std::shared_ptr<part>,std::unordered_map<std::string,metadatum>>>> parts)
    {
      
      std::shared_ptr<assembly> assem=std::make_shared<assembly>(name,snde_null_orientation3());

      /* ***!!!! Should we make sure that part names are unique? */
      std::unordered_map<std::string,metadatum> metadata;
      
      for (size_t cnt=0; cnt < parts->size();cnt++) {

	std::string postfix=std::string("");
	std::string partname;
	do {
	  partname=(*parts)[cnt].first->name+postfix;
	  postfix += "_"; // add additional trailing underscore
	} while (assem->pieces.find(partname) != assem->pieces.end());
	
	for (auto md: (*parts)[cnt].second) {
	  // prefix part name, perhaps a postfix, and "." onto metadata name
	  metadatum newmd(partname+"."+md.first,md.second);
	  assert(metadata.find(newmd.Name)==metadata.end()); // metadata should not exist already!
	  
	  metadata.emplace(newmd.Name,newmd);
	}
	assem->pieces.emplace(partname,std::static_pointer_cast<component>((*parts)[cnt].first));
      }
      
      return std::make_tuple(assem,metadata);
    }
    
  };


  ///* NOTE: Could have additional abstraction layer to accommodate 
  //   multi-resolution approximations */
  //class nurbspart : public component {
  //  nurbspart(const nurbspart &)=delete; /* copy constructor disabled */
  //  nurbspart& operator=(const nurbspart &)=delete; /* copy assignment disabled */
  //public:
  //  snde_index nurbspartnum;
  //  std::shared_ptr<geometry> geom;
  //
  //  nurbspart(std::shared_ptr<geometry> geom,std::string name,snde_index nurbspartnum)
  //  /* WARNING: This constructor takes ownership of the part and 
  //    subcomponents from the geometry database and (should) free them when 
  //    it is destroyed */
  //  {
  //    this->type=nurbs;
  //    this->geom=geom;
  //   this->name=name;
  //   //this->orientation=geom->geom.nurbsparts[nurbspartnum].orientation;
  //   this->nurbspartnum=nurbspartnum;
  // }
    
  // virtual void obtain_lock(std::shared_ptr<lockingprocess> process, snde_component_geom_mask_t readmask=SNDE_COMPONENT_GEOM_ALL,snde_component_geom_mask_t writemask=0,snde_component_geom_mask_t resizemask=0)
  // {
  //   /* writemask contains OR'd SNDE_COMPONENT_GEOM_xxx bits */
  //
  //    assert(0); /* not yet implemented */
  //   
  //  } 
  //  virtual ~nurbspart()
  //  {
  //   assert(0); /* not yet implemented */
  //  }
  //  
  //};


  
  
}


#endif /* SNDE_GEOMETRY_HPP */
