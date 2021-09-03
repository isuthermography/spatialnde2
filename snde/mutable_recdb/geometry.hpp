#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <map>
#include <tuple>
#include <string>
#include <atomic>

#include "snde/geometry_types.h"
#include "snde/geometrydata.h"
#include "snde/quaternion.h"
#include "snde/metadata.hpp"
#include "snde/infostore_or_component.hpp"

#include "snde/revision_manager.hpp"
#include "snde/normal_calculation.hpp"
#include "snde/inplanemat_calculation.hpp"
#include "snde/boxes_calculation.hpp"
#include "snde/projinfo_calculation.hpp"

#include "snde/arraymanager.hpp"
#include "snde/recdb_paths.hpp"

#include "snde/stringtools.hpp"



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


  class component; // Forward declaration
  class assembly; // Forward declaration
  class part; // Forward declaration
  class geometry_function; // Forward declaration
  // class nurbspart; // Forward declaration
  class mutableinfostore;
  class mutabledatastore;
  class mutableparameterizationstore;
  class mutablerecdb;
  
  class geometry {
  public:
    struct snde_geometrydata geom;

    std::shared_ptr<arraymanager> manager;
    // manager's lock manager handles all of the data arrays inside geom.
    // the manager pointer itself may not be changed once instantiated so it doesn't need locking.
    
    /* All arrays allocated by a particular allocator are locked together by manager->locker */
 

    // OBSOLETE: 
    //std::shared_ptr<rwlock> lock; // This is the object_trees_lock. In the locking order this PRECEDES all of the components. If you have this locked for read, then NONE of the object trees may be modified. If you have this locked for write then you may modify the object tree component lists/pointers INSIDE COMPONENTS THAT ARE ALSO WRITE-LOCKED... Corresponds to SNDE_INFOSTORE_OBJECT_TREES


    
    geometry(double tol,std::shared_ptr<arraymanager> manager)
    //lock(std::make_shared<rwlock>())
    {
      memset(&geom,0,sizeof(geom)); // reset everything to NULL
      this->manager=manager;
      geom.tol=tol;

      //manager->add_allocated_array((void **)&geom.assemblies,sizeof(*geom.assemblies),0);
      manager->add_allocated_array((void **)&geom.parts,sizeof(*geom.parts),0);

      manager->add_allocated_array((void **)&geom.topos,sizeof(*geom.topos),0);
      manager->add_allocated_array((void **)&geom.topo_indices,sizeof(*geom.topo_indices),0);

      

      std::set<snde_index> triangles_elemsizes;

      triangles_elemsizes.insert(sizeof(*geom.triangles));
      triangles_elemsizes.insert(sizeof(*geom.refpoints));
      triangles_elemsizes.insert(sizeof(*geom.maxradius));
      triangles_elemsizes.insert(sizeof(*geom.vertnormals));
      triangles_elemsizes.insert(sizeof(*geom.trinormals));
      triangles_elemsizes.insert(sizeof(*geom.inplanemats));

      manager->add_allocated_array((void **)&geom.triangles,sizeof(*geom.triangles),0,triangles_elemsizes);
      manager->add_follower_array((void **)&geom.triangles,(void **)&geom.refpoints,sizeof(*geom.refpoints));
      manager->add_follower_array((void **)&geom.triangles,(void **)&geom.maxradius,sizeof(*geom.maxradius));
      manager->add_follower_array((void **)&geom.triangles,(void **)&geom.vertnormals,sizeof(*geom.vertnormals));
      manager->add_follower_array((void **)&geom.triangles,(void **)&geom.trinormals,sizeof(*geom.trinormals));
      manager->add_follower_array((void **)&geom.triangles,(void **)&geom.inplanemats,sizeof(*geom.inplanemats));

      manager->add_allocated_array((void **)&geom.edges,sizeof(*geom.edges),0);


      std::set<snde_index> vertices_elemsizes;
      vertices_elemsizes.insert(sizeof(*geom.vertices));
      vertices_elemsizes.insert(sizeof(*geom.principal_curvatures));
      vertices_elemsizes.insert(sizeof(*geom.curvature_tangent_axes));
      vertices_elemsizes.insert(sizeof(*geom.vertex_edgelist_indices));
      manager->add_allocated_array((void **)&geom.vertices,sizeof(*geom.vertices),0,vertices_elemsizes);
      
      manager->add_follower_array((void **)&geom.vertices,(void **)&geom.principal_curvatures,sizeof(*geom.principal_curvatures));
      manager->add_follower_array((void **)&geom.vertices,(void **)&geom.curvature_tangent_axes,sizeof(*geom.curvature_tangent_axes));

      manager->add_follower_array((void **)&geom.vertices,(void **)&geom.vertex_edgelist_indices,sizeof(*geom.vertex_edgelist_indices));
      manager->add_allocated_array((void **)&geom.vertex_edgelist,sizeof(*geom.vertex_edgelist),0);
      

      
      std::set<snde_index> boxes_elemsizes;
      boxes_elemsizes.insert(sizeof(*geom.boxes));
      boxes_elemsizes.insert(sizeof(*geom.boxcoord));
      
      manager->add_allocated_array((void **)&geom.boxes,sizeof(*geom.boxes),0,boxes_elemsizes);
      manager->add_follower_array((void **)&geom.boxes,(void **)&geom.boxcoord,sizeof(*geom.boxcoord));
      
      manager->add_allocated_array((void **)&geom.boxpolys,sizeof(*geom.boxpolys),0);


      
      /* parameterization */
      manager->add_allocated_array((void **)&geom.uvs,sizeof(*geom.uvs),0);
      manager->add_allocated_array((void **)&geom.uv_patches,sizeof(*geom.uv_patches),0);
      manager->add_allocated_array((void **)&geom.uv_topos,sizeof(*geom.uv_topos),0);
      manager->add_allocated_array((void **)&geom.uv_topo_indices,sizeof(*geom.uv_topo_indices),0);
      
      std::set<snde_index> uv_triangles_elemsizes;

      uv_triangles_elemsizes.insert(sizeof(*geom.uv_triangles));
      uv_triangles_elemsizes.insert(sizeof(*geom.inplane2uvcoords));
      uv_triangles_elemsizes.insert(sizeof(*geom.uvcoords2inplane));

      manager->add_allocated_array((void **)&geom.uv_triangles,sizeof(*geom.uv_triangles),0,uv_triangles_elemsizes);
      manager->add_follower_array((void **)&geom.uv_triangles,(void **)&geom.inplane2uvcoords,sizeof(*geom.inplane2uvcoords));
      manager->add_follower_array((void **)&geom.uv_triangles,(void **)&geom.uvcoords2inplane,sizeof(*geom.uvcoords2inplane));
      //manager->add_follower_array((void **)&geom.uv_triangles,(void **)&geom.uv_patch_index,sizeof(*geom.uv_patch_index));

      manager->add_allocated_array((void **)&geom.uv_edges,sizeof(*geom.uv_edges),0);

      std::set<snde_index> uv_vertices_elemsizes;

      uv_vertices_elemsizes.insert(sizeof(*geom.uv_vertices));
      uv_vertices_elemsizes.insert(sizeof(*geom.uv_vertex_edgelist_indices));
      manager->add_allocated_array((void **)&geom.uv_vertices,sizeof(*geom.uv_vertices),0,uv_vertices_elemsizes);
      manager->add_follower_array((void **)&geom.uv_vertices,(void **)&geom.uv_vertex_edgelist_indices,sizeof(*geom.uv_vertex_edgelist_indices));

      manager->add_allocated_array((void **)&geom.uv_vertex_edgelist,sizeof(*geom.uv_vertex_edgelist),0);


      // ***!!! insert NURBS here !!!***

      std::set<snde_index> uv_boxes_elemsizes;
      uv_boxes_elemsizes.insert(sizeof(*geom.uv_boxes));
      uv_boxes_elemsizes.insert(sizeof(*geom.uv_boxcoord));
      manager->add_allocated_array((void **)&geom.uv_boxes,sizeof(*geom.uv_boxes),0,uv_boxes_elemsizes);
      manager->add_follower_array((void **)&geom.uv_boxes,(void **)&geom.uv_boxcoord,sizeof(*geom.uv_boxcoord));
      
      manager->add_allocated_array((void **)&geom.uv_boxpolys,sizeof(*geom.uv_boxpolys),0);

      //manager->add_allocated_array((void **)&geom.uv_images,sizeof(*geom.uv_images),0);


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


  static std::shared_ptr<immutable_metadata> reduce_partspecific_metadata(std::shared_ptr<immutable_metadata> metadata,std::string component_name)
  {

    if (!metadata) {
      metadata=std::make_shared<immutable_metadata>();
    }
				   
    std::shared_ptr<immutable_metadata> reduced_metadata=std::make_shared<immutable_metadata>(metadata->metadata); // not immutable while we are constructing it
      
    
    std::unordered_map<std::string,metadatum>::iterator next_name_metadatum;
    
    std::string name_with_dot=component_name+".";
    for (auto name_metadatum=reduced_metadata->metadata.begin();name_metadatum != reduced_metadata->metadata.end();name_metadatum=next_name_metadatum) {
      next_name_metadatum=name_metadatum;
      next_name_metadatum++;
      fprintf(stderr,"reduce_partspecific_metadata: got %s; name_with_dot=%s\n",name_metadatum->first.c_str(),name_with_dot.c_str());
      
      if (!strncmp(name_metadatum->first.c_str(),name_with_dot.c_str(),name_with_dot.size())) {
	/* this metadata entry name starts with this component name  + '.' */
	metadatum temp_copy=name_metadatum->second;
	reduced_metadata->metadata.erase(name_metadatum);
	
	// *** I believe since we are erasing before we are adding, a rehash should not be possible here (proscribed by the spec: https://stackoverflow.com/questions/13730470/how-do-i-prevent-rehashing-of-an-stdunordered-map-while-removing-elements) 
	// so we are OK and our iterators will remain valid
	
	/* give same entry to reduced_metadata, but with assembly name and dot stripped */
	temp_copy.Name = std::string(temp_copy.Name.c_str()+name_with_dot.size());
	reduced_metadata->metadata.emplace(temp_copy.Name,temp_copy);
	
      }
    }

    return reduced_metadata;
  }


  
  // ***!!! NOTE: class uv_images is not currently used.
  // it was intended for use when a single part/assembly pulls in
  // parameterization data (texture) from multiple other channels.
  // but the renderer does not (yet) support this. 
  
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

    //void set_image(snde_image *image)
    //// copies provided image struct
    //{
    //  snde_index index=image->imageidx;
    //  assert(index < numuvimages);
    //  images[index] = image;
    //}

    void free()
    {
      assert(images.size()==numuvimages);
      for (snde_index cnt=0; cnt < numuvimages;cnt++) {
	delete images[cnt];
	images[cnt]=nullptr;
      }
      if (firstuvimage != SNDE_INDEX_INVALID) {
	//geom->manager->free((void **)&geom->geom.uv_images,firstuvimage); //,numuvpatches);
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
	throw std::runtime_error("Should call free() method of uv_images object before it goes out of scope and the destructor is called");
      }
    }

    
  };

  


  // image_data abstraction is used both for rendering (to get texture) and
  // for image projection (to get projection buffer)
  class image_data {
    // abstract base class for rendering data corresponding to an snde_image
  public:
    image_data() {}
    image_data(const image_data &)=delete; // no copy constructor
    image_data & operator=(const image_data &)=delete; // no copy assignment

    // get_texture_image returns read-only copy
    virtual std::shared_ptr<snde_image> get_texture_image() {return nullptr;}
    
    virtual ~image_data() {}
  };


  class parameterization_patch: public std::enable_shared_from_this<parameterization_patch> {
  public:
    snde_index patchnum;
    std::shared_ptr<trm_dependency> boxes_function;
    std::shared_ptr<trm_dependency> projinfo_function;

    parameterization_patch(snde_index patchnum) :
      patchnum(patchnum)
    {
      
    }
    
    std::shared_ptr<trm_dependency> request_boxes(std::shared_ptr<mutablerecdb> recdb,std::string recdb_context,std::string recname,std::shared_ptr<geometry> geom,std::shared_ptr<parameterization> param,std::shared_ptr<trm> revman,cl_context context, cl_device_id device, cl_command_queue queue)
    {
      // must be called during a transaction!
      if (!boxes_function) {
	boxes_function = boxes_calculation_2d(recdb,recdb_context,recname,geom,revman,param,patchnum,context,device,queue);
      }
      return boxes_function;
    }


    
  };
  

  class parameterization : public lockable_infostore_or_component {
    // NOTE: everything in parameterization OR SUBCLASS that is required to explore the object graph (nothing, really, at the moment)
    // MUST be readable lockless in a thread-safe fashion.. that means consts or atomic shared pointers to consts that can't otherwise be modified.
    
  public:
    // NOTE: A parameterization cannot be shared among multiple parts,
    // because the meaning is dependent on the part geometry (see projinfo dependence on 3D geometry)
    class notifier {
    public:
      virtual void modified(std::shared_ptr<parameterization> param)=0;
      
    };

    std::shared_ptr<geometry> geom;
    snde_index idx; /* index of the parameterization in the geometry uv database -- we have ownership of this entry */
    snde_index numuvimages; // number of uv image "patches" IMMUTABLE AND MUST MATCH snde_parameterization DATA STRUCTURE
    std::vector<std::shared_ptr<parameterization_patch>> patches;
    
    std::shared_ptr<rwlock> lock; // managed by lockmanager

    std::shared_ptr<trm_dependency> projinfo_function; // TRM dependency that calculates projection info for this parameterization... null until request_projinfo() called. 
    
    std::set<std::weak_ptr<notifier>,std::owner_less<std::weak_ptr<notifier>>> notifiers; 
    

    bool destroyed;
    
    /* Should the mesheduv manage the snde_image data for the various uv patches? probably... */

    parameterization(std::shared_ptr<geometry> geom,snde_index idx,snde_index numuvimages) :
      lockable_infostore_or_component(SNDE_INFOSTORE_PARAMETERIZATIONS)
    /* WARNING: This constructor takes ownership of the snde_parameterization and 
       subcomponents from the geometry database and frees them when 
       it is destroyed */
    {
      this->geom=geom;
      this->idx=idx;
      this->numuvimages=numuvimages;
      this->lock=std::make_shared<rwlock>();
      destroyed=false;

      for (unsigned cnt=0; cnt < numuvimages; cnt++) {
	patches.push_back(std::make_shared<parameterization_patch>(cnt));
      }
    }


    virtual std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>
    explore_component_get_instances(std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> &component_set,
				     std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,
				     snde_orientation3 orientation,
				     std::shared_ptr<immutable_metadata> metadata,
				    std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names)> get_uv_imagedata)
    {
      std::shared_ptr<parameterization> our_ptr=std::dynamic_pointer_cast<parameterization>(shared_from_this());
      
      component_set.emplace(our_ptr);

      fprintf(stderr,"Explored parameterization\n");
      
      return std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>();
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

    std::vector<std::shared_ptr<trm_dependency>> request_boxes(std::shared_ptr<mutablerecdb> recdb,std::string recdb_context,std::string recname,std::shared_ptr<trm> revman,cl_context context, cl_device_id device, cl_command_queue queue)
    {
      // must be called during a transaction!
      std::vector<std::shared_ptr<trm_dependency>> ret;
      
      for (size_t cnt=0;cnt < patches.size();cnt++) {
	ret.push_back(patches.at(cnt)->request_boxes(recdb,recdb_context,recname,geom,std::dynamic_pointer_cast<parameterization>(shared_from_this()),revman,context,device,queue));
      }
      return ret;
    }


    std::shared_ptr<trm_dependency> request_projinfo(std::shared_ptr<mutablerecdb> recdb,std::string recdb_context,std::string recname,std::shared_ptr<part> partobj,std::shared_ptr<trm> revman,cl_context context, cl_device_id device, cl_command_queue queue)
    {
      // partobj must be the (unique!) part that this parameterization corresponds to. It need not be locked. 
      
      // must be called during a transaction!
      if (!projinfo_function) {
	projinfo_function = projinfo_calculation(recdb,recdb_context,recname,geom,revman,partobj,std::dynamic_pointer_cast<parameterization>(shared_from_this()),context,device,queue);
      }
      return projinfo_function;
    }


    
    virtual void add_notifier(std::shared_ptr<notifier> notify)
    {
      notifiers.emplace(notify);
      
    }
    
    /* virtual rwlock_token_set obtain_lock(std::shared_ptr<lockingprocess> process,std::shared_ptr<iterablerecrefs> recdb_reclist=null,std::string recdb_context="",snde_infostore_lock_mask_t readmask=SNDE_INFOSTORE_PARAMETERIZATIONS,snde_infostore_lock_mask_t writemask=0,temporary=false) // readmask/writemask contains OR'd SNDE_INFOSTORE_xxx bits 
    {
      // lock this parameterization
      // Assumes this is either the only parameterization being locked or the caller is taking care of the locking order
      
      std::shared_ptr<parameterization> our_ptr=shared_from_this();
      
      
      process->get_locks_lockable_mask(our_ptr,SNDE_INFOSTORE_PARAMETERIZATIONS,readmask,writemask);
      // ***!!! Need to support temporary flag and return meaningful value!!!1
    }
    
    */
    virtual void obtain_geom_lock(std::shared_ptr<lockingprocess> process, std::shared_ptr<iterablerecrefs> recdb_reclist=nullptr,std::string recdb_context="/",snde_infostore_lock_mask_t readmask=SNDE_COMPONENT_GEOM_ALL,snde_infostore_lock_mask_t writemask=0,snde_infostore_lock_mask_t resizemask=0) /* writemask contains OR'd SNDE_COMPONENT_GEOM_xxx bits */
    {
      // Nothing to do here as we only have parameterization (uv), not geometry 
    }
    
    virtual void obtain_uv_lock(std::shared_ptr<lockingprocess> process, std::shared_ptr<iterablerecrefs> recdb_reclist=nullptr,std::string recdb_context="/",snde_infostore_lock_mask_t readmask=SNDE_UV_GEOM_ALL, snde_infostore_lock_mask_t writemask=0, snde_infostore_lock_mask_t resizemask=0)
    // NOTE: obtain_uv_lock() is parallel in functionality to obtain_geom_lock()
    // (i.e. locks underlying UV geometry data structures), NOT component obtain_lock(), which just locks C++ structures.
    // ... Note that mutablegeomstore::obtain_lock DOES lock underlying regular geometry if requested but NOT uv... (at least not yet)

      
    {
      /* writemask contains OR'd SNDE_UV_GEOM_xxx bits */
      /* 
	 obtain locks from all our components... 
	 These have to be spawned so they can all obtain in parallel, 
	 following the locking order. 
      */

      snde_index patchnum;
      
      /* NOTE: Locking order here must follow order in geometry constructor (above) */
      assert(readmask & SNDE_UV_GEOM_UVS || writemask & SNDE_UV_GEOM_UVS); // Cannot do remainder of locking with out read access to uvs
      
      if (idx != SNDE_INDEX_INVALID) {
	process->get_locks_array_mask((void**)&geom->geom.uvs,SNDE_UV_GEOM_UVS,SNDE_UV_GEOM_UVS_RESIZE,readmask,writemask,resizemask,idx,1);
	
	if (geom->geom.uvs[idx].firstuvpatch != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.uv_patches,SNDE_UV_GEOM_UV_PATCHES,SNDE_UV_GEOM_UV_PATCHES_RESIZE,readmask,writemask,resizemask,geom->geom.uvs[idx].firstuvpatch,geom->geom.uvs[idx].numuvimages);
	}

	if (geom->geom.uvs[idx].first_uv_topo != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.uv_topos,SNDE_UV_GEOM_UV_TOPOS,SNDE_UV_GEOM_UV_TOPOS_RESIZE,readmask,writemask,resizemask,geom->geom.uvs[idx].first_uv_topo,geom->geom.uvs[idx].num_uv_topos);
	}

	if (geom->geom.uvs[idx].first_uv_topoidx != SNDE_INDEX_INVALID) {
	  process->get_locks_array_mask((void **)&geom->geom.uv_topo_indices,SNDE_UV_GEOM_UV_TOPO_INDICES,SNDE_UV_GEOM_UV_TOPO_INDICES_RESIZE,readmask,writemask,resizemask,geom->geom.uvs[idx].first_uv_topoidx,geom->geom.uvs[idx].num_uv_topoidxs);
	}

	if (geom->geom.uvs[idx].firstuvtri != SNDE_INDEX_INVALID) {
	  fprintf(stderr,"Attempting to lock UV_TRIANGLES\n");
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
	if (readmask & SNDE_UV_GEOM_UV_BOXES || readmask & SNDE_UV_GEOM_UV_BOXCOORD || readmask & SNDE_UV_GEOM_UV_BOXPOLYS ||
	    writemask & SNDE_UV_GEOM_UV_BOXES || writemask & SNDE_UV_GEOM_UV_BOXCOORD || writemask & SNDE_UV_GEOM_UV_BOXPOLYS) {

	  assert(readmask & SNDE_UV_GEOM_UV_PATCHES || writemask & SNDE_UV_GEOM_UV_PATCHES);
	  assert(geom->geom.uvs[idx].firstuvpatch != SNDE_INDEX_INVALID);
	  
	  for (patchnum=0; patchnum < geom->geom.uvs[idx].numuvimages;patchnum++) {
	    if (geom->geom.uv_patches[geom->geom.uvs[idx].firstuvpatch+patchnum].firstuvbox != SNDE_INDEX_INVALID) {
	      process->get_locks_array_mask((void **)&geom->geom.uv_boxes,SNDE_UV_GEOM_UV_BOXES,SNDE_UV_GEOM_UV_BOXES_RESIZE,readmask,writemask,resizemask,geom->geom.uv_patches[geom->geom.uvs[idx].firstuvpatch+patchnum].firstuvbox,geom->geom.uv_patches[geom->geom.uvs[idx].firstuvpatch+patchnum].numuvboxes);	  
	      process->get_locks_array_mask((void **)&geom->geom.uv_boxcoord,SNDE_UV_GEOM_UV_BOXCOORD,SNDE_UV_GEOM_UV_BOXES_RESIZE,readmask,writemask,resizemask,geom->geom.uv_patches[geom->geom.uvs[idx].firstuvpatch+patchnum].firstuvbox,geom->geom.uv_patches[geom->geom.uvs[idx].firstuvpatch+patchnum].numuvboxes);	  
	    }
	    if (geom->geom.uv_patches[geom->geom.uvs[idx].firstuvpatch+patchnum].firstuvboxpoly != SNDE_INDEX_INVALID) {
	      process->get_locks_array_mask((void **)&geom->geom.uv_boxpolys,SNDE_UV_GEOM_UV_BOXPOLYS,SNDE_UV_GEOM_UV_BOXPOLYS_RESIZE,readmask,writemask,resizemask,geom->geom.uv_patches[geom->geom.uvs[idx].firstuvpatch+patchnum].firstuvboxpoly,geom->geom.uv_patches[geom->geom.uvs[idx].firstuvpatch+patchnum].numuvboxpolys);	  
	    }
	  }
	}
      }
    }
    
    void free()
    {
      /* Free our entries in the geometry database */
      snde_index patchnum;

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

	if (geom->geom.uvs[idx].firstuvpatch != SNDE_INDEX_INVALID) {

	  
	  for (patchnum=0; patchnum < geom->geom.uvs[idx].numuvimages;patchnum++) {
	    snde_index patchidx = geom->geom.uvs[idx].firstuvpatch + patchnum;
	    
	    if (geom->geom.uv_patches[patchidx].firstuvbox != SNDE_INDEX_INVALID) {
	      geom->manager->free((void **)&geom->geom.uv_boxes,geom->geom.uv_patches[patchidx].firstuvbox); //,geom->geom.mesheduv->numuvboxes);
	      geom->geom.uv_patches[patchidx].firstuvbox = SNDE_INDEX_INVALID; 
	    }
	    
	    if (geom->geom.uv_patches[patchidx].firstuvboxpoly != SNDE_INDEX_INVALID) {
	      geom->manager->free((void **)&geom->geom.uv_boxpolys,geom->geom.uv_patches[patchidx].firstuvboxpoly); //,geom->geom.mesheduv->numuvboxpolys);
	      geom->geom.uv_patches[patchidx].firstuvboxpoly = SNDE_INDEX_INVALID;	    
	    }
	    
	    
	    //if (geom->geom.uv_patches[patchidx].firstuvboxcoord != SNDE_INDEX_INVALID) {
	    //  geom->manager->free((void **)&geom->geom.uv_boxcoord,geom->geom.uv_patches[patchidx].firstuvboxcoord); //,geom->geom.mesheduv->numuvboxcoords);
	    //  geom->geom.uv_patches[patchidx].firstuvboxcoord = SNDE_INDEX_INVALID;	    
	    //}

	  }
	  
	  geom->manager->free((void **)&geom->geom.uv_patches,geom->geom.uvs[idx].firstuvpatch); //,geom->geom.mesheduv->numuvtris);
	  geom->geom.uvs[idx].firstuvpatch = SNDE_INDEX_INVALID;	    
	}

	
	geom->manager->free((void **)&geom->geom.uvs,idx);// ,1);
	idx=SNDE_INDEX_INVALID;
	
	
      }

      //for (auto & name_patches : patches) {
      //name_patches.second->free();
      //}
      destroyed=true;

    }

    virtual ~parameterization()
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


  

  
  class component : public lockable_infostore_or_component { /* abstract base class for geometric components (assemblies, part) */
    // NOTE: everything in component OR SUBCLASS that is required to explore the object graph
    // MUST be readable lockless in a thread-safe fashion.. that means consts or atomic shared pointers to consts that can't otherwise be modified.

    
    // orientation model:
    // Each assembly has orientation
    // orientations of nested assemblys multiply
    // ... apply that product of quaternions to a vector in the part space  ...q1 q2 v q2^-1 q1^-1 for
    // inner assembly orientation q2, to get a vector in the world space.
    // ... apply element by element from inside out, the quaternion, then an offset, to a point in the part space
    // to get a point in the world space
  public:

    class notifier {
    public:
      virtual void modified(std::shared_ptr<component> comp)=0;
      
    };
    
    //const std::string name; // used for parameter paths ... form: assemblyname.assemblyname.partname.parameter as paramdict key... const so it is thread-safe

    
    //typedef enum {
    //  subassembly=0,
    //  nurbs=1,
    //  meshed=2,
    //} TYPE;

    //TYPE type;

    component() :
      lockable_infostore_or_component(SNDE_INFOSTORE_COMPONENTS)
      //name(name)
    {
      
    }
    //std::shared_ptr<rwlock> lock; // moved to superclass... managed by lockmanager... locks notifiers and other non-const, non-atomic (or atomic for write) elements of subclasses
    std::set<std::weak_ptr<notifier>,std::owner_less<std::weak_ptr<notifier>>> notifiers; 
    

    // If you have the geometry for this component (and subcomponents) locked via obtain_geom_lock() -- usually via superclass obtain_graph_lock(), ,get_instances will return you a vector of instances
    //virtual std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>> get_instances(std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,snde_orientation3 orientation, std::shared_ptr<immutable_metadata> metadata, std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>(std::shared_ptr<part> partdata,std::vector<std::string> parameterization_data_names)> get_param_data)=0;

    // obtain_geom_lock prototype is in superclass
    //virtual void obtain_geom_lock(std::shared_ptr<lockingprocess> process, std::shared_ptr<iterablerecrefs> recdb_recrefs=null,std::string recdb_context="",snde_infostore_lock_mask_t readmask=SNDE_COMPONENT_GEOM_ALL,snde_infostore_lock_mask_t writemask=0,snde_infostore_lock_mask_t resizemask=0)=0; /* writemask contains OR'd SNDE_COMPONENT_GEOM_xxx bits */

    // explore_component_get_instances() defined in superclass infostore_or_component.hpp
    virtual std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>
    explore_component_get_instances(std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> &component_set,
				     std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,
				     snde_orientation3 orientation,
				     std::shared_ptr<immutable_metadata> metadata,
				    std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names)> get_uv_imagedata)=0;

    
    virtual void modified()
    {
      // call this method to indicate that the component was modified

      for (auto notifier_obj: notifiers) {
	std::shared_ptr<notifier> notifier_obj_strong=notifier_obj.lock();
	if (notifier_obj_strong) {
	  notifier_obj_strong->modified(std::dynamic_pointer_cast<component>(shared_from_this()));
	}
      }
    }

    virtual void add_notifier(std::shared_ptr<notifier> notify)
    {
      notifiers.emplace(notify);
      
    }
    
    virtual ~component()
#if !defined(_MSC_VER) || _MSC_VER > 1800 // except for MSVC2013 and earlier
    noexcept(false)
#endif
      ;
  };

  class reccomponent: public component {
    // ****!!!!!! THIS CLASS IS PROBABLY OBSOLETE AND NOT NEEDED ANY MORE now that recinfostores
    // are the primary pointers to geometry anyway.
    
    // A component that is a mutablegeomstore in the same recdb as this geometry 
    // superclass defines:
    //   std::string name
    //   std::shared_ptr<rwlock> lock; 
    const std::string path; // can be absolute (leading /) or relative. constness makes it thread-safe

    reccomponent(std::string path) :
      component(),
      path(path)
    {

    }
    
    //virtual std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>> get_instances(std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,snde_orientation3 orientation, std::shared_ptr<immutable_metadata> metadata, std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>(std::shared_ptr<part> partdata,std::vector<std::string> parameterization_data_names)> get_uv_imagedata); // implementation in geometry.cpp


    virtual void obtain_geom_lock(std::shared_ptr<lockingprocess> process, std::shared_ptr<iterablerecrefs> recdb_reclist=nullptr,std::string recdb_context="/",snde_infostore_lock_mask_t readmask=SNDE_COMPONENT_GEOM_ALL,snde_infostore_lock_mask_t writemask=0,snde_infostore_lock_mask_t resizemask=0); /* writemask contains OR'd SNDE_COMPONENT_GEOM_xxx bits */ // moved to geometry.cpp

    virtual std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>
    explore_component_get_instances(std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> &component_set,
				    std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,
				     snde_orientation3 orientation,
				    std::shared_ptr<immutable_metadata> metadata,
				    std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names)> get_uv_imagedata); // moved to geomemtry.cpp

    /*
    virtual rwlock_token_set obtain_lock(std::shared_ptr<lockingprocess> process,std::shared_ptr<mutablerecdb> recdb=null,std::string recdb_context="",snde_infostore_lock_mask_t readmask=SNDE_INFOSTORE_COMPONENTS,snde_infostore_lock_mask_t writemask=0,temporary=false)
    {
      // attempt to obtain set of component pointers
      // including this component and all sub-components.
      // assumes no other component locks are held. Assumes SNDE_INFOSTORE_OBJECT_TREES is at least temporarily held for at least read,
      
      std::shared_ptr<component> our_ptr=shared_from_this(); 

      std::string full_path=recdb_path_join(recdb_context,path);
      std::shared_ptr<mutablegeomstore> geomstore = std::dynamic_pointer_cast<mutablegeomstore>(recdb->lookup(full_path));
     

      // ***!!!! Need to reorder locking of us vs geomstore according to locking order? ***!!!
      // NO: Need to use _explore_component method ***!!! 
      process->get_locks_lockable_mask(our_ptr,SNDE_INFOSTORE_COMPONENTS,readmask & SNDE_INFOSTORE_ALL,writemask & SNDE_INFOSTORE_ALL);
      if (geomstore) {
	
	geomstore->obtain_lock(process,recdb,full_path,readmask,writemask);
      } else {
	fprintf(stderr,"Warning: reccomponent:obtain_lock: geometry store %s not found\n",(char *)full_path);
      }
    }
    */
  };

  
  class part : public component {

    // NOTE: Part generally locked by holding the lock of its
    // ancestor mutablegeomstore (mutableinfostore)
    // this lock should be held when calling its methods
    
    part(const part &)=delete; /* copy constructor disabled */
    part& operator=(const part &)=delete; /* copy assignment disabled */
    
  public:
    std::shared_ptr<geometry> geom; /* ***!!! really necessary??? (used by addparameterization(), at least) */
    //snde_index idx; // index in the parts geometrydata array
    std::atomic<std::uint64_t> _idx; // atomic, so we can avoid locking the part in most cases. 
    
    //std::shared_ptr<const std::map<std::string,std::shared_ptr<parameterization>>> _parameterizations; /* atomic shared pointer to parameterization map */ /* NOTE: is a string (URI?) really the proper way to index parameterizations? ... may want to change this */

    std::shared_ptr<const std::set<std::string>> _parameterizations; // set of recording names relative to our context
    
    std::shared_ptr<trm_dependency> normal_function;
    std::shared_ptr<trm_dependency> inplanemat_function;
    std::shared_ptr<trm_dependency> curvature_function;

    std::shared_ptr<trm_dependency> boxes_function;

    
    //bool need_normals; // set if this part was loaded/created without normals being assigned, and therefore still needs normals
    bool destroyed;
    
    /* NOTE: May want to add cache of 
       openscenegraph geodes or drawables representing 
       this part */ 
    
    part(std::shared_ptr<geometry> geom,snde_index idx) :
    /* WARNING: This constructor takes ownership of the part (if given) and 
       subcomponents from the geometry database and (should) free them when 
       free() is called */
      geom(geom),
      component(),
      _idx(idx),
      _parameterizations(std::make_shared<std::set<std::string>>()),
      destroyed(false)
    {
    }

    std::shared_ptr<const std::set<std::string>> parameterizations()
    {
      return std::atomic_load(&_parameterizations);
    }

    virtual std::shared_ptr<std::set<std::string>> _begin_atomic_parameterizations_update()
    // part must be locked for write when calling this function
    {
      // Make copy of atomically-guarded data and return mutable copy
      return std::make_shared<std::set<std::string>>(*parameterizations());
    }

    virtual void _end_atomic_parameterizations_update(std::shared_ptr<const std::set<std::string>> new_parameterizations)
    {
      std::atomic_store(&_parameterizations,new_parameterizations);
    }

    snde_index idx()
    {
      return (snde_index)std::atomic_load(&_idx);
    }

    virtual void _atomic_idx_update(snde_index new_idx)
    {
      uint64_t new_idx_uint64(new_idx);
      std::atomic_store(&_idx,new_idx);
      
    }
    
    std::shared_ptr<trm_dependency> request_normals(std::shared_ptr<trm> revman,cl_context context, cl_device_id device, cl_command_queue queue)
    {
      // must be called during a transaction!
      if (!normal_function) {
	normal_function = normal_calculation(geom,revman,std::dynamic_pointer_cast<part>(shared_from_this()),context,device,queue);
      }
      return normal_function;
    }
    
    
    std::shared_ptr<trm_dependency> request_inplanemats(std::shared_ptr<trm> revman,cl_context context, cl_device_id device, cl_command_queue queue)
    {
      // must be called during a transaction!
      if (!inplanemat_function) {
	inplanemat_function = inplanemat_calculation(geom,revman,std::dynamic_pointer_cast<part>(shared_from_this()),context,device,queue);
      }
      return inplanemat_function;
    }

    std::shared_ptr<trm_dependency> request_boxes(std::shared_ptr<trm> revman,cl_context context, cl_device_id device, cl_command_queue queue)
    {
      // must be called during a transaction!
      if (!boxes_function) {
	boxes_function = boxes_calculation_3d(geom,revman,std::dynamic_pointer_cast<part>(shared_from_this()),context,device,queue);
      }
      return boxes_function;
    }

    virtual std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>
    explore_component_get_instances(std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> &component_set,
				    std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,
				    snde_orientation3 orientation,
				    std::shared_ptr<immutable_metadata> metadata,
				    std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names)> get_uv_imagedata); // implementation in geometry.cpp
    /*
    virtual rwlock_token_set obtain_lock(std::shared_ptr<lockingprocess> process,std::shared_ptr<mutablerecdb> recdb=null,std::string recdb_context="",snde_infostore_lock_mask_t readmask=SNDE_INFOSTORE_COMPONENTS,snde_infostore_lock_mask_t writemask=0,temporary=false) // readmask/writemask contains OR'd SNDE_INFOSTORE_xxx bits 
    {
      // attempt to obtain set of component pointers
      // including this component and all sub-components.
      // the obtain_lock on an assembly assumes the caller has at least a temporary readlock or writelock to SNDE_INFOSTORE_OBJECT_TREES, but that is not required to call obtain_lock() on a part. 
      // Assumes this is either the only component being locked or the caller is taking care of the locking order
      
      std::shared_ptr<component> our_ptr=shared_from_this();


      if (readmask & SNDE_INFOSTORE_COMPONENTS || writemask & SNDE_INFOSTORE_COMPONENTS) {
	process->get_locks_lockable_mask(our_ptr,SNDE_INFOSTORE_COMPONENTS,readmask & SNDE_INFOSTORE_ALL,writemask & SNDE_INFOSTORE_ALL);	
      }

      if (readmask & SNDE_COMPONENT_GEOM_ALL || writemask & SNDE_COMPONENT_GEOM_ALL) { // if ANY geometry requested... 
	obtain_geom_lock(process,recdb,recdb_context,readmask & SNDE_COMPONENT_GEOM_ALL,writemask & SNDE_COMPONENT_GEOM_ALL);
      }
      // *** need to support temporary parameter and returning a value ***!!! 
    }
    */
      
    virtual std::shared_ptr<mutableparameterizationstore> addparameterization(std::shared_ptr<mutablerecdb> recdb,std::string recdb_context,std::shared_ptr<snde::parameterization> parameterization,std::string name,const recmetadata &metadata);
    /*
    virtual std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>> get_instances(std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,snde_orientation3 orientation, std::shared_ptr<immutable_metadata> metadata, std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>(std::shared_ptr<part> partdata,std::vector<std::string> parameterization_data_names)> get_uv_imagedata) //,std::shared_ptr<std::unordered_map<std::string,metadatum>> metadata)
    {
      struct snde_partinstance ret=snde_partinstance{ .orientation=orientation,
	                                              .partnum = idx(),
						      .firstuvimage=SNDE_INDEX_INVALID,
						      .uvnum=SNDE_INDEX_INVALID,};
      std::shared_ptr<part> ret_ptr;

      ret_ptr = std::dynamic_pointer_cast<part>(shared_from_this());

      std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>> ret_vec;


      std::vector<std::string> parameterization_data_names;
      std::string uv_parameterization_channels=metadata->GetMetaDatumStr("uv_parameterization_channels","");
      // split comma-separated list of parameterization_data_names
      
      char *param_channels_c=strdup(uv_parameterization_channels.c_str());
      char *saveptr=NULL;
      for (char *tok=strtok_r(param_channels_c,",",&saveptr);tok;tok=strtok_r(NULL,",",&saveptr)) {
	parameterization_data_names.push_back(stripstr(tok));
      }

      ::free(param_channels_c); // :: means search in the global namespace for cstdlib free
      
      std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>> parameterization_data = get_param_data(std::dynamic_pointer_cast<part>(shared_from_this()),parameterization_data_names);
      //if (parameterization_data.find(name) != parameterization_data.end()) {
      //std::shared_ptr<uv_images> param_data=parameterization_data.at(name);
      //ret.uvnum = parameterizations.at(param_data->parameterization_name)->idx;
      //
      //ret.firstuvimage = param_data->firstuvimage;
      //}
      //{
//	std::string parameterization_name="";
//	
//	if (parameterizations.size() > 0) {
//	  parameterization_name=parameterizations.begin()->first;
//	}
//      
//	auto pname_entry=metadata->find(name+"."+"parameterization_name");
//	if (pname_entry != metadata->end()) {
//	  parameterization_name=pname_entry->second.Str(parameterization_name);
//	}
//	
//	auto pname_mesheduv=parameterizations.find(parameterization_name);
//	if (pname_mesheduv==parameterizations.end()) {
//	  ret.uvnum=SNDE_INDEX_INVALID;
//	} else {
//	  ret.uvnum=pname_mesheduv->second->idx;
//
	  // ***!!!!! NEED TO RETHINK HOW THIS WORKS.
	  // REALLY WANT TO IDENTIFY THE TEXTURE DATA FROM
	  // A CHANNEL NAME IN METADATA, THE PARAMETERIZATION
	  // FROM METADATA, AND HAVE SOME KIND OF
	  // CACHE OF TEXTURE DATA -> IMAGE TRANSFORMS
//	  auto iname_entry=metadata->find(name+"."+"images");
//	  if (iname_entry != metadata->end()) {
//	    std::string imagesname = pname_entry->second.Str("");
//	    
//	    std::shared_ptr<uv_images> images = pname_mesheduv->second->find_images(imagesname);
//	    
//	    if (!images) {
//	      throw std::runtime_error("part::get_instances():  Unknown UV images name: "+patchesname);
//	    }
//	    ret.firstuvimage=patches->firstuvpatch;
//	    //ret.numuvimages=patches->numuvpatches;
//	      
//	  }
//	  
//	}
 //     }
//      
//      //return std::vector<struct snde_partinstance>(1,ret);
//      ret_vec.push_back(std::tuple_cat(std::make_tuple(ret,ret_ptr),parameterization_data));
//      return ret_vec;
//    }
*/    

    virtual void obtain_geom_lock(std::shared_ptr<lockingprocess> process, std::shared_ptr<iterablerecrefs> recdb_reclist=nullptr,std::string recdb_context="/",snde_infostore_lock_mask_t readmask=SNDE_COMPONENT_GEOM_ALL, snde_infostore_lock_mask_t writemask=0, snde_infostore_lock_mask_t resizemask=0)
    {
      /* writemask contains OR'd SNDE_COMPONENT_GEOM_xxx bits */

      /* 
	 obtain locks from all our components... 
	 These have to be spawned so they can all obtain in parallel, 
	 following the locking order. 

	 You don't have to have the part locked in order to call this. But if it's not locked, it might 
	 change under you and point to different underlying geometry than what this obtains the locks for */

      snde_index idx=this->idx(); // just do the function call once
      /* NOTE: Locking order here must follow order in geometry constructor (above) */
      /* NOTE: Parallel Python implementation obtain_lock_pycpp 
	 must be maintained in geometry.i */

      assert(readmask & SNDE_COMPONENT_GEOM_PARTS || writemask & SNDE_COMPONENT_GEOM_PARTS); // Cannot do remainder of locking without read access to part

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
	  
	  if (geom->geom.vertnormals) {
	    process->get_locks_array_mask((void **)&geom->geom.vertnormals,SNDE_COMPONENT_GEOM_VERTNORMALS,SNDE_COMPONENT_GEOM_TRIS_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].firsttri,geom->geom.parts[idx].numtris);
	  }
	  
	  if (geom->geom.trinormals) {
	    process->get_locks_array_mask((void **)&geom->geom.trinormals,SNDE_COMPONENT_GEOM_TRINORMALS,SNDE_COMPONENT_GEOM_TRIS_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].firsttri,geom->geom.parts[idx].numtris);
	  }
	  
	  if (geom->geom.inplanemats) {
	    process->get_locks_array_mask((void **)&geom->geom.inplanemats,SNDE_COMPONENT_GEOM_INPLANEMATS,SNDE_COMPONENT_GEOM_TRIS_RESIZE,readmask,writemask,resizemask,geom->geom.parts[idx].firsttri,geom->geom.parts[idx].numtris);
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



    virtual void obtain_uv_lock(std::shared_ptr<lockingprocess> process, std::shared_ptr<iterablerecrefs> recdb_reclist=nullptr,std::string recdb_context="/",snde_infostore_lock_mask_t readmask=SNDE_COMPONENT_GEOM_ALL, snde_infostore_lock_mask_t writemask=0, snde_infostore_lock_mask_t resizemask=0)
    {
      // Does nothing because we are not a parameterization
      
    }

    

    void free() /* You must be certain that nothing could be using this part's database entries prior to free() */
    {

      snde_index idx=this->idx(); // just do the function call once

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
    
    // NOTE: assembly generally locked by holding the lock of its
    // ancestor mutablegeomstore (mutableinfostore)
    // this lock should be held when calling its methods
  public:
    // "pieces" vector uses name (recdb path relative to this context) and orientation
    // to set up multiple components within the assembly.
    // names can be repeated... From a metadata perspective
    // it would be like the name with an integer >= 2 concatenated, e.g. mypart2
    std::shared_ptr<const std::vector<std::tuple<std::string,snde_orientation3>>> _pieces; // atomic shared pointer for lock_free access
    //snde_orientation3 _orientation; /* orientation of this part/assembly relative to its parent */

    /* NOTE: May want to add cache of 
       openscenegraph group nodes representing 
       this assembly  */ 

    assembly() :
      component()
    {
      lock=std::make_shared<rwlock>();
      //this->type=subassembly;
      //this->_orientation=orientation;
      
    }
    

    assembly(std::shared_ptr<const std::vector<std::tuple<std::string,snde_orientation3>>> pieces) :
      //NOTE: The object pointed to by pieces MUST NOT BE CHANGED after being passed to this constructor!!!
      component(),
      _pieces(pieces)
    {
      lock=std::make_shared<rwlock>();
      //this->type=subassembly;
      //this->_orientation=orientation;
      
    }
    
//virtual snde_orientation3 orientation(void)
    //{
    //  return _orientation;
    //}

    std::shared_ptr<const std::vector<std::tuple<std::string,snde_orientation3>>> pieces()
    {
      return std::atomic_load(&_pieces);
    }

    virtual std::shared_ptr<std::vector<std::tuple<std::string,snde_orientation3>>> _begin_atomic_pieces_update()
    // component must be locked for write when calling this function
    {
      // Make copy of atomically-guarded data and return mutable copy
      return std::make_shared<std::vector<std::tuple<std::string,snde_orientation3>>>(*pieces());
      
    }

    virtual void _end_atomic_pieces_update(std::shared_ptr<const std::vector<std::tuple<std::string,snde_orientation3>>> new_pieces)
    {
      std::atomic_store(&_pieces,new_pieces);
    }

    
    virtual std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>
    explore_component_get_instances(std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> &component_set,
				    std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,
				    snde_orientation3 orientation,
				    std::shared_ptr<immutable_metadata> metadata,
				    std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names)> get_uv_imagedata);
    
    /*
    virtual rwlock_token_set obtain_lock(std::shared_ptr<lockingprocess> process,std::shared_ptr<mutablerecdb> recdb=null,std::string recdb_context="",snde_infostore_lock_mask_t readmask=SNDE_INFOSTORE_COMPONENTS|SNDE_INFOSTORE_INFOSTORE,snde_infostore_lock_mask_t writemask=0,bool temporary=false) // readmask/writemask contains OR'd SNDE_INFOSTORE_xxx bits. If temporary is set, then locks returned but not merged into process status.  
    {
      // attempt to obtain set of component pointers
      // including this component and all sub-components.
      // WARNING: consistency not guaranteed unless readmask or writemask contains SNDE_INFOSTORE_COMPONENTS|SNDE_INFOSTORE_INFOSTORE
      
      std::shared_ptr<component> our_ptr=shared_from_this(); 


      std::set<std::shared_ptr<component>,std::owner_less<std::shared_ptr<component>>> component_set;
      

      
      // Now the geometry (if applicable)
      if (readmask & SNDE_COMPONENT_GEOM_ALL || writemask & SNDE_COMPONENT_GEOM_ALL) { // if ANY geometry requested... 
	for (auto & comp: component_set) {
	  comp->obtain_geom_lock(process,recdb,recdb_context,readmask & SNDE_COMPONENT_GEOM_ALL,writemask & SNDE_COMPONENT_GEOM_ALL);
	  
	}
      }
      // ***!!! Need to support temporary parameter and return value
    }
    */
    
    /*
    virtual std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>> get_instances(std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,snde_orientation3 orientation, std::shared_ptr<immutable_metadata> metadata, std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>(std::shared_ptr<part> partdata,std::vector<std::string> parameterization_data_names)> get_param_data)
    {
      std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>> instances;

      
      for (auto & piece : *pieces()) {
	std::shared_ptr<immutable_metadata> reduced_metadata=reduce_metadata(metadata->metadata,piece_comp->name);


	// multiply externally given orientation by orientation of this piece
	snde_orientation3 neworientation;
	orientation_orientation_multiply(orientation,std::get<0>(piece.second),&neworientation);
	
	std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>>  newinstances=std::get<1>(piece.second)->get_instances(recdb_reclist,recdb_context,neworientation,reduced_metadata,get_param_data);
	instances.insert(instances.end(),newinstances.begin(),newinstances.end());
      }
      return instances;
    }
    */
    
    virtual void obtain_geom_lock(std::shared_ptr<lockingprocess> process, std::shared_ptr<iterablerecrefs> recdb_reclist=nullptr,std::string recdb_context="/",snde_infostore_lock_mask_t readmask=SNDE_COMPONENT_GEOM_ALL,snde_infostore_lock_mask_t writemask=0,snde_infostore_lock_mask_t resizemask=0);

    virtual void obtain_uv_lock(std::shared_ptr<lockingprocess> process, std::shared_ptr<iterablerecrefs> recdb_reclist=nullptr,std::string recdb_context="/",snde_infostore_lock_mask_t readmask=SNDE_COMPONENT_GEOM_ALL, snde_infostore_lock_mask_t writemask=0, snde_infostore_lock_mask_t resizemask=0)
    {
      // Does nothing because we are not a parameterization
      
    }

    
    virtual ~assembly()
    {
      
    }


    static std::shared_ptr<assembly> from_partlist(std::shared_ptr<mutablerecdb> recdb,std::string recdb_context,std::shared_ptr<std::vector<std::string>> partnames)
    {
      snde_orientation3 null_orientation;
      snde_null_orientation3(&null_orientation);

      std::shared_ptr<std::vector<std::tuple<std::string,snde_orientation3>>> pieces = std::make_shared<std::vector<std::tuple<std::string,snde_orientation3>>>();

      //std::unordered_map<std::string,metadatum> metadata;
      
      /* Make sure that part names are unique? */
      /*
      for (size_t cnt=0; cnt < parts->size();cnt++) {

	std::string postfix=std::string("");
	std::string partname;
	do {
	  partname=(*parts)[cnt].first->name+postfix;
	  postfix += "_"; // add additional trailing underscore
	} while (pieces->find(partname) != pieces->end());
	
	for (auto md: (*parts)[cnt].second) {
	  // prefix part name, perhaps a postfix, and "." onto metadata name
	  metadatum newmd(partname+"."+md.first,md.second);
	  assert(metadata.find(newmd.Name)==metadata.end()); // metadata should not exist already!
	  
	  metadata.emplace(newmd.Name,newmd);
	}
      */
      for (auto & partname : *partnames) {
	// ***!!! Should provide a way to give the partlist members different orientations
	// ***!!! Should provide a way to give duplicated members different metadata 
	pieces->emplace_back(std::make_tuple(partname,null_orientation));
	
	
      }
      std::shared_ptr<assembly> assem=std::make_shared<assembly>(pieces);
      
      return assem;
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
    
  // virtual void obtain_geom_lock(std::shared_ptr<lockingprocess> process, snde_infostore_lock_mask_t readmask=SNDE_COMPONENT_GEOM_ALL,snde_infostore_lock_mask_t writemask=0,snde_infostore_lock_mask_t resizemask=0)
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
