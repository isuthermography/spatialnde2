
#include "graphics_storage.hpp"

namespace snde {

  graphics_storage::graphics_storage(std::string recording_path,uint64_t recrevision,memallocator_regionid id,void **basearray,size_t elementsize,snde_index base_index,unsigned typenum,snde_index nelem,bool requires_locking_read,bool requires_locking_write,bool finalized) :
    recording_storage(recording_path,recrevision,id,basearray,elementsize,base_index,typenum,nelem,requires_locking_read,requires_locking_write,finalized)
    
  {
    
  }
  

  void *graphics_storage::dataaddr_or_null()
  {
    if (ref) {
      return ref->get_shiftedptr();
    } else {
      return shiftedarray;
    }
    
  }
  void *graphics_storage::cur_dataaddr()
  {
    if (ref) {
      return ref->get_shiftedptr();
    } else {
      if (shiftedarray) {
	return shiftedarray;
      }
    }
    // fallback
    return (void *)(((char *)_basearray) + elementsize*base_index);
    
  }

  void **graphics_storage::lockableaddr()
  {
    return _basearray;
  }
  
  std::shared_ptr<recording_storage> graphics_storage::obtain_nonmoving_copy_or_reference()
  {
    std::shared_ptr<recording_storage_reference> reference = std::make_shared<recording_storage_reference>(recording_path,recrevision,id,nelem,shared_from_this(),ref);

    return reference;
  }

  template <typename L,typename T>
  void graphics_storage_manager::add_arrays_given_sizes(memallocator_regionid *nextid,const std::set<snde_index> &elemsizes,L **leaderptr,T **arrayptr,const std::string &arrayname)
  {
    if ((void **)leaderptr == (void**)arrayptr) {
      manager->add_allocated_array(graphics_recgroup_path,0,*nextid,(void **)leaderptr,sizeof(**leaderptr),0,elemsizes);
    } else {
      manager->add_follower_array(*nextid,(void **)leaderptr,(void **)arrayptr,sizeof(**arrayptr));
      
    }
    arrayaddr_from_name.emplace(arrayname,(void **)arrayptr);
    arrayid_from_name.emplace(arrayname,*nextid);
    elemsize_from_name.emplace(arrayname,sizeof(T));
    
    (*nextid)++;
  }
  
  template <typename L,typename T,typename... Args>
  void graphics_storage_manager::add_arrays_given_sizes(memallocator_regionid *nextid,const std::set<snde_index> &elemsizes,L **leaderptr,T **arrayptr,const std::string &arrayname,Args... args)
  {
    if ((void **)leaderptr == (void**)arrayptr) {
      manager->add_allocated_array(graphics_recgroup_path,0,*nextid,(void **)leaderptr,sizeof(**leaderptr),0,elemsizes);
    } else {
      manager->add_follower_array(*nextid,(void **)leaderptr,(void **)arrayptr,sizeof(**arrayptr));
      
    }
    
    arrayaddr_from_name.emplace(arrayname,(void **)arrayptr);
    arrayid_from_name.emplace(arrayname,*nextid);
    elemsize_from_name.emplace(arrayname,sizeof(T));

    (*nextid)++;
    
    add_arrays_given_sizes(nextid,elemsizes,leaderptr,args...);
  }


  template <typename L,typename T>
  static void accumulate_sizes(std::set<snde_index> *accumulator,L **leaderptr,T **arrayptr,const std::string &arrayname)
  {
    accumulator->insert(sizeof(L));
    accumulator->insert(sizeof(T));
  }

  
  template <typename L,typename T,typename... Args>
  static void accumulate_sizes(std::set<snde_index> *accumulator,L **leaderptr,T **arrayptr,const std::string &arrayname,Args... args)
  {
    accumulator->insert(sizeof(L));
    accumulator->insert(sizeof(T));

    accumulate_sizes(accumulator,leaderptr,args...);
  }

  

  // Add a group of arrays to the storage manager that share allocations
  template <typename L,typename... Args>
  void graphics_storage_manager::add_grouped_arrays(memallocator_regionid *nextid,L **leaderptr,const std::string &leadername,Args... args)
  {
    std::set<snde_index> elemsizes;
    accumulate_sizes(&elemsizes,leaderptr,leaderptr,leadername,args...);

    add_arrays_given_sizes(nextid,elemsizes,leaderptr,leaderptr,leadername,args...);
  }

  

  graphics_storage_manager::graphics_storage_manager(const std::string &graphics_recgroup_path,std::shared_ptr<memallocator> memalloc,std::shared_ptr<allocator_alignment> alignment_requirements,std::shared_ptr<lockmanager> lockmgr,double tol):
    recording_storage_manager(), // superclass
    manager(std::make_shared<arraymanager>(memalloc,alignment_requirements,lockmgr)),
    graphics_recgroup_path(graphics_recgroup_path),
    geom() // Triggers value-initialization of .data which zero-initializes all members
  {

    memallocator_regionid next_region_id=0;
    
    geom.tol=tol;

    // Nominally for each leader array we do an add_allocated_array()
    // for each follower array we do an add_follower_array()
    // the add_allocated array takes a set of element sizes for all the followes
    // so that the allocation can be compatible with them

    

    // add_grouped_arrays automates the above for the leader and 0 or more followers. 
    add_grouped_arrays(&next_region_id,&geom.parts,"parts");
    
    //manager->add_allocated_array((void **)&geom.parts,sizeof(*geom.parts),0);
    
    
    //manager->add_allocated_array((void **)&geom.topos,sizeof(*geom.topos),0);
    //manager->add_allocated_array((void **)&geom.topo_indices,sizeof(*geom.topo_indices),0);

    add_grouped_arrays(&next_region_id,&geom.topos,"topos");
    add_grouped_arrays(&next_region_id,&geom.topo_indices,"topo_indices");
	
      

    //std::set<snde_index> triangles_elemsizes;
    
    //triangles_elemsizes.insert(sizeof(*geom.triangles));
    //triangles_elemsizes.insert(sizeof(*geom.refpoints));
    //triangles_elemsizes.insert(sizeof(*geom.maxradius));
    //triangles_elemsizes.insert(sizeof(*geom.vertnormals));
    //triangles_elemsizes.insert(sizeof(*geom.trinormals));
    //triangles_elemsizes.insert(sizeof(*geom.inplanemats));

    
    //manager->add_allocated_array((void **)&geom.triangles,sizeof(*geom.triangles),0,triangles_elemsizes);
    //manager->add_follower_array((void **)&geom.triangles,(void **)&geom.refpoints,sizeof(*geom.refpoints));
    //manager->add_follower_array((void **)&geom.triangles,(void **)&geom.maxradius,sizeof(*geom.maxradius));
    //manager->add_follower_array((void **)&geom.triangles,(void **)&geom.vertnormals,sizeof(*geom.vertnormals));
    //manager->add_follower_array((void **)&geom.triangles,(void **)&geom.trinormals,sizeof(*geom.trinormals));
    //manager->add_follower_array((void **)&geom.triangles,(void **)&geom.inplanemats,sizeof(*geom.inplanemats));

    add_grouped_arrays(&next_region_id,&geom.triangles,"triangles",
		       &geom.refpoints,"refpoints",
		       &geom.maxradius,"maxradius",
		       &geom.vertnormals,"vertnormals",
		       &geom.trinormals,"trinormals",
		       &geom.inplanemats,"inplanemats");
    
    add_grouped_arrays(&next_region_id,&geom.edges,"edges");
    
    
    add_grouped_arrays(&next_region_id,&geom.vertices,"vertices",
		       &geom.principal_curvatures,"principal_curvatures",
		       &geom.curvature_tangent_axes,"curvature_tangent_axes",
		       &geom.vertex_edgelist_indices,"vertex_edgelist_indices");

    add_grouped_arrays(&next_region_id,&geom.vertex_edgelist,"vertex_edgelist");

    add_grouped_arrays(&next_region_id,&geom.boxes,"boxes",
		       &geom.boxcoord,"boxcoord");

    add_grouped_arrays(&next_region_id,&geom.boxpolys,"boxpolys");
        
    
    /* parameterization */
    add_grouped_arrays(&next_region_id,&geom.uvs,"uvs");
    add_grouped_arrays(&next_region_id,&geom.uv_patches,"uv_patches");
    add_grouped_arrays(&next_region_id,&geom.uv_topos,"uv_topos");
    add_grouped_arrays(&next_region_id,&geom.uv_topo_indices,"uv_topo_indices");


    add_grouped_arrays(&next_region_id,&geom.uv_triangles,"uv_triangles",
		       &geom.inplane2uvcoords,"inplane2uvcoords",
		       &geom.uvcoords2inplane,"uvcoords2inplane");

    
    add_grouped_arrays(&next_region_id,&geom.uv_edges,"uv_edges");
    
    add_grouped_arrays(&next_region_id,&geom.uv_vertices,"uv_vertices",
		       &geom.uv_vertex_edgelist_indices,"uv_vertex_edgelist_indices");
    
    add_grouped_arrays(&next_region_id,&geom.uv_vertex_edgelist,"uv_vertex_edgelist");
    
    // ***!!! insert NURBS here !!!***


    add_grouped_arrays(&next_region_id,&geom.uv_boxes,"uv_boxes",
		       &geom.uv_boxcoord,"uv_boxcoord");

    add_grouped_arrays(&next_region_id,&geom.uv_boxpolys,"uv_boxpolys");

    
    
    //manager->add_allocated_array((void **)&geom.uv_images,sizeof(*geom.uv_images),0);
    
    
    /***!!! Insert uv patches and images here ***!!! */
    
    add_grouped_arrays(&next_region_id,&geom.vertex_arrays,"vertex_arrays");
    add_grouped_arrays(&next_region_id,&geom.texvertex_arrays,"texvertex_arrays");
    add_grouped_arrays(&next_region_id,&geom.texbuffer,"texbuffer");
    
    // ... need to initialize rest of struct...
    // Probably want an array manager class to handle all of this
    // initialization,
    // also creation and caching of OpenCL buffers and OpenGL buffers. 
    
    
  }


  graphics_storage_manager::~graphics_storage_manager()
  {
    // Destructor needs to wipe out manager's array pointers because they point into this geometry object, that
    // is being destroyed
    manager->cleararrays((void *)&geom,sizeof(geom));

  }


  // !!!*** how does math func grab space in a follower array???
  std::shared_ptr<recording_storage> graphics_storage_manager::allocate_recording(std::string recording_path,std::string array_name, // use "" for default array
										  uint64_t recrevision,
										  size_t elementsize,
										  unsigned typenum, // MET_...
										  snde_index nelem,
										  bool is_mutable) // returns (storage pointer,base_index); note that the recording_storage nelem may be different from what was requested.
  {
    // Will need to mark array as locking required for write, at least....

    std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(manager->locker);
    std::shared_ptr<lockholder> holder = std::make_shared<lockholder>();
    rwlock_token_set all_locks;

    void **arrayaddr = arrayaddr_from_name.at(array_name);
    if (elemsize_from_name.at(array_name) != elementsize) {
      throw snde_error("Mismatch between graphics array field %s element size with allocation: %u vs. %u",array_name,(unsigned)elemsize_from_name.at(array_name),(unsigned)elementsize);
    }
    
    
    holder->store_alloc(lockprocess->alloc_array_region(manager,arrayaddr,nelem,""));

    all_locks = lockprocess->finish();

    snde_index addr = holder->get_alloc(arrayaddr,"");
    
    
    unlock_rwlock_token_set(all_locks);

    std::shared_ptr<graphics_storage> retval = std::make_shared<graphics_storage>(recording_path,recrevision,arrayid_from_name.at(array_name),arrayaddr,elementsize,addr,typenum,nelem,is_mutable || manager->_memalloc->requires_locking_read,is_mutable || manager->_memalloc->requires_locking_write,false);

    // if not(requires_locking_write) we must switch the pointer to a nonmoving_copy_or_reference NOW because otherwise the array might be moved around as we try to write.
    // if not(requires_locking_read) we must switch the pointer to a nonmoving_copy_or_reference on finalization

    if (!retval->requires_locking_write) {
      retval->ref = manager->_memalloc->obtain_nonmoving_copy_or_reference(recording_path,recrevision,retval->id,retval->_basearray,*retval->_basearray,elementsize*addr,elementsize*nelem);
      
    } else if (!retval->requires_locking_read) {
      // unimplemented so-far. Will require finalization hook of some sort
      // Must not forget to also update the struct snde_array_info of the recording!!!
      assert(0);
    }

    
    return retval;
    
  }
  
};
