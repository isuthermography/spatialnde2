
#include "graphics_storage.hpp"

namespace snde {
  
  bool snde_doubleprec_coords()
  {
#ifdef SNDE_DOUBLEPREC_COORDS
    return true;
#else
    return false;
#endif
  }
  
  graphics_storage::graphics_storage(std::shared_ptr<graphics_storage_manager> graphman,std::shared_ptr<arraymanager> manager,std::shared_ptr<memallocator> memalloc,std::string recording_path,uint64_t recrevision,memallocator_regionid id,void **basearray,size_t elementsize,snde_index base_index,unsigned typenum,snde_index nelem,bool requires_locking_read,bool requires_locking_write,bool finalized) :
    recording_storage(recording_path,recrevision,id,basearray,elementsize,base_index,typenum,nelem,manager->locker,requires_locking_read,requires_locking_write,finalized),
    manager(manager),
    memalloc(memalloc),
    graphman(graphman)

    // Note: Called with array locks held
  {
    
  }
  
  graphics_storage::~graphics_storage()
  {
    //std::lock_guard<std::mutex> cmgr_lock(follower_cachemanagers_lock);
    // Don't need the lock because we are expiring!!!
    for (auto && cachemgr: follower_cachemanagers) {
      std::shared_ptr<cachemanager> cmgr_strong=cachemgr.lock();
      if (cmgr_strong) {
	cmgr_strong->notify_storage_expiration(lockableaddr(),base_index,nelem); // really has no effect (but we do it for completeness) as the openclcachemanager notify_storage_expiration ignores anything with a nonzero base-index (like all of our data)
      }
    }
    
  }
  void *graphics_storage::dataaddr_or_null()
  {
    std::shared_ptr<nonmoving_copy_or_reference> ref_ptr=ref();
    
    if (ref_ptr) {
      return ref_ptr->get_shiftedptr();
    } else {
      return shiftedarray;
    }
    
  }
  void *graphics_storage::cur_dataaddr()
  // warning -- may need to be locked for read or write as appropriate
  {
    std::shared_ptr<nonmoving_copy_or_reference> ref_ptr=ref();
    if (ref_ptr) {
      return ref_ptr->get_shiftedptr();
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

  snde_index graphics_storage::lockablenelem()
  {
    return manager->allocators()->at(_basearray).alloc->total_nelem();
  }

  std::shared_ptr<recording_storage> graphics_storage::obtain_nonmoving_copy_or_reference()
  // Note: May (generally should be) called with the underlying data array locked for read to prevent ptr from being modified under us
  {

    std::shared_ptr<nonmoving_copy_or_reference> ref_ptr=ref();

    if (!ref_ptr) {
      
      rwlock_token_set all_locks;
      
      // lock for read here to make the read of *_basearray safe
      if (requires_locking_read) {
	std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(manager->locker);
	std::shared_ptr<lockholder> holder = std::make_shared<lockholder>();
	holder->store(lockprocess->get_locks_read_array(_basearray));
	all_locks = lockprocess->finish();
	
      }
      ref_ptr=memalloc->obtain_nonmoving_copy_or_reference(recording_path,recrevision,id,_basearray,*_basearray,elementsize*base_index,elementsize*nelem);

      // locks released as context expires. 
    }
    
    std::shared_ptr<recording_storage_reference> reference = std::make_shared<recording_storage_reference>(recording_path,recrevision,id,nelem,shared_from_this(),ref_ptr);

    return reference;
  }


  void graphics_storage::mark_as_invalid(std::shared_ptr<cachemanager> already_knows,snde_index pos, snde_index numelem)
  // pos and numelem are relative to __this_recording__
  {
    std::set<std::weak_ptr<cachemanager>,std::owner_less<std::weak_ptr<cachemanager>>> fc_copy;
    {
      std::lock_guard<std::mutex> cmgr_lock(follower_cachemanagers_lock);
      fc_copy=follower_cachemanagers;
    }
    for (auto && cmgr: fc_copy) {
      std::shared_ptr<cachemanager> cmgr_strong=cmgr.lock();
      if (cmgr_strong) {
	if (cmgr_strong != already_knows) {
	  cmgr_strong->mark_as_invalid(lockableaddr(),base_index,pos,nelem);
	}
      }
    }
  }
  
  void graphics_storage::add_follower_cachemanager(std::shared_ptr<cachemanager> cachemgr)
  {
    {
      std::lock_guard<std::mutex> cmgr_lock(follower_cachemanagers_lock);
      follower_cachemanagers.emplace(cachemgr);
    }

    std::shared_ptr<graphics_storage_manager> graphman_strong=graphman.lock();
    if (graphman_strong) {
      graphman_strong->add_follower_cachemanager(cachemgr);
    }
  }


  
  std::shared_ptr<nonmoving_copy_or_reference> graphics_storage::ref()
  {
    return std::atomic_load(&_ref);
  }
  
  void graphics_storage::assign_ref(std::shared_ptr<nonmoving_copy_or_reference> ref)
  {
    std::atomic_store(&_ref,ref);
  }

  

  template <typename L,typename T>
  void graphics_storage_manager::add_arrays_given_sizes(memallocator_regionid *nextid,const std::set<snde_index> &elemsizes,L **leaderptr,T **arrayptr,const std::string &arrayname)
  // NOTE: CONTENTS MUST REMAIN PARALLEL TO ALTERNATE SPECIALIZATION OF add_arrays_given_sizes(), BELOW
  {
    if ((void **)leaderptr == (void**)arrayptr) {
      manager->add_allocated_array(graphics_recgroup_path,0,*nextid,(void **)leaderptr,sizeof(**leaderptr),0,elemsizes);
    } else {
      manager->add_follower_array(*nextid,(void **)leaderptr,(void **)arrayptr,sizeof(**arrayptr));
      
    }
    arrayaddr_from_name.emplace(arrayname,(void **)arrayptr);
    name_from_arrayaddr.emplace((void **)arrayptr,arrayname);
    arrayid_from_name.emplace(arrayname,*nextid);
    elemsize_from_name.emplace(arrayname,sizeof(T));

    (*nextid)++;



    // at this point everything has been added. Add a callback so if the memory pool is reallocated we can
    // notify the various cachemanagers
    // (this specialization is always the last, so it's safe to do it just here and not in the other specialization)
    std::shared_ptr<std::function<void(snde_index)>> pool_realloc_callback = std::make_shared<std::function<void(snde_index)>>([this,leaderptr](snde_index total_nelem) {
      /* Note: we're not managing context, but presumably the openclarrayinfo that is our key in buffer_map will prevent the context from being freed */
      // Whatever triggered the realloc had better have a write lock (or equivalent) on the array in which case nothing else should be reading 
      // or writing so this stuff should be safe. 
      
      this->mark_as_invalid(nullptr,(void**)leaderptr,0,SNDE_INDEX_INVALID);
      
      
      // now mark all the followers as invalid
      std::shared_ptr<std::multimap<void **,void **>> follower_map = this->manager->arrays_managed_by_allocator();

      std::multimap<void **,void **>::iterator begin,end,iter;
      std::tie(begin,end) = follower_map->equal_range((void**)leaderptr);

      for (iter=begin;iter != end;iter++) {
	if (iter->second != ((void**)leaderptr)) {
	  this->mark_as_invalid(nullptr,iter->second,0,SNDE_INDEX_INVALID);
	  
	}
      }
      
    });

    this->manager->allocators()->at((void **)leaderptr).alloc->register_pool_realloc_callback(pool_realloc_callback);
						       
    pool_realloc_callbacks.emplace(name_from_arrayaddr.at((void**)leaderptr),pool_realloc_callback);


  }
  
  template <typename L,typename T,typename... Args>
  void graphics_storage_manager::add_arrays_given_sizes(memallocator_regionid *nextid,const std::set<snde_index> &elemsizes,L **leaderptr,T **arrayptr,const std::string &arrayname,Args... args)
  // NOTE: CONTENTS MUST REMAIN PARALLEL TO ALTERNATE SPECIALIZATION OF add_arrays_given_sizes(), ABOVE
  {
    if ((void **)leaderptr == (void**)arrayptr) {
      manager->add_allocated_array(graphics_recgroup_path,0,*nextid,(void **)leaderptr,sizeof(**leaderptr),0,elemsizes);
    } else {
      manager->add_follower_array(*nextid,(void **)leaderptr,(void **)arrayptr,sizeof(**arrayptr));
      
    }
    
    arrayaddr_from_name.emplace(arrayname,(void **)arrayptr);
    name_from_arrayaddr.emplace((void **)arrayptr,arrayname);
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
    // unregister any reallocation callbacks (with leader arrays)
    for (auto && arrayname_realloccallback: pool_realloc_callbacks) {
      //void **arrayptr = arrayaddr_from_name.at(arrayname);
      manager->allocators()->at(arrayaddr_from_name.at(arrayname_realloccallback.first)).alloc->unregister_pool_realloc_callback(arrayname_realloccallback.second);
				       
    };

    pool_realloc_callbacks.clear();
    
    // Notify all caches that we are going away. 
    for (auto && cachemgr: follower_cachemanagers) {
      std::shared_ptr<cachemanager> cmgr_strong = cachemgr.lock();
      if (cmgr_strong) {
	for (auto && arrayname_arrayaddr: arrayaddr_from_name) {
	  void **arrayptr = arrayname_arrayaddr.second;
	  snde_index nelem = manager->allocators()->at(arrayptr).totalnelem();
	  cmgr_strong->notify_storage_expiration(arrayptr,0,nelem);
	}
      }
    }

    // Destructor needs to wipe out manager's array pointers because they point into this geometry object, that
    // is being destroyed
    
    manager->cleararrays((void *)&geom,sizeof(geom));
  }


  //  math funcs use this to  grab space in a follower array
  // NOTE: This doesn't currently prevent two math functions from
  // grabbing the same follower array -- which could lead to corruption
  std::shared_ptr<recording_storage> graphics_storage_manager::get_follower_storage(std::string recording_path,std::string leader_array_name,
										    std::string follower_array_name,
										    uint64_t recrevision,
										    snde_index base_index, // as allocated for leader_array
										    size_t elementsize,
										    unsigned typenum, // MET_...
										    snde_index nelem,
										    bool is_mutable)
  {
    void **leader_arrayaddr = arrayaddr_from_name.at(leader_array_name);
    void **follower_arrayaddr = arrayaddr_from_name.at(follower_array_name);
    
    if (elemsize_from_name.at(follower_array_name) != elementsize) {
      throw snde_error("Mismatch between graphics array field %s element size with allocation: %u vs. %u",follower_array_name,(unsigned)elemsize_from_name.at(follower_array_name),(unsigned)elementsize);
    }


    std::shared_ptr<graphics_storage> retval = std::make_shared<graphics_storage>(std::dynamic_pointer_cast<graphics_storage_manager>(shared_from_this()),manager,manager->_memalloc,recording_path,recrevision,arrayid_from_name.at(follower_array_name),follower_arrayaddr,elementsize,base_index,typenum,nelem,is_mutable || manager->_memalloc->requires_locking_read,is_mutable || manager->_memalloc->requires_locking_write,false);

    // see also parallel code in allocate_recording_lockprocesss, below
    if (!retval->requires_locking_write) {
      // switch pointer to a nonmoving copy or reference

      // assign _ref: 
      retval->assign_ref(manager->_memalloc->obtain_nonmoving_copy_or_reference(recording_path,recrevision,retval->id,retval->_basearray,*retval->_basearray,elementsize*base_index,elementsize*nelem));
      
    } else if (!retval->requires_locking_read) {
      // unimplemented so-far. This is the case of locking required for write, but not read,
      // as (for example) lock required on write to ensure graphics cache synchronization, or not writing to
      // graphics buffer while the entire buffer is mapped for read, etc. 
      // Will require finalization hook of some sort to switch the pointer to a non-moving copy or reference
      // Must not forget to also update the struct snde_array_info of the recording!!!
      assert(0);
    }

    return retval;
    
  }
  
  
  std::shared_ptr<recording_storage> graphics_storage_manager::allocate_recording_lockprocess(std::string recording_path,std::string array_name, // use "" for default array
											      uint64_t recrevision,
											      size_t elementsize,
											      unsigned typenum, // MET_...
											      snde_index nelem,
											      bool is_mutable,
											      std::shared_ptr<lockingprocess> lockprocess,
											      std::shared_ptr<lockholder> holder) // returns (storage pointer,base_index); note that the recording_storage nelem may be different from what was requested.
  {
    // Will need to mark array as locking required for write, at least....


    void **arrayaddr = arrayaddr_from_name.at(array_name);
    if (elemsize_from_name.at(array_name) != elementsize) {
      throw snde_error("Mismatch between graphics array field %s element size with allocation: %u vs. %u",array_name,(unsigned)elemsize_from_name.at(array_name),(unsigned)elementsize);
    }
    
    
    holder->store_alloc(lockprocess->alloc_array_region(manager,arrayaddr,nelem,""));


    snde_index addr = holder->get_alloc(arrayaddr,"");
    
    

    std::shared_ptr<graphics_storage> retval = std::make_shared<graphics_storage>(std::dynamic_pointer_cast<graphics_storage_manager>(shared_from_this()),manager,manager->_memalloc,recording_path,recrevision,arrayid_from_name.at(array_name),arrayaddr,elementsize,addr,typenum,nelem,is_mutable || manager->_memalloc->requires_locking_read,is_mutable || manager->_memalloc->requires_locking_write,false);

    // if not(requires_locking_write) we must switch the pointer to a nonmoving_copy_or_reference NOW because otherwise the array might be moved around as we try to write.
    // if not(requires_locking_read) we must switch the pointer to a nonmoving_copy_or_reference on finalization

    // see also parallel code in get_follower_storage, above
    if (!retval->requires_locking_write) {
      // switch pointer to a nonmoving copy or reference

      // assign _ref: 
      retval->assign_ref(manager->_memalloc->obtain_nonmoving_copy_or_reference(recording_path,recrevision,retval->id,retval->_basearray,*retval->_basearray,elementsize*addr,elementsize*nelem));
      
    } else if (!retval->requires_locking_read) {
      // unimplemented so-far. This is the case of locking required for write, but not read,
      // as (for example) lock required on write to ensure graphics cache synchronization, or not writing to
      // graphics buffer while the entire buffer is mapped for read, etc. 
      // Will require finalization hook of some sort to switch the pointer to a non-moving copy or reference
      // Must not forget to also update the struct snde_array_info of the recording!!!
      assert(0);
    }

    
    return retval;
    
  }

  
  std::shared_ptr<recording_storage> graphics_storage_manager::allocate_recording(std::string recording_path,std::string array_name, // use "" for default array
										  uint64_t recrevision,
										  size_t elementsize,
										  unsigned typenum, // MET_...
										  snde_index nelem,
										  bool is_mutable) // returns (storage pointer,base_index); note that the recording_storage nelem may be different from what was requested.
  {
    std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(manager->locker);
    std::shared_ptr<lockholder> holder = std::make_shared<lockholder>();
    rwlock_token_set all_locks;
    std::shared_ptr<recording_storage> retstorage;

    retstorage = allocate_recording_lockprocess(recording_path,array_name, // use "" for default array
						recrevision,
						elementsize,
						typenum, // MET_...
						nelem,
						is_mutable,
						lockprocess,
						holder);
    
    
    all_locks = lockprocess->finish();
    unlock_rwlock_token_set(all_locks);
    
    return retstorage;
  }


  void graphics_storage_manager::add_follower_cachemanager(std::shared_ptr<cachemanager> cachemgr)
  {
    {
      std::lock_guard<std::mutex> cmgr_lock(follower_cachemanagers_lock);
      follower_cachemanagers.emplace(cachemgr);
    }

  }



  void graphics_storage_manager::mark_as_invalid(std::shared_ptr<cachemanager> already_knows,void **arrayptr,snde_index pos,snde_index numelem)
  {
    std::set<std::weak_ptr<cachemanager>,std::owner_less<std::weak_ptr<cachemanager>>> fc_copy;
    {
      std::lock_guard<std::mutex> cmgr_lock(follower_cachemanagers_lock);
      fc_copy=follower_cachemanagers;
    }
    for (auto && cmgr: fc_copy) {
      std::shared_ptr<cachemanager> cmgr_strong=cmgr.lock();
      if (cmgr_strong) {
	if (cmgr_strong != already_knows) {
	  cmgr_strong->mark_as_invalid(arrayptr,0,pos,numelem);
	}
      }
    }


  }

  
};
