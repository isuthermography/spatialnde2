#include <vector>
#include <memory>

#ifndef SNDE_OPENSCENEGRAPH_PARAMETERIZATION_HPP
#define SNDE_OPENSCENEGRAPH_PARAMETERIZATION_HPP

namespace snde {

extern opencl_program texvertexarray_opencl_program;  // for now this is actualy defined in openscenegraph_geom.cpp.... if we create an openscenegraph_parameterization.cpp it should go there

  
static snde_index texvertexarray_from_uv_vertexarrayslocked(std::shared_ptr<geometry> geom,rwlock_token_set all_locks,snde_index uvnum,snde_index outaddr,snde_index outlen,cl_context context,cl_device_id device,cl_command_queue queue)
/* Should already have read locks on the part referenced by instance via obtain_lock() and the entire vertexarray locked for write */
/* Need to make copy... texvertexarray_... that operates on texture */
{

  snde_parameterization &uv = geom->geom.uvs[uvnum];


  assert(outlen==uv.numuvtris*6);
  
  //std::pair<snde_index,std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>>> addr_ptrs_tokens = geom->manager->alloc_arraylocked(all_locks,(void **)&geom->geom.vertex_arrays,part.numtris*9);
  
  //snde_index addr = addr_ptrs_tokens.first;
  //rwlock_token_set newalloc;

  // /* the vertex_arrays does not currently have any parallel-allocated arrays (these would have to be locked for write as well) */
  //assert(addr_ptrs_tokens.second.size()==1);
  //assert(addr_ptrs_tokens.second[0].first->value()==(void **)&geom->geom.vertex_arrays);
  //newalloc=addr_ptrs_tokens.second[0].second;

  cl_kernel texvertexarray_kern = texvertexarray_opencl_program.get_kernel(context,device);


  OpenCLBuffers Buffers(context,device,all_locks);
  
  // specify the arguments to the kernel, by argument number.
  // The third parameter is the array element to be passed
  // (actually comes from the OpenCL cache)
  
  Buffers.AddSubBufferAsKernelArg(geom->manager,texvertexarray_kern,0,(void **)&geom->geom.uvs,uvnum,1,false);
  
  
  Buffers.AddSubBufferAsKernelArg(geom->manager,texvertexarray_kern,1,(void **)&geom->geom.uv_triangles,uv.firstuvtri,uv.numuvtris,false);
  Buffers.AddSubBufferAsKernelArg(geom->manager,texvertexarray_kern,2,(void **)&geom->geom.uv_edges,uv.firstuvedge,uv.numuvedges,false);
  Buffers.AddSubBufferAsKernelArg(geom->manager,texvertexarray_kern,3,(void **)&geom->geom.uv_vertices,uv.firstuvvertex,uv.numuvvertices,false);

  Buffers.AddSubBufferAsKernelArg(geom->manager,texvertexarray_kern,4,(void **)&geom->geom.texvertex_arrays,outaddr,uv.numuvtris*6,true);
  
  
  size_t worksize=uv.numuvtris;
  cl_event kernel_complete=NULL;
  
  // Enqueue the kernel 
  cl_int err=clEnqueueNDRangeKernel(queue,texvertexarray_kern,1,NULL,&worksize,NULL,Buffers.NumFillEvents(),Buffers.FillEvents_untracked(),&kernel_complete);
  if (err != CL_SUCCESS) {
    throw openclerror(err,"Error enqueueing kernel");
  }
  /*** Need to mark as dirty; Need to Release Buffers once kernel is complete ****/
  
  clFlush(queue); /* trigger execution */
  Buffers.SubBufferDirty((void **)&geom->geom.texvertex_arrays,outaddr,uv.numuvtris*6,0,uv.numuvtris*6);
  
  Buffers.RemBuffers(kernel_complete,kernel_complete,true); 
  // Actually, we SHOULD wait for completion. (we are running in a compute thread, so waiting isn't really a problem)
  // (Are there unnecessary locks we can release first?)
  // ***!!! NOTE: Possible bug: If we manually step in the
  // debugger (gdb/PoCL) with between a RemBuffers(...,false)
  // and clWaitForEvents() then clWaitForEvents never returns.
  // ... If we just insert a sleep, though, clWaitForEvents works
  // fine. Perhaps a debugger/PoCL interaction? 
  //sleep(3);
  clWaitForEvents(1,&kernel_complete);
  //fprintf(stderr,"VertexArray kernel complete\n");
  
  
  clReleaseEvent(kernel_complete);

  // Release our reference to kernel, allowing it to be free'd
  clReleaseKernel(texvertexarray_kern);
  
  return outaddr; 
}

class osg_paramcacheentry : public std::enable_shared_from_this<osg_paramcacheentry> {
public:
  std::weak_ptr<osg_paramcacheentry> thisptr; /* Store this pointer so we can return it on demand... must be created by osg_texturecache */
  std::shared_ptr<osg_paramcacheentry> persistentptr; /* Store pointer here if we want persistence (otherwise leave as nullptr */

  std::weak_ptr<geometry> snde_geom;
  
  std::weak_ptr<parameterization> param;

  osg::ref_ptr<snde::OSGArray> TexCoordArray;
  std::shared_ptr<trm_dependency> texvertex_function; /* revision_manager function that renders winged edge structure into vertices */
  

  /* Remaining fields are updated when the vertex_function executes */
  struct snde_parameterization paramdata;
  snde_index cachedversion;

  osg_paramcacheentry(const osg_paramcacheentry &)=delete; /* copy constructor disabled */
  osg_paramcacheentry & operator=(const osg_paramcacheentry &)=delete; /* copy assignment disabled */

  osg_paramcacheentry(std::shared_ptr<geometry> snde_geom):
    snde_geom(snde_geom)
  {
    cachedversion=0;
  }


  void obtain_array_locks(std::shared_ptr<lockholder> holder,std::shared_ptr<lockingprocess_threaded> lockprocess,snde_component_geom_mask_t readmask, snde_component_geom_mask_t writemask,snde_component_geom_mask_t resizemask,bool include_texvertex_arrays, bool texvertex_arrays_write, bool texvertex_arrays_entire_array)
  // NOTE: This may be called from any thread!
  // NOTE: This does not lock the mutablegeomstore metadata!
  {
    std::shared_ptr<geometry> snde_geom_strong(snde_geom);
    std::shared_ptr<parameterization> param_strong(param);

    if (snde_geom_strong && param_strong) {
    
      /* Obtain lock for this component -- in parallel with our write lock on the vertex array, below */
      if (readmask != 0 || writemask != 0) {
	lockprocess->spawn( [ param_strong, lockprocess, readmask, writemask, resizemask ]() {
			      // Obtain lock for our parameterization
			      param_strong->obtain_lock(lockprocess,SNDE_UV_GEOM_UVS | readmask, writemask,resizemask);
			    });
      }
      
      if (include_texvertex_arrays && texvertex_arrays_write && !texvertex_arrays_entire_array) {
	/* Obtain write lock on vertex array output */
	rwlock_token_set texvertex_arrays_lock;
	lockholder_index texvertex_arrays_info;
	std::tie(texvertex_arrays_info,texvertex_arrays_lock) = lockprocess->get_locks_write_array_region((void **)&snde_geom_strong->geom.texvertex_arrays,texvertex_function->outputs[0].start,texvertex_function->outputs[0].len);//DataArray->offset,DataArray->nvec*3);
      } else if (include_texvertex_arrays && !texvertex_arrays_entire_array) {
	/* Obtain read lock on vertex array output */
	rwlock_token_set texvertex_arrays_lock;
	lockholder_index texvertex_arrays_info;
	std::tie(texvertex_arrays_info,texvertex_arrays_lock) = lockprocess->get_locks_read_array_region((void **)&snde_geom_strong->geom.texvertex_arrays,texvertex_function->outputs[0].start,texvertex_function->outputs[0].len);//,DataArray->offset,DataArray->nvec*3);
	
      } else if (include_texvertex_arrays && texvertex_arrays_write && !texvertex_arrays_entire_array) {
	rwlock_token_set texvertex_arrays_lock;
	lockholder_index texvertex_arrays_info;
	std::tie(texvertex_arrays_info,texvertex_arrays_lock) = lockprocess->get_locks_write_array((void **)&snde_geom_strong->geom.texvertex_arrays);      
      } else if (include_texvertex_arrays && !texvertex_arrays_entire_array) {
	rwlock_token_set texvertex_arrays_lock;
	lockholder_index texvertex_arrays_info;
	std::tie(texvertex_arrays_info,texvertex_arrays_lock) = lockprocess->get_locks_read_array((void **)&snde_geom_strong->geom.texvertex_arrays);      
      }
    }
  }

  rwlock_token_set obtain_array_locks(snde_component_geom_mask_t readmask, snde_component_geom_mask_t writemask,snde_component_geom_mask_t resizemask,bool include_texvertex_arrays, bool texvertex_arrays_write, bool texvertex_arrays_entire_array)
  // the geometry object_trees_lock should be held when this is called (but not necessarily by
  // this thread -- just to make sure it can't be changed) 
    
  // Locking the object_trees_lock should be taken care of by whoever is starting the transaction
  {
    std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
    std::shared_ptr<geometry> snde_geom_strong(snde_geom);
    std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(snde_geom_strong->manager->locker); // new locking process
    
    obtain_array_locks(holder,lockprocess,readmask,writemask,resizemask,include_texvertex_arrays,texvertex_arrays_write,texvertex_arrays_entire_array);
        
    rwlock_token_set all_locks=lockprocess->finish();

    return all_locks; // !!!*** Should we also return vertex_arrays_lock and/or _info? 
  }
  
  std::shared_ptr<osg_paramcacheentry> lock()
  {
    return shared_from_this();
  }
    
  ~osg_paramcacheentry()
  {
    std::shared_ptr<geometry> geom=snde_geom.lock();
    if (geom && TexCoordArray && TexCoordArray->offset != SNDE_INDEX_INVALID) {
      geom->manager->free((void **)&geom->geom.vertex_arrays,TexCoordArray->offset);
      TexCoordArray->offset=SNDE_INDEX_INVALID;
      TexCoordArray->nvec=0;
    }
  }
  
};


class osg_parameterizationcache: public std::enable_shared_from_this<osg_parameterizationcache> {
public:

  // param_cachedata is indexed by param->idx
  std::unordered_map<snde_index,osg_paramcacheentry> param_cachedata;
  std::shared_ptr<geometry> snde_geom;
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  
  std::mutex admin; // serialize references to param_cachedata because that could be used from any thread that drops the last reference to an paramcacheentry ... Need to think thread-safety of the instancecache through more carefully 

  
  osg_parameterizationcache(std::shared_ptr<geometry> snde_geom,
			    cl_context context,
			    cl_device_id device,
			    cl_command_queue queue) :
    snde_geom(snde_geom),
    context(context),
    device(device),
    queue(queue)
  {
    
  }

  
  std::shared_ptr<osg_paramcacheentry> lookup(std::shared_ptr<trm> rendering_revman,std::shared_ptr<parameterization> param) 
  {
    std::unordered_map<snde_index,osg_paramcacheentry>::iterator cache_entry;

    if (!param) return nullptr;
    
    std::unique_lock<std::mutex> adminlock(admin);
    
    
    cache_entry = param_cachedata.find(param->idx);
    if (cache_entry==param_cachedata.end()) {
      bool junk;
      std::tie(cache_entry,junk) = param_cachedata.emplace(std::piecewise_construct,
							   std::forward_as_tuple(param->idx),
							   std::forward_as_tuple(snde_geom));
      
      std::shared_ptr<osg_parameterizationcache> shared_cache = shared_from_this();
      
      // create shared pointer with custom deleter such that when
      // all references to this entry go away, we get called and can remove it
      // from the cache
      
      std::shared_ptr<osg_paramcacheentry> entry_ptr(&(cache_entry->second),
						     [ shared_cache ](osg_paramcacheentry *ent) { /* custom deleter... this is a parameter to the shared_ptr constructor, ... the osg_paramcachentry was created in emplace(), above.  */ 
						       std::unordered_map<snde_index,osg_paramcacheentry>::iterator foundent;
						       
						       std::lock_guard<std::mutex> adminlock(shared_cache->admin);

						       std::shared_ptr<parameterization> param_strong(ent->param);
						       
						       foundent = shared_cache->param_cachedata.find(param_strong->idx);
						       assert(foundent != shared_cache->param_cachedata.end()); /* cache entry should be in cache */
						       assert(ent == &foundent->second); /* should match what we are trying to delete */
						       // Note: cacheentry destructor being called while holding adminlock!
						       shared_cache->param_cachedata.erase(foundent); /* remove the element */ 
						       
						       } );
      
      cache_entry->second.thisptr=entry_ptr;
      cache_entry->second.snde_geom=snde_geom;
      cache_entry->second.param=param;
      cache_entry->second.TexCoordArray=new snde::OSGArray(snde_geom,(void **)&snde_geom->geom.texvertex_arrays,SNDE_INDEX_INVALID,sizeof(snde_rendercoord),2,0);

      std::vector<trm_arrayregion> initial_inputs;
      initial_inputs.push_back(trm_arrayregion(snde_geom->manager,(void **)&snde_geom->geom.uvs,param->idx,1));
      cache_entry->second.texvertex_function=
	rendering_revman->add_dependency_during_update(
						       // Function
						       // input parameters are:
						       // part
						       // triangles, based on part.firsttri and part.numtris
						       // edges, based on part.firstedge and part.numedges
						       // vertices, based on part.firstvertex and part.numvertices
						       
						       [ entry_ptr,param, shared_cache ] (snde_index newversion,std::shared_ptr<trm_dependency> dep,std::vector<rangetracker<markedregion>> &inputchangedregions) {
							 
							 // get inputs: param, uv_triangles, uv_edges, uv_vertices,
						      
							 // get output location from outputs
							 trm_arrayregion texvertex_array_out;
							 std::tie(texvertex_array_out) = extract_regions<rawregion>(dep->outputs);
							 assert((entry_ptr->TexCoordArray->elemsize==4 && (void**)entry_ptr->TexCoordArray->_ptr._float_ptr == (void **)&shared_cache->snde_geom->geom.texvertex_arrays && texvertex_array_out.array==(void **)&shared_cache->snde_geom->geom.texvertex_arrays) || (entry_ptr->TexCoordArray->elemsize==8 && (void**)entry_ptr->TexCoordArray->_ptr._double_ptr == (void **)&shared_cache->snde_geom->geom.texvertex_arrays) && texvertex_array_out.array==(void **)&shared_cache->snde_geom->geom.texvertex_arrays);
							 //// texvertex_array_out.start is counted in snde_coords, whereas
							 //// TexCoordArray is counted in vectors, so need to divide by 2
						   
							 //entry_ptr->TexCoordArray->offset = vertex_array_out.start;
							 //assert(vertex_array_out.len % 2 == 0);
							 //entry_ptr->TexCoordArray->nvec = vertex_array_out.len/2; // vertex_array_out.len is in number of coordinates; DataArray is counted in vectors
							 
							 
							 // Perform locking
							 rwlock_token_set all_locks=entry_ptr->obtain_array_locks(SNDE_UV_GEOM_UVS|SNDE_UV_GEOM_UV_TRIANGLES|SNDE_UV_GEOM_UV_EDGES|SNDE_UV_GEOM_UV_VERTICES,0,0,true,true,false);
							 //fprintf(stderr,"texvertexarray locked for write\n");
							 //fflush (stderr);
						      
							 entry_ptr->TexCoordArray->offset = texvertex_array_out.start;
							 entry_ptr->TexCoordArray->nvec = shared_cache->snde_geom->geom.uvs[param->idx].numuvtris*3; // DataArray is counted in terms of (x,y,z) vectors, so three sets of coordinates per triangle
							 assert(entry_ptr->TexCoordArray->nvec == texvertex_array_out.len/2);
							 // Should probably convert write lock to read lock and spawn this stuff off, maybe in a different thread (?) (WHY???) 						      
							 texvertexarray_from_uv_vertexarrayslocked(shared_cache->snde_geom,all_locks,dep->inputs[0].start,texvertex_array_out.start,texvertex_array_out.len,shared_cache->context,shared_cache->device,shared_cache->queue);
							 
							 
							 //entry_ptr->geom->setTexCoordArray(entry_ptr->TexCoordArray); /* tell OSG this is dirty ... (now handled by openscenegraph_geom cacheentry_function )*/
						      
							 
							 // ***!!! Should we express as tuple, then do tuple->vector conversion?
							 // ***!!! Can we extract the changed regions from the lower level notifications
							 // i.e. the cache_manager's mark_as_dirty() and/or mark_as_gpu_modified()???
							 
							 //std::vector<rangetracker<markedregion>> outputchangedregions;
							 
							 //outputchangedregions.emplace_back();
							 //outputchangedregions[0].mark_region(vertex_array_out.start,vertex_array_out.len);
							 //return outputchangedregions;
						       },
						       [ shared_cache, entry_ptr ] (std::vector<trm_struct_depend> struct_inputs,std::vector<trm_arrayregion> inputs) -> std::vector<trm_arrayregion> {
							 // Regionupdater function
							 // See Function input parameters, above
							 // Extract the first parameter (part) only
							 
							 // Perform locking
							 rwlock_token_set all_locks=entry_ptr->obtain_array_locks(SNDE_UV_GEOM_UVS,0,0,false,false,false);
							 
							 
							 // Note: We would really rather this
							 // be a reference but there is no good way to do that until C++17
							 // See: https://stackoverflow.com/questions/39103792/initializing-multiple-references-with-stdtie
							 snde_parameterization uv;
						      
							 // Construct the regions based on the part
							 std::tie(uv) = extract_regions<singleton<snde_parameterization>>(std::vector<trm_arrayregion>(inputs.begin(),inputs.begin()+1));
						      
							 std::vector<trm_arrayregion> new_inputs;
							 new_inputs.push_back(inputs[0]);
							 new_inputs.emplace_back(shared_cache->snde_geom->manager,(void **)&shared_cache->snde_geom->geom.uv_triangles,uv.firstuvtri,uv.numuvtris);
							 new_inputs.emplace_back(shared_cache->snde_geom->manager,(void **)&shared_cache->snde_geom->geom.uv_edges,uv.firstuvedge,uv.numuvedges);
							 new_inputs.emplace_back(shared_cache->snde_geom->manager,(void **)&shared_cache->snde_geom->geom.uv_vertices,uv.firstuvvertex,uv.numuvvertices);
							 return new_inputs;
							 
						       },
						       std::vector<trm_struct_depend>(), // struct_inputs
						       initial_inputs, // inputs
						       std::vector<trm_struct_depend>(), // struct_outputs
						       [ shared_cache,entry_ptr ](std::vector<trm_struct_depend> struct_inputs,std::vector<trm_arrayregion> inputs,std::vector<trm_struct_depend> struct_outputs, std::vector<trm_arrayregion> outputs) -> std::vector<trm_arrayregion> {  //, rwlock_token_set all_locks) {
							 // update_output_regions()
							 
							 rwlock_token_set all_locks=entry_ptr->obtain_array_locks(SNDE_UV_GEOM_UVS,0,0,true,true,true); // Must lock entire vertex_arrays here because we may need to reallocate it. Also when calling this we don't necessarily know the correct positioning. 
							 
									       
							 // Inputs: part
							 //         triangles
							 //         edges
							 //         vertices
							 // Outputs: vertex_arrays
							 snde_index numtris=0;
							 
							 if (inputs[0].len > 0) {
							   assert(inputs[0].len==1); // sizeof(snde_parameterization));
							   numtris=((struct snde_parameterization *)(*inputs[0].array))[inputs[0].start].numuvtris;
							 }
							 snde_index neededsize=numtris*6; // 6 vertex coords per triangle
							 
							 assert(outputs.size() <= 1);
							 if (outputs.size()==1) {
							   // already have an allocation 
							   //allocationinfo allocinfo = manager->allocators()->at((void**)&shared_cache->snde_geom->geom.vertex_arrays);
							   //snde_index alloclen = allocinfo.alloc->get_length(outputs[0].start);
							
							   snde_index alloclen = shared_cache->snde_geom->manager->get_length((void **)&shared_cache->snde_geom->geom.texvertex_arrays,outputs[0].start);
							   if (alloclen < neededsize) {
							     // too small... free this allocation... we will allocate new space below
							     //allocinfo->free(outputs[0].start);
							     shared_cache->snde_geom->manager->free((void **)&shared_cache->snde_geom->geom.texvertex_arrays,outputs[0].start);
							     outputs.erase(outputs.begin());
							   } else {
							     outputs[0].len=neededsize; // expand to needed size. 
							   }
							 }
							 
							 if (outputs.size() < 1) {										     //allocationinfo allocinfo = manager->allocators()->at(&shared_cache->snde_geom->geom.vertex_arrays);
							   std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>> allocation_vector;
							   // ***!!! Should we allocate extra space here for additional output? 
							   snde_index start;
							   std::tie(start,allocation_vector)=shared_cache->snde_geom->manager->alloc_arraylocked(all_locks,(void **)&shared_cache->snde_geom->geom.texvertex_arrays,neededsize); 
							   assert(allocation_vector.size()==1); // vertex_array shouldn't have any follower arrays
							   
							   
							   outputs.push_back(trm_arrayregion(shared_cache->snde_geom->manager,(void **)&shared_cache->snde_geom->geom.texvertex_arrays,start,neededsize));
							 }
							 
							 
							 return outputs;
						    },
						       [ shared_cache ](std::vector<trm_struct_depend> struct_inputs,std::vector<trm_arrayregion> inputs,std::vector<trm_arrayregion> outputs) {
							 if (outputs.size()==1) {
							   shared_cache->snde_geom->manager->free((void **)&shared_cache->snde_geom->geom.texvertex_arrays,outputs[0].start);
							   
							 }
							 
						       });
      
      
      return entry_ptr;
    } else {
      std::shared_ptr<osg_paramcacheentry> entry_ptr = cache_entry->second.lock();
      if (entry_ptr) {
	return entry_ptr;
      }
      else {
	// obsolete cache entry 
	param_cachedata.erase(cache_entry);
	adminlock.unlock();
	// recursive call to make a new cache entry
	return lookup(rendering_revman,param);
	
      }
    }
  }



};

}

#endif  // SNDE_OPENSCENEGRAPH_PARAMETERIZATION_HPP
