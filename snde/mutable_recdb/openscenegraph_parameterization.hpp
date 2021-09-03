#include <vector>
#include <memory>

#include "snde/revman_parameterization.hpp"

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

  //if (uv.numuvtris > 47759) {
  //  fprintf(stderr,"Triangle 47759 first vertex: (%f,%f)\n",geom->geom.texvertex_arrays[outaddr + 47759*6],geom->geom.texvertex_arrays[outaddr + 47759*6 + 1]);
  //}
  
  
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


  
  std::shared_ptr<osg_paramcacheentry> lock()
  {
    return shared_from_this();
  }
    
  ~osg_paramcacheentry()
  {
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
      
      std::weak_ptr<osg_paramcacheentry> entry_ptr_weak(entry_ptr);
      
      std::vector<trm_struct_depend> struct_inputs;
      struct_inputs.emplace_back(parameterization_dependency(rendering_revman,param));
      
      //std::vector<trm_arrayregion> initial_inputs;
      //initial_inputs.push_back(trm_arrayregion(snde_geom->manager,(void **)&snde_geom->geom.uvs,param->idx,1));
      cache_entry->second.texvertex_function=
	rendering_revman->add_dependency_during_update(
						       struct_inputs,
						       std::vector<trm_arrayregion>(), // inputs
						       std::vector<trm_struct_depend>(), // struct_outputs
						       // Function
						       // input parameters are:
						       // part
						       // triangles, based on part.firsttri and part.numtris
						       // edges, based on part.firstedge and part.numedges
						       // vertices, based on part.firstvertex and part.numvertices
						       
						       [ entry_ptr_weak,shared_cache ] (snde_index newversion,std::shared_ptr<trm_dependency> dep,const std::set<trm_struct_depend_key> &inputchangedstructs,const std::vector<rangetracker<markedregion>> &inputchangedregions,unsigned actions) {
							 
							 std::shared_ptr<osg_paramcacheentry> entry_ptr = entry_ptr_weak.lock();
							 std::shared_ptr<parameterization> param=get_parameterization_dependency(dep->struct_inputs[0]);

							 if (!entry_ptr || !param) {
							   // invalid inputs
							   // update inputs and outputs vectors to be empty as appropriate
							   
							   std::vector<trm_arrayregion> new_inputs;
							   dep->update_inputs(new_inputs);
							   
							   if (actions & STDA_IDENTIFYOUTPUTS) {
							     std::vector<trm_arrayregion> new_outputs;
							     dep->update_outputs(new_outputs);
							   }
							   return;
							   
							 }

							 std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
							 std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(shared_cache->snde_geom->manager->locker); // new locking process

							 // lock the parameterization for read
							 obtain_graph_lock(lockprocess,param,
									   std::vector<std::string>(),
									   std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>>(),
									   nullptr,"", // recdb and context only relevant for components which might have children we want to access
									   SNDE_INFOSTORE_PARAMETERIZATIONS|SNDE_UV_GEOM_UVS|((actions & STDA_EXECUTE) ? (SNDE_UV_GEOM_UV_TRIANGLES|SNDE_UV_GEOM_UV_EDGES|SNDE_UV_GEOM_UV_VERTICES):0),
									   0);
							 //param->obtain_lock(lockprocess);

							 // lock the UV data as needed
							 //if (actions & STDA_EXECUTE) {
							   
							 //  param->obtain_uv_lock(lockprocess,SNDE_UV_GEOM_UVS|SNDE_UV_GEOM_UV_TRIANGLES|SNDE_UV_GEOM_UV_EDGES|SNDE_UV_GEOM_UV_VERTICES);
							 //} else {
							 //  param->obtain_uv_lock(lockprocess,SNDE_UV_GEOM_UVS);
							 //  
							 //}

							 snde_parameterization &uvstruct = shared_cache->snde_geom->geom.uvs[param->idx];
							 if (actions & STDA_IDENTIFYOUTPUTS) {
							   holder->store_alloc(dep->realloc_output_if_needed(lockprocess,shared_cache->snde_geom->manager,0,(void **)&shared_cache->snde_geom->geom.texvertex_arrays,uvstruct.numuvtris*6,"texvertex_arrays"));
							   
							 }
							 rwlock_token_set all_locks=lockprocess->finish();

							 // inputs are: parameterization, uv_triangles,
							 // uv_edges, and uv_vertices
							 std::vector<trm_arrayregion> new_inputs;
							 new_inputs.emplace_back(shared_cache->snde_geom->manager,(void **)&shared_cache->snde_geom->geom.uvs,param->idx,1);
							 new_inputs.emplace_back(shared_cache->snde_geom->manager,(void **)&shared_cache->snde_geom->geom.uv_triangles,uvstruct.firstuvtri,uvstruct.numuvtris);
							 new_inputs.emplace_back(shared_cache->snde_geom->manager,(void **)&shared_cache->snde_geom->geom.uv_edges,uvstruct.firstuvedge,uvstruct.numuvedges);
							 new_inputs.emplace_back(shared_cache->snde_geom->manager,(void **)&shared_cache->snde_geom->geom.uv_vertices,uvstruct.firstuvvertex,uvstruct.numuvvertices);
							 dep->update_inputs(new_inputs);

							 if (actions & STDA_IDENTIFYOUTPUTS) {
							   snde_index neededsize=uvstruct.numuvtris*6; // 6 vertex coords per triangle
							   std::vector<trm_arrayregion> new_outputs;
							   dep->add_output_to_array(new_outputs,shared_cache->snde_geom->manager,holder,0,(void **)&shared_cache->snde_geom->geom.texvertex_arrays,"texvertex_arrays");
							   dep->update_outputs(new_outputs);

							   assert(new_outputs.at(0).array==(void **)&shared_cache->snde_geom->geom.texvertex_arrays);
							   fprintf(stderr,"texvertex array=0x%llx; start=%d; len=%d\n",(unsigned long long)((void *)&shared_cache->snde_geom->geom.texvertex_arrays),(int)new_outputs.at(0).start,(int)new_outputs.at(0).len);
							   
							   if (actions & STDA_EXECUTE) {
							     // function code
							     
							     // ***!!! Note: TexCoordArray->offset and nvec
							     // also set in openscenegraph_geom.hpp. This
							     // is redundant and should probably be cleaned up
							     entry_ptr->TexCoordArray->offset = dep->outputs[0].start;
							     entry_ptr->TexCoordArray->nvec = uvstruct.numuvtris*3; // DataArray is counted in terms of (x,y) vectors, so three sets of coordinates per triangle
							     assert(entry_ptr->TexCoordArray->nvec == dep->outputs[0].len/2);
							     // Should probably convert write lock to read lock and spawn this stuff off, maybe in a different thread (?) (WHY???) 						      
							     texvertexarray_from_uv_vertexarrayslocked(shared_cache->snde_geom,all_locks,dep->inputs[0].start,dep->outputs[0].start,dep->outputs[0].len,shared_cache->context,shared_cache->device,shared_cache->queue);
							     
							   }
							 }
						       },
						       [ entry_ptr_weak ] (trm_dependency  *dep)  {
							 // cleanup function

							 std::shared_ptr<osg_paramcacheentry> entry_ptr = entry_ptr_weak.lock();
							 if (entry_ptr && entry_ptr->TexCoordArray) {
							   entry_ptr->TexCoordArray->nvec=0;
							   entry_ptr->TexCoordArray->offset=SNDE_INDEX_INVALID;
							 }

							 // free our outputs
							 std::vector<trm_arrayregion> new_outputs;
							 dep->free_output(new_outputs,0);
							 dep->update_outputs(new_outputs);


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
