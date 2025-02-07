#ifdef SNDE_OPENCL
#include "snde/snde_types_h.h"
#include "snde/geometry_types_h.h"
#include "snde/vecops_h.h"
#include "snde/kdtree_knn_c.h"
#endif // SNDE_OPENCL


#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/geometrydata.h"

#include "snde/recstore.hpp"
#include "snde/recmath_cppfunction.hpp"
#include "snde/graphics_recording.hpp"
#include "snde/geometry_processing.hpp"

#ifdef SNDE_OPENCL
#include "snde/opencl_utils.hpp"
#include "snde/openclcachemanager.hpp"
#include "snde/recmath_compute_resource_opencl.hpp"
#include "snde/graphics_storage.hpp" // for snde_doubleprec_coords()
#endif


#include "snde/kdtree.hpp"
#include "snde/kdtree_knn.h"

namespace snde {


  int kce_compare(const void *kce_1,const void *kce_2)
  {
    const kdtree_construction_entry *kce1=(const kdtree_construction_entry *)kce_1;
    const kdtree_construction_entry *kce2=(const kdtree_construction_entry *)kce_2;

    if (kce1->depth < kce2->depth) {
      return -1;
    } else if (kce1->depth > kce2->depth) {
      return 1;
    } else if (kce1->cutting_vertex < kce2->cutting_vertex) {
      return -1;
    } else if (kce1->cutting_vertex > kce2->cutting_vertex) {
      return 1; 
    } else {
      assert(kce1->cutting_vertex==kce2->cutting_vertex);
      assert(kce1==kce2);
      return 0;
    }
	       
  }
  
  
  
  class kdtree_calculation: public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<snde_coord3>>> {
  public:
    kdtree_calculation(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec(rss,inst)
    {
      
    }
    
    // use default for decide_new_revision
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<ndtyped_recording_ref<snde_coord3>> vertices)
    {
      snde_index numvertices = vertices->layout.flattened_length();
      
      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numvertices*100, // data_bytes for transfer
							numvertices*(100), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
	};
      return std::make_pair(option_list,nullptr);
    }
  
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<ndtyped_recording_ref<snde_coord3>> vertices) 
    {
      // define_recs code
      //printf("define_recs()\n"); 
      std::shared_ptr<multi_ndarray_recording> result_rec;
      vertices->assert_no_scale_or_offset(this->inst->definition->definition_command);

      result_rec = create_recording_math<multi_ndarray_recording>(get_result_channel_path(0),rss,1);      
      result_rec->define_array(0,SNDE_RTN_SNDE_KDNODE,"vertex_kdtree");
      std::shared_ptr<ndtyped_recording_ref<snde_kdnode>> result_ref = std::dynamic_pointer_cast<ndtyped_recording_ref<snde_kdnode>>(result_rec->reference_ndarray(0));
      
      
      return std::make_shared<metadata_function_override_type>([ this,result_ref,vertices ]() {
	// metadata code
	std::shared_ptr<constructible_metadata> metadata = std::make_shared<constructible_metadata>();
	
	// don't mark metadata done here because we need to document the max depth
	
	return std::make_shared<lock_alloc_function_override_type>([ this,result_ref,vertices,metadata ]() {
	  // lock_alloc code
	  snde_index numvertices = vertices->layout.flattened_length();
	  
	  //std::shared_ptr<storage_manager> graphman = result_rec->assign_storage_manager();
	  
	  result_ref->allocate_storage({numvertices},false);
	  
	  
	  //parts_ref = recording->reference_ndarray("parts")

	  std::shared_ptr<meshed_part_recording> meshedpart; // will be meshedpart or nullptr depending on whether we are part of a meshed part recording
	  meshedpart = std::dynamic_pointer_cast<meshed_part_recording>(vertices->rec); 
	  

	  // ***!!! NOTE: We should probably consider putting the kdtree into a special graphics
	  // storage array and writing the indices into the part object ***!!!
	  
	  // locking is only required for certain recordings
	  // with special storage under certain conditions,
	  // however it is always good to explicitly request
	  // the locks, as the locking is a no-op if
	  // locking is not actually required.
	  std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> recrefs_to_lock = {
	    { vertices, false }, // first element is recording_ref, 2nd parameter is false for read, true for write
	    { result_ref, true }
	  };

	  if (meshedpart) {
	    // if vertices are from a meshed part, we will also write into our section
	    // of the struct snde_part so we need to lock it for write.
	    recrefs_to_lock.push_back(std::make_pair(meshedpart->reference_ndarray("parts"),true));
	  }
	  
	  rwlock_token_set locktokens = lockmgr->lock_recording_refs(recrefs_to_lock);
	  
	  return std::make_shared<exec_function_override_type>([ this,locktokens, result_ref,vertices,metadata,meshedpart ]() {
	    // exec code
	    snde_index numvertices = vertices->layout.flattened_length();
	    
	    kdtree_vertex<snde_coord3> *tree_vertices=(kdtree_vertex<snde_coord3> *)malloc(sizeof(kdtree_vertex<snde_coord3>)*numvertices);
	    
	    snde_index start_position=0;
	    snde_index end_position=numvertices;
	    
	    for (snde_index vertidx=0;vertidx < numvertices;vertidx++) {
	      tree_vertices[vertidx].original_index = vertidx;
	      tree_vertices[vertidx].pos=vertices->element(vertidx,false);
	    }
	    
	    
	    // Create kdtree_construction
	    kdtree_construction_entry *orig_tree=(kdtree_construction_entry *)malloc(sizeof(kdtree_construction_entry)*numvertices);;
	    snde_index tree_nextpos = 0;
	    
	    unsigned max_depth=0;
	    build_subkdtree<snde_coord3>(tree_vertices,orig_tree,&tree_nextpos,0,numvertices,3,0,&max_depth);
	    assert(tree_nextpos == numvertices);
	    
	    // Create copy of tree and sort it with the primary key being the depth.
	    // We do this to make the lookup process more cache-friendly
	    
	    // Copy kdtree_construction
	    kdtree_construction_entry *copy_tree=(kdtree_construction_entry *)malloc(sizeof(kdtree_construction_entry)*numvertices);
	    memcpy(copy_tree,orig_tree,sizeof(kdtree_construction_entry)*numvertices);
	    qsort(copy_tree,numvertices,sizeof(kdtree_construction_entry),kce_compare);
	    
	    // Go through the sorted copy, and modify the entry_index in the original
	    // to identify the sorted address
	    for (snde_index vertidx=0;vertidx < numvertices;vertidx++) {
	      orig_tree[copy_tree[vertidx].entry_index].entry_index = vertidx; 
	    }
	    // Now, the entry_index in orig_tree gives the index in the sorted copy
	    
	    // Now fix up the subtree indices, creating the result array from the copy
	    // following the sorted order
	    for (snde_index treeidx=0;treeidx < numvertices;treeidx++) {
	      snde_kdnode &treenode = result_ref->element(treeidx,false);
	      
	      // Need to use the orig tree entry_indexes to identify the sorted addresses
	      // for the left and right subtrees.
	      treenode.cutting_vertex = copy_tree[treeidx].cutting_vertex;
	      if (copy_tree[treeidx].left_subtree==SNDE_INDEX_INVALID) {
		treenode.left_subtree = SNDE_INDEX_INVALID;
	      } else {
		treenode.left_subtree = orig_tree[copy_tree[treeidx].left_subtree].entry_index;
	      }
	      
	      if (copy_tree[treeidx].right_subtree==SNDE_INDEX_INVALID) {
		treenode.right_subtree = SNDE_INDEX_INVALID;
	      } else {
		treenode.right_subtree = orig_tree[copy_tree[treeidx].right_subtree].entry_index;
	      }
	      
	    }
	    
	    
	    free(copy_tree);
	    free(tree_vertices);
	    free(orig_tree);

	    if (meshedpart) { 
	      assert(result_ref->rec->storage_manager == meshedpart->storage_manager); // must both be using the same graphics storage manager for this to be OK
	      // modify our indexes in partstruct (these are our responsibility)
	      snde_part &partstruct = meshedpart->reference_typed_ndarray<snde_part>("parts")->element(0);
	      partstruct.first_vertex_kdnode = result_ref->storage->base_index;
	      partstruct.num_vertex_kdnodes = result_ref->storage->nelem;
	      meshedpart->reference_ndarray("parts")->storage->mark_as_modified(nullptr,0,1,true); // indicate that we have modified this first element of "parts", invalidating caches. 

	    }
	      
	    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready()

	    metadata->AddMetaDatum(metadatum("kdtree_max_depth",(uint64_t)max_depth));
	    
	    result_ref->rec->metadata=metadata;
	    result_ref->rec->mark_metadata_done();
	    result_ref->rec->mark_data_ready();
	    
	  }); 
	});
      });
    };
    
  };
  
  std::shared_ptr<math_function> define_kdtree_calculation_function()
  {
    return std::make_shared<cpp_math_function>("snde.kdtree_calculation",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<kdtree_calculation>(rss,inst);
    });
    
  }


  SNDE_OCL_API std::shared_ptr<math_function> kdtree_calculation_function = define_kdtree_calculation_function();
  
  static int registered_kdtree_calculation_function = register_math_function(kdtree_calculation_function);


  void instantiate_vertex_kdtree(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags)
  {
    std::string context = recdb_path_context(loaded_geom->info->name);

    bool withmapping_flag = false;

    //if (all_processing_tags->find("vertex_kdtree_withmapping") != all_processing_tags->end()) {
    //   withmapping_flag = true; // withmapping flags enables generating a mapping from kdtree index to 
    //}
    
    std::shared_ptr<instantiated_math_function> instantiated = kdtree_calculation_function->instantiate( {
	std::make_shared<math_parameter_recording>("meshed","vertices")
      },
      {
	std::make_shared<std::string>("vertex_kdtree")
      },
      context,
      false, // is_mutable
      false, // ondemand
      false, // mdonly
      std::make_shared<math_definition>("instantiate_vertex_kdtree()"),
      {},
      nullptr);


    trans->recdb->add_math_function(trans,instantiated,true); // kdtree is generally hidden by default
    loaded_geom->processed_relpaths.emplace("kdtree","kdtree");
    
  }
  
  static int registered_vertex_kdtree_processor = register_geomproc_math_function("vertex_kdtree",instantiate_vertex_kdtree);
  //static int registered_vertex_kdtree_withmapping_processor = register_geomproc_math_function("vertex_kdtree_withmapping",instantiate_vertex_kdtree);
  
  


#ifdef SNDE_OPENCL
  static opencl_program knn_calculation_opencl("snde_kdtree_knn_opencl", { snde_types_h, geometry_types_h, vecops_h, kdtree_knn_c });
    
#endif // SNDE_OPENCL

#ifdef SNDE_OPENCL
  cl::Event perform_inline_ocl_knn_calculation(std::shared_ptr<assigned_compute_resource_opencl> opencl_resource,rwlock_token_set locktokens, std::shared_ptr<ndtyped_recording_ref<snde_kdnode>> kdtree, std::shared_ptr<ndtyped_recording_ref<uint32_t>> nodemask,std::shared_ptr<ndtyped_recording_ref<snde_coord3>> kdtree_vertices, OpenCLBuffers &Buffers,std::shared_ptr<ndtyped_recording_ref<snde_index>> search_point_indices,std::shared_ptr<ndtyped_recording_ref<snde_coord3>> search_points,cl::Event search_points_ready,std::shared_ptr<ndtyped_recording_ref<snde_index>> result_ref) // returns event that indicates result is ready (on GPU)
  {
    // NOTE: Must keep code parallel with perform_knn_calculation(), below!!!***
    snde_index num_search_points = search_points->layout.flattened_length();

    snde_index num_search_point_indices = num_search_points; // set equal to num_search_points if search_point_indices==nullptr
    if (search_point_indices) {
      // iterate over only search_points specified in search_point_indices
      num_search_point_indices = search_point_indices->layout.flattened_length();

      if (!search_point_indices->layout.is_contiguous()) {
	throw snde_error("search_point_indices array must be contiguous");
      }
    }

    if (!kdtree_vertices->layout.is_contiguous()) {
      throw snde_error("vertices array must be contiguous");
    }
    if (!kdtree->layout.is_contiguous()) {
      throw snde_error("kdtree array must be contiguous");
    }
    if (!search_points->layout.is_contiguous()) {
      throw snde_error("search_points array must be contiguous");
    }
    
    uint64_t max_depth=kdtree->rec->metadata->GetMetaDatumUnsigned("kdtree_max_depth",200);

    cl::Device knn_dev = opencl_resource->devices.at(0);
    cl::Kernel knn_kern = knn_calculation_opencl.get_kernel(opencl_resource->context,knn_dev);


    uint32_t stacksize_per_workitem = max_depth+1; // uint32_t because this is passed as a kernel arg, below
    size_t nodestacks_octwords_per_workitem = (stacksize_per_workitem*sizeof(snde_index) + 7)/8;
    size_t statestacks_octwords_per_workitem = (stacksize_per_workitem*sizeof(uint8_t) + 7)/8;
    size_t bboxstacks_octwords_per_workitem = (stacksize_per_workitem*(sizeof(snde_coord)*2) + 7)/8;
    size_t local_memory_octwords_per_workitem = nodestacks_octwords_per_workitem + statestacks_octwords_per_workitem + bboxstacks_octwords_per_workitem;
    
    //opencl_layout_workgroups_for_localmemory_1D(knn_dev,knn_kern,local_memory_octwords_per_workitem,num_search_point_indices);
    
    size_t kern_work_group_size,kernel_global_work_items;
    
    std::tie(kern_work_group_size,kernel_global_work_items) = opencl_layout_workgroups_for_localmemory_1D(knn_dev,
													  knn_kern,
													  local_memory_octwords_per_workitem,
													  num_search_point_indices);
    
      
    Buffers.AddBufferAsKernelArg(kdtree,knn_kern,0,false,false);
    if (nodemask) {
      Buffers.AddBufferAsKernelArg(nodemask,knn_kern,1,false,false);
    } else {
      knn_kern.setArg(1,sizeof(cl_mem),nullptr); // pass null for nodemask    
    }
    
    Buffers.AddBufferAsKernelArg(kdtree_vertices,knn_kern,2,false,false);
    // add local memory arrays 
    knn_kern.setArg(3,nodestacks_octwords_per_workitem*8*kern_work_group_size,nullptr);
    knn_kern.setArg(4,statestacks_octwords_per_workitem*8*kern_work_group_size,nullptr);
    knn_kern.setArg(5,bboxstacks_octwords_per_workitem*8*kern_work_group_size,nullptr);
      
    knn_kern.setArg(6,sizeof(stacksize_per_workitem),&stacksize_per_workitem);
    if (search_point_indices) {
      Buffers.AddBufferAsKernelArg(search_point_indices,knn_kern,7,false,false);
    } else {
      knn_kern.setArg(7,sizeof(cl_mem),nullptr);
    }
    Buffers.AddBufferAsKernelArg(search_points,knn_kern,8,false,false);
    Buffers.AddBufferAsKernelArg(result_ref,knn_kern,9,true,true);
    uint32_t opencl_ndim=3;
    knn_kern.setArg(10,sizeof(opencl_ndim),&opencl_ndim);
    uint32_t opencl_max_depth=max_depth;
    knn_kern.setArg(11,sizeof(opencl_max_depth),&opencl_max_depth);
    snde_index max_workitem_plus_one = num_search_point_indices; // remember this is set equal to num_search_points if search_point_indices==nullptr
    knn_kern.setArg(12,sizeof(max_workitem_plus_one),&max_workitem_plus_one);
      
    
      
    cl::Event kerndone;
    std::vector<cl::Event> FillEvents=Buffers.FillEvents();

    if (search_points_ready.get()) {
      FillEvents.push_back(search_points_ready);
    }
    
    cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(knn_kern,{},{ kernel_global_work_items },{ kern_work_group_size },&FillEvents,&kerndone);
    if (err != CL_SUCCESS) {
      throw openclerror(err,"Error enqueueing kernel");
    }
    opencl_resource->queues.at(0).flush(); /* trigger execution */
    // mark that the kernel has modified result_rec
    Buffers.BufferDirty(result_ref);
    
    return kerndone;
  }
  
#endif // SNDE_OPENCL

  void perform_knn_calculation(std::shared_ptr<assigned_compute_resource> compute_resource,rwlock_token_set locktokens, std::shared_ptr<ndtyped_recording_ref<snde_kdnode>> kdtree, std::shared_ptr<ndtyped_recording_ref<uint32_t>> nodemask,std::shared_ptr<ndtyped_recording_ref<snde_coord3>> kdtree_vertices, std::shared_ptr<ndtyped_recording_ref<snde_index>> search_point_indices,std::shared_ptr<ndtyped_recording_ref<snde_coord3>> search_points,std::shared_ptr<ndtyped_recording_ref<snde_index>> result_ref) // NOTE: nodemask and/or search_point_indices may be nullptr
  {
    // NOTE: Must keep code parallel with perform_inline_ocl_knn_calculation(), above!!!***
    snde_index num_search_points = search_points->layout.flattened_length();
    snde_index num_search_point_indices = num_search_points; // set equal to num_search_points if search_point_indices==nullptr
    if (search_point_indices) {

      // iterate over only search_points specified in search_point_indices
      num_search_point_indices = search_point_indices->layout.flattened_length();
      
      if (!search_point_indices->layout.is_contiguous()) {
	throw snde_error("search_point_indices array must be contiguous");
      }
    }
    
    if (!kdtree_vertices->layout.is_contiguous()) {
      throw snde_error("vertices array must be contiguous");
    }
    if (!kdtree->layout.is_contiguous()) {
      throw snde_error("kdtree array must be contiguous");
    }
    if (!search_points->layout.is_contiguous()) {
      throw snde_error("search_points array must be contiguous");
    }
    
    uint64_t max_depth=kdtree->rec->metadata->GetMetaDatumUnsigned("kdtree_max_depth",200);
    
#ifdef SNDE_OPENCL
    std::shared_ptr<assigned_compute_resource_opencl> opencl_resource=std::dynamic_pointer_cast<assigned_compute_resource_opencl>(compute_resource);
    if (opencl_resource && search_point_indices && search_point_indices->layout.flattened_length() > 0) {
      cl::Device knn_dev = opencl_resource->devices.at(0);

      OpenCLBuffers Buffers(opencl_resource->oclcache,opencl_resource->context,knn_dev,locktokens);

      cl::Event knn_calculation_done = perform_inline_ocl_knn_calculation(opencl_resource,
									  locktokens,
									  kdtree,
									  nodemask,
									  kdtree_vertices,
									  Buffers,
									  search_point_indices,
									  search_points,
									  cl::Event(), // pass empty event because the normal buffer load will cover it
									  result_ref);
									        
      // wait for kernel execution and transfers to complete
      Buffers.RemBuffers(knn_calculation_done,knn_calculation_done,true);
      
    } else {	    
#endif // SNDE_OPENCL
      
      snde_index *nodestack=(snde_index *)malloc((max_depth+1)*sizeof(snde_index));
      uint8_t *statestack=(uint8_t *)malloc((max_depth+1)*sizeof(uint8_t));
      snde_coord *bboxstack=(snde_coord *)malloc((max_depth+1)*sizeof(snde_coord)*2);

      if (search_point_indices) {
	// iterate over only search_points specified in search_point_indices
	snde_index num_search_point_indices = search_point_indices->layout.flattened_length();
	//snde_index *search_point_indices_raw = search_point_indices->shifted_arrayptr();
	
	for (snde_index searchidxidx=0;searchidxidx < num_search_point_indices;searchidxidx++) {
	  snde_index searchidx = search_point_indices->element(searchidxidx);
	  result_ref->element({searchidxidx}) = snde_kdtree_knn_one(kdtree->shifted_arrayptr(),
								    nodemask ? nodemask->shifted_arrayptr() : nullptr,
								    (snde_coord *)kdtree_vertices->shifted_arrayptr(),
								    nodestack,
								    statestack,
								    bboxstack,
								    &search_points->element(searchidx,false).coord[0],
								    //nullptr,
								    3,
								    max_depth,
								    searchidx);
	  
	}
	
      } else {
	// iterate over all search_points
	for (snde_index searchidx=0;searchidx < num_search_points;searchidx++) {
	  result_ref->element({searchidx}) = snde_kdtree_knn_one(kdtree->shifted_arrayptr(),
								 nodemask ? nodemask->shifted_arrayptr() : nullptr,
								 (snde_coord *)kdtree_vertices->shifted_arrayptr(),
								 nodestack,
								 statestack,
								 bboxstack,
								 &search_points->element(searchidx,false).coord[0],
								 //nullptr,
								 3,
								 max_depth,
								 searchidx);
	}
      }
      
      free(bboxstack);
      free(statestack);
      free(nodestack);
#ifdef SNDE_OPENCL
    }
#endif // SNDE_OPENCL
    
  }
  

  class knn_calculation: public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<snde_coord3>>,std::shared_ptr<ndtyped_recording_ref<snde_kdnode>>,std::shared_ptr<ndtyped_recording_ref<snde_coord3>>> {
    // parameters are vertices, kdtree built on those vertices, and points for searching
  public:
    knn_calculation(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec(rss,inst)
    {
      
    }
    
    // use default for decide_new_revision
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<ndtyped_recording_ref<snde_coord3>> vertices, std::shared_ptr<ndtyped_recording_ref<snde_kdnode>> kdtree,std::shared_ptr<ndtyped_recording_ref<snde_coord3>> search_points)
    {
      snde_index numvertices = vertices->layout.flattened_length();
      snde_index treesize = kdtree->layout.flattened_length();
      snde_index num_search_points = search_points->layout.flattened_length();
      
      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numvertices*sizeof(snde_coord3)+treesize*sizeof(snde_kdnode)+num_search_points*sizeof(snde_coord3), // data_bytes for transfer
							num_search_points*log(numvertices)*10.0, // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
							   0, //metadata_bytes
							   numvertices*sizeof(snde_coord3)+treesize*sizeof(snde_kdnode)+num_search_points*sizeof(snde_coord3), // data_bytes for transfer
							   0, // cpu_flops
							   num_search_points*log(numvertices)*10.0, // gpuflops
							   1, // max effective cpu cores
							   1, // useful_cpu_cores (min # of cores to supply
							   snde_doubleprec_coords()), // requires_doubleprec 
#endif // SNDE_OPENCL
	  
	};
      return std::make_pair(option_list,nullptr);
    }
  
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<ndtyped_recording_ref<snde_coord3>> vertices, std::shared_ptr<ndtyped_recording_ref<snde_kdnode>> kdtree,std::shared_ptr<ndtyped_recording_ref<snde_coord3>> search_points) 
    {
      // define_recs code
      //printf("define_recs()\n"); 
      std::shared_ptr<multi_ndarray_recording> result_rec;
      search_points->assert_no_scale_or_offset(this->inst->definition->definition_command);

      //result_rec = create_recording_math<multi_ndarray_recording>(get_result_channel_path(0),rss,1);      
      //result_rec->define_array(0,SNDE_RTN_SNDE_KDNODE,"vertex_kdtree");
      std::shared_ptr<ndtyped_recording_ref<snde_index>> result_ref = create_typed_ndarray_ref_math<snde_index>(this->get_result_channel_path(0),this->rss);
      
      
      return std::make_shared<metadata_function_override_type>([ this,result_ref,vertices,kdtree,search_points ]() {
	// metadata code
	std::unordered_map<std::string,metadatum> metadata;
	//printf("metadata()\n");
	
	result_ref->rec->metadata=std::make_shared<immutable_metadata>(metadata);
	result_ref->rec->mark_metadata_done();
	
	return std::make_shared<lock_alloc_function_override_type>([ this,result_ref,vertices,kdtree,search_points ]() {
	  // lock_alloc code
	  snde_index num_search_points = search_points->layout.flattened_length();
	  //snde_index numvertices = vertices->layout.flattened_length();
	  
	  //std::shared_ptr<storage_manager> graphman = result_rec->assign_storage_manager();
	  
	  result_ref->allocate_storage({num_search_points},false);
	  
	  
	  
	  // locking is only required for certain recordings
	  // with special storage under certain conditions,
	  // however it is always good to explicitly request
	  // the locks, as the locking is a no-op if
	  // locking is not actually required.
	  rwlock_token_set locktokens = lockmgr->lock_recording_refs({
	      { vertices, false }, // first element is recording_ref, 2nd parameter is false for read, true for write
	      { kdtree, false }, // first element is recording_ref, 2nd parameter is false for read, true for write
	      { search_points, false }, // first element is recording_ref, 2nd parameter is false for read, true for write
	      { result_ref, true }
	    },
#ifdef SNDE_OPENCL
	    std::dynamic_pointer_cast<assigned_compute_resource_opencl>(compute_resource) ? true:false
#else
	    false
#endif
	    );
	  
	  return std::make_shared<exec_function_override_type>([ this,locktokens, result_ref,vertices,kdtree,search_points ]() {
	    // exec code
	    //snde_index numvertices = vertices->layout.flattened_length();
	    perform_knn_calculation(compute_resource,locktokens,kdtree,nullptr,vertices,nullptr,search_points,result_ref);

	    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	    result_ref->rec->mark_data_ready();
	    
	  }); 
	});
      });
    };
    
  };
  
  std::shared_ptr<math_function> define_knn_calculation_function()
  {
    return std::make_shared<cpp_math_function>("snde.knn_calculation",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<knn_calculation>(rss,inst);
    });
    
  }
  
  SNDE_OCL_API std::shared_ptr<math_function> knn_calculation_function = define_knn_calculation_function();


  
  static int registered_knn_calculation_function = register_math_function(knn_calculation_function);

  
  
};
