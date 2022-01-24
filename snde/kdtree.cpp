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
	  std::make_shared<compute_resource_option_cpu>(0, //metadata_bytes 
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
	  
	  
	  // locking is only required for certain recordings
	  // with special storage under certain conditions,
	  // however it is always good to explicitly request
	  // the locks, as the locking is a no-op if
	  // locking is not actually required.
	  rwlock_token_set locktokens = lockmgr->lock_recording_refs({
	      { vertices, false }, // first element is recording_ref, 2nd parameter is false for read, true for write
	      { result_ref, true }
	    });
	  
	  return std::make_shared<exec_function_override_type>([ this,locktokens, result_ref,vertices,metadata ]() {
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
	    
	    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_as_ready()

	    metadata->AddMetaDatum(metadatum("kdtree_max_depth",(uint64_t)max_depth));
	    
	    result_ref->rec->metadata=metadata;
	    result_ref->rec->mark_metadata_done();
	    result_ref->rec->mark_as_ready();
	    
	  }); 
	});
      });
    };
    
  };
  
  std::shared_ptr<math_function> define_kdtree_calculation_function()
  {
    return std::make_shared<cpp_math_function>([] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<kdtree_calculation>(rss,inst);
    });
    
  }


  
  static int registered_kdtree_calculation_function = register_math_function("spatialnde2.kdtree_calculation",define_kdtree_calculation_function());





#ifdef SNDE_OPENCL
  static opencl_program knn_calculation_opencl("snde_kdtree_knn_opencl", { snde_types_h, geometry_types_h, vecops_h, kdtree_knn_c });
  static opencl_program testcase_opencl("snde_kdtree_knn_opencl", { R"FOO(

  struct snde_kdnode {
    unsigned long cutting_vertex;
    unsigned long left_subtree;
    unsigned long  right_subtree; 
  };



#ifdef __OPENCL_VERSION__
__kernel void snde_kdtree_knn_opencl(__local unsigned long *nodestacks, // (stacksize_per_workitem)*sizeof(unsigned long)*work_group_size
				     __local unsigned char *statestacks, // (stacksize_per_workitem)*sizeof(unsigned char)*work_group_size
				     __local float *bboxstacks, // (stacksize_per_workitem)*sizeof(float)*2*work_group_size,
				     unsigned stacksize_per_workitem)   // stacksize_per_workitem must be at least max_depth+1!!!
                                     
{ 
  unsigned long find_index = get_global_id(0);

  //__local float bboxstacks[288];



  size_t nodestacks_octwords_per_workitem = 6; //(stacksize_per_workitem*sizeof(unsigned long) + 7)/8;
  size_t statestacks_octwords_per_workitem = 1; //(stacksize_per_workitem*sizeof(unsigned char) + 7)/8;
  size_t bboxstacks_octwords_per_workitem = 6;//(stacksize_per_workitem*(sizeof(float)*2) + 7)/8;
  
  
  
  __local unsigned long *nodestack = (__local unsigned long *)(((__local unsigned char *)nodestacks) + get_local_id(0)*nodestacks_octwords_per_workitem*8);
  __local unsigned char *statestack = (__local unsigned char *)(((__local unsigned char *)statestacks) + get_local_id(0)*statestacks_octwords_per_workitem*8);
  __local float *bboxstack = (__local float *)(((__local unsigned char *)bboxstacks) + get_local_id(0)*2*bboxstacks_octwords_per_workitem*8);
  
  
  {
    unsigned depth=0;
    unsigned dimnum=0;
    unsigned previous_dimnum=0;

    printf("no=%u so=%u bo=%u\n",(unsigned)nodestacks_octwords_per_workitem,(unsigned)statestacks_octwords_per_workitem,(unsigned)bboxstacks_octwords_per_workitem);
    
    nodestack[0] = 0; // initial tree entry
    
#ifdef __OPENCL_VERSION__
    //if (get_global_id(0) < 2) {
    printf("initial global id: %d  depth = %d, node@0x%lx = %d\n",(int)get_global_id(0),(int)depth,(unsigned long)&nodestack[depth],(int)nodestack[depth]);
    printf("initial gid %d bboxstack@0x%lx; statestacks@0x%lx\n",(int)get_global_id(0),(unsigned long)&bboxstack[0],(unsigned long)&statestack[0]);
    
    //}
#endif
    
    
#ifdef __OPENCL_VERSION__
    //if (get_global_id(0) > 2) {
    bboxstack[0] = 1e3f;//my_infnan(-ERANGE); // -inf
    bboxstack[1] = 1e9f;//my_infnan(ERANGE); // inf
    //}
#endif
#ifdef __OPENCL_VERSION__
    //if (get_global_id(0) < 2) {
    printf("post_bboxstack global id: %d  depth = %d, node@0x%lx = %d\n",(int)get_global_id(0),(int)depth,(unsigned long)&nodestack[depth],(int)nodestack[depth]);
    printf("post_bboxstack gid %d bboxstack@0x%lx; statestacks@0x%lx\n",(int)get_global_id(0),(unsigned long)&bboxstack[0],(unsigned long)&statestack[0]);
    
    //}
#endif
  }
}    
#endif
    
)FOO"});
    
#endif // SNDE_OPENCL



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
	  std::make_shared<compute_resource_option_cpu>(0, //metadata_bytes 
							numvertices*sizeof(snde_coord3)+treesize*sizeof(snde_kdnode)+num_search_points*sizeof(snde_coord3), // data_bytes for transfer
							num_search_points*log(numvertices)*10.0, // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(0, //metadata_bytes
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
      result_rec = create_recording_math<multi_ndarray_recording>(get_result_channel_path(0),rss,1);      
      result_rec->define_array(0,SNDE_RTN_SNDE_KDNODE,"vertex_kdtree");
      std::shared_ptr<ndtyped_recording_ref<snde_index>> result_ref = create_typed_recording_ref_math<snde_index>(this->get_result_channel_path(0),this->rss);
      
      
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
	    true
#else
	    false
#endif
	    );
	  
	  return std::make_shared<exec_function_override_type>([ this,locktokens, result_ref,vertices,kdtree,search_points ]() {
	    // exec code
	    //snde_index numvertices = vertices->layout.flattened_length();
	    snde_index num_search_points = search_points->layout.flattened_length();

	    if (!vertices->layout.is_contiguous()) {
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
	    if (opencl_resource) {
	      
	      //snde_warning("knn executing in OpenCL!");
	      cl::Device knn_dev = opencl_resource->devices.at(0);
	      cl::Kernel knn_kern = knn_calculation_opencl.get_kernel(opencl_resource->context,knn_dev);
	      //cl::Kernel knn_kern = testcase_opencl.get_kernel(opencl_resource->context,knn_dev);
	      OpenCLBuffers Buffers(opencl_resource->oclcache,opencl_resource->context,knn_dev,locktokens);

	      size_t kern_work_group_size=0;
	      knn_kern.getWorkGroupInfo(knn_dev,CL_KERNEL_WORK_GROUP_SIZE,&kern_work_group_size);

	      uint32_t stacksize_per_workitem = max_depth+1; // uint32_t because this is passed as a kernel arg, below
	      size_t nodestacks_octwords_per_workitem = (stacksize_per_workitem*sizeof(snde_index) + 7)/8;
	      size_t statestacks_octwords_per_workitem = (stacksize_per_workitem*sizeof(uint8_t) + 7)/8;
	      size_t bboxstacks_octwords_per_workitem = (stacksize_per_workitem*(sizeof(snde_coord)*2) + 7)/8;
	      size_t local_memory_octwords_per_workitem = nodestacks_octwords_per_workitem + statestacks_octwords_per_workitem + bboxstacks_octwords_per_workitem;
	      
	      // limit workgroup size by local memory availability
	      cl_ulong local_mem_size=0;
	      knn_dev.getInfo(CL_DEVICE_LOCAL_MEM_SIZE,&local_mem_size);
	      size_t memory_workgroup_size_limit = local_mem_size/(8*local_memory_octwords_per_workitem);
	      if (memory_workgroup_size_limit < kern_work_group_size) {
		kern_work_group_size = memory_workgroup_size_limit;
	      }

	      
	      

	      size_t kernel_global_work_items = num_search_points;
	      
	      // limit the number of work items by the global size
	      if (kernel_global_work_items < kern_work_group_size) {
		kern_work_group_size = kernel_global_work_items; 	
	      }


	      size_t kern_preferred_size_multiple=0;
	      knn_kern.getWorkGroupInfo(knn_dev,CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,&kern_preferred_size_multiple);

	      if (kern_work_group_size > kern_preferred_size_multiple) {
		// Round work group size down to a multiple of of kern_preferred_size_multiple
		kern_work_group_size = kern_preferred_size_multiple * (kern_work_group_size/kern_preferred_size_multiple);
	      }
	      
	      
	      //kern_work_group_size = 1;

	      // for OpenCL 1.2 compatibility we make sure the number of global work items
	      // is a multiple of the work group size. i.e. we round the number of
	      // global work items up to the nearest multiple of kern_work_group_size
	      // (there is code in the kernel itself to ignore the excess work items) 
	      kernel_global_work_items = kern_work_group_size * ((kernel_global_work_items+kern_work_group_size-1)/kern_work_group_size);

	      
	      
	      Buffers.AddBufferAsKernelArg(kdtree,knn_kern,0,false,false);
	      Buffers.AddBufferAsKernelArg(vertices,knn_kern,1,false,false);
	      // add local memory arrays 
	      knn_kern.setArg(2,nodestacks_octwords_per_workitem*8*kern_work_group_size,nullptr);
	      knn_kern.setArg(3,statestacks_octwords_per_workitem*8*kern_work_group_size,nullptr);
	      knn_kern.setArg(4,bboxstacks_octwords_per_workitem*8*kern_work_group_size,nullptr);
	      
	      knn_kern.setArg(5,sizeof(stacksize_per_workitem),&stacksize_per_workitem);
	      Buffers.AddBufferAsKernelArg(search_points,knn_kern,6,false,false);
	      Buffers.AddBufferAsKernelArg(result_ref,knn_kern,7,true,true);
	      uint32_t opencl_ndim=3;
	      knn_kern.setArg(8,sizeof(opencl_ndim),&opencl_ndim);
	      uint32_t opencl_max_depth=max_depth;
	      knn_kern.setArg(9,sizeof(opencl_max_depth),&opencl_max_depth);
	      snde_index max_index_plus_one = kernel_global_work_items;
	      knn_kern.setArg(10,sizeof(max_index_plus_one),&max_index_plus_one);
	      

	      
	      cl::Event kerndone;
	      std::vector<cl::Event> FillEvents=Buffers.FillEvents();
	    
	      cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(knn_kern,{},{ kernel_global_work_items },{ kern_work_group_size },&FillEvents,&kerndone);
	      if (err != CL_SUCCESS) {
		throw openclerror(err,"Error enqueueing kernel");
	      }
	      opencl_resource->queues.at(0).flush(); /* trigger execution */
	      // mark that the kernel has modified result_rec
	      Buffers.BufferDirty(result_ref);
	      // wait for kernel execution and transfers to complete
	      Buffers.RemBuffers(kerndone,kerndone,true);
	      
	    } else {	    
#endif // SNDE_OPENCL
	      
	      snde_index *nodestack=(snde_index *)malloc((max_depth+1)*sizeof(snde_index));
	      uint8_t *statestack=(uint8_t *)malloc((max_depth+1)*sizeof(uint8_t));
	      snde_coord *bboxstack=(snde_coord *)malloc((max_depth+1)*sizeof(snde_coord)*2);
	      
	      for (snde_index searchidx=0;searchidx < num_search_points;searchidx++) {
		result_ref->element({searchidx}) = snde_kdtree_knn_one(kdtree->shifted_arrayptr(),
								       (snde_coord *)vertices->shifted_arrayptr(),
								       nodestack,
								       statestack,
								       bboxstack,
								       &search_points->element(searchidx,false).coord[0],
								       //nullptr,
								       3,
								       max_depth);
	      }
      
	      free(bboxstack);
	      free(statestack);
	      free(nodestack);
#ifdef SNDE_OPENCL
	    }
#endif // SNDE_OPENCL
	    
	    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_as_ready() 
	    result_ref->rec->mark_as_ready();
	    
	  }); 
	});
      });
    };
    
  };
  
  std::shared_ptr<math_function> define_knn_calculation_function()
  {
    return std::make_shared<cpp_math_function>([] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<knn_calculation>(rss,inst);
    });
    
  }


  
  static int registered_knn_calculation_function = register_math_function("spatialnde2.knn_calculation",define_knn_calculation_function());

  
  
};
