

#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/geometrydata.h"

#include "snde/recstore.hpp"
#include "snde/recmath_cppfunction.hpp"


#include "snde/kdtree.hpp"
#include "snde/kdtree_knn.h"

namespace snde {


  int kce_compare(const void *kce_1,const void *kce_2)
  {
    const kdtree_construction_entry *kce1=(const kdtree_construction_entry *)kce_1;
    const kdtree_construction_entry *kce2=(const kdtree_construction_entry *)kce_2;

    if (kce1->depth < kce2->depth) {
      return -1;
    } else if (kce1->depth < kce2->depth) {
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
	      treenode.left_subtree = orig_tree[copy_tree[treeidx].left_subtree].entry_index;
	      treenode.right_subtree = orig_tree[copy_tree[treeidx].right_subtree].entry_index;
	      
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
	    });
	  
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
								    nullptr,
								    3);
	    }
	    
	    free(bboxstack);
	    free(statestack);
	    free(nodestack);
	    
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
