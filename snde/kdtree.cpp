

#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/geometrydata.h"

#include "snde/recstore.hpp"
#include "snde/recmath_cppfunction.hpp"


#include "snde/kdtree.hpp"

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
	std::unordered_map<std::string,metadatum> metadata;
	//printf("metadata()\n");
	
	result_ref->rec->metadata=std::make_shared<immutable_metadata>(metadata);
	result_ref->rec->mark_metadata_done();
	
	return std::make_shared<lock_alloc_function_override_type>([ this,result_ref,vertices ]() {
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
	  
	  return std::make_shared<exec_function_override_type>([ this,locktokens, result_ref,vertices ]() {
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
	    
	    
	    build_subkdtree<snde_coord3>(tree_vertices,orig_tree,&tree_nextpos,0,numvertices,3,0);
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


  
  static int registered_kdtree_calculation_function = register_math_function(std::make_shared<registered_math_function>("spatialnde2.kdtree_calculation",define_kdtree_calculation_function));
  
  
  
};
