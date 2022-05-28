
#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/vecops.h"
#include "snde/recmath_cppfunction.hpp"
#include "snde/graphics_recording.hpp"
#include "snde/quaternion.h"

#include "snde/orientation_product.hpp"

namespace snde {


  class orientation_const_product: public recmath_cppfuncexec<snde_orientation3,std::shared_ptr<pose_channel_recording>,std::string> {
  public:
    orientation_const_product(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec(rss,inst)
    {
      
    }
    
    // use default for decide_new_revision
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(snde_orientation3 left,std::shared_ptr<pose_channel_recording> right,std::string untransformed)
    {

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(0, //metadata_bytes 
							3*sizeof(snde_orientation3), // data_bytes for transfer
							(50.0), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
	};
      return std::make_pair(option_list,nullptr);
    }
  
    std::shared_ptr<metadata_function_override_type> define_recs(snde_orientation3 left,std::shared_ptr<pose_channel_recording> right,std::string untransformed) 
  {
    // define_recs code
    //printf("define_recs()\n");
    std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> result_ref;
    result_ref = create_typed_subclass_recording_ref_math<pose_channel_recording,snde_orientation3>(get_result_channel_path(0),rss,recdb_path_join(recdb_path_context(right->info->name),right->channel_to_reorient));
    if (untransformed.size() > 0) {
      std::dynamic_pointer_cast<pose_channel_recording>(result_ref->rec)->set_untransformed_render_channel(untransformed);
    }
    
    return std::make_shared<metadata_function_override_type>([ this,result_ref,left, right ]() {
      // metadata code  -- copy from right, (will merge in updates from left if it is a recording)
      std::shared_ptr<constructible_metadata> metadata=std::make_shared<constructible_metadata>(*right->metadata);
      
      result_ref->rec->metadata=metadata;
      result_ref->rec->mark_metadata_done();
      
      return std::make_shared<lock_alloc_function_override_type>([ this,result_ref,left,right ]() {
	// lock_alloc code

	result_ref->allocate_storage({1},false);


	std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> right_ref = right->reference_typed_ndarray<snde_orientation3>();
	
	rwlock_token_set locktokens = lockmgr->lock_recording_refs({
	     // first element is recording_ref, 2nd parameter is false for read, true for write
	    { right_ref, false },
	    { result_ref, true },
	  },
	  false
	  );
	
	return std::make_shared<exec_function_override_type>([ this, result_ref, left, right, right_ref, locktokens ]() {
	  // exec code
	  
	  orientation_orientation_multiply(left,right_ref->element(0),&result_ref->element(0));
	  unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	  result_ref->rec->mark_data_ready();
	  
	}); 
      });
    });
  };
    
  };
  
  
  std::shared_ptr<math_function> define_spatialnde2_orientation_const_product_function()
  {
    return std::make_shared<cpp_math_function>([] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<orientation_const_product>(rss,inst);
    }); 
  }
  
  SNDE_API std::shared_ptr<math_function> orientation_const_product_function = define_spatialnde2_orientation_const_product_function();
  
  static int registered_orientation_const_product_function = register_math_function("spatialnde2.orientation_const_product",orientation_const_product_function);





  class orientation_rec_product: public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<snde_orientation3>>,std::shared_ptr<pose_channel_recording>,std::string> {
  public:
    orientation_rec_product(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec(rss,inst)
    {
      
    }
    
    // use default for decide_new_revision
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> left,std::shared_ptr<pose_channel_recording> right,std::string untransformed)
    {

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(0, //metadata_bytes 
							3*sizeof(snde_orientation3), // data_bytes for transfer
							(50.0), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
	};
      return std::make_pair(option_list,nullptr);
    }
  
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> left,std::shared_ptr<pose_channel_recording> right,std::string untransformed) 
  {
    // define_recs code
    //printf("define_recs()\n");
    std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> result_ref;
    result_ref = create_typed_subclass_recording_ref_math<pose_channel_recording,snde_orientation3>(get_result_channel_path(0),rss,recdb_path_join(recdb_path_context(right->info->name),right->channel_to_reorient));
    if (untransformed.size() > 0) {
      std::dynamic_pointer_cast<pose_channel_recording>(result_ref->rec)->set_untransformed_render_channel(untransformed);
    }
    
    return std::make_shared<metadata_function_override_type>([ this,result_ref,left, right ]() {
      // metadata code  -- copy from right, (will merge in updates from left if it is a recording)
      std::shared_ptr<constructible_metadata> metadata=std::make_shared<constructible_metadata>(*right->metadata);
      
      result_ref->rec->metadata=MergeMetadata(metadata,left->rec->metadata);
      result_ref->rec->mark_metadata_done();
      
      return std::make_shared<lock_alloc_function_override_type>([ this,result_ref,left,right ]() {
	// lock_alloc code

	result_ref->allocate_storage({1},false);

	std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> right_ref = right->reference_typed_ndarray<snde_orientation3>();

	rwlock_token_set locktokens = lockmgr->lock_recording_refs({
	     // first element is recording_ref, 2nd parameter is false for read, true for write
	    { left, false },
	    { right_ref, false },
	    { result_ref, true },
	  },
	  false
	  );
	
	return std::make_shared<exec_function_override_type>([ this, result_ref, left, right, right_ref, locktokens ]() {
	  // exec code
	  
	  orientation_orientation_multiply(left->element(0),right_ref->element(0),&result_ref->element(0));
	  unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	  result_ref->rec->mark_data_ready();
	  
	}); 
      });
    });
  };
    
  };
  
  
  std::shared_ptr<math_function> define_spatialnde2_orientation_rec_product_function()
  {
    return std::make_shared<cpp_math_function>([] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<orientation_rec_product>(rss,inst);
    }); 
  }
  
  SNDE_API std::shared_ptr<math_function> orientation_rec_product_function = define_spatialnde2_orientation_rec_product_function();
  
  static int registered_orientation_rec_product_function = register_math_function("spatialnde2.orientation_rec_product",orientation_rec_product_function);

  


};


