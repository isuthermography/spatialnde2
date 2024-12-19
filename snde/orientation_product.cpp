
#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/vecops.h"
#include "snde/recmath_cppfunction.hpp"
#include "snde/graphics_recording.hpp"
#include "snde/quaternion.h"

#include "snde/orientation_product.hpp"

namespace snde {


  class const_orientation_product: public recmath_cppfuncexec<snde_orientation3,std::shared_ptr<pose_channel_recording>,std::string> {
  public:
    const_orientation_product(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec(rss,inst)
    {
      
    }
    
    // use default for decide_new_revision
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(snde_orientation3 left,std::shared_ptr<pose_channel_recording> right,std::string untransformed)
    {

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
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
    result_ref = create_typed_subclass_ndarray_ref_math<pose_channel_recording,snde_orientation3>(get_result_channel_path(0),rss,recdb_path_join(recdb_path_context(right->info->name),right->channel_to_reorient));
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

	result_ref->allocate_storage({},false);


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
	  
	  snde_orientation3 right_orient = right_ref->element(0);
	  if (orientation_valid(left) && orientation_valid(right_orient)) {
	    orientation_orientation_multiply(left, right_orient, &result_ref->element(0));
	  }
	  else {
	    snde_invalid_orientation3(&result_ref->element(0));
	  }
	  
	  unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	  result_ref->rec->mark_data_ready();
	  
	}); 
      });
    });
  };
    
  };
  
  
  std::shared_ptr<math_function> define_spatialnde2_const_orientation_product_function()
  {
    return std::make_shared<cpp_math_function>("snde.const_orientation_product",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<const_orientation_product>(rss,inst);
    }); 
  }
  
  SNDE_API std::shared_ptr<math_function> const_orientation_product_function = define_spatialnde2_const_orientation_product_function();
  
  static int registered_const_orientation_product_function = register_math_function(const_orientation_product_function);





  class orientation_const_product: public recmath_cppfuncexec<std::shared_ptr<pose_channel_recording>,snde_orientation3,std::string> {
  public:
    orientation_const_product(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec(rss,inst)
    {
      
    }
    
    // use default for decide_new_revision
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<pose_channel_recording> left,snde_orientation3 right,std::string transformed)
    {

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							3*sizeof(snde_orientation3), // data_bytes for transfer
							(50.0), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
	};
      return std::make_pair(option_list,nullptr);
    }
  
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<pose_channel_recording> left,snde_orientation3 right, std::string transformed) 
  {
    // define_recs code
    //printf("define_recs()\n");

    if (!transformed.size()) {
      throw snde_error("orientation_const_product (%s -> %s): a channel to render is required",left->info->name,get_result_channel_path(0).c_str()); 
    }
    
    std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> result_ref;
    result_ref = create_typed_subclass_ndarray_ref_math<pose_channel_recording,snde_orientation3>(get_result_channel_path(0),rss,transformed);
    if (left->untransformed_channel) {
      std::string untransformed_full_name = recdb_path_join(recdb_path_context(left->info->name),*left->untransformed_channel);
      
      std::dynamic_pointer_cast<pose_channel_recording>(result_ref->rec)->set_untransformed_render_channel(untransformed_full_name);
    }
    
    return std::make_shared<metadata_function_override_type>([ this,result_ref,left, right ]() {
      // metadata code  -- copy from left)
      std::shared_ptr<constructible_metadata> metadata=std::make_shared<constructible_metadata>(*left->metadata);
      
      result_ref->rec->metadata=metadata;
      result_ref->rec->mark_metadata_done();
      
      return std::make_shared<lock_alloc_function_override_type>([ this,result_ref,left,right ]() {
	// lock_alloc code

	result_ref->allocate_storage({},false);


	std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> left_ref = left->reference_typed_ndarray<snde_orientation3>();
	
	rwlock_token_set locktokens = lockmgr->lock_recording_refs({
	     // first element is recording_ref, 2nd parameter is false for read, true for write
	    { left_ref, false },
	    { result_ref, true },
	  },
	  false
	  );
	
	return std::make_shared<exec_function_override_type>([ this, result_ref, left, left_ref, right, locktokens ]() {
	  // exec code
	  	  
	  snde_orientation3 left_orient = left_ref->element(0);
	  if (orientation_valid(left_orient) && orientation_valid(right)) {
	    orientation_orientation_multiply(left_orient,right,&result_ref->element(0));
	  }
	  else {
	    snde_invalid_orientation3(&result_ref->element(0));
	  }
	  

	  unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	  result_ref->rec->mark_data_ready();
	  
	}); 
      });
    });
  };
    
  };
  
  
  std::shared_ptr<math_function> define_spatialnde2_orientation_const_product_function()
  {
    return std::make_shared<cpp_math_function>("snde.orientation_const_product",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<orientation_const_product>(rss,inst);
    }); 
  }
  
  SNDE_API std::shared_ptr<math_function> orientation_const_product_function = define_spatialnde2_orientation_const_product_function();
  
  static int registered_orientation_const_product_function = register_math_function(orientation_const_product_function);






  

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
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
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
    result_ref = create_typed_subclass_ndarray_ref_math<pose_channel_recording,snde_orientation3>(get_result_channel_path(0),rss,recdb_path_join(recdb_path_context(right->info->name),right->channel_to_reorient));
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

	result_ref->allocate_storage({},false);

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

	  snde_orientation3 left_orient = left->element(0);
	  snde_orientation3 right_orient = right_ref->element(0);
	  if (orientation_valid(left_orient) && orientation_valid(right_orient)) {
	    orientation_orientation_multiply(left_orient, right_orient, &result_ref->element(0));
	  }
	  else {
	    snde_invalid_orientation3(&result_ref->element(0));
	  }
	  
	  unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	  result_ref->rec->mark_data_ready();
	  
	}); 
      });
    });
  };
    
  };
  
  
  std::shared_ptr<math_function> define_spatialnde2_orientation_rec_product_function()
  {
    return std::make_shared<cpp_math_function>("snde.orientation_rec_product",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<orientation_rec_product>(rss,inst);
    }); 
  }
  
  SNDE_API std::shared_ptr<math_function> orientation_rec_product_function = define_spatialnde2_orientation_rec_product_function();
  
  static int registered_orientation_rec_product_function = register_math_function(orientation_rec_product_function);

  




  class pose_follower: public recmath_cppfuncexec<std::shared_ptr<pose_channel_recording>,std::string,std::string> {
  public:
    pose_follower(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec(rss,inst)
    {
      
    }
    
    // use default for decide_new_revision
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<pose_channel_recording> leader,std::string follower_content,std::string untransfomed)
    {

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							3*sizeof(snde_orientation3), // data_bytes for transfer
							(50.0), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
	};
      return std::make_pair(option_list,nullptr);
    }
  
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<pose_channel_recording> leader,std::string follower_content,std::string untransformed) 
  {
    // define_recs code
    //printf("define_recs()\n");
    std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> result_ref;
    result_ref = create_typed_subclass_ndarray_ref_math<pose_channel_recording,snde_orientation3>(get_result_channel_path(0),rss,recdb_path_join(inst->channel_path_context,follower_content));
    if (untransformed.size() > 0) {
      std::dynamic_pointer_cast<pose_channel_recording>(result_ref->rec)->set_untransformed_render_channel(untransformed);
    }
    
    return std::make_shared<metadata_function_override_type>([ this,result_ref,leader ]() {
      // metadata code  -- copy from right, (will merge in updates from left if it is a recording)
      std::shared_ptr<constructible_metadata> metadata=std::make_shared<constructible_metadata>(*leader->metadata);
      
      result_ref->rec->metadata=metadata;
      result_ref->rec->mark_metadata_done();
      
      return std::make_shared<lock_alloc_function_override_type>([ this,result_ref,leader ]() {
	// lock_alloc code

	result_ref->allocate_storage({},false);


	std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> leader_ref = leader->reference_typed_ndarray<snde_orientation3>();
	
	rwlock_token_set locktokens = lockmgr->lock_recording_refs({
	     // first element is recording_ref, 2nd parameter is false for read, true for write
	    { leader_ref, false },
	    { result_ref, true },
	  },
	  false
	  );
	
	return std::make_shared<exec_function_override_type>([ this, result_ref, leader, leader_ref, locktokens ]() {
	  // exec code
	  
	  result_ref->element(0)=leader_ref->element(0);
	  unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	  result_ref->rec->mark_data_ready();
	  
	}); 
      });
    });
  };
    
  };
  
  
  std::shared_ptr<math_function> define_spatialnde2_pose_follower_function()
  {
    return std::make_shared<cpp_math_function>("snde.pose_follower",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<pose_follower>(rss,inst);
    }); 
  }
  
  SNDE_API std::shared_ptr<math_function> pose_follower_function = define_spatialnde2_pose_follower_function();
  
  static int registered_pose_follower_function = register_math_function(pose_follower_function);


  
};


