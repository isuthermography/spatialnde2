// Concept:
// * Need a way to insert offset (translation)
//   calibrations into a scene graph.
// * A math function that has the scene of interest as
//   one of its inputs and generates pose recordings on an output channel.
//   The output channel has to be part of the scene graph of the scene of
//   interest.
// * The math function is self-dependent and also has a dynamic dependency
//   on every element of its input scene graph except itself.
// * The output pose recordings contain translations, but no rotation.
// * The translations are determined from the average residual error
//   from calibration experiments.
// * Each calibration experiment provides coordinates for a landmark in
//   terms of a particular input scene graph node as well as actual
//   coordinates in terms of a different input scene graph node.
// * The output channel is yet a third input scene graph node.
// * To determine residual error, we need to transform the landmark
//   and actual coordinates into the coordinate frame of the output
//   channel. In the context of the calibration globalrevision, we
//   can then subtract them also considering the value of the preexisting
//   output. This gives a residual error that can be averaged with
//   residual errors from other calibration experiments.
// * Exactly one of the landmark coordinates or actual coordinates
//   must be within the scene graph of the output channel.
// * A positive translation moves the object within the scene graph
//   of the output channel in the +x, +y, +z direction within the
//   output channel. Therefore, if the translation is the average
//   residual error, we must define the residual error such that
//   a positive error means that the object within the output
//   channel scene graph needs to be moved in the positive direction.
// * Thus, we define residual error as (coordinates of object outside
//   the output channel scene graph) - (coordinates of object within
//   the output channel scene graph). Within this convention, if
//   the object within is located too far in the positive direction,
//   the residual error will be negative, and that negative correction
//   will line up the object as represented within and without the
//   output channel.
// * The calibration experiment data is passed to the math function via
//   a message sent within a transaction.
// * Note that two of these math functions within the same scene graph
//   can cause a math engine deadlock because each is dependent on every
//   element of its input scene except itself. Therefore, the math functions
//   will wait for each other and never execute.
// * To work around the potential math engine deadlock, we can program
//   the dynamic dependency calculations to only operate in a transaction
//   in which our particular math function was sent a message with
//   calibration data. Therefore, as long as we update multiple
//   calibrations in separate transactions, no deadlock will occur.



// Prerequisites:
// * Function for traversing a scene graph and building array of
//   oriented parts. This is needed to figure out the relative
//   transforms between the landmark, actual, and output channel
//   coordinate frames. DONE 
//    * Can use new methods for locking a scene graph to facilitate
//      the traversal.
// * Metadata and infrastructure for defining landmarks.
// * For testing, the ability to click through on a channel
//   and locate a point.

#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/vecops.h"
#include "snde/geometrydata.h"
#include "snde/quaternion.h"
#include "snde/recstore.hpp"
#include "snde/recmath_cppfunction.hpp"
#include "snde/graphics_recording.hpp"


#include "snde/offset_calibration.hpp"

namespace snde {


  class offset_calibration_creator_data : public recording_creator_data {
  public:

    snde_coord3 residual_sum; //sum of what the residuals would be with no
    // offset applied
    unsigned num_residuals;

    // snde_coord3 last_offset; 
    
    offset_calibration_creator_data() :
      residual_sum({{0.0,0.0,0.0}}),
      num_residuals(0)
    {
      
    }
    // rule of 3
    offset_calibration_creator_data& operator=(const offset_calibration_creator_data&) = delete;
    offset_calibration_creator_data(const offset_calibration_creator_data& orig) = delete;
    virtual ~offset_calibration_creator_data() = default;

  };
  
  offset_calibration_single_point::offset_calibration_single_point(std::string component_path0,
				    snde_coord3 local_coord_posn0,
				    std::string component_path1,
				    snde_coord3 local_coord_posn1) :
    math_instance_parameter(SNDE_MFPT_OFFSETCALIB),
    component_path{component_path0,component_path1},
    local_coord_posn{local_coord_posn0,local_coord_posn1}
  {
      
  }

  bool offset_calibration_single_point::operator==(const math_instance_parameter &ref) // used for comparing extra parameters to instantiated_math_functions
  {
    const offset_calibration_single_point *ref_ptr = dynamic_cast<const offset_calibration_single_point *>(&ref);
    
    if (!ref_ptr) {
	return false;
    }
    
    return (component_path[0] == ref_ptr->component_path[0] &&
	    component_path[1] == ref_ptr->component_path[1] &&
	    local_coord_posn[0].coord[0] == ref_ptr->local_coord_posn[0].coord[0] &&
	    local_coord_posn[0].coord[1] == ref_ptr->local_coord_posn[0].coord[1] &&
	    local_coord_posn[0].coord[2] == ref_ptr->local_coord_posn[0].coord[2] &&
	    local_coord_posn[1].coord[0] == ref_ptr->local_coord_posn[1].coord[0] &&
	    local_coord_posn[1].coord[1] == ref_ptr->local_coord_posn[1].coord[1] &&
	    local_coord_posn[1].coord[2] == ref_ptr->local_coord_posn[1].coord[2]);
  }
  
  bool offset_calibration_single_point::operator!=(const math_instance_parameter &ref)
  {
    return !((*this)==ref);
  }

  
  class offset_calibration : public recmath_cppfuncexec<std::shared_ptr<recording_base>,std::shared_ptr<recording_base>> {
  public:
    offset_calibration(std::shared_ptr<recording_set_state> rss, std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<recording_base>, std::shared_ptr<recording_base>>(rss, inst)
    {

    }
    
    std::pair<bool, std::shared_ptr<compute_options_function_override_type>> decide_execution(std::shared_ptr<recording_base> scene, std::shared_ptr<recording_base> to_transform)
    {
      bool just_starting = false;
      
      std::shared_ptr<recording_base> previous_recording = this->self_dependent_recordings.at(0);
      std::shared_ptr<pose_channel_recording> previous_recording_pose;
      std::shared_ptr<recording_creator_data> previous_recording_creator_data;
      std::shared_ptr<offset_calibration_creator_data> creator_data;
      
      if (!previous_recording) {
	just_starting = true;
      } else {
	{
	  std::lock_guard<std::mutex> prevrec_admin(previous_recording->admin);
	  previous_recording_creator_data = previous_recording->creator_data;
	}
	if (!previous_recording_creator_data) {
	  just_starting = true;
	} else {
	  creator_data = std::dynamic_pointer_cast<offset_calibration_creator_data>(previous_recording_creator_data);
	  if (!creator_data) {
	    just_starting = true;
	  } 
	}
      }

      auto clearcheck = msgs.find("clear");
      if (clearcheck != msgs.end()) {
	// create with blank creator data
	// treat like we are just starting
	just_starting = true;
      }

      
      if (just_starting) {
	creator_data = std::make_shared<offset_calibration_creator_data>();
      }
      // at this point, we have a creator_data pointer and
      // we know if we are just_starting

      bool have_update = false;

      std::shared_ptr<offset_calibration_single_point> single_point_offset;

      auto msg_it = msgs.find("single_point_offset");
      if (msg_it != msgs.end()) {
	
	single_point_offset = std::dynamic_pointer_cast<offset_calibration_single_point>(msg_it->second);
	if (single_point_offset) {
	  // a message we understand
	  have_update = true;
	}
      }

      if (just_starting && !have_update) {
	// just starting and no update: we will return a zero offset
	return std::make_pair(true,std::make_shared<compute_options_function_override_type>([this,to_transform,creator_data] {
	  

	  //snde_warning("offset_calibration: compute_options just_starting");

	  std::vector<std::shared_ptr<compute_resource_option>> option_list =
	    {
	      std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							    0, //metadata_bytes 
							    0, // data_bytes for transfer
							    0.0, // flops
							    1, // max effective cpu cores
							    1), // useful_cpu_cores (min # of cores to supply
		  
	    };
	  
	  
	  return std::make_pair(option_list,std::make_shared<define_recs_function_override_type>([this, to_transform, creator_data] { 
	    
	    // define_recs code
	    std::shared_ptr<pose_channel_recording> starting_result;
	    //snde_warning("offset_calibration: define_recs just_starting");
	    std::string channel_to_reorient = to_transform->info->name;
	    starting_result = create_recording_math<pose_channel_recording>(this->get_result_channel_path(0),this->rss,channel_to_reorient);
	    return std::make_shared<metadata_function_override_type>([ this, starting_result,creator_data ]() {
	      // metadata code

	      //snde_warning("avg: metadata just_starting");

	      starting_result->metadata=std::make_shared<immutable_metadata>();
	      starting_result->mark_metadata_done();
	      
	      return std::make_shared<lock_alloc_function_override_type>([ this, starting_result,creator_data ]() {
		// lock_alloc code
		std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> starting_ref = starting_result->reference_typed_ndarray<snde_orientation3>();
		starting_result->allocate_storage(0,{1},false);
		rwlock_token_set locktokens = lockmgr->lock_recording_refs({
		    { starting_ref,true}},
									   false
									   );
		
		
		return std::make_shared<exec_function_override_type>([ this, starting_result, starting_ref, locktokens,creator_data ]() {
		  // exec code

		  //std::shared_ptr<averaging_downsampler_creator_data<T>> new_creator_data = std::make_shared<averaging_downsampler_creator_data<T>>();
		  //snde_warning("avg: exec just_starting");

		  // new_creator_data->pending_recs.push_back(to_avg);
		  starting_result->creator_data = creator_data;
		  snde_orientation3 zero_offset;
		  snde_null_orientation3(&zero_offset);
		  starting_ref->element(0) = zero_offset;
		  
		  starting_result->mark_data_ready();
		  
		}); 
	      });
		  
	    });
	  }));
	}));
      } else if (have_update) {
	// this is back in decide_execution()
	// we will merge our offset in with any existing offset
	// return an execution tree for performing the offset update

	return std::make_pair(true,std::make_shared<compute_options_function_override_type>([this,scene,to_transform,previous_recording_pose,single_point_offset,creator_data] {
	  

	  //snde_warning("offset_calibration: compute_options just_starting");

	  std::vector<std::shared_ptr<compute_resource_option>> option_list =
	    {
	      std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							    0, //metadata_bytes 
							    0, // data_bytes for transfer
							    0.0, // flops
							    1, // max effective cpu cores
							    1), // useful_cpu_cores (min # of cores to supply
		  
	    };
	  
	  
	  return std::make_pair(option_list,std::make_shared<define_recs_function_override_type>([this,scene,to_transform,previous_recording_pose,single_point_offset,creator_data] { 
	    
	    // define_recs code
	    std::shared_ptr<pose_channel_recording> result;
	    //snde_warning("offset_calibration: define_recs just_starting");
	    std::string channel_to_reorient = to_transform->info->name;
	    result = create_recording_math<pose_channel_recording>(this->get_result_channel_path(0),this->rss,channel_to_reorient);
	    return std::make_shared<metadata_function_override_type>([ this,scene,to_transform,previous_recording_pose,single_point_offset,creator_data,channel_to_reorient,result ]() {
	      // metadata code

	      //snde_warning("offset_calibration: metadata just_starting");

	      result->metadata=std::make_shared<immutable_metadata>();
	      result->mark_metadata_done();
	      
	      return std::make_shared<lock_alloc_function_override_type>([ this,scene,to_transform,previous_recording_pose,single_point_offset,creator_data,channel_to_reorient,result ]() {
		// lock_alloc code
		result->allocate_storage(0,{1},false);

		// We are going to split the world into two parts:
		// "Outer", which is in the scenegraph but not within
		// the portion of the scenegraph transformed by our
		// offset, and "Inner" which is transformed by our
		// offset. 
		std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>> outer_lock_info;
		std::vector<std::pair<std::string,std::string>> our_channel_component_paths;
		std::tie(outer_lock_info,our_channel_component_paths) = traverse_scenegraph_orientationlocks_except_channelpaths(rss,scene->info->name,{get_result_channel_path(0)});

		if (our_channel_component_paths.size() == 0) {
		  throw snde_error("offset_calibration: Result channel %s is not present in scene", get_result_channel_path(0).c_str());
		} else if (our_channel_component_paths.size() > 1) {
		  throw snde_error("offset_calibration: Result channel %s is present more than once in scene.", get_result_channel_path(0).c_str());
		}
		std::string our_channel_path;
		std::string our_component_path;
		std::tie(our_channel_path,our_component_path) = *our_channel_component_paths.begin();

		std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>> inner_lock_info;

		std::tie(inner_lock_info,our_channel_component_paths) = traverse_scenegraph_orientationlocks_except_channelpaths(rss,scene->info->name,{get_result_channel_path(0)},our_component_path +  "/" + "channel_to_reorient");

		std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>> all_lock_info;
		// Merge outer_lock_info and inner_lock_info into all_lock_info. 
		all_lock_info.reserve( outer_lock_info.size() + inner_lock_info.size() ); // preallocate memory
		all_lock_info.insert( all_lock_info.end(), outer_lock_info.begin(), outer_lock_info.end() );
		all_lock_info.insert( all_lock_info.end(), inner_lock_info.begin(), inner_lock_info.end() );

		// Add our prior output to all_lock_info if present
		if (previous_recording_pose) {
		  all_lock_info.push_back(std::make_pair(previous_recording_pose,std::make_pair(0,false))); 
		}
		// Append our output array to all_lock_info
		all_lock_info.push_back(std::make_pair(result,std::make_pair(0,true))); // True = write enabled
		
		std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> result_ref = result->reference_typed_ndarray<snde_orientation3>();
		rwlock_token_set locktokens = lockmgr->lock_recording_arrays(all_lock_info,false);	
		
		return std::make_shared<exec_function_override_type>([ this,scene,to_transform,previous_recording_pose,single_point_offset,creator_data,channel_to_reorient,result,result_ref,locktokens ]() {
		  // exec code

		  //snde_warning("offset_calibration: exec just_starting");

		  snde_orientation3 our_transform;
		  if (previous_recording_pose) {
		    our_transform = previous_recording_pose->reference_typed_ndarray<snde_orientation3>()->element(0);
		  } else {
		    snde_null_orientation3(&our_transform);
		  }
		  
		  std::vector<std::tuple<std::string,std::string,snde_partinstance>> outer_channelpath_componentpath_instances;
		  std::vector<std::tuple<std::string,std::string,snde_orientation3>> our_channel_componentpath_orientations;
		  std::tie(outer_channelpath_componentpath_instances,our_channel_componentpath_orientations) = traverse_scenegraph_orientationlocked_except_channelpaths(rss,scene->info->name,{get_result_channel_path(0)});

		  assert(our_channel_componentpath_orientations.size() == 1); // Should always be true because we threw exceptions otherwise above. 

		  std::string our_channelpath;
		  std::string our_componentpath;
		  snde_orientation3 our_orientation;
		  std::tie(our_channelpath,our_componentpath,our_orientation) = *our_channel_componentpath_orientations.begin();
		  
		  std::vector<std::tuple<std::string,std::string,snde_partinstance>> inner_channelpath_componentpath_instances;
		  snde_orientation3 our_transformed_orientation;
		  orientation_orientation_multiply(our_orientation,our_transform,&our_transformed_orientation);
		  
		  std::tie(inner_channelpath_componentpath_instances,our_channel_componentpath_orientations) = traverse_scenegraph_orientationlocked_except_channelpaths(rss,scene->info->name,{get_result_channel_path(0)},our_componentpath + "/" + "channel_to_reorient",&our_transformed_orientation);




		  // Need to identify which entry in the
		  //single_point_offset (0 or 1) is inside and which
		  //is outside. Can do this by matching the
		  //component_paths.
		  std::string outer_componentpath;
		  snde_partinstance outer_instance;
		  snde_coord4 outer_coords;
		  bool got_outer = false;
		  
		  std::string inner_componentpath;
		  snde_partinstance inner_instance;
		  snde_coord4 inner_coords;
		  bool got_inner = false;
		  
		  for (auto && channelpath_componentpath_instance: outer_channelpath_componentpath_instances) {
		    for (int point_index = 0; point_index < 2; point_index++) {
		      if (std::get<1>(channelpath_componentpath_instance)==single_point_offset->component_path[point_index]) {
			if (got_outer) {
			  throw snde_error("offset_calibration: got multiple outer components at scenegraph paths %s and %s (channel %s) scene channel %s",outer_componentpath.c_str(),std::get<1>(channelpath_componentpath_instance).c_str(),std::get<0>(channelpath_componentpath_instance).c_str(),scene->info->name);
			}
			outer_componentpath = std::get<1>(channelpath_componentpath_instance);
			outer_instance = std::get<2>(channelpath_componentpath_instance);
			coord4_posn_from_coord3(single_point_offset->local_coord_posn[point_index],&outer_coords);
			got_outer = true;
		      }
	
		    }
		  }


		  
		  for (auto && channelpath_componentpath_instance: inner_channelpath_componentpath_instances) {
		    for (int point_index = 0; point_index < 2; point_index++) {
		      if (std::get<1>(channelpath_componentpath_instance)==single_point_offset->component_path[point_index]) {
			if (got_inner) {
			  throw snde_error("offset_calibration: got multiple inner components at scenegraph paths %s and %s (channel %s) scene channel %s",inner_componentpath.c_str(),std::get<1>(channelpath_componentpath_instance).c_str(),std::get<0>(channelpath_componentpath_instance).c_str(),scene->info->name);
			}
			inner_componentpath = std::get<1>(channelpath_componentpath_instance);
			inner_instance = std::get<2>(channelpath_componentpath_instance);
			coord4_posn_from_coord3(single_point_offset->local_coord_posn[point_index],&inner_coords);
			got_inner = true;
		      }
		    }
		  }

		  if (!got_outer) {
		    throw snde_error("offset_calibration: did not find an outer component for scene channel %s not inside %s",scene->info->name,get_result_channel_path(0).c_str());
		  }
		  
		  if (!got_inner) {
		    throw snde_error("offset_calibration: did not find an inner component for scene channel %s not inside %s",scene->info->name,get_result_channel_path(0).c_str());
		  }

		  
		  //Transform the coordinates of the
		  //outside entry into our coordinates.

		  snde_orientation3 outside_orientation = outer_instance.orientation;
		  // Outside_orientation multiplied on the right by a point
		  // in the coordinate frame of the outside point gives us
		  // the location of that point in scene coordinates.
		  // we treat it as scene_over_outside

		  // our_orientation works likewise and we treat it
		  // as scene_over_our

		  // therefore, to transform outside coordinates into
		  // our coordinates, we need our_over_outside.

		  // By dimensional analysis,
		  // our_over_outside = our_over_scene*scene_over_outside
		  // we obtain our_over_scene by inverting scene_over_our

		  snde_orientation3 our_over_scene;
		  orientation_inverse(our_orientation,&our_over_scene);
		  snde_orientation3 our_over_outside;
		  orientation_orientation_multiply(our_over_scene,outside_orientation,&our_over_outside);
		  snde_coord4 our_outer_coords;
		  orientation_apply_position(our_over_outside,outer_coords,&our_outer_coords);
		  
		  //Transform the
		  //coordinates of the inside entry into our
		  //coordinates as well. Will be the same as
		  //above.
		  snde_orientation3 inside_orientation = inner_instance.orientation;
		  snde_orientation3 our_over_inside;
		  orientation_orientation_multiply(our_over_scene,inside_orientation,&our_over_inside);
		  snde_coord4 our_inner_coords;
		  orientation_apply_position(our_over_inside,inner_coords,&our_inner_coords);

		  //Then subtract out the effect
		  //of our current transform from the inside entry
		  //coordinates. Now outside - inside =
		  //desired_transform.

		  snde_coord4 desired_transform;
		  subcoordcoord4(our_outer_coords, our_inner_coords, &desired_transform);
		  
		  //So we can add outside-inside to
		  //the residual_sum,
		  
		  creator_data->residual_sum.coord[0] += desired_transform.coord[0];
		  creator_data->residual_sum.coord[1] += desired_transform.coord[1];
		  creator_data->residual_sum.coord[2] += desired_transform.coord[2];
		  
		  //update num_residuals,

		  creator_data->num_residuals++;
		  
		  //and
		  //evaluate a new offset from
		  //residual_sum/num_residuals

		  snde_coord4 new_offset;
		  scalevec3(1.0/creator_data->num_residuals,&creator_data->residual_sum.coord[0],&new_offset.coord[0]);
		  new_offset.coord[3] = 0.0;

		  snde_orientation3 transform;
		  snde_null_orientation3(&transform);
		  transform.offset = new_offset;
		  
		  
		  //and store this in
		  //result_ref->element(0).
		  
		  
		  
		  result->creator_data = creator_data;
		  result_ref->element(0) = transform;
		  
		  result->mark_data_ready();
		  
		}); 
	      });
		  
	    });
	  }));
	}));
	
      } else {
	// this is back in decide_execution()
	// there is no update and we're not just starting
	// (or just cleared)
	// so we return false and do not execute.
	return std::make_pair(false,nullptr);
      }
    }
    



  };

  static bool offset_calibration_find_additional_deps(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst, math_function_status *mathstatus_ptr)
  {
    bool got_math_messages = false;
    {
      std::lock_guard<std::mutex> rss_admin(rss->admin);
      auto msg_it = rss->mathstatus.math_messages.find(inst);
      if (msg_it != rss->mathstatus.math_messages.end()) {
	got_math_messages = true;
      }
    }
    if (!got_math_messages) {
      return false;
    }

    //std::string result_channel_path = recdb_path_join(inst->channel_path_context,*inst->result_channel_paths.at(0));
    bool found_incomplete_deps = traverse_scenegraph_find_graphics_deps(rss,inst,mathstatus_ptr,{},{});

    return found_incomplete_deps;
  }
  
    
  std::shared_ptr<math_function> define_spatialnde2_offset_calibration_function()
  {
    std::shared_ptr<math_function> newfunc = std::make_shared<cpp_math_function>("snde.offset_calibration",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) -> std::shared_ptr<executing_math_function> {
	std::shared_ptr<offset_calibration> execfunc =  std::make_shared<offset_calibration>(rss,inst);

	return execfunc;
      });
    newfunc->self_dependent=true;
    newfunc->dynamic_dependency=true;
    newfunc->find_additional_deps=std::make_shared<std::function<bool(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst, math_function_status *mathstatus_ptr)>>(offset_calibration_find_additional_deps);
    newfunc->new_revision_optional=true;
    return newfunc;
  }
  
  SNDE_API std::shared_ptr<math_function> offset_calibration_function = define_spatialnde2_offset_calibration_function();
  
  static int registered_offset_calibration_function = register_math_function(offset_calibration_function);
  
  



  
  // To do list:
  // *need to implement execution tree for have_update case. DONE
  // *need to implement dynamic dependency DONE
  // *need to implement function for transmitting calibration points
  //  to the math function.






};
