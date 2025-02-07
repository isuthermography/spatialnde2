

#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/vecops.h"
#include "snde/geometry_ops.h"
#include "snde/geometrydata.h"
#include "snde/normal_calc.h"

#include "snde/recstore.hpp"
#include "snde/recmath_cppfunction.hpp"
#include "snde/graphics_recording.hpp"
#include "snde/graphics_storage.hpp"


#include "snde/averaging_downsampler.hpp"

namespace snde {
  /*
  class averaging_temporal_math_function: public cpp_math_function {
  public:
    averaging_temporal_math_function() :
      cpp_math_function([] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
	return make_cppfuncexec_floatingtypes<averaging_temporal_downsampler>(rss,inst);

      })
    {
      
    }

    // Rule of 3
    averaging_temporal_math_function(const averaging_temporal_math_function &) = delete;
    averaging_temporal_math_function& operator=(const averaging_temporal_math_function &) = delete; 
    virtual ~averaging_temporal_math_function();
    
  };

*/


  template <typename T>
  class averaging_downsampler_creator_data: public recording_creator_data {
  public:
    std::vector<std::shared_ptr<ndtyped_recording_ref<T>>> pending_recs;

    averaging_downsampler_creator_data() = default;
    // rule of 3
    averaging_downsampler_creator_data & operator=(const averaging_downsampler_creator_data &) = delete; 
    averaging_downsampler_creator_data(const averaging_downsampler_creator_data &orig) = delete;
    virtual ~averaging_downsampler_creator_data()=default; 

  };
  
  template <typename T>
  class averaging_temporal_downsampler: public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,uint64_t,snde_bool> {
  public:
    averaging_temporal_downsampler(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,uint64_t,snde_bool>(rss,inst)
    {
      
    }

    // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,uint64_t,snde_bool>::compute_options_function_override_type compute_options_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,uint64_t,snde_bool>::define_recs_function_override_type define_recs_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,uint64_t,snde_bool>::metadata_function_override_type metadata_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,uint64_t,snde_bool>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,uint64_t,snde_bool>::exec_function_override_type exec_function_override_type;

    
    std::pair<bool,std::shared_ptr<compute_options_function_override_type>> decide_execution(std::shared_ptr<ndtyped_recording_ref<T>> to_avg,uint64_t numavgs,snde_bool usedouble)
    {

      bool just_starting = false;
      bool new_recording = false;
      std::shared_ptr<recording_base> previous_recording = this->self_dependent_recordings.at(0);
      std::shared_ptr<multi_ndarray_recording> previous_recording_ndarray;
      std::shared_ptr<recording_creator_data> previous_recording_creator_data;
      std::shared_ptr<averaging_downsampler_creator_data<T>> creator_data;
      
      if (!previous_recording) {
	//snde_warning("avg: no previous recording");
	just_starting = true; 
      } else {
	//previous_recording_ndarray = std::dynamic_pointer_cast<multi_ndarray_recording>(previous_recording);
	//if (!previous_recording_ndarray) {
	//  just_starting = true;
	//} else {
	  //std::shared_ptr<math_parameter_recording> toavg_param = std::dynamic_pointer_cast<math_parameter_recording>(inst->parameters.at(0));
	  //size_t previous_recording_array_index = toavg_param->array_index;
	  //
	  //if (toavg_param->array_name.size() > 0) {
	  //  // override array_index
	  //  previous_recording_array_index = previous_recording_ndarray.name_mapping.at(toavg_param->array_name);
	  //  
	  //}
	{
	  std::lock_guard<std::mutex> prevrec_admin(previous_recording->admin);
	  previous_recording_creator_data = previous_recording->creator_data;
	}
	if (!previous_recording_creator_data) {
	  //snde_warning("avg: no previous creator_data");
	  just_starting = true; 
	} else {
	  creator_data = std::dynamic_pointer_cast<averaging_downsampler_creator_data<T>>(previous_recording_creator_data);
	  if (!creator_data) {
	    //snde_warning("avg: no previous compatible creator_data");
	    just_starting = true;
	  } else {
	    // access to creator_data synchronized by the implicit ordering of self-dependent math
	    //snde_warning("avg: pending_recs.size() = %llu; numavgs=%llu",(unsigned long long)creator_data->pending_recs.size(),(unsigned long long)numavgs);
	    
	    if (creator_data->pending_recs.size() > 0) {
	      // confirm compatibility
	      // Type matches inherently because of the way this is templated.
	      // Need to check the shape
	      if (to_avg->layout.dimlen != creator_data->pending_recs.at(creator_data->pending_recs.size()-1)->layout.dimlen) {
		// mismatch: clear out creator data
		creator_data->pending_recs.clear();
	      }
	    }
	    creator_data->pending_recs.push_back(to_avg);
	    if (creator_data->pending_recs.size() == numavgs) {
	      new_recording=true;
	    }
	    
	  }
	}
	
      }
      //snde_warning("avg: decide_new_revision, just_starting=%d new_recording=%d",(int)just_starting,(int)new_recording);
      if (just_starting) {
	return std::make_pair(true,std::make_shared<compute_options_function_override_type>([ this, just_starting, to_avg, numavgs, usedouble] {

	  //snde_warning("avg: compute_options just_starting");

	  std::vector<std::shared_ptr<compute_resource_option>> option_list =
	    {
	      std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							    0, //metadata_bytes 
							    0, // data_bytes for transfer
							    0.0, // flops
							    1, // max effective cpu cores
							    1), // useful_cpu_cores (min # of cores to supply
	      
	    };
	  
	  
	  return std::make_pair(option_list,std::make_shared<define_recs_function_override_type>([this, to_avg, numavgs, usedouble] { 
	    
	    // define_recs code
	    std::shared_ptr<null_recording> starting_result;
	    //snde_warning("avg: define_recs just_starting");
	  
	    to_avg->assert_no_scale_or_offset(this->inst->definition->definition_command);
	    
	    starting_result = create_recording_math<null_recording>(this->get_result_channel_path(0),this->rss);
	    return std::make_shared<metadata_function_override_type>([ this, to_avg, numavgs, usedouble, starting_result ]() {
	      // metadata code

	      //snde_warning("avg: metadata just_starting");

	      starting_result->metadata=std::make_shared<immutable_metadata>();
	      starting_result->mark_metadata_done();
	      
	      return std::make_shared<lock_alloc_function_override_type>([ this,to_avg, numavgs, usedouble, starting_result ]() {
		// lock_alloc code
	       
		
		return std::make_shared<exec_function_override_type>([ this, to_avg, numavgs, usedouble, starting_result ]() {
		  // exec code

		  std::shared_ptr<averaging_downsampler_creator_data<T>> new_creator_data = std::make_shared<averaging_downsampler_creator_data<T>>();
		  //snde_warning("avg: exec just_starting");

		  new_creator_data->pending_recs.push_back(to_avg);
		  starting_result->creator_data = new_creator_data;
		  starting_result->mark_data_ready();
		  
		}); 
	      });
	      
	    });
	  }));
	}));
      } else {
	// Not just starting
	assert(creator_data);
	return std::make_pair(new_recording,std::make_shared<compute_options_function_override_type>([ this, to_avg, creator_data, numavgs, usedouble ] {
	  snde_index num_elements = 1;
	  for (size_t dimnum=0;dimnum < to_avg->layout.dimlen.size();dimnum++) {
	    num_elements *= to_avg->layout.dimlen.at(dimnum);
	  }
	  
	  std::vector<std::shared_ptr<compute_resource_option>> option_list =
	    {
	      std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							    0, //metadata_bytes 
							    num_elements * sizeof(T) * (numavgs+1), // data_bytes for transfer
							    num_elements*(numavgs+1)*1.0, // flops
							    1, // max effective cpu cores
							    1), // useful_cpu_cores (min # of cores to supply
	      
	    };
	  
	  
	  return std::make_pair(option_list,std::make_shared<define_recs_function_override_type>([this, creator_data, numavgs, usedouble ] {
	    
	    
	    // define_recs code
	    std::shared_ptr<ndtyped_recording_ref<T>> result_ref;
	    creator_data->pending_recs.at(0)->assert_no_scale_or_offset(this->inst->definition->definition_command);

	    result_ref = create_typed_ndarray_ref_math<T>(this->get_result_channel_path(0),this->rss);
	    
	    return std::make_shared<metadata_function_override_type>([ this, creator_data, numavgs, usedouble, result_ref ]() {
	      // metadata code
	      
	      result_ref->rec->metadata=std::make_shared<immutable_metadata>(*creator_data->pending_recs.at(0)->rec->metadata);
	      result_ref->rec->mark_metadata_done();
	      
	      return std::make_shared<lock_alloc_function_override_type>([ this, creator_data, numavgs, usedouble, result_ref ]() {
		// lock_alloc code
		
		result_ref->allocate_storage(creator_data->pending_recs.at(0)->layout.dimlen,creator_data->pending_recs.at(0)->layout.is_f_contiguous()); 

		
		std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> to_lock;
		to_lock.reserve(creator_data->pending_recs.size()+1);
		
		for (auto && pending_ref: creator_data->pending_recs) {
		  to_lock.push_back(std::make_pair(pending_ref,false)); // lock for read
		}
		to_lock.push_back(std::make_pair(result_ref,true)); // lock for write
		
		rwlock_token_set locktokens = this->lockmgr->lock_recording_refs(to_lock);
		


			return std::make_shared<exec_function_override_type>([this, creator_data, numavgs, usedouble, result_ref, locktokens]() {
				// exec code		    
				std::vector<typename cppfunc_vector_underlying_type<T>::underlying_type* > input_pointers;

				for (size_t inpnum = 0; inpnum < numavgs; inpnum++) {
					input_pointers.push_back((typename cppfunc_vector_underlying_type<T>::underlying_type*)creator_data->pending_recs.at(inpnum)->void_shifted_arrayptr());

					if ((inpnum > 0 && creator_data->pending_recs.at(inpnum)->layout != creator_data->pending_recs.at(inpnum - 1)->layout) ||
						!creator_data->pending_recs.at(inpnum)->layout.is_contiguous()) {
						throw snde_error("averaging_downsampler(%s): Requires all inputs to have the same contiguous layout", this->get_result_channel_path(0).c_str());
					}

				}
				snde_index numelem = creator_data->pending_recs.at(0)->layout.flattened_length() * cppfunc_vector_multiplicity<T>();
				typename cppfunc_vector_underlying_type<T>::underlying_type** raw_input_pointers;
				typename cppfunc_vector_underlying_type<T>::underlying_type* raw_output_pointer;
				raw_input_pointers = input_pointers.data();
				raw_output_pointer = (typename cppfunc_vector_underlying_type<T>::underlying_type*)result_ref->void_shifted_arrayptr();
				if (creator_data->pending_recs.at(0)->layout != result_ref->layout) {
					throw snde_error("averaging_downsampler(%s): Requires input and output to have the same contiguous layout", this->get_result_channel_path(0).c_str());
				}

				snde_index elnum;
				if (usedouble) {
					for (elnum = 0; elnum < numelem; elnum++) {
						snde_float64 accum = 0.0;
						for (size_t inpnum = 0; inpnum < numavgs; inpnum++) {
							// Convert all types to float32 for averaging -- should this be double or should we make it an option to be double?
							accum += (snde_float64)raw_input_pointers[inpnum][elnum];
						}
						raw_output_pointer[elnum] = accum / numavgs;
					}
				}
				else {
					for (elnum = 0; elnum < numelem; elnum++) {
						snde_float32 accum = 0.0;
						for (size_t inpnum = 0; inpnum < numavgs; inpnum++) {
							// Convert all types to float32 for averaging -- should this be double or should we make it an option to be double?
							accum += (snde_float32)raw_input_pointers[inpnum][elnum];
						}
						raw_output_pointer[elnum] = accum / numavgs;
					}
				}			
				unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready()

				// clear out previous recording's creator data so that references within can go away		  
				{
					std::shared_ptr<recording_base> previous_recording = this->self_dependent_recordings.at(0);

					std::lock_guard<std::mutex> prevrec_admin(previous_recording->admin);
					previous_recording->creator_data = nullptr;
				}

				// Create new creator data for the next averaging. 
				std::shared_ptr<averaging_downsampler_creator_data<T>> new_creator_data = std::make_shared<averaging_downsampler_creator_data<T>>();
				result_ref->rec->creator_data = new_creator_data;

				result_ref->rec->mark_data_ready();
				//snde_warning("avg: Generated new result (rev %llu)",(unsigned long long)result_ref->rec->info->revision);
				});
	      });
	      
	    });
	  }));
	}));
      
      }
    }
  
  
    
  };
    
  
  std::shared_ptr<math_function> define_averaging_downsampler_function()
  {
    std::shared_ptr<math_function> newfunc = std::make_shared<cpp_math_function>("snde.averaging_downsampler",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      std::shared_ptr<executing_math_function> executing;
      
	  executing = make_cppfuncexec_integertypes<averaging_temporal_downsampler>(rss, inst);
	  if (!executing) {
      executing = make_cppfuncexec_floatingtypes<averaging_temporal_downsampler>(rss,inst);
	  }
      if (!executing) {
	executing = make_cppfuncexec_vectortypes<averaging_temporal_downsampler>(rss,inst);
      }
      
      if (!executing) {
	throw snde_error("In attempting to call math function %s, first parameter has unsupported data type.",inst->definition->definition_command.c_str());
      }
      return executing;
    });
    //newfunc->self_dependent=true;
    newfunc->new_revision_optional=true;
    return newfunc;
    
  }

  SNDE_API std::shared_ptr<math_function> averaging_downsampler_function=define_averaging_downsampler_function();

  
  static int registered_averaging_downsampler_function = register_math_function(averaging_downsampler_function);
  
  
  
  
};
