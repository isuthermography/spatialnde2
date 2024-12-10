

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


#include "snde/mean.hpp"

namespace snde {

  
  template <typename T>
  class mean: public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,std::vector<snde_index>,snde_bool> {
  public:
    mean(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>, std::vector<snde_index>, snde_bool>(rss,inst)
    {
      
    }

    // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,std::vector<snde_index>,snde_bool>::compute_options_function_override_type compute_options_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,std::vector<snde_index>,snde_bool>::define_recs_function_override_type define_recs_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,std::vector<snde_index>,snde_bool>::metadata_function_override_type metadata_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,std::vector<snde_index>,snde_bool>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,std::vector<snde_index>,snde_bool>::exec_function_override_type exec_function_override_type;

    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>, std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<ndtyped_recording_ref<T>> to_avg, std::vector<snde_index> axis = {}, snde_bool keepdims = false)
    {

      snde_index numbytes = 0;
      std::vector<snde_index> outdims = {};
      T junk;

      // Calcualte size of new array and determine mode of operation
      if (axis.size() == 0) {
	// We didn't define an axis
	numbytes = 1;
	outdims.push_back(1);
      }
      else {
	outdims = to_avg->layout.dimlen;
	std::sort(axis.begin(), axis.end(), std::greater<>());
	for (auto&& curaxis : axis) {
	  outdims.erase(outdims.begin() + curaxis);
	}
      }

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
      {
	std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
						      0, //metadata_bytes 
						      numbytes * sizeof(junk), // data_bytes for transfer
						      0.0, // flops
						      1, // max effective cpu cores
						      1), // useful_cpu_cores (min # of cores to supply
      };
      return std::make_pair(option_list, std::make_shared<define_recs_function_override_type>([this, to_avg, axis, keepdims, outdims]() {

	    // define_recs code
	    std::shared_ptr<ndtyped_recording_ref<T>> result_ref;

	    result_ref = create_typed_ndarray_ref_math<T>(this->get_result_channel_path(0),this->rss);
	    
	    return std::make_shared<metadata_function_override_type>([ this, to_avg, axis, keepdims, outdims, result_ref]() {
	      // metadata code
	      
	      result_ref->rec->metadata=std::make_shared<immutable_metadata>(*to_avg->rec->metadata);
	      result_ref->rec->mark_metadata_done();
	      
	      return std::make_shared<lock_alloc_function_override_type>([this, to_avg, axis, keepdims, outdims, result_ref]() {
		// lock_alloc code
		
		result_ref->allocate_storage(outdims,to_avg->layout.is_f_contiguous()); 

		
		std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> to_lock;
		
                to_lock.push_back(std::make_pair(to_avg,false)); // lock for read
		to_lock.push_back(std::make_pair(result_ref,true)); // lock for write
		
		rwlock_token_set locktokens = this->lockmgr->lock_recording_refs(to_lock);
		


			return std::make_shared<exec_function_override_type>([this, to_avg, axis, keepdims, outdims, result_ref, locktokens]() {
				// exec code		    

				snde_float64 avgval = 0.0;

				snde_index total = to_avg->layout.flattened_length();

				T* dataptr = static_cast<T*>(to_avg->void_shifted_arrayptr());

				for (snde_index i = 0; i < total; ++i) {
				  avgval += (snde_float64)dataptr[i];
				}

				avgval /= total;

				result_ref->element({ 0 }) = static_cast<T>(avgval);
	
				unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready()

				result_ref->rec->mark_data_ready();
				//snde_warning("avg: Generated new result (rev %llu)",(unsigned long long)result_ref->rec->info->revision);
				});
	      });
	      
	    });
	  }));
      
      };

  
  
    
  };
    
  
  std::shared_ptr<math_function> define_mean_function()
  {
    std::shared_ptr<math_function> newfunc = std::make_shared<cpp_math_function>("snde.mean",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      std::shared_ptr<executing_math_function> executing;
      
	  executing = make_cppfuncexec_integertypes<mean>(rss, inst);
	  if (!executing) {
      executing = make_cppfuncexec_floatingtypes<mean>(rss,inst);
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

  SNDE_API std::shared_ptr<math_function> mean_function=define_mean_function();

  
  static int registered_mean_function = register_math_function(mean_function);
  
  
  
  
};
