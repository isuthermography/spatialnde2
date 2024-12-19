

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


#include "snde/nd_accumulate_once.hpp"

namespace snde {


  template <typename T>
  class nd_accumulate_once_creator_data : public recording_creator_data {
  public:

    // Explicitly Keep first recording to track shape and data type
    std::shared_ptr<ndtyped_recording_ref<T>> first_rec;

    // Vector to hold pointers in emit only at end mode
    std::vector<std::shared_ptr<ndtyped_recording_ref<T>>> pending_recs;

    // Keep track of accumulated count
    uint64_t accumulated = 0;

    // Keep track of finished state
    bool finished = false;

    nd_accumulate_once_creator_data() = default;
    // rule of 3
    nd_accumulate_once_creator_data& operator=(const nd_accumulate_once_creator_data&) = delete;
    nd_accumulate_once_creator_data(const nd_accumulate_once_creator_data& orig) = delete;
    virtual ~nd_accumulate_once_creator_data() = default;

  };


  template <typename T>
  class nd_accumulate_once : public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>, std::vector<snde_index>, std::vector<snde_index>, snde_index, snde_bool, snde_bool, snde_bool, std::shared_ptr<snde::constructible_metadata>> {
  public:
    nd_accumulate_once(std::shared_ptr<recording_set_state> rss, std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>, std::vector<snde_index>, std::vector<snde_index>, snde_index, snde_bool, snde_bool, snde_bool, std::shared_ptr<snde::constructible_metadata>>(rss, inst)
    {

    }

    // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>, std::vector<snde_index>, std::vector<snde_index>, snde_index, snde_bool, snde_bool, snde_bool, std::shared_ptr<snde::constructible_metadata>>::compute_options_function_override_type compute_options_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>, std::vector<snde_index>, std::vector<snde_index>, snde_index, snde_bool, snde_bool, snde_bool, std::shared_ptr<snde::constructible_metadata>>::define_recs_function_override_type define_recs_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>, std::vector<snde_index>, std::vector<snde_index>, snde_index, snde_bool, snde_bool, snde_bool, std::shared_ptr<snde::constructible_metadata>>::metadata_function_override_type metadata_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>, std::vector<snde_index>, std::vector<snde_index>, snde_index, snde_bool, snde_bool, snde_bool, std::shared_ptr<snde::constructible_metadata>>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>, std::vector<snde_index>, std::vector<snde_index>, snde_index, snde_bool, snde_bool, snde_bool, std::shared_ptr<snde::constructible_metadata>>::exec_function_override_type exec_function_override_type;


    // Recording to Accumulate, Number of Recordings to Accumulate, Axis to Accumulate Along 
    // Axis to accumulate along can be -2, -1, 0, to_accum->layout.dimlen.size() - 1, or to_accum->layout.dimlen.size(), to_accum->layout.dimlen.size() + 1
    // Using -1 or to_accum->layout.dimlen.size() will add an additional axis to accumulate along
    // Using 0 or to_accum->layout.dimlen.size() - 1 will concatenate along the first or last axis
    // Using -2 or to_accum->layout.dimlen.size() + 1 will place all incoming recordings into new arrays instead
    std::pair<bool, std::shared_ptr<compute_options_function_override_type>> decide_execution(std::shared_ptr<ndtyped_recording_ref<T>> to_accum, std::vector<snde_index> accum_dimlen, std::vector<snde_index> assignmap, snde_index numaccum, snde_bool fortran_mode, snde_bool emit_when_complete_only, snde_bool auto_reset, std::shared_ptr<snde::constructible_metadata> metadata) {
      std::shared_ptr<recording_base> previous_recording = this->self_dependent_recordings.at(0);
      std::shared_ptr<recording_creator_data> previous_recording_creator_data;
      std::shared_ptr<nd_accumulate_once_creator_data<T>> creator_data;
      bool just_starting = false;
      bool will_run = true;

      auto clearcheck = executing_math_function::msgs.find("clear");
      if (clearcheck != executing_math_function::msgs.end()) {
	// Add check here for bool = True
	just_starting = true;
	if (emit_when_complete_only) {
	  std::shared_ptr<nd_accumulate_once_creator_data<T>> new_creator_data = std::make_shared<nd_accumulate_once_creator_data<T>>();
	  //snde_warning("avg: exec just_starting");
	  will_run = false;
	  new_creator_data->first_rec = to_accum;
	  new_creator_data->pending_recs.push_back(to_accum);
	  new_creator_data->accumulated = 1;
	  previous_recording->creator_data = new_creator_data;
		
	}
      }
      else {
	// Is there a previous recording
	if (!previous_recording) {  // No - flag that we are just starting
	  just_starting = true;
	  if (emit_when_complete_only) {
	    return std::make_pair(true,std::make_shared<compute_options_function_override_type>([ this, just_starting] {

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
	  
	  
	      return std::make_pair(option_list,std::make_shared<define_recs_function_override_type>([this] { 
	    
		// define_recs code
		std::shared_ptr<null_recording> starting_result;
		//snde_warning("avg: define_recs just_starting");
	  
		//to_avg->assert_no_scale_or_offset(this->inst->definition->definition_command);
	    
		starting_result = create_recording_math<null_recording>(this->get_result_channel_path(0),this->rss);
		return std::make_shared<metadata_function_override_type>([ this, starting_result ]() {
		  // metadata code

		  //snde_warning("avg: metadata just_starting");

		  starting_result->metadata=std::make_shared<immutable_metadata>();
		  starting_result->mark_metadata_done();
	      
		  return std::make_shared<lock_alloc_function_override_type>([ this, starting_result ]() {
		    // lock_alloc code
	       
		
		    return std::make_shared<exec_function_override_type>([ this, starting_result ]() {
		      // exec code

		      //std::shared_ptr<averaging_downsampler_creator_data<T>> new_creator_data = std::make_shared<averaging_downsampler_creator_data<T>>();
		      //snde_warning("avg: exec just_starting");

		      // new_creator_data->pending_recs.push_back(to_avg);
		      //starting_result->creator_data = new_creator_data;
		      starting_result->mark_data_ready();
		  
		    }); 
		  });
		  
		});
	      }));
	    }));
	  }
	}
      
	else {  // Yes -- there is a previous recording, let's check it
	{
	    // Get handle to creator data for previous recording
	    std::lock_guard<std::mutex> prevrec_admin(previous_recording->admin);
	    previous_recording_creator_data = previous_recording->creator_data;
	  }

	  // First check if we have auto_reset and we're finished
	  if (previous_recording_creator_data) {
	    creator_data = std::dynamic_pointer_cast<nd_accumulate_once_creator_data<T>>(previous_recording_creator_data);
	    if (auto_reset && creator_data->finished) {
	      creator_data = nullptr;
	      previous_recording_creator_data = nullptr;
	    }
	  }

	  // Did we get previous recording creator data
	  if (!previous_recording_creator_data) { // No

	    just_starting = true;

	    // If we are in emit only when complete mode, we need to return null recording with creator_data
	    if (emit_when_complete_only) {
		std::shared_ptr<nd_accumulate_once_creator_data<T>> new_creator_data = std::make_shared<nd_accumulate_once_creator_data<T>>();
		//snde_warning("avg: exec just_starting");
		will_run = false;
		new_creator_data->first_rec = to_accum;
		new_creator_data->pending_recs.push_back(to_accum);
		new_creator_data->accumulated = 1;
		previous_recording->creator_data = new_creator_data;

	    } 
	    
	  }
	  else { // Yes -- let's check it out

	    // Get the creator_data type casted and set up -- done above
	    // creator_data = std::dynamic_pointer_cast<nd_accumulate_once_creator_data<T>>(previous_recording_creator_data);

	    // Check if the finished flag is set -- then we shouldn't run
	    if (creator_data->finished) {
	      will_run = false;
	    }
	    else {  // No finished flag -- let's check for incoming recording
	      
	      // There are two modes of operation -- emitting while accumulating or only emitting when complete
	      // let's do a quick check here first to determine the mode of operation
	      if (emit_when_complete_only) {
		
		  if (creator_data->pending_recs.size() > 0) {
		    // confirm compatibility
		    // Type matches inherently because of the way this is templated.
		    // Need to check the shape
		    if (to_accum->layout.dimlen != creator_data->pending_recs.at(creator_data->pending_recs.size() - 1)->layout.dimlen) {
		      // mismatch: clear out creator data
		      creator_data->pending_recs.clear();
		    }
		  }
		  creator_data->pending_recs.push_back(to_accum);
		  will_run = false;

		  if (creator_data->pending_recs.size() >= numaccum) { // we are finishing here -- we should run
		    will_run = true;
		    just_starting = true;
		  }
		  
		
	      }
	      else {
		if (!to_accum) {
		  will_run = false;
		}
		else {  // Got a recording, let's check it's dimensions and data type for consistency
		  if (to_accum->layout.dimlen != creator_data->first_rec->layout.dimlen || to_accum->ndinfo()->typenum != creator_data->first_rec->ndinfo()->typenum) {
		    just_starting = true; // Something has changed -- let's restart
		  }
		  //else { // Everything looks good -- we should run and continue previous accumulation

		  //}
		}
	      }	      
	    }
	  }
	}
      }

      if (!will_run) { // Return back indicating we are not going to run
	return std::make_pair(false, nullptr);
      }
      else { // We are going to run
	return std::make_pair(true, std::make_shared<compute_options_function_override_type>([this, just_starting, previous_recording, creator_data, to_accum, accum_dimlen, assignmap, fortran_mode, emit_when_complete_only, numaccum, metadata]() {

	  snde_index nbytes = to_accum->storage->elementsize;

	  for (const auto& e : accum_dimlen)
	    nbytes *= e;

	  std::vector<std::shared_ptr<compute_resource_option>> option_list =
	  {
	    std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							  0, //metadata_bytes 
							  nbytes, // data_bytes for transfer
							  0.0, // flops
							  1, // max effective cpu cores
							  1), // useful_cpu_cores (min # of cores to supply

	  };
	  return std::make_pair(option_list, std::make_shared<define_recs_function_override_type>([this, just_starting, previous_recording, creator_data, to_accum, accum_dimlen, assignmap, fortran_mode, emit_when_complete_only, numaccum, metadata]() {
	    // define_recs code

	    std::shared_ptr<multi_ndarray_recording> result;

	    to_accum->assert_no_scale_or_offset(this->inst->definition->definition_command);

	    result = create_recording_math<multi_ndarray_recording>(this->get_result_channel_path(0), this->rss, 1);

	    return std::make_shared<metadata_function_override_type>([this, result, fortran_mode, accum_dimlen, just_starting, previous_recording, creator_data, to_accum, assignmap, emit_when_complete_only, numaccum, metadata]() {
	      // metadata code 

	      //snde_warning("avg: metadata just_starting");

	      std::shared_ptr<multi_ndarray_recording> previous_recording_ndarray;
	      previous_recording_ndarray = std::dynamic_pointer_cast<multi_ndarray_recording>(previous_recording);

	      /*
	      // In this mode -- we cannot make any assumptions about metadata.  Therefore, we require the user to supply it.
	      
	      std::shared_ptr<constructible_metadata> metadata;

	      if (!just_starting) {  // Make sure we have a previous array if we aren't just starting
		assert(previous_recording_ndarray);
		metadata = std::make_shared<constructible_metadata>(*previous_recording_ndarray->metadata);
	      }
	      else {
		metadata = std::make_shared<constructible_metadata>(*to_accum->rec->metadata);
	      }
	      */

	      result->metadata = metadata;
	      result->mark_metadata_done();

	      return std::make_shared<lock_alloc_function_override_type>([this, result, fortran_mode, accum_dimlen, just_starting, previous_recording, previous_recording_ndarray, creator_data, to_accum, assignmap, emit_when_complete_only, numaccum]() {
		// lock_alloc code

		std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>, bool>> to_lock;
		to_lock.reserve(2);

		// new array number
		snde_index arraynum = 0;

		// Lock the input for read access
		if (!emit_when_complete_only)
		{
		  to_lock.push_back(std::make_pair(to_accum, false));
		}

		// Allocate or assign all arrays
		if (just_starting) { // Just getting started -- allocate a new array
		  if (emit_when_complete_only) {
		    // we need to loop over all of the inputs to lock
		    for (snde_index i = 0; i < creator_data->pending_recs.size(); i++) {
		      to_lock.push_back(std::make_pair(creator_data->pending_recs.at(i), false));
		    }
		  }
		  result->define_array(0, to_accum->typenum);
		  result->allocate_storage(0, accum_dimlen, fortran_mode);
		  to_lock.push_back(std::make_pair(result->reference_ndarray(0), true));
		}
		else { // Re-use existing array
		  result->define_array(0, to_accum->typenum);
		  result->assign_storage_portion(previous_recording_ndarray->storage.at(0), 0, accum_dimlen, fortran_mode, 0);
		  to_lock.push_back(std::make_pair(result->reference_ndarray(0), true));
		}

		rwlock_token_set locktokens = this->lockmgr->lock_recording_refs(to_lock, false); // (false -> non_gpu) 

		if (emit_when_complete_only) {
		  return std::make_shared<exec_function_override_type>([this, locktokens, result, fortran_mode, accum_dimlen, just_starting, previous_recording, previous_recording_ndarray, creator_data, assignmap, numaccum]() {
		    
		      // exec code -- we need to loop over all accumulated arrays and copy the data
		      snde_index accumindex = 0;

		      snde_index elements_in_single_entry = 1;
		      for (auto&& axislen : creator_data->first_rec->layout.dimlen) {
			elements_in_single_entry *= axislen;
		      }
		      snde_index bytes_in_single_entry = elements_in_single_entry * creator_data->first_rec->ndinfo()->elementsize;


		      for (auto&& to_accum : creator_data->pending_recs) {


			// Copy data
			std::vector<snde_index> curelement(accum_dimlen.size());  //assignmap.at(accumindex)
			snde_index inputindex = assignmap.at(accumindex);
			if (fortran_mode) {
			  for (size_t i = 0; i < accum_dimlen.size(); ++i) {
			    curelement[i] = inputindex % accum_dimlen[i];
			    inputindex /= accum_dimlen[i];
			  }
			}
			else {
			  for (size_t i = accum_dimlen.size(); i > 0; --i) {
			    curelement[i-1] = inputindex % accum_dimlen[i-1];
			    inputindex /= accum_dimlen[i-1];
			  }
			}
			memcpy(result->element_dataptr(0, curelement), to_accum->void_shifted_arrayptr(), bytes_in_single_entry);
			

			accumindex += 1;

		      } 
		    

		    // Unlock all refs
		    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready()

		    // Create new creator_data
		    std::shared_ptr<nd_accumulate_once_creator_data<T>> new_creator_data = std::make_shared<nd_accumulate_once_creator_data<T>>();
		    new_creator_data->accumulated = creator_data->pending_recs.size();
		    new_creator_data->first_rec = creator_data->first_rec;    

		    // Set finished		    
		    new_creator_data->finished = true;
		   

		    // Assign creator_data
		    result->creator_data = new_creator_data;

		    // clear out previous recording's creator data so that references within can go away		  
		    {
		      std::lock_guard<std::mutex> prevrec_admin(previous_recording->admin);
		      previous_recording->creator_data = nullptr;
		    }

		    result->mark_data_ready();


		  });
		}
		else {
		  return std::make_shared<exec_function_override_type>([this, locktokens, result, arraynum, fortran_mode, accum_dimlen, just_starting, previous_recording, previous_recording_ndarray, creator_data, to_accum, assignmap, numaccum]() {
		    // exec code

		    snde_index accumindex = 0;
		    if (!just_starting) {
		      accumindex = creator_data->accumulated;
		    

		      snde_index elements_in_single_entry = 1;
		      for (auto&& axislen : to_accum->layout.dimlen) {
			elements_in_single_entry *= axislen;
		      }
		      snde_index bytes_in_single_entry = elements_in_single_entry * to_accum->ndinfo()->elementsize;
		    		    
		      // Copy data
		      std::vector<snde_index> curelement(accum_dimlen.size());  //assignmap.at(accumindex)
		      snde_index inputindex = assignmap.at(accumindex);
		      if (fortran_mode) {
			for (size_t i = 0; i < accum_dimlen.size(); ++i) {
			  curelement[i] = inputindex % accum_dimlen[i];
			  inputindex /= accum_dimlen[i];
			}
		      }
		      else {
			for (size_t i = accum_dimlen.size(); i > 0; --i) {
			  curelement[i-1] = inputindex % accum_dimlen[i-1];
			  inputindex /= accum_dimlen[i-1];
			}
		      }
		      memcpy(result->element_dataptr(arraynum, curelement), to_accum->void_shifted_arrayptr(), bytes_in_single_entry);

		    
		      // Mark Element as Modified
		      result->storage.at(0)->mark_as_modified(NULL, result->element_offset(0, curelement), elements_in_single_entry, true);
		    }

		    // Unlock all refs
		    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready()

		    // Create new creator_data
		    std::shared_ptr<nd_accumulate_once_creator_data<T>> new_creator_data = std::make_shared<nd_accumulate_once_creator_data<T>>();
		    if (just_starting) {
		      new_creator_data->accumulated = 0;
		      new_creator_data->first_rec = to_accum;
		    }
		    else {
		      if (creator_data->accumulated == 0) {
			new_creator_data->accumulated = 1;
			new_creator_data->first_rec = to_accum;
		      }
		      else {
			new_creator_data->accumulated = creator_data->accumulated + 1;
			new_creator_data->first_rec = creator_data->first_rec;
		      }
		      
		    }		  

		    // Check if finished
		    if (new_creator_data->accumulated >= numaccum) {
		      new_creator_data->finished = true;
		    }

		    // Assign creator_data
		    result->creator_data = new_creator_data;

		    // clear out previous recording's creator data so that references within can go away		  
		    if(previous_recording)
		    {
		      std::lock_guard<std::mutex> prevrec_admin(previous_recording->admin);
		      previous_recording->creator_data = nullptr;
		    }

		    result->mark_data_ready();

		    });
		  
		}
		});
	      });  
	  })); 
	}));
      }
    }
  };
    
  
  std::shared_ptr<math_function> define_nd_accumulate_once_function()
  {
    std::shared_ptr<math_function> newfunc = std::make_shared<cpp_math_function>("snde.nd_accumulate_once",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      std::shared_ptr<executing_math_function> executing;
      
      executing = make_cppfuncexec_floatingtypes<nd_accumulate_once>(rss,inst);
      if (!executing) {
	executing = make_cppfuncexec_vectortypes<nd_accumulate_once>(rss,inst);
      }
      if (!executing) {
	executing = make_cppfuncexec_complextypes<nd_accumulate_once>(rss,inst);
      }
      if (!executing) {
	throw snde_error("In attempting to call math function %s, first parameter has unsupported data type.",inst->definition->definition_command.c_str());
      }
      return executing;
    });
    newfunc->self_dependent=true;
    newfunc->new_revision_optional = true;
    return newfunc;
    
  }

  SNDE_API std::shared_ptr<math_function> nd_accumulate_once_function=define_nd_accumulate_once_function();

  static int registered_nd_accumulate_once_function = register_math_function(nd_accumulate_once_function);
  

  
  
};
