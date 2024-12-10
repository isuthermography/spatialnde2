

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


#include "snde/accumulate_once.hpp"

namespace snde {


  template <typename T>
  class accumulate_once_creator_data : public recording_creator_data {
  public:

    // Explicitly Keep first recording to track shape and data type
    std::shared_ptr<ndtyped_recording_ref<T>> first_rec;

    // Vector to hold pointers in emit only at end mode
    std::vector<std::shared_ptr<ndtyped_recording_ref<T>>> pending_recs;

    // Keep track of accumulated count
    uint64_t accumulated = 0;

    // Keep track of finished state
    bool finished = false;

    accumulate_once_creator_data() = default;
    // rule of 3
    accumulate_once_creator_data& operator=(const accumulate_once_creator_data&) = delete;
    accumulate_once_creator_data(const accumulate_once_creator_data& orig) = delete;
    virtual ~accumulate_once_creator_data() = default;

  };


  template <typename T>
  class accumulate_once : public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>, uint64_t, int64_t, snde_bool, snde_bool> {
  public:
    accumulate_once(std::shared_ptr<recording_set_state> rss, std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>, uint64_t, int64_t, snde_bool, snde_bool>(rss, inst)
    {

    }

    // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>, uint64_t, int64_t, snde_bool, snde_bool>::compute_options_function_override_type compute_options_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>, uint64_t, int64_t, snde_bool, snde_bool>::define_recs_function_override_type define_recs_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>, uint64_t, int64_t, snde_bool, snde_bool>::metadata_function_override_type metadata_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>, uint64_t, int64_t, snde_bool, snde_bool>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>, uint64_t, int64_t, snde_bool, snde_bool>::exec_function_override_type exec_function_override_type;


    // Recording to Accumulate, Number of Recordings to Accumulate, Axis to Accumulate Along 
    // Axis to accumulate along can be -2, -1, 0, to_accum->layout.dimlen.size() - 1, or to_accum->layout.dimlen.size(), to_accum->layout.dimlen.size() + 1
    // Using -1 or to_accum->layout.dimlen.size() will add an additional axis to accumulate along
    // Using 0 or to_accum->layout.dimlen.size() - 1 will concatenate along the first or last axis
    // Using -2 or to_accum->layout.dimlen.size() + 1 will place all incoming recordings into new arrays instead
    std::pair<bool, std::shared_ptr<compute_options_function_override_type>> decide_execution(std::shared_ptr<ndtyped_recording_ref<T>> to_accum, uint64_t numaccum, int64_t axis, snde_bool emit_when_complete_only, snde_bool auto_reset) {
      std::shared_ptr<recording_base> previous_recording = this->self_dependent_recordings.at(0);
      std::shared_ptr<recording_creator_data> previous_recording_creator_data;
      std::shared_ptr<accumulate_once_creator_data<T>> creator_data;
      bool just_starting = false;
      bool will_run = true;

      auto clearcheck = executing_math_function::msgs.find("clear");
      if (clearcheck != executing_math_function::msgs.end()) {
	// Add check here for bool = True
	just_starting = true;
	if (emit_when_complete_only) {
	  std::shared_ptr<accumulate_once_creator_data<T>> new_creator_data = std::make_shared<accumulate_once_creator_data<T>>();
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
	    creator_data = std::dynamic_pointer_cast<accumulate_once_creator_data<T>>(previous_recording_creator_data);
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
		std::shared_ptr<accumulate_once_creator_data<T>> new_creator_data = std::make_shared<accumulate_once_creator_data<T>>();
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
	    // creator_data = std::dynamic_pointer_cast<accumulate_once_creator_data<T>>(previous_recording_creator_data);

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
	return std::make_pair(true, std::make_shared<compute_options_function_override_type>([this, just_starting, previous_recording, creator_data, to_accum, numaccum, axis, emit_when_complete_only]() {

	  snde_index nbytes = to_accum->layout.flattened_length() * to_accum->storage->elementsize * numaccum;

	  std::vector<std::shared_ptr<compute_resource_option>> option_list =
	  {
	    std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							  0, //metadata_bytes 
							  nbytes, // data_bytes for transfer
							  0.0, // flops
							  1, // max effective cpu cores
							  1), // useful_cpu_cores (min # of cores to supply

	  };
	  return std::make_pair(option_list, std::make_shared<define_recs_function_override_type>([this, just_starting, previous_recording, creator_data, to_accum, numaccum, axis, emit_when_complete_only]() {
	    // define_recs code

	    std::shared_ptr<multi_ndarray_recording> result;
	    snde_index num_arrays = 0;
	    std::vector<snde_index> accum_dimlen;
	    bool fortran_mode = false;
	    uint8_t accum_mode = 0;
	    snde_index accumulated_increment = 1;

	    to_accum->assert_no_scale_or_offset(this->inst->definition->definition_command);
	    
	    // Let's determine mode of operation -- this should probably get cached in creator_data
	    if (axis == -2) {					    // Accumulate as new arrays at the beginning
	      if (just_starting) {
		num_arrays = 1;
	      }
	      else {
		num_arrays = creator_data->accumulated + 1;
	      }
	      if (emit_when_complete_only) {
		num_arrays = creator_data->pending_recs.size();
	      }
	      fortran_mode = false;
	      accum_mode = SNDE_ACCUM_NEW_NDARRAY;
	      accum_dimlen = std::vector<snde_index>(to_accum->layout.dimlen.begin(), to_accum->layout.dimlen.end());
	    }
	    else if (axis == -1) {				    // Accumulate along a new axis at beginning
	      num_arrays = 1;
	      fortran_mode = false;
	      accum_mode = SNDE_ACCUM_NEW_AXIS;
	      accum_dimlen.push_back(numaccum);
	      accum_dimlen.insert(accum_dimlen.end(), to_accum->layout.dimlen.begin(),  to_accum->layout.dimlen.end());
	    }
	    else if (axis == 0) {				    // Concatenate along an existing axis at beginning
	      num_arrays = 1;
	      fortran_mode = false;
	      accum_mode = SNDE_ACCUM_CONCATENATE;
	      accumulated_increment = to_accum->layout.dimlen.at(0);
	      if ((numaccum % accumulated_increment) != 0) {
		throw snde_error("accumulate_once: numaccum must be evenly divided by first dimension of to_accum");
	      }
	      accum_dimlen.push_back(numaccum);
	      accum_dimlen.insert(accum_dimlen.end(), to_accum->layout.dimlen.begin()+1, to_accum->layout.dimlen.end());
	    }
	    else if (axis == to_accum->layout.dimlen.size() - 1) {  // Concatenate along an existing axis at end
	      num_arrays = 1;
	      fortran_mode = true;
	      accum_mode = SNDE_ACCUM_CONCATENATE;
	      accumulated_increment = to_accum->layout.dimlen.back();
	      if ((numaccum % accumulated_increment) != 0) {
		throw snde_error("accumulate_once: numaccum must be evenly divided by last dimension of to_accum");
	      }
	      accum_dimlen.insert(accum_dimlen.end(), to_accum->layout.dimlen.begin(), to_accum->layout.dimlen.end()-1);
	      accum_dimlen.push_back(numaccum);
	    }
	    else if (axis == to_accum->layout.dimlen.size()) {	    // Accumulate along a new axis at end
	      num_arrays = 1;
	      fortran_mode = true;
	      accum_mode = SNDE_ACCUM_NEW_AXIS;
	      accum_dimlen.insert(accum_dimlen.end(), to_accum->layout.dimlen.begin(), to_accum->layout.dimlen.end());
	      accum_dimlen.push_back(numaccum);
	    }
	    else if (axis == to_accum->layout.dimlen.size() + 1) {  // Accumulate as new arrays at the end
	      if (just_starting) {
		num_arrays = 1;
	      }
	      else {
		num_arrays = creator_data->accumulated + 1;
	      }
	      if (emit_when_complete_only) {
		num_arrays = creator_data->pending_recs.size();
	      }
	      fortran_mode = true;
	      accum_mode = SNDE_ACCUM_NEW_NDARRAY;
	      accum_dimlen = std::vector<snde_index>(to_accum->layout.dimlen.begin(), to_accum->layout.dimlen.end());
	    }
	    else {
	      throw snde_error("accumulate_once: parameter 'axis' must be -2, -1, 0, to_accum->layout.dimlen.size() - 1, to_accum->layout.dimlen.size(), or to_accum->layout.dimlen.size() + 1");
	    }

	    result = create_recording_math<multi_ndarray_recording>(this->get_result_channel_path(0), this->rss, num_arrays);

	    return std::make_shared<metadata_function_override_type>([this, result, num_arrays, accumulated_increment, fortran_mode, accum_mode, accum_dimlen, just_starting, previous_recording, creator_data, to_accum, numaccum, axis, emit_when_complete_only]() {
	      // metadata code 

	      //snde_warning("avg: metadata just_starting");

	      std::shared_ptr<multi_ndarray_recording> previous_recording_ndarray;
	      previous_recording_ndarray = std::dynamic_pointer_cast<multi_ndarray_recording>(previous_recording);

	      std::shared_ptr<constructible_metadata> metadata;

	      if (!just_starting) {  // Make sure we have a previous array if we aren't just starting
		assert(previous_recording_ndarray);
		metadata = std::make_shared<constructible_metadata>(*previous_recording_ndarray->metadata);
	      }
	      else {
		metadata = std::make_shared<constructible_metadata>(*to_accum->rec->metadata);
	      }

	      result->metadata = metadata;
	      result->mark_metadata_done();

	      return std::make_shared<lock_alloc_function_override_type>([this, result, num_arrays, accumulated_increment, fortran_mode, accum_mode, accum_dimlen, just_starting, previous_recording, previous_recording_ndarray, creator_data, to_accum, numaccum, axis, emit_when_complete_only]() {
		// lock_alloc code

		std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>, bool>> to_lock;
		to_lock.reserve(num_arrays + 1);

		// new array number
		snde_index arraynum = 0;

		// Lock the input for read access
		if (!emit_when_complete_only)
		{
		  to_lock.push_back(std::make_pair(to_accum, false));
		}

		// Allocate or assign all arrays
		if ((accum_mode == SNDE_ACCUM_NEW_AXIS) || (accum_mode == SNDE_ACCUM_CONCATENATE)) { // Only One Array
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
		}
		else { // Multiple ND Arrays
		  if (just_starting) { // Just getting started -- allocate a new ndarray
		    if (emit_when_complete_only) {
		      // we need to loop over all of the arrays to be reused
		      for (snde_index i = 0; i < creator_data->pending_recs.size(); i++) {
			result->define_array(i, to_accum->typenum);
			//result->allocate_storage(i, accum_dimlen, fortran_mode);
			result->assign_storage_portion(creator_data->pending_recs.at(i)->storage, i, accum_dimlen, fortran_mode, 0);
			//to_lock.push_back(std::make_pair(result->reference_ndarray(i), true)); //no need to lock -- not copying
		      }
		    }
		    else {
		      result->define_array(0, to_accum->typenum);
		      result->allocate_storage(0, accum_dimlen, fortran_mode);
		      to_lock.push_back(std::make_pair(result->reference_ndarray(0), true));
		    }
		  }
		  else { // Previous data exists
		    snde_index cnt = 0;
		    for (int i = 0; i < num_arrays; i++) {  // Loop over all arrays

		      // Define This Array's Type
		      result->define_array(i, to_accum->typenum);  

		      // Let's allocate or assign storage
		      if ((i == 0 && fortran_mode) || (i == num_arrays-1 && (!fortran_mode))) { 
			// New array at beginning or end depending on fortran_mode
			result->allocate_storage(i, accum_dimlen, fortran_mode);
			to_lock.push_back(std::make_pair(result->reference_ndarray(i), true));
			arraynum = i;
		      }
		      else {
			// Existing array we need to copy over
			result->assign_storage_portion(previous_recording_ndarray->storage.at(cnt), i, accum_dimlen, fortran_mode, 0);
			to_lock.push_back(std::make_pair(result->reference_ndarray(i), false));
			cnt += 1;
		      }		      
		    }
		  }
		}	

		rwlock_token_set locktokens = this->lockmgr->lock_recording_refs(to_lock, false); // (false -> non_gpu) 

		if (emit_when_complete_only) {
		  return std::make_shared<exec_function_override_type>([this, locktokens, result, num_arrays, accumulated_increment, fortran_mode, accum_mode, accum_dimlen, just_starting, previous_recording, previous_recording_ndarray, creator_data, numaccum, axis]() {
		    
		    
		    if (accum_mode != SNDE_ACCUM_NEW_NDARRAY) {
		      // exec code -- we need to loop over all accumulated arrays and copy the data
		      snde_index accumindex = 0;

		      snde_index elements_in_single_entry = 1;
		      for (auto&& axislen : creator_data->first_rec->layout.dimlen) {
			elements_in_single_entry *= axislen;
		      }
		      snde_index bytes_in_single_entry = elements_in_single_entry * creator_data->first_rec->ndinfo()->elementsize;


		      for (auto&& to_accum : creator_data->pending_recs) {

			std::vector<snde_index> element_addr(accum_dimlen.size(), 0);


			snde_index newaxis = 0;  //c mode
			if (fortran_mode) {
			  newaxis = accum_dimlen.size() - 1;  // fortran mode
			}
			element_addr.at(newaxis) = accumindex;



			// Copy data
			memcpy(result->element_dataptr(0, element_addr), to_accum->void_shifted_arrayptr(), bytes_in_single_entry);
			

			accumindex += 1;

		      }

		    }
		    

		    // Unlock all refs
		    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready()

		    // Create new creator_data
		    std::shared_ptr<accumulate_once_creator_data<T>> new_creator_data = std::make_shared<accumulate_once_creator_data<T>>();
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
		  return std::make_shared<exec_function_override_type>([this, locktokens, result, arraynum, num_arrays, accumulated_increment, fortran_mode, accum_mode, accum_dimlen, just_starting, previous_recording, previous_recording_ndarray, creator_data, to_accum, numaccum, axis]() {
		    // exec code

		    snde_index accumindex = 0;
		    if (!just_starting) {
		      accumindex = creator_data->accumulated;
		    }

		    snde_index elements_in_single_entry = 1;
		    for (auto&& axislen : to_accum->layout.dimlen) {
		      elements_in_single_entry *= axislen;
		    }
		    snde_index bytes_in_single_entry = elements_in_single_entry * to_accum->ndinfo()->elementsize;

		    std::vector<snde_index> element_addr(accum_dimlen.size(), 0);

		    if (accum_mode == SNDE_ACCUM_NEW_NDARRAY) {
		      accumindex += 1;
		    }
		    else if (accum_mode == SNDE_ACCUM_NEW_AXIS || accum_mode == SNDE_ACCUM_CONCATENATE) {
		      snde_index newaxis = 0;  //c mode
		      if (fortran_mode) {
			newaxis = accum_dimlen.size() - 1;  // fortran mode
		      }
		      element_addr.at(newaxis) = accumindex;		    
		    }

		    // Copy data
		    memcpy(result->element_dataptr(arraynum, element_addr), to_accum->void_shifted_arrayptr(), bytes_in_single_entry);

		    // Flag as modified if needed
		    if ((accum_mode == SNDE_ACCUM_NEW_AXIS || accum_mode == SNDE_ACCUM_CONCATENATE) && (!just_starting)) {
		      // Mark Element as Modified
		      result->storage.at(arraynum)->mark_as_modified(NULL, result->element_offset(arraynum, element_addr), elements_in_single_entry, true);
		    }

		    // Unlock all refs
		    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready()

		    // Create new creator_data
		    std::shared_ptr<accumulate_once_creator_data<T>> new_creator_data = std::make_shared<accumulate_once_creator_data<T>>();
		    if (just_starting) {
		      new_creator_data->accumulated = accumulated_increment;
		      new_creator_data->first_rec = to_accum;
		    }
		    else {
		      new_creator_data->accumulated = creator_data->accumulated + accumulated_increment;
		      new_creator_data->first_rec = creator_data->first_rec;
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
    
  
  std::shared_ptr<math_function> define_accumulate_once_function()
  {
    std::shared_ptr<math_function> newfunc = std::make_shared<cpp_math_function>("snde.accumulate_once",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      std::shared_ptr<executing_math_function> executing;
      
      executing = make_cppfuncexec_floatingtypes<accumulate_once>(rss,inst);
      if (!executing) {
	executing = make_cppfuncexec_vectortypes<accumulate_once>(rss,inst);
      }
      if (!executing) {
	executing = make_cppfuncexec_complextypes<accumulate_once>(rss,inst);
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

  SNDE_API std::shared_ptr<math_function> accumulate_once_function=define_accumulate_once_function();

  static int registered_accumulate_once_function = register_math_function(accumulate_once_function);
  

  
  
};
