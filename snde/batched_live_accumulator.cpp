

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


#include "snde/batched_live_accumulator.hpp"

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
  class batched_live_accumulator: public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,uint64_t,uint64_t,bool,bool> {
  public:
    batched_live_accumulator(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,uint64_t,uint64_t,bool,bool>(rss,inst)
    {
      
    }

    // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,uint64_t>::compute_options_function_override_type compute_options_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,uint64_t>::define_recs_function_override_type define_recs_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,uint64_t>::metadata_function_override_type metadata_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,uint64_t>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,uint64_t>::exec_function_override_type exec_function_override_type;

    
    // just using the default for decide_execution

    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<ndtyped_recording_ref<T>> to_accum, uint64_t numaccum, uint64_t batch_size, bool fortran_mode,bool empty_first)
    {
      snde_index nbytes = to_accum->layout.flattened_length() * to_accum->storage->elementsize;

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(0, //metadata_bytes 
							nbytes*2, // data_bytes for transfer
							0.0, // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
	};
      return std::make_pair(option_list,nullptr);
    }

    
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<ndtyped_recording_ref<T>> to_accum, uint64_t numaccum, uint64_t batch_size, bool fortran_mode,bool empty_first) // fortran_mode true means accumulate a new last axis, false means accumulate a new first axis
    {

      // define_recs code
      bool just_starting = false;
      bool got_empty = false; 
      std::shared_ptr<multi_ndarray_recording> previous_recording_ndarray;
      
      std::shared_ptr<recording_base> previous_recording = this->self_dependent_recordings.at(0);

      std::vector<snde_index> accum_dimlen; // dimlen of arrays we are accumulating (may be length 0 for scalars)
      size_t accum_indexnum=0; // accumulation axis: 0 for C-mode, last for Fortran mode

      if (numaccum < batch_size) {
	throw snde_error("batched_live_accumulator numaccum (%llu) must be at least batch_size (%llu)",(unsigned long long)numaccum,(unsigned long long)batch_size);
      }

      snde_index numbatches = (numaccum+batch_size-1)/batch_size; // numbatches always rounds up

      if (numbatches < 2) {
	numbatches = 2; // don't allow less than 2 batches
      }
	
      snde_index previous_num_batches=0;
      snde_index first_batch_size = 0;
      snde_index last_batch_size = 0;
      snde_index previous_num_accumulated = 0;

      if (previous_recording) {
	previous_recording_ndarray = std::dynamic_pointer_cast<multi_ndarray_recording>(previous_recording);
	if (previous_recording_ndarray && previous_recording_ndarray->layouts.at(0).dimlen.size() >= 1) {
	  // check compatibility
	  
	  if (fortran_mode) {
	    // accumulation index is last index */
	    accum_dimlen = std::vector<snde_index>(previous_recording_ndarray->layouts.at(0).dimlen.begin(),previous_recording_ndarray->layouts.at(0).dimlen.end()-1);
	    accum_indexnum = accum_dimlen.size();
	  } else {
	    // C mode: accumulation index is first index */
	    accum_dimlen = std::vector<snde_index>(previous_recording_ndarray->layouts.at(0).dimlen.begin()+1,previous_recording_ndarray->layouts.at(0).dimlen.end());
	    accum_indexnum=0;
	  }

	  if (accum_dimlen == to_accum->layout.dimlen) {
	    // check type
	    if (previous_recording_ndarray->ndinfo(0)->typenum == to_accum->typenum) {

	      // check batch structure
	      // (number of ndarrays in previous recording)
	      previous_num_batches = previous_recording_ndarray->layouts.size();

	      // check batch sizes
	      snde_index batchnum;

	      for (batchnum=0;batchnum < previous_num_batches; batchnum++) {
		snde_index this_batch_length = previous_recording_ndarray->layouts.at(batchnum).dimlen.at(accum_indexnum);

		previous_num_accumulated += this_batch_length;
		
		if (this_batch_length > batch_size) {
		  // oversized batch: restart accumulation
		  
		  just_starting = true; // use as escape flag
		  break;
		}

		if (batchnum==0) {
		  first_batch_size = this_batch_length;
		} else if (batchnum == previous_num_batches-1) {
		  last_batch_size = this_batch_length;
		} else if (this_batch_length < batch_size) {
		  // undersized batch: restart accumulation
		  
		  just_starting = true; // use as escape flag
		  break;
		}

		std::vector<snde_index> this_batch_accum_dimlen;
		
		if (fortran_mode) {
		  // accumulation index is last index */
		  this_batch_accum_dimlen = std::vector<snde_index>(previous_recording_ndarray->layouts.at(batchnum).dimlen.begin(),previous_recording_ndarray->layouts.at(batchnum).dimlen.end()-1);
		} else {
		  // C mode: accumulation index is first index */
		  this_batch_accum_dimlen = std::vector<snde_index>(previous_recording_ndarray->layouts.at(batchnum).dimlen.begin()+1,previous_recording_ndarray->layouts.at(batchnum).dimlen.end());
		}

		if (this_batch_accum_dimlen != accum_dimlen) {
		  // batch shape mismatch: restart accumulation
		  
		  just_starting = true; // use as escape flag
		  break;
		}


		// check type

		if (previous_recording_ndarray->ndinfo(batchnum)->typenum != to_accum->typenum) {
		  // type mismatch: restart accumulation
		  just_starting = true; // use as escape flag
		  break;
		}

		// check layout
		if (fortran_mode) {
		  if (!previous_recording_ndarray->layouts.at(batchnum).is_f_contiguous()) {
		    // layout mismatch: restart accumulation
		    just_starting = true; 
		  }		  
		} else  {
		  // c-mode
		  if (!previous_recording_ndarray->layouts.at(batchnum).is_c_contiguous()) {
		    // layout mismatch: restart accumulation
		    just_starting = true; 
		  }		  
		  
		}
	      }

	      if (previous_num_accumulated > numaccum) {
		// total overflow
		just_starting = false; 
	      }

	      if (!just_starting) {
		// We are OK!
		// and have extracted all of the relevant shape
		// information and everything looks consistent
		// with our new entry
		
	      }
	      
	    } else {
	      // type mismatch
	      just_starting = true; 
	    }
	  }
	  else {
	    // shape mismatch; need to restart
	    just_starting=true;
	  }
	  
	} else if (previous_recording_ndarray) {
	  // previous empty recording
	  got_empty = true;
	  just_starting = true; 
	} else {
	  // no ndarray previous recording
	  just_starting = true; 
	}
      } else {
	// no previous recording
	just_starting = true; 
      }


      if (just_starting) {
	accum_dimlen = to_accum->layout.dimlen;

	if (fortran_mode) {
	  accum_indexnum = accum_dimlen.size();
	} else {
	  accum_indexnum = 0;
	}

	previous_num_batches=0;
	first_batch_size = 0;
	last_batch_size = 0; 

	previous_num_accumulated = 0;
      }

      // note: if there is only one batch so far, first_batch_size
      // will be its size and last_batch_size is zero 
      
      
      
      std::shared_ptr<multi_ndarray_recording> result;
      //snde_warning("avg: define_recs just_starting");


      result = create_recording_math<multi_ndarray_recording>(this->get_result_channel_path(0),this->rss,0); // set num_ndarrays to 0 initially

      snde_index new_num_batches=0;

      bool last_batch_overflow = false;
      bool first_batch_underflow = false; 
      
      if (empty_first && just_starting && !got_empty) {
	// starting result is just an empty recording
	new_num_batches = 0;
      } else {
	new_num_batches = previous_num_batches;

	if (!new_num_batches) {
	  new_num_batches = 1;  
	} else {
	  // check for last batch overflow
	  snde_index actual_last_batch_size = last_batch_size;
	  if (previous_num_batches==1) {
	    actual_last_batch_size = first_batch_size; 
	  }
	  if (actual_last_batch_size + 1 > batch_size) {
	    last_batch_overflow = true;
	    new_num_batches += 1;
	  }

	  // check for first batch underflow
	  if (previous_num_accumulated == numaccum && first_batch_size==1) {
	    // first batch underflows to empty in this step
	    first_batch_underflow = true;
	    new_num_batches -= 1; 
	  }
	}
      }
      
      result->set_num_ndarrays(new_num_batches);
      
      return std::make_shared<metadata_function_override_type>([ this, to_accum, previous_recording_ndarray, numaccum, batch_size, fortran_mode, empty_first, just_starting, got_empty, accum_dimlen, accum_indexnum, previous_num_batches, first_batch_size, last_batch_size, previous_num_accumulated, last_batch_overflow, first_batch_underflow, new_num_batches, result ]() {
	// metadata code

	//snde_warning("avg: metadata just_starting");

	std::shared_ptr<constructible_metadata> metadata = std::make_shared<constructible_metadata>();
	
	result->metadata=metadata;
	result->mark_metadata_done();
	
	return std::make_shared<lock_alloc_function_override_type>([ this, to_accum, previous_recording_ndarray, numaccum, batch_size, fortran_mode, empty_first, just_starting, got_empty, accum_dimlen, accum_indexnum, previous_num_batches, first_batch_size, last_batch_size, previous_num_accumulated, last_batch_overflow, first_batch_underflow, new_num_batches, result ]() {
	  // lock_alloc code
	  snde_index previous_batchnum,batchnum;

	  std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> to_lock;
	  to_lock.reserve(new_num_batches);

	  previous_batchnum = 0;
	  if (first_batch_underflow) {
	    previous_batchnum = 1;
	  }
	  
	  for (batchnum = 0; batchnum < new_num_batches; batchnum++,previous_batchnum++) {
	    result->define_array(batchnum,to_accum->typenum);
	    
	    bool remove_first_element_from_previous=false;
	    if (batchnum==0 && previous_num_accumulated == numaccum && !first_batch_underflow) {
	      // need to remove first element
	      remove_first_element_from_previous = true; 
	    }

	    bool entirely_new_batch = false;
	    if (batchnum==new_num_batches-1 && (last_batch_overflow || just_starting)) {
	      entirely_new_batch = true; 
	    }



	    if (entirely_new_batch) {

	      std::vector<snde_index> alloc_dimlen = accum_dimlen;
	      if (fortran_mode) {
		alloc_dimlen.push_back(batch_size);
	      } else {
		alloc_dimlen.insert(alloc_dimlen.begin(),batch_size);
	      }

	      // allocate the entire next batch
	      std::shared_ptr<recording_storage> newbatch_storage = result->allocate_storage(batchnum,alloc_dimlen,fortran_mode); 

	      // then reach in and only use the first element of it along the accumulation axis
	      std::vector<snde_index> newbatch_dimlen = alloc_dimlen;
	      newbatch_dimlen.at(accum_indexnum)=1;

	      result->assign_storage_portion(newbatch_storage,batchnum,newbatch_dimlen,fortran_mode,newbatch_storage->base_index);
	      
	      // (Note that the rest of the storage is there; just unused. 
	    } else {
	      // Use the pre-existing batch

	      snde_index elements_in_single_entry=1;
	      for (auto &&axislen: accum_dimlen) {
		elements_in_single_entry *= axislen;
	      }
	      
	      snde_index oldbatch_baseindex = previous_recording_ndarray->ndinfo(previous_batchnum)->base_index;
	      std::vector<snde_index> oldbatch_dimlen = previous_recording_ndarray->layouts.at(previous_batchnum).dimlen;
	      if (remove_first_element_from_previous) {

		oldbatch_baseindex += elements_in_single_entry;
		oldbatch_dimlen.at(accum_indexnum) -= 1;
	      }
	      
	      result->assign_storage_portion(previous_recording_ndarray->storage.at(previous_batchnum),batchnum,oldbatch_dimlen,fortran_mode,oldbatch_baseindex);
	    }

	    to_lock.push_back(std::make_pair(result->reference_ndarray(batchnum),batchnum==new_num_batches-1));
	    
	  }

	  rwlock_token_set locktokens = this->lockmgr->lock_recording_refs(to_lock,false); // (false -> non_gpu) 

	  
	  return std::make_shared<exec_function_override_type>([ this, to_accum, previous_recording_ndarray, numaccum, batch_size, fortran_mode, empty_first, just_starting, got_empty, accum_dimlen, accum_indexnum, previous_num_batches, first_batch_size, last_batch_size, previous_num_accumulated, last_batch_overflow, first_batch_underflow, new_num_batches, result, locktokens ]() {
	    // exec code

	    // copy data into new element
	    snde_index batchnum = new_num_batches-1;
	    std::vector<snde_index> element_addr(accum_dimlen.size()+1,0); // initialize index with zeros
	    // but our accumulation index should point at the last element	    
	    element_addr.at(accum_indexnum) = result->layouts.at(batchnum).dimlen.at(accum_indexnum)-1;


	    snde_index elements_in_single_entry=1;
	    for (auto &&axislen: accum_dimlen) {
	      elements_in_single_entry *= axislen;
	    }
	    snde_index bytes_in_single_entry = elements_in_single_entry*to_accum->ndinfo()->elementsize;
	    
	    memcpy(result->element_dataptr(batchnum,element_addr),to_accum->void_shifted_arrayptr(),bytes_in_single_entry);
	    
	    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_as_ready()
	    
	    result->mark_as_ready();
	    
	  }); 
	});
	      
      });
    }
  };
    
  
  std::shared_ptr<math_function> define_batched_live_accumulator_function()
  {
    std::shared_ptr<math_function> newfunc = std::make_shared<cpp_math_function>([] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      std::shared_ptr<executing_math_function> executing;
      
      executing = make_cppfuncexec_floatingtypes<batched_live_accumulator>(rss,inst);
      if (!executing) {
	executing = make_cppfuncexec_vectortypes<batched_live_accumulator>(rss,inst);
      }
      if (!executing) {
	executing = make_cppfuncexec_complextypes<batched_live_accumulator>(rss,inst);
      }
      if (!executing) {
	throw snde_error("In attempting to call math function %s, first parameter has unsupported data type.",inst->definition->definition_command.c_str());
      }
      return executing;
    });
    newfunc->self_dependent=true;
    return newfunc;
    
  }

  SNDE_API std::shared_ptr<math_function> batched_live_accumulator_function=define_batched_live_accumulator_function();

  static int registered_batched_live_accumulator_function = register_math_function("spatialnde2.batched_live_accumulator",batched_live_accumulator_function);
  
  
  
  
};