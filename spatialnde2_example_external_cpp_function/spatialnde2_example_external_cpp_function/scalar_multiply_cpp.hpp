#ifndef SNDE_EXAMPLE_EXT_CPP_FUNC_SCALAR_MULTIPLY_HPP

#include "snde/recmath.hpp"
#include "snde/snde_error.hpp"
#include "snde/recmath_cppfunction.hpp"



namespace snde2_fn_ex {

  template <typename T>
  class scalar_multiply: public snde::recmath_cppfuncexec<std::shared_ptr<snde::ndtyped_recording_ref<T>>,snde_float64>
  {
  public:
    scalar_multiply(std::shared_ptr<snde::recording_set_state> rss,std::shared_ptr<snde::instantiated_math_function> inst) :
      snde::recmath_cppfuncexec<std::shared_ptr<snde::ndtyped_recording_ref<T>>,snde_float64>(rss,inst)
    {
      
    }
    
    // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
    typedef typename snde::recmath_cppfuncexec<std::shared_ptr<snde::ndtyped_recording_ref<T>>,snde_float64>::metadata_function_override_type metadata_function_override_type;
    typedef typename snde::recmath_cppfuncexec<std::shared_ptr<snde::ndtyped_recording_ref<T>>,snde_float64>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename snde::recmath_cppfuncexec<std::shared_ptr<snde::ndtyped_recording_ref<T>>,snde_float64>::exec_function_override_type exec_function_override_type;
  
    // just using the default for decide_new_revision and compute_options
    
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<snde::ndtyped_recording_ref<T>> recording, snde_float64 multiplier) 
    {
      // define_recs code
      snde::snde_debug(SNDE_DC_APP,"define_recs()");

      // recordings can contain scale and offset metadata. If these
      // are provided, you must either properly scale and offset
      // the recording when you use it, or assert that the scale
      // is 1.0 and the offset is 0.0 with the
      // assert_no_scale_or_offset() method, which will throw an
      // exception otherwise, e.g.
      //  recording->assert_no_scale_or_offset(this->inst->definition->definition_command);
      // (In this case, we do proper scaling using the
      // get_ampl_scale_offset() method below)
      // Use of "this" in the next line for the same reason as the typedefs, above
      std::shared_ptr<snde::ndtyped_recording_ref<T>> result_rec = snde::create_typed_ndarray_ref_math<T>(this->get_result_channel_path(0),this->rss);
      // ***!!! Should provide means to set allocation manager !!!***
      
      return std::make_shared<metadata_function_override_type>([ this,result_rec,recording,multiplier ]() {
	// metadata code
	std::unordered_map<std::string,snde::metadatum> metadata;
	snde::snde_debug(SNDE_DC_APP,"metadata()");
	metadata.emplace("Test_metadata_entry",snde::metadatum("Test_metadata_entry",3.14));
	
	result_rec->rec->metadata=std::make_shared<snde::immutable_metadata>(metadata);
	result_rec->rec->mark_metadata_done();
	
	return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,recording,multiplier ]() {
	  // lock_alloc code
	  
	  result_rec->allocate_storage(recording->layout.dimlen);

	  // locking is only required for certain recordings
	  // with special storage under certain conditions,
	  // however it is always good to explicitly request
	  // the locks, as the locking is a no-op if
	  // locking is not actually required. 
	  snde::rwlock_token_set locktokens = this->lockmgr->lock_recording_refs({
	      { recording, false }, // first element is recording_ref, 2nd parameter is false for read, true for write 
	      { result_rec, true },
	    });
	  

	  
	  return std::make_shared<exec_function_override_type>([ this,locktokens,result_rec,recording,multiplier ]() {
	    // exec code

	    snde_float64 scale,offset;
	    std::tie(scale,offset)=recording->get_ampl_scale_offset();
	    for (snde_index pos=0;pos < recording->layout.dimlen.at(0);pos++){
	      result_rec->element({pos}) = (recording->element({pos})*scale + offset) * multiplier;
	    }
	    
	    snde::unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 

	    result_rec->rec->mark_data_ready();
	  }); 
	});
      });
    }
    
    
    
  };
  
  
  
  std::shared_ptr<snde::math_function> define_scalar_multiply()
  {
    return std::make_shared<snde::cpp_math_function>("spatialnde2_example_external_cpp_function.scalar_multiply_function",1,[] (std::shared_ptr<snde::recording_set_state> rss,std::shared_ptr<snde::instantiated_math_function> inst) {
      return snde::make_cppfuncexec_floatingtypes<scalar_multiply>(rss,inst);
    });
    
  }


  static std::shared_ptr<snde::math_function> scalar_multiply_function=define_scalar_multiply();

  // Register the math function into the C++ database
  // This should use the same python-accessible name for
  // maximum interoperability

  static int registered_scalar_multiply_function = register_math_function(scalar_multiply_function);

};

#endif // SNDE_EXAMPLE_EXT_CPP_FUNC_SCALAR_MULTIPLY_HPP
