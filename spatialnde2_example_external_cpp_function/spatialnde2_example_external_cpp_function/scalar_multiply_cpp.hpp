#include "snde/snde_error.hpp"
#include "snde/recmath_cppfunction.hpp"
#include "snde/recmath.hpp"

namespace snde2_fn_ex {


  template <typename T>
  class scalar_multiply: public snde::recmath_cppfuncexec<std::shared_ptr<snde::ndtyped_recording_ref<T>>,snde_float64>
  {
  public:
    scalar_multiply(std::shared_ptr<snde::recording_set_state> wss,std::shared_ptr<snde::instantiated_math_function> inst) :
      snde::recmath_cppfuncexec<std::shared_ptr<snde::ndtyped_recording_ref<T>>,snde_float64>(wss,inst)
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
      // Use of "this" in the next line for the same reason as the typedefs, above
      std::shared_ptr<snde::ndtyped_recording_ref<T>> result_rec = snde::ndtyped_recording_ref<T>::create_recording_math(this->get_result_channel_path(0),this->wss);
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
	  
	  return std::make_shared<exec_function_override_type>([ this,result_rec,recording,multiplier ]() {
	    // exec code
	    for (snde_index pos=0;pos < recording->layout.dimlen.at(0);pos++){
	      result_rec->element({pos}) = recording->element({pos}) * multiplier;
	    }
	    result_rec->rec->mark_as_ready();
	  }); 
	});
      });
    }
    
    
    
  };
  
  
  
  std::shared_ptr<snde::math_function> define_scalar_multiply()
  {
    return std::make_shared<snde::cpp_math_function>([] (std::shared_ptr<snde::recording_set_state> wss,std::shared_ptr<snde::instantiated_math_function> inst) {
      return snde::make_cppfuncexec_floatingtypes<scalar_multiply>(wss,inst);
    },
      true,
      false,
      false);
    
  }

}
