#include <thread>
#include <cmath>

#include "recstore.hpp"
#include "recmath_cppfunction.hpp"

using namespace snde;


const double scalefactor=4.5;

template <typename T>
class multiply_by_scalar: public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording<T>>,snde_float64>
{
public:
  multiply_by_scalar(std::shared_ptr<recording_set_state> wss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<ndtyped_recording<T>>,snde_float64>(wss,inst)
  {

  }

  // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
  // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
  typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording<T>>,snde_float64>::metadata_function_override_type metadata_function_override_type;
  typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording<T>>,snde_float64>::lock_alloc_function_override_type lock_alloc_function_override_type;
  typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording<T>>,snde_float64>::exec_function_override_type exec_function_override_type;
  
  // just using the default for decide_new_revision and compute_options
 
  std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<ndtyped_recording<T>> recording, snde_float64 multiplier) 
  {
    // define_recs code
    snde_debug(SNDE_DC_APP,"define_recs()");
    // Use of "this" in the next line for the same reason as the typedefs, above
    std::shared_ptr<ndtyped_recording<T>> result_rec = ndtyped_recording<T>::create_recording(*this->inst->result_channel_paths.at(0),this->wss);
    // ***!!! Should provide means to set allocation manager !!!***
    
    return std::make_shared<metadata_function_override_type>([ this,result_rec,recording,multiplier ]() {
      // metadata code
      std::unordered_map<std::string,metadatum> metadata;
      snde_debug(SNDE_DC_APP,"metadata()");
      metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));
      
      result_rec->metadata=std::make_shared<immutable_metadata>(metadata);
      result_rec->mark_metadata_done();
      
      return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,recording,multiplier ]() {
	  // lock_alloc code
	  
	  result_rec->allocate_storage(recording->layout.dimlen);
	  
	  return std::make_shared<exec_function_override_type>([ this,result_rec,recording,multiplier ]() {
	    // exec code
	    for (snde_index pos=0;pos < recording->layout.dimlen.at(0);pos++){
	      result_rec->element({pos}) = recording->element({pos}) * multiplier;
	    }
	    result_rec->mark_as_ready();
	  }); 
	});
      });
  }
  
  
  
};




int main(int argc, char *argv[])
{
  size_t len=100;
  std::shared_ptr<snde::recdatabase> recdb=std::make_shared<snde::recdatabase>();
  std::shared_ptr<snde::ndtyped_recording<snde_float32>> test_rec_32;
  std::shared_ptr<snde::ndtyped_recording<snde_float64>> test_rec_64;

  recdb->default_storage_manager = std::make_shared<recording_storage_manager_shmem>();
  recdb->compute_resources->compute_resources.push_back(std::make_shared<available_compute_resource_cpu>(recdb,recdb->compute_resources,SNDE_CR_CPU,std::thread::hardware_concurrency()));
  recdb->compute_resources->start();

  std::shared_ptr<math_function> multiply_by_scalar_function = std::make_shared<cpp_math_function>([] (std::shared_ptr<recording_set_state> wss,std::shared_ptr<instantiated_math_function> inst) {
												     return make_cppfuncexec_floatingtypes<multiply_by_scalar>(wss,inst);
												   },
												   true,
												   false,
												   false);
  
  std::shared_ptr<instantiated_math_function> scaled_channel_function = multiply_by_scalar_function->instantiate({
      std::make_shared<math_parameter_recording>("/test_channel"),
      std::make_shared<math_parameter_double_const>(scalefactor),
    },
    { std::make_shared<std::string>("/scaled channel") },
    "",
    false,
    false,
    false,
    std::make_shared<math_definition>("c++ definition"),
    "");
  
  
  
  snde::active_transaction transact(recdb); // Transaction RAII holder

  recdb->add_math_function(scaled_channel_function,false);
  
  std::shared_ptr<snde::channelconfig> testchan_config=std::make_shared<snde::channelconfig>("/test_channel", "main", (void *)&main,false);
  
  std::shared_ptr<snde::channel> testchan = recdb->reserve_channel(testchan_config);

  // demonstrate alternative ways to create the recording
  test_rec_32 = std::dynamic_pointer_cast<ndtyped_recording<snde_float32>>(ndarray_recording::create_typed_recording(recdb,testchan,(void *)&main,SNDE_RTN_FLOAT32));
  std::shared_ptr<snde::globalrevision> globalrev = transact.end_transaction();


  snde::active_transaction transact2(recdb); // Transaction RAII holder

  test_rec_64 = ndtyped_recording<snde_float64>::create_recording(recdb,testchan,(void *)&main);
  std::shared_ptr<snde::globalrevision> globalrev2 = transact2.end_transaction();

  
  test_rec_32->metadata=std::make_shared<snde::immutable_metadata>();
  test_rec_32->mark_metadata_done();
  test_rec_32->allocate_storage(std::vector<snde_index>{len});

  
  test_rec_64->metadata=std::make_shared<snde::immutable_metadata>();
  test_rec_64->mark_metadata_done();
  test_rec_64->allocate_storage(std::vector<snde_index>{len});
  
  for (size_t cnt=0;cnt < len; cnt++) {

    // demonstrating alternative array interfaces
    test_rec_32->assign_double({cnt},100.0*sin(cnt));
    
    test_rec_64->element({cnt}) = -46.0*sin(cnt);
    
  }
  test_rec_32->mark_as_ready();
  test_rec_64->mark_as_ready();

  snde_debug(SNDE_DC_APP,"About to wait_complete()");
  globalrev->wait_complete();
  globalrev2->wait_complete();

  snde_debug(SNDE_DC_APP,"wait_complete() done");
  std::shared_ptr<ndarray_recording> scaled_rec_32 = std::dynamic_pointer_cast<ndarray_recording>(globalrev->get_recording("/scaled channel"));
  
  for (size_t cnt=0;cnt < len; cnt++) {
    double math_function_value = scaled_rec_32->element_double({cnt});
    double recalc_value = (float)(test_rec_32->element_double({cnt})*scalefactor);
    printf(" %f \t \t %f\n",recalc_value,math_function_value);
    assert(math_function_value == recalc_value);
  }

  
  std::shared_ptr<ndarray_recording> scaled_rec_64 = std::dynamic_pointer_cast<ndarray_recording>(globalrev2->get_recording("/scaled channel"));
  
  for (size_t cnt=0;cnt < len; cnt++) {
    double math_function_value = scaled_rec_64->element_double({cnt});
    double recalc_value = (double)(test_rec_64->element_double({cnt})*scalefactor);
    printf(" %f \t \t %f\n",recalc_value,math_function_value);
    assert(math_function_value == recalc_value);
  }
  
  snde_debug(SNDE_DC_APP,"Exiting.");
  return 0;
}
