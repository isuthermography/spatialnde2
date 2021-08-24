#include <thread>
#include <cmath>

#include "wfmstore.hpp"
#include "wfmmath_cppfunction.hpp"

using namespace snde;


const double scalefactor=4.5;

template <typename T>
class multiply_by_scalar: public wfmmath_cppfuncexec<std::shared_ptr<ndtyped_waveform<T>>,snde_float64>
{
public:
  multiply_by_scalar(std::shared_ptr<waveform_set_state> wss,std::shared_ptr<instantiated_math_function> inst) :
      wfmmath_cppfuncexec<std::shared_ptr<ndtyped_waveform<T>>,snde_float64>(wss,inst)
  {

  }

  // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
  // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
  typedef typename wfmmath_cppfuncexec<std::shared_ptr<ndtyped_waveform<T>>,snde_float64>::metadata_function_type metadata_function_type;
  typedef typename wfmmath_cppfuncexec<std::shared_ptr<ndtyped_waveform<T>>,snde_float64>::lock_alloc_function_type lock_alloc_function_type;
  typedef typename wfmmath_cppfuncexec<std::shared_ptr<ndtyped_waveform<T>>,snde_float64>::exec_function_type exec_function_type;
  
  // just using the default for decide_new_revision and compute_options
 
  std::shared_ptr<metadata_function_type> define_wfms(std::shared_ptr<ndtyped_waveform<T>> waveform, snde_float64 multiplier) 
  {
    // define_wfms code
    snde_debug(SNDE_DC_APP,"define_wfms()");
    // Use of "this" in the next line for the same reason as the typedefs, above
    std::shared_ptr<ndtyped_waveform<T>> result_wfm = ndtyped_waveform<T>::create_waveform(*this->inst->result_channel_paths.at(0),this->wss);
    // ***!!! Should provide means to set allocation manager !!!***
    
    return std::make_shared<metadata_function_type>([ this,result_wfm ](std::shared_ptr<ndtyped_waveform<T>> waveform, snde_float64 multiplier) {
      // metadata code
      std::unordered_map<std::string,metadatum> metadata;
      snde_debug(SNDE_DC_APP,"metadata()");
      metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));
      
      result_wfm->metadata=std::make_shared<immutable_metadata>(metadata);
      result_wfm->mark_metadata_done();
      
      return std::make_shared<lock_alloc_function_type>([ this,result_wfm ](std::shared_ptr<ndtyped_waveform<T>> waveform, snde_float64 multiplier) {
	  // lock_alloc code
	  
	  result_wfm->allocate_storage(waveform->layout.dimlen);
	  
	  return std::make_shared<exec_function_type>([ this,result_wfm ](std::shared_ptr<ndtyped_waveform<T>> waveform, snde_float64 multiplier) {
	    // exec code
	    for (snde_index pos=0;pos < waveform->layout.dimlen.at(0);pos++){
	      result_wfm->element({pos}) = waveform->element({pos}) * multiplier;
	    }
	    result_wfm->mark_as_ready();
	  }); 
	});
      });
  }
  
  
  
};




int main(int argc, char *argv[])
{
  size_t len=100;
  std::shared_ptr<snde::wfmdatabase> wfmdb=std::make_shared<snde::wfmdatabase>();
  std::shared_ptr<snde::ndtyped_waveform<snde_float32>> test_wfm_32;
  std::shared_ptr<snde::ndtyped_waveform<snde_float64>> test_wfm_64;

  wfmdb->default_storage_manager = std::make_shared<waveform_storage_manager_shmem>();
  wfmdb->compute_resources->compute_resources.push_back(std::make_shared<available_compute_resource_cpu>(wfmdb,wfmdb->compute_resources,SNDE_CR_CPU,std::thread::hardware_concurrency()));
  wfmdb->compute_resources->start();

  std::shared_ptr<math_function> multiply_by_scalar_function = std::make_shared<cpp_math_function>(1,
												   [] (std::shared_ptr<waveform_set_state> wss,std::shared_ptr<instantiated_math_function> inst) {
												     return make_cppfuncexec_floatingtypes<multiply_by_scalar>(wss,inst);
												   },
												   true,
												   false,
												   false);
  
  std::shared_ptr<instantiated_math_function> scaled_channel_function = multiply_by_scalar_function->instantiate({
      std::make_shared<math_parameter_waveform>("/test_channel"),
      std::make_shared<math_parameter_double_const>(scalefactor),
    },
    { std::make_shared<std::string>("/scaled channel") },
    "",
    false,
    false,
    false,
    std::make_shared<math_definition>("c++ definition"),
    "");
  
  
  
  snde::active_transaction transact(wfmdb); // Transaction RAII holder

  wfmdb->add_math_function(scaled_channel_function,false);
  
  std::shared_ptr<snde::channelconfig> testchan_config=std::make_shared<snde::channelconfig>("/test_channel", "main", (void *)&main,false);
  
  std::shared_ptr<snde::channel> testchan = wfmdb->reserve_channel(testchan_config);

  // demonstrate alternative ways to create the waveform
  test_wfm_32 = std::dynamic_pointer_cast<ndtyped_waveform<snde_float32>>(ndarray_waveform::create_typed_waveform(wfmdb,testchan,(void *)&main,SNDE_WTN_FLOAT32));
  std::shared_ptr<snde::globalrevision> globalrev = transact.end_transaction();


  snde::active_transaction transact2(wfmdb); // Transaction RAII holder

  test_wfm_64 = ndtyped_waveform<snde_float64>::create_waveform(wfmdb,testchan,(void *)&main);
  std::shared_ptr<snde::globalrevision> globalrev2 = transact2.end_transaction();

  
  test_wfm_32->metadata=std::make_shared<snde::immutable_metadata>();
  test_wfm_32->mark_metadata_done();
  test_wfm_32->allocate_storage(std::vector<snde_index>{len});

  
  test_wfm_64->metadata=std::make_shared<snde::immutable_metadata>();
  test_wfm_64->mark_metadata_done();
  test_wfm_64->allocate_storage(std::vector<snde_index>{len});
  
  for (size_t cnt=0;cnt < len; cnt++) {

    // demonstrating alternative array interfaces
    test_wfm_32->assign_double({cnt},100.0*sin(cnt));
    
    test_wfm_64->element({cnt}) = -46.0*sin(cnt);
    
  }
  test_wfm_32->mark_as_ready();
  test_wfm_64->mark_as_ready();

  snde_debug(SNDE_DC_APP,"About to wait_complete()");
  globalrev->wait_complete();
  globalrev2->wait_complete();

  snde_debug(SNDE_DC_APP,"wait_complete() done");
  std::shared_ptr<ndarray_waveform> scaled_wfm_32 = std::dynamic_pointer_cast<ndarray_waveform>(globalrev->get_waveform("/scaled channel"));
  
  for (size_t cnt=0;cnt < len; cnt++) {
    double math_function_value = scaled_wfm_32->element_double({cnt});
    double recalc_value = (float)(test_wfm_32->element_double({cnt})*scalefactor);
    printf(" %f \t \t %f\n",recalc_value,math_function_value);
    assert(math_function_value == recalc_value);
  }

  
  std::shared_ptr<ndarray_waveform> scaled_wfm_64 = std::dynamic_pointer_cast<ndarray_waveform>(globalrev2->get_waveform("/scaled channel"));
  
  for (size_t cnt=0;cnt < len; cnt++) {
    double math_function_value = scaled_wfm_64->element_double({cnt});
    double recalc_value = (double)(test_wfm_64->element_double({cnt})*scalefactor);
    printf(" %f \t \t %f\n",recalc_value,math_function_value);
    assert(math_function_value == recalc_value);
  }
  
  snde_debug(SNDE_DC_APP,"Exiting.");
  return 0;
}
