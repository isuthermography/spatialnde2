#include <thread>
#include <cmath>

#include "wfmstore.hpp"
#include "wfmmath_cppfunction.hpp"

using namespace snde;


const double scalefactor=4.5;

class multiply_by_scalar: public wfmmath_cppfuncexec<std::shared_ptr<ndtyped_waveform<snde_float32>>,snde_float64>
{
public:
  multiply_by_scalar(std::shared_ptr<waveform_set_state> wss,std::shared_ptr<instantiated_math_function> inst) :
      wfmmath_cppfuncexec(wss,inst)
  {

  }
  
  std::pair<bool,std::shared_ptr<compute_options_function_type>> decide_new_revision(std::shared_ptr<ndtyped_waveform<snde_float32>> waveform, snde_float64 multiplier)
  {
    return std::make_pair(true,nullptr);
  }

  std::pair<std::list<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_wfms_function_type>> compute_options(std::shared_ptr<ndtyped_waveform<snde_float32>> waveform, snde_float64 multiplier)
  // This is just a representation of the default
  {
    std::list<std::shared_ptr<compute_resource_option>> option_list = { std::make_shared<compute_resource_option_cpu>(SNDE_CR_CPU,0,0,nullptr,0,1,1) };
    return std::make_pair(option_list,nullptr);
  }


  std::shared_ptr<metadata_function_type> define_wfms(std::shared_ptr<ndtyped_waveform<snde_float32>> waveform, snde_float64 multiplier) 
  {
    // define_wfms code
    printf("define_wfms()\n");
    std::shared_ptr<ndtyped_waveform<snde_float32>> result_wfm = ndtyped_waveform<snde_float32>::create_waveform(*inst->result_channel_paths.at(0),wss);
    // ***!!! Should provide means to set allocation manager !!!***
    
    return std::make_shared<metadata_function_type>([ this,result_wfm ](std::shared_ptr<ndtyped_waveform<snde_float32>> waveform, snde_float64 multiplier) {
      // metadata code
      std::unordered_map<std::string,metadatum> metadata;
      printf("metadata()\n");
      metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));
      
      result_wfm->metadata=std::make_shared<immutable_metadata>(metadata);
      result_wfm->mark_metadata_done();
      
      return std::make_shared<lock_alloc_function_type>([ this,result_wfm ](std::shared_ptr<ndtyped_waveform<snde_float32>> waveform, snde_float64 multiplier) {
	  // lock_alloc code
	  
	  result_wfm->allocate_storage(waveform->layout.dimlen);
	  
	  return std::make_shared<exec_function_type>([ this,result_wfm ](std::shared_ptr<ndtyped_waveform<snde_float32>> waveform, snde_float64 multiplier) {
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
  std::shared_ptr<snde::ndarray_waveform> test_wfm;

  wfmdb->default_storage_manager = std::make_shared<waveform_storage_manager_shmem>();
  wfmdb->compute_resources->compute_resources.push_back(std::make_shared<available_compute_resource_cpu>(wfmdb,wfmdb->compute_resources,SNDE_CR_CPU,std::thread::hardware_concurrency()));
  wfmdb->compute_resources->start();

  std::shared_ptr<math_function> multiply_by_scalar_function = std::make_shared<cpp_math_function>(1,
												   [] (std::shared_ptr<waveform_set_state> wss,std::shared_ptr<instantiated_math_function> inst) {
												     return std::make_shared<multiply_by_scalar>(wss,inst);
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
  test_wfm = ndarray_waveform::create_typed_waveform(wfmdb,testchan,(void *)&main,SNDE_WTN_FLOAT32);
  std::shared_ptr<snde::globalrevision> globalrev = transact.end_transaction();

  test_wfm->metadata=std::make_shared<snde::immutable_metadata>();
  test_wfm->mark_metadata_done();
  test_wfm->allocate_storage(std::vector<snde_index>{len});

  for (size_t cnt=0;cnt < len; cnt++) {
    test_wfm->assign_double({cnt},100.0*sin(cnt));
    
  }
  test_wfm->mark_as_ready();

  printf("About to wait_complete()\n");
  fflush(stdout);
  globalrev->wait_complete();

  printf("wait_complete() done\n");
  fflush(stdout);
  std::shared_ptr<ndarray_waveform> scaled_wfm = std::dynamic_pointer_cast<ndarray_waveform>(globalrev->get_waveform("/scaled channel"));
  
  for (size_t cnt=0;cnt < len; cnt++) {
    double math_function_value = scaled_wfm->element_double({cnt});
    double recalc_value = (float)(test_wfm->element_double({cnt})*scalefactor);
    printf(" %f \t \t %f\n",recalc_value,math_function_value);
    assert(math_function_value == recalc_value);
  }
  
  printf("Exiting.\n");
  return 0;
}
