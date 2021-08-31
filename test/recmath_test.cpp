#include <thread>
#include <cmath>

#include "recstore.hpp"
#include "recmath_cppfunction.hpp"

using namespace snde;


const double scalefactor=4.5;

class multiply_by_scalar: public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording<snde_float32>>,snde_float64>
{
public:
  multiply_by_scalar(std::shared_ptr<recording_set_state> wss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec(wss,inst)
  {

  }
  
  std::pair<bool,std::shared_ptr<compute_options_function_override_type>> decide_new_revision(std::shared_ptr<ndtyped_recording<snde_float32>> recording, snde_float64 multiplier)
  {
    return std::make_pair(true,nullptr);
  }

  std::pair<std::list<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<ndtyped_recording<snde_float32>> recording, snde_float64 multiplier)
  // This is just a representation of the default
  {
    std::list<std::shared_ptr<compute_resource_option>> option_list = { std::make_shared<compute_resource_option_cpu>(SNDE_CR_CPU,0,0,nullptr,0,1,1) };
    return std::make_pair(option_list,nullptr);
  }


  std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<ndtyped_recording<snde_float32>> recording, snde_float64 multiplier) 
  {
    // define_recs code
    printf("define_recs()\n");
    std::shared_ptr<ndtyped_recording<snde_float32>> result_rec = ndtyped_recording<snde_float32>::create_recording(*inst->result_channel_paths.at(0),wss);
    // ***!!! Should provide means to set allocation manager !!!***
    
    return std::make_shared<metadata_function_override_type>([ this,result_rec,recording,multiplier ]() {
      // metadata code
      std::unordered_map<std::string,metadatum> metadata;
      printf("metadata()\n");
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
  std::shared_ptr<snde::ndarray_recording> test_rec;

  recdb->default_storage_manager = std::make_shared<recording_storage_manager_shmem>();
  recdb->compute_resources->compute_resources.push_back(std::make_shared<available_compute_resource_cpu>(recdb,recdb->compute_resources,SNDE_CR_CPU,std::thread::hardware_concurrency()));
  recdb->compute_resources->start();

  std::shared_ptr<math_function> multiply_by_scalar_function = std::make_shared<cpp_math_function>(1,
												   [] (std::shared_ptr<recording_set_state> wss,std::shared_ptr<instantiated_math_function> inst) {
												     return std::make_shared<multiply_by_scalar>(wss,inst);
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
  test_rec = ndarray_recording::create_typed_recording(recdb,testchan,(void *)&main,SNDE_RTN_FLOAT32);
  std::shared_ptr<snde::globalrevision> globalrev = transact.end_transaction();

  test_rec->metadata=std::make_shared<snde::immutable_metadata>();
  test_rec->mark_metadata_done();
  test_rec->allocate_storage(std::vector<snde_index>{len});

  for (size_t cnt=0;cnt < len; cnt++) {
    test_rec->assign_double({cnt},100.0*sin(cnt));
    
  }
  test_rec->mark_as_ready();

  printf("About to wait_complete()\n");
  fflush(stdout);
  globalrev->wait_complete();

  printf("wait_complete() done\n");
  fflush(stdout);
  std::shared_ptr<ndarray_recording> scaled_rec = std::dynamic_pointer_cast<ndarray_recording>(globalrev->get_recording("/scaled channel"));
  
  for (size_t cnt=0;cnt < len; cnt++) {
    double math_function_value = scaled_rec->element_double({cnt});
    double recalc_value = (float)(test_rec->element_double({cnt})*scalefactor);
    printf(" %f \t \t %f\n",recalc_value,math_function_value);
    assert(math_function_value == recalc_value);
  }
  
  printf("Exiting.\n");
  return 0;
}
