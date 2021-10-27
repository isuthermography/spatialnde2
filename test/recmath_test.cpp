#include <thread>
#include <cmath>

#include "recstore.hpp"
#include "recmath_cppfunction.hpp"

using namespace snde;


const double scalefactor=4.5;

class multiply_by_scalar: public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<snde_float32>>,snde_float64>
{
public:
  multiply_by_scalar(std::shared_ptr<recording_set_state> wss,std::shared_ptr<instantiated_math_function> inst) :
    recmath_cppfuncexec(wss,inst)
  {

  }
  
  std::pair<bool,std::shared_ptr<compute_options_function_override_type>> decide_new_revision(std::shared_ptr<ndtyped_recording_ref<snde_float32>> recording, snde_float64 multiplier)
  {
    return std::make_pair(true,nullptr);
  }

  std::pair<std::list<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<ndtyped_recording_ref<snde_float32>> recording, snde_float64 multiplier)
  // This is just a representation of the default
  {
    std::list<std::shared_ptr<compute_resource_option>> option_list = { std::make_shared<compute_resource_option_cpu>(SNDE_CR_CPU,0,0,nullptr,0,1,1) };
    return std::make_pair(option_list,nullptr);
  }


  std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<ndtyped_recording_ref<snde_float32>> recording, snde_float64 multiplier) 
  {
    // define_recs code
    printf("define_recs()\n");
    std::shared_ptr<ndtyped_recording_ref<snde_float32>> result_rec;
    result_rec = ndtyped_recording_ref<snde_float32>::create_recording_math(get_result_channel_path(0),wss);
    // ***!!! Should provide means to set allocation manager !!!***
    
    return std::make_shared<metadata_function_override_type>([ this,result_rec,recording,multiplier ]() {
      // metadata code
      std::unordered_map<std::string,metadatum> metadata;
      printf("metadata()\n");
      metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));
      
      result_rec->rec->metadata=std::make_shared<immutable_metadata>(metadata);
      result_rec->rec->mark_metadata_done();
      
      return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,recording,multiplier ]() {
	  // lock_alloc code
	  
	  result_rec->allocate_storage(recording->layout.dimlen);

	  // locking is only required for certain recordings
	  // with special storage under certain conditions,
	  // however it is always good to explicitly request
	  // the locks, as the locking is a no-op if
	  // locking is not actually required. 
	  rwlock_token_set locktokens = lockmgr->lock_recording_refs({
	      { recording, false }, // first element is recording_ref, 2nd parameter is false for read, true for write 
	      { result_rec, true },
	    });
	  
	  return std::make_shared<exec_function_override_type>([ this,locktokens, result_rec,recording,multiplier ]() {
	    // exec code
	    for (snde_index pos=0;pos < recording->layout.dimlen.at(0);pos++){
	      result_rec->element({pos}) = recording->element({pos}) * multiplier;
	    }
	    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_as_ready() 
	    result_rec->rec->mark_as_ready();
	  }); 
	});
      });
  }
  
  
  
};




int main(int argc, char *argv[])
{
  size_t len=100;
  std::shared_ptr<snde::recdatabase> recdb=std::make_shared<snde::recdatabase>();
  std::shared_ptr<snde::ndarray_recording_ref> test_rec;
  
  recdb->default_storage_manager = std::make_shared<recording_storage_manager_shmem>();
  recdb->compute_resources->compute_resources.push_back(std::make_shared<available_compute_resource_cpu>(recdb,recdb->compute_resources,SNDE_CR_CPU,std::thread::hardware_concurrency()));
  recdb->compute_resources->start();
  
  std::shared_ptr<math_function> multiply_by_scalar_function = std::make_shared<cpp_math_function>([] (std::shared_ptr<recording_set_state> wss,std::shared_ptr<instantiated_math_function> inst) {
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
    nullptr);
  
  
  
  snde::active_transaction transact(recdb); // Transaction RAII holder
  
  recdb->add_math_function(scaled_channel_function,false);
  
  std::shared_ptr<snde::channelconfig> testchan_config=std::make_shared<snde::channelconfig>("/test_channel", "main", (void *)&main,false);
  
  std::shared_ptr<snde::channel> testchan = recdb->reserve_channel(testchan_config);
  test_rec = multi_ndarray_recording::create_typed_recording(recdb,testchan,(void *)&main,SNDE_RTN_FLOAT32);
  std::shared_ptr<snde::globalrevision> globalrev = transact.end_transaction();

  test_rec->rec->metadata=std::make_shared<snde::immutable_metadata>();
  test_rec->rec->mark_metadata_done();
  test_rec->allocate_storage(std::vector<snde_index>{len});

  // locking is only required for certain recordings
  // with special storage under certain conditions,
  // however it is always good to explicitly request
  // the locks, as the locking is a no-op if
  // locking is not actually required.
  // Note that requiring locking for read is extremely rare
  // and won't apply to normal channels. Requiring locking
  // for write is relatively common. 
  {
    rwlock_token_set locktokens = recdb->lockmgr->lock_recording_refs({
	{ test_rec, true }, // first element is recording_ref, 2nd parameter is false for read, true for write 
      });
    for (size_t cnt=0;cnt < len; cnt++) {
      test_rec->assign_double({cnt},100.0*sin(cnt));
      
    }
    // locktokens automatically dropped as it goes out of scope
    // must drop before mark_as_ready()

  }
  test_rec->rec->mark_as_ready();

  printf("About to wait_complete()\n");
  fflush(stdout);
  globalrev->wait_complete();

  printf("wait_complete() done\n");
  fflush(stdout);
  std::shared_ptr<ndarray_recording_ref> scaled_rec = globalrev->get_recording_ref("/scaled channel");
    
  // verify it is OK to read these channels without locking
  assert(!scaled_rec->ndinfo()->requires_locking_read);
  assert(!test_rec->ndinfo()->requires_locking_read);
  for (size_t cnt=0;cnt < len; cnt++) {
    double math_function_value = scaled_rec->element_double({cnt});
    double recalc_value = (float)(test_rec->element_double({cnt})*scalefactor);
    printf(" %f \t \t %f\n",recalc_value,math_function_value);
    assert(math_function_value == recalc_value);
  }
  
  printf("Exiting.\n");
  return 0;
}
