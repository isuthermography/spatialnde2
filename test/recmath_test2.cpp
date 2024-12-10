#include <thread>
#include <cmath>

#include "recstore.hpp"
#include "recmath_cppfunction.hpp"
#include "allocator.hpp"
#include "snde/recstore_setup.hpp"

using namespace snde;


const double scalefactor=4.5;

template <typename T>
class multiply_by_scalar: public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,snde_float64>
{
public:
  multiply_by_scalar(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,snde_float64>(rss,inst)
  {

  }

  // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
  // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
  typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,snde_float64>::define_recs_function_override_type define_recs_function_override_type;
  typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,snde_float64>::metadata_function_override_type metadata_function_override_type;
  typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,snde_float64>::lock_alloc_function_override_type lock_alloc_function_override_type;
  typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,snde_float64>::exec_function_override_type exec_function_override_type;
  
  // just using the default for decide_new_revision

  // compute_options
   std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<ndtyped_recording_ref<snde_float32>> recording, snde_float64 multiplier)
  {
    // This is a slight improvement over the default, which doesn't know the computational complexity of the calculation. 
    snde_index numentries = recording->layout.flattened_length();
    std::vector<std::shared_ptr<compute_resource_option>> option_list =
      {
	std::make_shared<compute_resource_option_cpu>(std::set<std::string>(),
						      0, //metadata_bytes 
						      numentries*sizeof(snde_float32)*2, // data_bytes for transfer
						      (snde_float64)numentries, // flops
						      1, // max effective cpu cores
						      1), // useful_cpu_cores (min # of cores to supply
      };
    return std::make_pair(option_list,nullptr);
  }

  std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<ndtyped_recording_ref<T>> recording, snde_float64 multiplier) 
  {
    // define_recs code
    snde_debug(SNDE_DC_APP,"define_recs()");
    // Use of "this" in the next line for the same reason as the typedefs, above
    std::shared_ptr<ndtyped_recording_ref<T>> result_rec = create_typed_ndarray_ref_math<T>(this->get_result_channel_path(0),this->rss);
    // ***!!! Should provide means to set allocation manager !!!***
    
    return std::make_shared<metadata_function_override_type>([ this,result_rec,recording,multiplier ]() {
      // metadata code
      std::unordered_map<std::string,metadatum> metadata;
      snde_debug(SNDE_DC_APP,"metadata()");
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
	  rwlock_token_set locktokens = this->lockmgr->lock_recording_refs({
	      { recording, false }, // first element is recording_ref, 2nd parameter is false for read, true for write 
	      { result_rec, true },
	    });

	  
	  return std::make_shared<exec_function_override_type>([ this,locktokens,result_rec,recording,multiplier ]() {
	    // exec code
	    for (snde_index pos=0;pos < recording->layout.dimlen.at(0);pos++){
	      result_rec->element({pos}) = (T)(recording->element({pos}) * multiplier);
	    }
	    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	    result_rec->rec->mark_data_ready();
	  }); 
	});
      });
  }
  
  
  
};




int main(int argc, char *argv[])
{
  size_t len=100;

  std::shared_ptr<snde::recdatabase> recdb=std::make_shared<snde::recdatabase>();
  setup_cpu(recdb,{},std::thread::hardware_concurrency());
  setup_storage_manager(recdb);
  setup_math_functions(recdb,{});
  recdb->startup();

  
  std::shared_ptr<snde::ndtyped_recording_ref<snde_float32>> test_rec_32;
  std::shared_ptr<snde::ndtyped_recording_ref<snde_float64>> test_rec_64;


  std::shared_ptr<math_function> multiply_by_scalar_function = std::make_shared<cpp_math_function>("multiply_by_scalar",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
    std::shared_ptr<executing_math_function> executing;
    executing = make_cppfuncexec_floatingtypes<multiply_by_scalar>(rss,inst);
    if (!executing) {
      throw snde_error("In attempting to call math function %s, first parameter has unsupported data type.",inst->definition->definition_command.c_str());
    }
    return executing;
  });
  
  std::shared_ptr<instantiated_math_function> scaled_channel_function = multiply_by_scalar_function->instantiate({
      std::make_shared<math_parameter_recording>("/test_channel"),
      std::make_shared<math_parameter_double_const>(scalefactor),
    },
    { std::make_shared<std::string>("/scaled channel") },
    "/",
    false,
    false,
    false,
    std::make_shared<math_definition>("c++ definition"),
    {},
    nullptr);
  
  
  
  std::shared_ptr<snde::active_transaction> transact=recdb->start_transaction(); // Transaction RAII holder

  recdb->add_math_function(transact,scaled_channel_function,false);
  
  std::shared_ptr<snde::channelconfig> testchan_config=std::make_shared<snde::channelconfig>("/test_channel", "main",false);
  
  std::shared_ptr<snde::reserved_channel> testchan = recdb->reserve_channel(transact,testchan_config);

  // demonstrate alternative ways to create the recording
  test_rec_32 = std::dynamic_pointer_cast<ndtyped_recording_ref<float>>(create_ndarray_ref(transact,testchan,SNDE_RTN_FLOAT32));
  std::shared_ptr<snde::globalrevision> globalrev = transact->end_transaction()->globalrev_available();


  std::shared_ptr<snde::active_transaction> transact2=recdb->start_transaction(); // Transaction RAII holder


  test_rec_64 = create_typed_ndarray_ref<snde_float64>(transact2,testchan);
  std::shared_ptr<snde::globalrevision> globalrev2 = transact2->end_transaction()->globalrev_available();

  
  test_rec_32->rec->metadata=std::make_shared<snde::immutable_metadata>();
  test_rec_32->rec->mark_metadata_done();
  test_rec_32->allocate_storage(std::vector<snde_index>{len});

  
  test_rec_64->rec->metadata=std::make_shared<snde::immutable_metadata>();
  test_rec_64->rec->mark_metadata_done();
  test_rec_64->allocate_storage(std::vector<snde_index>{len});


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
	{ test_rec_32, true }, // first element is recording_ref, 2nd parameter is false for read, true for write 
	{ test_rec_64, true },
      });
    
    for (size_t cnt=0;cnt < len; cnt++) {
      
      // demonstrating alternative array interfaces
      test_rec_32->assign_double({cnt},100.0*sin(cnt));
      
      test_rec_64->element({cnt}) = -46.0*sin(cnt);
      
    }
    // locktokens automatically dropped as it goes out of scope
    // must drop before mark_as_ready()
  }
  test_rec_32->rec->mark_data_ready();
  test_rec_64->rec->mark_data_ready();

  snde_debug(SNDE_DC_APP,"About to wait_complete()");
  globalrev->wait_complete();
  globalrev2->wait_complete();

  snde_debug(SNDE_DC_APP,"wait_complete() done");
  std::shared_ptr<ndarray_recording_ref> scaled_rec_32 = globalrev->get_ndarray_ref("/scaled channel");

  
  // verify it is OK to read these channels without locking
  assert(!scaled_rec_32->ndinfo()->requires_locking_read);
  assert(!test_rec_32->ndinfo()->requires_locking_read);
  for (size_t cnt=0;cnt < len; cnt++) {
    double math_function_value = scaled_rec_32->element_double({cnt});
    double recalc_value = (float)(test_rec_32->element_double({cnt})*scalefactor);
    printf(" %f \t \t %f\n",recalc_value,math_function_value);
    assert(math_function_value == recalc_value);
  }

  
  std::shared_ptr<ndarray_recording_ref> scaled_rec_64 = globalrev2->get_ndarray_ref("/scaled channel");

  // verify it is OK to read these channels without locking
  assert(!scaled_rec_64->ndinfo()->requires_locking_read);
  assert(!test_rec_64->ndinfo()->requires_locking_read);
  
  for (size_t cnt=0;cnt < len; cnt++) {
    double math_function_value = scaled_rec_64->element_double({cnt});
    double recalc_value = (double)(test_rec_64->element_double({cnt})*scalefactor);
    printf(" %f \t \t %f\n",recalc_value,math_function_value);
    assert(math_function_value == recalc_value);
  }
  
  snde_debug(SNDE_DC_APP,"Exiting.");
  return 0;
}
