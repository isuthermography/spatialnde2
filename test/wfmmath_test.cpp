#include <cmath>
#include "wfmstore.hpp"
#include "wfmmath_cppfunction.hpp"

using namespace snde;


class multiply_by_scalar: public wfmmath_cppfuncexec<std::shared_ptr<ndtyped_waveform<snde_float32>>,snde_float64>
{
public:
  multiply_by_scalar(std::shared_ptr<waveform_set_state> wss,std::shared_ptr<instantiated_math_function> inst,bool is_mutable,bool mdonly) :
      wfmmath_cppfuncexec(wss,inst,is_mutable,mdonly)
  {

  }
  
  std::pair<bool,compute_options_function_type*> decide_new_revision(ndtyped_waveform<snde_float32> waveform, snde_float64 multiplier)
  {
    return std::make_pair(true,nullptr);
  }
  
};




int main(int argc, char *argv[])
{
  size_t len=100;
  std::shared_ptr<snde::wfmdatabase> wfmdb=std::make_shared<snde::wfmdatabase>();
  std::shared_ptr<snde::ndarray_waveform> test_wfm;

  wfmdb->default_storage_manager = std::make_shared<waveform_storage_manager_shmem>();

  std::shared_ptr<math_function> multiply_by_scalar_function = std::make_shared<cpp_math_function>(1,
												   [] (std::shared_ptr<waveform_set_state> wss,std::shared_ptr<instantiated_math_function> inst,bool is_mutable,bool mdonly) {
												     return std::make_shared<multiply_by_scalar>(wss,inst,is_mutable,mdonly);
												   },
												   true,
												   false,
												   false);
  
  std::shared_ptr<instantiated_math_function> scaled_channel_function = multiply_by_scalar_function->instantiate({
      std::make_shared<math_parameter_waveform>("/test_channel"),
      std::make_shared<math_parameter_double_const>(4.5),
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

  globalrev->wait_complete();
  return 0;
}
