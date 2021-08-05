#include <cmath>
#include "wfmstore.hpp"

using namespace snde;

int main(int argc, char *argv[])
{
  size_t len=100;
  std::shared_ptr<snde::wfmdatabase> wfmdb=std::make_shared<snde::wfmdatabase>();
  std::shared_ptr<snde::ndarray_waveform> test_wfm;

  
  snde::active_transaction transact(wfmdb); // Transaction RAII holder
  std::shared_ptr<snde::channelconfig> testchan_config=std::make_shared<snde::channelconfig>("test channel", "main", (void *)&main,false);
  
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
  
  return 0;
}
