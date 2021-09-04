#include <thread>
#include <cmath>
#include "recstore.hpp"

using namespace snde;



int main(int argc, char *argv[])
{
  size_t len=100;
  std::shared_ptr<snde::recdatabase> recdb=std::make_shared<snde::recdatabase>();
  std::shared_ptr<snde::ndarray_recording> test_rec;

  recdb->default_storage_manager = std::make_shared<recording_storage_manager_shmem>();
  recdb->compute_resources->compute_resources.push_back(std::make_shared<available_compute_resource_cpu>(recdb,recdb->compute_resources,SNDE_CR_CPU,std::thread::hardware_concurrency()));
  recdb->compute_resources->start();

  snde::active_transaction transact(recdb); // Transaction RAII holder
  std::shared_ptr<snde::channelconfig> testchan_config=std::make_shared<snde::channelconfig>("test channel", "main", (void *)&main,false);
  
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

  globalrev->wait_complete();
  return 0;
}
