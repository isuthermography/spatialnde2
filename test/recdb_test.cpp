#include <thread>
#include <cmath>
#include "snde/recstore.hpp"
#include "snde/allocator.hpp"
#include "snde/recstore_setup.hpp"

using namespace snde;



int main(int argc, char *argv[])
{
  size_t len=100;
  std::shared_ptr<snde::recdatabase> recdb=std::make_shared<snde::recdatabase>();
  setup_cpu(recdb,std::thread::hardware_concurrency());
  setup_storage_manager(recdb);
  setup_math_functions(recdb, {});
  recdb->startup();
    
  std::shared_ptr<snde::ndarray_recording_ref> test_rec;


  snde::active_transaction transact(recdb); // Transaction RAII holder
  std::shared_ptr<snde::channelconfig> testchan_config=std::make_shared<snde::channelconfig>("test channel", "main", (void *)&main,false);
  
  std::shared_ptr<snde::channel> testchan = recdb->reserve_channel(testchan_config);
  test_rec = create_recording_ref(recdb,testchan,(void *)&main,SNDE_RTN_FLOAT32);
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
    // (must drop before mark_as_ready())
  }
  test_rec->rec->mark_as_ready();
  
  globalrev->wait_complete();
  return 0;
}
