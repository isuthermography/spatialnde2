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
  setup_cpu(recdb,{},std::thread::hardware_concurrency());
  setup_storage_manager(recdb);
  setup_math_functions(recdb, {});
  recdb->startup();
    


  std::shared_ptr<snde::active_transaction> transact=recdb->start_transaction(); 
  
  std::shared_ptr<snde::reserved_channel> testchan = recdb->define_channel(transact,"/test channel", "main");
  std::shared_ptr<snde::ndarray_recording_ref> test_ref = snde::create_ndarray_ref(transact,testchan,SNDE_RTN_FLOAT32);
  std::shared_ptr<snde::globalrevision> globalrev = transact->end_transaction()->globalrev_available();

  std::shared_ptr<snde::constructible_metadata> test_rec_metadata = std::make_shared<snde::constructible_metadata>();
  test_rec_metadata->AddMetaDatum(snde::metadatum_dbl("ande_array-axis0_offset",0.0));
    
  test_ref->rec->metadata = test_rec_metadata;
  test_ref->rec->mark_metadata_done();
  test_ref->allocate_storage(std::vector<snde_index>{len},false);

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
	{ test_ref, true }, // first element is recording_ref, 2nd parameter is false for read, true for write 
      });

    for (size_t cnt=0;cnt < len; cnt++) {
      test_ref->assign_double({cnt},100.0*sin(cnt));
      
    }
    // locktokens automatically dropped as it goes out of scope
    // (must drop before mark_data_ready())
  }
  test_ref->rec->mark_data_ready();
  
  globalrev->wait_complete();
  return 0;
}
