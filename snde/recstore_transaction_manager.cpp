#include <mutex>
#include <memory>

#include "snde/recstore.hpp"


#include "snde/recstore_transaction_manager.hpp"

namespace snde {
  transaction_manager::~transaction_manager()
  {
    
  }

  ordered_transaction_manager::ordered_transaction_manager(std::shared_ptr<recdatabase> recdb) :
  transaction_manager_background_end_mustexit(false),
    recdb(recdb)
  {
    transaction_manager_background_end_thread = std::thread([this]() { transaction_manager_background_end_code(); });
  }
  
  std::shared_ptr<transaction> ordered_transaction_manager::start_transaction(std::shared_ptr<recdatabase> recdb)
  {
    std::unique_lock<movable_mutex> tr_lock_acquire(transaction_lock);

    tr_lock_acquire.swap(transaction_lock_holder); // transfer lock into holder

    uint64_t previous_globalrev_index = 0;
    {
      std::lock_guard<std::mutex> manager_lock(admin);
      std::lock_guard<std::mutex> recdb_lock(recdb->admin);
      
      
      assert(!trans);
      
      trans=std::make_shared<ordered_transaction>();
      //trans = ordered_trans;
      
      std::shared_ptr<globalrevision> previous_globalrev;

      if (recdb->_globalrevs.size()) {
	// if there are any globalrevs (otherwise we are starting the first!)
	std::map<uint64_t,std::shared_ptr<globalrevision>>::iterator last_globalrev_ptr = recdb->_globalrevs.end();
	--last_globalrev_ptr; // change from final+1 to final entry
	previous_globalrev_index = last_globalrev_ptr->first;
	previous_globalrev = last_globalrev_ptr->second;

      } else {
	// this is the first globalrev
	previous_globalrev = nullptr; 
      }
      trans->prerequisite_state->rss_assign(previous_globalrev);
      trans->globalrev = previous_globalrev_index+1;
      trans->rss_unique_index = rss_get_unique(); // will be transferred into the rss during end_transaction()
    }
    return trans;

  }

  void ordered_transaction_manager::end_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<transaction> trans_base)
  {
    std::shared_ptr<globalrevision> globalrev_ptr;
    struct transaction_notifies trans_notifies;
    std::shared_ptr<ordered_transaction> ordered_trans = std::dynamic_pointer_cast<ordered_transaction>(trans_base);
    assert(trans && trans == ordered_trans);

    std::unique_lock<std::mutex> recdb_admin(recdb->admin);
    std::lock_guard<std::mutex> manager_lock(admin);
    // Give every recording created in the transaction its revision index

    {
      std::lock_guard<std::mutex> transaction_admin_lock(trans->admin);
      for (auto && name_recptr: trans->new_recordings) {
	std::shared_ptr<channel> chan = recdb->_channels.at(name_recptr.first);
	assert(name_recptr.second->info_revision == SNDE_REVISION_INVALID); // for this transaction manager, revisions are assigned now (except for math channels)
	uint64_t new_revision = ++chan->latest_revision; // atomic variable so it is safe to pre-increment
	std::lock_guard<std::mutex> recording_lock(name_recptr.second->admin);
	name_recptr.second->info->revision = new_revision;
	name_recptr.second->info_revision = new_revision;
      }

      recdb_admin.unlock();
    }
    
    std::tie(globalrev_ptr,trans_notifies) = trans->_realize_transaction(recdb,trans->globalrev);
    
    transaction_lock_holder.unlock();
    trans->_notify_transaction_globalrev(recdb,globalrev_ptr,trans_notifies);

    std::lock_guard<std::mutex> transaction_admin_lock(trans->admin);
    trans->prerequisite_state = nullptr;
    trans->our_state_reference = nullptr;
    trans = nullptr;
  }

  void ordered_transaction_manager::notify_background_end_fcn(std::shared_ptr<active_transaction> act_trans)
  {
    // std::shared_ptr<ordered_transaction> ordered_trans = std::dynamic_pointer_cast<ordered_transaction>(act_trans->trans);
    transaction_manager_background_end_condition.notify_all();
  }

  ordered_transaction_manager::~ordered_transaction_manager()
  {
    // Trigger transaction_manager_background_end_thread to die, then join() it. 
    {
      std::lock_guard<std::mutex> transaction_manager_background_end_lockholder(transaction_manager_background_end_lock);

      transaction_manager_background_end_mustexit=true;
      transaction_manager_background_end_condition.notify_all();
    }

    transaction_manager_background_end_thread.join();
  }

  void ordered_transaction_manager::transaction_manager_background_end_code()
  {

    set_thread_name(nullptr,"snde2 otm tbec");

    std::unique_lock<std::mutex> transaction_manager_background_end_lockholder(transaction_manager_background_end_lock);
    
    //fprintf(stderr,"tbec() starting\n");

    while (true) {
      //fprintf(stderr,"tbec() waiting\n");
      transaction_manager_background_end_condition.wait(transaction_manager_background_end_lockholder,[this] { return transaction_manager_background_end_mustexit || (trans && trans->transaction_background_end_fcn); });
      //fprintf(stderr,"tbec() wakeup\n");

      if (transaction_manager_background_end_mustexit) {
	return;
      }

      std::unique_lock<std::mutex> transaction_background_end_lockholder(trans->transaction_background_end_lock);

      if (trans->transaction_background_end_fcn) {
	std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<void> params)> transaction_background_end_fcn = trans->transaction_background_end_fcn;
	std::shared_ptr<void> transaction_background_end_params = trans->transaction_background_end_params;
       
	transaction_background_end_lockholder.unlock();
	transaction_manager_background_end_lockholder.unlock();

	std::shared_ptr<recdatabase> recdb_strong = recdb.lock();
	if (!recdb_strong) {
	  return;
	}

	transaction_background_end_fcn(recdb_strong,transaction_background_end_params);
	
	transaction_manager_background_end_lockholder.lock();
	transaction_background_end_lockholder.lock();
	
	// empty the std::function
	trans->transaction_background_end_fcn = std::function<void(std::shared_ptr<recdatabase> recdb, std::shared_ptr<void> params)>();
	trans->transaction_background_end_params = nullptr;
	//std::shared_ptr<active_transaction> transaction_background_end_acttrans_copy = transaction_background_end_acttrans;
	//transaction_background_end_acttrans = nullptr;
	transaction_background_end_lockholder.unlock();
	transaction_manager_background_end_lockholder.unlock();

	//transaction_background_end_acttrans_copy->end_transaction();

	end_transaction(recdb_strong,trans);
	transaction_manager_background_end_lockholder.lock();
      } else {
	transaction_background_end_lockholder.unlock();
      }
    }
    //fprintf(stderr,"gmnnc() exit\n");

  }
}
