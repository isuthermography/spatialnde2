#include <mutex>
#include <memory>

#include "snde/recstore.hpp"


#include "snde/recstore_transaction_manager.hpp"

namespace snde {
  bool measurement_time_ptr_less::operator()(const std::shared_ptr<measurement_time> a,const std::shared_ptr<measurement_time> b) const
  {
    return (*a) < b;
  }

  transaction_manager::transaction_manager()
  {
    started=false;
  }

  std::shared_ptr<transaction_manager> transaction_manager::upcast()
  {
    return shared_from_this();
  }
  
  transaction_manager::~transaction_manager()
  {
    
  }

  ordered_transaction_manager::ordered_transaction_manager() :
  transaction_manager_background_end_mustexit(false)
  {

  }
  
  void ordered_transaction_manager::startup(std::shared_ptr<recdatabase> recdb)
  {
    {
      std::lock_guard<std::mutex> transmgr_admin(admin);
      started=true;
      this->recdb=recdb;
    }
    transaction_manager_background_end_thread = std::thread([this]() { transaction_manager_background_end_code(); });
    
    // insert an empty globalrev so there always is one
    snde::active_transaction transact(recdb);
    transact.end_transaction();
  }

  std::shared_ptr<std::thread::id> ordered_transaction_manager::trans_thread_owner()
  {
    return std::atomic_load(&_trans_thread_owner);
  }

  
  std::shared_ptr<transaction> ordered_transaction_manager::start_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<measurement_time> timestamp)
  {
    std::shared_ptr<std::thread::id> owner=trans_thread_owner();
    if (owner && *owner == std::this_thread::get_id()) {
      throw snde_error("spatialnde2 ordered_transaction_manager: a single thread cannot have two transactions open simultaneously");
    }
    std::unique_lock<movable_mutex> tr_lock_acquire(transaction_lock);

    tr_lock_acquire.swap(transaction_lock_holder); // transfer lock into holder

    uint64_t previous_globalrev_index = 0;
    {
      std::lock_guard<std::mutex> manager_lock(admin);
      std::lock_guard<std::mutex> recdb_lock(recdb->admin);
      
      
      assert(!trans);
      
      trans=std::make_shared<ordered_transaction>(recdb);
      //trans = ordered_trans;
      trans->math->trans=trans; // assign it here because the trans constructor can't access it's own shared pointer
      std::atomic_store(&_trans_thread_owner,std::make_shared<std::thread::id>(std::this_thread::get_id()));
      
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
      trans->globalrev_index = previous_globalrev_index+1;
      trans->rss_unique_index = rss_get_unique(); // will be transferred into the rss during end_transaction()
    }
    return trans;

  }

  void ordered_transaction_manager::end_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<transaction> trans_base)
  {
    std::shared_ptr<globalrevision> globalrev_ptr;
    struct transaction_notifies trans_notifies;
    std::shared_ptr<ordered_transaction> ordered_trans = std::dynamic_pointer_cast<ordered_transaction>(trans_base);
    std::shared_ptr<transaction> trans_local;

    {
      std::lock_guard<std::mutex> manager_lock(admin);
      trans_local=trans;
      assert(trans && trans == ordered_trans);
      std::shared_ptr<std::thread::id> owner=trans_thread_owner();
      
      if (owner && *owner != std::this_thread::get_id()) {
        throw snde_error("ordered_transaction_manager::end_transaction(): transaction must be ended in the same thread as it was started.");
      }
    }
    
    // std::unique_lock<std::mutex> recdb_admin(recdb->admin);
    
    
    std::tie(globalrev_ptr,trans_notifies) = trans_local->_realize_transaction(recdb,ordered_trans->globalrev_index);
    
    {
      std::lock_guard<std::mutex> manager_lock(admin);
      std::shared_ptr<std::thread::id> empty_id;
      std::atomic_store(&_trans_thread_owner,empty_id);
      trans = nullptr;
    }

    std::unique_lock<movable_mutex> tr_lock_release;
    transaction_lock_holder.swap(tr_lock_release);
    tr_lock_release.unlock();
 

 
  
    trans_local->_notify_transaction_globalrev(recdb,globalrev_ptr,trans_notifies);

    {
      std::lock_guard<std::mutex> transaction_admin_lock(trans_local->admin);
      trans_local->prerequisite_state = nullptr;
      trans_local->our_state_reference = nullptr;
    }
    
   
  }

  void ordered_transaction_manager::notify_background_end_fcn(std::shared_ptr<active_transaction> act_trans)
  {
    std::shared_ptr<std::thread::id> owner=trans_thread_owner();
      
    if (*owner != std::this_thread::get_id()) {
      throw snde_error("ordered_transaction_manager: run_in_background_and_end_transaction(): transaction must be ended in the same thread as it was started.");
    }
    owner = nullptr;
    std::atomic_store(&_trans_thread_owner,owner);
      
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
    if (started) {
      transaction_manager_background_end_thread.join();
    }
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

      //if (trans->transaction_background_end_fcn) {
      std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<void> params)> transaction_background_end_fcn = trans->transaction_background_end_fcn;
      std::shared_ptr<void> transaction_background_end_params = trans->transaction_background_end_params;
       
      transaction_background_end_lockholder.unlock();
      transaction_manager_background_end_lockholder.unlock();

      std::shared_ptr<recdatabase> recdb_strong = recdb.lock();
      if (!recdb_strong) {
	return;
      }
      if (transaction_background_end_fcn) {
       
	transaction_background_end_fcn(recdb_strong,transaction_background_end_params);
      }
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
      //} else {
      //transaction_background_end_lockholder.unlock();
      //}
    }
    //fprintf(stderr,"gmnnc() exit\n");

  }

  ordered_transaction::ordered_transaction(std::shared_ptr<recdatabase> recdb) :
  transaction(recdb),
  globalrev_index(SNDE_INDEX_INVALID)
  {

  }
  
  timed_transaction_manager::timed_transaction_manager(std::shared_ptr<measurement_clock> clock,double latency_secs) :
    transaction_manager_background_end_mustexit(false),
    transaction_end_thread_mustexit(false),
    clock(clock),
    latency_secs(latency_secs)
  {

  }
  
  void timed_transaction_manager::startup(std::shared_ptr<recdatabase> recdb)
  {
    this->recdb=recdb;

    
    // insert an empty globalrev so there always is one
    snde::active_transaction transact(recdb);
    std::shared_ptr<transaction> trans=transact.end_transaction();
    std::shared_ptr<timed_transaction> timed_trans=std::dynamic_pointer_cast<timed_transaction>(trans);
    if (!timed_trans){
      throw snde_error("timed_transaction_manager startup(): Transaction is from a different kind of manager. Is this transaction manager registered with the correct recording database?");
    }
    {
      std::lock_guard<std::mutex> transmgr_admin(admin);
      started=true;
    
      transaction_map.erase(timed_trans->timestamp);
    }
    _actually_end_transaction(recdb,timed_trans);
    int threadpool_num_threads=8;
    for (int threadcnt=0;threadcnt<threadpool_num_threads;threadcnt++) {
      transaction_manager_background_end_thread_pool.push_back(std::thread([this]() { transaction_manager_background_end_code(); }));

    }

    transaction_end_thread = std::thread([this]() { transaction_end_thread_code(); });
  }

   std::shared_ptr<transaction> timed_transaction_manager::start_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<measurement_time> timestamp)
  {

    std::shared_ptr<timed_transaction> trans;

    if (!timestamp) {
      timestamp=clock->get_current_time();
    }

    uint64_t previous_globalrev_index = 0;
    {
      std::lock_guard<std::mutex> manager_lock(admin);
      std::lock_guard<std::mutex> recdb_lock(recdb->admin);
      
      
      
      
      trans=std::make_shared<timed_transaction>(recdb);
      trans->math->trans=trans; // assign it here because the trans constructor can't access it's own shared pointer

      // acquire the transaction lock for our new transaction.
      // technically this violates the locking order but there is no
      // chance it could create an actual problem because
      // the transaction is brand new and hasn't been published.
      std::unique_lock<movable_mutex> tr_lock_acquire(trans->transaction_lock);

      tr_lock_acquire.swap(trans->transaction_lock_holder); // transfer lock into holder
      
      trans->rss_unique_index = rss_get_unique(); // will be transferred into the rss during end_transaction()
      trans->timestamp=timestamp;

      transaction_map.emplace(timestamp,trans);
     
    }
    return trans;

  }
  void timed_transaction_manager::end_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<transaction> trans_base)
  {
    
    std::shared_ptr<timed_transaction> timed_trans = std::dynamic_pointer_cast<timed_transaction>(trans_base);
    assert(timed_trans);
    // std::lock_guard<std::mutex> trans_admin(timed_trans->admin);
    if (timed_trans->ended) {
      throw snde_error("transaction ended more than one time");
    }
    timed_trans->ended=true;

  }

  void timed_transaction_manager::_actually_end_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<timed_transaction> timed_trans)
  {
    std::shared_ptr<globalrevision> globalrev_ptr;
    struct transaction_notifies trans_notifies;
    uint64_t globalrev_index;
    std::shared_ptr<globalrevision> previous_globalrev;
     
    {
      std::unique_lock<std::mutex> recdb_admin(recdb->admin);
      uint64_t previous_globalrev_index = 0;
      
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
     
      globalrev_index = previous_globalrev_index+1;
    }

    {
      std::lock_guard<std::mutex> trans_admin(timed_trans->admin);
      timed_trans->prerequisite_state->rss_assign(previous_globalrev);
    }
    
    
    std::tie(globalrev_ptr,trans_notifies) = timed_trans->_realize_transaction(recdb,globalrev_index);
    
   
    std::unique_lock<movable_mutex> tr_lock_release;
    timed_trans->transaction_lock_holder.swap(tr_lock_release);
    tr_lock_release.unlock();
    timed_trans->_notify_transaction_globalrev(recdb,globalrev_ptr,trans_notifies);
    {
      std::lock_guard<std::mutex> transaction_admin_lock(timed_trans->admin);
      timed_trans->prerequisite_state = nullptr;
      timed_trans->our_state_reference = nullptr;
    }
  }

  void timed_transaction_manager::notify_background_end_fcn(std::shared_ptr<active_transaction> act_trans)
  {
    // std::shared_ptr<ordered_transaction> ordered_trans = std::dynamic_pointer_cast<ordered_transaction>(act_trans->trans);
    std::shared_ptr<transaction> trans=act_trans->trans;
    // std::lock_guard<std::mutex> trans_admin(trans->admin);

    std::lock_guard<std::mutex> trans_mgr_bkgnd(transaction_manager_background_end_lock);
    std::shared_ptr<timed_transaction> timed_trans = std::dynamic_pointer_cast<timed_transaction>(trans);
    if (!timed_trans) {
      throw snde_error("Wrong transaction class");
    }
    transaction_background_end_queue.push_front(timed_trans);
    
    
    transaction_manager_background_end_condition.notify_one();
  }

  timed_transaction_manager::~timed_transaction_manager()
  {
    {
      std::lock_guard<std::mutex> transmgr_admin(admin);
      transaction_end_thread_mustexit=true;
    }
    // Trigger transaction_manager_background_end_thread to die, then join() it. 
    {
      std::lock_guard<std::mutex> transaction_manager_background_end_lockholder(transaction_manager_background_end_lock);

      transaction_manager_background_end_mustexit=true;
      transaction_manager_background_end_condition.notify_all();
    }
    if (started) {
      for (auto && background_end_thread: transaction_manager_background_end_thread_pool) {
	background_end_thread.join();
      }
      transaction_end_thread.join();
    }
  }

  void timed_transaction_manager::transaction_manager_background_end_code()
  {

    set_thread_name(nullptr,"snde2 ttm tbec");

    std::unique_lock<std::mutex> transaction_manager_background_end_lockholder(transaction_manager_background_end_lock);
    
    //fprintf(stderr,"tbec() starting\n");

    while (true) {
      //fprintf(stderr,"tbec() waiting\n");
      transaction_manager_background_end_condition.wait(transaction_manager_background_end_lockholder,[this] { return transaction_manager_background_end_mustexit || !transaction_background_end_queue.empty(); });
      //fprintf(stderr,"tbec() wakeup\n");

      if (transaction_manager_background_end_mustexit) {
	return;
      }
      std::shared_ptr<timed_transaction> trans=transaction_background_end_queue.back();
      transaction_background_end_queue.pop_back();
      std::unique_lock<std::mutex> transaction_background_end_lockholder(trans->transaction_background_end_lock);

      // if (trans->transaction_background_end_fcn) {
      std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<void> params)> transaction_background_end_fcn = trans->transaction_background_end_fcn;
      std::shared_ptr<void> transaction_background_end_params = trans->transaction_background_end_params;
       
      transaction_background_end_lockholder.unlock();
      transaction_manager_background_end_lockholder.unlock();

      std::shared_ptr<recdatabase> recdb_strong = recdb.lock();
      if (!recdb_strong) {
	return;
      }
      if (transaction_background_end_fcn) {
	transaction_background_end_fcn(recdb_strong,transaction_background_end_params);
      }
	
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
      // } else {
      // transaction_background_end_lockholder.unlock();
      // }
    }
    //fprintf(stderr,"gmnnc() exit\n");

  }

  void timed_transaction_manager::transaction_end_thread_code()
  {
    // Thread needs to iterate through the transaction_map . The first transaction can be actually_ended if it is older than the latency and it has been ended. Otherwise, wait for the lesser of half the latency or the time when that first transaction will be older than the latency.;
    std::shared_ptr<measurement_time> last_iteration_timestamp;
    std::shared_ptr<measurement_time> last_transaction_timestamp;

    double wait_time;
    
    while (1) {
      
      std::unique_lock<std::mutex> manager_lock(admin);
      if (transaction_end_thread_mustexit) {
	
	break;
      }
      wait_time = 0.0;
      
      
      std::shared_ptr<measurement_time> iteration_timestamp= clock->get_current_time();
      //snde_warning("ttm: timestamp=%.20g",iteration_timestamp->seconds_since_epoch());
      auto transaction_map_it = transaction_map.begin();

      if (transaction_map_it != transaction_map.end()) {
	auto & time_transaction = *transaction_map_it;
	double age = iteration_timestamp->difference_seconds(time_transaction.first);
	//snde_warning("ttm: age=%g", age);
	if (age > latency_secs && time_transaction.second->ended) {
	  // this transaction is the first in the map and is older than latency and has been ended
	  //snde_warning("ttm: ending transaction");
	  std::shared_ptr<measurement_time> transaction_timestamp = time_transaction.first;
	  std::shared_ptr<timed_transaction> trans = time_transaction.second;
	  transaction_map.erase(transaction_map_it);
	  
	  manager_lock.unlock();
	  std::shared_ptr<recdatabase> recdb_strong = recdb.lock();
	  if (! recdb_strong){
	    return;
	  }
	  if (last_transaction_timestamp && transaction_timestamp->difference_seconds(last_transaction_timestamp) < 0.0) {
	    // this transaction precedes our previous transaction
	    snde_warning("timed_transaction_manager: attempting to actually_end_transaction out-of-order. Transaction discarded. Perhaps the latency parameter is too small?");
	    continue;
	  }
	  last_transaction_timestamp = transaction_timestamp;
	  //snde_warning("ttm: actually end transaction");
	  _actually_end_transaction(recdb_strong,trans);
	  //snde_warning("ttm: fully ended");
	  continue;
	} else if (age <= latency_secs) {
	  wait_time = latency_secs - age;
	  if (wait_time > latency_secs/2.0) {
	    wait_time = latency_secs/2.0;
	  }
	} else if (!time_transaction.second->ended) {
	  wait_time = (latency_secs/4.0);
	}
	//snde_warning("ttm: wait_time=%g",wait_time);
	

      } else {
	wait_time = latency_secs/2.0;
      }
      last_iteration_timestamp = iteration_timestamp;
      manager_lock.unlock();

      clock->sleep_for(wait_time);
      

      
    }
																							     
  }
  
  timed_transaction::timed_transaction(std::shared_ptr<recdatabase> recdb) :
    transaction(recdb),
    ended(false)
  {

  }

  void timed_transaction::update_timestamp(std::shared_ptr<transaction_manager> transmgr,std::shared_ptr<measurement_time> new_timestamp)
  {
    // Needs to find the relevant entry in transaction_map, remove it and replace it with the new time stamp, in addition to updating the time stamp directly.
    std::shared_ptr<timed_transaction_manager> ttr = std::dynamic_pointer_cast<timed_transaction_manager>(transmgr);
    std::lock_guard<std::mutex> transmgr_admin(transmgr->admin); // lock the transaction manager structure.
    std::lock_guard<std::mutex> trans_admin(admin); // lock the transaction structure.
    bool updated = false;
    
    for (auto transaction_map_it = ttr->transaction_map.find(timestamp); transaction_map_it != ttr->transaction_map.end(); transaction_map_it++) {
      if (*transaction_map_it->first != new_timestamp) {
	break; // we are beyond the matching entries in the multimap
	
      }
      if (transaction_map_it->second.get() == this) {
	updated = true;
	std::shared_ptr<timed_transaction> shared_this = transaction_map_it->second;
	ttr->transaction_map.erase(transaction_map_it);

	shared_this->timestamp = new_timestamp; // !!! check locking
	ttr->transaction_map.emplace(new_timestamp,shared_this);
	break;
      }

    }

    if (!updated) {
      throw snde_error("update_timestamp(): transaction not found in map.");
    }
    
     
  }
  
}
