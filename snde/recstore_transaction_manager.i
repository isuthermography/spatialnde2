%shared_ptr(snde::transaction_manager);
snde_rawaccessible(snde::transaction_manager);

%shared_ptr(snde::ordered_transaction_manager);
snde_rawaccessible(snde::ordered_transaction_manager);

%shared_ptr(snde::transaction);
snde_rawaccessible(snde::transaction);

%shared_ptr(snde::ordered_transaction);
snde_rawaccessible(snde::ordered_transaction);

%shared_ptr(snde::timed_transaction_manager);
snde_rawaccessible(snde::timed_transaction_manager);

%shared_ptr(snde::timed_transaction);
snde_rawaccessible(snde::timed_transaction);

%{
#include "snde/recstore_transaction_manager.hpp"
%}


namespace snde {

  class ordered_transaction;
  
  class transaction_manager {
  public:
    //std::mutex admin; // locks member variables of subclasses; between transaction_lock (in ordered_transaction_manager) and recdb admin lock in locking order
    virtual std::shared_ptr<transaction> start_transaction(std::shared_ptr<recdatabase> recdb) = 0;
    virtual void end_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<transaction> trans) = 0;
    virtual void notify_background_end_fcn(std::shared_ptr<active_transaction> trans) = 0;
    virtual ~transaction_manager();

  };

  class ordered_transaction_manager: public transaction_manager {
  public:
    // transaction_lock is a movable_mutex so we have the freedom to unlock
    // it from a different thread than we used to lock it.
    //movable_mutex transaction_lock; // ***!!! Before any dataguzzler-python module context locks, etc. Before the recdb admin lock. Note the distinction between this and the admin lock of the class transaction.
    //std::unique_lock<movable_mutex> transaction_lock_holder;
    std::shared_ptr<ordered_transaction> trans; // only valid while transaction_lock is held. But changing/accessing also requires the transaction_manager admin lock

    //std::mutex transaction_manager_background_end_lock; // last in the locking order except for transaction_background_end_lock. locks the condition variable and bool below. 
    //std::condition_variable transaction_manager_background_end_condition;
    // managing the thread that can run stuff at the end of a transaction
    //std::thread transaction_manager_background_end_thread; // used by active_transaction::run_in_background_and_end_transaction()
    bool transaction_manager_background_end_mustexit;


    //std::weak_ptr<recdatabase> recdb;
    ordered_transaction_manager(std::shared_ptr<recdatabase> recdb);
    
    virtual std::shared_ptr<transaction> start_transaction(std::shared_ptr<recdatabase> recdb);

    virtual void end_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<transaction> trans);

    virtual void notify_background_end_fcn(std::shared_ptr<active_transaction> trans);

    virtual ~ordered_transaction_manager();
    virtual void transaction_manager_background_end_code();
    
  };

  class ordered_transaction: public transaction {
  public:
    //std::shared_ptr<globalrevision> previous_globalrev;
    uint64_t globalrev; // globalrev index for this transaction. Immutable once published
    
    //ordered_transaction();

  };


  class timed_transaction_manager: public transaction_manager {
  public:

  };


  class timed_transaction: public transaction {
  public:

  };

};
