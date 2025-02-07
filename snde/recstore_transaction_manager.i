%shared_ptr(snde::measurement_time);
snde_rawaccessible(snde::measurement_time);

%shared_ptr(snde::measurement_clock);
snde_rawaccessible(snde::measurement_clock);

%shared_ptr(snde::measurement_time_cpp<std::chrono::system_clock>);
snde_rawaccessible(snde::measurement_time_cpp<std::chrono::system_clock>);

%shared_ptr(snde::measurement_clock_cpp<std::chrono::system_clock>);
snde_rawaccessible(snde::measurement_clock_cpp<std::chrono::system_clock>);

%shared_ptr(snde::measurement_time_cpp<std::chrono::steady_clock>);
snde_rawaccessible(snde::measurement_time_cpp<std::chrono::steady_clock>);

%shared_ptr(snde::measurement_clock_cpp<std::chrono::steady_clock>);
snde_rawaccessible(snde::measurement_clock_cpp<std::chrono::steady_clock>);

%shared_ptr(snde::measurement_time_cpp<std::chrono::high_resolution_clock>);
snde_rawaccessible(snde::measurement_time_cpp<std::chrono::high_resolution_clock>);

%shared_ptr(snde::measurement_clock_cpp<std::chrono::high_resolution_clock>);
snde_rawaccessible(snde::measurement_clock_cpp<std::chrono::high_resolution_clock>);

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

%typemap(out) PyObject* snde::measurement_time::Py_Downcast {
  $result=$1;
  Py_INCREF($result);
#if SWIG_VERSION >= 0x040100 // Directors disabled for swig older than 4.1.0 because of uncompilable code
  director = nullptr; // disable ownership release
#endif
 }
namespace snde {

  class ordered_transaction;
#if SWIG_VERSION >= 0x040100 // Directors disabled for swig older than 4.1.0 because of uncompilable code
  %feature("director") measurement_time;
#endif
  class measurement_time {
  public:
    // immutable
    std::string epoch_start_iso8601; // may be empty string
    measurement_time(std::string epoch_start_iso8601);
    virtual ~measurement_time()=default;
    virtual double seconds_since_epoch();

    virtual double difference_seconds(std::shared_ptr<measurement_time> to_subtract);
    
    virtual const bool operator==(const std::shared_ptr<measurement_time> &rhs);

    virtual const bool operator!=(const std::shared_ptr<measurement_time> &rhs);

    virtual const bool operator<(const std::shared_ptr<measurement_time> &rhs);
    virtual const bool operator<=(const std::shared_ptr<measurement_time> &rhs);

    virtual const bool operator>(const std::shared_ptr<measurement_time> &rhs);

    virtual const bool operator>=(const std::shared_ptr<measurement_time> &rhs);
    virtual PyObject *Py_Downcast();
  };
#if SWIG_VERSION >= 0x040100 // Directors disabled for swig older than 4.1.0 because of uncompilable code
  %feature("director") measurement_clock;
#endif
  class measurement_clock {
  public:
    //std::mutex admin; // Locks member variables of this and subclasses; last in the locking order.
    std::string epoch_start_iso8601; // may be empty string
    measurement_clock(std::string epoch_start_iso8601);
    
    virtual ~measurement_clock()=default;
    virtual std::shared_ptr<measurement_time> get_current_time();
    virtual void sleep_for(double seconds);

  };
  template <typename T>
  class measurement_time_cpp: public measurement_time {
  public:
    //typedef T clock_type;
    //std::chrono::time_point<T> _point;

    //measurement_time_cpp(std::chrono::time_point<T> point,std::string epoch_start_iso8601) ;

    virtual ~measurement_time_cpp()=default;
    virtual double seconds_since_epoch();

    virtual double difference_seconds(std::shared_ptr<measurement_time> to_subtract);
    
    virtual const bool operator==(const std::shared_ptr<measurement_time> &rhs);

    virtual const bool operator!=(const std::shared_ptr<measurement_time> &rhs);

    virtual const bool operator<(const std::shared_ptr<measurement_time> &rhs);
    virtual const bool operator<=(const std::shared_ptr<measurement_time> &rhs);

    virtual const bool operator>(const std::shared_ptr<measurement_time> &rhs);

    virtual const bool operator>=(const std::shared_ptr<measurement_time> &rhs);
  };

template <typename T>
  class measurement_clock_cpp: public measurement_clock {
    // can instantiate against any of the standard C++ clocks
    // for example, std::chrono::system_clock,std::chrono::steady_clock,
    // std::chrono::high_resolution_clock
  public:
    typedef T clock_type;
    measurement_clock_cpp(std::string epoch_start_iso8601);
    virtual std::shared_ptr<measurement_time> get_current_time();
    virtual void sleep_for(double seconds);
  };
  %template(measurement_clock_cpp_system) snde::measurement_clock_cpp<std::chrono::system_clock>;
  %template(measurement_clock_cpp_steady) snde::measurement_clock_cpp<std::chrono::steady_clock>;
  %template(measurement_clock_cpp_high_resolution) snde::measurement_clock_cpp<std::chrono::high_resolution_clock>;
  class transaction_manager {
  public:
    //std::mutex admin; // locks member variables of subclasses; between transaction_lock (in ordered_transaction_manager) and recdb admin lock in locking order
    transaction_manager();
    virtual void startup(std::shared_ptr<recdatabase> recdb)=0;
    std::shared_ptr<transaction_manager> upcast(); //because of a swig bug where subclasses can't be assigned into the recording db object.
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
    ordered_transaction_manager();
       virtual void startup(std::shared_ptr<recdatabase> recdb);
    virtual std::shared_ptr<transaction> start_transaction(std::shared_ptr<recdatabase> recdb);

    virtual void end_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<transaction> trans);

    virtual void notify_background_end_fcn(std::shared_ptr<active_transaction> trans);

    virtual ~ordered_transaction_manager();
    virtual void transaction_manager_background_end_code();
    
  };
  
    
    
  class ordered_transaction: public transaction {
  public:
    //std::shared_ptr<globalrevision> previous_globalrev;
    uint64_t globalrev_index; // globalrev index for this transaction. Immutable once published
    
    //ordered_transaction();
    ordered_transaction(std::shared_ptr<recdatabase> recdb);
    // rule of 3
    ordered_transaction& operator=(const ordered_transaction &) = delete; 
    ordered_transaction(const ordered_transaction &orig) = delete;
    virtual ~ordered_transaction()=default;
  };


  class timed_transaction_manager: public transaction_manager {
  public:
    // std::mutex transaction_manager_background_end_lock; // last in the locking order except for transaction_background_end_lock. locks the condition variable and bool below.

   

   
    
    // std::condition_variable transaction_manager_background_end_condition;
    // managing the thread that can run stuff at the end of a transaction
    // std::vector<std::thread> transaction_manager_background_end_thread_pool; // used by active_transaction::run_in_background_and_end_transaction()
    bool transaction_manager_background_end_mustexit;
    std::list<std::shared_ptr<timed_transaction>> transaction_background_end_queue; // locked by transaction_manager_background_end_lock
    
    // std::multimap<std::shared_ptr<measurement_time>,std::shared_ptr<timed_transaction>,measurement_time_ptr_less> transaction_map;

    // std::thread transaction_end_thread;

    bool transaction_end_thread_mustexit; //locked by transaction manager admin lock
    
    std::weak_ptr<recdatabase> recdb;
    std::shared_ptr<measurement_clock> clock;
    double latency_secs;
    
    timed_transaction_manager(std::shared_ptr<measurement_clock> clock,double latency_secs);
    virtual void startup(std::shared_ptr<recdatabase> recdb);
    virtual std::shared_ptr<transaction> start_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<measurement_time> timestamp=nullptr);

    virtual void end_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<transaction> trans);

    virtual void _actually_end_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<timed_transaction> timed_trans);

    virtual void notify_background_end_fcn(std::shared_ptr<active_transaction> trans);
    // rule of 3
    timed_transaction_manager& operator=(const timed_transaction_manager &) = delete; 
    timed_transaction_manager(const timed_transaction_manager &orig) = delete;
    virtual ~timed_transaction_manager();
    virtual void transaction_manager_background_end_code();

    virtual void transaction_end_thread_code();
    
    

  };


  class timed_transaction: public transaction {
  public:
        // uint64_t globalrev_index; // globalrev index for this transaction. Immutable once published
     // transaction_lock is a movable_mutex so we have the freedom to unlock
    // it from a different thread than we used to lock it.
    // movable_mutex transaction_lock; // ***!!! Before any dataguzzler-python module context locks, etc. Before the recdb admin lock. Note the distinction between this and the admin lock of the class transaction.
    // std::unique_lock<movable_mutex> transaction_lock_holder;

    std::shared_ptr<measurement_time> timestamp;
    // std::atomic<bool> ended;
    bool ended;
    timed_transaction(std::shared_ptr<recdatabase> recdb);
    // rule of 3
    timed_transaction& operator=(const timed_transaction &) = delete; 
    timed_transaction(const timed_transaction &orig) = delete;
    virtual ~timed_transaction()=default;

    
    
    virtual void update_timestamp(std::shared_ptr<transaction_manager> transmgr,std::shared_ptr<measurement_time> new_timestamp);

  };

};
