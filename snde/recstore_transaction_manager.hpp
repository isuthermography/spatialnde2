#ifndef SNDE_RECSTORE_TRANSACTION_MANAGER_HPP
#define SNDE_RECSTORE_TRANSACTION_MANAGER_HPP
#include <chrono>

#ifdef SNDE_PYTHON_SUPPORT_ENABLED
#include <Python.h>
#endif

namespace snde {

  class ordered_transaction;
  class timed_transaction;

  class measurement_time {
  public:
    // immutable
    std::string epoch_start_iso8601; // may be empty string
    measurement_time(std::string epoch_start_iso8601):
      epoch_start_iso8601(epoch_start_iso8601)
    {

    }
    virtual ~measurement_time()=default;
    virtual double seconds_since_epoch()
    {
      return std::numeric_limits<double>::quiet_NaN();
    }
    virtual double difference_seconds(std::shared_ptr<measurement_time> to_subtract)
    {
      return std::numeric_limits<double>::quiet_NaN();
    }
    
    virtual const bool operator==(const std::shared_ptr<measurement_time> &rhs)
    {
      throw snde_error("Invalid abstract base class comparison");
    }

    virtual const bool operator!=(const std::shared_ptr<measurement_time> &rhs)
    {
      throw snde_error("Invalid abstract base class comparison");
    }
    virtual const bool operator>(const std::shared_ptr<measurement_time> &rhs)
    {
      throw snde_error("Invalid abstract base class comparison");
    }
    virtual const bool operator>=(const std::shared_ptr<measurement_time> &rhs)
    {
      throw snde_error("Invalid abstract base class comparison");
    }
    virtual const bool operator<(const std::shared_ptr<measurement_time> &rhs)
    {
      throw snde_error("Invalid abstract base class comparison");
    }
    virtual const bool operator<=(const std::shared_ptr<measurement_time> &rhs)
    {
      throw snde_error("Invalid abstract base class comparison");
    }
#ifdef SNDE_PYTHON_SUPPORT_ENABLED
    virtual PyObject *Py_Downcast()
    {
      return nullptr;
    }
#endif
  };

  class measurement_time_ptr_less {
  public:
    bool operator()(const std::shared_ptr<measurement_time> a,const std::shared_ptr<measurement_time> b) const;
  };
  
  class measurement_clock {
  public:
    std::mutex admin; // Locks member variables of this and subclasses; last in the locking order.
    std::string epoch_start_iso8601; // may be empty string
    measurement_clock(std::string epoch_start_iso8601) :
      epoch_start_iso8601(epoch_start_iso8601)
    {

    }
    
    virtual ~measurement_clock()=default;
    virtual std::shared_ptr<measurement_time> get_current_time()
    {
      throw snde_error("Abstract base class get_current_time()");
    }
    virtual void sleep_for(double seconds)
    {
      throw snde_error("Abstract base class sleep_for()");
    }

  };

  template <typename T>
  class measurement_time_cpp: public measurement_time {
  public:
    typedef T clock_type;
    std::chrono::time_point<T> _point;

    measurement_time_cpp(std::chrono::time_point<T> point,std::string epoch_start_iso8601) :
      _point(point),
      measurement_time(epoch_start_iso8601)
    {


    }
    virtual const bool operator==(const std::shared_ptr<measurement_time> &rhs)
    {
      std::shared_ptr<measurement_time_cpp> rhscast=std::dynamic_pointer_cast<measurement_time_cpp>(rhs);
      if (!rhscast) {
	throw snde_error("Type mismatch in time comparison: %s vs. %s",demangle_type_name(typeid(*this).name()).c_str(),demangle_type_name(typeid(rhs.get()).name()).c_str());
      }
      return _point==rhscast->_point;
    }

    virtual const bool operator!=(const std::shared_ptr<measurement_time> &rhs)
    {
      std::shared_ptr<measurement_time_cpp> rhscast=std::dynamic_pointer_cast<measurement_time_cpp>(rhs);
      if (!rhscast) {
	throw snde_error("Type mismatch in time comparison: %s vs. %s",demangle_type_name(typeid(*this).name()).c_str(),demangle_type_name(typeid(rhs.get()).name()).c_str());
      }
      return _point!=rhscast->_point;
    }

    virtual const bool operator<(const std::shared_ptr<measurement_time> &rhs)
    {
      std::shared_ptr<measurement_time_cpp> rhscast=std::dynamic_pointer_cast<measurement_time_cpp>(rhs);
      if (!rhscast) {
	throw snde_error("Type mismatch in time comparison: %s vs. %s",demangle_type_name(typeid(*this).name()).c_str(),demangle_type_name(typeid(rhs.get()).name()).c_str());
      }
      return _point<rhscast->_point;
    }

    virtual const bool operator<=(const std::shared_ptr<measurement_time> &rhs)
    {
      std::shared_ptr<measurement_time_cpp> rhscast=std::dynamic_pointer_cast<measurement_time_cpp>(rhs);
      if (!rhscast) {
	throw snde_error("Type mismatch in time comparison: %s vs. %s",demangle_type_name(typeid(*this).name()).c_str(),demangle_type_name(typeid(rhs.get()).name()).c_str());
      }
      return _point<=rhscast->_point;
    }

    virtual const bool operator>(const std::shared_ptr<measurement_time> &rhs)
    {
      std::shared_ptr<measurement_time_cpp> rhscast=std::dynamic_pointer_cast<measurement_time_cpp>(rhs);
      if (!rhscast) {
	throw snde_error("Type mismatch in time comparison: %s vs. %s",demangle_type_name(typeid(*this).name()).c_str(),demangle_type_name(typeid(rhs.get()).name()).c_str());
      }
      return _point>rhscast->_point;
    }

    virtual const bool operator>=(const std::shared_ptr<measurement_time> &rhs)
    {
      std::shared_ptr<measurement_time_cpp> rhscast=std::dynamic_pointer_cast<measurement_time_cpp>(rhs);
      if (!rhscast) {
	throw snde_error("Type mismatch in time comparison: %s vs. %s",demangle_type_name(typeid(*this).name()).c_str(),demangle_type_name(typeid(rhs.get()).name()).c_str());
      }
      return _point>=rhscast->_point;
    }
    
    virtual double seconds_since_epoch()
    {
      std::chrono::duration<typename T::rep,typename T::period> dur=_point.time_since_epoch();
      return dur.count()*1.0*T::period::num/T::period::den;
    }
    virtual double difference_seconds(std::shared_ptr<measurement_time> to_subtract)
    {
      std::chrono::duration<typename T::rep,typename T::period> dur=_point.time_since_epoch();
      std::shared_ptr<measurement_time_cpp> to_subtract_cast=std::dynamic_pointer_cast<measurement_time_cpp>(to_subtract);
      if (!to_subtract_cast) {
	throw snde_error("Type mismatch in time subtraction: %s vs. %s",demangle_type_name(typeid(*this).name()).c_str(),demangle_type_name(typeid(to_subtract.get()).name()).c_str());
      }
      std::chrono::duration<typename T::rep,typename T::period> to_subtract_dur=to_subtract_cast->_point.time_since_epoch();
      return (dur.count()-to_subtract_dur.count())*1.0*T::period::num/T::period::den;
    }
  };

  // seconds_to_clock_duration<T> creates a duration object for a
  // clock of the specified type with the specified minimum delay
  // in seconds. If the clock does not use a floating point period,
  // then the duration is rounded up to the next multiple of its period.
  
  template <typename T>
  typename std::enable_if<std::chrono::treat_as_floating_point<typename T::rep>::value,std::chrono::duration<typename T::rep,typename T::period>>::type
  seconds_to_clock_duration(double seconds)
  {
    return std::chrono::duration<typename T::rep,typename T::period>((typename T::rep)(seconds*T::period::den/T::period::num));
  }

  
  template <typename T>
  typename std::enable_if<!std::chrono::treat_as_floating_point<typename T::rep>::value,std::chrono::duration<typename T::rep,typename T::period>>::type
  seconds_to_clock_duration(double seconds)
  {
    return std::chrono::duration<typename T::rep,typename T::period>((typename T::rep)(std::ceil(seconds*T::period::den/T::period::num)));
  }
  
  template <typename T>
  class measurement_clock_cpp: public measurement_clock {
    // can instantiate against any of the standard C++ clocks
    // for example, std::chrono::system_clock,std::chrono::steady_clock,
    // std::chrono::high_resolution_clock
  public:
    typedef T clock_type;
    measurement_clock_cpp(std::string epoch_start_iso8601) :
      measurement_clock(epoch_start_iso8601)
    {



    }
    virtual std::shared_ptr<measurement_time> get_current_time()
    {
      return std::make_shared<measurement_time_cpp<T>>(T::now(),epoch_start_iso8601);
    }
    virtual void sleep_for(double seconds)
    {
      
	
      std::chrono::duration<typename T::rep,typename T::period> dur(seconds_to_clock_duration<T>(seconds));

      std::this_thread::sleep_for(dur);
      
    }
  };
      
  class transaction_manager: public std::enable_shared_from_this<transaction_manager> {
  public:
    std::mutex admin; // locks member variables of subclasses; between transaction_lock (in ordered_transaction_manager) and recdb admin lock in locking order. Precedes the admin lock of a transaction proper.
    std::atomic<bool> started;
    transaction_manager();
    virtual void startup(std::shared_ptr<recdatabase> recdb) =0;
    std::shared_ptr<transaction_manager> upcast(); //because of a swig bug where subclasses can't be assigned into the recording db object.
    virtual std::shared_ptr<transaction> start_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<measurement_time> timestamp=nullptr) = 0;
    virtual void end_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<transaction> trans) = 0;
    virtual void notify_background_end_fcn(std::shared_ptr<active_transaction> trans) = 0;
    // rule of 3
    transaction_manager& operator=(const transaction_manager &) = delete; 
    transaction_manager(const transaction_manager &orig) = delete;
    virtual ~transaction_manager();

  };

  class ordered_transaction_manager: public transaction_manager {
  public:
    // transaction_lock is a movable_mutex so we have the freedom to unlock
    // it from a different thread than we used to lock it.
    movable_mutex transaction_lock; // ***!!! Before any dataguzzler-python module context locks, etc. Before the recdb admin lock. Note the distinction between this and the admin lock of the class transaction.
    std::unique_lock<movable_mutex> transaction_lock_holder;
    std::shared_ptr<ordered_transaction> trans; // only valid while transaction_lock is held. But changing/accessing also requires the transaction_manager admin lock
    std::shared_ptr<std::thread::id> _trans_thread_owner; // atomic shared pointer; use trans_thread_owner() accessor

    std::mutex transaction_manager_background_end_lock; // last in the locking order except for transaction_background_end_lock. locks the condition variable and bool below. 
    std::condition_variable transaction_manager_background_end_condition;
    // managing the thread that can run stuff at the end of a transaction
    std::thread transaction_manager_background_end_thread; // used by active_transaction::run_in_background_and_end_transaction()
    bool transaction_manager_background_end_mustexit;


    std::weak_ptr<recdatabase> recdb;
    ordered_transaction_manager();
    virtual void startup(std::shared_ptr<recdatabase> recdb);
    std::shared_ptr<std::thread::id> trans_thread_owner();
    virtual std::shared_ptr<transaction> start_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<measurement_time> timestamp=nullptr);

    virtual void end_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<transaction> trans);

    virtual void notify_background_end_fcn(std::shared_ptr<active_transaction> trans);
    // rule of 3
    ordered_transaction_manager& operator=(const ordered_transaction_manager &) = delete; 
    ordered_transaction_manager(const ordered_transaction_manager &orig) = delete;
    virtual ~ordered_transaction_manager();
    virtual void transaction_manager_background_end_code();
    
  };

  class ordered_transaction: public transaction {
  public:
    //std::shared_ptr<globalrevision> previous_globalrev;
    uint64_t globalrev_index; // globalrev index for this transaction. Immutable once published
    
    ordered_transaction(std::shared_ptr<recdatabase> recdb);
    // rule of 3
    ordered_transaction& operator=(const ordered_transaction &) = delete; 
    ordered_transaction(const ordered_transaction &orig) = delete;
    virtual ~ordered_transaction()=default;
  };


  class timed_transaction_manager: public transaction_manager {
  public:
    std::mutex transaction_manager_background_end_lock; // last in the locking order except for transaction_background_end_lock. locks the condition variable and bool below.

   

   
    
    std::condition_variable transaction_manager_background_end_condition;
    // managing the thread that can run stuff at the end of a transaction
    std::vector<std::thread> transaction_manager_background_end_thread_pool; // used by active_transaction::run_in_background_and_end_transaction()
    bool transaction_manager_background_end_mustexit;
    std::list<std::shared_ptr<timed_transaction>> transaction_background_end_queue; // locked by transaction_manager_background_end_lock
    
    std::multimap<std::shared_ptr<measurement_time>,std::shared_ptr<timed_transaction>,measurement_time_ptr_less> transaction_map;

    std::thread transaction_end_thread;

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
    movable_mutex transaction_lock; // ***!!! Before any dataguzzler-python module context locks, etc. Before the recdb admin lock. Note the distinction between this and the admin lock of the class transaction.
    std::unique_lock<movable_mutex> transaction_lock_holder;

    std::shared_ptr<measurement_time> timestamp;
    std::atomic<bool> ended;
    timed_transaction(std::shared_ptr<recdatabase> recdb);
    // rule of 3
    timed_transaction& operator=(const timed_transaction &) = delete; 
    timed_transaction(const timed_transaction &orig) = delete;
    virtual ~timed_transaction()=default;

    
    
    virtual void update_timestamp(std::shared_ptr<transaction_manager> transmgr,std::shared_ptr<measurement_time> new_timestamp);
  };

};

#endif //SNDE_RECSTORE_TRANSACTION_MANAGER_HPP
