%shared_ptr(snde::rwlock);
snde_rawaccessible(snde::rwlock);
%shared_ptr(snde::rwlock_token_content);
snde_rawaccessible(snde::rwlock_token_content);
%shared_ptr(snde::rwlock_token_set_content);
snde_rawaccessible(snde::rwlock_token_set_content);

%{
  
#include "lock_types.hpp"
%}


namespace snde {

  class rwlock;  // forward declaration

  class rwlock_lockable {
    // private to rwlock
  public:
    int _writer;
    rwlock *_rwlock_obj;
    
    rwlock_lockable(rwlock *lock,int writer);

    void lock();  // implemented in lockmanager.cpp
    void unlock(); // implemented in lockmanager.cpp
  };


    
  class rwlock /*: public std::enable_shared_from_this<rwlock>  */ {
  public:
    // loosely motivated by https://stackoverflow.com/questions/11032450/how-are-read-write-locks-implemented-in-pthread
    // * Instantiate snde::rwlock()
    // * To read lock, define a std::unique_lock<snde::rwlock> readlock(rwlock_object)
    // * To write lock, define a std::unique_lock<snde::rwlock_writer> readlock(rwlock_object.writer)
    // Can alternatively use lock_guard if you don't need to be able to unlock within the context

    std::deque<std::condition_variable *> threadqueue;
    int writelockpending;
    int writelockcount;
    int readlockcount;
    //size_t regionstart; // if in subregions only
    //size_t regionend; // if in subregions only

    //std::mutex admin;  (swig-incompatible)
    rwlock_lockable reader;
    rwlock_lockable writer;




    // For weak readers to support on-GPU caching of data, we would maintain a list of some sort
    // of these weak readers. Before we give write access, we would have to ask each weak reader
    // to relinquish. Then when write access ends by unlock or downgrade, we offer each weak
    // reader the ability to recache.

    // in addition/alternatively the allocator concept could allocate memory directly on board the GPU...
    // but this alone would be problematic for e.g. triangle lists that need to be accessed both by
    // a GPU renderer and GPGPU computation.


    rwlock();
    
    void _wait_for_top_of_queue(std::condition_variable *cond,std::unique_lock<std::mutex> *adminlock);
    
    void lock_reader();

    void clone_reader();
    void unlock_reader();

    //void writer_append_region(snde_index firstelem, snde_index numelems);
    //void lock_writer(snde_index firstelem,snde_index numelems);

    void lock_writer();

    void downgrade();

    void sidegrade();

    void clone_writer();
    void unlock_writer();

  };

  typedef long lockindex_t; // ***!!!! Should really be int64_t, but we do this to work around swig bugs...

  //typedef std::unique_lock<rwlock_lockable> rwlock_token_content;
  class rwlock_token_content {

  };
  typedef std::shared_ptr<rwlock_token_content> rwlock_token;

  //typedef std::unordered_map<rwlock_lockable *,rwlock_token> rwlock_token_set_content;
  class rwlock_token_set_content {}; // rwlock_token_set_content is fully abstract from the SWIG perspective
  
  //typedef std::shared_ptr<rwlock_token_set_content> rwlock_token_set;
  // Persistent token of lock ownership
  //typedef std::shared_ptr<std::unique_lock<rwlock_lockable>> rwlock_token;
  // Set of tokens
  typedef std::shared_ptr<rwlock_token_set_content> rwlock_token_set;
};

// On the python side, no distinction between the class and shared_ptr
%pythoncode %{
  rwlock_token_set = rwlock_token_set_content
%}

/*
%typemap(out) snde::rwlock_token_set {
  std::shared_ptr<snde::rwlock_token_set_content> *smartresult = bool(result) ? new std::shared_ptr<snde::rwlock_token_set_content>(result SWIG_NO_NULL_DELETER_SWIG_POINTER_NEW) :0;
  %set_output(SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(std::shared_ptr< snde::rwlock_token_set_content > *), SWIG_POINTER_NEW|SWIG_POINTER_OWN));

}*/

namespace snde{


  /* rwlock_token_set semantics: 
     The rwlock_token_set contains a reference-counted set of locks, specific 
     to a particular thread. They can be passed around, copied, etc.
     at will within a thread context. 
     
     A lock is unlocked when: 
        (a) unlock_rwlock_token_set() is called on an rwlock_token_set containing 
	    the lock, or 
        (b) All references to all rwlock_token_sets containing the lock
	    are released (by release_rwlock_token_set()) or go out of scope
     Note that it is an error (std::system_error) to call unlock_rwlock_token_set()
     on an rwlock_token_set that contains any locks that already have been
     unlocked (i.e. by a prior call to unlock_rwlock_token_set()) 

     It is possible to pass a token_set to another thread, but 
     only by creating a completely independent cloned copy and 
     completely delegating the cloned copy to the other thread.

     The cloned copy is created with clone_rwlock_token_set(). 
     The unlock()ing the cloned copy is completely separate from
     unlocking the original.
*/

  static inline void release_rwlock_token_set(rwlock_token_set &tokens);

  static inline void unlock_rwlock_token_set(rwlock_token_set tokens);
  



  static inline snde::rwlock_token_set empty_rwlock_token_set(void);

  static inline bool check_rwlock_token_set(rwlock_token_set tokens);
  

  static inline void merge_into_rwlock_token_set(rwlock_token_set accumulator, rwlock_token_set tomerge);

  static inline rwlock_token_set clone_rwlock_token_set(rwlock_token_set orig);
};


//%template(rwlock_token_content) std::unique_lock<rwlock_lockable>;
//%template(rwlock_token) std::shared_ptr<snde::rwlock_token_content>;  
//%template(rwlock_token_set) std::shared_ptr<snde::rwlock_token_set_content>;  


