#ifndef SNDE_LOCK_TYPES_HPP
#define SNDE_LOCK_TYPES_HPP

#include <memory>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

namespace snde {

  class rwlock;  // forward declaration

  class rwlock_lockable {
    // private to rwlock
  public:
    int _writer;
    rwlock *_rwlock_obj;

    rwlock_lockable(rwlock *lock,int writer) {
      _writer=writer;
      _rwlock_obj=lock;
    }

    void lock();  // implemented in lockmanager.cpp
    void unlock(); // implemented in lockmanager.cpp
  };

    
  class rwlock: public std::enable_shared_from_this<rwlock> {
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

    std::mutex admin;
    rwlock_lockable reader;
    rwlock_lockable writer;

    std::shared_ptr<rwlock> keepalive; /* this shared_ptr is a reference to us, that prevents us from being free'd
					  We keep it set so long as the lock is locked. 
					  Thus if the lock is otherwise destroyed while it is held, 
					  it will continue to exist until it is released */
    

    //std::deque<std::function<void(snde_index firstelem,snde_index numelems)>> _dirtynotify; /* locked by admin, but functions should be called with admin lock unlocked (i.e. copy the deque before calling). This write lock will be locked. NOTE: numelems of SNDE_INDEX_INVALID means to the end of the array  */

    // For weak readers to support on-GPU caching of data, we would maintain a list of some sort
    // of these weak readers. Before we give write access, we would have to ask each weak reader
    // to relinquish. Then when write access ends by unlock or downgrade, we offer each weak
    // reader the ability to recache.

    // in addition/alternatively the allocator concept could allocate memory directly on board the GPU...
    // but this alone would be problematic for e.g. triangle lists that need to be accessed both by
    // a GPU renderer and GPGPU computation.


    rwlock() :
      writelockpending(0),
      reader(this,0),
      writer(this,1)
    {
      writelockcount=0;
      readlockcount=0;
    }

    void _wait_for_top_of_queue(std::condition_variable *cond,std::unique_lock<std::mutex> *adminlock) {

      do {
	cond->wait(*adminlock);
      } while(threadqueue.front() != cond);
      threadqueue.pop_front();
    }

    void lock_reader() { // lock for read
      std::unique_lock<std::mutex> adminlock(admin);


      std::condition_variable cond;

      if (writelockcount > 0 || !threadqueue.empty()) {
	/* add us to end of queue if locked for writing or there is a writer in the queue */
	threadqueue.push_back(&cond);
	_wait_for_top_of_queue(&cond,&adminlock);
      }

      /* been through queue once... now keep us on front of queue
         until no-longer locked for writing */
      while(writelockcount > 0) {

	threadqueue.push_front(&cond);
	_wait_for_top_of_queue(&cond,&adminlock);
      }

      readlockcount++;

      if (readlockcount==1) {
	keepalive=shared_from_this();
	// keep us from being destroyed until we are fully unlocked
      }
      /* Notify front of queue in case it is a reader and can
	 read in parallel with us */
      if (!threadqueue.empty()) {
        threadqueue.front()->notify_all();
      }
    }

    void clone_reader()
    {
      std::unique_lock<std::mutex> adminlock(admin);

      if (readlockcount < 1) {
	throw std::invalid_argument("Can only clone readlock that has positive readlockcount");
      }

      readlockcount++;

    }

    void unlock_reader() {
      // unlock for read
      std::unique_lock<std::mutex> adminlock(admin);
      readlockcount--;

      if (!threadqueue.empty()) {
        threadqueue.front()->notify_all();
      }

      if (!writelockcount && !readlockcount) {
	/* temporarily hold a reference on the stack, so
	   our object doesn't disappear until we return. 
	   Also drop adminlock whihc references our object */
	
	std::shared_ptr<rwlock> this_ptr;
	this_ptr=shared_from_this();
	keepalive=nullptr;
	adminlock.unlock();
	adminlock.release();
	
      }
    }

    //    void writer_mark_as_dirty(cachemanager *cache_with_valid_data,snde_index firstelem, snde_index numelems) {
    //  // was writer_append_region()
    //  if (writelockcount < 1) {
    //	throw std::invalid_argument("Can only append to region of something locked for write");
    //      }
    //      std::unique_lock<std::mutex> adminlock(admin);
    //      /* Should probably merge with preexisting entries here... */
    //
    //      _dirtyregions.mark_region(firstelem,numelems,cache_with_valid_data);
    //
    //    }

    //void lock_writer(snde_index firstelem,snde_index numelems) {
    // // lock for write
    // // WARNING no actual element granularity; firstelem and
    //  // numelems are stored so as to enable correct dirty
    //  // notifications
    //  std::unique_lock<std::mutex> adminlock(admin);


    //  std::condition_variable cond;

    //  if (writelockcount > 0 || readlockcount > 0) {
    //	/* add us to end of queue if locked for reading or writing */
    //	threadqueue.push_back(&cond);
    //	_wait_for_top_of_queue(&cond,&adminlock);
    //  }

    //  /* been through queue once... now keep us on front of queue
    //     until no-longer locked for writing */
    //  while(writelockcount > 0 || readlockcount > 0) {
    //
    //  threadqueue.push_front(&cond);
    //  _wait_for_top_of_queue(&cond,&adminlock);
    //  }

    //  _dirtyregions.mark_region(firstelem,numelems);

    //  writelockcount++;
    //}

    void lock_writer()
    {
     // lock for write
     // WARNING: Caller is responsible for doing any dirty-marking!!!
      std::unique_lock<std::mutex> adminlock(admin);


      std::condition_variable cond;

      if (writelockcount > 0 || readlockcount > 0) {
	/* add us to end of queue if locked for reading or writing */
	threadqueue.push_back(&cond);
	_wait_for_top_of_queue(&cond,&adminlock);
      }

      /* been through queue once... now keep us on front of queue
         until no-longer locked for writing */
      while(writelockcount > 0 || readlockcount > 0) {

	threadqueue.push_front(&cond);
	_wait_for_top_of_queue(&cond,&adminlock);
      }

      writelockcount++;
      
      if (writelockcount==1) {
	keepalive=shared_from_this();
	// keep us from being destroyed until we are fully unlocked
      }
    }


    void downgrade() {
      std::unique_lock<std::mutex> adminlock(admin);
      if (writelockcount < 1) {
	throw std::invalid_argument("Can only downgrade lock that has positive writelockcount");
      }
      writelockcount--;
      readlockcount++;

      /* notify waiters that they might be able to read now */
      if (!threadqueue.empty()) {
        threadqueue.front()->notify_all();
      }
    }

    void sidegrade() {
      // Get read access while we already have write access
      std::unique_lock<std::mutex> adminlock(admin);
      if (writelockcount < 1) {
	throw std::invalid_argument("Can only sidegrade lock that has positive writelockcount");
      }
      readlockcount++;

    }

    void clone_writer()
    {
      std::unique_lock<std::mutex> adminlock(admin);

      if (writelockcount < 1) {
	throw std::invalid_argument("Can only clone lock that has positive writelockcount");
      }

      writelockcount++;

    }

    void unlock_writer() {
      // unlock for write
      std::unique_lock<std::mutex> adminlock(admin);

      if (writelockcount < 1) {
	throw std::invalid_argument("Can only unlock lock that has positive writelockcount");
      }


      //if (_dirtynotify.size() > 0 && writelockcount==1) {
      //std::deque<std::function<void(snde_index firstelem,snde_index numelems)>> dirtynotifycopy(_dirtynotify);

      //// make thread-safe copy of dirtyregions;
      //// since we are locked for write nobody had better
      //// be messing with dirtyregions
      //
      //adminlock.unlock();

      //for (auto & callback: dirtynotifycopy) {
      //for (auto & region: _dirtyregions) {
      //	    if (region.second->regionend==SNDE_INDEX_INVALID) { // to infinity
      //callback(region.second->regionstart,SNDE_INDEX_INVALID);// repeat callback for each dirty region
      //
      //} else {
      //callback(region.second->regionstart,region.second->regionend-region.second->regionstart);// repeat callback for each dirty region
      //}
      //}
      //}

      //adminlock.lock();
      //}
      
      writelockcount--;
      //
      //if (!writelockcount) {
      //_dirtyregions.clear_all();
      //}

      if (!writelockcount && !threadqueue.empty()) {
        threadqueue.front()->notify_all();
      }

      if (!writelockcount && !readlockcount) {
	/* temporarily hold a reference on the stack, so
	   our object doesn't disappear until we return. 
	   Also drop adminlock whihc references our object */
	
	std::shared_ptr<rwlock> this_ptr;
	this_ptr=shared_from_this();
	keepalive=nullptr;
	adminlock.unlock();
	adminlock.release();
	
      }
    }


  };


  typedef int64_t lockindex_t; 
  
  // Persistent token of lock ownership
  typedef std::unique_lock<rwlock_lockable> rwlock_token_content;
  typedef std::shared_ptr<rwlock_token_content> rwlock_token;

  
  // Set of tokens
  typedef std::unordered_map<rwlock_lockable *,rwlock_token> rwlock_token_set_content;
  typedef std::shared_ptr<rwlock_token_set_content> rwlock_token_set;




  
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

  static inline void release_rwlock_token_set(rwlock_token_set &tokens)
  /* release_token_set releases our references to a particular 
     token set. Once all references are released, any locked tokens
     become unlocked */
  {
    tokens.reset(); // release our reference 
  }

  static inline void unlock_rwlock_token_set(rwlock_token_set tokens)
  /* unlock_token_set() explicitly unlocks all underlying tokens. 
     It is an error (std::system_error) to unlock more than once. 
     All other references to these tokens also become unlocked */
  {
    for (auto & lockableptr_token : *tokens) {
      lockableptr_token.second->unlock();
    }
  }
  
  //static inline rwlock_token_set_mark_as_dirty(rwlock_token_set tokens,cachemanager *cache_with_valid_data,snde_index firstelem, snde_index numelems)
  //{
  //  for (auto & lockableptr_token : *tokens) {
  //    assert(lockableptr_token.second->owns_lock()); /* lock should be locked in order to mark it as dirty */
  //    /* so long as it is locked, it had better exist, so the pointer is valid */
  //    lockable_ptr_token.first->_rwlock_obj->writer_mark_as_dirty(cache_with_valid_data,firstelem,numelems);
  //  }

  //}

  
  static inline rwlock_token_set empty_rwlock_token_set(void)
  {
    return std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
  }

  static inline bool check_rwlock_token_set(rwlock_token_set tokens)
  /* checks to make sure that referenced tokens are actually locked.
     returns true if all are locked, false if any are unlocked */
  {
    bool all_locked=true;
    
    for (auto & lockableptr_token : *tokens) {
      all_locked = all_locked && lockableptr_token.second->owns_lock();
    }
    return all_locked; 
  }
  
  

  static inline void merge_into_rwlock_token_set(rwlock_token_set accumulator, rwlock_token_set tomerge)
  {
    std::unordered_map<rwlock_lockable *,rwlock_token>::iterator foundtoken;
      
    for (std::unordered_map<rwlock_lockable *,rwlock_token>::iterator lockable_token=tomerge->begin();lockable_token != tomerge->end();lockable_token++) {

      foundtoken=accumulator->find(lockable_token->first);
      if (foundtoken != accumulator->end() && foundtoken->first->_writer) {
	/* already have this key in accumulator , and what we already have is a write lock
	   ... can't do better than this so do nothing */
      } else {
	/* pull into accumulator */
	(*accumulator)[lockable_token->first]=lockable_token->second;
      }
    }
  }

  
  static inline rwlock_token_set clone_rwlock_token_set(rwlock_token_set orig)
  {
    /* Clone a rwlock_token_set so that the copy can be delegated
       to a thread. Note that once a write lock is delegated to another
       thread, the locking is no longer useful for ensuring that writes
       only occur from one thread unless the original token set
       is immediately released (by orig.reset() and on all copies) */

    rwlock_token_set copy(new std::unordered_map<rwlock_lockable *,rwlock_token>());
    rwlock_lockable *lockable;

    for (std::unordered_map<rwlock_lockable *,rwlock_token>::iterator lockable_token=orig->begin();lockable_token != orig->end();lockable_token++) {
      lockable=lockable_token->first;
      /* clone the lockable */
      if (lockable->_writer) {
	lockable->_rwlock_obj->clone_writer();
      } else {
	lockable->_rwlock_obj->clone_reader();
      }
      /* now make a rwlock_token representing the clone
	 and put it in the copy */
      (*copy)[lockable]=std::make_shared<std::unique_lock<rwlock_lockable>>(*lockable,std::adopt_lock);
    }

    return copy;
  }


  class movable_mutex {
    // This class acts like a mutex, but once locked
    // can be unlocked from another thread without an error
    // meets the BasicLockable named requirement
  public:
    bool mutex_is_owned;

    std::mutex mio_lock;
    std::condition_variable mio_cond;

    movable_mutex() :
      mutex_is_owned(false)
    {

    }
    
    void lock()
    {
      std::unique_lock<std::mutex> mio_lock_holder(mio_lock);
      
      mio_cond.wait(mio_lock_holder,[ this ] {return !mutex_is_owned;});

      mutex_is_owned = true; 
    }

    void unlock()
    {
      std::lock_guard<std::mutex> mio_lock_holder(mio_lock);
      assert(mutex_is_owned);
      mutex_is_owned = false;

      mio_cond.notify_all();
    }
  };
  
};

#endif /* LOCK_TYPES_HPP */
