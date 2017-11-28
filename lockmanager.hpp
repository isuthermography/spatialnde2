#ifndef SNDE_LOCKMANAGER
#define SNDE_LOCKMANAGER


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

  struct arrayregion {
    void **array;
    size_t indexstart;
    size_t indexend;
  };

  class rwlock {
  public:
    // loosely motivated by https://stackoverflow.com/questions/11032450/how-are-read-write-locks-implemented-in-pthread
    // * Instantiate snde::rwlock()
    // * To read lock, define a std::unique_lock<snde::rwlock> readlock(rwlock_object)
    // * To write lock, define a std::unique_lock<snde::rwlock_writer> readlock(rwlock_object.writer)
    // Can alternatively use lock_guard if you don't need to be able to unlock within the context

    std::deque<std::condition_variable *> threadqueue;
    int writelockcount;
    int readlockcount;
    //size_t regionstart; // if in subregions only
    //size_t regionend; // if in subregions only
    std::mutex admin;
    rwlock_lockable reader;
    rwlock_lockable writer;

    // For weak readers to support on-GPU caching of data, we would maintain a list of some sort
    // of these weak readers. Before we give write access, we would have to ask each weak reader
    // to relinquish. Then when write access ends by unlock or downgrade, we offer each weak
    // reader the ability to recache.

    // in addition/alternatively the allocator concept could allocate memory directly on board the GPU...
    // but this alone would be problematic for e.g. triangle lists that need to be accessed both by
    // a GPU renderer and GPGPU computation.


    rwlock() :
      reader(this,0),
      writer(this,1)
    {
      writelockcount=0;
      readlockcount=0;
    }

    void _wait_for_top_of_queue(std::condition_variable *cond,std::unique_lock<std::mutex> *adminlock) {

      while(threadqueue.front() != cond) {
	cond->wait(*adminlock);
      }
      threadqueue.pop_front();
    }

    void lock_reader() { // lock for read
      std::unique_lock<std::mutex> adminlock(admin);


      std::condition_variable cond;

      if (writelockcount > 0) {
	/* add us to end of queue if locked for writing */
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

      /* Notify front of queue in case it is a reader and can
	 read in parallel with us */
      if (!threadqueue.empty()) {
        threadqueue.front()->notify_all();
      }
    }

    void unlock_reader() {
      // unlock for read
      std::unique_lock<std::mutex> adminlock(admin);
      readlockcount--;

      if (!threadqueue.empty()) {
        threadqueue.front()->notify_all();
      }
    }

    void lock_writer() {
      // lock for write
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
    }

    void downgrade() {
      if (writelockcount != 1) {
	throw std::invalid_argument("Can only downgrade lock that has writelockcount of 1");
      }
      writelockcount--;
      readlockcount++;

      /* notify waiters that they might be able to read now */
      if (!threadqueue.empty()) {
        threadqueue.front()->notify_all();
      }
    }

    void unlock_writer() {
      // unlock for write
      std::unique_lock<std::mutex> adminlock(admin);

      writelockcount--;

      if (!threadqueue.empty()) {
        threadqueue.front()->notify_all();
      }
    }


  };


  // Persistent token of lock ownership
  typedef std::shared_ptr<std::lock_guard<rwlock_lockable>> rwlock_token;

  // Set of tokens
  typedef std::shared_ptr<std::map<rwlock_lockable *,rwlock_token>> rwlock_token_set;



  class arraylock {
  public:
    //std::mutex admin; /* locks access to full_array and subregions members.. obsolete.... just use the lockmanager's admin lock */

    rwlock full_array;
    //std::vector<rwlock> subregions;

    arraylock() {

    }
  };



  class lockmanager {
    /* Manage all of the locks/mutexes/etc. for a class
       which contains a bunch of arrays handled by
       snde::allocator */


    /* Need to add capability for long-term persistent weak read locks
       that can be notified that they need to go away (e.g. to support
       reallocation). These
       can be used to keep data loaded into a rendering pipline
       or an opencl context */

    /* This class handles multiple arrays.


       We define a read/write lock per array, or more precisely
       per allocated array.

       All arrays managed by a particular allocator
       are implicitly locked by locking the primary
       array for that allocator

       The locking order is determined by the reverse of the (order of the
       calls to addarray() which should generally match
       the order of the arrays in your structure that holds them).

       Our API provides a means to request locking of
       sub-regions of an array, but for now these
       simply translate to nested requests to keep the
       entire array locked.... That means you can't lock
       some regions for read and some regions for write
       of the same array simultaneously.

       You can lock all arrays together with get_locks_write_all()
       or get_locks_read_all()

       You can lock one particular array with get_locks_read_array()
       or get_locks_write_array(). Please note that you must
       follow the locking order convention when doing this
       (or already hold the locks, if you just want another lock token
       for a particular array or region)

       You can lock a region of one particular array with
       get_locks_read_array_region() or get_locks_write_array_region().
       Please note that you must follow the locking order convention
       when doing this (or already hold the locks, if you just
       want another lock token for a particular array or region)
       The locking order convention for regions is that later regions
       must be locked prior to earlier regions.
       (Please note that the current version does not actually implement
       region-level granularity)

       You can lock several arrays with get_locks_read_arrays(),
       get_locks_write_arrays(), get_locks_read_arrays_regions(),
       or get_locks_write_arrays_regions(). Assuming either
       nothing is locked beforehand or everything is locked beforehand,
       locking order is not an issue because these functions
       sort your vector of arrays into the proper order.

       These functions return an rwlock_token_set, which is really
       a std::shared_ptr. When the rwlock_token_set and any copies/
       assignments/etc. go out of scope, the lock is released.
       Think of the rwlock_token_set as a handle to what you have
       locked. You can add references to the handle, which will
       prevent those tokens from being released. That is distinct
       from locking the same locks a second time, which gives
       you a new rwlock_token_set that can be passed around
       separately. Conceptually a rwlock_token_set is a single
       set of access rights. If you want to delegate access
       rights to somewhere else, lock the same resources
       a second time (reference the rights you already have),
       release your own rights as appropriate, and pass the
       new rights (optionally downgraded) on to the destination.

       If you hold some locks and want more, or want multiple locks
       on the same resource, you should pass the rwlock_token_set(s)
       that you have as parameters to the locking function. Each
       locking function has a version that accepts a rwlock_token_set
       and a second version that accepts a vector of rwlock_token_sets.
       The locking function returns a new rwlock_token_set with your
       new locks and/or references to the preexisting locks.

       a write lock on an array implies a write lock on
       all regions of the array. Likewise a read lock on
       an array implies a read lock on all regions of the
       array. This means that locking a sub region is really
       more like allowing you  to do  a partial unlock, unless
       you don't have the whole array locked when you lock the
       sub-region

       Note that lock upgrading is not permitted; if you
       have a read lock on any part of the array, you may
       not get a write lock on that part (or the whole array)

       Lock downgrading is possible, with the downgrade_to_read()
       method. This downgrades an entire rwlock_token_set from
       write access to read access. Note that no other references
       to the underlying locks should exist, or an
       std::invalid_argument exception will be thrown.
       a single downgrade_to_read() call downgrades all references
       to the referenced rwlock_token_set.


       concurrency and multithreading
       ------------------------------
       Initialization (calls to addarray()) is currently not
       thread safe; all initialization should occur from a single
       thread before calls from other threads can be made.

       Once initialization is complete, locks can be acquired
       from any thread. Be aware that the rwlock_token_set
       objects themselves are not thread safe. They should
       either be accessed from a single thread, or
       they can be delegated from one thread to another
       (with appropriate synchronization) at which point
       they can be used from the other thread. IN THE
       CURRENT VERSION, BECAUSE THE std::lock_guard USED
       IN rwlock_token IS NOT THREAD SAFE, YOU MUST
       RELEASE ALL OTHER rwlock_token_sets YOU HAVE
       BEFORE DELEGATING A rwlock_token_set TO ANOTHER
       THREAD! (this constraint may be removed in a
       later version). NOTE THAT TO RELEASE A
       rwlock_token_set it must either go out of scope
       or its .reset() method must be called.

       One example of such delegation is when
       data will be processed by a GPU and an arbitrary
       thread may do a callback.
    */


    /* Note that the existance of other data structures
       can implicitly define and maintains the existance of
       array regions e.g. the vertex region mentioned in a
       part structure indicates that those array elements
       exist. That tells you where to find the regions,
       but you still need to lock them to protect against
       e.g. a reallocation process that might be taking the
       data and moving it around to accommodate more space. */


    /* Thoughts:
       The default lock order is last to first, because later defined
       structures (e.g. parts) will generally be higher level...
       so as you traverse, you can lock the part database, figure out
       which vertices, lock those, and perhaps narrow or release your
       hold on the part database.
    */


    /* NOTE: All lockmanager initialization (defining the arrays)
       for a particular class must be done from a single thread,
       before others may
       do any locking */

    /* These next few elements may ONLY be modified during
       initialization phase (single thread, etc.) */
  public:
    std::vector<void **> _arrays; /* get array pointer from index */
    std::map<void **,size_t> _arrayidx; /* get array index from pointer */
    std::deque<arraylock> _locks; /* get lock from index */
    std::map<rwlock *,size_t> _idx_from_lockptr; /* get array index from lock pointer */


    //std::mutex admin; /* synchronizes access to lockmanager
    //			 data structures (arrays, arrayidx,locks).
    //			 this is locked BEFORE the adminlocks within the
    //			 locks array. -- not necessary because we specify
    //                   that all such initialization must be done from
    //                   one thread on startup */

    lockmanager() {

    }

    void addarray(void **array) {
      // array is pointer to pointer to array data, because
      // the pointer to pointer remains fixed even as the array itself may be reallocated
      /* WARNING: ALL addarray() CALLS MUST BE ON INITIALIZATION
	 FROM INITIALIZATION THREAD, BEFORE OTHER METHODS MAY
	 BE CALLED! */
      size_t idx;
      //std::lock_guard<std::mutex> lock(admin);

      assert(_arrayidx.count(array)==0);

      idx=_arrays.size();
      _arrays.push_back(array);

      _arrayidx[array]=idx;



      _locks.emplace_back();  /* Create arraylock object */

      _idx_from_lockptr[&_locks.back().full_array]=idx;

    }

    void set_array_size(void **Arrayptr,size_t elemsize,snde_index nelem) {
      // We don't currently care about the array size
    }



    rwlock_token  _get_lock_read_array(std::vector<rwlock_token_set> priors, size_t arrayidx)
    {
      // prior is like a rwlock_token_set **
      rwlock_lockable *lockobj=&_locks[arrayidx].full_array.reader;

      for (std::vector<rwlock_token_set>::iterator prior=priors.begin(); prior != priors.end(); prior++) {
	//

	if ((**prior).count(lockobj)) {
	  /* If we have this token */
	  /* return a reference */
	  return (**prior)[lockobj];
	}
      }

      /* if we got here, we do not have a token, need to make one */
      return rwlock_token(new std::lock_guard<rwlock_lockable>(*lockobj));

    }


    rwlock_token_set get_locks_read_array(std::vector<rwlock_token_set> priors, void **array)
    {
      rwlock_token_set token_set=std::make_shared<std::map<rwlock_lockable *,rwlock_token>>();;

      (*token_set)[&_locks[_arrayidx[array]].full_array.reader]=_get_lock_read_array(priors,_arrayidx[array]);

      return token_set;
    }

    rwlock_token_set get_locks_read_arrays(std::vector<rwlock_token_set> priors, std::vector<void **> arrays)
    {
      rwlock_token_set token_set=std::make_shared<std::map<rwlock_lockable *,rwlock_token>>();;

      std::vector<void **> sorted_arrays(arrays);

      std::sort(sorted_arrays.begin(), sorted_arrays.end(),
          [ this ] (void **a, void **b) { return _arrayidx[a] < _arrayidx[b]; });


      for (std::reverse_iterator<std::vector<void **>::iterator> array=sorted_arrays.rbegin(); array != sorted_arrays.rend(); array++) {
	(*token_set)[&_locks[_arrayidx[*array]].full_array.reader]=_get_lock_read_array(priors,_arrayidx[*array]);
      }
      return token_set;
    }


    rwlock_token_set get_locks_read_array_region(std::vector<rwlock_token_set> priors, void **array,size_t indexstart,size_t indexend)
    {
      // We do not currently implement region-granular locking
      return get_locks_read_array(priors,array);
    }

    rwlock_token_set get_locks_read_arrays_region(std::vector<rwlock_token_set> priors, std::vector<struct arrayregion> arrays)
    {
      // We do not currently implement region-granular locking
      std::vector<void **> arrayptrs(arrays.size());

      int cnt=0;
      for (std::vector<struct arrayregion>::iterator array=arrays.begin();array != arrays.end();array++,cnt++) {
	arrayptrs[cnt]=array->array;
      }

      return get_locks_read_arrays(priors,arrayptrs);
    }


    rwlock_token_set get_locks_read_arrays_region(rwlock_token_set prior, std::vector<struct arrayregion> arrays)
    {
      std::vector<rwlock_token_set> priors; //prior);
      priors.emplace_back(prior);

      return get_locks_read_arrays_region(priors,arrays);

    }

    rwlock_token_set get_locks_read_arrays_region(std::vector<struct arrayregion> arrays)
    {
      std::vector<rwlock_token_set> priors;

      return get_locks_read_arrays_region(priors,arrays);

    }




    rwlock_token_set get_locks_read_all(std::vector<rwlock_token_set> priors)
    {
      rwlock_token_set tokens=std::make_shared<std::map<rwlock_lockable *,rwlock_token>>();

      for (ssize_t cnt=_arrays.size()-1;cnt >= 0;cnt--) {
	(*tokens)[&_locks[cnt].full_array.reader]=_get_lock_read_array(priors,cnt);
      }
      return tokens;
    }

    rwlock_token_set get_locks_read_array(rwlock_token_set prior, void **array)
    {
      std::vector<rwlock_token_set> priors;
      priors.emplace_back(prior);

      return get_locks_read_array(priors,array);
    }

    rwlock_token_set get_locks_read_arrays(rwlock_token_set prior, std::vector<void **> arrays)
    {
      std::vector<rwlock_token_set> priors;
      priors.emplace_back(prior);

      return get_locks_read_arrays(priors,arrays);
    }


    rwlock_token_set get_locks_read_all(rwlock_token_set prior)
    {
      std::vector<rwlock_token_set> priors;
      priors.emplace_back(prior);

      return get_locks_read_all(priors);
    }

    rwlock_token_set get_locks_read_array(void **array)
    {
      return get_locks_read_array(std::vector<rwlock_token_set>(),array);
    }

    rwlock_token_set get_locks_read_all()
    {
      return get_locks_read_all(std::vector<rwlock_token_set>());
    }



    rwlock_token  _get_lock_write_array(std::vector<rwlock_token_set> priors, size_t arrayidx)
    {
      // prior is like a rwlock_token_set **
      rwlock_lockable *lockobj=&_locks[arrayidx].full_array.writer;

      for (std::vector<rwlock_token_set>::iterator prior=priors.begin(); prior != priors.end(); prior++) {
	//

	if ((**prior).count(lockobj)) {
	  /* If we have this token */
	  /* return a reference */
	  return (**prior)[lockobj];
	}
      }

      /* if we got here, we do not have a token, need to make one */
      return rwlock_token(new std::lock_guard<rwlock_lockable>(*lockobj));

    }

    rwlock_token_set get_locks_write_array(std::vector<rwlock_token_set> priors, void **array)
    {
      rwlock_token_set token_set=std::make_shared<std::map<rwlock_lockable *,rwlock_token>>();;

      (*token_set)[&_locks[_arrayidx[array]].full_array.writer]=_get_lock_write_array(priors,_arrayidx[array]);

      return token_set;
    }

    rwlock_token_set get_locks_write_arrays(std::vector<rwlock_token_set> priors, std::vector<void **> arrays)
    {
      rwlock_token_set token_set=std::make_shared<std::map<rwlock_lockable *,rwlock_token>>();;

      std::vector<void **> sorted_arrays(arrays);

      std::sort(sorted_arrays.begin(), sorted_arrays.end(),
          [ this ] (void **a, void **b) { return _arrayidx[a] < _arrayidx[b]; });


      for (std::reverse_iterator<std::vector<void **>::iterator> array=sorted_arrays.rbegin(); array != sorted_arrays.rend(); array++) {
	(*token_set)[&_locks[_arrayidx[*array]].full_array.writer]=_get_lock_write_array(priors,_arrayidx[*array]);
      }
      return token_set;
    }

    rwlock_token_set get_locks_write_array_region(std::vector<rwlock_token_set> priors, void **array,size_t indexstart,size_t indexend)
    {
      // We do not currently implement region-granular locking
      return get_locks_write_array(priors,array);
    }

    rwlock_token_set get_locks_write_arrays_region(std::vector<rwlock_token_set> priors, std::vector<struct arrayregion> arrays)
    {
      // We do not currently implement region-granular locking
      std::vector<void **> arrayptrs(arrays.size());
      int cnt=0;

      for (std::vector<struct arrayregion>::iterator array=arrays.begin();array != arrays.end();array++,cnt++) {
	arrayptrs[cnt]=array->array;
      }

      return get_locks_write_arrays(priors,arrayptrs);
    }


    rwlock_token_set get_locks_write_arrays_region(rwlock_token_set prior, std::vector<struct arrayregion> arrays)
    {
      std::vector<rwlock_token_set> priors;
      priors.emplace_back(prior);

      return get_locks_write_arrays_region(priors,arrays);

    }

    rwlock_token_set get_locks_write_arrays_region(std::vector<struct arrayregion> arrays)
    {
      std::vector<rwlock_token_set> priors;

      return get_locks_write_arrays_region(priors,arrays);

    }


    rwlock_token_set get_locks_write_all(std::vector<rwlock_token_set> priors)
    {
      rwlock_token_set tokens=std::make_shared<std::map<rwlock_lockable *,rwlock_token>>();

      for (ssize_t cnt=_arrays.size()-1;cnt >= 0;cnt--) {
	(*tokens)[&_locks[cnt].full_array.writer]=_get_lock_write_array(priors,cnt);
      }
      return tokens;
    }


    rwlock_token_set get_locks_write_array(rwlock_token_set prior, void **array)
    {
      std::vector<rwlock_token_set> priors;
      priors.emplace_back(prior);

      return get_locks_write_array(priors,array);
    }

    rwlock_token_set get_locks_write_arrays(rwlock_token_set prior, std::vector<void **> arrays)
    {
      std::vector<rwlock_token_set> priors;
      priors.emplace_back(prior);

      return get_locks_write_arrays(priors,arrays);
    }


    rwlock_token_set get_locks_write_all(rwlock_token_set prior)
    {
      std::vector<rwlock_token_set> priors;
      priors.emplace_back(prior);

      return get_locks_write_all(priors);
    }


    rwlock_token_set get_locks_write_array(void **array)
    {
      return get_locks_write_array(std::vector<rwlock_token_set>(),array);
    }


    rwlock_token_set get_locks_write_all()
    {
      return get_locks_write_all(std::vector<rwlock_token_set>());
    }

    void downgrade_to_read(rwlock_token_set locks) {
      /* locks within the token_set MUST NOT be referenced more than once */

      for (std::map<rwlock_lockable *,rwlock_token>::iterator lockable_token=locks->begin();lockable_token != locks->end();lockable_token++) {
	// lockable_token.first is reference to the lockable
	// lockable_token.second is reference to the token
	if (lockable_token->second.use_count() != 1) {
	  throw std::invalid_argument("Locks referenced by more than one token_set may not be downgraded");
	  lockable_token->first->_rwlock_obj->downgrade();
	}
      }

    }

  };



}


#endif /* SNDE_LOCKMANAGER */
