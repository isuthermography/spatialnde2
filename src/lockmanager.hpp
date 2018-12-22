#ifndef SNDE_LOCKMANAGER_HPP
#define SNDE_LOCKMANAGER_HPP

#include <cstdint>
#include <cassert>

#include <condition_variable>
#include <functional>
#include <deque>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <algorithm>
#include <map>
#include <thread> // Remember to add -pthread to the flags

#include "snde_error.hpp"
#include "geometry_types.h"
#include "lock_types.hpp"
#include "rangetracker.hpp"

/* general locking api information:

 * Locking can be done at various levels of granularity:  All arrays
(get_locks_..._all()), multiple arrays (get_locks_..._arrays()), individual arrays (get_locks_..._array()),
individual regions of individual arrays (get_locks_..._array_region),
or multiple regions of multiple arrays (get_locks_..._arrays_region).
(region-granularity is currently present in the API but an attempt to lock a region actually locks the entire array)
 * Can obtain either a read lock or a write lock. Multiple readers are
allowed simultaneously, but only one writer is possible at a time.
i.e. use get_locks_read_...() vs. get_locks_write_...()
 * Locks must be acquired in a specified order to avoid deadlock. See
https://stackoverflow.com/questions/1951275/would-you-explain-lock-ordering
The order is from the top of the struct snde_geometrydata  to the bottom. Once you own a lock for a particular array, you may not lock an array farther up.
 * Within an array, locks are ordered from smallest to largest index.
 * Allocating space in an array requires a write lock on the entire array, as it may cause the array to be reallocated if it must be expanded.
 * Ownership of a lock is denoted by an rwlock_token, but
   as a single locking operation may return multiple locks,
   the locking operations return an rwlock_token_set
 * When obtaining additional locks after you already own one,
   you must pass the preexisting locks to the locking function,
   as the "prior" or "priors" argument. The preexisting locks are unaffected
   (but it is OK to relock them -- the locks will nest)
 * the rwlock_token and rwlock_token_set are NOT thread safe.
   If you want to pass locks acquired in one thread to another
   thread you can create an rwlock_token_set from your locking
   operation, and then call lockmanager->clone_rwlock_token_set() to
   create an independent clone of the rwlock_token_set that can
   then be safely used by another thread.

 * Currently locks are only implemented to array granularity.
 * In the current implementation attempting to simulaneously read lock one part of an array
   and write lock another part of the same array may deadlock.
 * No useful API to do region-by-region locks of the arrays exists
   so far. Such an API would allow identifying all of the sub-regions
   of all arrays that correspond to the parts (objects) of interest.
   Problem is, the parts need to be locked in order and this ordered
   locking must proceed in parallel for all objects. This would be
   a mess. Suggested approach: Use "resumable functions" once they
   make it into the C++ standard. There are also various
   workarounds to implement closures/resumable functions,
   but they tend to be notationally messy,
   https://github.com/vmilea/CppAsync or require Boost++ (boost.fiber)

 * If we wanted to implement region-granular locking we could probably use the
   rangetracker class to identify and track locked regions.


 */

namespace snde {
  class lockholder_index; // forward declaration 
  
  typedef  std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>,std::allocator< std::pair< void **const,size_t > > > voidpp_size_map;
  typedef voidpp_size_map::iterator voidpp_size_map_iterator;
  
  class arraymanager; // forward declaration


    struct arrayregion {
    void **array;
    snde_index indexstart;
    snde_index numelems;
    
  };

  class markedregion  {
  public:
    snde_index regionstart;
    snde_index regionend;
    
    markedregion(snde_index regionstart,snde_index regionend)
    {
      this->regionstart=regionstart;
      this->regionend=regionend;
    }

    bool attempt_merge(markedregion &later)
    {
      assert(later.regionstart==regionend);
      regionend=later.regionend;
      return true;
    }
    std::shared_ptr<markedregion> sp_breakup(snde_index breakpoint)
    /* breakup method ends this region at breakpoint and returns
       a new region starting at from breakpoint to the prior end */
    {
      std::shared_ptr<markedregion> newregion=std::make_shared<markedregion>(breakpoint,regionend);
      regionend=breakpoint;

      return newregion;
    }
    markedregion breakup(snde_index breakpoint)
    /* breakup method ends this region at breakpoint and returns
       a new region starting at from breakpoint to the prior end */
    {
      markedregion newregion(breakpoint,regionend);
      regionend=breakpoint;

      return newregion;
    }

    bool operator<(const markedregion & other) const {
      if (regionstart < other.regionstart) return true;
      return false;

    }
  };


  //  class dirtyregion: public markedregion  {
  //  public:
  //    cachemanager *cache_with_valid_data; /* Do not dereference this pointer... NULL means the main CPU store is the one with the valid data */
  //    cl_event FlushDoneEvent;
  //    bool FlushDoneEventComplete;
  //    dirtyregion(cachemanager *cache_with_valid_data,snde_index regionstart, snde_index regionend) : markedregion(regionstart,regionend)
  //    {
  //      this->cache_with_valid_data=cache_with_valid_data;
  //    }
  //  };

  class arraylock {
  public:
    std::mutex admin; /* locks access to subregions, lock after everything; see also whole_array_write */

    //rwlock full_array;
    //std::vector<rwlock> subregions;

    std::shared_ptr<rwlock> wholearray; /* This is in the locking order with the arrays. In
				 order to modify subregions you must hold this AND all subregions AND admin (above) 
				 for write... Note: Not used for dirty tracking (put dirty stuff in subregions!) */
    std::map<markedregion,std::shared_ptr<rwlock>> subregions;
    
    arraylock() {
      wholearray=std::make_shared<rwlock>();
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
       Objects themselves are not thread safe. They should
       either be accessed from a single thread, or
       they can be delegated from one thread to another
       (with appropriate synchronization) at which point
       they can be used from the other thread. IN THE
       CURRENT VERSION, BECAUSE THE std::unique_lock USED
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


    
    /* Synchronization model for __arrays, __arrayidx, 
       and __locks: Atomic shared pointer for 
       the content for reading. To change the content, lock the 
       admin mutex, make a complete copy, then 
       switch the atomic pointer. 

       non-atomic shared pointer copy retrieved by the allocators(), 
       allocation_arrays(), and arrays_managed_by_allocator() methods
    */
    /* These next few elements may ONLY be modified during
       initialization phase (single thread, etc.) */
  public:
    std::mutex admin; // Should ONLY be held when rearranging __arrays/__arrayidx/__locks

    /* DO NOT ACCESS THESE ARRAYS DIRECTLY... ALWAYS USE THE ACCESSORS FOR READ OR THE
       _begin_atomic_update()/_end_atomic_update() FOR WRITE */
    // Note: locking index and locking order position is defined based
    // on index into the _arrays() vector and _locks() deque. Note that entries MAY NOT
    // BE REMOVED because that would change succeeding indices,
    // but they may in the future support disabling, and entries may be added
    // (using _begin_atomic_update(), etc.)
    std::shared_ptr<std::vector<void **>> __arrays; /* atomic shared pointer to get array pointer from index */
    //std::unordered_map<void **,size_t> _arrayidx; /* get array index from pointer */
    std::shared_ptr<std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>>> __arrayidx; /* atomic shared pointer to get array index from pointer */

    std::shared_ptr<std::deque<std::shared_ptr<arraylock>>> __locks; /* atomic shared pointer to get lock from index */
    //std::unordered_map<rwlock *,size_t> _idx_from_lockptr; /* get array index from lock pointer */

    // basic lockmanager data structures (arrays, arrayidx,locks).
    //			 do not require locking for read because they are 
    //                   immutable (replaced when changed)
    
    // ... But _locks[...].subregions is mutable
    // and is locked with locks[...].mutex  mutex.
    
    lockmanager() {
      std::atomic_store(&__arrays,std::make_shared<std::vector<void **>>());
      std::atomic_store(&__arrayidx,std::make_shared<std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>>>());;
      std::atomic_store(&__locks,std::make_shared<std::deque<std::shared_ptr<arraylock>>>());
    }

    /* Accessors for atomic shared pointers */
    std::shared_ptr<std::vector<void **>> _arrays()
    {
      /* get array pointer from index */    
      return std::atomic_load(&__arrays);
    }
    std::shared_ptr<std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>>> _arrayidx()
    {
      /* get array index from pointer */    
      return std::atomic_load(&__arrayidx);
    }
    
    std::shared_ptr<std::deque<std::shared_ptr<arraylock>>> _locks()
    {
      /* Get lock from index */
      return std::atomic_load(&__locks);
    }

    std::tuple<std::shared_ptr<std::vector<void **>>,
	       std::shared_ptr<std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>>>,
	       std::shared_ptr<std::deque<std::shared_ptr<arraylock>>>> _begin_atomic_update()
    // adminlock must be locked when calling this function...
    // it returns new copies of the atomically-guarded data
    {

      // Make copies of atomically-guarded data 
      std::shared_ptr<std::vector<void **>> new__arrays=std::make_shared<std::vector<void **>>(*_arrays());
      std::shared_ptr<std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>>> new__arrayidx=std::make_shared<std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>>>(*_arrayidx());
      std::shared_ptr<std::deque<std::shared_ptr<arraylock>>> new__locks=std::make_shared<std::deque<std::shared_ptr<arraylock>>>(*_locks());      
      
      return std::make_tuple(new__arrays,new__arrayidx,new__locks);
    }

    void _end_atomic_update(std::shared_ptr<std::vector<void **>> new__arrays,
			     std::shared_ptr<std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>>> new__arrayidx,
			     std::shared_ptr<std::deque<std::shared_ptr<arraylock>>> new__locks)
    // adminlock must be locked when calling this function...
    {

      // replace old with new

      std::atomic_store(&__arrays,new__arrays);
      std::atomic_store(&__arrayidx,new__arrayidx);
      std::atomic_store(&__locks,new__locks);
      
    }

    size_t get_array_idx(void **array)
    {
      auto arrayidx = _arrayidx();
      assert(arrayidx->find(array) != arrayidx->end());
      return (*arrayidx)[array];
    }

    void addarray(void **array) {
      // array is pointer to pointer to array data, because
      // the pointer to pointer remains fixed even as the array itself may be reallocated
      size_t idx;
      std::lock_guard<std::mutex> lock(admin);

      // Make copies of atomically-guarded data 
      std::shared_ptr<std::vector<void **>> new__arrays;
      std::shared_ptr<std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>>> new__arrayidx;
      std::shared_ptr<std::deque<std::shared_ptr<arraylock>>> new__locks;

      std::tie(new__arrays,new__arrayidx,new__locks) = _begin_atomic_update();
      
      assert(new__arrayidx->count(array)==0);
      
      idx=new__arrays->size();
      new__arrays->push_back(array);

      (*new__arrayidx)[array]=idx;
      

      
      new__locks->emplace_back(std::make_shared<arraylock>());  /* Create arraylock object */

      //_idx_from_lockptr[&_locks.back().full_array]=idx;
      // replace old with new
      _end_atomic_update(new__arrays,new__arrayidx,new__locks);


    }

    bool is_region_granular(void) /* Return whether locking is really granular on a region-by-region basis (true) or just on an array-by-array basis (false) */
    {
      return true;
    }

    void set_array_size(void **Arrayptr,size_t elemsize,snde_index nelem) {
      // We don't currently care about the array size
    }

    //void writer_mark_as_dirty(cachemanager *cache,void **arrayptr,snde_index pos,snde_index size)
    //{
    //  size_t arrayidx=_arrayidx.at(array);
    //
    //
    //  std::unique_lock<std::mutex> arrayadminlock(_locks[arrayidx].admin);
    //
    //  std::map<markedregion,rwlock>::iterator iter=manager->locker->_locks[arrayidx].subregions.lower_bound(pos);

    // if (pos < iter.first.regionstart) { /* probably won't happen due to array layout process, but just in case */
    //assert(iter != _locks[arrayidx].subregions.begin());
    //	iter--;
    //  }
    //
    //// iterate over the subregions of this arraylock
    //  for (;iter != manager->locker->_locks[arrayidx].subregions.end() && iter->first.regionstart < writeregion.second->regionend;iter++) {
    //snde_index regionstart=dirtyregion->regionstart;
    //snde_index regionend=dirtyregion->regionend;
    //
    //
    //if (iter->first.regionstart > regionstart) {
    //regionstart=iter->first.regionstart;
    //}
    //if (regionend > iter->first.regionend) {
    //regionend=iter->first.regionend;
    //}
    //
    //iter->second.writer_mark_as_dirty(this,regionstart,regionend-regionstart);
    //
    //}
    
    //}
    
    rwlock_token newallocation(rwlock_token_set all_locks,void **arrayptr,snde_index pos,snde_index size,snde_index elemsize)
    {
      /* callback from allocator */
      /* returns locked token */
      /* Entire array should be write locked in order to call this */
      size_t arrayidx=_arrayidx()->at(arrayptr);

      std::shared_ptr<arraylock> thislock=_locks()->at(arrayidx);
      std::unique_lock<std::mutex> lock(thislock->admin); // lock the subregions field

      
      
      
      // Notification from allocator
      markedregion region(pos,pos+size);
      
      assert(thislock->subregions.count(region)==0);

      // add new rwlock to subregion
      thislock->subregions.emplace(std::piecewise_construct,
				   std::forward_as_tuple(pos,pos+size),
				   std::forward_as_tuple());

      thislock->subregions.at(region)=std::make_shared<rwlock>();
      std::shared_ptr<rwlock> rwlockobj=thislock->subregions.at(region);
      
      rwlock_token retval(new std::unique_lock<rwlock_lockable>(rwlockobj->writer));
      (*all_locks)[&rwlockobj->writer]=retval;
      
      return retval;
    }
    
    void realloc_down_allocation(void **arrayptr, snde_index addr,snde_index orignelem, snde_index newnelem)
    {
      /* callback from allocator */
      /* callback from allocator */
      size_t idx=_arrayidx()->at(arrayptr);
      std::shared_ptr<arraylock> thislock=_locks()->at(idx);
      std::lock_guard<std::mutex> lock(thislock->admin);

      // notification from allocator that an allocation has shrunk

      // Find pointer for lock for this region
      std::shared_ptr<rwlock> lock_ptr = thislock->subregions.at(markedregion(addr,addr+orignelem));
      
      // Remove this pointer from the subregions map
      thislock->subregions.erase(markedregion(addr,addr+orignelem));

      // Reinsert it with the new size
      thislock->subregions.emplace(std::make_pair(markedregion(addr,addr+newnelem),lock_ptr));
      
    }
    
    void freeallocation(void **arrayptr,snde_index pos, snde_index size,snde_index elemsize)
    {
      /* callback from allocator */
      size_t idx=_arrayidx()->at(arrayptr);
      std::shared_ptr<arraylock> thislock=_locks()->at(idx);
      std::lock_guard<std::mutex> lock(thislock->admin);


      // notification from allocator
      thislock->subregions.erase(markedregion(pos,pos+size));
      
    }


    rwlock_token  _get_preexisting_lock_read_array_lockobj(rwlock_token_set all_locks, size_t arrayidx,std::shared_ptr<rwlock> rwlockobj)
    /* returns NULL if there is no such preexisting read lock or
       there is no preexisting write lock that is convertable to a read lock */
    {      // must hold write lock on entire array... returns write lock on new allocation
      // and position

      // prior is like a rwlock_token_set **
      
      rwlock_lockable *lockobj=&rwlockobj->reader;
      rwlock_lockable *writelockobj=&rwlockobj->writer;
      rwlock_token writelocktoken;

	
      if ((*all_locks).count(lockobj)) {
	/* If we have this token */
	/* return a reference */
	return (*all_locks)[lockobj];
      }
      if ((*all_locks).count(writelockobj)) {
	/* There is a write lock for this token */
	writelocktoken=(*all_locks)[writelockobj];
      }
      
      /* if we got here, we do not have a token, need to make one */
      if (writelocktoken) {
	/* There is a write token, but not a read token */
	rwlockobj->sidegrade(); /* add read capability to write lock */
        /* this capability is important to avoid deadlocking
           if a single owner locks one subregion for read and
           another subregion for write, then so long as the write
           lock was done first, it will not deadlock. Unfortunately
           this doesn't cover all situations because the locking order
           specifies that earlier blocks should be allocated first,
           and the earlier block may not be the write block. */
	/* now create a new reader token that adopts the read capability
	   we just added */
	rwlock_token retval=std::make_shared<std::unique_lock<rwlock_lockable>>(*lockobj,std::adopt_lock);
	(*all_locks)[lockobj]=retval;
	
	return retval;
      }
      
      return std::shared_ptr<std::unique_lock<rwlock_lockable>>();

    }

    
    std::pair<rwlock_lockable *,rwlock_token>  _get_preexisting_lock_read_array_region(rwlock_token_set all_locks, size_t arrayidx,snde_index pos,snde_index size)
    /* returns NULL if there is no such preexisting read lock or
       there is no preexisting write lock that is convertable to a read lock */
    {
      // prior is like a rwlock_token_set **
      std::shared_ptr<arraylock> thislock=_locks()->at(arrayidx);
      std::unique_lock<std::mutex> lock(thislock->admin); // lock the subregions field
      
      std::shared_ptr<rwlock> rwlockobj = thislock->subregions.at(markedregion(pos,pos+size));
      lock.unlock();
      return std::make_pair(&rwlockobj->reader,_get_preexisting_lock_read_array_lockobj(all_locks, arrayidx,rwlockobj));
    }
    


    std::pair<rwlock_lockable *,rwlock_token>  _get_lock_read_array_region(rwlock_token_set all_locks, size_t arrayidx,snde_index pos,snde_index size)
    {

      rwlock_token retval;
      rwlock_lockable *lockobj;
      std::tie(lockobj,retval)=_get_preexisting_lock_read_array_region(all_locks,arrayidx,pos,size);
      
      if (retval == nullptr) {
	
	retval = rwlock_token(new std::unique_lock<rwlock_lockable>(*lockobj));
	(*all_locks)[lockobj]=retval;
      }
      return std::make_pair(lockobj,retval);
    }
    

    rwlock_token_set get_locks_read_array(rwlock_token_set all_locks, void **array)
    {
      rwlock_token_set token_set=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
      
      size_t idx=_arrayidx()->at(array);
      std::shared_ptr<arraylock> alock=_locks()->at(idx);


      // First, get wholearray lock
      
      rwlock_token wholearray_token = _get_preexisting_lock_read_array_lockobj(all_locks,idx,alock->wholearray);
      if (wholearray_token==nullptr) {
	wholearray_token = rwlock_token(new std::unique_lock<rwlock_lockable>(alock->wholearray->reader));
	(*all_locks)[&alock->wholearray->reader]=wholearray_token;
      }
      (*token_set)[&alock->wholearray->reader]=wholearray_token;
            
      /* now that we have wholearray_lock, nobody else can do allocations, etc. that may modify subregions, 
	 so we can safely iterate over it without holding the admin lock */
      
      for (auto & markedregion_rwlock : alock->subregions) {
	rwlock_lockable *lockableptr;
	rwlock_token token;
	std::tie(lockableptr,token)=_get_lock_read_array_region(all_locks,idx,markedregion_rwlock.first.regionstart,markedregion_rwlock.first.regionend-markedregion_rwlock.first.regionstart);
	(*token_set)[lockableptr]=token;
      }
      return token_set;
    }

    rwlock_token_set get_preexisting_locks_read_array(rwlock_token_set all_locks, void **array)
    {
      rwlock_token_set token_set=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();

      //assert(_arrayidx.find(array) != _arrayidx.end());
      size_t idx=_arrayidx()->at(array);
      std::shared_ptr<arraylock> alock=_locks()->at(idx);

      // First, get wholearray lock      
      rwlock_token wholearray_token = _get_preexisting_lock_read_array_lockobj(all_locks,idx,alock->wholearray);
      if (wholearray_token==nullptr) {
	throw std::invalid_argument("Must have valid preexisting wholearray lock (this may be a locking order violation)");
      }

      (*token_set)[&alock->wholearray->reader]=wholearray_token;
      
      /* now that we have wholearray_lock, nobody else can do allocations, etc. that may modify subregions, 
	 so we can safely iterate over it without holding the admin lock */
      
      for (auto & markedregion_rwlock : alock->subregions) {	
      
	rwlock_lockable *lockableptr;
	rwlock_token preexisting_lock;
	std::tie(lockableptr,preexisting_lock)=_get_preexisting_lock_read_array_region(all_locks,idx,markedregion_rwlock.first.regionstart,markedregion_rwlock.first.regionend-markedregion_rwlock.first.regionstart);
	
	if (preexisting_lock==nullptr) {
	  throw std::invalid_argument("Must have valid preexisting lock (this may be a locking order violation)");
	}
	//(*token_set)[&markedregion_rwlock.second.reader]=preexisting_lock;
	(*token_set)[lockableptr]=preexisting_lock;

      }
      return token_set;
    }


    
    rwlock_token_set get_preexisting_locks_read_array_region(rwlock_token_set all_locks, void **array,snde_index indexstart,snde_index numelems)
    {
      rwlock_lockable *lockobj;
      rwlock_token retval;

      rwlock_token_set token_set=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();

      if (indexstart==0 && numelems==SNDE_INDEX_INVALID) {
	return get_preexisting_locks_read_array(all_locks,array);
      }

      size_t arrayidx=_arrayidx()->at(array);

      std::tie(lockobj,retval) =  _get_preexisting_lock_read_array_region(all_locks, arrayidx,indexstart,numelems);

      if (retval==nullptr) {
	throw std::invalid_argument("Must have valid preexisting lock (this may be a locking order violation)");
      }
      (*token_set)[lockobj]=retval;
	
      return token_set;
    }

    rwlock_token_set get_locks_read_array_region(rwlock_token_set all_locks, void **array,snde_index indexstart,snde_index numelems)
    {
      rwlock_lockable *lockobj;

      rwlock_token retval;

      if (indexstart==0 && numelems==SNDE_INDEX_INVALID) {
	return get_locks_read_array(all_locks,array);
      }

      
      rwlock_token_set token_set=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
      size_t arrayidx=_arrayidx()->at(array);

      std::tie(lockobj,retval) =  _get_lock_read_array_region(all_locks, arrayidx,indexstart,numelems);

      (*token_set)[lockobj]=retval;
      return token_set;
    }



    rwlock_token_set get_locks_read_all(rwlock_token_set all_locks)
    {
      rwlock_token_set tokens=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();

      for (size_t cnt=0;cnt < _arrays()->size();cnt++) {
	rwlock_token_set thisset = get_locks_read_array(all_locks,(*_arrays())[cnt]);
	merge_into_rwlock_token_set(tokens,thisset);
      }
      return tokens;
    }
    
    rwlock_token  _get_preexisting_lock_write_array_lockobj(rwlock_token_set all_locks, size_t arrayidx,std::shared_ptr<rwlock> rwlockobj)
    {
      // but we do store the bounds for notification purposes (NOT ANYMORE; dirty marking is now explicit)

      // prior is like a rwlock_token_set **
      rwlock_token token;
      
      rwlock_lockable *lockobj=&rwlockobj->writer;

      if ((*all_locks).count(lockobj)) {	  /* If we have this token */
      	  /* return a reference */
	  //(**prior)[lockobj]

	return (*all_locks)[lockobj];
      }
      
      return std::shared_ptr<std::unique_lock<rwlock_lockable>>(); /* return nullptr if there is no preexisting lock */
    }
    

    std::pair<rwlock_lockable *,rwlock_token>  _get_preexisting_lock_write_array_region(rwlock_token_set all_locks, size_t arrayidx,snde_index indexstart,snde_index numelems)
    {
      // We currently implement region-granular locking
      // but we do store the bounds for notification purposes (NOT ANYMORE; dirty marking is now explicit)

      // prior is like a rwlock_token_set **
      std::shared_ptr<arraylock> thislock=_locks()->at(arrayidx);
      std::unique_lock<std::mutex> lock(thislock->admin); // lock the subregions field

      std::shared_ptr<rwlock> rwlockobj = thislock->subregions.at(markedregion(indexstart,indexstart+numelems));
      lock.unlock();
      return std::make_pair(&rwlockobj->writer,_get_preexisting_lock_write_array_lockobj(all_locks, arrayidx,rwlockobj));
      
    }
    
    std::pair<rwlock_lockable *,rwlock_token>  _get_lock_write_array_region(rwlock_token_set all_locks, size_t arrayidx,snde_index pos,snde_index size)
    {
      // but we do store the bounds for notification purposes (NOT ANYMORE; dirty marking is now explicit)

      rwlock_token retval;
      rwlock_lockable *lockobj;


      std::tie(lockobj,retval)=_get_preexisting_lock_write_array_region(all_locks,arrayidx,pos,size);
      if (retval == nullptr) {
	
	/* if we got here, we do not have a token, need to make one */
	retval = std::make_shared<std::unique_lock<rwlock_lockable>>(*lockobj);
	(*all_locks)[lockobj]=retval;
	

      }
      // Dirty marking now must be done explicitly
      //lockobj->_rwlock_obj->writer_append_region(indexstart,numelems);
      return std::make_pair(lockobj,retval);
      
    }

    
    rwlock_token_set get_locks_write_array(rwlock_token_set all_locks, void **array) {
      rwlock_token_set token_set=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
      
      size_t idx=_arrayidx()->at(array);
      std::shared_ptr<arraylock> alock=_locks()->at(idx);
      
      // First, get wholearray lock
      
      rwlock_token wholearray_token = _get_preexisting_lock_write_array_lockobj(all_locks,idx,alock->wholearray);
      if (wholearray_token==nullptr) {
	wholearray_token = rwlock_token(new std::unique_lock<rwlock_lockable>(alock->wholearray->writer));
	(*all_locks)[&alock->wholearray->writer]=wholearray_token;
      }
      (*token_set)[&alock->wholearray->writer]=wholearray_token;
      
      /* now that we have wholearray_lock, nobody else can do allocations, etc. that may modify subregions, 
	 so we can safely iterate over it without holding the admin lock */
      
      for (auto & markedregion_rwlock : alock->subregions) {
	
	//(*token_set)[&markedregion_rwlock.second.writer]
	rwlock_lockable *lockableptr;
	rwlock_token token;
	std::tie(lockableptr,token)=_get_lock_write_array_region(all_locks,idx,markedregion_rwlock.first.regionstart,markedregion_rwlock.first.regionend-markedregion_rwlock.first.regionstart);
	(*token_set)[lockableptr]=token;
      }
      return token_set;
    }

    rwlock_token_set get_preexisting_locks_write_array(rwlock_token_set all_locks, void **array) {
      rwlock_token_set token_set=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();

      //assert(_arrayidx.find(array) != _arrayidx.end());
      size_t idx=_arrayidx()->at(array);
      std::shared_ptr<arraylock> alock=_locks()->at(idx);

      // First, get wholearray lock      
      rwlock_token wholearray_token = _get_preexisting_lock_write_array_lockobj(all_locks,idx,alock->wholearray);
      if (wholearray_token==nullptr) {
	throw std::invalid_argument("Must have valid preexisting wholearray lock (this may be a locking order violation)");
      }

      (*token_set)[&alock->wholearray->writer]=wholearray_token;
      
      /* now that we have wholearray_lock, nobody else can do allocations, etc. that may modify subregions, 
	 so we can safely iterate over it without holding the admin lock */
      
      for (auto & markedregion_rwlock : alock->subregions) {	
	rwlock_lockable *lockableptr;
	rwlock_token preexisting_lock;
      
	std::tie(lockableptr,preexisting_lock)=_get_preexisting_lock_write_array_region(all_locks,idx,markedregion_rwlock.first.regionstart,markedregion_rwlock.first.regionend-markedregion_rwlock.first.regionstart);
	
	if (preexisting_lock==nullptr) {
	  throw std::invalid_argument("Must have valid preexisting lock (this may be a locking order violation)");
	}
	(*token_set)[&markedregion_rwlock.second->writer]=preexisting_lock;
      }
      return token_set;
    }

    rwlock_token_set get_preexisting_locks_write_array_region(rwlock_token_set all_locks, void **array,snde_index indexstart,snde_index numelems)
    {
      rwlock_lockable *lockobj;
      rwlock_token retval;

      if (indexstart==0 && numelems==SNDE_INDEX_INVALID) {
	return get_preexisting_locks_write_array(all_locks,array);
      }

      rwlock_token_set token_set=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
      size_t arrayidx=_arrayidx()->at(array);

      std::tie(lockobj,retval) =  _get_preexisting_lock_write_array_region(all_locks, arrayidx,indexstart,numelems);

      if (retval==nullptr) {
	throw std::invalid_argument("Must have valid preexisting lock (this may be a locking order violation)");
      }
      (*token_set)[lockobj]=retval;
	
      return token_set;

    }

    rwlock_token_set get_locks_write_array_region(rwlock_token_set all_locks, void **array,snde_index indexstart,snde_index numelems)
    {
      rwlock_lockable *lockobj;

      rwlock_token retval;

      if (indexstart==0 && numelems==SNDE_INDEX_INVALID) {
	return get_locks_write_array(all_locks,array);
      }
      
      rwlock_token_set token_set=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
      size_t arrayidx=_arrayidx()->at(array);

      std::tie(lockobj,retval) =  _get_lock_write_array_region(all_locks, arrayidx,indexstart,numelems);

      (*token_set)[lockobj]=retval;
      return token_set;


    }




    rwlock_token_set get_locks_write_all(rwlock_token_set all_locks)
    {
      rwlock_token_set tokens=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();

      for (size_t cnt=0;cnt < _arrays()->size();cnt++) {
	rwlock_token_set thisset = get_locks_write_array(all_locks,(*_arrays())[cnt]);
	merge_into_rwlock_token_set(tokens,thisset);
      }
      return tokens;
    }



    void downgrade_to_read(rwlock_token_set locks) {
      /* locks within the token_set MUST NOT be referenced more than once.... That means you must
	 have released your all_locks rwlock_token_set object*/

      for (std::unordered_map<rwlock_lockable *,rwlock_token>::iterator lockable_token=locks->begin();lockable_token != locks->end();lockable_token++) {
	// lockable_token.first is reference to the lockable
	// lockable_token.second is reference to the token
	if (lockable_token->second.use_count() != 1) {
	  throw std::invalid_argument("Locks referenced by more than one token_set may not be downgraded");
	  lockable_token->first->_rwlock_obj->downgrade();
	}
      }

    }

  };
  

  class lockingprocess {
      /* lockingprocess is a tool for performing multiple locking
         for multiple objects while satisfying the required
         locking order */

      /* (There was going to be an opencl_lockingprocess that was to be derived
         from this class, but it was cancelled) */
  public:
    //lockingprocess();
    //lockingprocess(const lockingprocess &)=delete; /* copy constructor disabled */
    //lockingprocess& operator=(const lockingprocess &)=delete; /* copy assignment disabled */

    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_write_array(void **array)=0;
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_write_array_region(void **array,snde_index indexstart,snde_index numelems)=0;
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_read_array(void **array)=0;
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_read_array_region(void **array,snde_index indexstart,snde_index numelems)=0;
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_array_region(void **array,bool write,snde_index indexstart,snde_index numelems)=0;
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_array(void **array,bool write)=0;
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_array_mask(void **array,uint64_t maskentry,uint64_t resizemaskentry,uint64_t readmask,uint64_t writemask,uint64_t resizemask,snde_index indexstart,snde_index numelems)=0;

    virtual std::vector<std::tuple<lockholder_index,rwlock_token_set,std::string>> alloc_array_region(std::shared_ptr<arraymanager> manager,void **allocatedptr,snde_index nelem,std::string allocid)=0;
    virtual void spawn(std::function<void(void)> f)=0;

    virtual ~lockingprocess()=0;

    
  };



    class lockingposition {
    public:
      size_t arrayidx; /* index of array we want to lock, or numeric_limits<size_t>.max() */
      snde_index idx_in_array; /* index within array, or SNDE_INDEX_INVALID*/
      bool write; /* are we trying to lock for write? */

      lockingposition()
      {
	arrayidx=0;
	idx_in_array=0;
	write=0;
      }
      
      lockingposition(size_t arrayidx,snde_index idx_in_array,bool write)
      {
        this->arrayidx=arrayidx;
        this->idx_in_array=idx_in_array;
        this->write=write;
      }

      bool operator<(const lockingposition & other) const {
        if (arrayidx < other.arrayidx) return true;
        if (arrayidx > other.arrayidx) return false;

        if (idx_in_array < other.idx_in_array) return true;
        if (idx_in_array > other.idx_in_array) return false;

        if (write && !other.write) return true;
        if (!write && other.write) return false;

        /* if we got here, everything is equal, i.e. not less than */
        return false;
      }
    };



#ifdef SNDE_LOCKMANAGER_COROUTINES_THREADED

  /* ***!!! Should create alternate implementation based on boost stackful coroutines ***!!! */
  /* ***!!! Should create alternate implementation based on C++ resumable functions proposal  */

  class lockingprocess_threaded: public lockingprocess {
    /* lockingprocess is a tool for performing multiple locking
       for multiple objects while satisfying the required
       locking order */
    
      /* (There was going to be an opencl_lockingprocess that was to be derived
         from this class, but it was cancelled) */
 
    lockingprocess_threaded(const lockingprocess_threaded &)=delete; /* copy constructor disabled */
    lockingprocess_threaded& operator=(const lockingprocess_threaded &)=delete; /* copy assignment disabled */
    
  public:
    //std::shared_ptr<arraymanager> _manager;
    std::shared_ptr<lockmanager> _lockmanager;
    std::mutex _mutex;
    //std::condition_variable _cv;
    std::multimap<lockingposition,std::condition_variable *> _waitingthreads; // locked by _mutex
    std::deque<std::condition_variable *> _runnablethreads; // locked by _mutex.... first entry is the running thread, which is listed as NULL (no cv needed because it is running).
    
    std::deque<std::thread *> _threadarray; // locked by _mutex

    /* ****!!!!!! Need to separate all_tokens into all_tokens and used_tokens, as in 
       the Python implementation in lockmanager.i. Also allow re-locking of stuff in all_tokens */
    rwlock_token_set all_tokens; /* these are all the tokens we have acquired along the way */
    rwlock_token_set used_tokens; /* these are all the tokens we are actually returning */
    
    std::shared_ptr<std::vector<rangetracker<markedregion>>> arrayreadregions; /* indexed by arrayidx */
    std::shared_ptr<std::vector<rangetracker<markedregion>>> arraywriteregions; /* indexed by arrayidx */

    
    lockingposition lastlockingposition; /* for diagnosing locking order violations */

    std::unique_lock<std::mutex> _executor_lock;
    
      /* The way this works is we spawn off parallel threads for
         each locking task. The individual threads can execute up
         until a lock attempt. All threads must have reached the
         lock attempt (or be finished) before any locking can
         occur, and then only the thread seeking the earliest lock
         (earliest in the locking order) may execute. That thread
         can then execute up until its next lock attempt and
         the process repeats.

         The mapping between locking positions and threads
         is stored in a std::multimap _waitingthreads (locked by _mutex).
         Each thread involved always either has an entry in _waitingthreads
         or is counted in _runnablethreads.

         To avoid synchronization hassles, only one thread can
         actually run at a time (locked by _mutex and managed by
         _executor_lock when running user code)

      */



    lockingprocess_threaded(std::shared_ptr<lockmanager> lockmanager);
    
    virtual bool _barrier(lockingposition lockpos); //(size_t arrayidx,snde_index pos,bool write);

    virtual void *pre_callback();
    
    virtual void post_callback(void *state);

    virtual void *prelock();
    
    virtual void postunlock(void *prelockstate);

    virtual std::pair<lockholder_index,rwlock_token_set>  get_locks_write_array(void **array);

    virtual std::pair<lockholder_index,rwlock_token_set>  get_locks_write_array_region(void **array,snde_index indexstart,snde_index numelems);


    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_read_array(void **array);

    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_read_array_region(void **array,snde_index indexstart,snde_index numelems);

    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_array_region(void **array,bool write,snde_index indexstart,snde_index numelems);

    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_array(void **array,bool write);
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_array_mask(void **array,uint64_t maskentry,uint64_t resizemaskentry,uint64_t readmask,uint64_t writemask,uint64_t resizemask,snde_index indexstart,snde_index numelems);

    virtual std::vector<std::tuple<lockholder_index,rwlock_token_set,std::string>> alloc_array_region(std::shared_ptr<arraymanager> manager,void **allocatedptr,snde_index nelem,std::string allocid);

    virtual void spawn(std::function<void(void)> f);
      

    virtual rwlock_token_set finish();
    virtual ~lockingprocess_threaded();

    };


#endif

}




namespace snde {

  class lockholder_index {
  public:
    void **array;
    bool write;
    snde_index startidx;
    snde_index numelem;

    lockholder_index() :
      array(NULL), write(false), startidx(SNDE_INDEX_INVALID), numelem(SNDE_INDEX_INVALID)
    {

    }
    
    lockholder_index(void **_array,bool _write, snde_index _startidx,snde_index _numelem) :
      array(_array), write(_write), startidx(_startidx), numelem(_numelem)
    {

    }

    // equality operator for std::unordered_map
    bool operator==(const lockholder_index b) const
    {
      return b.array==array && b.write==write && b.startidx==startidx && b.numelem==numelem;
    }
  };

  
}

/* provide hash implementation for indices used for lockholder */
namespace std {
  template <> struct hash<snde::lockholder_index>
  {
    size_t operator()(const snde::lockholder_index & x) const
    {

      return std::hash<void *>{}((void *)x.array) + std::hash<bool>{}(x.write) + std::hash<snde_index>{}(x.startidx) + std::hash<snde_index>{}(x.numelem);
    }
  };
}


namespace std {
  template <> struct hash<std::pair<void **,std::string>>
  {
    size_t operator()(const std::pair<void **,std::string> & x) const
    {
      void **param;
      std::string allocid;
      
      std::tie(param,allocid)=x;

      return std::hash<void *>{}((void *)param) + std::hash<std::string>{}(allocid);
    }
  };
}


namespace snde{ 
  class lockholder {
  public:
    std::unordered_map<lockholder_index,rwlock_token_set> values;
    std::unordered_map<std::pair<void **,std::string>,snde_index> allocvalues;

    std::string as_string() {
      std::string ret="";

      ret+=ssprintf("snde::lockholder at 0x%lx with  %d allocations and %d locks\n",(unsigned long)this,(int)allocvalues.size(),(int)values.size());
      ret+=ssprintf("---------------------------------------------------------------------------\n");
      for (auto & array_startidx : allocvalues) {
	ret+=ssprintf("allocation for array 0x%lx allocid=\"%s\" startidx=%lu\n",(unsigned long)array_startidx.first.first,array_startidx.first.second.c_str(),(unsigned long)array_startidx.second);
      }
      for (auto & idx_tokens : values) {
	ret+=ssprintf("locks for array 0x%lx write=%s startidx=%lu numelem=%lu\n",(unsigned long)idx_tokens.first.array,idx_tokens.first.write ? "true": "false",(unsigned long)idx_tokens.first.startidx,(unsigned long)idx_tokens.first.numelem);
      }
      return ret;
    }
    
    bool has_lock(void **array,bool write,snde_index indexstart,snde_index numelem)
    {
      return !(values.find(lockholder_index(array,write,indexstart,numelem))==values.end());
    }
    
    bool has_alloc(void **array,std::string allocid)
    {
      return !(allocvalues.find(std::make_pair(array,allocid))==allocvalues.end());
    }
    
    void store(void **array,bool write,snde_index indexstart,snde_index numelem,rwlock_token_set locktoken)
    {
      values[lockholder_index(array,write,indexstart,numelem)]=locktoken;
    }
    void store(lockholder_index array_write_startidx_numelem_tokens,rwlock_token_set locktoken)
    {
      
      values[array_write_startidx_numelem_tokens]=locktoken;
    }

    void store(std::pair<lockholder_index,rwlock_token_set> idx_locktoken)
    {
      lockholder_index array_write_startidx_numelem_tokens;
      rwlock_token_set locktoken;

      std::tie(array_write_startidx_numelem_tokens,locktoken)=idx_locktoken;
      store(array_write_startidx_numelem_tokens,locktoken);
    }

    //void store_name(std::string nameoverride,std::pair<std::string,rwlock_token_set> namevalue)
    //{
    //  values[nameoverride]=namevalue.second;
    //
    //}

    void store_alloc(void **array,bool write,snde_index startidx,snde_index numelem,rwlock_token_set tokens,std::string allocid)
    {
      
      values[lockholder_index(array,write,startidx,numelem)]=tokens;
      allocvalues[std::make_pair(array,allocid)]=startidx;
    }
    void store_alloc(lockholder_index idx,rwlock_token_set tokens,std::string allocid)
    {
      store_alloc(idx.array,
		  idx.write,
		  idx.startidx,
		  idx.numelem,
		  tokens,
		  allocid);
    }
    void store_alloc(std::tuple<lockholder_index,rwlock_token_set,std::string> idx_tokens_allocid)
    {
      lockholder_index idx;
      rwlock_token_set tokens;
      std::string allocid;
      std::tie(idx,tokens,allocid)=idx_tokens_allocid;
      store_alloc(idx,tokens,allocid);
    }
    
    void store_alloc(std::vector<std::tuple<lockholder_index,rwlock_token_set,std::string>> vector_idx_tokens_allocid)
    {
      size_t cnt;
      for (cnt=0; cnt < vector_idx_tokens_allocid.size();cnt++) {
	store_alloc(vector_idx_tokens_allocid[cnt]);
      }
    }
    
    //void store_addr(void **array,std::pair<rwlock_token_set,snde_index> tokens_addr)
    //{
    //  store_addr(array,std::get<0>(tokens_addr),std::get<1>(tokens_addr));
    //}

    rwlock_token_set get(void **array,bool write,snde_index indexstart,snde_index numelem)
    {
      std::unordered_map<lockholder_index,rwlock_token_set>::iterator value=values.find(lockholder_index(array,write,indexstart,numelem));

      if (value==values.end()) {
	throw std::runtime_error("Specified array and region with given writable status not found in lockholder. Was it locked with the same parameters?");
      }
      return value->second;
    }
    rwlock_token_set get_alloc_lock(void **array,snde_index numelem,std::string allocid)
    {
      std::unordered_map<std::pair<void **,std::string>,snde_index>::iterator allocvalue=allocvalues.find(std::make_pair(array,allocid));
      if (allocvalue==allocvalues.end()) {
	
	throw std::runtime_error("Specified array allocation and ID not found in lockholder. Are the array pointer and ID correct?");
      }
      return get(array,true,allocvalue->second,numelem);
    }

    snde_index get_alloc(void **array,std::string allocid)
    {
      std::unordered_map<std::pair<void **,std::string>,snde_index>::iterator allocvalue=allocvalues.find(std::make_pair(array,allocid));
      if (allocvalue==allocvalues.end()) {
	throw std::runtime_error("Specified array allocation and ID not found in lockholder. Are the array pointer and ID correct?");
      }
      return allocvalue->second;
    }
    //rwlock_token_set operator[](void ** array)
    //{
    //  return values.at(array);      
    //}

  };
  

}


#endif /* SNDE_LOCKMANAGER_HPP */
