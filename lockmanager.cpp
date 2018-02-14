
#include <cstdint>

#include "geometry_types.h"
#include "lockmanager.hpp"

using namespace snde;

// These are defined here to avoid forward reference problems. 
void snde::rwlock_lockable::lock() {
  if (_writer) {
    _rwlock_obj->lock_writer();
  } else {
    _rwlock_obj->lock_reader();
  }
}

void snde::rwlock_lockable::unlock() {
  if (_writer) {
    _rwlock_obj->unlock_writer();
  } else {
    _rwlock_obj->unlock_reader();
  }
}


//snde::lockingprocess::lockingprocess()
//{
//
//};


snde::lockingprocess::~lockingprocess()
{

}



#ifdef SNDE_LOCKMANAGER_COROUTINES_THREADED
snde::lockingprocess_threaded::lockingprocess_threaded(std::shared_ptr<lockmanager> manager) :
  arrayreadregions(std::make_shared<std::vector<rangetracker<markedregion>>>(manager->_arrays.size())),
  arraywriteregions(std::make_shared<std::vector<rangetracker<markedregion>>>(manager->_arrays.size())),
  lastlockingposition(0,0,true),
  _executor_lock(_mutex)
{
  this->_lockmanager=manager;
  
  
  all_tokens=empty_rwlock_token_set();
        /* locking can occur in original thread, so
	   include us in _runnablethreads  */
  _runnablethreads.emplace_back((std::condition_variable *)NULL);
  
  /* since we return as the running thread,
     we return with the mutex locked via _executor_lock (from constructor, above). */
  
}




void snde::lockingprocess_threaded::_barrier(lockingposition lockpos) //(size_t arrayidx,snde_index pos,bool write)
{
  
  /* Since we must be the running thread in order to take
     this call, take the lock from _executor_lock */
  std::unique_lock<std::mutex> lock;
  std::condition_variable ourcv;
  
  lock.swap(_executor_lock);
  
  /* at the barrier, we are no longer runnable... */
  //assert(_runnablethreads[0]==std::this_thread::id);
  _runnablethreads.pop_front();
  
  /* ... but we are waiting so register on the waiting multimap */
  _waitingthreads.emplace(std::make_pair(lockpos,&ourcv));
  
  /* notify first runnable or waiting thread that we have gone into waiting mode too */
  if (_runnablethreads.size() > 0) {
    _runnablethreads[0]->notify_all();
  } else {
    if (_waitingthreads.size() > 0) {
      (*_waitingthreads.begin()).second->notify_all();
	  }
  }
  
  /* now wait for us to be the lowest available waiter AND the runnablethreads to be empty */
  while ((*_waitingthreads.begin()).second != &ourcv && _runnablethreads.size() == 0) {
    ourcv.wait(lock);
  }
  
  /* because the wait terminated we must be first on the waiting list */
  _waitingthreads.erase(_waitingthreads.begin());
  
  /* put ourself on the running list, which must be empty */
  /* since we will be running, we are listed as NULL */
  _runnablethreads.emplace_front((std::condition_variable *)NULL);
  
  /* give our lock back to _executor_lock since we are running */
  lock.swap(_executor_lock);
  
  assert(!(lockpos < lastlockingposition)); /* This assert diagnoses a locking order violation */

  /* mark this position as our new last locking position */
  lastlockingposition=lockpos;
}

rwlock_token_set snde::lockingprocess_threaded::get_locks_write_array(void **array)
{
  rwlock_token_set newset;
  _barrier(lockingposition(_lockmanager->_arrayidx[array],0,true));
  newset = _lockmanager->get_locks_write_array_region(all_tokens,array,0,SNDE_INDEX_INVALID);
  
  (*arraywriteregions)[_lockmanager->_arrayidx[array]].mark_all(SNDE_INDEX_INVALID);
  
  return newset;
}

rwlock_token_set snde::lockingprocess_threaded::get_locks_write_array_region(void **array,snde_index indexstart,snde_index numelems)
{
  rwlock_token_set newset;
  if (_lockmanager->is_region_granular()) {
    _barrier(lockingposition(_lockmanager->_arrayidx[array],indexstart,true));
  } else {
    _barrier(lockingposition(_lockmanager->_arrayidx[array],0,true));
  }
  newset = _lockmanager->get_locks_write_array_region(all_tokens,array,indexstart,numelems);
  
  (*arraywriteregions)[_lockmanager->_arrayidx[array]].mark_region(indexstart,numelems);
  
  return newset;
}

rwlock_token_set snde::lockingprocess_threaded::get_locks_read_array(void **array)
{
  rwlock_token_set newset;
  _barrier(lockingposition(_lockmanager->_arrayidx[array],0,false));
  newset = _lockmanager->get_locks_read_array_region(all_tokens,array,0,SNDE_INDEX_INVALID);
	
  (*arrayreadregions)[_lockmanager->_arrayidx[array]].mark_all(SNDE_INDEX_INVALID);
  
  
  return newset;
}


rwlock_token_set snde::lockingprocess_threaded::get_locks_read_array_region(void **array,snde_index indexstart,snde_index numelems)
      {
        rwlock_token_set newset;

        if (_lockmanager->is_region_granular()) {
	  _barrier(lockingposition(_lockmanager->_arrayidx[array],indexstart,false));
        } else {
	  _barrier(lockingposition(_lockmanager->_arrayidx[array],0,false));
        }

        newset = _lockmanager->get_locks_read_array_region(all_tokens,array,indexstart,numelems);

        (*arrayreadregions)[_lockmanager->_arrayidx[array]].mark_region(indexstart,numelems);

        return newset;
      }

void snde::lockingprocess_threaded::spawn(std::function<void(void)> f)
{
  /* Since we must be the running thread in order to take
     this call, take the lock from _executor_lock */
  std::unique_lock<std::mutex> lock;
  std::condition_variable ourcv;
  
  lock.swap(_executor_lock);
  
  
  /* Consider ourselves no longer first on the runnable stack (delegate to the thread) */
  _runnablethreads.pop_front();

  /* We will be runnable... */
  _runnablethreads.push_front(&ourcv);
  
  /* but this top entry represents the new thread */
  _runnablethreads.push_front(NULL);
  
  std::thread *newthread=new std::thread([f,this]() {
      std::unique_lock<std::mutex> subthreadlock(this->_mutex);
      
      /* We start out as the running thread, so swap our lock
	 into executor_lock */
      subthreadlock.swap(this->_executor_lock);
      f();
      /* spawn code done... swap lock back into us, where it will
	       be unlocked on return */
      subthreadlock.swap(this->_executor_lock);
      
      /* Since we were running, the NULL at the front of
	 _runnablethreads represents us. Remove this */
      this->_runnablethreads.pop_front();
      
      /* ... and notify whomever is first to take over execution */
      if (this->_runnablethreads.size() > 0) {
	_runnablethreads[0]->notify_all();
      } else {
	if (this->_waitingthreads.size() > 0) {
	  (*this->_waitingthreads.begin()).second->notify_all();
	}
	    }
    });
  _threadarray.emplace_back(newthread);
  
  /* Wait for us to make it back to the front of the runnables */
  while (_runnablethreads[0] != &ourcv) {
    ourcv.wait(lock);
  }
  /* we are now at the front of the runnables. Switch to
     running state by popping us off and pushing NULL */
  
  _runnablethreads.pop_front();
  _runnablethreads.push_front(NULL);
  
  /* Now swap our lock back into the executor lock */
  lock.swap(_executor_lock);
  
  /* return and continue executing */
}


std::tuple<rwlock_token_set,std::shared_ptr<std::vector<rangetracker<markedregion>>>,std::shared_ptr<std::vector<rangetracker<markedregion>>>> snde::lockingprocess_threaded::finish()
{
  /* Since we must be the running thread in order to take this call,
     take the lock from _executor_lock */
  /* returns tuple: token_set, arrayreadregions, arraywriteregions */
  
  /* if finish() has already been called once, _executor_lock is
     unlocked and swap() is a no-op */
  std::unique_lock<std::mutex> lock;
  
  lock.swap(_executor_lock);
  
  /* We no longer count as a runnable thread from here-on.
     if finish() has already been called size(_runnablethreads)==0 */
  if (_runnablethreads.size() > 0) {
    //assert(_runnablethreads[0]==std::this_thread::id);
    _runnablethreads.pop_front();
  }
  
  /* join each of the threads */
  while (_threadarray.size() > 0) {
    std::thread *tojoin = _threadarray[0];
    
    lock.unlock();
    tojoin->join();
    
    lock.lock();
    _threadarray.erase(_threadarray.begin());
    
    delete tojoin;
  }
  
  rwlock_token_set retval=all_tokens;
  
  release_rwlock_token_set(all_tokens); /* drop references to the tokens from the structure */
  
  std::shared_ptr<std::vector<rangetracker<markedregion>>> readregions=arrayreadregions;
  arrayreadregions.reset();
  
  std::shared_ptr<std::vector<rangetracker<markedregion>>> writeregions=arraywriteregions;
  arraywriteregions.reset();
  
  return std::make_tuple(retval,readregions,writeregions);
}


snde::lockingprocess_threaded::~lockingprocess_threaded()
{
  /* make sure threads are all cleaned up and finish() has been called */
  finish();
}
#endif // SNDE_LOCKMANAGER_COROUTINES_THREADED
