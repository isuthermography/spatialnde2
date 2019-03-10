
#include <cstdint>
#include <utility>

#include "geometry_types.h"
#include "lockmanager.hpp"
#include "arraymanager.hpp"
#include "geometry.hpp"
#include "mutablewfmstore.hpp"


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
snde::lockingprocess_threaded::lockingprocess_threaded(std::shared_ptr<lockmanager> lockmanager) :
  //arrayreadregions(std::make_shared<std::vector<rangetracker<markedregion>>>(manager->locker->_arrays.size())),
  //arraywriteregions(std::make_shared<std::vector<rangetracker<markedregion>>>(manager->locker->_arrays.size())),
  lastlockingposition(),
  _executor_lock(_mutex)
{
  this->_lockmanager=lockmanager;

  //fprintf(stderr,"Creating new lockingprocess_threaded()\n");
  
  all_tokens=empty_rwlock_token_set();
  used_tokens=empty_rwlock_token_set();
        /* locking can occur in original thread, so
	   include us in _runnablethreads  */
  _runnablethreads.emplace_back((std::condition_variable *)NULL);
  
  /* since we return as the running thread,
     we return with the mutex locked via _executor_lock (from constructor, above). */
  
}


snde::lockingprocess_threaded::lockingprocess_threaded(std::shared_ptr<lockmanager> lockmanager,rwlock_token_set all_locks) :
  //arrayreadregions(std::make_shared<std::vector<rangetracker<markedregion>>>(manager->locker->_arrays.size())),
  //arraywriteregions(std::make_shared<std::vector<rangetracker<markedregion>>>(manager->locker->_arrays.size())),
  lastlockingposition(),
  _executor_lock(_mutex)
{
  this->_lockmanager=lockmanager;

  //fprintf(stderr,"Creating new lockingprocess_threaded()\n");
  
  all_tokens=all_locks;
  used_tokens=empty_rwlock_token_set();
        /* locking can occur in original thread, so
	   include us in _runnablethreads  */
  _runnablethreads.emplace_back((std::condition_variable *)NULL);
  
  /* since we return as the running thread,
     we return with the mutex locked via _executor_lock (from constructor, above). */
  
}




bool snde::lockingprocess_threaded::_barrier(lockingposition lockpos) //(lockindex_t idx,snde_index pos,bool write)
/* returns preexisting_only */
{
  bool preexisting_only=false;

  //fprintf(stderr,"Reached barrier... lockpos=%d\n",lockpos.idx);
  
  /* Since we must be the running thread in order to take
     this call, take the lock from _executor_lock */
  std::unique_lock<std::mutex> lock;
  std::condition_variable ourcv;

  void *prelockstate=prelock();
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
  while ((*_waitingthreads.begin()).second != &ourcv || _runnablethreads.size() > 0) {
    ourcv.wait(lock);
  }
  
  //fprintf(stderr,"Proceeding from barrier... lockpos=%d\n",lockpos.idx);
  /* because the wait terminated we must be first on the waiting list */
  _waitingthreads.erase(_waitingthreads.begin());
  
  /* put ourself on the running list, which must be empty */
  /* since we will be running, we are listed as NULL */
  _runnablethreads.emplace_front((std::condition_variable *)NULL);
  
  /* Should be able to allow 
     locking order violation if locks exist... 
     implement by barrier return value */
  //assert(!(lockpos < lastlockingposition)); /* This assert diagnoses a locking order violation */
  preexisting_only = lockpos < lastlockingposition;
  
  /* mark this position as our new last locking position */
  if (!preexisting_only) {
    lastlockingposition=lockpos;
  }
  
  /* give our lock back to _executor_lock since we are running */
  lock.swap(_executor_lock);
  postunlock(prelockstate);

  return preexisting_only;
}

void *snde::lockingprocess_threaded::pre_callback()
{
  return NULL;
}

void snde::lockingprocess_threaded::post_callback(void *state)
{

}


void *snde::lockingprocess_threaded::prelock()
{
  return NULL;
}

void snde::lockingprocess_threaded::postunlock(void *prelockstate)
{

}

std::pair<lockholder_index,rwlock_token_set>  snde::lockingprocess_threaded::get_locks_write_lockable(std::shared_ptr<mutableinfostore> infostore)
{
  rwlock_token_set newset;
  bool preexisting_only;
  
  preexisting_only=_barrier(lockingposition(infostore,true));
  if (preexisting_only) {
    newset = _lockmanager->get_preexisting_locks_write_lockable(all_tokens,infostore);
  } else {
    newset = _lockmanager->get_locks_write_lockable(all_tokens,infostore);
  }
  merge_into_rwlock_token_set(used_tokens,newset);
  
  //(*arraywriteregions)[_lockmanager->_arrayidx[array]].mark_all(SNDE_INDEX_INVALID);
  
  return std::make_pair(lockholder_index(infostore.get(),true),newset);
}


std::pair<lockholder_index,rwlock_token_set>  snde::lockingprocess_threaded::get_locks_write_lockable(std::shared_ptr<geometry> geom)
{
  rwlock_token_set newset;
  bool preexisting_only;
  
  preexisting_only=_barrier(lockingposition(geom,true));
  if (preexisting_only) {
    newset = _lockmanager->get_preexisting_locks_write_lockable(all_tokens,geom);
  } else {
    newset = _lockmanager->get_locks_write_lockable(all_tokens,geom);
  }
  merge_into_rwlock_token_set(used_tokens,newset);
  
  //(*arraywriteregions)[_lockmanager->_arrayidx[array]].mark_all(SNDE_INDEX_INVALID);
  
  return std::make_pair(lockholder_index(geom.get(),true),newset);
}


std::pair<lockholder_index,rwlock_token_set>  snde::lockingprocess_threaded::get_locks_write_lockable(std::shared_ptr<component> comp)
{
  rwlock_token_set newset;
  bool preexisting_only;
  
  preexisting_only=_barrier(lockingposition(comp,true));
  if (preexisting_only) {
    newset = _lockmanager->get_preexisting_locks_write_lockable(all_tokens,comp);
  } else {
    newset = _lockmanager->get_locks_write_lockable(all_tokens,comp);
  }
  merge_into_rwlock_token_set(used_tokens,newset);
  
  //(*arraywriteregions)[_lockmanager->_arrayidx[array]].mark_all(SNDE_INDEX_INVALID);
  
  return std::make_pair(lockholder_index(comp.get(),true),newset);
}


std::pair<lockholder_index,rwlock_token_set>  snde::lockingprocess_threaded::get_locks_write_lockable(std::shared_ptr<parameterization> param)
{
  rwlock_token_set newset;
  bool preexisting_only;
  
  preexisting_only=_barrier(lockingposition(param,true));
  if (preexisting_only) {
    newset = _lockmanager->get_preexisting_locks_write_lockable(all_tokens,param);
  } else {
    newset = _lockmanager->get_locks_write_lockable(all_tokens,param);
  }
  merge_into_rwlock_token_set(used_tokens,newset);
  
  //(*arraywriteregions)[_lockmanager->_arrayidx[array]].mark_all(SNDE_INDEX_INVALID);
  
  return std::make_pair(lockholder_index(param.get(),true),newset);
}



std::pair<lockholder_index,rwlock_token_set>  snde::lockingprocess_threaded::get_locks_write_array(void **array)
{
  rwlock_token_set newset;
  bool preexisting_only;
  auto arrayidx = _lockmanager->_arrayidx();
  assert(arrayidx->find(array) != arrayidx->end());
  
  preexisting_only=_barrier(lockingposition((*arrayidx)[array],0,true));
  if (preexisting_only) {
    newset = _lockmanager->get_preexisting_locks_write_array_region(all_tokens,array,0,SNDE_INDEX_INVALID);
  } else {
    newset = _lockmanager->get_locks_write_array_region(all_tokens,array,0,SNDE_INDEX_INVALID);
  }
  merge_into_rwlock_token_set(used_tokens,newset);
  
  //(*arraywriteregions)[_lockmanager->_arrayidx[array]].mark_all(SNDE_INDEX_INVALID);
  
  return std::make_pair(lockholder_index(array,true,0,SNDE_INDEX_INVALID),newset);
}

std::pair<lockholder_index,rwlock_token_set> snde::lockingprocess_threaded::get_locks_write_array_region(void **array,snde_index indexstart,snde_index numelems)
{
  rwlock_token_set newset;
  bool preexisting_only;
  auto arrayidx = _lockmanager->_arrayidx();
  assert(arrayidx->find(array) != arrayidx->end());

  
  if (_lockmanager->is_region_granular()) {
    preexisting_only=_barrier(lockingposition((*arrayidx)[array],indexstart,true));
  } else {
    preexisting_only=_barrier(lockingposition((*arrayidx)[array],0,true));
  }
  if (preexisting_only) {
    newset = _lockmanager->get_preexisting_locks_write_array_region(all_tokens,array,indexstart,numelems);
  } else {
    newset = _lockmanager->get_locks_write_array_region(all_tokens,array,indexstart,numelems);
  }
  merge_into_rwlock_token_set(used_tokens,newset);
  
  //(*arraywriteregions)[_lockmanager->_arrayidx[(void*)array]].mark_region(indexstart,numelems);
  
  return std::make_pair(lockholder_index(array,true,indexstart,numelems),newset);
}


std::pair<lockholder_index,rwlock_token_set>  snde::lockingprocess_threaded::get_locks_read_lockable(std::shared_ptr<mutableinfostore> infostore)
{
  rwlock_token_set newset;
  bool preexisting_only;
  
  preexisting_only=_barrier(lockingposition(infostore,false));
  if (preexisting_only) {
    newset = _lockmanager->get_preexisting_locks_read_lockable(all_tokens,infostore);
  } else {
    newset = _lockmanager->get_locks_read_lockable(all_tokens,infostore);
  }
  merge_into_rwlock_token_set(used_tokens,newset);
  
  //(*arraywriteregions)[_lockmanager->_arrayidx[array]].mark_all(SNDE_INDEX_INVALID);
  
  return std::make_pair(lockholder_index(infostore.get(),false),newset);
}

std::pair<lockholder_index,rwlock_token_set>  snde::lockingprocess_threaded::get_locks_read_lockable(std::shared_ptr<geometry> geom)
{
  rwlock_token_set newset;
  bool preexisting_only;
  
  preexisting_only=_barrier(lockingposition(geom,false));
  if (preexisting_only) {
    newset = _lockmanager->get_preexisting_locks_read_lockable(all_tokens,geom);
  } else {
    newset = _lockmanager->get_locks_read_lockable(all_tokens,geom);
  }
  merge_into_rwlock_token_set(used_tokens,newset);
  
  //(*arraywriteregions)[_lockmanager->_arrayidx[array]].mark_all(SNDE_INDEX_INVALID);
  
  return std::make_pair(lockholder_index(geom.get(),false),newset);
}


rwlock_token_set snde::lockingprocess_threaded::get_locks_read_lockable_temporary(std::shared_ptr<geometry> geom)
// ***!!!! Need to add temporary set to all_locks **!!!!
{
  rwlock_token_set newset;
  bool preexisting_only;
  
  _barrier(lockingposition(geom,false));
  newset = _lockmanager->get_locks_read_lockable(all_tokens,geom);

  // because tokens are temporary they do NOT get merged into used_tokens.
  // all_tokens gets released in lockingprocess_threaded.finish()
  
  
  //(*arraywriteregions)[_lockmanager->_arrayidx[array]].mark_all(SNDE_INDEX_INVALID);

  return newset; 
}

std::pair<lockholder_index,rwlock_token_set>  snde::lockingprocess_threaded::get_locks_read_lockable(std::shared_ptr<component> comp)
{
  rwlock_token_set newset;
  bool preexisting_only;
  
  preexisting_only=_barrier(lockingposition(comp,false));
  if (preexisting_only) {
    newset = _lockmanager->get_preexisting_locks_read_lockable(all_tokens,comp);
  } else {
    newset = _lockmanager->get_locks_read_lockable(all_tokens,comp);
  }
  merge_into_rwlock_token_set(used_tokens,newset);
  
  //(*arraywriteregions)[_lockmanager->_arrayidx[array]].mark_all(SNDE_INDEX_INVALID);
  
  return std::make_pair(lockholder_index(comp.get(),false),newset);
}

std::pair<lockholder_index,rwlock_token_set>  snde::lockingprocess_threaded::get_locks_read_lockable(std::shared_ptr<parameterization> param)
{
  rwlock_token_set newset;
  bool preexisting_only;
  
  preexisting_only=_barrier(lockingposition(param,false));
  if (preexisting_only) {
    newset = _lockmanager->get_preexisting_locks_read_lockable(all_tokens,param);
  } else {
    newset = _lockmanager->get_locks_read_lockable(all_tokens,param);
  }
  merge_into_rwlock_token_set(used_tokens,newset);
  
  //(*arraywriteregions)[_lockmanager->_arrayidx[array]].mark_all(SNDE_INDEX_INVALID);
  
  return std::make_pair(lockholder_index(param.get(),false),newset);
}


std::pair<lockholder_index,rwlock_token_set> snde::lockingprocess_threaded::get_locks_read_array(void **array)
{
  rwlock_token_set newset;
  bool preexisting_only;
  
  auto arrayidx = _lockmanager->_arrayidx();
  assert(arrayidx->find(array) != arrayidx->end());

  
  preexisting_only=_barrier(lockingposition((*arrayidx)[array],0,false));
  if (preexisting_only) {
    newset = _lockmanager->get_preexisting_locks_read_array_region(all_tokens,array,0,SNDE_INDEX_INVALID);
  } else {
    newset = _lockmanager->get_locks_read_array_region(all_tokens,array,0,SNDE_INDEX_INVALID);
  }
  merge_into_rwlock_token_set(used_tokens,newset);
	
  //(*arrayreadregions)[(*arrayidx)[(void*)array]].mark_all(SNDE_INDEX_INVALID);
  
  
  return std::make_pair(lockholder_index(array,false,0,SNDE_INDEX_INVALID),newset);
}


std::pair<lockholder_index,rwlock_token_set> snde::lockingprocess_threaded::get_locks_read_array_region(void **array,snde_index indexstart,snde_index numelems)
{
  bool preexisting_only;
  rwlock_token_set newset;

  auto arrayidx = _lockmanager->_arrayidx();
  assert(arrayidx->find(array) != arrayidx->end());
  
  if (_lockmanager->is_region_granular()) {
    preexisting_only=_barrier(lockingposition((*arrayidx)[array],indexstart,false));
  } else {
    preexisting_only=_barrier(lockingposition((*arrayidx)[array],0,false));
  }
  
  if (preexisting_only) {
    newset = _lockmanager->get_preexisting_locks_read_array_region(all_tokens,array,indexstart,numelems);
  } else {
    newset = _lockmanager->get_locks_read_array_region(all_tokens,array,indexstart,numelems);
  }
  merge_into_rwlock_token_set(used_tokens,newset);

  //(*arrayreadregions)[(*arrayidx)[(void*)array]].mark_region(indexstart,numelems);
  
  return std::make_pair(lockholder_index(array,false,indexstart,numelems),newset);
}


std::pair<lockholder_index,rwlock_token_set>  snde::lockingprocess_threaded::get_locks_array_region(void **array,bool write,snde_index indexstart,snde_index numelems)
{
  if (write) {
    
    return get_locks_write_array_region(array,indexstart,numelems);
  } else {
    return get_locks_read_array_region(array,indexstart,numelems);
  }
}

std::pair<lockholder_index,rwlock_token_set> snde::lockingprocess_threaded::get_locks_array(void **array,bool write)
{
  if (write) {
    
    return get_locks_write_array_region(array,0,SNDE_INDEX_INVALID);
  } else {
    return get_locks_read_array_region(array,0,SNDE_INDEX_INVALID);
  }
}

std::pair<lockholder_index,rwlock_token_set> lockingprocess_threaded::get_locks_array_mask(void **array,uint64_t maskentry,uint64_t resizemaskentry,uint64_t readmask,uint64_t writemask,uint64_t resizemask,snde_index indexstart,snde_index numelems)
{
  if (resizemask & resizemaskentry) {
    // if resizing must lock whole array for write
    return get_locks_array(array,true);
  } else if (writemask & maskentry) {
    // Lock region for write
    return get_locks_array_region(array,true,indexstart,numelems);
  } else if (readmask & maskentry) {
    // lock region for read
    return get_locks_array_region(array,false,indexstart,numelems);
    
  } else {
    return std::make_pair(lockholder_index(),empty_rwlock_token_set());
  }
}

std::pair<lockholder_index,rwlock_token_set> lockingprocess_threaded::get_locks_lockable_mask(std::shared_ptr<mutableinfostore> infostore,uint64_t maskentry,uint64_t readmask,uint64_t writemask)
{
  if (writemask & maskentry) {
    // Lock component for write
    if (maskentry==SNDE_INFOSTORE_INFOSTORE) {
      return get_locks_write_lockable(infostore);
    } else {
      assert(0); // Invalid maskentry for this function
    }
  } else if (readmask & maskentry) {
    // lock component for read
    if (maskentry==SNDE_INFOSTORE_INFOSTORE) {
      return get_locks_read_lockable(infostore);
    } else {    
      assert(0); // Invalid maskentry for this function
    }
  } else {
    return std::make_pair(lockholder_index(),empty_rwlock_token_set());
  }
}

std::pair<lockholder_index,rwlock_token_set> lockingprocess_threaded::get_locks_lockable_mask(std::shared_ptr<geometry> geom,uint64_t maskentry,uint64_t readmask,uint64_t writemask)
{
  if (writemask & maskentry) {
    // Lock component for write
    if (maskentry==SNDE_INFOSTORE_OBJECT_TREES) {
      return get_locks_write_lockable(geom);
    } else {
      assert(0); // Invalid maskentry for this function
    }
  } else if (readmask & maskentry) {
    // lock component for read
    if (maskentry==SNDE_INFOSTORE_OBJECT_TREES) {
      return get_locks_read_lockable(geom);
      
    } else {    
      assert(0); // Invalid maskentry for this function
    }
  } else {
    return std::make_pair(lockholder_index(),empty_rwlock_token_set());
  }
}

std::pair<lockholder_index,rwlock_token_set> lockingprocess_threaded::get_locks_lockable_mask(std::shared_ptr<component> comp,uint64_t maskentry,uint64_t readmask,uint64_t writemask)
{
  if (writemask & maskentry) {
    // Lock component for write
    if (maskentry==SNDE_INFOSTORE_COMPONENTS) {
      return get_locks_write_lockable(comp);
    } else {
      assert(0); // Invalid maskentry for this function
    }
  } else if (readmask & maskentry) {
    // lock component for read
    if (maskentry==SNDE_INFOSTORE_COMPONENTS) {
      return get_locks_read_lockable(comp);
      
    } else {    
      assert(0); // Invalid maskentry for this function
    }
  } else {
    return std::make_pair(lockholder_index(),empty_rwlock_token_set());
  }
}


std::vector<std::tuple<lockholder_index,rwlock_token_set,std::string>> snde::lockingprocess_threaded::alloc_array_region(std::shared_ptr<arraymanager> manager,void **allocatedptr,snde_index nelem,std::string allocid)
/* Note: returns write lock tokens for ALL arrays allocated by the allocated array referred to by allocatedptr */
/* This allocates the entire allocatedptr array for write AND any other arrays that are allocated in parallel. 
   If you will be locking anything else prior to the end of the last such array, this 
   MUST be called from a spawn()'d thread of its own then unlocks most of the elements.
   (Note that it only keeps what is needed and releases the rest  */
{

  std::vector<std::tuple<lockholder_index,rwlock_token_set,std::string>> retval;
  //  std::map<void **,rwlock_token_set> whole_array_tokens;
  
  size_t numarrays_locked=0;

  rwlock_token_set ret_tokens;
  snde_index ret_addr;
  auto arrays_managed_by_allocator = manager->arrays_managed_by_allocator();

  /* iterate over all arrays managed by this same allocator */
  for (auto allocator_managed=arrays_managed_by_allocator->lower_bound(allocatedptr);allocator_managed != arrays_managed_by_allocator->end() && allocator_managed->first==allocatedptr;allocator_managed++) {

    /* lock entire array */
    _lockmanager->get_locks_write_array_region(all_tokens,allocator_managed->second,0,SNDE_INDEX_INVALID);
    numarrays_locked++;

  }

  assert(numarrays_locked > 0); /* if this fails, you probably called with a follower array pointer, not an allocator array pointer */

  /* perform allocation */
  std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>> vector_arrayptr_tokens;
  std::tie(ret_addr,vector_arrayptr_tokens) = manager->alloc_arraylocked(all_tokens,allocatedptr,nelem);

  // iterate over returned pointers and tokens
  for (auto & arrayptr_tokens : vector_arrayptr_tokens) {    
    retval.emplace_back(std::make_tuple(lockholder_index(arrayptr_tokens.first->value(),true,ret_addr,nelem),arrayptr_tokens.second,allocid));
    /* Merge in the returned tokens to our class's all_tokens object... we'll release all of the others */
    /* (these are write tokens) */
    merge_into_rwlock_token_set(used_tokens,arrayptr_tokens.second);

  }
  

  /* (all other tokens release automatically as they go out of scope) */
  
  return retval;
}

void snde::lockingprocess_threaded::spawn(std::function<void(void)> f)
{
  /* Since we must be the running thread in order to take
     this call, take the lock from _executor_lock */
  std::unique_lock<std::mutex> lock;
  std::condition_variable ourcv;
  
  void *prelockstate=prelock();
  lock.swap(_executor_lock);
  
  
  /* Consider ourselves no longer first on the runnable stack (delegate to the thread) */
  _runnablethreads.pop_front();

  /* We will be runnable... */
  _runnablethreads.push_front(&ourcv);
  
  /* but this top entry represents the new thread */
  _runnablethreads.push_front(NULL);
  //fprintf(stderr,"Spawning thread\n");
  std::thread *newthread=new std::thread([f,this]() {
      std::unique_lock<std::mutex> subthreadlock(this->_mutex);
      
      /* in this context, pre_callback() and post_callback() handle the 
         Python GIL, if applicable */
      /* We start out as the running thread, so swap our lock
	 into executor_lock */
      subthreadlock.swap(this->_executor_lock);
      void *state=this->pre_callback();
      f();
      this->post_callback(state);
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

  //fprintf(stderr,"Proceeding after spawn\n");

  /* we are now at the front of the runnables. Switch to
     running state by popping us off and pushing NULL */
  
  _runnablethreads.pop_front();
  _runnablethreads.push_front(NULL);
  
  /* Now swap our lock back into the executor lock */
  lock.swap(_executor_lock);
  postunlock(prelockstate);
  /* return and continue executing */
}


rwlock_token_set snde::lockingprocess_threaded::finish()
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
  
  /* notify first runnable or waiting thread that it can proceed */
  if (_runnablethreads.size() > 0) {
    _runnablethreads[0]->notify_all();
  } else {
    if (_waitingthreads.size() > 0) {
      (*_waitingthreads.begin()).second->notify_all();
    }
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
  
  rwlock_token_set retval=used_tokens;
  
  release_rwlock_token_set(all_tokens); /* drop references to the tokens from the structure */
  release_rwlock_token_set(used_tokens); /* drop references to the tokens from the structure */
  
  //std::shared_ptr<std::vector<rangetracker<markedregion>>> readregions=arrayreadregions;
  //arrayreadregions.reset();
  
  //std::shared_ptr<std::vector<rangetracker<markedregion>>> writeregions=arraywriteregions;
  //arraywriteregions.reset();
  
  //return std::make_tuple(retval,readregions,writeregions);
  return retval;
}


snde::lockingprocess_threaded::~lockingprocess_threaded()
{
  /* make sure threads are all cleaned up and finish() has been called */
  finish();
}
#endif // SNDE_LOCKMANAGER_COROUTINES_THREADED
