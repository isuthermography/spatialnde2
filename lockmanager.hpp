#ifndef SNDE_LOCKMANAGER
#define SNDE_LOCKMANAGER

namespace snde {
  class rwlock {
    std::vector<std::condition_variable> threadqueue;
    int writelockcount;
    int readlockcount; 
  };

  
  class arraylock {
    std::mutex admin; /* locks access to full_array and subregions */
    
    rwlock full_array;
    std::vector<rwlock> subregions;
    
  };
  

  class lockmanager {
    /* Manage all of the locks/mutexes/etc. for a class 
       which contains a bunch of arrays handled by 
       snde::allocator */

    /* ***!!! Need to add reference management to lockmanager
       so regions can be automatically freed when array 
       region goes away */

    /* Also add capability for long-term persistent read locks
       that can be notified that they need to go away. These 
       can be used to keep data loaded into a rendering pipline
       or an opencl context */

    /* We define a read/write mutex per array. It is 
       assumed that all arrays are part of 
       a single class or object. 
       
       Our API provides a means to request locking of 
       sub-regions of an array, but for now these 
       simply translate to nested requests to keep the 
       entire array locked. 
    */

    /* All arrays managed by a particular allocator 
       are implicitly locked by locking the primary 
       array for that allocator */
    

    /* There is a specific locking order that must 
       be followed.  All locking requests for arrays
       must be ordered by the order of initialization 
       of the array to the lock manager. Within that, locking 
       of subregions should be done by starting address  */

    /* Note: since subregions may be passed to other threads (or GPU)
       for processing, they need to be handled by semaphores, 
       not mutexes */

    /* a write lock on an array implies a write lock on 
       all regions of the array. Likewise a read lock on 
       an array implies a read lock on all regions of the
       array. This means that locking a sub region is really 
       more like indicating a partial unlock, unless 
       you don't have the whole array locked when you lock the 
       sub-region */
    /* Note that lock upgrading is not permitted; if you 
       have a read lock on any part of the array, you may 
       not get a write lock on that part (or the whole array)
    */
    /* Note that the locks are not counted... you may not 
       lock again, once you already have a lock (except
       for locking a subregion once you have the parent) */

    /* Note that the existance of other data structures
       can implicitly define and maintains the existance of 
       array regions e.g. the vertex region mentioned in a 
       part structure indicates that those array elements 
       exist. Then you can freely lock those elements 
       without locking the structure as a whole. */
    
    
    /* NOTE: All lockmanager initialization (defining the arrays)
       for a particular class must be done from a single thread, 
       before others may 
       do any locking */ 
    
    /* These next few elements may ONLY be modified during
       initialization phase (single thread, etc.) */
  public:
    std::vector<void *> arrays; /* get array pointer from index */
    std::map<void *,int> arrayidx; /* get array index from pointer */
    std::vector<arraylock> locks; /* get lock from index */

    
    std::mutex managermutex; /* synchronizes access to lockmanager
				data structures (specifically which?) */

    void set_array_size(void *Arrayptr,size_t elemsize,snde_index nelem) {
      
    }
    
  };
  

  
}


#endif /* SNDE_LOCKMANAGER */
