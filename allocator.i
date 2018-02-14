%shared_ptr(snde::allocator);
//%shared_ptr(snde::cmemallocator);

%{
  
#include "allocator.hpp"
%}


namespace snde {

  struct arrayinfo {
    void **arrayptr;
    size_t elemsize;
    bool destroyed; /* locked by allocatormutex */
  };

  class allocator /* : public allocatorbase*/ {

    std::mutex allocatormutex; // Always final mutex in locking order; protects the free list 
    
  public: 
    snde_index _firstfree;
    snde_index _totalnchunks;

    std::shared_ptr<memallocator> _memalloc;
    std::shared_ptr<lockmanager> _locker; // could be NULL if there is no locker
    std::deque<std::shared_ptr<std::function<void(snde_index)>>> realloccallbacks; // locked by allocatormutex

    bool destroyed;
    /* 
       Should lock things on allocation...
     
       Will probably need separate lock for main *_arrayptr so we can 
       wait on everything relinquishing that in order to do a realloc. 
     
    */
  
  
    //void **_arrayptr;
    //size_t _elemsize;

    // The arrays member is genuinely public for read access and
    // may be iterated over. Note that it may only be written
    // from the single thread during the initialization phase
    std::deque<struct arrayinfo> arrays;
    
    
    /* Freelist structure ... But read via memcpy to avoid 
       aliasing problems... 
       snde_index freeblocksize  (in chunks)
       snde_index nextfreeblockstart ... nextfreeblockstart of last block should be _totalnchunks
    */
    snde_index _allocchunksize; // size of chunks we allocate, in numbers of elements
  
    allocator(std::shared_ptr<memallocator> memalloc,std::shared_ptr<lockmanager> locker,void **arrayptr,size_t elemsize,snde_index totalnelem);

    allocator(const allocator &)=delete; /* copy constructor disabled */
    allocator& operator=(const allocator &)=delete; /* assignment disabled */

    size_t add_other_array(void **arrayptr, size_t elsize);

    size_t num_arrays(void);
    
    void remove_array(void **arrayptr);
    
    void _realloc(snde_index newnchunks);

    snde_index total_nelem();
    
  
    snde_index alloc(snde_index nelem);
    
    //void register_realloc_callback(std::shared_ptr<std::function<void(snde_index)>> callback);

    //void unregister_realloc_callback(std::shared_ptr<std::function<void(snde_index)>> callback);

    
    void free(snde_index addr,snde_index nelem);
    ~allocator();
  };
}

