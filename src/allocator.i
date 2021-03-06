%shared_ptr(snde::allocator);
%shared_ptr(snde::allocator_alignment);
%shared_ptr(snde::alloc_voidpp);
%shared_ptr(std::deque<struct arrayinfo>);
//%shared_ptr(snde::cmemallocator);

%{
  
#include "allocator.hpp"
%}


namespace snde {

  
  class allocator_alignment {
  public:
    std::vector<unsigned> address_alignment; /* required address alignments, in bytes */

    allocator_alignment();
    void add_requirement(unsigned alignment);

    unsigned get_alignment();
    
  };


  struct arrayinfo {
    void **arrayptr;
    size_t elemsize;
    bool destroyed; /* locked by allocatormutex */
  };

  class alloc_voidpp {
  public:
    void **ptr;
    alloc_voidpp(void **_ptr);
    alloc_voidpp();
    void **value();
  };

  
  class allocation {
    /* protected by allocator's allocatormutex */
  public:
    snde_index regionstart;
    snde_index regionend;
    
    allocation(const allocation &)=delete; /* copy constructor disabled */
    allocation& operator=(const allocation &)=delete; /* copy assignment disabled */
    
    allocation(snde_index regionstart,snde_index regionend,snde_index nelem);

    bool attempt_merge(allocation &later);
    
    /* breakup method ends this region at breakpoint and returns
       a new region starting at from breakpoint to the prior end */
    std::shared_ptr<allocation> sp_breakup(snde_index breakpoint,snde_index nelem);
    ~allocation();
    
    
  };
  
  

  
  class allocator /* : public allocatorbase*/ {

    std::mutex allocatormutex; // Always final mutex in locking order; protects the free list 
    
  public: 
    snde_index _totalnchunks;

    std::shared_ptr<memallocator> _memalloc;
    std::shared_ptr<lockmanager> _locker; // could be NULL if there is no locker
    std::deque<std::shared_ptr<std::function<void(snde_index)>>> pool_realloc_callbacks; // locked by allocatormutex

    bool destroyed;
    /* 
       Should lock things on allocation...
     
       Will probably need separate lock for main *_arrayptr so we can 
       wait on everything relinquishing that in order to do a realloc. 
     
    */
  
  
    //void **_arrayptr;
    //size_t _elemsize;

    // The arrays member is genuinely public for read access through
    // the arrays() accessor and
    // may be iterated over. Note that it may only be written
    // to when allocatormutex is locked.
    // Elements may never be deleted, so as to maintain numbering.
    // The shared_ptr around _arrays
    // is atomic, so it may be freely read through the accessor
    //std::shared_ptr<std::deque<struct arrayinfo>> _arrays; // atomic shared_ptr 

    
    /* Freelist structure ... 
    */
    rangetracker<allocation> allocations;
    rangetracker<allocation> allocations_unmerged;
    snde_index _allocchunksize; // size of chunks we allocate, in numbers of elements
  
    allocator(std::shared_ptr<memallocator> memalloc,std::shared_ptr<lockmanager> locker,std::shared_ptr<allocator_alignment> alignment,void **arrayptr,size_t elemsize,snde_index totalnelem,const std::set<snde_index>& follower_elemsizes);

    allocator(const allocator &)=delete; /* copy constructor disabled */
    allocator& operator=(const allocator &)=delete; /* assignment disabled */

    // accessor for atomic _arrays member
    std::shared_ptr<std::deque<struct arrayinfo>> arrays();
    
    size_t add_other_array(void **arrayptr, size_t elsize);

    size_t num_arrays(void);
    
    void remove_array(void **arrayptr);
    
    void _pool_realloc(snde_index newnchunks);

    snde_index total_nelem();
    
  
    // This next one gives SWIG trouble because of confusion over whether snde_index is an unsigned long or an unsigned long long
    //std::pair<snde_index,std::vector<std::pair<alloc_voidpp,rwlock_token_set>>> alloc_arraylocked(rwlock_token_set all_locks,snde_index nelem);
    
    //void register_pool_realloc_callback(std::shared_ptr<std::function<void(snde_index)>> callback);

    //void unregister_pool_realloc_callback(std::shared_ptr<std::function<void(snde_index)>> callback);

    
    void free(snde_index addr);
    ~allocator();
  };
}

