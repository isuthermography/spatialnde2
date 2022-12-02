//%shared_ptr(snde::cachemanager);
%shared_ptr(snde::arraymanager);
%shared_ptr(std::unordered_map<void **,void **>);
%shared_ptr(std::multimap<void **,void **>);

%template(vector_arrayptr_tokenset) std::vector<std::pair<std::shared_ptr<snde::alloc_voidpp>,snde::rwlock_token_set>>;
%template(arrayptr_tokenset) std::pair<std::shared_ptr<snde::alloc_voidpp>,snde::rwlock_token_set>;
%shared_ptr(std::pair<std::shared_ptr<snde::alloc_voidpp>,snde::rwlock_token_set>);
%{
  
#include "arraymanager.hpp"
%}

using namespace snde;

namespace snde {

  class markedregion;
  
// Handle allocation routines that return both rwlock_token_set and index
%typemap(out) std::pair<snde::rwlock_token_set,snde_index> {
    $result = PyTuple_New(2);
    // Substituted code for converting cl_context here came
    // from a typemap substitution "$typemap(out,cl_context)"
    snde::rwlock_token_set result0 = std::get<0>(*&$1);
    snde::rwlock_token_set *smartresult0 = result0 ? new snde::rwlock_token_set(result0) : 0;
    
    PyTuple_SetItem($result,0,SWIG_NewPointerObj(SWIG_as_voidptr(smartresult0),$descriptor(rwlock_token_set *),SWIG_POINTER_OWN));

    snde_index result1= std::get<1>(*&$1);
    PyTuple_SetItem($result,1,PyLong_FromUnsignedLongLong(result1));


}

  
  typedef void **ArrayPtr;
  static inline ArrayPtr ArrayPtr_fromint(unsigned long long intval); 
  

  std::string AddrStr(void **ArrayPtr);
  
  class allocationinfo  { 
  public:
    std::shared_ptr<snde::allocator> alloc;
    size_t elemsize;
    size_t _totalnelem;
    size_t allocindex; // index into alloc->arrays
    
    allocationinfo();
    allocationinfo(std::shared_ptr<snde::allocator> alloc,size_t elemsize,size_t _totalnelem,size_t allocindex);
    allocationinfo(const allocationinfo &orig);
    //allocationinfo& operator=(const allocationinfo &orig);
  };


  class pending_modified_tracker {
  public:
    //std::mutex admin;
    //rangetracker<markedregion> pending_modified;

    void mark_as_pending_modified(snde_index base_index,snde_index len);

    void clear_pending_modified(snde_index base_index,snde_index len);


    rangetracker<markedregion> find_pending_modified(snde_index base_index,snde_index len);
  };
  

  class arraymanager  {
  public: 
    /* Look up allocator by __allocated_array_pointer__ only */
    /* mapping index is arrayptr, returns allocator */
    /* in this simple manager, allocators _memalloc and locker are presumed to be fixed
       after single-threaded startup, so we don't worry about thread safety */
    // don't wrap allocators because of SWIG bug
    //std::unordered_map<void **,allocationinfo> allocators;
    std::shared_ptr<snde::memallocator> _memalloc;
    std::shared_ptr<snde::lockmanager> locker;
    std::shared_ptr<allocator_alignment> alignment_requirements;
    std::size_t maxaddressbytes;

    //std::mutex admin; /* serializes access to caches */
    //std::unordered_map<void **,void **> allocation_arrays; // look up the arrays used for allocation
    //std::multimap<void **,void **> arrays_managed_by_allocator; // look up the managed arrays by the allocator array... ordering is as the arrays are created, which follows the locking order

    
    //std::unordered_map<std::string,std::shared_ptr<snde::cachemanager>> _caches;


    arraymanager(std::shared_ptr<memallocator> memalloc,std::shared_ptr<allocator_alignment> alignment_requirements,size_t maxaddressbytes,std::shared_ptr<lockmanager> locker=null);

    // accessor methods
    // virtual std::shared_ptr<std::unordered_map<void **,allocationinfo>> allocators(); // don't wrap allocators because of swig bug
    
    virtual std::shared_ptr<std::unordered_map<void **,void **>> allocation_arrays();
    
    virtual std::shared_ptr<std::multimap<void **,void **>> arrays_managed_by_allocator();


    virtual void add_allocated_array(std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,void **arrayptr,size_t elemsize,snde_index totalnelem,const std::set<snde_index>& follower_elemsizes = std::set<snde_index>());
    virtual void add_follower_array(memallocator_regionid id,void **allocatedptr,void **arrayptr,size_t elemsize);

    
    virtual void add_unmanaged_array(void **arrayptr,size_t elemsize,snde_index totalnelem);
    //virtual void mark_as_dirty(cachemanager *owning_cache_or_null,void **arrayptr,snde_index pos,snde_index len);
    //virtual void dirty_alloc(std::shared_ptr<lockholder> holder,void **arrayptr,std::string allocid, snde_index numelem);
    virtual snde_index get_elemsize(void **arrayptr);
    virtual snde_index get_total_nelem(void **arrayptr);

    virtual void realloc_down(void **allocatedptr,snde_index addr,snde_index orignelem, snde_index newnelem);

    // This next one gives SWIG trouble because of confusion over whether snde_index is an unsigned long or an unsigned long long
    //virtual std::pair<snde_index,std::vector<std::pair<void **,rwlock_token_set>>> alloc_arraylocked(snde::rwlock_token_set all_locks,void **allocatedptr,snde_index nelem);
    virtual std::vector<std::pair<std::shared_ptr<snde::alloc_voidpp>,rwlock_token_set>> alloc_arraylocked_swigworkaround(snde::rwlock_token_set all_locks,void **allocatedptr,snde_index nelem,snde_index *OUTPUT);


    virtual snde_index get_length(void **allocatedptr,snde_index addr);
    
    virtual void free(void **allocatedptr,snde_index addr);

    virtual void clear();

    virtual void remove_unmanaged_array(void **basearray);

    virtual void cleararrays(void *structaddr, size_t structlen);    
    //virtual std::shared_ptr<snde::cachemanager> get_cache(std::string name);

    //virtual bool has_cache(std::string name);
    //virtual void set_undefined_cache(std::string name,std::shared_ptr<snde::cachemanager> cache);
    virtual ~arraymanager();
  };



  
}

