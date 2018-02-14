
%shared_ptr(snde::cachemanager);
%shared_ptr(snde::arraymanager);


%{
  
#include "arraymanager.hpp"
%}

using namespace snde;

namespace snde {

  typedef void **ArrayPtr;
  static inline ArrayPtr ArrayPtr_fromint(unsigned long long intval); 
  
  class cachemanager { /* abstract base class for cache managers */
  public:
    virtual ~cachemanager() {};
    
  };

  class allocationinfo  { 
  public:
    std::shared_ptr<snde::allocator> alloc;
    size_t allocindex; // index into alloc->arrays
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

    //std::mutex admin; /* serializes access to caches */

    
    //std::unordered_map<std::string,std::shared_ptr<snde::cachemanager>> _caches;


    arraymanager(std::shared_ptr<snde::memallocator> memalloc);


    virtual void add_allocated_array(void **arrayptr,size_t elemsize,snde_index totalnelem);
    virtual void add_follower_array(void **allocatedptr,void **arrayptr,size_t elemsize);
    virtual snde_index alloc(void **allocatedptr,snde_index nelem);

    virtual void free(void **allocatedptr,snde_index addr,snde_index nelem);

    virtual void clear();

    virtual void cleararrays(void *structaddr, size_t structlen);    
    virtual std::shared_ptr<snde::cachemanager> get_cache(std::string name);

    virtual bool has_cache(std::string name);
    virtual void set_cache(std::string name,std::shared_ptr<snde::cachemanager> cache);
    virtual ~arraymanager();
  };



  
}

