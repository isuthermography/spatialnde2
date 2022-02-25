%shared_ptr(snde::cachemanager);
snde_rawaccessible(snde::cachemanager);
%shared_ptr(snde::cached_recording);
snde_rawaccessible(snde::cached_recording);

%{
  
#include "cached_recording.hpp"
%}

namespace snde {

  class recording_storage; // recstore_storage.hpp
  class cachemanager /*: public std::enable_shared_from_this<cachemanager> */ { /* abstract base class for cache managers */
  public:
    cachemanager() = default;
    
    virtual void mark_as_invalid(void **arrayptr,snde_index base_index,snde_index pos,snde_index numelem)=0;
    virtual void notify_storage_expiration(void **arrayptr,snde_index base_index,snde_index nelem)=0;
    // Rule of 3
    cachemanager(const cachemanager &) = delete;  // CC and CAO are deleted because we don't anticipate needing them. 
    cachemanager& operator=(const cachemanager &) = delete; 
    virtual ~cachemanager()=default;
    
  };

  std::string get_cache_name(const std::string &base);


  
  class cached_recording /* : public std::enable_shared_from_this<cached_recording>*/ {
  public:
    // abstract base class, used by recording_storage 
    cached_recording() = default;

    // rule of 3
    cached_recording(const cached_recording &orig) = delete;
    cached_recording& operator=(const cached_recording &) = delete;
    virtual ~cached_recording() = default;

  };

  


};
