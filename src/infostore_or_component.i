%shared_ptr(snde::lockable_infostore_or_component);

%{
  
#include "infostore_or_component.hpp"
%}

%nodefaultctor snde::lockable_infostore_or_component; // lockable_infostore_or_component is abstract so it shouldn't have default constructor created

namespace snde {
  class part;
  class component;
  class parameterization;
  class image_data;
  class iterablewfmrefs;
  class immutable_metadata;
  class mutabledatastore;

  class lockable_infostore_or_component /* : public std::enable_shared_from_this<lockable_infostore_or_component> */ {
  public:
    uint64_t lic_mask;  // the mask corresponding to this specific type of infostore or component e.g. SNDE_INFOSTORE_INFOSTORES, etc. 
    std::shared_ptr<rwlock> lock; // managed by lockmanager... locks notifiers and other non-const, non-atomic (or atomic for write) elements of subclasses
    
    //lockable_infostore_or_component(uint64_t lic_mask); // don't define constructor for swig because class is abstract

    virtual ~lockable_infostore_or_component();



  };
}
