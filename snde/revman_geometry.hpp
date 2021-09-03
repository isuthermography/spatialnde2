// revman_geometry.hpp -- glue connecting the revision manager (trm) with the geometry store)
// so that you can make revisions dependent on the component database
#include <string>
#include <memory>

#include "snde/revision_manager.hpp"
#include "snde/geometry.hpp"

#ifndef SNDE_REVMAN_GEOMETRY_HPP
#define SNDE_REVMAN_GEOMETRY_HPP

namespace snde {
  
class trm_component_key: public trm_struct_depend_keyimpl_base
// dependency key on geometry component... 
{
public:
  trm_component_key(const trm_component_key &)=delete; // no copy constructor
  trm_component_key & operator=(const trm_component_key &)=delete; // no copy assignment
 
  std::weak_ptr<component> comp;
  

  trm_component_key(std::shared_ptr<component> comp) :
    comp(comp),
    trm_struct_depend_keyimpl_base()
  {
    
  }

  
  virtual bool less_than(const trm_struct_depend_keyimpl_base &other) const
  {
    // called to identify mapping location of the trm_struct_depend.
    // both l&r should be our class
    const trm_component_key *op = dynamic_cast<const trm_component_key *>(&other);

    assert(op);

    return comp.owner_before(op->comp);
    
  }
  
};

class trm_component_notifier: public component::notifier,public trm_struct_depend_notifier {
  // inherited members:
  //   from component::notifier:
  //     (none)
  //   from trm_struct_depend_notifier: 
  //     std::weak_ptr<trm> recipient;
  //     trm_struct_depend_key key;
  //
  //  key has a member keyimpl that can be dynamically pointer casted to trm_component_key 

  // notifier has the potential to (but doesnt) store the value(s) of interest and only
  // propagate the notification if the value has changed
public:
  
  trm_component_notifier(const trm_component_notifier &)=delete; // no copy constructor
  trm_component_notifier & operator=(const trm_component_notifier &)=delete; // no copy assignment
  
  trm_component_notifier(std::shared_ptr<trm> recipient,std::shared_ptr<component> comp) :
    component::notifier(),
    trm_struct_depend_notifier(recipient,trm_struct_depend_key(std::make_shared<trm_component_key>(comp)))
    
  {

  }

  virtual void modified(std::shared_ptr<component> comp)
  {
    std::shared_ptr<trm_component_key> our_key;
    our_key=std::dynamic_pointer_cast<trm_component_key>(key.keyimpl);
    assert(our_key);
    std::shared_ptr<component> our_key_comp;
    our_key_comp=our_key->comp.lock();
    
    assert(our_key_comp && comp==our_key_comp);

    //recipient->mark_struct_depend_as_modified(key);
    trm_notify();
  }

  virtual ~trm_component_notifier() {}
};


static trm_struct_depend geom_dependency(std::shared_ptr<trm> revman, std::shared_ptr<component> comp)
{
  std::shared_ptr<trm_component_notifier> notifier = std::make_shared<trm_component_notifier>(revman,comp);

  comp->add_notifier(notifier);
  
  return std::make_pair(notifier->key,notifier);
}

static std::shared_ptr<component> get_geom_dependency(const trm_struct_depend &depend)
// May return nullptr if component has been deleted 
{
  std::shared_ptr<trm_component_notifier> notifier=std::dynamic_pointer_cast<trm_component_notifier>(depend.second);
  assert(notifier);
  
  std::shared_ptr<trm_component_key> our_key;
  our_key=std::dynamic_pointer_cast<trm_component_key>(notifier->key.keyimpl);
  assert(our_key);

  return our_key->comp.lock();
}




  
}
#endif // SNDE_REVMAN_RECSTORE_HPP

