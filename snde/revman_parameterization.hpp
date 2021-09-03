// revman_parameterization.hpp -- glue connecting the revision manager (trm) with surface parameterizations in the geometry store)
// so that you can make revisions dependent on the parameterizations
#include <string>
#include <memory>

#include "snde/revision_manager.hpp"
#include "snde/geometry.hpp"

#ifndef SNDE_REVMAN_PARAMETERIZATION_HPP
#define SNDE_REVMAN_PARAMETERIZATION_HPP

namespace snde {
  
class trm_parameterization_key: public trm_struct_depend_keyimpl_base
// dependency key on geometry parameterization... 
{
public:
  trm_parameterization_key(const trm_parameterization_key &)=delete; // no copy constructor
  trm_parameterization_key & operator=(const trm_parameterization_key &)=delete; // no copy assignment
 
  std::weak_ptr<parameterization> param;
  

  trm_parameterization_key(std::shared_ptr<parameterization> param) :
    param(param),
    trm_struct_depend_keyimpl_base()
  {
    
  }

  
  virtual bool less_than(const trm_struct_depend_keyimpl_base &other) const
  {
    // called to identify mapping location of the trm_struct_depend.
    // both l&r should be our class
    const trm_parameterization_key *op = dynamic_cast<const trm_parameterization_key *>(&other);

    assert(op);
    
    return param.owner_before(op->param);
    
  }
  
};

class trm_parameterization_notifier: public parameterization::notifier,public trm_struct_depend_notifier {
  // inherited members:
  //   from parameterization::notifier:
  //     (none)
  //   from trm_struct_depend_notifier: 
  //     std::weak_ptr<trm> recipient;
  //     trm_struct_depend_key key;
  //
  //  key has a member keyimpl that can be dynamically pointer casted to trm_parameterization_key 

  // notifier has the potential to (but doesnt) store the value(s) of interest and only
  // propagate the notification if the value has changed
public:
  
  trm_parameterization_notifier(const trm_parameterization_notifier &)=delete; // no copy constructor
  trm_parameterization_notifier & operator=(const trm_parameterization_notifier &)=delete; // no copy assignment
  
  trm_parameterization_notifier(std::shared_ptr<trm> recipient,std::shared_ptr<parameterization> param) :
    parameterization::notifier(),
    trm_struct_depend_notifier(recipient,trm_struct_depend_key(std::make_shared<trm_parameterization_key>(param)))
    
  {
    
  }
  
  virtual void modified(std::shared_ptr<parameterization> param)
  {
    std::shared_ptr<trm_parameterization_key> our_key;
    our_key=std::dynamic_pointer_cast<trm_parameterization_key>(key.keyimpl);
    assert(our_key);
    std::shared_ptr<parameterization> our_key_param;
    our_key_param=our_key->param.lock();
    
    assert(our_key_param && param==our_key_param);

    //recipient->mark_struct_depend_as_modified(key);
    trm_notify();
  }
  
  virtual ~trm_parameterization_notifier() {}
};


static trm_struct_depend parameterization_dependency(std::shared_ptr<trm> revman, std::shared_ptr<parameterization> param)
{
  std::shared_ptr<trm_parameterization_notifier> notifier = std::make_shared<trm_parameterization_notifier>(revman,param);
  
  param->add_notifier(notifier);
  
  return std::make_pair(notifier->key,notifier);
}
  
static std::shared_ptr<parameterization> get_parameterization_dependency(const trm_struct_depend &depend)
// May return nullptr if parameterization has been deleted 
{
  std::shared_ptr<trm_parameterization_notifier> notifier=std::dynamic_pointer_cast<trm_parameterization_notifier>(depend.second);
  assert(notifier);
  
  std::shared_ptr<trm_parameterization_key> our_key;
  our_key=std::dynamic_pointer_cast<trm_parameterization_key>(notifier->key.keyimpl);
  assert(our_key);
  
  return our_key->param.lock();
}
  
}
#endif // SNDE_REVMAN_PARAMETERIZATION_HPP

