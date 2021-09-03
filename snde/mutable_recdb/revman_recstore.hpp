// revman_recstore.hpp -- glue connecting the revision manager (trm) with the mutablerecstore)
// so that you can make revisions dependent on recording (implictly metadata because the array dependence
// should generally handle the data)
#include <string>
#include <memory>

#include "snde/revision_manager.hpp"
#include "snde/mutablerecstore.hpp"

#ifndef SNDE_REVMAN_RECSTORE_HPP
#define SNDE_REVMAN_RECSTORE_HPP

namespace snde {
  
class trm_mutablerec_key: public trm_struct_depend_keyimpl_base
// dependency key on mutablerec metadata... 
{
public:
  trm_mutablerec_key(const trm_mutablerec_key &)=delete; // no copy constructor
  trm_mutablerec_key & operator=(const trm_mutablerec_key &)=delete; // no copy assignment
 
  // recdb + recname define the recording whose metadata we are interested in,
  // and want a notification if the metadata changes
  std::weak_ptr<mutablerecdb> recdb;
  
  std::string recfullname; 

  trm_mutablerec_key(std::shared_ptr<mutablerecdb> recdb,std::string recfullname) :
    recdb(recdb),
    recfullname(recfullname),
    trm_struct_depend_keyimpl_base()
  {
    
  }


  virtual bool less_than(const trm_struct_depend_keyimpl_base &other) const
  {
    // called to identify mapping location of the trm_struct_depend.
    // both l&r should be our class
    const trm_mutablerec_key *op = dynamic_cast<const trm_mutablerec_key *>(&other);

    assert(op);

    if (recdb.owner_before(op->recdb)) return true; 
    if (op->recdb.owner_before(recdb)) return false; 

    // if recdb's equal, compare the strings
    return recfullname < op->recfullname;
    
    
  }
  
};

class trm_mutablerec_notifier: public recdirty_notification_receiver,public trm_struct_depend_notifier {
  // inherited members:
  //   from recdirty_notification_receiver:
  //     std::string recfullname;
  //   from trm_struct_depend_notifier: 
  //     std::weak_ptr<trm> recipient;
  //     trm_struct_depend_key key;
  //
  //  key has a member keyimpl that can be dynamically pointer casted to trm_mutablerec_key 

  // notifier has the potential to (but doesnt) store the value(s) of interest and only
  // propagate the notification if the value has changed
public:
  
  trm_mutablerec_notifier(const trm_mutablerec_notifier &)=delete; // no copy constructor
  trm_mutablerec_notifier & operator=(const trm_mutablerec_notifier &)=delete; // no copy assignment
  
  trm_mutablerec_notifier(std::shared_ptr<trm> recipient,std::shared_ptr<mutablerecdb> recdb,std::string recfullname) :
    recdirty_notification_receiver(recfullname),
    trm_struct_depend_notifier(recipient,trm_struct_depend_key(std::make_shared<trm_mutablerec_key>(recdb,recfullname)))
    
  {

  }

  virtual void mark_as_dirty(std::shared_ptr<mutableinfostore> infostore)
  {
    assert(infostore->fullname==recfullname);

    //recipient->mark_struct_depend_as_modified(key);
    trm_notify();
  }

  virtual ~trm_mutablerec_notifier() {}
};


static trm_struct_depend rec_dependency(std::shared_ptr<trm> revman, std::shared_ptr<mutablerecdb> recdb,std::string recfullname)
{
  std::shared_ptr<trm_mutablerec_notifier> notifier = std::make_shared<trm_mutablerec_notifier>(revman,recdb,recfullname);

  recdb->add_dirty_notification_receiver(notifier);
  
  return std::make_pair(notifier->key,notifier);
}


static std::tuple<std::shared_ptr<mutablerecdb>, std::shared_ptr<mutableinfostore>> get_rec_dependency(const trm_struct_depend &depend)
{
  std::shared_ptr<mutablerecdb> recdb = std::dynamic_pointer_cast<trm_mutablerec_key>(depend.first.keyimpl)->recdb.lock();
  std::shared_ptr<mutableinfostore> input=recdb->lookup(std::dynamic_pointer_cast<trm_mutablerec_key>(depend.first.keyimpl)->recfullname);

  return std::make_tuple(recdb,input);
}
  
}
#endif // SNDE_REVMAN_RECSTORE_HPP

