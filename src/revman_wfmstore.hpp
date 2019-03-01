// revman_wfmstore.hpp -- glue connecting the revision manager (trm) with the mutablewfmstore)
// so that you can make revisions dependent on waveform (implictly metadata because the array dependence
// should generally handle the data)
#include <string>
#include <memory>

#include "revision_manager.hpp"
#include "mutablewfmstore.hpp"

#ifndef SNDE_REVMAN_WFMSTORE_HPP
#define SNDE_REVMAN_WFMSTORE_HPP

namespace snde {
  
class trm_mutablewfm_key: public trm_struct_depend_keyimpl_base
// dependency key on mutablewfm metadata... 
{
public:
  trm_mutablewfm_key(const trm_mutablewfm_key &)=delete; // no copy constructor
  trm_mutablewfm_key & operator=(const trm_mutablewfm_key &)=delete; // no copy assignment
 
  // wfmdb + wfmname define the waveform whose metadata we are interested in,
  // and want a notification if the metadata changes
  std::weak_ptr<mutablewfmdb> wfmdb;
  
  std::string wfmfullname; 

  trm_mutablewfm_key(std::shared_ptr<mutablewfmdb> wfmdb,std::string wfmfullname) :
    wfmdb(wfmdb),
    wfmfullname(wfmfullname),
    trm_struct_depend_keyimpl_base()
  {
    
  }


  virtual bool less_than(const trm_struct_depend_keyimpl_base &other) const
  {
    // called to identify mapping location of the trm_struct_depend.
    // both l&r should be our class
    const trm_mutablewfm_key *op = dynamic_cast<const trm_mutablewfm_key *>(&other);

    assert(op);

    if (wfmdb.owner_before(op->wfmdb)) return true; 
    if (op->wfmdb.owner_before(wfmdb)) return false; 

    // if wfmdb's equal, compare the strings
    return wfmfullname < op->wfmfullname;
    
    
  }
  
};

class trm_mutablewfm_notifier: public wfmdirty_notification_receiver,public trm_struct_depend_notifier {
  // inherited members:
  //   from wfmdirty_notification_receiver:
  //     std::string wfmfullname;
  //   from trm_struct_depend_notifier: 
  //     std::weak_ptr<trm> recipient;
  //     trm_struct_depend_key key;
  //
  //  key has a member keyimpl that can be dynamically pointer casted to trm_mutablewfm_key 

  // notifier has the potential to (but doesnt) store the value(s) of interest and only
  // propagate the notification if the value has changed
public:
  
  trm_mutablewfm_notifier(const trm_mutablewfm_notifier &)=delete; // no copy constructor
  trm_mutablewfm_notifier & operator=(const trm_mutablewfm_notifier &)=delete; // no copy assignment
  
  trm_mutablewfm_notifier(std::shared_ptr<trm> recipient,std::shared_ptr<mutablewfmdb> wfmdb,std::string wfmfullname) :
    wfmdirty_notification_receiver(wfmfullname),
    trm_struct_depend_notifier(recipient,trm_struct_depend_key(std::make_shared<trm_mutablewfm_key>(wfmdb,wfmfullname)))
    
  {

  }

  virtual void mark_as_dirty(std::shared_ptr<mutableinfostore> infostore)
  {
    assert(infostore->fullname==wfmfullname);

    //recipient->mark_struct_depend_as_modified(key);
    trm_notify();
  }

  virtual ~trm_mutablewfm_notifier() {}
};


static trm_struct_depend wfm_dependency(std::shared_ptr<trm> revman, std::shared_ptr<mutablewfmdb> wfmdb,std::string wfmfullname)
{
  std::shared_ptr<trm_mutablewfm_notifier> notifier = std::make_shared<trm_mutablewfm_notifier>(revman,wfmdb,wfmfullname);

  wfmdb->add_dirty_notification_receiver(notifier);
  
  return std::make_pair(notifier->key,notifier);
}


}
#endif // SNDE_REVMAN_WFMSTORE_HPP

