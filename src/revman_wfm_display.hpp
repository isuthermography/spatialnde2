// revman_wfmstore.hpp -- glue connecting the revision manager (trm) with wfm_display
// so that you can make revisions dependent on display parameters
#include <string>
#include <memory>

#include "revision_manager.hpp"
#include "wfm_display.hpp"

#ifndef SNDE_REVMAN_WFM_DISPLAY_HPP
#define SNDE_REVMAN_WFM_DISPLAY_HPP

namespace snde {

class trm_wfmdisplay_key: public trm_struct_depend_keyimpl_base
// dependency key on wfm_display modification... 
{
public:
  std::shared_ptr<display_channel> displaychan;
  
  trm_wfmdisplay_key(const trm_wfmdisplay_key &)=delete; // no copy constructor
  trm_wfmdisplay_key & operator=(const trm_wfmdisplay_key &)=delete; // no copy assignment
   
  //std::string wfmfullname; 

  trm_wfmdisplay_key(std::shared_ptr<display_channel> displaychan) :
    displaychan(displaychan),
    trm_struct_depend_keyimpl_base()
  {
    
  }


  virtual bool less_than(const trm_struct_depend_keyimpl_base &other) const
  {
    // called to identify mapping location of the trm_struct_depend.
    // both l&r should be our class
    const trm_wfmdisplay_key *op = dynamic_cast<const trm_wfmdisplay_key *>(&other);

    assert(op);
    
    return displaychan.owner_before(op->displaychan);
    //if (displaychan.owner_before(op->displaychan)) return true; 
    //if (op->displaychan.owner_before(displaychan)) return false; 

    //// if wfmdb's equal, compare the strings
    //return wfmname < op->wfmname;
    
    
  }
  
};

class trm_wfmdisplay_notifier: public wfmdisplay_notification_receiver,public trm_struct_depend_notifier {
  // inherited members:
  //   from wfmdisplay_notification_receiver:
  //   from trm_struct_depend_notifier: 
  //     std::weak_ptr<trm> recipient;
  //     trm_struct_depend_key key;
  //
  //  key has a member keyimpl that can be dynamically pointer casted to trm_mutablewfm_md_key 

  // notifier has the potential to (but doesnt) store the value(s) of interest and only
  // propagate the notification if the value has changed
public:
  
  trm_wfmdisplay_notifier(const trm_wfmdisplay_notifier &)=delete; // no copy constructor
  trm_wfmdisplay_notifier & operator=(const trm_wfmdisplay_notifier &)=delete; // no copy assignment
  
   trm_wfmdisplay_notifier(std::shared_ptr<trm> recipient,std::shared_ptr<display_channel> displaychan) :
    wfmdisplay_notification_receiver(),
    trm_struct_depend_notifier(recipient,trm_struct_depend_key(std::make_shared<trm_wfmdisplay_key>(displaychan)))
    
  {

  }

  virtual void mark_as_dirty(std::shared_ptr<display_channel> dirtychan)
  {

    std::shared_ptr<trm_wfmdisplay_key> keyimpl=std::dynamic_pointer_cast<trm_wfmdisplay_key>(key.keyimpl);

    assert(keyimpl);

    std::shared_ptr<display_channel> keyimpl_displaychan_strong(keyimpl->displaychan);
    assert(keyimpl_displaychan_strong && dirtychan==keyimpl_displaychan_strong);
    
    /* ***!!! Since this should generally be done in a transaction, should we start one here??? */
    // No... for now we assume our caller has done so 
    
    //recipient->mark_struct_depend_as_modified(key);
    trm_notify();
  }
  
  virtual ~trm_wfmdisplay_notifier() {}
};


static trm_struct_depend display_channel_dependency(std::shared_ptr<trm> revman, std::shared_ptr<display_channel> displaychan)
{

  std::shared_ptr<trm_wfmdisplay_notifier> notifier = std::make_shared<trm_wfmdisplay_notifier>(revman,displaychan);
  
  displaychan->add_adjustment_dep(notifier);
  
  return std::make_pair(notifier->key,notifier);
}

}

#endif // SNDE_REVMAN_WFM_DISPLAY_HPP
