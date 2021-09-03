// revman_recstore.hpp -- glue connecting the revision manager (trm) with rec_display
// so that you can make revisions dependent on display parameters
#include <string>
#include <memory>

#include "snde/revision_manager.hpp"
#include "snde/rec_display.hpp"

#ifndef SNDE_REVMAN_REC_DISPLAY_HPP
#define SNDE_REVMAN_REC_DISPLAY_HPP

namespace snde {

class trm_recdisplay_key: public trm_struct_depend_keyimpl_base
// dependency key on rec_display modification... 
{
public:
  std::shared_ptr<display_channel> displaychan;
  
  trm_recdisplay_key(const trm_recdisplay_key &)=delete; // no copy constructor
  trm_recdisplay_key & operator=(const trm_recdisplay_key &)=delete; // no copy assignment
   
  //std::string recfullname; 

  trm_recdisplay_key(std::shared_ptr<display_channel> displaychan) :
    displaychan(displaychan),
    trm_struct_depend_keyimpl_base()
  {
    
  }


  virtual bool less_than(const trm_struct_depend_keyimpl_base &other) const
  {
    // called to identify mapping location of the trm_struct_depend.
    // both l&r should be our class
    const trm_recdisplay_key *op = dynamic_cast<const trm_recdisplay_key *>(&other);

    assert(op);
    
    return displaychan.owner_before(op->displaychan);
    //if (displaychan.owner_before(op->displaychan)) return true; 
    //if (op->displaychan.owner_before(displaychan)) return false; 

    //// if recdb's equal, compare the strings
    //return recname < op->recname;
    
    
  }
  
};

class trm_recdisplay_notifier: public recdisplay_notification_receiver,public trm_struct_depend_notifier {
  // inherited members:
  //   from recdisplay_notification_receiver:
  //   from trm_struct_depend_notifier: 
  //     std::weak_ptr<trm> recipient;
  //     trm_struct_depend_key key;
  //
  //  key has a member keyimpl that can be dynamically pointer casted to trm_mutablerec_md_key 

  // notifier has the potential to (but doesnt) store the value(s) of interest and only
  // propagate the notification if the value has changed
public:
  
  trm_recdisplay_notifier(const trm_recdisplay_notifier &)=delete; // no copy constructor
  trm_recdisplay_notifier & operator=(const trm_recdisplay_notifier &)=delete; // no copy assignment
  
   trm_recdisplay_notifier(std::shared_ptr<trm> recipient,std::shared_ptr<display_channel> displaychan) :
    recdisplay_notification_receiver(),
    trm_struct_depend_notifier(recipient,trm_struct_depend_key(std::make_shared<trm_recdisplay_key>(displaychan)))
    
  {

  }

  virtual void mark_as_dirty(std::shared_ptr<display_channel> dirtychan)
  {

    std::shared_ptr<trm_recdisplay_key> keyimpl=std::dynamic_pointer_cast<trm_recdisplay_key>(key.keyimpl);

    assert(keyimpl);

    std::shared_ptr<display_channel> keyimpl_displaychan_strong(keyimpl->displaychan);
    assert(keyimpl_displaychan_strong && dirtychan==keyimpl_displaychan_strong);
    
    /* ***!!! Since this should generally be done in a transaction, should we start one here??? */
    // No... for now we assume our caller has done so 
    
    //recipient->mark_struct_depend_as_modified(key);
    trm_notify();
  }
  
  virtual ~trm_recdisplay_notifier() {}
};


static trm_struct_depend display_channel_dependency(std::shared_ptr<trm> revman, std::shared_ptr<display_channel> displaychan)
{

  std::shared_ptr<trm_recdisplay_notifier> notifier = std::make_shared<trm_recdisplay_notifier>(revman,displaychan);
  
  displaychan->add_adjustment_dep(notifier);
  
  return std::make_pair(notifier->key,notifier);
}


static std::shared_ptr<display_channel> get_display_channel_dependency(const trm_struct_depend &depend)
{
  std::shared_ptr<trm_recdisplay_notifier> notifier=std::dynamic_pointer_cast<trm_recdisplay_notifier>(depend.second);
  assert(notifier);

  std::shared_ptr<trm_recdisplay_key> our_key;
  our_key=std::dynamic_pointer_cast<trm_recdisplay_key>(notifier->key.keyimpl);
  assert(our_key);

  return our_key->displaychan;

}
  
}

#endif // SNDE_REVMAN_REC_DISPLAY_HPP
