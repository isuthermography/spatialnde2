#include "wfmstore.hpp"
#include "notify.hpp"
#include "utils.hpp"



namespace snde {

  /*
  template <typename T>
  class notifier {
    std::shared_ptr<T> changing_object; 
    std::function<bool(std::shared_ptr<T>)> criteria_satisfied; // call criteria_satisifed(changing_object) to check criteria
    
    notifier(std::shared_ptr<T> changing_object) :
      changing_object(changing_object,std::function<bool(std::shared_ptr<T>)> criteria_satisfied)
    {

    }
  };
  */
    channel_notification_criteria::channel_notification_criteria() :
    waveformset_complete(false)
  {
    
  }

  // copy assignment operator -- copies but ignores non-copyable mutex
  channel_notification_criteria & channel_notification_criteria::operator=(const channel_notification_criteria &orig)
  {
    std::lock_guard<std::mutex> orig_admin(orig.admin);

    waveformset_complete = orig.waveformset_complete;
    metadataonly_channels = orig.metadataonly_channels;
    fullyready_channels = orig.fullyready_channels;

    return *this;
  }
  
  // copy constructor -- copies but ignores non-copyable mutex
  channel_notification_criteria::channel_notification_criteria(const channel_notification_criteria &orig)
  {
    std::lock_guard<std::mutex> orig_admin(orig.admin);

    waveformset_complete = orig.waveformset_complete;
    metadataonly_channels = orig.metadataonly_channels;
    fullyready_channels = orig.fullyready_channels;
    
  }

  void channel_notification_criteria::add_waveformset_complete() 
  {
    // only allowed during creation so we don't worry about locking
    waveformset_complete=true;
    
  }

  void channel_notification_criteria::add_completion_channel(std::shared_ptr<waveform_set_state> wss,std::string channelname)
  // satisfied once the specified channel reaches the current (when this criteria is defined) definition of completion for that channel (mdonly vs fullyready)
  // Checks the current definition of completion and calls add_fullyready_channel or add_metadataonly_channel as appropriate
  {
    std::map<std::string,std::shared_ptr<instantiated_math_function>>::iterator math_function_it;
    bool mdonly = false; 

    math_function_it = wss->mathstatus.math_functions->defined_math_functions.find(channelname);
    if (math_function_it != wss->mathstatus.math_functions->defined_math_functions.end()) {
      // channel is a math channel
      
      std::lock_guard<std::mutex> wssadmin(wss->admin);
      math_function_status &mathstatus = wss->mathstatus.function_status.at(math_function_it->second);
      mdonly = mathstatus.execfunc->mdonly;
    }
    
    if (mdonly) {
      add_metadataonly_channel(channelname);
    } else {
      add_fullyready_channel(channelname);

    }
  }
  
  void channel_notification_criteria::add_fullyready_channel(std::string channelname) 
  {
    // only allowed during creation so we don't worry about locking
    fullyready_channels.emplace(channelname);    
  }

  void channel_notification_criteria::add_metadataonly_channel(std::string channelname) 
  {
    // only allowed during creation so we don't worry about locking
    metadataonly_channels.emplace(channelname);    
  }

  channel_notify::channel_notify() :
    criteria()
  {

  }
  
  channel_notify::channel_notify(const channel_notification_criteria &criteria_to_copy) :
    criteria(criteria_to_copy)
  {
    
  }

  void channel_notify::notify_metadataonly(const std::string &channelpath) // notify this notifier that the given channel has satisified metadataonly (not usually modified by subclass)
  {

    bool generate_notify=false;
    {
      std::lock_guard<std::mutex> criteria_admin(criteria.admin);
      
      std::unordered_set<std::string>::iterator mdo_channels_it = criteria.metadataonly_channels.find(channelpath);
      
      assert(mdo_channels_it != criteria.metadataonly_channels.end()); // should only be able to do a notify on something in the set!
      criteria.metadataonly_channels.erase(mdo_channels_it);

      if (!criteria.metadataonly_channels.size() && !criteria.fullyready_channels.size() && !criteria.waveformset_complete) {
	// all criteria removed; ready for notification
	generate_notify=true;
      }
    }
    if (generate_notify) {
      perform_notify();
    }
  }
  
  void channel_notify::notify_ready(const std::string &channelpath) // notify this notifier that the given channel has satisified ready (not usually modified by subclass)
  {
    bool generate_notify=false;
    {
      std::lock_guard<std::mutex> criteria_admin(criteria.admin);
      
      std::unordered_set<std::string>::iterator fullyready_channels_it = criteria.fullyready_channels.find(channelpath);
      
      assert(fullyready_channels_it != criteria.fullyready_channels.end()); // should only be able to do a notify on something in the set!
      criteria.fullyready_channels.erase(fullyready_channels_it);

      if (!criteria.metadataonly_channels.size() && !criteria.fullyready_channels.size() && !criteria.waveformset_complete) {
	// all criteria removed; ready for notification
	       
	generate_notify=true;
      }
    }
    if (generate_notify) {
      //printf("cn::notify_ready();\n");
      //fflush(stdout);
      perform_notify();
    }
  }
  
  void channel_notify::notify_waveformset_complete() // notify this notifier that all waveforms in this set are complete.
  {
    bool generate_notify=false;
    {
      std::lock_guard<std::mutex> criteria_admin(criteria.admin);

      assert(criteria.waveformset_complete);
      criteria.waveformset_complete=false; // criterion is now satisfied, so we no longer need to wait for it

      
      if (!criteria.metadataonly_channels.size() && !criteria.fullyready_channels.size() && !criteria.waveformset_complete) {
	// all criteria removed; ready for notification
	generate_notify=true;
      }
    }
    if (generate_notify) {
      //printf("cn::notify_ws_complete();\n");
      //fflush(stdout);
      perform_notify();
    }

  }

  void channel_notify::check_waveformset_complete(std::shared_ptr<waveform_set_state> wss)
  {
    bool generate_notify=false;

    {
      std::lock_guard<std::mutex> wss_admin(wss->admin);
      std::lock_guard<std::mutex> criteria_admin(criteria.admin);
      
      // check if all waveforms are ready;
      bool all_ready = !wss->wfmstatus.defined_waveforms.size() && !wss->wfmstatus.instantiated_waveforms.size();      
      //printf("cn::all_ready;\n");
      //fflush(stdout);
      
      if (criteria.waveformset_complete && all_ready) {
	criteria.waveformset_complete = false;
	wss->waveformset_complete_notifiers.erase(shared_from_this());
      }
      
      if (!criteria.metadataonly_channels.size() && !criteria.fullyready_channels.size() && !criteria.waveformset_complete) {
	// all criteria removed; ready for notification
	generate_notify=true;
      }
    }
    if (generate_notify) {
      //printf("cn::check_ws_complete();\n");
      //fflush(stdout);
      perform_notify();
    }
    
    
  }
  bool channel_notify::_check_all_criteria_locked(std::shared_ptr<waveform_set_state> wss,bool notifies_already_applied_to_wss)
  // Internal only: Should be called with wss admin lock and criteria admin locks locked. Returns true if an immediate notification is due
  {
    bool generate_notify=false;
    std::vector<std::string> mdonly_satisfied;
    std::vector<std::string> fullyready_satisfied;

    snde_debug(SNDE_DC_NOTIFY,"channel_notify::_check_all_criteria_locked(0x%lx)",(unsigned long)(wss.get()));
    for (auto && md_channelname: criteria.metadataonly_channels) {
      channel_state & chanstate = wss->wfmstatus.channel_map.at(md_channelname);
      
      if (chanstate.waveform_is_complete(true)) {
	if (notifies_already_applied_to_wss) {
	  std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_metadataonly = chanstate.begin_atomic_notify_about_this_channel_metadataonly_update();	  
	  notify_about_this_channel_metadataonly->erase(shared_from_this());
	  chanstate.end_atomic_notify_about_this_channel_metadataonly_update(notify_about_this_channel_metadataonly);
	}
	mdonly_satisfied.push_back(md_channelname);
      }
    }
    
    for (auto && fr_channelname: criteria.fullyready_channels) {
      channel_state & chanstate = wss->wfmstatus.channel_map.at(fr_channelname);
      
      if (chanstate.waveform_is_complete(false)) {
	if (notifies_already_applied_to_wss) {
	  std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_ready = chanstate.begin_atomic_notify_about_this_channel_ready_update();
	  
	  notify_about_this_channel_ready->erase(shared_from_this());
	  chanstate.end_atomic_notify_about_this_channel_ready_update(notify_about_this_channel_ready);
	}
	fullyready_satisfied.push_back(fr_channelname);
	
      }
    }
    
    // check if all waveforms are ready;
    bool all_ready = !wss->wfmstatus.defined_waveforms.size() && !wss->wfmstatus.instantiated_waveforms.size();
    
    // update criteria according to satisfied conditions
    for (auto && md_channelname: mdonly_satisfied) {
      criteria.metadataonly_channels.erase(md_channelname);
    }
    
    for (auto && fr_channelname: fullyready_satisfied) {
      criteria.fullyready_channels.erase(fr_channelname);
    }
    
    if (criteria.waveformset_complete && all_ready) {
      criteria.waveformset_complete = false; 
      if (notifies_already_applied_to_wss) {
	wss->waveformset_complete_notifiers.erase(shared_from_this());
      }
    }
    
    if (!criteria.metadataonly_channels.size() && !criteria.fullyready_channels.size() && !criteria.waveformset_complete) {
      // all criteria removed; ready for notification
      generate_notify=true;
    }
    snde_debug(SNDE_DC_NOTIFY,"cacl() mdoc_size=%d frc_size=%d wait_complete=%s defined=%d instantiated=%d; returns %s",(int)criteria.metadataonly_channels.size(),(int)criteria.fullyready_channels.size(),(criteria.waveformset_complete) ? "true": "false",(int)wss->wfmstatus.defined_waveforms.size(),(int)wss->wfmstatus.instantiated_waveforms.size(), (generate_notify) ? "true":"false");
    return generate_notify;
  }
  
    
  

  void channel_notify::check_all_criteria(std::shared_ptr<waveform_set_state> wss)
  {
    bool generate_notify=false;
	

    {
      std::lock_guard<std::mutex> wss_admin(wss->admin);
      std::lock_guard<std::mutex> criteria_admin(criteria.admin);

      generate_notify=_check_all_criteria_locked(wss,true);
      
      
    }
    
    if (generate_notify) {
      perform_notify();
    }

  }
  
  std::shared_ptr<channel_notify> channel_notify::notify_copier()
  {
    throw snde_error("Copier must be provided for repetitive channel notifications");
    return nullptr;
  }
  
  void channel_notify::apply_to_wss(std::shared_ptr<waveform_set_state> wss) // apply this notification process to a particular waveform_set_state. WARNING: May trigger the notification immediately
  {
    bool generate_notify;
    {
      std::lock_guard<std::mutex> wss_admin(wss->admin);
      std::lock_guard<std::mutex> criteria_admin(criteria.admin);

      generate_notify=_check_all_criteria_locked(wss,false);
      
      // Add criteria to this waveform set state


      for (auto && md_channelname: criteria.metadataonly_channels) {
	channel_state & chanstate = wss->wfmstatus.channel_map.at(md_channelname);
      
	std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_metadataonly = chanstate.begin_atomic_notify_about_this_channel_metadataonly_update();	  
	notify_about_this_channel_metadataonly->emplace(shared_from_this());
	chanstate.end_atomic_notify_about_this_channel_metadataonly_update(notify_about_this_channel_metadataonly);		
      }


      
      for (auto && fr_channelname: criteria.fullyready_channels) {
	channel_state & chanstate = wss->wfmstatus.channel_map.at(fr_channelname);
      
	std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_ready = chanstate.begin_atomic_notify_about_this_channel_ready_update();
	  
	notify_about_this_channel_ready->emplace(shared_from_this());
	chanstate.end_atomic_notify_about_this_channel_ready_update(notify_about_this_channel_ready);
      }
    }

      
    if (criteria.waveformset_complete) {
      wss->waveformset_complete_notifiers.emplace(shared_from_this());
    }
    
    
    if (generate_notify) {
      perform_notify();
    }
  }

  std::shared_ptr<channel_notify> repetitive_channel_notify::create_notify_instance()
  // this default implementation uses the channel_notify's notify_copier() to create the instance
  {
    return notify->notify_copier();
  }

  promise_channel_notify::promise_channel_notify(const std::vector<std::string> &mdonly_channels,const std::vector<std::string> &ready_channels,bool wfmset_complete)
  {
    for (auto && mdonly_channel: mdonly_channels) {
      criteria.add_metadataonly_channel(mdonly_channel);
    }
    for (auto && ready_channel: ready_channels) {
      criteria.add_fullyready_channel(ready_channel);
    }
    if (wfmset_complete) {
      criteria.add_waveformset_complete();
    }
  }

  void promise_channel_notify::perform_notify()
  {
    promise.set_value();
  }
  
  _unchanged_channel_notify::_unchanged_channel_notify(std::weak_ptr<wfmdatabase> wfmdb,std::shared_ptr<globalrevision> subsequent_globalrev,channel_state &current_channelstate,channel_state & sg_channelstate,bool mdonly) :
    wfmdb(wfmdb),
    subsequent_globalrev(subsequent_globalrev),
    current_channelstate(current_channelstate),
    sg_channelstate(sg_channelstate),
    mdonly(mdonly)
  {
    if (mdonly) {
      criteria.add_metadataonly_channel(sg_channelstate.config->channelpath);
    } else {
      criteria.add_fullyready_channel(sg_channelstate.config->channelpath);
    }
  }
  
  void _unchanged_channel_notify::perform_notify()
  {
    {
      std::lock_guard<std::mutex> subsequent_globalrev_admin(subsequent_globalrev->admin);
    
      // Pass completed waveform from this channel_state to subsequent_globalrev's channelstate
      sg_channelstate.end_atomic_wfm_update(current_channelstate.wfm());

      std::unordered_map<std::shared_ptr<channelconfig>,channel_state *>::iterator def_wfms_it = subsequent_globalrev->wfmstatus.defined_waveforms.find(current_channelstate.config);
      assert(def_wfms_it != subsequent_globalrev->wfmstatus.defined_waveforms.end()); // should be in defined_waveforms prior to the notifications

      assert(def_wfms_it->second == &sg_channelstate);
      
      if (mdonly  && !sg_channelstate.waveform_is_complete(false)) {
	// if we are mdonly and waveform is only complete through mdonly
	assert(sg_channelstate.waveform_is_complete(true));
	subsequent_globalrev->wfmstatus.metadataonly_waveforms.emplace(current_channelstate.config,&sg_channelstate);	
      } else {
	// waveform must be complete
	assert(sg_channelstate.waveform_is_complete(false));
	subsequent_globalrev->wfmstatus.completed_waveforms.emplace(current_channelstate.config,&sg_channelstate);	
      }
      
      subsequent_globalrev->wfmstatus.defined_waveforms.erase(def_wfms_it);
    }  
    sg_channelstate.issue_nonmath_notifications(subsequent_globalrev);

    std::shared_ptr<wfmdatabase> wfmdb_strong=wfmdb.lock();
    if (wfmdb_strong) {
      sg_channelstate.issue_math_notifications(wfmdb_strong,subsequent_globalrev);
    }
  }

  _previous_globalrev_done_notify::_previous_globalrev_done_notify(std::weak_ptr<wfmdatabase> wfmdb,std::shared_ptr<globalrevision> previous_globalrev,std::shared_ptr<globalrevision> current_globalrev) :
    wfmdb(wfmdb),
    previous_globalrev(previous_globalrev),
    current_globalrev(current_globalrev)
  {
    criteria.add_waveformset_complete();
  }
    
  void _previous_globalrev_done_notify::perform_notify()
  {
    {
      std::shared_ptr<wfmdatabase> wfmdb_strong=wfmdb.lock();
      if (!wfmdb_strong) return; 
      std::unique_lock<std::mutex> computeresources_admin(*wfmdb_strong->compute_resources->admin);

      // go through compute_resources blocked_list, removing the blocked computations
      // and queueing them up. 
      std::multimap<uint64_t,std::shared_ptr<pending_computation>>::iterator blocked_it; 
      while ((blocked_it = wfmdb_strong->compute_resources->blocked_list.begin()) != wfmdb_strong->compute_resources->blocked_list.end() && blocked_it->first <= previous_globalrev->globalrev) {
	std::shared_ptr<pending_computation> blocked_computation = blocked_it->second;
	wfmdb_strong->compute_resources->blocked_list.erase(blocked_it);
	computeresources_admin.unlock();
	wfmdb_strong->compute_resources->_queue_computation_internal(blocked_computation);
	computeresources_admin.lock();
      }
    }

    // clear previous_globalrev's prerequisite state now that previous_globalrev is entirely ready
    previous_globalrev->atomic_prerequisite_state_clear();
  }
  


};
