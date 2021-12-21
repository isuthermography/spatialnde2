#include "snde/recstore.hpp"
#include "snde/notify.hpp"
#include "snde/utils.hpp"



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
    recordingset_complete(false)
  {
    
  }

  // copy assignment operator -- copies but ignores non-copyable mutex
  channel_notification_criteria & channel_notification_criteria::operator=(const channel_notification_criteria &orig)
  {
    std::lock_guard<std::mutex> orig_admin(orig.admin);

    recordingset_complete = orig.recordingset_complete;
    metadataonly_channels = orig.metadataonly_channels;
    fullyready_channels = orig.fullyready_channels;

    return *this;
  }
  
  // copy constructor -- copies but ignores non-copyable mutex
  channel_notification_criteria::channel_notification_criteria(const channel_notification_criteria &orig)
  {
    std::lock_guard<std::mutex> orig_admin(orig.admin);

    recordingset_complete = orig.recordingset_complete;
    metadataonly_channels = orig.metadataonly_channels;
    fullyready_channels = orig.fullyready_channels;
    
  }

  void channel_notification_criteria::add_recordingset_complete() 
  {
    // only allowed during creation so we don't worry about locking
    recordingset_complete=true;
    
  }

  void channel_notification_criteria::add_completion_channel(std::shared_ptr<recording_set_state> rss,std::string channelname)
  // satisfied once the specified channel reaches the current (when this criteria is defined) definition of completion for that channel (mdonly vs fullyready)
  // Checks the current definition of completion and calls add_fullyready_channel or add_metadataonly_channel as appropriate
  {
    std::map<std::string,std::shared_ptr<instantiated_math_function>>::iterator math_function_it;
    bool mdonly = false; 

    math_function_it = rss->mathstatus.math_functions->defined_math_functions.find(channelname);
    if (math_function_it != rss->mathstatus.math_functions->defined_math_functions.end()) {
      // channel is a math channel
      
      std::lock_guard<std::mutex> rssadmin(rss->admin);
      math_function_status &mathstatus = rss->mathstatus.function_status.at(math_function_it->second);
      mdonly = mathstatus.mdonly;
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

      if (!criteria.metadataonly_channels.size() && !criteria.fullyready_channels.size() && !criteria.recordingset_complete) {
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

      if (!criteria.metadataonly_channels.size() && !criteria.fullyready_channels.size() && !criteria.recordingset_complete) {
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
  
  void channel_notify::notify_recordingset_complete() // notify this notifier that all recordings in this set are complete.
  {
    bool generate_notify=false;
    {
      std::lock_guard<std::mutex> criteria_admin(criteria.admin);

      assert(criteria.recordingset_complete);
      criteria.recordingset_complete=false; // criterion is now satisfied, so we no longer need to wait for it

      
      if (!criteria.metadataonly_channels.size() && !criteria.fullyready_channels.size() && !criteria.recordingset_complete) {
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

  void channel_notify::check_recordingset_complete(std::shared_ptr<recording_set_state> rss)
  {
    bool generate_notify=false;

    {
      std::lock_guard<std::mutex> rss_admin(rss->admin);
      std::lock_guard<std::mutex> criteria_admin(criteria.admin);
      
      // check if all recordings are ready;
      bool all_ready = !rss->recstatus.defined_recordings.size() && !rss->recstatus.instantiated_recordings.size();      
      //printf("cn::all_ready;\n");
      //fflush(stdout);
      
      if (criteria.recordingset_complete && all_ready) {
	criteria.recordingset_complete = false;
	rss->recordingset_complete_notifiers.erase(shared_from_this());
      }
      
      if (!criteria.metadataonly_channels.size() && !criteria.fullyready_channels.size() && !criteria.recordingset_complete) {
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
  bool channel_notify::_check_all_criteria_locked(std::shared_ptr<recording_set_state> rss,bool notifies_already_applied_to_rss)
  // Internal only: Should be called with rss admin lock and criteria admin locks locked. Returns true if an immediate notification is due
  {
    bool generate_notify=false;
    std::vector<std::string> mdonly_satisfied;
    std::vector<std::string> fullyready_satisfied;

    snde_debug(SNDE_DC_NOTIFY,"channel_notify::_check_all_criteria_locked(0x%lx)",(unsigned long)(rss.get()));
    for (auto && md_channelname: criteria.metadataonly_channels) {
      channel_state & chanstate = rss->recstatus.channel_map.at(md_channelname);
      
      if (chanstate.recording_is_complete(true)) {
	if (notifies_already_applied_to_rss) {
	  std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_metadataonly = chanstate.begin_atomic_notify_about_this_channel_metadataonly_update();	  
	  notify_about_this_channel_metadataonly->erase(shared_from_this());
	  chanstate.end_atomic_notify_about_this_channel_metadataonly_update(notify_about_this_channel_metadataonly);
	}
	mdonly_satisfied.push_back(md_channelname);
      }
    }
    
    for (auto && fr_channelname: criteria.fullyready_channels) {
      channel_state & chanstate = rss->recstatus.channel_map.at(fr_channelname);
      
      if (chanstate.recording_is_complete(false)) {
	if (notifies_already_applied_to_rss) {
	  std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_ready = chanstate.begin_atomic_notify_about_this_channel_ready_update();
	  
	  notify_about_this_channel_ready->erase(shared_from_this());
	  chanstate.end_atomic_notify_about_this_channel_ready_update(notify_about_this_channel_ready);
	}
	fullyready_satisfied.push_back(fr_channelname);
	
      }
    }
    
    // check if all recordings are ready;
    bool all_ready = !rss->recstatus.defined_recordings.size() && !rss->recstatus.instantiated_recordings.size();
    
    // update criteria according to satisfied conditions
    for (auto && md_channelname: mdonly_satisfied) {
      criteria.metadataonly_channels.erase(md_channelname);
    }
    
    for (auto && fr_channelname: fullyready_satisfied) {
      criteria.fullyready_channels.erase(fr_channelname);
    }
    
    if (criteria.recordingset_complete && all_ready) {
      criteria.recordingset_complete = false; 
      if (notifies_already_applied_to_rss) {
	rss->recordingset_complete_notifiers.erase(shared_from_this());
      }
    }
    
    if (!criteria.metadataonly_channels.size() && !criteria.fullyready_channels.size() && !criteria.recordingset_complete) {
      // all criteria removed; ready for notification
      generate_notify=true;
    }
    snde_debug(SNDE_DC_NOTIFY,"cacl() mdoc_size=%d frc_size=%d wait_complete=%s defined=%d instantiated=%d; returns %s",(int)criteria.metadataonly_channels.size(),(int)criteria.fullyready_channels.size(),(criteria.recordingset_complete) ? "true": "false",(int)rss->recstatus.defined_recordings.size(),(int)rss->recstatus.instantiated_recordings.size(), (generate_notify) ? "true":"false");
    return generate_notify;
  }
  
    
  

  void channel_notify::check_all_criteria(std::shared_ptr<recording_set_state> rss)
  {
    bool generate_notify=false;
	

    {
      std::lock_guard<std::mutex> rss_admin(rss->admin);
      std::lock_guard<std::mutex> criteria_admin(criteria.admin);

      generate_notify=_check_all_criteria_locked(rss,true);
      
      
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
  
  void channel_notify::apply_to_rss(std::shared_ptr<recording_set_state> rss) // apply this notification process to a particular recording_set_state. WARNING: May trigger the notification immediately
  {
    bool generate_notify;
    {
      std::lock_guard<std::mutex> rss_admin(rss->admin);
      std::lock_guard<std::mutex> criteria_admin(criteria.admin);

      generate_notify=_check_all_criteria_locked(rss,false);
      
      // Add criteria to this recording set state


      for (auto && md_channelname: criteria.metadataonly_channels) {
	channel_state & chanstate = rss->recstatus.channel_map.at(md_channelname);
      
	std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_metadataonly = chanstate.begin_atomic_notify_about_this_channel_metadataonly_update();	  
	notify_about_this_channel_metadataonly->emplace(shared_from_this());
	chanstate.end_atomic_notify_about_this_channel_metadataonly_update(notify_about_this_channel_metadataonly);		
      }


      
      for (auto && fr_channelname: criteria.fullyready_channels) {
	channel_state & chanstate = rss->recstatus.channel_map.at(fr_channelname);
      
	std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_ready = chanstate.begin_atomic_notify_about_this_channel_ready_update();
	  
	notify_about_this_channel_ready->emplace(shared_from_this());
	chanstate.end_atomic_notify_about_this_channel_ready_update(notify_about_this_channel_ready);
      }
    }

      
    if (criteria.recordingset_complete) {
      rss->recordingset_complete_notifiers.emplace(shared_from_this());
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

  promise_channel_notify::promise_channel_notify(const std::vector<std::string> &mdonly_channels,const std::vector<std::string> &ready_channels,bool recset_complete)
  {
    for (auto && mdonly_channel: mdonly_channels) {
      criteria.add_metadataonly_channel(mdonly_channel);
    }
    for (auto && ready_channel: ready_channels) {
      criteria.add_fullyready_channel(ready_channel);
    }
    if (recset_complete) {
      criteria.add_recordingset_complete();
    }
  }

  void promise_channel_notify::perform_notify()
  {
    // not impossible that we would have a second (superfluous) notification
    // so we explicitly ignore that.
    try {
      promise.set_value();
    } catch (const std::future_error &e) {
      if (e.code() != std::future_errc::promise_already_satisfied) {
	throw; // rethrow all other exceptions
      }
    }
  }
  
  _unchanged_channel_notify::_unchanged_channel_notify(std::weak_ptr<recdatabase> recdb,std::shared_ptr<globalrevision> subsequent_globalrev,channel_state &current_channelstate,channel_state & sg_channelstate,bool mdonly) :
    recdb(recdb),
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
    
      // Pass completed recording from this channel_state to subsequent_globalrev's channelstate
      sg_channelstate.end_atomic_rec_update(current_channelstate.rec());

      std::unordered_map<std::shared_ptr<channelconfig>,channel_state *>::iterator def_recs_it = subsequent_globalrev->recstatus.defined_recordings.find(current_channelstate.config);
      assert(def_recs_it != subsequent_globalrev->recstatus.defined_recordings.end()); // should be in defined_recordings prior to the notifications

      assert(def_recs_it->second == &sg_channelstate);
      
      if (mdonly  && !sg_channelstate.recording_is_complete(false)) {
	// if we are mdonly and recording is only complete through mdonly
	assert(sg_channelstate.recording_is_complete(true));
	subsequent_globalrev->recstatus.metadataonly_recordings.emplace(current_channelstate.config,&sg_channelstate);	
      } else {
	// recording must be complete
	assert(sg_channelstate.recording_is_complete(false));
	subsequent_globalrev->recstatus.completed_recordings.emplace(current_channelstate.config,&sg_channelstate);	
      }
      
      subsequent_globalrev->recstatus.defined_recordings.erase(def_recs_it);
    }  
    sg_channelstate.issue_nonmath_notifications(subsequent_globalrev);

    std::shared_ptr<recdatabase> recdb_strong=recdb.lock();
    if (recdb_strong) {
      sg_channelstate.issue_math_notifications(recdb_strong,subsequent_globalrev,false);
    }
  }

  /*
  _previous_globalrev_nolongerneeded_notify::_previous_globalrev_nolongerneeded_notify(std::weak_ptr<recdatabase> recdb,std::shared_ptr<globalrevision> previous_globalrev,std::shared_ptr<globalrevision> current_globalrev) :
    recdb(recdb),
    previous_globalrev(previous_globalrev),
    current_globalrev(current_globalrev)
  {
    criteria.add_recordingset_complete();
  }
    
  void _previous_globalrev_nolongerneeded_notify::perform_notify()
  {
    {
      std::shared_ptr<recdatabase> recdb_strong=recdb.lock();
      if (!recdb_strong) return; 

    }
  }
  */
  
  _globalrev_complete_notify::_globalrev_complete_notify(std::weak_ptr<recdatabase> recdb,std::shared_ptr<globalrevision> globalrev) :
    recdb(recdb),
    globalrev(globalrev)
  {
    criteria.add_recordingset_complete();
    
  }

  void _globalrev_complete_notify::perform_notify()
  {
    // This notification indicates that the attached globalrevision
    // has reached ready state


    std::shared_ptr<recdatabase> recdb_strong=recdb.lock();
    if (!recdb_strong) return;

    std::lock_guard<std::mutex> recdb_admin(recdb_strong->admin);

    snde_debug(SNDE_DC_RECDB,"_globalrev_complete_notify::perform_notify(); globalrev=%llu; notify_globalrev=%llu",(unsigned long long)globalrev->globalrev,(unsigned long long)recdb_strong->monitoring_notify_globalrev);

    globalrev->atomic_prerequisite_state_clear(); // once we are ready, we no longer care about any prerequisite state, so that can be free'd as needed. 


    // Perform any moniitoring notifications
    if (globalrev->globalrev == recdb_strong->monitoring_notify_globalrev+1) {
      // next globalrev for monitoring is ready

      
      snde_debug(SNDE_DC_RECDB,"Issuing monitoring notifications");

      std::shared_ptr<globalrevision> complete_globalrev = globalrev;
      std::shared_ptr<globalrevision> last_complete_globalrev;
      std::shared_ptr<globalrev_mutable_lock> globalrev_mutable_lock_null; // permanently a nullptr

      while (complete_globalrev) {
	std::set<std::weak_ptr<monitor_globalrevs>,std::owner_less<std::weak_ptr<monitor_globalrevs>>> monitoring_deadptrs;

	last_complete_globalrev = complete_globalrev;

	// Mark complete_globalrev as now the latest ready globalrev
	std::atomic_store(&recdb_strong->_latest_ready_globalrev,complete_globalrev);

	
	std::shared_ptr<globalrev_mutable_lock> complete_globalrev_mutable_recordings_lock; 

	{
	  std::lock_guard<std::mutex> complete_globalrev_admin(complete_globalrev->admin);
	  complete_globalrev_mutable_recordings_lock = complete_globalrev->mutable_recordings_need_holder;
	}
	
	for (auto && monitor_globalrev_weak: recdb_strong->monitoring) {
	  std::shared_ptr<monitor_globalrevs> monitor_globalrev = monitor_globalrev_weak.lock();
	  if (!monitor_globalrev) {
	    // dead ptr, mark it as to be removed
	    monitoring_deadptrs.emplace(monitor_globalrev_weak);
	  } else {
	    // perform notification
	    {
	      std::lock_guard<std::mutex> monitor_admin(monitor_globalrev->admin);
	      if (monitor_globalrev->inhibit_mutable) {
		monitor_globalrev->pending.emplace(complete_globalrev->globalrev,std::make_tuple(complete_globalrev,complete_globalrev_mutable_recordings_lock));
	      } else {
		// no need to inhibit modifications to mutable waveforms; pass nullptr in place of mutable_recordings_need_holder
		monitor_globalrev->pending.emplace(complete_globalrev->globalrev,std::make_tuple(complete_globalrev,globalrev_mutable_lock_null));
	      }
	    }
	    monitor_globalrev->ready_globalrev.notify_all();
	  }	
	  
	}
	
	// remove all dead pointers from monitoring list
	for (auto && monitoring_deadptr: monitoring_deadptrs) {
	  recdb_strong->monitoring.erase(monitoring_deadptr);
	}


	// perform the quick notifications
	for (auto && quicknotify: recdb_strong->ready_globalrev_quicknotifies_called_recdb_locked) {
	  (*quicknotify)(recdb_strong,complete_globalrev);
	}
	
	// clear the reference in complete_globalrev to the mutable_recordings_need_holder
	// now once all of the monitoring is done, the globalrev_mutable_lock gets destroyed,
	// triggering the globalrev_mutablenotneeded_thread to requeue any blocked computations
	{
	  std::lock_guard<std::mutex> complete_globalrev_admin(complete_globalrev->admin);
	  complete_globalrev->mutable_recordings_need_holder = nullptr;
	}

	
	recdb_strong->monitoring_notify_globalrev = complete_globalrev->globalrev;
      
	// Is the next globalrev done?
	uint64_t nextrev_index = complete_globalrev->globalrev+1;

	std::map<uint64_t,std::shared_ptr<globalrevision>>::iterator nextglob_it;
	nextglob_it = recdb_strong->_globalrevs.find(nextrev_index);
	if (nextglob_it != recdb_strong->_globalrevs.end()) {
	  std::shared_ptr<globalrevision> next_globalrev = nextglob_it->second;
	    
	  bool all_ready = !next_globalrev->recstatus.defined_recordings.size() && !next_globalrev->recstatus.instantiated_recordings.size();

	  if (all_ready) {
	    complete_globalrev = next_globalrev;
	  } else {
	    complete_globalrev = nullptr;
	  }
	} else {
	  complete_globalrev = nullptr;
	}

	// loop back, notifying about the next globalrev if it is ready. 
      }
      
      // We can now remove any prior globalrevisions from the
      // recdatabase's _globalrevs map, as they are thoroughly
      // obsolete
      
      std::map<uint64_t,std::shared_ptr<globalrevision>>::iterator _gr_it;
      
      while ((_gr_it=recdb_strong->_globalrevs.begin()) != recdb_strong->_globalrevs.end() && _gr_it->first < last_complete_globalrev->globalrev) {
	snde_debug(SNDE_DC_RECDB,"Globalrev %llu removed from database",(unsigned long long)_gr_it->first);
	recdb_strong->_globalrevs.erase(_gr_it);
      }
    }
    
    
  }

  monitor_globalrevs::monitor_globalrevs(std::shared_ptr<globalrevision> first,bool inhibit_mutable) :
    next_globalrev_index(first->globalrev),
    inhibit_mutable(inhibit_mutable),
    active(true)
  {
    
  }

  std::tuple<std::shared_ptr<globalrevision>,std::shared_ptr<globalrev_mutable_lock>> monitor_globalrevs::wait_next_inhibit_mutable(std::shared_ptr<recdatabase> recdb)
  {
    std::unique_lock<std::mutex> monitor_admin(admin);
    if (!active) {
      throw snde_error("Waiting on inactive globalrev monitor");
    }

    std::map<uint64_t,std::tuple<std::shared_ptr<globalrevision>,std::shared_ptr<globalrev_mutable_lock>>>::iterator nextpending;
    while ((nextpending=pending.begin()) == pending.end()) {
      ready_globalrev.wait(monitor_admin);
    }
    
    std::tuple<std::shared_ptr<globalrevision>,std::shared_ptr<globalrev_mutable_lock>> retval = nextpending->second;
    pending.erase(nextpending);
    
    return retval;
  }


  std::shared_ptr<globalrevision> monitor_globalrevs::wait_next(std::shared_ptr<recdatabase> recdb)
  {
    std::tuple<std::shared_ptr<globalrevision>,std::shared_ptr<globalrev_mutable_lock>> globalrev_mutablelock = wait_next_inhibit_mutable(recdb);

    return std::get<0>(globalrev_mutablelock);
  }
  
  void monitor_globalrevs::close(std::shared_ptr<recdatabase> recdb)
  {
    std::unique_lock<std::mutex> monitor_admin(admin,std::defer_lock);
    {
      std::lock_guard<std::mutex> recdb_admin(recdb->admin);
      monitor_admin.lock();
      active = false;
      recdb->monitoring.erase(shared_from_this());
    }
    pending.clear();
    
  }

  
};
