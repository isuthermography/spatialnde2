#include "snde/recstore_display_transforms.hpp"
#include "snde/notify.hpp"

namespace snde {

  void recstore_display_transforms::update(std::shared_ptr<recdatabase> recdb,std::shared_ptr<globalrevision> globalrev,const std::vector<display_requirement> &requirements)
  // define a new with_display_transforms member based on a new globalrev and a list of display requirements.
  // the globalrev is presumed to be fullyready.
  // Likewise the previous rss in with_display_transforms is also presumed to be fullyready
  {
    std::shared_ptr<globalrevision> previous_globalrev = latest_globalrev;
    std::shared_ptr<recording_set_state> previous_with_transforms = with_display_transforms;
    std::map<std::string,std::shared_ptr<channelconfig>> all_channels_by_name;

    latest_globalrev = globalrev;

    if (!previous_globalrev) {
      
      previous_globalrev=std::make_shared<globalrevision>(0,nullptr,recdb,instantiated_math_database(),std::map<std::string,channel_state>(),nullptr);
    }

    if (!previous_with_transforms) {
      previous_with_transforms = std::make_shared<recording_set_state>(recdb,instantiated_math_database(),std::map<std::string,channel_state>(),nullptr);
    }

    std::unordered_set<std::shared_ptr<channelconfig>> unknownchanged_channels;
    std::unordered_set<std::shared_ptr<channelconfig>> changed_channels_need_dispatch;
    std::unordered_set<std::shared_ptr<channelconfig>> ignored_channels;
    std::unordered_set<std::shared_ptr<channelconfig>> explicitly_updated_channels; // no explicitly updated channels in this case


    // set of math functions not known to be changed or unchanged
    std::unordered_set<std::shared_ptr<instantiated_math_function>> unknownchanged_math_functions; 
    
    // set of math functions known to be (definitely) changed
    std::unordered_set<std::shared_ptr<instantiated_math_function>> changed_math_functions; 


    // Should we combine the full set of math functions or just use the display ones?
    // Just the display ones for now
    instantiated_math_database initial_mathdb;


    // assemble the channel_map
    std::map<std::string,channel_state> initial_channel_map;

    // First from the current set of channels out of globalrev
    for (auto && channame_chanstate: globalrev->recstatus.channel_map) {
      initial_channel_map.emplace(std::piecewise_construct,
				  std::forward_as_tuple(channame_chanstate.first),
				  std::forward_as_tuple(channame_chanstate.second._channel,channame_chanstate.second.config,channame_chanstate.second.rec(),false));

      all_channels_by_name.emplace(std::piecewise_construct,
				   std::forward_as_tuple(channame_chanstate.first),
				   std::forward_as_tuple(channame_chanstate.second.config));

      
      // Check to see here if there was a change from previous_globalrev to globalrev
      
      // If so it should go into changed_channels_need_dispatch
      // Otherwise should go into ignored_channels and NOT unknownchanged_channels. 

      auto prev_it = previous_globalrev->recstatus.channel_map.find(channame_chanstate.first);
      if (prev_it != previous_globalrev->recstatus.channel_map.end() && prev_it->second.rec() == channame_chanstate.second.rec()) {
	// channel is unchanged
	ignored_channels.emplace(channame_chanstate.second.config);
      } else {
	// channel modified
	changed_channels_need_dispatch.emplace(channame_chanstate.second.config);
      }

      
      // also the new channel_map pointer should be placed into the completed_recordings map of the new with_display_transforms's recstatus
      // (done below)
    }
    


    // ... and second from the display requirements
    for (auto && dispreq: requirements) {
      if (dispreq.renderable_function) {
	assert(dispreq.channelpath != dispreq.renderable_channelpath);

	std::shared_ptr<channelconfig> renderableconfig;

	// search for pre-existing channel in previous_with_transforms
	auto preexist_it = previous_with_transforms->recstatus.channel_map.find(dispreq.renderable_channelpath);
	// to reuse, we have to find something of the same name where the math_fcns compare by value, indicating the same function and parameters
	if (preexist_it != previous_with_transforms->recstatus.channel_map.end() && *preexist_it->second.config->math_fcn == *dispreq.renderable_function) {
	  // reuse old config
	  renderableconfig = preexist_it->second.config;

	  // mark this channel as maybe needing data
	  unknownchanged_channels.emplace(renderableconfig);
	  
	  // mark this function as maybe needing to execute
	  unknownchanged_math_functions.emplace(renderableconfig->math_fcn);


	  
	} else {
	  // need to make new config
	  renderableconfig = std::make_shared<channelconfig>(dispreq.renderable_channelpath,
											    "recstore_display_transform",
											    (void *)this,
											    true, // hidden
											    nullptr); // storage_manager
	  renderableconfig->math=true;
	  renderableconfig->math_fcn = dispreq.renderable_function;
	  renderableconfig->ondemand=true;
	  renderableconfig->data_mutable=false; // don't support mutable rendering functions for now... maybe in the future

	  // mark this channel as needing data
	  changed_channels_need_dispatch.emplace(renderableconfig);
	  
	  // mark this function as needing to execute
	  changed_math_functions.emplace(renderableconfig->math_fcn);
	}

	// add to initial_mathdb
	initial_mathdb.defined_math_functions.emplace(dispreq.renderable_channelpath,renderableconfig->math_fcn);
	
	// add to initial_channel_map
	initial_channel_map.emplace(std::piecewise_construct,
				    std::forward_as_tuple(dispreq.renderable_channelpath),
				    std::forward_as_tuple(nullptr,renderableconfig,nullptr,false));

	// also the new channel_map pointer should be placed into the defined_recordings map of the rss's recstatus

	all_channels_by_name.emplace(std::piecewise_construct,
				     std::forward_as_tuple(dispreq.renderable_channelpath),
				     std::forward_as_tuple(renderableconfig));
	

      }


      
    
    }

    
    // build a class recording_set_state using this new channel_map

    
    with_display_transforms = std::make_shared<recording_set_state>(recdb,initial_mathdb,initial_channel_map,nullptr);
    with_display_transforms->mathstatus.math_functions->_rebuild_dependency_map(); // (not automatically done on construction)

    // For everything we copied in from the globalrev (above),
    // mark it in the completed_recordings map
    for (auto && channame_chanstate: globalrev->recstatus.channel_map) {
      auto wdt_chanmap_iter = with_display_transforms->recstatus.channel_map.find(channame_chanstate.first);

      assert(wdt_chanmap_iter != with_display_transforms->recstatus.channel_map.end());

      if (!wdt_chanmap_iter->second.recording_is_complete(false)) {
	// must be mdonly
	with_display_transforms->recstatus.metadataonly_recordings.emplace(channame_chanstate.second.config,&wdt_chanmap_iter->second);
      } else {
	with_display_transforms->recstatus.completed_recordings.emplace(channame_chanstate.second.config,&wdt_chanmap_iter->second);
      }
    }

    // For everything from the requirements, mark it in the defined_recordings map
    for (auto && dispreq: requirements) {
      if (dispreq.renderable_function) {
	auto wdt_chanmap_iter = with_display_transforms->recstatus.channel_map.find(dispreq.renderable_channelpath);
	assert(wdt_chanmap_iter != with_display_transforms->recstatus.channel_map.end());
	
	with_display_transforms->recstatus.defined_recordings.emplace(wdt_chanmap_iter->second.config,&wdt_chanmap_iter->second);
	
      }
    }
    
    // defined unknownchanged_channels (every display channel)
    // defined unknownchanged_math_functions (every display math function)

    // set of channels definitely changed, according to whether we've dispatched them in our graph search
    // for possibly dependent channels 
    //std::unordered_set<std::shared_ptr<channelconfig>> changed_channels_need_dispatch;
    //std::unordered_set<std::shared_ptr<channelconfig>> changed_channels_dispatched;


    
    // make sure hash tables won't rehash and screw up iterators or similar
    changed_channels_need_dispatch.reserve(changed_channels_need_dispatch.size()+unknownchanged_channels.size()+ignored_channels.size());
    
    std::unordered_set<std::shared_ptr<channelconfig>> changed_channels_dispatched;
    changed_channels_dispatched.reserve(changed_channels_need_dispatch.size()+unknownchanged_channels.size()+ignored_channels.size());
    
    // all modified channels/recordings from the globalrevs have been put in changed_channels_need_dispatched and removed them from ignored_channels
    // if they have changed between the two globalrevs 
    // channels which haven't changed have been imported into the globalrev, removed from unknownchanged_channels, 
    // and placed into the ignored_channels list. 

    
    // Pull all channels from the globalrev, taking them out of unknownchanged_channels
    // and putting them in to ...
    
    // set of ready channels
    std::unordered_set<channel_state *> ready_channels; // references into the new_rss->recstatus.channel_map

    std::vector<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>> ready_to_execute;
    bool all_ready=false;


    build_rss_from_functions_and_channels(recdb,
					  previous_with_transforms,
					  with_display_transforms,
					  all_channels_by_name,
					  // set of channels definitely changed, according to whether we've dispatched them in our graph search
					  // for possibly dependent channels 
					  &changed_channels_need_dispatch,
					  &changed_channels_dispatched,
					  // set of channels not yet known to be changed
					  &unknownchanged_channels,
					  // set of math functions not known to be changed or unchanged
					  &unknownchanged_math_functions,
					  // set of math functions known to be (definitely) changed
					  &changed_math_functions,
					  &explicitly_updated_channels,
					  &ready_channels,
					  &ready_to_execute,&all_ready);
    // Perform notifies that unchanged copied recordings from prior revs are now ready
    // (and that globalrev is ready if there is nothing pending!)
    for (auto && readychan : ready_channels) { // readychan is a channel_state &
      readychan->issue_nonmath_notifications(with_display_transforms);
    }

    // queue up everything we marked as ready_to_execute
    for (auto && ready_rss_ready_fcn: ready_to_execute) {
      // Need to queue as a pending_computation
      std::shared_ptr<recording_set_state> ready_rss;
      std::shared_ptr<instantiated_math_function> ready_fcn;

      std::tie(ready_rss,ready_fcn) = ready_rss_ready_fcn;
      recdb->compute_resources->queue_computation(recdb,ready_rss,ready_fcn);
    }

  
    // Check if everything is done; issue notification
    if (all_ready) {
      std::unique_lock<std::mutex> rss_admin(globalrev->admin);
      std::unordered_set<std::shared_ptr<channel_notify>> recordingset_complete_notifiers=std::move(globalrev->recordingset_complete_notifiers);
      with_display_transforms->recordingset_complete_notifiers.clear();
      rss_admin.unlock();

      for (auto && channel_notify_ptr: recordingset_complete_notifiers) {
	channel_notify_ptr->notify_recordingset_complete();
      }
    }

    //// get notified so as to remove entries from available_compute_resource_database blocked_list
    //// once the previous globalrev is complete.
    //if (previous_globalrev) {
    //  std::shared_ptr<_previous_globalrev_nolongerneeded_notify> prev_nolongerneeded_notify = std::make_shared<_previous_globalrev_nolongerneeded_notify>(recdb,previous_globalrev,globalrev);
    //
    //  prev_nolongerneeded_notify->apply_to_rss(previous_globalrev);
    //}

    // Set up notification when this globalrev is complete
    // So that we can remove obsolete entries from the _globalrevs
    // database and so we can notify anyone monitoring
    // that there is a ready globalrev.
    
    //std::shared_ptr<_globalrev_complete_notify> complete_notify=std::make_shared<_globalrev_complete_notify>(recdb,globalrev);
    
    //complete_notify->apply_to_rss(globalrev);



    
  }
  
  

};
