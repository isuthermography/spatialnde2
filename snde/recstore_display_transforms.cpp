#include "snde/recstore_display_transforms.hpp"
#include "snde/notify.hpp"
#include "snde/display_requirements.hpp"

namespace snde {

  static void _flattenreq_accumulate_reqs(std::shared_ptr<display_requirement> req,std::vector<std::shared_ptr<display_requirement>> *accumulator)
  {
    accumulator->push_back(req);

    for (auto && subreq: req->sub_requirements) {
      if (subreq) {
	_flattenreq_accumulate_reqs(subreq,accumulator);
      } else {
	snde_warning("recstore_display_transforms.cpp:_flattenreq_accumulate_reqs(): Got null sub-requirement of %s",req->channelpath.c_str());
      }
    }
  }
  
  static std::vector<std::shared_ptr<display_requirement>> flatten_requirements(const std::map<std::string,std::shared_ptr<display_requirement>> &requirements)
  {
    std::vector<std::shared_ptr<display_requirement>> retval; 
    for (auto && chanpath_req: requirements) {
      _flattenreq_accumulate_reqs(chanpath_req.second,&retval);
    }
    return retval;
  }


  static std::unordered_map<std::pair<std::string,rendermode_ext>,std::shared_ptr<display_requirement>,chanpathmodeext_hash> merge_requirements(const std::vector<std::shared_ptr<display_requirement>> &requirements)
    {
      // Merge together all requirements that are asking for exactly the same thing
      // (e.g. more than one rendered object might need exactly the same texture)

      // The renderer still keeps the original tree, but this is used to uniquely
      // define the math functions, etc. to achieve the requirements

      // the merge criteria are the recording name and the extended rendermode

      std::unordered_map<std::pair<std::string,rendermode_ext>,std::shared_ptr<display_requirement>,chanpathmodeext_hash> retval;

      for (auto && requirement: requirements) {
	std::pair<std::string,rendermode_ext> key = std::make_pair(requirement->channelpath,requirement->mode);

	std::unordered_map<std::pair<std::string,rendermode_ext>,std::shared_ptr<display_requirement>,chanpathmodeext_hash>::iterator existing = retval.find(key);
	if (existing != retval.end()) {
	  // found matching entry already
	  // verify compatibility

	  // check that renderable_channelpaths match
	  if (requirement->renderable_channelpath != existing->second->renderable_channelpath) {
	    // would have pointer equality if they are both nullptr
	    if (!requirement->renderable_channelpath && existing->second->renderable_channelpath) {
	      throw snde_error("recstore_display_transforms::update()/merge_requirements(): Attempting to merge requirements with incompatible renderable channelpaths nullptr and %s",existing->second->renderable_channelpath->c_str());
	      
	    }
	    if (requirement->renderable_channelpath && !existing->second->renderable_channelpath) {
	      throw snde_error("recstore_display_transforms::update()/merge_requirements(): Attempting to merge requirements with incompatible renderable channelpaths %s and nullptr",requirement->renderable_channelpath->c_str());
	    }
	    
	    // neither can be nullptr at this point	    
	    if (*requirement->renderable_channelpath != *existing->second->renderable_channelpath) {
	      throw snde_error("recstore_display_transforms::update()/merge_requirements(): Attempting to merge requirements with incompatible renderable channelpaths %s and %s",requirement->renderable_channelpath->c_str(),existing->second->renderable_channelpath->c_str());
	    }
	  }


	  // check that renderable_functions match
	  if (requirement->renderable_function != existing->second->renderable_function) {
	    // would have pointer equality if they are both nullptr
	    if (!requirement->renderable_function && existing->second->renderable_function) {
	      throw snde_error("recstore_display_transforms::update()/merge_requirements(): Attempting to merge requirements with incompatible renderable functions nullptr and %s",existing->second->renderable_function->result_channel_paths.at(0)->c_str());
	      
	    }
	    if (requirement->renderable_function && !existing->second->renderable_function) {
	      throw snde_error("recstore_display_transforms::update()/merge_requirements(): Attempting to merge requirements with incompatible renderable functions %s and nullptr",requirement->renderable_function->result_channel_paths.at(0)->c_str());
	      
	    }
	    
	    // neither can be nullptr at this point	    
	    if (*requirement->renderable_function != *existing->second->renderable_function) { // uses operator== on the instantiated_math_function

	      throw snde_error("recstore_display_transforms::update()/merge_requirements(): Attempting to merge requirements with incompatible renderable functions %s and %s",requirement->renderable_function->result_channel_paths.at(0)->c_str(),existing->second->renderable_function->result_channel_paths.at(0)->c_str());
	      
	    }
	  }
	  // (sub_requirements should already have been flattened at this point. They are still present,
	  // but we will get to them via the flat iteration)
	} else {
	  // At this point there is no existing match 
	  // (If there had been a key match and the existing data didn't match
	  // we would have thrown an exception above)

	  // So we need to add the new entry

	  retval.emplace(key,requirement);
	}
      }

      return retval; // all duplicate requirements eliminated
      
    }

  recstore_display_transforms::recstore_display_transforms() :
    starting_revision(rss_get_unique())
  {

  }
  
  void recstore_display_transforms::update(std::shared_ptr<recdatabase> recdb,std::shared_ptr<globalrevision> globalrev,const std::map<std::string,std::shared_ptr<display_requirement>> &requirements)
  // define a new with_display_transforms member based on a new globalrev and a list of display requirements.
  // the globalrev is presumed to be fullyready.
  // Likewise the previous rss in with_display_transforms is also presumed to be fullyready
  {
    std::shared_ptr<globalrevision> previous_globalrev = latest_ready_globalrev;
    std::shared_ptr<recording_set_state> previous_with_transforms = with_display_transforms;
    std::map<std::string,std::shared_ptr<channelconfig>> all_channels_by_name;

    std::vector<std::shared_ptr<display_requirement>> flattened_requirements = flatten_requirements(requirements);

    // need to merge display_requirements according to channelpath and mode.
    // while doing so should verify that renderable_channelpaths match and renderable_function definition and parameters match. 
    std::unordered_map<std::pair<std::string,rendermode_ext>,std::shared_ptr<display_requirement>,chanpathmodeext_hash> merged_requirements = merge_requirements(flattened_requirements);
    
    latest_ready_globalrev = globalrev;

    if (!previous_globalrev) {
      // dummy previous globalrev
      previous_globalrev=std::make_shared<globalrevision>(0,nullptr,recdb,instantiated_math_database(),std::map<std::string,channel_state>(),nullptr,0);
    }

    if (!previous_with_transforms) {
      // dummy previous recording_set_state
      previous_with_transforms = std::make_shared<recording_set_state>(recdb,instantiated_math_database(),std::map<std::string,channel_state>(),nullptr,0,0);
    }

    std::unordered_set<std::shared_ptr<channelconfig>> unknownchanged_channels;
    std::unordered_set<std::shared_ptr<channelconfig>> changed_channels_need_dispatch;
    std::unordered_set<std::shared_ptr<channelconfig>> unchanged_channels;
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

    auto globalrev_channel_map = globalrev->recstatus.channel_map;
    
    // First from the current set of channels out of globalrev
    for (auto && channame_chanstate: *globalrev_channel_map) {
      initial_channel_map.emplace(std::piecewise_construct,
				  std::forward_as_tuple(channame_chanstate.first),
				  std::forward_as_tuple(channame_chanstate.second.chan,channame_chanstate.second.config,channame_chanstate.second.rec(),false));

      all_channels_by_name.emplace(std::piecewise_construct,
				   std::forward_as_tuple(channame_chanstate.first),
				   std::forward_as_tuple(channame_chanstate.second.config));

      
      // Check to see here if there was a change from previous_globalrev to globalrev
      
      // If so it should go into changed_channels_need_dispatch
      // Otherwise should go into unchanged_channels and NOT unknownchanged_channels. 

      auto prev_it = previous_globalrev->recstatus.channel_map->find(channame_chanstate.first);
      if (prev_it != previous_globalrev->recstatus.channel_map->end() && prev_it->second.rec() == channame_chanstate.second.rec()) {
	// channel is unchanged
	snde_debug(SNDE_DC_DISPLAY,"Channel %s is unchanged",channame_chanstate.first.c_str());
	unchanged_channels.emplace(channame_chanstate.second.config);
      } else {
	// channel modified
	snde_debug(SNDE_DC_DISPLAY,"Channel %s is modified",channame_chanstate.first.c_str());
	changed_channels_need_dispatch.emplace(channame_chanstate.second.config);
      }

      
      // also the new channel_map pointer should be placed into the completed_recordings map of the new with_display_transforms's recstatus
      // (done below)
    }
    


    // ... and second from the display requirements
    for (auto && dispkey_dispreq: merged_requirements) {
      std::shared_ptr<display_requirement> dispreq=dispkey_dispreq.second;
      if (dispreq->renderable_function) {
	assert(dispreq->channelpath != *dispreq->renderable_channelpath);

	snde_debug(SNDE_DC_DISPLAY,"recstore_display_transforms::update() got renderable_function %s->%s",dispreq->channelpath.c_str(),dispreq->renderable_channelpath->c_str());
	
	std::shared_ptr<channelconfig> renderableconfig;

	// search for pre-existing channel in previous_with_transforms
	auto preexist_it = previous_with_transforms->recstatus.channel_map->find(*dispreq->renderable_channelpath);
	// to reuse, we have to find something of the same name where the math_fcns compare by value, indicating the same function and parameters
	if (preexist_it != previous_with_transforms->recstatus.channel_map->end() && *preexist_it->second.config->math_fcn == *dispreq->renderable_function) {
	  
	  snde_debug(SNDE_DC_DISPLAY,"recstore_display_transforms::update() found old config");

	  // reuse old config
	  renderableconfig = preexist_it->second.config;

	  // mark this channel as maybe needing data
	  unknownchanged_channels.emplace(renderableconfig);
	  
	  // mark this function as maybe needing to execute
	  unknownchanged_math_functions.emplace(renderableconfig->math_fcn);


	  
	} else {
	  // need to make new config
	  snde_debug(SNDE_DC_DISPLAY,"recstore_display_transforms::update() making new config");

	  renderableconfig = std::make_shared<channelconfig>(*dispreq->renderable_channelpath,
											    "recstore_display_transform",
							     
											    true, // hidden
											    nullptr); // storage_manager
	  renderableconfig->math=true;
	  renderableconfig->math_fcn = dispreq->renderable_function;
	  renderableconfig->ondemand=true;
	  renderableconfig->data_mutable=false; // don't support mutable rendering functions for now... maybe in the future

	  // mark this channel as needing data
	  changed_channels_need_dispatch.emplace(renderableconfig);
	  
	  // mark this function as needing to execute
	  changed_math_functions.emplace(renderableconfig->math_fcn);
	}

	// add to initial_mathdb
	initial_mathdb.defined_math_functions.emplace(*dispreq->renderable_channelpath,renderableconfig->math_fcn);

	// Get a class reserved_channel to represent this math function
	std::shared_ptr<reserved_channel> rdt_channel;
	std::unordered_map<std::string,std::shared_ptr<reserved_channel>>::iterator existing_channel = rdt_channels.find(*dispreq->renderable_channelpath);
	if (existing_channel != rdt_channels.end()) {
	  rdt_channel = existing_channel->second;
	  {
	    std::lock_guard<std::mutex> rdt_admin(rdt_channel->admin);
	    rdt_channel->begin_atomic_proposed_config_update<channelconfig>();
	    rdt_channel->end_atomic_proposed_config_update(renderableconfig);

	    rdt_channel->begin_atomic_realized_config_update<channelconfig>();
	    rdt_channel->end_atomic_realized_config_update(renderableconfig);
	  }
	    
	} else {
	  rdt_channel=std::make_shared<reserved_channel>();
	  rdt_channel->begin_atomic_proposed_config_update<channelconfig>();
	  rdt_channel->end_atomic_proposed_config_update(renderableconfig);
	  rdt_channel->begin_atomic_realized_config_update<channelconfig>();
	  rdt_channel->end_atomic_realized_config_update(renderableconfig);
	  std::shared_ptr<channel> pseudochannel=std::make_shared<channel>(*dispreq->renderable_channelpath,rdt_channel);
	  pseudochannel->begin_atomic_realized_owner_update();
	  pseudochannel->end_atomic_realized_owner_update(rdt_channel);
	  rdt_channel->chan=pseudochannel;
	  
	  rdt_channel->chan->latest_revision = starting_revision;
	  
	}
	
	// add to initial_channel_map
	initial_channel_map.emplace(std::piecewise_construct,
				    std::forward_as_tuple(*dispreq->renderable_channelpath),
				    std::forward_as_tuple(rdt_channel,renderableconfig,nullptr,false));

	// also the new channel_map pointer should be placed into the defined_recordings map of the rss's recstatus

	all_channels_by_name.emplace(std::piecewise_construct,
				     std::forward_as_tuple(*dispreq->renderable_channelpath),
				     std::forward_as_tuple(renderableconfig));
	

      }


      
    
    }

    
    // build a class recording_set_state using this new channel_map

    
    with_display_transforms = std::make_shared<recording_set_state>(recdb,initial_mathdb,initial_channel_map,nullptr,previous_globalrev->globalrev,rss_get_unique());


    // We have to play transaction manager here because there isn't an actual transaction involved.
    with_display_transforms->our_state_reference = std::make_shared<rss_reference>(with_display_transforms);

    // snde_warning("rss 0x%lx gets new rss_reference", (unsigned long) with_display_transforms.get());
    // We need to make sure that our_state_reference goes away when the rss is complete. Use clear_osr_notify to trigger a call back.
    std::shared_ptr<recording_set_state> with_display_transforms_ref = with_display_transforms;
    
    with_display_transforms->mathstatus.math_functions->_rebuild_dependency_map(recdb,true); // (not automatically done on construction)

    // For everything we copied in from the globalrev (above),
    // mark it in the completed_recordings map
    //auto globalrev_channel_map = globalrev->recstatus.channel_map; // already defined above
    for (auto && channame_chanstate: *globalrev_channel_map) {
      auto wdt_chanmap_iter = with_display_transforms->recstatus.channel_map->find(channame_chanstate.first);

      assert(wdt_chanmap_iter != with_display_transforms->recstatus.channel_map->end());

      if (!wdt_chanmap_iter->second.recording_is_complete(false)) {
	// must be mdonly
	with_display_transforms->recstatus.metadataonly_recordings.emplace(channame_chanstate.second.config,&wdt_chanmap_iter->second);
      } else {
	with_display_transforms->recstatus.completed_recordings.emplace(channame_chanstate.second.config,&wdt_chanmap_iter->second);
      }
    }

    // For everything from the requirements, mark it in the defined_recordings map
    for (auto && dispkey_dispreq: merged_requirements) {
      std::shared_ptr<display_requirement> dispreq=dispkey_dispreq.second;
      if (dispreq->renderable_function) {
	auto wdt_chanmap_iter = with_display_transforms->recstatus.channel_map->find(*dispreq->renderable_channelpath);
	assert(wdt_chanmap_iter != with_display_transforms->recstatus.channel_map->end());
	
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
    changed_channels_need_dispatch.reserve(changed_channels_need_dispatch.size()+unknownchanged_channels.size()+unchanged_channels.size());
    
    std::unordered_set<std::shared_ptr<channelconfig>> changed_channels_dispatched;
    changed_channels_dispatched.reserve(changed_channels_need_dispatch.size()+unknownchanged_channels.size()+unchanged_channels.size());
    
    // all modified channels/recordings from the globalrevs have been put in changed_channels_need_dispatched and removed them from unchanged_channels
    // if they have changed between the two globalrevs 
    // channels which haven't changed have been imported into the globalrev, removed from unknownchanged_channels, 
    // and placed into the unchanged_channels list. 

    
    // Pull all channels from the globalrev, taking them out of unknownchanged_channels
    // and putting them in to ...
    
    // set of ready channels
    std::unordered_set<channel_state *> ready_channels; // references into the new_rss->recstatus.channel_map

    std::vector<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>> ready_to_execute;
    std::set<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>>,mncn_lessthan> may_need_completion_notification;
    bool all_ready=false;

    snde_debug(SNDE_DC_DISPLAY,"recstore_display_transforms::update calling build_rss_from_functions_and_channels() with %d unknownchanged_math_functions and %d changed_math_functions and %d ccnd and %d ccd",unknownchanged_math_functions.size(),changed_math_functions.size(),changed_channels_need_dispatch.size(),changed_channels_dispatched.size());

    
    build_rss_from_functions_and_channels(recdb,
					  previous_with_transforms,
					  with_display_transforms,
					  all_channels_by_name,
					  // set of channels definitely changed, according to whether we've dispatched them in our graph search
					  // for possibly dependent channels 
					  &changed_channels_need_dispatch,
					  &changed_channels_dispatched,
					  // set of channels known to be unchanged (pass as empty)
					  &unchanged_channels,
					  // set of channels not yet known to be changed
					  &unknownchanged_channels,
					  // set of math functions not known to be changed or unchanged
					  &unknownchanged_math_functions,
					  // set of math functions known to be (definitely) changed
					  &changed_math_functions,
					  &explicitly_updated_channels,
					  &ready_channels,
					  &ready_to_execute,
					  &may_need_completion_notification,
					  &all_ready,
					  true); // enable_ondemand

    snde_debug(SNDE_DC_DISPLAY,"recstore_display_transforms::update build_rss_from_functions_and_channels() complete with %d unknownchanged_math_functions and %d changed_math_functions",unknownchanged_math_functions.size(),changed_math_functions.size());

    // with_display_transforms->_update_recstatus__rss_admin_transaction_admin_locked()
    // isn't needed here because in the math calculations, the resulting
    // rss pointer has been known from the beginning, so the state should
    // be in the correct bin already. 

    // Perform notifies that unchanged copied recordings from prior revs are now ready
    // (and that with_display_transforms is ready if there is nothing pending!)
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

    // get notified when with_display_transforms is ready so that we can clear our_state_reference
    std::shared_ptr<callback_channel_notify> clear_osr_notify = std::make_shared<callback_channel_notify>(std::vector<std::string>(),std::vector<std::string>(),true,[with_display_transforms_ref] (){with_display_transforms_ref->our_state_reference = nullptr;});
    clear_osr_notify->apply_to_rss(with_display_transforms);

    
    // Run any possibly needed completion notifications
    for (auto && complete_rss_complete_execfunc: may_need_completion_notification) {
      std::shared_ptr<recording_set_state> complete_rss;
      std::shared_ptr<math_function_execution> complete_execfunc;

      std::tie(complete_rss,complete_execfunc) = complete_rss_complete_execfunc;
      execution_complete_notify_single_referencing_rss(recdb,complete_execfunc,complete_execfunc->mdonly,true,complete_rss);
    }

    
    // Check if everything is done; issue notification
    if (all_ready) {
      std::unique_lock<std::mutex> rss_admin(with_display_transforms->admin);
      std::unordered_set<std::shared_ptr<channel_notify>> recordingset_complete_notifiers=std::move(with_display_transforms->recordingset_complete_notifiers);
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
