#include "snde/recmath_compute_resource.hpp"
#include "snde/recstore.hpp"
#include "snde/recmath.hpp"

namespace snde {
  compute_resource_option::compute_resource_option(unsigned type,std::set<std::string> execution_tags, size_t metadata_bytes,size_t data_bytes) :
    type(type),
    execution_tags(execution_tags),
    metadata_bytes(metadata_bytes),
    data_bytes(data_bytes)
  {

  }


  compute_resource_option_cpu::compute_resource_option_cpu(std::set<std::string> execution_tags,
							   size_t metadata_bytes,
							   size_t data_bytes,
							   snde_float64 flops,
							   size_t max_effective_cpu_cores,
							   size_t useful_cpu_cores) :
    compute_resource_option(SNDE_CR_CPU,execution_tags,metadata_bytes,data_bytes),
    flops(flops),
    max_effective_cpu_cores(max_effective_cpu_cores),
    useful_cpu_cores(useful_cpu_cores)
  {

  }

  bool compute_resource_option_cpu::compatible_with(std::shared_ptr<available_compute_resource> available)
  {
    if (type==available->type) {
      assert(std::dynamic_pointer_cast<available_compute_resource_cpu>(available));
      return true;
    }
    return false;
  }


  _compute_resource_option_cpu_combined::_compute_resource_option_cpu_combined(std::set<std::string> execution_tags,
									       size_t metadata_bytes,
									       size_t data_bytes,
									       snde_float64 flops,
									       size_t max_effective_cpu_cores,
									       size_t useful_cpu_cores,
									       std::shared_ptr<compute_resource_option> orig,
									       std::shared_ptr<assigned_compute_resource> orig_assignment) :
    compute_resource_option_cpu(execution_tags,metadata_bytes,data_bytes,flops,
				max_effective_cpu_cores,useful_cpu_cores),
    orig(orig),
    orig_assignment(orig_assignment)
  {
    
  }


#define SNDE_FRS_INVALID 0  
#define SNDE_FRS_DEFINED (1<<1)  // Can binary OR these numbers together for FromResultState but NOT ToResultState
#define SNDE_FRS_INSTANTIATED (1<<2)
#define SNDE_FRS_METADATAONLY (1<<3)
#define SNDE_FRS_COMPLETED (1<<4)
#define SNDE_FRS_ALL (1<<5) // ToResultState only
#define SNDE_FRS_ANY (SNDE_FRS_DEFINED|SNDE_FRS_INSTANTIATED|SNDE_FRS_METADATAONLY|SNDE_FRS_COMPLETED) // FromResultState only

  static void _transfer_function_result_state_single_locked_referencing_rss(std::shared_ptr<math_function_execution> execfunc,int FromResultState,int ToResultState, const std::vector<std::shared_ptr<recording_base>> &result_channel_recs,std::shared_ptr<recording_set_state> single_referencing_rss,std::unique_lock<std::mutex> *referencing_rss_lock)
  // MUST be called with the single_referencing_rss locked (provide the unique_lock as referencing_rss_lock
  // MAY be called with execfunc locked. Remember execfunc is after the rss's in the locking order
  {

    snde_debug(SNDE_DC_RECMATH,"_transfer_function_result_state_single_locked_referencing_rss(execfunc=0x%llx,single_referencing_rss=0x%llx,ToResultState=%d)",(unsigned long long)execfunc.get(),(unsigned long long)single_referencing_rss.get(),(int)ToResultState);
    
    for (size_t cnt=0;cnt < execfunc->inst->result_channel_paths.size(); cnt++) {
      std::shared_ptr<std::string> result_channel_path_ptr = execfunc->inst->result_channel_paths.at(cnt);
      
      if (result_channel_path_ptr) {
	channel_state &referencing_rss_channel_state = single_referencing_rss->recstatus.channel_map->at(recdb_path_join(execfunc->inst->channel_path_context,*result_channel_path_ptr));
	// none should have preassigned recordings --not true (?) because parallel update or end_transaction() could be going on? Actually no, due to locking and synchronization
	//assert(!referencing_rss_channel_state.rec());
	size_t numerased;
	
	if (FromResultState & SNDE_FRS_DEFINED) {
	  numerased = single_referencing_rss->recstatus.defined_recordings.erase(referencing_rss_channel_state.config);
	  if (FromResultState == SNDE_FRS_DEFINED) {
	    assert(numerased==1); 
	  }	  
	}
	
	if (FromResultState & SNDE_FRS_INSTANTIATED) {
	  numerased = single_referencing_rss->recstatus.instantiated_recordings.erase(referencing_rss_channel_state.config);
	  if (FromResultState == SNDE_FRS_INSTANTIATED) {
	    assert(numerased==1); 
	  }	  
	}
	
	if (FromResultState & SNDE_FRS_METADATAONLY) {
	  numerased = single_referencing_rss->recstatus.defined_recordings.erase(referencing_rss_channel_state.config);
	  if (FromResultState == SNDE_FRS_METADATAONLY) {
	    assert(numerased==1); 
	  }	  
	}
	
	if (ToResultState == SNDE_FRS_INSTANTIATED) {
	  single_referencing_rss->recstatus.instantiated_recordings.emplace(referencing_rss_channel_state.config,&referencing_rss_channel_state);
	}
	if (ToResultState == SNDE_FRS_METADATAONLY) {
	  single_referencing_rss->recstatus.metadataonly_recordings.emplace(referencing_rss_channel_state.config,&referencing_rss_channel_state);
	}
	if (ToResultState == SNDE_FRS_COMPLETED || ToResultState == SNDE_FRS_ALL) {
	  
	  single_referencing_rss->recstatus.completed_recordings.emplace(referencing_rss_channel_state.config,&referencing_rss_channel_state);
	}
	
	if (ToResultState == SNDE_FRS_INSTANTIATED || ToResultState==SNDE_FRS_ALL || (ToResultState == SNDE_FRS_COMPLETED && (FromResultState & SNDE_FRS_DEFINED))) {
	  referencing_rss_channel_state.end_atomic_rec_update(result_channel_recs.at(cnt));
	  //referencing_rss_channel_state.revision = std::make_shared<uint64_t>(result_channel_recs.at(cnt)->info->revision); (now implicit in the above)
	}
      }
    }

    // transfer completed flag to this RSS 
    if (ToResultState == SNDE_FRS_ALL || ToResultState == SNDE_FRS_COMPLETED) {
      math_function_status &referencing_rss_function_status = single_referencing_rss->mathstatus.function_status.at(execfunc->inst);
      referencing_rss_function_status.complete = true; 
    }
    
  }
  

  void join_rss_into_function_result_state(std::shared_ptr<math_function_execution> execfunc,std::shared_ptr<recording_set_state> source_rss,std::shared_ptr<recording_set_state> new_rss)
  // source_rss must already be on the execfunc's referencing_rss list
  {

    snde_debug(SNDE_DC_RECMATH,"_join_rss_into_function_result_state(execfunc=0x%llx,source_rss=0x%llx,new_rss=0x%llx)",(unsigned long long)execfunc.get(),(unsigned long long)source_rss.get(),(unsigned long long)new_rss.get());

    std::unique_lock<std::mutex> new_rss_admin(new_rss->admin);

    int FromResultState=SNDE_FRS_ANY;
    int ToResultState=SNDE_FRS_DEFINED;

    std::lock_guard<std::mutex> execfunc_admin(execfunc->admin);
    execfunc->referencing_rss.emplace(new_rss);

    if (execfunc->instantiated) {
      ToResultState = SNDE_FRS_INSTANTIATED;
    }
    if (execfunc->metadataonly_complete) {
      ToResultState = SNDE_FRS_METADATAONLY;
    }
    if (execfunc->fully_complete) {
      ToResultState=SNDE_FRS_COMPLETED;
    }


    std::vector<std::shared_ptr<recording_base>> result_channel_recs;
    
    for (size_t cnt=0;cnt < execfunc->inst->result_channel_paths.size(); cnt++) {
      std::shared_ptr<std::string> result_channel_path_ptr = execfunc->inst->result_channel_paths.at(cnt);
      
      // note source_rss is NOT locked
      channel_state &source_rss_channel_state = source_rss->recstatus.channel_map->at(recdb_path_join(execfunc->inst->channel_path_context,*result_channel_path_ptr));
      result_channel_recs.push_back(source_rss_channel_state.rec());
      
    }
      
    _transfer_function_result_state_single_locked_referencing_rss(execfunc,FromResultState,ToResultState, result_channel_recs,new_rss,&new_rss_admin);
  }
						   
  void _transfer_function_result_state(std::shared_ptr<math_function_execution> execfunc,std::shared_ptr<recording_set_state> ignore_rss,int FromResultState,int ToResultState, const std::vector<std::shared_ptr<recording_base>> &result_channel_recs) // result_channel_recs should be same length as inst->result_channel_paths and hold the results we want to program in. Programming in only occurs if ToResultState is SNDE_FRS_INSTANTIATED or SNDE_FRS_ALL
  {

    std::set<std::weak_ptr<recording_set_state>,std::owner_less<std::weak_ptr<recording_set_state>>> referencing_rss_copy; // will have all recording set states that reference this executing_math_function

    snde_debug(SNDE_DC_RECMATH,"_transfer_function_result_state(execfunc=0x%llx,ignore_rss=0x%llx,FromResultState=%d,ToResultState=%d)",(unsigned long long)execfunc.get(),(unsigned long long)ignore_rss.get(),(int)FromResultState,(int)ToResultState);

    
    {
      std::lock_guard<std::mutex> execfunc_admin(execfunc->admin);
      referencing_rss_copy = execfunc->referencing_rss;
      
      if (ToResultState==SNDE_FRS_INSTANTIATED || ToResultState==SNDE_FRS_ALL) {
	execfunc->instantiated=true;
      }
      if (ToResultState==SNDE_FRS_METADATAONLY) {
	execfunc->metadata_executed=true;
	execfunc->metadataonly_complete=true;
      }
      if (ToResultState==SNDE_FRS_COMPLETED || ToResultState==SNDE_FRS_ALL) {
	execfunc->metadata_executed=true;
	execfunc->fully_complete=true;
      }
      
      
    }

    // assign recordings to all referencing rss recordings (should all still exist)  -- but it turns out sometimes maybe not (dynamic calcs?)
    for (auto && referencing_rss_weak: referencing_rss_copy) {
      //std::shared_ptr<recording_set_state> referencing_rss_strong(referencing_rss_weak);
      std::shared_ptr<recording_set_state> referencing_rss_strong=referencing_rss_weak.lock();
      if (!referencing_rss_strong) {
	//snde_warning("recmath_compute_resource.cpp: _tfrs: referencing_rss is already expired!");
	continue;
      }

      if (referencing_rss_strong==ignore_rss) {
	continue; // ignore ignored-rss
      }
      
      std::unique_lock<std::mutex> referencing_rss_admin(referencing_rss_strong->admin);

      
      _transfer_function_result_state_single_locked_referencing_rss(execfunc,FromResultState,ToResultState,result_channel_recs,referencing_rss_strong,&referencing_rss_admin);      
      
    }

    
  }


  void execution_complete_notify_single_referencing_rss(std::shared_ptr<recdatabase> recdb,std::shared_ptr<math_function_execution> execfunc,bool mdonly,bool possibly_redundant,std::shared_ptr<recording_set_state> single_referencing_rss)
  {
    snde_debug(SNDE_DC_NOTIFY,"execution_complete_notify_single_referencing_rss on rss 0x%llx execfunc 0x%llx %s",(unsigned long long)single_referencing_rss.get(),(unsigned long long)execfunc.get(),execfunc->inst->definition->definition_command.c_str());
    
    //snde_debug(SNDE_DC_RECMATH,"qc: already finished rss notify %llx",(unsigned long long)referencing_rss_strong.get());
    //snde_debug(SNDE_DC_RECMATH|SNDE_DC_NOTIFY,"pool code: rss notify %llx",(unsigned long long)referencing_rss_strong.get());
    std::string math_function_channel0_name="(nullptr)";
    if (execfunc->inst->result_channel_paths.at(0)) {
      math_function_channel0_name=*execfunc->inst->result_channel_paths.at(0);
    }
    snde_debug(SNDE_DC_RECMATH|SNDE_DC_NOTIFY,"math function %s rss notify %llx",math_function_channel0_name.c_str(),(unsigned long long)single_referencing_rss.get());
    // Issue function completion notification
    if (recdb) {
      single_referencing_rss->mathstatus.notify_math_function_executed(recdb,single_referencing_rss,execfunc->inst,mdonly,possibly_redundant); 
    }

    for (auto && result_channel_path_ptr: execfunc->inst->result_channel_paths) {
      channel_state &chanstate = single_referencing_rss->recstatus.channel_map->at(recdb_path_join(execfunc->inst->channel_path_context,*result_channel_path_ptr));
      //chanstate.issue_math_notifications(recdb,ready_rss); // taken care of by notify_math_function_executed(), above
      chanstate.issue_nonmath_notifications(single_referencing_rss);
      
    }
    
    snde_debug(SNDE_DC_NOTIFY,"execution_complete_notify_single_referencing_rss complete on rss 0x%llx execfunc 0x%llx %s",(unsigned long long)single_referencing_rss.get(),(unsigned long long)execfunc.get(),execfunc->inst->definition->definition_command.c_str());
    
  }
  

  static void _execution_complete_notify(std::shared_ptr<recdatabase> recdb,std::shared_ptr<math_function_execution> execfunc,bool mdonly,bool possibly_redundant)
  {
    std::set<std::weak_ptr<recording_set_state>,std::owner_less<std::weak_ptr<recording_set_state>>> referencing_rss_copy; // will have all recording set states that reference this executing_math_function
    
    {
      std::lock_guard<std::mutex> execfunc_admin(execfunc->admin);
      referencing_rss_copy = execfunc->referencing_rss;
      
      
    }

    snde_debug(SNDE_DC_NOTIFY,"_execution_complete_notify on execfunc 0x%llx %s",(unsigned long long)execfunc.get(),execfunc->inst->definition->definition_command.c_str());

    
    for (auto && referencing_rss_weak: referencing_rss_copy) {
      //std::shared_ptr<recording_set_state> referencing_rss_strong(referencing_rss_weak);
      std::shared_ptr<recording_set_state> referencing_rss_strong=referencing_rss_weak.lock();
      if (!referencing_rss_strong) {
	//snde_warning("recmath_compute_resource.cpp: _ecn: referencing_rss is already expired!");
	continue;
      }

      execution_complete_notify_single_referencing_rss(recdb,execfunc,mdonly,possibly_redundant,referencing_rss_strong);
    }
    
    snde_debug(SNDE_DC_NOTIFY,"_execution_complete_notify complete on execfunc 0x%llx %s",(unsigned long long)execfunc.get(),execfunc->inst->definition->definition_command.c_str());

  }
  
  void _wrap_up_execution(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_set_state> ready_rss,std::shared_ptr<instantiated_math_function> ready_fcn, std::vector<std::shared_ptr<recording_base>> result_channel_recs,bool possibly_redundant)
  // ***!!!!! You must have the execution ticket -- "true" value from try_execution_ticket() in order to call this
  {

    
    assert(result_channel_recs.size() == ready_fcn->result_channel_paths.size());
    
    // make sure all of the result_channel_recs are marked as complete
    for (size_t resultcnt=0; resultcnt < result_channel_recs.size();resultcnt++) {
      
      std::shared_ptr<recording_base> &result_channel_rec = result_channel_recs.at(resultcnt);
      std::shared_ptr<std::string> path_ptr = ready_fcn->result_channel_paths.at(resultcnt);
      std::string full_path;
      if (path_ptr) {
	
	full_path = recdb_path_join(ready_fcn->channel_path_context,*path_ptr);
	channel_state &chanstate = ready_rss->recstatus.channel_map->at(full_path);
	
	if (!chanstate.rec()) {
	  assert(!chanstate.revision()); // potential pitfall here if we promise a new revision but don't deliver it (?)
	  chanstate.end_atomic_rec_update(result_channel_rec);
	  //chanstate.revision = std::make_shared<uint64_t>(result_channel_rec->info->revision); (now implicit in the above)
	} else {
	  result_channel_rec = chanstate.rec();
	  chanstate.updated=true;
	}
	
      }
      if (result_channel_rec) {
	if (!(result_channel_rec->info_state & SNDE_RECF_STATICMETADATAREADY)) {
	  result_channel_rec->mark_metadata_done();
	}
	if (!(result_channel_rec->info_state & SNDE_RECF_DATAREADY)) {
	  result_channel_rec->mark_data_ready();
	}
      } else if (path_ptr) {
	result_channel_rec = create_recording_math<null_recording>(full_path,ready_rss);
	result_channel_rec->mark_metadata_done();
	result_channel_rec->mark_data_ready();
	
      }
    }
    
    std::shared_ptr<math_function_execution> execfunc;
    {
      std::lock_guard<std::mutex> ready_rss_admin(ready_rss->admin);
      math_function_status &ready_rss_status = ready_rss->mathstatus.function_status.at(ready_fcn);

      execfunc = ready_rss_status.execfunc;
      assert(execfunc); // execfunc should have been assigned (At the latest) by check_dep_fcn_ready(), which  should have been called before us.
      
    }
    assert(execfunc->inst == ready_fcn);
    
    _transfer_function_result_state(execfunc,nullptr,SNDE_FRS_ANY,SNDE_FRS_COMPLETED,result_channel_recs);
    
    // mark math_status and execfunc as complete
    execfunc->metadata_executed = true;
    execfunc->fully_complete = true;
    {
      std::lock_guard<std::mutex> ready_rss_admin(ready_rss->admin);
      math_function_status &ready_rss_status = ready_rss->mathstatus.function_status.at(ready_fcn);

      {
	std::shared_ptr<globalrevision> gv=std::dynamic_pointer_cast<globalrevision>(ready_rss);
	if (gv) {
	  snde_debug(SNDE_DC_RECMATH,"_wrap_up_execution: Marking %s as complete in globalrev %u",ready_fcn->definition->definition_command.c_str(),(unsigned)gv->globalrev);
	}
      }
      
      ready_rss_status.complete=true;
    }
    

    {
      std::lock_guard<std::mutex> execfunc_admin(execfunc->admin);
      
      // clear execfunc->rss to eliminate reference loop once at least metadata execution is complete
      execfunc->rss = nullptr;
      if (execfunc->execution_tracker) { // execution_tracker might be null if initiate_execution() failed. 
	execfunc->execution_tracker->rss = nullptr;

	// clear out self_dependent_recordings to eliminate infinite historical references now that execution is completed
	execfunc->execution_tracker->self_dependent_recordings.clear();

      }
      execfunc->execution_tracker = nullptr; 

      execfunc->executing = false; // release the execution ticket
    }
    
    
    // issue notifications
    
    // ... in all referencing rss's
    // !!!**** the false here is mdonly -- may need to set that properly
    _execution_complete_notify(recdb,execfunc,false,possibly_redundant);

    
    
    
  }
  

  
  
  available_compute_resource_database::available_compute_resource_database() :
    admin(std::make_shared<std::mutex>()),
    started(false)
  {

  }

  
  void available_compute_resource_database::set_cpu_resource(std::shared_ptr<available_compute_resource_cpu> cpu_resource)
  {
    assert(!cpu);
    cpu=cpu_resource;
    add_resource(cpu);
  }

  void available_compute_resource_database::add_resource(std::shared_ptr<available_compute_resource> new_resource)
  {

    int new_priority;
    bool new_fallback_flag;
    std::string new_fallback_message;

    std::tie(new_priority,new_fallback_flag,new_fallback_message) = new_resource->get_dispatch_priority();
    
    std::lock_guard<std::mutex> adminlock(*admin);
    auto compute_resource_it = compute_resources.begin();
    
    if (new_fallback_flag && (compute_resource_it==compute_resources.end() || compute_resource_it->first >= SNDE_ACRP_CPU)) {
      // got a fallback with nothing better than a CPU already in place.
      // issue warning message
      snde_warning(new_fallback_message);
    }
    
    compute_resources.emplace(new_priority,new_resource);
  }

  bool available_compute_resource_database::_queue_computation_into_database_acrdb_locked(uint64_t globalrev,std::shared_ptr<pending_computation> computation,const std::vector<std::shared_ptr<compute_resource_option>> &compute_options)
  // returns true if we successfully queued it into at least one place. 
  {

    


    // Algorithm: sort matches between compute_resource_options and
    // available_compute_resources (compute_resource_option->compatible_with())
    // according to the number of execution tags from the
    // (function_to_execute and compute_resource_option) that match
    // tags from the (available_compute_resource). We then select
    // all available_compute_resources with the highest number of
    // matching execution tags. 

    // We will do this by placing vectors of
    // (compute_resource_option,available_compute_resource) pairs in a
    // std::map indexed by the number of matching tags.
    // Then we just pull out the last entry in the map
    // and the corresponding vector of options and resources
    // are used to dispatch the computation. 

    // Here is the map:
    std::map<size_t,std::vector<std::pair<std::shared_ptr<compute_resource_option>,std::shared_ptr<available_compute_resource>>>> resources_by_tags_matched;
    
    std::shared_ptr <available_compute_resource> selected_resource;
    std::shared_ptr <compute_resource_option> selected_option;
    for (auto && compute_option: compute_options) { // compute_option is a shared_ptr<compute_resource_option>
   
      std::set<std::string> execution_tags;
      // Add the execution tags from the function definition
      execution_tags.insert(computation->function_to_execute->inst->execution_tags.begin(),computation->function_to_execute->inst->execution_tags.end());
      // Add the execution tags from the compute_option provided by the function itself
      execution_tags.insert(compute_option->execution_tags.begin(),compute_option->execution_tags.end());

      

     
      for (auto && compute_resource: compute_resources) { // compute_resource.second is a shared_ptr<available_compute_resource>
	if (compute_option->compatible_with(compute_resource.second)) {
	  // Count the number of matches between execution_tags and our available_compute_resource
	  std::set<std::string> matching_tags;
	  
	  std::set_intersection(execution_tags.begin(),execution_tags.end(),
				compute_resource.second->tags.begin(), compute_resource.second->tags.end(),
				std::inserter(matching_tags,matching_tags.begin()));
	  size_t num_matches = matching_tags.size();
	
	  auto resource_map_it = resources_by_tags_matched.find(num_matches);
	  if (resource_map_it == resources_by_tags_matched.end()){
	    // no entry for this num_matches so far
	    bool success;
	    std::tie(resource_map_it,success) = resources_by_tags_matched.emplace(num_matches,std::vector<std::pair<std::shared_ptr<compute_resource_option>,std::shared_ptr<available_compute_resource>>>());
	  }
	  resource_map_it->second.emplace_back(std::make_pair(compute_option,compute_resource.second));
	}

      }
    }
    auto selected_resources=resources_by_tags_matched.end();
    selected_resources--; // last actual element is end()-1
    if (selected_resources==resources_by_tags_matched.end()){

      // snde_warning("_queue_computation_into_database_acrdb_locked: No available_compute_resource matches compute job for %s!!!",computation->function_to_execute->inst->definition->definition_command.c_str());
      return false;
    } else {
      for (auto && option_and_resource: selected_resources->second) {
	std::tie(selected_option,selected_resource)=option_and_resource;
	
	selected_resource->prioritized_computations.emplace(std::make_pair(globalrev*SNDE_CR_PRIORITY_REDUCTION_LIMIT + computation->priority_reduction,
									   std::make_tuple(std::weak_ptr<pending_computation>(computation),selected_option)));
	//compute_resource_lock.unlock();  (???What was this supposed to do???)
	//selected_resource->computations_added.notify_one();
	  snde_debug(SNDE_DC_COMPUTE_DISPATCH,"_queue_computation_into_database_acrdb_locked: Notifying ACRD of changes");



      }
    
   
      notify_acrd_of_changes_to_prioritized_computations();

	

	  

      todo_list.emplace(computation);
      return true;
     
      
    }


  }


  static std::pair<snde_index,bool> queue_computation_get_globalrev_index(std::shared_ptr<recdatabase> recdb, std::shared_ptr<recording_set_state> ready_rss)
  {
    uint64_t globalrev_index=0;
    bool ready_rss_is_globalrev = false; 
    std::shared_ptr<globalrevision> globalrev_ptr = std::dynamic_pointer_cast<globalrevision>(ready_rss);
    if (!globalrev_ptr) {
      //throw snde_error("recording_set_state does not appear to be associated with any global revision");

      // In this case, we are not a globalrev. We are probably a recstore_display_transform. So we use the latest_defined_globalrev() so we queue with
      // stuff currently being computed, rather than taking priority
      // by using our actual originating globalrev (which
      // might well be a lower number)...
      //
      // Alternatively we could change this to have the recording_set_state
      // store its originating_globalrev (to be set in recstore_display_transform)
      // and use the priority from this, but that would potentially allow
      // interactive viewing to lock out background compute, which isn't
      // necessarily a good thing.

      // A better solution might be to dedicate one compute thread solely
      // to interactive viewing. If that is done, then even tons of
      // background compute wouldn't completely block out the display.
      //

      globalrev_index = recdb->latest_defined_globalrev()->globalrev;
    } else {
      ready_rss_is_globalrev=true;
      globalrev_index = globalrev_ptr->globalrev;
    }

    return std::make_pair(globalrev_index,ready_rss_is_globalrev);
  }
  
  void available_compute_resource_database::_queue_computation_internal(std::shared_ptr<recdatabase> recdb,std::shared_ptr<pending_computation> &computation) // NOTE: Sets computation to nullptr once queued
  {
    snde_debug(SNDE_DC_RECMATH,"_queue_computation_internal: %s globalrev %llu",computation->function_to_execute->inst->definition->definition_command.c_str(),(unsigned long long)computation->globalrev);

    uint64_t globalrev_index=0;
    bool computation_rss_is_globalrev = false; 

    std::tie(globalrev_index,computation_rss_is_globalrev) = queue_computation_get_globalrev_index(recdb,computation->recstate);

    
    // get the compute options
    //std::list<std::shared_ptr<compute_resource_option>> compute_options = computation->function_to_execute->get_compute_options();
    std::vector<std::shared_ptr<compute_resource_option>> compute_options = computation->function_to_execute->execution_tracker->perform_compute_options();

    
    // we can execute anything immutable, anything that is not part of a globalrev (i.e. ondemand), or once the prior globalrev is fully ready
    // (really we just need to make sure all immutable recordings in the prior globalrev are ready, but there isn't currently a good
    // way to do that)
    std::shared_ptr<recording_set_state> prior_globalrev;
    if (computation->recstate->prerequisite_state()) {
      prior_globalrev=computation->recstate->prerequisite_state()->rss(); // only actually prior_globalrev if computation_rss_is_globalrev
    }

    std::shared_ptr<globalrevision> prior_globalrev_globalrev=std::dynamic_pointer_cast<globalrevision>(prior_globalrev);
    
    std::lock_guard<std::mutex> acrdb_admin(*admin);

    // ***!!!! Really here we need to see if we want to run it in
    // mutable or immutable mode !!!***
    //if (!computation->function_to_execute->inst->fcn->mandatory_mutable || !computation_rss_is_globalrev || !prior_globalrev || prior_globalrev->ready) {
    
    if (!computation->function_to_execute->is_mutable || !computation_rss_is_globalrev || !prior_globalrev || (prior_globalrev->ready && (!prior_globalrev_globalrev || !prior_globalrev_globalrev->mutable_recordings_still_needed))) {

      if (!_queue_computation_into_database_acrdb_locked(globalrev_index,computation,compute_options)) {
      
  	throw snde_error("No suitable compute resource found for math function %s",computation->function_to_execute->inst->definition->definition_command.c_str());
      }
      
      
    } else {
      // blocked... we have to wait for previous revision to at least
      // complete its mutable recordings and then for any consumers
      // of its mutable recordings to be finished. This is identified
      // when the globalrevision's mutable_recordings_need_holder --
      // which is passed out to all the monitor_globalrevs with the
      // inhibit_mutable flag set -- expires, triggering the
      // blocked_computations for that globalrev (stored in the
      // recdatabase compute_resources blocked_list) to be queued
      // in recstore.cpp: recdatabase::globalrev_mutablenotneeded_code()
      // by calling this function again. 

      blocked_list.emplace(globalrev_index*SNDE_CR_PRIORITY_REDUCTION_LIMIT + computation->priority_reduction,computation);
    }
    computation=nullptr; // release shared pointer prior to releasing acrdb_admin lock so that it will expire from pending_computation lists when extracted.
  }


  void available_compute_resource_database::queue_computation(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_set_state> ready_rss,std::shared_ptr<instantiated_math_function> ready_fcn)
  // Take an identified function ready_fcn that is ready to be computed
  // (all inputs are complete) for execution in the context of ready_rss 
  // and queue it for actual execution by the worker threads
  {

    snde_index globalrev_index;
    bool ready_rss_is_globalrev = false;

    snde_debug(SNDE_DC_RECMATH,"queue_computation");

    
    std::tie(globalrev_index,ready_rss_is_globalrev) = queue_computation_get_globalrev_index(recdb,ready_rss);
    

    bool is_mutable=false;
    bool mdonly;
    std::shared_ptr<math_function_execution> execfunc;
    {
      std::unique_lock<std::mutex> ready_rss_admin(ready_rss->admin);
      math_function_status &ready_rss_status = ready_rss->mathstatus.function_status.at(ready_fcn);

      assert(ready_rss_status.execfunc); // execfunc should have been assigned (At the latest) by check_dep_fcn_ready(), which  should have been called before us. 
      
      execfunc = ready_rss_status.execfunc;
      
      snde_debug(SNDE_DC_RECMATH,"execfunc=0x%llx; rss=0x%llx",(unsigned long long)execfunc.get(),(unsigned long long)ready_rss.get());
      snde_debug(SNDE_DC_RECMATH,"ready_to_execute:%d",(int)ready_rss_status.ready_to_execute);
      if (ready_rss_status.ready_to_execute && execfunc->try_execution_ticket()) {

	// we are taking care of execution -- we have the execution ticket!
	assert(ready_rss_status.execution_demanded); 
	ready_rss_status.ready_to_execute=false;

	// Move rss->mathstatus->math_msgs into executing_math_funciton for this math funciton
	// Extract to local variable here and assign into executing_math_function down on 586
	std::unordered_map<std::string, std::shared_ptr<math_instance_parameter>> math_messages;
	auto msglookup = ready_rss->mathstatus.math_messages.find(ready_fcn);
	if (msglookup != ready_rss->mathstatus.math_messages.end()) {
	  math_messages = msglookup->second;
	}
	
	//std::shared_ptr<recording_set_state> prerequisite_rss = ready_rss->prerequisite_state()->rss();
	
	ready_rss_admin.unlock();


	try {
	  execfunc->execution_tracker = ready_fcn->fcn->initiate_execution(ready_rss,ready_fcn);


	  if (!execfunc->execution_tracker) {
	    // initiate_execution failed
	    snde_warning("initiate_execution() failed and returned nullptr in function %s. This usually means a waveform parameter had an unsupported data type.",ready_fcn->definition->definition_command.c_str());

	    // This code here is identical to the math parameter mismatch catch below
	    // and should probably be refactored
	    std::vector<std::shared_ptr<recording_base>> result_channel_recs;
	    while (result_channel_recs.size() < execfunc->inst->result_channel_paths.size()) {
	      // Create a null recording for any undefined results
	      std::shared_ptr<std::string> path_ptr = execfunc->inst->result_channel_paths.at(result_channel_recs.size());
	      if (path_ptr) {
		result_channel_recs.push_back(create_recording_math<null_recording>(recdb_path_join(execfunc->inst->channel_path_context,*path_ptr),ready_rss));
	      } else {
		result_channel_recs.push_back(nullptr);
	      }
	      
	    }
	    _transfer_function_result_state(execfunc,nullptr,SNDE_FRS_ANY,SNDE_FRS_INSTANTIATED,result_channel_recs);
	    
	    _wrap_up_execution(recdb,ready_rss,ready_fcn,result_channel_recs,true); // true is possibly_redundant
	    return;

	    
	  } else {
	    // add it here
	    execfunc->execution_tracker->msgs = math_messages;
	  }
	  
	} catch (const math_parameter_mismatch &exc) {
	  if (typeid(exc)==typeid(silent_math_parameter_mismatch)) {
	    snde_debug(SNDE_DC_RECMATH,"Silent math parameter mismatch in initiate_execution(): %s (function %s)",exc.shortwhat(),execfunc->inst->definition->definition_command.c_str());
	  } else  {
	    snde_warning("Math parameter mismatch in initiate_execution(): %s (function %s)",exc.shortwhat(),execfunc->inst->definition->definition_command.c_str());
	  }
	  snde_debug(SNDE_DC_RECMATH,"Full backtrace: %s",exc.what());
	  
	  // This code here is identical to the initiate_execution() null return case above
	  // and should probably be refactored
	  std::vector<std::shared_ptr<recording_base>> result_channel_recs;
	  while (result_channel_recs.size() < execfunc->inst->result_channel_paths.size()) {
	    // Create a null recording for any undefined results
	    std::shared_ptr<std::string> path_ptr = execfunc->inst->result_channel_paths.at(result_channel_recs.size());
	    if (path_ptr) {
	      result_channel_recs.push_back(create_recording_math<null_recording>(recdb_path_join(execfunc->inst->channel_path_context,*path_ptr),ready_rss));
	    } else {
	      result_channel_recs.push_back(nullptr);
	    }

	  }
	  _transfer_function_result_state(execfunc,nullptr,SNDE_FRS_ANY,SNDE_FRS_INSTANTIATED,result_channel_recs);
	  
	  _wrap_up_execution(recdb,ready_rss,ready_fcn,result_channel_recs,true); // true is possibly_redundant
	  return;

	}
	bool actually_execute=true;
	
	// Check new_revision_optional
	if (ready_fcn->fcn->new_revision_optional) {
	  // Need to check if it s OK to execute
	  //#define SNDE_RCR_DISABLE_EXCEPTION_HANDLING 
#ifndef SNDE_RCR_DISABLE_EXCEPTION_HANDLING
	  try {
#endif
	    actually_execute = execfunc->execution_tracker->perform_decide_execution();
#ifndef SNDE_RCR_DISABLE_EXCEPTION_HANDLING
	  } catch(const std::exception &exc) {
	    // Only consider exceptions derived from std::exception because there's no general way to print anything else, so we might as well just crash in that case. 
	    // func is our math_function_execution
	    snde_warning("Exception class %s caught in perform_decide_execution(): %s (function %s)",typeid(exc).name(),exc.what(),execfunc->inst->definition->definition_command.c_str());
	    actually_execute=false;
	  }
#endif
	  
	}
	
	if (!actually_execute) {
	  // because new_revision_optional and mdonly are incompatible (see comment in instantiated_math_function
	  // constructor, the prior revision deriving from our self-dependency must be fully ready, and since
	  // we're not executing, we must be fully ready too. 

	  // grab recording results from execfunc->execution_tracker->self_dependent_recordings
	  assert(ready_fcn->result_channel_paths.size()==execfunc->execution_tracker->self_dependent_recordings.size());
	  std::vector<std::shared_ptr<recording_base>> result_channel_recs=execfunc->execution_tracker->self_dependent_recordings;
	  
	  _wrap_up_execution(recdb,ready_rss,ready_fcn,result_channel_recs,false); // false is possibly_redundant
	  snde_debug(SNDE_DC_RECMATH,"qc: already finished");
	  return;
	}
	std::shared_ptr<pending_computation> computation = std::make_shared<pending_computation>(execfunc,ready_rss,globalrev_index,SNDE_CR_PRIORITY_NORMAL);
	
	
	_queue_computation_internal(recdb,computation); // this call sets computation to nullptr; execution ticket delegated to the queued computation
	
	
      } else {
	// Somebody else is taking care of computation (or not ready to execute)
	snde_debug(SNDE_DC_RECMATH,"qc: somebody else");
	
	// no need to do anything; just return
      }
      // warning: ready_rss_admin lock may or may not be held here 
    }
    
  }

  void available_compute_resource_database::start()
  // start all of the compute_resources
  {
    {
      std::lock_guard<std::mutex> acrd_admin(*admin);

      if (started) {
	throw snde_error("compute_resources are already running!");
      }

      started = true; 

    }
    std::multimap<int,std::shared_ptr<available_compute_resource>> compute_resources_copy;
    {
      std::lock_guard<std::mutex> acrd_admin(*admin);
      
      compute_resources_copy = compute_resources;
    }

    for (auto && compute_resource: compute_resources_copy) {
      compute_resource.second->start();
    }


    // start our dispatch thread
    // instantiate dispatch thread
    dispatcher_thread = std::thread([this]() { dispatch_code(); });
    dispatcher_thread.detach(); // we won't be join()ing this thread
    
  }


  void available_compute_resource_database::dispatch_code()
  {
      
    bool no_actual_dispatch = false;
    std::shared_ptr<available_compute_resource_database> acrd_keepinmemory=shared_from_this();  // this shared_ptr prevents the available_compute_resource_database object from getting released, which would make "this" become invalid under us. It also creates a memory leak unless there is a way to signal to this thread that it should return.

    set_thread_name(nullptr,"snde2 acrd dispatch");


    std::unique_lock<std::mutex> admin_lock(*admin);
    while(true) {
      
      if (no_actual_dispatch) {
	snde_debug(SNDE_DC_COMPUTE_DISPATCH,"waiting on computations_added_or_completed");
	computations_added_or_completed.wait(admin_lock);
	snde_debug(SNDE_DC_COMPUTE_DISPATCH,"wakeup on computations_added_or_completed");
      }

      no_actual_dispatch = true; 

      for (auto && compute_resource: compute_resources) {
	// Note: If we drop acrd_admin, the compute_resources list
	// might change under us and we would then be required to break
	// out of this loop.

	// try to dispatch via this resource
	if (compute_resource.second->dispatch_code(admin_lock)) {
	  // success!   (dispatched something, or removed an expred computation from queue)
	  no_actual_dispatch=false;
	  break;
	}
	
      }
      snde_debug(SNDE_DC_COMPUTE_DISPATCH,"dispatch loop terminated; no_actual_dispatch = %d, todo_list.size()=%u",(int)no_actual_dispatch,(unsigned)todo_list.size());

      
    }
  }


  void available_compute_resource_database::notify_acrd_of_changes_to_prioritized_computations() // should be called WITH ACRD's admin lock held
  {
    computations_added_or_completed.notify_one();
  }
  
  pending_computation::pending_computation(std::shared_ptr<math_function_execution> function_to_execute,std::shared_ptr<recording_set_state> recstate,uint64_t globalrev,uint64_t priority_reduction) :
    function_to_execute(function_to_execute),
    recstate(recstate),
    globalrev(globalrev),
    priority_reduction(priority_reduction)
  {

  }


  available_compute_resource::available_compute_resource(std::shared_ptr<recdatabase> recdb,unsigned type,std::set<std::string> tags) :
    recdb(recdb),
    acrd_admin(recdb->compute_resources->admin),
    acrd(recdb->compute_resources),
    type(type),
    tags(tags)
  {
    
  }

  available_compute_resource_cpu::available_compute_resource_cpu(std::shared_ptr<recdatabase> recdb,std::set<std::string> tags,size_t total_cpu_cores_available) :
    available_compute_resource(recdb,SNDE_CR_CPU,tags),
    total_cpu_cores_available(total_cpu_cores_available)
  {

  }

  void available_compute_resource_cpu::start()
  {
    
    size_t cnt;

    std::lock_guard<std::mutex> acrd_lock(*acrd_admin); // because threads will start, and we need to lock them out while we fill up the vector data structures
    for (cnt=0; cnt < total_cpu_cores_available;cnt++) {
      functions_using_cores.push_back(nullptr);
      thread_triggers.emplace_back(std::make_shared<std::condition_variable>());
      thread_actions.push_back(std::make_tuple((std::shared_ptr<recording_set_state>)nullptr,(std::shared_ptr<math_function_execution>)nullptr,(std::shared_ptr<assigned_compute_resource_cpu>)nullptr));
      available_threads.emplace_back(std::thread([this](size_t n){ pool_code(n); },cnt));
      set_thread_name(&available_threads.at(cnt),ssprintf("snde2 acrd %2.2d",cnt));

      available_threads.at(cnt).detach(); // we won't be join()ing these threads
      
    }

  }
  

  size_t available_compute_resource_cpu::_number_of_free_cpus()
  // Must call with ACRD admin lock locked
  {
    size_t number_of_free_cpus=0;

    for (auto && exec_fcn: functions_using_cores) {
      if (!exec_fcn) {
	number_of_free_cpus++;
      }
    }
    return number_of_free_cpus;
  }

  std::shared_ptr<assigned_compute_resource_cpu> available_compute_resource_cpu::_assign_cpus(std::shared_ptr<math_function_execution> function_to_execute,size_t number_of_cpus)
  // called with acrd admin lock held
  {
    size_t cpu_index=0;
    std::vector<size_t> cpu_assignments; 
    for (auto && exec_fcn: functions_using_cores) {
      if (!exec_fcn) {
	// this cpu is available
	cpu_assignments.push_back(cpu_index);
	//function_to_execute->cpu_cores.push_back(cpu_index);
	number_of_cpus--;

	if (!number_of_cpus) {
	  break;
	}
      }
      cpu_index++;
    }
    assert(!number_of_cpus); // should have been able to assign everything
    
    return std::make_shared<assigned_compute_resource_cpu>(shared_from_this(),cpu_assignments);

  }

  void available_compute_resource_cpu::_dispatch_threads_from_pool(std::shared_ptr<recording_set_state> recstate,std::shared_ptr<math_function_execution> function_to_execute,std::shared_ptr<assigned_compute_resource_cpu> assigned_cpu_resource,size_t first_thread_index)
  // Must be called with acrd_admin lock held
  {
    //printf("_dispatch_thread()!\n");

    
    //std::lock_guard<std::mutex> admin_lock(*acrd_admin); // lock assumed to be already held
      
    // assign ourselves to functions_using_cores;
    for (auto && core_index: assigned_cpu_resource->assigned_cpu_core_indices) {
      assert(functions_using_cores.at(core_index)==nullptr);
      functions_using_cores.at(core_index) = function_to_execute; 
    }
    
    std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>,std::shared_ptr<assigned_compute_resource_cpu>> &this_thread_action = thread_actions.at(first_thread_index);
    assert(this_thread_action==std::make_tuple((std::shared_ptr<recording_set_state>)nullptr,(std::shared_ptr<math_function_execution>)nullptr,(std::shared_ptr<assigned_compute_resource_cpu>)nullptr));
    
    this_thread_action = std::make_tuple(recstate,function_to_execute,assigned_cpu_resource);
    
    
  
    //printf("triggering thread %d\n",first_thread_index);
    thread_triggers.at(first_thread_index)->notify_one();
  }
  
  bool available_compute_resource_cpu::dispatch_code(std::unique_lock<std::mutex> &acrd_admin_lock)
  {
    // ***!!! Would be very beneficial here to be more sophisticated, perhas dedicating a few cores to preferentially
    // accept ondemand (i.e. rendering) jobs and/or to go with GPU jobs. 
    
    std::shared_ptr<available_compute_resource_database> acrd_strong=acrd.lock();
    if (!acrd_strong) return false;

    snde_debug(SNDE_DC_COMPUTE_DISPATCH,"CPU Dispatch, %u computations",(unsigned)prioritized_computations.size());
    if (prioritized_computations.size() > 0) {

      // ***!!! Instead of just looking at the top entry, we could loop here,
      // looking deeper at least through the current globalrev until we find something dispatchable. 
      
      std::multimap<uint64_t,std::tuple<std::weak_ptr<pending_computation>,std::shared_ptr<compute_resource_option>>>::iterator this_computation_it = prioritized_computations.begin();
      std::weak_ptr<pending_computation> this_computation_weak;
      std::shared_ptr<compute_resource_option> compute_option;

      std::tie(this_computation_weak,compute_option) = this_computation_it->second;
      std::shared_ptr<pending_computation> this_computation = this_computation_weak.lock();
      if (!this_computation) {
	// pointer expired; computation has been handled elsewhere
	prioritized_computations.erase(this_computation_it); // remove from our list
	snde_debug(SNDE_DC_COMPUTE_DISPATCH,"CPU Dispatched expired computation");
	
	return true; // removing from prioritized_computations counts as an actual dispatch
      } else {
	// got this_computation and compute_option to possibly try.
	// Check if we have enough cores available for compute_option
	std::shared_ptr<compute_resource_option_cpu> compute_option_cpu=std::dynamic_pointer_cast<compute_resource_option_cpu>(compute_option);
	// this had better be one of our pointers...
	assert(compute_option_cpu);
	
	// For now, just blindly use the useful # of cpu cores
	size_t free_cores = _number_of_free_cpus();

	snde_debug(SNDE_DC_COMPUTE_DISPATCH,"Checking execution of %s: free_cores=%llu; enough_available=%d",this_computation->function_to_execute->inst->definition->definition_command.c_str(),(unsigned long long)free_cores,(int)(compute_option_cpu->useful_cpu_cores <= free_cores || free_cores == total_cpu_cores_available));
	
	if (compute_option_cpu->useful_cpu_cores <= free_cores || free_cores == total_cpu_cores_available) {	    
	  // we have enough cores available (or all of them)
	  // !!!*** Would make sense here to limit the cores going to a single computation to slightly
	  // fewer than the total so as to allow single-core GPU jobs to execute in parallel. 
	  std::shared_ptr<math_function_execution> function_to_execute=this_computation->function_to_execute;
	  std::shared_ptr<recording_set_state> recstate=this_computation->recstate;
	  
	  prioritized_computations.erase(this_computation_it); // take charge of this computation
	  acrd_strong->todo_list.erase(this_computation); // remove from todo list so pointer can expire
	  this_computation = nullptr; // force pointer to expire so nobody else tries this computation;
	  
	  std::shared_ptr<assigned_compute_resource_cpu> assigned_cpus = _assign_cpus(function_to_execute,std::min(compute_option_cpu->useful_cpu_cores,total_cpu_cores_available));

	  std::shared_ptr<_compute_resource_option_cpu_combined> combined_resource = std::dynamic_pointer_cast<_compute_resource_option_cpu_combined>(compute_option);
	  if (combined_resource) {
	    // We are just part of the underlying resource, which is combined_resource->orig
	    function_to_execute->execution_tracker->compute_resource = combined_resource->combine_cpu_assignment(assigned_cpus);
	    function_to_execute->execution_tracker->selected_compute_option = combined_resource->orig;
	  
	  } else {
	    function_to_execute->execution_tracker->compute_resource = assigned_cpus;
	    function_to_execute->execution_tracker->selected_compute_option = compute_option;
	  }
	  _dispatch_threads_from_pool(recstate,function_to_execute,assigned_cpus,assigned_cpus->assigned_cpu_core_indices.at(0));
	  
	  snde_debug(SNDE_DC_COMPUTE_DISPATCH,"CPU Dispatched computation for %s rss %llx",function_to_execute->inst->definition->definition_command.c_str(),(unsigned long long)function_to_execute->rss.get());
	  
	  return true; // dispatched execution, so don't wait at next iteration. 
	}

	this_computation = nullptr; // remove all references to pending_computation before we release the lock
      }
      
      
    }
    snde_debug(SNDE_DC_COMPUTE_DISPATCH,"CPU did not dispatch any computation");

    return false;
  }
  
  std::tuple<int,bool,std::string> available_compute_resource_cpu::get_dispatch_priority() // Get the dispatch priority of this compute resource. Smaller or more negative numbers are higher priority. See SNDE_ACRP_XXXX, above. Returns (dispatch_priority,fallback_flag,fallback_message).
  {
    return std::make_tuple(SNDE_ACRP_CPU,false,"");
  }
  
  void available_compute_resource_cpu::pool_code(size_t threadidx)
  {
    std::shared_ptr<available_compute_resource> acr_keepinmemory=shared_from_this();  // this shared_ptr prevents the available_compute_resource object from getting released, which would make "this" become invalid under us.  It also creates a memory leak unless there is a way to signal to this thread that it should return. 

    
    //printf("pool_code_startup!\n");
    std::unique_lock<std::mutex> admin_lock(*acrd_admin);
    while(true) {
      

      std::shared_ptr<recording_set_state> recstate;
      std::shared_ptr<math_function_execution> func;
      std::shared_ptr<assigned_compute_resource_cpu> assigned_compute_cpu;

      std::tie(recstate,func,assigned_compute_cpu) = thread_actions.at(threadidx);
      while (!recstate) {
	thread_triggers.at(threadidx)->wait(admin_lock);
	//printf("pool_code_wakeup!\n");
	// (will pull in the recstate on the next loop through)
	std::tie(recstate,func,assigned_compute_cpu) = thread_actions.at(threadidx);
      }

      assert(func->execution_tracker->compute_resource); // should have a compute_resource defined

      std::shared_ptr<recording_set_state> func_rss = func->rss; // in case someone else notices that the rss is now complete and expires the rss pointer on us

      assert(func_rss);
      
      //printf("Pool code got thread action\n");
      // Remove parameters from thread_actions. This can safely happen here because our cores are still reserved in functions_using_cores until release_assigned_resources() is called below. 
      thread_actions.at(threadidx)=std::make_tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>,std::shared_ptr<assigned_compute_resource_cpu>>(nullptr,nullptr,nullptr);
      
      // not worrying about cpu affinity yet.
      
      //std::shared_ptr<recording_set_state> prerequisite_rss=recstate->prerequisite_state()->rss();
      
      std::vector<std::shared_ptr<recording_base>> result_channel_recs;
      bool mdonly = false;
      bool failed = false;


      // temporarily assign recdb during function execution, so it is available to the function code
      func->execution_tracker->recdb = recstate->recdb_weak.lock();
      if (!func->execution_tracker->recdb) {
	continue; // nowhere to put result if recdb is gone
      }

      // Set the build-time variable SNDE_RCR_DISABLE_EXCEPTION_HANDLING to disable the try {} ... catch{} block in math execution so that you can capture the offending scenario in the debugger
      //#define SNDE_RCR_DISABLE_EXCEPTION_HANDLING
#ifndef SNDE_RCR_DISABLE_EXCEPTION_HANDLING
      try {
#endif
	
	// Need also our assigned_compute_resource (func->compute_resource)
	admin_lock.unlock();
	
	// will eventually need the lock manager from somewhere(!!!***???)
	mdonly = func->mdonly;
	//bool is_mutable;
	//bool mdonly_executed;
	
	//// Get the mdonly and is_mutable flags from the math_function_status 
	//{
	//std::lock_guard<std::mutex> rss_admin(recstate->admin);

	
	//math_function_status &our_status = recstate->mathstatus.function_status.at(func->inst);
	//  mdonly = our_status.mdonly;
	//  is_mutable = our_status.is_mutable;
	// mdonly_executed = our_status.mdonly_executed; 
	//
	//}
	
	if (!func->metadata_executed) {
	  func->execution_tracker->perform_define_recs();
	  
	  // grab generated recordings and move them from defined_recordings to instantiated_recordings list
	  {
	    std::lock_guard<std::mutex> rssadmin(func_rss->admin);
	    for (auto && result_channel_path_ptr: func->inst->result_channel_paths) {
	      if (result_channel_path_ptr) {
		
		channel_state &chanstate = func_rss->recstatus.channel_map->at(recdb_path_join(func->inst->channel_path_context,*result_channel_path_ptr));
		if (!chanstate.rec()) {
		  // don't have a recording
		  // Function was supposed to create this but didn't.
		  // Create a null recording
		  chanstate.end_atomic_rec_update(create_recording_math<null_recording>(recdb_path_join(func->inst->channel_path_context,*result_channel_path_ptr),func_rss));
		  
		}
		size_t erased_from_defined_recordings = func_rss->recstatus.defined_recordings.erase(chanstate.config);
		    
		//assert(erased_from_defined_recordings==1); // recording should have been listed in defined_recordings
		func_rss->recstatus.instantiated_recordings.emplace(chanstate.config,&chanstate);
		
	      }
	    }
	  }
	  for (auto && result_channel_path_ptr: func->inst->result_channel_paths) {
	    if (result_channel_path_ptr) {
	      channel_state &chanstate = func_rss->recstatus.channel_map->at(recdb_path_join(func->inst->channel_path_context,*result_channel_path_ptr));
	      result_channel_recs.push_back(chanstate.rec());
	      chanstate.updated=true; // write that we are updating this output channel !!!*** If we allow an update that doesn't change the recording to be considered as no change, then we would need to consider that along with the updated flag here !!!***
	    } else {
	      result_channel_recs.push_back(nullptr);
	    }
	  }
	  
	  //func->instantiated=true; (now assigned while holding lock in _transfer_function_result_state())
	  // NOTE that the result state could already have jumped ahead to INSTANTIATED by end_transaction()
	  // so we need to accommodate that
	  _transfer_function_result_state(func,nullptr,SNDE_FRS_DEFINED|SNDE_FRS_INSTANTIATED,SNDE_FRS_INSTANTIATED,result_channel_recs);
	  
	  
	  func->execution_tracker->perform_metadata();
	  func->metadata_executed=true;
	  
	  if (mdonly) {
	    
	    // assign recordings to all referencing rss recordings (should all still exist)
	    // (only needed if we're not doing it below)
	    //func->metadataonly_complete = true;  (now assigned while holding lock in _transfer_function_result_state())
	    // NOTE that the result state could already have jumped ahead to METADATAONLY by end_transaction()
	    // so we need to accommodate that
	    _transfer_function_result_state(func,nullptr,SNDE_FRS_INSTANTIATED|SNDE_FRS_METADATAONLY,SNDE_FRS_METADATAONLY,result_channel_recs); // note: result_channel_recs ignored at this phase
	  }
	  
	  
	} else {
	  
	  for (auto && result_channel_path_ptr: func->inst->result_channel_paths) {
	    if (result_channel_path_ptr) {
	      channel_state &chanstate = func_rss->recstatus.channel_map->at(recdb_path_join(func->inst->channel_path_context,*result_channel_path_ptr));
	      result_channel_recs.push_back(chanstate.rec());
	    } else {
	      result_channel_recs.push_back(nullptr);
	    }
	  }
	  
	}
	
	
	if (!mdonly) {
	  func->execution_tracker->perform_lock_alloc();
	  func->execution_tracker->perform_exec();
	  
	  //func->fully_complete = true; (now assigned while holding lock in _transfer_function_result_state())
	  // NOTE that the result state could already have jumped ahead to COMPLETED by end_transaction()
	  // so we need to accommodate that
	  _transfer_function_result_state(func,nullptr,SNDE_FRS_INSTANTIATED|SNDE_FRS_METADATAONLY|SNDE_FRS_COMPLETED,SNDE_FRS_COMPLETED,result_channel_recs); // note: result_channel_recs ignored at this phase
	  
	  // clear out self_dependent_recordings to eliminate infinite historical references now that execution is completed
	  std::lock_guard<std::mutex> func_admin(func->admin);
	  func->execution_tracker->self_dependent_recordings.clear();
	  
	} else {
	  // self-dependency incompatible with mdonly -- otherwise we would have infinite historical references
	  
	  assert(!func->execution_tracker->self_dependent_recordings.size());
	}
	
	
	// Mark execution as no longer ongoing in the math_function_status 
	{
	  std::unique_lock<std::mutex> rss_admin(recstate->admin);
	  std::unique_lock<std::mutex> func_admin(func->admin);
	  
	  math_function_status &our_status = recstate->mathstatus.function_status.at(func->inst);
	  if (mdonly && !func->mdonly) {
	    // execution status changed from mdonly to non-mdonly behind our back...
	    mdonly=false;
	    // finish up execution before we mark as finished
	    func_admin.unlock();
	    rss_admin.unlock();
	    
	    func->execution_tracker->perform_lock_alloc();
	    func->execution_tracker->perform_exec();
	    
	    //func->fully_complete = true; (now assigned while holding lock in _transfer_function_result_state())
	    // NOTE that the result state could already have jumped ahead to METADATAONLY by end_transaction()
	    // so we need to accommodate that
	    
	    _transfer_function_result_state(func,nullptr,SNDE_FRS_INSTANTIATED|SNDE_FRS_METADATAONLY|SNDE_FRS_COMPLETED,SNDE_FRS_COMPLETED,result_channel_recs); // note: result_channel_recs ignored at this phase	      
	    
	    rss_admin.lock();
	    func_admin.lock();
	    // clear out self_dependent_recordings to eliminate infinite historical references now that execution is completed
	    func->execution_tracker->self_dependent_recordings.clear();
	    
	  }
	  our_status.complete=true;
	  
	}
	
	// clear execfunc->rss to eliminate reference loop once at least metadata execution is complete for an mdonly recording
	// or once all execution is complete for a regular recording
	if (func->metadataonly_complete || func->fully_complete) {
	  std::lock_guard<std::mutex> func_admin(func->admin);
	  
	  func->rss = nullptr;
	  func->execution_tracker->rss = nullptr;
	  
	}
	func->execution_tracker->recdb = nullptr; // eliminate reference loop
	
#ifndef SNDE_RCR_DISABLE_EXCEPTION_HANDLING
      }  catch(const std::exception &exc) {
	// Only consider exceptions derived from std::exception because there's no general way to print anything else, so we might as well just crash in that case. 
	// func is our math_function_execution
	snde_warning("Exception class %s caught in math thread pool: %s (function %s)",typeid(exc).name(),exc.what(),func->inst->definition->definition_command.c_str());

	func->execution_tracker->recdb = nullptr; // eliminate reference loop
	
	// mark as complete
	//func->fully_complete = true; -- now done in _transfer_function_result_state()
	
	while (result_channel_recs.size() < func->inst->result_channel_paths.size()) {
	  // Create a null recording for any undefined results
	  std::shared_ptr<std::string> path_ptr = func->inst->result_channel_paths.at(result_channel_recs.size());
	  if (path_ptr) {
	    result_channel_recs.push_back(create_recording_math<null_recording>(recdb_path_join(func->inst->channel_path_context,*path_ptr),func_rss));
	  } else {
	    result_channel_recs.push_back(nullptr);
	  }
	  
	}
	//_transfer_function_result_state(func,nullptr,SNDE_FRS_ANY,SNDE_FRS_INSTANTIATED,result_channel_recs);
	
	// make sure all of the result_channel_recs are marked as complete
	for (auto && result_channel_rec: result_channel_recs) {
	  if (result_channel_rec) {
	    if (!result_channel_rec->metadata) {
	      result_channel_rec->metadata = std::make_shared<immutable_metadata>();
	    }
	    if (!(result_channel_rec->info_state & SNDE_RECF_STATICMETADATAREADY)) {
	      result_channel_rec->mark_metadata_done();
	    }
	    if (!(result_channel_rec->info_state & SNDE_RECF_DATAREADY)) {
	      result_channel_rec->mark_data_ready();
	    }
	  }
	}
	
	
	// mark all the waveforms as complete in our and all dependent rss's. 
	_transfer_function_result_state(func,nullptr,SNDE_FRS_ANY,SNDE_FRS_ALL,result_channel_recs); 
	{
	  std::unique_lock<std::mutex> rss_admin(recstate->admin);
	  std::unique_lock<std::mutex> func_admin(func->admin);
	  
	  // set function status as complete
	  math_function_status &our_status = recstate->mathstatus.function_status.at(func->inst);
	  our_status.complete=true;
	  
	  // clear out self_dependent_recordings to eliminate infinite historical references
	  
	  func->execution_tracker->self_dependent_recordings.clear();
	  //func->execution_tracker = nullptr;  cleared below
	  
	  // clear out references to the rss to allow releasing memory
	  //func->rss = nullptr; (now done below)
	  
	}
	failed = true; 
      }
#endif // SNDE_RCR_DISABLE_EXCEPTION_HANDLING
      func->executing = false; // release the execution ticket
      
      
      std::shared_ptr<recdatabase> recdb_strong = recdb.lock();
      
      snde_debug(SNDE_DC_RECMATH,"Pool code completed math function %s",func->inst->definition->definition_command.c_str());
      //fflush(stdout);
      
      
      // Need to do notifications that the math function finished in all referencing rss's
      _execution_complete_notify(recdb_strong,func,mdonly,false); // false is possibly_redundant

      snde_debug(SNDE_DC_RECMATH,"Pool code completed notifications for math function %s",func->inst->definition->definition_command.c_str());
      
      //printf("Pool code completed notification\n");
      //fflush(stdout);
      // Completion notification:
      //  * removing ourselves from functions_using_cores and triggering computations_added_or_completed
      admin_lock.lock();
      
      // release compute resources
      func->execution_tracker->compute_resource->release_assigned_resources(admin_lock);
      
      admin_lock.unlock();
      if (failed || !func->metadataonly_complete) {
	
	// This function isn't going to execute again -- clear its execution tracker and prevent it from keeping its rss in memory
	std::lock_guard<std::mutex> func_admin(func->admin);
	func->execution_tracker = nullptr;
	func->rss = nullptr; 
      }
	
      
      admin_lock.lock();
      // Notify that we are done
      std::shared_ptr<available_compute_resource_database> acrd_strong=acrd.lock();
      if (acrd_strong) {
	acrd_strong->notify_acrd_of_changes_to_prioritized_computations();
      }
    
    }
  }

  
  assigned_compute_resource::assigned_compute_resource(unsigned type,std::shared_ptr<available_compute_resource> resource) :
    type(type),
    resource(resource)
  {
    
  }
  
  assigned_compute_resource_cpu::assigned_compute_resource_cpu(std::shared_ptr<available_compute_resource> resource,const std::vector<size_t> &assigned_cpu_core_indices) :
    assigned_compute_resource(SNDE_CR_CPU,resource),
    assigned_cpu_core_indices(assigned_cpu_core_indices)
  {
    
  }

  void assigned_compute_resource_cpu::release_assigned_resources(std::unique_lock<std::mutex> &acrd_admin_holder) // resources referenced below no longer meaningful once this is called. Must be called with acrd admin lock locked
  {
    // remove ourselves from functions_using_cores
    //for (size_t corenum=0;corenum < total_cpu_cores_available;corenum++) {
    //  if (functions_using_cores.at(corenum) == func) {
    //functions_using_cores.at(corenum) = nullptr; 
    //  }
    //}

    std::shared_ptr<available_compute_resource_cpu> cpu_resource = std::dynamic_pointer_cast<available_compute_resource_cpu>(resource);
    assert(cpu_resource); // types should always match
    
    for (auto && coreindex: assigned_cpu_core_indices) {
      cpu_resource->functions_using_cores.at(coreindex) = nullptr; 
    }

  }

};
