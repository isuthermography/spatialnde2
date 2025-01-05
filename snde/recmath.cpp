#include "snde/recmath.hpp"
#include "snde/recstore.hpp"

namespace snde {
  // std::set<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>>>
  bool mncn_lessthan::operator()( const std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>> & lhs,
				  const std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>> & rhs ) const noexcept
  {
    // std::shared_ptr<recording_set_state> l_rss, r_rss;
    // std::shared_ptr<math_function_execution> l_execfunc, r_execfunc;

    // std::tie(l_rss, l_execfunc)=lhs;

    if (std::owner_less<std::shared_ptr<recording_set_state>>()(std::get<0>(lhs),std::get<0>(rhs))){
      return true;
    }
    if (std::owner_less<std::shared_ptr<recording_set_state>>()(std::get<0>(rhs),std::get<0>(lhs))){
      return false;
    }
    if (std::owner_less<std::shared_ptr<math_function_execution>>()(std::get<1>(lhs),std::get<1>(rhs))){
      return true;
    }
    return false;
  }
  
  
  math_instance_parameter::math_instance_parameter(unsigned paramtype) :
    paramtype(paramtype)
  {

  }

  list_math_instance_parameter::list_math_instance_parameter(std::vector<std::shared_ptr<math_instance_parameter>> list) :
    math_instance_parameter(SNDE_MFPT_VECTOR),
    list(list)
  {

  }

  bool list_math_instance_parameter::operator==(const math_instance_parameter &ref) // used for comparing extra parameters to instantiated_math_functions
  {

    const list_math_instance_parameter *lref = dynamic_cast<const list_math_instance_parameter *>(&ref);

    if (!lref) {
      return false;
    }
    
    if (list.size() != lref->list.size()) {
      return false;
    }

    for (size_t idx=0;idx < list.size();idx++) {
      if (*list[idx] != *lref->list[idx]) {
	return false;
      }
    }
    return true;
    
  }
  
  bool list_math_instance_parameter::operator!=(const math_instance_parameter &ref)
  {
    return !(*this==ref);
  }
  
  dict_math_instance_parameter::dict_math_instance_parameter(std::unordered_map<std::string, std::shared_ptr<math_instance_parameter>> dict) :
    math_instance_parameter(SNDE_MFPT_MAP),
    dict(dict)
  {
    
  }

  bool dict_math_instance_parameter::operator==(const math_instance_parameter &ref) // used for comparing extra parameters to instantiated_math_functions
  {

    const dict_math_instance_parameter *dref = dynamic_cast<const dict_math_instance_parameter *>(&ref);
    
    if (!dref) {
      return false;
    }

    
    if (dict.size() != dref->dict.size()) {
      return false;
    }
    
    for (auto && str_subparam: dict) {
      auto ref_iter = dref->dict.find(str_subparam.first);
      if (ref_iter == dref->dict.end()) {
	return false;
      }
      if (*ref_iter->second != *str_subparam.second) {
	return false;
      }
    }

    return true;
  }
  bool dict_math_instance_parameter::operator!=(const math_instance_parameter &ref)
  {
    return !(*this==ref);
  }

  string_math_instance_parameter::string_math_instance_parameter(std::string value) :
    math_instance_parameter(SNDE_MFPT_STR),
    value(value)
  {

  }
  
  bool string_math_instance_parameter::operator==(const math_instance_parameter &ref) // used for comparing extra parameters to instantiated_math_functions
  {
    const string_math_instance_parameter *sref = dynamic_cast<const string_math_instance_parameter *>(&ref);

    if (!sref) {
      return false;
    }

    return value == sref->value;
  }
  
  bool string_math_instance_parameter::operator!=(const math_instance_parameter &ref)
  {
    return !(*this==ref);
  }
  
  int_math_instance_parameter::int_math_instance_parameter(int64_t value) :
    math_instance_parameter(SNDE_MFPT_INT),
    value(value)
  {
    
  }

  bool int_math_instance_parameter::operator==(const math_instance_parameter &ref) // used for comparing extra parameters to instantiated_math_functions
  {

    const int_math_instance_parameter *iref = dynamic_cast<const int_math_instance_parameter *>(&ref);

    if (!iref) {
      return false;
    }

    return value==iref->value;
  }
  
  bool int_math_instance_parameter::operator!=(const math_instance_parameter &ref)
  {
    return !(*this==ref);
  }

  double_math_instance_parameter::double_math_instance_parameter(double value) :
    math_instance_parameter(SNDE_MFPT_DBL),
    value(value)
  {

  }
  
  bool double_math_instance_parameter::operator==(const math_instance_parameter &ref) // used for comparing extra parameters to instantiated_math_functions
  {
    const double_math_instance_parameter *dref = dynamic_cast<const double_math_instance_parameter *>(&ref);

    if (!dref) {
      return false;
    }
    return value==dref->value;
  }
  
  bool double_math_instance_parameter::operator!=(const math_instance_parameter &ref)
  {
    return !(*this==ref);
  }

  
  math_function::math_function(std::string function_name, size_t num_results,const std::vector<std::pair<std::string,unsigned>> &param_names_types,std::function<std::shared_ptr<executing_math_function>(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> instantiated)> initiate_execution) :
    function_name(function_name),
    num_results(num_results),
    param_names_types(param_names_types),
    initiate_execution(initiate_execution)
  {
    new_revision_optional=false;
    pure_optionally_mutable=false;
    mandatory_mutable=false;
    self_dependent=false;
    mdonly_allowed = false;
    dynamic_dependency = false;
  }

  math_definition::math_definition(std::string definition_command) :
    definition_command(definition_command)
  {

  }
  std::shared_ptr<math_definition> math_definition::rebuild(std::shared_ptr<instantiated_math_function> fcn)
  {
    return std::make_shared<math_definition>(*this);
  }
  
  bool math_definition::operator==(const math_definition &ref)
  {
    return definition_command==ref.definition_command;
  }

  bool math_definition::operator!=(const math_definition &ref)
  {
    return !(*this==ref);
  }

  
  instantiated_math_function::instantiated_math_function(const std::vector<std::shared_ptr<math_parameter>> & parameters,
							 const std::vector<std::shared_ptr<std::string>> & result_channel_paths,
							 std::string channel_path_context,
							 bool is_mutable,
							 bool ondemand,
							 bool mdonly,
							 std::shared_ptr<math_function> fcn,
							 std::shared_ptr<math_definition> definition,
							 std::set<std::string> execution_tags,
							 std::shared_ptr<math_instance_parameter> extra_params) :
    parameters(parameters.begin(),parameters.end()),
    result_channel_paths(result_channel_paths.begin(),result_channel_paths.end()),
    channel_path_context(channel_path_context),
    disabled(false),
    is_mutable(is_mutable),
    self_dependent(fcn->new_revision_optional || is_mutable || fcn->self_dependent || fcn->dynamic_dependency),
    ondemand(ondemand),
    mdonly(mdonly),
    fcn(fcn),
    definition(definition),
    execution_tags(execution_tags),
    extra_params(extra_params)
  {

    if (mdonly) {
      if (fcn->new_revision_optional) {
	// mdonly is (maybe) incompatible with new_revision_optional because if we decide not to execute
	// we need a prior result to fall back on. But execution is managed based on changed
	// inputs (math_function_execution) where the prior incomplete result would require importing
	// and replacing our current math_function_execution/executing_math_function pair,
	// which might be possible but we currently don't allow. If we were to do this, any
	// subsequent revisions already created but referring to our current math_function_execution/executing_math_function pair
	// would need to be identified and updated
	throw snde_error("Metadata only function may not be new_revision_optional");
      }
    }
  }
  
  
  std::shared_ptr<instantiated_math_function> instantiated_math_function::clone(bool definition_change)
  {
    // This code needs to be repeated in every derived class with the make_shared<> referencing the derived class
    std::shared_ptr<instantiated_math_function> copy = std::make_shared<instantiated_math_function>(*this);

    // see comment at start of definition of class instantiated_math_function
    if (definition_change && definition) {
      assert(!copy->original_function);
      copy->original_function = shared_from_this();
      copy->definition = nullptr; 
    }
    return copy;
  }

  bool instantiated_math_function::operator==(const instantiated_math_function &ref)
  {

    if (parameters.size() != ref.parameters.size()) {
      return false;
    }

    for (size_t pnum=0;pnum < parameters.size();pnum++) {
      if (*parameters[pnum] != *ref.parameters[pnum]) {
	return false;
      }
    }

    if (result_channel_paths.size() != ref.result_channel_paths.size()) {
      return false;
    }
    
    for (size_t rnum=0;rnum < result_channel_paths.size();rnum++) {
      if (*result_channel_paths[rnum] != *ref.result_channel_paths[rnum]) {
	return false;
      }
    }

    if (result_mutability != ref.result_mutability) {
      return false;
    }

    if (channel_path_context != ref.channel_path_context) {
      return false;
    }

    if (disabled != ref.disabled) {
      return false;
    }

    if (is_mutable != ref.is_mutable) {
      return false;
    }


    if (self_dependent != ref.self_dependent) {
      return false;
    }

    if (ondemand != ref.ondemand) {
      return false;
    }

    if (mdonly != ref.mdonly) {
      return false;
    }

    if (fcn != ref.fcn) {
      // Note that here we compare the (smart) math_function pointers,
      // not the structure contents -- so this test will fail
      // if the math_function object got redefined
      return false;
    }

    if (*definition != *ref.definition) {
      // use math_definition::operator==()
      return false;
    }


    if ((extra_params && !ref.extra_params) || (!extra_params && ref.extra_params)) {
      return false;
    }

    if (!extra_params && !ref.extra_params) {
      return true;
    }

    if (*extra_params != *ref.extra_params) {
      // use math_instance_parameter::operator()
      return false;
    }

    return true;
  }

  bool instantiated_math_function::operator!=(const instantiated_math_function &ref)
  {
    return !(*this==ref);
  }

  static std::shared_ptr<instantiated_math_function> _rdm_get_updated_fcn(std::unordered_map<std::shared_ptr<instantiated_math_function>,std::shared_ptr<instantiated_math_function>> *updated_fcns, std::shared_ptr<instantiated_math_function> key)
  {
    auto updated_it=updated_fcns->find(key);
    if (updated_it==updated_fcns->end()) {
      bool junk;
      std::tie(updated_it,junk)=updated_fcns->emplace(key,key->clone(false)); // definition_changed is false because we will be replacing the definition with its rebuild() method later.
    }
    return updated_it->second;
  }
  
  void instantiated_math_database::_rdm_remove_from_definition(std::unordered_map<std::shared_ptr<instantiated_math_function>,std::shared_ptr<instantiated_math_function>> *updated_fcns,size_t result_index,std::shared_ptr<instantiated_math_function> orig_fcn_ptr)
  {
    // recdb should be locked when calling this method.
    std::shared_ptr<instantiated_math_function> clone=_rdm_get_updated_fcn(updated_fcns,orig_fcn_ptr);
    std::string result_name=recdb_path_join(orig_fcn_ptr->channel_path_context,*orig_fcn_ptr->result_channel_paths.at(result_index));
    // remove from the defined_math_functions, if present
    auto defined_fcns_it=defined_math_functions.find(result_name);

    if (defined_fcns_it != defined_math_functions.end() && defined_fcns_it->second == orig_fcn_ptr) {
      // it matches... remove
      defined_math_functions.erase(defined_fcns_it);
    }
    clone->result_channel_paths.at(result_index)=nullptr; // zero out the relevant result channel
    clone->definition=clone->definition->rebuild(clone);
  }
  // rebuild all_dependencies_of_channel hash table. Must be called any time any of the defined_math_functions changes. May only be called for the instantiated_math_database within the main recording database, and the main recording database admin lock must be locked when this is called. 
  void instantiated_math_database::_rebuild_dependency_map(std::shared_ptr<recdatabase> recdb,bool allow_non_reserved_math_channels)
  {

    all_dependencies_of_channel.clear();
    all_dependencies_of_function.clear();
    mdonly_functions.clear();
    
    std::unordered_set<std::shared_ptr<instantiated_math_function>> all_fcns;
    std::unordered_set<std::shared_ptr<instantiated_math_function>> extraneous_fcns;
    std::unordered_map<std::shared_ptr<instantiated_math_function>,std::shared_ptr<instantiated_math_function>> updated_fcns; 
    
    for (auto&& channame_math_fcn_ptr: defined_math_functions) {
      std::shared_ptr<instantiated_math_function> fcn_ptr = channame_math_fcn_ptr.second;

      all_fcns.emplace(fcn_ptr);

    }
    // clean up any result_channel_paths that are no longer valid.
    for (auto && fcn_ptr: all_fcns) {
      size_t num_active_results_chans =0;
      for (size_t resultpath_index=0;resultpath_index<fcn_ptr->result_channel_paths.size();resultpath_index++) {
        std::shared_ptr<std::string> resultpath_ptr=fcn_ptr->result_channel_paths[resultpath_index];
      	if (resultpath_ptr) {
          std::string resultpath=recdb_path_join(fcn_ptr->channel_path_context,*resultpath_ptr);
          std::shared_ptr<reserved_channel> reserved_current=recdb->lookup_math_channel_recdb_locked(resultpath);
          if (reserved_current) {
            std::shared_ptr<channelconfig> config_current=reserved_current->realized_config();
            if (config_current->math_fcn == fcn_ptr) {
              num_active_results_chans++;
            }
            else {
              _rdm_remove_from_definition(&updated_fcns,resultpath_index,fcn_ptr);
              // resultpath_ptr=nullptr;
            }
          } else {
            if (allow_non_reserved_math_channels) {
              num_active_results_chans++;
            } else {
              _rdm_remove_from_definition(&updated_fcns,resultpath_index,fcn_ptr);
              // resultpath_ptr=nullptr;
            }
          }
        } else {
          _rdm_remove_from_definition(&updated_fcns,resultpath_index,fcn_ptr);
          // resultpath_ptr=nullptr;
        }
      }
      if (num_active_results_chans==0) {
        // All result channels disabled or turned off
        extraneous_fcns.emplace(fcn_ptr);
      }
    }
    all_fcns.clear();
    for (auto&& fcn_ptr: extraneous_fcns) {
      
      //  Need to remove these from the instantiated math database and updated_fcns
      // already removed from defined_math_functions by rdm_remove_from_definition(); other database entries are fully rebuilt below. 
      auto updated_it = updated_fcns.find(fcn_ptr);
      if (updated_it != updated_fcns.end()) {
        updated_fcns.erase(updated_it);
      }
      
    }
    //  Need to replace updated_fcnsin the instantiated_mathdatabase
    for (auto && updatedold_updatednew: updated_fcns) {
      std::shared_ptr<instantiated_math_function> updatedold,updatednew;

      std::tie(updatedold,updatednew)=updatedold_updatednew;

      for (auto && result_channel_path_ptr: updatednew->result_channel_paths) {
        if (result_channel_path_ptr) {
          auto defined_it=defined_math_functions.find(recdb_path_join(updatedold->channel_path_context,*result_channel_path_ptr));
          if (defined_it != defined_math_functions.end()) {
            defined_math_functions.erase(defined_it);
          }
          
        }
      }
    }
    for (auto&& channame_math_fcn_ptr: defined_math_functions) {
      std::shared_ptr<instantiated_math_function> fcn_ptr = channame_math_fcn_ptr.second;

      all_fcns.emplace(fcn_ptr);

    }
   

 
    for (auto && fcn_ptr: all_fcns) {
      if (fcn_ptr->mdonly) {
	mdonly_functions.emplace(fcn_ptr);
      }
      for (auto && parameter_ptr: fcn_ptr->parameters) {

	std::set<std::string> prereqs = parameter_ptr->get_prerequisites(fcn_ptr->channel_path_context);
	for (auto && prereq: prereqs) {
	  auto adc_it = all_dependencies_of_channel.find(prereq);
	  if (adc_it == all_dependencies_of_channel.end()) {
	    // need to insert new element
	    adc_it = all_dependencies_of_channel.emplace(std::piecewise_construct,
							 std::forward_as_tuple(prereq),
							 std::forward_as_tuple()).first;
	    
	  }
	  // add this function as dependent on the prerequisite
	  adc_it->second.emplace(fcn_ptr);
	}
      }
    }

    for (auto && fcn_ptr: all_fcns) {
      // going through each math function, look up all result_channels
      auto adf_it = all_dependencies_of_function.find(fcn_ptr);
      assert(adf_it == all_dependencies_of_function.end()); // shouldn't be possible for this to already be populated
      adf_it = all_dependencies_of_function.emplace(std::piecewise_construct,
						    std::forward_as_tuple(fcn_ptr),
						    std::forward_as_tuple()).first;
      size_t num_active_result_chans=0;
      for (auto && resultpath_ptr : fcn_ptr->result_channel_paths) {
        if (resultpath_ptr) {
          
          std::string resultpath=recdb_path_join(fcn_ptr->channel_path_context,*resultpath_ptr);
          
          //std::shared_ptr<reserved_channel> reserved_current=recdb->lookup_math_channel_recdb_locked(resultpath);
          //std::shared_ptr<channelconfig> config_current=reserved_current->realized_config();
           
 
              
          // for this result channel, look up all dependent functions
          auto adc_it = all_dependencies_of_channel.find(resultpath);
          if (adc_it != all_dependencies_of_channel.end()) {
            // iterate over all dependencies
            for (auto && dep_fcn: adc_it->second) { // dep_fcn is a shared_ptr<instantiated_math_function>
              // mark dep_fcn as dependent on fcn_ptr
              adf_it->second.emplace(dep_fcn);
            }
          }
        }
      }
    

    }
   
  }

  math_function_status::math_function_status(bool self_dependent) :
    num_modified_prerequisites(0),
    mdonly(false),
    self_dependent(self_dependent),
    // is_mutable(is_mutable),
    execution_demanded(false),
    ready_to_execute(false),
    complete(false)    
  {

  }
  
  math_status::math_status(std::shared_ptr<instantiated_math_database> math_functions,const std::map<std::string,channel_state> & channel_map) :
    math_functions(math_functions)
  {

    
    std::atomic_store(&_external_dependencies_on_function,std::make_shared<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>>());

    std::atomic_store(&_external_dependencies_on_channel,std::make_shared<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>>());

    std::atomic_store(&_extra_internal_dependencies_on_channel,std::make_shared<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::shared_ptr<instantiated_math_function>>>>());
    
    // put all math functions into function_status and _external_dependencies databases and into pending_functions?
    for (auto && math_function_name_ptr: math_functions->defined_math_functions) {
      std::shared_ptr<instantiated_math_function> instfcn = math_function_name_ptr.second;

      // self-dependency: does not include self-dependencies caused by:
      //    * The lack of initial knowledge about whether the function will execute
      //      (generally because it is dependent directly or indirectly on something
      //      else that might or might not execute)
      //self_dependent = instfcn->self_dependent 
      
      if (function_status.find(math_function_name_ptr.second) == function_status.end()) {
	function_status.emplace(std::piecewise_construct, // ***!!! Need to fill out prerequisites somewhere, but not sure constructor is the place!!!*** ... NO. prerequisites are filled out in end_transaction(). 
				std::forward_as_tuple(math_function_name_ptr.second),
				std::forward_as_tuple(instfcn->self_dependent)); // .second->mdonly,math_function_name_ptr.second->is_mutable)); // Set the mutable flag according to the instantiated_math_function (do we really need this flag still???) 
	_external_dependencies_on_function->emplace(std::piecewise_construct,
						    std::forward_as_tuple(math_function_name_ptr.second),
						    std::forward_as_tuple());
	
	
	if (math_function_name_ptr.second->mdonly) {
	  mdonly_pending_functions.emplace(math_function_name_ptr.second);
	} else {
	  snde_debug(SNDE_DC_RECMATH,"mathstatus 0x%llx %s is pending",(unsigned long long)this,math_function_name_ptr.first.c_str());
	  pending_functions.emplace(math_function_name_ptr.second);	  
	}
	
	
      }
    }

    for (auto && channel_path_channel_state: channel_map) {
      _external_dependencies_on_channel->emplace(std::piecewise_construct,
						 std::forward_as_tuple(channel_path_channel_state.second.config),
						 std::forward_as_tuple());
      
    }
  }
  
  std::string math_status::print_math_status(std::shared_ptr<recording_set_state> rss, bool verbose)
  {
    std::string print_buf;
    std::lock_guard<std::mutex> rss_lock(rss->admin);

    print_buf += ssprintf("\n");
    print_buf += ssprintf("math function status\n");
    print_buf += ssprintf("--------------------\n");

    for (auto && fcn_fcn_status: function_status) {
      std::shared_ptr<instantiated_math_function> fcn = std::get<0>(fcn_fcn_status);
      math_function_status &fcn_status = std::get<1>(fcn_fcn_status);

      if (verbose || !fcn_status.complete){
	print_buf += ssprintf("function %s: %lu modified prerequisites\n",fcn->definition->definition_command.c_str(),(unsigned long)fcn_status.num_modified_prerequisites);
	print_buf += ssprintf("---------------------------------\n");
	print_buf += ssprintf("mdonly=%d; self_dep=%d; exec_demand=%d; ready=%d; complete=%d\n",(int)fcn_status.mdonly,(int)fcn_status.self_dependent,(int)fcn_status.execution_demanded,(int)fcn_status.ready_to_execute,(int)fcn_status.complete);
	for (auto && prereq: fcn_status.missing_prerequisites){
	  print_buf += ssprintf("prerequisite %s missing\n",prereq->channelpath.c_str());
	}
	
	for (auto && extchan_prereq: fcn_status.missing_external_channel_prerequisites){
	  std::shared_ptr<recording_set_state> extchan_prereq_rss;
	  std::shared_ptr<channelconfig> extchan_prereq_chan;
	  std::tie(extchan_prereq_rss,extchan_prereq_chan)=extchan_prereq;
	  
	  std::shared_ptr<globalrevision> globalrev = std::dynamic_pointer_cast<globalrevision>(extchan_prereq_rss);
	  if (globalrev) {
	    print_buf += ssprintf("external prerequisite globalrev %llu channel %s missing\n",(unsigned long long)globalrev->globalrev,extchan_prereq_chan->channelpath.c_str());
	  }
	  else {
	    print_buf += ssprintf("external prerequisite non-globalrev channel %s missing\n",extchan_prereq_chan->channelpath.c_str());
	  }
	}

	for (auto && extfunc_prereq: fcn_status.missing_external_function_prerequisites){
	  std::shared_ptr<recording_set_state> extfunc_prereq_rss;
	  std::shared_ptr<instantiated_math_function> extfunc_prereq_fcn;
	  std::tie(extfunc_prereq_rss,extfunc_prereq_fcn)=extfunc_prereq;
	  
	  std::shared_ptr<globalrevision> globalrev = std::dynamic_pointer_cast<globalrevision>(extfunc_prereq_rss);
	  if (globalrev) {
	    print_buf += ssprintf("external prerequisite globalrev %llu function %s missing\n",(unsigned long long)globalrev->globalrev,extfunc_prereq_fcn->definition->definition_command.c_str());
	  }
	  else {
	    print_buf += ssprintf("external prerequisite non-globalrev function %s missing\n",extfunc_prereq_fcn->definition->definition_command.c_str());
	  }
	}
	print_buf += ssprintf("\n");
      }
      
    }
    
    print_buf += ssprintf("total functions: %lu; %lu accounted for\n",(unsigned long)function_status.size(),(unsigned long)(pending_functions.size() + mdonly_pending_functions.size() + completed_functions.size() + completed_mdonly_functions.size()));

    if (pending_functions.size() > 0) {
      print_buf += ssprintf("pending functions (%lu)\n",(unsigned long)pending_functions.size());
      print_buf += ssprintf("-------------------------\n");
      for (auto && fcn: pending_functions) {
	print_buf += ssprintf("  %s\n",fcn->definition->definition_command.c_str());
      }
    }

    print_buf += ssprintf("\n");
    
    if (mdonly_pending_functions.size() > 0) {
      print_buf += ssprintf("mdonly pending functions (%lu)\n",(unsigned long)mdonly_pending_functions.size());
      print_buf += ssprintf("-----------------------------\n");
      
      for (auto && fcn: mdonly_pending_functions) {
	print_buf += ssprintf("  %s\n",fcn->definition->definition_command.c_str());
      }
    }

    print_buf += ssprintf("\n");

    if (completed_functions.size() > 0) {
      if (verbose) {
	
	print_buf += ssprintf("completed functions (%lu)\n",(unsigned long)completed_functions.size());
	print_buf += ssprintf("-------------------------------\n");
	
	for (auto && fcn: completed_functions) {
	  print_buf += ssprintf("  %s\n",fcn->definition->definition_command.c_str());
	}
      }
      else {
	print_buf += ssprintf("%lu completed functions\n",(unsigned long)completed_functions.size());
      }
    }

    print_buf += ssprintf("\n");

    if (completed_mdonly_functions.size() > 0) {
      if (verbose) {
	
	print_buf += ssprintf("completed mdonly functions (%lu)\n",(unsigned long)completed_mdonly_functions.size());
	print_buf += ssprintf("--------------------------------\n");
	
	for (auto && fcn: completed_mdonly_functions) {
	  print_buf += ssprintf("  %s\n",fcn->definition->definition_command.c_str());
	}
      }
      else {
	print_buf += ssprintf("%lu completed mdonly functions\n",(unsigned long)completed_mdonly_functions.size());
      }
    }

    print_buf += ssprintf("\n");
    return print_buf;
  }
  
  std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> math_status::begin_atomic_external_dependencies_on_channel_update() // must be called with recording_set_state's admin lock held
  {
    std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> orig = external_dependencies_on_channel(); 
    return std::make_shared<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>>(*orig);    
  }

  
  
  void math_status::end_atomic_external_dependencies_on_channel_update(std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> newextdep)
// must be called with recording_set_state's admin lock held
  {
    std::atomic_store(&_external_dependencies_on_channel,newextdep);
  }
  
  std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> math_status::external_dependencies_on_channel()
  {
    return std::atomic_load(&_external_dependencies_on_channel);
  }

  
  std::shared_ptr<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> math_status::begin_atomic_external_dependencies_on_function_update() // must be called with recording_set_state's admin lock held
  {
    return std::make_shared<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>>(*external_dependencies_on_function());
  }
  
  void math_status::end_atomic_external_dependencies_on_function_update(std::shared_ptr<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> newextdep)
// must be called with recording_set_state's admin lock held
  {
    std::atomic_store(&_external_dependencies_on_function,newextdep);
  }
  
  std::shared_ptr<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> math_status::external_dependencies_on_function()
  {
    return std::atomic_load(&_external_dependencies_on_function);
  }



  std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::shared_ptr<instantiated_math_function>>>> math_status::begin_atomic_extra_internal_dependencies_on_channel_update() // must be called with recording_set_state's admin lock held
  {
    return std::make_shared<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::shared_ptr<instantiated_math_function>>>>(*extra_internal_dependencies_on_channel());
  }
  
  void math_status::end_atomic_extra_internal_dependencies_on_channel_update(std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::shared_ptr<instantiated_math_function>>>> newextdep)
// must be called with recording_set_state's admin lock held
  {
    std::atomic_store(&_extra_internal_dependencies_on_channel,newextdep);
  }

  std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::shared_ptr<instantiated_math_function>>>> math_status::extra_internal_dependencies_on_channel()
  {
    return std::atomic_load(&_extra_internal_dependencies_on_channel);
  }


  
  void math_status::notify_math_function_executed(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_set_state> recstate,std::shared_ptr<instantiated_math_function> fcn,bool mdonly,bool possibly_redundant)
  // possibly_redundant is set when our notification is a result of
  // an exception or similar and hence might be redundant
  {
    std::shared_ptr<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> external_dependencies_on_function;

    std::set<std::tuple<std::shared_ptr<channelconfig>,std::shared_ptr<instantiated_math_function>>> extra_internal_deps;
    
    {
      std::lock_guard<std::mutex> rss_admin(recstate->admin);
      math_function_status &our_status = recstate->mathstatus.function_status.at(fcn);


      our_status.complete = true; // trust that whoever called us knows what they were talking about
      // ***!!!!! Need to refactor the status transfer so that we can
      // explicitly check it; also need to be OK with status transfer
      // being ahead of us (possibly another thread)
      //
      // Also need in recstore.cpp:end_transaction()
      // to go through and check if this needs to be redone
      // after execfunc set. 
      
      // Find this fcn in matstatus [mdonly_]pending_functions and remove it, adding it to the completed block
      std::unordered_set<std::shared_ptr<instantiated_math_function>>::iterator pending_it; 
      if (mdonly) {
	pending_it = recstate->mathstatus.mdonly_pending_functions.find(fcn);
	if (pending_it==recstate->mathstatus.mdonly_pending_functions.end()) {
	  throw snde_error("Math function %s executed mdonly without being in mdonly_pending_functions queue",fcn->definition->definition_command.c_str());
	}
	recstate->mathstatus.mdonly_pending_functions.erase(pending_it);
	recstate->mathstatus.completed_mdonly_functions.emplace(fcn);
      } else {
	pending_it = recstate->mathstatus.pending_functions.find(fcn);	
	if (pending_it==recstate->mathstatus.pending_functions.end() && !possibly_redundant) {
	  // We might need to remove this error as realistically it is likely given all the possible
	  // race conditions required by locking/unlocking etc. that redundant notifications may happen,
	  // in which case we might (rarely) legitimately get duplicate notifications
	  // this is here in the interim to troubleshoot excessive duplicate notifications
	  // (they should be extremely rare)
	  snde_debug(SNDE_DC_RECMATH,"mathstatus for error 0x%llx",(unsigned long long)&recstate->mathstatus);
	  //throw snde_error("Math function %s executed without being in pending_functions queue",fcn->definition->definition_command.c_str());
	  snde_debug(SNDE_DC_RECMATH,"Got math function execution notification for math function %s not in pending_functions queue. Probably a redundant notification.",fcn->definition->definition_command.c_str());
	  assert(recstate->mathstatus.completed_functions.find(fcn) != recstate->mathstatus.completed_functions.end());
	} else if (pending_it!=recstate->mathstatus.pending_functions.end()) {
	  recstate->mathstatus.pending_functions.erase(pending_it);
	  recstate->mathstatus.completed_functions.emplace(fcn);
	}
	std::shared_ptr<globalrevision> recglobalrev = std::dynamic_pointer_cast<globalrevision>(recstate);
	if (recglobalrev) {
	  snde_debug(SNDE_DC_RECMATH,"Math function %s in globalrev %llu completed and removed from pending_functions (mathstatus 0x%llx)",fcn->definition->definition_command.c_str(),(unsigned long long)recglobalrev->globalrev,(unsigned long long)&recstate->mathstatus);
	}
      }

      // These assigned before calling this function
      //our_status.metadata_executed=true;
      //if (mdonly && our_status.mdonly) {
      //our_status.complete=true;
      //}
      
      // look up anything dependent on this function's execution
      external_dependencies_on_function = recstate->mathstatus.external_dependencies_on_function();
      extra_internal_deps = our_status.extra_internal_dependencies_on_function;
      
    } // release lock

    
    std::vector<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>> ready_to_execute;
    std::set<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>>,mncn_lessthan> may_need_completion_notification;
    // Search for external dependencies on this function; accumulate in ready_to_execute
    
    std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>::iterator ext_dep_it = external_dependencies_on_function->find(fcn);    
    assert(ext_dep_it != external_dependencies_on_function->end()); // should always have a vector, even if it's empty


    std::shared_ptr<globalrevision> globalrevstate=std::dynamic_pointer_cast<globalrevision>(recstate);
    if (globalrevstate) {
      snde_debug(SNDE_DC_RECMATH,"globalrev %d checking external dependencies on %s (0x%llx): got %d",
		 globalrevstate->globalrev,
		 fcn->definition->definition_command.c_str(),
		 (unsigned long long)fcn.get(),
		 ext_dep_it->second.size());
    }
    
    for (auto && rss_fcn: ext_dep_it->second) {
      std::weak_ptr<recording_set_state> ext_dep_rss_weak;
      std::shared_ptr<instantiated_math_function> ext_dep_fcn;

      std::tie(ext_dep_rss_weak,ext_dep_fcn) = rss_fcn;
      std::shared_ptr<recording_set_state> ext_dep_rss=ext_dep_rss_weak.lock();

      if (ext_dep_rss) {
	std::unique_lock<std::mutex> dep_rss_admin(ext_dep_rss->admin);
	math_function_status &ext_dep_status = ext_dep_rss->mathstatus.function_status.at(ext_dep_fcn);

	std::set<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>::iterator
	  dependent_prereq_it = ext_dep_status.missing_external_function_prerequisites.find(std::make_tuple((std::shared_ptr<recording_set_state>)recstate,(std::shared_ptr<instantiated_math_function>)fcn));
	
	std::shared_ptr<globalrevision> ext_dep_globalrev=std::dynamic_pointer_cast<globalrevision>(ext_dep_rss);
      
	if (ext_dep_globalrev) {
	  snde_debug(SNDE_DC_RECMATH,"Checking if %s in globalrev %llu can now execute; rte size=%llu",ext_dep_fcn->definition->definition_command.c_str(),(unsigned long long)ext_dep_globalrev->globalrev,(unsigned long long)ready_to_execute.size());
	}
	
	
	if (dependent_prereq_it != ext_dep_status.missing_external_function_prerequisites.end()) {
	  snde_debug(SNDE_DC_RECMATH, "Erasing mefp %s from %s",fcn->definition->definition_command.c_str(),ext_dep_fcn->definition->definition_command.c_str());
	  ext_dep_status.missing_external_function_prerequisites.erase(dependent_prereq_it);
	}
	ext_dep_rss->mathstatus.check_dep_fcn_ready(recdb,ext_dep_rss,ext_dep_fcn,&ext_dep_status,ready_to_execute,may_need_completion_notification,dep_rss_admin);
	if (ext_dep_globalrev) {
	  snde_debug(SNDE_DC_RECMATH,"After checking if %s in globalrev %llu can now execute, rte size=%llu",ext_dep_fcn->definition->definition_command.c_str(),(unsigned long long)ext_dep_globalrev->globalrev,(unsigned long long)ready_to_execute.size());
	}
      }
    }

    // Go through any extra_internal_deps, removing us as a missing prerequisite. Then see if the function is ready to run. 
    {
      std::unique_lock<std::mutex> rss_admin(recstate->admin);
      for (auto && dep: extra_internal_deps) {
	std::shared_ptr<channelconfig> dep_channel;
	std::shared_ptr<instantiated_math_function> dep_math;

	std::tie(dep_channel,dep_math) = dep;
	math_function_status &dep_status = recstate->mathstatus.function_status.at(dep_math);
	// Remove us as a missing prerequisite...For all our result_channel_paths.
	for (auto && our_result: fcn->result_channel_paths) {
	  if (!our_result) {
	    continue;
	  }
	  channel_state &stat = recstate->recstatus.channel_map->at(*our_result);
	  auto missing_prereq_it = dep_status.missing_prerequisites.find(stat.config);
	  if (missing_prereq_it != dep_status.missing_prerequisites.end()) {
	    dep_status.missing_prerequisites.erase(missing_prereq_it);
	    bool channel_modified = stat.updated;
	    if (channel_modified) {
	      // Don't count self-dependencies here (at least not unless the definition has changed) ... but shouldn't be possible to get this code path on a self-dependency
	      dep_status.num_modified_prerequisites++;
	    }
	  }
	}
	
	recstate->mathstatus.check_dep_fcn_ready(recdb,recstate,dep_math,&dep_status,ready_to_execute,may_need_completion_notification,rss_admin);
  
      }
    }
    // Queue computations from dependent functions
    
    for (auto && ready_rss_ready_fcn: ready_to_execute) {
      // Need to queue as a pending_computation
      std::shared_ptr<recording_set_state> ready_rss;
      std::shared_ptr<instantiated_math_function> ready_fcn;

      std::tie(ready_rss,ready_fcn) = ready_rss_ready_fcn;
      recdb->compute_resources->queue_computation(recdb,ready_rss,ready_fcn); // will only queue if we got the execution ticket
    }
    // Run any possibly needed completion notifications
    for (auto && complete_rss_complete_execfunc: may_need_completion_notification) {
      std::shared_ptr<recording_set_state> complete_rss;
      std::shared_ptr<math_function_execution> complete_execfunc;

      std::tie(complete_rss,complete_execfunc) = complete_rss_complete_execfunc;
      execution_complete_notify_single_referencing_rss(recdb,complete_execfunc,complete_execfunc->mdonly,true,complete_rss);
    }

    // Go through the function's output channels and issue suitable notifications
    for (auto && result_channel_relpath: fcn->result_channel_paths) {
      std::string result_channel_path = recdb_path_join(fcn->channel_path_context,*result_channel_relpath);
      channel_state &chanstate = recstate->recstatus.channel_map->at(result_channel_path);
      // ***!!!! If we wanted to support math functions that decide
      // not to change their outputs, we could do so here
      // by changing the truth value for each channel receiving
      // the math notification according to whether the channel was
      // updated (would also have to treat function significantly
      // like execution_optional)
      chanstate.issue_math_notifications(recdb,recstate);
    }

    	    //execution_complete_notify_single_referencing_rss(recdb,execfunc,execfunc->mdonly,false,prior_state,dep_rss);

    /*
    std::set<std::weak_ptr<recording_set_state>,std::owner_less<std::weak_ptr<recording_set_state>>> referencing_rss_copy; // will have all recording set states that reference this executing_math_function


    {
      std::lock_guard<std::mutex> execfunc_admin(execfunc->admin);
      referencing_rss_copy = execfunc->referencing_rss;
    }

    for (auto && referencing_rss_weak: referencing_rss_copy) {
      //std::shared_ptr<recording_set_state> referencing_rss_strong(referencing_rss_weak);
      std::shared_ptr<recording_set_state> referencing_rss_strong=referencing_rss_weak.lock();
      if (!referencing_rss_strong) {
	//snde_warning("recmath_compute_resource.cpp: _tfrs: referencing_rss is already expired!");
	continue;
      }

      if (referencing_rss_strong == execfunc->rss) {
	// no need for main rss
	continue
      }

      join_rss_into_function_result_state(execfunc,execfunc->rss,referencing_rss_strong);
    }
    */
  }

  void math_status::check_dep_fcn_ready(std::shared_ptr<recdatabase> recdb,
					std::shared_ptr<recording_set_state> dep_rss,
					std::shared_ptr<instantiated_math_function> dep_fcn,
					math_function_status *mathstatus_ptr,
					std::vector<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>> &ready_to_execute_appendvec,
					std::set<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>>,mncn_lessthan> &may_need_completion_notification_set,
					std::unique_lock<std::mutex> &dep_rss_admin_holder)
  //check_dep_fcn_ready() is called from issue_math_notifications_check_dependent_channel()
  // via issue_math_notifications() for math dependencies within the rss, and also directly
  // for external_dependencies_on_channel to evaluate whether a math function that is
  // dependent on something that just changed state is now ready to execute.

  // check_dep_fcn_ready() accumulates these in the ready_to_execute_append_vec
  // It is also called by build_rss_from_functions_and_channels()
  // (which is part of end_transaction()) in recstore.cpp to likewise determine
  // what math functions can be immediately executed.

  // check_dep_fcn_ready() is also called by notify_math_function_executed()
  // to help accumulate a vector for external_dependencies_on_function.
  // Recall that (as of this writing) external_dependencies_on_function is
  // used solely for self dependencies, because self dependencies are not
  // really compatible with external_dependencies_on_channel.

  // We have a recursive loop because check_dep_fcn_ready() can call
  // execution_complete_notify_single_referencing_rss(), which in turn, can
  // call the notify_math_function_executed() method of mathstatus that in
  // turn calls check_dep_fcn_ready() both directly and inderectly via
  // chanstate.issue_math_notifications()

  // check_dep_fcn_ready()'s calls to execution_complete_notify_single_referencing_rss()
  // are done in part to address a race condition whereby the execution_function
  // might have been marked as complete but perhaps the result is not fully transfered
  // (?). The notify_single_referencing_rss() is also done
  // when it is determined that the function does not
  // actually need to run and the pre-existing result
  // can be copied in.

  // One possible solution would be to queue up
  // calls to execution_complete_notify_single_referencing_rss()
  // so that they only get called once at the end.
  // The queue should be done similar to the
  // ready_to_execute queue.

  // assumes dep_rss admin lock is already held
  // checks if prerequisites are done and thus this function is ready to
  // execute.
  //
  // dep_rss admin lock holder passed in as a parameter
  // because we need to be able to temporarily drop that
  // lock -- for example when calling the dynamic_dependency
  // dependency walker 
  // !!!*** Also needs to consider -- for mathstatuses with no execfunc
  // whether execution is entirely unnecessary -- if so copy in execfunc from
  // previous globalrev, (needs to be an additional parameter?). For a
  // dynamic_dependency also needs to call dependency walker method
  // so as to extend missing_prerequisites and add to num_modified_prerequisites
  // as appropriate
  // Needs to somehow pass notification instructions back to caller, as
  // notification probably can't be issued with dep_rss admin lock held.
    
  {
    if (dep_rss->ready) return; // unnecessary call


    if (mathstatus_ptr->execfunc) {
      std::unique_lock<std::mutex> execfunc_admin(mathstatus_ptr->execfunc->admin);
      if (mathstatus_ptr->execfunc->metadata_executed || mathstatus_ptr->execfunc->rss != dep_rss) {
	// this is already executing or it's someone else's responsibility to execute.
	// however we can check if it is complete

	if (mathstatus_ptr->execfunc->fully_complete) {
	  //mathstatus_ptr->complete = true;
	  may_need_completion_notification_set.emplace(std::make_tuple(dep_rss,mathstatus_ptr->execfunc));
	  
	  // std::shared_ptr<math_function_execution> execfunc=mathstatus_ptr->execfunc;
	  
	  //  execfunc_admin.unlock();
	  
	  // dep_rss_admin_holder.unlock();
	  // execution_complete_notify_single_referencing_rss(recdb,execfunc,execfunc->mdonly,true,dep_rss); // Used to CRASH HERE... modified execfunc structure to store execution output independent of rss, so that we can pull it in after rss is forgotten. 
							   
	  // dep_rss_admin_holder.lock();
	}

	return;
      }
    }
    
    std::shared_ptr<globalrevision> dep_globalrev = std::dynamic_pointer_cast<globalrevision>(dep_rss);
    if (dep_globalrev) {
      snde_debug(SNDE_DC_RECMATH,"recmath: check_dep_fcn_ready(%s); num_modified_prerequisites=%llu; globalrev=%llu; %u missing prereqs, %u mecp %u mefp",dep_fcn->definition->definition_command.c_str(),(unsigned long long)mathstatus_ptr->num_modified_prerequisites,(unsigned long long)dep_globalrev->globalrev,(unsigned)mathstatus_ptr->missing_prerequisites.size(),(unsigned)mathstatus_ptr->missing_external_channel_prerequisites.size(),(unsigned)mathstatus_ptr->missing_external_function_prerequisites.size());
      
    } else {
      snde_debug(SNDE_DC_RECMATH,"recmath: check_dep_fcn_ready(%s); num_modified_prerequisites=%llu",dep_fcn->definition->definition_command.c_str(),(unsigned long long)mathstatus_ptr->num_modified_prerequisites);
    }
    mathstatus_ptr->execution_demanded = mathstatus_ptr->execution_demanded || (mathstatus_ptr->num_modified_prerequisites > 0); 
     
    
    if (!mathstatus_ptr->missing_prerequisites.size() &&
	!mathstatus_ptr->missing_external_channel_prerequisites.size() &&
	!mathstatus_ptr->missing_external_function_prerequisites.size()) {
      
      // all currently listed prerequisites are ready

      if (!mathstatus_ptr->execfunc) {
	// if it could be a dynamic dependency, then call the relevant
	// method and see if more dependencies can be found.
	// if so, make recursive call back to this function and return
	if (dep_fcn->fcn->find_additional_deps) {
	  dep_rss_admin_holder.unlock();
	  bool more_deps = (*dep_fcn->fcn->find_additional_deps)(dep_rss,dep_fcn,mathstatus_ptr);
	  
	  dep_rss_admin_holder.lock();
	  
	  if (more_deps) {
	    check_dep_fcn_ready(recdb,dep_rss,dep_fcn,mathstatus_ptr,ready_to_execute_appendvec,may_need_completion_notification_set,dep_rss_admin_holder);
	    return;
	  }
	}
	// Check if prerequisites are unchanged from prior rev
	//  ... in this case copy in execfunc from prior rev
	// (all prerequisites should at least be in place)
	
	// Note: We know the the math function definition itself is unchanged, because otherwise
	// execfunc would have been assigned in recstore.cpp end_transaction()
	// So all we have to check here is the full list of prerequisites
	
	std::shared_ptr<recording_set_state> prior_state = dep_rss->prerequisite_state()->rss();
	assert(prior_state); 

	
	bool need_recalc = false; 
	for (auto && dep_fcn_param: dep_fcn->parameters) {
	  std::shared_ptr<math_parameter_recording> dep_fcn_rec_param = std::dynamic_pointer_cast<math_parameter_recording>(dep_fcn_param);

	  if (dep_fcn_rec_param) {
	    std::string dep_fcn_param_fullpath = recdb_path_join(dep_fcn->channel_path_context,dep_fcn_rec_param->channel_name);
	    
	    // look up the parameter/prerequisite in our current rss
	    channel_state &paramstate = dep_rss->recstatus.channel_map->at(dep_fcn_param_fullpath);
	    assert(paramstate.revision());


	    //channel_state &parampriorstate = prior_state->recstatus.channel_map->at(dep_fcn_param_fullpath);
	    //assert(parampriorstate.revision());

	    if (paramstate.updated) {
	      snde_debug(SNDE_DC_RECMATH,"recmath: %s need recalc due to %s updated.",dep_fcn->definition->definition_command.c_str(),dep_fcn_param_fullpath.c_str());
	      // Alternate conditional
	      //if ( (*paramstate.revision()) != (*parampriorstate.revision())) {
	      //snde_debug(SNDE_DC_RECMATH,"recmath: %s need recalc due to changed revision of %s.",dep_fcn->definition->definition_command.c_str(),dep_fcn_param_fullpath.c_str());
	      
	      // parameter has changed: Need a recalculation
	      

	      need_recalc=true;
	      break;
	    }
	  }
	}

	mathstatus_ptr->execution_demanded = mathstatus_ptr->execution_demanded || need_recalc;
	
	if (!mathstatus_ptr->execution_demanded) {
	  // copy in execfunc from prior state
	  // (we can look it up based on our instantiated_math_function because
	  // we would already have an execfunc in place if our instantiated_math_function
	  // had changed)
	  math_function_status &prior_status = prior_state->mathstatus.function_status.at(dep_fcn);
	  std::shared_ptr<math_function_execution> execfunc;

	  dep_rss_admin_holder.unlock();

	  {
	    std::lock_guard<std::mutex> prior_admin(prior_state->admin);
	    execfunc=prior_status.execfunc;
	    
	  }

	  
	  // register with execfunc's registering_rss so we get completion notifications
	  
	  // 1/10/22 ***!!! if execfunc is complete and didn't run then we are complete too and will need
	  // to copy the recordings into place notify anybody dependent on those
	  // recordings of the completion.
	  // if execfunc is not complete then we have to make sure that this registration
	  // in reference_rss will make those things happen when execfunc is finally complete.
	  
	  join_rss_into_function_result_state(execfunc,prior_state,dep_rss);
	  //{   
	  //  // (now done by join_rss_into_function_result_state())
	  //  std::lock_guard<std::mutex> execfunc_admin(execfunc->admin);
	  //  execfunc->referencing_rss.emplace(std::weak_ptr<recording_set_state>(dep_rss));
	  //
	  //}


	  dep_rss_admin_holder.lock();

	  // assign execfunc  into our status
	  mathstatus_ptr->execfunc = execfunc;
	  mathstatus_ptr->complete=execfunc->fully_complete; 

	  dep_rss_admin_holder.unlock();

	  // if execfunc is already complete, we need to
	  // trigger the notifications now.

	  //dep_rss_admin_holder.lock();
	  /*
	  if ((mathstatus_ptr->mdonly && execfunc->metadata_executed) ||
	      (!mathstatus_ptr->mdonly && execfunc->fully_complete)) {
	    // execfunc is already complete so we may have to take care of notifications
	    
	    mathstatus_ptr->complete=true;

	    // issue notifications
	    dep_rss_admin_holder.unlock();
	    for (auto && result_channel_relpath: dep_fcn->result_channel_paths) {
	      std::string result_channel_path = recdb_path_join(dep_fcn->channel_path_context,*result_channel_relpath);
	      channel_state &chanstate = dep_rss->recstatus.channel_map->at(recdb_path_join(dep_fcn->channel_path_context,result_channel_path));
	      
	      //chanstate.issue_math_notifications(recdb,ready_rss); // taken care of by notify_math_function_executed(), below
	      chanstate.issue_nonmath_notifications(dep_rss);

	    }
	    dep_rss_admin_holder.lock();
	  } 
	  */
	  //dep_rss_admin_holder.unlock();

	  // Notify that the our channels are all set, in case they weren't already notified
	  if ((mathstatus_ptr->mdonly && execfunc->metadata_executed) ||
	      (!mathstatus_ptr->mdonly && execfunc->fully_complete)) {

	    //execution_complete_notify_single_referencing_rss(recdb,execfunc,execfunc->mdonly,false,dep_rss);

	    // execfunc is already complete so we may have to take care of notifications
	    // Issue function completion notification
	    //dep_rss->mathstatus.notify_math_function_executed(recdb,dep_rss,execfunc->inst,execfunc->mdonly,true,prior_state);
	    // execution_complete_notify_single_referencing_rss(recdb,execfunc,execfunc->mdonly,true,dep_rss);

	    may_need_completion_notification_set.emplace(std::make_tuple(dep_rss,execfunc));
	  }
	  dep_rss_admin_holder.lock();
	  
	  return;
	} else {
	  // execution is demanded and we are ready to go... or at least set-up the execfunc
	  std::shared_ptr<globalrevision> dep_globalrev = std::dynamic_pointer_cast<globalrevision>(dep_rss);
	  if (dep_globalrev) {
	    snde_debug(SNDE_DC_RECMATH,"check_dep_fcn_ready(): creating math_function_execution for globalrev %llu for %s",(unsigned long long)dep_globalrev->globalrev,dep_fcn->definition->definition_command.c_str());
	  }
	  mathstatus_ptr->execfunc = std::make_shared<math_function_execution>(dep_rss,dep_fcn,mathstatus_ptr->mdonly,dep_fcn->is_mutable);

	}
	
	// set ready_to_execute (below)
	// so that queue_computation() will be called by our
	// caller
	//

	
      } else {
	// we have an execfunc... Are we the right context to do the
	// execution (or maybe execution has already begun?)
	std::lock_guard<std::mutex> execfunc_admin(mathstatus_ptr->execfunc->admin);
	if (mathstatus_ptr->execfunc->rss != dep_rss) {
	  // not the right context or already executed (and execfunc->rss is nullptr). 
	  return; 
	}
      }
      
      mathstatus_ptr->ready_to_execute=true;
      
      ready_to_execute_appendvec.emplace_back(std::make_tuple(dep_rss,dep_fcn));  // recording_set_state,  instantiated_math_function
    }
    
  }

  

  math_function_execution::math_function_execution(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst,bool mdonly,bool is_mutable) :
    rss(rss),
    rss_channel_map(rss->recstatus.channel_map),
    inst(inst),
    executing(false),
    is_mutable(is_mutable),
    mdonly(mdonly),
    instantiated(false),
    metadata_executed(false),
    metadataonly_complete(false),
    fully_complete(false)
    // automatically adds rss to referencing_rss set
  {
    referencing_rss.emplace(std::weak_ptr<recording_set_state>(rss));

  }

  
  executing_math_function::executing_math_function(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst,std::shared_ptr<lockmanager> lockmgr) :
    rss(rss),
    inst(inst),
    lockmgr(lockmgr)
  {
    std::shared_ptr<recording_base> null_rec;

    
    // initialize self_dependent_recordings, if applicable
    if (inst && inst->self_dependent) {
      std::shared_ptr<recording_set_state> prereq_state=rss->prerequisite_state()->rss();

      auto prereq_state_function_status_it = prereq_state->mathstatus.function_status.find(inst);
      if (prereq_state_function_status_it == prereq_state->mathstatus.function_status.end()) {
	// math function definition has changed. Don't try to pull in self-dependent recordings
	for (auto &&result_channel_path_ptr: inst->result_channel_paths) {
	  self_dependent_recordings.push_back(null_rec);
	}
      } else {
	for (auto &&result_channel_path_ptr: inst->result_channel_paths) {
	  if (result_channel_path_ptr) {
	    channel_state &chanstate = prereq_state->recstatus.channel_map->at(recdb_path_join(inst->channel_path_context, *result_channel_path_ptr));
	    self_dependent_recordings.push_back(chanstate.rec());
	  } else {
	    self_dependent_recordings.push_back(null_rec);
	  }
	}
      }
    }
    

  }

  std::string executing_math_function::get_result_channel_path(size_t result_index)
  {
    return recdb_path_join(inst->channel_path_context,*inst->result_channel_paths.at(result_index));
  }
  
  size_t executing_math_function::get_num_result_channels()
  {
    return inst->result_channel_paths.size();
  }


  
  pending_math_definition_result_channel::pending_math_definition_result_channel(std::shared_ptr<pending_math_definition> definition,size_t result_channel_index):
    definition(definition),
    result_channel_index(result_channel_index),
    existing_mode(false),
    existing_mode_math_definition(""),
    existing_mode_channel_name("")
  {
    
  }


  pending_math_definition_result_channel::pending_math_definition_result_channel(std::string existing_mode_math_definition,std::string existing_mode_channel_name):
    definition(nullptr),
    result_channel_index(0),
    existing_mode(true),
    existing_mode_math_definition(existing_mode_math_definition),
    existing_mode_channel_name(existing_mode_channel_name)
  {

  }
  
  python_math_definition::python_math_definition(std::string function_name,std::vector<std::string> args) :
    math_definition(""),
    function_name(function_name),
    args(args) 
  {
    

  }


  std::shared_ptr<math_definition> python_math_definition::rebuild(std::shared_ptr<instantiated_math_function> fcn)
  {
    std::shared_ptr<python_math_definition> copy=std::make_shared<python_math_definition>(*this);
    copy->evaluate(fcn,fcn->result_channel_paths);
    return copy;
  }

  void python_math_definition::evaluate(std::shared_ptr<instantiated_math_function> instantiated,std::vector<std::shared_ptr<std::string>> result_channel_paths)
  {
    definition_command = "";
    if (result_channel_paths.size()==1 && result_channel_paths[0]){
      definition_command += "trans.math[" + escape_to_quoted_string(*result_channel_paths[0]) +"] = ";
    }
    else {
      definition_command += "(";
      for (size_t num = 0; num < result_channel_paths.size(); num++) {
        if (result_channel_paths[num]) {
          definition_command += "trans.math[" + escape_to_quoted_string(*result_channel_paths[num]) +"],";
        } else{
          definition_command += "junk,";
        }
      }
      definition_command += ") = ";
    }

    definition_command += function_name +"(";

    for (auto && arg: args) {

      definition_command += arg +",";
    }

    if (instantiated->is_mutable){

      definition_command += "mutable = True,";
    }
    
    if (instantiated->execution_tags.size() > 0) {
      
      definition_command += "execution_tags = [";
      
      for (auto && tag: instantiated->execution_tags) {
        
        definition_command += escape_to_quoted_string(tag) +",";
      }
      
      definition_command += "],";
    }

    if (instantiated->extra_params) {
      definition_command += "extra_params = extra_params_not_implemented_in_pending_math_definition,";
    }
    // Remove the final trailing comma
    if (definition_command[definition_command.size() -1] == ','){
      definition_command = definition_command.substr(0,definition_command.size() -1);
    }
    definition_command += ")";


  }
  
  void pending_math_intermediate_channels::append(std::string channel_name, std::shared_ptr<pending_math_definition_result_channel> result_chan)
  {
    if (result_chan->existing_mode){
      throw snde_error("Invalid result_chan");
    }
    intermediate_channels.push_back(std::make_pair(channel_name,result_chan));

  }
  
  pending_math_definition::pending_math_definition(std::string function_name,std::vector<std::string> args,std::shared_ptr<pending_math_intermediate_channels> intermediate_channels,size_t num_results) :
    python_math_definition(function_name,args),
    instantiated(nullptr),
    intermediate_channels(intermediate_channels->intermediate_channels),
    num_results(num_results)
  {
    

  }

  void pending_math_definition::evaluate(std::shared_ptr<instantiated_math_function> instantiated,std::vector<std::shared_ptr<std::string>> result_channel_paths)
  {
    python_math_definition::evaluate(instantiated,result_channel_paths);

      
    // Clear out our storage to eliminate reference loops
    instantiated = nullptr;
    intermediate_channels.clear();
  }
  
  // Helper function for traverse_scenegraph_find_graphics_deps
  static bool tsfgd_helper(std::shared_ptr<recording_set_state> rss,std::string channel_fullpath,const std::vector<std::string> &processing_tags,const std::set<std::string> &filter_channels,std::shared_ptr<std::set<std::string>> *deps_out)
  {
    bool found_incomplete_deps = false; 
    auto prereq_chan_it = rss->recstatus.channel_map->find(channel_fullpath);
    if (prereq_chan_it != rss->recstatus.channel_map->end()){
      // Try to find an existing recording.
      std::shared_ptr<recording_base> rec = prereq_chan_it->second.rec();

      // Check if we are filtering this channel
      auto fc_it = filter_channels.find(channel_fullpath);
      bool filtering = (fc_it != filter_channels.end());
      if (!filtering && (!rec || ((rec->info_state & SNDE_RECS_FULLYREADY) != SNDE_RECS_FULLYREADY))) {
	// We are not filtering this channel and the recording is not present or not fully ready.
	(*deps_out)->emplace(channel_fullpath);
	found_incomplete_deps = true;
      }
      else {
	// Either we are filtering this channel or the recording is present and fully ready.
	// Obtain list of referenced channels.
	// traverse recursively
	std::shared_ptr<std::set<std::string>> componentparts = rec->graphics_componentpart_channels(rss,processing_tags);
	
	for (auto && component_fullpath: *componentparts) {
	  found_incomplete_deps = tsfgd_helper(rss,component_fullpath,processing_tags,filter_channels,deps_out) || found_incomplete_deps;
	}
      }
    }
    return found_incomplete_deps; 
  }

    void annotate_incomplete_deps(std::shared_ptr<recording_set_state> rss, std::shared_ptr<instantiated_math_function> inst, math_function_status *mathstatus_ptr, const std::vector<std::string> &processing_tags, std::shared_ptr<std::set<std::string>> deps)
  {
    // We have now used the helper to find that we are dependent on
    // some additional channels (if found_incomplete_deps is
    // true).That means that we have to annotate ourselves as having
    // this extra dependency, and also that we have to arrange for
    // notifications once the recordings we are dependent on become
    // complete.
    //
    // We annotate ourselves by adding to missing_prerequisites
    // we arrange for notifications depending on whether the recording
    // we are waiting for exists or not. If the recording came from
    // a transaction, then it must exist. If it came from math, then
    // it may exist. Either way, if it exists we find its
    // originating_rss and add ourselves to the
    // external_dependencies_on_channel if the originating_rss
    // was not us. If the originating_rss was us, we add the new
    // dependency to extra_internal_dependencies_on_channel.
    // 
    // If rec does not exist then it must be the outcome of a math
    // function. We add the new dependency to the
    // extra_internal_dependencies_on_function of our rss. That way
    // when the function result is applied to our rss, it will see
    // the new dependency triggering a check whether the math function
    // we are working on can now execute. 
    //
    
    // First pass is to put anything relevant into external_dependencies_on_channel of an originating_rss while holding our rss unlocked.

    for (auto && dep: *deps) {
      auto chan_it = rss->recstatus.channel_map->find(dep);
      if (chan_it == rss->recstatus.channel_map->end()) {
	snde_warning("traverse_scenegraph_find_graphics_deps(): channel %s not found in RSS.", dep.c_str());
      }
      else {
	std::shared_ptr<recording_base> rec=chan_it->second.rec();

	if (rec){
	  if ((rec->info_state & SNDE_RECS_FULLYREADY) != SNDE_RECS_FULLYREADY){
	    std::shared_ptr<recording_set_state> originating_rss;
	    {
	      std::lock_guard<std::mutex> rec_admin(rec->admin);
	      originating_rss = rec->originating_state->rss();
	    }
	    if (originating_rss && originating_rss != rss) {
	      std::lock_guard<std::mutex> orss_admin(originating_rss->admin);
	      auto edoc = originating_rss->mathstatus.begin_atomic_external_dependencies_on_channel_update();
	      std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>::iterator edoc_it;
	      bool edoc_inserted;
	      std::tie(edoc_it,edoc_inserted)=edoc->emplace(chan_it->second.config,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>());
	      edoc_it->second.push_back(std::make_tuple(rss,inst));
	      rss->mathstatus.end_atomic_external_dependencies_on_channel_update(edoc);
	    }
	  }
	}
      }
    }

    // Now that other rss's know to notify us, it is safe
    // for us to note down our dependence on them. 
    for (auto && dep: *deps) {
      auto chan_it = rss->recstatus.channel_map->find(dep);
      if (chan_it == rss->recstatus.channel_map->end()) {
	snde_warning("traverse_scenegraph_find_graphics_deps(): channel %s not found in RSS.", dep.c_str());
      }
      else {
	std::unique_lock<std::mutex> rss_admin(rss->admin);
	std::shared_ptr<recording_base> rec=chan_it->second.rec();

	if (rec){
	  if ((rec->info_state & SNDE_RECS_FULLYREADY) != SNDE_RECS_FULLYREADY){
	    mathstatus_ptr->missing_prerequisites.emplace(chan_it->second.config);
	    std::shared_ptr<recording_set_state> originating_rss;
	    {
	      std::lock_guard<std::mutex> rec_admin(rec->admin);
	      originating_rss = rec->originating_state->rss();
	    }
	    if (originating_rss == rss) {
	     
	      auto eidoc=rss->mathstatus.begin_atomic_extra_internal_dependencies_on_channel_update();

	      std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::shared_ptr<instantiated_math_function>>>::iterator eidoc_it;
	      bool inserted = false;
	      std::tie(eidoc_it,inserted)=eidoc->emplace(chan_it->second.config,std::vector<std::shared_ptr<instantiated_math_function>>());
	      eidoc_it->second.push_back(inst);
	      rss->mathstatus.end_atomic_extra_internal_dependencies_on_channel_update(eidoc);
	    }
	    else {
	      // Already added to external_dependencies_on_channel of
	      // the originating_rss
	      
	    }
	  }
	}
	else {
	  // rec does not exist; it must be an uncertain result
	  // of a math function.
	  mathstatus_ptr->missing_prerequisites.emplace(chan_it->second.config);
	  std::shared_ptr<instantiated_math_function> dep_fcn = rss->mathstatus.math_functions->defined_math_functions.at(dep);
	  rss->mathstatus.function_status.at(dep_fcn).extra_internal_dependencies_on_function.emplace(std::make_tuple(chan_it->second.config,inst));
	  // This way, when execution_complete_notify_single_referencing_rss() looks at our rss, notify_math_function_executed() will realize that our function needs to be considered.
	    
	}
      }
    } 
  }
  
  // Dynamic dependencies have a find_additional_deps() method
  // that is needed to identify implicit dependencies.
  // This function is designed to fill that role for graphics
  // dependencies. Wrap this with a lambda that knows your
  // processing tags and assign it when the math_function is
  // created.
  //
  // Note that this only considers recordings that are not ready
  // as if the recording is ready any dependence is already
  // satisfied. 
  bool traverse_scenegraph_find_graphics_deps(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst,math_function_status *mathstatus_ptr,const std::vector<std::string> &processing_tags,const std::set<std::string> &filter_channels)
  {
    bool found_incomplete_deps = false; 

    std::shared_ptr<std::set<std::string>> deps;

    
    for (auto && parameter: inst->parameters) {
      // get prerequisite channels for this parameter
      std::set<std::string> prereq_channels = parameter->get_prerequisites(inst->channel_path_context);
      for (auto && prereq_channel: prereq_channels) {
	std::string prereq_channel_fullpath = recdb_path_join(inst->channel_path_context,prereq_channel);
	found_incomplete_deps = tsfgd_helper(rss,prereq_channel_fullpath,processing_tags,filter_channels,&deps) || found_incomplete_deps;
      }
    }

    annotate_incomplete_deps(rss,inst,mathstatus_ptr,processing_tags,deps);

    return found_incomplete_deps;
  }

  // traverse_scenegraph_find_graphics_deps() searches all explicit
  // dependencies (parameters of the math function) for referenced graphics.
  // traverse_scenegraph_find_graphics_deps_under_channel() searches
  // only within a specified channel
  
  bool traverse_scenegraph_find_graphics_deps_under_channel(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst,math_function_status *mathstatus_ptr,const std::vector<std::string> &processing_tags,const std::set<std::string> &filter_channels,std::string channel_path)
  {
    bool found_incomplete_deps = false; 

    std::shared_ptr<std::set<std::string>> deps;

    
    found_incomplete_deps = tsfgd_helper(rss,channel_path,processing_tags,filter_channels,&deps); 
      
    

    annotate_incomplete_deps(rss,inst,mathstatus_ptr,processing_tags,deps);

    return found_incomplete_deps;
  }

  
  /*  registered_math_function::registered_math_function(std::string name,std::function<std::shared_ptr<math_function>()> builder_function) :
    name(name),
    builder_function(builder_function)
  {
    
  }*/

  static std::shared_ptr<math_function_registry_map> *_math_function_registry; // default-initialized to nullptr;

  static std::mutex& math_function_registry_mutex()
  {
    // take advantage of the fact that since C++11 initialization of function statics
    // happens on first execution and is guaranteed thread-safe. This lets us
    // work around the "static initialization order fiasco" using the
    // "construct on first use idiom".
    // We just use regular pointers, which are safe from the order fiasco,
    // but we need some way to bootstrap thread-safety, and this mutex
    // is it. 
    static std::mutex regmutex; 
    return regmutex; 
  }

  
  std::shared_ptr<math_function_registry_map> math_function_registry()
  {
    std::mutex &regmutex = math_function_registry_mutex();
    std::lock_guard<std::mutex> reglock(regmutex);

    if (!_math_function_registry) {
      _math_function_registry = new std::shared_ptr<math_function_registry_map>(std::make_shared<math_function_registry_map>());
      
    }
    return *_math_function_registry;
  }

  
  int register_math_function(std::shared_ptr<math_function> fcn)
  // returns value so can be used as an initializer
  {
    math_function_registry(); // make sure registry is defined

    std::mutex &regmutex = math_function_registry_mutex();
    std::lock_guard<std::mutex> reglock(regmutex);

    // copy map and update then publish the copy
    std::shared_ptr<math_function_registry_map> new_map = std::make_shared<math_function_registry_map>(**_math_function_registry);

    std::unordered_map<std::string,std::shared_ptr<math_function>>::iterator old_function;

    old_function = new_map->find(fcn->function_name);
    if (old_function != new_map->end()) {
      snde_warning("Overriding already-registered math function %s in the static registry",fcn->function_name.c_str());
      new_map->erase(old_function);
    }
    
    new_map->emplace(fcn->function_name,fcn);

    *_math_function_registry = new_map;
    return 0;
  }

};
