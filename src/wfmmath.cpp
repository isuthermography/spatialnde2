#include "wfmmath.hpp"


namespace snde {
  math_function::math_function(size_t num_results,const std::list<std::tuple<std::string,unsigned>> &param_names_types) :
    num_results(num_results),
    param_names_types(param_names_types)
  {
    new_revision_optional=false;
    pure_optionally_mutable=false;
    mandatory_mutable=false;
    self_dependent=false; 
  }

  instantiated_math_function() :
    disabled(false),
    ondemand(false),
    mdonly(false)
  {
    
  }
  
  virtual std::shared_ptr<instantiated_math_function> instantiated_math_function::clone()
  {
    // This code needs to be repeated in every derived class with the make_shared<> referencing the derived class
    std::shared_ptr<instantiated_math_function> copy = std::make_shared<instaniated_math_function>(*this);

    // see comment at start of definition of class instantiated_math_function
    if (definition) {
      assert(!copy->original_function);
      copy->original_function = definition;
      copy->definition = nullptr; 
    }
    return copy;
  }

  // rebuild all_dependencies_of_channel hash table. Must be called any time any of the defined_math_functions changes. May only be called for the instantiated_math_database within the main waveform database, and the main waveform database admin lock must be locked when this is called. 
  void instantiated_math_database::rebuild_dependency_map()
  {
#error not yet implemented
  }

  math_function_status::math_function_status(bool mdonly,bool is_mutable) :
    mdonly(mdonly),
    is_mutable(is_mutable),
    ready_to_execute(false),
    execution_demanded(false),
    metadataonly_complete(false),
    complete(false)    
  {

  }
  
  math_status::math_status(std::shared_ptr<instantiated_math_database> math_functions,const std::map<std::string,channel_state> & channel_map) :
    math_functions(math_functions)
  {

    // put all math functions into function_status and _external_dependencies databases and into pending_functions?
    for (auto && math_function_ptr: math_function->defined_math_functions) {
      if (function_status.find(math_function_ptr) != function_status.end()) {
	function_status.emplace(std::piecewise_construct, // ***!!! Need to fill out prerequisites somewhere, but not sure constructor is the place!!!*** ... NO. prerequisites are filled out in end_transaction(). 
				std::forward_as_tuple(math_function_ptr),
				std::forward_as_tuple(math_function_ptr->mdonly,math_function_ptr->fcn->mandatory_mutable)); // For now we only set the mutable flag for mandatory_mutable functions. In the future we can also enable it when suitable for pure_optionally_mutable functions. 
	_external_dependencies_on_function.emplace(std::piecewise_construct,
						   std::forward_as_tuple(math_function_ptr),
						   std::forward_as_tuple());

	
	if (math_function_ptr->mdonly) {
	  mdonly_pending_functions.emplace(math_function_ptr);
	} else {
	  pending_functions.emplace(math_function_ptr);	  
	}
	
	
      }
    }

    for (auto && channel_path_channel_state: channel_map) {
      _external_dependencies_on_channel.emplace(std::piecewise_construct,
						std::forward_as_tuple(channel_path_channel_state.second->config),
						std::forward_as_tuple());
      
    }
  }

};
