#include "snde/recstore_setup.hpp"

#ifndef _WIN32
#include "shared_memory_allocator_posix.hpp"
#endif // !_WIN32

namespace snde {

  void setup_cpu(std::shared_ptr<recdatabase> recdb,size_t nthreads)
  {


    std::shared_ptr<available_compute_resource_cpu> cpu = std::make_shared<available_compute_resource_cpu>(recdb,nthreads);
    
    recdb->compute_resources->set_cpu_resource(cpu);

    
  }

  
  void setup_storage_manager(std::shared_ptr<recdatabase> recdb)
  {
#ifdef _WIN32
#pragma message("No shared memory allocator available for Win32 yet. Using regular memory instead")
    recdb->lowlevel_alloc=std::make_shared<cmemallocator>();
    
#else
    recdb->lowlevel_alloc=std::make_shared<shared_memory_allocator_posix>();
#endif
    recdb->default_storage_manager=std::make_shared<recording_storage_manager_simple>(recdb->lowlevel_alloc,recdb->lockmgr,recdb->alignment_requirements);
    
  }
 
  void setup_math_functions(std::shared_ptr<recdatabase> recdb,
			    std::vector<std::pair<std::string,std::shared_ptr<math_function>>> custom_math_funcs)
  {

    // First, include all functions from the compiled-in registry
    

    
    std::map<std::string,std::shared_ptr<math_function>> additional_functions;
    
    std::shared_ptr<function_registry_map> compiled_in=math_function_registry();
    for (auto && name_registeredbuilder: *compiled_in) {
      const std::string &name = name_registeredbuilder.first;
      std::shared_ptr<registered_math_function> registeredbuilder = name_registeredbuilder.second;

      std::shared_ptr<math_function> additional_math_function = registeredbuilder->builder_function();

      additional_functions.emplace(name,additional_math_function);
      
    }


    {
      std::lock_guard<std::mutex> recdb_admin(recdb->admin);
      std::shared_ptr<std::map<std::string,std::shared_ptr<math_function>>> new_math_functions=recdb->_begin_atomic_math_functions_update();
      
      for (auto && name_additional_fcn: additional_functions) {
	const std::string &name = name_additional_fcn.first;
	std::shared_ptr<math_function> fcn = name_additional_fcn.second;
	
	std::map<std::string,std::shared_ptr<math_function>>::iterator old_function = new_math_functions->find(name);
	if (old_function != new_math_functions->end()) {
	  throw snde_error("Attempting to overwrite existing math function %s",name.c_str());
	}
	
	new_math_functions->emplace(name,fcn);
      }


      for (auto && name_custom_fcn: custom_math_funcs) {
	std::string &name = name_custom_fcn.first;
	std::shared_ptr<math_function> fcn = name_custom_fcn.second;
	
	std::map<std::string,std::shared_ptr<math_function>>::iterator old_function = new_math_functions->find(name);
	if (old_function != new_math_functions->end()) {
	  throw snde_error("Attempting to overwrite existing math function %s",name.c_str());
	}
	
	new_math_functions->emplace(name,fcn);
      }

      // set the atomic shared pointer
      recdb->_end_atomic_math_functions_update(new_math_functions);
      
    }
  }

  
  
};
