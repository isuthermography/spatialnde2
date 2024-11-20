#include "snde/recstore_setup.hpp"

#ifndef _WIN32
#include "shared_memory_allocator_posix.hpp"
#else
#include "shared_memory_allocator_win32.hpp"
#endif // !_WIN32

namespace snde {

  void setup_cpu(std::shared_ptr<recdatabase> recdb,std::set<std::string> tags,size_t nthreads)
  {

    std::set<std::string> cpu_tags(tags);
    cpu_tags.emplace("CPU");
    cpu_tags.emplace("OpenMP");
      
    std::shared_ptr<available_compute_resource_cpu> cpu = std::make_shared<available_compute_resource_cpu>(recdb,cpu_tags,nthreads);
    
    recdb->compute_resources->set_cpu_resource(cpu);

    
  }

  
  void setup_storage_manager(std::shared_ptr<recdatabase> recdb)
  {
#ifdef _WIN32
//#pragma message("No shared memory allocator available for Win32 yet. Using regular memory instead")
//    recdb->lowlevel_alloc=std::make_shared<cmemallocator>();
    recdb->lowlevel_alloc = std::make_shared<shared_memory_allocator_win32>();
    
#else
    recdb->lowlevel_alloc=std::make_shared<shared_memory_allocator_posix>();
#endif
    recdb->default_storage_manager=std::make_shared<recording_storage_manager_simple>(recdb->lowlevel_alloc,recdb->lockmgr,recdb->alignment_requirements);

    recdb->nonlocking_storage_manager = recdb->default_storage_manager; // default storage manager from previous line is nonlocking so we can use it safely
    
  }
 
  void setup_math_functions(std::shared_ptr<recdatabase> recdb,
			    std::vector<std::pair<std::string,std::shared_ptr<math_function>>> custom_math_funcs)
  {


    {
      std::lock_guard<std::mutex> recdb_admin(recdb->admin);
      std::shared_ptr<math_function_registry_map> new_math_functions=recdb->_begin_atomic_available_math_functions_update();


      for (auto && name_custom_fcn: custom_math_funcs) {
	std::string &name = name_custom_fcn.first;
	std::shared_ptr<math_function> fcn = name_custom_fcn.second;
	
	math_function_registry_map::iterator old_function = new_math_functions->find(name);
	if (old_function != new_math_functions->end()) {
	  snde_warning("Overwriting existing math function %s",name.c_str());
	  new_math_functions->erase(old_function);
	}
	
	new_math_functions->emplace(name,fcn);
      }

      // set the atomic shared pointer
      recdb->_end_atomic_available_math_functions_update(new_math_functions);
      
    }
  }

  
  
};
