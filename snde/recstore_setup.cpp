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
    std::shared_ptr<memallocator> lowlevel_alloc=std::make_shared<cmemallocator>();
    
#else
    std::shared_ptr<memallocator> lowlevel_alloc=std::make_shared<shared_memory_allocator_posix>();
#endif
    recdb->default_storage_manager=std::make_shared<recording_storage_manager_simple>(lowlevel_alloc,recdb->lockmgr,recdb->alignment_requirements);
    
  }
  
};
