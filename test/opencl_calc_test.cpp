#include <unistd.h>


#include "openscenegraph_geom.hpp"
#include "revision_manager.hpp"
#include "arraymanager.hpp"
#include "normal_calculation.hpp"
#include "inplanemat_calculation.hpp"
#include "projinfo_calculation.hpp"
#include "x3d.hpp"

using namespace snde;

// The original purpose of this was to be able to run oclgrind, which isn't thread safe (contrary to spec)
// That still doesn't work, but it is useful for testing calculations with valgrind via pocl
// or testing out calculation inconsistency by running different OCL platforms


int main(int argc, char **argv)
{
  cl_context context;
  cl_device_id device;
  std::string clmsgs;
  snde_index revnum;
  
  std::shared_ptr<geometry> geom;
  std::shared_ptr<trm> revision_manager; /* transactional revision manager */


  if (argc < 2) {
    fprintf(stderr,"USAGE: %s <x3d_file.x3d>\n", argv[0]);
    fprintf(stderr,"\n");
    fprintf(stderr,"Supply a single x3d file with a single IndexedFaceSet or IndexedTriangleSet geometry\n");
    exit(1);
  }

  std::tie(context,device,clmsgs) = get_opencl_context(":GPU:",true,NULL,NULL);

  fprintf(stderr,"%s",clmsgs.c_str());

  
  

  std::shared_ptr<memallocator> lowlevel_alloc;
  std::shared_ptr<allocator_alignment> alignment_requirements;
  std::shared_ptr<arraymanager> manager;
  
  // lowlevel_alloc performs the actual host-side memory allocations
  lowlevel_alloc=std::make_shared<cmemallocator>();


  // alignment requirements specify constraints on allocation
  // block sizes
  alignment_requirements=std::make_shared<allocator_alignment>();
  // Each OpenCL device can impose an alignment requirement...
  add_opencl_alignment_requirement(alignment_requirements,device);
  
  // the arraymanager handles multiple arrays, including
  //   * Allocating space, reallocating when needed
  //   * Locking (implemented by manager.locker)
  //   * On-demand caching of array data to GPUs 
  manager=std::make_shared<arraymanager>(lowlevel_alloc,alignment_requirements);

  // geom is a C++ wrapper around a C data structure that
  // contains multiple arrays to be managed by the
  // arraymanager. These arrays are managed in
  // groups. All arrays in a group are presumed
  // to have parallel content, and are allocated,
  // freed, and locked in parallel.

  // Note that this initialization (adding arrays to
  // the arraymanager) is presumed to occur in a single-
  // threaded environment, whereas execution can be
  // freely done from multiple threads (with appropriate
  // locking of resources) 
  geom=std::make_shared<geometry>(1e-6,manager);

  std::shared_ptr<mutablerecdb> recdb = std::make_shared<mutablerecdb>();
  
  revision_manager=std::make_shared<trm>(); /* transactional revision manager */

  // Create a command queue for the specified context and device. This logic
  // tries to obtain one that permits out-of-order execution, if available.
  cl_int clerror=0;
  
  cl_command_queue queue=clCreateCommandQueue(context,device,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&clerror);
  if (clerror==CL_INVALID_QUEUE_PROPERTIES) {
    queue=clCreateCommandQueue(context,device,0,&clerror);
    
  }

  //std::vector<std::shared_ptr<trm_dependency>> normal_calcs;

  
  std::shared_ptr<std::vector<std::shared_ptr<mutableinfostore>>> part_infostores;
  {
    revision_manager->Start_Transaction();
    
    //std::shared_ptr<std::vector<trm_arrayregion>> modified = std::make_shared<std::vector<trm_arrayregion>>();

    
    part_infostores = x3d_load_geometry(geom,argv[1],recdb,"/",false,true); // !!!*** Try enable vertex reindexing !!!***
    
    revnum=revision_manager->End_Transaction();
    revision_manager->Wait_Computation(revnum);
  }

  std::shared_ptr<trm_dependency> normal_calc_depend; // in larger context so that it doesn't disappear on us from going out of scope
  std::shared_ptr<trm_dependency> inplanemat_calc_depend; // in larger context so that it doesn't disappear on us from going out of scope
  std::shared_ptr<trm_dependency> projinfo_calc_depend; // in larger context so that it doesn't disappear on us from going out of scope
  {
    std::shared_ptr<mutablegeomstore> x3d_part_store;
    std::shared_ptr<part> x3d_part;

    std::shared_ptr<mutableparameterizationstore> x3d_param_store;
    std::shared_ptr<parameterization> x3d_param;

    for (auto & infostore : *part_infostores) {
      if (std::dynamic_pointer_cast<mutablegeomstore>(infostore)) {
	x3d_part_store = std::dynamic_pointer_cast<mutablegeomstore>(infostore);
	//x3d_part = std::dynamic_pointer_cast<part>(x3d_part_store->comp());
	x3d_part = std::dynamic_pointer_cast<part>(x3d_part_store->comp());
      }
      if (std::dynamic_pointer_cast<mutableparameterizationstore>(infostore)) {
	x3d_param_store = std::dynamic_pointer_cast<mutableparameterizationstore>(infostore);
	x3d_param = x3d_param_store->param();
      }
    }

    
    revision_manager->Start_Transaction();
    
    //std::shared_ptr<std::vector<trm_arrayregion>> modified = std::make_shared<std::vector<trm_arrayregion>>();

    normal_calc_depend = normal_calculation(geom,revision_manager,x3d_part,context,device,queue);
    
    revnum=revision_manager->End_Transaction();
    revision_manager->Wait_Computation(revnum);

    
    revision_manager->Start_Transaction();    
    //std::shared_ptr<std::vector<trm_arrayregion>> modified = std::make_shared<std::vector<trm_arrayregion>>();

    inplanemat_calc_depend = inplanemat_calculation(geom,revision_manager,x3d_part,context,device,queue);
    
    revnum=revision_manager->End_Transaction();
    revision_manager->Wait_Computation(revnum);

    revision_manager->Start_Transaction();    
    //std::shared_ptr<std::vector<trm_arrayregion>> modified = std::make_shared<std::vector<trm_arrayregion>>();

    projinfo_calc_depend = projinfo_calculation(recdb,"/",x3d_part_store->fullname,geom,revision_manager,x3d_part,x3d_param,context,device,queue);
    
    revnum=revision_manager->End_Transaction();
    revision_manager->Wait_Computation(revnum);

  }


  exit(0);

}
