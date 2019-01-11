#include <QApplication>
#include <QMainWindow>

#include "qtwfmviewer.hpp"

#include "revision_manager.hpp"
#include "arraymanager.hpp"
#include "pngimage.hpp"

using namespace snde;

int main(int argc, char **argv)
{
  cl_context context;
  cl_device_id device;
  std::string clmsgs;
  snde_index revnum;
  

  if (argc < 2) {
    fprintf(stderr,"USAGE: %s <png_file.png>\n", argv[0]);
    exit(1);
  }

  std::tie(context,device,clmsgs) = get_opencl_context("::",true,NULL,NULL);

  fprintf(stderr,"%s",clmsgs.c_str());


  

  std::shared_ptr<memallocator> lowlevel_alloc;
  std::shared_ptr<allocator_alignment> alignment_requirements;
  std::shared_ptr<arraymanager> manager;
  std::shared_ptr<geometry> geom;
  
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

  geom=std::make_shared<geometry>(1e-6,manager);
  
  std::shared_ptr<trm> revision_manager=std::make_shared<trm>(); /* transactional revision manager */

  // Create a command queue for the specified context and device. This logic
  // tries to obtain one that permits out-of-order execution, if available.
  cl_int clerror=0;
  
  cl_command_queue queue=clCreateCommandQueue(context,device,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&clerror);
  if (clerror==CL_INVALID_QUEUE_PROPERTIES) {
    queue=clCreateCommandQueue(context,device,0,&clerror);
    
  }
  
  std::shared_ptr<mutablewfmdb> wfmdb = std::make_shared<mutablewfmdb>();
  
  revision_manager->Start_Transaction();
  std::shared_ptr<mutabledatastore> pngstore = ReadPNG(manager,"PNGFile",argv[1]);
  wfmdb->addinfostore(pngstore);
  revision_manager->End_Transaction();
  
  QApplication qapp(argc,argv);
  QMainWindow window;
  QTWfmViewer *Viewer = new QTWfmViewer(wfmdb,geom,revision_manager,context,device,queue,&window);
  window.setCentralWidget(Viewer);
  window.show();

  return qapp.exec();
 
}
