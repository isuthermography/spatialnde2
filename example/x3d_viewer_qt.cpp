#include <QApplication>
#include <QMainWindow>
#include <QStyleFactory>

#include "qtrecviewer.hpp"

#include "revision_manager.hpp"
#include "arraymanager.hpp"
#include "x3d.hpp"
#include "openscenegraph_geom.hpp"

using namespace snde;

void StdErrOutput(QtMsgType type, const QMessageLogContext &context, const QString &msg)
{
    QByteArray localMsg = msg.toLocal8Bit();
    switch (type) {
    case QtDebugMsg:
        fprintf(stderr, "Debug: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    case QtInfoMsg:
        fprintf(stderr, "Info: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    case QtWarningMsg:
        fprintf(stderr, "Warning: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    case QtCriticalMsg:
        fprintf(stderr, "Critical: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    case QtFatalMsg:
        fprintf(stderr, "Fatal: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    }
}

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

       

  // The OpenCL device specification has the format:
  // <Platform or Vendor>:<CPU or GPU or ACCELERATOR>:<Device name or number>
  // double precision support is required. 
  
  // You can get the relevant platform/vendor/device identifiers using the "clinfo" program
  // Any parameter specifically specified becomes a requirement. Just specifying
  // "::" will prefer a GPU 
  std::tie(context,device,clmsgs) = get_opencl_context("::",true,NULL,NULL);

  fprintf(stderr,"%s",clmsgs.c_str());


  QApplication qapp(argc,argv);
  
  
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
  
  std::shared_ptr<mutablerecdb> recdb = std::make_shared<mutablerecdb>();

  std::shared_ptr<trm> revision_manager=std::make_shared<trm>(); /* transactional revision manager */


  // Create a command queue for the specified context and device. This logic
  // tries to obtain one that permits out-of-order execution, if available.
  cl_int clerror=0;
  
  cl_command_queue queue=clCreateCommandQueue(context,device,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&clerror);
  if (clerror==CL_INVALID_QUEUE_PROPERTIES) {
    queue=clCreateCommandQueue(context,device,0,&clerror);
    
  }
  

  //std::shared_ptr<std::vector<std::pair<std::shared_ptr<part>,std::unordered_map<std::string,metadatum>>>> parts;
  std::shared_ptr<std::vector<std::shared_ptr<mutableinfostore>>> part_infostores;
  revision_manager->Start_Transaction();
  part_infostores = x3d_load_geometry(geom,argv[1],recdb,"/",false,true); // !!!*** Try enable vertex reindexing !!!***
  revision_manager->End_Transaction();


  std::shared_ptr<osg_instancecache> geomcache;
  std::shared_ptr<osg_texturecache> texcache=std::make_shared<osg_texturecache>(geom,revision_manager,recdb,context,device,queue);

  std::shared_ptr<osg_parameterizationcache> paramcache=std::make_shared<osg_parameterizationcache>(geom,context,device,queue);
  
  geomcache=std::make_shared<osg_instancecache>(geom,recdb,paramcache,context,device,queue);


  
  QMainWindow window;

  ////hardwire QT style
  //qapp.setStyle(QStyleFactory::create("Fusion"));
  window.setAttribute(Qt::WA_AcceptTouchEvents, true);
  QTRecViewer *Viewer = new QTRecViewer(recdb,geom,revision_manager,context,device,queue,&window);

  
  {
    std::shared_ptr<mutablegeomstore> assem_infostore;
    assem_infostore=mutablegeomstore::from_partlist(recdb,"/",geom,"LoadedX3D",part_infostores);
    
    
    
    //geom->object_trees.insert(std::make_pair("LoadedX3D",assem));
    revision_manager->Start_Transaction();
    
    //std::shared_ptr<mutablegeomstore> LoadedX3D = std::make_shared<mutablegeomstore>("LoadedX3D","/LoadedX3D",recmetadata(md),geom,assem);
    
    recdb->addinfostore(assem_infostore);
    
    //for (auto & part_md : *parts) {
    //  // add normal calculation for each part from the .x3d file
    //  // warning: we don't do anything explicit here to make sure that the parts
    //  // last as long as the nomral_calculation objects....
    //  normal_calcs.push_back(normal_calculation(geom,revision_manager,part_md.first,context,device,queue));
    //}
    
    
    osg::ref_ptr<snde::OSGComponent> OSGComp;
    // ***!!!!! (is OSGComp used? I don't think so)
    //OSGComp=new snde::OSGComponent(geom,geomcache,paramcache,texcache,recdb,revision_manager,LoadedX3D->fullname,Viewer->display); // OSGComp must be created during a transaction...

    
    revnum=revision_manager->End_Transaction();
  }

  qInstallMessageHandler(StdErrOutput);

  //qapp.setNavigationMode(Qt::NavigationModeNone);
  window.setCentralWidget(Viewer);
  window.show();

  return qapp.exec();
 
}
