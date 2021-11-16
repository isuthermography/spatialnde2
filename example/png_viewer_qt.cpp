#include <QApplication>
#include <QMainWindow>
#include <QStyleFactory>

#include "qtrecviewer.hpp"

#include "revision_manager.hpp"
#include "arraymanager.hpp"
#include "pngimage.hpp"

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
  
  std::shared_ptr<mutablerecdb> recdb = std::make_shared<mutablerecdb>();

  std::shared_ptr<trm> revision_manager=std::make_shared<trm>(); /* transactional revision manager */


  // Create a command queue for the specified context and device. This logic
  // tries to obtain one that permits out-of-order execution, if available.
  cl_int clerror=0;
  
  cl_command_queue queue=clCreateCommandQueue(context,device,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&clerror);
  if (clerror==CL_INVALID_QUEUE_PROPERTIES) {
    queue=clCreateCommandQueue(context,device,0,&clerror);
    
  }
  
  
  revision_manager->Start_Transaction();
  std::shared_ptr<mutabledatastore> pngstore = ReadPNG(manager,"PNGFile","PNGFile",argv[1]);
  recdb->addinfostore(pngstore);

  std::shared_ptr<mutabledatastore> pngstore2 = ReadPNG(manager,"PNGFile2","PNGFile2",argv[2]);
  recdb->addinfostore(pngstore2);
  revision_manager->End_Transaction();

  qInstallMessageHandler(StdErrOutput);
     
  QApplication qapp(argc,argv);

  //qapp.setNavigationMode(Qt::NavigationModeNone);
  
  QMainWindow window;

  ////hardwire QT style
  //qapp.setStyle(QStyleFactory::create("Fusion"));
  window.setAttribute(Qt::WA_AcceptTouchEvents, true);
  QTRecViewer *Viewer = new QTRecViewer(recdb,geom,revision_manager,context,device,queue,&window);
  window.setCentralWidget(Viewer);
  window.show();

  return qapp.exec();
 
}
