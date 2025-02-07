#include <QApplication>
#include <QMainWindow>
#include <QStyleFactory>

#include "snde/qtrecviewer.hpp"
#include "snde/recstore_setup.hpp"
#ifdef SNDE_OPENCL
#include "snde/recstore_setup_opencl.hpp"
#endif

#include "snde/x3d.hpp"

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
  

  if (argc < 2) {
    fprintf(stderr,"USAGE: %s <x3d_file.x3d>\n", argv[0]);
    exit(1);
  }
       
  std::shared_ptr<recdatabase> recdb; 
  recdb=std::make_shared<snde::recdatabase>();
  setup_cpu(recdb,{},std::thread::hardware_concurrency());
#ifdef SNDE_OPENCL
  setup_opencl(recdb,{},false,8,nullptr); // limit to 8 parallel jobs. Could replace nullptr with OpenCL platform name
  //#warning "GPU acceleration temporarily disabled for viewer."
#endif
  setup_storage_manager(recdb);
  std::shared_ptr<graphics_storage_manager> graphman=std::make_shared<graphics_storage_manager>("/",recdb->lowlevel_alloc,recdb->alignment_requirements,recdb->lockmgr,1e-8,2e9);
  //recdb->default_storage_manager = graphman;
  
  setup_math_functions(recdb,{});
  recdb->startup();

  std::shared_ptr<snde::active_transaction> transact=recdb->start_transaction(); // Transaction RAII holder

  
  std::shared_ptr<loaded_part_geometry_recording> part_recording = x3d_load_geometry(transact,graphman,argv[1],0,"main","/loaded_x3d/",nullptr,{ /* "reindex_vertices", */ "reindex_tex_vertices" } ); 

  std::shared_ptr<snde::globalrevision> globalrev = transact->end_transaction()->globalrev();
  

  QCoreApplication::setAttribute(Qt::AA_UseDesktopOpenGL); // OpenSceneGraph requires UseDesktopOpenGL, I think
  QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts); // Eliminate annoying QT warning message
  QApplication qapp(argc,argv);  
  QMainWindow window;

  ////hardwire QT style
  //qapp.setStyle(QStyleFactory::create("Fusion"));
  window.setAttribute(Qt::WA_AcceptTouchEvents, true);
  QTRecViewer *Viewer = new QTRecViewer(recdb,&window);
  
  
  
  qInstallMessageHandler(StdErrOutput);
  
  //qapp.setNavigationMode(Qt::NavigationModeNone);
  window.setCentralWidget(Viewer);
  window.show();

  qapp.exec();

  return 0;
}
