#include <QApplication>
#include <QMainWindow>
#include <QStyleFactory>

#include "snde/qtrecviewer.hpp"
#include "snde/recstore_setup.hpp"

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
  snde_index revnum;
  

  if (argc < 2) {
    fprintf(stderr,"USAGE: %s <x3d_file.x3d>\n", argv[0]);
    exit(1);
  }

       
  std::shared_ptr<recdatabase> recdb; 
  recdb=std::make_shared<snde::recdatabase>();
  setup_cpu(recdb,std::thread::hardware_concurrency());
  #warning "GPU acceleration temporarily disabled for viewer."
  //setup_opencl(recdb,false,8,nullptr); // limit to 8 parallel jobs. Could replace nullptr with OpenCL platform name
  setup_storage_manager(recdb);
  std::shared_ptr<graphics_storage_manager> graphman=std::make_shared<graphics_storage_manager>("/",recdb->lowlevel_alloc,recdb->alignment_requirements,recdb->lockmgr,1e-8);
  recdb->default_storage_manager = graphman;
  
  setup_math_functions(recdb,{});
  recdb->startup();

  snde::active_transaction transact(recdb); // Transaction RAII holder

  
  std::vector<std::shared_ptr<textured_part_recording>> part_recordings = x3d_load_geometry(recdb,graphman,argv[1],"main",(void *)&main,"/",false,true); // !!!*** Try enable vertex reindexing !!!***

  std::shared_ptr<snde::globalrevision> globalrev = transact.end_transaction();
  

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