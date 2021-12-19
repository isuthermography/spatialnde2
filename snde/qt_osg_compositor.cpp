
#include "snde/qt_osg_compositor.hpp"
#include "snde/openscenegraph_renderer.hpp"

namespace snde {

  qt_osg_worker_thread::qt_osg_worker_thread(qt_osg_compositor *comp,QObject *parent) :
    QThread(parent),
    comp(comp)
  {

  }
  
  void qt_osg_worker_thread::run()
  {
    comp->worker_code();
  }

  void qt_osg_worker_thread::emit_need_update()
  {
    emit compositor_need_update();
  }


  
  static bool confirm_threaded_opengl(bool enable_threaded_opengl)
  {
    bool platform_support = QOpenGLContext::supportsThreadedOpenGL();

    if (enable_threaded_opengl && !platform_support) {
      snde_warning("qt_osg_compositor: Threaded OpenGL disabled because of a lack of platform support");
      return false;
    }

    return enable_threaded_opengl;
  }
  
  qt_osg_compositor::qt_osg_compositor(std::shared_ptr<recdatabase> recdb,
				       std::shared_ptr<display_info> display,
				       osg::ref_ptr<osgViewer::Viewer> Viewer,osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
				       bool threaded,bool enable_threaded_opengl,
				       QPointer<QTRecViewer> Parent_Viewer,QWidget *parent/*=nullptr*/) :
    QOpenGLWidget(parent),
    osg_compositor(recdb,display,Viewer,new osg_ParanoidGraphicsWindowEmbedded(0,0,width(),height()),
		   threaded,confirm_threaded_opengl(enable_threaded_opengl),defaultFramebufferObject() /* probably 0 as QT OGL hasn't started... we'll assign an updated value in initializeGL() below */ ),
    RenderContext(nullptr),
    qt_worker_thread(nullptr),
    Parent_Viewer(Parent_Viewer)
  {
    
  }

  qt_osg_compositor::~qt_osg_compositor()
  {
    // call stop before any other destruction happens so that our objects are still valid
    stop();

    // our superclass will call stop() again but it won't matter because the above call
    // will have already dealt with everything. 
  }
  
  void qt_osg_compositor::initializeGL()
  {
    // called once our context is created by QT and after any
    // reparenting (which would trigger a new context)

    if (threaded && enable_threaded_opengl) {
      RenderContext = new QOpenGLContext(this);
      RenderContext->setShareContext(context());
      RenderContext->setScreen(screen());
      DummyOffscreenSurface = new QOffscreenSurface(screen(),this);
      DummyOffscreenSurface->setFormat(context()->format());
      DummyOffscreenSurface->create();
      
      RenderContext->create();
      
      LayerDefaultFramebufferObject = RenderContext->defaultFramebufferObject();
    } else {
      LayerDefaultFramebufferObject = context()->defaultFramebufferObject();
    }
    
    start(); // make sure threads are going

    
    if (threaded && enable_threaded_opengl) {
      assert(qt_worker_thread);
      RenderContext->moveToThread(qt_worker_thread);
      DummyOffscreenSurface->moveToThread(qt_worker_thread);
    }
  }


  void qt_osg_compositor::worker_code()
  {
    {
      std::lock_guard<std::mutex> adminlock(admin);
      worker_thread_id = std::make_shared<std::thread::id>(std::this_thread::get_id());
      
      execution_notify.notify_all(); // notify parent we have set the worker id
    }
    // regular dispatch
    dispatch(false,true,true);

    // termination
    // notify parent that we are done by clearing the worker id
    
    {
      std::lock_guard<std::mutex> adminlock(admin);
      worker_thread_id = nullptr;
      
      execution_notify.notify_all(); // notify parent we have set the worker id
    }
  }

  
  void qt_osg_compositor::_start_worker_thread()
  {
    // start worker thread as a QThread instead of std::thread
    if (threaded) {
      //qt_worker_thread = QThread::create([ this ]() { this->worker_code(); });
      //qt_worker_thread->setParent(this);
      qt_worker_thread = new qt_osg_worker_thread(this,this); // just calls worker_code() method
      // connect output signal of worker thread to this (QOpenGLWidget update slot)
      connect(qt_worker_thread,&qt_osg_worker_thread::compositor_need_update,this,&qt_osg_compositor::update);
      qt_worker_thread->start();
      

      // Wait for worker thread to set it's ID (protected by admin lock) 
      {
	std::unique_lock<std::mutex> adminlock(admin);
	execution_notify.wait(adminlock,[ this ]() { return (bool)worker_thread_id; });
      }

      
    } else {
      worker_thread_id = std::make_shared<std::thread::id>(std::this_thread::get_id());
    }

    threads_started=true; 
  }

  void qt_osg_compositor::_join_worker_thread()
  {
    if (threaded && threads_started) {
      // worker thread clearing its ID is our handshake that it is finished.
      
      
      {
	std::unique_lock<std::mutex> adminlock(admin);
	execution_notify.wait(adminlock,[ this ]() { return (bool)!worker_thread_id; });
      }
      
      // now that it is done and returning, call deleteLater() on it
      qt_worker_thread->deleteLater();
      qt_worker_thread = nullptr; // no longer responsible for thread object (qt main loop will perform actual join, etc.) 
    }
    // single threaded again from here on. 
    
    threads_started=false;
    worker_thread_id=nullptr; 

  }

  void qt_osg_compositor::perform_ondemand_calcs()
  {
    // wrap osg_compositor::perform_ondemand_calcs so that after layer
    // ondemand calcs are done we will rerender.

    // Only needed if threaded is enabled but threaded_opengl is not enabled because in that
    // circumstance we are in a different thread and need to trigger the main thread
    // to get a paint callback to do the rendering.
    
    osg_compositor::perform_ondemand_calcs();

    if (threaded && !enable_threaded_opengl) {
      qt_worker_thread->emit_need_update();
    }
    
  }

  void qt_osg_compositor::perform_layer_rendering()
  {
    // wrap osg_compositor::perform_layer_rendering so that after layer
    // rendering is done we will repaint.

    // Only needed if threaded_opengl is enabled because in that
    // circumstance we are in a different thread and need to trigger the main thread
    // to get a paint callback to do the compositing.

    if (threaded && enable_threaded_opengl) {
      // This is in the worker thread and we are allowed to
      // make OpenGL calls here
      // ... but only after making our QOpenGLContext current.
      RenderContext->makeCurrent(DummyOffscreenSurface);
    }
    
    osg_compositor::perform_layer_rendering();

    if (threaded && enable_threaded_opengl) {

      // undo the makeCurrent above
      RenderContext->doneCurrent();
      
      // if we are doing the rendering in a separate thread,
      // then we need to wake up the main loop now so it
      // can do compositing next.
      qt_worker_thread->emit_need_update();
    }
    
  }


  void qt_osg_compositor::paintGL()
  {
    // mark that at minimum we need a recomposite 
    {
      std::lock_guard<std::mutex> adminlock(admin);
      need_recomposite=true;
    }
    // execute up to one full rendering pass but don't allow waiting in the QT main thread main loop
    dispatch(true,false,false);
  }
}
