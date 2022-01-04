#include <QMouseEvent>
#include <QWheelEvent>
#include <QTouchEvent>
#include <QGuiApplication>

#include "snde/qt_osg_compositor.hpp"
#include "snde/openscenegraph_renderer.hpp"
#include "snde/colormap.h"
#include "snde/rec_display.hpp"

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
				       osg::ref_ptr<osgViewer::Viewer> Viewer,
				       bool threaded,bool enable_threaded_opengl,
				       QPointer<QTRecViewer> Parent_QTRViewer,QWidget *parent/*=nullptr*/) :
    QOpenGLWidget(parent),
    osg_compositor(recdb,display,Viewer,new osg_ParanoidGraphicsWindowEmbedded(0,0,width(),height()),
		   threaded,confirm_threaded_opengl(enable_threaded_opengl),defaultFramebufferObject() /* probably 0 as QT OGL hasn't started... we'll assign an updated value in initializeGL() below */ ),
    RenderContext(nullptr),
    qt_worker_thread(nullptr),
    Parent_QTRViewer(Parent_QTRViewer)
  {
    Viewer->setThreadingModel(osgViewer::Viewer::SingleThreaded);
    Viewer->getCamera()->setViewport(new osg::Viewport(0,0,width(),height()));
    Viewer->getCamera()->setGraphicsContext(GraphicsWindow);
    
    //AnimTimer = new QTimer(this);
    //AnimTimer->setInterval(16); // 16 ms ~ 60 Hz

    setMouseTracking(true); // ???
    setAttribute(Qt::WA_AcceptTouchEvents,true);

    //QObject::connect(AnimTimer,SIGNAL(timeout()),this,SLOT(update()));

  }

  qt_osg_compositor::~qt_osg_compositor()
  {
    // call stop before any other destruction happens so that our objects are still valid
    stop();

    // our superclass will call stop() again but it won't matter because the above call
    // will have already dealt with everything. 
  }

  void qt_osg_compositor::trigger_rerender()
  {

    // perform OSG event traversal prior to rendering so as to be able to process
    // mouse events, etc. BEFORE compositing
    snde_debug(SNDE_DC_RENDERING,"trigger_rerender()");
    Viewer->eventTraversal();
    
    osg_compositor::trigger_rerender();

    if (!threaded) {
      // if not threaded, we need a paint callback
      update();
    }
  }

  void qt_osg_compositor::initializeGL()
  {
    // called once our context is created by QT and after any
    // reparenting (which would trigger a new context)

    // tell our graphics window that OpenGL has been initialized. 
    (dynamic_cast<osg_ParanoidGraphicsWindowEmbedded *>(GraphicsWindow.get()))->gl_is_available();
    if (threaded && enable_threaded_opengl) {
      RenderContext = new QOpenGLContext();
      RenderContext->setShareContext(context());
      QScreen* pScreen = QGuiApplication::screenAt(mapToGlobal({ width() / 2,0 }));
      RenderContext->setScreen(pScreen);
      DummyOffscreenSurface = new QOffscreenSurface(pScreen);
      DummyOffscreenSurface->setFormat(context()->format());
      DummyOffscreenSurface->create();
      
      RenderContext->create();
      
      LayerDefaultFramebufferObject = RenderContext->defaultFramebufferObject();
    } else {
      LayerDefaultFramebufferObject = context()->defaultFramebufferObject();
    }
    
    start(); // make sure threads are going

    // This next code moved into _start_worker_thread() so it can happen
    // guaranteed before the thread tries to access RenderContext
    //if (threaded && enable_threaded_opengl) {
    //  assert(qt_worker_thread);
    //  RenderContext->moveToThread(qt_worker_thread);
    //  DummyOffscreenSurface->moveToThread(qt_worker_thread);
    //}
  }


  void qt_osg_compositor::worker_code()
  {
    {
      std::unique_lock<std::mutex> adminlock(admin);
      worker_thread_id = std::make_shared<std::thread::id>(std::this_thread::get_id());
      
      execution_notify.notify_all(); // notify parent we have set the worker id

      // now wait for parent to set the responsibility_mapping and notify us

      execution_notify.wait( adminlock, [ this ]() { return responsibility_mapping.size() > 0; });
    }

    
    
    // regular dispatch
    try {
      dispatch(false,true,true);
    } catch (const std::exception &exc) {
      snde_warning("Exception class %s caught in osg_compositor::worker_code: %s",typeid(exc).name(),exc.what());
      
      
    }
    
    // termination
    // notify parent that we are done by clearing the worker id
    
    {
      std::lock_guard<std::mutex> adminlock(admin);
      worker_thread_id = nullptr;
      
      execution_notify.notify_all(); // notify parent we have set the worker id
    }
  }

  
  void qt_osg_compositor::_start_worker_thread(std::unique_lock<std::mutex> *adminlock)
  {
    // start worker thread as a QThread instead of std::thread
    if (threaded) {
      //qt_worker_thread = QThread::create([ this ]() { this->worker_code(); });
      //qt_worker_thread->setParent(this);
      qt_worker_thread = new qt_osg_worker_thread(this,this); // just calls worker_code() method
      // connect output signal of worker thread to this (QOpenGLWidget update slot)
      bool success;
      success = connect(qt_worker_thread,&qt_osg_worker_thread::compositor_need_update,this,&qt_osg_compositor::update);
      assert(success);
      qt_worker_thread->start();
      

      // Wait for worker thread to set it's ID (protected by admin lock) 
      execution_notify.wait(*adminlock,[ this ]() { return (bool)worker_thread_id; });
      
      
    } else {
      worker_thread_id = std::make_shared<std::thread::id>(std::this_thread::get_id());
    }

    // Move the rendering context and dummy surface to our newly created thread
    if (threaded && enable_threaded_opengl) {
      assert(qt_worker_thread);
      RenderContext->moveToThread(qt_worker_thread);
      DummyOffscreenSurface->moveToThread(qt_worker_thread);

      snde_debug(SNDE_DC_RENDERING,"RC and DOC: movetothread 0x%llx 0x%llx 0x%llx",(unsigned long long)RenderContext,(unsigned long long)DummyOffscreenSurface,(unsigned long long)qt_worker_thread);
    }

    
    threads_started=true; 
    // Note: worker_thread will still be waiting for us to setup the thread_responsibilities
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

  void qt_osg_compositor::perform_ondemand_calcs(std::unique_lock<std::mutex> *adminlock)
  {
    // wrap osg_compositor::perform_ondemand_calcs so that after layer
    // ondemand calcs are done we will rerender.

    // Only needed if threaded is enabled but threaded_opengl is not enabled because in that
    // circumstance we are in a different thread and need to trigger the main thread
    // to get a paint callback to do the rendering.
    
    osg_compositor::perform_ondemand_calcs(adminlock);

    //if (threaded && !enable_threaded_opengl) {
    //  qt_worker_thread->emit_need_update();
    //}
    
  }

  void qt_osg_compositor::perform_layer_rendering(std::unique_lock<std::mutex> *adminlock)
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
    
    osg_compositor::perform_layer_rendering(adminlock);

    if (threaded && enable_threaded_opengl) {

      // undo the makeCurrent above
      RenderContext->doneCurrent();
      
      // if we are doing the rendering in a separate thread,
      // then we need to wake up the main loop now so it
      // can do compositing next.
      //qt_worker_thread->emit_need_update();
    }
    
  }


  void qt_osg_compositor::perform_compositing(std::unique_lock<std::mutex> *adminlock)
  {

    
    osg_compositor::perform_compositing(adminlock);

    // The code below was needed at the layer level; may or may not be needed here. 
    // Push a dummy event prior to the frame on the queue
    // without this we can't process events on our pseudo-GraphicsWindow because
    // osgGA::EventQueue::takeEvents() looks for an event prior to the cutoffTime
    // when selecting events to take. If it doesn't find any then you don't get any
    // events (?).
    // The cutofftime comes from renderer->Viewer->_frameStamp->getReferenceTime()
    osg::ref_ptr<osgGA::Event> dummy_event = new osgGA::Event();
    dummy_event->setTime(Viewer->getFrameStamp()->getReferenceTime()-1.0);
    GraphicsWindow->getEventQueue()->addEvent(dummy_event);

    snde_debug(SNDE_DC_RENDERING,"Dummy events added; need_recomposite=%d",(int)need_recomposite);
    
    // enable continuous updating if requested 
    /*
    if (request_continuous_update) {
      if (!AnimTimer->isActive()) {
	AnimTimer->start();
	fprintf(stderr,"Starting animation timer!\n");
      }
    } else {
      fprintf(stderr,"Manipulator not animating\n");
      if (AnimTimer->isActive()) {
	AnimTimer->stop();
      }
      
    }

    */
  }


  void qt_osg_compositor::wake_up_ondemand_locked(std::unique_lock<std::mutex> *adminlock)
  {
    if (threaded) {
      execution_notify.notify_all();
    }
  }
  
  void qt_osg_compositor::wake_up_renderer_locked(std::unique_lock<std::mutex> *adminlock)
  {
    if (threaded && enable_threaded_opengl) {
      execution_notify.notify_all();
    } else if (threaded && !enable_threaded_opengl) {

      // Need GUI update
      adminlock->unlock();
      if (std::this_thread::get_id() == *worker_thread_id) {
	qt_worker_thread->emit_need_update();
      } else {
	// emit update(); // (commented out because if we are the GUI thread then we are already awake!)
      }
      adminlock->lock();

    }

  }

  void qt_osg_compositor::wake_up_compositor_locked(std::unique_lock<std::mutex> *adminlock)
  {
    adminlock->unlock();

    // Need GUI update
    if (std::this_thread::get_id() == *worker_thread_id) {
      qt_worker_thread->emit_need_update();
    } else {
      // emit update(); // (commented out because if we are the GUI thread then we are already awake!)
    }
    adminlock->lock();
  }

  void qt_osg_compositor::clean_up_renderer_locked(std::unique_lock<std::mutex> *adminlock)
  {
    
    osg_compositor::clean_up_renderer_locked(adminlock);

    if (threaded && enable_threaded_opengl) {
      delete RenderContext; // OK because it's not owned by another QObject
      delete DummyOffscreenSurface; // OK because it's not owned by another QObject
    }
  }

  
  void qt_osg_compositor::paintGL()
  {
    // mark that at minimum we need a recomposite
    snde_debug(SNDE_DC_RENDERING,"paintGL()");
    {
      std::lock_guard<std::mutex> adminlock(admin);
      need_recomposite=true;
    }
    GraphicsWindow->setDefaultFboId(defaultFramebufferObject()); // nobody should be messing with the graphicswindow but this thread 
    // execute up to one full rendering pass but don't allow waiting in the QT main thread main loop
    dispatch(true,false,false);
  }

  void qt_osg_compositor::resizeGL(int width, int height)
  {
    // ***!!!! BUG: compositor gets its size through resize_width and
    // resize_height after a proper resize operation here,
    // but display_requirements.cpp pulls from
    // display->drawareawidth and display->drawareaheight, which may be
    // different and aren't sync'd properly. We do a dumb
    // sync inside resize_compositor() below.
    
    //GraphicsWindow->getEventQueue()->windowResize(0,0,width,height);
    //GraphicsWindow->resized(0,0,width,height);
    //Camera->setViewport(0,0,width,height);
    display->set_pixelsperdiv(width,height);
    
    resize_compositor(width,height);
    
    trigger_rerender();
  }




  void qt_osg_compositor::mouseMoveEvent(QMouseEvent *event)
  {
    // translate Qt mouseMoveEvent to OpenSceneGraph
    snde_debug(SNDE_DC_EVENT,"Generating mousemotion");
    GraphicsWindow->getEventQueue()->mouseMotion(event->x(), event->y()); //,event->timestamp()/1000.0);
    
    // for some reason drags with the middle mouse button pressed
    // get the buttons field filtered out (?)
    
    // should we only update if a button is pressed??
    //fprintf(stderr,"buttons=%llx\n",(unsigned long long)event->buttons());
    // !!!*** NOTE:  "throwing" works if we make the trigger_rerender here unconditional
    if (event->buttons()) {
      trigger_rerender();
    }
  }
  
  void qt_osg_compositor::mousePressEvent(QMouseEvent *event)
  {
    int button;
    switch(event->button()) {
    case Qt::LeftButton:
      button=1;
      break;
      
    case Qt::MiddleButton:
      button=2;
      break;
      
    case Qt::RightButton:
      button=3;
      break;
      
    default:
      button=0;
      
      
    }

    snde_debug(SNDE_DC_EVENT,"Mouse press event (%d,%d,%d)",event->x(),event->y(),button);
    
    GraphicsWindow->getEventQueue()->mouseButtonPress(event->x(),event->y(),button); //,event->timestamp()/1000.0);
    
    trigger_rerender();
    
    // Can adapt QT events -> OSG events here
    // would do e.g.
    //GraphicsWindow->getEventQueue()->mouseButtonPress(event->x(),event->y(),button#)
    // Would also want to forward mouseButtonRelease() 
  }
  
  void qt_osg_compositor::mouseReleaseEvent(QMouseEvent *event)
  {
    int button;
    switch(event->button()) {
    case Qt::LeftButton:
      button=1;
      break;
      
    case Qt::MiddleButton:
      button=2;
      break;
      
    case Qt::RightButton:
      button=3;
      break;
	
    default:
      button=0;
      
      
    }
    
    snde_debug(SNDE_DC_EVENT,"Mouse release event (%d,%d,%d)",event->x(),event->y(),button);
    GraphicsWindow->getEventQueue()->mouseButtonRelease(event->x(),event->y(),button); //,event->timestamp()/1000.0);
    
    trigger_rerender();
    
      // Can adapt QT events -> OSG events here
      // would do e.g.
      //GraphicsWindow->getEventQueue()->mouseButtonPress(event->x(),event->y(),button#)
      // Would also want to forward mouseButtonRelease() 
  }
  
  void qt_osg_compositor::wheelEvent(QWheelEvent *event)
  {
    GraphicsWindow->getEventQueue()->mouseScroll( (event->angleDelta().y() > 0) ?
						  osgGA::GUIEventAdapter::SCROLL_UP :
						  osgGA::GUIEventAdapter::SCROLL_DOWN);
    //event->timestamp()/1000.0);
    trigger_rerender();
    
  }
  
  
  bool qt_osg_compositor::event(QEvent *event)
  {
    if (event->type()==QEvent::TouchBegin || event->type()==QEvent::TouchUpdate || event->type()==QEvent::TouchEnd) {
      QList<QTouchEvent::TouchPoint> TouchPoints = static_cast<QTouchEvent *>(event)->touchPoints();
      
      //double timestamp=static_cast<QInputEvent *>(event)->timestamp()/1000.0;
      
      for (auto & TouchPoint: TouchPoints) {
	
	if (TouchPoint.state()==Qt::TouchPointPressed) {
	  GraphicsWindow->getEventQueue()->touchBegan(TouchPoint.id(),osgGA::GUIEventAdapter::TOUCH_BEGAN,TouchPoint.pos().x(),TouchPoint.pos().y(),1); //,timestamp);
	} else if (TouchPoint.state()==Qt::TouchPointMoved) {
	  GraphicsWindow->getEventQueue()->touchMoved(TouchPoint.id(),osgGA::GUIEventAdapter::TOUCH_MOVED,TouchPoint.pos().x(),TouchPoint.pos().y(),1); //,timestamp);
	  
	} else if (TouchPoint.state()==Qt::TouchPointStationary) {
	  GraphicsWindow->getEventQueue()->touchMoved(TouchPoint.id(),osgGA::GUIEventAdapter::TOUCH_STATIONERY,TouchPoint.pos().x(),TouchPoint.pos().y(),1); //,timestamp);
	  
	} else if (TouchPoint.state()==Qt::TouchPointReleased) {
	  GraphicsWindow->getEventQueue()->touchEnded(TouchPoint.id(),osgGA::GUIEventAdapter::TOUCH_ENDED,TouchPoint.pos().x(),TouchPoint.pos().y(),1); //,timestamp);
	  
	}
      }
      trigger_rerender();
      return true;
    } else {
      
      return QOpenGLWidget::event(event);
    }
  }
  
  
  /*
  void qt_osg_compositor::ClearPickedOrientation()
  {
    // notification from picker to clear any marked orientation
    if (QTViewer->GeomRenderer) {
      QTViewer->GeomRenderer->ClearPickedOrientation();
    }
  }
  */

  void qt_osg_compositor::rerender()
  // QT slot indicating that rerendering is needed
  {
    snde_debug(SNDE_DC_RENDERING,"qt_osg_compositor: Got rerender");
    trigger_rerender();

    if (!threaded) {
      emit update(); // in non-threaded mode we have to go into paintGL() to initiate the update (otherwise sub-thread will take care of it for us)
    }
  }
  
  void qt_osg_compositor::update()
  // QT slot indicating that we should do a display update, i.e. a re-composite
  {
    snde_debug(SNDE_DC_RENDERING,"qt_osg_compositor::update()");
    QOpenGLWidget::update(); // re-composite done inside paintGL();
  }

  
}
