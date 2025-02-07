#include <QMouseEvent>
#include <QWheelEvent>
#include <QTouchEvent>
#include <QGuiApplication>

#include "snde/qt_osg_compositor.hpp"
#include "snde/openscenegraph_renderer.hpp"
#include "snde/colormap.h"
#include "snde/rec_display.hpp"
#include "snde/display_requirements.hpp"

#include "snde/python_support.hpp"

namespace snde {

  qt_osg_worker_thread::qt_osg_worker_thread(qt_osg_compositor *comp,QObject *parent) :
    QThread(parent),
    comp(comp)
  {

  }
  
  void qt_osg_worker_thread::run()
  {
    {
	SNDE_BeginDropPythonGILBlock
    set_thread_name(nullptr,"snde2 qt_osg worker");

    comp->worker_code();
    SNDE_EndDropPythonGILBlock
	}
  }

  void qt_osg_worker_thread::emit_need_update()
  {
    {
	SNDE_BeginDropPythonGILBlock
    emit compositor_need_update();
    SNDE_EndDropPythonGILBlock
	}
  }


  
  static bool confirm_threaded_opengl(bool enable_threaded_opengl)
  {
    {
	SNDE_BeginDropPythonGILBlock
    bool platform_support = QOpenGLContext::supportsThreadedOpenGL();

    if (enable_threaded_opengl && !platform_support) {
      snde_warning("qt_osg_compositor: Threaded OpenGL disabled because of a lack of platform support");
      return false;
    }

    return enable_threaded_opengl;
    SNDE_EndDropPythonGILBlock
	}
  }
  
  qt_osg_compositor::qt_osg_compositor(std::shared_ptr<recdatabase> recdb,
				       std::shared_ptr<display_info> display,
				       osg::ref_ptr<osgViewer::Viewer> Viewer,
				       bool threaded,bool enable_threaded_opengl,
				       bool enable_shaders,
				       QPointer<QTRecViewer> Parent_QTRViewer,QWidget *parent/*=nullptr*/) :
    QOpenGLWidget(parent),
    osg_compositor(recdb,display,Viewer,new osg_ParanoidGraphicsWindowEmbedded(0,0,width(),height()),
		   threaded,confirm_threaded_opengl(enable_threaded_opengl),
		   enable_shaders,defaultFramebufferObject() /* probably 0 as QT OGL hasn't started... we'll assign an updated value in initializeGL() and paintGL() below */ ),
    RenderContext(nullptr),
    qt_worker_thread(nullptr),
    Parent_QTRViewer(QPointer<QTRecViewer>(Parent_QTRViewer))
  {
    {
	SNDE_BeginDropPythonGILBlock
    Viewer->setThreadingModel(osgViewer::Viewer::SingleThreaded);
    Viewer->getCamera()->setViewport(new osg::Viewport(0,0,width(),height()));
    Viewer->getCamera()->setGraphicsContext(GraphicsWindow);
    
    //AnimTimer = new QTimer(this);
    //AnimTimer->setInterval(16); // 16 ms ~ 60 Hz

    setMouseTracking(true); // ???
    setAttribute(Qt::WA_AcceptTouchEvents,true);

    //QObject::connect(AnimTimer,SIGNAL(timeout()),this,SLOT(update()));
    SNDE_EndDropPythonGILBlock
	}
  }

  qt_osg_compositor::~qt_osg_compositor()
  {
    {
	SNDE_BeginDropPythonGILBlock
    // call stop before any other destruction happens so that our objects are still valid
    stop();

    // our superclass will call stop() again but it won't matter because the above call
    // will have already dealt with everything. 
    SNDE_EndDropPythonGILBlock
	}
  }

  void qt_osg_compositor::trigger_rerender()
  {
    {
	SNDE_BeginDropPythonGILBlock
    // perform OSG event traversal prior to rendering so as to be able to process
    // mouse events, etc. BEFORE compositing
    snde_debug(SNDE_DC_RENDERING,"trigger_rerender()");
    Viewer->eventTraversal();
    
    osg_compositor::trigger_rerender();

    if (!threaded) {
      // if not threaded, we need a paint callback
      update();
    }
    SNDE_EndDropPythonGILBlock
	}
  }

  void qt_osg_compositor::initializeGL()
  {
    {
	SNDE_BeginDropPythonGILBlock
    // called once our context is created by QT and after any
    // reparenting (which would trigger a new context)

    // tell our graphics window that OpenGL has been initialized. 
    (dynamic_cast<osg_ParanoidGraphicsWindowEmbedded *>(GraphicsWindow.get()))->gl_is_available();
    if (threaded && enable_threaded_opengl) {
      RenderContext = new QOpenGLContext();
      RenderContext->setShareContext(context());
      QScreen* pScreen = QGuiApplication::screenAt(mapToGlobal(QPoint{ width() / 2,0 }));
      RenderContext->setScreen(pScreen);
      DummyOffscreenSurface = new QOffscreenSurface(pScreen);
      DummyOffscreenSurface->setFormat(context()->format());
      DummyOffscreenSurface->create();
      
      RenderContext->create();
      
      LayerDefaultFramebufferObject = RenderContext->defaultFramebufferObject();
    } else {
      LayerDefaultFramebufferObject = context()->defaultFramebufferObject(); // doesn't seem to be valid yet, but we reassign every call to paintGL() as well
    }
    
    start(); // make sure threads are going

    // This next code moved into _start_worker_thread() so it can happen
    // guaranteed before the thread tries to access RenderContext
    //if (threaded && enable_threaded_opengl) {
    //  assert(qt_worker_thread);
    //  RenderContext->moveToThread(qt_worker_thread);
    //  DummyOffscreenSurface->moveToThread(qt_worker_thread);
    //}
    SNDE_EndDropPythonGILBlock
	}
  }


  void qt_osg_compositor::worker_code()
  {
    {
	SNDE_BeginDropPythonGILBlock
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
    SNDE_EndDropPythonGILBlock
	}
  }

  
  void qt_osg_compositor::_start_worker_thread(std::unique_lock<std::mutex> *adminlock)
  {
    {
	SNDE_BeginDropPythonGILBlock
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
    SNDE_EndDropPythonGILBlock
	}
  }

  void qt_osg_compositor::_join_worker_thread()
  {
    {
	SNDE_BeginDropPythonGILBlock
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
    SNDE_EndDropPythonGILBlock
	}

  }

  void qt_osg_compositor::perform_ondemand_calcs(std::unique_lock<std::mutex> *adminlock)
  {
    {
	SNDE_BeginDropPythonGILBlock
    // wrap osg_compositor::perform_ondemand_calcs so that after layer
    // ondemand calcs are done we will rerender.

    // Only needed if threaded is enabled but threaded_opengl is not enabled because in that
    // circumstance we are in a different thread and need to trigger the main thread
    // to get a paint callback to do the rendering.
    
    osg_compositor::perform_ondemand_calcs(adminlock);

    //if (threaded && !enable_threaded_opengl) {
    //  qt_worker_thread->emit_need_update();
    //}
    SNDE_EndDropPythonGILBlock
	}
  }

  void qt_osg_compositor::perform_layer_rendering(std::unique_lock<std::mutex> *adminlock)
  {
    {
	SNDE_BeginDropPythonGILBlock
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
    SNDE_EndDropPythonGILBlock
	}
  }


  void qt_osg_compositor::perform_compositing(std::unique_lock<std::mutex> *adminlock)
  {
    {
	SNDE_BeginDropPythonGILBlock
    
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
    SNDE_EndDropPythonGILBlock
	}
  }


  void qt_osg_compositor::wake_up_ondemand_locked(std::unique_lock<std::mutex> *adminlock)
  {
    {
	SNDE_BeginDropPythonGILBlock
    if (threaded) {
      execution_notify.notify_all();
    }
    SNDE_EndDropPythonGILBlock
	}
  }
  
  void qt_osg_compositor::wake_up_renderer_locked(std::unique_lock<std::mutex> *adminlock)
  {
    {
	SNDE_BeginDropPythonGILBlock
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
    SNDE_EndDropPythonGILBlock
	}

  }

  void qt_osg_compositor::wake_up_compositor_locked(std::unique_lock<std::mutex> *adminlock)
  {
    {
	SNDE_BeginDropPythonGILBlock
    adminlock->unlock();

    // Need GUI update
    if (std::this_thread::get_id() == *worker_thread_id && qt_worker_thread) {
      qt_worker_thread->emit_need_update();
    } else {
      //emit update(); // (commented out because if we are the GUI thread then we are already awake!)
    }
    adminlock->lock();
    SNDE_EndDropPythonGILBlock
	}
  }

  void qt_osg_compositor::clean_up_renderer_locked(std::unique_lock<std::mutex> *adminlock)
  {
    {
	SNDE_BeginDropPythonGILBlock
    osg_compositor::clean_up_renderer_locked(adminlock);

    if (threaded && enable_threaded_opengl) {
      delete RenderContext; // OK because it's not owned by another QObject
      delete DummyOffscreenSurface; // OK because it's not owned by another QObject
    }
    SNDE_EndDropPythonGILBlock
	}
  }

  
  void qt_osg_compositor::paintGL()
  {
    {
	SNDE_BeginDropPythonGILBlock
    // mark that at minimum we need a recomposite
    snde_debug(SNDE_DC_RENDERING,"paintGL()");
    {
      std::lock_guard<std::mutex> adminlock(admin);
      need_recomposite=true;
    }
    GraphicsWindow->setDefaultFboId(defaultFramebufferObject()); // nobody should be messing with the graphicswindow but this thread
    if (threaded && enable_threaded_opengl) {
      LayerDefaultFramebufferObject = RenderContext->defaultFramebufferObject();
    } else {
      LayerDefaultFramebufferObject = defaultFramebufferObject();
    }
    // execute up to one full rendering pass but don't allow waiting in the QT main thread main loop
    dispatch(true,false,false);
    snde_debug(SNDE_DC_RENDERING,"paintGL() returning.");
    SNDE_EndDropPythonGILBlock
	}
  }

  void qt_osg_compositor::resizeGL(int width, int height)
  {
    {
	SNDE_BeginDropPythonGILBlock
    // ***!!!! BUG: compositor gets its size through resize_width and
    // resize_height after a proper resize operation here,
    // but display_requirements.cpp pulls from
    // display->drawareawidth and display->drawareaheight, which may be
    // different and aren't sync'd properly. We do a dumb
    // sync inside resize_compositor() below.
    
    //GraphicsWindow->getEventQueue()->windowResize(0,0,width,height);
    //GraphicsWindow->resized(0,0,width,height);
    //Camera->setViewport(0,0,width,height);
    display->set_pixelsperdiv(width*devicePixelRatio(),height*devicePixelRatio());
    
    resize_compositor(width*devicePixelRatio(),height*devicePixelRatio());
    
    trigger_rerender();
    SNDE_EndDropPythonGILBlock
	}
  }




  void qt_osg_compositor::mouseMoveEvent(QMouseEvent *event)
  {
    {
	SNDE_BeginDropPythonGILBlock
    // translate Qt mouseMoveEvent to OpenSceneGraph
    snde_debug(SNDE_DC_EVENT,"Generating mousemotion");
    GraphicsWindow->getEventQueue()->mouseMotion(event->x()*devicePixelRatio(), event->y()*devicePixelRatio()); //,event->timestamp()/1000.0);
    
    // for some reason drags with the middle mouse button pressed
    // get the buttons field filtered out (?)
    
    // should we only update if a button is pressed??
    //fprintf(stderr,"buttons=%llx\n",(unsigned long long)event->buttons());
    // !!!*** NOTE:  "throwing" works if we make the trigger_rerender here unconditional
    if (event->buttons()) {
      trigger_rerender();
    }
    SNDE_EndDropPythonGILBlock
	}
  }
  
  void qt_osg_compositor::mousePressEvent(QMouseEvent *event)
  {
    {
	SNDE_BeginDropPythonGILBlock
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

    snde_debug(SNDE_DC_EVENT,"Mouse press event (%d,%d,%d)",event->x()*devicePixelRatio(),event->y()*devicePixelRatio(),button);
    
    GraphicsWindow->getEventQueue()->mouseButtonPress(event->x()*devicePixelRatio(),event->y()*devicePixelRatio(),button); //,event->timestamp()/1000.0);
    
    trigger_rerender();
    
    // Can adapt QT events -> OSG events here
    // would do e.g.
    //GraphicsWindow->getEventQueue()->mouseButtonPress(event->x(),event->y(),button#)
    // Would also want to forward mouseButtonRelease() 
    SNDE_EndDropPythonGILBlock
	}
  }
  
  void qt_osg_compositor::mouseReleaseEvent(QMouseEvent *event)
  {
    {
	SNDE_BeginDropPythonGILBlock
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
    
    snde_debug(SNDE_DC_EVENT,"Mouse release event (%d,%d,%d)",event->x()*devicePixelRatio(),event->y()*devicePixelRatio(),button);
    GraphicsWindow->getEventQueue()->mouseButtonRelease(event->x()*devicePixelRatio(),event->y()*devicePixelRatio(),button); //,event->timestamp()/1000.0);
    
    trigger_rerender();
    
      // Can adapt QT events -> OSG events here
      // would do e.g.
      //GraphicsWindow->getEventQueue()->mouseButtonPress(event->x(),event->y(),button#)
      // Would also want to forward mouseButtonRelease() 
    SNDE_EndDropPythonGILBlock
	}
  }
  
  void qt_osg_compositor::wheelEvent(QWheelEvent *event)
  {
    {
	SNDE_BeginDropPythonGILBlock
    GraphicsWindow->getEventQueue()->mouseScroll( (event->angleDelta().y() > 0) ?
						  osgGA::GUIEventAdapter::SCROLL_UP :
						  osgGA::GUIEventAdapter::SCROLL_DOWN);
    //event->timestamp()/1000.0);
    trigger_rerender();
    SNDE_EndDropPythonGILBlock
	}
    
  }
  
  
  bool qt_osg_compositor::event(QEvent *event)
  {
    {
	SNDE_BeginDropPythonGILBlock
    if (event->type()==QEvent::TouchBegin || event->type()==QEvent::TouchUpdate || event->type()==QEvent::TouchEnd) {
      QList<QTouchEvent::TouchPoint> TouchPoints = static_cast<QTouchEvent *>(event)->touchPoints();
      
      //double timestamp=static_cast<QInputEvent *>(event)->timestamp()/1000.0;
      
      for (auto & TouchPoint: TouchPoints) {
	
	if (TouchPoint.state()==Qt::TouchPointPressed) {
	  GraphicsWindow->getEventQueue()->touchBegan(TouchPoint.id(),osgGA::GUIEventAdapter::TOUCH_BEGAN,TouchPoint.pos().x()*devicePixelRatio(),TouchPoint.pos().y()*devicePixelRatio(),1); //,timestamp);
	} else if (TouchPoint.state()==Qt::TouchPointMoved) {
	  GraphicsWindow->getEventQueue()->touchMoved(TouchPoint.id(),osgGA::GUIEventAdapter::TOUCH_MOVED,TouchPoint.pos().x()*devicePixelRatio(),TouchPoint.pos().y()*devicePixelRatio(),1); //,timestamp);
	  
	} else if (TouchPoint.state()==Qt::TouchPointStationary) {
	  GraphicsWindow->getEventQueue()->touchMoved(TouchPoint.id(),osgGA::GUIEventAdapter::TOUCH_STATIONERY,TouchPoint.pos().x()*devicePixelRatio(),TouchPoint.pos().y()*devicePixelRatio(),1); //,timestamp);
	  
	} else if (TouchPoint.state()==Qt::TouchPointReleased) {
	  GraphicsWindow->getEventQueue()->touchEnded(TouchPoint.id(),osgGA::GUIEventAdapter::TOUCH_ENDED,TouchPoint.pos().x()*devicePixelRatio(),TouchPoint.pos().y()*devicePixelRatio(),1); //,timestamp);
	  
	}
      }
      trigger_rerender();
      return true;
    } else {
      
      return QOpenGLWidget::event(event);
    }
    SNDE_EndDropPythonGILBlock
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
    {
	SNDE_BeginDropPythonGILBlock
    snde_debug(SNDE_DC_RENDERING,"qt_osg_compositor: Got rerender");
    trigger_rerender();

    if (!threaded) {
      emit update(); // in non-threaded mode we have to go into paintGL() to initiate the update (otherwise sub-thread will take care of it for us)
    }
    SNDE_EndDropPythonGILBlock
	}
  }
  
  void qt_osg_compositor::update()
  // QT slot indicating that we should do a display update, i.e. a re-composite
  {
    {
	SNDE_BeginDropPythonGILBlock
    snde_debug(SNDE_DC_RENDERING,"qt_osg_compositor::update()");
    QOpenGLWidget::update(); // re-composite done inside paintGL();
    SNDE_EndDropPythonGILBlock
	}
  }




  // Register the pre-existing tracking_pose_recording_display_handler in display_requirement.cpp/hpp as the display handler for osg_compositor_view_tracking_pose_recording
  static int register_qtocvtpr_display_handler = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING,typeid(qt_osg_compositor_view_tracking_pose_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
	return std::make_shared<tracking_pose_recording_display_handler>(display,displaychan,base_rss);
      }));
  
  
  qt_osg_compositor_view_tracking_pose_recording::qt_osg_compositor_view_tracking_pose_recording(struct recording_params params,size_t info_structsize,std::string channel_to_reorient, std::string component_name,QSharedPointer<qt_osg_compositor> compositor) :
    tracking_pose_recording(params,info_structsize,channel_to_reorient,component_name),
    compositor(compositor)
  {
    rec_classes.push_back(recording_class_info("snde::qt_osg_compositor_view_tracking_pose_recording",typeid(qt_osg_compositor_view_tracking_pose_recording),ptr_to_new_shared_impl<qt_osg_compositor_view_tracking_pose_recording>));


  }
  
  snde_orientation3 qt_osg_compositor_view_tracking_pose_recording::get_channel_to_reorient_pose(std::shared_ptr<recording_set_state> rss) const
  // NOTE: This function can be called from other threads (in the display_requirement evaluation, for example)
  // and it DOES access a QWidget class (qt_osg_compositor). This is safe because the QWidget is protected
  // by QSharedPointer with deleteLater() as its deleter and we are only calling a custom method that
  // is thread-safe. 
  {
    {
	SNDE_BeginDropPythonGILBlock
    snde_orientation3 retval;

    snde_invalid_orientation3(&retval);
    QSharedPointer<qt_osg_compositor> compositor_strong = compositor.toStrongRef();
    
    
    if (compositor_strong.isNull()) {
      return retval;
    }

    std::string chanpath(info->name);

    std::string channel_to_reorient_fullpath = recdb_path_join(chanpath,channel_to_reorient);

    snde_orientation3 channel_to_reorient_campose = compositor_strong->get_camera_pose(channel_to_reorient_fullpath);
    // channel_to_reorient_campose represents the orientation of the camera
    // for the channel_to_reorient channel relative to the channel_to_reorient object.
    // i.e. you give it coordinates relative to the ctt camera and it gives
    // you coordinates relative to the ctt object.
    // i.e. it has units (ctt channel object coords)/(ctt channel camera coords)

    
    snde_orientation3 follower_channel_campose = compositor_strong->get_camera_pose(chanpath);
    // follower_channel_campose represents the orientation of the camera
    // in our follower channel relative to follower channel coordinates.
    // i.e. you give it coordinates relative to the follower channel camera and it gives
    // you coordinates relative to the follower channel frame, which is the same as the component frame.
    // i.e. it has units (follower channel object coords)/(follower channel camera coords)

    // Consider the following scenario:
    //  * We have a "world" reference coordinate frame around our component channel
    //  * retval represents "world" reference coordinates over channel-to-track object coordinates
    //  * i.e. represents (world coordinates)/(ctt object coords)
    //  * Our channel-to-track channel is thought of as rotated into some position in the "world"
    //    but we just get camera position relative to the channel to track object.
    //    Thus channel_to_reorient_campose has units (ctt object coords)/(ctt channel camera coords)
    //  * follower_channel_campose has units (world coords)/(follower channel camera coords)
    //  * Our constraint is that the ctt object should appear in the same position relative to the
    //    follower camera as it appears relative to the channel to track camera. Since
    //    we just gave the ctt object a unique orientation, now the ctt camera and follower camera
    //    also have to have the same orientation. So treat follower channel camera coords as just "camera coords"
    //
    //  follower_channel_campose = (world coords/camera coords)
    //  retval = (world coords)/(ctt object coords)
    //  channel_to_reorient_campose = (ctt object coords)/((ctt channel)? camera coords)
    //
    //  retval = follower_channel_campos / channel_to_reorient_campose  = world coords/ctt object coords


    // retval = follower_channel_campose * inv(channel_to_reorient_campose)
    if (0) {
      // This code builds retval by converting
      // our orientations into 4x4 matrices 
      snde_coord4 follower_channel_campose_rotmtx[4]; // index identifies which column (data stored column-major)
      orientation_build_rotmtx(follower_channel_campose,follower_channel_campose_rotmtx);
      
      snde_coord4 channel_to_reorient_campose_rotmtx[4]; // index identifies which column (data stored column-major)
      orientation_build_rotmtx(channel_to_reorient_campose,channel_to_reorient_campose_rotmtx);
      
      
      
      
      osg::Matrixd retval_osgmtx;
      if (0) {
	
	// Build OpenSceneGraph matrices and use OSG
	// to calculate follower_channel_campose * inv(channel_to_reorient_campose)
	
	// ...following backwards OSG operator ordering convention:
	// retval = inv(channel_to_reorient_campose) * follower_channel_campose
	
	osg::Matrixd follower_channel_campose_osgmtx(&follower_channel_campose_rotmtx[0].coord[0]);
	osg::Matrixd channel_to_reorient_campose_osgmtx(&channel_to_reorient_campose_rotmtx[0].coord[0]);
	
	osg::Matrixd channel_to_reorient_campose_inverse_osgmtx=osg::Matrixd::inverse(channel_to_reorient_campose_osgmtx);
	
	
	osg::Matrixd osg_product(follower_channel_campose_osgmtx);
	osg_product.preMult(channel_to_reorient_campose_inverse_osgmtx);
	retval_osgmtx = osg_product;
      }

      // calculate follower_channel_campose * inv(channel_to_reorient_campose)
      // invert channel_to_reorient_campose
      snde_coord4 channel_to_reorient_campose_rotmtx_inverse[4] = { { 1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}}; // when we solve for this in Ax=b it will turn into the inverse matrix
      size_t pivots[4];
      // note: fmatrixsolve destroys channel_to_reorient_campose_rotmtx
      fmatrixsolve(&channel_to_reorient_campose_rotmtx[0].coord[0],&channel_to_reorient_campose_rotmtx_inverse[0].coord[0],4,4,pivots,false);
      
      // multcmat44 is row major so like OSG we have to do it backwards
      snde_coord4 retval_cmat[4];
      multcmat44(&channel_to_reorient_campose_rotmtx_inverse[0].coord[0],&follower_channel_campose_rotmtx[0].coord[0],&retval_cmat[0].coord[0]);
      retval_osgmtx = osg::Matrixd(&retval_cmat[0].coord[0]);
      
      
      osg::Vec3d translation;
      osg::Quat rotation;
      osg::Vec3d scale;
      osg::Quat scale_orientation;

      retval_osgmtx.decompose(translation,rotation,scale,scale_orientation);
      
      //snde_orientation3 retval;
      retval.offset.coord[0]=translation.x();
      retval.offset.coord[1]=translation.y();
      retval.offset.coord[2]=translation.z();
      retval.offset.coord[3]=1.0;
      
      retval.quat.coord[1]=rotation.x();
      retval.quat.coord[2]=rotation.y();
      retval.quat.coord[3]=rotation.z();
      retval.quat.coord[0]=rotation.w();
    }

    // retval = follower_channel_campose * inv(channel_to_reorient_campose)
    snde_orientation3 channel_to_reorient_campose_inverse;
    orientation_inverse(channel_to_reorient_campose,&channel_to_reorient_campose_inverse);

    orientation_orientation_multiply(follower_channel_campose,channel_to_reorient_campose_inverse,&retval);
    
    
    return retval; 

    // ***!!!! comments below are obsolete !!!***
    // The orientation we return will be used as the
    // (follower channel ctt object coords)/(ctt channel object coords)
    //
    // and we want (follower channel ctt object coords)/(follower channel camera coords)
    // to match (ctt channel object coords)/(ctt channel camera coords), i.e.
    // the ctt object appears in the same orientation in the follower channel
    //

    // This gives us an equation: (follower channel ctt object coords)/(follower channel camera coords) = (ctt channel object coords)/(ctt channel camera coords) = channel_to_reorient_campose
    //
    // and we have follower_channel_campose = (follower channel object coords)/(follower channel camera coords)
    // and we have retval = (follower channel ctt object coords)/(ctt channel object coords)
    // So (follower channel ctt object coords) = retval * (ctt channel object coords)
    //
    // From the eq. (follower channel ctt object coords)/(follower channel camera coords) = channel_to_reorient_campose
    //
    // so  retval * (ctt channel object coords)/(follower channel camera coords) = channel_to_reorient_campose
    
    // from above, follower_channel_campose = (follower channel object coords)/(follower channel camera coords)
    // therefore  (follower channel camera coords) = (follower channel object coords)/follower_channel_campose  


    // therefore retval * (ctt channel object coords)*follower_channel_campose/(follower channel object coords) = channel_to_reorient_campose
    // Therefore retval = channel_to_reorient_campose/follower_channel_campose * (follower channel object coords)/(ctt channel object coords)?
    
    // Therefore retval = channel_to_reorient_campose * follower_channel_campose * (follower channel object coords)/(ctt channel object coords)?

    // channel_to_reorient_campose / follower_channel_campose = (ctt channel object coords)/(ctt channel camera coords) * (follower channel camera coords) / (follower channel object coords) = (ctt channel object coords)/(follower channel object coords) * (follower channel camera coords)/(ctt channel camera coords)
    SNDE_EndDropPythonGILBlock
	}

  }


  
}
