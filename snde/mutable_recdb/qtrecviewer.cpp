#include <osgDB/WriteFile>

#include "snde/qtrecviewer.hpp"

namespace snde {
  /*
  QTRecGraphicsWindow::QTRecGraphicsWindow(int x,int y,int width,int height,QTRecRender *Renderer) :
      osgViewer::GraphicsWindowEmbedded(x,y,width,height),Renderer(Renderer)
    {
      
    }

  void QTRecGraphicsWindow::requestRedraw()
  {
    Renderer->update();
  }

  void QTRecGraphicsWindow::requestContinuousUpdate(bool needed)
  {
    if (needed && Renderer->AnimTimer) {
      if (!Renderer->AnimTimer->isActive()) {
	Renderer->AnimTimer->start();
	fprintf(stderr,"Starting animation timer!\n");
      }
    } else {
      fprintf(stderr,"Manipulator not animating\n");
      if (Renderer->AnimTimer && Renderer->AnimTimer->isActive()) {
	Renderer->AnimTimer->stop();
      }
      
    }
    
  }
  */
  
  QTRecRender::QTRecRender(osg::ref_ptr<osg::Node> RootNode, QTRecViewer *QTViewer,QWidget *parent)
      : QOpenGLWidget(parent),
	osg_renderer(new osgViewer::GraphicsWindowEmbedded(x(),y(),width(),height()),
		     RootNode,
		     false),
	picker(new osg_picker(this,QTViewer->display)),
	QTViewer(QTViewer)
    {
      AnimTimer = new QTimer(this);
      AnimTimer->setInterval(16); // 16 ms ~ 60 Hz
      
      Camera->setViewport(0,0,width(),height());

      SetProjectionMatrix();

      
      setMouseTracking(true); // ???

      setAttribute(Qt::WA_AcceptTouchEvents,true);
      //Viewer->addEventHandler(picker); // adding now handled by osg_picker constructor...
 
      Viewer->realize();
      QObject::connect(AnimTimer,SIGNAL(timeout()),this,SLOT(update()));
    }
  
  /* virtual */ void QTRecRender::ClearPickedOrientation() // in qtrecviewer.cpp
  {
    // notification from picker to clear any marked orientation
    if (QTViewer->GeomRenderer) {
      QTViewer->GeomRenderer->ClearPickedOrientation();
    }
  }
  
  /* virtual */ void QTRecRender::paintGL()
  {
    //fprintf(stderr,"paintGL()\n");
    
    if (Viewer.valid()) {
      /* Because our data is marked as DYNAMIC, so long as we have it 
	 locked during viewer->frame() we should be OK */
      
      //std::lock_guard<std::mutex> object_trees_lock(geom->object_trees_lock);
      
      //std::shared_ptr<lockholder> holder=std::make_shared<lockholder>();
      //std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(geom->manager->locker); // new locking process
      //OSGComp->LockVertexArraysTextures(holder,lockprocess);
      //rwlock_token_set all_locks=lockprocess->finish();
      
      
 
      QTViewer->update_rec_list();
      QTViewer->rendering_revman->Start_Transaction();
      QTViewer->update_renderer(); // this eats the events...
      /* !!!*** Need to be able to auto-lock all inputs as 
	 we start End_Transaction to ensure a consistent state at the end */
      /* IDEA: lock acquisition is already designed to be able to accommodate 
	 locks already held. Can use same infrastructure. Just need to forbid 
	 explicit unlocking */ 
      snde_index revnum = QTViewer->rendering_revman->End_Transaction();
      /* !!!*** Need some way to auto-lock output at the end of 
         the transaction to ensure data consistency while rendering */ 
      /* ... perhaps based on output from update_renderer() ?  */
      QTViewer->rendering_revman->Wait_Computation(revnum);

      //osgDB::writeNodeFile(*RootNode,"/tmp/qtrecviewer.osg");
      
      assert(!Camera->getViewMatrix().isNaN());
      fprintf(stderr,"Render! empty=%d\n",(int)GraphicsWindow->getEventQueue()->empty());
      if (!GraphicsWindow->getEventQueue()->empty()) {
	fprintf(stderr,"About to process events\n");
      }

      {
	std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(QTViewer->sndegeom->manager->locker); // new locking process
	//OSGComp->LockVertexArraysTextures(holder,lockprocess);
	QTViewer->lock_renderer(lockprocess);
	rwlock_token_set all_locks=lockprocess->finish();
	
	Viewer->frame();
      }

      if (Viewer->compat34GetRequestContinousUpdate()) {//(Viewer->getRequestContinousUpdate()) { // Manipulator->isAnimating doesn't work for some reason(?)
      // ideally we should implement for OpenSceneGraph the GUIActionAdapter
      // class, passing that somehow to the osg View, and handling the
      // requestContinousUpdate() call instead of checking manipulator
      // animation directly.. 
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

      fprintf(stderr,"Render complete; empty=%d\n",(int)GraphicsWindow->getEventQueue()->empty());
      
      //unlock_rwlock_token_set(all_locks); // Drop our locks 
      
      //QOpenGLWidget::paintGL();  // necessary? 
    }
  }
    
  /* virtual */ void QTRecRender::resizeGL(int width,int height)
  {
    GraphicsWindow->getEventQueue()->windowResize(x(),y(),width,height);
    GraphicsWindow->resized(x(),y(),width,height);
    Camera->setViewport(0,0,width,height);
    SetProjectionMatrix();
    QTViewer->update();
    //QTViewer->
  }


  
  qtrec_position_manager::~qtrec_position_manager()
  {
    
  }


  bool QTRecSelector::eventFilter(QObject *object,QEvent *event)
  {
    if (event->type()==QEvent::FocusIn) {
      //fprintf(stderr,"FocusIn\n");

      if (object==RadioButton) {
	Viewer->set_selected(this);
      }
    }
    if (event->type()==QEvent::KeyRelease || event->type()==QEvent::KeyPress) {
      QKeyEvent *key = static_cast<QKeyEvent *>(event);
	switch(key->key()) {
	case Qt::Key_Left:
	  if (event->type()==QEvent::KeyPress) {
	    Viewer->posmgr->HorizZoomOut(false);
	  }
	  return true;

	case Qt::Key_Right:
	  if (event->type()==QEvent::KeyPress) {
	    //fprintf(stderr,"Press right\n");
	    Viewer->posmgr->HorizZoomIn(false);
	  }
	  // else fprintf(stderr,"Release right\n");
	  return true;
	  
	  
	case Qt::Key_Down:
	  if (event->type()==QEvent::KeyPress) {
	    if (Viewer->posmgr->selected_channel) {
	      std::shared_ptr<mutableinfostore> chan_data;
	      chan_data = Viewer->recdb->lookup(Viewer->posmgr->selected_channel->FullName());

	      std::shared_ptr<mutabledatastore> datastore=std::dynamic_pointer_cast<mutabledatastore>(chan_data);
	      if (datastore) {
		if (datastore->dimlen.size() > 1) {
		  // image data... decrease contrast instead
		  Viewer->LessContrast(false);
		  return true;
		}
	      }
	    }
	    Viewer->posmgr->VertZoomOut(false);
	  }
	  return true;

	case Qt::Key_Up:
	  if (event->type()==QEvent::KeyPress) {

	    if (Viewer->posmgr->selected_channel) {
	      std::shared_ptr<mutableinfostore> chan_data;
	      chan_data = Viewer->recdb->lookup(Viewer->posmgr->selected_channel->FullName());

	      std::shared_ptr<mutabledatastore> datastore=std::dynamic_pointer_cast<mutabledatastore>(chan_data);
	      if (datastore) {
		if (datastore->dimlen.size() > 1) {
		  // image data... increase contrast instead
		  Viewer->MoreContrast(false);
		  return true;
		}
	      }
	    }
	    Viewer->posmgr->VertZoomIn(false);
	  }
	  return true;

	case Qt::Key_Home:
	  if (event->type()==QEvent::KeyPress) {
	    Viewer->posmgr->HorizSliderActionTriggered(QAbstractSlider::SliderSingleStepAdd);
	  }
	  return true;
	  
	case Qt::Key_End:
	  if (event->type()==QEvent::KeyPress) {
	    Viewer->posmgr->HorizSliderActionTriggered(QAbstractSlider::SliderSingleStepSub);
	  }
	  return true;

	case Qt::Key_PageUp:
	  if (event->type()==QEvent::KeyPress) {
	    Viewer->posmgr->VertSliderActionTriggered(QAbstractSlider::SliderSingleStepAdd);
	  }
	  return true;

	case Qt::Key_PageDown:
	  if (event->type()==QEvent::KeyPress) {
	    Viewer->posmgr->VertSliderActionTriggered(QAbstractSlider::SliderSingleStepSub);
	  }
	  return true;

	case Qt::Key_Insert:
	  if (event->type()==QEvent::KeyPress) {
	    Viewer->Brighten(false);
	  }
	  return true;

	case Qt::Key_Delete:
	  if (event->type()==QEvent::KeyPress) {
	    Viewer->Darken(false);
	  }
	  return true;

	default:
	  return QFrame::eventFilter(object,event);
	  
	}
    }

    
    return QFrame::eventFilter(object,event);
  }
  
  
}
