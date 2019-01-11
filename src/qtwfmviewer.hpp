#include <QString>
#include <QWidget>
#include <QOpenGLWidget>
#include <QMouseEvent>
#include <QFrame>
#include <QRadioButton>
#include <QLayout>
#include <QVBoxLayout>
#include <QtUiTools/QUiLoader>

#include <osg/Array>
#include <osgViewer/Viewer>
#include <osgGA/TrackballManipulator>

#include "wfm_display.hpp"
#include "colormap.h"

#include "openscenegraph_geom.hpp"
#include "openscenegraph_data.hpp"

#ifndef SNDE_QTWFMVIEWER_HPP
#define SNDE_QTWFMVIEWER_HPP

namespace snde {

  // See https://vicrucann.github.io/tutorials/
  // See https://gist.github.com/vicrucann/874ec3c0a7ba4a814bd84756447bc798 "OpenSceneGraph + QOpenGLWidget - minimal example"
  // and http://forum.openscenegraph.org/viewtopic.php?t=16549 "QOpenGLWidget in osgQt"
  // and http://forum.openscenegraph.org/viewtopic.php?t=15097 "OSG 3.2.1 and Qt5 Widget integration"

  class QTWfmViewer; // forward declaration
  
  class QTWfmRender : public QOpenGLWidget {
  public:
    osg::ref_ptr<osgViewer::GraphicsWindowEmbedded> GraphicsWindow;
    osg::ref_ptr<osgViewer::Viewer> Viewer;
    osg::ref_ptr<osg::Camera> Camera;
    osg::ref_ptr<osg::Node> RootNode;
    osg::ref_ptr<osgGA::TrackballManipulator> Manipulator;
    QTWfmViewer *QTViewer; 
    bool twodimensional;
    
    
    QTWfmRender(osg::ref_ptr<osg::Node> RootNode, QTWfmViewer *QTViewer,QWidget *parent=0)
      : QOpenGLWidget(parent),
	GraphicsWindow(new osgViewer::GraphicsWindowEmbedded(x(),y(),width(),height())),
	Viewer(new osgViewer::Viewer()),
	Camera(new osg::Camera()),
	QTViewer(QTViewer),
	RootNode(RootNode),
	twodimensional(false)
    {
      Camera->setViewport(0,0,width(),height());

      // set background color to blueish
      Camera->setClearColor(osg::Vec4(.1,.1,.3,1.0));
      SetProjectionMatrix();
      Camera->setGraphicsContext(GraphicsWindow);

      Viewer->setCamera(Camera);
      Manipulator = new osgGA::TrackballManipulator();
      Viewer->setThreadingModel(osgViewer::Viewer::SingleThreaded);
      Viewer->setSceneData(RootNode);
      
      setMouseTracking(true); // ???

      Viewer->realize();
    }

    
    
    void SetTwoDimensional(bool twod)
    {
      twodimensional=twod;
      SetProjectionMatrix();

      if (twod) {
	Viewer->setCameraManipulator(nullptr);
	Camera->setViewMatrix(osg::Matrixd::identity());
      } else {
	Viewer->setCameraManipulator(Manipulator);
	
      }
    }
    
    void SetProjectionMatrix(void)
    {
      if (twodimensional) {
	Camera->setProjectionMatrixAsOrtho(0,static_cast<double>(width()),0,static_cast<double>(height()),-10.0,1000.0);
	
      } else {
	Camera->setProjectionMatrixAsPerspective(30.0,static_cast<float>(this->width())/static_cast<float>(this->height()),-1.0,1000.0);
	
      }
    }
	
    void SetRootNode(osg::ref_ptr<osg::Node> RootNode)
    {
      this->RootNode=RootNode;
      if (RootNode) {
	Viewer->setSceneData(RootNode);
      }
    }
    
  protected:
    virtual void paintGL(); // rendeirng code in qtwfmviewer.cpp

    virtual void resizeGL(int width,int height); // code in qtwfmviewer.cpp

    virtual void initializeGL()
    {
      // any opengl initialization here...
      
    }

    virtual void mouseMoveEvent(QMouseEvent *event)
    {
      //getEventQueue()->mouseMotion(event->x(), event->y());
    }

    virtual void mousePressEvent(QMouseEvent *event)
    {
      switch(event->button()) {
      case Qt::LeftButton:

	break;
      case Qt::RightButton:

	break;
	  
      }
      // Can adapt QT events -> OSG events here
      // would do e.g.
      //GraphicsWindow->getEventQueue()->mouseButtonPress(event->x(),event->y(),button#)
      // Would also want to forward mouseButtonRelease() 
    }

    virtual bool event(QEvent *event)
    {
      return QOpenGLWidget::event(event);
    }
  };
  class QTWfmSelector : public QFrame {
  public:
    std::string Name;
    QRadioButton *RadioButton;
    bool touched_during_update; // has this QTWfmSelector been touched during the latest update pass
    QPalette basepalette;
    QPalette selectedpalette;
    bool selected;
    WfmColor wfmcolor; 
    
    QTWfmSelector(std::string Name,WfmColor wfmcolor,QWidget *parent=0) :
      QFrame(parent),
      RadioButton(new QRadioButton(QString::fromStdString(Name),this)),
      Name(Name),
      basepalette(palette()),
      touched_during_update(true),
      selected(false),
      wfmcolor(wfmcolor)
    {
      setFrameShadow(QFrame::Shadow::Raised);
      setFrameShape(QFrame::Shape::Box);
      setLineWidth(1);
      setMidLineWidth(2);
      RadioButton->setAutoExclusive(false);
      QLayout *Layout=new QVBoxLayout();
      setLayout(Layout);
      Layout->addWidget(RadioButton);

      
      //setStyleSheet(QString("");
      selectedpalette = basepalette;
      selectedpalette.setColor(QPalette::Mid,basepalette.color(QPalette::Mid).lighter(150));
      
      setcolor(wfmcolor);
      setselected(selected);
    }
      
    void setcolor(WfmColor newcolor)
    {
      float Rscaled=round_to_uchar(newcolor.R*255.0);
      float Gscaled=round_to_uchar(newcolor.G*255.0);
      float Bscaled=round_to_uchar(newcolor.B*255.0);
      setStyleSheet(QString::fromStdString("QRadioButton { color:rgb(" + std::to_string((int)Rscaled) + ", " + std::to_string((int)Gscaled) + ", " + std::to_string((int)Bscaled) + "); } "));

      wfmcolor=newcolor;
    }
      
    void setselected(bool newselected)
    {
      selected=newselected;
      if (selected) {
	setPalette(selectedpalette);
	RadioButton->setPalette(basepalette);
      } else {
	setPalette(basepalette);
      }
    }
  };
  
    
  class QTWfmViewer : public QWidget {
    Q_OBJECT;
  public:
    QTWfmRender *OSGWidget;
    std::shared_ptr<mutablewfmdb> wfmdb;
    std::shared_ptr<display_info> info;
    std::shared_ptr<mutableinfostore> selected;
    std::shared_ptr<snde::geometry> sndegeom;
    std::shared_ptr<trm> rendering_revman;

    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    
    std::unordered_map<std::string,QTWfmSelector *> Selectors; // indexed by FullName
    
    QHBoxLayout *layout;
    QWidget *DesignerTree;
    QWidget *WfmListScrollAreaContent;
    QVBoxLayout *WfmListScrollAreaLayout;
    osg::ref_ptr<OSGData> DataRenderer; 
    osg::ref_ptr<OSGComponent> GeomRenderer;

    std::shared_ptr<osg_instancecache> geomcache;
    
    QTWfmViewer(std::shared_ptr<mutablewfmdb> wfmdb,std::shared_ptr<snde::geometry> sndegeom,std::shared_ptr<trm> rendering_revman,cl_context context, cl_device_id device, cl_command_queue queue,QWidget *parent =0)
      : QWidget(parent),
	wfmdb(wfmdb),
	sndegeom(sndegeom),
	rendering_revman(rendering_revman),
	context(context),
	device(device),
	queue(queue)   /* !!!*** Should either use reference tracking objects for context, device, and queue, or explicitly reference them and dereference them in the destructor !!!*** */ 
    {

      geomcache = std::make_shared<osg_instancecache>(sndegeom,context,device,queue);
      
      QFile file(":/qtwfmviewer.ui");
      file.open(QFile::ReadOnly);
      
      QUiLoader loader;
      
      DesignerTree = loader.load(&file,this);
      file.close();

      layout = new QHBoxLayout();
      layout->addWidget(DesignerTree);
      setLayout(layout);

      QGridLayout *viewerGridLayout = DesignerTree->findChild<QGridLayout*>("viewerGridLayout");
      // we want to add our QOpenGLWidget to the 0,0 entry of the QGridLayout

      OSGWidget=new QTWfmRender(nullptr,this,viewerGridLayout->parentWidget());
      viewerGridLayout->addWidget(OSGWidget,0,0);
      
      WfmListScrollAreaContent=DesignerTree->findChild<QWidget *>("wfmListScrollAreaContent");
      WfmListScrollAreaLayout=new QVBoxLayout();
      info = std::make_shared<display_info>(wfmdb);
      //setWindowTitle(tr("QTWfmViewer"));


      update_wfm_list();
      rendering_revman->Start_Transaction();
      update_renderer();
      rendering_revman->End_Transaction();
    }

    void update_data_renderer(const std::vector<std::shared_ptr<display_channel>> &currentwfmlist)
    // Must be called inside a transaction!
    {
      GeomRenderer=nullptr; /* remove geom rendering while rendering data */ 

      if (!DataRenderer) {
	DataRenderer=new OSGData(rendering_revman);
      }
      DataRenderer->update(sndegeom,info,currentwfmlist,info->pixelsperdiv(OSGWidget->width(),OSGWidget->height()),OSGWidget->width(),OSGWidget->height(),context,device,queue);
      OSGWidget->SetTwoDimensional(true);
      OSGWidget->SetRootNode(DataRenderer);
    }

    void update_geom_renderer(std::shared_ptr<display_channel> geomchan,std::shared_ptr<mutablegeomstore> geomstore,const std::vector<std::shared_ptr<display_channel>> &currentwfmlist)
    // Must be called inside a transaction!
    {
      DataRenderer->clearcache(); // empty out data renderer
      
      if (!GeomRenderer  || (GeomRenderer && GeomRenderer->comp != geomstore->comp)) {
	// component mismatch: Need new GeomRenderer
	GeomRenderer=new OSGComponent(sndegeom,geomcache,geomstore->comp,rendering_revman);
      }
      OSGWidget->SetTwoDimensional(false);
      OSGWidget->SetRootNode(GeomRenderer);
    }
    
    void update_renderer()
    // Must be called inside a transaction!
    {
      std::vector<std::shared_ptr<display_channel>> currentwfmlist = info->update(wfmdb,selected,false,false);
      std::shared_ptr<display_channel> geomchan;
      // Is any channel a mutablegeomstore? i.e. do we render a 3D geometry
      std::shared_ptr<mutablegeomstore> geom;
      
      for (size_t pos=0;pos < currentwfmlist.size();pos++) {
	if (currentwfmlist[pos]->chan_data && std::dynamic_pointer_cast<mutablegeomstore>(currentwfmlist[pos]->chan_data)) {
	  geom=std::dynamic_pointer_cast<mutablegeomstore>(currentwfmlist[pos]->chan_data);
	  geomchan=currentwfmlist[pos];
	}
      }

      if (geom) {
	update_geom_renderer(geomchan,geom,currentwfmlist);
      } else {
	update_data_renderer(currentwfmlist);
      }
    }
    
    void update_wfm_list()  
    {
      std::vector<std::shared_ptr<display_channel>> currentwfmlist = info->update(wfmdb,nullptr,true,false);

      // clear touched flag for all selectors
      for(auto & selector: Selectors) {
	selector.second->touched_during_update=false;
      }

      // iterate over wfm list
      size_t pos=0;
      for (auto & displaychan: currentwfmlist) {
	std::lock_guard<std::mutex> displaychanlock(displaychan->displaychan_mutex);

	auto selector_iter = Selectors.find(displaychan->FullName);
	if (selector_iter == Selectors.end()) {
	  // create a new selector
	  QTWfmSelector *NewSel = new QTWfmSelector(displaychan->FullName,WfmColorTable[displaychan->ColorIdx],WfmListScrollAreaContent);
	  WfmListScrollAreaLayout->insertWidget(pos,NewSel);
	  Selectors[displaychan->FullName]=NewSel;
	  
	}

	QTWfmSelector *Sel = Selectors[displaychan->FullName];
	Sel->touched_during_update=true;
	
	if (WfmListScrollAreaLayout->indexOf(Sel) != pos) {
	  /* entry is out-of-order */
	  WfmListScrollAreaLayout->removeWidget(Sel);
	  WfmListScrollAreaLayout->insertWidget(pos,Sel);
	}

	if (Sel->wfmcolor != WfmColorTable[displaychan->ColorIdx]) {
	  Sel->setcolor(WfmColorTable[displaychan->ColorIdx]);
	}
	if (displaychan->Enabled != Sel->RadioButton->isChecked()) {
	  Sel->RadioButton->setChecked(displaychan->Enabled);
	  // NOTE: because we call setChecked() with the displaychan locked
	  // we must be sure that the displaychan update comes from the "clicked"
	  // signal, NOT the "checked" signal, lest we get a deadlock here	  
	}	
	
	pos++;
      }

      // re-iterate through selectors, removing any that weren't touched
      std::unordered_map<std::string,QTWfmSelector *>::iterator selector_iter, next_selector_iter;
  
      for(selector_iter=Selectors.begin(); selector_iter != Selectors.end(); selector_iter = next_selector_iter) {
	next_selector_iter=selector_iter;
	next_selector_iter++;

	if (!selector_iter->second->touched_during_update) {
	  // delete widget -- will auto-remove itself from widget tree
	  delete selector_iter->second;
	  // remove from selectors map
	  Selectors.erase(selector_iter);
	}
	
      }
    }


  };
    
  

  
    


}
#endif // SNDE_QTWFMVIEWER_HPP
