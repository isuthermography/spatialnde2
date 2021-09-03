#include <ios>
#include <cmath>

#include <QString>
#include <QWidget>
#include <QOpenGLWidget>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QTouchEvent>
#include <QFrame>
#include <QRadioButton>
#include <QToolButton>
#include <QPushButton>
#include <QAbstractSlider>
#include <QLineEdit>
#include <QSlider>
#include <QLayout>
#include <QVBoxLayout>
#include <QtUiTools/QUiLoader>
#include <QTimer>

#include <osg/Array>
#include <osgViewer/Viewer>
#include <osgGA/TrackballManipulator>

#include "rec_display.hpp"
#include "colormap.h"

#include "openscenegraph_geom.hpp"
#include "openscenegraph_data.hpp"
#include "openscenegraph_renderer.hpp"
#include "openscenegraph_picker.hpp"

#ifndef SNDE_QTRECVIEWER_HPP
#define SNDE_QTRECVIEWER_HPP

namespace snde {

  // See https://vicrucann.github.io/tutorials/
  // See https://gist.github.com/vicrucann/874ec3c0a7ba4a814bd84756447bc798 "OpenSceneGraph + QOpenGLWidget - minimal example"
  // and http://forum.openscenegraph.org/viewtopic.php?t=16549 "QOpenGLWidget in osgQt"
  // and http://forum.openscenegraph.org/viewtopic.php?t=15097 "OSG 3.2.1 and Qt5 Widget integration"

  class QTRecViewer; // forward declaration
  class QTRecRender; // forward declaration

  /*
  class QTRecGraphicsWindow: public osgViewer::GraphicsWindowEmbedded {
  public:
    QTRecRender *Renderer;
    
    QTRecGraphicsWindow(int x,int y,int width,int height,QTRecRender *Renderer); 

    virtual void requestRedraw();

    virtual void requestContinuousUpdate(bool needed=true);

  };
  */
  class QTRecRender : public QOpenGLWidget, public osg_renderer {
  public:
    QTRecViewer *QTViewer; 
    QTimer *AnimTimer; 
    
    // member variables from osg_renderer:
    // osg::ref_ptr<osgViewer::GraphicsWindowEmbedded> GraphicsWindow;
    // osg::ref_ptr<osgViewer::Viewer> Viewer;
    // osg::ref_ptr<osg::Camera> Camera;
    // osg::ref_ptr<osg::Node> RootNode;
    // osg::ref_ptr<osgGA::TrackballManipulator> Manipulator; // manipulator for 3D mode
    // bool twodimensional;

    // methods from osg_renderer:
    //void SetRootNode(osg::ref_ptr<osg::Node> RootNode)

    osg::ref_ptr<osg_picker> picker;
    
    
    QTRecRender(osg::ref_ptr<osg::Node> RootNode, QTRecViewer *QTViewer,QWidget *parent=0); // Note: Constructor body moved to qtrecviewer.cpp because of reference to QTViewer->display

    
    void update()
    {
      fprintf(stderr,"QOpenGLWidget update()\n");
      fprintf(stderr,"Manipulator animating: %d\n",(int)Manipulator->isAnimating());
      QOpenGLWidget::update();

    }
    
    void SetTwoDimensional(bool twod)
    {
      twodimensional=twod;
      SetProjectionMatrix();

      if (twod) {
	Viewer->setCameraManipulator(nullptr);
	Camera->setViewMatrix(osg::Matrixd::identity());
      } else {
	if (Viewer->getCameraManipulator() != Manipulator) {
	  Viewer->setCameraManipulator(Manipulator);
	}
      }
    }
    
    void SetProjectionMatrix(void)
    {
      if (twodimensional) {
	Camera->setProjectionMatrixAsOrtho(0,static_cast<double>(width()),0,static_cast<double>(height()),-10.0,1000.0);
	
      } else {
	Camera->setProjectionMatrixAsPerspective(30.0,static_cast<float>(this->width())/static_cast<float>(this->height()),0.1,1000.0);
	
      }
    }

  virtual void ClearPickedOrientation(); // in qtrecviewer.cpp

    
  protected:
    virtual void paintGL(); // rendeirng code in qtrecviewer.cpp

    virtual void resizeGL(int width,int height); // code in qtrecviewer.cpp

    virtual void initializeGL()
    {
      // any opengl initialization here...
      
    }

    
    virtual void mouseMoveEvent(QMouseEvent *event)
    {
      // translate Qt mouseMoveEvent to OpenSceneGraph
      GraphicsWindow->getEventQueue()->mouseMotion(event->x(), event->y()); //,event->timestamp()/1000.0);

      // for some reason drags with the middle mouse button pressed
      // get the buttons field filtered out (?)
      
      // should we only update if a button is pressed??
      fprintf(stderr,"buttons=%llx\n",(unsigned long long)event->buttons());
      if (event->buttons()) {
	update();
      }
    }

    virtual void mousePressEvent(QMouseEvent *event)
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

      GraphicsWindow->getEventQueue()->mouseButtonPress(event->x(),event->y(),button); //,event->timestamp()/1000.0);

      update();
      
      // Can adapt QT events -> OSG events here
      // would do e.g.
      //GraphicsWindow->getEventQueue()->mouseButtonPress(event->x(),event->y(),button#)
      // Would also want to forward mouseButtonRelease() 
    }

    virtual void mouseReleaseEvent(QMouseEvent *event)
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

      GraphicsWindow->getEventQueue()->mouseButtonRelease(event->x(),event->y(),button); //,event->timestamp()/1000.0);

      update();
      
      // Can adapt QT events -> OSG events here
      // would do e.g.
      //GraphicsWindow->getEventQueue()->mouseButtonPress(event->x(),event->y(),button#)
      // Would also want to forward mouseButtonRelease() 
    }

    virtual void wheelEvent(QWheelEvent *event)
    {
      GraphicsWindow->getEventQueue()->mouseScroll( (event->delta() > 0) ?
						    osgGA::GUIEventAdapter::SCROLL_UP :
						    osgGA::GUIEventAdapter::SCROLL_DOWN);
      //event->timestamp()/1000.0);
      
    }


    virtual bool event(QEvent *event)
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
	update();
	return true;
      } else {
	
	return QOpenGLWidget::event(event);
      }
    }
  };

  

  
  class QTRecSelector : public QFrame {
  public:
    std::string Name;
    QTRecViewer *Viewer;
    QRadioButton *RadioButton;
    bool touched_during_update; // has this QTRecSelector been touched during the latest update pass
    QPalette basepalette;
    QPalette selectedpalette;
    bool selected;
    RecColor reccolor; 
    
    QTRecSelector(QTRecViewer *Viewer,std::string Name,RecColor reccolor,QWidget *parent=0) :
      QFrame(parent),
      Viewer(Viewer),
      RadioButton(new QRadioButton(QString::fromStdString(Name),this)),
      Name(Name),
      basepalette(palette()),
      touched_during_update(true),
      selected(false),
      reccolor(reccolor)
    {
      setFrameShadow(QFrame::Shadow::Raised);
      setFrameShape(QFrame::Shape::Box);
      setLineWidth(1);
      setMidLineWidth(2);
      setSizePolicy(QSizePolicy(QSizePolicy::Minimum,QSizePolicy::Minimum));
      RadioButton->setAutoExclusive(false);
      QLayout *Layout=new QVBoxLayout();
      setLayout(Layout);
      Layout->addWidget(RadioButton);

      // eventfilter monitors for focus activation of the Selector and for key presses for display adjustment. Also inhibits use of cursor keys to move focus around
      RadioButton->installEventFilter(this);
      
      //setStyleSheet(QString("");
      selectedpalette = basepalette;
      selectedpalette.setColor(QPalette::Mid,basepalette.color(QPalette::Mid).lighter(150));
      
      //setcolor(reccolor);
      setselected(selected);
    }
      
    void setcolor(RecColor newcolor)
    {
      float Rscaled=round_to_uchar(newcolor.R*255.0);
      float Gscaled=round_to_uchar(newcolor.G*255.0);
      float Bscaled=round_to_uchar(newcolor.B*255.0);

      std::string CSScolor = "rgb("+ std::to_string((int)Rscaled) + ", " + std::to_string((int)Gscaled) + ", " + std::to_string((int)Bscaled) + ")";

      std::string BorderColor;

      if (selected) {
	BorderColor=CSScolor;
      } else {
	BorderColor="gray";
      }
      
      setStyleSheet(QString::fromStdString("QRadioButton { color:"+CSScolor+"; }\n" + "QFrame { border: 2px solid " + BorderColor + "; }\n"));

      reccolor=newcolor;
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
      setcolor(reccolor);
    }

    
    bool eventFilter(QObject *object,QEvent *event); // in qtrecviewer.cpp
  };



  /* Position management
     Position slider has 10001 (call it nsteps) integer steps. 
     let RelPos = (Position-(nsteps-1)/2)
     Mapping between scaled position and RelPos: 
     

     dScaledPos/dRelPos = 10^(C*ScaledPos/(num_div*unitsperdiv))

     dScaledPos/10^(C*ScaledPos/(num_div*unitsperdiv)) = dRelPos

     integral_SP1^SP2 10^-(C*ScaledPos/(num_div*unitsperdiv))dScaledPos = RelPos2-RelPos1

     let u = -(C*ScaledPos/(num_div*unitsperdiv))
     du = -(C/(num_div*unitsperdiv))*dScaledPos
     dScaledPos = -(num_div*unitsperdiv)/C du 

     -(num_div*unitsperdiv)/C integral_SP=SP1^SP=SP2 10^u du = RelPos2-RelPos1
     
     -(num_div*unitsperdiv)/C 10^u/ln 10|_SP=SP1^SP=SP2 = RelPos2-RelPos1

     (num_div*unitsperdiv)/C 10^(-C*ScaledPos1/(num_div*unitsperdiv)))/ln(10) - (num_div*unitsperdiv)/C 10^(-C*ScaledPos2/(num_div*unitsperdiv)))/ln(10) = RelPos2-RelPos1
     

     Let ScaledPos1 = 0.0
     Let RelPos1 = 0
     (num_div*unitsperdiv)/(C*ln(10)) (  1 - 10^(-C*ScaledPos2/(num_div*unitsperdiv))) = RelPos2

     Use absolute values
     RelPos2 = sgn(ScaledPos2)*(num_div*unitsperdiv)/(C*ln(10)) (  1 - 10^(-C*|ScaledPos2|/(num_div*unitsperdiv)))

     Use absolute values, substitute C formula from below
     RelPos2 = sgn(ScaledPos2)*((nsteps-1)/2 + 1) (  1 - 10^(-|ScaledPos2|/(ln(10)*((nsteps-1)/2+1))))

     at ScaledPos2=infinity,   RelPos2 = (num_div*unitsperdiv)/(C*ln(10))
     ... This should correspond to RelPos2 = (nsteps-1)/2 + 1
     5001 = (num_div*unitsperdiv)/(C*ln(10))
     C = (num_div*unitsperdiv)/(ln(10)*[ (nsteps-1)/2 + 1])

     
     Inverse formula (on absolute values)
     ScaledPos2 = -log10( 1 - RelPos2/((nsteps-1)/2 + 1) )*ln(10)*((nsteps-1)/2+1)
     ScaledPos2 = -ln( 1 - RelPos2/((nsteps-1)/2 + 1) )*((nsteps-1)/2+1)

     Check forward formula:
     [ 1-exp(-ScaledPos2/((nsteps-1)/2+1)) ]*((nsteps-1)/2+1) = RelPos2

     ... Power doesn't matter!
   */

  static int SliderPosFromScaledPos(double ScaledPos,double unitsperdiv, double num_div,double power,int nsteps)
  {
    double retdbl=0.0;
    if (ScaledPos < 0) retdbl = -((nsteps-1)/2 + 1)*(1-exp(-fabs(ScaledPos)/(((nsteps-1)/2+1))));
    else retdbl = ((nsteps-1)/2 + 1)*(1-exp(-fabs(ScaledPos)/(((nsteps-1)/2+1))));

    // shift to 0...(nsteps-1) from -(nsteps-1)/2..(nsteps-1)/2

    retdbl+=round((nsteps-1.0)/2.0);

    if (retdbl < 0.0) retdbl=0.0;
    if (retdbl >= nsteps-1) retdbl=nsteps-1;

    return (int)retdbl;
  }

  static inline std::tuple<int,double> round_to_zoom_digit(double val)
  // Rounds an integer 0-10 to the nearest valid zoom digit (1,2, or 5)
  // returns (index,rounded) where index is 0, 1, or 2
  {
    int index=0;
    int intval=(int)val;

    assert(intval==val); // inputs should be small integers
    
    switch (intval) {
    case 0:
    case 1:
      val=1.0;
      index=0;
      break;
      
    case 2:
    case 3:
      val=2.0;
      index=1;
      break;
      
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
      val=5.0;
      index=2;
      break;
      
    default:
      assert(0); // means val is invalid (not an integer 0..10)
    }
    return std::make_tuple(index,val);
  }

  
  class qtrec_position_manager: public QObject
  {
    Q_OBJECT

  public:    
    std::shared_ptr<display_info> display;
    std::shared_ptr<display_channel> selected_channel;
    QAbstractSlider *HorizSlider; // owned by our parent
    QAbstractSlider *VertSlider; // owned by our parent

    QAbstractSlider *HorizZoom; // owned by our parent
    //QToolButton *HorizZoomInButton; // owned by our parent
    //QToolButton *HorizZoomOutButton; // owned by our parent
    QAbstractSlider *VertZoom; // owned by our parent
    //QToolButton *VertZoomInButton; // owned by our parent
    //QToolButton *VertZoomOutButton; // owned by our parent
    
    double power; /* determines nature of curve mapping between slider position and motion */
    int nsteps;
    int nzoomsteps;
    // zoom values go e.g. for nzoomsteps=7, in REVERSE ORDER units per division of 
    // 1e-1, 2e-1, 5e-1, 1e0, 2e0, 5e0, 1e1

    
    qtrec_position_manager(std::shared_ptr<display_info> display,QAbstractSlider *HorizSlider,QAbstractSlider *VertSlider,QAbstractSlider *HorizZoom,
			   //QToolButton *HorizZoomInButton, QToolButton *HorizZoomOutButton,
			   QAbstractSlider *VertZoom
			   //QToolButton *VertZoomInButton,QToolButton *VertZoomOutButton
			   ) :
      display(display),
      HorizSlider(HorizSlider),
      VertSlider(VertSlider),
      HorizZoom(HorizZoom),
      //HorizZoomInButton(HorizZoomInButton),
      //HorizZoomOutButton(HorizZoomOutButton),
      VertZoom(VertZoom)
      //VertZoomInButton(VertZoomInButton),
      //VertZoomOutButton(VertZoomOutButton)
    {
      power=100.0;
      nsteps=1000;
      nzoomsteps=43; // should be multiple of 6+1, e.g. 2*3*7=42, add 1 -> 43


      assert((nzoomsteps-1) % 6 == 0); // enforce multiple of 6+1
      
      HorizSlider->setRange(0,nsteps-1);
      VertSlider->setRange(0,nsteps-1);

      VertZoom->setRange(0,nzoomsteps-1);
      HorizZoom->setRange(0,nzoomsteps-1);
    }

    ~qtrec_position_manager();


    std::tuple<double,bool> GetHorizScale()
    {
     double horizscale = 2.0/display->horizontal_divisions;
     bool horizpixelflag=false;
     
     if (selected_channel) {
       std::shared_ptr<display_axis> a = display->GetFirstAxis(selected_channel->FullName());
       std::lock_guard<std::mutex> axislock(a->unit->admin);
       horizscale = a->unit->scale;
       horizpixelflag = a->unit->pixelflag;
     }
     
     return std::make_tuple(horizscale,horizpixelflag);
    }

    std::tuple<double,bool> GetVertScale()
    {
      double vertscale=0.0;
      bool success=false;
      bool vertpixelflag=false;
      if (selected_channel) {
	std::tie(success,vertscale,vertpixelflag) = display->GetVertScale(selected_channel);
	
      }

      if (!success) {
	std::lock_guard<std::mutex> adminlock(display->admin);

	vertscale = 2.0/display->vertical_divisions;
	vertpixelflag=false;
      }
     
      
      return std::make_tuple(vertscale,vertpixelflag);
    }

    void SetHorizScale(double horizscale,bool horizpixelflag)
    {
      //fprintf(stderr,"SetHorizScale %.2g\n",horizscale);
      if (selected_channel) {
	std::shared_ptr<display_axis> a = display->GetFirstAxis(selected_channel->FullName());
	{
	  std::lock_guard<std::mutex> adminlock(a->unit->admin);
	  a->unit->scale = horizscale;
	  a->unit->pixelflag = horizpixelflag;
	}
	selected_channel->mark_as_dirty();
      }
    }
    

    void SetVertScale(double vertscale,bool vertpixelflag)
    {
     if (selected_channel) {
       display->SetVertScale(selected_channel,vertscale,vertpixelflag);
       
     }
    }


    double GetScaleFromZoomPos(int zoompos,bool pixelflag)     
    {
      // see comment under definition of nzoomsteps for step definition
      double scale;
      int forwardzoompos = nzoomsteps-1 - zoompos; // regular zoompos is REVERSED so higher numbers mean fewer unitsperdiv... this one is FORWARD so higher numbers mean more unitsperdiv
      
      const unsigned zoommultiplier[] = {1,2,5};
      double zoompower = forwardzoompos/3 - (nzoomsteps-1)/6;
      scale = pow(10,zoompower)*zoommultiplier[forwardzoompos % 3];

      //fprintf(stderr,"GetScaleFromZoom(%d)=%f\n",zoompos,scale);
      //if (pixelflag) {
      //scale /= display->pixelsperdiv;
      //}


      return scale;
    }
    
    int GetZoomPosFromScale(double scale, bool pixelflag)
    {
      
      // see comment under definition of nzoomsteps for step definition
      //double unitsperdiv = scale; // a->unit->scale;
      //if (pixelflag) { // a->unit->pixelflag
      //unitsperdiv *= display->pixelsperdiv;
      //}
      
      double zoompower_floor = floor(log(scale)/log(10.0));
      double zoompower_ceil = ceil(log(scale)/log(10.0));
      
      double leadingdigit_floor;
      int leadingdigit_flooridx;
      std::tie(leadingdigit_flooridx,leadingdigit_floor) = round_to_zoom_digit(round(scale/pow(10,zoompower_floor)));
      
      double leadingdigit_ceil;
      int leadingdigit_ceilidx;
      std::tie(leadingdigit_ceilidx,leadingdigit_ceil) = round_to_zoom_digit(round(scale/pow(10,zoompower_ceil)));
      // Now find whichever reconstruction is closer, floor or ceil
      double floordist = fabs(leadingdigit_floor*pow(10,zoompower_floor)-scale);
      double ceildist = fabs(leadingdigit_ceil*pow(10,zoompower_ceil)-scale);
      
      int forwardsliderpos; // regular zoompos is REVERSED so higher numbers mean fewer unitsperdiv... this one is FORWARD so higher numbers mean more unitsperdiv
      if (floordist < ceildist) {
	forwardsliderpos = (nzoomsteps-1)/2 + ((int)zoompower_floor)*3 + leadingdigit_flooridx;
      } else {
	forwardsliderpos = (nzoomsteps-1)/2 + ((int)zoompower_ceil)*3 + leadingdigit_ceilidx;
	
      }
      if (forwardsliderpos >= nzoomsteps) {
	forwardsliderpos=nzoomsteps-1;
      }
      if (forwardsliderpos < 0) {
	forwardsliderpos=0;
      }

      return nzoomsteps-1 - forwardsliderpos; // return properly REVERSED sliderpos
    }
    
    
    std::tuple<double,double,double> GetHorizEdges()
    {
      double LeftEdge = -1.0;
      double RightEdge = 1.0;
      double horizunitsperdiv = (RightEdge-LeftEdge)/display->horizontal_divisions;

      if (selected_channel) {
	std::shared_ptr<display_axis> a = display->GetFirstAxis(selected_channel->FullName());

	{
	  std::lock_guard<std::mutex> adminlock(a->unit->admin);
	  horizunitsperdiv = a->unit->scale;
	  if (a->unit->pixelflag) horizunitsperdiv *= display->pixelsperdiv;
	}
	
	double CenterCoord;
	
	{
	  std::lock_guard<std::mutex> adminlock(a->admin);
	  CenterCoord=a->CenterCoord;
	}
	
	LeftEdge=CenterCoord-horizunitsperdiv*display->horizontal_divisions/2;
	RightEdge=CenterCoord+horizunitsperdiv*display->horizontal_divisions/2;

	//fprintf(stderr,"LeftEdge=%f, RightEdge=%f\n",LeftEdge,RightEdge);
	
      }
      return std::make_tuple(LeftEdge,RightEdge,horizunitsperdiv);
    }


    std::tuple<double,double,double> GetVertEdges()
    {
      double BottomEdge = -1.0;
      double TopEdge = 1.0;
      double vertunitsperdiv = (TopEdge-BottomEdge)/display->horizontal_divisions;

      if (selected_channel) {
	vertunitsperdiv = display->GetVertUnitsPerDiv(selected_channel);
	
	std::lock_guard<std::mutex> adminlock(selected_channel->admin);
	if (selected_channel->VertZoomAroundAxis) {
	  BottomEdge=-selected_channel->Position*vertunitsperdiv-vertunitsperdiv*display->vertical_divisions/2;
	  TopEdge=-selected_channel->Position*vertunitsperdiv+vertunitsperdiv*display->vertical_divisions/2;	
	} else {
	  BottomEdge=selected_channel->VertCenterCoord-vertunitsperdiv*display->vertical_divisions/2;
	  TopEdge=selected_channel->VertCenterCoord+vertunitsperdiv*display->vertical_divisions/2;	
	  
	}
      }
      return std::make_tuple(BottomEdge,TopEdge,vertunitsperdiv);
    }
  
    
    void trigger()
    {
      double LeftEdgeRec,RightEdgeRec,horizunitsperdiv;
      double BottomEdgeRec,TopEdgeRec,vertunitsperdiv;

      if (!selected_channel) return; 
      
      std::shared_ptr<mutableinfostore> chan_data;
      chan_data = display->recdb->lookup(selected_channel->FullName());

      if (!chan_data || std::dynamic_pointer_cast<mutablegeomstore>(chan_data)) return; // none of this matters for 3D rendering

      std::tie(LeftEdgeRec,RightEdgeRec,horizunitsperdiv)=GetHorizEdges();
      std::tie(BottomEdgeRec,TopEdgeRec,vertunitsperdiv)=GetVertEdges();

      /* NOTE: LeftEdgeRec and RightEdgeRec are in recording coordinates (with units)
	 so the farther right the recording is scrolled, the more negative 
	 LeftEdgeRec and RightEdgeRec are.
	 
	 By contrast, LeftEdgeInt and RightEdgeInt are the other way around and we 
	 negate and interchange them to make things work nicely

      */
      int LeftEdgeInt = SliderPosFromScaledPos(LeftEdgeRec/horizunitsperdiv,horizunitsperdiv, display->horizontal_divisions,power,nsteps);
      int RightEdgeInt = SliderPosFromScaledPos(RightEdgeRec/horizunitsperdiv,horizunitsperdiv, display->horizontal_divisions,power,nsteps);

      int BottomEdgeInt = SliderPosFromScaledPos(-BottomEdgeRec/vertunitsperdiv,vertunitsperdiv, display->vertical_divisions,power,nsteps);
      int TopEdgeInt = SliderPosFromScaledPos(-TopEdgeRec/vertunitsperdiv,vertunitsperdiv, display->vertical_divisions,power,nsteps);
      

      //fprintf(stderr,"LeftEdgeInt=%d; RightEdgeInt=%d\n",LeftEdgeInt,RightEdgeInt);
      
      bool horizblocked=HorizSlider->blockSignals(true); // prevent our change from propagating back -- because the slider being integer based will screw up the correct values
      HorizSlider->setMaximum(nsteps-(RightEdgeInt-LeftEdgeInt)-1);
      HorizSlider->setSliderPosition(LeftEdgeInt);
      HorizSlider->setPageStep(RightEdgeInt-LeftEdgeInt);
      //emit NewHorizSliderPosition(LeftEdgeInt);
      HorizSlider->blockSignals(horizblocked);

      //fprintf(stderr,"LeftEdgeInt=%d RightEdgeInt=%d width=%d\n",LeftEdgeInt,RightEdgeInt,RightEdgeInt-LeftEdgeInt);
      bool vertblocked=VertSlider->blockSignals(true); // prevent our change from propagating back -- because the slider being integer based will screw up the correct values
      VertSlider->setMaximum(nsteps-(TopEdgeInt-BottomEdgeInt)-1);
      VertSlider->setSliderPosition(BottomEdgeInt);
      VertSlider->setPageStep(TopEdgeInt-BottomEdgeInt);
      VertSlider->blockSignals(vertblocked);
      //fprintf(stderr,"BottomEdgeInt=%d TopEdgeInt=%d width=%d\n",BottomEdgeInt,TopEdgeInt,TopEdgeInt-BottomEdgeInt);
      //emit NewVertSliderPosition(BottomEdgeInt);
      //fprintf(stderr,"Emitting NewPosition()\n");


      double horizscale;
      bool horizpixelflag;
      std::tie(horizscale,horizpixelflag) = GetHorizScale();

      //fprintf(stderr,"HorizScale=%f\n",horizscale);
      
      double vertscale;
      bool vertpixelflag;
      std::tie(vertscale,vertpixelflag) = GetVertScale();

      int horiz_zoom_pos = GetZoomPosFromScale(horizscale, horizpixelflag);
      //fprintf(stderr,"Set Horiz Zoom sliderpos: %d\n",horiz_zoom_pos);
      
      HorizZoom->setSliderPosition(horiz_zoom_pos);


      int vert_zoom_pos = GetZoomPosFromScale(vertscale, vertpixelflag);
      fprintf(stderr,"Set vert Zoom sliderpos: %d\n",vert_zoom_pos);
      VertZoom->setSliderPosition(vert_zoom_pos);

      
      emit NewPosition();
    }

    void set_selected(std::shared_ptr<display_channel> chan)
    {
      selected_channel=chan;

      trigger();
      
    }
  public slots:

    void HorizSliderActionTriggered(int action)
    {
      double HorizPosn = HorizSlider->sliderPosition()-(nsteps-1)/2.0;

      double CenterCoord=0.0;
      double horizunitsperdiv=1.0;
      
      std::shared_ptr<display_axis> a = nullptr;
      if (selected_channel) {
	{
	  std::lock_guard<std::mutex> adminlock(selected_channel->admin);
	}
	a = display->GetFirstAxis(selected_channel->FullName());
	{
	  std::lock_guard<std::mutex> adminlock(a->unit->admin);
	  horizunitsperdiv = a->unit->scale;
	  if (a->unit->pixelflag) horizunitsperdiv *= display->pixelsperdiv;
	}
	
	{
	  std::lock_guard<std::mutex> adminlock(a->admin);
	  CenterCoord = a->CenterCoord;
	}
      }

      switch(action) {
      case QAbstractSlider::SliderSingleStepAdd:
	if (a) {
	  std::lock_guard<std::mutex> adminlock(a->admin);
	  a->CenterCoord+=horizunitsperdiv;
	}
	
	break;
      case QAbstractSlider::SliderSingleStepSub:
	if (a) {
	  std::lock_guard<std::mutex> adminlock(a->admin);
	  a->CenterCoord-=horizunitsperdiv;
	}
	break;

      case QAbstractSlider::SliderPageStepAdd:
	if (a) {
	  std::lock_guard<std::mutex> adminlock(a->admin);
	  a->CenterCoord+=horizunitsperdiv*display->horizontal_divisions;
	}
	break;
      case QAbstractSlider::SliderPageStepSub:
	if (a) {
	  std::lock_guard<std::mutex> adminlock(a->admin);
	  a->CenterCoord-=horizunitsperdiv*display->horizontal_divisions;
	}	
	break;

      case QAbstractSlider::SliderMove:
	if (a) {
	  std::lock_guard<std::mutex> adminlock(a->admin);
	  double LeftEdgeRec=-1.0;
	  if (HorizPosn < 0.0) {
	    LeftEdgeRec = log(1.0 - fabs(HorizPosn)/((nsteps-1)/2.0 + 1.0) )*((nsteps-1)/2+1)*horizunitsperdiv;
	  } else {
	    LeftEdgeRec = -log(1.0 - HorizPosn/((nsteps-1)/2.0 + 1.0) )*((nsteps-1)/2+1)*horizunitsperdiv;
	    
	  }
	  a->CenterCoord = (LeftEdgeRec + horizunitsperdiv*display->horizontal_divisions/2);
	  //fprintf(stderr,"HorizSliderMove: Setting CenterCoord to %f\n",a->CenterCoord);
	}
	
	break;
      }
      //if (a) {
      //fprintf(stderr,"HorizCenterCoord=%f\n",a->CenterCoord);
      //}
      trigger();
    }

    void VertSliderActionTriggered(int action)
    {
      double VertPosn = VertSlider->sliderPosition() - (nsteps-1)/2.0;

      double CenterCoord=0.0;
      double vertunitsperdiv=1.0;
      
      std::shared_ptr<display_axis> a = nullptr;
      if (selected_channel) {
	vertunitsperdiv = display->GetVertUnitsPerDiv(selected_channel);
	
	
      }

      switch(action) {
      case QAbstractSlider::SliderSingleStepAdd:
	if (selected_channel) {
	  std::lock_guard<std::mutex> adminlock(selected_channel->admin);
	  if (selected_channel->VertZoomAroundAxis) {
	    selected_channel->Position++;
	  } else {
	    selected_channel->VertCenterCoord -= vertunitsperdiv;
	  }
	}
	
	break;
      case QAbstractSlider::SliderSingleStepSub:
	if (selected_channel) {
	  std::lock_guard<std::mutex> adminlock(selected_channel->admin);
	  if (selected_channel->VertZoomAroundAxis) {
	    selected_channel->Position--;
	  } else {
	    selected_channel->VertCenterCoord += vertunitsperdiv;
	  }
	}
	break;

      case QAbstractSlider::SliderPageStepAdd:
	if (selected_channel) {
	  size_t vertical_divisions;
	  {
	    std::lock_guard<std::mutex> adminlock(display->admin);
	    vertical_divisions = display->vertical_divisions;
	  }
	  std::lock_guard<std::mutex> adminlock(selected_channel->admin);
	  if (selected_channel->VertZoomAroundAxis) {
	    selected_channel->Position+=vertical_divisions;
	  } else {
	    selected_channel->VertCenterCoord -= vertunitsperdiv*vertical_divisions;	    
	  }
	}
	break;
      case QAbstractSlider::SliderPageStepSub:
	if (selected_channel) {
	  size_t vertical_divisions;
	  {
	    std::lock_guard<std::mutex> adminlock(display->admin);
	    vertical_divisions = display->vertical_divisions;
	  }
	  std::lock_guard<std::mutex> adminlock(selected_channel->admin);
	  if (selected_channel->VertZoomAroundAxis) {
	    selected_channel->Position-=vertical_divisions;
	  } else {
	    selected_channel->VertCenterCoord += vertunitsperdiv*vertical_divisions;	    
	  }
	}	
	break;
	
      case QAbstractSlider::SliderMove:
	double BottomEdgeRec=1.0;

	size_t vertical_divisions;
	{
	  std::lock_guard<std::mutex> adminlock(display->admin);
	  vertical_divisions = display->vertical_divisions;
	}
	
	if (selected_channel) {
	  if (VertPosn < 0.0) {
	    BottomEdgeRec = -log(1.0 - fabs(VertPosn)/((nsteps-1)/2.0 + 1.0) )*((nsteps-1)/2+1)*vertunitsperdiv;
	  } else {
	    BottomEdgeRec = log(1.0 - VertPosn/((nsteps-1)/2.0 + 1.0) )*((nsteps-1)/2+1)*vertunitsperdiv;
	  }
	  
	  if (selected_channel->VertZoomAroundAxis) {
	    std::lock_guard<std::mutex> adminlock(selected_channel->admin);
	    selected_channel->Position = -BottomEdgeRec/vertunitsperdiv + vertical_divisions/2;
	    //fprintf(stderr,"Position = %f vert units\n",selected_channel->Position);
	  } else {
	    std::lock_guard<std::mutex> adminlock(selected_channel->admin);
	    selected_channel->VertCenterCoord = BottomEdgeRec + vertunitsperdiv*vertical_divisions/2;
	    //fprintf(stderr,"BottomEdgeRec=%f; VertCenterCoord=%f\n",BottomEdgeRec,selected_channel->VertCenterCoord);
	  }
	  

	}
	break;
      }
      selected_channel->mark_as_dirty();
      trigger();
      
    }

  
    void HorizZoomActionTriggered(int action)
    {
      
      double horizscale;
      bool horizpixelflag;
      std::tie(horizscale,horizpixelflag) = GetHorizScale();
      
      int horiz_zoom_pos = GetZoomPosFromScale(horizscale, horizpixelflag);

      double rounded_scale = GetScaleFromZoomPos(horiz_zoom_pos,horizpixelflag);

      switch(action) {
      case QAbstractSlider::SliderSingleStepAdd:
      case QAbstractSlider::SliderPageStepAdd:

	if (rounded_scale > horizscale) {
	  // round up
	  SetHorizScale(rounded_scale,horizpixelflag);	  
	} else {
	  // Step up
	  double new_scale = GetScaleFromZoomPos(horiz_zoom_pos+1,horizpixelflag);
	  SetHorizScale(new_scale,horizpixelflag);	  
	}
	break;
	
      case QAbstractSlider::SliderSingleStepSub:
      case QAbstractSlider::SliderPageStepSub:
	if (rounded_scale < horizscale) {
	  // round down
	  SetHorizScale(rounded_scale,horizpixelflag);	  
	} else {
	  // Step down
	  double new_scale = GetScaleFromZoomPos(horiz_zoom_pos-1,horizpixelflag);
	  SetHorizScale(new_scale,horizpixelflag);	  
	}
	break;

      case QAbstractSlider::SliderMove:
	fprintf(stderr,"Got Horiz Zoom slidermove: %d\n",HorizZoom->sliderPosition());

	double HorizZoomPosn = GetScaleFromZoomPos(HorizZoom->sliderPosition(),horizpixelflag);
	SetHorizScale(HorizZoomPosn,horizpixelflag);	  
		
	break;
      }
      trigger();
    }


    void VertZoomActionTriggered(int action)
    {
      
      double vertscale;
      bool vertpixelflag;
      std::tie(vertscale,vertpixelflag) = GetVertScale();
      
      int vert_zoom_pos = GetZoomPosFromScale(vertscale, vertpixelflag);

      double rounded_scale = GetScaleFromZoomPos(vert_zoom_pos,vertpixelflag);
      
      switch(action) {
      case QAbstractSlider::SliderSingleStepAdd:
      case QAbstractSlider::SliderPageStepAdd:
	
	if (rounded_scale > vertscale) {
	  // round up
	  SetVertScale(rounded_scale,vertpixelflag);	  
	} else {
	  // Step up
	  double new_scale = GetScaleFromZoomPos(vert_zoom_pos+1,vertpixelflag);
	  SetVertScale(new_scale,vertpixelflag);	  
	}
	break;
	
      case QAbstractSlider::SliderSingleStepSub:
      case QAbstractSlider::SliderPageStepSub:
	if (rounded_scale < vertscale) {
	  // round down
	  SetVertScale(rounded_scale,vertpixelflag);	  
	} else {
	  // Step down
	  double new_scale = GetScaleFromZoomPos(vert_zoom_pos-1,vertpixelflag);
	  SetVertScale(new_scale,vertpixelflag);	  
	}
	break;

      case QAbstractSlider::SliderMove:
	double VertZoomPosn = GetScaleFromZoomPos(VertZoom->sliderPosition(),vertpixelflag);
	SetVertScale(VertZoomPosn,vertpixelflag);	  
		
	break;
      }
      trigger();
    }

    void VertZoomIn(bool)
    {
      //fprintf(stderr,"VertZoomIn()\n");
      VertZoomActionTriggered(QAbstractSlider::SliderSingleStepAdd);      
    }
    
    void VertZoomOut(bool)
    {
      VertZoomActionTriggered(QAbstractSlider::SliderSingleStepSub);      
    }

    void HorizZoomIn(bool)
    {
      //fprintf(stderr,"HorizZoomIn()\n");
      HorizZoomActionTriggered(QAbstractSlider::SliderSingleStepAdd);      
    }
    
    void HorizZoomOut(bool)
    {
      HorizZoomActionTriggered(QAbstractSlider::SliderSingleStepSub);      
    }


    
    signals:
    void NewPosition();
    //void NewHorizSliderPosition(int value);
    //void NewVertSliderPosition(int value);
  };
    


  
    
  class QTRecViewer : public QWidget {
    Q_OBJECT
  public:
    QTRecRender *OSGWidget;
    std::shared_ptr<mutablerecdb> recdb;
    std::shared_ptr<display_info> display;
    std::string selected; // name of selected channel
    std::shared_ptr<snde::geometry> sndegeom;
    std::shared_ptr<trm> rendering_revman;

    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    
    std::unordered_map<std::string,QTRecSelector *> Selectors; // indexed by FullName
    
    QHBoxLayout *layout;
    QWidget *DesignerTree;
    QWidget *RecListScrollAreaContent;
    QVBoxLayout *RecListScrollAreaLayout;
    //   QSpacerItem *RecListScrollAreaBottomSpace;
    QLineEdit *ViewerStatus;
    osg::ref_ptr<OSGData> DataRenderer; 
    osg::ref_ptr<OSGComponent> GeomRenderer;

    std::shared_ptr<qtrec_position_manager> posmgr; 
    
    std::shared_ptr<osg_instancecache> geomcache;
    std::shared_ptr<osg_parameterizationcache> paramcache;
    std::shared_ptr<osg_texturecache> texcache;
    
    QTRecViewer(std::shared_ptr<mutablerecdb> recdb,std::shared_ptr<snde::geometry> sndegeom,std::shared_ptr<trm> rendering_revman,cl_context context, cl_device_id device, cl_command_queue queue,QWidget *parent =0)
      : QWidget(parent),
	recdb(recdb),
	sndegeom(sndegeom),
	rendering_revman(rendering_revman),
	context(context),
	device(device),
	queue(queue)   /* !!!*** Should either use reference tracking objects for context, device, and queue, or explicitly reference them and dereference them in the destructor !!!*** */ 
    {

      texcache=std::make_shared<osg_texturecache>(sndegeom,rendering_revman,recdb,context,device,queue);
      paramcache=std::make_shared<osg_parameterizationcache>(sndegeom,context,device,queue);
      
      geomcache = std::make_shared<osg_instancecache>(sndegeom,recdb,paramcache,context,device,queue);
      
      QFile file(":/qtrecviewer.ui");
      file.open(QFile::ReadOnly);
      
      QUiLoader loader;
      
      DesignerTree = loader.load(&file,this);
      file.close();

      // Set all widgets in DesignerTree to have a focusPolicy of Qt::NoFocus
      QList<QWidget *> DTSubWidgets = DesignerTree->findChildren<QWidget *>();
      for (auto DTSubWid = DTSubWidgets.begin();DTSubWid != DTSubWidgets.end();DTSubWid++) {
	(*DTSubWid)->setFocusPolicy(Qt::NoFocus);
      }

      layout = new QHBoxLayout();
      layout->addWidget(DesignerTree);
      setLayout(layout);

      QGridLayout *viewerGridLayout = DesignerTree->findChild<QGridLayout*>("viewerGridLayout");
      // we want to add our QOpenGLWidget to the 1,0 entry of the QGridLayout

      display = std::make_shared<display_info>(recdb);

      OSGWidget=new QTRecRender(nullptr,this,viewerGridLayout->parentWidget());
      viewerGridLayout->addWidget(OSGWidget,1,0);
      
      RecListScrollAreaContent=DesignerTree->findChild<QWidget *>("recListScrollAreaContent");
      RecListScrollAreaLayout=new QVBoxLayout();
      RecListScrollAreaContent->setLayout(RecListScrollAreaLayout);

      
      //setWindowTitle(tr("QTRecViewer"));

      ViewerStatus=DesignerTree->findChild<QLineEdit *>("ViewerStatus");

      QAbstractSlider *HorizSlider = DesignerTree->findChild<QAbstractSlider*>("horizontalScrollBar");
      QAbstractSlider *VertSlider = DesignerTree->findChild<QAbstractSlider*>("verticalScrollBar");

      QAbstractSlider *HorizZoom = DesignerTree->findChild<QAbstractSlider*>("horizZoomSlider");
      QAbstractSlider *VertZoom = DesignerTree->findChild<QAbstractSlider*>("vertZoomSlider");
      QToolButton *HorizZoomInButton = DesignerTree->findChild<QToolButton*>("horizZoomInButton");
      QToolButton *HorizZoomOutButton = DesignerTree->findChild<QToolButton*>("horizZoomOutButton");
      QToolButton *VertZoomInButton = DesignerTree->findChild<QToolButton*>("vertZoomInButton");
      QToolButton *VertZoomOutButton = DesignerTree->findChild<QToolButton*>("vertZoomOutButton");

      QToolButton *DarkenButton = DesignerTree->findChild<QToolButton*>("DarkenButton");
      QToolButton *ResetIntensityButton = DesignerTree->findChild<QToolButton*>("ResetIntensityButton");
      QToolButton *BrightenButton = DesignerTree->findChild<QToolButton*>("BrightenButton");
      QToolButton *LessContrastButton = DesignerTree->findChild<QToolButton*>("LessContrastButton");
      QToolButton *MoreContrastButton = DesignerTree->findChild<QToolButton*>("MoreContrastButton");

      
      
      // Force slider up and down arrows to be together, by some fixed-size QML magic...
      HorizSlider->setStyleSheet(QString::fromStdString("QScrollBar:horizontal { \n"
							"   border: 2px;\n"
							"   height: 20px;\n"
							"   margin: 0px 60px 0px 0px;\n"
							"}\n"
							"QScrollBar::add-line:horizontal {\n"
							"   width: 20px;\n"
							"   border: 2px outset black;\n"
							"   subcontrol-position: right;\n"
							"   subcontrol-origin: margin;\n"
							"}\n"
							"QScrollBar::sub-line:horizontal {\n"
							"   width: 20px;\n"
							"   border: 2px outset black;\n"
							"   subcontrol-position: top right;\n"
							"   subcontrol-origin: margin;\n"
							"   position: absolute;\n"
							"   right: 30px;\n"
							
							"}\n"
							"QScrollBar::left-arrow:horizontal {\n"
							"   width: 18px;\n"
							"   height: 18px;\n"
							"   color: black;\n"
							//"   background: pink;\n"
							"   image: url(\":/larrow.png\");\n"
							"}\n"
							"QScrollBar::right-arrow:horizontal {\n"
							"   width: 18px;\n"
							"   height: 18px;\n"
							"   color: black;\n"
							//"   background: pink;\n"
							"   image: url(\":/rarrow.png\");\n"
							"}\n"));
      
      VertSlider->setStyleSheet(QString::fromStdString("QScrollBar:vertical { \n"
							"   border: 2px;\n"
							"   width: 20px;\n"
							"   margin: 0px 0px 60px 0px;\n"
							"}\n"
							"QScrollBar::add-line:vertical {\n"
							"   height: 20px;\n"
							"   border: 2px outset black;\n"
							"   subcontrol-position: bottom;\n"
							"   subcontrol-origin: margin;\n"
							"}\n"
							"QScrollBar::sub-line:vertical {\n"
							"   height: 20px;\n"
							"   border: 2px outset black;\n"
							"   subcontrol-position: bottom;\n"
							"   subcontrol-origin: margin;\n"
							"   position: absolute;\n"
							"   bottom: 30px;\n"
							
							"}\n"
							"QScrollBar::up-arrow:vertical {\n"
							"   width: 18px;\n"
							"   height: 18px;\n"
							"   color: black;\n"
							//"   background: pink;\n"
							"   image: url(\":/uarrow.png\");\n"
							"}\n"
							"QScrollBar::down-arrow:vertical {\n"
							"   width: 18px;\n"
							"   height: 18px;\n"
							"   color: black;\n"
							//"   background: pink;\n"
							"   image: url(\":/darrow.png\");\n"
							"}\n"));

      posmgr = std::make_shared<qtrec_position_manager>(display,HorizSlider,VertSlider,HorizZoom,VertZoom);
      QObject::connect(HorizSlider,SIGNAL(actionTriggered(int)),
		       posmgr.get(), SLOT(HorizSliderActionTriggered(int)));

      QObject::connect(VertSlider,SIGNAL(actionTriggered(int)),
		       posmgr.get(), SLOT(VertSliderActionTriggered(int)));
      

      QObject::connect(HorizZoom,SIGNAL(actionTriggered(int)),
		       posmgr.get(), SLOT(HorizZoomActionTriggered(int)));
      QObject::connect(VertZoom,SIGNAL(actionTriggered(int)),
		       posmgr.get(), SLOT(VertZoomActionTriggered(int)));

      QObject::connect(VertZoomInButton,SIGNAL(clicked(bool)),
		       posmgr.get(), SLOT(VertZoomIn(bool)));
      QObject::connect(HorizZoomInButton,SIGNAL(clicked(bool)),
		       posmgr.get(), SLOT(HorizZoomIn(bool)));
      
      QObject::connect(VertZoomOutButton,SIGNAL(clicked(bool)),
		       posmgr.get(), SLOT(VertZoomOut(bool)));
      QObject::connect(HorizZoomOutButton,SIGNAL(clicked(bool)),
		       posmgr.get(), SLOT(HorizZoomOut(bool)));

      QObject::connect(DarkenButton,SIGNAL(clicked(bool)),
		       this, SLOT(Darken(bool)));
      QObject::connect(ResetIntensityButton,SIGNAL(clicked(bool)),
		       this, SLOT(ResetIntensity(bool)));
      QObject::connect(BrightenButton,SIGNAL(clicked(bool)),
		       this, SLOT(Brighten(bool)));

      QObject::connect(LessContrastButton,SIGNAL(clicked(bool)),
		       this, SLOT(LessContrast(bool)));
      QObject::connect(MoreContrastButton,SIGNAL(clicked(bool)),
		       this, SLOT(MoreContrast(bool)));
      
      
      
      QObject::connect(posmgr.get(),SIGNAL(NewPosition()),
		       OSGWidget,SLOT(update()));

      QObject::connect(this,SIGNAL(NeedRedraw()),
		       OSGWidget,SLOT(update()));

      
      QObject::connect(posmgr.get(),SIGNAL(NewPosition()),
		       this,SLOT(UpdateViewerStatus()));

      
      // eventfilter monitors for keypresses
      //      installEventFilter(this);
      
      update_rec_list();
      rendering_revman->Start_Transaction();
      update_renderer();
      rendering_revman->End_Transaction();
      posmgr->trigger();
      
    }

    
    std::shared_ptr<display_channel> FindDisplayChan(QTRecSelector *Selector)
    {

      if (!Selector) return nullptr;
      
      auto ci_iter = display->channel_info.find(Selector->Name);
      if (ci_iter != display->channel_info.end()) {
	auto & displaychan = ci_iter->second;
	
	if (displaychan->FullName()==Selector->Name) {
	  //auto selector_iter = Selectors.find(displaychan->FullName);
	  //if (selector_iter != Selectors.end() && selector_iter->second==Selector) {
	  return displaychan;
	}
      }
      return nullptr; 
    }
    
    void set_selected(QTRecSelector *Selector)
    // assumes Selector already highlighted 
    {

      std::shared_ptr<display_channel> displaychan=FindDisplayChan(Selector);
      
      Selector->setselected(true);
      //std::shared_ptr<iterablerecrefs> reclist=recdb->reclist();
      
      //for (auto reciter=reclist->begin();reciter != reclist->end();reciter++) {
      //std::shared_ptr<mutableinfostore> infostore=*reciter;
      posmgr->set_selected(displaychan);
      selected = displaychan->FullName();
      //}
      
      deselect_other_selectors(Selector);

      //UpdateViewerStatus(); // Now taken care of by posmgr->set_selected()'s call to trigger which emits into this slot
    }
    
    void update_data_renderer(const std::vector<std::shared_ptr<display_channel>> &currentreclist)
    // Must be called inside a transaction!
    {
      GeomRenderer=nullptr; /* remove geom rendering while rendering data */ 

      if (!DataRenderer) {
	DataRenderer=new OSGData(display,rendering_revman);
      }
      display->set_pixelsperdiv(OSGWidget->width(),OSGWidget->height());
      DataRenderer->update(sndegeom,recdb,selected,currentreclist,OSGWidget->width(),OSGWidget->height(),context,device,queue);
      OSGWidget->SetTwoDimensional(true);
      OSGWidget->SetRootNode(DataRenderer);
    }

    void update_geom_renderer(std::shared_ptr<display_channel> geomchan,std::string recname /*std::shared_ptr<mutablegeomstore> geomstore*/,const std::vector<std::shared_ptr<display_channel>> &currentreclist)
    // Must be called inside a transaction!
    {
      if (DataRenderer) {
	DataRenderer->clearcache(); // empty out data renderer
      }
      
      if (!GeomRenderer  || (GeomRenderer && GeomRenderer->recname != recname)) {
	// component mismatch: Need new GeomRenderer
	fprintf(stderr,"New OSGComponent()\n");
	GeomRenderer=new OSGComponent(sndegeom,geomcache,paramcache,texcache,recdb,rendering_revman,recname /*geomstore*/,display);
      }
      OSGWidget->SetTwoDimensional(false);
      OSGWidget->SetRootNode(GeomRenderer);
    }
    
    void update_renderer()
    // Must be called inside a transaction!
    {
      std::vector<std::shared_ptr<display_channel>> currentreclist = display->update(selected,false,false);
      std::shared_ptr<display_channel> geomchan;
      // Is any channel a mutablegeomstore? i.e. do we render a 3D geometry
      std::shared_ptr<mutablegeomstore> geom;
      
      for (size_t pos=0;pos < currentreclist.size();pos++) {
	std::shared_ptr<mutableinfostore> chan_data;
	chan_data = display->recdb->lookup(currentreclist[pos]->FullName());
	
	if (chan_data && std::dynamic_pointer_cast<mutablegeomstore>(chan_data)) {
	  geom=std::dynamic_pointer_cast<mutablegeomstore>(chan_data);
	  geomchan=currentreclist[pos];
	}
      }
      
      if (geom) {
	update_geom_renderer(geomchan,geomchan->FullName(),currentreclist);
      } else {
	update_data_renderer(currentreclist);
      }
    }

    void lock_renderer(std::shared_ptr<lockingprocess_threaded> lockprocess)
    {
      if (OSGWidget->twodimensional) {
	// datarenderer
	DataRenderer->LockGraphics(lockprocess);
      } else {
	// geomrenderer
	GeomRenderer->LockVertexArraysTextures(lockprocess);
      }
    }
    
    void update_rec_list()  
    {
      std::vector<std::shared_ptr<display_channel>> currentreclist = display->update("",true,false);

      // clear touched flag for all selectors
      for(auto & selector: Selectors) {
	selector.second->touched_during_update=false;
      }

      // iterate over rec list
      size_t pos=0;
      for (auto & displaychan: currentreclist) {
	std::lock_guard<std::mutex> displaychanlock(displaychan->admin);

	auto selector_iter = Selectors.find(displaychan->FullName());
	if (selector_iter == Selectors.end()) {
	  // create a new selector
	  QTRecSelector *NewSel = new QTRecSelector(this,displaychan->FullName(),RecColorTable[displaychan->ColorIdx],RecListScrollAreaContent);
	  RecListScrollAreaLayout->insertWidget(pos,NewSel);
	  Selectors[displaychan->FullName()]=NewSel;
	  QObject::connect(NewSel->RadioButton,SIGNAL(clicked(bool)),
			   this,SLOT(SelectorClicked(bool)));
	  //QObject::connect(NewSel->RadioButton,SIGNAL(toggled(bool)),
	  //this,SLOT(SelectorClicked(bool)));
	}
      
	QTRecSelector *Sel = Selectors[displaychan->FullName()];
	Sel->touched_during_update=true;
	
	if (RecListScrollAreaLayout->indexOf(Sel) != pos) {
	  /* entry is out-of-order */
	  RecListScrollAreaLayout->removeWidget(Sel);
	  RecListScrollAreaLayout->insertWidget(pos,Sel);
	}

	if (Sel->reccolor != RecColorTable[displaychan->ColorIdx]) {
	  Sel->setcolor(RecColorTable[displaychan->ColorIdx]);
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
      std::unordered_map<std::string,QTRecSelector *>::iterator selector_iter, next_selector_iter;
  
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


      // remove anything else from the tree, including our stretch

      while (pos < RecListScrollAreaLayout->count()) {
	RecListScrollAreaLayout->removeItem(RecListScrollAreaLayout->itemAt(pos));
      }

      // add our stretch
      RecListScrollAreaLayout->addStretch(10);
    }

    void deselect_other_selectors(QTRecSelector *Selected)
    {
      for (auto & name_sel : Selectors) {
	if (name_sel.second != Selected) {
	  name_sel.second->setselected(false);
	}
      }
    }
  
  public slots:
    void UpdateViewerStatus()
    {
      double horizscale;
      bool horizpixelflag;
      std::string statusline="";
      bool needjoin=false;

      if (posmgr->selected_channel) {
	std::shared_ptr<mutableinfostore> chan_data;
	chan_data = recdb->lookup(posmgr->selected_channel->FullName());

	std::shared_ptr<display_axis> a = display->GetFirstAxis(posmgr->selected_channel->FullName());	
	std::shared_ptr<mutabledatastore> datastore=std::dynamic_pointer_cast<mutabledatastore>(chan_data);

	
	
	if (a) {
	  {
	    std::lock_guard<std::mutex> adminlock(a->unit->admin);
	    horizscale = a->unit->scale;
	    horizpixelflag = a->unit->pixelflag;
	  }
	  // Gawdawful C++ floating point formatting
	  //std::stringstream inipos;
	  //inipos << std::defaultfloat << std::setprecision(6) << a->CenterCoord;
	  
	  //std::stringstream horizscalestr;
	  //horizscalestr << std::defaultfloat << std::setprecision(6) << a->unit->scale;
	  //fprintf(stderr,"unitprint: %s\n",a->unit->unit.print(false).c_str());
	  
	  {
	    std::lock_guard<std::mutex> adminlock(a->admin);
	    
	    statusline += a->abbrev+"0=" + PrintWithSIPrefix(a->CenterCoord,a->unit->unit.print(false),3) + " " + PrintWithSIPrefix(horizscale,a->unit->unit.print(false),3);
	  }
	  if (horizpixelflag) {
	    statusline += "/px";
	  } else {
	    statusline += "/div";
	  }
	  needjoin=true;
	}
	
	if (datastore) {
	  size_t ndim=0;
	  {
	    rwlock_token_set datastore_tokens=empty_rwlock_token_set();
	    sndegeom->manager->locker->get_locks_read_lockable(datastore_tokens,datastore);
	    ndim=datastore->dimlen.size();
	  }
	  if (ndim > 0) {
	    if (ndim==1) {
	      a = display->GetAmplAxis(posmgr->selected_channel->FullName());
	      
	      if (a) {
		double scalefactor=0.0;
		{
		  std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
		  scalefactor=posmgr->selected_channel->Scale;
		}
		bool pixelflag=false;
		{
		  std::lock_guard<std::mutex> adminlock(a->admin);
		  pixelflag=a->unit->pixelflag;
		}
		if (needjoin) {
		  statusline += " | ";
		}
		double vertunitsperdiv=scalefactor;
		if (pixelflag) {		  
		  std::lock_guard<std::mutex> adminlock(display->admin);
		  vertunitsperdiv*=display->pixelsperdiv;
		}
		
		//std::stringstream inipos;
		double inipos;
		{
		  std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
		  if (posmgr->selected_channel->VertZoomAroundAxis) {
		    //inipos << std::defaultfloat << std::setprecision(6) << posmgr->selected_channel->Position*vertunitsperdiv;
		    inipos = posmgr->selected_channel->Position*vertunitsperdiv;
		  } else {
		    //inipos << std::defaultfloat << std::setprecision(6) << posmgr->selected_channel->VertCenterCoord;
		    inipos = posmgr->selected_channel->VertCenterCoord;
		  }
		}
		
		{
		  std::lock_guard<std::mutex> adminlock(a->unit->admin);
		  //std::stringstream vertscalestr;
		  //vertscalestr << std::defaultfloat << std::setprecision(6) << scalefactor;
		  
		  statusline += a->abbrev+"0=" + PrintWithSIPrefix(inipos,a->unit->unit.print(false),3) + " " + PrintWithSIPrefix(scalefactor,a->unit->unit.print(false),3);
		  //statusline += a->abbrev+"0=" + inipos.str() + " " + vertscalestr.str() + a->unit->unit.print(false);
		}
		if (horizpixelflag) {
		  statusline += "/px";
		} else {
		  statusline += "/div";
		}
		needjoin=true;
	      }
	    } else {
	      // ndim > 1
	      a=display->GetSecondAxis(posmgr->selected_channel->FullName());
	      if (a) {
		if (needjoin) {
		  statusline += " | ";
		}
		double scalefactor;
		double vertunitsperdiv;
		bool pixelflag=false;

		{
		  std::lock_guard<std::mutex> adminlock(a->unit->admin);
		  scalefactor=a->unit->scale;
		  vertunitsperdiv=scalefactor;
		  
		  pixelflag=a->unit->pixelflag;
		}

		{
		  std::lock_guard<std::mutex> adminlock(display->admin);
		  if (pixelflag) vertunitsperdiv*=display->pixelsperdiv;
		}
		
		//std::stringstream inipos;
		double inipos;
		{
		  std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
		  if (posmgr->selected_channel->VertZoomAroundAxis) {
		    //inipos << std::defaultfloat << std::setprecision(6) << posmgr->selected_channel->Position*vertunitsperdiv;
		    inipos = posmgr->selected_channel->Position*vertunitsperdiv;
		  } else {
		    //inipos << std::defaultfloat << std::setprecision(6) << posmgr->selected_channel->VertCenterCoord;
		    inipos = posmgr->selected_channel->VertCenterCoord;	      
		    
		  }
		}
		
		//std::stringstream vertscalestr;
		//vertscalestr << std::defaultfloat << std::setprecision(6) << scalefactor;
		
		//statusline += a->abbrev+"0=" + inipos.str() + " " + vertscalestr.str() + a->unit->unit.print(false);
		std::lock_guard<std::mutex> adminlock(a->admin);
		statusline += a->abbrev+"0=" + PrintWithSIPrefix(inipos,a->unit->unit.print(false),3) + " " + PrintWithSIPrefix(scalefactor,a->unit->unit.print(false),3);
		
		if (horizpixelflag) {
		  statusline += "/px";
		} else {
		  statusline += "/div";
		}
		needjoin=true;
		
		
	      }

	      double scalefactor;
	      //std::shared_ptr<mutableinfostore> chan_data;
	      {
		std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
		//chan_data = posmgr->selected_channel->chan_data;
		scalefactor=posmgr->selected_channel->Scale;
	      }
	      
	      a=display->GetAmplAxis(posmgr->selected_channel->FullName());
	      
	      if (a) {
		if (needjoin) {
		  statusline += " | ";
		}
		//double intensityunitsperdiv=scalefactor;
	      
		//if (a->unit->pixelflag) vertunitsperdiv*=display->pixelsperdiv;
		
		std::lock_guard<std::mutex> adminlock(a->admin);
		//statusline += a->abbrev+"0=" + inipos.str() + " " + intscalestr.str() + a->unit->unit.print(false) + "/intensity";
		statusline += a->abbrev+"0=" + PrintWithSIPrefix(posmgr->selected_channel->Offset,a->unit->unit.print(false),3) + " " + PrintWithSIPrefix(scalefactor,a->unit->unit.print(false),3) + "/intensity";
		
		needjoin=true;
		
		
	      }
	    }
	    
	  }
	}
	
      }
      
      ViewerStatus->setText(QString::fromStdString(statusline));
      
    }

    void SelectorClicked(bool checked)
    {
      //fprintf(stderr,"SelectorClicked()\n");
      QObject *obj = sender();      
      for (auto & name_selector: Selectors) {
	if (name_selector.second->RadioButton==obj) {

	  std::shared_ptr<display_channel> displaychan = FindDisplayChan(name_selector.second);
	  if (displaychan) {
	    {
	      std::lock_guard<std::mutex> adminlock(displaychan->admin);
	      displaychan->Enabled = checked;
	    }
	    displaychan->mark_as_dirty();
	    OSGWidget->update();
	  }
	}
      }
      
      
    }

    void Darken(bool checked)
    {
      if (posmgr->selected_channel) {
	std::shared_ptr<mutableinfostore> chan_data;
	chan_data = recdb->lookup(posmgr->selected_channel->FullName());

	//{
	//  std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
	//  chan_data=posmgr->selected_channel->chan_data;
	//}
	if (chan_data) {
	  
	  std::shared_ptr<display_axis> a=display->GetAmplAxis(posmgr->selected_channel->FullName());
	  std::shared_ptr<mutabledatastore> datastore=std::dynamic_pointer_cast<mutabledatastore>(chan_data);

	  size_t ndim=0;
	  if (datastore) {
	    rwlock_token_set datastore_tokens=empty_rwlock_token_set();
	    sndegeom->manager->locker->get_locks_read_lockable(datastore_tokens,datastore);
	    ndim=datastore->dimlen.size();
	    
	  }
	  
	  if (a && ndim > 1) {
	    {
	      std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
	      posmgr->selected_channel->Offset += posmgr->selected_channel->Scale/8.0;
	    }
	    posmgr->selected_channel->mark_as_dirty();
	    UpdateViewerStatus();
	    emit NeedRedraw();
	  }
	}
      }
    }

    void ResetIntensity(bool checked)
    {
      if (posmgr->selected_channel) {

	std::shared_ptr<mutableinfostore> chan_data;
	
	chan_data = recdb->lookup(posmgr->selected_channel->FullName());

	if (chan_data) {
	  
	  std::shared_ptr<display_axis> a=display->GetAmplAxis(posmgr->selected_channel->FullName());
	  std::shared_ptr<mutabledatastore> datastore=std::dynamic_pointer_cast<mutabledatastore>(chan_data);
	  size_t ndim=0;
	  if (datastore) {
	    rwlock_token_set datastore_tokens=empty_rwlock_token_set();
	    sndegeom->manager->locker->get_locks_read_lockable(datastore_tokens,datastore);
	    ndim=datastore->dimlen.size();
	  }
	  if (a && ndim > 1) {
	    {
	      std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
	      posmgr->selected_channel->Offset = 0.0;
	    }
	    posmgr->selected_channel->mark_as_dirty();
	    // ***!!! Should probably look at intensity bounds for channel instead ***!!!
	    UpdateViewerStatus();
	    emit NeedRedraw();
	  }
	}

      }
    }
    
    void Brighten(bool checked)
    {
      if (posmgr->selected_channel) {
	std::shared_ptr<mutableinfostore> chan_data;
	chan_data = recdb->lookup(posmgr->selected_channel->FullName());
	

	if (chan_data) {
	  
	  std::shared_ptr<display_axis> a=display->GetAmplAxis(posmgr->selected_channel->FullName());
	  std::shared_ptr<mutabledatastore> datastore=std::dynamic_pointer_cast<mutabledatastore>(chan_data);
	  
	  size_t ndim=0;
	  if (datastore) {
	    
	    rwlock_token_set datastore_tokens=empty_rwlock_token_set();
	    sndegeom->manager->locker->get_locks_read_lockable(datastore_tokens,datastore);
	    ndim=datastore->dimlen.size();
	    
	  }
	  if (a && ndim > 1) {
	    {
	      std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
	      posmgr->selected_channel->Offset -= posmgr->selected_channel->Scale/8.0;
	    }
	    posmgr->selected_channel->mark_as_dirty();
	    UpdateViewerStatus();
	    emit NeedRedraw();
	  }
	}
      }
    }

    void LessContrast(bool checked)
    {
      if (posmgr->selected_channel) {
	std::shared_ptr<mutableinfostore> chan_data;
	chan_data = recdb->lookup(posmgr->selected_channel->FullName());
	double Scale;
	{
	  std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
	  Scale=posmgr->selected_channel->Scale;
	}
	
	if (chan_data) {
	  
	  std::shared_ptr<mutabledatastore> datastore=std::dynamic_pointer_cast<mutabledatastore>(chan_data);
	  
	  std::shared_ptr<display_axis> a=display->GetAmplAxis(posmgr->selected_channel->FullName());
	  size_t ndim=0;
	  if (datastore) {	    
	    rwlock_token_set datastore_tokens=empty_rwlock_token_set();
	    sndegeom->manager->locker->get_locks_read_lockable(datastore_tokens,datastore);
	    ndim=datastore->dimlen.size();
	    
	  }
	  if (a && datastore && ndim > 1) {
	    //double contrastpower_floor = floor(log(posmgr->selected_channel->scale)/log(10.0));
	    double contrastpower_ceil = floor(log(Scale)/log(10.0));
	    
	    //double leadingdigit_floor;
	    //int leadingdigit_flooridx;
	    
	    //std::tie(leadingdigit_flooridx,leadingdigit_floor) = round_to_zoom_digit(round(posmgr->selected_channel->scale/pow(10,contrastpower_floor)));

	    // Less contrast -> more scale (i.e. more physical quantity/unit intensity
	    
	    double leadingdigit_ceil;
	    int leadingdigit_ceilidx;
	    
	    std::tie(leadingdigit_ceilidx,leadingdigit_ceil) = round_to_zoom_digit(round(Scale/pow(10,contrastpower_ceil)));
	    
	    double difference = leadingdigit_ceil*pow(10,contrastpower_ceil) - Scale;
	    if (fabs(difference/Scale) < .1) {
	      // no significant change from the ceil operation
	      // bump up by one notch
	      
	      const double newleadingdigits[]={2.0,5.0,10.0};
	      leadingdigit_ceil = newleadingdigits[leadingdigit_ceilidx];
	      
	      
	    }

	    {
	      std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
	      posmgr->selected_channel->Scale = leadingdigit_ceil*pow(10,contrastpower_ceil);
	    }
	    posmgr->selected_channel->mark_as_dirty();
	    UpdateViewerStatus();
	  
	    emit NeedRedraw();
	  }
	}
      }
    }

    void MoreContrast(bool checked)
    {
      if (posmgr->selected_channel) {
	std::shared_ptr<mutableinfostore> chan_data;
	chan_data = recdb->lookup(posmgr->selected_channel->FullName());
	double Scale;
	{
	  std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
	  Scale=posmgr->selected_channel->Scale;
	}
	
	if (chan_data) {
	  
	  std::shared_ptr<mutabledatastore> datastore=std::dynamic_pointer_cast<mutabledatastore>(chan_data);
	  
	  std::shared_ptr<display_axis> a=display->GetAmplAxis(posmgr->selected_channel->FullName());
	  size_t ndim=0;
	  if (datastore) {	    
	    rwlock_token_set datastore_tokens=empty_rwlock_token_set();
	    sndegeom->manager->locker->get_locks_read_lockable(datastore_tokens,datastore);
	    ndim=datastore->dimlen.size();
	    
	  }
	  if (a && datastore && ndim > 1) {
	    double contrastpower_floor = floor(log(Scale)/log(10.0));
	    
	    double leadingdigit_floor;
	    int leadingdigit_flooridx;
	    
	    std::tie(leadingdigit_flooridx,leadingdigit_floor) = round_to_zoom_digit(round(Scale/pow(10,contrastpower_floor)));
	    
	    // More contrast -> less scale (i.e. less physical quantity/unit intensity)
	    
	    
	    double difference = leadingdigit_floor*pow(10,contrastpower_floor) - Scale;
	    if (fabs(difference/Scale) < .1) {
	      // no significant change from the floor operation
	      // bump down by one notch
	      
	      const double newleadingdigits[]={0.5,1.0,2.0};
	      leadingdigit_floor = newleadingdigits[leadingdigit_flooridx];
	      
	      
	    }
	    
	    {
	      std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
	      posmgr->selected_channel->Scale = leadingdigit_floor*pow(10,contrastpower_floor);
	    }
	    posmgr->selected_channel->mark_as_dirty();
	    UpdateViewerStatus();
	    emit NeedRedraw();
	  }
	}
      }
    }

  signals:
    void NeedRedraw();
    

  };
  
  

  }
#endif // SNDE_QTRECVIEWER_HPP
