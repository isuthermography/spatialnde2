//#include <osgDB/WriteFile>

#include <QKeyEvent>

#include "snde/qtrecviewer_support.hpp"
#include "snde/qtrecviewer.hpp"
#include "snde/qt_osg_compositor.hpp"
#include "snde/python_support.hpp"

namespace snde {
  
    
  QTRecSelector::QTRecSelector(QTRecViewer *Viewer,std::string Name,RecColor reccolor,QWidget *parent/*=0*/) :
    QFrame(parent),
    Viewer(Viewer),
    RadioButton(new QRadioButton(QString::fromStdString(Name),this)),
    Name(Name),
    basepalette(palette()),
    touched_during_update(true),
    selected(false),
    reccolor(reccolor)
  {
    {
	SNDE_BeginDropPythonGILBlock
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
    SNDE_EndDropPythonGILBlock
	}
  }

  
  void QTRecSelector::setcolor(RecColor newcolor)
  {
    {
	SNDE_BeginDropPythonGILBlock
    float Rscaled=snde_round_to_uchar(newcolor.R*255.0);
    float Gscaled=snde_round_to_uchar(newcolor.G*255.0);
    float Bscaled=snde_round_to_uchar(newcolor.B*255.0);
    
    std::string CSScolor = "rgb("+ std::to_string((int)Rscaled) + ", " + std::to_string((int)Gscaled) + ", " + std::to_string((int)Bscaled) + ")";
    
    std::string BorderColor;
    
    if (selected) {
      BorderColor=CSScolor;
    } else {
      BorderColor="gray";
    }

    setStyleSheet(QString::fromStdString("QRadioButton { color:" + CSScolor + "; }\n" + "QFrame { border: 2px solid " + BorderColor + "; }\n"));
    
    reccolor=newcolor;
    SNDE_EndDropPythonGILBlock
	}
  }
  


  void QTRecSelector::setselected(bool newselected)
  {
    {
	SNDE_BeginDropPythonGILBlock
    selected=newselected;
    if (selected) {
      setPalette(selectedpalette);
      RadioButton->setPalette(basepalette);
    } else {
      setPalette(basepalette);
    }
    setcolor(reccolor);
    SNDE_EndDropPythonGILBlock
	}
  }
  
  bool QTRecSelector::eventFilter(QObject *object,QEvent *event)
  {
    {
	SNDE_BeginDropPythonGILBlock
    if (event->type()==QEvent::FocusIn) {
      //fprintf(stderr,"FocusIn\n");
      QFocusEvent* focusEvent = static_cast<QFocusEvent*>(event);

      if (object==RadioButton && focusEvent->reason() != Qt::FocusReason::OtherFocusReason) {
	Viewer->set_selected(this);
      }
    }
    if (event->type()==QEvent::KeyRelease || event->type()==QEvent::KeyPress) {
      QKeyEvent *key = dynamic_cast<QKeyEvent *>(event);
      assert(key); // dynamic_cast should have passed!
      switch(key->key()) {
      case Qt::Key_Left:
	if (event->type()==QEvent::KeyPress) {
	  Viewer->posmgr->HorizZoomOut(false);
	}
	return true;
	
      case Qt::Key_Right:
	if (event->type()==QEvent::KeyPress) {
	  Viewer->posmgr->HorizZoomIn(false);
	}
	
	// else fprintf(stderr,"Release right\n");
	return true;


      case '.':
    if (event->type() == QEvent::KeyPress) {
      Viewer->NextFrame(false);
    }

    return true;


      case ',':
    if (event->type() == QEvent::KeyPress) {
      Viewer->PreviousFrame(false);
    }
    
    return true;
	
	
      case Qt::Key_Down:
	if (event->type()==QEvent::KeyPress) {
	  if (Viewer->posmgr->selected_channel) {
	    std::unique_lock<std::mutex> chanadmin(Viewer->posmgr->selected_channel->admin);
	    if (Viewer->posmgr->selected_channel->render_mode == SNDE_DCRM_IMAGE || Viewer->posmgr->selected_channel->render_mode == SNDE_DCRM_GEOMETRY) {
	      // image or geometry data... decrease contrast instead
	      chanadmin.unlock();
	      Viewer->LessContrast(false);
	      
	    } else /* if (Viewer->posmgr->selected_channel->render_mode == SNDE_DCRM_WAVEFORM)*/ {
	      chanadmin.unlock();
	      Viewer->posmgr->VertZoomOut(false);
	    }
	  }
	}
	return true;
	
      case Qt::Key_Up:
	if (event->type()==QEvent::KeyPress) {
	  
	  if (Viewer->posmgr->selected_channel) {
	    std::unique_lock<std::mutex> chanadmin(Viewer->posmgr->selected_channel->admin);
	    if (Viewer->posmgr->selected_channel->render_mode == SNDE_DCRM_IMAGE || Viewer->posmgr->selected_channel->render_mode == SNDE_DCRM_GEOMETRY) {
	      // image or geometry data... increase contrast instead
	      chanadmin.unlock();
	      Viewer->MoreContrast(false);
	    } else /* if (Viewer->posmgr->selected_channel->render_mode == SNDE_DCRM_WAVEFORM)*/ {	      
	      chanadmin.unlock();
	      Viewer->posmgr->VertZoomIn(false);
	    }
	  }
	  
	}
	return true;
	
      case Qt::Key_Home:
	if (event->type()==QEvent::KeyPress) {
	  Viewer->posmgr->HorizSliderActionTriggered(QAbstractSlider::SliderSingleStepSub);
	}
	return true;
	
      case Qt::Key_End:
	if (event->type()==QEvent::KeyPress) {
	  Viewer->posmgr->HorizSliderActionTriggered(QAbstractSlider::SliderSingleStepAdd);
	}
	return true;
	
      case Qt::Key_PageUp:
	if (event->type()==QEvent::KeyPress) {
	  Viewer->posmgr->VertSliderActionTriggered(QAbstractSlider::SliderSingleStepSub);
	}
	return true;
	
      case Qt::Key_PageDown:
	if (event->type()==QEvent::KeyPress) {
	  Viewer->posmgr->VertSliderActionTriggered(QAbstractSlider::SliderSingleStepAdd);
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

      case 'c':
      case 'C':
	if (event->type()==QEvent::KeyPress) {
	  Viewer->RotateColormap(false);
	}
	return true;

      case 'o':
      case'O':
          if (event->type() == QEvent::KeyPress) {
              Viewer->SetOffsetToMean(false);
          }
          return true;
	
      default:
	return QFrame::eventFilter(object,event);
	
      }
    }

    
    return QFrame::eventFilter(object,event);
    SNDE_EndDropPythonGILBlock
	}
  }
  




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

  std::tuple<int,double> round_to_zoom_digit(double val)
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
  };


    
  qtrec_position_manager::qtrec_position_manager(std::shared_ptr<display_info> display,QAbstractSlider *HorizSlider,QAbstractSlider *VertSlider,QAbstractSlider *HorizZoom,
						 //QToolButton *HorizZoomInButton, QToolButton *HorizZoomOutButton,
						 QAbstractSlider *VertZoom,
						 //QToolButton *VertZoomInButton,QToolButton *VertZoomOutButton
						 QTRecViewer *Parent_Viewer,
						 QObject *Parent
						 ) :
    QObject(Parent),
    display(display),
    HorizSlider(HorizSlider),
    VertSlider(VertSlider),
    HorizZoom(HorizZoom),
    //HorizZoomInButton(HorizZoomInButton),
    //HorizZoomOutButton(HorizZoomOutButton),
    VertZoom(VertZoom),
    //VertZoomInButton(VertZoomInButton),
    //VertZoomOutButton(VertZoomOutButton)
    Parent_Viewer(Parent_Viewer)
  {
    {
	SNDE_BeginDropPythonGILBlock
    power=100.0;
    nsteps=100;
    nzoomsteps=43; // should be multiple of 6+1, e.g. 2*3*7=42, add 1 -> 43
    
    
    assert((nzoomsteps-1) % 6 == 0); // enforce multiple of 6+1
    
    HorizSlider->setRange(0,nsteps-1);
    VertSlider->setRange(0,nsteps-1);
    
    VertZoom->setRange(0,nzoomsteps-1);
    HorizZoom->setRange(0,nzoomsteps-1);
    SNDE_EndDropPythonGILBlock
	}
  }
  
  
  std::tuple<double,bool> qtrec_position_manager::GetHorizScale()
  {
    {
	SNDE_BeginDropPythonGILBlock
    double horizscale = 2.0/display->horizontal_divisions;
    bool horizpixelflag=false;
    bool success=false;
    
    if (selected_channel) {
      if (selected_channel->render_mode != SNDE_DCRM_GEOMETRY) {
	std::shared_ptr<display_axis> a = display->GetFirstAxis(selected_channel->FullName);
	std::lock_guard<std::mutex> axislock(a->unit->admin);
	horizscale = a->unit->scale;
	horizpixelflag = a->unit->pixelflag;
      } else {
	// geometry -- just clone the vertical scale
	std::tie(success,horizscale) = display->GetRenderScale(selected_channel);
      }
    }
    
    return std::make_tuple(horizscale,horizpixelflag);
    SNDE_EndDropPythonGILBlock
	}
  }
  
  std::tuple<double,bool> qtrec_position_manager::GetVertScale()
  {
    {
	SNDE_BeginDropPythonGILBlock
    double vertscale=0.0;
    bool success=false;
    bool vertpixelflag=false;
    if (selected_channel) {
      if (selected_channel->render_mode != SNDE_DCRM_GEOMETRY) {
	std::tie(success,vertscale,vertpixelflag) = display->GetVertScale(selected_channel);
      } else {
	std::tie(success,vertscale) = display->GetRenderScale(selected_channel);

      }
    }
    
    if (!success) {
      std::lock_guard<std::mutex> adminlock(display->admin);
      
      vertscale = 2.0/display->vertical_divisions;
      vertpixelflag=false;
    }
    
    
    return std::make_tuple(vertscale,vertpixelflag);
    SNDE_EndDropPythonGILBlock
	}
  }
  



  void qtrec_position_manager::SetHorizScale(double horizscale,bool horizpixelflag)
  {
    {
	SNDE_BeginDropPythonGILBlock
    snde_debug(SNDE_DC_VIEWER,"SetHorizScale %.2g",horizscale);
    assert(horizscale > 0.0);
    if (selected_channel) {
      if (selected_channel->render_mode != SNDE_DCRM_GEOMETRY) {
	std::shared_ptr<display_axis> a = display->GetFirstAxis(selected_channel->FullName);
	{
	  std::lock_guard<std::mutex> adminlock(a->unit->admin);
	  a->unit->scale = horizscale;
	  a->unit->pixelflag = horizpixelflag;
	}
      } else {
	display->SetRenderScale(selected_channel,horizscale,horizpixelflag);
      }
      //selected_channel->mark_as_dirty();
    }
    SNDE_EndDropPythonGILBlock
	}
  }

  void qtrec_position_manager::SetVertScale(double vertscale,bool vertpixelflag)
  {
    {
	SNDE_BeginDropPythonGILBlock
    snde_debug(SNDE_DC_VIEWER,"SetVertScale()");
    if (selected_channel) {
      snde_debug(SNDE_DC_VIEWER,"SetVertScale(); selected_channel");
      if (selected_channel->render_mode != SNDE_DCRM_GEOMETRY) {
	display->SetVertScale(selected_channel,vertscale,vertpixelflag);
      } else {
	display->SetRenderScale(selected_channel,vertscale,vertpixelflag);

      }
    }
    SNDE_EndDropPythonGILBlock
	}
  }


  double qtrec_position_manager::GetScaleFromZoomPos(int zoompos,bool pixelflag)     
  {
    {
	SNDE_BeginDropPythonGILBlock
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
    SNDE_EndDropPythonGILBlock
	}
  }
  

  int qtrec_position_manager::GetZoomPosFromScale(double scale, bool pixelflag)
  {
    {
	SNDE_BeginDropPythonGILBlock
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

    int retval = nzoomsteps-1 - forwardsliderpos; // return properly REVERSED sliderpos

    if (retval >= nzoomsteps) {
      retval = nzoomsteps-1;
    }
    if (retval < 0) {
      retval = 0;
    }
    return retval;
    SNDE_EndDropPythonGILBlock
	}
  }



  std::tuple<double,double,double> qtrec_position_manager::GetHorizEdges()
  {
    {
	SNDE_BeginDropPythonGILBlock
    double LeftEdge = -1.0;
    double RightEdge = 1.0;
    double horizunitsperdiv = (RightEdge-LeftEdge)/display->horizontal_divisions;
    
    if (selected_channel) {
      std::shared_ptr<display_axis> a = display->GetFirstAxis(selected_channel->FullName);
      
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
    SNDE_EndDropPythonGILBlock
	}
  }


  std::tuple<double,double,double> qtrec_position_manager::GetVertEdges()
  {
    {
	SNDE_BeginDropPythonGILBlock
    double BottomEdge = -1.0;
    double TopEdge = 1.0;
    double vertunitsperdiv = (TopEdge-BottomEdge)/display->horizontal_divisions;
    
    if (selected_channel) {
      vertunitsperdiv = display->GetVertUnitsPerDiv(selected_channel);

      int render_mode;
      {
	std::lock_guard<std::mutex> selchan_admin(selected_channel->admin);

	render_mode = selected_channel->render_mode;
      }

      std::lock_guard<std::mutex> adminlock(selected_channel->admin);
      if (selected_channel->VertZoomAroundAxis || render_mode == SNDE_DCRM_GEOMETRY) {
	BottomEdge=-selected_channel->Position*vertunitsperdiv-vertunitsperdiv*display->vertical_divisions/2;
	TopEdge=-selected_channel->Position*vertunitsperdiv+vertunitsperdiv*display->vertical_divisions/2;	
      } else {
	BottomEdge=selected_channel->VertCenterCoord-vertunitsperdiv*display->vertical_divisions/2;
	TopEdge=selected_channel->VertCenterCoord+vertunitsperdiv*display->vertical_divisions/2;	
	
      }
    }
    return std::make_tuple(BottomEdge,TopEdge,vertunitsperdiv);
    SNDE_EndDropPythonGILBlock
	}
  }
  

  void qtrec_position_manager::trigger()
  {
    {
	SNDE_BeginDropPythonGILBlock

    // TODO -- Potential Bug Issue !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // This method should be moved into QtRecViewer proper because it manipulates
    // QWidget objects, which needs to be done from the main thread.  It should
    // use a signal instead and this code be in a slot of QtRecViewer.

    // We should also document the thread semantics better too.

    double LeftEdgeRec,RightEdgeRec,horizunitsperdiv;
    double BottomEdgeRec,TopEdgeRec,vertunitsperdiv;
    
    if (!selected_channel) {
      snde_debug(SNDE_DC_VIEWER,"Emitting Newposition()");
      Parent_Viewer->OSGWidget->set_selected_channel("");
      emit NewPosition(); // blank out any currently selected channel
      
      return; 
    }

    Parent_Viewer->OSGWidget->set_selected_channel(selected_channel->FullName);

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
    snde_debug(SNDE_DC_VIEWER,"Set vert Zoom sliderpos: %d",vert_zoom_pos);
    VertZoom->setSliderPosition(vert_zoom_pos);
    
    
    snde_debug(SNDE_DC_VIEWER,"Emitting Newposition()");
    emit NewPosition();
    SNDE_EndDropPythonGILBlock
	}
  }

  void qtrec_position_manager::set_selected(std::shared_ptr<display_channel> chan)
  {
    {
	SNDE_BeginDropPythonGILBlock
    selected_channel=chan;
    
    trigger();
    SNDE_EndDropPythonGILBlock
	}
  }
  
  void qtrec_position_manager::HorizSliderActionTriggered(int action)
  {
    {
	SNDE_BeginDropPythonGILBlock
    //double HorizPosn = HorizSlider->sliderPosition()-(nsteps-1)/2.0;
    double HorizPosn = (nsteps-1)/2.0 - HorizSlider->sliderPosition();
    
    double CenterCoord=0.0;
    double horizunitsperdiv=1.0;
      
    std::shared_ptr<display_axis> a = nullptr;
    if (selected_channel) {
      int render_mode;
      {
	std::lock_guard<std::mutex> selchan_admin(selected_channel->admin);

	render_mode = selected_channel->render_mode;
      }
      snde_debug(SNDE_DC_VIEWER,"HSAT: render_mode=%d",render_mode);
      if (render_mode == SNDE_DCRM_IMAGE || render_mode == SNDE_DCRM_WAVEFORM || render_mode == SNDE_DCRM_SCALAR) {

	{
	  std::lock_guard<std::mutex> adminlock(selected_channel->admin);
	}
	a = display->GetFirstAxis(selected_channel->FullName);
	{
	  std::lock_guard<std::mutex> adminlock(a->unit->admin);
	  horizunitsperdiv = a->unit->scale;
	  if (a->unit->pixelflag) horizunitsperdiv *= display->pixelsperdiv;
	}
	
	{
	  std::lock_guard<std::mutex> adminlock(a->admin);
	  CenterCoord = a->CenterCoord;
	}
      
	
	switch(action) {
	case QAbstractSlider::SliderSingleStepAdd:
	  if (a) {
	    std::lock_guard<std::mutex> adminlock(a->admin);
	    a->CenterCoord-=horizunitsperdiv;
	  }
	  
	  break;
	case QAbstractSlider::SliderSingleStepSub:
	  if (a) {
	    std::lock_guard<std::mutex> adminlock(a->admin);
	    a->CenterCoord+=horizunitsperdiv;
	  }
	  break;
	  
	case QAbstractSlider::SliderPageStepAdd:
	  if (a) {
	    std::lock_guard<std::mutex> adminlock(a->admin);
	    a->CenterCoord+=horizunitsperdiv*display->horizontal_divisions/2.0;
	  }
	  break;
	case QAbstractSlider::SliderPageStepSub:
	  if (a) {
	    std::lock_guard<std::mutex> adminlock(a->admin);
	    a->CenterCoord-=horizunitsperdiv*display->horizontal_divisions/2.0;
	  }	
	  break;
	  
	case QAbstractSlider::SliderMove:
	  if (a) {
	    std::lock_guard<std::mutex> adminlock(a->admin);
	    double LeftEdgeRec=-1.0;
	    if (HorizPosn < 0.0) {
	      LeftEdgeRec = -log(1.0 - fabs(HorizPosn)/((nsteps-1)/2.0 + 1.0) )*((nsteps-1)/2+1)*horizunitsperdiv;
	    } else {
	      LeftEdgeRec = log(1.0 - HorizPosn/((nsteps-1)/2.0 + 1.0) )*((nsteps-1)/2+1)*horizunitsperdiv;
	      
	    }
	    a->CenterCoord = (LeftEdgeRec + horizunitsperdiv*display->horizontal_divisions/2);
	    //fprintf(stderr,"HorizSliderMove: Setting CenterCoord to %f\n",a->CenterCoord);
	  }
	  
	  break;
	}
      } else if (render_mode == SNDE_DCRM_GEOMETRY) {
	switch(action) {
	case QAbstractSlider::SliderSingleStepAdd:
	  {
	    std::lock_guard<std::mutex> adminlock(selected_channel->admin);
	    selected_channel->HorizPosition++;
	  }
	  break;
	  
	case QAbstractSlider::SliderSingleStepSub:
	  {
	    std::lock_guard<std::mutex> adminlock(selected_channel->admin);
	    selected_channel->HorizPosition--;
	  }
	  break;
      
	case QAbstractSlider::SliderPageStepAdd:
	  {
	    size_t horizontal_divisions;
	    {
	      std::lock_guard<std::mutex> adminlock(display->admin);
	      horizontal_divisions = display->horizontal_divisions;
	    }
	    std::lock_guard<std::mutex> adminlock(selected_channel->admin);
	    selected_channel->HorizPosition-=horizontal_divisions/2.0;
	  }
	  break;
	  
	case QAbstractSlider::SliderPageStepSub:
	  {
	    size_t horizontal_divisions;
	    {
	      std::lock_guard<std::mutex> adminlock(display->admin);
	      horizontal_divisions = display->horizontal_divisions;
	    }
	    std::lock_guard<std::mutex> adminlock(selected_channel->admin);
	    selected_channel->HorizPosition+=horizontal_divisions/2.0;
	  }	
	  break;
	  
	case QAbstractSlider::SliderMove:
	  {
	    double LeftEdgeRec=1.0;	  
	    size_t horizontal_divisions;
	    {
	      std::lock_guard<std::mutex> adminlock(display->admin);
	      horizontal_divisions = display->horizontal_divisions;
	    }
	    
	    if (HorizPosn < 0.0) {
	      LeftEdgeRec = -log(1.0 - fabs(HorizPosn)/((nsteps-1)/2.0 + 1.0) )*((nsteps-1)/2+1)*horizunitsperdiv;
	    } else {
	      LeftEdgeRec = log(1.0 - HorizPosn/((nsteps-1)/2.0 + 1.0) )*((nsteps-1)/2+1)*horizunitsperdiv;
	    }
	    
	    std::lock_guard<std::mutex> adminlock(selected_channel->admin);
	    selected_channel->HorizPosition = -LeftEdgeRec/horizunitsperdiv + horizontal_divisions/2;
	    snde_debug(SNDE_DC_VIEWER,"HorizPosition = %f horiz units",selected_channel->HorizPosition);
	  }	
	  break;
	}
      }
    }
    //if (a) {
    //fprintf(stderr,"HorizCenterCoord=%f\n",a->CenterCoord);
    //}
    trigger();
    SNDE_EndDropPythonGILBlock
	}
  }
  
  void qtrec_position_manager::VertSliderActionTriggered(int action)
  {
    {
	SNDE_BeginDropPythonGILBlock
    double VertPosn = VertSlider->sliderPosition() - (nsteps-1)/2.0;
    
    double CenterCoord=0.0;
    double vertunitsperdiv=1.0;
    int render_mode=SNDE_DCRM_INVALID;

    std::shared_ptr<display_axis> a = nullptr;
    if (selected_channel) {
      vertunitsperdiv = display->GetVertUnitsPerDiv(selected_channel);
      
      std::lock_guard<std::mutex> selchan_admin(selected_channel->admin);
      
      render_mode = selected_channel->render_mode;
    
      
    }
    
    switch(action) {
    case QAbstractSlider::SliderSingleStepAdd:
      if (selected_channel) {
	std::lock_guard<std::mutex> adminlock(selected_channel->admin);
	if (selected_channel->VertZoomAroundAxis || render_mode == SNDE_DCRM_GEOMETRY) {
	  selected_channel->Position--;
	} else {
	  selected_channel->VertCenterCoord += vertunitsperdiv;
	}
      }
      snde_debug(SNDE_DC_VIEWER,"VertPosition = %f",selected_channel->Position);
      break;
    case QAbstractSlider::SliderSingleStepSub:
      if (selected_channel) {
	std::lock_guard<std::mutex> adminlock(selected_channel->admin);
	if (selected_channel->VertZoomAroundAxis || render_mode == SNDE_DCRM_GEOMETRY) {
	  selected_channel->Position++;
	} else {
	  selected_channel->VertCenterCoord -= vertunitsperdiv;
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
	if (selected_channel->VertZoomAroundAxis || render_mode == SNDE_DCRM_GEOMETRY) {
	  selected_channel->Position+=vertical_divisions/2.0;
	  } else {
	  selected_channel->VertCenterCoord -= vertunitsperdiv*vertical_divisions/2.0;	    
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
	if (selected_channel->VertZoomAroundAxis || render_mode == SNDE_DCRM_GEOMETRY) {
	  selected_channel->Position-=vertical_divisions/2.0;
	} else {
	  selected_channel->VertCenterCoord += vertunitsperdiv*vertical_divisions/2.0;	    
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
	
	if (selected_channel->VertZoomAroundAxis || render_mode == SNDE_DCRM_GEOMETRY) {
	  std::lock_guard<std::mutex> adminlock(selected_channel->admin);
	  selected_channel->Position = -BottomEdgeRec/vertunitsperdiv + vertical_divisions/2;
	  //snde_warning("Position = %f vert units",selected_channel->Position);
	} else {
	  std::lock_guard<std::mutex> adminlock(selected_channel->admin);
	  selected_channel->VertCenterCoord = BottomEdgeRec + vertunitsperdiv*vertical_divisions/2;
	  //fprintf(stderr,"BottomEdgeRec=%f; VertCenterCoord=%f\n",BottomEdgeRec,selected_channel->VertCenterCoord);
	}
	
	
      }
      break;
    }
    //selected_channel->mark_as_dirty();
    trigger();
    SNDE_EndDropPythonGILBlock
	}
    
  }

  void qtrec_position_manager::HorizZoomActionTriggered(int action)
  {
    {
	SNDE_BeginDropPythonGILBlock
    double horizscale;
    bool horizpixelflag;
    std::tie(horizscale,horizpixelflag) = GetHorizScale();
    
    int horiz_zoom_pos = GetZoomPosFromScale(horizscale, horizpixelflag);
    
    double rounded_scale = GetScaleFromZoomPos(horiz_zoom_pos,horizpixelflag);
    
    switch(action) {
    case QAbstractSlider::SliderSingleStepAdd:
    case QAbstractSlider::SliderPageStepAdd:
      
      if (rounded_scale > horizscale*1.01) {
	// round up
	snde_debug(SNDE_DC_VIEWER,"Zooming in, selected channel=\"%s\" round up; rounded_scale = %f, horizscale*1.01=%f",selected_channel->FullName.c_str(),rounded_scale,horizscale*1.01);
	SetHorizScale(rounded_scale,horizpixelflag);	  
      } else {
	// Step up
	if (horiz_zoom_pos+1 < nzoomsteps) {
	  snde_debug(SNDE_DC_VIEWER,"Zooming in, selected channel=\"%s\" step up",selected_channel->FullName.c_str());
	  double new_scale = GetScaleFromZoomPos(horiz_zoom_pos+1,horizpixelflag);
	  SetHorizScale(new_scale,horizpixelflag);
	}
      }
      break;
      
    case QAbstractSlider::SliderSingleStepSub:
    case QAbstractSlider::SliderPageStepSub:
      if (rounded_scale < horizscale*.99) {
	// round down
	snde_debug(SNDE_DC_VIEWER,"horiz slider: round down");
	SetHorizScale(rounded_scale,horizpixelflag);	  
      } else {
	// Step down
	snde_debug(SNDE_DC_VIEWER,"horiz slider: step down");
	if (horiz_zoom_pos > 0) {
	  double new_scale = GetScaleFromZoomPos(horiz_zoom_pos-1,horizpixelflag);
	  SetHorizScale(new_scale,horizpixelflag);
	}
      }
      break;
      
    case QAbstractSlider::SliderMove:
      snde_debug(SNDE_DC_VIEWER,"Got Horiz Zoom slidermove: %d\n",HorizZoom->sliderPosition());
      
      double HorizZoomPosn = GetScaleFromZoomPos(HorizZoom->sliderPosition(),horizpixelflag);
      SetHorizScale(HorizZoomPosn,horizpixelflag);	  
      
      break;
    }
    trigger();
    SNDE_EndDropPythonGILBlock
	}
  }



  void qtrec_position_manager::VertZoomActionTriggered(int action)
  {
    {
	SNDE_BeginDropPythonGILBlock
    double vertscale;
    bool vertpixelflag;
    std::tie(vertscale,vertpixelflag) = GetVertScale();
    
    int vert_zoom_pos = GetZoomPosFromScale(vertscale, vertpixelflag);
    
    double rounded_scale = GetScaleFromZoomPos(vert_zoom_pos,vertpixelflag);
    
    switch(action) {
    case QAbstractSlider::SliderSingleStepAdd:
    case QAbstractSlider::SliderPageStepAdd:
      
      if (rounded_scale > vertscale*1.01) {
	// round up
	snde_debug(SNDE_DC_VIEWER,"vert slider: round up");
	SetVertScale(rounded_scale,vertpixelflag);	  
      } else {
	// Step up
	snde_debug(SNDE_DC_VIEWER,"vert slider: step up");
	if (vert_zoom_pos+1 < nzoomsteps) {
	  double new_scale = GetScaleFromZoomPos(vert_zoom_pos+1,vertpixelflag);
	  SetVertScale(new_scale,vertpixelflag);
	}  
      }
      break;
      
    case QAbstractSlider::SliderSingleStepSub:
    case QAbstractSlider::SliderPageStepSub:
      if (rounded_scale < vertscale*.99) {
	// round down
	snde_debug(SNDE_DC_VIEWER,"vert slider: round down: rounded=%f; vert=%f",rounded_scale,vertscale);
	SetVertScale(rounded_scale,vertpixelflag);	  
	std::tie(vertscale,vertpixelflag) = GetVertScale();
	snde_debug(SNDE_DC_VIEWER,"vert slider_post_round: vert=%f",vertscale);
      } else {
	// Step down
	snde_debug(SNDE_DC_VIEWER,"vert slider: step down");
	if (vert_zoom_pos > 0) {
	  double new_scale = GetScaleFromZoomPos(vert_zoom_pos-1,vertpixelflag);
	  SetVertScale(new_scale,vertpixelflag);
	}
      }
      break;
      
    case QAbstractSlider::SliderMove:
      snde_debug(SNDE_DC_VIEWER,"Got Vert Zoom slidermove: %d",VertZoom->sliderPosition());
      double VertZoomPosn = GetScaleFromZoomPos(VertZoom->sliderPosition(),vertpixelflag);
      SetVertScale(VertZoomPosn,vertpixelflag);	  
      
      break;
    }
    trigger();
    SNDE_EndDropPythonGILBlock
	}
  }
  
  void qtrec_position_manager::VertZoomIn(bool)
  {
    {
	SNDE_BeginDropPythonGILBlock
    //fprintf(stderr,"VertZoomIn()\n");
    VertZoomActionTriggered(QAbstractSlider::SliderSingleStepAdd);      
    SNDE_EndDropPythonGILBlock
	}
  }

  void qtrec_position_manager::VertZoomOut(bool)
  {
    {
	SNDE_BeginDropPythonGILBlock
    VertZoomActionTriggered(QAbstractSlider::SliderSingleStepSub);      
    SNDE_EndDropPythonGILBlock
	}
  }


  void qtrec_position_manager::HorizZoomIn(bool)
  {
    {
	SNDE_BeginDropPythonGILBlock
    snde_debug(SNDE_DC_VIEWER,"HorizZoomIn()");
    HorizZoomActionTriggered(QAbstractSlider::SliderSingleStepAdd);

    if (selected_channel) {
      int render_mode;
      {
	std::lock_guard<std::mutex> selchan_admin(selected_channel->admin);
	render_mode = selected_channel->render_mode;
      }
    }
    SNDE_EndDropPythonGILBlock
	}
  }


  void qtrec_position_manager::HorizZoomOut(bool)
  {
    {
	SNDE_BeginDropPythonGILBlock
    HorizZoomActionTriggered(QAbstractSlider::SliderSingleStepSub);      
    if (selected_channel) {
      int render_mode;
      {
	std::lock_guard<std::mutex> selchan_admin(selected_channel->admin);
	render_mode = selected_channel->render_mode;
      }

      if (render_mode == SNDE_DCRM_IMAGE) {
	// for images we vertically zoom along with horizontal (at least for now)
	// unless the vertical and horizontal axes are already the same
	std::shared_ptr<display_axis> a = display->GetFirstAxis(selected_channel->FullName);
	std::shared_ptr<display_axis> b = display->GetSecondAxis(selected_channel->FullName);

	//if (a != b) { // don't actually want to trigger other because we get the vert scale from the horiz scale anyway
	//  VertZoomActionTriggered(QAbstractSlider::SliderSingleStepSub);	  
	//}
	
      }
    }
    SNDE_EndDropPythonGILBlock
	}
  }


};

