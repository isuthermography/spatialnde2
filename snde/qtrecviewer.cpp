//#include <osgDB/WriteFile>

#include <QFile>
#include <QUiLoader>

#include "snde/qtrecviewer_support.hpp"
#include "snde/qtrecviewer.hpp"
#include "snde/qt_osg_compositor.hpp"

#include "snde/python_support.hpp"

namespace snde {
  
  QTRecViewer::QTRecViewer(std::shared_ptr<recdatabase> recdb,QWidget *parent /*=0*/)
    : QWidget(parent),
      recdb(recdb)
  {

    // unless OSG_NOTIFY_LEVEL or OSGNOTIFYLEVEL is set
    // raise the OpenSceneGraph notify level to WARN
    // to avoid the annoying State::reset() messages coming at
    // NOTICE level
    char *OSG_NOTIFY_LEVEL_value = std::getenv("OSG_NOTIFY_LEVEL");
    char *OSGNOTIFYLEVEL_value = std::getenv("OSGNOTIFYLEVEL");

    if ((!OSG_NOTIFY_LEVEL_value || !strlen(OSG_NOTIFY_LEVEL_value)) &&
	(!OSGNOTIFYLEVEL_value || !strlen(OSGNOTIFYLEVEL_value))) {
      // environment variable not set
      osg::setNotifyLevel(osg::NotifySeverity::WARN);
    }
    
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
    
    OSGWidget=QSharedPointer<qt_osg_compositor>(new qt_osg_compositor(recdb,display,new osgViewer::Viewer(),
								      true, // threaded; try true
								      true, // enable_threaded_opengl
								      false, // enable_shaders
								      this,this),&QObject::deleteLater); // Note that we have a parent widget set on an object we are reference counting with QSharedPointer for ownership instead so it can be safely referenced from other threads. This is inherently dangerous. Our escape is that we explicitly deparent OSGWidget in our destructor, so the parenting is irrelevant. 
    
    viewerGridLayout->addWidget(OSGWidget.get(),1,0);
    // Even if we didn't parent OSGWidget above, addWidget() will have automatically parented OSGWidget, which we don't want, so we explicitly set the parent back to nullptr
    // in our destructor so that deletion is really managed by the qsharedpointer
  
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
    QToolButton *ColormapButton = DesignerTree->findChild<QToolButton*>("ColormapButton");
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
							"QScrollBar::sub-line:vertical {\n"
							"   height: 20px;\n"
							"   border: 2px outset black;\n"
							"   subcontrol-position: bottom;\n"
							"   subcontrol-origin: margin;\n"
							"   position: absolute;\n"
							"   bottom: 30px;\n"
							"}\n"
							"QScrollBar::add-line:vertical {\n"
							"   height: 20px;\n"
							"   border: 2px outset black;\n"
							"   subcontrol-position: bottom;\n"
							"   subcontrol-origin: margin;\n"
							"}\n"
							"QScrollBar::down-arrow:vertical {\n"
							"   width: 18px;\n"
							"   height: 18px;\n"
							"   color: black;\n"
							//"   background: pink;\n"
							"   image: url(\":/darrow.png\");\n"
							"}\n"
							"QScrollBar::up-arrow:vertical {\n"
							"   width: 18px;\n"
							"   height: 18px;\n"
							"   color: black;\n"
							//"   background: pink;\n"
							"   image: url(\":/uarrow.png\");\n"
							"}\n"
						     ));

    posmgr = new qtrec_position_manager(display,HorizSlider,VertSlider,HorizZoom,VertZoom,this,this);

    //OSGWidget->start();  // we can't call start() ourselves... we have to wait for InitializeGL()

    ready_globalrev_quicknotify = std::make_shared<std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<globalrevision>)>>( [ this ](std::shared_ptr<recdatabase> recdb,std::shared_ptr<globalrevision>) {
      snde_debug(SNDE_DC_VIEWER,"Invoking qtrecviewer update_rec_list slot because of globalrev update");
      
      QMetaObject::invokeMethod(this,"update_rec_list",Qt::QueuedConnection); // trigger update_rec_list slot in QT main loop
    });
    
    recdb->register_ready_globalrev_quicknotifies_called_recdb_locked(ready_globalrev_quicknotify);

    bool success=false;
    
    success=QObject::connect(HorizSlider,SIGNAL(actionTriggered(int)),
			     posmgr, SLOT(HorizSliderActionTriggered(int)));
    assert(success);

    success=QObject::connect(VertSlider,SIGNAL(actionTriggered(int)),
			     posmgr, SLOT(VertSliderActionTriggered(int)));
    assert(success);

    
    success = QObject::connect(HorizZoom,SIGNAL(actionTriggered(int)),
			       posmgr, SLOT(HorizZoomActionTriggered(int)));
    assert(success);

    success = QObject::connect(VertZoom,SIGNAL(actionTriggered(int)),
			       posmgr, SLOT(VertZoomActionTriggered(int)));
    assert(success);
    
    success = QObject::connect(VertZoomInButton,SIGNAL(clicked(bool)),
		     posmgr, SLOT(VertZoomIn(bool)));
    assert(success);

    success = QObject::connect(HorizZoomInButton,SIGNAL(clicked(bool)),
		     posmgr, SLOT(HorizZoomIn(bool)));
    assert(success);

    success = QObject::connect(VertZoomOutButton,SIGNAL(clicked(bool)),
		     posmgr, SLOT(VertZoomOut(bool)));
    assert(success);

    success = QObject::connect(HorizZoomOutButton,SIGNAL(clicked(bool)),
		     posmgr, SLOT(HorizZoomOut(bool)));
    assert(success);
    
    success = QObject::connect(DarkenButton,SIGNAL(clicked(bool)),
		     this, SLOT(Darken(bool)));
    assert(success);

    success = QObject::connect(ResetIntensityButton,SIGNAL(clicked(bool)),
		     this, SLOT(ResetIntensity(bool)));
    assert(success);


    success = QObject::connect(BrightenButton,SIGNAL(clicked(bool)),
		     this, SLOT(Brighten(bool)));
    assert(success);

    success = QObject::connect(LessContrastButton,SIGNAL(clicked(bool)),
		     this, SLOT(LessContrast(bool)));
    assert(success);
    success = QObject::connect(ColormapButton,SIGNAL(clicked(bool)),
		     this, SLOT(RotateColormap(bool)));
    assert(success);

    success = QObject::connect(MoreContrastButton,SIGNAL(clicked(bool)),
		     this, SLOT(MoreContrast(bool)));
    assert(success);
    
    
    
    success = QObject::connect(posmgr,SIGNAL(NewPosition()),
			       OSGWidget.get(),SLOT(rerender()));
    assert(success);
    
    success = QObject::connect(this,SIGNAL(NeedRedraw()),
			       OSGWidget.get(),SLOT(rerender()));
    assert(success);
    
    
    success = QObject::connect(posmgr,SIGNAL(NewPosition()),
			       this,SLOT(UpdateViewerStatus()));
    assert(success);
    
    success = QObject::connect(this,SIGNAL(NewGlobalRev()),
			       OSGWidget.get(),SLOT(rerender()));
    assert(success);
    
    success = QObject::connect(this,SIGNAL(NewGlobalRev()),
			       this,SLOT(UpdateViewerStatus()));    
    assert(success);
    
    // eventfilter monitors for keypresses
    //      installEventFilter(this);

    
    update_rec_list();
    //rendering_revman->Start_Transaction();
    //update_renderer();
    //rendering_revman->End_Transaction();
    
    posmgr->trigger();
    
  }

  
  QTRecViewer::~QTRecViewer()
  {

    // Explicitly deparent OSGWidget so that its lifetime
    // is actually controlled by the QSharedPointer, which
    // will call deleteLater() once it goes out of scope. 
    OSGWidget->setParent(nullptr); 


    std::shared_ptr<recdatabase> recdb_strong = recdb.lock();

    if (recdb_strong) {
      // Most stuff taken care of by QT parenting but...    
      recdb_strong->unregister_ready_globalrev_quicknotifies_called_recdb_locked(ready_globalrev_quicknotify);
    
      ready_globalrev_quicknotify=nullptr;
    }
  }

  std::shared_ptr<display_channel> QTRecViewer::FindDisplayChan(std::string channame) {

	  auto ci_iter = display->channel_info.find(channame);
	  if (ci_iter != display->channel_info.end()) {
		  auto& displaychan = ci_iter->second;

		  if (displaychan->FullName == channame) {
			  //auto selector_iter = Selectors.find(displaychan->FullName);
				//if (selector_iter != Selectors.end() && selector_iter->second==Selector) {
			  return displaychan;
		  }
	  }
	  return nullptr;
  }

  std::shared_ptr<display_channel> QTRecViewer::FindDisplayChan(QTRecSelector *Selector)
  {
	  if (!Selector) return nullptr;
	  return FindDisplayChan(Selector->Name);
  } 
  
  void QTRecViewer::set_selected(QTRecSelector *Selector)
  // assumes Selector already highlighted 
  {
    {
	SNDE_BeginDropPythonGILBlock
    
    std::shared_ptr<display_channel> displaychan=FindDisplayChan(Selector);
   

    Selector->setselected(true);
    //std::shared_ptr<iterablerecrefs> reclist=recdb->reclist();
    
    //for (auto reciter=reclist->begin();reciter != reclist->end();reciter++) {
    //std::shared_ptr<mutableinfostore> infostore=*reciter;
    posmgr->set_selected(displaychan);
    OSGWidget->set_selected_channel(displaychan->FullName);
    selected = displaychan->FullName;
    //}
    
    deselect_other_selectors(Selector);
    
    //UpdateViewerStatus(); // Now taken care of by posmgr->set_selected()'s call to trigger which emits into this slot
    SNDE_EndDropPythonGILBlock
	}
  }


  void QTRecViewer::set_selected(std::string channame) {

    {
	SNDE_BeginDropPythonGILBlock

    auto selector_iter = Selectors.find(channame);
    if (selector_iter != Selectors.end()) {
      auto& selector = selector_iter->second;
      selector->RadioButton->setFocus(Qt::FocusReason::OtherFocusReason); // Workaround required to ensure that focus doesn't change to a radio button when the window regains focus -- this triggers a change of the selection
      set_selected(selector);  
    }
    else {
      throw snde_error("Unable to find channel % s", channame.c_str());
    }

    SNDE_EndDropPythonGILBlock
	}
  }
  
  


  void QTRecViewer::update_rec_list()  
  {
    {
	SNDE_BeginDropPythonGILBlock
    std::shared_ptr<recdatabase> recdb_strong=recdb.lock();

    snde_debug(SNDE_DC_VIEWER,"QTRecViewer executing update_rec_list()");
    
    if (!recdb_strong) return;

    // !!!*** NOTE: osg_compositor allso calls display->update(), but with all three bools false
    // ... is it a problem that both modules are calling display->update??? (and perhaps from different threads!)
    // Answer: It is OK. display->update just does internal housekeeping on the display_info list
    // of channels. The bools just determine which channels are returned. 
    std::vector<std::shared_ptr<display_channel>> currentreclist;
    std::vector<std::shared_ptr<display_channel>> junk;
    std::tie(currentreclist,junk) = display->get_channels(recdb_strong->latest_globalrev() /* actually the latest ready globalrev */,selected,false,false,true,false);
    
    // clear touched flag for all selectors
    for(auto & selector: Selectors) {
      selector.second->touched_during_update=false;
    }
    
    // iterate over rec list
    size_t pos=0;
    for (auto & displaychan: currentreclist) {
      std::lock_guard<std::mutex> displaychanlock(displaychan->admin);
      
      auto selector_iter = Selectors.find(displaychan->FullName);
      if (selector_iter == Selectors.end()) {
	  // create a new selector
	QTRecSelector *NewSel = new QTRecSelector(this,displaychan->FullName,RecColorTable[displaychan->ColorIdx],RecListScrollAreaContent);
	RecListScrollAreaLayout->insertWidget(pos,NewSel);
	Selectors[displaychan->FullName]=NewSel;
	QObject::connect(NewSel->RadioButton,SIGNAL(clicked(bool)),
			 this,SLOT(SelectorClicked(bool)));
	//QObject::connect(NewSel->RadioButton,SIGNAL(toggled(bool)),
	//this,SLOT(SelectorClicked(bool)));
      }
      
      QTRecSelector *Sel = Selectors[displaychan->FullName];
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

    snde_debug(SNDE_DC_VIEWER,"QTRecViewer update_rec_list() emitting NewGlobalRev()");
    emit NewGlobalRev();

    SNDE_EndDropPythonGILBlock
	}
    
  }
  
  void QTRecViewer::deselect_other_selectors(QTRecSelector *Selected)
  {
    {
	SNDE_BeginDropPythonGILBlock
    for (auto & name_sel : Selectors) {
      if (name_sel.second != Selected) {
	name_sel.second->setselected(false);
      }
    }
    SNDE_EndDropPythonGILBlock
	}
  }


  snde_orientation3 QTRecViewer::get_camera_pose(std::string channelpath)
  {
    {
	SNDE_BeginDropPythonGILBlock
    return OSGWidget->get_camera_pose(channelpath);
    SNDE_EndDropPythonGILBlock
	}
  }

  void QTRecViewer::set_camera_pose(std::string channelpath,const snde_orientation3 &newpose)
  {
    {
	SNDE_BeginDropPythonGILBlock
    OSGWidget->set_camera_pose(channelpath,newpose);


    emit NeedRedraw();
    SNDE_EndDropPythonGILBlock
	}
  }

  snde_coord QTRecViewer::get_rotation_center_dist(std::string channelpath)
  {
    {
	SNDE_BeginDropPythonGILBlock
    return OSGWidget->get_rotation_center_dist(channelpath);
    SNDE_EndDropPythonGILBlock
	}
  }

  void QTRecViewer::set_rotation_center_dist(std::string channelpath,snde_coord newcenterdist)
  {
    {
	SNDE_BeginDropPythonGILBlock
    OSGWidget->set_rotation_center_dist(channelpath,newcenterdist);


    emit NeedRedraw();
    SNDE_EndDropPythonGILBlock
	}
  }

  
  void QTRecViewer::UpdateViewerStatus()
  {
    {
	SNDE_BeginDropPythonGILBlock
    double horizscale;
    bool horizpixelflag;
    std::string statusline="";
    bool needjoin=false;

    std::shared_ptr<recdatabase> recdb_strong=recdb.lock();
    std::shared_ptr<ndarray_recording_ref> rec = NULL;
    
    

    //if (!recdb_strong) return;

    if (posmgr->selected_channel) {
      //std::shared_ptr<mutableinfostore> chan_data;
      //chan_data = recdb->lookup(posmgr->selected_channel->FullName);

    // display revision -- Note, this recording ref is used below
    // Also, this isn't necessarily linked to what is actually being displayed on screen
    // This should be reworked so that it is linked to a specific globalrev being actively
    // rendered by the scope

    // Also... Line 509 below... returns a value regardless of whether there is a valid recording.  It probably shouldn't.  Adding a check on line 514 for now.
    if (recdb_strong){
        auto latest_globalrev = recdb_strong->latest_globalrev();
	    rec = latest_globalrev->check_for_recording_ref(posmgr->selected_channel->FullName, 0);
    
		if (recdb_strong && rec) {
			statusline += "Revision: " + std::to_string(rec->rec->info->revision) + "/" + std::to_string(latest_globalrev->globalrev);
			needjoin = true;
		}
    }

      int render_mode;
      bool chan_enabled;
      {
	std::lock_guard<std::mutex> selchan_admin(posmgr->selected_channel->admin);

	render_mode = posmgr->selected_channel->render_mode;
	chan_enabled = posmgr->selected_channel->Enabled;
      }
      
      std::shared_ptr<display_axis> a = display->GetFirstAxis(posmgr->selected_channel->FullName);	
      //std::shared_ptr<mutabledatastore> datastore=std::dynamic_pointer_cast<mutabledatastore>(chan_data);
      
      
      
      if (recdb_strong && rec && a) {
	{
	  std::lock_guard<std::mutex> adminlock(a->unit->admin);
	  horizscale = a->unit->scale;
	  horizpixelflag = a->unit->pixelflag;
	  snde_debug(SNDE_DC_VIEWER,"Horizontal axis: a=%s",a->axis.c_str());
	}
	// Gawdawful C++ floating point formatting
	//std::stringstream inipos;
	//inipos << std::defaultfloat << std::setprecision(6) << a->CenterCoord;
	
	//std::stringstream horizscalestr;
	//horizscalestr << std::defaultfloat << std::setprecision(6) << a->unit->scale;
	  //fprintf(stderr,"unitprint: %s\n",a->unit->unit.print(false).c_str());
	  
	{
	  std::lock_guard<std::mutex> adminlock(a->admin);
	  
      if(needjoin) {
          statusline += " | ";
      }

	  statusline += a->abbrev+"=" + PrintWithSIPrefix(a->CenterCoord,a->unit->unit.print(false),3) + " " + PrintWithSIPrefix(horizscale,a->unit->unit.print(false),3);
	}
	if (horizpixelflag) {
	  statusline += "/px";
	} else {
	  statusline += "/div";
	}
	needjoin=true;
      }
      
      if (render_mode==SNDE_DCRM_WAVEFORM) {
	a = display->GetAmplAxis(posmgr->selected_channel->FullName);
	
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
	    
	    statusline += a->abbrev+"=" + PrintWithSIPrefix(inipos,a->unit->unit.print(false),3) + " " + PrintWithSIPrefix(scalefactor,a->unit->unit.print(false),3);
	    //statusline += a->abbrev+"0=" + inipos.str() + " " + vertscalestr.str() + a->unit->unit.print(false);
	  }
	  if (horizpixelflag) {
	    statusline += "/px";
	  } else {
	    statusline += "/div";
	  }
	  needjoin=true;
	}
      } else if (render_mode == SNDE_DCRM_IMAGE) {
	// ndim > 1
	a=display->GetSecondAxis(posmgr->selected_channel->FullName);
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
	    snde_debug(SNDE_DC_VIEWER,"Image: Vertical axis: a=%s",a->axis.c_str());
	  
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
	  statusline += a->abbrev+"=" + PrintWithSIPrefix(inipos,a->unit->unit.print(false),3) + " " + PrintWithSIPrefix(scalefactor,a->unit->unit.print(false),3);
	  
	  if (horizpixelflag) {
	    statusline += "/px";
	  } else {
	    statusline += "/div";
	  }
	  needjoin=true;
	  
	  
	}



	// Try to Add Third Axis if it Exists

	// There is an obvious race condition issue here where this could change between line 632 and line 636 -- fix this later
	if (rec && rec->ndinfo()->ndim >= 3) {
		double offset2, scale2;
		std::string UnitName;
		std::tie(a,offset2,scale2,UnitName) = display->GetThirdAxis(posmgr->selected_channel->FullName);
		if (a) {
			if (needjoin) {
				statusline += " | ";
			}

			double displayframe;
			{
			  	std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
				displayframe = posmgr->selected_channel->DisplayFrame;
			}

			//std::stringstream vertscalestr;
			//vertscalestr << std::defaultfloat << std::setprecision(6) << scalefactor;

			//statusline += a->abbrev+"0=" + inipos.str() + " " + vertscalestr.str() + a->unit->unit.print(false);
			std::lock_guard<std::mutex> adminlock(a->admin);
			statusline += a->abbrev + "=" + PrintWithSIPrefix(offset2 + scale2 * displayframe, a->unit->unit.print(false), 3);

			needjoin = true;


		}

	}
	
	double scalefactor;
	//std::shared_ptr<mutableinfostore> chan_data;
	{
	  std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
	  //chan_data = posmgr->selected_channel->chan_data;
	  scalefactor=posmgr->selected_channel->Scale;
	}
	
	a=display->GetAmplAxis(posmgr->selected_channel->FullName);
	
	// Only Show Amplitude If It Can Be Adjusted -- Not Colormapping with RGBA Image Directly
	
	if (rec && rec->storage->typenum != SNDE_RTN_SNDE_RGBA) {

		if (a) {
			if (needjoin) {
				statusline += " | ";
			}
			//double intensityunitsperdiv=scalefactor;

			//if (a->unit->pixelflag) vertunitsperdiv*=display->pixelsperdiv;

			std::lock_guard<std::mutex> adminlock(a->admin);
			//statusline += a->abbrev+"0=" + inipos.str() + " " + intscalestr.str() + a->unit->unit.print(false) + "/intensity";
			snde_debug(SNDE_DC_VIEWER, "Image: Amplitude axis: a=%s", a->axis.c_str());

			statusline += a->abbrev + "=" + PrintWithSIPrefix(posmgr->selected_channel->Offset, a->unit->unit.print(false), 3) + " " + PrintWithSIPrefix(scalefactor, a->unit->unit.print(false), 3) + "/intensity";

			needjoin = true;


		}


	}

      } else if (render_mode == SNDE_DCRM_GEOMETRY) {

	double scalefactor;
	//std::shared_ptr<mutableinfostore> chan_data;
	{
	  std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
	  //chan_data = posmgr->selected_channel->chan_data;
	  scalefactor=posmgr->selected_channel->Scale;
	}
	
	a=display->GetAmplAxis(posmgr->selected_channel->FullName);
	
	if (a) {
	  if (needjoin) {
	    statusline += " | ";
	  }
	  //double intensityunitsperdiv=scalefactor;
	  
	  //if (a->unit->pixelflag) vertunitsperdiv*=display->pixelsperdiv;
	  
	  std::lock_guard<std::mutex> adminlock(a->admin);
	  //statusline += a->abbrev+"0=" + inipos.str() + " " + intscalestr.str() + a->unit->unit.print(false) + "/intensity";
	  snde_debug(SNDE_DC_VIEWER,"Image: Amplitude axis: a=%s",a->axis.c_str());

	  statusline += a->abbrev+"=" + PrintWithSIPrefix(posmgr->selected_channel->Offset,a->unit->unit.print(false),3) + " " + PrintWithSIPrefix(scalefactor,a->unit->unit.print(false),3) + "/intensity";
	  
	  needjoin=true;
	  
	  
	}

	
      } else if (render_mode == SNDE_DCRM_SCALAR) {
	statusline = "Scalar Value";
      } else if (render_mode == SNDE_DCRM_PHASEPLANE) {
	a=display->GetAmplAxis(posmgr->selected_channel->FullName);
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
	    snde_debug(SNDE_DC_VIEWER,"Phase plane: Amplitude axis: a=%s",a->axis.c_str());
	  
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
	  statusline += a->abbrev+"=" + PrintWithSIPrefix(inipos,a->unit->unit.print(false),3) + " " + PrintWithSIPrefix(scalefactor,a->unit->unit.print(false),3);
	  
	  if (pixelflag) {
	    statusline += "/px";
	  } else {
	    statusline += "/div";
	  }
	  needjoin=true;
	  
	  
	}
	
	  
	
      }  else {
	if (chan_enabled) {
	  // How about we display something in the status line instead of printing an annoying error message to stderr
	  //snde_warning("qtrecviewer: invalid render_mode: %d on channel %s (0x%llx)",render_mode,posmgr->selected_channel->FullName.c_str(),(unsigned long long)((uintptr_t)posmgr->selected_channel.get()));
	  if (needjoin) {
	    statusline += " | ";
	  }
	  statusline += " Invalid Recording";
	}
      }
      
    
      
      
    }
    
    ViewerStatus->setText(QString::fromStdString(statusline));

    SNDE_EndDropPythonGILBlock
	}
      
  }

  void QTRecViewer::SelectorClicked(bool checked)
  {
    {
	SNDE_BeginDropPythonGILBlock
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
	  //displaychan->mark_as_dirty();
	  emit NeedRedraw();
	}
      }
    }
    
    SNDE_EndDropPythonGILBlock
	}
  }
  

  void QTRecViewer::Darken(bool checked)
  {
    {
	SNDE_BeginDropPythonGILBlock
    if (posmgr->selected_channel) {
      
      int render_mode;
      {
	std::lock_guard<std::mutex> selchan_admin(posmgr->selected_channel->admin);

	render_mode = posmgr->selected_channel->render_mode;
      }
      

      
      std::shared_ptr<display_axis> a=display->GetAmplAxis(posmgr->selected_channel->FullName);
      
	
      if (a && (render_mode == SNDE_DCRM_IMAGE || render_mode == SNDE_DCRM_GEOMETRY)) {
	{
	  std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
	  posmgr->selected_channel->Offset += posmgr->selected_channel->Scale/8.0;
	}
	//posmgr->selected_channel->mark_as_dirty();
	UpdateViewerStatus();
	emit NeedRedraw();
      }
    }

    SNDE_EndDropPythonGILBlock
	}
  }
  
  

  void QTRecViewer::ResetIntensity(bool checked)
  {
    {
	SNDE_BeginDropPythonGILBlock
    if (posmgr->selected_channel) {
      
      
      int render_mode;
      {
	std::lock_guard<std::mutex> selchan_admin(posmgr->selected_channel->admin);

	render_mode = posmgr->selected_channel->render_mode;
      }
      
      std::shared_ptr<display_axis> a=display->GetAmplAxis(posmgr->selected_channel->FullName);
      if (a && (render_mode == SNDE_DCRM_IMAGE  || render_mode == SNDE_DCRM_GEOMETRY)) {
	{
	  std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
	  posmgr->selected_channel->Offset = 0.0;
	}
	//posmgr->selected_channel->mark_as_dirty();
	// ***!!! Should probably look at intensity bounds for channel instead ***!!!
	UpdateViewerStatus();
	emit NeedRedraw();
      }
    }
      
    SNDE_EndDropPythonGILBlock
	}
  }


  void QTRecViewer::SetOffsetToMean(bool checked)
  {
    {
	SNDE_BeginDropPythonGILBlock
	  if (posmgr->selected_channel) {


		  int render_mode;
		  {
			  std::lock_guard<std::mutex> selchan_admin(posmgr->selected_channel->admin);

			  render_mode = posmgr->selected_channel->render_mode;
		  }

		  std::shared_ptr<display_axis> a = display->GetAmplAxis(posmgr->selected_channel->FullName);

		  std::shared_ptr<recdatabase> recdb_strong = recdb.lock();

		  if (a && recdb_strong && (render_mode == SNDE_DCRM_IMAGE || render_mode == SNDE_DCRM_WAVEFORM)) {
			  snde_float64 mean = 0.0;
			  std::shared_ptr<snde::globalrevision> rev = recdb_strong->latest_globalrev();
			  if (rev) {
				  std::shared_ptr<snde::ndarray_recording_ref> ref = rev->get_ndarray_ref(posmgr->selected_channel->FullName, 0);
				  if (ref) {
					  snde_index n = ref->layout.flattened_length();
					  snde_debug(SNDE_DC_VIEWER, "SetOffsetToMean: n = %llu", n);
					  switch (ref->typenum) {
					  case SNDE_RTN_FLOAT32:
					  case SNDE_RTN_FLOAT64:
						  for (snde_index i = 0; i < n; i++) {
							  mean += ref->element_double(i);
						  }
						  break;
					  case SNDE_RTN_UINT8:
					  case SNDE_RTN_UINT16:
					  case SNDE_RTN_UINT32:
					  case SNDE_RTN_UINT64:
						  for (snde_index i = 0; i < n; i++) {
							  mean += (double)ref->element_unsigned(i);
						  }
						  break;
					  case SNDE_RTN_INT16:
					  case SNDE_RTN_INT32:
					  case SNDE_RTN_INT64:
						  for (snde_index i = 0; i < n; i++) {
							  mean += (double)ref->element_int(i);
						  }
						  break;
					  }
					  if (n != 0) {
						  mean /= double(n);
					  }
					  else {
						  mean = 0.0;
					  }
				  }
			  }

			  {
				  std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
				  if (render_mode == SNDE_DCRM_IMAGE) {
					  posmgr->selected_channel->Offset = mean;
				  }
				  else if(render_mode == SNDE_DCRM_WAVEFORM){
					  posmgr->selected_channel->VertCenterCoord = mean;
				  }
				  
			  }
			  //posmgr->selected_channel->mark_as_dirty();
			  // ***!!! Should probably look at intensity bounds for channel instead ***!!!
			  UpdateViewerStatus();
			  emit NeedRedraw();
		  }
	  }

    SNDE_EndDropPythonGILBlock
	}
  }

  
  void QTRecViewer::Brighten(bool checked)
  {
    {
	SNDE_BeginDropPythonGILBlock
    if (posmgr->selected_channel) {
      int render_mode;
      {
	std::lock_guard<std::mutex> selchan_admin(posmgr->selected_channel->admin);

	render_mode = posmgr->selected_channel->render_mode;
      }

      
      std::shared_ptr<display_axis> a=display->GetAmplAxis(posmgr->selected_channel->FullName);
      
      if (a && (render_mode == SNDE_DCRM_IMAGE || render_mode == SNDE_DCRM_GEOMETRY)) {
	{
	  std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
	  posmgr->selected_channel->Offset -= posmgr->selected_channel->Scale/8.0;
	}
	//posmgr->selected_channel->mark_as_dirty();
	UpdateViewerStatus();
	emit NeedRedraw();
      }
    }
    SNDE_EndDropPythonGILBlock
	}
  }
  

  float QTRecViewer::GetChannelContrast(std::string channelpath) {
    {
	SNDE_BeginDropPythonGILBlock

	  std::shared_ptr<display_channel> displaychan = FindDisplayChan(channelpath);
	  if (!displaychan) {
		  throw snde_error("QTRecViewer::GetChannelContrast -- Channel %s not found", channelpath.c_str());
	  }
	  float retval;
	  {
		  std::lock_guard<std::mutex> adminlock(displaychan->admin);
		  retval = displaychan->Scale;
	  }
	  return retval;
    SNDE_EndDropPythonGILBlock
	}
  }

  void QTRecViewer::SetChannelContrast(std::string channelpath, float contrast) {
    {
	SNDE_BeginDropPythonGILBlock
	  std::shared_ptr<display_channel> displaychan = FindDisplayChan(channelpath);
	  if (!displaychan) {
		  throw snde_error("QTRecViewer::GetChannelContrast -- Channel %s not found", channelpath.c_str());
	  }
	  {
		  std::lock_guard<std::mutex> adminlock(displaychan->admin);
		  displaychan->Scale = contrast;
	  }
	  UpdateViewerStatus();
	  emit NeedRedraw();
    SNDE_EndDropPythonGILBlock
	}
  }


  float QTRecViewer::GetChannelBrightness(std::string channelpath) {
    {
	SNDE_BeginDropPythonGILBlock
	  std::shared_ptr<display_channel> displaychan = FindDisplayChan(channelpath);
	  if (!displaychan) {
		  throw snde_error("QTRecViewer::GetChannelBrightness -- Channel %s not found", channelpath.c_str());
	  }
	  float retval;
	  {
		  std::lock_guard<std::mutex> adminlock(displaychan->admin);
		  retval = displaychan->Offset;
	  }
	  return retval;
	  SNDE_EndDropPythonGILBlock
	}
  }

  void QTRecViewer::EnableChannel(std::string channelpath) {
    {
	SNDE_BeginDropPythonGILBlock
	  std::shared_ptr<display_channel> displaychan = FindDisplayChan(channelpath);
	  if (!displaychan) {
		  throw snde_error("QTRecViewer::EnableChannel -- Channel %s not found", channelpath.c_str());
	  }
	  displaychan->set_enabled(true);
	  emit NeedRedraw();
	  SNDE_EndDropPythonGILBlock
	}
  }

  void QTRecViewer::DisableChannel(std::string channelpath) {
    {
	SNDE_BeginDropPythonGILBlock
	  std::shared_ptr<display_channel> displaychan = FindDisplayChan(channelpath);
	  if (!displaychan) {
		  throw snde_error("QTRecViewer::EnableChannel -- Channel %s not found", channelpath.c_str());
	  }
	  displaychan->set_enabled(false);
	  emit NeedRedraw();
	  SNDE_EndDropPythonGILBlock
	}
  }

  void QTRecViewer::SetChannelBrightness(std::string channelpath, float brightness) {
    {
	SNDE_BeginDropPythonGILBlock
	  std::shared_ptr<display_channel> displaychan = FindDisplayChan(channelpath);
	  if (!displaychan) {
		  throw snde_error("QTRecViewer::GetChannelBrightness -- Channel %s not found", channelpath.c_str());
	  }
	  {
		  std::lock_guard<std::mutex> adminlock(displaychan->admin);
		  displaychan->Offset = brightness;
	  }
	  UpdateViewerStatus();
	  emit NeedRedraw();
	  SNDE_EndDropPythonGILBlock
	}
  }


  void QTRecViewer::LessContrast(bool checked)
  {
    {
	SNDE_BeginDropPythonGILBlock
    if (posmgr->selected_channel) {

      int render_mode;
      {
	std::lock_guard<std::mutex> selchan_admin(posmgr->selected_channel->admin);

	render_mode = posmgr->selected_channel->render_mode;
      }


      double Scale;
      {
	std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
	Scale=posmgr->selected_channel->Scale;
      }
      
      
      std::shared_ptr<display_axis> a=display->GetAmplAxis(posmgr->selected_channel->FullName);
      if (a && (render_mode == SNDE_DCRM_IMAGE || render_mode == SNDE_DCRM_GEOMETRY)) {
	//double contrastpower_floor = floor(log(posmgr->selected_channel->scale)/log(10.0));
	double contrastpower_ceil = ceil(log(Scale)/log(10.0));
	
	//double leadingdigit_floor;
	//int leadingdigit_flooridx;
	
	//std::tie(leadingdigit_flooridx,leadingdigit_floor) = round_to_zoom_digit(round(posmgr->selected_channel->scale/pow(10,contrastpower_floor)));
	
	// Less contrast -> more scale (i.e. more physical quantity/unit intensity
	
	double leadingdigit_ceil;
	int leadingdigit_ceilidx;
	
	std::tie(leadingdigit_ceilidx,leadingdigit_ceil) = round_to_zoom_digit(round(Scale/pow(10,contrastpower_ceil)));
	
	double difference = leadingdigit_ceil*pow(10,contrastpower_ceil) - Scale;
	snde_debug(SNDE_DC_VIEWER,"LessContrast: difference/Scale = %f",fabs(difference/Scale));
	if (fabs(difference/Scale) < .1) {
	  // no significant change from the ceil operation
	  // bump up by one notch
	  
	  const double newleadingdigits[]={2.0,5.0,10.0};
	  snde_debug(SNDE_DC_VIEWER,"LessContrast: Bumping up leadingdigit_ceil from %f to %f.",leadingdigit_ceil,newleadingdigits[leadingdigit_ceilidx]);
	  leadingdigit_ceil = newleadingdigits[leadingdigit_ceilidx];

	  
	}
	
	{
	  std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
	  posmgr->selected_channel->Scale = leadingdigit_ceil*pow(10,contrastpower_ceil);
	}
	//posmgr->selected_channel->mark_as_dirty();
	UpdateViewerStatus();
	
	emit NeedRedraw();
      }
    }
    SNDE_EndDropPythonGILBlock
	}
  }

  void QTRecViewer::RotateColormap(bool checked)
  {
    {
	SNDE_BeginDropPythonGILBlock
    if (posmgr->selected_channel) {
      std::lock_guard<std::mutex> selchan_admin(posmgr->selected_channel->admin);
      posmgr->selected_channel->ColorMap++;
      if (posmgr->selected_channel->ColorMap >= SNDE_COLORMAP_MAXPLUSONE) {
	posmgr->selected_channel->ColorMap=0;
      }
    }
    
    emit NeedRedraw();
    SNDE_EndDropPythonGILBlock
	}
  }

  void QTRecViewer::NextFrame(bool checked) 
  {
    {
	SNDE_BeginDropPythonGILBlock
	  if (posmgr->selected_channel) {
		  {
			  std::lock_guard<std::mutex> selchan_admin(posmgr->selected_channel->admin);
		      posmgr->selected_channel->DisplayFrame++; // No need to get a lock here to check we can increment -- it'll be stopped later
		  }
		  UpdateViewerStatus();
		  emit NeedRedraw();
	  }
    SNDE_EndDropPythonGILBlock
	}
  }

  void QTRecViewer::PreviousFrame(bool checked)
  {
    {
	SNDE_BeginDropPythonGILBlock
	  if (posmgr->selected_channel) {
		  {
			  std::lock_guard<std::mutex> selchan_admin(posmgr->selected_channel->admin);
			  if (posmgr->selected_channel->DisplayFrame > 0) {
				  posmgr->selected_channel->DisplayFrame--;
			  }
		  }
		  UpdateViewerStatus();
		  emit NeedRedraw();
	  }
    SNDE_EndDropPythonGILBlock
	}
  }
  
  void QTRecViewer::MoreContrast(bool checked)
  {
    {
	SNDE_BeginDropPythonGILBlock
    if (posmgr->selected_channel) {

      int render_mode;
      {
	std::lock_guard<std::mutex> selchan_admin(posmgr->selected_channel->admin);

	render_mode = posmgr->selected_channel->render_mode;
      }


      double Scale;
      {
	std::lock_guard<std::mutex> adminlock(posmgr->selected_channel->admin);
	Scale=posmgr->selected_channel->Scale;
      }
      
      std::shared_ptr<display_axis> a=display->GetAmplAxis(posmgr->selected_channel->FullName);
      if (a && (render_mode == SNDE_DCRM_IMAGE || render_mode == SNDE_DCRM_GEOMETRY)) {
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
	//posmgr->selected_channel->mark_as_dirty();
	UpdateViewerStatus();
	emit NeedRedraw();
      }
    }
    SNDE_EndDropPythonGILBlock
	}
  }



  
}
