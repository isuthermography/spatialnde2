#include <osgGA/GUIEventAdapter>

#include "snde/openscenegraph_renderer.hpp"
#include "snde/openscenegraph_compositor.hpp"
#include "snde/rec_display.hpp"
#include "snde/recstore_display_transforms.hpp"
#include "snde/openscenegraph_layerwindow.hpp"
#include "snde/openscenegraph_image_renderer.hpp"
#include "snde/openscenegraph_geom_renderer.hpp"
#include "snde/display_requirements.hpp"

namespace snde {

  static const std::unordered_map<int,int> osg_special_keymappings({
      {osgGA::GUIEventAdapter::KEY_Left,SNDE_RDK_LEFT},
      {osgGA::GUIEventAdapter::KEY_Right,SNDE_RDK_RIGHT},
      {osgGA::GUIEventAdapter::KEY_Up,SNDE_RDK_UP},
      {osgGA::GUIEventAdapter::KEY_Down,SNDE_RDK_DOWN},
      {osgGA::GUIEventAdapter::KEY_Page_Up,SNDE_RDK_PAGEUP},
      {osgGA::GUIEventAdapter::KEY_Page_Down,SNDE_RDK_PAGEDOWN},
      {osgGA::GUIEventAdapter::KEY_Home,SNDE_RDK_HOME},
      {osgGA::GUIEventAdapter::KEY_End,SNDE_RDK_END},
      {osgGA::GUIEventAdapter::KEY_Insert,SNDE_RDK_INSERT},
      {osgGA::GUIEventAdapter::KEY_Delete,SNDE_RDK_DELETE},
      {osgGA::GUIEventAdapter::KEY_BackSpace,SNDE_RDK_BACKSPACE},
      {osgGA::GUIEventAdapter::KEY_Return,SNDE_RDK_ENTER},
      {osgGA::GUIEventAdapter::KEY_Tab,SNDE_RDK_TAB},
      {osgGA::GUIEventAdapter::KEY_Escape,SNDE_RDK_ESC},
    });
  

  osg_compositor_eventhandler::osg_compositor_eventhandler(std::shared_ptr<osg_compositor> comp,std::shared_ptr<display_info> display) :
      comp(comp),
      display(display)
  {
    
  }

  bool osg_compositor_eventhandler::handle(const osgGA::GUIEventAdapter &eventadapt,
					   osgGA::GUIActionAdapter &actionadapt)
  {

    std::shared_ptr<osg_compositor> comp_strong = comp.lock();
    if (!comp_strong) {
      return false;
    }


    std::string selected_channel;
    {
      std::lock_guard<std::mutex> comp_admin(comp_strong->admin);
      selected_channel = comp_strong->selected_channel;
    }
    //snde_warning("osg_compositor: Handling event: %s %d",selected_channel.c_str(),eventadapt.getEventType());

    if (!selected_channel.size()) {
      return false; // no event processing if no channel selected
    }
    //if (eventadapt.getEventType() == osgGA::GUIEventAdapter::FRAME) {
    //  snde_warning("osg_eventhandler frame: Empty=%d",(int)comp_strong->GraphicsWindow->getEventQueue()->empty());
    //}
    
    if (eventadapt.getEventType() == osgGA::GUIEventAdapter::KEYDOWN) {
      bool shift=(bool)(eventadapt.getModKeyMask() & osgGA::GUIEventAdapter::MODKEY_SHIFT);
      bool ctrl=(bool)(eventadapt.getModKeyMask() & osgGA::GUIEventAdapter::MODKEY_CTRL);
      bool alt=(bool)(eventadapt.getModKeyMask() & osgGA::GUIEventAdapter::MODKEY_ALT);

      snde_warning("osg_compositor: Handling keyboard event");

      if (eventadapt.getKey() >= ' ' && eventadapt.getKey() <= '~') {
	// ASCII space...tilde  includes all regular
	// characters of both cases
	display->handle_key_down(selected_channel,eventadapt.getKey(),shift,ctrl,alt);
      } else {
	std::unordered_map<int,int>::const_iterator special_it;

	special_it = osg_special_keymappings.find(eventadapt.getKey());
	if (special_it != osg_special_keymappings.end()) {
	  display->handle_special_down(selected_channel,special_it->second,shift,ctrl,alt);
	  return true; 
	}
      }
    } else if (eventadapt.getEventType() == osgGA::GUIEventAdapter::PUSH ||
	       eventadapt.getEventType() == osgGA::GUIEventAdapter::RELEASE ||
	       eventadapt.getEventType() == osgGA::GUIEventAdapter::DOUBLECLICK ||
	       eventadapt.getEventType() == osgGA::GUIEventAdapter::DRAG) {
      // mouseclick or drag... pass it on to the selected channel

      std::shared_ptr<osg_renderer> renderer;

      {
	std::lock_guard<std::mutex> adminlock(comp_strong->admin); // locking required for renderers field
	std::map<std::string,std::shared_ptr<osg_renderer>>::iterator renderer_it;
	renderer_it=comp_strong->renderers->find(selected_channel);
	if (renderer_it != comp_strong->renderers->end()) {
	  renderer = renderer_it->second;
	}
      }
      
      //snde_warning("osg_compositor: Forwarding mouse event to 0x%lx for %s",(unsigned long)renderer->EventQueue.get(),selected_channel.c_str());

      if (renderer) {
	// OSG event queues seem to be thread safe (see e.g. locking in src/osgGA/EventQueue.cpp)
	// so this should be OK
	renderer->EventQueue->addEvent(new osgGA::GUIEventAdapter(eventadapt,osg::CopyOp::DEEP_COPY_OBJECTS));
	return true;
      }
      
    }

    return false;
  }


  osg_compositor::osg_compositor(std::shared_ptr<recdatabase> recdb,
				 std::shared_ptr<display_info> display,
				 osg::ref_ptr<osgViewer::Viewer> Viewer, // use an osgViewerCompat34()
				 osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
				 bool threaded,
				 bool platform_supports_threaded_opengl) : // ***!!! NOTE: don't set platform_supports_threaded_opengl unless you have arranged some means for the worker thread to operate in a different OpenGL context that shares textures with the main context !!!***
    recdb(recdb),
    display(display),
    GraphicsWindow(GraphicsWindow),
    Viewer(Viewer),
    RootGroup(new osg::Group()),
    threaded(threaded),
    platform_supports_threaded_opengl(platform_supports_threaded_opengl),
    next_state(SNDE_OSGRCS_WAITING),
    need_update(true),
    need_recomposite(true),
    Camera(Viewer->getCamera()),
    display_transforms(std::make_shared<recstore_display_transforms>()),
    RenderCache(nullptr),
    width(0), // assigned in perform_ondemand_calcs
    height(0),
    borderwidthpixels(0)
  {
    
    Camera->setGraphicsContext(GraphicsWindow);
    
    // set background color to blueish
    Camera->setClearColor(osg::Vec4(.1,.1,.3,1.0));
    
    // need to enable culling so that linesegmentintersector (openscenegraph_picker)
    // behavior matches camera behavior
    // (is this efficient?)
    Camera->setComputeNearFarMode( osg::CullSettings::COMPUTE_NEAR_FAR_USING_PRIMITIVES );
    Camera->setCullingMode(osg::CullSettings::ENABLE_ALL_CULLING);
    
    Viewer->setThreadingModel(osgViewer::Viewer::SingleThreaded);
    Viewer->setSceneData(RootGroup);
    
    /* Two dimensional initialization */
    GraticuleTransform = new osg::MatrixTransform();
    osg::ref_ptr<osg::Geode> GraticuleThickGeode = new osg::Geode();
    GraticuleTransform->addChild(GraticuleThickGeode);
    osg::ref_ptr<osg::Geometry> GraticuleThickGeom = new osg::Geometry();
    GraticuleThickGeode->addDrawable(GraticuleThickGeom);
    osg::ref_ptr<osg::Geode> GraticuleThinGeode = new osg::Geode();
    GraticuleTransform->addChild(GraticuleThinGeode);
    osg::ref_ptr<osg::Geometry> GraticuleThinGeom = new osg::Geometry();
    GraticuleThinGeode->addDrawable(GraticuleThinGeom);
    
    osg::ref_ptr<osg::StateSet> GraticuleThinStateSet=GraticuleThinGeode->getOrCreateStateSet();
    GraticuleThinStateSet->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    osg::ref_ptr<osg::LineWidth> GraticuleThinLineWidth=new osg::LineWidth();
    GraticuleThinLineWidth->setWidth(display->borderwidthpixels);
    GraticuleThinStateSet->setAttributeAndModes(GraticuleThinLineWidth,osg::StateAttribute::ON);
    GraticuleThinGeom->setStateSet(GraticuleThinStateSet);
    
    osg::ref_ptr<osg::StateSet> GraticuleThickStateSet=GraticuleThickGeode->getOrCreateStateSet();
    GraticuleThickStateSet->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    osg::ref_ptr<osg::LineWidth> GraticuleThickLineWidth=new osg::LineWidth();
    GraticuleThickLineWidth->setWidth(display->borderwidthpixels*2);
    GraticuleThickStateSet->setAttributeAndModes(GraticuleThickLineWidth,osg::StateAttribute::ON);
    GraticuleThickGeom->setStateSet(GraticuleThickStateSet);
    
    osg::ref_ptr<osg::Vec4Array> GraticuleColorArray=new osg::Vec4Array();
    GraticuleColorArray->push_back(osg::Vec4(1.0,1.0,1.0,1.0));
    GraticuleThinGeom->setColorArray(GraticuleColorArray,osg::Array::BIND_OVERALL);
    GraticuleThinGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    GraticuleThickGeom->setColorArray(GraticuleColorArray,osg::Array::BIND_OVERALL);
    GraticuleThickGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    
    // Units in these coordinates are 5 per division
    osg::ref_ptr<osg::Vec3dArray> ThinGridLineCoords=new osg::Vec3dArray();
      // horizontal thin grid lines
    for (size_t cnt=0; cnt <= display->vertical_divisions;cnt++) {
      double Pos;
      Pos = -1.0*display->vertical_divisions*5.0/2.0 + cnt*5.0;
      ThinGridLineCoords->push_back(osg::Vec3d(-1.0*display->horizontal_divisions*5.0/2.0,Pos,0));
      ThinGridLineCoords->push_back(osg::Vec3d(display->horizontal_divisions*5.0/2.0,Pos,0));
    }
    // vertical thin grid lines
    for (size_t cnt=0; cnt <= display->horizontal_divisions;cnt++) {
      double Pos;
      Pos = -1.0*display->horizontal_divisions*5.0/2.0 + cnt*5.0;
      ThinGridLineCoords->push_back(osg::Vec3d(Pos,-1.0*display->vertical_divisions*5.0/2.0,0));
      ThinGridLineCoords->push_back(osg::Vec3d(Pos,display->vertical_divisions*5.0/2.0,0));
    }
    
    // horizontal thin minidiv lines
    for (size_t cnt=0; cnt <= display->vertical_divisions*5;cnt++) {
      double Pos;
      Pos = -1.0*display->vertical_divisions*5.0/2.0 + cnt;
      ThinGridLineCoords->push_back(osg::Vec3d(-0.5,Pos,0));
      ThinGridLineCoords->push_back(osg::Vec3d(0.5,Pos,0));
    }
    // vertical thin minidiv lines
    for (size_t cnt=0; cnt <= display->horizontal_divisions*5;cnt++) {
      double Pos;
      Pos = -1.0*display->horizontal_divisions*5.0/2.0 + cnt;
      ThinGridLineCoords->push_back(osg::Vec3d(Pos,-0.5,0));
      ThinGridLineCoords->push_back(osg::Vec3d(Pos,0.5,0));
    }
    
    osg::ref_ptr<osg::Vec3dArray> ThickGridLineCoords=new osg::Vec3dArray();
    // horizontal main cross line
    ThickGridLineCoords->push_back(osg::Vec3d(-1.0*display->horizontal_divisions*5.0/2.0,0.0,0.0));
    ThickGridLineCoords->push_back(osg::Vec3d(display->horizontal_divisions*5.0/2.0,0.0,0.0));
    
    // vertical main cross line
    ThickGridLineCoords->push_back(osg::Vec3d(0.0,-1.0*display->vertical_divisions*5.0/2.0,0.0));
    ThickGridLineCoords->push_back(osg::Vec3d(0.0,display->vertical_divisions*5.0/2.0,0.0));
    
    
    
    osg::ref_ptr<osg::DrawArrays> GraticuleThinLines = new osg::DrawArrays(osg::PrimitiveSet::LINES,0,ThinGridLineCoords->size());
    osg::ref_ptr<osg::DrawArrays> GraticuleThickLines = new osg::DrawArrays(osg::PrimitiveSet::LINES,0,ThickGridLineCoords->size());
    
    GraticuleThinGeom->addPrimitiveSet(GraticuleThinLines);
    GraticuleThickGeom->addPrimitiveSet(GraticuleThickLines);
    
    GraticuleThinGeom->setVertexArray(ThinGridLineCoords);
    GraticuleThickGeom->setVertexArray(ThickGridLineCoords);
    SetPickerCrossHairs();
    
    
    
    // Caller should set camera viewport,
    // implement SetProjectionMatrix(),
    // SetTwoDimensional()
    // and make initial calls to those functions
    // from their constructor,
    // then call Viewer->realize();

    
  }

  osg_compositor::~osg_compositor()
  {
    //if (threaded) {
    stop();
    //}
  }

  void osg_compositor::trigger_update()
  {
    {
      std::lock_guard<std::mutex> adminlock(admin);
      need_update=true;
    }
    
  }


  void osg_compositor::wait_render()
  {
    // should have already triggered a render. 
    dispatch(true,true,false);
  }

  void osg_compositor::set_selected_channel(const std::string &selected_name)
  {
    // selected_name should be "" if nothing is selected
    std::lock_guard<std::mutex> adminlock(admin);
    selected_channel = selected_name;
  }


  std::string osg_compositor::get_selected_channel()
  {
    // selected_name should be "" if nothing is selected
    std::lock_guard<std::mutex> adminlock(admin);
    return selected_channel;
  }

  void osg_compositor::perform_ondemand_calcs()
  {
    assert(this_thread_ok_for(SNDE_OSGRCS_ONDEMANDCALCS));
    // NOTE: This function shouldn't make ANY OpenSceneGraph/OpenGL calls, directly or indirectly (!!!)

    std::shared_ptr<recdatabase> recdb_strong = recdb.lock();
    if (!recdb_strong) return;
    
    std::shared_ptr<globalrevision> globalrev = recdb_strong->latest_ready_globalrev();
    
    display->set_current_globalrev(globalrev);

    std::string selected_channel_copy;
    {
      std::lock_guard<std::mutex> adminlock(admin);
      selected_channel_copy=selected_channel;
    }
    
    channels_to_display = display->update(globalrev,selected_channel_copy,false,false,false);

    ColorIdx_by_channelpath.clear();
    
    {
      std::lock_guard<std::mutex> disp_admin(display->admin);
      width = display->drawareawidth;
      height = display->drawareaheight;
      borderwidthpixels = display->borderwidthpixels;

      for (auto && display_chan: channels_to_display) {
	std::lock_guard<std::mutex> dispchan_admin(display_chan->admin);

	ColorIdx_by_channelpath.emplace(display_chan->FullName,display_chan->ColorIdx);
	
      }
      
    }

    display_reqs = traverse_display_requirements(display,globalrev,channels_to_display);


    display_transforms->update(recdb_strong,globalrev,display_reqs);


    // perform all the transforms
    display_transforms->with_display_transforms->wait_complete(); 
    
  }

  
  void osg_compositor::perform_layer_rendering()
  {
    assert(this_thread_ok_for(SNDE_OSGRCS_RENDERING));
    // perform a render ... this method DOES call OpenSceneGraph and requires a valid OpenGL context

    if (!RenderCache) {
      RenderCache = std::make_shared<osg_rendercache>();
    }


    {
      std::lock_guard<std::mutex> adminlock(admin);
      if (!renderers) {
	renderers = std::make_shared<std::map<std::string,std::shared_ptr<osg_renderer>>>();
      }
    }

    
    if (!layer_rendering_rendered_textures) {
      layer_rendering_rendered_textures = std::make_shared<std::map<std::string,std::pair<osg::ref_ptr<osg::Texture2D>,GLuint>>>();
    }
    RenderCache->mark_obsolete();

    // ***!!! NEED TO grab all locks that might be needed at this point, following the correct locking order ***!!!
    
    // This would be by iterating over the display_requirements
    // and either verifying that none of them have require_locking
    // or by accumulating needed lock specs into an ordered set
    // or ordered map, and then locking them in the proper order. 
    
    for (auto && display_req: display_reqs) {
      // look up renderer
      
      std::map<std::string,std::shared_ptr<osg_renderer>>::iterator renderer_it;
      std::shared_ptr<osg_renderer> renderer;
      
      {
	std::unique_lock<std::mutex> adminlock(admin); // locking required for renderers field
	renderer_it=renderers->find(display_req.second->channelpath);

	if (renderer_it==renderers->end() || renderer_it->second->type != display_req.second->renderer_type) {
	  //snde_warning("compositor: New renderer for %s rit_type = %d; drt = %d",display_req.second->channelpath.c_str(),renderer_it==renderers->end() ? -1 : renderer_it->second->type,display_req.second->renderer_type);
	  adminlock.unlock();
	  // Need a new renderer
	  osg::ref_ptr<osgViewer::Viewer> Viewer(new osgViewerCompat34());
	  osg::ref_ptr<osg_layerwindow> LW=new osg_layerwindow(Viewer,nullptr,width,height,false);
	  Viewer->getCamera()->setGraphicsContext(LW);
	  Viewer->getCamera()->setViewport(new osg::Viewport(0,0,width,height));	
	  LW->setup_camera(Viewer->getCamera());
	  
	  if (display_req.second->renderer_type == SNDE_DRRT_IMAGE) {
	    renderer=std::make_shared<osg_image_renderer>(Viewer,LW,display_req.second->channelpath);
	  } else if (display_req.second->renderer_type == SNDE_DRRT_GEOMETRY) {
	    renderer=std::make_shared<osg_geom_renderer>(Viewer,LW,display_req.second->channelpath);
	    
	  } else {
	    snde_warning("osg_compositor: invalid render type SNDE_DRRT_#%d",display_req.second->renderer_type);
	    continue;
	    
	  }
	  adminlock.lock();

	  renderers->erase(display_req.second->channelpath);
	  renderers->emplace(display_req.second->channelpath,renderer);
	} else {	
	  // use pre-existing renderer
	  renderer=renderer_it->second;
	}
      }
      
      // perform rendering
      std::shared_ptr<osg_rendercacheentry> cacheentry;
      bool modified;

      std::tie(cacheentry,modified) = renderer->prepare_render(display_transforms->with_display_transforms,RenderCache,display_reqs,width,height);

      // rerender either if there is a modification to the tree, or if we have OSG events (such as rotating, etc)
      snde_warning("compositor about to render %s: 0x%lx empty=%d",display_req.second->channelpath.c_str(),(unsigned long)renderer->EventQueue.get(),renderer->EventQueue->empty());
      if (cacheentry && (modified || !renderer->EventQueue->empty())) {
	renderer->frame();

	// Push a dummy event prior to the frame on the queue
	// without this we can't process events on our pseudo-GraphicsWindow because
	// osgGA::EventQueue::takeEvents() looks for an event prior to the cutoffTime
	// when selecting events to take. If it doesn't find any then you don't get any
	// events (?).
	// The cutofftime comes from renderer->Viewer->_frameStamp->getReferenceTime()
	osg::ref_ptr<osgGA::Event> dummy_event = new osgGA::Event();
	dummy_event->setTime(renderer->Viewer->getFrameStamp()->getReferenceTime()-1.0);
	renderer->EventQueue->addEvent(dummy_event);

	//std::tie(cacheentry,modified) = renderer->prepare_render(display_transforms->with_display_transforms,RenderCache,display_reqs,width,height);
	//renderer->frame();
	
	// store our generated texture and its ID
	osg::ref_ptr<osg::Texture2D> generated_texture = dynamic_cast<osg_layerwindow *>(renderer->GraphicsWindow.get())->outputbuf;
	layer_rendering_rendered_textures->erase(display_req.second->channelpath);
	
	layer_rendering_rendered_textures->emplace(std::piecewise_construct,
						   std::forward_as_tuple(display_req.second->channelpath),
						   std::forward_as_tuple(std::make_pair(generated_texture,generated_texture->getTextureObject(renderer->GraphicsWindow->getState()->getContextID())->id())));
      }
      
      
    }

    RenderCache->erase_obsolete();
  }


  void osg_compositor::perform_compositing()
  {
    assert(this_thread_ok_for(SNDE_OSGRCS_COMPOSITING));
    if (width != Camera->getViewport()->width() || height != Camera->getViewport()->height()) {
      GraphicsWindow->getEventQueue()->windowResize(0,0,width,height);
      GraphicsWindow->resized(0,0,width,height);
      Camera->setViewport(0,0,width,height);
      
    }

    //snde_warning("perform_compositing: Empty=%d",(int)GraphicsWindow->getEventQueue()->empty());

    
    Camera->setProjectionMatrixAsOrtho(0,width,0,height,0.0,10000.0); // check last two parameters 

    osg::ref_ptr<osg::Group> group=new osg::Group();
    double depth=-1.0*(channels_to_display.size()+1);  // start negative to be compatible with usual OpenGL coordinate frame where negative z's are in front of the cameras
    snde_warning("starting compositing loop:");

    group->getOrCreateStateSet()->setMode(GL_DEPTH_TEST,osg::StateAttribute::OFF);
    
    for (auto && displaychan: channels_to_display) {
      std::map<std::string,std::pair<osg::ref_ptr<osg::Texture2D>,GLuint>>::iterator tex_it;
      std::map<std::string,std::shared_ptr<display_requirement>>::iterator dispreq_it;
      
      tex_it = layer_rendering_rendered_textures->find(displaychan->FullName);
      dispreq_it = display_reqs.find(displaychan->FullName);
     
      if (tex_it != layer_rendering_rendered_textures->end() && dispreq_it != display_reqs.end()) {

	std::shared_ptr<display_requirement> dispreq = dispreq_it->second;
	
	// see https://stackoverflow.com/questions/63992608/displaying-qt-quick-content-inside-openscenegraph-scene
	// and https://github.com/samdavydov/qtquick-osg/blob/master/widget.cpp
	// for an apparently working example of creating an osg::Texture2D from a shared context texture
	
	// Create the texture based on the shared ID.
 
	osg::ref_ptr<osg::Texture2D> tex = new osg::Texture2D();
	osg::ref_ptr<osg::Texture::TextureObject> texobj = new osg::Texture::TextureObject(tex, tex_it->second.second, GL_TEXTURE_2D);
	
	texobj->setAllocated();
	tex->setTextureObject(GraphicsWindow->getState()->getContextID(),texobj);

	snde_warning("borderbox: width=%d,height=%d",width,height);
	/* !!!*** NOTE: Apparently had trouble previously with double precision vs single precision arrays (?) */
	
	// Z position of border is -0.5 relative to image, so it appears on top
	// around edge

	float borderbox_xleft = dispreq->spatial_position->x-borderwidthpixels/2.0;
	if (borderbox_xleft < 0.5) {
	  borderbox_xleft = 0.5;
	}

	float borderbox_xright = dispreq->spatial_position->x+dispreq->spatial_position->width+borderwidthpixels/2.0;
	if (borderbox_xright > width-0.5) {
	  borderbox_xright = width-0.5;
	}

	float borderbox_ybot = dispreq->spatial_position->y-borderwidthpixels/2.0;
	if (borderbox_ybot < 0.5) {
	  borderbox_ybot = 0.5;
	}

	float borderbox_ytop = dispreq->spatial_position->y+dispreq->spatial_position->height+borderwidthpixels/2.0;
	if (borderbox_ytop > height-0.5) {
	  borderbox_ytop = height-0.5;
	}


	snde_warning("borderbox: xleft=%f,ybot=%f,xright=%f,ytop=%f",borderbox_xleft,borderbox_ybot,borderbox_xright,borderbox_ytop);
		
	osg::ref_ptr<osg::Vec3Array> BorderCoords=new osg::Vec3Array(8);
	(*BorderCoords)[0]=osg::Vec3(borderbox_xleft,borderbox_ybot,depth-0.5);
	(*BorderCoords)[1]=osg::Vec3(borderbox_xright,borderbox_ybot,depth-0.5);
	
	(*BorderCoords)[2]=osg::Vec3(borderbox_xright,borderbox_ybot,depth-0.5);
	(*BorderCoords)[3]=osg::Vec3(borderbox_xright,borderbox_ytop,depth-0.5);
	
	(*BorderCoords)[4]=osg::Vec3(borderbox_xright,borderbox_ytop,depth-0.5);
	(*BorderCoords)[5]=osg::Vec3(borderbox_xleft,borderbox_ytop,depth-0.5);
	
	(*BorderCoords)[6]=osg::Vec3(borderbox_xleft,borderbox_ytop,depth-0.5);
	(*BorderCoords)[7]=osg::Vec3(borderbox_xleft,borderbox_ybot,depth-0.5);

	osg::ref_ptr<osg::Geode> bordergeode = new osg::Geode();
	osg::ref_ptr<osg::Geometry> bordergeom = new osg::Geometry();

	osg::ref_ptr<osg::DrawArrays> borderdraw = new osg::DrawArrays(osg::PrimitiveSet::LINES,0,8); // # is number of lines * number of coordinates per line
	bordergeom->setVertexArray(BorderCoords);
	bordergeom->addPrimitiveSet(borderdraw);
	bordergeode->addDrawable(bordergeom);
	bordergeode->getOrCreateStateSet()->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
	bordergeode->getOrCreateStateSet()->setAttributeAndModes(new osg::LineWidth(borderwidthpixels),osg::StateAttribute::ON);
	group->addChild(bordergeode);


	osg::ref_ptr<osg::Vec4Array> BorderColorArray=new osg::Vec4Array();
	size_t ColorIdx = ColorIdx_by_channelpath.at(displaychan->FullName);
	BorderColorArray->push_back(osg::Vec4(RecColorTable[ColorIdx].R,RecColorTable[ColorIdx].G,RecColorTable[ColorIdx].B,1.0));
	bordergeom->setColorArray(BorderColorArray,osg::Array::BIND_OVERALL);
	bordergeom->setColorBinding(osg::Geometry::BIND_OVERALL);
	
	// Image coordinates, from actual corners, counterclockwise,
	// Two triangles    
	osg::ref_ptr<osg::Vec3Array> ImageCoords=new osg::Vec3Array(6);
	osg::ref_ptr<osg::Vec2Array> ImageTexCoords=new osg::Vec2Array(6);
	
	(*ImageCoords)[0]=osg::Vec3(dispreq->spatial_position->x,
				     dispreq->spatial_position->y,
				     depth);
	(*ImageCoords)[1]=osg::Vec3(dispreq->spatial_position->x + dispreq->spatial_position->width,
				     dispreq->spatial_position->y,
				     depth);
	(*ImageCoords)[2]=osg::Vec3(dispreq->spatial_position->x,
				     dispreq->spatial_position->y + dispreq->spatial_position->height,
				     depth);
	
	(*ImageTexCoords)[0]=osg::Vec2(0,0);
	(*ImageTexCoords)[1]=osg::Vec2(1,0);
	(*ImageTexCoords)[2]=osg::Vec2(0,1);
      
	// upper-right triangle 
	(*ImageCoords)[3]=osg::Vec3(dispreq->spatial_position->x + dispreq->spatial_position->width,
				     dispreq->spatial_position->y + dispreq->spatial_position->height,
				     depth);
	(*ImageCoords)[4]=osg::Vec3(dispreq->spatial_position->x,
				     dispreq->spatial_position->y + dispreq->spatial_position->height,
				     depth);
	(*ImageCoords)[5]=osg::Vec3(dispreq->spatial_position->x + dispreq->spatial_position->width,
				     dispreq->spatial_position->y,
				     depth);
	(*ImageTexCoords)[3]=osg::Vec2(1,1);
	(*ImageTexCoords)[4]=osg::Vec2(0,1);
	(*ImageTexCoords)[5]=osg::Vec2(1,0);


	osg::ref_ptr<osg::Geode> ImageGeode = new osg::Geode();
	osg::ref_ptr<osg::Geometry> ImageGeom = new osg::Geometry();
	osg::ref_ptr<osg::DrawArrays> ImageTris = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,6); // # is number of triangles * number of coordinates per triangle
	osg::ref_ptr<osg::Texture2D> ImageTexture = tex; // imported texture from above

	ImageGeom->addPrimitiveSet(ImageTris);
	ImageGeom->setVertexArray(ImageCoords);
	ImageGeom->setTexCoordArray(0,ImageTexCoords);
	ImageGeode->getOrCreateStateSet()->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
	ImageGeode->getOrCreateStateSet()->setTextureAttributeAndModes(0,ImageTexture,osg::StateAttribute::ON);
	osg::ref_ptr<osg::Vec4Array> ColorArray=new osg::Vec4Array();
	ColorArray->push_back(osg::Vec4(1.0,1.0,1.0,1.0));
	ImageGeom->setColorArray(ColorArray,osg::Array::BIND_OVERALL);
	ImageGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
	ImageGeode->addDrawable(ImageGeom);

	group->addChild(ImageGeode);
	
	snde_warning("osg_compositor::perform_compositing(): Rendered channel %s",displaychan->FullName.c_str());
	
	depth--; // next layer is deeper (though maybe we should flip this around and build from deep to shallow to make transparency work better)
      } else {
	snde_warning("osg_compositor::perform_compositing(): Did not find rendered layer for channel %s",displaychan->FullName.c_str());
      }
      
    }
    
    //snde_warning("perform_compositing2: Empty=%d",(int)GraphicsWindow->getEventQueue()->empty());

    //Viewer->setSceneData(group); setSceneData() seems to clear our event queue, so instead we swap out the contents of a persistent group (RootGroup) that was set as the scene data in the constructor

    
    if (RootGroup->getNumChildren()) {
      RootGroup->removeChildren(0,1);
    }
    RootGroup->addChild(group);
    
    
    //snde_warning("drawing frame: Empty=%d",(int)GraphicsWindow->getEventQueue()->empty());
    Viewer->frame();
    
  }

  bool osg_compositor::this_thread_ok_for_locked(int action)
  {

    assert(action==next_state);  // not strictly necessary, but why else would we be checking??
    
    const std::set<int> &thread_ok_actions = responsibility_mapping.at(std::this_thread::get_id());

    return thread_ok_actions.find(action) != thread_ok_actions.end();
  }

  bool osg_compositor::this_thread_ok_for(int action)
  {
    std::lock_guard<std::mutex> compositor_admin(admin);

    return this_thread_ok_for_locked(action);
  }

  
  void osg_compositor::dispatch(bool return_if_idle,bool wait, bool loop_forever)
  {
    if (!wait) {
      assert(!loop_forever); // to loop forever we have to be waiting too
      // wait && !loop_forever performs up to one wait, one dispatch sequence
      // up to completion of compositing
    }

    std::unique_lock<std::mutex> adminlock(admin);
    bool executed_something=false;
    bool executed_compositing=false; 

    if (return_if_idle && next_state==SNDE_OSGRCS_WAITING && !need_recomposite && !need_update) {
      // idle and return_if_idle flag
      return;
    }
    
    while (next_state != SNDE_OSGRCS_EXIT) {
      while (!this_thread_ok_for_locked(next_state)) {
	if ( (wait && loop_forever) || (wait && !loop_forever && !executed_something)) {
	  execution_notify.wait(adminlock);
	} else {
	  return; // caller said not to wait
	}
      }
      
      if (next_state == SNDE_OSGRCS_WAITING) {
	if (need_recomposite) {
	  next_state = SNDE_OSGRCS_COMPOSITING;
	  need_recomposite=false;
	  executed_something=true; 
	}
	if (need_update) {
	  next_state = SNDE_OSGRCS_ONDEMANDCALCS;
	  need_update = false;
	  executed_something=true; 
	}
      } else if (next_state == SNDE_OSGRCS_ONDEMANDCALCS) {
	adminlock.unlock();
	try {
	  perform_ondemand_calcs();
	} catch(const std::exception &e) {
	  snde_warning("Exception in ondemand rendering calculations: %s",e.what());
	}
	adminlock.lock();
	executed_something=true;
	if (next_state == SNDE_OSGRCS_ONDEMANDCALCS) {
	  // otherwise we don't want to interrupt a cleanup/exit command
	  next_state = SNDE_OSGRCS_RENDERING;
	}
	
      } else if (next_state==SNDE_OSGRCS_RENDERING) {
	adminlock.unlock();
	try {
	  perform_layer_rendering();
	} catch(const std::exception &e) {
	  snde_warning("Exception in compositor layer rendering operations: %s",e.what());
	}
	adminlock.lock();
	executed_something=true; 
	if (next_state == SNDE_OSGRCS_RENDERING) {
	  // otherwise we don't want to interrupt a cleanup/exit command
	  next_state = SNDE_OSGRCS_COMPOSITING;
	}
	
      } else if (next_state==SNDE_OSGRCS_COMPOSITING) {
	adminlock.unlock();
	try {
	  perform_compositing();
	} catch(const std::exception &e) {
	  snde_warning("Exception in compositing operations: %s",e.what());
	}
	adminlock.lock();
	executed_something=true; 
	executed_compositing=true; 
	if (next_state == SNDE_OSGRCS_COMPOSITING) {
	  // otherwise we don't want to interrupt a cleanup/exit command
	  next_state = SNDE_OSGRCS_WAITING;
	}
      } else if (next_state==SNDE_OSGRCS_COMPOSITING_CLEANUP) {
	//compositing_textures = nullptr; // This free's the various OSG objects. We have to be careful to do it from this thread

	next_state = SNDE_OSGRCS_RENDERING_CLEANUP;	
      }
      else if (next_state==SNDE_OSGRCS_RENDERING_CLEANUP) {
	RenderCache = nullptr;
	renderers = nullptr;
	layer_rendering_rendered_textures = nullptr;
	
	next_state = SNDE_OSGRCS_EXIT;	
      }


      
      execution_notify.notify_all();

      if (wait && !loop_forever && executed_compositing) {
	// finished a pass
	return; 
      }
    }
    
  }
  
  
  void osg_compositor::worker_code()
  {
    dispatch(false,true,true);
  }

  void osg_compositor::start()
  {
    std::lock_guard<std::mutex> adminlock(admin);

    if (!eventhandler) {
      eventhandler = new osg_compositor_eventhandler(shared_from_this(),display);
      Viewer->addEventHandler(eventhandler);

      Viewer->getCamera()->setAllowEventFocus(false); // prevent camera from eating our events and interfering
    }
    if (threaded) {

      
      worker_thread = std::make_shared<std::thread>([ this ]() { this->worker_code(); });

      if (platform_supports_threaded_opengl) {
	responsibility_mapping.emplace(worker_thread->get_id(),std::set<int>{SNDE_OSGRCS_WAITING,SNDE_OSGRCS_ONDEMANDCALCS,SNDE_OSGRCS_RENDERING,SNDE_OSGRCS_RENDERING_CLEANUP,SNDE_OSGRCS_EXIT});
	
	responsibility_mapping.emplace(std::this_thread::get_id(),std::set<int>{SNDE_OSGRCS_COMPOSITING,SNDE_OSGRCS_COMPOSITING_CLEANUP,SNDE_OSGRCS_EXIT});
      } else {
	responsibility_mapping.emplace(worker_thread->get_id(),std::set<int>{SNDE_OSGRCS_WAITING,SNDE_OSGRCS_ONDEMANDCALCS,SNDE_OSGRCS_EXIT});
	responsibility_mapping.emplace(std::this_thread::get_id(),std::set<int>{SNDE_OSGRCS_COMPOSITING,SNDE_OSGRCS_RENDERING,SNDE_OSGRCS_COMPOSITING_CLEANUP,SNDE_OSGRCS_RENDERING_CLEANUP,SNDE_OSGRCS_EXIT});
      }

      
    } else {
      responsibility_mapping.emplace(std::this_thread::get_id(),std::set<int>{SNDE_OSGRCS_WAITING,SNDE_OSGRCS_ONDEMANDCALCS,SNDE_OSGRCS_COMPOSITING,SNDE_OSGRCS_RENDERING,SNDE_OSGRCS_COMPOSITING_CLEANUP,SNDE_OSGRCS_RENDERING_CLEANUP,SNDE_OSGRCS_EXIT});
      
    }


    
  }

  void osg_compositor::stop()
  {
    // Get us into cleanup mode.
    {
      std::lock_guard<std::mutex> adminlock(admin);
      next_state = SNDE_OSGRCS_COMPOSITING_CLEANUP;
      execution_notify.notify_all();
    }

    dispatch(false,true,true); // perform any cleanup actions that are our thread's responsibility
    
    if (threaded) {
      worker_thread->join();
      worker_thread=nullptr;
    }
  }

  
  void osg_compositor::SetPickerCrossHairs()
  {
    
    PickerCrossHairs = new osg::MatrixTransform();
    osg::ref_ptr<osg::Geode> CrossHairsGeode = new osg::Geode();
    osg::ref_ptr<osg::Geometry> CrossHairsGeom = new osg::Geometry();
    osg::ref_ptr<osg::StateSet> CrossHairsStateSet = CrossHairsGeode->getOrCreateStateSet();
    PickerCrossHairs->addChild(CrossHairsGeode);
    CrossHairsGeode->addDrawable(CrossHairsGeom);
    CrossHairsStateSet->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    osg::ref_ptr<osg::LineWidth> CrossHairsLineWidth=new osg::LineWidth();
    CrossHairsLineWidth->setWidth(4);
    CrossHairsStateSet->setAttributeAndModes(CrossHairsLineWidth,osg::StateAttribute::ON);
    CrossHairsGeom->setStateSet(CrossHairsStateSet);
    osg::ref_ptr<osg::Vec4Array> CrossHairsColorArray=new osg::Vec4Array();
    CrossHairsColorArray->push_back(osg::Vec4(1.0,1.0,1.0,1.0)); // R, G, B, A
    CrossHairsGeom->setColorArray(CrossHairsColorArray,osg::Array::BIND_OVERALL);
    CrossHairsGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    
    
    osg::ref_ptr<osg::Vec3Array> CrossHairsLinesCoords=new osg::Vec3Array();
    CrossHairsLinesCoords->push_back(osg::Vec3(-10.0,-10.0,0.0));
    CrossHairsLinesCoords->push_back(osg::Vec3(10.0,10.0,0.0));
    CrossHairsLinesCoords->push_back(osg::Vec3(-10.0,10.0,0.0));
    CrossHairsLinesCoords->push_back(osg::Vec3(10.0,-10.0,0.0));
    
    osg::ref_ptr<osg::DrawArrays> CrossHairsLines = new osg::DrawArrays(osg::PrimitiveSet::LINES,0,CrossHairsLinesCoords->size());
    
    CrossHairsGeom->addPrimitiveSet(CrossHairsLines);
    CrossHairsGeom->setVertexArray(CrossHairsLinesCoords);
    
  }
  


};
