#include <osgGA/GUIEventAdapter>
#include <osgViewer/Renderer>
#include <osgUtil/ShaderGen>

#include <Eigen/Dense>
#include <osgDB/WriteFile>

#include "snde/recstore.hpp"
#include "snde/openscenegraph_renderer.hpp"
#include "snde/openscenegraph_compositor.hpp"
#include "snde/rec_display.hpp"
#include "snde/recstore_display_transforms.hpp"
#include "snde/openscenegraph_layerwindow.hpp"
#include "snde/openscenegraph_2d_renderer.hpp"
#include "snde/openscenegraph_geom_renderer.hpp"
#include "snde/display_requirements.hpp"
#include "snde/quaternion.h"

#ifndef GL_VERSION_3_2
typedef struct __GLsync* GLsync;
typedef uint64_t GLuint64;  // may need #include <stdint.h>
#endif

#ifndef GL_VERSION_4_3
typedef void (APIENTRY* GLDEBUGPROC)(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);
#endif

// access default shaders from OpenSceneGraph
// (WARNING: These might go into a namespace sometime!!!)
extern char shadergen_frag[];
extern char shadergen_vert[];


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
  

  osg_compositor_eventhandler::osg_compositor_eventhandler(osg_compositor *comp,std::shared_ptr<display_info> display) :
      comp(comp),
      compositor_dead(false),
      display(display)
  {
    
  }

  bool osg_compositor_eventhandler::handle(const osgGA::GUIEventAdapter &eventadapt,
					   osgGA::GUIActionAdapter &actionadapt)
  {

    
    
    if (compositor_dead) {
      return false; 
    }


    std::string selected_channel;
    {
      std::lock_guard<std::mutex> comp_admin(comp->admin);
      selected_channel = comp->selected_channel;
    }
    snde_debug(SNDE_DC_EVENT,"osg_compositor: Handling event: %s %d",selected_channel.c_str(),eventadapt.getEventType());

    if (!selected_channel.size()) {
      return false; // no event processing if no channel selected
    }
    //if (eventadapt.getEventType() == osgGA::GUIEventAdapter::FRAME) {
    //  snde_warning("osg_eventhandler frame: Empty=%d",(int)comp->GraphicsWindow->getEventQueue()->empty());
    //}
    
    if (eventadapt.getEventType() == osgGA::GUIEventAdapter::KEYDOWN) {
      bool shift=(bool)(eventadapt.getModKeyMask() & osgGA::GUIEventAdapter::MODKEY_SHIFT);
      bool ctrl=(bool)(eventadapt.getModKeyMask() & osgGA::GUIEventAdapter::MODKEY_CTRL);
      bool alt=(bool)(eventadapt.getModKeyMask() & osgGA::GUIEventAdapter::MODKEY_ALT);

      snde_debug(SNDE_DC_EVENT,"osg_compositor: Handling keyboard event");

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
      snde_debug(SNDE_DC_EVENT,"osg_compositor: Handling mouse event");

      std::shared_ptr<osg_renderer> renderer;

      {
	std::lock_guard<std::mutex> adminlock(comp->admin); // locking required for renderers field
	std::map<std::string,std::shared_ptr<osg_renderer>>::iterator renderer_it;
	renderer_it=comp->renderers->find(selected_channel);
	if (renderer_it != comp->renderers->end()) {
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
				 bool enable_threaded_opengl,
				 bool enable_shaders,
				 GLuint LayerDefaultFramebufferObject/*=0*/) : // ***!!! NOTE: don't set enabled_threaded_opengl unless you have arranged some means for the worker thread to operate in a different OpenGL context that shares textures with the main context. This would probably have to be done by a context !!!***
    recdb(recdb),
    display(display),
    GraphicsWindow(GraphicsWindow),
    Viewer(Viewer),
    RootGroup(new osg::Group()),
    threaded(threaded),
    enable_threaded_opengl(enable_threaded_opengl),
    enable_shaders(enable_shaders),
    LayerDefaultFramebufferObject(LayerDefaultFramebufferObject),
    next_state(SNDE_OSGRCS_WAITING),
    threads_started(false),
    need_rerender(true),
    need_recomposite(true),
    need_resize(true),
    resize_width(0),
    resize_height(0),
    Camera(Viewer->getCamera()),
    display_transforms(std::make_shared<recstore_display_transforms>()),
    RenderCache(nullptr),
    compositor_width(0), // assigned in perform_layer_rendering from resize_width and height
    compositor_height(0),
    borderwidthpixels(0),
    request_continuous_update(false)
  {
    {
      std::lock_guard<std::mutex> displayadmin(display->admin);
      resize_width = display->drawareawidth;
      resize_height = display->drawareaheight;
    }
    
    Camera->setGraphicsContext(GraphicsWindow);
    
    // set background color to blueish
    Camera->setClearColor(osg::Vec4(.1,.1,.3,1.0));
    //Camera->setClearColor(osg::Vec4(.3,.1,.1,1.0));
    Camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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
    osg::ref_ptr<osg::Vec3Array> ThinGridLineCoords=new osg::Vec3Array();
      // horizontal thin grid lines
    double xpos = display->horizontal_divisions * 5.0 / 2.0;
    for (size_t cnt=0; cnt <= display->vertical_divisions;cnt++) {
      double Pos;
      Pos = -1.0*display->vertical_divisions*5.0/2.0 + cnt*5.0;
      ThinGridLineCoords->push_back(osg::Vec3(-1.0*display->horizontal_divisions*5.0/2.0,Pos,0));
      ThinGridLineCoords->push_back(osg::Vec3(display->horizontal_divisions*5.0/2.0,Pos,0));

      /*
      ThinGridLineCoords->push_back(osg::Vec3(-xpos - display->borderwidthpixels, Pos - display->borderwidthpixels, 0));
      ThinGridLineCoords->push_back(osg::Vec3(xpos - display->borderwidthpixels, Pos - display->borderwidthpixels, 0));
      ThinGridLineCoords->push_back(osg::Vec3(-xpos + display->borderwidthpixels, Pos + display->borderwidthpixels, 0));
      ThinGridLineCoords->push_back(osg::Vec3(-xpos + display->borderwidthpixels, Pos + display->borderwidthpixels, 0));
      ThinGridLineCoords->push_back(osg::Vec3(xpos - display->borderwidthpixels, Pos - display->borderwidthpixels, 0));
      ThinGridLineCoords->push_back(osg::Vec3(xpos + display->borderwidthpixels, Pos + display->borderwidthpixels, 0));    
      */
    }
    // vertical thin grid lines
    for (size_t cnt=0; cnt <= display->horizontal_divisions;cnt++) {
      double Pos;
      Pos = -1.0*display->horizontal_divisions*5.0/2.0 + cnt*5.0;
      ThinGridLineCoords->push_back(osg::Vec3(Pos,-1.0*display->vertical_divisions*5.0/2.0,0));
      ThinGridLineCoords->push_back(osg::Vec3(Pos,display->vertical_divisions*5.0/2.0,0));
    }
    
    // horizontal thin minidiv lines
    for (size_t cnt=0; cnt <= display->vertical_divisions*5;cnt++) {
      double Pos;
      Pos = -1.0*display->vertical_divisions*5.0/2.0 + cnt;
      ThinGridLineCoords->push_back(osg::Vec3(-0.5,Pos,0));
      ThinGridLineCoords->push_back(osg::Vec3(0.5,Pos,0));
    }
    // vertical thin minidiv lines
    for (size_t cnt=0; cnt <= display->horizontal_divisions*5;cnt++) {
      double Pos;
      Pos = -1.0*display->horizontal_divisions*5.0/2.0 + cnt;
      ThinGridLineCoords->push_back(osg::Vec3(Pos,-0.5,0));
      ThinGridLineCoords->push_back(osg::Vec3(Pos,0.5,0));
    }
    
    osg::ref_ptr<osg::Vec3Array> ThickGridLineCoords=new osg::Vec3Array();
    // horizontal main cross line
    ThickGridLineCoords->push_back(osg::Vec3(-1.0*display->horizontal_divisions*5.0/2.0,0.0,0.0));
    ThickGridLineCoords->push_back(osg::Vec3(display->horizontal_divisions*5.0/2.0,0.0,0.0));
    
    // vertical main cross line
    ThickGridLineCoords->push_back(osg::Vec3(0.0,-1.0*display->vertical_divisions*5.0/2.0,0.0));
    ThickGridLineCoords->push_back(osg::Vec3(0.0,display->vertical_divisions*5.0/2.0,0.0));
    
    
    
    osg::ref_ptr<osg::DrawArrays> GraticuleThinLines = new osg::DrawArrays(osg::PrimitiveSet::LINES,0,ThinGridLineCoords->size());
    osg::ref_ptr<osg::DrawArrays> GraticuleThickLines = new osg::DrawArrays(osg::PrimitiveSet::LINES,0,ThickGridLineCoords->size());
    
    GraticuleThinGeom->addPrimitiveSet(GraticuleThinLines);
    GraticuleThickGeom->addPrimitiveSet(GraticuleThickLines);
    
    GraticuleThinGeom->setVertexArray(ThinGridLineCoords);
    GraticuleThickGeom->setVertexArray(ThickGridLineCoords);
    SetPickerCrossHairs();


    RootGroup->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);
    RootGroup->getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    
    
    
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
    snde_debug(SNDE_DC_RENDERING,"~osg_compositor()");
    // NOTE: If we are actually a qt_osg_compositor(), that destructor wold have already called stop() once!
    stop();
    snde_debug(SNDE_DC_RENDERING,"~osg_compositor(): stop() complete");
    //}
    if (eventhandler) {
      eventhandler->compositor_dead=true;
    }
  }

  void osg_compositor::trigger_rerender()
  {  // NOTE: Don't call this except from the main GUI thread because subclasses (qt_osg_render()) call non-threadsafe eventhandling code in the derived version
    {
      std::lock_guard<std::mutex> adminlock(admin);
      need_rerender=true;
      execution_notify.notify_all();
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

  void osg_compositor::perform_ondemand_calcs(std::unique_lock<std::mutex> *adminlock)
  {
    std::shared_ptr<globalrevision> globalrev; 
    assert(this_thread_ok_for_locked(SNDE_OSGRCS_ONDEMANDCALCS));
    // NOTE: This function shouldn't make ANY OpenSceneGraph/OpenGL calls, directly or indirectly (!!!)

    adminlock->unlock();
    
    std::shared_ptr<recdatabase> recdb_strong = recdb.lock();
    if (!recdb_strong) {
      adminlock->lock();
      return;
    }
    try {
      
      globalrev = recdb_strong->latest_globalrev(); // obtain latest ready globalrev
      
      //display->set_current_globalrev(globalrev); (redundant with display->update)
      
      std::string selected_channel_copy;
      {
	std::lock_guard<std::mutex> adminlock(admin);
	selected_channel_copy=selected_channel;
      }

      std::vector<std::shared_ptr<display_channel>> mutable_channels;

      std::shared_ptr<globalrev_mutable_lock> incomplete_mutable_recording_locker; // used if we have mutable recordings that are incomplete to hold them from being modified once completed while we do our computations
      rwlock_token_set complete_mutable_recording_locks;
      
  
      std::tie(channels_to_display,mutable_channels) = display->get_channels(globalrev,selected_channel_copy,true,true,false,false);

      // !!!*** Really we need to check for anything mutable in the display requirements not just the raw channel list ***!!!
      
      if (mutable_channels.size() > 0) {
	// if we have a mutable channel we have to go to the latest defined globalrev,
	// not the latest ready globalrev so that we don't stall the computation
	// pipeline (display math functions are prioritized with its globalrev index
	// and can therefore hold mutable recording locks, delaying subsequent globalrev
	// mutable computations).

	{
	  // we hold the recdb admin lock while getting the channels so that
	  // another globalrev can't be defined before we have the chance
	  // to register our callback to read-lock things. 
	  std::lock_guard<std::mutex> recdb_admin(recdb_strong->admin);
	  
	  // switch to latest defined globalrev
	  globalrev = recdb_strong->latest_defined_globalrev();
	  std::lock_guard<std::mutex> globalrev_admin(globalrev->admin);
	  incomplete_mutable_recording_locker=globalrev->mutable_recordings_need_holder;
	  
	  if (globalrev->ready) {
	    // if the globalrev is ready we can update the list of mutable channels
	    std::tie(channels_to_display,mutable_channels) = display->get_channels(globalrev,selected_channel_copy,true,true,false,false);
	  } else {
	    // otherwise we hope it hasn't changed. !!!*** Alternatively can we get the mutability from the math definitions somehow? 
	    std::vector<std::shared_ptr<display_channel>> junk;
	    std::tie(channels_to_display,junk) = display->get_channels(globalrev,selected_channel_copy,false,true,false,false);
	    
	  }
	  assert(incomplete_mutable_recording_locker);
	  /*  This stuff is now unnecessary because the mutable_recordings_need_holder now 
	      sticks around until a new latest_define_globalrev is assigned */
	  /*
	  if (!incomplete_mutable_recording_locker) {
	    // if the locker has expired then this globalrev may already be complete and
	    // we can hold the mutable recordings by locking them explicitly for read
	    
	    assert(globalrev->ready);
	    
	    // !!!*** Should we have a more general process that can operate on mutable
	    // structures besides multi_ndarray_recording?
	    std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> recrefs;
	    
	    for (auto && mutable_channel : mutable_channels) {
	      std::shared_ptr<recording_base> mutable_rec = globalrev->get_recording(mutable_channel->FullName);
	      std::shared_ptr<multi_ndarray_recording> mutable_ndarray = std::dynamic_pointer_cast<multi_ndarray_recording>(mutable_rec);
	      
	      if (mutable_ndarray) {
		for (size_t array_index=0; array_index < mutable_ndarray->layouts.size();array_index++) {
		  recrefs.push_back(std::make_pair(mutable_ndarray->reference_ndarray(array_index),false)); // will lock this ref for read
		}
	      }
	    }
	    
	    
	    complete_mutable_recording_locks = globalrev->lockmgr->lock_recording_refs(recrefs,false);
	  } */

	  
	} // release the locks...
	
	globalrev->wait_complete(); // wait for this globalrev to become complete before finishing render process
      }
    } catch (const std::exception &exc) {
      snde_warning("Exception class %s caught in osg_compositor::perform_ondemand_calcs: getting complete globalrev %s",typeid(exc).name(),exc.what());
      
      
    }

    try {
      
    
      ColorIdx_by_channelpath.clear();
      
      {
	std::lock_guard<std::mutex> disp_admin(display->admin);
	borderwidthpixels = display->borderwidthpixels;
	
	for (auto && display_chan: channels_to_display) {
	  std::lock_guard<std::mutex> dispchan_admin(display_chan->admin);
	  
	  ColorIdx_by_channelpath.emplace(display_chan->FullName,display_chan->ColorIdx);
	  
	}
	
      }
      
      display_reqs = traverse_display_requirements(display,globalrev,channels_to_display);
      
      
    } catch (const std::exception &exc) {
      snde_warning("Exception class %s caught in osg_compositor::perform_ondemand_calcs: colors and display requirements %s",typeid(exc).name(),exc.what());
      
      
    }
    try {
      display_transforms->update(recdb_strong,globalrev,display_reqs);
    } catch (const std::exception &exc) {
      snde_warning("Exception class %s caught in osg_compositor::perform_ondemand_calcs: display update %s",typeid(exc).name(),exc.what());
      
      
    }
    try {
      
      // perform all the transforms
      display_transforms->with_display_transforms->wait_complete(); 


      // Closing this block frees any locks holding current versions of mutable
      // recordings in place.
      
    } catch (const std::exception &exc) {
      snde_warning("Exception class %s caught in osg_compositor::perform_ondemand_calcs: wait %s",typeid(exc).name(),exc.what());
      
      
    }
    adminlock->lock();
 
  }
  
  
  void osg_compositor::perform_layer_rendering(std::unique_lock<std::mutex> *adminlock)
  {
    assert(this_thread_ok_for_locked(SNDE_OSGRCS_RENDERING));
    // perform a render ... this method DOES call OpenSceneGraph and requires a valid OpenGL context

    adminlock->unlock();
    try {
      bool resizing=false; 
      {
	
	std::lock_guard<std::mutex> adminlock(admin); // locking required for resize_width
	
	if (need_resize) {
	  compositor_width = resize_width;
	  compositor_height = resize_height;
	  resizing=true;
	}
      }
      
      
      if (!RenderCache) {
	RenderCache = std::make_shared<osg_rendercache>();
      }
      
      
      {
	//std::lock_guard<std::mutex> adminlock(admin);
	adminlock->lock();
	if (!renderers) {
	  renderers = std::make_shared<std::map<std::string,std::shared_ptr<osg_renderer>>>();
	}
	adminlock->unlock();
      }
      
      
      if (!layer_rendering_rendered_textures) {
	layer_rendering_rendered_textures = std::make_shared<std::map<std::string,std::pair<osg::ref_ptr<osg::Texture2D>,GLuint>>>();
      }
      
      
      bool new_request_continuous_update = false; 
      
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
	
	osg::ref_ptr<osgViewerCompat34> LayerViewer;
	{
	  std::unique_lock<std::mutex> adminlock2(admin); // locking required for renderers field
	  renderer_it=renderers->find(display_req.second->channelpath);
	
	  if (renderer_it==renderers->end() || renderer_it->second->type != display_req.second->renderer_type) {
	    //snde_warning("compositor: New renderer for %s rit_type = %d; drt = %d",display_req.second->channelpath.c_str(),renderer_it==renderers->end() ? -1 : renderer_it->second->type,display_req.second->renderer_type);
	    adminlock2.unlock();
	    // Need a new renderer
	    LayerViewer = new osgViewerCompat34();
	    LayerViewer->setThreadingModel(osgViewer::Viewer::SingleThreaded);
	    osg::ref_ptr<osg_layerwindow> LW=new osg_layerwindow(LayerViewer,nullptr,compositor_width,compositor_height,false);
	    LW->setDefaultFboId(LayerDefaultFramebufferObject);
	    
	    LayerViewer->getCamera()->setGraphicsContext(LW);
	    LayerViewer->getCamera()->setViewport(0,0,0,0);
	    //LayerViewer->getCamera()->setViewport(new osg::Viewport(0,0,compositor_width,compositor_height));	 (let the renderer set the viewport so it can detect modifications
	    
	    LW->setup_camera(LayerViewer->getCamera());
	    
	    if (display_req.second->renderer_type == SNDE_DRRT_2D) {
	      renderer=std::make_shared<osg_2d_renderer>(LayerViewer,LW,display_req.second->channelpath,enable_shaders);
	    } else if (display_req.second->renderer_type == SNDE_DRRT_GEOMETRY) {
	      renderer=std::make_shared<osg_geom_renderer>(LayerViewer,LW,display_req.second->channelpath,enable_shaders);
	      
	    } else {
	      snde_warning("osg_compositor: invalid render type SNDE_DRRT_#%d",display_req.second->renderer_type);
	      continue;
	      
	    }
	    adminlock2.lock();

	    auto FutCamPoseIt = FutureChannelCamPose.find(display_req.second->channelpath);
	    if (FutCamPoseIt != FutureChannelCamPose.end()) {
	      renderer->AssignNewCameraPose(FutCamPoseIt->second);
	    }
	    auto FutRotCtrIt = FutureChannelRotationCenterDist.find(display_req.second->channelpath);
	    if (FutRotCtrIt != FutureChannelRotationCenterDist.end()) {
	      renderer->AssignNewRotationCenterDist(FutRotCtrIt->second);
	    }
	    
	    renderers->erase(display_req.second->channelpath);
	    renderers->emplace(display_req.second->channelpath,renderer);
	  } else {	
	    // use pre-existing renderer
	    renderer=renderer_it->second;

	    osg::ref_ptr<osg_layerwindow> LW = dynamic_cast<osg_layerwindow *>(renderer->GraphicsWindow.get());

	    assert(LW);
	    LW->setDefaultFboId(LayerDefaultFramebufferObject); // fixup the default FBO as it might have changed e.g. if the window was resized. 

	    // Force re-setup on camera
	    // Not sure why this is necessary but without it, when you enlarge the window
	    // after creation it only draws up to the original size of the rendering area
	    // for this layer.
	    
	    dynamic_cast<osgViewer::Renderer *>(renderer->Viewer->getCamera()->getRenderer())->setCameraRequiresSetUp(true);
	    

	    
	    // old debugging stuff
	    /*
	    osg::ref_ptr<osg::Camera> oldcamera = renderer->Viewer->getCamera();
	    {
	      osg::ref_ptr<osgViewer::Viewer> oldviewer=renderer->Viewer;
	      assert(renderer->GraphicsWindow->getState());
	      //renderer->Viewer = new osgViewerCompat34(*dynamic_cast<osgViewerCompat34*>(renderer->Viewer.get())); //new osgViewerCompat34();
	      //renderer->Viewer->getCamera()->setRenderer(new osgViewer::Renderer(renderer->Viewer->getCamera()));
	      renderer->Viewer->setThreadingModel(osgViewer::Viewer::SingleThreaded);
	      
	      //oldviewer->setCamera(nullptr); // need to clear the camera before the oldviewer goes away lest it kill our graphics contexts
	    }
	    //renderer->Viewer->setCamera(new osg::Camera());
	    //renderer->Viewer->getCamera()->setGraphicsContext(renderer->GraphicsWindow);
	    //renderer->Viewer->getCamera()->setViewport(0,0,0,0);
	    //renderer->Viewer->setCamera(oldcamera);
	    //renderer->Camera = renderer->Viewer->getCamera();
	    
	    assert(renderer->GraphicsWindow->getState());
	    osg::ref_ptr<osg_layerwindow> LW=(dynamic_cast<osg_layerwindow *>(renderer->GraphicsWindow.get()));
	    LW->Viewer=renderer->Viewer;
	    LW->predraw->Viewer=renderer->Viewer;
	    LW->postdraw->Viewer=renderer->Viewer;
	    //LW->setup_camera(renderer->Viewer->getCamera());
	    //LW->setState(new osg::State());
	    //LW->getState()->setGraphicsContext(LW);
	    //LW->getState()->setContextID(osg::GraphicsContext::createNewContextID());
	    */
	    
	    /*
            osg::ref_ptr<osg_layerwindow> LW=new osg_layerwindow(renderer->Viewer,nullptr,compositor_width,compositor_height,false);
	    
	    LW->setDefaultFboId(LayerDefaultFramebufferObject);
	    
	    renderer->Viewer->getCamera()->setGraphicsContext(LW);
	    renderer->Viewer->getCamera()->setViewport(0,0,0,0);
	    //LayerViewer->getCamera()->setViewport(new osg::Viewport(0,0,compositor_width,compositor_height));	 (let the renderer set the viewport so it can detect modifications
	    
	    LW->setup_camera(renderer->Viewer->getCamera());
	    

	    renderer=std::make_shared<osg_2d_renderer>(renderer->Viewer,LW,display_req.second->channelpath);
	    */

	    

	    LayerViewer = dynamic_cast<osgViewerCompat34 *>(renderer->Viewer.get());
	    assert(LayerViewer);
	    if (resizing) {
	      // let the renderer do the resize itself because that way it can detect
	      // size changes and force a rerender (via "modified" bool)
	      //renderer->GraphicsWindow->resized(0,0,compositor_width,compositor_height);
	      //LayerViewer->getCamera()->setViewport(new osg::Viewport(0,0,compositor_width,compositor_height));
	      
	    }
	  }
	}
	
	// perform rendering
	std::shared_ptr<osg_rendercacheentry> cacheentry;
	std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> locks_required;
	bool modified;
	
	std::tie(cacheentry,locks_required,modified) = renderer->prepare_render(display_transforms->with_display_transforms,RenderCache,display_reqs,compositor_width,compositor_height);
	
	// rerender either if there is a modification to the tree, or if we have OSG events (such as rotating, etc)
	snde_debug(SNDE_DC_RENDERING,"compositor about to render %s: 0x%llx 0x%llx modified=%d; empty=%d",display_req.second->channelpath.c_str(),(unsigned long long)cacheentry.get(),(unsigned long long)renderer->EventQueue.get(),(int)modified,(int)renderer->EventQueue->empty());
	if (cacheentry && (modified || !renderer->EventQueue->empty())) {
	  {
	    std::shared_ptr<recdatabase> recdb_strong(recdb);
	    if (recdb_strong) {
	      rwlock_token_set frame_locks = recdb_strong->lockmgr->lock_recording_refs(locks_required,false /*bool gpu_access */); // gpu_access is false because that is only needed for gpgpu calculations like OpenCL where we might be trying to map the entire scene data in one large all-encompassing array

	      //if (display_req.second->channelpath=="/graphics/projection") {
	      //osgDB::writeNodeFile(*renderer->Viewer->getSceneData(),"/tmp/projection.osg");
		//std::cout << "ViewMatrix:\n " << Eigen::Map<Eigen::Matrix4d>(renderer->Camera->getViewMatrix().ptr()) << "\n";
		//std::cout << "InverseViewMatrix:\n " << Eigen::Map<Eigen::Matrix4d>(renderer->Camera->getInverseViewMatrix().ptr()) << "\n";
	      //}
	      
	      renderer->frame();
	    }
	  }
	  // Push a dummy event prior to the frame on the queue
	  // without this we can't process events on our pseudo-GraphicsWindow because
	  // osgGA::EventQueue::takeEvents() looks for an event prior to the cutoffTime
	  // when selecting events to take. If it doesn't find any then you don't get any
	  // events (?).
	  // The cutofftime comes from renderer->Viewer->_frameStamp->getReferenceTime()
	  osg::ref_ptr<osgGA::Event> dummy_event = new osgGA::Event();
	  dummy_event->setTime(renderer->Viewer->getFrameStamp()->getReferenceTime()-1.0);
	  renderer->EventQueue->addEvent(dummy_event);
	  
	  //std::tie(cacheentry,modified) = renderer->prepare_render(display_transforms->with_display_transforms,RenderCache,display_reqs,compositor_width,compositor_height);
	  //renderer->frame();
	  
	  // store our generated texture and its ID
	  osg::ref_ptr<osg::Texture2D> generated_texture = dynamic_cast<osg_layerwindow *>(renderer->GraphicsWindow.get())->outputbuf;
	  layer_rendering_rendered_textures->erase(display_req.second->channelpath);
	  
	  layer_rendering_rendered_textures->emplace(std::piecewise_construct,
						     std::forward_as_tuple(display_req.second->channelpath),
						     std::forward_as_tuple(std::make_pair(generated_texture,generated_texture->getTextureObject(renderer->GraphicsWindow->getState()->getContextID())->id())));
	  

	  if (LayerViewer->compat34GetRequestContinousUpdate()) {//(Viewer->getRequestContinousUpdate()) { // Manipulator->isAnimating doesn't work for some reason(?)
	    new_request_continuous_update = true; 
	  }
	}
	
	
      }

      RenderCache->erase_obsolete();
      
      request_continuous_update = new_request_continuous_update;
    } catch (const std::exception &exc) {
      snde_warning("Exception class %s caught in osg_compositor::perform_layer_rendering: %s",typeid(exc).name(),exc.what());
      
      
    }


    
    GLsync (*ext_glFenceSync)(GLenum condition,GLbitfield flags) = (GLsync (*)(GLenum condition, GLbitfield flags))osg::getGLExtensionFuncPtr("glFenceSync");

    GLenum (*ext_glClientWaitSync)(GLsync sync, GLbitfield Flags, GLuint64 timeout) = (GLenum (*)(GLsync sync, GLbitfield Flags, GLuint64 timeout))osg::getGLExtensionFuncPtr("glClientWaitSync");
    
    if (ext_glFenceSync && ext_glClientWaitSync) {
      
      // Wait for all layers to finish rendering
      // This means that we wait here in the rendering thread.
      // otherwise the compositor will have to wait, eating up the
      // main GUI thread
      GLsync syncobj = ext_glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);

      ext_glClientWaitSync(syncobj,0,5000000000*0.01); // wait up to 5 seconds
    }

    adminlock->lock();
  }


  void osg_compositor::perform_compositing(std::unique_lock<std::mutex> *adminlock)
  {
    assert(this_thread_ok_for_locked(SNDE_OSGRCS_COMPOSITING));

    adminlock->unlock();

    
    if (compositor_width != Camera->getViewport()->width() || compositor_height != Camera->getViewport()->height()) {
      GraphicsWindow->getEventQueue()->windowResize(0,0,compositor_width,compositor_height);
      GraphicsWindow->resized(0,0,compositor_width,compositor_height);
      Camera->setViewport(0,0,compositor_width,compositor_height);
      
    }

    //snde_warning("perform_compositing: Empty=%d",(int)GraphicsWindow->getEventQueue()->empty());

    if (enable_shaders) {
      // Start with OSG 3.6 built-in shaders
      //CompositingShaderProgram = new osg::Program();
      //CompositingShaderProgram->addShader(new osg::Shader(osg::Shader::VERTEX, shadergen_vert));
      //CompositingShaderProgram->addShader(new osg::Shader(osg::Shader::FRAGMENT, shadergen_frag));
      
      // Apply ShaderProgram to our camera
      // and add the required diffuseMap uniform
      osg::ref_ptr<osg::StateSet> CameraStateSet = Camera->getOrCreateStateSet();
      //CameraStateSet->setAttribute(CompositingShaderProgram);
      //CameraStateSet->addUniform(new osg::Uniform("diffuseMap",0));

      // Apply ShaderGen stateset transformation to the camera
      // This transforms basic lighting, fog, and texture
      // to shader defines.
      osgUtil::ShaderGenVisitor ShaderGen;
      ShaderGen.assignUberProgram(CameraStateSet);
      // (Alternatively I think this would be equivalent to
      // Camera->accept(ShaderGen);
      ShaderGen.apply(*Camera);
      
      
    }
    
    Camera->setProjectionMatrixAsOrtho(0,compositor_width,0,compositor_height,0.0,10000.0); // check last two parameters 

    osg::ref_ptr<osg::Group> group=new osg::Group();
    double depth=-1.0*(channels_to_display.size()+1);  // start negative to be compatible with usual OpenGL coordinate frame where negative z's are in front of the cameras
    snde_debug(SNDE_DC_RENDERING,"starting compositing loop:");

    group->getOrCreateStateSet()->setMode(GL_DEPTH_TEST,osg::StateAttribute::OFF);

    // Add Graticule
    double horizontal_padding = (compositor_width - display->horizontal_divisions * display->pixelsperdiv) / 2.0;
    double vertical_padding = (compositor_height - display->vertical_divisions * display->pixelsperdiv) / 2.0;
    GraticuleTransform->setMatrix(osg::Matrixd(display->pixelsperdiv / 5.0, 0, 0, 0,
        0, display->pixelsperdiv / 5.0, 0, 0,
        0, 0, 1, 0,
        horizontal_padding + display->pixelsperdiv * display->horizontal_divisions / 2.0 - 0.5, vertical_padding + display->pixelsperdiv * display->vertical_divisions / 2.0 - 0.5, depth - 0.5, 1)); // ***!!! are -0.5's and negative sign in front of layer_index correct?  .... fix here and in transformmtx, above. 
    group->addChild(GraticuleTransform);

    std::vector<osg::ref_ptr<osg::Texture2D>> temporary_texture_references;
    
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
          snde_debug(SNDE_DC_RENDERING, "Compositor: using layer texture object ID #%u", (unsigned)tex_it->second.second);
          osg::ref_ptr<osg::Texture::TextureObject> texobj = new osg::Texture::TextureObject(tex, tex_it->second.second, GL_TEXTURE_2D);

          texobj->setAllocated();
          tex->setTextureObject(GraphicsWindow->getState()->getContextID(), texobj);
          temporary_texture_references.push_back(tex); // store the reference so we can clear it prior to deletion and avoid the annoying message


          if (displaychan->render_mode == SNDE_DCRM_IMAGE)
          {
              snde_debug(SNDE_DC_RENDERING, "borderbox: width=%d,height=%d", compositor_width, compositor_height);
              /* !!!*** NOTE: Apparently had trouble previously with double precision vs single precision arrays (?) */

              // Z position of border is -0.5 relative to image, so it appears on top
              // around edge

              snde_debug(SNDE_DC_RENDERING, "borderbox: spatial x = %d; y = %d; width = %d; height = %d", dispreq->spatial_position->x, dispreq->spatial_position->y, dispreq->spatial_position->width, dispreq->spatial_position->height);

              float borderbox_xleft = dispreq->spatial_position->x - borderwidthpixels / 2.0;
              if (borderbox_xleft < 0.5) {
                  borderbox_xleft = 0.5;
              }

              float borderbox_xright = dispreq->spatial_position->x + dispreq->spatial_position->width + borderwidthpixels / 2.0;
              if (borderbox_xright > compositor_width - 0.5) {
                  borderbox_xright = compositor_width - 0.5;
              }

              float borderbox_ybot = dispreq->spatial_position->y - borderwidthpixels / 2.0;
              if (borderbox_ybot < 0.5) {
                  borderbox_ybot = 0.5;
              }

              float borderbox_ytop = dispreq->spatial_position->y + dispreq->spatial_position->height + borderwidthpixels / 2.0;
              if (borderbox_ytop > compositor_height - 0.5) {
                  borderbox_ytop = compositor_height - 0.5;
              }


              snde_debug(SNDE_DC_RENDERING, "borderbox: xleft=%f,ybot=%f,xright=%f,ytop=%f", borderbox_xleft, borderbox_ybot, borderbox_xright, borderbox_ytop);

              osg::ref_ptr<osg::Vec3Array> BorderCoords = new osg::Vec3Array(8);
              (*BorderCoords)[0] = osg::Vec3(borderbox_xleft, borderbox_ybot, depth + 0.5);
              (*BorderCoords)[1] = osg::Vec3(borderbox_xright, borderbox_ybot, depth + 0.5);

              (*BorderCoords)[2] = osg::Vec3(borderbox_xright, borderbox_ybot, depth + 0.5);
              (*BorderCoords)[3] = osg::Vec3(borderbox_xright, borderbox_ytop, depth + 0.5);

              (*BorderCoords)[4] = osg::Vec3(borderbox_xright, borderbox_ytop, depth + 0.5);
              (*BorderCoords)[5] = osg::Vec3(borderbox_xleft, borderbox_ytop, depth + 0.5);

              (*BorderCoords)[6] = osg::Vec3(borderbox_xleft, borderbox_ytop, depth + 0.5);
              (*BorderCoords)[7] = osg::Vec3(borderbox_xleft, borderbox_ybot, depth + 0.5);

              osg::ref_ptr<osg::Geode> bordergeode = new osg::Geode();
              osg::ref_ptr<osg::Geometry> bordergeom = new osg::Geometry();

              osg::ref_ptr<osg::DrawArrays> borderdraw = new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 8); // # is number of lines * number of coordinates per line
              bordergeom->setVertexArray(BorderCoords);
              bordergeom->addPrimitiveSet(borderdraw);
              bordergeode->addDrawable(bordergeom);
              bordergeode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
              bordergeode->getOrCreateStateSet()->setAttributeAndModes(new osg::LineWidth(borderwidthpixels), osg::StateAttribute::ON);
              group->addChild(bordergeode);

              osg::ref_ptr<osg::Vec4Array> BorderColorArray = new osg::Vec4Array();
              size_t ColorIdx = ColorIdx_by_channelpath.at(displaychan->FullName);
              BorderColorArray->push_back(osg::Vec4(RecColorTable[ColorIdx].R, RecColorTable[ColorIdx].G, RecColorTable[ColorIdx].B, 1.0));
              bordergeom->setColorArray(BorderColorArray, osg::Array::BIND_OVERALL);
              bordergeom->setColorBinding(osg::Geometry::BIND_OVERALL);

          }

    // Image coordinates, from actual corners, counterclockwise,
	// Two triangles    
	osg::ref_ptr<osg::Vec3Array> ImageCoords=new osg::Vec3Array(6);
	osg::ref_ptr<osg::Vec2Array> ImageTexCoords=new osg::Vec2Array(6);


	float image_xleft = dispreq->spatial_position->x;
	if (image_xleft < 0.0) {
	  image_xleft = 0.0;
	}
	
	float image_xright = dispreq->spatial_position->x+dispreq->spatial_position->width;
	if (image_xright > compositor_width) {
	  image_xright = compositor_width;
	}
	
	float image_ybot = dispreq->spatial_position->y;
	if (image_ybot < 0.0) {
	  image_ybot = 0.0;
	}
	
	float image_ytop = dispreq->spatial_position->y+dispreq->spatial_position->height;
	if (image_ytop > compositor_height) {
	  image_ytop = compositor_height;
	}
	
	(*ImageCoords)[0]=osg::Vec3(image_xleft,
				     image_ybot,
				     depth);
	(*ImageCoords)[1]=osg::Vec3(image_xright,
				     image_ybot,
				     depth);
	(*ImageCoords)[2]=osg::Vec3(image_xleft,
				     image_ytop,
				     depth);
	
	(*ImageTexCoords)[0]=osg::Vec2(0,0);
	(*ImageTexCoords)[1]=osg::Vec2(1,0);
	(*ImageTexCoords)[2]=osg::Vec2(0,1);
      
	// upper-right triangle 
	(*ImageCoords)[3]=osg::Vec3(image_xright,
				    image_ytop,
				    depth);
	(*ImageCoords)[4]=osg::Vec3(image_xleft,
				    image_ytop,
				    depth);
	(*ImageCoords)[5]=osg::Vec3(image_xright,
				    image_ybot,
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
	
	snde_debug(SNDE_DC_RENDERING,"osg_compositor::perform_compositing(): Rendered channel %s",displaychan->FullName.c_str());
	
	depth++; // next layer is shallower (less negative)
      } else {
	snde_debug(SNDE_DC_RENDERING,"osg_compositor::perform_compositing(): Did not find rendered layer for channel %s",displaychan->FullName.c_str());
      }
      
    }

        
    //snde_warning("perform_compositing2: Empty=%d",(int)GraphicsWindow->getEventQueue()->empty());

    //Viewer->setSceneData(group); setSceneData() seems to clear our event queue, so instead we swap out the contents of a persistent group (RootGroup) that was set as the scene data in the constructor
    
    
    if (RootGroup->getNumChildren()) {
      RootGroup->removeChildren(0,1);
    }
    RootGroup->addChild(group);
    

    if (enable_shaders) {

      // Apply use of shaders instead of old-style lighting and texture to the modified tree
      osgUtil::ShaderGenVisitor ShaderGen;
      // This transforms basic lighting, fog, and texture
      // to shader defines.

      // The shader stateset was already applied to
      // the camera in the constructor. 

      // (Alternatively I think this would be equivalent to
      /// ShaderGen.apply(RootTransform);
      RootGroup->accept(ShaderGen);

    }
    
    snde_debug(SNDE_DC_RENDERING,"Compositor drawing frame: numlayers = %d, Empty=%d",(int)group->getNumChildren(),(int)GraphicsWindow->getEventQueue()->empty());
    Viewer->frame();
    //snde_debug(SNDE_DC_RENDERING,"Compositor frame draw completed: Empty=%d",(int)GraphicsWindow->getEventQueue()->empty());

    for (auto && texobj: temporary_texture_references) {
      // undo the tex->setTextureObject() from perform_compositing(). This eliminates the nasty message about releaseTextureObject() being unimplemented
      texobj->setTextureObject(GraphicsWindow->getState()->getContextID(),nullptr);
    }
    temporary_texture_references.clear();
    
    adminlock->lock();
    snde_debug(SNDE_DC_RENDERING,"Compositing complete; need_recomposite=%d",(int)need_recomposite);


    
    

    
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

  void osg_compositor::wake_up_ondemand_locked(std::unique_lock<std::mutex> *adminlock)
  {
    if (threaded) {
      execution_notify.notify_all();
    }
  }
  
  void osg_compositor::wake_up_renderer_locked(std::unique_lock<std::mutex> *adminlock)
  {
    if (threaded || enable_threaded_opengl) {
      execution_notify.notify_all();
    }

  }

  void osg_compositor::wake_up_compositor_locked(std::unique_lock<std::mutex> *adminlock)
  {

  }
  
  void osg_compositor::clean_up_renderer_locked(std::unique_lock<std::mutex> *adminlock)
  {
    assert(this_thread_ok_for_locked(SNDE_OSGRCS_RENDERING_CLEANUP)); // Cleanup OK really necessary for qt_osg_compositor::clean_up_renderer_locked because that deletes RenderContext and DummyOffscreenSurface
    
    RenderCache = nullptr;
    renderers = nullptr;
    layer_rendering_rendered_textures = nullptr;
    
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

    if (return_if_idle && next_state==SNDE_OSGRCS_WAITING && !need_recomposite && !need_rerender) {
      // idle and return_if_idle flag
      return;
    }
    // WARNING: This function can be a bit confusing because it is called both from the GUI thread
    // and the ondemand/rendering thread. Threads pick up the next_state based on what the
    // particular thread is allowed to do (based on the responsibility_mapping as evaluated in
    // this_thread_ok_for()). They may also have to trigger the other thread to wake up (see
    // wake_up...() methods).
    //
    // It's a bit complicated (especially with some methods overridden in qt_osg_compositor),
    // but it allows a single codebase to handle both threaded and non-threaded rendering,
    // as well as layering a GUI on top that doesn't cooperate particularly well with OSG. 
    
    while (next_state != SNDE_OSGRCS_EXIT) {
      snde_debug(SNDE_DC_RENDERING,"start ttofl: %d next_state:%d need_recomposite: %d, need_rerender: %d",this_thread_ok_for_locked(next_state),next_state,need_recomposite,need_rerender);
      while ((!this_thread_ok_for_locked(next_state)  && !(next_state==SNDE_OSGRCS_WAITING && (need_recomposite || need_rerender))) || (next_state==SNDE_OSGRCS_WAITING && !need_recomposite && !need_rerender)) {
	if ( (wait && loop_forever) || (wait && !loop_forever && !executed_something)) {
	  snde_debug(SNDE_DC_RENDERING,"osg_compositor tid %d waiting",std::this_thread::get_id());
	  execution_notify.wait(adminlock);
	} else {
	  return; // caller said not to wait
	}
	snde_debug(SNDE_DC_RENDERING,"ttofl: %d next_state:%d need_recomposite: %d, need_rerender: %d",this_thread_ok_for_locked(next_state),next_state,need_recomposite,need_rerender);
      }
      snde_debug(SNDE_DC_RENDERING,"osg_compositor tid %d wakeup",std::this_thread::get_id());
      
      if (next_state == SNDE_OSGRCS_WAITING) {
	if (need_recomposite) {
	  next_state = SNDE_OSGRCS_COMPOSITING;
	  need_recomposite=false;
	  executed_something=true;
	  
	}
	if (need_rerender) {
	  // need_update overrides need_recomposite 
	  next_state = SNDE_OSGRCS_ONDEMANDCALCS;
	  need_rerender = false;
	  executed_something=true; 
	}
	if (next_state==SNDE_OSGRCS_ONDEMANDCALCS) {
	  wake_up_ondemand_locked(&adminlock);
	} else if (next_state == SNDE_OSGRCS_COMPOSITING) {
	  wake_up_compositor_locked(&adminlock);
	}
      } else if (next_state == SNDE_OSGRCS_ONDEMANDCALCS) {
	//try {
	  perform_ondemand_calcs(&adminlock);
          //} catch(const std::exception &e) {
	  //snde_warning("Exception in ondemand rendering calculations: %s",e.what());
          //}
	executed_something=true;
	if (next_state == SNDE_OSGRCS_ONDEMANDCALCS) {
	  // otherwise we don't want to interrupt a cleanup/exit command
	  next_state = SNDE_OSGRCS_RENDERING;
	  if (threaded && !enable_threaded_opengl) { 
	    wake_up_renderer_locked(&adminlock);
	  }
	}
	
      } else if (next_state==SNDE_OSGRCS_RENDERING) {
	try {
	  perform_layer_rendering(&adminlock);
	} catch(const std::exception &e) {
	  snde_warning("Exception in compositor layer rendering operations: %s",e.what());
	}
	executed_something=true; 
	if (next_state == SNDE_OSGRCS_RENDERING) {
	  // otherwise we don't want to interrupt a cleanup/exit command
	  next_state = SNDE_OSGRCS_COMPOSITING;
	  if (threaded && enable_threaded_opengl) {
	    wake_up_compositor_locked(&adminlock);
	  }
	}
	
      } else if (next_state==SNDE_OSGRCS_COMPOSITING) {
	try {
	  need_recomposite=false; // without this there is a race condition above and we can end up with need_recomposite still set afterward, potentially.
	  perform_compositing(&adminlock);
	} catch(const std::exception &e) {
	  snde_warning("Exception in compositing operations: %s",e.what());
	}
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

	clean_up_renderer_locked(&adminlock);

	// This code moved into clean_up_renderer_lockes()
	//RenderCache = nullptr;
	//renderers = nullptr;
	//layer_rendering_rendered_textures = nullptr;
	
	next_state = SNDE_OSGRCS_EXIT;	
      }
    

    
      //execution_notify.notify_all();
    
      if (!loop_forever && executed_compositing) {
	// finished a pass
	return; 
      }
    }
    
  }
  
  
  void osg_compositor::worker_code()
  {
    
    dispatch(false,true,true);
  }

  void osg_compositor::_start_worker_thread(std::unique_lock<std::mutex> *adminlock)
  // adminlock is held by the passed unique_lock 
  {
    if (threaded) {
      worker_thread = std::make_shared<std::thread>([ this ]() { this->worker_code(); });
      set_thread_name(worker_thread.get(),"snde2 osg_comp worker");

      worker_thread_id = std::make_shared<std::thread::id>(worker_thread->get_id());
    } else {
      worker_thread_id = std::make_shared<std::thread::id>(std::this_thread::get_id());
    }
    
    threads_started=true;

    // Note: worker_thread will still be waiting for us to setup the thread_responsibilities
  }
  
  void osg_compositor::_join_worker_thread()
  {
    if (threaded && threads_started) {
      worker_thread->join();
      worker_thread=nullptr;
    }
    threads_started=false;
    worker_thread_id=nullptr; 
    
  }


  void osg_compositor::resize_compositor(int width, int height)
  {

    // ***!!!! BUG: compositor gets its size through resize_width and
    // resize_height after a proper resize operation here,
    // but display_requirements.cpp pulls from
    // display->drawareawidth and display->drawareaheight, which may be
    // different and aren't sync'd properly except we just force it here
    {
      std::lock_guard<std::mutex> display_admin(display->admin);
      display->drawareawidth = width;
      display->drawareaheight = height;
    }
    // different and aren't sync'd properly
    std::lock_guard<std::mutex> adminlock(admin);

    need_resize=true;
    resize_width=width;
    resize_height=height;
  }

  snde_orientation3 osg_compositor::get_camera_pose(std::string channel_path)
  // get the camera pose (or a null orientation) for the given channel
  {

    osg::Matrixd CamPose;
    {
      std::lock_guard<std::mutex> compositor_admin(admin); // required for access to renderers
      
      auto renderer_it = renderers->find(channel_path);
      if (renderer_it != renderers->end()) {

	//CamPose = renderer_it->second->Camera->getInverseViewMatrix(); // camera pose is the inverse of the view matrix
	CamPose = renderer_it->second->GetLastCameraPose();
      } else {
	// channel not found - check for FutureChannelCamPose
	auto FutCamPoseIt = FutureChannelCamPose.find(channel_path);
	if (FutCamPoseIt != FutureChannelCamPose.end()) {
	  CamPose = FutCamPoseIt->second;
	} else {

	  // return null orientation
	  snde_orientation3 null;
	  snde_null_orientation3(&null);
	  return null;
	}
      }
      
    }
    
    osg::Vec3d translation;
    osg::Quat rotation;
    osg::Vec3d scale;
    osg::Quat scale_orientation;

    CamPose.decompose(translation,rotation,scale,scale_orientation);

    snde_orientation3 retval;
    retval.offset.coord[0]=translation.x();
    retval.offset.coord[1]=translation.y();
    retval.offset.coord[2]=translation.z();
    retval.offset.coord[3]=1.0;

    retval.quat.coord[1]=rotation.x();
    retval.quat.coord[2]=rotation.y();
    retval.quat.coord[3]=rotation.z();
    retval.quat.coord[0]=rotation.w();

    return retval;
  }


  void osg_compositor::set_camera_pose(std::string channel_path,const snde_orientation3 &newpose)
  // set the camera pose for the given channel
  {
    {
      std::lock_guard<std::mutex> compositor_admin(admin); // required for access to renderers
      
      snde_coord4 rotmtx[4];
      orientation_build_rotmtx(newpose,rotmtx);
      osg::Matrixd OSGCamPose(&rotmtx[0].coord[0]);

      
      auto renderer_it = renderers->find(channel_path);
      if (renderer_it != renderers->end()) {
	renderer_it->second->AssignNewCameraPose(OSGCamPose);

      } else {
	// channel not found - check for FutureChannelCamPose
	auto FutCamPoseIt = FutureChannelCamPose.find(channel_path);
	if (FutCamPoseIt != FutureChannelCamPose.end()) {
	  FutCamPoseIt->second = OSGCamPose;
	} else {
	  FutureChannelCamPose.emplace(channel_path,OSGCamPose);
	}
      }

    }
  }


  snde_coord osg_compositor::get_rotation_center_dist(std::string channel_path) // get the viewer rotation center
  {
    snde_coord RotCenterDist;
    {
      std::lock_guard<std::mutex> compositor_admin(admin); // required for access to renderers
      
      auto renderer_it = renderers->find(channel_path);
      if (renderer_it != renderers->end()) {
	RotCenterDist = renderer_it->second->GetLastRotationCenterDist();
	
      } else {
	// channel not found - check for FutureChannelRotationCenterDist
	auto FutRotCtrIt = FutureChannelRotationCenterDist.find(channel_path);
	if (FutRotCtrIt != FutureChannelRotationCenterDist.end()) {
	  RotCenterDist = FutRotCtrIt->second;
	} else {

	  // channel not found -- return 1 meter distance
	  return 1.0;
	}
      }
      
    }
    

    return RotCenterDist;

  }
  void osg_compositor::set_rotation_center_dist(std::string channel_path,snde_coord newcenterdist)
  {
    {
      std::lock_guard<std::mutex> compositor_admin(admin); // required for access to renderers
      
      
      auto renderer_it = renderers->find(channel_path);
      if (renderer_it != renderers->end()) {
	renderer_it->second->AssignNewRotationCenterDist(newcenterdist);
	
      } else {
	// channel not found - check for FutureChannelRotCenter
	auto FutRotCtrIt = FutureChannelRotationCenterDist.find(channel_path);
	if (FutRotCtrIt != FutureChannelRotationCenterDist.end()) {
	  FutRotCtrIt->second = newcenterdist;
	} else {
	  FutureChannelRotationCenterDist.emplace(channel_path,newcenterdist);
	}

      }
      
    }

  }

  void osg_compositor::start()
  {
    
    std::unique_lock<std::mutex> adminlock(admin);
    
    if (!eventhandler) {
      adminlock.unlock();
      eventhandler = new osg_compositor_eventhandler(this,display);
      adminlock.lock();
      Viewer->addEventHandler(eventhandler);
      
      Viewer->getCamera()->setAllowEventFocus(false); // prevent camera from eating our events and interfering
    }
  
    if (!threads_started) {
      _start_worker_thread(&adminlock); // sets threads_started

      // assign thread responsibilities 
      if (threaded) {
	
	
	if (enable_threaded_opengl) {
	  responsibility_mapping.emplace(*worker_thread_id,std::set<int>{SNDE_OSGRCS_WAITING,SNDE_OSGRCS_ONDEMANDCALCS,SNDE_OSGRCS_RENDERING,SNDE_OSGRCS_RENDERING_CLEANUP,SNDE_OSGRCS_EXIT});
	  
	  responsibility_mapping.emplace(std::this_thread::get_id(),std::set<int>{SNDE_OSGRCS_COMPOSITING,SNDE_OSGRCS_COMPOSITING_CLEANUP,SNDE_OSGRCS_EXIT});
	} else {
	  responsibility_mapping.emplace(*worker_thread_id,std::set<int>{SNDE_OSGRCS_WAITING,SNDE_OSGRCS_ONDEMANDCALCS,SNDE_OSGRCS_EXIT});
	  responsibility_mapping.emplace(std::this_thread::get_id(),std::set<int>{SNDE_OSGRCS_COMPOSITING,SNDE_OSGRCS_RENDERING,SNDE_OSGRCS_COMPOSITING_CLEANUP,SNDE_OSGRCS_RENDERING_CLEANUP,SNDE_OSGRCS_EXIT});
	}
	
	
      } else {
	responsibility_mapping.emplace(std::this_thread::get_id(),std::set<int>{SNDE_OSGRCS_WAITING,SNDE_OSGRCS_ONDEMANDCALCS,SNDE_OSGRCS_COMPOSITING,SNDE_OSGRCS_RENDERING,SNDE_OSGRCS_COMPOSITING_CLEANUP,SNDE_OSGRCS_RENDERING_CLEANUP,SNDE_OSGRCS_EXIT});
	
      }
      execution_notify.notify_all(); // notify subthread that responsibility_mapping is set-up 

    }

    
  }

  void osg_compositor::stop()
  {
    // Get us into cleanup mode. In case this is a second stop call,
    // if it would depend on the worker thread, we only do it if the worker
    // thread is still alive
    if (!threaded || (threaded && threads_started)) {
      {
	std::lock_guard<std::mutex> adminlock(admin);
	next_state = SNDE_OSGRCS_COMPOSITING_CLEANUP;
	execution_notify.notify_all();
      }

      dispatch(false,true,true); // perform any cleanup actions that are our thread's responsibility
    }
    _join_worker_thread();
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
