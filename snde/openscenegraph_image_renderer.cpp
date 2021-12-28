
#include "snde/openscenegraph_image_renderer.hpp"
#include "snde/rec_display.hpp"
#include "snde/display_requirements.hpp"

namespace snde {


  


  osg_image_renderer::osg_image_renderer(osg::ref_ptr<osgViewer::Viewer> Viewer, // use an osgViewerCompat34()
					 osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
					 std::string channel_path) :
    osg_renderer(Viewer,GraphicsWindow,channel_path,SNDE_DRRT_IMAGE)
  {
    
    EventQueue=GraphicsWindow->getEventQueue();
    Camera->setGraphicsContext(GraphicsWindow);
    
    // set background color to blueish
    Camera->setClearColor(osg::Vec4(.1,.1,.3,1.0));    
    Camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Camera->setCullingMode(osg::CullSettings::NO_CULLING); // otherwise triangles we use for rendering can get culled as we zoom in far enough that their vertices go off-screen?
    
    Viewer->setThreadingModel(osgViewer::Viewer::SingleThreaded); // OSG single threaded mode is REQUIRED (!!!)
    
    //Viewer->setRunFrameScheme(osg::ON_DEMAND); // ***!!! With this OSG looks at whether it thinks a new render is needed based on scene graph changes and only renders if necessary.
    
    Viewer->setCameraManipulator(nullptr);
    Camera->setViewMatrix(osg::Matrixd::identity());

    Viewer->realize();
      
  }


  // actually rendering is done by osg_image_renderer::frame() which just calls Viewer->frame()
  
  std::tuple<std::shared_ptr<osg_rendercacheentry>,bool>
  osg_image_renderer::prepare_render(//std::shared_ptr<recdatabase> recdb,
				     std::shared_ptr<recording_set_state> with_display_transforms,
				     //std::shared_ptr<display_info> display,
				     std::shared_ptr<osg_rendercache> RenderCache,
				     const std::map<std::string,std::shared_ptr<display_requirement>> &display_reqs,
				     size_t width, // width of viewport in pixels
				     size_t height) // height of viewport in pixels
  // returns cache entry used, and bool that is true if it is new or modified
  {
    // look up render cache.
    std::map<std::string,std::shared_ptr<display_requirement>>::const_iterator got_req;

    got_req=display_reqs.find(channel_path);
    if (got_req==display_reqs.end()) {
      snde_warning("openscenegraph_image_renderer: Was not possible to transform channel \"%s\" into something renderable",channel_path.c_str());
      return std::make_pair(nullptr,true);
    }
    
    std::shared_ptr<display_requirement> display_req =got_req->second;
    osg_renderparams params{
      //recdb,
      RenderCache,
      with_display_transforms,
      //display,
      
      display_req->spatial_bounds->left,
      display_req->spatial_bounds->right,
      display_req->spatial_bounds->bottom,
      display_req->spatial_bounds->top,
      width,
      height,
      
    };

    snde_warning("image render width bounds: left: %f right: %f bottom: %f top: %f",
		 display_req->spatial_bounds->left,
		 display_req->spatial_bounds->right,
		 display_req->spatial_bounds->bottom,
		 display_req->spatial_bounds->top);


    std::shared_ptr<osg_rendercacheentry> imageentry;
    bool modified=false;
    
    if (display_req->spatial_bounds->bottom >= display_req->spatial_bounds->top ||
	display_req->spatial_bounds->left >= display_req->spatial_bounds->right) {
      // negative or zero display area

      if (RootGroup->getNumChildren()) {

	RootGroup->removeChildren(0,RootGroup->getNumChildren());
      }

      modified = true; 
    } else { // Positive display area 
      std::tie(imageentry,modified) = RenderCache->GetEntry(params,display_req);
    
      /// NOTE: to adjust size, first send event, then 
      //   change viewport:
      if (display_req->spatial_position->width != Camera->getViewport()->width() || display_req->spatial_position->height != Camera->getViewport()->height()) {
	GraphicsWindow->getEventQueue()->windowResize(0,0,display_req->spatial_position->width,display_req->spatial_position->height);
	GraphicsWindow->resized(0,0,display_req->spatial_position->width,display_req->spatial_position->height);
	Camera->setViewport(0,0,display_req->spatial_position->width,display_req->spatial_position->height);
	modified = true;
	snde_warning("image position: width: %d height: %d",
		     display_req->spatial_position->width,
		     display_req->spatial_position->height);

      }
      
      Camera->setProjectionMatrixAsOrtho(display_req->spatial_bounds->left,display_req->spatial_bounds->right,display_req->spatial_bounds->bottom,display_req->spatial_bounds->top,-10.0,1000.0);
      
      
      
      if (imageentry) {
	
	std::shared_ptr<osg_rendercachegroupentry> imagegroup = std::dynamic_pointer_cast<osg_rendercachegroupentry>(imageentry); 
	
	if (imagegroup) {
	  //if (Viewer->getSceneData() != imagegroup->osg_group){
	  //  //group=imagegroup;
	  //  Viewer->setSceneData(imagegroup->osg_group);
	  //}
	  if (RootGroup->getNumChildren() && RootGroup->getChild(0) != imagegroup->osg_group) {
	    RootGroup->removeChildren(0,1);
	  }
	  if (!RootGroup->getNumChildren()) {
	    RootGroup->addChild(imagegroup->osg_group);
	  }
	  
	  if (!Viewer->getSceneData()) {
	    Viewer->setSceneData(RootGroup);
	  }
	  
	} else {
	  snde_warning("openscenegraph_image_renderer: cache entry not convertable to an osg_group rendering channel \"%s\"",channel_path.c_str());
	}
      } else {
	snde_warning("openscenegraph_image_renderer: cache entry not available rendering channel \"%s\"",channel_path.c_str());
	
      }
	
    }
    
    return std::make_pair(imageentry,modified);
    
  }
  

  
};
