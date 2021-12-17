
#include "snde/openscenegraph_image_renderer.hpp"
#include "snde/rec_display.hpp"
#include "snde/display_requirements.hpp"

namespace snde {


  


  osg_image_renderer::osg_image_renderer(osg::ref_ptr<osgViewer::Viewer> Viewer, // use an osgViewerCompat34()
					 osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
					 std::string channel_path) :
    osg_renderer(Viewer,GraphicsWindow,channel_path,SNDE_DRRT_IMAGE)
  {
    
    Camera->setGraphicsContext(GraphicsWindow);
    
    // set background color to blueish
    Camera->setClearColor(osg::Vec4(.1,.1,.3,1.0));    
    Camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
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



    std::shared_ptr<osg_rendercacheentry> imageentry;
    bool modified;
    std::tie(imageentry,modified) = RenderCache->GetEntry(params,display_req);
    
    
    /// NOTE: to adjust size, first send event, then 
    //   change viewport:
    if (width != Camera->getViewport()->width() || height != Camera->getViewport()->height()) {
      GraphicsWindow->getEventQueue()->windowResize(0,0,width,height);
      GraphicsWindow->resized(0,0,width,height);
      Camera->setViewport(0,0,width,height);
      modified = true; 
    }
    
    Camera->setProjectionMatrixAsOrtho(display_req->spatial_bounds->left,display_req->spatial_bounds->right,display_req->spatial_bounds->bottom,display_req->spatial_bounds->top,-10.0,1000.0);

    

    if (imageentry) {

      std::shared_ptr<osg_rendercachegroupentry> imagegroup = std::dynamic_pointer_cast<osg_rendercachegroupentry>(imageentry); 

      if (imagegroup) {
	if (Viewer->getSceneData() != imagegroup->osg_group){
	  //group=imagegroup;
	  Viewer->setSceneData(imagegroup->osg_group);
	}
    
      } else {
	snde_warning("openscenegraph_image_renderer: cache entry not convertable to an osg_group rendering channel \"%s\"",channel_path.c_str());
      }
    } else {
      snde_warning("openscenegraph_image_renderer: cache entry not available rendering channel \"%s\"",channel_path.c_str());
      
    }
	

    return std::make_pair(imageentry,modified);
    
  }
  

  
};
