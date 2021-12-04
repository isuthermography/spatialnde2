
#include "snde/openscenegraph_geom_renderer.hpp"
#include "snde/rec_display.hpp"

namespace snde {


  
  // ***!!! Should make the channel to display a parameter to the renderere ***!!!

  osg_3d_renderer::osg_3d_renderer(osg::ref_ptr<osgViewer::Viewer> Viewer, // use an osgViewerCompat34()
				   osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
				   std::shared_ptr<osg_rendercache> RenderCache,
				   std::string channel_path) :
    channel_path(channel_path),
    GraphicsWindow(GraphicsWindow),
    //Viewer(new osgViewer::Viewer()),
    Viewer(Viewer),
    Manipulator(new osgGA::TrackballManipulator()),
    Camera(Viewer->getCamera()),
    //RootNode(RootNode),
    RenderCache(RenderCache)
  {
    
    Camera->setGraphicsContext(GraphicsWindow);
    
    // set background color to blueish
    Camera->setClearColor(osg::Vec4(.1,.1,.3,1.0));    


    // need to enable culling so that linesegmentintersector (openscenegraph_picker)
    // behavior matches camera behavior
    // (is this efficient?)
    Camera->setComputeNearFarMode( osg::CullSettings::COMPUTE_NEAR_FAR_USING_PRIMITIVES );
    Camera->setCullingMode(osg::CullSettings::ENABLE_ALL_CULLING);

    Viewer->setThreadingModel(osgViewer::Viewer::SingleThreaded); // We handle threading ourselves
    Viewer->setCameraManipulator(Manipulator);
    //Viewer->setRunFrameScheme(osg::ON_DEMAND); // ***!!! With this OSG looks at whether it thinks a new render is needed based on scene graph changes and only renders if necessary.

    //Viewer->setSceneData(RootNode);

    
    /* Two dimensional initialization */
    //Transform = new osg::MatrixTransform();
    //Transform->...
    //Viewer->setSceneData(Transform);
    //group = new osg::group();
    //Transform->addChild(Geode)
    //Viewer->setSceneData(group);


    // This stuff needs to be moved into renderer->perform_render
    // Caller should set camera viewport,
    // implement SetProjectionMatrix(),
    // SetTwoDimensional()
    // and make initial calls to those functions
    // from their constructor,
    // then call Viewer->realize();
    Viewer->setCameraManipulator(nullptr);
    Camera->setViewMatrix(osg::Matrixd::identity());

    Viewer->realize();


      
  }

  
  
  void osg_3d_renderer::perform_render(std::shared_ptr<recdatabase> recdb,
					  //std::shared_ptr<recstore_display_transforms> display_transforms,
					  std::shared_ptr<recording_set_state> with_display_transforms,
					  //std::shared_ptr<display_channel> channel_to_display,
					  //std::string channel_path,
					  std::shared_ptr<display_info> display,
					  const std::map<std::string,std::shared_ptr<display_requirement>> &display_reqs,
					  double left, // left of viewport in channel horizontal units
					  double right, // right of viewport in channel horizontal units
					  double bottom, // bottom of viewport in channel vertical units
					  double top, // top of viewport in channel vertical units
					  size_t width, // width of viewport in pixels
					  size_t height) // height of viewport in pixels
  {
    // look up render cache.
    std::shared_ptr<display_requirement> display_req = display_reqs.at(channel_path);
    osg_renderparams params{
      recdb,
      RenderCache,
      with_display_transforms,
      display,
      
      left,
      right,
      bottom,
      top,
      width,
      height,
      
    };
    
    std::shared_ptr<osg_rendercacheentry> renderentry = RenderCache->GetEntry(params,display_req);
    if (renderentry) {
      std::shared_ptr<osg_rendercachegroupentry> rendergroup = std::dynamic_pointer_cast<osg_rendercachegroupentry>(renderentry);
      
      if (rendergroup) {
	if (Viewer->getSceneData() != rendergroup->osg_group){
	  //group=imagegroup->osg_group;
	  Viewer->setSceneData(rendergroup->osg_group);
	}
	/*
	  if (Geode && Geode != imagegeode) {
	  Transform->removechild(Geode);
	  }
	  if (Geode != imagegeode) {
	  Geode=imagegeode;
	  Transform->addchild(imagegeode);
	  }*/
	
	/* double left;
	   double right;
	   double bottom;
	   double top;
	*/
	
	Camera->setProjectionMatrixAsOrtho(left,right,bottom,top,-10.0,1000.0);
	Camera->setViewport(0,0,width,height);
      } else {
	snde_warning("openscenegraph_3d_renderer: cache entry not convertable to an osg_group rendering channel %s",channel_path.c_str());
      }
    } else {
      snde_warning("openscenegraph_3d_renderer: cache entry not available rendering channel %s",channel_path.c_str());
      
    }
    Viewer->frame();
    
  }
  
  
  
};
