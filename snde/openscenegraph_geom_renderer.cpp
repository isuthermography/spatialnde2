
#include "snde/openscenegraph_geom_renderer.hpp"
#include "snde/rec_display.hpp"
#include "snde/display_requirements.hpp"

namespace snde {


  

  osg_geom_renderer::osg_geom_renderer(osg::ref_ptr<osgViewer::Viewer> Viewer, // use an osgViewerCompat34()
				   osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
				   std::string channel_path) :
    osg_renderer(Viewer,GraphicsWindow,channel_path,SNDE_DRRT_GEOMETRY),
    Manipulator(new osgGA::TrackballManipulator())
    //firstrun(true)
  {
    
    //Camera->setGraphicsContext(GraphicsWindow);
    
    // set background color to blueish
    Camera->setClearColor(osg::Vec4(.1,.1,.3,1.0));    
    Camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    

    // need to enable culling so that linesegmentintersector (openscenegraph_picker)
    // behavior matches camera behavior
    // (is this efficient?)
    // NOTE: Having this on made rendering fail !!!***
    //Camera->setComputeNearFarMode( osg::CullSettings::COMPUTE_NEAR_FAR_USING_PRIMITIVES );
    //Camera->setCullingMode(osg::CullSettings::ENABLE_ALL_CULLING);

    Viewer->setThreadingModel(osgViewer::Viewer::SingleThreaded); // We handle threading ourselves... this is REQUIRED
    Viewer->setCameraManipulator(Manipulator);

    Viewer->realize();
  }

  
  
  std::tuple<std::shared_ptr<osg_rendercacheentry>,bool>
  osg_geom_renderer::prepare_render(//std::shared_ptr<recdatabase> recdb,
				  std::shared_ptr<recording_set_state> with_display_transforms,
				  //std::shared_ptr<display_info> display,
				  std::shared_ptr<osg_rendercache> RenderCache,
				  const std::map<std::string,std::shared_ptr<display_requirement>> &display_reqs,
				  size_t width, // width of viewport in pixels
				  size_t height) // height of viewport in pixels
  {
    // look up render cache.
    std::shared_ptr<display_requirement> display_req = display_reqs.at(channel_path);
    osg_renderparams params{
      //recdb,
      RenderCache,
      with_display_transforms,
      
      0.0, //left,
      0.0, //right,
      0.0, //bottom,
      0.0, //top,
      width,
      height,
      
    };

    std::shared_ptr<osg_rendercacheentry> renderentry;
    bool modified;
    
    std::tie(renderentry,modified) = RenderCache->GetEntry(params,display_req);
    
    
    if (width != Camera->getViewport()->width() || height != Camera->getViewport()->height()) {
      GraphicsWindow->getEventQueue()->windowResize(0,0,width,height);
      GraphicsWindow->resized(0,0,width,height);
      Camera->setViewport(0,0,width,height);
      modified = true;
    }
    Camera->setProjectionMatrixAsPerspective(30.0f,((double)width)/height,1.0f,10000.0f); // !!!*** Check last two parameters
    
    
    if (renderentry) {
      std::shared_ptr<osg_rendercachegroupentry> rendergroup = std::dynamic_pointer_cast<osg_rendercachegroupentry>(renderentry);
      
      if (rendergroup) {
	if (Viewer->getSceneData() != rendergroup->osg_group){
	  Viewer->setSceneData(rendergroup->osg_group);
	}
	
	
      } else {
	snde_warning("openscenegraph_3d_renderer: cache entry not convertable to an osg_group rendering channel %s",channel_path.c_str());
      }
    } else {
      snde_warning("openscenegraph_3d_renderer: cache entry not available rendering channel %s",channel_path.c_str());
      
    }
    
    //if (firstrun) {
    //  Viewer->realize();
    //  firstrun=false;
    //}

    /*    osg::Matrixd viewmat = Camera->getViewMatrix();
    {
      int i,j;
      printf("Camera view matrix:\n");
      for (i=0; i < 4; i++) {
	for (j=0; j < 4; j++) {
	  printf("%8f ",viewmat(i,j));
	}
	printf("\n");
      }
      
      }*/


    return std::make_tuple(renderentry,modified);
    
  }
  
  
  
};
