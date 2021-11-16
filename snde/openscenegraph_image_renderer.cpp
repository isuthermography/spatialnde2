
#include "snde/openscenegraph_image_renderer.hpp"
#include "snde/rec_display.hpp"

namespace snde {


  


  osg_image_renderer::osg_image_renderer(osg::ref_ptr<osgViewer::Viewer> Viewer, // use an osgViewerCompat34()
					 osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
					 std::shared_ptr<osg_rendercache> RenderCache) :
    GraphicsWindow(GraphicsWindow),
    //Viewer(new osgViewer::Viewer()),
    Viewer(Viewer),
    Camera(Viewer->getCamera()),
    //RootNode(RootNode),
    RenderCache(RenderCache)
  {
    
    Camera->setGraphicsContext(GraphicsWindow);
    
    // set background color to blueish
    Camera->setClearColor(osg::Vec4(.1,.1,.3,1.0));    
    
    Viewer->setThreadingModel(osgViewer::Viewer::SingleThreaded); // Could try more sophisticated rendering model later. 
    //Viewer->setCameraManipulator(Manipulator);
    //Viewer->setRunFrameScheme(osg::ON_DEMAND); // ***!!! With this OSG looks at whether it thinks a new render is needed based on scene graph changes and only renders if necessary. 
    
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

  
  
  void osg_image_renderer::perform_render(std::shared_ptr<recdatabase> recdb,
					  //std::shared_ptr<recstore_display_transforms> display_transforms,
					  std::shared_ptr<recording_set_state> with_display_transforms,
					  //std::shared_ptr<display_channel> channel_to_display,
					  std::string channel_path,
					  std::shared_ptr<display_info> display,
					  double left, // left of viewport in channel horizontal units
					  double right, // right of viewport in channel horizontal units
					  double bottom, // bottom of viewport in channel vertical units
					  double top, // top of viewport in channel vertical units
					  size_t width, // width of viewport in pixels
					  size_t height) // height of viewport in pixels
  {
    // look up render cache.
    osg::ref_ptr<osg::Group> imagegroup = RenderCache->GetEntry(with_display_transforms,channel_path,SNDE_DRM_RGBAIMAGE,left,right,bottom,top);

    if (Viewer->getSceneData() != imagegroup){
      group=imagegroup;
      Viewer->setSceneData(group);
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
    
    
    Viewer->frame();
    
  }
  

  
};
