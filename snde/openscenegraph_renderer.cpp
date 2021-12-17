
#include "snde/openscenegraph_renderer.hpp"

namespace snde {


  
  osg_renderer::osg_renderer(osg::ref_ptr<osgViewer::Viewer> Viewer, // use an osgViewerCompat34()
			     osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
			     std::string channel_path,int type) :
    Viewer(Viewer),
    Camera(Viewer->getCamera()),
    GraphicsWindow(GraphicsWindow),
    channel_path(channel_path)
  {
    
  }
  
  void osg_renderer::frame()
  {
    Viewer->frame();
  }
  
  
  
};
