
#include "snde/openscenegraph_renderer.hpp"

namespace snde {


  void GLAPIENTRY OGLMessageCallback( GLenum source,
				      GLenum type,
				      GLuint id,
				      GLenum severity,
				      GLsizei length,
				      const GLchar* message,
				      const void* userParam)
  {
    // could change this to snde_debug(SNDE_DC_RENDERING,...
    snde_warning("OPENGL MESSAGE: %s type = 0x%x, severity = 0x%x, message = %s\n",
		 ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "" ),
		 type, severity, message );
  }
  
  
  
  osg_renderer::osg_renderer(osg::ref_ptr<osgViewer::Viewer> Viewer, // use an osgViewerCompat34()
			     osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
			     std::string channel_path,int type) :
    Viewer(Viewer),
    RootTransform(new osg::MatrixTransform()),
    Camera(Viewer->getCamera()),
    GraphicsWindow(GraphicsWindow),
    channel_path(channel_path),
    type(type)
  {
    
  }
  
  void osg_renderer::frame()
  {
    Viewer->frame();
  }
  
  
  
};
