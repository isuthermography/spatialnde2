
#ifndef SNDE_OPENSCENEGRAPH_RENDERER_HPP
#define SNDE_OPENSCENEGRAPH_RENDERER_HPP

// This is missing from the OSG OpenGL headers on Anaconda on Windows for some reason... it is the only thing missing...
#ifndef GL_RGB8UI
#define GL_RGB8UI 0x8D7D
#endif

#include <osgViewer/Renderer>
#include <osgViewer/Viewer>


#include "snde/recstore.hpp"
#include "snde/openscenegraph_rendercache.hpp"


namespace snde {

  class osg_renderer;
  
  class osgViewerCompat34: public osgViewer::Viewer {
    // derived version of osgViewer::Viewer that gives compat34GetRequestContinousUpdate()
    // alternative to osg v3.6 getRequestContinousUpdate()
  public:
    bool compat34GetRequestContinousUpdate()
    {
      return _requestContinousUpdate;
    }
  };



  class osg_renderer {
  public:
    // base class for renderers

    osg::ref_ptr<osgViewer::Viewer> Viewer;
    osg::ref_ptr<osg::Camera> Camera;
    osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow;
    std::string channel_path;

    int type; // see SNDE_DRRT_XXXXX in rec_display.hpp


    osg_renderer(osg::ref_ptr<osgViewer::Viewer> Viewer, // use an osgViewerCompat34()
		 osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
		 std::string channel_path,int type);
    osg_renderer(const osg_renderer &) = delete;
    osg_renderer & operator=(const osg_renderer &) = delete;
    virtual ~osg_renderer() = default; 
    
    virtual std::tuple<std::shared_ptr<osg_rendercacheentry>,bool>
    prepare_render(//std::shared_ptr<recdatabase> recdb,
		   std::shared_ptr<recording_set_state> with_display_transforms,
		   //std::shared_ptr<display_info> display,
		   std::shared_ptr<osg_rendercache> RenderCache,
		   const std::map<std::string,std::shared_ptr<display_requirement>> &display_reqs,
		   size_t width,
		   size_t height)=0;

    virtual void frame();

  };


}

#endif // SNDE_OPENSCENEGRAPH_RENDERER_HPP



