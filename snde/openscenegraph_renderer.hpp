
#ifndef SNDE_OPENSCENEGRAPH_RENDERER_HPP
#define SNDE_OPENSCENEGRAPH_RENDERER_HPP

// This is missing from the OSG OpenGL headers on Anaconda on Windows for some reason... it is the only thing missing...
#ifndef GL_RGB8UI
#define GL_RGB8UI 0x8D7D
#endif

#include <osgViewer/Renderer>
#include <osgViewer/Viewer>
#include <osg/Group>
#include <osg/MatrixTransform>

#include "snde/recstore.hpp"
#include "snde/openscenegraph_rendercache.hpp"


namespace snde {

  class osg_renderer;
  
  class osgViewerCompat34: public osgViewer::Viewer {
    // derived version of osgViewer::Viewer that gives compat34GetRequestContinousUpdate()
    // alternative to osg v3.6 getRequestContinousUpdate()
  public:

    osgViewerCompat34() = default;
    osgViewerCompat34(const osgViewerCompat34 &orig) :
      osgViewer::Viewer(orig)
    {
      //_frameStamp = new osg::FrameStamp;
      //_frameStamp->setFrameNumber(0);
      //_frameStamp->setReferenceTime(0);
      //_frameStamp->setSimulationTime(0);
      _frameStamp = orig._frameStamp;

      //_eventVisitor = new osgGA::EventVisitor;
      //_eventVisitor->setActionAdapter(this);
      //_eventVisitor->setFrameStamp(_frameStamp.get());
      
      //_updateVisitor = new osgUtil::UpdateVisitor;
      //_updateVisitor->setFrameStamp(_frameStamp.get());
      _updateVisitor = orig._updateVisitor;
    }

    bool compat34GetRequestContinousUpdate()
    {
      return _requestContinousUpdate;
    }
  };



  class osg_renderer {
  public:
    // base class for renderers

    osg::ref_ptr<osgViewer::Viewer> Viewer;
    osg::ref_ptr<osg::MatrixTransform> RootTransform; // Need Root group because swapping out SceneData clears event queue
    osg::ref_ptr<osg::Camera> Camera;
    osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow;
    osg::ref_ptr<osgGA::EventQueue> EventQueue; // we keep a separate pointer to the event queue because getEventQueue() may not e thread safe but the EventQueue itself seems to be. 
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


  class osg_ParanoidGraphicsWindowEmbedded: public osgViewer::GraphicsWindowEmbedded {
  public:
    std::atomic<bool> gl_initialized;
    
    osg_ParanoidGraphicsWindowEmbedded(int x, int y, int width, int height) :
      osgViewer::GraphicsWindowEmbedded(x,y,width,height),
      gl_initialized(false)
    {

    }

    void gl_is_available()
    {
      gl_initialized=true;
    }
    
    void gl_not_available()
    {
      gl_initialized=false;
    }

    virtual const char* libraryName() const { return "snde"; }
    
    virtual const char* className() const { return "osg_ParanoidGraphicsWindowEmbedded"; }

    virtual bool makeCurrentImplementation()
    {
      // paranoid means no assumption that the state hasn't been messed with behind our backs
      
      //if (!referenceCount()) {
	// this means we might be in the destructor, in which case there might not be a valid
	// OpenGL context, and we should just return
      //return false;
      //}
      if (!gl_initialized) {
	return false;
      }
      
      getState()->reset(); // the OSG-expected state for THIS WINDOW may have been messed up (e.g. by another window). So we need to reset the assumptions about the OpenGL state

      // !!!*** reset() above may be unnecessarily pessimistic, dirtying all array buffers, etc. (why???)

      getState()->apply();

      getState()->initializeExtensionProcs();
      
      // make sure the correct framebuffer is bound... but only if we actually have extensions
      if (getState()->_extensionMap.size() > 0) {
	getState()->get<osg::GLExtensions>()->glBindFramebuffer(GL_FRAMEBUFFER_EXT, getDefaultFboId());
      }
      return true;
    }
    
  };

}

#endif // SNDE_OPENSCENEGRAPH_RENDERER_HPP



