
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

  void SetupOGLMessageCallback();


  class osg_renderer {
  public:
    // base class for renderers.... Not generally thread safe, like OpenSceneGraph
    // except for GetLastCameraPose() method and AssignNewCameraPose()

    osg::ref_ptr<osgViewer::Viewer> Viewer;
    osg::ref_ptr<osg::MatrixTransform> RootTransform; // Need Root group because swapping out SceneData clears event queue
    osg::ref_ptr<osg::Camera> Camera;
    osg::ref_ptr<osgGA::CameraManipulator> Manipulator;
    osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow;
    osg::ref_ptr<osgGA::EventQueue> EventQueue; // we keep a separate pointer to the event queue because getEventQueue() may not e thread safe but the EventQueue itself seems to be. 
    std::string channel_path;

    std::mutex LCP_NCP_mutex; // protects _LastCameraPose and _NewCameraPose; last in the locking order; definitely after the compositor admin lock
    osg::Matrixd _LastCameraPose;
    snde_coord _LastRotationCenterDist;
    osg::ref_ptr<osg::RefMatrixd> _NewCameraPose; // if not nullptr, use this new camera pose on next render.
    std::shared_ptr<snde_coord> _NewRotationCenterDist; // if not nullptr, use this new rotation center on next render.

    int type; // see SNDE_DRRT_XXXXX in rec_display.hpp
    bool enable_shaders;
    osg::ref_ptr<osg::Program> ShaderProgram;
    

    osg_renderer(osg::ref_ptr<osgViewer::Viewer> Viewer, // use an osgViewerCompat34()
		 osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
		 osg::ref_ptr<osgGA::CameraManipulator> Manipulator,
		 std::string channel_path,int type,bool enable_shaders);
    osg_renderer(const osg_renderer &) = delete;
    osg_renderer & operator=(const osg_renderer &) = delete;
    virtual ~osg_renderer() = default; 
    
    virtual std::tuple<std::shared_ptr<osg_rendercacheentry>,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>>,bool>  // returns cacheentry,locks_required,modified
    prepare_render(//std::shared_ptr<recdatabase> recdb,
		   std::shared_ptr<recording_set_state> with_display_transforms,
		   //std::shared_ptr<display_info> display,
		   std::shared_ptr<osg_rendercache> RenderCache,
		   const std::map<std::string,std::shared_ptr<display_requirement>> &display_reqs,
		   size_t width,
		   size_t height)=0;

    
    
    virtual void frame();
    virtual osg::Matrixd GetLastCameraPose();
    virtual snde_coord GetLastRotationCenterDist();
    virtual void AssignNewCameraPose(const osg::Matrixd &newpose);
    virtual void AssignNewRotationCenterDist(snde_coord newrotctrdist);
    
  };


  class osg_SyncableState: public osg::State {
    // State class that can be synchronized to a corrupted state
    // (because we keep switching the state between our various
    // osg_layerwindows and (in the non-threaded-opengl case)
    // our rendering window.
    //
    // The problem is that osg::State::reset() doesn't mark the state
    // bits as invalid and needing to be rewritten in all cases --
    // it just flip the state bits and marks them as dirty -- so
    // that if the next desired state is flipped from our current
    // state it will fail to be rewritten. The solution, per
    // https://osg-users.openscenegraph.narkive.com/zHUDa1nx/state-reset
    // is this derived class with a method that can be called after
    // osg::State::reset() to make sure the programmed mode bits
    // match the actual state

    // It is possible (maybe not very likely, but possible)
    // that something similar will have to be done with the texture
    // state....
    
  public:
    void SyncModeBits();
    
  };


  
  
  class osg_ParanoidGraphicsWindowEmbedded: public osgViewer::GraphicsWindowEmbedded {
  public:
    std::atomic<bool> gl_initialized;
    
    osg_ParanoidGraphicsWindowEmbedded(int x, int y, int width, int height);

    
    // this prevents osg::GraphicsWindowEmbedded::init() from being called in the superclass constructor
    // so we can create the osg::State ourselves in our constructor. Also prevents access until
    // QT has initialized OpenGL for us. 
    //virtual bool valid() const
    //{
    //  return gl_initialized; 
    //}
    
    
    void gl_is_available();
    
    void gl_not_available();

    virtual const char* libraryName() const { return "snde"; }
    
    virtual const char* className() const { return "osg_ParanoidGraphicsWindowEmbedded"; }

    virtual bool makeCurrentImplementation();

    virtual bool releaseContextImplementation();
    
  };

}

#endif // SNDE_OPENSCENEGRAPH_RENDERER_HPP



