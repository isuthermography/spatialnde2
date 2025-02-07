#ifdef _MSC_VER
  #define GLAPI WINGDIAPI
  #define GLAPIENTRY APIENTRY
  #define NOMINMAX
  #include <Windows.h>
  #include <GL/glew.h>
#endif

#include <osg/Version>
#include <osg/GL>
#include <osg/GLExtensions>
#include <osgGA/OrbitManipulator>

#if OPENSCENEGRAPH_MAJOR_VERSION >= 3 && OPENSCENEGRAPH_MINOR_VERSION >= 6
#include <osg/VertexArrayState>
#endif

#include "snde/openscenegraph_renderer.hpp"
#ifdef __APPLE__
//GLDEBUGPROC not defined
typedef void (*GLDEBUGPROC)(GLenum source,
            GLenum type,
            GLuint id,
            GLenum severity,
            GLsizei length,
            const GLchar *message,
            const void *userParam);
#endif

namespace snde {


  static void
#ifdef WIN32
  GLAPIENTRY
#endif
  OGLMessageCallback( GLenum source,
				      GLenum type,
				      GLuint id,
				      GLenum severity,
				      GLsizei length,
				      const GLchar* message,
				      const void* userParam)
  {
    
    if (type==GL_DEBUG_TYPE_ERROR) {
      snde_warning("OPENGL MESSAGE: %s type = 0x%x, severity = 0x%x, message = %s\n",
		 ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "" ),
		 type, severity, message );
      
    } else {
      
      snde_debug(SNDE_DC_RENDERING,"OPENGL MESSAGE: %s type = 0x%x, severity = 0x%x, message = %s\n",
		 ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "" ),
		 type, severity, message );
    }
  }
  
  
  void SetupOGLMessageCallback()
  {
    void (*ext_glDebugMessageCallback)(GLDEBUGPROC callback, void* userParam) = (void (*)(GLDEBUGPROC callback, void *userParam))osg::getGLExtensionFuncPtr("glDebugMessageCallback");
    if (ext_glDebugMessageCallback) {
      glEnable(GL_DEBUG_OUTPUT);
      ext_glDebugMessageCallback(&OGLMessageCallback,0);
    }
    
  }
  
  osg_renderer::osg_renderer(osg::ref_ptr<osgViewer::Viewer> Viewer, // use an osgViewerCompat34()
			     osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
			     osg::ref_ptr<osgGA::CameraManipulator> Manipulator,
			     std::string channel_path,int type,bool enable_shaders) :
    Viewer(Viewer),
    RootTransform(new osg::MatrixTransform()),
    Camera(Viewer->getCamera()),
    GraphicsWindow(GraphicsWindow),
    Manipulator(Manipulator),
    channel_path(channel_path),
    type(type),
    enable_shaders(enable_shaders)
  {
    if (enable_shaders) {
      ShaderProgram = new osg::Program();
    }

    
    _LastCameraPose = Camera->getInverseViewMatrix(); // camera pose is the inverse of the view matrix
    _LastRotationCenterDist=1.0; // reasonable value
    if (Manipulator) {
      osgGA::OrbitManipulator *Manip = dynamic_cast<osgGA::OrbitManipulator*>(Manipulator.get());
      if (Manip) {
	_LastRotationCenterDist = Manip->getDistance();
      }
    }
  }
  
  void osg_renderer::frame()
  {
    osg::ref_ptr<osg::RefMatrixd> NewCameraPose;
    std::shared_ptr<snde_coord> NewRotationCenterDist;
    {
      std::lock_guard<std::mutex> LCP_NCP_lock(LCP_NCP_mutex);
      
      NewCameraPose = _NewCameraPose;
      _NewCameraPose = nullptr;

      NewRotationCenterDist = _NewRotationCenterDist;
      _NewRotationCenterDist = nullptr;

    }

    if (NewCameraPose && Manipulator) {
      Manipulator->setByMatrix(*NewCameraPose);
    }
    if (NewRotationCenterDist && Manipulator) {
      osgGA::OrbitManipulator *Manip = dynamic_cast<osgGA::OrbitManipulator*>(Manipulator.get());
      if (Manip) {
	// Make sure the new rotation center doesn't change the camera pose
	osg::Matrixd CamMtx = Manip->getMatrix();
	
	//snde_warning("rendering %s setting rotation center to (%f,%f,%f)",channel_path.c_str(),NewRotationCenter->x(),NewRotationCenter->y(),NewRotationCenter->z());

	// really all we set with the center is the distance to the
	// rotation point because we don't allow the
	// rotation center application to change the camera pose
	//double distance = sqrt(pow(NewRotationCenter->x()-CamMtx(3,0),2)+pow(NewRotationCenter->y()-CamMtx(3,1),2) + pow(NewRotationCenter->z()-CamMtx(3,2),2));
	
	Manip->setDistance(*NewRotationCenterDist);
	Manip->setByMatrix(CamMtx);
      }
    }
    
    Viewer->frame();

    {
      std::lock_guard<std::mutex> LCP_NCP_lock(LCP_NCP_mutex);
      _LastCameraPose = Camera->getInverseViewMatrix(); // camera pose is the inverse of the view matrix
      if (Manipulator) {
	osgGA::OrbitManipulator *Manip = dynamic_cast<osgGA::OrbitManipulator*>(Manipulator.get());
	if (Manip) {
	  _LastRotationCenterDist = Manip->getDistance();
	  //snde_warning("rendering %s got rotation center of (%f,%f,%f)",channel_path.c_str(),_LastRotationCenter.x(),_LastRotationCenter.y(),_LastRotationCenter.z());
	}
      }
    }
  }
  
  osg::Matrixd osg_renderer::GetLastCameraPose()
  {
    {
      std::lock_guard<std::mutex> LCP_NCP_lock(LCP_NCP_mutex);
      return _LastCameraPose;
    }
    
  }

  void osg_renderer::AssignNewCameraPose(const osg::Matrixd &newpose)
  {
    {
      std::lock_guard<std::mutex> LCP_NCP_lock(LCP_NCP_mutex);
      _NewCameraPose = new osg::RefMatrixd(newpose);
      _LastCameraPose = newpose; 
    }
    
  }


  snde_coord osg_renderer::GetLastRotationCenterDist()
  {
    {
      std::lock_guard<std::mutex> LCP_NCP_lock(LCP_NCP_mutex);
      return _LastRotationCenterDist;
    }
    
  }

  void osg_renderer::AssignNewRotationCenterDist(snde_coord newcenterdist)
  {
    {
      std::lock_guard<std::mutex> LCP_NCP_lock(LCP_NCP_mutex);
      _NewRotationCenterDist = std::make_shared<snde_coord>(newcenterdist);
      _LastRotationCenterDist = newcenterdist; 
    }
    
  }


  
  void osg_SyncableState::SyncModeBits()
  {
    // (see header for explanation)
    // call this after calling reset()
    for (auto && ModeEntry: _modeMap) {
      const osg::StateAttribute::GLMode & MEMode = ModeEntry.first;
      ModeStack & MEStack = ModeEntry.second;
      if (MEStack.last_applied_value) {
	glEnable(MEMode);
      } else {
	glDisable(MEMode);	  
      }
      // Now this mode is synchronized with what OSG thinks
      // is the last applied value
    }
  }
  
  
  osg_ParanoidGraphicsWindowEmbedded::osg_ParanoidGraphicsWindowEmbedded(int x, int y, int width, int height) :
    osgViewer::GraphicsWindowEmbedded(x,y,width,height),
    gl_initialized(false)
  {
    osg::ref_ptr<osg_SyncableState> window_state;
    window_state = new osg_SyncableState() ;
    setState(window_state);
    
    assert(!window_state->getStateSetStackSize());
    
    window_state->setGraphicsContext(this);
    
    if (_traits->sharedContext.valid()) {
      window_state->setContextID(_traits->sharedContext->getState()->getContextID());
      
    } else {
      window_state->setContextID(osg::GraphicsContext::createNewContextID());
    }
  }
  
  void osg_ParanoidGraphicsWindowEmbedded::gl_is_available()
  {
    gl_initialized=true;
  }

  void osg_ParanoidGraphicsWindowEmbedded::gl_not_available()
  {
    gl_initialized=false;
  }



  bool osg_ParanoidGraphicsWindowEmbedded::makeCurrentImplementation()
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
    
    
    GLint drawbuf;
    glGetIntegerv(GL_DRAW_BUFFER,&drawbuf);
    snde_debug(SNDE_DC_RENDERING,"paranoidgraphicswindowembedded makecurrent glDrawBuffer is %x",(unsigned)drawbuf);
    GLint drawframebuf;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&drawframebuf);
    snde_debug(SNDE_DC_RENDERING,"paranoidgraphicswindowembedded makecurrent glDrawFrameBuffer is %d compared to defaultFBO %d",(int)drawframebuf,(int)getState()->getGraphicsContext()->getDefaultFboId());
    
    
    assert(!getState()->getStateSetStackSize());
    
    // Just in case our operations make changes to the
    // otherwise default state, we push this state onto
    // the OpenGL state stack so we can pop it off at the end. 
    glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    
    
    getState()->reset(); // the OSG-expected state for THIS WINDOW may have been messed up (e.g. by another window). So we need to reset the assumptions about the OpenGL state
	
	getState()->initializeExtensionProcs();
#if OPENSCENEGRAPH_MAJOR_VERSION >= 3 && OPENSCENEGRAPH_MINOR_VERSION >= 6
    // OSG 3.6.0 and above use a new VertexArrayState object that doesn't get
    // properly dirty()'d by reset()
    osg::ref_ptr<osg::VertexArrayState> VAS = getState()->getCurrentVertexArrayState();
    if (VAS) {
      VAS->dirty(); // doesn't actually do anything
      getState()->disableAllVertexArrays();
    }
#endif
    
    osg::ref_ptr<osg_SyncableState> window_state = dynamic_cast<osg_SyncableState *>(getState());
    window_state->SyncModeBits();
    
    // !!!*** reset() above may be unnecessarily pessimistic, dirtying all array buffers, etc. (why???)
    getState()->apply();
    
    
    
    
    SetupOGLMessageCallback();
    
    // make sure the correct framebuffer is bound... but only if we actually have extensions
    if (getState()->_extensionMap.size() > 0) {
      getState()->get<osg::GLExtensions>()->glBindFramebuffer(GL_FRAMEBUFFER_EXT, getDefaultFboId());
    }
    
    
    getState()->pushStateSet(new osg::StateSet());
    
    
    
    return true;
  }
  
  bool osg_ParanoidGraphicsWindowEmbedded::releaseContextImplementation()
  {
    //assert(getState()->getStateSetStackSize()==1);
    //getState()->popStateSet();
    assert(getState()->getStateSetStackSize() <= 1); // -- can be 1 because viewer->frame() pops all statesets; can be 0 on deletion
    
    
    // return OpenGL to default state
    getState()->popAllStateSets();
    getState()->apply();
    
    
    GLint drawbuf;
    glGetIntegerv(GL_DRAW_BUFFER,&drawbuf);
    snde_debug(SNDE_DC_RENDERING,"paranoidgraphicswindowembedded releasecontext glDrawBuffer is %x",(unsigned)drawbuf);
    GLint drawframebuf;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&drawframebuf);
    snde_debug(SNDE_DC_RENDERING,"paranoidgraphicswindowembedded releasecontext glDrawFrameBuffer is %d compared to defaultFBO %d",(int)drawframebuf,(int)getState()->getGraphicsContext()->getDefaultFboId());

    // it would be cleaner to explicitly remove our OGLMessageCallback here

    
    glPopAttrib();
    glPopClientAttrib();
    return true;
  }
  
  
};
