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

#if OPENSCENEGRAPH_MAJOR_VERSION >= 3 && OPENSCENEGRAPH_MINOR_VERSION >= 6
#include <osg/VertexArrayState>
#endif

#include "snde/openscenegraph_renderer.hpp"

namespace snde {


  static void GLAPIENTRY OGLMessageCallback( GLenum source,
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
			     std::string channel_path,int type,bool enable_shaders) :
    Viewer(Viewer),
    RootTransform(new osg::MatrixTransform()),
    Camera(Viewer->getCamera()),
    GraphicsWindow(GraphicsWindow),
    channel_path(channel_path),
    type(type),
    enable_shaders(enable_shaders)
  {
    if (enable_shaders) {
      ShaderProgram = new osg::Program();
    }
  }
  
  void osg_renderer::frame()
  {
    Viewer->frame();
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
    
    getState()->initializeExtensionProcs();
    
    
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
