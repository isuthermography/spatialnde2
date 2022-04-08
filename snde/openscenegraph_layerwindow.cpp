#include <osg/Version>
#include <osg/Texture>
#include <osg/FrameBufferObject>

#if OPENSCENEGRAPH_MAJOR_VERSION >= 3 && OPENSCENEGRAPH_MINOR_VERSION >= 6
#include <osg/VertexArrayState>
#endif

#include "snde/openscenegraph_layerwindow.hpp"
#include "snde/openscenegraph_renderer.hpp"

#ifndef GL_DRAW_FRAMEBUFFER_BINDING
#define GL_DRAW_FRAMEBUFFER_BINDING 0x8CA6   // in case GL headers are missing this
#endif 

namespace snde {
  
  osg_layerwindow_postdraw_callback::osg_layerwindow_postdraw_callback(osg::ref_ptr<osgViewer::Viewer> Viewer, osg::ref_ptr<osg::Texture2D> outputbuf_tex) :
    Viewer(Viewer),
    outputbuf_tex(outputbuf_tex)
  {
    readback_pixels=std::make_shared<std::shared_ptr<std::vector<unsigned char>>>();
  }


  void osg_layerwindow_postdraw_callback::operator()(osg::RenderInfo &Info) const
  {
    //OSG_INFO << "postDraw()\n";
    
    if (outputbuf_tex) {
      // Reading the image back (optional, but needed for testing)
      
      // https://groups.google.com/g/osg-users/c/OomZxLrRDGk :
      // I haven't done what you want before but the way I'd tackle it would be
      // to use a Camera post draw callback to call
      // state.applyAttribute(texture); <---- NOTE: This caused huge problems; instead we use texture.getTextureObject().bind()
      // image->readImageFromCurrentTexture(..).
      
      // Alternative:
      //https://www.khronos.org/opengl/wiki/Framebuffer_Object_Extension_Examples#glReadPixels
      
      //GLint drawbuf;
      //glGetIntegerv(GL_DRAW_BUFFER,&drawbuf);
      //OSG_INFO << std::string("glDrawBuffer is now ")+std::to_string(drawbuf) +"\n";
      
      //GLint drawframebuf;
      //glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&drawframebuf);
      //OSG_INFO << std::string("glDrawFrameBuffer is now ")+std::to_string(drawframebuf) +"\n";
      
      osg::ref_ptr<osg::FrameBufferObject> FBO;
      
      // get the OpenSceneGraph osg::Camera::FRAME_BUFFER_OBJECT
      // RenderTargetImplementation's FBO, by looking up the renderer
      // based on our viewer. 
      osgViewer::Renderer *Rend = dynamic_cast<osgViewer::Renderer *>(Viewer->getCamera()->getRenderer());
      osgUtil::RenderStage *Stage = Rend->getSceneView(0)->getRenderStage();
      
      FBO = Stage->getFrameBufferObject();
      
      
      FBO->apply(*Info.getState(),osg::FrameBufferObject::DRAW_FRAMEBUFFER);
      FBO->apply(*Info.getState(),osg::FrameBufferObject::READ_FRAMEBUFFER);	
      
      
      
      outputbuf_tex->getTextureObject(Info.getState()->getContextID())->bind();
      *readback_pixels=std::make_shared<std::vector<unsigned char>>(outputbuf_tex->getTextureWidth()*outputbuf_tex->getTextureHeight()*4);
      glReadPixels(0,0,outputbuf_tex->getTextureWidth(),outputbuf_tex->getTextureHeight(),GL_RGBA,GL_UNSIGNED_BYTE,(*readback_pixels)->data());

      
      
      //FILE *fh=fopen("/tmp/foo.img","wb");
      //fwrite((*readback_pixels)->data(),1,(outputbuf_tex->getTextureWidth()*outputbuf_tex->getTextureHeight()*4),fh);
      //fclose(fh);
    

      
      // disable the Fbo ... like disableFboAfterRendering
      
      GLuint fboId = Info.getState()->getGraphicsContext()->getDefaultFboId();
      Info.getState()->get<osg::GLExtensions>()->glBindFramebuffer(GL_FRAMEBUFFER_EXT, fboId);
    }
    
    // If we had bound our own framebuffer in the predraw callback then
    // this next call would make sense (except that we should really
    // use the GraphicsContext's default FBO, not #0
    //Info.getState()->get<osg::GLExtensions>()->glBindFramebuffer( GL_FRAMEBUFFER_EXT, 0 );
    
    // switch drawing back to the OSG-intended buffer
    //glDrawBuffer(*DrawBufferSave); // ... except this causes errors, I think because of asymmetries in OpenSceneGraph
    
  }
  


  osg_layerwindow_predraw_callback::osg_layerwindow_predraw_callback(osg::ref_ptr<osgViewer::Viewer> Viewer,osg::ref_ptr<osg::Texture2D> outputbuf,bool readback) :
      Viewer(Viewer),
      outputbuf(outputbuf),
      readback(readback)
  {
    
  }


  void osg_layerwindow_predraw_callback::operator()(osg::RenderInfo &Info) const
  {
    //OSG_INFO << "preDraw()\n";

    
    //GLint drawbuf;
    //glGetIntegerv(GL_DRAW_BUFFER,&drawbuf);
    //*DrawBufferSave = (GLenum)drawbuf;
    
    osg::ref_ptr<osg::FrameBufferObject> FBO;
    
    // get the OpenSceneGraph osg::Camera::FRAME_BUFFER_OBJECT
    // RenderTargetImplementation's FBO, by looking up the renderer
    // based on our viewer. 
    osgViewer::Renderer *Rend = dynamic_cast<osgViewer::Renderer *>(Viewer->getCamera()->getRenderer());
    osgUtil::RenderStage *Stage = Rend->getSceneView(0)->getRenderStage();


    // since we are now using InitialCallback not Predraw callback
    // we need to run RenderStage::runCameraSetup() early to generate
    // the FBO, as InitialCallback is called before it 
    // (the _cameraRequiresSetUp flag will prevent a double call)
    Stage->runCameraSetUp(Info);
    
    Stage->setDisableFboAfterRender(false); // Need to leave the Fbo in place for post_render cameras


    // OSG doesn't actually resize the texture so we need to do that ourselves in case it has changed
    osg::Texture::TextureObject *OutputBufTexObj = outputbuf->getTextureObject(Info.getState()->getContextID());
    if (OutputBufTexObj) {
      // (If OutputBufTexObj doesn't exist then it will be created with the correct parameters when needed
      // and this is unnecessary)
      OutputBufTexObj->bind();
      snde_debug(SNDE_DC_RENDERING,"osg_layerwindow: Calling glTexImage2D (id %d) with width=%d, height=%d",(int)OutputBufTexObj->id(),outputbuf->getTextureWidth(), outputbuf->getTextureHeight());
      // NOTE: If we start getting an OpenGL error from this next line
      // it means that the hack to prevent osg/Texture2D.cpp from creating
      // an unresizable texture with glTexStorage2D() has failed (see init() function below) 
      glTexImage2D( GL_TEXTURE_2D, 0, outputbuf->getInternalFormat(),
	    	    outputbuf->getTextureWidth(), outputbuf->getTextureHeight(),
	    	    outputbuf->getBorderWidth(),
	    	    outputbuf->getInternalFormat(),
	    	    outputbuf->getSourceType(),nullptr);

    }
    FBO = Stage->getFrameBufferObject();
    
    // Our attachment here overrides the RenderBuffer that OSG's FBO
    // RenderTargetImplementation created automatically, but that's OK. 
    FBO->setAttachment(osg::Camera::COLOR_BUFFER, osg::FrameBufferAttachment(outputbuf.get()));
    
    // If we had created our own FBO, we would need to attach our own
    // DEPTH_BUFFER, but this is irrelevant because OSG's FBO RTI
    // already created and attached one for us. 
    //FBO->setAttachment(osg::Camera::DEPTH_BUFFER, osg::FrameBufferAttachment(depthbuf.get()));
    
    // setup as the draw framebuffer -- this may be redundant but makes
    // sure the FBO is properly configured in the OpenGL state. 
    FBO->apply(*Info.getState(),osg::FrameBufferObject::DRAW_FRAMEBUFFER);
    //if (readback) {

    // get GL errors from OSG if we don't also set the read framebuffer
    FBO->apply(*Info.getState(),osg::FrameBufferObject::READ_FRAMEBUFFER);	
      //}
    //assert(glGetError()== GL_NO_ERROR);
    
    // Verify the framebuffer configuration (Draw mode)
    GLenum status = Info.getState()->get<osg::GLExtensions>()->glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER_EXT);
    
    if (status != GL_FRAMEBUFFER_COMPLETE_EXT) {
      if (status==GL_FRAMEBUFFER_UNSUPPORTED_EXT) {
	throw snde_error("osg_layerwindow: Framebuffer configuration not supported by OpenGL implementation");
      } else {
	throw snde_error("osg_layerwindow: Unknown framebuffer error: %x",(unsigned)status);
	
      }
      
    }
    
    //assert(glGetError()== GL_NO_ERROR);
    
    if (readback) {
      // Verify the framebuffer configuration (Read mode)
      
      GLenum status = Info.getState()->get<osg::GLExtensions>()->glCheckFramebufferStatus(GL_READ_FRAMEBUFFER_EXT);
      
      if (status != GL_FRAMEBUFFER_COMPLETE_EXT) {
	if (status==GL_FRAMEBUFFER_UNSUPPORTED_EXT) {
	  throw snde_error("osg_layerwindow: Framebuffer configuration not supported by OpenGL implementation");
	} else {
	  throw snde_error("osg_layerwindow: Unknown framebuffer error: %x",(unsigned)status);
	  
	}
	
      }
    }
    
    //assert(glGetError()== GL_NO_ERROR);
    
    // Make sure we are drawing onto the FBO attachment
    glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

    //assert(glGetError()== GL_NO_ERROR);

    //glViewport( 0, 0, outputbuf->getTextureWidth(),outputbuf->getTextureHeight());
    //glClearColor(.5,.4,.3,1.0);
    //glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
 
    
    // debugging
    //glViewport(0,0,outputbuf->getTextureWidth(),outputbuf->getTextureHeight());
    //assert(glGetError()== GL_NO_ERROR);
    
    
    // These next few lines can be used for debugging to make
    // sure are settings are surviving the render process
    //GLint drawbuf;
    //glGetIntegerv(GL_DRAW_BUFFER,&drawbuf);
    //OSG_INFO << std::string("glDrawBuffer set to ")+std::to_string(drawbuf) +"\n";
    
    //GLint drawframebuf;
    //glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&drawframebuf);
    //OSG_INFO << std::string("glDrawFrameBuffer set to ")+std::to_string(drawframebuf) +"\n";
    
    
    // These next few lines can be used along with the commented-out
    // readback, below, to confirm proper FBO operation before
    // starting the full rendering process
    //glClearColor(.3,.4,.5,1.0);
    //glViewport( 0, 0, outputbuf->getTextureWidth(),outputbuf->getTextureHeight());
    //glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    
    
    //outputbuf->getTextureObject(Info.getState()->getContextID())->bind();
    
    //void *readback_pixels=calloc(1,(outputbuf->getTextureWidth()*outputbuf->getTextureHeight()*4));
    //glReadPixels(0,0,outputbuf->getTextureWidth(),outputbuf->getTextureHeight(),GL_RGBA,GL_UNSIGNED_BYTE,readback_pixels);
    //
    //FILE *fh=fopen("/tmp/foo.img","wb");
    //fwrite(readback_pixels,1,(outputbuf->getTextureWidth()*outputbuf->getTextureHeight()*4),fh);
    //fclose(fh);
    
    
    
  }



  
  osg_layerwindow::osg_layerwindow(osg::ref_ptr<osgViewer::Viewer> Viewer,osg::ref_ptr<osg::GraphicsContext> shared_context,int width, int height,bool readback) :
    Viewer(Viewer),
    readback(readback),
    osgViewer::GraphicsWindow(),
    shared_context(shared_context)
  {
    // Two ways to run this: Use 
    // Cam->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT).
    // or 
    // Cam->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER)
    // in setup_camera(), below.
    //
    // For now we use the former. In this case we rely on the OSG
    // renderer (RenderStage.cpp) to create our FBO that we render into.
    // This is because the FBO-aware renderer would override any attempt
    // to install our own FBO renderer.
    //
    // We just need to attach our own texture buffer to save the rendered
    // output. This is done by setting up pre-draw and post-draw
    // callbacks in the camera (below). 
    
    // I believe it would also work if we ran this in
    // FRAME_BUFFER mode instead. In this case we would have to build
    // our own FBO. Preliminary testing indicated that it works to
    // install the FBO using the PreDrawCallback (with the FBO-naive
    // FRAME_BUFFER renderer). We would also need to create our own
    // depth RenderBuffer. All of this is present but commented
    // out 
    
    _traits = new GraphicsContext::Traits();
    _traits->x = 0;
    _traits->y = 0;
    _traits->width = width;
    _traits->height = height;
    
    
    
    _traits->windowDecoration=false;
    _traits->doubleBuffer=false;
    _traits->sharedContext=shared_context;
    _traits->vsync=false;
    
    init();
    
    //Cam->setReadBuffer()
    //Cam->attach(osg::Camera::COLOR_BUFFER0,outputbuf);
    
    //Cam->attach(osg::Camera::DEPTH_BUFFER, depthbuf);
  }


  void osg_layerwindow::setup_camera(osg::ref_ptr<osg::Camera> Cam)
  {
    
    Cam->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
    //Cam->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER);
    Cam->setClearMask(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    Cam->setClearColor(osg::Vec4f(0, 0, 0, 0));
    //Cam->setPreDrawCallback(predraw);
    //Cam->setPostDrawCallback(postdraw);
    Cam->setInitialDrawCallback(predraw);
    Cam->setFinalDrawCallback(postdraw);
    Cam->setDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
    Cam->setReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
    
    
    
    // Potentially a 3rd option would be to use OSG's
    // RTT functionality directly by attaching
    // buffers to the camera. I had no indication
    // this would actually work. 
    //Cam->attach(osg::Camera::COLOR_BUFFER0,outputbuf);
    //Cam->attach(osg::Camera::DEPTH_BUFFER, depthbuf);
    
    Cam->setGraphicsContext(this);
    
  }

  void osg_layerwindow::clear_from_camera(osg::ref_ptr<osg::Camera> Cam)
  {
    //Cam->setPreDrawCallback(nullptr);
    //Cam->setPostDrawCallback(nullptr);
    Cam->setInitialDrawCallback(nullptr);
    Cam->setFinalDrawCallback(nullptr);
    
    Cam->setGraphicsContext(nullptr);
  }


  void osg_layerwindow::resizedImplementation(int x,int y,
					      int width,
					      int height)
  {

    
    assert(x==0  && y==0);
    
    if (_traits) {
      if (width != _traits->width || height != _traits->height) {
	
	_traits->width = width;
	_traits->height = height;
	//depthbuf->setSize(width,height);

	// There seem (?) to be driver bugs where if you attempt grow an existing texture
	// only the original corner is valid. So we try just completely swapping out the texture

	
	outputbuf->setTextureSize(width,height);

	/*
	outputbuf = new osg::Texture2D(); // !!!*** see also init() where outputbuf is initially created ***!!!
	outputbuf->setTextureSize(_traits->width,_traits->height);
	outputbuf->setSourceFormat(GL_RGBA);
	//outputbuf->setInternalFormat(GL_RGBA8UI); // using this causes  GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT 0x8CD6 error
	outputbuf->setInternalFormat(GL_RGBA);
	outputbuf->setWrap(osg::Texture::WRAP_S,osg::Texture::CLAMP_TO_EDGE);
	outputbuf->setWrap(osg::Texture::WRAP_T,osg::Texture::CLAMP_TO_EDGE);
	outputbuf->setSourceType(GL_UNSIGNED_BYTE);
	outputbuf->setFilter(osg::Texture::MIN_FILTER,osg::Texture::LINEAR);
	outputbuf->setFilter(osg::Texture::MAG_FILTER,osg::Texture::LINEAR);
	
	predraw->outputbuf=outputbuf;
	if (readback) {
	  postdraw->outputbuf_tex = outputbuf;
	}
	*/
	
	//snde_warning("layerwindow: setting texture size to %d by %d", width,height);
	Viewer->getCamera()->setViewport(0,0,width,height);//  (redundant)

      }
      
      osgViewer::GraphicsWindow::resizedImplementation(0,0,width,height);
      
    } else {
      // not currently possible
      _traits = new osg::GraphicsContext::Traits();
      _traits->x=0;
      _traits->y=0;
      _traits->width=width;
      _traits->height=height;
      _traits->windowDecoration=false;
      _traits->doubleBuffer=false;
      _traits->sharedContext=shared_context;
      _traits->vsync=false;
      init();
      osgViewer::GraphicsWindow::resized(0,0,width,height);
      
    }
  }


  void osg_layerwindow::init()
  {
    
    if (valid()) {
      
      
      osg::ref_ptr<osg::State> ourstate=new osg_SyncableState();
      ourstate->setGraphicsContext(this);

      // for debugging only -- good for tracking down any opengl errors
      // identified by OSG
      //ourstate->setCheckForGLErrors(osg::State::CheckForGLErrors::ONCE_PER_ATTRIBUTE);
      
      
      // Use (and increment the usage count) of the shared context, if given
      if (shared_context) {
	ourstate->setContextID(shared_context->getState()->getContextID());
	incrementContextIDUsageCount(ourstate->getContextID());
      } else {	
	ourstate->setContextID(osg::GraphicsContext::createNewContextID());
      }
      setState(ourstate);
      
      ourstate->initializeExtensionProcs();

      // Hack to prevent osg/Texture2D.cpp from creating an unresizable texture with glTexStorage2D by disable use of TextureStorage entirely
      ourstate->get<osg::GLExtensions>()->isTextureStorageEnabled=false; 
      
      
      outputbuf = new osg::Texture2D(); // !!!*** See also resizedImplementation() where outputbuf is re-created
      outputbuf->setTextureSize(_traits->width,_traits->height);
      outputbuf->setSourceFormat(GL_RGBA);
      //outputbuf->setInternalFormat(GL_RGBA8UI); // using this causes  GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT 0x8CD6 error
      outputbuf->setInternalFormat(GL_RGBA);
      outputbuf->setWrap(osg::Texture::WRAP_S,osg::Texture::CLAMP_TO_EDGE);
      outputbuf->setWrap(osg::Texture::WRAP_T,osg::Texture::CLAMP_TO_EDGE);
      outputbuf->setSourceType(GL_UNSIGNED_BYTE);
      outputbuf->setFilter(osg::Texture::MIN_FILTER,osg::Texture::LINEAR);
      outputbuf->setFilter(osg::Texture::MAG_FILTER,osg::Texture::LINEAR);
      
      
      // Create our own FBO for rendering into 
      //FBO = new osg::FrameBufferObject();
      
      // Attach FBO to our outputbuf texture
      //FBO->setAttachment(osg::Camera::COLOR_BUFFER0, osg::FrameBufferAttachment(outputbuf.get()));
      // Need to apply() so that the FBO is actually created
      //FBO->apply(*ourstate,osg::FrameBufferObject::DRAW_FRAMEBUFFER);
      
      // This must be after the setAttachment() and apply() or the FBO won't actually have been created. Set's the GraphicsWindow/GraphicsContext default FBO. 
      // setDefaultFboId(FBO->getHandle(ourstate->getContextID()));
      //OSG_INFO << "Default FBO ID: " + std::to_string(FBO->getHandle(ourstate->getContextID())) + "\n";
      
      // undo binding until we need it. If using QT we would want to bind to the QOpenGLContext's default framebuffer, not 0. 
      //ourstate->get<osg::GLExtensions>()->glBindFramebuffer( GL_FRAMEBUFFER_EXT, 0 );
      
      // Create our depth buffer that we will need to attach to the FBO
      //depthbuf = new osg::RenderBuffer(_traits->width,_traits->height,GL_DEPTH_COMPONENT24);
      
      
      // Create callbacks for use in setup_camera();
      predraw = new osg_layerwindow_predraw_callback(Viewer,outputbuf,readback);
      //predraw->DrawBufferSave=std::make_shared<GLenum>(GL_FRONT);
      postdraw = new osg_layerwindow_postdraw_callback(Viewer,(readback/*=true*/ /* ***!!!! */ )  ? outputbuf : nullptr); 
      //postdraw->DrawBufferSave=predraw->DrawBufferSave;

    }
  }


  const char *osg_layerwindow::libraryName() const
  {
    return "snde";
  }


  const char *osg_layerwindow::className() const
  {
    return "osg_layerwindow";
  }

  bool osg_layerwindow::valid() const
  {
    return true; 
  }

  bool osg_layerwindow::makeCurrentImplementation()
  {
    OSG_INFO << "makeCurrent()\n";


    GLint drawbuf;
    glGetIntegerv(GL_DRAW_BUFFER,&drawbuf);
    snde_debug(SNDE_DC_RENDERING,"osg_layerwindow makecurrent glDrawBuffer is %x",(unsigned)drawbuf);
    GLint drawframebuf;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&drawframebuf);
    snde_debug(SNDE_DC_RENDERING,"osg_layerwindow makecurrent glDrawFrameBuffer is %d compared to defaultFBO %d",(int)drawframebuf,(int)getState()->getGraphicsContext()->getDefaultFboId());

    
    // Just in case our operations make changes to the
    // otherwise default state, we push this state onto
    // the OpenGL state stack so we can pop it off at the end. 
    glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    assert(!getState()->getStateSetStackSize());

    
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

    getState()->apply();


    // Make sure our debug message callback in in place
    SetupOGLMessageCallback();
    

    getState()->pushStateSet(new osg::StateSet());

    //FBO->setAttachment(osg::Camera::COLOR_BUFFER0, osg::FrameBufferAttachment(outputbuf.get()));
    //FBO->setAttachment(osg::Camera::DEPTH_BUFFER, osg::FrameBufferAttachment(depthbuf.get()));
    
    // setup as the draw framebuffer
    //FBO->apply(*getState(),osg::FrameBufferObject::DRAW_FRAMEBUFFER);
    //if (readback) {
    //FBO->apply(*getState(),osg::FrameBufferObject::READ_FRAMEBUFFER);	
    //}
    
    //GLenum status = getState()->get<osg::GLExtensions>()->glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
    
    //if (status != GL_FRAMEBUFFER_COMPLETE) {
    //if (status==GL_FRAMEBUFFER_UNSUPPORTED) {
    //  throw snde_error("osg_layerwindow: Framebuffer configuration not supported by OpenGL implementation");
    //} else {
    //  throw snde_error("osg_layerwindow: Unknown framebuffer error: %x",(unsigned)status);
    //  
    //}
    //
    //}
    
    
    //if (readback) {
    //
    //GLenum status = getState()->get<osg::GLExtensions>()->glCheckFramebufferStatus(GL_READ_FRAMEBUFFER);
    
    //if (status != GL_FRAMEBUFFER_COMPLETE) {
    //  if (status==GL_FRAMEBUFFER_UNSUPPORTED) {
    //    throw snde_error("osg_layerwindow: Framebuffer configuration not supported by OpenGL implementation");
    //  } else {
    //    throw snde_error("osg_layerwindow: Unknown framebuffer error: %x",(unsigned)status);
    //    
    //  }
    
    //}
    //}
    
    
    
    return true;
  }

  bool osg_layerwindow::releaseContextImplementation()
  {

    //assert(getState()->getStateSetStackSize()==1);
    //getState()->popStateSet();
    assert(getState()->getStateSetStackSize() <= 1); // -- can be 1 because viewer->frame() pops all statesets; can be 0 on deletion

    // return OpenGL to default state
    getState()->popAllStateSets();
    getState()->apply();

    //OSG_INFO << "releaseContext()\n";
    //outputbuf->getTextureObject(getState()->getContextID())->bind();
    
    //getState()->get<osg::GLExtensions>()->glBindFramebuffer( GL_FRAMEBUFFER_EXT, 0 );

    // OSG leaves the drawbuffer set to the attachment (?), which can be problematic
    //glDrawBuffer(GL_FRONT);
    
    GLint drawbuf;
    glGetIntegerv(GL_DRAW_BUFFER,&drawbuf);
    snde_debug(SNDE_DC_RENDERING,"osg_layerwindow releasecontext glDrawBuffer is %x",(unsigned)drawbuf);
    GLint drawframebuf;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&drawframebuf);
    snde_debug(SNDE_DC_RENDERING,"osg_layerwindow releasecontext glDrawFrameBuffer is %d compared to defaultFBO %d",(int)drawframebuf,(int)getState()->getGraphicsContext()->getDefaultFboId());

    // it would be cleaner to explicitly remove our OGLMessageCallback here
    
    
    
    glPopAttrib();
    glPopClientAttrib();
    
    return true;
  }
  
  bool osg_layerwindow::realizeImplementation()
  {
    return true;
  }
  
  bool osg_layerwindow::isRealizedImplementation() const
  {
    return true;
  }

  void osg_layerwindow::closeImplementation()
  {
    
  }

  void osg_layerwindow::swapBuffersImplementation()
  {
    
  }

  void osg_layerwindow::grabFocus()
  {
    
  }


  void osg_layerwindow::grabFocusIfPointerInWindow()
  {
    
  }

  void osg_layerwindow::raiseWindow()
  {
    
  }
  
  
};
