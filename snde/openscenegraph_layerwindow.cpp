#include <osg/Texture>

#include "snde/openscenegraph_layerwindow.hpp"


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
      
      // disable the Fbo ... like disableFboAfterRendering
      
      GLuint fboId = Info.getState()->getGraphicsContext()->getDefaultFboId();
      Info.getState()->get<osg::GLExtensions>()->glBindFramebuffer(GL_FRAMEBUFFER_EXT, fboId);
    }
    
    // If we had bound our own framebuffer in the predraw callback then
    // this next call would make sense (except that we should really
    // use the GraphicsContext's default FBO, not #0
    //Info.getState()->get<osg::GLExtensions>()->glBindFramebuffer( GL_FRAMEBUFFER_EXT, 0 );
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
    
    
    osg::ref_ptr<osg::FrameBufferObject> FBO;
    
    // get the OpenSceneGraph osg::Camera::FRAME_BUFFER_OBJECT
    // RenderTargetImplementation's FBO, by looking up the renderer
    // based on our viewer. 
    osgViewer::Renderer *Rend = dynamic_cast<osgViewer::Renderer *>(Viewer->getCamera()->getRenderer());
    osgUtil::RenderStage *Stage = Rend->getSceneView(0)->getRenderStage();
    
    Stage->setDisableFboAfterRender(true); // need to drop back to the default FBO so we can do compositing


    // OSG doesn't actually resize the texture so we need to do that ourselves in case it has changed
    osg::Texture::TextureObject *OutputBufTexObj = outputbuf->getTextureObject(Info.getState()->getContextID());
    if (OutputBufTexObj) {
      // (If OutputBufTexObj doesn't exist then it will be created with the correct parameters when needed
      // and this is unnecessary)
      OutputBufTexObj->bind();      
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
    GLenum status = Info.getState()->get<osg::GLExtensions>()->glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
    
    if (status != GL_FRAMEBUFFER_COMPLETE) {
      if (status==GL_FRAMEBUFFER_UNSUPPORTED) {
	throw snde_error("osg_layerwindow: Framebuffer configuration not supported by OpenGL implementation");
      } else {
	throw snde_error("osg_layerwindow: Unknown framebuffer error: %x",(unsigned)status);
	
      }
      
    }
    
    //assert(glGetError()== GL_NO_ERROR);
    
    if (readback) {
      // Verify the framebuffer configuration (Read mode)
      
      GLenum status = Info.getState()->get<osg::GLExtensions>()->glCheckFramebufferStatus(GL_READ_FRAMEBUFFER);
      
      if (status != GL_FRAMEBUFFER_COMPLETE) {
	if (status==GL_FRAMEBUFFER_UNSUPPORTED) {
	  throw snde_error("osg_layerwindow: Framebuffer configuration not supported by OpenGL implementation");
	} else {
	  throw snde_error("osg_layerwindow: Unknown framebuffer error: %x",(unsigned)status);
	  
	}
	
      }
    }
    
    //assert(glGetError()== GL_NO_ERROR);
    
    // Make sure we are drawing onto the FBO attachment
    glDrawBuffer(GL_COLOR_ATTACHMENT0);

    //assert(glGetError()== GL_NO_ERROR);


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
    Cam->setPreDrawCallback(predraw);
    Cam->setPostDrawCallback(postdraw);
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
    Cam->setPreDrawCallback(nullptr);
    Cam->setPostDrawCallback(nullptr);
    
    Cam->setGraphicsContext(nullptr);
  }


  void osg_layerwindow::resizedImplementation(int x,int y,
					      int width,
					      int height)
  {

    
    assert(x==0  && y==0);
    
    if (_traits) {
      if (width != _traits->width || height != _traits->height) {
	
	//depthbuf->setSize(width,height);
	outputbuf->setTextureSize(width,height);
	snde_warning("layerwindow: setting texture size to %d by %d", width,height);
	//Cam->setViewport(0,0,pix_width,height);
	
	
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
      
      
      osg::ref_ptr<osg::State> ourstate=new osg::State();
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
      
	
      outputbuf = new osg::Texture2D();
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
      postdraw = new osg_layerwindow_postdraw_callback(Viewer,readback ? outputbuf : nullptr); 
      

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
    
    getState()->reset(); // the OSG-expected state for THIS WINDOW may have been messed up (e.g. by another window). So we need to reset the assumptions about the OpenGL state

    // !!!*** reset() above may be unnecessarily pessimistic, dirtying all array buffers, etc. (why???)
    
    getState()->apply();
    
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
    //OSG_INFO << "releaseContext()\n";
    //outputbuf->getTextureObject(getState()->getContextID())->bind();
    
    //getState()->get<osg::GLExtensions>()->glBindFramebuffer( GL_FRAMEBUFFER_EXT, 0 );
    
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
