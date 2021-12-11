
#ifndef SNDE_OPENSCENEGRAPH_RENDERER_HPP
#define SNDE_OPENSCENEGRAPH_RENDERER_HPP

// This is missing from the OSG OpenGL headers on Anaconda on Windows for some reason... it is the only thing missing...
#ifndef GL_RGB8UI
#define GL_RGB8UI 0x8D7D
#endif

#include <osgViewer/GraphicsWindow>
#include <osgViewer/Viewer>
#include <osg/Texture2D>
#include <osgGA/TrackballManipulator>
#include <osg/MatrixTransform>
#include <osgUtil/SceneView>
#include <osgViewer/Renderer>

#include "snde/recstore.hpp"
#include "snde/openscenegraph_rendercache.hpp"

// Threading approach:
// Can delegate most rendering (exc. final compositing) to
// a separate thread, but be warned. Per QT, not all graphics
// drivers are thread-safe, so best to check the QT capability report
//
// Either way, we use a condition variable to signal a sub-thread
// that an update has occurred. This sub-thread then takes the
// latest ready globalrev and parameters and does all of the
// calculations/updates. If rendering is considered thread-safe,
// then this same sub-thread renders each of the layers in
// OpenGL Framebuffers and then
// triggers a QT redraw in the main GUI thread.
// The main thread redraw sees that an update is available,
// performs compositing, and flags the sub-thread that it is OK
// for it to continue.
//
// If rendering is not considered thread-safe, then the rendering
// (but not the calculations/updates) are moved into the main thread
// redraw but otherwise the process proceeds identically)

// There is a potential issue if QT wants a paint while the other thread is 

// (Note that nothing in this module is QT-specific)

// general info about opengl off-screen rendering:
// https://stackoverflow.com/questions/9742840/what-are-the-steps-necessary-to-render-my-scene-to-a-framebuffer-objectfbo-and
// OpenSceneGraph: Use FrameBufferObject, such as in
// https://github.com/openscenegraph/OpenSceneGraph/blob/master/examples/osgfpdepth/osgfpdepth.cpp
// or RTT: http://beefdev.blogspot.com/2012/01/render-to-texture-in-openscenegraph.html
// *** order-independent-transparency depth peeling example https://github.com/openscenegraph/OpenSceneGraph/blob/34a1d8bc9bba5c415c4ff590b3ea5229fa876ba8/examples/osgoit/DepthPeeling.cpp

// https://github.com/openscenegraph/OpenSceneGraph/blob/master/examples/osgmultiplerendertargets/osgmultiplerendertargets.cpp

// OSG fbo creation: https://github.com/openscenegraph/OpenSceneGraph/blob/3141cea7c102cf7431a9fa1b55414aa4ff2f6495/examples/osgfpdepth/osgfpdepth.cpp except this creates a depth texture

// Basically, you generate a framebuffer handle, (glGenFramebuffers, in osg FrameBufferObject::apply()
// a texture handle, glGenTextures (presumably osg::Texture2D?)
// and a render buffer (depth) handle. glGenRenderbuffers (RenderBuffer::getObjectID) 

// The framebuffer must be bound to the current context (camera operation?)
// Likewise the texure and depth buffers must be bound.
// Texture resolution defined by glTexImage2D. (osg::Texture2D)
// Consider setting texture min_filter and mag_filter parameters.
// Use glFramebufferTexture2D to attach the framebuffer
// to the texture.
// Use glRenderBufferStorage to define the depth buffer resolution
// Use glFramebufferRenderBuffer to attach the renderbuffer to the framebuffer
// Use glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) to verify supported mode
// set up glViewport according to geometry
//
// in OSG the various attachments are done with FrameBufferAttachment::attach in FrameBufferObject.cpp which is triggered by FrameBufferObject.setAttachment()
// rttCamera->setRenderTargetImplementation(Camera::FRAME_BUFFER_OBJECT)
// camera->attach(Camera::COLOR_BUFFER, colorTexture.get(), 0, 0, false,
//                   config.coverageSamples, config.depthSamples);
// camera->attach(Camera::DEPTH_BUFFER, depthTexture.get()
// See also FboTest in osgmemorytest.cpp
// Also: https://stackoverflow.com/questions/31640707/sequential-off-screen-rendering-screen-capture-without-windowing-system-using

// Viewer: Must setThredingModel(SingleThreaded); !!!

// Compositor specifics
// --------------------
// Consider two threads (these are the same thread
// for the GLUT compositor example, but are
// different threads in the QT GUI
//   A: Main GUI Thread + Compositor thread
//   B: Layer Rendering thread
//
// The threads interact using condition variables
// with the main gui thread being the leader and
// the layer rendering thread being the follower.
// The layer rendering thread is only allowed to
// execute when delegated to by the main GUI thread.
// When the main GUI thread has delegated to the
// layer rendering thread, the main GUI thread
// is not itself allowed to do any rendering. 
// 
// Compositor initialized in main GUI thread
// Rendercache initialized in layer rendering thread
//
// Layer rendering thread maintains a set of
// osg::Viewers, with osg_layerwindows or the
// osg_qtoffscreenlayer subclass as their
// "osg::GraphicsWindow", one per currently defined
// channel (These release their rootnode if they
// are not activated for rendering). These are
// all created in the layer rendering thread and
// thus may (only) execute in that thread. NOTE:
// the QOffscreenSurface must be created or destroyed
// only in the main GUI thread so that functionality
// will have to be delegated (!) -- Delegate with QtConcurrent
// to a thread pool containing just the main GUI thread
// and wait on the QFuture.
//
// The Layer rendering thread also provides the
// graticule layer.
//
// When an update is triggered, the layer rendering
// thread triggers the relevant recstore_display_transforms
// and waits for completion. Then it goes through
// all activated channels, finds their corresponding
// osg::Viewer and calls Viewer.frame() to render
// each to their corresponding framebuffer object.
// The layer rendering thread then notifies the
// main GUI thread that it is done.
//
// The main GUI thread can then assemble the layers
// from the corresponding framebuffer object textures
// and render into the QOpenGLWidget.
//
// In the class the notification methods just call
// the other thread code directly; a subclass implements the
// thread dispatching.

// NOTES:
// Aternate option: Need to set RenderStage FrameBufferObject  and perhaps remove DisableFboAfterRender
// Renderstage accessed through osgViewer::Renderer::getSceneView(0 or 1)->GetRenderStage() -- which was set in SceneView->setDefaults()...
// Renderer probably accessible through camera::getRenderer
// (Renderer is a GraphicsOperation), created in View constructor

// want doCopyTexture = false... requries "callingContext=useContext" (?) but that tries to enable pBuffer ... No... instead don't assign texture, just
// bind it at last minute? ... but bound texture required for _rtt?
// maybe no read_fbo? 
//
// Apply texture binding to _fbo in predrawcallback? 
// overriding the provided RenderBuffer?
// Then do readback in postdraw... 

namespace snde {
  class osgViewerCompat34: public osgViewer::Viewer {
    // derived version of osgViewer::Viewer that gives compat34GetRequestContinousUpdate()
    // alternative to osg v3.6 getRequestContinousUpdate()
  public:
    bool compat34GetRequestContinousUpdate()
    {
      return _requestContinousUpdate;
    }
  };



  class osg_layerwindow_postdraw_callback: public osg::Camera::DrawCallback {
  public:
    osg::ref_ptr<osgViewer::Viewer> Viewer;
    osg::ref_ptr<osg::Texture2D> outputbuf_tex;
    std::shared_ptr<std::shared_ptr<std::vector<unsigned char>>> readback_pixels; // double pointer to work around const callbacks
    osg_layerwindow_postdraw_callback(osg::ref_ptr<osgViewer::Viewer> Viewer, osg::ref_ptr<osg::Texture2D> outputbuf_tex) :
      Viewer(Viewer),
      outputbuf_tex(outputbuf_tex)
    {
      readback_pixels=std::make_shared<std::shared_ptr<std::vector<unsigned char>>>();
    }
    
    virtual void operator()(osg::RenderInfo &Info) const
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
	
      }

      // If we had bound our own framebuffer in the predraw callback then
      // this next call would make sense (except that we should really
      // use the GraphicsContext's default FBO, not #0
      //Info.getState()->get<osg::GLExtensions>()->glBindFramebuffer( GL_FRAMEBUFFER_EXT, 0 );
    }
  };

  class osg_layerwindow_predraw_callback: public osg::Camera::DrawCallback {
  public:
    osg::ref_ptr<osgViewer::Viewer> Viewer;
    osg::ref_ptr<osg::Texture2D> outputbuf;
    //osg::ref_ptr<osg::RenderBuffer> depthbuf;
    bool readback;
    
    osg_layerwindow_predraw_callback(osg::ref_ptr<osgViewer::Viewer> Viewer,osg::ref_ptr<osg::Texture2D> outputbuf,bool readback) :
      Viewer(Viewer),
      outputbuf(outputbuf),
      readback(readback)
    {

    }

    virtual void operator()(osg::RenderInfo &Info) const
    {
      //OSG_INFO << "preDraw()\n";


      osg::ref_ptr<osg::FrameBufferObject> FBO;

      // get the OpenSceneGraph osg::Camera::FRAME_BUFFER_OBJECT
      // RenderTargetImplementation's FBO, by looking up the renderer
      // based on our viewer. 
      osgViewer::Renderer *Rend = dynamic_cast<osgViewer::Renderer *>(Viewer->getCamera()->getRenderer());
      osgUtil::RenderStage *Stage = Rend->getSceneView(0)->getRenderStage();
      
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
      if (readback) {
	FBO->apply(*Info.getState(),osg::FrameBufferObject::READ_FRAMEBUFFER);	
      }

      // Verify the framebuffer configuration (Draw mode)
      GLenum status = Info.getState()->get<osg::GLExtensions>()->glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
	
      if (status != GL_FRAMEBUFFER_COMPLETE) {
	if (status==GL_FRAMEBUFFER_UNSUPPORTED) {
	  throw snde_error("osg_layerwindow: Framebuffer configuration not supported by OpenGL implementation");
	} else {
	  throw snde_error("osg_layerwindow: Unknown framebuffer error: %x",(unsigned)status);
	  
	}
	
      }


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

      
      // Make sure we are drawing onto the FBO attachment
      glDrawBuffer(GL_COLOR_ATTACHMENT0);


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
  };

  
  class osg_layerwindow: public osgViewer::GraphicsWindow {
  public:
    osg::ref_ptr<osgViewer::Viewer> Viewer;
    osg::ref_ptr<osg::GraphicsContext> shared_context;
    bool readback;
    
    //osg::ref_ptr<osg::FrameBufferObject> FBO;
    osg::ref_ptr<osg::Texture2D> outputbuf;
    osg::ref_ptr<osg::RenderBuffer> depthbuf;

    //osg::ref_ptr<osg::Image> readback_img;


    osg::ref_ptr<osg_layerwindow_predraw_callback> predraw;
    osg::ref_ptr<osg_layerwindow_postdraw_callback> postdraw;

    // NOTE: There is a test of osg_layerwindow functionality
    // in tests/osg_layerwindow_test.cpp
    osg_layerwindow(osg::ref_ptr<osgViewer::Viewer> Viewer,osg::ref_ptr<osg::GraphicsContext> shared_context,int width, int height,bool readback) :
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

    void setup_camera(osg::ref_ptr<osg::Camera> Cam)
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

    void clear_from_camera(osg::ref_ptr<osg::Camera> Cam)
    {
      Cam->setPreDrawCallback(nullptr);
      Cam->setPostDrawCallback(nullptr);
      
      Cam->setGraphicsContext(nullptr);
    }

    void resizedImplementation(int x,int y,
			       int width,
			       int height)
    {


      assert(x==0  && y==0);
      
      if (_traits) {
	if (width != _traits->width || height != _traits->height) {
	  
	  depthbuf->setSize(width,height);
	  outputbuf->setTextureSize(width,height);	  
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

    void init()
    {

      if (valid()) {

	
	osg::ref_ptr<osg::State> ourstate=new osg::State();
	ourstate->setGraphicsContext(this);

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
      
    virtual const char *libraryName() const { return "snde"; }
    virtual const char *className() const { return "osg_layerwindow"; }

    virtual bool valid() const
    {
      return true; 
    }

    bool makeCurrentImplementation()
    {
      OSG_INFO << "makeCurrent()\n";

      getState();

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

    bool releaseContextImplementation()
    {
      //OSG_INFO << "releaseContext()\n";
      //outputbuf->getTextureObject(getState()->getContextID())->bind();
      
      //getState()->get<osg::GLExtensions>()->glBindFramebuffer( GL_FRAMEBUFFER_EXT, 0 );

      return true;
    }

    bool realizeImplementation()
    {
      return true;
    }
    bool isRealizedImplementation() const
    {
      return true;
    }
    void closeImplementation()
    {
      
    }
    void swapBuffersImplementation()
    {
      
    }
    void grabFocus()
    {

    }
    void grabFocusIfPointerInWindow()
    {

    }
    void raiseWindow()
    {

    }

    //~osg_layerwindow()
    //{
    //  
    //  
    //}
    
  };
  
  class osg_compositor { // Used as a base class for QTRecRender, which also inherits from QOpenGLWidget
  public:
    osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow;
    //osg::ref_ptr<osgViewer::Viewer> Viewer;
    osg::ref_ptr<osgViewerCompat34> Viewer;
    osg::ref_ptr<osg::Camera> Camera;
    //osg::ref_ptr<osg::Node> RootNode;

    // PickerCrossHairs and GraticuleTransform for 2D image and 1D waveform rendering
    osg::ref_ptr<osg::MatrixTransform> PickerCrossHairs;
    osg::ref_ptr<osg::MatrixTransform> GraticuleTransform; // entire graticule hangs off of this!
    
    
    std::shared_ptr<osg_rendercache> RenderCache;
    std::shared_ptr<recording_set_state> RenderingState; // RenderingState is a recording set state that branches off the globalrev being rendered, and has any data transform channels needed for rendering added to it. The next RenderingState comes will come from the latest globalrev plus any needed data transform channels. If those data transform channels are unchanged since the prior rendering state they can be imported from this stored pointer. 

    osg_compositor(osg::ref_ptr<osgViewer::Viewer> Viewer,osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
		 //osg::ref_ptr<osg::Node> RootNode,
		 bool twodimensional);

    void perform_render(std::shared_ptr<recdatabase> recdb,
			const std::vector<display_requirement> & display_reqs,
			std::shared_ptr<recstore_display_transforms> display_transforms,
			const std::vector<std::shared_ptr<display_channel>> &channels_to_display, 
			std::shared_ptr<display_info> display,
			bool singlechannel);
    
    void SetPickerCrossHairs();
    
    void SetRootNode(osg::ref_ptr<osg::Node> RootNode);

    /* NOTE: to actually render, do any geometry updates, 
       then call Viewer->frame() */
    /* NOTE: to adjust size, first send event, then 
       change viewport:

    GraphicsWindow->getEventQueue()->windowResize(x(),y(),width,height);
    GraphicsWindow->resized(x(),y(),width,height);
    Camera->setViewport(0,0,width,height);
    SetProjectionMatrix();

    */




    virtual void ClearPickedOrientation()
    {
      // notification from picker to clear any marked orientation
      // probably needs to be reimplemented by derived classes
    }

    std::tuple<double,double> GetPadding(size_t drawareawidth,size_t drawareaheight);

    std::tuple<double,double> GetScalefactors(std::string recname);

    osg::Matrixd GetChannelTransform(std::string recname,std::shared_ptr<display_channel> displaychan,size_t drawareawidth,size_t drawareaheight,size_t layer_index);
    
    
  };


}

#endif // SNDE_OPENSCENEGRAPH_RENDERER_HPP



