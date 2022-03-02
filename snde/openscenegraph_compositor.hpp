#ifndef SNDE_OPENSCENEGRAPH_COMPOSITOR_HPP
#define SNDE_OPENSCENEGRAPH_COMPOSITOR_HPP

#include <osgViewer/GraphicsWindow>
#include <osgViewer/Viewer>
#include <osg/Texture2D>
#include <osgGA/TrackballManipulator>
#include <osg/MatrixTransform>
#include <osgUtil/SceneView>
#include <osgViewer/Renderer>


#include "snde/recstore.hpp"
#include "snde/openscenegraph_rendercache.hpp"


namespace snde {

  class osg_compositor;
  class osg_renderer; // openscenegraph_renderer.hpp
  class osgViewerCompat34; // openscenegraph_renderer.hpp
  
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


  class osg_compositor_eventhandler: public osgGA::GUIEventHandler {
  public:
    osg_compositor *comp; // use the compositor_dead flag (only set in main GUI loop thread) instead of a weak_ptr so that osg_compositor is compatible with both shared_ptr and QT ownership models. compositor_dead will be set in the destructor of the osg_compositor
    bool compositor_dead; // used as a validity flag for comp
    std::shared_ptr<display_info> display;
    
    osg_compositor_eventhandler(osg_compositor *comp,std::shared_ptr<display_info> display);

    osg_compositor_eventhandler(const osg_compositor_eventhandler &) = delete;
    osg_compositor_eventhandler & operator=(const osg_compositor_eventhandler &) = delete;
    virtual ~osg_compositor_eventhandler() = default; 

    virtual bool handle(const osgGA::GUIEventAdapter &eventadapt,
			osgGA::GUIActionAdapter &actionadapt);
  };
  
  

  // ****!!!!! Need resize method.. should call display->set_pixelsperdiv() !!!****
  // note that qt_osg_compositor derives from this and specializes it. 
  class osg_compositor { // Used as a base class for QTRecRender, which also inherits from QOpenGLWidget
  public:

    std::weak_ptr<recdatabase> recdb; // immutable once initialized
    // These pointers (not necessarily content) are immutable once initialized
    // and represent the composited view. In the QT environment they need
    // to be initialized from the main GUI thread (SNDE_OSGRCS_COMPOSITING
    // context). They generally are only to be worked with from the main thread
    // (compositing step)
    osg::ref_ptr<osgViewer::Viewer> Viewer;
    osg::ref_ptr<osg::Group> RootGroup;
    osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow;
    osg::ref_ptr<osg::Camera> Camera;
    osg::ref_ptr<osg_compositor_eventhandler> eventhandler; 

    std::map<std::string,osg::Matrixd> FutureChannelCamPose; // locked by admin mutex
    std::map<std::string,snde_coord> FutureChannelRotationCenterDist; // locked by admin mutex
    
    
    std::shared_ptr<display_info> display;
    std::string selected_channel; // locked by admin lock.... maybe selected channel should instead be handled like compositor_width, compositor_height, etc. below. or stored in display_info copy we will eventually have? 
    
    bool threaded;
    bool enable_threaded_opengl;
    bool enable_shaders;
    GLuint LayerDefaultFramebufferObject; // default FBO # for layer renderers 

    // PickerCrossHairs and GraticuleTransform for 2D image and 1D waveform rendering
    // They are initialized in the compositor's constructor and immutable thereafter
    osg::ref_ptr<osg::MatrixTransform> PickerCrossHairs;
    osg::ref_ptr<osg::MatrixTransform> GraticuleTransform; // entire graticule hangs off of this!


    // these are initialized in the ONDEMANDCALCS step and referenced in subsequent steps
    // (perhaps from different threads, but there will have been mutex/condition variable
    // synchronization since)
    std::vector<std::shared_ptr<display_channel>> channels_to_display;
    std::map<std::string,std::shared_ptr<display_requirement>> display_reqs;
    std::shared_ptr<recstore_display_transforms> display_transforms;

    // Render cache is maintained by the rendering step which runs in the rendering thread
    std::shared_ptr<osg_rendercache> RenderCache; // Should be freed (set to nullptr) ONLY by layer rendering thread with the proper OpenGL context loaded

    // Renderers maintained by the rendering step which runs in the rendering thread
    std::shared_ptr<std::map<std::string,std::shared_ptr<osg_renderer>>> renderers; // Should be freed (set to nullptr) ONLY by layer rendering thread with the proper OpenGL context loaded. This is locked by the admin lock because it may be accessed by event handling threads to pass events on to the viewer
    

    // Renderers created by rendering step which runs in the rendering thread, but
    // the integer texture ID numbers are used in the compositing step
    std::shared_ptr<std::map<std::string,std::pair<osg::ref_ptr<osg::Texture2D>,GLuint>>> layer_rendering_rendered_textures; // should be freed ONLY by layer rendering thread. Indexed by channel name; texture pointer valid in layer rendering thread and opengl texture ID number.

    //std::shared_ptr<std::map<std::string,std::pair<osg::ref_ptr<osg::Texture2D>,GLuint>>> compositing_textures; // should be freed ONLY by compositing thread. Indexed by channel name. Texture pointer valid in compositing thread

    // CompositingShaderProgram is owned by the compositing step.
    osg::ref_ptr<osg::Program> CompositingShaderProgram;

    
    // Rendering consists of four phases, which may be in different threads,
    // but must proceed sequentially with one thread handing off to the next
    // 1. Waiting for trigger; the trigger indicates that an update
    //    is necessary, such as the need_rerender or need_recomposite
    //    flags below. It can come from QT, from the presence of a new
    //    ready globalrevision, etc.  Any thread can do the trigger,
    //    by locking the admin mutex, setting the flag, and calling the
    //    state_update notify method. 
    // 2. Identification of channels to render and waiting for on-demand
    //    calculations (e.g. colormapping) to become ready
    // 3. Rendering of enabled channels onto textures.
    //    (Note: Easy optimization opportunity: Don't re-render if scene
    //     hasn't changed -- i.e. same cached elements) 
    // 4. Compositing of enabled textures onto final output.
    //    In the QT integration this final step must be done in the
    //    main GUI event loop. The others do not have to, unless the
    //    platform doesn't support threaded OpenGL, in which case
    //    step #3 also has to be in the main GUI event loop.

    //std::mutex execution_lock; // Whoever is executing an above step must own th execution_lock. It is early in the locking order, prior to recdb locks (see lockmanager.hpp)

    std::mutex admin; // the admin lock primarily locks next_state, execution_notify, need_rerender, need_recomposite
    std::condition_variable execution_notify; //paired with the admin lock, above
    int next_state; // When you are ready to start doing one of the above
    // operations, acquire the admin lock and check next_state. If next_state
    // is your responsibility (i.e. if
    // responsibility_mapping.at(std::this_thread::get_id) contains next_state,
    // then you take over execution. Clear need_rerender or need_recomposite
    // as appropriate and then drop the admin lock and start executing.
    // If next_state is not your responsibility, 
    // you need to either drop the admin lock and do other stuff, or
    // keep the admin lock but wait on the execution_notify condition
    // variable. Once you finish, re-acquire the admin lock and
    // (checking first to see if next_state has been modified to exceed 
    // SNDE_SGRCS_CLEANUP, in which case you should return to the main loop and
    // initate cleanup procedures if possible)
    // set next_state to one of the SNDE_OSGRCS_xxxxs
    // below representing the next step. Finally drop the admin lock and
    // call the state_update notify method, which triggers the execution_notify
    // condition variable with notify_all() (and subclasses might do other
    // notification mechanisms as well such as QT slots). 

    // reponsibility_mapping needs to be pre-configured with
    // which threads take which responsibilities (SNDE_OSGRCS... below)
    // So any given thread needs to check if it can take responsibility
    // for a given next_state. Only a single thread should every be allowed
    // to have responsibility for any given state. 
    std::map<std::thread::id,std::set<int>> responsibility_mapping; // this is immutable once created


    bool threads_started; // whether we have performed the thread/responsibility_mapping initialization (regardless of if we are using a threaded model)

    // These all locked with admin lock
    bool need_rerender; 
    bool need_recomposite;
    bool need_resize; // pull new size from resize_width and resize_height, below

    int resize_width;  // set with need_resize
    int resize_height; 
    
#define SNDE_OSGRCS_WAITING 1 // Entry #1, above. This one is special
    // in that the responsible thread should be waiting on the
    // condition variable and checking the need_rerender and
    // need_recomposite bools, and marking the next state as
    // SNDE_OSGRCS_ONDEMANDCALCS or SNDE_OSGRCS_COMPOSITING as
    // appropriate if one of those is set.
#define SNDE_OSGRCS_ONDEMANDCALCS 2 // Entry #2, above
#define SNDE_OSGRCS_RENDERING 3 // Entry #3, above
#define SNDE_OSGRCS_COMPOSITING 4 // Entry #4, above
    //#define SNDE_OSGRCS_CLEANUP 5
#define SNDE_OSGRCS_COMPOSITING_CLEANUP 6 // command to worker thread(s) to cleanup
#define SNDE_OSGRCS_RENDERING_CLEANUP 7 // command to worker thread(s) to cleanup
#define SNDE_OSGRCS_EXIT 8 // command to worker thread(s) to exit
    

    std::shared_ptr<std::thread> worker_thread; // NOTE: Not used by QT subclass
    std::shared_ptr<std::thread::id> worker_thread_id; // C++ id of worker thread (set even by QT subclass); Immutable once published (after start())
    
    
    size_t compositor_width;  // updated in peform_ondemand_calcs; Should only be modified there, and read by whoever is running a step
    size_t compositor_height;
    double borderwidthpixels;
    std::map<std::string,size_t> ColorIdx_by_channelpath; // updated in perform_ondemand_calc;

    std::atomic_bool request_continuous_update; // updated in perform_layer_rendering(); true if any renderer is requesting continuous updates
    
    
    // ***!!! NOTE: don't set platform_supports_threaded_opengl unless you have arranged some means for the worker thread to operate in a different OpenGL context that shares textures with the main context !!!***
    // NOTE 2: This will be (is?) subclassed by a QT version that does just that.
    // ***!!!! Should provide some means to set the default framebuffer for the various GraphicsWindows ***!!!
    osg_compositor(std::shared_ptr<recdatabase> recdb,
		   std::shared_ptr<display_info> display,
		   osg::ref_ptr<osgViewer::Viewer> Viewer,osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
		   bool threaded,bool enable_threaded_opengl,
		   bool enable_shaders,
		   GLuint LayerDefaultFramebufferObject=0);

    osg_compositor(const osg_compositor &) = delete;
    osg_compositor & operator=(const osg_compositor &) = delete;
    virtual ~osg_compositor();
    

    virtual void trigger_rerender();
    virtual void wait_render();
    virtual void set_selected_channel(const std::string &selected_name);
    virtual std::string get_selected_channel();
    virtual void perform_ondemand_calcs(std::unique_lock<std::mutex> *adminlock);
    virtual void perform_layer_rendering(std::unique_lock<std::mutex> *adminlock);
    virtual void perform_compositing(std::unique_lock<std::mutex> *adminlock);
    virtual bool this_thread_ok_for_locked(int action);
    virtual bool this_thread_ok_for(int action);
    virtual void wake_up_ondemand_locked(std::unique_lock<std::mutex> *adminlock);
    virtual void wake_up_renderer_locked(std::unique_lock<std::mutex> *adminlock);
    virtual void wake_up_compositor_locked(std::unique_lock<std::mutex> *adminlock);
    virtual void clean_up_renderer_locked(std::unique_lock<std::mutex> *adminlock);
 

    virtual void dispatch(bool return_if_idle,bool wait, bool loop_forever);
    virtual void worker_code();
    virtual void _start_worker_thread(std::unique_lock<std::mutex> *adminlock);
    virtual void _join_worker_thread();
    virtual void resize_compositor(int width, int height);

    virtual snde_orientation3 get_camera_pose(std::string channel_path); // get the camera pose (or a null orientation) for the given channel
    virtual void set_camera_pose(std::string channel_path,const snde_orientation3 &newpose);
    virtual snde_coord get_rotation_center_dist(std::string channel_path); // get the viewer rotation center
    virtual void set_rotation_center_dist(std::string channel_path,snde_coord newcenterdist);
    
    

    virtual void start();
    virtual void stop();
    virtual void SetPickerCrossHairs();

    //void SetPickerCrossHairs();
    
    //void SetRootNode(osg::ref_ptr<osg::Node> RootNode);



    
    //virtual void ClearPickedOrientation()
    //{
    //  // notification from picker to clear any marked orientation
    // // probably needs to be reimplemented by derived classes
    //}
    
    //std::tuple<double,double> GetPadding(size_t drawareawidth,size_t drawareaheight);

    //std::tuple<double,double> GetScalefactors(std::string recname);

    //osg::Matrixd GetChannelTransform(std::string recname,std::shared_ptr<display_channel> displaychan,size_t drawareawidth,size_t drawareaheight,size_t layer_index);
    
    
  };

  
};




#endif // SNDE_OPENSCENEGRAPH_COMPOSITOR_HPP
