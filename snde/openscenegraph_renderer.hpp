
#ifndef SNDE_OPENSCENEGRAPH_RENDERER_HPP
#define SNDE_OPENSCENEGRAPH_RENDERER_HPP

#include <osgViewer/GraphicsWindow>
#include <osgViewer/Viewer>
#include <osg/Texture2D>
#include <osgGA/TrackballManipulator>
#include <osg/MatrixTransform>

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

  class osg_layerwindow: public osgViewer::GraphicsWindow {
  public:
    osg::ref_ptr<osg::Camera> Cam;
    osg::ref_ptr<osg::Texture2D> outputbuf;
    osg::ref_ptr<osg::GraphicsContext> shared_context;
    osg_layerwindow(osg::ref_ptr<osg::GraphicsContext> shared_context) :
      osgViewer::GraphicsWindow(),
      shared_context(shared_context)
    {
      Cam = new osg::Camera();
      Cam->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
      outputbuf = new osg::Texture2D();
      outputbuf->setSourceFormat(GL_RGBA);
      outputbuf->setInternalFormat(GL_RGB8UI);
      outputbuf->setSourceType(GL_UNSIGNED_BYTE);
      Cam->attach(osg::Camera::COLOR_BUFFER0,outputbuf);
      
    }

    void set_size_and_bounds(int pix_x,
			     int pix_y,
			     int pix_width,
			     int pix_height)
    {
      if (_traits) {
	resized(pix_x,pix_y,pix_width,pix_height);
      } else {
	_traits = new osg::GraphicsContext::Traits();
	_traits->x=pix_x;
	_traits->y=pix_y;
	_traits->width=pix_width;
	_traits->height=pix_height;
	_traits->windowDecoration=false;
	_traits->doubleBuffer=false;
	_traits->sharedContext=shared_context;
	_traits->vsync=false;
	init();
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
	
      }
    }
      
    virtual const char *libraryName() const { return "snde"; }
    virtual const char *className() const { return "osg_layerwindow"; }


    bool makeCurrentImplementation()
    {
      return true;
    }

    bool releaseContextImplementation()
    {
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
  
  class osg_renderer { // Used as a base class for QTRecRender, which also inherits from QOpenGLWidget
  public:
    osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow;
    //osg::ref_ptr<osgViewer::Viewer> Viewer;
    osg::ref_ptr<osgViewerCompat34> Viewer;
    osg::ref_ptr<osg::Camera> Camera;
    //osg::ref_ptr<osg::Node> RootNode;
    osg::ref_ptr<osgGA::TrackballManipulator> Manipulator; // manipulator for 3D mode

    // PickerCrossHairs and GraticuleTransform for 2D image and 1D waveform rendering
    osg::ref_ptr<osg::MatrixTransform> PickerCrossHairs;
    osg::ref_ptr<osg::MatrixTransform> GraticuleTransform; // entire graticule hangs off of this!

    
    std::shared_ptr<osg_rendercache> RenderCache;
    std::shared_ptr<recording_set_state> RenderingState; // RenderingState is a recording set state that branches off the globalrev being rendered, and has any data transform channels needed for rendering added to it. The next RenderingState comes will come from the latest globalrev plus any needed data transform channels. If those data transform channels are unchanged since the prior rendering state they can be imported from this stored pointer. 

    bool twodimensional;
    
    osg_renderer(osg::ref_ptr<osgViewer::Viewer> Viewer,osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
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



