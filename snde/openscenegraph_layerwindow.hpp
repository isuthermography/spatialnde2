#ifndef SNDE_OPENSCENEGRAPH_LAYERWINDOW_HPP
#define SNDE_OPENSCENEGRAPH_LAYERWINDOW_HPP


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
  class osg_renderer;

  
  class osg_layerwindow_postdraw_callback: public osg::Camera::DrawCallback {
  public:
    osg::ref_ptr<osgViewer::Viewer> Viewer;
    osg::ref_ptr<osg::Texture2D> outputbuf_tex;
    std::shared_ptr<std::shared_ptr<std::vector<unsigned char>>> readback_pixels; // double pointer to work around const callbacks
    //std::shared_ptr<GLenum> DrawBufferSave; // spot where the pre-draw callback saved the current value of glDrawBuffer
    osg_layerwindow_postdraw_callback(osg::ref_ptr<osgViewer::Viewer> Viewer, osg::ref_ptr<osg::Texture2D> outputbuf_tex);
    
    virtual void operator()(osg::RenderInfo &Info) const;
  };

  class osg_layerwindow_predraw_callback: public osg::Camera::DrawCallback {
  public:
    osg::ref_ptr<osgViewer::Viewer> Viewer;
    osg::ref_ptr<osg::Texture2D> outputbuf;
    //osg::ref_ptr<osg::RenderBuffer> depthbuf;
    //std::shared_ptr<GLenum> DrawBufferSave; // spot to save the current value of glDrawBuffer for the postdraw callback
    bool readback;
    
    osg_layerwindow_predraw_callback(osg::ref_ptr<osgViewer::Viewer> Viewer,osg::ref_ptr<osg::Texture2D> outputbuf,bool readback);

    virtual void operator()(osg::RenderInfo &Info) const;
  };

  
  class osg_layerwindow: public osgViewer::GraphicsWindow {
  public:
    osg::ref_ptr<osgViewer::Viewer> Viewer;
    osg::ref_ptr<osg::GraphicsContext> shared_context;
    bool readback;
    
    //osg::ref_ptr<osg::FrameBufferObject> FBO;
    osg::ref_ptr<osg::Texture2D> outputbuf;
    //osg::ref_ptr<osg::RenderBuffer> depthbuf;

    //osg::ref_ptr<osg::Image> readback_img;


    osg::ref_ptr<osg_layerwindow_predraw_callback> predraw;
    osg::ref_ptr<osg_layerwindow_postdraw_callback> postdraw;

    // NOTE: There is a test of osg_layerwindow functionality
    // in tests/osg_layerwindow_test.cpp
    osg_layerwindow(osg::ref_ptr<osgViewer::Viewer> Viewer,osg::ref_ptr<osg::GraphicsContext> shared_context,int width, int height,bool readback);
    osg_layerwindow(const osg_layerwindow &) = delete;
    osg_layerwindow & operator=(const osg_layerwindow &) = delete;
    virtual ~osg_layerwindow()=default;
    
    
    virtual void setup_camera(osg::ref_ptr<osg::Camera> Cam);
    virtual void clear_from_camera(osg::ref_ptr<osg::Camera> Cam);
    virtual void resizedImplementation(int x,int y,
				       int width,
				       int height);
    virtual void init();
      
    virtual const char *libraryName() const;
    virtual const char *className() const;
    virtual bool valid() const;
    
    virtual bool makeCurrentImplementation();
    virtual bool releaseContextImplementation();

    virtual bool realizeImplementation();
    virtual bool isRealizedImplementation() const;
    virtual void closeImplementation();
    virtual void swapBuffersImplementation();
    virtual void grabFocus();
    virtual void grabFocusIfPointerInWindow();
    virtual void raiseWindow();
    
    //{
    //  
    //  
    //}
    
  };



};



#endif // SNDE_OPENSCENEGRAPH_LAYERWINDOW_HPP
