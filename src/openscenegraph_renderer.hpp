
#ifndef SNDE_OPENSCENEGRAPH_RENDERER_HPP
#define SNDE_OPENSCENEGRAPH_RENDERER_HPP


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

  
  class osg_renderer {
  public:
    osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow;
    //osg::ref_ptr<osgViewer::Viewer> Viewer;
    osg::ref_ptr<osgViewerCompat34> Viewer;
    osg::ref_ptr<osg::Camera> Camera;
    osg::ref_ptr<osg::Node> RootNode;
    osg::ref_ptr<osgGA::TrackballManipulator> Manipulator; // manipulator for 3D mode


    bool twodimensional;
    
    osg_renderer(osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
		 osg::ref_ptr<osg::Node> RootNode,
		 bool twodimensional) :
      GraphicsWindow(GraphicsWindow),
      //Viewer(new osgViewer::Viewer()),
      Viewer(new osgViewerCompat34()),
      Camera(Viewer->getCamera()),
      RootNode(RootNode),
      twodimensional(twodimensional)
    {
      
      Camera->setGraphicsContext(GraphicsWindow);

      // set background color to blueish
      Camera->setClearColor(osg::Vec4(.1,.1,.3,1.0));

      // need to enable culling so that linesegmentintersector (openscenegraph_picker)
      // behavior matches camera behavior
      // (is this efficient?)
      Camera->setComputeNearFarMode( osg::CullSettings::COMPUTE_NEAR_FAR_USING_PRIMITIVES );
      Camera->setCullingMode(osg::CullSettings::ENABLE_ALL_CULLING);
      //Viewer->setCamera(Camera);

      Manipulator = new osgGA::TrackballManipulator();
      Viewer->setThreadingModel(osgViewer::Viewer::SingleThreaded);
      Viewer->setSceneData(RootNode);
      Viewer->setCameraManipulator(Manipulator);
      
      // Caller should set camera viewport,
      // implement SetProjectionMatrix(),
      // SetTwoDimensional()
      // and make initial calls to those functions
      // from their constructor,
      // then call Viewer->realize();
    }

    
    void SetRootNode(osg::ref_ptr<osg::Node> RootNode)
    {
      this->RootNode=RootNode;
      if (RootNode) {
	if (Viewer->getSceneData() != RootNode) {
	  Viewer->setSceneData(RootNode);
	}
      }
    }

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
    
    
  };


}

#endif // SNDE_OPENSCENEGRAPH_RENDERER_HPP



