#include <osgUtil/ShaderGen>
#include <osg/ShapeDrawable>

#include "snde/openscenegraph_geom_renderer.hpp"
#include "snde/rec_display.hpp"
#include "snde/display_requirements.hpp"


// access default shaders from OpenSceneGraph
// (WARNING: These might go into a namespace sometime!!!)
//extern char shadergen_frag[];
//extern char shadergen_vert[];

namespace snde {



  osg_geom_renderer::osg_geom_renderer(osg::ref_ptr<osgViewer::Viewer> Viewer, // use an osgViewerCompat34()
				       osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
				       std::string channel_path,bool enable_shaders) :
    osg_renderer(Viewer,GraphicsWindow,new osgGA::TrackballManipulator(),channel_path,SNDE_DRRT_GEOMETRY,enable_shaders),
    hudwidth(150),
    hudheight(150)
    //firstrun(true)
  {


    BuildCoordAxes();
    BuildCoordAxes_HUD();
    
    EventQueue=GraphicsWindow->getEventQueue();
    //Camera->setGraphicsContext(GraphicsWindow);
    
    // set background color to blueish
    Camera->setClearColor(osg::Vec4(.1,.1,.3,1.0));    
    Camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    

    // need to enable culling so that linesegmentintersector (openscenegraph_picker)
    // behavior matches camera behavior
    // (is this efficient?)
    // NOTE: Having this on made rendering fail !!!***
    //Camera->setComputeNearFarMode( osg::CullSettings::COMPUTE_NEAR_FAR_USING_PRIMITIVES );
    //Camera->setCullingMode(osg::CullSettings::ENABLE_ALL_CULLING);

    Viewer->setThreadingModel(osgViewer::Viewer::SingleThreaded); // We handle threading ourselves... this is REQUIRED
    Viewer->setCameraManipulator(Manipulator);
    (dynamic_cast<osgGA::OrbitManipulator *>(Manipulator.get()))->setAllowThrow(false); // leave at false unless/until we get the other animation infrastructure working (basically relevant event callbacks compatible with the OSG timers)
    //Viewer->addEventHandler(Manipulator);

    CoordAxesCamera->setGraphicsContext(GraphicsWindow);
    CoordAxesCamera->setViewport(0,0,hudwidth,hudheight);
    //CoordAxesCamera->setClearColor(osg::Vec4(.3,.1,.1,1.0));
    
    //CoordAxesCamera->setRenderTargetImplementation(Camera->getRenderTargetImplementation());
    //CoordAxesCamera->setDrawBuffer(Camera->getDrawBuffer());
    //CoordAxesCamera->setReadBuffer(Camera->getReadBuffer());
    //Viewer->addSlave(CoordAxesCamera,false);
    
    Viewer->realize();

    if (enable_shaders) {
      // Start with OSG 3.6 built-in shaders
      //ShaderProgram = new osg::Program();
      //ShaderProgram->addShader(new osg::Shader(osg::Shader::VERTEX, shadergen_vert));
      //ShaderProgram->addShader(new osg::Shader(osg::Shader::FRAGMENT, shadergen_frag));
      
      // Apply ShaderProgram to our camera
      // and add the required diffuseMap uniform
      osg::ref_ptr<osg::StateSet> CameraStateSet = Camera->getOrCreateStateSet();
      //CameraStateSet->setAttribute(ShaderProgram);
      //CameraStateSet->addUniform(new osg::Uniform("diffuseMap",0));

      // Apply ShaderGen stateset transformation to the camera
      // This transforms basic lighting, fog, and texture
      // to shader defines.
      osgUtil::ShaderGenVisitor ShaderGen;
      ShaderGen.assignUberProgram(CameraStateSet);
      // (Alternatively I think this would be equivalent to
      // Camera->accept(ShaderGen);
      ShaderGen.apply(*Camera);

    }
  }

  
  
  std::tuple<std::shared_ptr<osg_rendercacheentry>,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>>,bool>
  osg_geom_renderer::prepare_render(//std::shared_ptr<recdatabase> recdb,
				  std::shared_ptr<recording_set_state> with_display_transforms,
				  //std::shared_ptr<display_info> display,
				  std::shared_ptr<osg_rendercache> RenderCache,
				  const std::map<std::string,std::shared_ptr<display_requirement>> &display_reqs,
				  size_t width, // width of full-display viewport in pixels
				  size_t height) // height of full-display viewport in pixels
  {
    // look up render cache.
    std::shared_ptr<display_requirement> display_req = display_reqs.at(channel_path);
    osg_renderparams params{
      //recdb,
      RenderCache,
      with_display_transforms,
      
      display_req->spatial_bounds->left,
      display_req->spatial_bounds->right,
      display_req->spatial_bounds->bottom,
      display_req->spatial_bounds->top,
      width,
      height,
      
    };

    std::shared_ptr<osg_rendercacheentry> renderentry;
    bool modified=false;
    std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> locks_required; 
    


    if (display_req->spatial_bounds->bottom >= display_req->spatial_bounds->top ||
	display_req->spatial_bounds->left >= display_req->spatial_bounds->right) {
      // negative or zero display area
      if (RootTransform->getNumChildren()) {
	
	RootTransform->removeChildren(0,RootTransform->getNumChildren());
      }
      

      modified = true; 
    } else { // Positive display area 
      
      std::tie(renderentry,modified) = RenderCache->GetEntry(params,display_req,&locks_required);
      
      if (display_req->spatial_position->width != Camera->getViewport()->width() || display_req->spatial_position->height != Camera->getViewport()->height()) {
	GraphicsWindow->getEventQueue()->windowResize(0,0,display_req->spatial_position->width,display_req->spatial_position->height);
	GraphicsWindow->resized(0,0,display_req->spatial_position->width,display_req->spatial_position->height);
	Camera->setViewport(0,0,display_req->spatial_position->width,display_req->spatial_position->height);
	
	CoordAxesCamera->setViewport(0, // per OpenGL spec Viewport is measured from lower left, so the correct origin here is (0,0)
				     0, // (display_req->spatial_position->height > hudheight) ? (display_req->spatial_position->height-hudheight) : 0,
				     //(hudwidth > display_req->spatial_position->width) ? display_req->spatial_position->width : hudwidth,
				     //(hudheight > display_req->spatial_position->height) ? display_req->spatial_position->height : hudheight,
				     hudwidth,hudheight);
	
	modified = true;
      }
      Camera->setProjectionMatrixAsPerspective(30.0f,((double)width)/height,1.0f,10000.0f); // !!!*** Check last two parameters
      //snde_warning("setProjectionMatrixAsPerspective aspect ratio = %f",((double)width)/height);
      
      OrientCoordAxes(); // update orientation of coordinate axes

      {
	osg::Vec3f eye,center,up;
	Camera->getViewMatrixAsLookAt(eye,center,up);
	snde_debug(SNDE_DC_RENDERING,"Camera View on %s: eye: %f %f %f center: %f %f %f up: %f %f %f",channel_path.c_str(),eye[0],eye[1],eye[2],center[0],center[1],center[2],up[0],up[1],up[2]); 
      }
      
      // Create projection matrix.
      // Focal behavior comes from the spatial position width/height
      double fovy = 30.0; // degrees
      double aspect = display_req->spatial_position->drawareawidth/display_req->spatial_position->drawareaheight;
      double znear = 1.0;
      double zfar = 10000.0;
      
      double xshift = (display_req->spatial_bounds->right + display_req->spatial_bounds->left)/(display_req->spatial_bounds->right - display_req->spatial_bounds->left);
      double yshift = (display_req->spatial_bounds->top + display_req->spatial_bounds->bottom)/(display_req->spatial_bounds->top - display_req->spatial_bounds->bottom);
      
      double f = 1.0/tan(fovy/2.0); // per gluPerspective man page
      
      
      osg::Matrixf ProjectionMatrix(f/aspect, 0, 0, 0,  // per gluPerspecitve man page
				    0,        f, 0, 0,  // (remember osg Matrices appear transposed!)
				    0,        0, (zfar+znear)/(znear-zfar), -1,
				    -xshift/4.0,-yshift/4.0, (2*zfar*znear)/(znear-zfar), 0);  // View as the image shifts off screen is wrong. I don't think we're doing the xshift/yshift correctly
      
      //Camera->setProjectionMatrix(ProjectionMatrix);
      
      // (Do we need to set modified flag if the projection matrix changes???)
      
      if (renderentry) {
	std::shared_ptr<osg_rendercachegroupentry> rendergroup = std::dynamic_pointer_cast<osg_rendercachegroupentry>(renderentry);
	
	if (rendergroup) {
	  //if (Viewer->getSceneData() != rendergroup->osg_group){
	  //Viewer->setSceneData(rendergroup->osg_group);
	  //}
	  
	  if (RootTransform->getNumChildren() && RootTransform->getChild(0) != rendergroup->osg_group) {
	    RootTransform->removeChildren(0,RootTransform->getNumChildren());
	  }
	  if (!RootTransform->getNumChildren()) {
	    RootTransform->addChild(rendergroup->osg_group);
	    RootTransform->addChild(CoordAxesCamera);
	  }
	  if (!Viewer->getSceneData()) {
	    Viewer->setSceneData(RootTransform);
	  }

	  {
	    osg::GraphicsOperation *Op=CoordAxesCamera->getRenderer(); 
	    snde_debug(SNDE_DC_RENDERING,"CoordAxesCamera->getRenderer is 0x%llx",(unsigned long long)Op);
	  }
	} else {
	  snde_warning("openscenegraph_3d_renderer: cache entry not convertable to an osg_group rendering channel %s",channel_path.c_str());
	}
      } else {
	snde_warning("openscenegraph_3d_renderer: cache entry not available rendering channel %s",channel_path.c_str());
	
      }
      
      //if (firstrun) {
      //  Viewer->realize();
      //  firstrun=false;
      //}

    /*    osg::Matrixd viewmat = Camera->getViewMatrix();
    {
      int i,j;
      printf("Camera view matrix:\n");
      for (i=0; i < 4; i++) {
	for (j=0; j < 4; j++) {
	  printf("%8f ",viewmat(i,j));
	}
	printf("\n");
      }
      
      }*/
    }

    if (modified && enable_shaders) {
      // Apply use of shaders instead of old-style lighting and texture to the modified tree
      osgUtil::ShaderGenVisitor ShaderGen;
      // This transforms basic lighting, fog, and texture
      // to shader defines.

      // The shader stateset was already applied to
      // the camera in the constructor. 

      // (Alternatively I think this would be equivalent to
      /// ShaderGen.apply(RootTransform);
      RootTransform->accept(ShaderGen);
    }
    
    return std::make_tuple(renderentry,locks_required,modified);
    
  }
  
  void osg_geom_renderer::BuildCoordAxes()
  {
    double cylradius=0.1;
    double cyllength=0.8; // out of 1.0 total length
    double coneheight=1.0-cyllength;
    double coneradius=0.2;
    
    
    osg::ref_ptr<osg::TessellationHints> Hints = new osg::TessellationHints();
    Hints->setDetailRatio(0.5);
    
    CoordAxes = new osg::MatrixTransform();
    osg::ref_ptr<osg::Geode> CoordAxesGeode = new osg::Geode();
    osg::ref_ptr<osg::StateSet> CoordAxesStateSet = CoordAxesGeode->getOrCreateStateSet();
    CoordAxes->addChild(CoordAxesGeode);
    
    osg::ref_ptr<osg::Cylinder> RedCyl=new osg::Cylinder(osg::Vec3(cyllength/2.0,0.0,0.0),cylradius,cyllength);
    // Cylinder by default is in +z axis; we want +x axis;
    // so we rotate around +y axis
    RedCyl->setRotation(osg::Quat(M_PI/2.0,osg::Vec3(0.0,1.0,0.0)));
    osg::ref_ptr<osg::ShapeDrawable> RedCylinder = new osg::ShapeDrawable(RedCyl,Hints);
    RedCylinder->setColor(osg::Vec4(1.0,0.0,0.0,1.0));
    CoordAxesGeode->addDrawable(RedCylinder);
    
    osg::ref_ptr<osg::Cone> RedCon=new osg::Cone(osg::Vec3(cyllength+0.25*coneheight,0.0,0.0),coneradius,coneheight);
    RedCon->setRotation(osg::Quat(M_PI/2.0,osg::Vec3(0.0,1.0,0.0)));
    osg::ref_ptr<osg::ShapeDrawable> RedCone = new osg::ShapeDrawable(RedCon,Hints);
    RedCone->setColor(osg::Vec4(1.0,0.0,0.0,1.0));
    CoordAxesGeode->addDrawable(RedCone);
    
    osg::ref_ptr<osg::Cylinder> GreenCyl = new osg::Cylinder(osg::Vec3(0.0,cyllength/2.0,0.0),cylradius,cyllength);
    // Cylinder by default is in +z axis; we want +y axis;
    // so we rotate around -x axis
    GreenCyl->setRotation(osg::Quat(M_PI/2.0,osg::Vec3(-1.0,0.0,0.0)));
    osg::ref_ptr<osg::ShapeDrawable> GreenCylinder = new osg::ShapeDrawable(GreenCyl,Hints);
    GreenCylinder->setColor(osg::Vec4(0.0,1.0,0.0,1.0));
    CoordAxesGeode->addDrawable(GreenCylinder);
    
    osg::ref_ptr<osg::Cone> GreenCon = new osg::Cone(osg::Vec3(0.0,cyllength+0.25*coneheight,0.0),coneradius,coneheight);
    GreenCon->setRotation(osg::Quat(M_PI/2.0,osg::Vec3(-1.0,0.0,0.0)));
    osg::ref_ptr<osg::ShapeDrawable> GreenCone = new osg::ShapeDrawable(GreenCon,Hints);
    GreenCone->setColor(osg::Vec4(0.0,1.0,0.0,1.0));
    CoordAxesGeode->addDrawable(GreenCone);
    
    osg::ref_ptr<osg::Cylinder> BlueCyl = new osg::Cylinder(osg::Vec3(0.0,0.0,cyllength/2.0),cylradius,cyllength);
    osg::ref_ptr<osg::ShapeDrawable> BlueCylinder = new osg::ShapeDrawable(BlueCyl,Hints);
    // Cylinder by default is in +z axis; no need to rotate!
    BlueCylinder->setColor(osg::Vec4(0.0,0.0,1.0,1.0));
    CoordAxesGeode->addDrawable(BlueCylinder);
    
    osg::ref_ptr<osg::Cone> BlueCon = new osg::Cone(osg::Vec3(0.0,0.0,cyllength+0.25*coneheight),coneradius,coneheight);
    osg::ref_ptr<osg::ShapeDrawable> BlueCone = new osg::ShapeDrawable(BlueCon,Hints);
    // Cylinder by default is in +z axis; no need to rotate!
    BlueCone->setColor(osg::Vec4(0.0,0.0,1.0,1.0));
    CoordAxesGeode->addDrawable(BlueCone);
    
    
  }

  void osg_geom_renderer::OrientCoordAxes()
  {
    osg::Vec3d translation;
    osg::Quat rotation;
    osg::Vec3d scale;
    osg::Quat scale_orientation;

    // get the view matrix from the main camera and decompose it into components
    Camera->getViewMatrix().decompose(translation,rotation,scale,scale_orientation);

    // Apply the same rotation to our CoordAxesCamera (ignoring rotation, scale, etc.

    osg::Matrixd CoordAxesView = osg::Matrixd::rotate(rotation);

    // step back a bit from the origin in CoordAxesView -- pushing
    // the Coord Axes (which are at the origin) into the
    // view of the camera
    CoordAxesView(3,2) = -6.0;
    CoordAxesCamera->setViewMatrix(CoordAxesView);
    
  }
  
  void osg_geom_renderer::BuildCoordAxes_HUD()
  {
    // build a "Heads up display" to show the coordinate axes
    // roughly following the OSG HUD example
    // https://github.com/openscenegraph/OpenSceneGraph/blob/master/examples/osghud/osghud.cpp
    // and XYZ axis drawing suggestions
    // https://community.khronos.org/t/draw-xyz-axis-in-corner/55901
    
    
    CoordAxesCamera = new osg::Camera;
    CoordAxesCamera->setProjectionMatrixAsPerspective(30.0f,((double)hudwidth)/hudheight,1.0f,10000.0f); // !!!*** Check last two parameters

    CoordAxesCamera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);

    OrientCoordAxes();
    
    // Clear depth buffer
    CoordAxesCamera->setClearMask(/*GL_COLOR_BUFFER_BIT| */GL_DEPTH_BUFFER_BIT);

    // mark as post render
    CoordAxesCamera->setRenderOrder(osg::Camera::POST_RENDER);

    // no event focus
    CoordAxesCamera->setAllowEventFocus(false);

    CoordAxesCamera->addChild(CoordAxes);
  }
  
};
