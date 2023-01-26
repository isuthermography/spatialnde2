#include <osgUtil/ShaderGen>

#include "snde/openscenegraph_2d_renderer.hpp"
#include "snde/rec_display.hpp"
#include "snde/display_requirements.hpp"
 

namespace snde {




  osg_2d_renderer::osg_2d_renderer(osg::ref_ptr<osgViewer::Viewer> Viewer, // use an osgViewerCompat34()
					 osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
					 std::string channel_path,bool enable_shaders) :
    osg_renderer(Viewer,GraphicsWindow,nullptr,channel_path,SNDE_DRRT_2D,enable_shaders)
  {
    
    EventQueue=GraphicsWindow->getEventQueue();
    Camera->setGraphicsContext(GraphicsWindow);
    
    // set background color to blueish
    Camera->setClearColor(osg::Vec4(.1,.1,.3,0.0f));    
    Camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Camera->setCullingMode(osg::CullSettings::NO_CULLING); // otherwise triangles we use for rendering can get culled as we zoom in far enough that their vertices go off-screen?
    
    Viewer->setThreadingModel(osgViewer::Viewer::SingleThreaded); // OSG single threaded mode is REQUIRED (!!!)
    
    //Viewer->setRunFrameScheme(osg::ON_DEMAND); // ***!!! With this OSG looks at whether it thinks a new render is needed based on scene graph changes and only renders if necessary.
    
    Viewer->setCameraManipulator(nullptr);
    Camera->setViewMatrix(osg::Matrixd::identity());

    Viewer->realize();

    if (enable_shaders) {
      // Start with OSG 3.6 built-in shaders (ShaderProgram created in parent class constructor)
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


  // actually rendering is done by osg_2d_renderer::frame() which just calls Viewer->frame()
  
  std::tuple<std::shared_ptr<osg_rendercacheentry>,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>>,bool>
  osg_2d_renderer::prepare_render(//std::shared_ptr<recdatabase> recdb,
				     std::shared_ptr<recording_set_state> with_display_transforms,
				     //std::shared_ptr<display_info> display,
				     std::shared_ptr<osg_rendercache> RenderCache,
				     const std::map<std::string,std::shared_ptr<display_requirement>> &display_reqs,
				     size_t width, // width of viewport in pixels
				     size_t height) // height of viewport in pixels
  // returns cache entry used, and bool that is true if it is new or modified
  {
    // look up render cache.
    std::map<std::string,std::shared_ptr<display_requirement>>::const_iterator got_req;
    std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> locks_required;
    
    got_req=display_reqs.find(channel_path);
    if (got_req==display_reqs.end()) {
      snde_warning("openscenegraph_2d_renderer: Was not possible to transform channel \"%s\" into something renderable",channel_path.c_str());
      return std::make_tuple(nullptr,locks_required,true);
    }
    
    std::shared_ptr<display_requirement> display_req =got_req->second;
    osg_renderparams params{
      //recdb,
      RenderCache,
      with_display_transforms,
      //display,
      
      display_req->spatial_bounds->left,
      display_req->spatial_bounds->right,
      display_req->spatial_bounds->bottom,
      display_req->spatial_bounds->top,
      width,
      height,
      
    };

    snde_debug(SNDE_DC_RENDERING,"image render width bounds: left: %f right: %f bottom: %f top: %f",
		 display_req->spatial_bounds->left,
		 display_req->spatial_bounds->right,
		 display_req->spatial_bounds->bottom,
		 display_req->spatial_bounds->top);


    std::shared_ptr<osg_rendercacheentry> imageentry;
    bool modified=false;
    
    if (display_req->spatial_bounds->bottom >= display_req->spatial_bounds->top ||
	display_req->spatial_bounds->left >= display_req->spatial_bounds->right) {
      // negative or zero display area

      if (RootTransform->getNumChildren()) {

	RootTransform->removeChildren(0,RootTransform->getNumChildren());
      }

      modified = true; 
    } else { // Positive display area 
      std::tie(imageentry,modified) = RenderCache->GetEntry(params,display_req,&locks_required);
    
      /// NOTE: to adjust size, first send event, then 
      //   change viewport:
      
      snde_debug(SNDE_DC_RENDERING,"width=%d; height=%d; spatial_position->width=%d; spatial_position->height=%d",width,height,display_req->spatial_position->width,display_req->spatial_position->height);
      
      if (display_req->spatial_position->width != Camera->getViewport()->width() || display_req->spatial_position->height != Camera->getViewport()->height()) {
	GraphicsWindow->getEventQueue()->windowResize(0,0,display_req->spatial_position->width,display_req->spatial_position->height);
	GraphicsWindow->resized(0,0,display_req->spatial_position->width,display_req->spatial_position->height);
	Camera->setViewport(0,0,display_req->spatial_position->width,display_req->spatial_position->height);
	modified = true;
	snde_debug(SNDE_DC_RENDERING,"image position: width: %d height: %d",
		     display_req->spatial_position->width,
		     display_req->spatial_position->height);

      }
      
      Camera->setProjectionMatrixAsOrtho(display_req->spatial_bounds->left,display_req->spatial_bounds->right,display_req->spatial_bounds->bottom,display_req->spatial_bounds->top,-10.0,1000.0);

      // rcocc = renderbox_coords_over_channel_coords
      //Eigen::Matrix<double,3,3,Eigen::RowMajor> rcocc = display_req->spatial_transform->renderarea_coords_over_channel_coords*display_req->spatial_transform->renderarea_coords_over_renderbox_coords.inverse();

      // need to convert from Eigen 3x3 matrix representing 2D projective coordinates to an OpenSceneGraph Matrixd
      // representing 3D projective coordinates.
      // Remember that OSG takes entries fortran-order (row major) 
      /*RootTransform->setMatrix(osg::Matrixd(rcocc.coeff(0,0), rcocc.coeff(1,0), 0.0, 0.0, // first column
					    rcocc.coeff(0,1), rcocc.coeff(1,1), 0.0, 0.0, // second column 
					    0.0, 0.0, 1.0, 0.0, // third column
					    rcocc.coeff(0,2), rcocc.coeff(1,2), 0.0, 1.0)); // offsets
					    snde_debug(SNDE_DC_RENDERING,"Transform: %8.4f  %8.4f  %8.4f\n           %8.4f  %8.4f %8.4f\n	       %8.4f  %8.4f %8.4f",rcocc.coeff(0,0),rcocc.coeff(0,1),rcocc.coeff(0,2),rcocc.coeff(1,0),rcocc.coeff(1,1),rcocc.coeff(1,2),rcocc.coeff(2,0),rcocc.coeff(2,1),rcocc.coeff(2,2));  */ 
      if (imageentry) {
	
	std::shared_ptr<osg_rendercachegroupentry> imagegroup = std::dynamic_pointer_cast<osg_rendercachegroupentry>(imageentry); 
	
	if (imagegroup) {
	  //if (Viewer->getSceneData() != imagegroup->osg_group){
	  //  //group=imagegroup;
	  //  Viewer->setSceneData(imagegroup->osg_group);
	  //}
	  if (RootTransform->getNumChildren() && RootTransform->getChild(0) != imagegroup->osg_group) {
	    RootTransform->removeChildren(0,1);
	  }
	  if (!RootTransform->getNumChildren()) {
	    RootTransform->addChild(imagegroup->osg_group);
	  }
	  
	  if (!Viewer->getSceneData()) {
	    Viewer->setSceneData(RootTransform);
	  }
	  
	} else {
	  snde_warning("openscenegraph_2d_renderer: cache entry not convertable to an osg_group rendering channel \"%s\"",channel_path.c_str());
	}
      } else {
	snde_warning("openscenegraph_2d_renderer: cache entry not available rendering channel \"%s\"",channel_path.c_str());
	
      }
	
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

    
    return std::make_tuple(imageentry,locks_required,modified);
    
  }
  

  
};
