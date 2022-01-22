
#ifndef SNDE_OPENSCENEGRAPH_IMAGE_RENDERER_HPP
#define SNDE_OPENSCENEGRAPH_IMAGE_RENDERER_HPP


#include "snde/openscenegraph_rendercache.hpp"
#include "snde/openscenegraph_renderer.hpp" // for osgViewerCompat34

namespace snde {
  
  class osg_image_renderer : public osg_renderer{ 
  public:

    // From osg_renderer base class:
    //osg::ref_ptr<osgViewer::Viewer> Viewer;
    //osg::ref_ptr<osg::Camera> Camera;
    //osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow;
    //std::string channel_path;
    //int type; // see SNDE_DRRT_XXXXX in rec_display.hpp


    //osg::ref_ptr<osg::Group> group;
    
        
    osg_image_renderer(osg::ref_ptr<osgViewer::Viewer> Viewer,osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
		       std::string channel_path);
    osg_image_renderer(const osg_image_renderer &) = delete;
    osg_image_renderer & operator=(const osg_image_renderer &) = delete;
    virtual ~osg_image_renderer() = default; 
    

    std::tuple<std::shared_ptr<osg_rendercacheentry>,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>>,bool>
    prepare_render(//std::shared_ptr<recdatabase> recdb,
		   std::shared_ptr<recording_set_state> with_display_transforms,
		   //std::shared_ptr<display_info> display,
		   std::shared_ptr<osg_rendercache> RenderCache,
		   const std::map<std::string,std::shared_ptr<display_requirement>> &display_reqs,
		   size_t width, // width of viewport in pixels
		   size_t height); // height of viewport in pixels
    
    /* NOTE: to actually render, do any geometry updates, 
       then call prepare_render() then call frame() (implemented in osg_renderer superclass */

    
  };


}

#endif // SNDE_OPENSCENEGRAPH_IMAGE_RENDERER_HPP



