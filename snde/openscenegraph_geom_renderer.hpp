
#ifndef SNDE_OPENSCENEGRAPH_GEOM_RENDERER_HPP
#define SNDE_OPENSCENEGRAPH_GEOM_RENDERER_HPP


#include "snde/openscenegraph_rendercache.hpp"
#include "snde/openscenegraph_renderer.hpp" // for osgViewerCompat34

#include <osgGA/TrackballManipulator>

namespace snde {
  
  class osg_geom_renderer: public osg_renderer { 
  public:

    // inherited members from osg_renderer
    //osg::ref_ptr<osgViewer::Viewer> Viewer;
    //osg::ref_ptr<osg::Camera> Camera;
    //osg::ref_ptr<osgGA::CameraManipulator> Manipulator;
    //osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow;
    //std::string channel_path;
    //
    //int type; // see SNDE_DRRT_XXXXX in rec_display.hpp

    osg::ref_ptr<osg::Group> group;
    osg::ref_ptr<osg::MatrixTransform> CoordAxes; // red, green, and blue coordinate axes drawn in lower left hand corner
    osg::ref_ptr<osg::Camera> CoordAxesCamera;

    size_t hudwidth;
    size_t hudheight;
    
    //    bool firstrun;
    
    osg_geom_renderer(osg::ref_ptr<osgViewer::Viewer> Viewer,osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
		      std::string channel_path,bool enable_shaders);
    osg_geom_renderer(const osg_geom_renderer &) = delete;
    osg_geom_renderer & operator=(const osg_geom_renderer &) = delete;
    virtual ~osg_geom_renderer() = default; 
    
    
    std::tuple<std::shared_ptr<osg_rendercacheentry>,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>>,bool>
    prepare_render(//std::shared_ptr<recdatabase> recdb,
		   std::shared_ptr<recording_set_state> with_display_transforms,
		   //std::shared_ptr<display_info> display,
		   std::shared_ptr<osg_rendercache> RenderCache,
		   const std::map<std::string,std::shared_ptr<display_requirement>> &display_reqs,
		   size_t width, // width of viewport in pixels
		   size_t height); // height of viewport in pixels
    
    /* NOTE: to actually render, do any geometry updates, 
       then call Viewer->frame() */
    /* NOTE: to adjust size, first send event, then 
       change viewport:

    GraphicsWindow->getEventQueue()->windowResize(x(),y(),width,height);
    GraphicsWindow->resized(x(),y(),width,height);
    Camera->setViewport(0,0,width,height);
    SetProjectionMatrix();

    */

    void BuildCoordAxes();

    void OrientCoordAxes();

    void BuildCoordAxes_HUD();

    /*
    std::tuple<double,double> GetPadding(size_t drawareawidth,size_t drawareaheight);

    std::tuple<double,double> GetScalefactors(std::string recname);

    osg::Matrixd GetChannelTransform(std::string recname,std::shared_ptr<display_channel> displaychan,size_t drawareawidth,size_t drawareaheight,size_t layer_index);
    */
    
  };


}

#endif // SNDE_OPENSCENEGRAPH_GEOM_RENDERER_HPP



