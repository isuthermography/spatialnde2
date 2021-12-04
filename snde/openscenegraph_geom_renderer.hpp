
#ifndef SNDE_OPENSCENEGRAPH_GEOM_RENDERER_HPP
#define SNDE_OPENSCENEGRAPH_GEOM_RENDERER_HPP


#include "snde/openscenegraph_rendercache.hpp"
#include "snde/openscenegraph_renderer.hpp" // for osgViewerCompat34

#include <osgGA/TrackballManipulator>

namespace snde {
  
  class osg_3d_renderer { 
  public:
    std::string channel_path;
    osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow;
    //osg::ref_ptr<osgViewer::Viewer> Viewer;
    osg::ref_ptr<osgViewer::Viewer> Viewer;
    osg::ref_ptr<osgGA::TrackballManipulator> Manipulator;
    osg::ref_ptr<osg::Camera> Camera;


    //osg::ref_ptr<osg::MatrixTransform> Transform;
    osg::ref_ptr<osg::Group> group;
    
    
    std::shared_ptr<osg_rendercache> RenderCache;
    
    osg_3d_renderer(osg::ref_ptr<osgViewer::Viewer> Viewer,osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
		    std::shared_ptr<osg_rendercache> RenderCache,
		    std::string channel_path);
    
    
    void perform_render(std::shared_ptr<recdatabase> recdb,
			//std::shared_ptr<recstore_display_transforms> display_transforms,
			std::shared_ptr<recording_set_state> with_display_transforms,
			//std::shared_ptr<display_channel> channel_to_display,
			//std::string channel_path,
			std::shared_ptr<display_info> display,
			const std::map<std::string,std::shared_ptr<display_requirement>> &display_reqs,
			double left, // left of viewport in channel horizontal units
			double right, // right of viewport in channel horizontal units
			double bottom, // bottom of viewport in channel vertical units
			double top, // top of viewport in channel vertical units
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



    std::tuple<double,double> GetPadding(size_t drawareawidth,size_t drawareaheight);

    std::tuple<double,double> GetScalefactors(std::string recname);

    osg::Matrixd GetChannelTransform(std::string recname,std::shared_ptr<display_channel> displaychan,size_t drawareawidth,size_t drawareaheight,size_t layer_index);
    
    
  };


}

#endif // SNDE_OPENSCENEGRAPH_GEOM_RENDERER_HPP



