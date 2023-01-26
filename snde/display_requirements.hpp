#ifndef SNDE_DISPLAY_REQUIREMENTS_HPP
#define SNDE_DISPLAY_REQUIREMENTS_HPP

#include <memory>
#include <mutex>
#include <typeindex>

#include <Eigen/Dense>


#include "snde/recstore.hpp"
#include "snde/rendermode.hpp"

namespace snde {

  class instantiated_math_function; // recmath.hpp
  class recording_display_handler_base;
  class registered_recording_display_handler;
  class display_info; // rec_display.hpp
  class display_channel; // rec_display.hpp
  
  class display_spatial_position {
  public:
    size_t x; // position of lowerleft of channel renderingbox vs lowerleft of full
              // rendering area in pixels
    size_t y; 
    size_t width; // size of channel renderingbox in pixels
    size_t height;
    
    size_t drawareawidth; // size of entire drawing area in pixels
    size_t drawareaheight;
  };

  class display_spatial_transform {
  public:
    // This defines the transforms from rendered channel coordinates or rendered 3D coordinates 
    // to rendering area pixel coordinates as a 3x3 matrix in 2D inhomogeneous coordinates.
    //
    
    // Note that the pixel area goes from (0,0) to (width,height) inclusive, with the
    // first pixel center at (0.5,0.5) (see
    // https://www.realtimerendering.com/blog/the-center-of-the-pixel-is-0-50-5/ )
    //
    // This is based on a glViewport(0,0,width,height)
    // and a glOrtho(0.0,width,0.0,height,...)
    //
    // So for images or waveforms this transform represents:
    //   pixels-rel-lower-left-of-channel-renderingbox divided by channel coordinates
    //

    // rendering area is the visible domain to be rendered to
    // render box is the portion into which we are rendering this channel
    
    // NOTE: Stored in fortran order (row-major), so it may appear transposed from C++
    Eigen::Matrix<double,3,3,Eigen::RowMajor> renderarea_coords_over_renderbox_coords; // renderarea in pixels relative to its lower-left; renderbox in pixels relative to its lower left
    Eigen::Matrix<double,3,3,Eigen::RowMajor> renderarea_coords_over_channel_coords; // renderarea in pixels relative to its lowerleft, channel relative to its native data origin

    bool operator==(const display_spatial_transform &other) const
    {
      return renderarea_coords_over_renderbox_coords == other.renderarea_coords_over_renderbox_coords;
    }
    
    bool operator!=(const display_spatial_transform &other) const
    {
      return !(*this == other);
    }
    
  };

  class display_channel_rendering_bounds {
  public:
    // These are the parameters to glOrtho()
    double left; // in channel units
    double right;
    double bottom;
    double top;
  };
  

  std::tuple<std::shared_ptr<display_spatial_position>,std::shared_ptr<display_spatial_transform>,std::shared_ptr<display_channel_rendering_bounds>> spatial_transforms_for_image_channel(size_t drawareawidth,size_t drawareaheight,size_t horizontal_divisions,size_t vertical_divisions,double x_center_channel_units,double y_chanposn_divs,bool y_chan_vertzoomaroundaxis,double y_chan_vertcentercoord,double xunitscale,double yunitscale,double pixelsperdiv,bool horizontal_pixelflag, bool vertical_pixelflag,bool vert_zoom_around_axis,double dataleftedge_chanunits,double datarightedge_chanunits,double databottomedge_chanunits,double datatopedge_chanunits);

  std::tuple<std::shared_ptr<display_spatial_position>,std::shared_ptr<display_spatial_transform>,std::shared_ptr<display_channel_rendering_bounds>> spatial_transforms_for_waveform_channel(size_t drawareawidth,size_t drawareaheight,size_t horizontal_divisions,size_t vertical_divisions,double x_center_channel_units,double y_chanposn_divs,bool y_chan_vertzoomaroundaxis,double y_chan_vertcentercoord,double xunitscale,double yunitscale,double pixelsperdiv,bool horizontal_pixelflag, bool vertical_pixelflag,bool vert_zoom_around_axis);

  std::tuple<std::shared_ptr<display_spatial_position>,std::shared_ptr<display_spatial_transform>,std::shared_ptr<display_channel_rendering_bounds>> spatial_transforms_for_3d_channel(size_t drawareawidth,size_t drawareaheight,double x_chanposn_divs,double y_chanposn_divs,double mag,double pixelsperdiv);

  std::shared_ptr<std::multimap<rendergoal,std::shared_ptr<registered_recording_display_handler>>> recording_display_handler_registry();
  
  int register_recording_display_handler(rendergoal goal,std::shared_ptr<registered_recording_display_handler> handler);


  
  class display_requirement {
  public:
    // ***!!! The channelpath and mode (with extended parameters) should uniquely define
    // the rendering configuration (generated on-demand channels via the renderable function
    // and renderable channelpath). The rendermode_ext mode ends up being the key for the
    // rendercache -- which stores renderable OSG trees. So anything that might affect 
    // the OSG tree has to go into mode (generally via the extension and a renderparams subclass).
    //
    // Even if the mode is the same and the rendercache ends up reusing its same output,
    // we still need to reexecute the OSG tree (i.e. rerender) if spatial_position,
    // spatial_transform, or spatial_bounds (below) change, if the display area
    // geometry changes, or if the 3D OSG Manipulator changes position. 
    
    std::string channelpath;
    rendermode_ext mode; // see rendermode.hpp; contains parameter block
    std::shared_ptr<recording_base> original_recording;
    std::shared_ptr<recording_display_handler_base> display_handler;

    int renderer_type; // see SNDE_DRRT_xxxx
#define SNDE_DRRT_INVALID 0
    //#define SNDE_DRRT_WAVEFORM 1  // waveform actually should be handled identically to image at the moment so we just use that
    //#define SNDE_DRRT_IMAGE 2
#define SNDE_DRRT_2D 1
#define SNDE_DRRT_GEOMETRY 3
    
    // A re-render can be avoided if (a) mode matches, (b) spatial_transform is matches (both null OK), (c) osg_rendercacheentry is the same pointer, (d) for 3D renders the manipulator orientation matches, and (e) display area width and height are unchanged.
    std::shared_ptr<display_spatial_position> spatial_position; // for root requirements only, specifies the  the drawing area for this channel in integer pixels (not including surrounding highlight box) relative to the lower left corner of our rendering area. 
    std::shared_ptr<display_spatial_transform> spatial_transform; // for root requirements only, this documents the spatial transformations between render area pixel coordinates, channel coordinates, and renderbox pixel coordinates. 
    std::shared_ptr<display_channel_rendering_bounds> spatial_bounds; // for root requirements only. These are the bounds in channel units of what is being (or to be ) displayed. For 3D rendering channels, "channel units" are interpreted to be pixels at the 3D rendering resolution
    
    std::shared_ptr<std::string> renderable_channelpath;  // shared_ptr aspect could probably be removed now that we default it to the channelpath
    std::shared_ptr<instantiated_math_function> renderable_function;
    std::vector<std::shared_ptr<display_requirement>> sub_requirements;
    
    display_requirement(std::string channelpath,rendermode_ext mode,std::shared_ptr<recording_base> original_recording,std::shared_ptr<recording_display_handler_base> display_handler) :
      channelpath(channelpath),
      mode(mode),
      original_recording(original_recording),
      display_handler(display_handler),
      renderer_type(SNDE_DRRT_INVALID),
      renderable_channelpath(std::make_shared<std::string>(channelpath)) // (default to input channel; often overridden)
    {

    }

    // We could make display-requirement polymorphic by giving it a virtual destructor... Should we?
    // Probably not because any additional information should be passed in the parameters that are part
    // of the extended rendermode 
  };


  class recording_display_handler_base : public std::enable_shared_from_this<recording_display_handler_base> {
  public:
    std::shared_ptr<display_info> display;
    std::shared_ptr<display_channel> displaychan;
    std::shared_ptr<recording_set_state> base_rss;

    size_t drawareawidth;
    size_t drawareaheight;
    
    
    recording_display_handler_base(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) :
      display(display),
      displaychan(displaychan),
      base_rss(base_rss)
    {
      
    }
    
    virtual ~recording_display_handler_base()=default; // polymorphic

    virtual std::shared_ptr<display_requirement> get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent)=0;
  };
  
  class registered_recording_display_handler {
  public:
    std::function<std::shared_ptr<recording_display_handler_base>(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss)> display_handler_factory;
    // more stuff to go here as the basis for selecting a display handler when there are multiple options

    registered_recording_display_handler(std::function<std::shared_ptr<recording_display_handler_base>(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss)> display_handler_factory) :
      display_handler_factory(display_handler_factory)
    {

    }
    
  };


  class null_recording_recording_display_handler: public recording_display_handler_base {
  public:
    // From recording_display_handler_base
    //std::shared_ptr<display_info> display;
    //std::shared_ptr<display_channel> displaychan;
    //std::shared_ptr<recording_set_state> base_rss;
    
    null_recording_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss);
    
    virtual ~null_recording_recording_display_handler()=default; // polymorphic

    virtual std::shared_ptr<display_requirement> get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent);

    
  };

  
  class multi_ndarray_recording_display_handler: public recording_display_handler_base {
  public:
    // From recording_display_handler_base
    //std::shared_ptr<display_info> display;
    //std::shared_ptr<display_channel> displaychan;
    //std::shared_ptr<recording_set_state> base_rss;
    
    multi_ndarray_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss);
    
    virtual ~multi_ndarray_recording_display_handler()=default; // polymorphic

    virtual std::shared_ptr<display_requirement> get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent);

    
  };


  class fusion_ndarray_recording_display_handler: public recording_display_handler_base {
  public:
    // From recording_display_handler_base
    //std::shared_ptr<display_info> display;
    //std::shared_ptr<display_channel> displaychan;
    //std::shared_ptr<recording_set_state> base_rss;
    
    fusion_ndarray_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss);
    
    virtual ~fusion_ndarray_recording_display_handler()=default; // polymorphic

    virtual std::shared_ptr<display_requirement> get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent);

    
  };



  
  class meshed_part_recording_display_handler: public recording_display_handler_base {
  public:
    // From recording_display_handler_base
    //std::shared_ptr<display_info> display;
    //std::shared_ptr<display_channel> displaychan;
    //std::shared_ptr<recording_set_state> base_rss;
    
    meshed_part_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss);
    
    virtual ~meshed_part_recording_display_handler()=default; // polymorphic

    virtual std::shared_ptr<display_requirement> get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent);

    
  };



  class meshed_parameterization_recording_display_handler: public recording_display_handler_base {
  public:
    // From recording_display_handler_base
    //std::shared_ptr<display_info> display;
    //std::shared_ptr<display_channel> displaychan;
    //std::shared_ptr<recording_set_state> base_rss;
    
    meshed_parameterization_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss);
    
    virtual ~meshed_parameterization_recording_display_handler()=default; // polymorphic

    virtual std::shared_ptr<display_requirement> get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent);

    
  };


  class textured_part_recording_display_handler: public recording_display_handler_base {
  public:
    // From recording_display_handler_base
    //std::shared_ptr<display_info> display;
    //std::shared_ptr<display_channel> displaychan;
    //std::shared_ptr<recording_set_state> base_rss;
    
    textured_part_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss);
    
    virtual ~textured_part_recording_display_handler()=default; // polymorphic

    virtual std::shared_ptr<display_requirement> get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent);

    
  };


  class assembly_recording_display_handler: public recording_display_handler_base {
  public:
    // From recording_display_handler_base
    //std::shared_ptr<display_info> display;
    //std::shared_ptr<display_channel> displaychan;
    //std::shared_ptr<recording_set_state> base_rss;
    
    assembly_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss);
    
    virtual ~assembly_recording_display_handler()=default; // polymorphic

    virtual std::shared_ptr<display_requirement> get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent);

    
  };

  
  class tracking_pose_recording_display_handler: public recording_display_handler_base {
  public:
    // From recording_display_handler_base
    //std::shared_ptr<display_info> display;
    //std::shared_ptr<display_channel> displaychan;
    //std::shared_ptr<recording_set_state> base_rss;
    
    tracking_pose_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss);
    
    virtual ~tracking_pose_recording_display_handler()=default; // polymorphic

    virtual std::shared_ptr<display_requirement> get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent);

    
  };


  class pose_channel_recording_display_handler: public recording_display_handler_base {
  public:
    // From recording_display_handler_base
    //std::shared_ptr<display_info> display;
    //std::shared_ptr<display_channel> displaychan;
    //std::shared_ptr<recording_set_state> base_rss;
    
    pose_channel_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss);
    
    virtual ~pose_channel_recording_display_handler()=default; // polymorphic

    virtual std::shared_ptr<display_requirement> get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent);

    
  };


  

  std::shared_ptr<display_requirement> traverse_display_requirement(std::shared_ptr<display_info> display, std::shared_ptr<recording_set_state> base_rss, std::shared_ptr<display_channel> displaychan, std::string simple_goal, std::shared_ptr<renderparams_base> params_from_parent); // simple_goal such as SNDE_SRG_RENDERING
  
  // Go through the vector of channels we want to display,
  // and figure out
  // (a) all channels that will be necessary, and
  // (b) the math function (if necessary) to render to rgba, and
  // (c) the name of the renderable rgba channel
  std::map<std::string, std::shared_ptr<display_requirement>> traverse_display_requirements(std::shared_ptr<display_info> display, std::shared_ptr<recording_set_state> base_rss /* (usually a globalrev) */, const std::vector<std::shared_ptr<display_channel>>& displaychans);

};

#endif // SNDE_DISPLAY_REQUIREMENTS_HPP
