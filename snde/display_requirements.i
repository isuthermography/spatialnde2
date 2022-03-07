%shared_ptr(snde::display_requirement);
snde_rawaccessible(snde::display_requirement)
%shared_ptr(snde::recording_display_handler_base);
snde_rawaccessible(snde::recording_display_handler_base)

%shared_ptr(snde::registered_recording_display_handler);
snde_rawaccessible(snde::registered_recording_display_handler)

%{
#include "snde/display_requirements.hpp"

%}
  

namespace snde {


  class recording_display_handler_base;
  class registered_recording_display_handler;

  class display_spatial_position;
  class display_spatial_transform;
  class display_channel_rendering_bounds;
  class display_info;
  class display_channel;

  std::shared_ptr<std::multimap<rendergoal,std::shared_ptr<registered_recording_display_handler>>> recording_display_handler_registry();
  
  int register_recording_display_handler(rendergoal goal,std::shared_ptr<registered_recording_display_handler> handler);

  
  struct display_requirement {
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
#define SNDE_DRRT_2D 2
#define SNDE_DRRT_GEOMETRY 3
    
    // A re-render can be avoided if (a) mode matches, (b) spatial_transform is matches (both null OK), (c) osg_rendercacheentry is the same pointer, (d) for 3D renders the manipulator orientation matches, and (e) display area width and height are unchanged.
    std::shared_ptr<display_spatial_position> spatial_position; // for root requirements only, specifies the  the drawing area for this channel in integer pixels (not including surrounding highlight box) relative to the lower left corner of our rendering area. 
    std::shared_ptr<display_spatial_transform> spatial_transform; // for root requirements only, this documents the spatial transformations between render area pixel coordinates, channel coordinates, and renderbox pixel coordinates. 
    std::shared_ptr<display_channel_rendering_bounds> spatial_bounds; // for root requirements only. These are the bounds in channel units of what is being (or to be ) displayed. For 3D rendering channels, "channel units" are interpreted to be pixels at the 3D rendering resolution
    
    std::shared_ptr<std::string> renderable_channelpath;
    std::shared_ptr<instantiated_math_function> renderable_function;
    std::vector<std::shared_ptr<display_requirement>> sub_requirements;
    
    display_requirement(std::string channelpath,rendermode_ext mode,std::shared_ptr<recording_base> original_recording,std::shared_ptr<recording_display_handler_base> display_handler);

    // We could make display-requirement polymorphic by giving it a virtual destructor... Should we?
    // Probably not because any additional information should be passed in the parameters that are part
    // of the extended rendermode 
  };


  class recording_display_handler_base /*: public std::enable_shared_from_this<recording_display_handler_base> */ {
  public:
    std::shared_ptr<display_info> display;
    std::shared_ptr<display_channel> displaychan;
    std::shared_ptr<recording_set_state> base_rss;

    size_t drawareawidth;
    size_t drawareaheight;
    
    
    recording_display_handler_base(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss);
    virtual ~recording_display_handler_base()=default; // polymorphic

    virtual std::shared_ptr<display_requirement> get_display_requirement(std::string simple_goal,std::shared_ptr<renderparams_base> params_from_parent)=0;
  };
  
  class registered_recording_display_handler {
  public:
    std::function<std::shared_ptr<recording_display_handler_base>(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss)> display_handler_factory;
    // more stuff to go here as the basis for selecting a display handler when there are multiple options

    registered_recording_display_handler(std::function<std::shared_ptr<recording_display_handler_base>(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss)> display_handler_factory);
    
  };
  
  

};
