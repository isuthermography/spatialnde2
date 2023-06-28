#ifndef SNDE_REC_DISPLAY_HPP
#define SNDE_REC_DISPLAY_HPP

#include <memory>
#include <mutex>
#include <typeindex>


#include "snde/units.hpp"
#include "snde/rec_display_colormap.hpp"
//#include "snde/normal_calculation.hpp"
//#include "snde/rec_display_vertex_functions.hpp"

//#include "snde/mutablerecstore.hpp"
//#include "snde/revision_manager.hpp"

// ***!!!! This should probably have some major rework, using a snapshot of the settings
// at the start of a rendering pass to inform the rendering pass, and taking the output
// from the pass (including on-demand waveforms, etc.) to update the settings as appropriate.
// i.e. all of the get_ndarray_ref()'s should be eliminated. The locking should be eliminated
// (since once snapshotted it will only be accessed from one thread at a time), etc. 


namespace snde {

  class globalrevision; // recstore.hpp
  class display_channel; // forward declaration

  //typedef std::unordered_map<std::pair<std::string,rendermode>,std::pair<std::shared_ptr<recording_base>,std::shared_ptr<image_reference>>,chanpathmode_hash> chanpathmode_rectexref_dict;
  
  struct display_unit {
    display_unit(units unit,
		 double scale,
		 bool pixelflag);
    
    std::mutex admin; // protects scale; other members should be immutable. After the display_info and display_channel and display_axis locks in the locking order
    
    units unit;
    double scale; // Horizontal units per division (if not pixelflag) or units per pixel (if pixelflag)
    bool pixelflag;
  };
  
  struct display_axis {
    
    std::mutex admin; // protects CenterCoord; other members should be immutable. After the display_info and display_channel locks in the locking order but before the display_unit locks
    std::string axis;
    std::string abbrev;
    std::shared_ptr<display_unit> unit;
    bool has_abbrev;
    
    double CenterCoord; // horizontal coordinate (in axis units) of the center of the display
    double DefaultOffset;
    double DefaultUnitsPerDiv; // Should be 1.0, 2.0, or 5.0 times a power of 10
    
    // double MousePosn;
    
    
    display_axis(const std::string &axis,
		 const std::string &abbrev,
		 std::shared_ptr<display_unit> unit,
		 bool has_abbrev,
		 double CenterCoord,
		 double DefaultOffset,
		 double DefaultUnitsPerDiv);
  };
  

  /*
  class recdisplay_notification_receiver {
    // abstract base class
  public:
    recdisplay_notification_receiver();
    
    recdisplay_notification_receiver(const recdisplay_notification_receiver &)=delete; // no copy constructor
    recdisplay_notification_receiver & operator=(const recdisplay_notification_receiver &)=delete; // no copy assignment
    virtual ~recdisplay_notification_receiver() = default;
    
    virtual void mark_as_dirty(std::shared_ptr<display_channel> dirtychan)=0;
  };
  */
  
  class display_channel: public std::enable_shared_from_this<display_channel> {
    
    
    //std::shared_ptr<std::string> _FullName; // Atomic shared pointer pointing to full name, including slash separating tree elements
    //std::shared_ptr<mutableinfostore> chan_data;
  public:
    const std::string FullName; // immutable so you do not need to hold the admin lock to access this
    
    
    float Scale; // vertical axis scaling for 1D recs; color axis scaling for 2D recordings; units/pixel if pixelflag is set is set for the axis/units, units/div (or equivalently units/intensity) if pixelflag is not set. Also magnification for 3d channels
    float RenderScale; // scaling of the 3D rendering
    float Position; // vertical offset on display, in divisions. To get in units, multiply by GetVertUnitsPerDiv(Chan) USED ONLY IF VertZoomAroundAxis is true
    float HorizPosition; // horizonal offset onf display, in divisions. Used only on 3D projection channels
    float VertCenterCoord; // Vertical position, in vertical axis units, of center of display. USE ONLY IF VertZoomAroundAxis is false;
    bool VertZoomAroundAxis; // if false, when we zoom vertically the location of the display center remains fixed; if true when we zoom vertically the location of object's horizontal axis (y value of 0) remains fixed
    float Offset; // >= 2d only, intensity offset on display, in amplitude units
    float Alpha; // alpha transparency of channel: 1: fully visible, 0: fully transparent
    size_t ColorIdx; // index into color table for how this channel is colored
    
    
    bool Enabled; // Is this channel currently visible
    // unsigned long long currevision;
    size_t DisplayFrame; // Frame # to display
    size_t DisplaySeq; // Sequence # to display
    // NeedAxisScales
    size_t ColorMap; // colormap selection
    
    int render_mode; // see SNDE_DCRM_XXXX, below
#define SNDE_DCRM_INVALID 0
#define SNDE_DCRM_SCALAR 1
#define SNDE_DCRM_WAVEFORM 2
#define SNDE_DCRM_IMAGE 3
#define SNDE_DCRM_GEOMETRY 4
#define SNDE_DCRM_PHASEPLANE 5
    

    //std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>> adjustment_deps; // these trm_dependencies should be triggered when these parameters are changed. *** SHOULD BE REPLACED BY revman_rec_display method
    
    // NOTE: Adjustement deps needs to be cleaned periodically of lost weak pointers!
    // receivers in adjustment_deps should be called during a transaction! */
    //std::set<std::weak_ptr<recdisplay_notification_receiver>,std::owner_less<std::weak_ptr<recdisplay_notification_receiver>>> adjustment_deps;
    
    std::mutex admin; // protects all members, as the display_channel
    // may be accessed from transform threads, not just the GUI thread
    
    
    display_channel(const std::string &FullName,//std::shared_ptr<mutableinfostore> chan_data,
		    float Scale,float Position,float HorizPosition,float VertCenterCoord,bool VertZoomAroundAxis,float Offset,float Alpha,
		    size_t ColorIdx,bool Enabled, size_t DisplayFrame,size_t DisplaySeq,
		    size_t ColorMap,int render_mode);
    
    void set_enabled(bool Enabled);
    
    
    
    
    //void UpdateFullName(const std::string &new_FullName)
    //{
    //  std::shared_ptr<std::string> New_NamePtr = std::make_shared<std::string>(new_FullName);
    //  std::atomic_store(&_FullName,New_NamePtr);
    //}
    
    //void add_adjustment_dep(std::shared_ptr<recdisplay_notification_receiver> notifier);
    void mark_as_dirty();
    
  };
  

  struct display_posn {
    // represents a position on the display, such as a clicked location
    // or the current mouse position
    std::shared_ptr<display_axis> Horiz;
    double HorizPosn;  // position on Horiz axis
    
    std::shared_ptr<display_axis> Vert;
    double VertPosn;  // position on Vert axis
    
    //std::shared_ptr<display_axis> Intensity;
    //double IntensityPosn;  // recorded intensity at this position
    
  };
  
  std::string PrintWithSIPrefix(double val, const std::string &unitabbrev, int sigfigs);
  
  struct RecColor {
  double R,G,B;
    
    friend bool operator==(const RecColor &lhs, const RecColor &rhs);
    friend bool operator!=(const RecColor &lhs, const RecColor &rhs);
  };


  
  static const RecColor RecColorTable[]={
    {1.0,0.0,0.0}, /* Red */
    {0.0,0.4,1.0}, /* Blue */
    {0.0,.6,0.0}, /* Green */
    {.6,.6,0.0}, /* Yellow */
    {0.0,.6,.6}, /* Cyan */
    {1.0,0.0,1.0}, /* Magenta */
  };
  

  // special key definitions
#define SNDE_RDK_LEFT 100
#define SNDE_RDK_RIGHT 101
#define SNDE_RDK_UP 102
#define SNDE_RDK_DOWN 103
#define SNDE_RDK_PAGEUP 104
#define SNDE_RDK_PAGEDOWN 105
#define SNDE_RDK_HOME 106
#define SNDE_RDK_END 107
#define SNDE_RDK_INSERT 108
#define SNDE_RDK_DELETE 109
#define SNDE_RDK_BACKSPACE 110
#define SNDE_RDK_ENTER 111
#define SNDE_RDK_TAB 112
#define SNDE_RDK_ESC 113
  
  
  
  class display_info {
  public:
    
    mutable std::mutex admin; // locks access to below structure. Late in the locking order; after recdb and RSS/globalrev but prior to GIL and prior to the display_channel locks; 
    size_t unique_index;
    std::vector<std::shared_ptr<display_unit>>  UnitList;
    std::vector<std::shared_ptr<display_axis>>  AxisList;
    size_t NextColor;
    size_t horizontal_divisions;
    size_t vertical_divisions;
    float borderwidthpixels;
    double pixelsperdiv;
    size_t drawareawidth;
    size_t drawareaheight;
    display_posn selected_posn; 
    
    std::weak_ptr<recdatabase> recdb;
    //std::shared_ptr<globalrevision> current_globalrev;
    //uint64_t current_globalrev_index;
    
    std::unordered_map<std::string,std::shared_ptr<display_channel>> channel_info;
    //std::vector<std::string> channel_layer_order; // index is nominal order, string is full channel name
    
    const std::shared_ptr<math_function> vertnormalarray_function; // immutable
    const std::shared_ptr<math_function> colormapping_function; // immutable
    const std::shared_ptr<math_function> fusion_colormapping_function; // immutable
    const std::shared_ptr<math_function> pointcloud_colormapping_function; // immutable
    const std::shared_ptr<math_function> vertexarray_function; // immutable
    const std::shared_ptr<math_function> texvertexarray_function; // immutable
    
    display_info(std::shared_ptr<recdatabase> recdb);
    //void set_current_globalrev(std::shared_ptr<globalrevision> globalrev);
    
    void set_selected_posn(const display_posn &markerposn);
    
    display_posn get_selected_posn() const;
    
    std::shared_ptr<display_channel> lookup_channel(const std::string& recfullname);
    
    void set_pixelsperdiv(size_t drawareawidth,size_t drawareaheight);
    
    std::shared_ptr<display_channel> _add_new_channel(const std::string &fullname);  // must be called with admin lock locked. 
    
    std::pair<std::vector<std::shared_ptr<display_channel>>,std::vector<std::shared_ptr<display_channel>>> get_channels(std::shared_ptr<globalrevision> globalrev,const std::string &selected, bool check_for_mutable,bool selected_last,bool include_disabled,bool include_hidden);  
    
    std::shared_ptr<display_unit> FindUnitLocked(const std::string &name);

    std::shared_ptr<display_unit> FindUnit(const std::string &name);
    
    
    std::shared_ptr<display_axis> FindAxisLocked(const std::string &axisname,const std::string &unitname);
    std::shared_ptr<display_axis> FindAxis(const std::string &axisname,const std::string &unitname);
    
    std::shared_ptr<display_axis> GetFirstAxis(const std::string &fullname /*std::shared_ptr<mutableinfostore> rec */);
    std::shared_ptr<display_axis> GetFirstAxisLocked(const std::string &fullname /*std::shared_ptr<mutableinfostore> rec */);
    
    std::shared_ptr<display_axis> GetSecondAxis(const std::string &fullname);
    
    
    std::shared_ptr<display_axis> GetSecondAxisLocked(const std::string &fullname);
    
    
    std::tuple<std::shared_ptr<display_axis>, double, double, std::string> GetThirdAxis(const std::string &fullname);
    
    std::shared_ptr<display_axis> GetThirdAxisLocked(const std::string &fullname);
    
    std::shared_ptr<display_axis> GetFourthAxis(const std::string &fullname);
    
    std::shared_ptr<display_axis> GetFourthAxisLocked(const std::string &fullname);
    
    std::shared_ptr<display_axis> GetAmplAxis(const std::string &fullname);
    
    
    std::shared_ptr<display_axis> GetAmplAxisLocked(const std::string &fullname);
    
    void SetVertScale(std::shared_ptr<display_channel> c,double scalefactor,bool pixelflag);
    
    
    std::tuple<bool,double,bool> GetVertScale(std::shared_ptr<display_channel> c);  // returns (success,scalefactor,pixelflag)
    void SetRenderScale(std::shared_ptr<display_channel> c,double scale, bool ignored_pixelflag);

    std::tuple<bool,double> GetRenderScale(std::shared_ptr<display_channel> c);
    
    double GetVertUnitsPerDiv(std::shared_ptr<display_channel> c);

    void handle_key_down(const std::string &selected_channel,int key,bool shift,bool alt,bool ctrl);
    void handle_special_down(const std::string &selected_channel,int special,bool shift,bool alt,bool ctrl);
    
  };
  

}
#endif // SNDE_REC_DISPLAY_HPP
