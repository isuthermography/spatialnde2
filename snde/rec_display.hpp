#ifndef SNDE_REC_DISPLAY_HPP
#define SNDE_REC_DISPLAY_HPP

#include <memory>
#include <mutex>
#include <typeindex>

#include "snde/units.hpp"
#include "snde/rec_display_colormap.hpp"
#include "snde/normal_calculation.hpp"
#include "snde/rec_display_vertex_functions.hpp"
#include "snde/recstore.hpp"
#include "snde/rendermode.hpp"

//#include "snde/mutablerecstore.hpp"
//#include "snde/revision_manager.hpp"

namespace snde {

  struct display_channel; // forward declaration
  class instantiated_math_function; // recmath.hpp

  class recording_display_handler_base;

  //typedef std::unordered_map<std::pair<std::string,rendermode>,std::pair<std::shared_ptr<recording_base>,std::shared_ptr<image_reference>>,chanpathmode_hash> chanpathmode_rectexref_dict;
  
struct display_unit {
  display_unit(units unit,
	       double scale,
	       bool pixelflag) :
    unit(unit),
    scale(scale),
    pixelflag(pixelflag)
  {

  }
  
  std::mutex admin; // protects scale; other members should be immutable
  
  units unit;
  double scale; // Units per division (if not pixelflag) or units per pixel (if pixelflag)
  bool pixelflag;
};

struct display_axis {

  std::mutex admin; // protects CenterCoord; other members should be immutable
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
	       double DefaultUnitsPerDiv) :
    axis(axis),
    abbrev(abbrev),
    unit(unit),
    has_abbrev(has_abbrev),
    CenterCoord(CenterCoord),
    DefaultOffset(DefaultOffset),
    DefaultUnitsPerDiv(DefaultUnitsPerDiv)
  {

  }

};


class recdisplay_notification_receiver {
  // abstract base class
public:
  recdisplay_notification_receiver()
  {
    
  }
  
  recdisplay_notification_receiver(const recdisplay_notification_receiver &)=delete; // no copy constructor
  recdisplay_notification_receiver & operator=(const recdisplay_notification_receiver &)=delete; // no copy assignment
  
  virtual void mark_as_dirty(std::shared_ptr<display_channel> dirtychan) {} ;
  virtual ~recdisplay_notification_receiver() {} ;
};

struct display_channel: public std::enable_shared_from_this<display_channel> {

  
  //std::shared_ptr<std::string> _FullName; // Atomic shared pointer pointing to full name, including slash separating tree elements
  //std::shared_ptr<mutableinfostore> chan_data;
  const std::string FullName; // immutable so you do not need to hold the admin lock to access this
  
  
  float Scale; // vertical axis scaling for 1D recs; color axis scaling for 2D recordings; units/pixel if pixelflag is set is set for the axis/units, units/div (or equivalently units/intensity) if pixelflag is not set
  float Position; // vertical offset on display, in divisions. To get in units, multiply by GetVertUnitsPerDiv(Chan) USED ONLY IF VertZoomAroundAxis is true
  float VertCenterCoord; // Vertical position, in vertical axis units, of center of display. USE ONLY IF VertZoomAroundAxis is false;
  bool VertZoomAroundAxis;
  float Offset; // >= 2d only, intensity offset on display, in amplitude units
  float Alpha; // alpha transparency of channel: 1: fully visible, 0: fully transparent
  size_t ColorIdx; // index into color table for how this channel is colored
  
  bool Enabled; // Is this channel currently visible
  // unsigned long long currevision;
  size_t DisplayFrame; // Frame # to display
  size_t DisplaySeq; // Sequence # to display
  // NeedAxisScales
  size_t ColorMap; // colormap selection

  //std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>> adjustment_deps; // these trm_dependencies should be triggered when these parameters are changed. *** SHOULD BE REPLACED BY revman_rec_display method

  // NOTE: Adjustement deps needs to be cleaned periodically of lost weak pointers!
  // receivers in adjustment_deps should be called during a transaction! */
  std::set<std::weak_ptr<recdisplay_notification_receiver>,std::owner_less<std::weak_ptr<recdisplay_notification_receiver>>> adjustment_deps;

  std::mutex admin; // protects all members, as the display_channel
  // may be accessed from transform threads, not just the GUI thread

  
  display_channel(const std::string &FullName,//std::shared_ptr<mutableinfostore> chan_data,
		  float Scale,float Position,float VertCenterCoord,bool VertZoomAroundAxis,float Offset,float Alpha,
		  size_t ColorIdx,bool Enabled, size_t DisplayFrame,size_t DisplaySeq,
		  size_t ColorMap) :
    FullName(FullName),
    //chan_data(chan_data),
    Scale(Scale),
    Position(Position),
    VertCenterCoord(VertCenterCoord),
    VertZoomAroundAxis(VertZoomAroundAxis),
    Offset(Offset),
    Alpha(Alpha),
    ColorIdx(ColorIdx),
    Enabled(Enabled),
    DisplayFrame(DisplayFrame),
    DisplaySeq(DisplaySeq),
    ColorMap(ColorMap)

  {

  }


  void set_enabled(bool Enabled)
  {
    std::lock_guard<std::mutex> dc_lock(admin);
    this->Enabled=Enabled;
  }

  //void UpdateFullName(const std::string &new_FullName)
  //{
  //  std::shared_ptr<std::string> New_NamePtr = std::make_shared<std::string>(new_FullName);
  //  std::atomic_store(&_FullName,New_NamePtr);
  //}
  
  void add_adjustment_dep(std::shared_ptr<recdisplay_notification_receiver> notifier)
  {
    std::lock_guard<std::mutex> dc_lock(admin);
    adjustment_deps.emplace(notifier);

    // filter out any dead pointers
    std::set<std::weak_ptr<recdisplay_notification_receiver>,std::owner_less<std::weak_ptr<recdisplay_notification_receiver>>>::iterator next;
    for (auto it=adjustment_deps.begin(); it != adjustment_deps.end(); it=next) {
      next=it;
      next++;
      
      if (it->expired()) {
	adjustment_deps.erase(it);
      }
      
    }
    
  }

  void mark_as_dirty()
  {
    std::vector<std::shared_ptr<recdisplay_notification_receiver>> tonotify;
    {
      std::lock_guard<std::mutex> dc_lock(admin);
      std::set<std::weak_ptr<recdisplay_notification_receiver>,std::owner_less<std::weak_ptr<recdisplay_notification_receiver>>>::iterator next;
      
      for (auto it=adjustment_deps.begin(); it != adjustment_deps.end(); it=next) {
	next=it;
	next++;
      
	if (std::shared_ptr<recdisplay_notification_receiver> rcvr = it->lock()) {
	  tonotify.push_back(rcvr);
	} else {
	  adjustment_deps.erase(it);
	  
	}
	
      }
    }
    for (auto & notify: tonotify) {
      notify->mark_as_dirty(shared_from_this());
    }
  }

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

struct RecColor {
  double R,G,B;

  friend bool operator==(const RecColor &lhs, const RecColor &rhs)
  {
    return (lhs.R == rhs.R && lhs.G==rhs.G && lhs.B==rhs.B);
  }
  friend bool operator!=(const RecColor &lhs, const RecColor &rhs)
  {
    return !(lhs==rhs);
  }
};

  static const RecColor RecColorTable[]={
					 {1.0,0.0,0.0}, /* Red */
					 {0.0,0.4,1.0}, /* Blue */
					 {0.0,1.0,0.0}, /* Green */
					 {1.0,1.0,0.0}, /* Yellow */
					 {0.0,1.0,1.0}, /* Cyan */
					 {1.0,0.0,1.0}, /* Magenta */
  };
  

static std::string PrintWithSIPrefix(double val, const std::string &unitabbrev, int sigfigs)
  {
    std::string buff="";
    
    if (val < 0) {
      val=-val;
      buff += "-";
    }

    std::stringstream numberbuf;
    
    if (val >= 1.0 && val < 1000.0) {
      numberbuf << std::defaultfloat << std::setprecision(sigfigs) << val;
      buff += numberbuf.str() + unitabbrev;
    } else if (val >= 1000.0 && val < 1.e6) {
      numberbuf << std::defaultfloat << std::setprecision(sigfigs) << val*1e-3;
      buff += numberbuf.str() + "k" + unitabbrev;
    } else if (val >= 1.e6 && val < 1.e9) {
      numberbuf << std::defaultfloat << std::setprecision(sigfigs) << val*1e-6;
      buff += numberbuf.str() + "M" + unitabbrev;
    } else if (val >= 1.e9) {
      numberbuf << std::defaultfloat << std::setprecision(sigfigs) << val*1e-9;
      buff += numberbuf.str() + "G" + unitabbrev;
    } else if (val >=1e-3 && val < 1.0) {
      numberbuf << std::defaultfloat << std::setprecision(sigfigs) << val*1e3;
      buff += numberbuf.str() + "m" + unitabbrev;
    } else if (val >= 1e-6 && val < 1.e-3) {
      numberbuf << std::defaultfloat << std::setprecision(sigfigs) << val*1e6;
      buff += numberbuf.str() + "u" + unitabbrev;
    } else if (val >= 1e-9 && val < 1.e-6) {
      numberbuf << std::defaultfloat << std::setprecision(sigfigs) << val*1e9;
      buff += numberbuf.str() + "n" + unitabbrev;
    } else if (val >= 1e-12 && val < 1.e-9) {
      numberbuf << std::defaultfloat << std::setprecision(sigfigs) << val*1e12;
      buff += numberbuf.str() + "p" + unitabbrev;
    } else {
      numberbuf << std::defaultfloat << std::setprecision(sigfigs) << val;
      buff += numberbuf.str() + unitabbrev;
    }

    return buff;
  }
  
  
  
class display_info {
public:
  
  std::mutex admin; // locks access to below structure. Late in the locking order but prior to GIL.
  size_t unique_index;
  std::vector<std::shared_ptr<display_unit>>  UnitList;
  std::vector<std::shared_ptr<display_axis>>  AxisList;
  size_t NextColor;
  size_t horizontal_divisions;
  size_t vertical_divisions;
  float borderwidthpixels;
  double pixelsperdiv;
  display_posn selected_posn; 
  
  std::weak_ptr<recdatabase> recdb;
  std::shared_ptr<globalrevision> current_globalrev;
 
  std::unordered_map<std::string,std::shared_ptr<display_channel>> channel_info;
  std::vector<std::string> channel_layer_order; // index is nominal order, string is full channel name

  const std::shared_ptr<math_function> vertnormals_function; // immutable
  const std::shared_ptr<math_function> colormapping_function; // immutable
  const std::shared_ptr<math_function> vertexarray_function; // immutable
  const std::shared_ptr<math_function> texvertexarray_function; // immutable
  
  display_info(std::shared_ptr<recdatabase> recdb) :
    recdb(recdb),
    selected_posn(display_posn{
      nullptr,
      0.0,
      nullptr,
      0.0,
    }),
    vertnormals_function(define_vertnormals_recording_function()),
    colormapping_function(define_colormap_recording_function()),
    vertexarray_function(define_vertexarray_recording_function()),
    texvertexarray_function(define_texvertexarray_recording_function())

  {
    static std::mutex di_index_mutex; // created atomically on first access per c++ spec
    static size_t di_index; // only one copy across entire application
    {
      std::lock_guard<std::mutex> di_index_lock(di_index_mutex);
      unique_index = di_index++;
    }
    
    NextColor=0;

    // numbers of divisions should be even!
    vertical_divisions=8;
    horizontal_divisions=10;
    borderwidthpixels=2.0; // width of the border line, in pixels
    
    // Insert some basic common units
    
    UnitList.push_back(std::make_shared<display_unit>(units::parseunits("meters"),0.1,false));
    UnitList.push_back(std::make_shared<display_unit>(units::parseunits("seconds"),0.5,false));
    UnitList.push_back(std::make_shared<display_unit>(units::parseunits("Hertz"),10e3,false));
    UnitList.push_back(std::make_shared<display_unit>(units::parseunits("pixels"),1.0,true));
    UnitList.push_back(std::make_shared<display_unit>(units::parseunits("unitless"),1.0,false));
    UnitList.push_back(std::make_shared<display_unit>(units::parseunits("meters/second"),1.0,false));
    UnitList.push_back(std::make_shared<display_unit>(units::parseunits("Kelvin"),50,false));
    UnitList.push_back(std::make_shared<display_unit>(units::parseunits("Pascals"),50e3,false));
    UnitList.push_back(std::make_shared<display_unit>(units::parseunits("Volts"),1.0,false));
    UnitList.push_back(std::make_shared<display_unit>(units::parseunits("Amps"),1.0,false));
    UnitList.push_back(std::make_shared<display_unit>(units::parseunits("Newtons"),5.0,false));
    UnitList.push_back(std::make_shared<display_unit>(units::parseunits("Newton-meters"),5.0,false));
    UnitList.push_back(std::make_shared<display_unit>(units::parseunits("Degrees"),20.0,false));
    UnitList.push_back(std::make_shared<display_unit>(units::parseunits("Arbitrary"),10.0,false));

    // Define some well-known axes
    AxisList.push_back(std::make_shared<display_axis>("time","t",FindUnit("seconds"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("frequency","f",FindUnit("Hertz"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("X Position","x",FindUnit("meters"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("Y Position","y",FindUnit("meters"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("X Position","x",FindUnit("pixels"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("Y Position","y",FindUnit("pixels"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("U Position","u",FindUnit("meters"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("V Position","v",FindUnit("meters"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("U Position","u",FindUnit("pixels"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("V Position","v",FindUnit("pixels"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("Index","i",FindUnit("unitless"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("Velocity","v",FindUnit("meters/second"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("Temperature","T",FindUnit("Kelvin"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("Pressure","p",FindUnit("Pascals"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("Voltage","v",FindUnit("Volts"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("Current","i",FindUnit("Amps"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("Force","F",FindUnit("Newtons"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("Moment","M",FindUnit("Newton-meters"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("Angle","ang",FindUnit("degrees"),false,0.0,0.0,1.0));
    AxisList.push_back(std::make_shared<display_axis>("Intensity","I",FindUnit("arbitrary"),false,0.0,0.0,1.0));

    pixelsperdiv=1.0; // will need to be updated by set_pixelsperdiv
  }

  void set_current_globalrev(std::shared_ptr<globalrevision> globalrev)
  {
    std::lock_guard<std::mutex> adminlock(admin);
    this->current_globalrev=globalrev;
  }

  void set_selected_posn(const display_posn &markerposn)
  {
    std::lock_guard<std::mutex> adminlock(admin);

    selected_posn = markerposn;
  }

  display_posn get_selected_posn()
  {
    std::lock_guard<std::mutex> adminlock(admin);

    return selected_posn;
  }
  
  std::shared_ptr<display_channel> lookup_channel(const std::string &recfullname)
  {
    std::unique_lock<std::mutex> adminlock(admin);
    auto iter = channel_info.find(recfullname);

    if (iter==channel_info.end()) {
      // not found in list
      admin.unlock();
      // Does the rec exist in the recdb?

      std::shared_ptr<recording_base> rec;
      rec = current_globalrev->get_recording(recfullname);
      return  _add_new_channel(recfullname,rec);
    }
    return iter->second;
  }
  void set_pixelsperdiv(size_t drawareawidth,size_t drawareaheight)
  {
    std::lock_guard<std::mutex> adminlock(admin);
    double pixelsperdiv_width = drawareawidth*1.0/horizontal_divisions;
    double pixelsperdiv_height = drawareaheight*1.0/vertical_divisions;

    pixelsperdiv = std::min(pixelsperdiv_width,pixelsperdiv_height);
  }


  std::shared_ptr<display_channel> _add_new_channel(const std::string &fullname,std::shared_ptr<recording_base> rec)
  // must be called with admin lock locked. 
  {
    
    
    size_t _NextColor,NewNextColor;
    {
      //std::lock_guard<std::mutex> adminlock(admin);
      _NextColor=NextColor;
      NewNextColor = (NextColor + 1) % (sizeof(RecColorTable)/sizeof(RecColorTable[0]));
      NextColor=NewNextColor;
    }
    
    // create a new display_channel
    std::shared_ptr<display_channel> new_display_channel=std::make_shared<display_channel>(fullname,/*info, */  1.0,0.0,0.0,false,0.0,1.0,_NextColor,true,0,0,0);
    
    
    //std::lock_guard<std::mutex> adminlock(admin);
    channel_info.emplace(fullname,new_display_channel);
    channel_layer_order.push_back(fullname);
    return new_display_channel;
  }
    
  std::vector<std::shared_ptr<display_channel>> update(std::shared_ptr<globalrevision> globalrev,const std::string &selected, bool selected_only,bool include_disabled,bool include_hidden)

  // for now globalrev is assumed to be fully ready (!)
    
  // if include_disabled is set, disabled channels will be included
  // if selected is not the empty string, it will be moved to the front of the list
  // to be rendered on top 
    
  // STILL NEED TO IMPLEMENT THE FOLLOWING: 
  // if include_hidden is set, hidden channels will be included
  {

    std::vector<std::shared_ptr<display_channel>> retval;
    std::shared_ptr<display_channel> selected_chan=nullptr;

    std::unordered_set<std::string> channels_considered;
    
    set_current_globalrev(globalrev);


    std::lock_guard<std::mutex> adminlock(admin);

    
    // go through reclist backwards, assembling reclist
    // so last entries in reclist will be first 
    // (except for selected_chan which will be very first, if given).

    std::vector<std::string>::reverse_iterator cl_next_iter;
    
    for (auto cl_name_iter=channel_layer_order.rbegin();cl_name_iter != channel_layer_order.rend();cl_name_iter=cl_next_iter) {
      cl_next_iter = cl_name_iter+1;
      const std::string &cl_name = *cl_name_iter;
      
      auto ci_iter = channel_info.find(cl_name);
      auto reciter = globalrev->recstatus.channel_map.find(cl_name);

      if (reciter==globalrev->recstatus.channel_map.end()) {
	// channel is gone; remove from channel_info
	channel_info.erase(cl_name);

	//auto clo_iter = std::find(channel_layer_order.begin(),channel_layer_order.end(),cl_name);
	//assert(clo_iter != channel_layer_order.end()); // if this trips, then somehow channel_info and channel_layer_order weren't kept parallel

	channel_layer_order.erase(cl_next_iter.base()); // cl_next_iter points to the previous element of the sequence, but the .base() returns an iterator pointing at the subsequent element, so we erase the element we wanted to. Also note that this erasure invalidates all subsequent iterators (but the only one we are going to use is cl_next_iter, which isn't subsequent. 
      } else {

	assert(ci_iter != channel_info.end()); // if this trips, then somehow channel_info and channel_layer_order weren't kept parallel

	channels_considered.emplace(cl_name);
	
	std::shared_ptr<recording_base> rec=reciter->second.rec();
	const std::string &fullname=reciter->first;
	
	if ((include_disabled || ci_iter->second->Enabled) || (selected_only && fullname==selected)) {
	  if (selected.size() > 0 && fullname == selected) {
	    retval.insert(retval.begin(),ci_iter->second);
	    //assert(!selected_chan);
	    //selected_chan = ci_iter->second;
	  } else if (!selected_only) {
	    retval.push_back(ci_iter->second);
	    //retval.insert(retval.begin(),ci_iter->second);
	  }
	}
	
	
      }
    }


    for (auto reciter = globalrev->recstatus.channel_map.begin(); reciter != globalrev->recstatus.channel_map.end(); reciter++) {
      
      const std::string &fullname=reciter->first;
      channel_state &chanstate=reciter->second;
      if (channels_considered.find(fullname) != channels_considered.end()) {
	// not already in our list
	std::shared_ptr<display_channel> new_display_channel=_add_new_channel(fullname,chanstate.rec());
	
	if ((include_disabled || new_display_channel->Enabled) || (selected_only && fullname==selected)) {
	  if (selected.size() > 0 && fullname == selected) {
	    retval.insert(retval.begin(),new_display_channel);
	    //assert(!selected_chan);
	    //selected_chan = new_display_channel;
	  } else if (!selected_only) {
	    retval.push_back(new_display_channel);
	  }
	  
	}
	
	
      }
    }
      
    
    return retval;
  }
  
  
  
  std::shared_ptr<display_unit> FindUnit(const std::string &name)
  {
    units u=units::parseunits(name);

    std::lock_guard<std::mutex> adminlock(admin);	  
    for (auto & uptr: UnitList) {
      if (units::compareunits(uptr->unit,u)==1.0) {
	return uptr; 
      }
    }
    
    // if we are still executing, we need to create a unit
    std::shared_ptr<display_unit> uptr=std::make_shared<display_unit>(u,1.0,false);

    UnitList.push_back(uptr);

    return uptr;
  }


  std::shared_ptr<display_axis> FindAxis(const std::string &axisname,const std::string &unitname)
  {
    units u=units::parseunits(unitname);

    std::shared_ptr<display_unit> unit=FindUnit(unitname);

    std::lock_guard<std::mutex> adminlock(admin);

    for (auto & ax: AxisList) {
      if (axisname==ax->axis && units::compareunits(u,ax->unit->unit)==1.0) {
	return ax;
      }
    }

    // need to create axis
    std::shared_ptr<display_axis> ax_ptr=std::make_shared<display_axis>(axisname,"",unit,false,0.0,0.0,1.0);

    AxisList.push_back(ax_ptr);

    return ax_ptr;
  }

  std::shared_ptr<display_axis> GetFirstAxis(const std::string &fullname /*std::shared_ptr<mutableinfostore> rec */)
  {
    std::shared_ptr<ndarray_recording_ref> rec;
    try {
      rec = current_globalrev->get_recording_ref(fullname);
    } catch (snde_error &) {
      // no such reference
      return FindAxis("Time","seconds");
    }
    
    std::string AxisName = rec->rec->metadata->GetMetaDatumStr("Coord1","Time");
    std::string UnitName = rec->rec->metadata->GetMetaDatumStr("Units1","seconds");

    return FindAxis(AxisName,UnitName);
  }

  std::shared_ptr<display_axis> GetSecondAxis(const std::string &fullname)
  {
    std::shared_ptr<ndarray_recording_ref> rec;
    try {
      rec = current_globalrev->get_recording_ref(fullname);
    } catch (snde_error &) {
      // no such reference
      return FindAxis("Time","seconds");
    }


    std::string AxisName = rec->rec->metadata->GetMetaDatumStr("Coord2","Time");
    std::string UnitName = rec->rec->metadata->GetMetaDatumStr("Units2","seconds");

    return FindAxis(AxisName,UnitName);
  }

  std::shared_ptr<display_axis> GetThirdAxis(const std::string &fullname)
  {
    std::shared_ptr<ndarray_recording_ref> rec;
    try {
      rec = current_globalrev->get_recording_ref(fullname);
    } catch (snde_error &) {
      // no such reference
      return FindAxis("Time","seconds");
    }

    std::string AxisName = rec->rec->metadata->GetMetaDatumStr("Coord3","Time");
    std::string UnitName = rec->rec->metadata->GetMetaDatumStr("Units3","seconds");

    return FindAxis(AxisName,UnitName);
  }

  std::shared_ptr<display_axis> GetFourthAxis(const std::string &fullname)
  {
    std::shared_ptr<ndarray_recording_ref> rec;
    try {
      rec = current_globalrev->get_recording_ref(fullname);
    } catch (snde_error &) {
      // no such reference
      return FindAxis("Time","seconds");
    }

    std::string AxisName = rec->rec->metadata->GetMetaDatumStr("Coord4","Time");
    std::string UnitName = rec->rec->metadata->GetMetaDatumStr("Units4","seconds");

    return FindAxis(AxisName,UnitName);
  }

  std::shared_ptr<display_axis> GetAmplAxis(const std::string &fullname)
  {
    std::shared_ptr<ndarray_recording_ref> rec;
    try {
      rec = current_globalrev->get_recording_ref(fullname);
    } catch (snde_error &) {
      // no such reference
      return FindAxis("Time","seconds");
    }

    std::string AxisName = rec->rec->metadata->GetMetaDatumStr("AmplCoord","Voltage");
    std::string UnitName = rec->rec->metadata->GetMetaDatumStr("AmplUnits","Volts");

    return FindAxis(AxisName,UnitName);
  }

  void SetVertScale(std::shared_ptr<display_channel> c,double scalefactor,bool pixelflag)
  {
    std::shared_ptr<display_axis> a;
    const std::string chan_name = c->FullName;
    
    std::shared_ptr<ndarray_recording_ref> chan_data;
    try {
      chan_data = current_globalrev->get_recording_ref(chan_name);
    } catch (snde_error &) {
      // no such reference
      return;
    }

    size_t ndim=chan_data->layout.dimlen.size();

    /*  set the scaling of whatever unit is used on the vertical axis of this channel */
    if (ndim==1) {
      a = GetAmplAxis(chan_name);
      //if (pixelflag) {
      //  c->UnitsPerDiv=scalefactor/pixelsperdiv;
      //} else {
      std::lock_guard<std::mutex> adminlock(c->admin);
      c->Scale=scalefactor;	  
      //}
	
      return;
    } else if (ndim==0) {
      return;
    } else if (ndim >= 2) {
      /* image or image array */
      a=GetSecondAxis(chan_name);
      //if (pixelflag)
      //  a->unit->scale=scalefactor/pixelsperdiv;
      //else
      std::lock_guard<std::mutex> adminlock(a->unit->admin);
      a->unit->scale=scalefactor;
      return;
    } else {
      assert(0);
      return;
    }
    
    
  }
  
  
  std::tuple<bool,double,bool> GetVertScale(std::shared_ptr<display_channel> c)
  // returns (success,scalefactor,pixelflag)
  {
    std::shared_ptr<display_axis> a;
    double scalefactor;
    bool success=false;

    const std::string &chan_name = c->FullName;
    
    std::shared_ptr<ndarray_recording_ref> chan_data;
    try {
      chan_data = current_globalrev->get_recording_ref(chan_name);
    } catch (snde_error &) {
      // no such reference
      return std::make_tuple(false,0.0,false);
    }
    
    size_t ndim=chan_data->layout.dimlen.size();

    /* return the units/div of whatever unit is used on the vertical axis of this channel */
    if (ndim==1) {
      a = GetAmplAxis(chan_name);
      //if (a->unit->pixelflag) {
      //  scalefactor=c->UnitsPerDiv*pixelsperdiv;
      ///} else {
      {
	std::lock_guard<std::mutex> adminlock(c->admin);
	scalefactor=c->Scale;
      }
      //}
      std::lock_guard<std::mutex> adminlock(a->admin);
      return std::make_tuple(true,scalefactor,a->unit->pixelflag);
    } else if (ndim==0) {
      //return std::make_tuple(true,1.0,false);
      return std::make_tuple(false,0.0,false);
    } else if (ndim >= 2) {
      /* image or image array */
      a=GetSecondAxis(chan_name);
      //if (a->unit->pixelflag)
      //  scalefactor=a->unit->scale*pixelsperdiv;
      //else
      {
	std::lock_guard<std::mutex> adminlock(a->unit->admin);
	scalefactor=a->unit->scale;
	return std::make_tuple(true,scalefactor,a->unit->pixelflag);
      }
    } else {
      assert(0);
      return std::make_tuple(false,0.0,false);
    }
    return std::make_tuple(false,0.0,false);    
  }
  

  
  double GetVertUnitsPerDiv(std::shared_ptr<display_channel> c)
  {
    std::shared_ptr<display_axis> a;
    double scalefactor;
    double UnitsPerDiv;

    
    const std::string &chan_name = c->FullName;
    
    std::shared_ptr<ndarray_recording_ref> chan_data;
    try {
      chan_data = current_globalrev->get_recording_ref(chan_name);
    } catch (snde_error &) {
      // no such reference
      return 0.0;
    }


    size_t ndim=chan_data->layout.dimlen.size();

    /* return the units/div of whatever unit is used on the vertical axis of this channel */
    if (ndim==1) {
      a = GetAmplAxis(chan_name);
      {
	std::lock_guard<std::mutex> adminlock(c->admin);
	scalefactor=c->Scale;
      }
	
      UnitsPerDiv=scalefactor;
      
      std::lock_guard<std::mutex> adminlock(a->admin);
      if (a->unit->pixelflag) {
	UnitsPerDiv *= pixelsperdiv;
      } 
      return UnitsPerDiv;
    } else if (ndim==0) {
      return 1.0;
    } else if (ndim >= 2) {
      /* image or image array */
      a=GetSecondAxis(chan_name);
      {
	std::lock_guard<std::mutex> adminlock(a->unit->admin);
	scalefactor=a->unit->scale;
	UnitsPerDiv=scalefactor;
	if (a->unit->pixelflag) {
	  UnitsPerDiv *= pixelsperdiv;
	}
      }
      return UnitsPerDiv;
    } else {
      assert(0);
      return 0.0;
    }
    
    return 0.0;
  }
};


  struct display_requirement {
    // ***!!! The channelpath and mode (with extended parameters) should uniquely define
    // the needed output (generated on-demand channels via the renderable function
    // and renderable channelpath)
    
    std::string channelpath;
    rendermode_ext mode; // see rendermode.hpp; contains parameter block
    std::shared_ptr<recording_base> original_recording;
    std::shared_ptr<recording_display_handler_base> display_handler; 
    //std::shared_ptr<image_reference> imgref; // image reference, out of original_recording; or nullptr; CAN WE GET RID OF THIS????
    
    std::shared_ptr<std::string> renderable_channelpath;
    std::shared_ptr<instantiated_math_function> renderable_function;
    std::vector<std::shared_ptr<display_requirement>> sub_requirements;
    
    display_requirement(std::string channelpath,rendermode_ext mode,std::shared_ptr<recording_base> original_recording,std::shared_ptr<recording_display_handler_base> display_handler) :
      channelpath(channelpath),
      mode(mode),
      original_recording(original_recording),
      display_handler(display_handler)
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
    recording_display_handler_base(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) :
      display(display),
      displaychan(displaychan),
      base_rss(base_rss)
    {
      
    }
    
    virtual ~recording_display_handler_base()=default; // polymorphic

    virtual std::shared_ptr<display_requirement> get_display_requirement(int simple_goal,std::shared_ptr<renderparams_base> params_from_parent)=0;
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
  
  class multi_ndarray_recording_display_handler: public recording_display_handler_base {
  public:
    // From recording_display_handler_base
    //std::shared_ptr<display_info> display;
    //std::shared_ptr<display_channel> displaychan;
    //std::shared_ptr<recording_set_state> base_rss;
    
    multi_ndarray_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss);
    
    virtual ~multi_ndarray_recording_display_handler()=default; // polymorphic

    virtual std::shared_ptr<display_requirement> get_display_requirement(int simple_goal,std::shared_ptr<renderparams_base> params_from_parent);

    
  };



  class meshed_part_recording_display_handler: public recording_display_handler_base {
  public:
    // From recording_display_handler_base
    //std::shared_ptr<display_info> display;
    //std::shared_ptr<display_channel> displaychan;
    //std::shared_ptr<recording_set_state> base_rss;
    
    meshed_part_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss);
    
    virtual ~meshed_part_recording_display_handler()=default; // polymorphic

    virtual std::shared_ptr<display_requirement> get_display_requirement(int simple_goal,std::shared_ptr<renderparams_base> params_from_parent);

    
  };



  class meshed_parameterization_recording_display_handler: public recording_display_handler_base {
  public:
    // From recording_display_handler_base
    //std::shared_ptr<display_info> display;
    //std::shared_ptr<display_channel> displaychan;
    //std::shared_ptr<recording_set_state> base_rss;
    
    meshed_parameterization_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss);
    
    virtual ~meshed_parameterization_recording_display_handler()=default; // polymorphic

    virtual std::shared_ptr<display_requirement> get_display_requirement(int simple_goal,std::shared_ptr<renderparams_base> params_from_parent);

    
  };


  class textured_part_recording_display_handler: public recording_display_handler_base {
  public:
    // From recording_display_handler_base
    //std::shared_ptr<display_info> display;
    //std::shared_ptr<display_channel> displaychan;
    //std::shared_ptr<recording_set_state> base_rss;
    
    textured_part_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss);
    
    virtual ~textured_part_recording_display_handler()=default; // polymorphic

    virtual std::shared_ptr<display_requirement> get_display_requirement(int simple_goal,std::shared_ptr<renderparams_base> params_from_parent);

    
  };


  class assembly_recording_display_handler: public recording_display_handler_base {
  public:
    // From recording_display_handler_base
    //std::shared_ptr<display_info> display;
    //std::shared_ptr<display_channel> displaychan;
    //std::shared_ptr<recording_set_state> base_rss;
    
    assembly_recording_display_handler(std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss);
    
    virtual ~assembly_recording_display_handler()=default; // polymorphic

    virtual std::shared_ptr<display_requirement> get_display_requirement(int simple_goal,std::shared_ptr<renderparams_base> params_from_parent);

    
  };


  std::shared_ptr<display_requirement> traverse_display_requirement(std::shared_ptr<display_info> display,std::shared_ptr<recording_set_state> base_rss,std::shared_ptr<display_channel> displaychan, int simple_goal,std::shared_ptr<renderparams_base> params_from_parent); // simple_goal such as SNDE_SRG_RENDERING

  
  // Go through the vector of channels we want to display,
  // and figure out
  // (a) all channels that will be necessary, and
  // (b) the math function (if necessary) to render to rgba, and
  // (c) the name of the renderable rgba channel
  std::map<std::string,std::shared_ptr<display_requirement>> traverse_display_requirements(std::shared_ptr<display_info> display,std::shared_ptr<recording_set_state> base_rss /* (usually a globalrev) */, const std::vector<std::shared_ptr<display_channel>> &displaychans);

}
#endif // SNDE_REC_DISPLAY_HPP
