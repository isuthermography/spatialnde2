#include <memory>
#include <mutex>
#include <typeindex>

#include "units.hpp"
#include "mutablewfmstore.hpp"
#include "revision_manager.hpp"

#ifndef SNDE_WFM_DISPLAY_HPP
#define SNDE_WFM_DISPLAY_HPP

namespace snde {
  
struct display_unit {
  units unit;
  double scale; // Units per division (if not pixelflag) or units per pixel (if pixelflag)
  bool pixelflag;
};

struct display_axis {
  std::string axis;
  std::string abbrev;
  std::shared_ptr<display_unit> unit;
  bool has_abbrev;

  double CenterCoord; // horizontal coordinate of the center of the display
  double DefaultOffset;
  double DefaultUnitsPerDiv; // Should be 1.0, 2.0, or 5.0 times a power of 10

  // double MousePosn;
};


struct display_channel {

  
  std::string FullName; // full name, including colon (or slash?) separating tree elements
  std::shared_ptr<mutableinfostore> chan_data;
  
  float UnitsPerDiv; // vertical axis scaling for 1D wfms; color axis scaling for 2D waveforms
  float Position; // vertical offset on display, in divisions. To get in units, multiply by GetVertUnitsPerDiv(Chan)
  float Offset; // >= 2d only, intensity offset on display, in amplitude units
  float Alpha; // alpha transparency of channel: 1: fully visible, 0: fully transparent
  size_t ColorIdx; // index into color table for how this channel is colored
  bool Enabled; // Is this channel currently visible
  // unsigned long long currevision;
  size_t DisplayFrame; // Frame # to display
  size_t DisplaySeq; // Sequence # to display
  // NeedAxisScales
  size_t ColorMap; // colormap selection

  std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>> adjustment_deps; // these trm_dependencies should be triggered when these parameters are changed.

  std::mutex displaychan_mutex; // protects all members, as the display_channel
  // may be accessed from transform threads, not just the GUI thread

  
  display_channel(std::string FullName,std::shared_ptr<mutableinfostore> chan_data,
		  float UnitsPerDiv,float Position,float Offset,float Alpha,
		  size_t ColorIdx,bool Enabled, size_t DisplayFrame,size_t DisplaySeq,
		  size_t ColorMap) :
    FullName(FullName),
    chan_data(chan_data),
    UnitsPerDiv(UnitsPerDiv),
    Position(Position),
    Offset(Offset),
    Alpha(Alpha),
    ColorIdx(ColorIdx),
    Enabled(Enabled),
    DisplayFrame(DisplayFrame),
    DisplaySeq(DisplaySeq),
    ColorMap(ColorMap)

  {

  }

};


struct display_posn {
  // represents a position on the display, such as a clicked location
  // or the current mouse position
  std::shared_ptr<display_axis> Horiz;
  double HorizPosn;  // position on Horiz axis

  std::shared_ptr<display_axis> Vert;
  double VertPosn;  // position on Vert axis
  
  std::shared_ptr<display_axis> Intensity;
  double IntensityPosn;  // recorded intensity at this position
  
};

struct WfmColor {
  double R,G,B;

  friend bool operator==(const WfmColor &lhs, const WfmColor &rhs)
  {
    return (lhs.R == rhs.R && lhs.G==rhs.G && lhs.B==rhs.B);
  }
  friend bool operator!=(const WfmColor &lhs, const WfmColor &rhs)
  {
    return !(lhs==rhs);
  }
};

  static const WfmColor WfmColorTable[]={
					 {1.0,0.0,0.0}, /* Red */
					 {0.0,0.4,1.0}, /* Blue */
					 {0.0,1.0,0.0}, /* Green */
					 {1.0,1.0,0.0}, /* Yellow */
					 {0.0,1.0,1.0}, /* Cyan */
					 {1.0,0.0,1.0}, /* Magenta */
  };
  

  
class display_info {
public:
  
  std::vector<std::shared_ptr<display_unit>>  UnitList;
  std::vector<std::shared_ptr<display_axis>>  AxisList;
  size_t NextColor;
  size_t horizontal_divisions;
  size_t vertical_divisions;
  float borderwidthpixels;
  

  std::unordered_map<std::string,std::shared_ptr<display_channel>> channel_info;

  display_info(std::shared_ptr<mutablewfmdb> wfmdb)
  {
    NextColor=0;

    // numbers of divisions should be even!
    vertical_divisions=8;
    horizontal_divisions=10;
    borderwidthpixels=2.0; // width of the border line, in pixels
    
    // Insert some basic common units
    
    UnitList.push_back(std::make_shared<display_unit>(display_unit{units::parseunits("meters"),0.1,false}));
    UnitList.push_back(std::make_shared<display_unit>(display_unit{units::parseunits("seconds"),0.5,false}));
    UnitList.push_back(std::make_shared<display_unit>(display_unit{units::parseunits("Hertz"),10e3,false}));
    UnitList.push_back(std::make_shared<display_unit>(display_unit{units::parseunits("pixels"),1.0,true}));
    UnitList.push_back(std::make_shared<display_unit>(display_unit{units::parseunits("unitless"),1.0,false}));
    UnitList.push_back(std::make_shared<display_unit>(display_unit{units::parseunits("meters/second"),1.0,false}));
    UnitList.push_back(std::make_shared<display_unit>(display_unit{units::parseunits("Kelvin"),50,false}));
    UnitList.push_back(std::make_shared<display_unit>(display_unit{units::parseunits("Pascals"),50e3,false}));
    UnitList.push_back(std::make_shared<display_unit>(display_unit{units::parseunits("Volts"),1.0,false}));
    UnitList.push_back(std::make_shared<display_unit>(display_unit{units::parseunits("Amps"),1.0,false}));
    UnitList.push_back(std::make_shared<display_unit>(display_unit{units::parseunits("Newtons"),5.0,false}));
    UnitList.push_back(std::make_shared<display_unit>(display_unit{units::parseunits("Newton-meters"),5.0,false}));
    UnitList.push_back(std::make_shared<display_unit>(display_unit{units::parseunits("Degrees"),20.0,false}));
    UnitList.push_back(std::make_shared<display_unit>(display_unit{units::parseunits("Arbitrary"),10.0,false}));

    // Define some well-known axes
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"time","t",FindUnit("seconds"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"frequency","f",FindUnit("Hertz"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"X Position","x",FindUnit("meters"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"Y Position","y",FindUnit("meters"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"X Position","x",FindUnit("pixels"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"Y Position","y",FindUnit("pixels"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"U Position","u",FindUnit("meters"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"V Position","v",FindUnit("meters"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"U Position","u",FindUnit("pixels"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"V Position","v",FindUnit("pixels"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"Index","i",FindUnit("unitless"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"Velocity","v",FindUnit("meters/second"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"Temperature","T",FindUnit("Kelvin"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"Pressure","p",FindUnit("Pascals"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"Voltage","v",FindUnit("Volts"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"Current","i",FindUnit("Amps"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"Force","F",FindUnit("Newtons"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"Moment","M",FindUnit("Newton-meters"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"Angle","ang",FindUnit("degrees"),false,0.0,0.0,1.0}));
    AxisList.push_back(std::make_shared<display_axis>(display_axis{"Intensity","I",FindUnit("arbitrary"),false,0.0,0.0,1.0}));
    
  }

  double pixelsperdiv(size_t drawareawidth,size_t drawareaheight)
  {
    double pixelsperdiv_width = drawareawidth*1.0/horizontal_divisions;
    double pixelsperdiv_height = drawareawidth*1.0/vertical_divisions;

    return std::min(pixelsperdiv_width,pixelsperdiv_height);
  }

  std::vector<std::shared_ptr<display_channel>> update(std::shared_ptr<mutablewfmdb> wfmdb,std::shared_ptr<mutableinfostore> selected, bool include_disabled,bool include_hidden)

  // STILL NEED TO IMPLEMENT THE FOLLOWING: 
  // if selected is not nullptr, it will be moved to the head of the list
  // if include_disabled is set, disabled channels will be included
  // if include_hidden is set, hidden channels will be included
  {

    std::vector<std::shared_ptr<display_channel>> retval;
    
    std::shared_ptr<iterablewfmrefs> wfmlist=wfmdb->wfmlist();

    for (auto wfmiter=wfmlist->begin();wfmiter != wfmlist->end();wfmiter++) {
      std::shared_ptr<mutableinfostore> info=*wfmiter;
      std::string fullname=wfmiter.get_full_name();

      // do we have this wfm in our channel_info database? 
      auto ci_iter = channel_info.find(fullname);
      if (ci_iter != channel_info.end() && ci_iter->second->chan_data==info) {
	retval.push_back(ci_iter->second);
      } else {
	// create a new display_channel
	std::shared_ptr<display_channel> new_display_channel=std::make_shared<display_channel>(fullname,info,1.0,0.0,0.0,1.0,NextColor,true,0,0,0);
	NextColor = (NextColor + 1) % (sizeof(WfmColorTable)/sizeof(WfmColorTable[0]));

	channel_info[fullname]=new_display_channel;
	retval.push_back(new_display_channel);
      }
      
    }
    return retval;
  }
  
  
  
  std::shared_ptr<display_unit> FindUnit(std::string name)
  {
    units u=units::parseunits(name);

    for (auto & uptr: UnitList) {
      if (units::compareunits(uptr->unit,u)==1.0) {
	return uptr; 
      }
    }

    // if we are still executing, we need to create a unit
    std::shared_ptr<display_unit> uptr=std::make_shared<display_unit>(display_unit{u,1.0,false});

    UnitList.push_back(uptr);

    return uptr;
  }


  std::shared_ptr<display_axis> FindAxis(std::string axisname,std::string unitname)
  {
    units u=units::parseunits(unitname);

    for (auto & ax: AxisList) {
      if (axisname==ax->axis && units::compareunits(u,ax->unit->unit)==1.0) {
	return ax;
      }
    }

    // need to create axis
    std::shared_ptr<display_axis> ax_ptr=std::make_shared<display_axis>(display_axis{axisname,"",FindUnit(unitname),false,0.0,0.0,1.0});

    AxisList.push_back(ax_ptr);

    return ax_ptr;
  }

  std::shared_ptr<display_axis> GetFirstAxis(std::shared_ptr<mutableinfostore> wfm)
  {
    std::string AxisName = wfm->metadata.GetMetaDatumStr("Coord1","Time");
    std::string UnitName = wfm->metadata.GetMetaDatumStr("Units1","seconds");

    return FindAxis(AxisName,UnitName);
  }

  std::shared_ptr<display_axis> GetSecondAxis(std::shared_ptr<mutableinfostore> wfm)
  {
    std::string AxisName = wfm->metadata.GetMetaDatumStr("Coord2","Time");
    std::string UnitName = wfm->metadata.GetMetaDatumStr("Units2","seconds");

    return FindAxis(AxisName,UnitName);
  }

  std::shared_ptr<display_axis> GetThirdAxis(std::shared_ptr<mutableinfostore> wfm)
  {
    std::string AxisName = wfm->metadata.GetMetaDatumStr("Coord3","Time");
    std::string UnitName = wfm->metadata.GetMetaDatumStr("Units3","seconds");

    return FindAxis(AxisName,UnitName);
  }

  std::shared_ptr<display_axis> GetFourthAxis(std::shared_ptr<mutableinfostore> wfm)
  {
    std::string AxisName = wfm->metadata.GetMetaDatumStr("Coord4","Time");
    std::string UnitName = wfm->metadata.GetMetaDatumStr("Units4","seconds");

    return FindAxis(AxisName,UnitName);
  }

  double GetVertUnitsPerDiv(std::shared_ptr<display_channel> c,double pixelsperdiv)
  {
    std::shared_ptr<display_axis> a;
    double scalefactor;

    std::shared_ptr<mutabledatastore> datastore;

    datastore=std::dynamic_pointer_cast<mutabledatastore>(c->chan_data);

    if (datastore) {
      size_t ndim=datastore->dimlen.size();

      /* return the units/div of whatever unit is used on the vertical axis of this channel */
      if (ndim==1) {
	return c->UnitsPerDiv;
      } else if (ndim==0) {
	return 1.0;
      } else if (ndim >= 2) {
	/* image or image array */
	a=GetSecondAxis(c->chan_data);
	if (a->unit->pixelflag)
	  scalefactor=a->unit->scale;
	else
	  scalefactor=a->unit->scale/pixelsperdiv;
	return scalefactor*pixelsperdiv;
      } else {
	assert(0);
	return 0.0;
      }
    }
    return 0.0;
  }
};

}
#endif // SNDE_WFM_DISPLAY_HPP
