#include "snde/rec_display.hpp"
#include "snde/recstore.hpp"

namespace snde {
  display_unit::display_unit(units unit,
			     double scale,
			     bool pixelflag) :
    unit(unit),
    scale(scale),
    pixelflag(pixelflag)
  {
    
  }


  display_axis::display_axis(const std::string &axis,
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

  /*
  recdisplay_notification_receiver::recdisplay_notification_receiver()
  {
    
  }
  */



  display_channel::display_channel(const std::string &FullName,//std::shared_ptr<mutableinfostore> chan_data,
				   float Scale,float Position,float HorizPosition,float VertCenterCoord,bool VertZoomAroundAxis,float Offset,float Alpha,
				   size_t ColorIdx,bool Enabled, size_t DisplayFrame,size_t DisplaySeq,
				   size_t ColorMap,int render_mode) :
    FullName(FullName),
    //chan_data(chan_data),
    Scale(Scale),
    Position(Position),
    HorizPosition(HorizPosition),
    VertCenterCoord(VertCenterCoord),
    VertZoomAroundAxis(VertZoomAroundAxis),
    Offset(Offset),
    Alpha(Alpha),
    ColorIdx(ColorIdx),
    Enabled(Enabled),
    DisplayFrame(DisplayFrame),
    DisplaySeq(DisplaySeq),
    ColorMap(ColorMap),
    render_mode(render_mode)

  {

  }


  void display_channel::set_enabled(bool Enabled)
  {
    std::lock_guard<std::mutex> dc_lock(admin);
    this->Enabled=Enabled;
  }

  /*
  void display_channel::add_adjustment_dep(std::shared_ptr<recdisplay_notification_receiver> notifier)
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
  */


  /*
  void display_channel::mark_as_dirty()
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
  */

  std::string PrintWithSIPrefix(double val, const std::string &unitabbrev, int sigfigs)
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



  
  bool operator==(const RecColor &lhs, const RecColor &rhs)
  {
    return (lhs.R == rhs.R && lhs.G==rhs.G && lhs.B==rhs.B);
  }

  bool operator!=(const RecColor &lhs, const RecColor &rhs)
  {
    return !(lhs==rhs);
  }





  display_info::display_info(std::shared_ptr<recdatabase> recdb) :
    recdb(recdb),
    selected_posn(display_posn{
      nullptr,
      0.0,
      nullptr,
      0.0,
    }),
    vertnormals_function(recdb->math_functions()->at("spatialnde2.vertnormals")),
    colormapping_function(recdb->math_functions()->at("spatialnde2.colormap")),
    vertexarray_function(recdb->math_functions()->at("spatialnde2.meshedpart_vertexarray")),
    texvertexarray_function(recdb->math_functions()->at("spatialnde2.meshedparameterization_texvertexarray"))

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
    drawareawidth=1.0; // also updated by set_pixelsperdiv
    drawareaheight=1.0;
  }

  void display_info::set_current_globalrev(std::shared_ptr<globalrevision> globalrev)
  {
    std::lock_guard<std::mutex> adminlock(admin);
    this->current_globalrev=globalrev;
  }


  void display_info::set_selected_posn(const display_posn &markerposn)
  {
    std::lock_guard<std::mutex> adminlock(admin);

    selected_posn = markerposn;
  }


  display_posn display_info::get_selected_posn() const
  {
    std::lock_guard<std::mutex> adminlock(admin);

    return selected_posn;
  }


  std::shared_ptr<display_channel> display_info::lookup_channel(const std::string &recfullname)
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

  void display_info::set_pixelsperdiv(size_t new_drawareawidth,size_t new_drawareaheight)
  {
    std::lock_guard<std::mutex> adminlock(admin);

    drawareawidth = new_drawareawidth;
    drawareaheight = new_drawareaheight;
    
    double pixelsperdiv_width = drawareawidth*1.0/horizontal_divisions;
    double pixelsperdiv_height = drawareaheight*1.0/vertical_divisions;

    
    
    pixelsperdiv = std::min(pixelsperdiv_width,pixelsperdiv_height);
  }


  std::shared_ptr<display_channel> display_info::_add_new_channel(const std::string &fullname,std::shared_ptr<recording_base> rec)
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
    std::shared_ptr<display_channel> new_display_channel=std::make_shared<display_channel>(fullname,/*info, */  1.0,0.0,0.0,0.0,false,0.0,1.0,_NextColor,false,0,0,0,SNDE_DCRM_INVALID);
    
    
    //std::lock_guard<std::mutex> adminlock(admin);
    channel_info.emplace(fullname,new_display_channel);
    channel_layer_order.push_back(fullname);
    return new_display_channel;
  }



  std::vector<std::shared_ptr<display_channel>> display_info::update(std::shared_ptr<globalrevision> globalrev,const std::string &selected, bool selected_only,bool include_disabled,bool include_hidden)

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
      if (channels_considered.find(fullname) == channels_considered.end()) {
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
  


  std::shared_ptr<display_unit> display_info::FindUnitLocked(const std::string &name)
  {
    units u=units::parseunits(name);

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



  std::shared_ptr<display_unit> display_info::FindUnit(const std::string &name)
  {
    std::lock_guard<std::mutex> adminlock(admin);
    return FindUnitLocked(name);
  };


  std::shared_ptr<display_axis> display_info::FindAxisLocked(const std::string &axisname,const std::string &unitname)
  {
    units u=units::parseunits(unitname);

    std::shared_ptr<display_unit> unit=FindUnitLocked(unitname);

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



  std::shared_ptr<display_axis> display_info::FindAxis(const std::string &axisname,const std::string &unitname)
  {

    std::lock_guard<std::mutex> adminlock(admin);
    
    return FindAxisLocked(axisname,unitname);
  }



  std::shared_ptr<display_axis> display_info::GetFirstAxis(const std::string &fullname /*std::shared_ptr<mutableinfostore> rec */)
  {
    std::shared_ptr<ndarray_recording_ref> rec;
    try {
      rec = current_globalrev->get_recording_ref(fullname);
    } catch (snde_error &) {
      // no such reference
      return FindAxis("Time","seconds");
    }
    
    std::string AxisName = rec->rec->metadata->GetMetaDatumStr("nde_axis0_coord","Time");
    std::string UnitName = rec->rec->metadata->GetMetaDatumStr("nde_axis0_units","seconds");

    return FindAxis(AxisName,UnitName);
  }



  std::shared_ptr<display_axis> display_info::GetFirstAxisLocked(const std::string &fullname /*std::shared_ptr<mutableinfostore> rec */)
  {
    std::shared_ptr<ndarray_recording_ref> rec;
    try {
      rec = current_globalrev->get_recording_ref(fullname);
    } catch (snde_error &) {
      // no such reference
      return FindAxisLocked("Time","seconds");
    }
    
    std::string AxisName = rec->rec->metadata->GetMetaDatumStr("nde_axis0_coord","Time");
    std::string UnitName = rec->rec->metadata->GetMetaDatumStr("nde_axis0_units","seconds");

    return FindAxisLocked(AxisName,UnitName);
  }



  std::shared_ptr<display_axis> display_info::GetSecondAxis(const std::string &fullname)
  {
    std::shared_ptr<ndarray_recording_ref> rec;
    try {
      rec = current_globalrev->get_recording_ref(fullname);
    } catch (snde_error &) {
      // no such reference
      return FindAxis("Time","seconds");
    }


    std::string AxisName = rec->rec->metadata->GetMetaDatumStr("nde_axis1_coord","Time");
    std::string UnitName = rec->rec->metadata->GetMetaDatumStr("nde_axis1_units","seconds");

    return FindAxis(AxisName,UnitName);
  }

  
  std::shared_ptr<display_axis> display_info::GetSecondAxisLocked(const std::string &fullname)
  {
    std::shared_ptr<ndarray_recording_ref> rec;
    try {
      rec = current_globalrev->get_recording_ref(fullname);
    } catch (snde_error &) {
      // no such reference
      return FindAxis("Time","seconds");
    }


    std::string AxisName = rec->rec->metadata->GetMetaDatumStr("nde_axis1_coord","Time");
    std::string UnitName = rec->rec->metadata->GetMetaDatumStr("nde_axis1_units","seconds");

    return FindAxisLocked(AxisName,UnitName);
  }


  std::shared_ptr<display_axis> display_info::GetThirdAxis(const std::string &fullname)
  {
    std::shared_ptr<ndarray_recording_ref> rec;
    try {
      rec = current_globalrev->get_recording_ref(fullname);
    } catch (snde_error &) {
      // no such reference
      return FindAxis("Time","seconds");
    }

    std::string AxisName = rec->rec->metadata->GetMetaDatumStr("nde_axis2_coord","Time");
    std::string UnitName = rec->rec->metadata->GetMetaDatumStr("nde_axis2_units","seconds");

    return FindAxis(AxisName,UnitName);
  }


  std::shared_ptr<display_axis> display_info::GetThirdAxisLocked(const std::string &fullname)
  {
    std::shared_ptr<ndarray_recording_ref> rec;
    try {
      rec = current_globalrev->get_recording_ref(fullname);
    } catch (snde_error &) {
      // no such reference
      return FindAxis("Time","seconds");
    }

    std::string AxisName = rec->rec->metadata->GetMetaDatumStr("nde_axis2_coord","Time");
    std::string UnitName = rec->rec->metadata->GetMetaDatumStr("nde_axis2_units","seconds");

    return FindAxisLocked(AxisName,UnitName);
  }

  std::shared_ptr<display_axis> display_info::GetFourthAxis(const std::string &fullname)
  {
    std::shared_ptr<ndarray_recording_ref> rec;
    try {
      rec = current_globalrev->get_recording_ref(fullname);
    } catch (snde_error &) {
      // no such reference
      return FindAxis("Time","seconds");
    }

    std::string AxisName = rec->rec->metadata->GetMetaDatumStr("nde_axis3_coord","Time");
    std::string UnitName = rec->rec->metadata->GetMetaDatumStr("nde_axis3_units","seconds");

    return FindAxis(AxisName,UnitName);
  }


  std::shared_ptr<display_axis> display_info::GetFourthAxisLocked(const std::string &fullname)
  {
    std::shared_ptr<ndarray_recording_ref> rec;
    try {
      rec = current_globalrev->get_recording_ref(fullname);
    } catch (snde_error &) {
      // no such reference
      return FindAxis("Time","seconds");
    }

    std::string AxisName = rec->rec->metadata->GetMetaDatumStr("nde_axis3_coord","Time");
    std::string UnitName = rec->rec->metadata->GetMetaDatumStr("nde_axis3_units","seconds");

    return FindAxisLocked(AxisName,UnitName);
  }



  std::shared_ptr<display_axis> display_info::GetAmplAxis(const std::string &fullname)
  {
    std::shared_ptr<ndarray_recording_ref> rec;
    try {
      rec = current_globalrev->get_recording_ref(fullname);
    } catch (snde_error &) {
      // no such reference
      return FindAxis("Time","seconds");
    }

    std::string AxisName = rec->rec->metadata->GetMetaDatumStr("nde_amplcoord","Voltage");
    std::string UnitName = rec->rec->metadata->GetMetaDatumStr("nde_amplunits","Volts");

    return FindAxis(AxisName,UnitName);
  }


  std::shared_ptr<display_axis> display_info::GetAmplAxisLocked(const std::string &fullname)
  {
    std::shared_ptr<ndarray_recording_ref> rec;
    try {
      rec = current_globalrev->get_recording_ref(fullname);
    } catch (snde_error &) {
      // no such reference
      return FindAxis("Time","seconds");
    }

    std::string AxisName = rec->rec->metadata->GetMetaDatumStr("nde_amplcoord","Voltage");
    std::string UnitName = rec->rec->metadata->GetMetaDatumStr("nde_amplunits","Volts");

    return FindAxisLocked(AxisName,UnitName);
  }


  void display_info::SetVertScale(std::shared_ptr<display_channel> c,double scalefactor,bool pixelflag)
  {
    std::shared_ptr<display_axis> a;
    snde_warning("display->SetVertScale()");

    const std::string chan_name = c->FullName;
    int render_mode;
    {

      std::lock_guard<std::mutex> chanadmin(c->admin);
      render_mode = c->render_mode;
    }

    snde_warning("SetVertScale(): scalefactor=%f; render_mode=%d",scalefactor,render_mode);

    /*  set the scaling of whatever unit is used on the vertical axis of this channel */
    if (render_mode==SNDE_DCRM_INVALID) {
      // do nothing
    } else if (render_mode==SNDE_DCRM_WAVEFORM) {
      a = GetAmplAxis(chan_name);
      //if (pixelflag) {
      //  c->UnitsPerDiv=scalefactor/pixelsperdiv;
      //} else {
      std::lock_guard<std::mutex> adminlock(c->admin);
      c->Scale=scalefactor;	  
      //}
	
      return;
    } else if (render_mode==SNDE_DCRM_SCALAR) {
      return;
    } else if (render_mode==SNDE_DCRM_IMAGE) {
      /* image or image array */
      a=GetSecondAxis(chan_name);
      //if (pixelflag)
      //  a->unit->scale=scalefactor/pixelsperdiv;
      //else
      std::lock_guard<std::mutex> adminlock(a->unit->admin);
      a->unit->scale=scalefactor;
      return;
    } else if (render_mode==SNDE_DCRM_GEOMETRY) {
      c->Scale = scalefactor;
    } else {
      assert(0);
      return;
    }
    
    
  }


  std::tuple<bool,double,bool> display_info::GetVertScale(std::shared_ptr<display_channel> c)
  // returns (success,scalefactor,pixelflag)
  {
    std::shared_ptr<display_axis> a;
    double scalefactor;
    bool success=false;

    const std::string &chan_name = c->FullName;
    int render_mode;
    {

      std::lock_guard<std::mutex> chanadmin(c->admin);
      render_mode = c->render_mode;
    }
    
    

    /* return the units/div of whatever unit is used on the vertical axis of this channel */
    if (render_mode==SNDE_DCRM_INVALID) {
      // do nothing
    } else if (render_mode==SNDE_DCRM_WAVEFORM) {
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
    } else if (render_mode==SNDE_DCRM_SCALAR) {
      //return std::make_tuple(true,1.0,false);
      return std::make_tuple(false,0.0,false);
    } else if (render_mode == SNDE_DCRM_IMAGE) {
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
    } else if (render_mode == SNDE_DCRM_GEOMETRY) {
      {
	{
	  std::lock_guard<std::mutex> adminlock(c->admin);
	  scalefactor=c->Scale;
	}
	return std::make_tuple(true,scalefactor,false); // a->unit->pixelflag);
      }
      
    } else {
      assert(0);
      return std::make_tuple(false,0.0,false);
    }
    return std::make_tuple(false,0.0,false);    
  }



  double display_info::GetVertUnitsPerDiv(std::shared_ptr<display_channel> c)
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


    int render_mode;
    {

      std::lock_guard<std::mutex> chanadmin(c->admin);
      render_mode = c->render_mode;
    }
    

    /* return the units/div of whatever unit is used on the vertical axis of this channel */
    if (render_mode==SNDE_DCRM_INVALID) {

    } else if (render_mode==SNDE_DCRM_WAVEFORM) {
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
    } else if (render_mode == SNDE_DCRM_SCALAR) {
      return 1.0;
    } else if (render_mode==SNDE_DCRM_IMAGE) {
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
    } else if (render_mode==SNDE_DCRM_GEOMETRY) {
      
    } else {
      assert(0);
      return 0.0;
    }
    
    return 0.0;
  }

  
  void display_info::handle_key_down(const std::string &selected_channel,int key,bool shift,bool alt,bool ctrl)
  {
    
  }

  
  void display_info::handle_special_down(const std::string &selected_channel,int special,bool shift,bool alt,bool ctrl)
  {
    // Primarily we do this through the QT GUI but it's also possible to implement straight in
    // OSG if needed (because that gives a better way to
    // ensure consistency with sliders, etc.) This is currenly more of an example than anything. 
    
    std::shared_ptr<display_axis> a = GetFirstAxis(selected_channel);
    std::shared_ptr<display_unit> u;
    double scale;
    {
      std::lock_guard<std::mutex> axisadm(a->admin);
      u=a->unit;
      std::lock_guard<std::mutex> unitadm(u->admin);
      scale = u->scale;
    }
    
    switch (special) {
    case SNDE_RDK_LEFT:
      double perdiv=scale;
      int power=(int)(log(perdiv)/log(10.0) + 1000.01) -1000;
      int bigdigit = (int)(0.5+perdiv/pow(10,power));

      switch(bigdigit) {
      case 1:
	bigdigit=2;
	break;
	
      case 2:
	bigdigit=5;
	break; 
	
      case 5: 
	bigdigit=10;
	break;
	
      case 10: 
	bigdigit=20;
	break;
	
      }

      {
	std::lock_guard<std::mutex> unitadm(u->admin);
	u->scale=pow(10,power)*bigdigit;
      }
      break;
      
    }
  }

  
};
