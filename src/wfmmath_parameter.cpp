#include <typeinfo>

#include "snde_error.hpp"
#include "wfmdb_paths.hpp"

namespace snde {

  math_parameter::math_parameter(unsigned paramtype) :
    paramtype(paramtype)
  {

  }

  virtual std::string math_parameter::get_string(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context)
  {
    raise snde_error("Cannot get string value from parameter of class %s",(char *)typeid(*this).name()); 
  }
  
  virtual int64_t math_parameter::get_int(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context)
  {
    raise snde_error("Cannot get integer value from parameter of class %s",(char *)typeid(*this).name()); 
    
  }
  
  virtual double math_parameter::get_double(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context)
  {
    raise snde_error("Cannot get double value from parameter of class %s",(char *)typeid(*this).name()); 

  }
  
  virtual std::shared_ptr<waveform> math_parameter::get_waveform(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context) // should only return ready waveforms
  {
    raise snde_error("Cannot get waveform value from parameter of class %s",(char *)typeid(*this).name()); 

  }
  virtual std::set<std::string> math_parameter::get_prerequisites(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context) // obtain immediate dependencies of this parameter (absolute path channel names); typically only the waveform
  {
    return std::set<std::string>(); // default to no prerequisites
  }


  math_parameter_string_const::math_parameter_string_const(std::string string_constant) :
    string_constant(string_constant)
  {

  }


  virtual std::string math_parameter_string_const::get_string(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context)
  {
    return string_constant;
  }



  math_parameter_int_const::math_parameter_int_const(int64_t int_constant) :
    int_constant(int_constant)
  {

  }
  
  virtual int64_t math_parameter_int_const::get_int(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context)
  {
    return int_constant;
  }
  
  math_parameter_double_const::math_parameter_double_const(double double_constant) :
    double_constant(double_constant)
  {

  }

  virtual double math_parameter_double_const::get_double(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context)
  {
    return double_constant;
  }

  math_parameter_waveform::math_parameter_waveform(std::string channel_name) :
    channel_name(channel_name)
  {

  }


  virtual std::set<std::string> math_parameter_waveform::get_prerequisites(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context)
  {
    std::set<std::string> retval;
    retval.emplace(channel_name);
    return retval;
  }

  virtual std::shared_ptr<waveform> math_parameter_waveform::get_waveform(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context) // should only return ready waveforms because we shouldn't be called until our deps are ready. 
  {
    std::shared_ptr<waveform> wfm;
    std::string fullpath = wfmdb_path_join(channel_path_context,channel_name);
    {
      std::lock_guard<std::mutex> wsslock(wss->admin);
      wfm=wss->channel_map.at(fullpath).wfm;
    }
    return wfm; 
  }
  
}
