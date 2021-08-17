#include <typeinfo>

#include "wfmmath_parameter.hpp"
#include "wfmmath.hpp"
#include "wfmstore.hpp"

namespace snde {

  math_parameter::math_parameter(unsigned paramtype) :
    paramtype(paramtype)
  {

  }

  std::string math_parameter::get_string(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index)
  {
    throw snde_error("Cannot get string value from parameter of class %s for parameter %d of %s",(char *)typeid(*this).name(),parameter_index,fcn_def->definition_command.c_str()); 
  }
  
  int64_t math_parameter::get_int(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index)
  {
    throw snde_error("Cannot get integer value from parameter of class %s for parameter %d of %s",(char *)typeid(*this).name(),parameter_index,fcn_def->definition_command.c_str()); 
    
  }
  
  double math_parameter::get_double(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index)
  {
    throw snde_error("Cannot get double value from parameter of class %s for parameter %d of %s",(char *)typeid(*this).name(),parameter_index,fcn_def->definition_command.c_str()); 

  }
  
  std::shared_ptr<waveform_base> math_parameter::get_waveform(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index) // should only return ready waveforms
  {
    throw snde_error("Cannot get waveform value from parameter of class %s for parameter %d of %s",(char *)typeid(*this).name(),parameter_index,fcn_def->definition_command.c_str()); 

  }
  std::set<std::string> math_parameter::get_prerequisites(/*std::shared_ptr<waveform_set_state> wss,*/ const std::string &channel_path_context) // obtain immediate dependencies of this parameter (absolute path channel names); typically only the waveform
  {
    return std::set<std::string>(); // default to no prerequisites
  }


  math_parameter_string_const::math_parameter_string_const(std::string string_constant) :
    math_parameter(SNDE_MFPT_STR),
    string_constant(string_constant)
  {

  }


  std::string math_parameter_string_const::get_string(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index)
  // parameter_index human interpreted parameter number, starting at 1, for error messages only
  {
    return string_constant;
  }



  math_parameter_int_const::math_parameter_int_const(int64_t int_constant) :
    math_parameter(SNDE_MFPT_INT),
    int_constant(int_constant)
  {

  }
  
  int64_t math_parameter_int_const::get_int(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index)
  // parameter_index human interpreted parameter number, starting at 1, for error messages only
  {
    return int_constant;
  }
  
  math_parameter_double_const::math_parameter_double_const(double double_constant) :
    math_parameter(SNDE_MFPT_DBL),
    double_constant(double_constant)
  {

  }

  double math_parameter_double_const::get_double(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index)
  // parameter_index human interpreted parameter number, starting at 1, for error messages only
  {
    return double_constant;
  }

  math_parameter_waveform::math_parameter_waveform(std::string channel_name) :
    math_parameter(SNDE_MFPT_WAVEFORM),
    channel_name(channel_name)
  {

  }


  std::set<std::string> math_parameter_waveform::get_prerequisites(/*std::shared_ptr<waveform_set_state> wss,*/ const std::string &channel_path_context)
  {
    std::set<std::string> retval;
    retval.emplace(channel_name);
    return retval;
  }

  std::shared_ptr<waveform_base> math_parameter_waveform::get_waveform(std::shared_ptr<waveform_set_state> wss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index) // should only return ready waveforms because we shouldn't be called until our deps are ready.
  // parameter_index human interpreted parameter number, starting at 1, for error messages only
  {
    std::shared_ptr<waveform_base> wfm;
    std::string fullpath = wfmdb_path_join(channel_path_context,channel_name);
    {
      //std::lock_guard<std::mutex> wsslock(wss->admin); // Think this locking is actually unnecessary
      wfm=wss->wfmstatus.channel_map.at(fullpath).wfm();
    }
    return wfm; 
  }
  
}
