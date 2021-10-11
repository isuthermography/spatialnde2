#ifndef RECMATH_PARAMETER_HPP
#define RECMATH_PARAMETER_HPP

#include <set>
#include <memory>
#include <unordered_map>

#include "snde/snde_error.hpp"
#include "snde/recdb_paths.hpp"

namespace snde {

  // forward declarations
  class recording_base; // defined in recstore.hpp
  class recording_set_state; // defined in recstore.hpp
  class math_definition; // defined in recmath.hpp
  
  class math_parameter {
  public:
    unsigned paramtype; // SNDE_MFPT_XXX from recmath.hpp


    math_parameter(unsigned paramtype);
    // Rule of 3
    math_parameter(const math_parameter &) = delete;
    math_parameter& operator=(const math_parameter &) = delete; 
    virtual ~math_parameter()=default;  // virtual destructor required so we can be subclassed

    // default implementations that just raise runtime_error
    // function definition and parameter index are just for the error message
    // NOTE: To add support for more parameter types,
    // need to add entries here as well as modify
    // recmath_cppfunction.cpp/cpp_math_function() constructor to accept them
    // and recmath_cppfunction.hpp templates to call the appropriate
    // additional get_...() methods. Also make sure they have a SNDE_WTN entry
    // in recording.h
    virtual std::string get_string(std::shared_ptr<recording_set_state> wss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index); // parameter_index human interpreted parameter number, starting at 1, for error messages only
    virtual int64_t get_int(std::shared_ptr<recording_set_state> wss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index); // parameter_index human interpreted parameter number, starting at 1, for error messages only
    virtual double get_double(std::shared_ptr<recording_set_state> wss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index); // parameter_index human interpreted parameter number, starting at 1, for error messages only
    virtual std::shared_ptr<recording_base> get_recording(std::shared_ptr<recording_set_state> wss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index); // should only return ready recordings because we shouldn't be called until dependencies are ready // parameter_index human interpreted parameter number, starting at 1, for error messages only

    // default implementations that returns an empty set
    virtual std::set<std::string> get_prerequisites(/*std::shared_ptr<recording_set_state> wss,*/ const std::string &channel_path_context); // obtain immediate prerequisites of this parameter (absolute path channel names); typically only the recording
  };


  class math_parameter_string_const: public math_parameter {
  public:
    std::string string_constant;

    math_parameter_string_const(std::string string_constant);
    virtual std::string get_string(std::shared_ptr<recording_set_state> wss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index);
    
  };


  class math_parameter_int_const: public math_parameter {
  public:
    int64_t int_constant;

    math_parameter_int_const(int64_t int_constant);
    virtual int64_t get_int(std::shared_ptr<recording_set_state> wss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index);
    
  };

  class math_parameter_double_const: public math_parameter {
  public:
    double double_constant;

    math_parameter_double_const(double double_constant);
    virtual double get_double(std::shared_ptr<recording_set_state> wss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index);
    
  };


  class math_parameter_recording: public math_parameter {
  public:
    std::string channel_name; // ***!!! MUST BE COMBINED WITH channel_path_context from instantiated_math_function ***!!!

    math_parameter_recording(std::string channel_name);
    virtual std::shared_ptr<recording_base> get_recording(std::shared_ptr<recording_set_state> wss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index); // should only return ready recordings. parameter_index starting at 1, just for printing messages
    virtual std::set<std::string> get_prerequisites(/*std::shared_ptr<recording_set_state> wss,*/ const std::string &channel_path_context); // obtain immediate prerequisites of this parameter (absolute path channel names); typically only the recording
    
  };

  // ***!!! Could have more classes here to implement e.g. parameters derived from metadata, expressions involving metadata, etc. 

};

#endif // RECMATH_PARAMETER_HPP

