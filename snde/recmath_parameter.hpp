#ifndef RECMATH_PARAMETER_HPP
#define RECMATH_PARAMETER_HPP

#include <set>
#include <memory>
#include <unordered_map>

#include "snde/metadata.hpp"
#include "snde/snde_error.hpp"
#include "snde/recdb_paths.hpp"
#include "snde/geometry_types.h" // for snde_index

namespace snde {

  // forward declarations
  class recording_base; // defined in recstore.hpp
  class recording_set_state; // defined in recstore.hpp
  class math_definition; // defined in recmath.hpp
  class ndarray_recording_ref; // defined in recstore.hpp

  
  std::string escape_to_quoted_string(std::string to_quote);

  
  class math_parameter {
  public:
    unsigned paramtype; // SNDE_MFPT_XXX from recmath.hpp


    math_parameter(unsigned paramtype);
    // Rule of 3
    math_parameter(const math_parameter &) = delete;
    math_parameter& operator=(const math_parameter &) = delete; 
    virtual ~math_parameter()=default;  // virtual destructor required so we can be subclassed

    virtual std::string generate_parsible()=0; // generate a parsible string for Python that can be used for redefining the math function.

    virtual bool operator==(const math_parameter &ref)=0; // used for comparing parameters to instantiated_math_functions
    virtual bool operator!=(const math_parameter &ref)=0;

    
    // default implementations that just raise runtime_error
    // function definition and parameter index are just for the error message
    // NOTE: To add support for more parameter types,
    // need to add entries here as well as modify
    // recmath_cppfunction.cpp/cpp_math_function() constructor to accept them
    // and recmath_cppfunction.hpp templates to call the appropriate
    // additional get_...() methods. Also make sure they have a SNDE_RTN entry
    // in recording.h and an entry in _convert_math_param() in recmath.i
    virtual std::string get_string(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index); // parameter_index human interpreted parameter number, starting at 1, for error messages only
    virtual int64_t get_int(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index); // parameter_index human interpreted parameter number, starting at 1, for error messages only

    virtual uint64_t get_unsigned(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index); // parameter_index human interpreted parameter number, starting at 1, for error messages only

    virtual double get_double(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index); // parameter_index human interpreted parameter number, starting at 1, for error messages only

    virtual snde_bool get_bool(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index); // parameter_index human interpreted parameter number, starting at 1, for error messages only

    
    virtual snde_coord3 get_vector(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index); // parameter_index human interpreted parameter number, starting at 1, for error messages only
    
    virtual snde_orientation3 get_orientation(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index); // parameter_index human interpreted parameter number, starting at 1, for error messages only

    virtual std::vector<snde_index> get_indexvec(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index);

    virtual std::shared_ptr<constructible_metadata> get_metadata(std::shared_ptr<recording_set_state> rss, const std::string& channel_path_context, const std::shared_ptr<math_definition>& fcn_def, size_t parameter_index);
    
    virtual std::shared_ptr<recording_base> get_recording(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index); // should only return ready recordings because we shouldn't be called until dependencies are ready // parameter_index human interpreted parameter number, starting at 1, for error messages only
    virtual std::shared_ptr<ndarray_recording_ref> get_ndarray_recording_ref(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index); // should only return ready recordings. parameter_index starting at 1, just for printing messages

    // default implementations that returns an empty set
    virtual std::set<std::string> get_prerequisites(/*std::shared_ptr<recording_set_state> rss,*/ const std::string &channel_path_context); // obtain immediate prerequisites of this parameter (absolute path channel names); typically only the recording
  };


  class math_parameter_string_const: public math_parameter {
  public:
    std::string string_constant;

    math_parameter_string_const(std::string string_constant);
    virtual std::string generate_parsible(); // generate a parsible string for Python that can be used for redefining the math function.
    virtual std::string get_string(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index);


    virtual bool operator==(const math_parameter &ref); // used for comparing parameters to instantiated_math_functions
    virtual bool operator!=(const math_parameter &ref);

    
  };


  class math_parameter_int_const: public math_parameter {
  public:
    int64_t int_constant;

    math_parameter_int_const(int64_t int_constant);
    virtual std::string generate_parsible(); // generate a parsible string for Python that can be used for redefining the math function.
    virtual int64_t get_int(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index);
    virtual snde_bool get_bool(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index); // parameter_index human interpreted parameter number, starting at 1, for error messages only

    virtual bool operator==(const math_parameter &ref); // used for comparing parameters to instantiated_math_functions
    virtual bool operator!=(const math_parameter &ref);

  };


  class math_parameter_unsigned_const: public math_parameter {
  public:
    uint64_t unsigned_constant;

    math_parameter_unsigned_const(uint64_t unsigned_constant);
    virtual std::string generate_parsible(); // generate a parsible string for Python that can be used for redefining the math function.
    virtual uint64_t get_unsigned(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index);
    virtual snde_bool get_bool(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index); // parameter_index human interpreted parameter number, starting at 1, for error messages only

    virtual bool operator==(const math_parameter &ref); // used for comparing parameters to instantiated_math_functions
    virtual bool operator!=(const math_parameter &ref);

  };

  class math_parameter_sndeindex_const : public math_parameter {
  public:
    snde_index index_constant;

    math_parameter_sndeindex_const(snde_index index);
    virtual std::string generate_parsible(); // generate a parsible string for Python that can be used for redefining the math function.
    virtual uint64_t get_unsigned(std::shared_ptr<recording_set_state> rss, const std::string& channel_path_context, const std::shared_ptr<math_definition>& fcn_def, size_t parameter_index);

    virtual bool operator==(const math_parameter& ref); // used for comparing parameters to instantiated_math_functions
    virtual bool operator!=(const math_parameter& ref);

  };

  
  class math_parameter_double_const: public math_parameter {
  public:
    double double_constant;

    math_parameter_double_const(double double_constant);
    virtual std::string generate_parsible(); // generate a parsible string for Python that can be used for redefining the math function.
    virtual double get_double(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index);

    virtual bool operator==(const math_parameter &ref); // used for comparing parameters to instantiated_math_functions
    virtual bool operator!=(const math_parameter &ref);

  };


  class math_parameter_bool_const: public math_parameter {
  public:
    snde_bool bool_constant;

    math_parameter_bool_const(snde_bool bool_constant);
    virtual std::string generate_parsible(); // generate a parsible string for Python that can be used for redefining the math function.
    virtual snde_bool get_bool(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index);

    virtual bool operator==(const math_parameter &ref); // used for comparing parameters to instantiated_math_functions
    virtual bool operator!=(const math_parameter &ref);

  };


  
  class math_parameter_vector_const: public math_parameter {
  public:
    snde_coord3 vector_constant;

    math_parameter_vector_const(const snde_coord3 vector_constant);
    virtual std::string generate_parsible(); // generate a parsible string for Python that can be used for redefining the math function.
    virtual snde_coord3 get_vector(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index);

    virtual bool operator==(const math_parameter &ref); // used for comparing parameters to instantiated_math_functions
    virtual bool operator!=(const math_parameter &ref);

  };

  class math_parameter_orientation_const: public math_parameter {
  public:
    snde_orientation3 orientation_constant;

    math_parameter_orientation_const(const snde_orientation3 orientation_constant);
    virtual std::string generate_parsible(); // generate a parsible string for Python that can be used for redefining the math function.
    virtual snde_orientation3 get_orientation(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index);

    virtual bool operator==(const math_parameter &ref); // used for comparing parameters to instantiated_math_functions
    virtual bool operator!=(const math_parameter &ref);

  };

  class math_parameter_indexvec_const: public math_parameter {
  public:
    std::vector<snde_index> indexvec;

    math_parameter_indexvec_const(const std::vector<snde_index> & indexvec);
    virtual std::string generate_parsible(); // generate a parsible string for Python that can be used for redefining the math function.
    virtual std::vector<snde_index> get_indexvec(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index);

    virtual bool operator==(const math_parameter &ref); // used for comparing parameters to instantiated_math_functions
    virtual bool operator!=(const math_parameter &ref);

  };


  class math_parameter_metadata_const : public math_parameter {
  public:
    std::shared_ptr<snde::constructible_metadata> metadata;

    math_parameter_metadata_const(std::shared_ptr<snde::constructible_metadata> metadata);
    virtual std::string generate_parsible(); // generate a parsible string for Python that can be used for redefining the math function.
    virtual std::shared_ptr<snde::constructible_metadata> get_metadata(std::shared_ptr<recording_set_state> rss, const std::string& channel_path_context, const std::shared_ptr<math_definition>& fcn_def, size_t parameter_index);

    virtual bool operator==(const math_parameter& ref); // used for comparing parameters to instantiated_math_functions
    virtual bool operator!=(const math_parameter& ref);

  };


  class math_parameter_recording: public math_parameter {
  public: // Can refer to either an entire recording or a single ndarray. Which is implicit; if converted to a recording_base it will show the whole thing.
    std::string channel_name; // ***!!! MUST BE COMBINED WITH channel_path_context from instantiated_math_function ***!!!
    size_t array_index;
    std::string array_name; // overrides array_index if size(array_name) > 0
    
    math_parameter_recording(std::string channel_name);
    math_parameter_recording(std::string channel_name,size_t array_index);
    math_parameter_recording(std::string channel_name,std::string array_name);

    virtual bool operator==(const math_parameter &ref); // used for comparing parameters to instantiated_math_functions
    virtual bool operator!=(const math_parameter &ref);
    virtual std::string generate_parsible(); // generate a parsible string for Python that can be used for redefining the math function.
    virtual std::shared_ptr<recording_base> get_recording(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index); // should only return ready recordings. parameter_index starting at 1, just for printing messages
    virtual std::shared_ptr<ndarray_recording_ref> get_ndarray_recording_ref(std::shared_ptr<recording_set_state> rss, const std::string &channel_path_context,const std::shared_ptr<math_definition> &fcn_def, size_t parameter_index); // should only return ready recordings. parameter_index starting at 1, just for printing messages

    virtual std::set<std::string> get_prerequisites(/*std::shared_ptr<recording_set_state> rss,*/ const std::string &channel_path_context); // obtain immediate prerequisites of this parameter (absolute path channel names); typically only the recording
    
  };

  // ***!!! Could have more classes here to implement e.g. parameters derived from metadata, expressions involving metadata, etc. 

};

#endif // RECMATH_PARAMETER_HPP

