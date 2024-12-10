%shared_ptr(snde::recmath_cppfuncexec_base);
snde_rawaccessible(snde::recmath_cppfuncexec_base);
%shared_ptr(snde::cpp_math_function);
snde_rawaccessible(snde::cpp_math_function);
%shared_ptr(snde::instantiated_cpp_math_function);
snde_rawaccessible(snde::instantiated_cpp_math_function);
  
%{
  #include "recmath_cppfunction.hpp"
%}
namespace snde {
  class recmath_cppfuncexec_base : public executing_math_function {
  public:

    // executing_math_function defines these class members:
    //   std::shared_ptr<recording_set_state> rss; // recording set state in which we are executing
    //   std::shared_ptr<instantiated_math_function> inst;     // This attribute is immutable once published
    //   bool is_mutable;
    //   bool mdonly; 
    //   std::shared_ptr<assigned_compute_resource> compute_resource; // locked by acrd's admin lock

    // !!!*** May still need self_dependent_recordings but perhaps not here... (moved to executing_math_function

    recmath_cppfuncexec_base(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst);

    recmath_cppfuncexec_base(const recmath_cppfuncexec_base &) = delete;
    recmath_cppfuncexec_base& operator=(const recmath_cppfuncexec_base &) = delete; 
    virtual ~recmath_cppfuncexec_base()=default;  // virtual destructor required so we can be subclassed

    virtual std::vector<unsigned> determine_param_types()=0;

    //virtual std::list<std::shared_ptr<compute_resource_option>> get_compute_options(); ... actually implemetnted in subclass

    //virtual std::list<std::shared_ptr<compute_resource_option>> perform_compute_options(); // calls subclass methods; !!!***

  };


  class cpp_math_function: public math_function {
  public:
    //bool supports_cpu;
    //bool supports_opencl;
    //bool supports_cuda;
    cpp_math_function(std::string function_name,size_t num_results,std::function<std::shared_ptr<executing_math_function>(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> instantiated)> initiate_execution);
    //		      bool supports_cpu,
    //		      bool supports_opencl,
    //		      bool supports_cuda);

    // Rule of 3
    cpp_math_function(const cpp_math_function &) = delete;
    cpp_math_function& operator=(const cpp_math_function &) = delete; 
    virtual ~cpp_math_function()=default;  // virtual destructor required so we can be subclassed

    virtual std::shared_ptr<instantiated_math_function> instantiate(const std::vector<std::shared_ptr<math_parameter>> & parameters,
								    const std::vector<std::shared_ptr<std::string>> & result_channel_paths,
								    std::string channel_path_context,
								    bool is_mutable,
								    bool ondemand,
								    bool mdonly,
								    std::shared_ptr<math_definition> definition,
								    std::set<std::string> execution_tags,
								    std::shared_ptr<math_instance_parameter> extra_params);
    
    // initiate_execution is now a function pointer member of our superclass
    //virtual std::shared_ptr<executing_math_function> initiate_execution(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> instantiated); // actually returns pointer to class recmath_cppfuncexec<...>

    // !!!*** How to concisely extract parameter types from template instantiated by
    // initiate_execution?
    // Idea: Have it construct a structure with rss and instantiated set to nullptr,
    // then interrogate that. 
  };
  
  
  class instantiated_cpp_math_function: public instantiated_math_function {
  public:
    //bool enable_cpu;
    //bool enable_opencl;
    //bool enable_cuda;

    instantiated_cpp_math_function(const std::vector<std::shared_ptr<math_parameter>> & parameters,
				   const std::vector<std::shared_ptr<std::string>> & result_channel_paths,
				   std::string channel_path_context,
				   bool is_mutable,
				   bool ondemand,
				   bool mdonly,
				   std::shared_ptr<math_function> fcn,
				   std::shared_ptr<math_definition> definition,
				   std::set<std::string> execution_tags,
				   std::shared_ptr<math_instance_parameter> extra_params);
    
    // rule of 3
    instantiated_cpp_math_function(const instantiated_cpp_math_function &)=default; // for use in clone() method
    instantiated_cpp_math_function& operator=(const instantiated_cpp_math_function &) = delete; 
    virtual ~instantiated_cpp_math_function()=default;  // virtual destructor required so we can be subclassed
    
    virtual std::shared_ptr<instantiated_math_function> clone(bool definition_change=true); // only clone with definition_change=false for enable/disable of the function
  };


};

