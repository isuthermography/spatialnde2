#ifndef SNDE_RECMATH_CPPFUNCTION_HPP
#define SNDE_RECMATH_CPPFUNCTION_HPP

#include <utility>
#include <typeindex>
#include <variant>
#include <optional>

//#if __cplusplus < 201402L
//#error This header requires C++14! (for std::index_sequence)
//#endif

#include "snde/metadata.hpp"
#include "snde/recstore.hpp"
#include "snde/recmath.hpp"

namespace snde {
  // The problem of how to define a function can be thought of as having several
  // "axes", and the solution/API must successfully consider all of them
  //  * Axis 1: Variable arguments: API must accommodate functions with different
  //    numbers and types of parameters, and present a reasonably notation
  //    to the programmer for how such functions are written
  //  * Axis 2: Multiple phases: API must accommodate multiple phases of the
  //    the calculation: Deciding whether to instantiate a new revision; locking
  //    and/or allocating storage; executing the calculation. The intermediates
  //    of one phase need to be accessible to subsequent phases. 
  //  * Axis 3: Multiple compute environments: locking/allocation and execution
  //    may vary according to the (runtime) determination of compute environment
  //    (CPU,OpenCL,CUDA, MPI, etc.)
  //  * Axis 4: Parameter type templating. Should be possible to automatically
  //    generate and run-time select template instantiation for certain
  //    parameter type alternatives (e.g. float vs. double) and possibly
  //    result type alternatives as well.


  // Solution:
  // ---------
  // A base class from which you derive your function class.
  // The class has methods for the different phases; the base class
  // provides default implementations that may just throw an error.
  // The base class also provides storage for various extrinsic
  // parameters of the calculation, as well as implicit self-dependencies
  // and the like. 
  //
  // Each of these methods returns a pair<result_tuple,next_phase_function>
  // where next_phase_function is generally a lambda to implement the
  // next phase of the calculation. The returned subsequent phase overrides
  // any explicit class method. 
  
  // A variadic template is used to instantiate your class given
  // a set of parameters. 


  // The class can have a derived subclass of recmath's instantiated_math_function
  // with custom information for this particular c++ function.
  // Note that while instantiated_math_function is treated as immutable once
  // published, your additions don't necessarily need to be, provided that you manipulate
  // them safely (lock at the end of the locking order, conventions that limit manipulation
  // to function execution for a function with a self-dependency, etc.)

  
  // Concrete description of how this all works:
  // -------------------------------------------
  // C++ definable math functions (class cpp_math_function, derived from class math_function)
  // are defined by a lambda which instantiates a function-specific class (which must be derived
  // from class executing_math_function) given a recording_set_state (such as a globalrevision)
  // and an instantiated_math_function.
  //
  // The function-specific class is derived from recmath_cppfuncexec<types...> which is derived
  // from recmath_cppfuncexec_base, in turn derived from executing_math_function.
  // The C++ math function class is only instantiated (by calling the aforementioned lambda)
  // once the parameters are available and defined by the recording_set_state.
  //
  // As mentioned, the C++ math function class is derived from the recmath_cppfuncexec<types...>
  // template class. The template parameters are the types of the parameters the function
  // expects (parameter signature). In recmath_test.cpp, multiply_by scalar derives from:
  //   recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<snde_float32>>,snde_float64>
  // indicating that the function expects a reference to a recording of snde_float32 values as
  // its first parameter, and a single snde_float64 value as its second parameter. 
  //
  // Instantiating the function-specific class is the responsibility of the lambda passed
  // to the cpp_math_function constructor. So long as there is only one non-template function-
  // specific class, this is a trivial instantiation as illustrated in recmath_test.cpp.
  // However the lambda can also look at the values of the parameters in the
  // recording_set_state and use those to select different template instantiations or even
  // different classes entirely.
  //
  // An example of selecting different template instantiations is shown in recmath_test2, with
  // its templated multiply_by_scalar class. In this case, the lambda uses the
  // make_cppfuncexec_floatingtypes<T> template function to choose a template instantiation
  // for the single-template-parameter multiply_by_scalar class from the various floating point
  // types (snde_float32, snde_float64, and snde_float16 (if enabled))
  // according to the actual data type of the first parameter.
  //
  // The function parameters are extracted on construction of the function-specific class
  // via the recmath_cppfuncexec<Ts...> constructor by the rmcfe_get_parameters<Ts...>()
  // parameter builder which calls the rmcfe_tuple_builder_helper<> recursive templates
  // to construct the variadic template "parameters" tuple of the function-specific class's
  // recmath_cppfuncexec<Ts...> base class. The rmcfe_tuple_builder_helper<> has
  // specializations for the various types which might plausibly be parameters in the
  // function parameter signature, with corresponding code to extract the parameter
  // values from the function definition (instantiated_math_function) or defined
  // recordings (recording_set_state). 
  
  
  // ***!!! Need subclass for instantiated_math_function
  // with instantiation data for the cpp math function.
  // ... Specifically, a lamdba to create the recmath_cppfuncexec
  // ... flags to represent raw, opencl, and cuda options
  // Alternate methods for raw, opencl, and cuda versions?
  
  // creation of data structures to return representing those options.
  // ***!!! (NO: Type of input recordings may change. Single lambda may not be adequate?)
  
  // ***!!! Need template and helper to instantiate specializations for multiple
  // recording types, select them on the basis of input recording type
  //  THIS helper should be what is called in the lambda. !!!***

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

  // recursive parameter tuple builder for recmath_cppfuncexec
  // first, declare the template

  // NOTE: If you get a linker error: undefined reference to `snde::rmcfe_tuple_builder_helper<>::rmcfe_tuple_builder(...
  // Then it probably means you are defining a math function which
  // takes a parameter for which there is no matching template
  // in the full specialization list (below)
  template <typename... Rest>
  struct rmcfe_tuple_builder_helper {
    std::tuple<std::tuple<Rest...>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index);
  };
  
  // recursive definition
  template <typename T,typename... Rest>
  struct rmcfe_tuple_builder_helper<T,Rest...> {
    std::tuple<std::tuple<T,Rest...>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::tuple<T> this_tuple;
      std::tuple<Rest...> rest_tuple;
      size_t nextparam_index,endparam_index;
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam,endparam;
      std::tie(this_tuple,nextparam,nextparam_index) = rmcfe_tuple_builder_helper<T>().rmcfe_tuple_builder(rss,thisparam,end,channel_path_context,definition,thisparam_index); // call full specialization **!!! NOTE: If you get a segmentation fault on this line, it usually means there is a math function using a parameter type that isn't supported by one of the specializations below !!!***
      std::tie(rest_tuple,endparam,endparam_index) = rmcfe_tuple_builder_helper<Rest...>().rmcfe_tuple_builder(rss,nextparam,end,channel_path_context,definition,nextparam_index);
      
      return std::make_tuple(std::tuple_cat(this_tuple,rest_tuple),endparam,endparam_index);
    }
  };
  
  // full specialization for each concrete parameter type
  template <>
  struct rmcfe_tuple_builder_helper<std::string> {  
    std::tuple<std::tuple<std::string>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      if (thisparam==end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy string parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple((*thisparam)->get_string(rss,channel_path_context,definition,thisparam_index)),nextparam,thisparam_index+1);
    }
  };

  template <>
  struct rmcfe_tuple_builder_helper<int64_t> {
    std::tuple<std::tuple<int64_t>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      if (thisparam==end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy integer parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple((*thisparam)->get_int(rss,channel_path_context,definition,thisparam_index)),nextparam,thisparam_index+1);
    }
  };


  template <>
  struct rmcfe_tuple_builder_helper<int> {
    std::tuple<std::tuple<int>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      if (thisparam==end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy integer parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple((*thisparam)->get_int(rss,channel_path_context,definition,thisparam_index)),nextparam,thisparam_index+1);
    }
  };

  // Note -- the below templates have been set up to capture all scenarios for the definition of snde_index and uint64_t which may conflict with eachother

  template <>
  struct rmcfe_tuple_builder_helper<unsigned long> {
    std::tuple<std::tuple<unsigned long>, std::vector<std::shared_ptr<math_parameter>>::iterator, size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss, std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end, const std::string& channel_path_context, const std::shared_ptr<math_definition>& definition, size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam = thisparam;
      if (thisparam == end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy unsigned integer parameter #%d of %s", (int)thisparam_index, definition->definition_command.c_str());
      }
      nextparam++;
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple((*thisparam)->get_unsigned(rss, channel_path_context, definition, thisparam_index)), nextparam, thisparam_index + 1);
    }
  };


    template <>
  struct rmcfe_tuple_builder_helper<unsigned long long> {
    std::tuple<std::tuple<unsigned long long>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      if (thisparam==end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy unsigned integer parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple((*thisparam)->get_unsigned(rss,channel_path_context,definition,thisparam_index)),nextparam,thisparam_index+1);
    }
  };


  template <>
  struct rmcfe_tuple_builder_helper<unsigned> {
    std::tuple<std::tuple<unsigned>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      if (thisparam==end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy unsigned integer parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple((*thisparam)->get_unsigned(rss,channel_path_context,definition,thisparam_index)),nextparam,thisparam_index+1);
    }
  };

  
  template <>
  struct rmcfe_tuple_builder_helper<double> {
    std::tuple<std::tuple<double>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      
      if (thisparam==end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy double precision parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple((*thisparam)->get_double(rss,channel_path_context,definition,thisparam_index)),nextparam,thisparam_index+1);
    }
  };

  template <>
  struct rmcfe_tuple_builder_helper<float> {
    std::tuple<std::tuple<float>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      
      if (thisparam==end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy double precision parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple((*thisparam)->get_double(rss,channel_path_context,definition,thisparam_index)),nextparam,thisparam_index+1);
    }
  };

  // Specialization for an snde_bool

  template <>
  struct rmcfe_tuple_builder_helper<snde_bool> {
    std::tuple<std::tuple<snde_bool>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      
      if (thisparam==end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy snde_bool parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple((*thisparam)->get_bool(rss,channel_path_context,definition,thisparam_index)),nextparam,thisparam_index+1);
    }
  };



  
  // specialzation for a vector snde_coord3
  
  template <>
  struct rmcfe_tuple_builder_helper<snde_coord3> {
    std::tuple<std::tuple<snde_coord3>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      
      if (thisparam==end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy vector parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple((*thisparam)->get_vector(rss,channel_path_context,definition,thisparam_index)),nextparam,thisparam_index+1);
    }
  };


  // specialzation for a pose snde_orientation3  
  template <>
  struct rmcfe_tuple_builder_helper<snde_orientation3> {
    std::tuple<std::tuple<snde_orientation3>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      
      if (thisparam==end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy vector parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple((*thisparam)->get_orientation(rss,channel_path_context,definition,thisparam_index)),nextparam,thisparam_index+1);
    }
  };

  

  // specialzation for an index vector std::vector<snde_index>
  template <>
  struct rmcfe_tuple_builder_helper<std::vector<snde_index>> {
    std::tuple<std::tuple<std::vector<snde_index>>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      
      if (thisparam==end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy index vector parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple((*thisparam)->get_indexvec(rss,channel_path_context,definition,thisparam_index)),nextparam,thisparam_index+1);
    }
  };
  

  // partial specialization for a recording_base or subclasses thereof
  template <typename T> // T should be a recording_base or subclass
  struct rmcfe_tuple_builder_helper<std::shared_ptr<T>> {
    std::tuple<std::tuple<std::shared_ptr<T>>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      
      if (thisparam==end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy recording parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;

      std::shared_ptr<recording_base> this_rec = (*thisparam)->get_recording(rss,channel_path_context,definition,thisparam_index);
      std::shared_ptr<T> rec_subclass = std::dynamic_pointer_cast<T>(this_rec);
      if (!rec_subclass) {
	throw math_parameter_mismatch("Recording parameter %s relative to %s (which is a %s) is not convertible to %s",std::dynamic_pointer_cast<math_parameter_recording>(*thisparam)->channel_name.c_str(),channel_path_context.c_str(),typeid(*this_rec.get()).name(),typeid(T).name());
      }

      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple(rec_subclass),nextparam,thisparam_index+1);
    }
  };


  // partial specialization for a recording_base or subclasses thereof
  template <> // T should be a recording_base or subclass
  struct rmcfe_tuple_builder_helper<std::shared_ptr<snde::constructible_metadata>> {
    std::tuple<std::tuple<std::shared_ptr<snde::constructible_metadata>>, std::vector<std::shared_ptr<math_parameter>>::iterator, size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss, std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end, const std::string& channel_path_context, const std::shared_ptr<math_definition>& definition, size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam = thisparam;

      if (thisparam == end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy metadata parameter #%d of %s", (int)thisparam_index, definition->definition_command.c_str());
      }
      nextparam++;

      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple((*thisparam)->get_metadata(rss, channel_path_context, definition, thisparam_index)), nextparam, thisparam_index + 1);
    }
  };


  /*
  // specialization for a recording_base
  template <>
  struct rmcfe_tuple_builder_helper<std::shared_ptr<recording_base>> {
    std::tuple<std::tuple<std::shared_ptr<recording_base>>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      
      if (thisparam==end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy recording parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,thisparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple((*thisparam)->get_recording(rss,channel_path_context,definition,thisparam_index)),nextparam,thisparam_index+1);
    }
  };


  // specialization for a meshed_part_recording
  template <>
  struct rmcfe_tuple_builder_helper<std::shared_ptr<meshed_part_recording>> {
    std::tuple<std::tuple<std::shared_ptr<meshed_part_recording>>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      
      if (thisparam==end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy recording parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;

      std::shared_ptr<meshed_part_recording> mpr = std::dynamic_pointer_cast<meshed_part_recording>((*thisparam)->get_recording(rss,channel_path_context,definition,thisparam_index));
      if (!mpr) {
	throw math_parameter_mismatch("Recording parameter %s relative to %s is not convertible to a meshed_part_recording",channel_path_context.c_str(),std::dynamic_pointer_cast<math_parameter_recording>(*thisparam)->channel_name.c_str());
      }

      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple(mnr),nextparam,thisparam_index+1);
    }
  };

  
  // specialization for an multi_ndarray_recording
  template <>
  struct rmcfe_tuple_builder_helper<std::shared_ptr<multi_ndarray_recording>> {
    std::tuple<std::tuple<std::shared_ptr<multi_ndarray_recording>>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      
      if (thisparam==end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy recording parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;

      std::shared_ptr<multi_ndarray_recording> mnr = std::dynamic_pointer_cast<multi_ndarray_recording>((*thisparam)->get_recording(rss,channel_path_context,definition,thisparam_index));
      if (!mnr) {
	throw math_parameter_mismatch("Recording parameter %s relative to %s is not convertible to a multi_ndarray_recording",channel_path_context.c_str(),std::dynamic_pointer_cast<math_parameter_recording>(*thisparam)->channel_name.c_str());
      }

      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple(mnr),nextparam,thisparam_index+1);
    }
  };

  */
    // specialization for an ndarray_recording_ref
  template <>
  struct rmcfe_tuple_builder_helper<std::shared_ptr<ndarray_recording_ref>> {
    std::tuple<std::tuple<std::shared_ptr<ndarray_recording_ref>>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      
      if (thisparam==end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy recording parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;

      std::shared_ptr<ndarray_recording_ref> ret = (*thisparam)->get_ndarray_recording_ref(rss,channel_path_context,definition,thisparam_index);
      
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple(ret),nextparam,thisparam_index+1);
    }
  };


  
  // partial specialization for an ndtyped_recording_ref<T>
  template <typename T>
  struct rmcfe_tuple_builder_helper<std::shared_ptr<ndtyped_recording_ref<T>>> {
    std::tuple<std::tuple<std::shared_ptr<ndtyped_recording_ref<T>>>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;

      std::shared_ptr<ndtyped_recording_ref<T>> ret;
      
      if (thisparam==end) {
	throw math_parameter_mismatch("Not enough parameters provided to satisfy recording parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      std::shared_ptr<ndarray_recording_ref> nrr = (*thisparam)->get_ndarray_recording_ref(rss,channel_path_context,definition,thisparam_index);

      ret = std::dynamic_pointer_cast<ndtyped_recording_ref<T>>(nrr);
      if (!ret) {
	//assert(0);
	throw math_parameter_mismatch("Recording parameter %s relative to %s is not convertible to an ndtyped_recording_ref<%s>",std::dynamic_pointer_cast<math_parameter_recording>(*thisparam)->channel_name.c_str(),channel_path_context.c_str(),demangle_type_name(typeid(T).name()).c_str());
	
      }
      
      return std::make_tuple(std::make_tuple(ret),nextparam,thisparam_index+1);
    }
  };


  // specialization for a blank at the end, which g++ seems to want (?)
  /*
  template <>
  struct rmcfe_tuple_builder_helper<> {
    std::tuple<std::tuple<>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> rmcfe_tuple_builder(std::shared_ptr<recording_set_state> rss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
    
      if (thisparam!=end) {
	throw math_parameter_mismatch("Too many parameters provided to satisfy integer parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = rmcfe_tuple_builder(rss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple(),thisparam,thisparam_index);
    }
  };
  */
  
  template <typename... Ts>
  std::tuple<Ts...> rmcfe_get_parameters(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst)
  {
    // extract the parameters from the recording_set_state, store them in parameters tuple
    std::vector<std::shared_ptr<math_parameter>>::iterator param_extract_last;
    //std::tie(parameters,param_extract_last)

    if (!inst) { // accomodate no instance; used on startup to probe available parameters
      return std::tuple<Ts...>();
    }
    
    std::tuple<std::tuple<Ts...>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> parameters_param_extract_last = rmcfe_tuple_builder_helper<Ts...>().rmcfe_tuple_builder(rss,inst->parameters.begin(),inst->parameters.end(),inst->channel_path_context,inst->definition,1);
    
    param_extract_last = std::get<1>(parameters_param_extract_last);
    if (param_extract_last != inst->parameters.end()) {
      throw math_parameter_mismatch("Too many parameters provided for %s",inst->definition->definition_command.c_str());
    }
    
    return std::get<0>(parameters_param_extract_last);
  }
  
  // https://stackoverflow.com/questions/16868129/how-to-store-variadic-template-arguments  (see update from aschepler)
  // We are depending on C++14 here for std::index_sequence_for
  template <typename... Ts>
  class recmath_cppfuncexec: public recmath_cppfuncexec_base {
    // represents execution of a c++ function
    // derive your implementation from this templated base class
    // Instantiate the template according to your function arguments
    // e.g.
    // class multiply_by_scalar: public recmath_cppfuncexec_base<ndtyped_recording<float>,float> {};
    // or even as a template
    // template <typename T> class multiply_by_scalar: public recmath_cppfuncexec_base<ndtyped_recording<T>,float> {};

    // ***!!! Because we will be deriving classes from this class, any code
    // in here has to be very careful, because since templates can't be virtual
    // any function we call in this code will only see these methods, not
    // behavior overridden the the derived class. 
    
  public:

    std::tuple<Ts...> parameters;
      
    recmath_cppfuncexec(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec_base(rss,inst),
      parameters(rmcfe_get_parameters<Ts...>(rss,inst)),
      compute_options_function(nullptr),
      define_recs_function(nullptr),
      metadata_function(nullptr),
      lock_alloc_function(nullptr),
      exec_function(nullptr)
    {
      
    }

    // rule of 3
    recmath_cppfuncexec(const recmath_cppfuncexec &) = delete;
    recmath_cppfuncexec & operator = (const recmath_cppfuncexec &) = delete;
    virtual ~recmath_cppfuncexec()=default;

    typedef std::function<void()> exec_function_override_type;
    typedef std::function<std::shared_ptr<exec_function_override_type>()> lock_alloc_function_override_type; 
    typedef std::function<std::shared_ptr<lock_alloc_function_override_type>()> metadata_function_override_type;
    typedef std::function<std::shared_ptr<metadata_function_override_type>()> define_recs_function_override_type;
    typedef std::function<std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>>()> compute_options_function_override_type; 
    typedef std::function<std::pair<bool,std::shared_ptr<compute_options_function_override_type>>()> decide_execution_function_override_type; 
    
    // The function pointers stored here (if they are valid) override the methods below.
    // This way decide_execution_function can return a metadata_function which returns a
    // lock_alloc_function, which returns an exec_function all constructed in a chain of lambdas
    // such that captured parameters can persist throughout the chain. It's OK for the
    // returned functions to be empty, in which case the class methods will be used. 
    //decide_execution_function_override_type decide_execution_function;
    // Don't need a pointer for decide_execution because there's nothing that
    // could cause it to be overriden.
    // decide_execution in any case is only used if new_revision_optional set in the math_function
    // !!!*** If adding more function pointers here, be sure to initialize them  to nullptr in the constructor !!!***
    std::shared_ptr<compute_options_function_override_type> compute_options_function;
    std::shared_ptr<define_recs_function_override_type> define_recs_function;
    std::shared_ptr<metadata_function_override_type> metadata_function;
    std::shared_ptr<lock_alloc_function_override_type> lock_alloc_function;
    std::shared_ptr<exec_function_override_type> exec_function;

    
    
    virtual std::vector<unsigned> determine_param_types()
    {
      return std::vector<unsigned>({ rtn_fromtype<Ts>()... }); // NOTE: If you get an exception thrown at this line, it probably means that one of the parameters to your math function is not a type in the typemaps list, or a recording or recording reference
    }
	
	
    
    // NOTE: any captured variables passed by decide_execution to lock_alloc_function should be "smart" so that they don't leak if subsequent lambdas are never called because we returned false
    // NOTE: If you choose to override decide_execution, the decision should be made
    // quickly without going through the full calculations. 
    virtual std::pair<bool,std::shared_ptr<compute_options_function_override_type>> decide_execution(Ts...) // only used if new_revision_optional set in the math_function
    {
      // default implementation returns true and null compute_options method
      return std::make_pair(true,nullptr);
    }

    // call decide_execution, passing parameters from tuple (see stackoverflow link, above)
    template <std::size_t... Indexes>
    std::pair<bool,std::shared_ptr<compute_options_function_override_type>> call_decide_execution(std::tuple<Ts...>& tup,std::index_sequence<Indexes...>)
    {
      return decide_execution(std::get<Indexes>(tup)...);
    }
    
    template <std::size_t... Indexes>
    std::pair<bool,std::shared_ptr<compute_options_function_override_type>> call_decide_execution(std::tuple<Ts...>& tup)
    {
      return call_decide_execution(tup,std::index_sequence_for<Ts...>{});
    }
        
    virtual bool perform_decide_execution()
    {
      bool new_revision;
      std::tie(new_revision,compute_options_function)=call_decide_execution(parameters);

      return new_revision;
    }

    // likewise if you override compute_options, this one should not do much and finish quickly
    virtual std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(Ts...)
    {
      std::vector<std::shared_ptr<compute_resource_option>> option_list = { std::make_shared<compute_resource_option_cpu>(std::set<std::string>(),0,0,0.0,1,1) };
      return std::make_pair(option_list,nullptr);
    }

    // call compute_options, passing parameters from tuple (see stackoverflow link, above)
    template <std::size_t... Indexes>
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> call_compute_options(std::tuple<Ts...>& tup,std::index_sequence<Indexes...>)
    {
      if (compute_options_function) {
	return (*compute_options_function)();

      } else {
	return compute_options(std::get<Indexes>(tup)...);
      }
    }
    
    template <std::size_t... Indexes>
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> call_compute_options(std::tuple<Ts...>& tup)
    {
      return call_compute_options(tup,std::index_sequence_for<Ts...>{});
    }
        
    virtual std::vector<std::shared_ptr<compute_resource_option>> perform_compute_options()
    {
      std::vector<std::shared_ptr<compute_resource_option>> opts;
      std::tie(opts,define_recs_function)=call_compute_options(parameters);

      return opts;
    }


    virtual std::shared_ptr<metadata_function_override_type> define_recs(Ts...)
    {
      return nullptr;
    }

    
    // call define_recs, passing parameters from tuple (see stackoverflow link, above)
    template <std::size_t... Indexes>
    std::shared_ptr<metadata_function_override_type> call_define_recs(std::tuple<Ts...>& tup,std::index_sequence<Indexes...>)
    {
      if (define_recs_function) {
	return (*define_recs_function)();

      } else {
	return define_recs(std::get<Indexes>(tup)...);
      }
    }
    
    template <std::size_t... Indexes>
    std::shared_ptr<metadata_function_override_type> call_define_recs(std::tuple<Ts...>& tup)
    {
      return call_define_recs(tup,std::index_sequence_for<Ts...>{});
    }
        
    virtual void perform_define_recs()
    {
      metadata_function=call_define_recs(parameters);

    }


    
    virtual std::shared_ptr<lock_alloc_function_override_type> metadata(Ts...)
    // NOTE: Your metadata implementation is only required to actually
    // set all metadata if the function is mdonly. If you do
    // set all metadata you should call the mark_metadata_done
    // on all output recordings
    {
      //// default implementation returns lock_alloc method
      //return std::make_shared<lock_alloc_function_override_type>([ this ](Ts&... ts) -> std::shared_ptr<exec_function_override_type> {
      //return lock_alloc(ts...);
      //});
      return nullptr;
    }
    

    // call metadata, passing parameters from tuple (see stackoverflow link, above)
    template <std::size_t... Indexes>
    std::shared_ptr<lock_alloc_function_override_type> call_metadata(std::tuple<Ts...>& tup,std::index_sequence<Indexes...>)
    {
      if (metadata_function) {
	return (*metadata_function)();

      } else {
	return metadata(std::get<Indexes>(tup)...);
      }
    }
    
    template <std::size_t... Indexes>
    std::shared_ptr<lock_alloc_function_override_type> call_metadata(std::tuple<Ts...>& tup)
    {
      return call_metadata(tup,std::index_sequence_for<Ts...>{});
    }
        
    virtual void perform_metadata()
    {
      lock_alloc_function=call_metadata(parameters);

    }


    
    // don't override if you implement decide_execution() and return a suitable lock_alloc() from that.
    // NOTE: it is OK for lock_alloc to be a no-op and do your locking/allocation inline in the execution
    virtual std::shared_ptr<exec_function_override_type>lock_alloc(Ts... ts) {
      throw snde_error("lock_alloc method must be provided or returned from metadata function");

    }

    // call lock_alloc, passing parameters from tuple (see stackoverflow link, above)
    template <std::size_t... Indexes>
    std::shared_ptr<exec_function_override_type> call_lock_alloc(std::tuple<Ts...>& tup,std::index_sequence<Indexes...>)
    {
      if (lock_alloc_function) {
	return (*lock_alloc_function)();

      } else {
	return lock_alloc(std::get<Indexes>(tup)...);
      }
    }
    
    template <std::size_t... Indexes>
    std::shared_ptr<exec_function_override_type> call_lock_alloc(std::tuple<Ts...>& tup)
    {
      return call_lock_alloc(tup,std::index_sequence_for<Ts...>{});
    }
        
    virtual void perform_lock_alloc()
    {
      exec_function=call_lock_alloc(parameters);

    }


    
    // generally don't override; just return lambda from lock_alloc instead
    virtual void exec(Ts...)
    {
      throw snde_error("exec method should be overridden or returned from lock_alloc");

    }


    // call exec, passing parameters from tuple (see stackoverflow link, above)
    template <std::size_t... Indexes>
    void call_exec(std::tuple<Ts...>& tup,std::index_sequence<Indexes...>)
    {
      if (exec_function) {
	(*exec_function)();

      } else {
	exec(std::get<Indexes>(tup)...);
      }
    }
    
    template <std::size_t... Indexes>
    void call_exec(std::tuple<Ts...>& tup)
    {
      call_exec(tup,std::index_sequence_for<Ts...>{});
    }
        
    virtual void perform_exec()
    {
      call_exec(parameters);

    }

    
  };


  // This template allows you to write a math function once
  // that auto-detects whether its first input is snde_float32 or
  // snde_float64 and runs the correct version automatically
  template <template <typename...> class CppFuncClass, typename... Args>  
  std::shared_ptr<executing_math_function> make_cppfuncexec_floatingtypes(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst)
  {
    if (!inst) {
      // initial call with no instantiation to probe parameters; just use float32 case
      return std::make_shared<CppFuncClass<snde_float32,Args...>>(rss,inst);

    }
    
    std::shared_ptr<math_parameter> firstparam = inst->parameters.at(0);

    assert(firstparam->paramtype==SNDE_MFPT_RECORDING);

    std::shared_ptr<math_parameter_recording> firstparam_rec = std::dynamic_pointer_cast<math_parameter_recording>(firstparam);

    assert(firstparam_rec);
    
    std::shared_ptr<ndarray_recording_ref> firstparam_rec_val = firstparam_rec->get_ndarray_recording_ref(rss,inst->channel_path_context,inst->definition,1);

    if (!firstparam_rec_val) { // Won't ever happen because get_ndarray_recording_ref() now throws the exception itself
      throw snde_error("In attempting to call math function %s, first parameter %s is not an ndarray recording",inst->definition->definition_command.c_str(),firstparam_rec->channel_name.c_str());
    }

    switch (firstparam_rec_val->ndinfo()->typenum) {
    case SNDE_RTN_FLOAT32:
      return std::make_shared<CppFuncClass<snde_float32,Args...>>(rss,inst);

    case SNDE_RTN_FLOAT64:
      return std::make_shared<CppFuncClass<snde_float64,Args...>>(rss,inst);

#ifdef SNDE_HAVE_FLOAT16
    case SNDE_RTN_FLOAT16:
      return std::make_shared<CppFuncClass<snde_float16,Args...>>(rss,inst);
#endif
      
    default:
      //throw snde_error("In attempting to call math function %s, first parameter %s has non-floating point type %s",inst->definition->definition_command.c_str(),firstparam_rec->channel_name.c_str(),rtn_typenamemap.at(firstparam_rec_val->ndinfo()->typenum).c_str());
      return nullptr;
    }
  }



  // This template allows you to write a math function once
  // that auto-detects whether its first input is int32_t or
  // int64_t and runs the correct version automatically
  template <template <typename...> class CppFuncClass, typename... Args>
  std::shared_ptr<executing_math_function> make_cppfuncexec_integertypes(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst)
  {
    if (!inst) {
      // initial call with no instantiation to probe parameters; just use int32 case
      return std::make_shared<CppFuncClass<int32_t,Args...>>(rss,inst);

    }
    
    std::shared_ptr<math_parameter> firstparam = inst->parameters.at(0);

    assert(firstparam->paramtype==SNDE_MFPT_RECORDING);

    std::shared_ptr<math_parameter_recording> firstparam_rec = std::dynamic_pointer_cast<math_parameter_recording>(firstparam);

    assert(firstparam_rec);
    
    std::shared_ptr<ndarray_recording_ref> firstparam_rec_val = firstparam_rec->get_ndarray_recording_ref(rss,inst->channel_path_context,inst->definition,1);

    if (!firstparam_rec_val) { // Won't ever happen because get_ndarray_recording_ref() now throws the exception itself
      throw snde_error("In attempting to call math function %s, first parameter %s is not an ndarray recording",inst->definition->definition_command.c_str(),firstparam_rec->channel_name.c_str());
    }

    switch (firstparam_rec_val->ndinfo()->typenum) {
    case SNDE_RTN_UINT64:
      return std::make_shared<CppFuncClass<uint64_t, Args...>>(rss,inst);

    case SNDE_RTN_INT64:
      return std::make_shared<CppFuncClass<int64_t, Args...>>(rss,inst);

    case SNDE_RTN_UINT32:
      return std::make_shared<CppFuncClass<uint32_t, Args...>>(rss,inst);

    case SNDE_RTN_INT32:
      return std::make_shared<CppFuncClass<int32_t, Args...>>(rss,inst);

    case SNDE_RTN_UINT16:
      return std::make_shared<CppFuncClass<uint16_t, Args...>>(rss,inst);
      
    case SNDE_RTN_INT16:
      return std::make_shared<CppFuncClass<int16_t, Args...>>(rss,inst);

    case SNDE_RTN_UINT8:
      return std::make_shared<CppFuncClass<uint8_t, Args...>>(rss,inst);

    case SNDE_RTN_INT8:
      return std::make_shared<CppFuncClass<int8_t, Args...>>(rss,inst);

    default:
      //throw snde_error("In attempting to call math function %s, first parameter %s has non-floating point type %s",inst->definition->definition_command.c_str(),firstparam_rec->channel_name.c_str(),rtn_typenamemap.at(firstparam_rec_val->ndinfo()->typenum).c_str());
      return nullptr;
    }
  }


  // This template allows you to write a math function once
  // that auto-detects whether its first input is complex float32 or
  // complex float64 and runs the correct version automatically
  template <template <typename...> class CppFuncClass, typename... Args>
  std::shared_ptr<executing_math_function> make_cppfuncexec_complextypes(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst)
  {
    if (!inst) {
      // initial call with no instantiation to probe parameters; just use int32 case
      return std::make_shared<CppFuncClass<snde_complexfloat32,Args...>>(rss,inst);

    }
    
    std::shared_ptr<math_parameter> firstparam = inst->parameters.at(0);

    assert(firstparam->paramtype==SNDE_MFPT_RECORDING);

    std::shared_ptr<math_parameter_recording> firstparam_rec = std::dynamic_pointer_cast<math_parameter_recording>(firstparam);

    assert(firstparam_rec);
    
    std::shared_ptr<ndarray_recording_ref> firstparam_rec_val = firstparam_rec->get_ndarray_recording_ref(rss,inst->channel_path_context,inst->definition,1);

    if (!firstparam_rec_val) { // Won't ever happen because get_ndarray_recording_ref() now throws the exception itself
      throw snde_error("In attempting to call math function %s, first parameter %s is not an ndarray recording",inst->definition->definition_command.c_str(),firstparam_rec->channel_name.c_str());
    }

    switch (firstparam_rec_val->ndinfo()->typenum) {
    case SNDE_RTN_COMPLEXFLOAT32:
      return std::make_shared<CppFuncClass<snde_complexfloat32, Args...>>(rss,inst);

    case SNDE_RTN_COMPLEXFLOAT64:
      return std::make_shared<CppFuncClass<snde_complexfloat64, Args...>>(rss,inst);

    case SNDE_RTN_SNDE_COMPLEXIMAGEDATA:
      return std::make_shared<CppFuncClass<snde_compleximagedata, Args...>>(rss,inst);

      
    default:
      //throw snde_error("In attempting to call math function %s, first parameter %s has non-floating point type %s",inst->definition->definition_command.c_str(),firstparam_rec->channel_name.c_str(),rtn_typenamemap.at(firstparam_rec_val->ndinfo()->typenum).c_str());
      return nullptr;
    }
  }

  
  

  template <typename T>
  inline size_t cppfunc_vector_multiplicity()
  {
    throw snde_error("This function does not support vectors of type %s",typeid(T).name());
  }
  template<>
  inline size_t cppfunc_vector_multiplicity<uint8_t>()
  {
      return 1;
  }
  template<>
  inline size_t cppfunc_vector_multiplicity<uint16_t>()
  {
      return 1;
  }
  template<>
  inline size_t cppfunc_vector_multiplicity<snde_float32>()
  {
    return 1;
  }
  template<>
  inline size_t cppfunc_vector_multiplicity<snde_float64>()
  {
    return 1;
  }
  
  template<>
  inline size_t cppfunc_vector_multiplicity<snde_coord2>()
  {
    return 2;
  }
  
  template<>
  inline size_t cppfunc_vector_multiplicity<snde_coord3>()
  {
    return 3;
  }

  template <typename T> struct cppfunc_vector_underlying_type {
    typedef T underlying_type;
  };
  template <> struct cppfunc_vector_underlying_type<snde_coord3> {
    typedef snde_coord underlying_type;    
  };
  template <> struct cppfunc_vector_underlying_type<snde_coord2> {
    typedef snde_coord underlying_type;    
  };


  // This template allows you to write a math function once
  // that auto-detects whether its first input is snde_float32 or
  // snde_float64 or one of our vector types and runs the correct version automatically
  // see also the vector evaluation templates immediately above as they may be convenient
  // (see averaging_downsampler.cpp for an example)
  template <template <typename...> class CppFuncClass, typename... Args>
  std::shared_ptr<executing_math_function> make_cppfuncexec_vectortypes(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst)
  {
    if (!inst) {
      // initial call with no instantiation to probe parameters; just use float32 case
      return std::make_shared<CppFuncClass<snde_float32,Args...>>(rss,inst);

    }
    
    std::shared_ptr<math_parameter> firstparam = inst->parameters.at(0);

    assert(firstparam->paramtype==SNDE_MFPT_RECORDING);

    std::shared_ptr<math_parameter_recording> firstparam_rec = std::dynamic_pointer_cast<math_parameter_recording>(firstparam);

    assert(firstparam_rec);
    
    std::shared_ptr<ndarray_recording_ref> firstparam_rec_val = firstparam_rec->get_ndarray_recording_ref(rss,inst->channel_path_context,inst->definition,1);

    if (!firstparam_rec_val) { // Won't ever happen because get_ndarray_recording_ref() now throws the exception itself
      throw snde_error("In attempting to call math function %s, first parameter %s is not an ndarray recording",inst->definition->definition_command.c_str(),firstparam_rec->channel_name.c_str());
    }

    switch (firstparam_rec_val->ndinfo()->typenum) {
    case SNDE_RTN_SNDE_COORD2:
      return std::make_shared<CppFuncClass<snde_coord2, Args...>>(rss,inst);

    case SNDE_RTN_SNDE_COORD3:
      return std::make_shared<CppFuncClass<snde_coord3, Args...>>(rss,inst);

      
    default:
      return nullptr;
    }
  }

  // Use of variants to identify parameters so as to instantiate the correct type of a math function


  template <typename ...Args>
  struct optvariant_union;

  template <typename ...Args1,typename ...Args2>
  struct optvariant_union<std::optional<std::variant<Args1...>>,std::optional<std::variant<Args2...>>> {

    using type = std::optional<std::variant<Args1...,Args2...>>;
  };

  template <typename ...Variants>
  void variant_merge(Variants... args)
  {
    throw snde_error("variant_merge not implemented for this many parameters");
  }

  template <typename Var1,typename Var2>
  typename optvariant_union<Var1,Var2>::type variant_merge(Var1 val1,Var2 val2)
  {
    if (val1.has_value()) {
      return typename optvariant_union<Var1,Var2>::type::value_type(std::visit([] (auto &arg) -> typename optvariant_union<Var1,Var2>::type::value_type { return std::move(arg); },val1.value()));
    }
    if (val2.has_value()) {
      return typename optvariant_union<Var1,Var2>::type::value_type(std::visit([] (auto &arg) -> typename optvariant_union<Var1,Var2>::type::value_type { return std::move(arg); },val2.value()));
    }
    return typename optvariant_union<Var1,Var2>::type();
  }

  template <typename Var1,typename Var2,typename Var3>
  typename optvariant_union<typename optvariant_union<Var1,Var2>::type,Var3>::type variant_merge(Var1 val1,Var2 val2,Var3 val3)
  {
    auto val12 = variant_merge(val1,val2);
    return variant_merge(val12,val3);
    
  }
  
  using ref_float_var = std::optional<std::variant<std::shared_ptr<ndtyped_recording_ref<snde_float32>>,std::shared_ptr<ndtyped_recording_ref<snde_float64>>
#ifdef SNDE_HAVE_FLOAT16
    ,std::shared_ptr<ndtyped_recording_ref<snde_float16>>
#endif // SNDE_HAVE_FLOAT16
                                                    >>;
  using ref_signed_var = std::optional<std::variant<std::shared_ptr<ndtyped_recording_ref<int8_t>>,std::shared_ptr<ndtyped_recording_ref<int16_t>>,std::shared_ptr<ndtyped_recording_ref<int32_t>>,std::shared_ptr<ndtyped_recording_ref<int64_t>>>>;

  using ref_unsigned_var = std::optional<std::variant<std::shared_ptr<ndtyped_recording_ref<uint8_t>>,std::shared_ptr<ndtyped_recording_ref<uint16_t>>,std::shared_ptr<ndtyped_recording_ref<uint32_t>>,std::shared_ptr<ndtyped_recording_ref<uint64_t>>>>;

  using ref_integer_var = typename optvariant_union<ref_signed_var,ref_unsigned_var>::type;

  using ref_real_var = typename optvariant_union<ref_float_var,ref_integer_var>::type;
  
  template <template <typename...> class CppFuncClass,typename Arg0, typename Arg1, typename... ExtraArgs>
  std::shared_ptr<executing_math_function> make_cppfuncexec_twovariants(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst,Arg0 variant0, Arg1 variant1)
  {
    if (!inst) {
      // initial call with no instantiation to probe parameters; just use float32 case
      return std::make_shared<CppFuncClass<snde_float32,snde_float32,ExtraArgs...>>(rss,inst);

    }
    return std::visit([rss,inst](auto && variant0, auto && variant1) -> std::shared_ptr<executing_math_function> {
            using Type1 = typename std::decay_t<decltype(variant0)>::element_type::dtype;
            using Type2 = typename std::decay_t<decltype(variant1)>::element_type::dtype;
            return std::make_shared<CppFuncClass<Type1,Type2,ExtraArgs...>>(rss,inst);

          }, variant0.value(),variant1.value());

  }
  

  std::shared_ptr<ndarray_recording_ref>  math_param_ref(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst,snde_index param_num);
  
  ref_float_var math_param_ref_float(std::shared_ptr<ndarray_recording_ref> param_ref_val);

  ref_signed_var math_param_ref_signed(std::shared_ptr<ndarray_recording_ref> param_ref_val);

  ref_unsigned_var math_param_ref_unsigned(std::shared_ptr<ndarray_recording_ref> param_ref_val);

  ref_integer_var math_param_ref_integer(std::shared_ptr<ndarray_recording_ref> param_ref_val);

  // Real, meaning floating point or integer
  ref_real_var math_param_ref_real(std::shared_ptr<ndarray_recording_ref> param_ref_val);
  
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
				   //bool enable_cpu,bool enable_opencl,bool enable_cuda);
    
    // rule of 3
    instantiated_cpp_math_function(const instantiated_cpp_math_function &)=default; // for use in clone() method
    instantiated_cpp_math_function& operator=(const instantiated_cpp_math_function &) = delete; 
    virtual ~instantiated_cpp_math_function()=default;  // virtual destructor required so we can be subclassed

    // If we had any elements that would distinguish us by functionality we would have to implement comparison operators to accommodate comparisons of rendering functions
    //virtual operator==(const instantiated_math_function &ref);
    //virtual operator!=(const instantiated_math_function &ref);

    
    virtual std::shared_ptr<instantiated_math_function> clone(bool definition_change=true); // only clone with definition_change=false for enable/disable of the function
  };


};

  

#endif // SNDE_RECMATH_CPPFUNCTION_HPP
