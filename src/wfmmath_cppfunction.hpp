#ifndef SNDE_WFMMATH_CPPFUNCTION_HPP
#define SNDE_WFMMATH_CPPFUNCTION_HPP

#include <utility>
#include <typeindex>

//#if __cplusplus < 201402L
//#error This header requires C++14! (for std::index_sequence)
//#endif


#include "wfmstore.hpp"
#include "wfmmath.hpp"

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


  // The class can have a derived subclass of wfmmath's instantiated_math_function
  // with custom information for this particular c++ function.
  // Note that while instantiated_math_function is treated as immutable once
  // published, your additions don't necessarily need to be, provided that you manipulate
  // them safely (lock at the end of the locking order, conventions that limit manipulation
  // to function execution for a function with a self-dependency, etc.)


  // ***!!! Need subclass for instantiated_math_function
  // with instantiation data for the cpp math function.
  // ... Specifically, a lamdba to create the wfmmath_cppfuncexec
  // ... flags to represent raw, opencl, and cuda options
  // Alternate methods for raw, opencl, and cuda versions?
  
  // creation of data structures to return representing those options.
  // ***!!! (NO: Type of input waveforms may change. Single lambda may not be adequate?)
  
  // ***!!! Need template and helper to instantiate specializations for multiple
  // waveform types, select them on the basis of input waveform type
  //  THIS helper should be what is called in the lambda. !!!***

  class wfmmath_cppfuncexec_base : public executing_math_function {
  public:

    // executing_math_function defines these class members:
    //   std::shared_ptr<waveform_set_state> wss; // waveform set state in which we are executing
    //   std::shared_ptr<instantiated_math_function> inst;     // This attribute is immutable once published
    //   bool is_mutable;
    //   bool mdonly; 
    //   std::shared_ptr<assigned_compute_resource> compute_resource; // locked by acrd's admin lock

    // !!!*** May still need self_dependent_waveforms but perhaps not here... (moved to executing_math_function

    wfmmath_cppfuncexec_base(std::shared_ptr<waveform_set_state> wss,std::shared_ptr<instantiated_math_function> inst);

    wfmmath_cppfuncexec_base(const wfmmath_cppfuncexec_base &) = delete;
    wfmmath_cppfuncexec_base& operator=(const wfmmath_cppfuncexec_base &) = delete; 
    virtual ~wfmmath_cppfuncexec_base()=default;  // virtual destructor required so we can be subclassed

    virtual std::vector<unsigned> determine_param_types()=0;

    //virtual std::list<std::shared_ptr<compute_resource_option>> get_compute_options(); ... actually implemetnted in subclass

    //virtual std::list<std::shared_ptr<compute_resource_option>> perform_compute_options(); // calls subclass methods; !!!***

  };

  // recursive parameter tuple builder for wfmmath_cppfuncexec
  // first, declare the template
  template <typename... Rest>
  struct wmcfe_tuple_builder_helper {
    std::tuple<std::tuple<Rest...>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> wmcfe_tuple_builder(std::shared_ptr<waveform_set_state> wss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index);
  };
  
  // recursive definition
  template <typename T,typename... Rest>
  struct wmcfe_tuple_builder_helper<T,Rest...> {
    std::tuple<std::tuple<T,Rest...>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> wmcfe_tuple_builder(std::shared_ptr<waveform_set_state> wss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::tuple<T> this_tuple;
      std::tuple<Rest...> rest_tuple;
      size_t nextparam_index,endparam_index;
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam,endparam;
      std::tie(this_tuple,nextparam,nextparam_index) = wmcfe_tuple_builder_helper<T>().wmcfe_tuple_builder(wss,thisparam,end,channel_path_context,definition,thisparam_index); // call full specialization
      std::tie(rest_tuple,endparam,endparam_index) = wmcfe_tuple_builder_helper<Rest...>().wmcfe_tuple_builder(wss,nextparam,end,channel_path_context,definition,nextparam_index);
      
      return std::make_tuple(std::tuple_cat(this_tuple,rest_tuple),endparam,endparam_index);
    }
  };
  
  // full specialization for each concrete parameter type
  template <>
  struct wmcfe_tuple_builder_helper<std::string> {  
    std::tuple<std::tuple<std::string>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> wmcfe_tuple_builder(std::shared_ptr<waveform_set_state> wss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      if (thisparam==end) {
	throw snde_error("Not enough parameters provided to satisfy string parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = wmcfe_tuple_builder(wss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple((*thisparam)->get_string(wss,channel_path_context,definition,thisparam_index)),nextparam,thisparam_index+1);
    }
  };

  template <>
  struct wmcfe_tuple_builder_helper<int64_t> {
    std::tuple<std::tuple<int64_t>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> wmcfe_tuple_builder(std::shared_ptr<waveform_set_state> wss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      if (thisparam==end) {
	throw snde_error("Not enough parameters provided to satisfy integer parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = wmcfe_tuple_builder(wss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple((*thisparam)->get_int(wss,channel_path_context,definition,thisparam_index)),nextparam,thisparam_index+1);
    }
  };

  template <>
  struct wmcfe_tuple_builder_helper<double> {
    std::tuple<std::tuple<double>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> wmcfe_tuple_builder(std::shared_ptr<waveform_set_state> wss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      
      if (thisparam==end) {
	throw snde_error("Not enough parameters provided to satisfy double precision parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = wmcfe_tuple_builder(wss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple((*thisparam)->get_double(wss,channel_path_context,definition,thisparam_index)),nextparam,thisparam_index+1);
    }
  };
  
  // specialization for a waveform_base
  template <>
  struct wmcfe_tuple_builder_helper<std::shared_ptr<waveform_base>> {
    std::tuple<std::tuple<std::shared_ptr<waveform_base>>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> wmcfe_tuple_builder(std::shared_ptr<waveform_set_state> wss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      
      if (thisparam==end) {
	throw snde_error("Not enough parameters provided to satisfy waveform parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = wmcfe_tuple_builder(wss,thisparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple((*thisparam)->get_waveform(wss,channel_path_context,definition,thisparam_index)),nextparam,thisparam_index+1);
    }
  };
  
  // specialization for an ndarray_waveform
  template <>
  struct wmcfe_tuple_builder_helper<std::shared_ptr<ndarray_waveform>> {
    std::tuple<std::tuple<std::shared_ptr<ndarray_waveform>>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> wmcfe_tuple_builder(std::shared_ptr<waveform_set_state> wss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      
      if (thisparam==end) {
	throw snde_error("Not enough parameters provided to satisfy waveform parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = wmcfe_tuple_builder(wss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple(std::dynamic_pointer_cast<ndarray_waveform>((*thisparam)->get_waveform(wss,channel_path_context,definition,thisparam_index))),nextparam,thisparam_index+1);
    }
  };
    
  // partial specialization for an ntyped_waveform<T>
  template <typename T>
  struct wmcfe_tuple_builder_helper<std::shared_ptr<ndtyped_waveform<T>>> {
    std::tuple<std::tuple<std::shared_ptr<ndtyped_waveform<T>>>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> wmcfe_tuple_builder(std::shared_ptr<waveform_set_state> wss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
      std::vector<std::shared_ptr<math_parameter>>::iterator nextparam=thisparam;
      
      if (thisparam==end) {
	throw snde_error("Not enough parameters provided to satisfy waveform parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      nextparam++;
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = wmcfe_tuple_builder(wss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple(std::dynamic_pointer_cast<ndtyped_waveform<T>>((*thisparam)->get_waveform(wss,channel_path_context,definition,thisparam_index))),nextparam,thisparam_index+1);
    }
  };


  // specialization for a blank at the end, which g++ seemss to want (?)
  template <>
  struct wmcfe_tuple_builder_helper<> {
    std::tuple<std::tuple<>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> wmcfe_tuple_builder(std::shared_ptr<waveform_set_state> wss,std::vector<std::shared_ptr<math_parameter>>::iterator thisparam, std::vector<std::shared_ptr<math_parameter>>::iterator end,const std::string &channel_path_context,const std::shared_ptr<math_definition> &definition,size_t thisparam_index)
    {
    
      if (thisparam!=end) {
	throw snde_error("Too many parameters provided to satisfy integer parameter #%d of %s",(int)thisparam_index,definition->definition_command.c_str());
      }
      // return statement implements the following:
      //std::tie(this_tuple,nextparam) = wmcfe_tuple_builder(wss,firstparam,end,channel_path_context);
      return std::make_tuple(std::make_tuple(),thisparam,thisparam_index);
    }
  };

  
  template <typename... Ts>
  std::tuple<Ts...> wmcfe_get_parameters(std::shared_ptr<waveform_set_state> wss,std::shared_ptr<instantiated_math_function> inst)
  {
    // extract the parameters from the waveform_set_state, store them in parameters tuple
    std::vector<std::shared_ptr<math_parameter>>::iterator param_extract_last;
    //std::tie(parameters,param_extract_last)

    if (!inst) { // accomodate no instance; used on startup to probe available parameters
      return std::tuple<Ts...>();
    }
    
    std::tuple<std::tuple<Ts...>,std::vector<std::shared_ptr<math_parameter>>::iterator,size_t> parameters_param_extract_last = wmcfe_tuple_builder_helper<Ts...>().wmcfe_tuple_builder(wss,inst->parameters.begin(),inst->parameters.end(),inst->channel_path_context,inst->definition,1);
    
    param_extract_last = std::get<1>(parameters_param_extract_last);
    if (param_extract_last != inst->parameters.end()) {
      throw snde_error("Too many parameters provided for %s",inst->definition->definition_command.c_str());
    }
    
    return std::get<0>(parameters_param_extract_last);
  }
  
  // https://stackoverflow.com/questions/16868129/how-to-store-variadic-template-arguments  (see update from aschepler)
  // We are depending on C++14 here for std::index_sequence_for
  template <typename... Ts>
  class wfmmath_cppfuncexec: public wfmmath_cppfuncexec_base {
    // represents execution of a c++ function
    // derive your implementation from this templated base class
    // Instantiate the template according to your function arguments
    // e.g.
    // class multiply_by_scalar: public wfmmath_cppfuncexec_base<ndtyped_waveform<float>,float> {};
    // or even as a template
    // template <typename T> class multiply_by_scalar: public wfmmath_cppfuncexec_base<ndtyped_waveform<T>,float> {};

    // ***!!! Because we will be deriving classes from this class, any code
    // in here has to be very careful, because since templates can't be virtual
    // any function we call in this code will only see these methods, not
    // behavior overridden the the derived class. 
    
  public:

    std::tuple<Ts...> parameters;
      
    wfmmath_cppfuncexec(std::shared_ptr<waveform_set_state> wss,std::shared_ptr<instantiated_math_function> inst) :
      wfmmath_cppfuncexec_base(wss,inst),
      compute_options_function(nullptr),
      define_wfms_function(nullptr),
      metadata_function(nullptr),
      lock_alloc_function(nullptr),
      exec_function(nullptr),
      parameters(wmcfe_get_parameters<Ts...>(wss,inst))
    {
      
    }

    // rule of 3
    wfmmath_cppfuncexec(const wfmmath_cppfuncexec &) = delete;
    wfmmath_cppfuncexec & operator = (const wfmmath_cppfuncexec &) = delete;
    virtual ~wfmmath_cppfuncexec()=default;

    typedef std::function<void(Ts&...)> exec_function_type;
    typedef std::function<std::shared_ptr<exec_function_type>(Ts&...)> lock_alloc_function_type; 
    typedef std::function<std::shared_ptr<lock_alloc_function_type>(Ts&...)> metadata_function_type;
    typedef std::function<std::shared_ptr<metadata_function_type>(Ts&...)> define_wfms_function_type;
    typedef std::function<std::pair<std::list<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_wfms_function_type>>(Ts&...)> compute_options_function_type; 
    typedef std::function<std::pair<bool,std::shared_ptr<compute_options_function_type>>(Ts&...)> decide_execution_function_type; 
    
    // The function pointers stored here (if they are valid) override the methods below.
    // This way decide_execution_function can return a metadata_function which returns a
    // lock_alloc_function, which returns an exec_function all constructed in a chain of lambdas
    // such that captured parameters can persist throughout the chain. It's OK for the
    // returned functions to be empty, in which case the class methods will be used. 
    //decide_execution_function_type decide_execution_function;
    // Don't need a pointer for decide_execution because there's nothing that
    // could cause it to be overriden.
    // decide_execution in any case is only used if new_revision_optional set in the math_function
    // !!!*** If adding more function pointers here, be sure to initialize them  to nullptr in the constructor !!!***
    std::shared_ptr<compute_options_function_type> compute_options_function;
    std::shared_ptr<define_wfms_function_type> define_wfms_function;
    std::shared_ptr<metadata_function_type> metadata_function;
    std::shared_ptr<lock_alloc_function_type> lock_alloc_function;
    std::shared_ptr<exec_function_type> exec_function;

    
    
    virtual std::vector<unsigned> determine_param_types()
    {
      return std::vector<unsigned>({ wtn_fromtype<Ts>()... }); // NOTE: If you get an exception thrown at this line, it probably means that one of the parameters to your math function 
    }
	
	
    
    // NOTE: any captured variables passed to lock_alloc_function should be "smart" so that they don't leak if subsequent lambdas are never called because we returned false
    // NOTE: If you choose to override decide_execution, the decision should be made
    // quickly without going through the full calculations. 
    virtual std::pair<bool,std::shared_ptr<compute_options_function_type>> decide_execution(Ts...) // only used if new_revision_optional set in the math_function
    {
      // default implementation returns true and null compute_options method
      return std::make_pair(true,nullptr);
    }

    // call decide_execution, passing parameters from tuple (see stackoverflow link, above)
    template <std::size_t... Indexes>
    std::pair<bool,std::shared_ptr<compute_options_function_type>> call_decide_execution(std::tuple<Ts...>& tup,std::index_sequence<Indexes...>)
    {
      return decide_execution(std::get<Indexes>(tup)...);
    }
    
    template <std::size_t... Indexes>
    std::pair<bool,std::shared_ptr<compute_options_function_type>> call_decide_execution(std::tuple<Ts...>& tup)
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
    virtual std::pair<std::list<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_wfms_function_type>> compute_options(Ts...)
    {
      std::list<std::shared_ptr<compute_resource_option>> option_list = { std::make_shared<compute_resource_option_cpu>(SNDE_CR_CPU,0,0,nullptr,0,1,1) };
      return std::make_pair(option_list,nullptr);
    }

    // call compute_options, passing parameters from tuple (see stackoverflow link, above)
    template <std::size_t... Indexes>
    std::pair<std::list<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_wfms_function_type>> call_compute_options(std::tuple<Ts...>& tup,std::index_sequence<Indexes...>)
    {
      if (compute_options_function) {
	return (*compute_options_function)(std::get<Indexes>(tup)...);

      } else {
	return compute_options(std::get<Indexes>(tup)...);
      }
    }
    
    template <std::size_t... Indexes>
    std::pair<std::list<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_wfms_function_type>> call_compute_options(std::tuple<Ts...>& tup)
    {
      return call_compute_options(tup,std::index_sequence_for<Ts...>{});
    }
        
    virtual std::list<std::shared_ptr<compute_resource_option>> perform_compute_options()
    {
      std::list<std::shared_ptr<compute_resource_option>> opts;
      std::tie(opts,define_wfms_function)=call_compute_options(parameters);

      return opts;
    }


    virtual std::shared_ptr<metadata_function_type> define_wfms(Ts...)
    {
      return nullptr;
    }

    
    // call define_wfms, passing parameters from tuple (see stackoverflow link, above)
    template <std::size_t... Indexes>
    std::shared_ptr<metadata_function_type> call_define_wfms(std::tuple<Ts...>& tup,std::index_sequence<Indexes...>)
    {
      if (define_wfms_function) {
	return (*define_wfms_function)(std::get<Indexes>(tup)...);

      } else {
	return define_wfms(std::get<Indexes>(tup)...);
      }
    }
    
    template <std::size_t... Indexes>
    std::shared_ptr<metadata_function_type> call_define_wfms(std::tuple<Ts...>& tup)
    {
      return call_define_wfms(tup,std::index_sequence_for<Ts...>{});
    }
        
    virtual void perform_define_wfms()
    {
      metadata_function=call_define_wfms(parameters);

    }


    
    virtual std::shared_ptr<lock_alloc_function_type> metadata(Ts...)
    // NOTE: Your metadata implementation is only required to actually
    // set all metadata if the function is mdonly. If you do
    // set all metadata you should call the mark_metadata_done
    // on all output waveforms
    {
      //// default implementation returns lock_alloc method
      //return std::make_shared<lock_alloc_function_type>([ this ](Ts&... ts) -> std::shared_ptr<exec_function_type> {
      //return lock_alloc(ts...);
      //});
      return nullptr;
    }
    

    // call metadata, passing parameters from tuple (see stackoverflow link, above)
    template <std::size_t... Indexes>
    std::shared_ptr<lock_alloc_function_type> call_metadata(std::tuple<Ts...>& tup,std::index_sequence<Indexes...>)
    {
      if (metadata_function) {
	return (*metadata_function)(std::get<Indexes>(tup)...);

      } else {
	return metadata(std::get<Indexes>(tup)...);
      }
    }
    
    template <std::size_t... Indexes>
    std::shared_ptr<lock_alloc_function_type> call_metadata(std::tuple<Ts...>& tup)
    {
      return call_metadata(tup,std::index_sequence_for<Ts...>{});
    }
        
    virtual void perform_metadata()
    {
      lock_alloc_function=call_metadata(parameters);

    }


    
    // don't override if you implement decide_execution() and return a suitable lock_alloc() from that.
    virtual std::shared_ptr<exec_function_type>lock_alloc(Ts... ts) {
      throw snde_error("lock_alloc method must be provided or returned from metadata function");

    }

    // call lock_alloc, passing parameters from tuple (see stackoverflow link, above)
    template <std::size_t... Indexes>
    std::shared_ptr<exec_function_type> call_lock_alloc(std::tuple<Ts...>& tup,std::index_sequence<Indexes...>)
    {
      if (lock_alloc_function) {
	return (*lock_alloc_function)(std::get<Indexes>(tup)...);

      } else {
	return lock_alloc(std::get<Indexes>(tup)...);
      }
    }
    
    template <std::size_t... Indexes>
    std::shared_ptr<exec_function_type> call_lock_alloc(std::tuple<Ts...>& tup)
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
	(*exec_function)(std::get<Indexes>(tup)...);

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
  template <template<class> class CppFuncClass>
  std::shared_ptr<executing_math_function> make_cppfuncexec_floatingtypes(std::shared_ptr<waveform_set_state> wss,std::shared_ptr<instantiated_math_function> inst)
  {
    if (!inst) {
      // initial call with no instantiation to probe parameters; just use float32 case
      return std::make_shared<CppFuncClass<snde_float32>>(wss,inst);

    }
    
    std::shared_ptr<math_parameter> firstparam = inst->parameters.at(0);

    assert(firstparam->paramtype==SNDE_MFPT_WAVEFORM);

    std::shared_ptr<math_parameter_waveform> firstparam_wfm = std::dynamic_pointer_cast<math_parameter_waveform>(firstparam);

    assert(firstparam_wfm);
    
    std::shared_ptr<ndarray_waveform> firstparam_wfm_val = std::dynamic_pointer_cast<ndarray_waveform>(firstparam_wfm->get_waveform(wss,inst->channel_path_context,inst->definition,1));

    if (!firstparam_wfm_val) {
      throw snde_error("In attempting to call math function %s, first parameter %s is not an ndarray waveform",inst->definition->definition_command.c_str(),firstparam_wfm->channel_name.c_str());
    }

    switch (firstparam_wfm_val->ndinfo()->typenum) {
    case SNDE_WTN_FLOAT32:
      return std::make_shared<CppFuncClass<snde_float32>>(wss,inst);

    case SNDE_WTN_FLOAT64:
      return std::make_shared<CppFuncClass<snde_float64>>(wss,inst);

#ifdef SNDE_HAVE_FLOAT16
    case SNDE_WTN_FLOAT16:
      return std::make_shared<CppFuncClass<snde_float16>>(wss,inst);    
#endif
      
    default:
      throw snde_error("In attempting to call math function %s, first parameter %s does not have floating point type %s",inst->definition->definition_command.c_str(),firstparam_wfm->channel_name.c_str(),wtn_typenamemap.at(firstparam_wfm_val->ndinfo()->typenum).c_str());
    }
  }
  

  class cpp_math_function: public math_function {
  public:
    bool supports_cpu;
    bool supports_opencl;
    bool supports_cuda;
    cpp_math_function(size_t num_results,
		      std::function<std::shared_ptr<executing_math_function>(std::shared_ptr<waveform_set_state> wss,std::shared_ptr<instantiated_math_function> instantiated)> initiate_execution,
		      bool supports_cpu,
		      bool supports_opencl,
		      bool supports_cuda);

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
								    std::string extra_params);
    
    // initiate_execution is now a function pointer member of our superclass
    //virtual std::shared_ptr<executing_math_function> initiate_execution(std::shared_ptr<waveform_set_state> wss,std::shared_ptr<instantiated_math_function> instantiated); // actually returns pointer to class wfmmath_cppfuncexec<...>

    // !!!*** How to concisely extract parameter types from template instantiated by
    // initiate_execution?
    // Idea: Have it construct a structure with wss and instantiated set to nullptr,
    // then interrogate that. 
  };
  
  
  class instantiated_cpp_math_function: public instantiated_math_function {
  public:
    bool enable_cpu;
    bool enable_opencl;
    bool enable_cuda;

    instantiated_cpp_math_function(const std::vector<std::shared_ptr<math_parameter>> & parameters,
				   const std::vector<std::shared_ptr<std::string>> & result_channel_paths,
				   std::string channel_path_context,
				   bool is_mutable,
				   bool ondemand,
				   bool mdonly,
				   std::shared_ptr<math_function> fcn,
				   std::shared_ptr<math_definition> definition,
				   bool enable_cpu,bool enable_opencl,bool enable_cuda);
    
    // rule of 3
    instantiated_cpp_math_function(const instantiated_cpp_math_function &)=default; // for use in clone() method
    instantiated_cpp_math_function& operator=(const instantiated_cpp_math_function &) = delete; 
    virtual ~instantiated_cpp_math_function()=default;  // virtual destructor required so we can be subclassed
    
    virtual std::shared_ptr<instantiated_math_function> clone(bool definition_change=true); // only clone with definition_change=false for enable/disable of the function
  };


};

  

#endif // SNDE_WFMMATH_CPPFUNCTION_HPP
