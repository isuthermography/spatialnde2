#include "snde/recmath_cppfunction.hpp"

namespace snde {
  
  recmath_cppfuncexec_base::recmath_cppfuncexec_base(std::shared_ptr<recording_set_state> wss,std::shared_ptr<instantiated_math_function> inst) :
    executing_math_function(wss,inst)
  {
    
  }



  cpp_math_function::cpp_math_function(std::function<std::shared_ptr<executing_math_function>(std::shared_ptr<recording_set_state> wss,std::shared_ptr<instantiated_math_function> instantiated)> initiate_execution,
				       bool supports_cpu,
				       bool supports_opencl,
				       bool supports_cuda) :
    math_function(std::list<std::tuple<std::string,unsigned>>(),initiate_execution),
    supports_cpu(supports_cpu),
    supports_opencl(supports_opencl),
    supports_cuda(supports_cuda)
  {
    // perform test creation of a recmath_cppfuncexec_base to extract the parameter list
    std::shared_ptr<recmath_cppfuncexec_base> testexec = std::dynamic_pointer_cast<recmath_cppfuncexec_base>(initiate_execution(nullptr,nullptr));

    // Get vector of param type numbers from 
    std::vector<unsigned> param_types_vec = testexec->determine_param_types();

    for (size_t paramnum=0; paramnum < param_types_vec.size();paramnum++) {

      unsigned paramtype = param_types_vec.at(paramnum);
      if (paramtype != SNDE_RTN_STRING && paramtype != SNDE_RTN_INT64 && paramtype != SNDE_RTN_FLOAT64 && paramtype != SNDE_RTN_RECORDING && paramtype != SNDE_RTN_RECORDING_REF) {
	throw snde_error("Type %s is not supported as a math function parameter",rtn_typenamemap.at(paramtype).c_str());
      }
      param_names_types.emplace_back(ssprintf("Param%d",paramnum+1),paramtype);
      
    }

  }


  std::shared_ptr<instantiated_math_function> cpp_math_function::instantiate(const std::vector<std::shared_ptr<math_parameter>> & parameters,
									     const std::vector<std::shared_ptr<std::string>> & result_channel_paths,
									     std::string channel_path_context,
									     bool is_mutable,
									     bool ondemand,
									     bool mdonly,
									     std::shared_ptr<math_definition> definition,
									     std::shared_ptr<math_instance_parameter> extra_params) 
  {
    //std::shared_ptr<cpp_math_function> cpp_fcn=std::dynamic_pointer_cast<cpp_math_function>(fcn);
    return std::make_shared<instantiated_cpp_math_function>(parameters,
							    result_channel_paths,
							    channel_path_context,
							    is_mutable,
							    ondemand,
							    mdonly,
							    shared_from_this(),
							    definition,
							    extra_params,
							    supports_cpu,
							    supports_opencl,
							    supports_cuda); // so far just enable everything supported by the underlying function
  }
  
  instantiated_cpp_math_function::instantiated_cpp_math_function(const std::vector<std::shared_ptr<math_parameter>> & parameters,
								 const std::vector<std::shared_ptr<std::string>> & result_channel_paths,
								 std::string channel_path_context,
								 bool is_mutable,
								 bool ondemand,
								 bool mdonly,
								 std::shared_ptr<math_function> fcn,
								 std::shared_ptr<math_definition> definition,
								 std::shared_ptr<math_instance_parameter> extra_params,
								 bool enable_cpu,bool enable_opencl,bool enable_cuda) :
    instantiated_math_function(parameters,result_channel_paths,channel_path_context,is_mutable,ondemand,mdonly,fcn,definition,extra_params),
    enable_cpu(enable_cpu),
    enable_opencl(enable_opencl),
    enable_cuda(enable_cuda)
  {
    
  }

  
  std::shared_ptr<instantiated_math_function> instantiated_cpp_math_function::clone(bool definition_change) // only clone with definition_change=false for enable/disable of the function
  {
    std::shared_ptr<instantiated_cpp_math_function> copy = std::make_shared<instantiated_cpp_math_function>(*this);

    // see comment at start of definition of class instantiated_math_function
    if (definition_change && definition) {
      assert(!copy->original_function);
      copy->original_function = shared_from_this();
      copy->definition = nullptr; 
    }
    return copy;
    
  }
  
  
};
