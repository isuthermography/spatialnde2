#include "snde/recmath_cppfunction.hpp"

#include "snde/recmath.hpp"
namespace snde {
  
  recmath_cppfuncexec_base::recmath_cppfuncexec_base(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
    executing_math_function(rss,inst,(rss) ? rss->lockmgr:nullptr)
  {
    
  }



  cpp_math_function::cpp_math_function(std::string function_name, size_t num_results,std::function<std::shared_ptr<executing_math_function>(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> instantiated)> initiate_execution) :
    //				       bool supports_cpu,
    //				       bool supports_opencl,
    //				       bool supports_cuda) :
    math_function(function_name,num_results,std::vector<std::pair<std::string,unsigned>>(),initiate_execution)
    //supports_cpu(supports_cpu),
    //supports_opencl(supports_opencl),
    //supports_cuda(supports_cuda)
  {
    // perform test creation of a recmath_cppfuncexec_base to extract the parameter list
    std::shared_ptr<recmath_cppfuncexec_base> testexec = std::dynamic_pointer_cast<recmath_cppfuncexec_base>(initiate_execution(nullptr,nullptr));

    // Get vector of param type numbers from 
    std::vector<unsigned> param_types_vec = testexec->determine_param_types();

    for (size_t paramnum=0; paramnum < param_types_vec.size();paramnum++) {

      unsigned paramtype = param_types_vec.at(paramnum);
      if (paramtype != SNDE_RTN_STRING && paramtype != SNDE_RTN_INT64 && paramtype != SNDE_RTN_UINT64 && paramtype != SNDE_RTN_INT32 && paramtype != SNDE_RTN_UINT32 && paramtype != SNDE_RTN_FLOAT64 && paramtype != SNDE_RTN_FLOAT32 && paramtype != SNDE_RTN_INDEXVEC && paramtype != SNDE_RTN_RECORDING && paramtype != SNDE_RTN_RECORDING_REF && paramtype != SNDE_RTN_CONSTRUCTIBLEMETADATA && paramtype != SNDE_RTN_SNDE_COORD3 && paramtype != SNDE_RTN_SNDE_ORIENTATION3 && paramtype != SNDE_RTN_SNDE_BOOL && paramtype != SNDE_RTN_UINT8) {
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
									     std::set<std::string> execution_tags,
									     std::shared_ptr<math_instance_parameter> extra_params) 
  {
    //std::shared_ptr<cpp_math_function> cpp_fcn=std::dynamic_pointer_cast<cpp_math_function>(fcn);
    // std::set<std::string> execution_tags_set;
    // for (auto && execution_tag: execution_tags) {
    // execution_tags_set.emplace(execution_tag);
    // }
    return std::make_shared<instantiated_cpp_math_function>(parameters,
							    result_channel_paths,
							    channel_path_context,
							    is_mutable,
							    ondemand,
							    mdonly,
							    shared_from_this(),
							    definition,
							    execution_tags,
							    extra_params);
    //							    supports_cpu,
    //							    supports_opencl,
    //							    supports_cuda); // so far just enable everything supported by the underlying function
  }
  
  instantiated_cpp_math_function::instantiated_cpp_math_function(const std::vector<std::shared_ptr<math_parameter>> & parameters,
								 const std::vector<std::shared_ptr<std::string>> & result_channel_paths,
								 std::string channel_path_context,
								 bool is_mutable,
								 bool ondemand,
								 bool mdonly,
								 std::shared_ptr<math_function> fcn,
								 std::shared_ptr<math_definition> definition,
								 std::set<std::string> execution_tags,
								 std::shared_ptr<math_instance_parameter> extra_params) :
								 //bool enable_cpu,bool enable_opencl,bool enable_cuda) :
    instantiated_math_function(parameters,result_channel_paths,channel_path_context,is_mutable,ondemand,mdonly,fcn,definition,execution_tags,extra_params)
    //    enable_cpu(enable_cpu),
    //    enable_opencl(enable_opencl),
    //    enable_cuda(enable_cuda)
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
  
  std::shared_ptr<ndarray_recording_ref>  math_param_ref(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst,snde_index param_num)
  {
    if (param_num >= inst->parameters.size()) {
      throw snde_error("Insufficient parameters for index %u",(unsigned)param_num);

    }
    std::shared_ptr<math_parameter> param = inst->parameters.at(param_num);

    if (param->paramtype!=SNDE_MFPT_RECORDING) {
      throw math_parameter_mismatch("Parameter %u is not a recording",(unsigned)(param_num+1));
    }

    std::shared_ptr<math_parameter_recording> param_rec = std::dynamic_pointer_cast<math_parameter_recording>(param);
    assert(param_rec);
    
    std::shared_ptr<ndarray_recording_ref> param_ref_val = param_rec->get_ndarray_recording_ref(rss,inst->channel_path_context,inst->definition,param_num+1);
    return param_ref_val;
  }
  
  ref_float_var math_param_ref_float(std::shared_ptr<ndarray_recording_ref> param_ref_val)
  {
    assert(param_ref_val); // Won't ever happen because get_ndarray_recording_ref() now throws the exception itself
    switch (param_ref_val->ndinfo()->typenum) {
    case SNDE_RTN_FLOAT32:
      return ref_float_var(param_ref_val->reference_typed_ndarray<snde_float32>());

    case SNDE_RTN_FLOAT64:
      return ref_float_var(param_ref_val->reference_typed_ndarray<snde_float64>());
                           

#ifdef SNDE_HAVE_FLOAT16
    case SNDE_RTN_FLOAT16:
      return ref_float_var(param_ref_val->reference_typed_ndarray<snde_float32>()); 
#endif
      
    }
    return ref_float_var();
  }
  
  ref_signed_var math_param_ref_signed(std::shared_ptr<ndarray_recording_ref> param_ref_val)
  {
    assert(param_ref_val); // Won't ever happen because get_ndarray_recording_ref() now throws the exception itself
    switch (param_ref_val->ndinfo()->typenum) {
    case SNDE_RTN_INT8:
      return ref_signed_var(param_ref_val->reference_typed_ndarray<int8_t>());

    case SNDE_RTN_INT16:
      return ref_signed_var(param_ref_val->reference_typed_ndarray<int16_t>());

    case SNDE_RTN_INT32:
      return ref_signed_var(param_ref_val->reference_typed_ndarray<int32_t>());
                           

    case SNDE_RTN_INT64:
      return ref_signed_var(param_ref_val->reference_typed_ndarray<int64_t>());

    }
    return ref_signed_var();
  }

  ref_unsigned_var math_param_ref_unsigned(std::shared_ptr<ndarray_recording_ref> param_ref_val)
  {
    assert(param_ref_val); // Won't ever happen because get_ndarray_recording_ref() now throws the exception itself
    switch (param_ref_val->ndinfo()->typenum) {
    case SNDE_RTN_UINT8:
      return ref_unsigned_var(param_ref_val->reference_typed_ndarray<uint8_t>());

    case SNDE_RTN_UINT16:
      return ref_unsigned_var(param_ref_val->reference_typed_ndarray<uint16_t>());

    case SNDE_RTN_UINT32:
      return ref_unsigned_var(param_ref_val->reference_typed_ndarray<uint32_t>());
                           

    case SNDE_RTN_UINT64:
      return ref_unsigned_var(param_ref_val->reference_typed_ndarray<uint64_t>());

    }
    return ref_unsigned_var();
  }

  ref_integer_var math_param_ref_integer(std::shared_ptr<ndarray_recording_ref> param_ref_val)
  {
    ref_signed_var param_signed = math_param_ref_signed(param_ref_val);
    ref_unsigned_var param_unsigned = math_param_ref_unsigned(param_ref_val);
    ref_integer_var combined = variant_merge(param_signed,param_unsigned);
    return combined;
  }

  // Real, meaning floating point or integer
  ref_real_var math_param_ref_real(std::shared_ptr<ndarray_recording_ref> param_ref_val)
  {
    ref_float_var param_float = math_param_ref_float(param_ref_val);
    ref_integer_var param_integer = math_param_ref_integer(param_ref_val);
    ref_real_var combined = variant_merge(param_float,param_integer);
    return combined;
  }
};
