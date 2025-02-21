

#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/vecops.h"
#include "snde/geometry_ops.h"
#include "snde/geometrydata.h"
#include "snde/normal_calc.h"
#include "snde/snde_types_h.h"

#include "snde/recstore.hpp"
#include "snde/recmath_cppfunction.hpp"
#include "snde/graphics_recording.hpp"
#include "snde/graphics_storage.hpp"

#ifdef SNDE_OPENCL
#include "snde/opencl_utils.hpp"
#include "snde/openclcachemanager.hpp"
#include "snde/recmath_compute_resource_opencl.hpp"
#endif

#include "snde/arithmetic.hpp"

namespace snde {

  template <typename Type1, typename Type2> class addition;
  template <typename Type1, typename Type2> class subtraction;

  template <typename Type1,typename Type2,typename Enable = void>
  struct larger_float {
    typedef snde_float32 type;
  };

  
  template <typename Type1,typename Type2>
  struct larger_float<Type1,Type2,typename std::enable_if<(std::is_floating_point<Type1>::value && std::is_floating_point<Type2>::value && sizeof(Type1) >= sizeof(Type2))>> {
    typedef Type1 type;
  };

  template <typename Type1,typename Type2>
  struct larger_float<Type1,Type2,typename std::enable_if<(std::is_floating_point<Type1>::value && std::is_floating_point<Type2>::value && sizeof(Type1) < sizeof(Type2))>> {
    typedef Type2 type;
  };

  template <typename Type1,typename Type2>
  struct larger_float<Type1,Type2,typename std::enable_if<(std::is_floating_point<Type1>::value && !std::is_floating_point<Type2>::value)>> {
    typedef Type1 type;
  };

  template <typename Type1,typename Type2>
  struct larger_float<Type1,Type2,typename std::enable_if<(!std::is_floating_point<Type1>::value && std::is_floating_point<Type2>::value)>> {
    typedef Type2 type;
  };


  template <typename result_type, typename Type1, typename Type2, typename Enable=void>
  struct make_signed_if_any_signed {


  };
  
  template <typename result_type, typename Type1, typename Type2>
  struct make_signed_if_any_signed<result_type, Type1, Type2,typename std::enable_if<(std::is_signed<Type1>::value || std::is_signed<Type2>::value)>::type> {
    typedef typename std::make_signed<result_type>::type type;

  };
  template <typename result_type, typename Type1, typename Type2>
  struct make_signed_if_any_signed<result_type, Type1, Type2,typename std::enable_if<(!std::is_signed<Type1>::value && !std::is_signed<Type2>::value)>::type> {
    typedef result_type type;

  };
  
  template <typename Type1, typename Type2, typename Enable = void>
  struct larger_int {
    typedef void type;

  };



  template <typename Type1, typename Type2>
  struct larger_int<Type1, Type2, typename std::enable_if<(sizeof(Type2) > sizeof(Type1))>::type> {
    typedef Type2 type;
  };
  template <typename Type1, typename Type2>
  struct larger_int<Type1, Type2, typename std::enable_if<(sizeof(Type2) <= sizeof(Type1))>::type> {
    typedef Type1 type;
  };
    
  template <typename Op, typename Type1, typename Type2,  typename Enable = void>
  struct arithmetic_result_type {
    typedef Type1 result_type;
  };

  template <typename Op, typename Type1, typename Type2>
  struct arithmetic_result_type<Op,Type1,Type2, typename std::enable_if<(std::is_floating_point<Type1>::value || std::is_floating_point<Type2>::value)>::type> {
    typedef typename larger_float<Type2,Type1>::type result_type;
  };


  template <typename Type1, typename Type2>
  struct arithmetic_result_type<addition<Type1,Type2>,Type1,Type2, typename std::enable_if<(!std::is_floating_point<Type1>::value && !std::is_floating_point<Type2>::value)>::type> {
    typedef typename make_signed_if_any_signed<typename larger_int<Type2,Type1>::type, Type1, Type2>::type result_type;
  };

  template <typename Type1, typename Type2>
  struct arithmetic_result_type<subtraction<Type1,Type2>,Type1,Type2, typename std::enable_if<(!std::is_floating_point<Type1>::value && !std::is_floating_point<Type2>::value)>::type> {
    typedef typename std::make_signed<typename larger_int<Type2,Type1>::type>::type result_type;
  };
  

  
  class arithmetic_operation {
  public:
    const std::string name;
    const std::string c_code;

    arithmetic_operation(std::string name,std::string c_code) :
      name(name),
      c_code(c_code)
    {

    }
  };
  
  class arithmetic_binary_operation: public arithmetic_operation {
  public:
    // const std::string name;
    // const std::string c_code;

    arithmetic_binary_operation(std::string name,std::string c_code) :
      arithmetic_operation(name,c_code)
    {

    }
#ifdef SNDE_OPENCL
    template <typename Op,typename Type1, typename Type2>
    std::shared_ptr<opencl_program> op_program(size_t ndim)
    {

      return build_typed_opencl_program<Type1,Type2>(ssprintf("snde.arithmetic.%s.%dd",name.c_str(),(int)ndim),(std::function<std::shared_ptr<opencl_program>(std::string,std::string)>)[ndim,this] (std::string ocltype1, std::string ocltype2) {
          auto typemap_it = rtn_typemap.find(typeid(typename arithmetic_result_type<Op,Type1,Type2>::result_type));
        if (typemap_it == rtn_typemap.end()) {
          throw snde_error("Can't dynamically build typed opencl programs without typemap entry");
        }
        auto ocltypemap_it = rtn_ocltypemap.find(typemap_it->second);
        if (ocltypemap_it == rtn_ocltypemap.end()) {
          throw snde_error("Can't dynamically build typed opencl programs without OpenCL typemap entry");
        }
        return std::make_shared<opencl_program>("arithmetic_kern",std::vector<std::string>({
              snde_types_h,
              "\ntypedef "+ocltype1+" Type1;\n",
              "typedef "+ocltype2+" Type2;\n",
              "typedef "+ocltypemap_it->second+" result_type;\n",
              ssprintf("#define NDIM %d\n", (int)ndim),
              c_code,
              
              "\n__kernel void arithmetic_kern(\n"
              "  __global const snde_index *result_dims,\n"
              "  __global const snde_index *left_strides,\n"
              "  __global const snde_index *right_strides,\n"
              "  __global const snde_index *result_strides,\n"
              "  __global const Type1 *left,\n"
              "  __global const Type2 *right,\n"
              "  __global result_type *result)\n"
              "{\n"
              "  snde_index idx=get_global_id(0);\n"
              "  snde_index dim_indexes[NDIM];\n"
              "  int dim;\n"
              "  snde_index left_index = 0;\n"
              "  snde_index right_index = 0;\n"
              "  snde_index result_index = 0;\n"

              "  if (result_strides[0] < result_strides[NDIM-1]) {\n"
              "    // Fortran order\n"
              "    for (dim=0; dim < NDIM; dim++) {\n"
              
              "      dim_indexes[dim] = idx % result_dims[dim];\n"
              "      idx = idx / result_dims[dim];\n"
              // "  printf(\"idx = %u \\n\",(unsigned)idx);\n"
              "      left_index += dim_indexes[dim]*left_strides[dim];\n"
              "      right_index += dim_indexes[dim]*right_strides[dim];\n"
              "      result_index += dim_indexes[dim]*result_strides[dim];\n"
              "    }\n"
              "  } else {\n"
              "    // C order\n"
              "    for (dim=NDIM-1; dim >=0 ; dim++) {\n"
              
              "      dim_indexes[dim] = idx % result_dims[dim];\n"
              "      idx = idx / result_dims[dim];\n"
              // "  printf(\"idx = %u \\n\",(unsigned)idx);\n"
              "      left_index += dim_indexes[dim]*left_strides[dim];\n"
              "      right_index += dim_indexes[dim]*right_strides[dim];\n"
              "      result_index += dim_indexes[dim]*result_strides[dim];\n"
              "    }\n"
              "  }\n"
              "  result[result_index] = perform_op(left[left_index], right[right_index]);\n"
              "}\n"
            }));
              
        
      });
        
    }
#endif // SNDE_OPENCL
  };
  template <typename Type1, typename Type2>
  class addition: public arithmetic_binary_operation{
  public:
    typedef typename arithmetic_result_type<addition,Type1,Type2>::result_type result_type;
    

    addition() :
      arithmetic_binary_operation("addition",
                           "result_type perform_op(Type1 left, Type2 right)\n"
                           "{\n"
                           "return left+right;\n"
                           "}\n")
    {

    }
    
    result_type perform_op(Type1 left, Type2 right)
    {
      return left+right;
    }


  };


  template <typename Type1, typename Type2>
  class subtraction: public arithmetic_binary_operation{
  public:
    typedef typename arithmetic_result_type<subtraction,Type1,Type2>::result_type result_type;
    

    subtraction() :
      arithmetic_binary_operation("subtraction",
                           "result_type perform_op(Type1 left, Type2 right)\n"
                           "{\n"
                           "return left-right;\n"
                           "}\n")
    {

    }
    
    result_type perform_op(Type1 left, Type2 right)
    {
      return left-right;
    }


  };

  
  template <typename Op,typename Type1,typename Type2> //T is the data type we are operating on. O is the operation we are performing
  class arithmetic_binary_op: public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<Type1>>,std::shared_ptr<ndtyped_recording_ref<Type2>>> {
  public:
    typedef typename arithmetic_result_type<Op,Type1,Type2>::result_type result_type;
    
    arithmetic_binary_op(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<Type1>>, std::shared_ptr<ndtyped_recording_ref<Type2>>>(rss,inst)
    {
      
    }

    // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
      typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<Type1>>,std::shared_ptr<ndtyped_recording_ref<Type2>>>::compute_options_function_override_type compute_options_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<Type1>>,std::shared_ptr<ndtyped_recording_ref<Type2>>>::define_recs_function_override_type define_recs_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<Type1>>,std::shared_ptr<ndtyped_recording_ref<Type2>>>::metadata_function_override_type metadata_function_override_type;
   typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<Type1>>,std::shared_ptr<ndtyped_recording_ref<Type2>>>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<Type1>>,std::shared_ptr<ndtyped_recording_ref<Type2>>>::exec_function_override_type exec_function_override_type;

    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>, std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<ndtyped_recording_ref<Type1>> left,std::shared_ptr<ndtyped_recording_ref<Type2>> right)
    {

      snde_index num_elements = 1;
      std::vector<snde_index> outdims = {};
      std::vector<bool> left_broadcast, right_broadcast;
      
      Type1 junk1;
      Type2 junk2;
      result_type junk3;
      

      // Calculate size of new array and determine mode of operation
      
     
      outdims = left->layout.dimlen;
      if (left->layout.dimlen.size() !=right->layout.dimlen.size()) {
        throw snde_error(ssprintf("arithmetic_binary_op %s on %s: incompatible numbers of dimensions index: %u and %u", Op().name.c_str(), this->inst->definition->definition_command.c_str(),(unsigned) left->layout.dimlen.size(), (unsigned)right->layout.dimlen.size()));
      }
      for (size_t dimnum = 0;dimnum<outdims.size();dimnum++) {
        if(right->layout.dimlen.at(dimnum) != outdims.at(dimnum)) {
          outdims.at(dimnum) = std::max(outdims.at(dimnum),right->layout.dimlen.at(dimnum));
          if (left->layout.dimlen.at(dimnum) == 1) {
            left_broadcast.push_back(true);
          } else {
            left_broadcast.push_back(false);
          }
          if (right->layout.dimlen.at(dimnum) == 1) {
            right_broadcast.push_back(true);
          } else {
            right_broadcast.push_back(false);
          }
          if (right->layout.dimlen.at(dimnum) != 1 && left->layout.dimlen.at(dimnum) != 1){
            //two incompatible dimension lengths
            throw snde_error(ssprintf("arithmetic_binary_op %s on %s: incompatible lengths of dimension index %u: %u and %u", Op().name.c_str(), this->inst->definition->definition_command.c_str(), (unsigned)dimnum, (unsigned)left->layout.dimlen.at(dimnum), (unsigned)right->layout.dimlen.at(dimnum)));

          }

        } else {
          left_broadcast.push_back(false);
          right_broadcast.push_back(false);
        }
        num_elements *= outdims.at(dimnum);
         
      }

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
      {
	std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
						      0, //metadata_bytes 
						      num_elements * (sizeof(junk1)+sizeof(junk2)+sizeof(junk3)), // data_bytes for transfer
						      num_elements, // flops
						      1, // max effective cpu cores
						      1), // useful_cpu_cores (min # of cores to supply
   
#ifdef SNDE_OPENCL
        std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
                                                       0, //metadata_bytes
  						      num_elements * (sizeof(junk1)+sizeof(junk2)+sizeof(junk3)), // data_bytes for transfer

                                                      0, // cpu_flops
                                                       num_elements, // guflops
                                                       1, // max effective cpu cores
                                                       1, // useful_cpu_cores (min # of cores to supply
                                                       (std::is_floating_point<Type1>::value && (sizeof(Type1) > sizeof(float))) || (std::is_floating_point<Type2>::value && (sizeof(Type2) > sizeof(float)))  ), // requires_doubleprec 
#endif // SNDE_OPENCL
      };
      return std::make_pair(option_list, std::make_shared<define_recs_function_override_type>([this, num_elements, left, right, outdims, left_broadcast, right_broadcast]() {

	    // define_recs code
	    std::shared_ptr<ndtyped_recording_ref<result_type>> result_ref;

	    result_ref = create_typed_ndarray_ref_math<result_type>(this->get_result_channel_path(0),this->rss);
	    
	    return std::make_shared<metadata_function_override_type>([ this, num_elements, left, right, outdims, left_broadcast, right_broadcast, result_ref]() {
	      // metadata code
	      std::shared_ptr<constructible_metadata> metadata = MergeMetadata(left->rec->metadata, right->rec->metadata);
	      result_ref->rec->metadata=metadata;
	      result_ref->rec->mark_metadata_done();
	      
	      return std::make_shared<lock_alloc_function_override_type>([this, num_elements, left, right, outdims, left_broadcast, right_broadcast, result_ref]() {
		// lock_alloc code
#ifdef SNDE_OPENCL
                std::shared_ptr<assigned_compute_resource_opencl> opencl_resource = std::dynamic_pointer_cast<assigned_compute_resource_opencl>(this->compute_resource);
                bool using_gpu = opencl_resource != nullptr;
#else
                bool using_gpu = false;
#endif	
		result_ref->allocate_storage(outdims,left->layout.is_f_contiguous()); 

		
		std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> to_lock;
		
                to_lock.push_back(std::make_pair(left,false)); // lock for read
                to_lock.push_back(std::make_pair(right,false)); // lock for read
		to_lock.push_back(std::make_pair(result_ref,true)); // lock for write
		
		rwlock_token_set locktokens = this->lockmgr->lock_recording_refs(to_lock,using_gpu);
		


                return std::make_shared<exec_function_override_type>([this, num_elements, left, right, outdims, left_broadcast, right_broadcast, result_ref, locktokens]() {
                  // exec code
                  Op binary_op;
                  size_t ndim = outdims.size();
                  std::vector<snde_index> left_strides(left->layout.strides);
                  std::vector<snde_index> right_strides(right->layout.strides);
                  for (size_t dimcnt; dimcnt<ndim; dimcnt++) {
                    if (left_broadcast.at(dimcnt)) {
                      left_strides.at(dimcnt) = 0;
                    }
                    if (right_broadcast.at(dimcnt)) {
                      right_strides.at(dimcnt) = 0;
                    }
                  }
                  
#ifdef SNDE_OPENCL
                  std::shared_ptr<assigned_compute_resource_opencl> opencl_resource = std::dynamic_pointer_cast<assigned_compute_resource_opencl>(this->compute_resource);
                  if (opencl_resource &&  num_elements > 10000) {

	     
                    std::shared_ptr<opencl_program> arithmetic_prog = binary_op.template op_program<Op,Type1,Type2>(ndim);
	     
	      
                    cl::Kernel arithmetic_kern = arithmetic_prog->get_kernel(opencl_resource->context,opencl_resource->devices.at(0));
	      
                    OpenCLBuffers Buffers(opencl_resource->oclcache,opencl_resource->context,opencl_resource->devices.at(0),locktokens);
                    cl_int err = 0;
                    std::vector<cl::Event> Events;
                    cl::Buffer result_dims_mem(opencl_resource->context,CL_MEM_READ_ONLY,sizeof(snde_index)*ndim);
                    err = opencl_resource->queues.at(0).enqueueWriteBuffer(result_dims_mem, CL_FALSE, 0, sizeof(snde_index) * ndim, outdims.data(), &Events);
		    if (err != CL_SUCCESS) {
		      throw openclerror(err, "Error transferring kernel data");
		    }
                    cl::Buffer left_strides_mem(opencl_resource->context,CL_MEM_READ_ONLY,sizeof(snde_index)*ndim);
                    err = opencl_resource->queues.at(0).enqueueWriteBuffer(left_strides_mem, CL_FALSE, 0, sizeof(snde_index) * ndim, left_strides.data(), &Events);
		    if (err != CL_SUCCESS) {
		      throw openclerror(err, "Error transferring kernel data");
		    }
                    cl::Buffer right_strides_mem(opencl_resource->context,CL_MEM_READ_ONLY,sizeof(snde_index)*ndim);
                    err = opencl_resource->queues.at(0).enqueueWriteBuffer(right_strides_mem, CL_FALSE, 0, sizeof(snde_index) * ndim, right_strides.data(), &Events);
		    if (err != CL_SUCCESS) {
		      throw openclerror(err, "Error transferring kernel data");
		    }
                    cl::Buffer result_strides_mem(opencl_resource->context,CL_MEM_READ_ONLY,sizeof(snde_index)*ndim);
                    err = opencl_resource->queues.at(0).enqueueWriteBuffer(result_strides_mem, CL_FALSE, 0, sizeof(snde_index) * ndim, result_ref->layout.strides.data(), &Events);
		    if (err != CL_SUCCESS) {
		      throw openclerror(err, "Error transferring kernel data");
		    }
                    arithmetic_kern.setArg(0,result_dims_mem);
                    arithmetic_kern.setArg(1,left_strides_mem);
                    arithmetic_kern.setArg(2,right_strides_mem);
                    arithmetic_kern.setArg(3,result_strides_mem);
                    Buffers.AddBufferAsKernelArg(left,arithmetic_kern,4,false,false);
                    Buffers.AddBufferAsKernelArg(right,arithmetic_kern,5,false,false);	      
                    Buffers.AddBufferAsKernelArg(result_ref,arithmetic_kern,6,true,true);
	      
                    cl::Event kerndone;
                    std::vector<cl::Event> FillEvents=Buffers.FillEvents();
                    //combine FillEvents into our existing vector
                    Events.insert(Events.end(),FillEvents.begin(),FillEvents.end());
                    
                    err = opencl_resource->queues.at(0).enqueueNDRangeKernel(arithmetic_kern,{},{ num_elements },{},&Events,&kerndone);	      
                    if (err != CL_SUCCESS) {
                      throw openclerror(err,"Error enqueueing kernel");
                    }
                    opencl_resource->queues.at(0).flush(); /* trigger execution */
                    // mark that the kernel has modified result_rec
                    Buffers.BufferDirty(result_ref);
                    // wait for kernel execution and transfers to complete
                    Buffers.RemBuffers(kerndone,kerndone,true);

	     
                  } else {	    
#endif // SNDE_OPENCL
                    snde_index idx;

                    std::vector<snde_index> result_pos;//, right_pos, result_pos;
                    snde_index left_index = 0, right_index = 0, result_index = 0;
                    for (size_t dimnum = 0; dimnum<ndim; dimnum++) {
                      // left_pos.push_back(0);
                      //right_pos.push_back(0);
                      result_pos.push_back(0);

                    }
                    Type1 *left_data=left->shifted_arrayptr();
                    Type2 *right_data=right->shifted_arrayptr();
                    result_type *result_data=result_ref->shifted_arrayptr();
                    for (idx=0;idx<num_elements;idx++) {
                      result_data[result_index]=binary_op.perform_op(left_data[left_index],right_data[right_index]);
                      for (int dimnum = ndim-1; dimnum>=0; dimnum--) {
                        if (!left_broadcast[dimnum]) {
                          
                          //left_pos[dimnum]++;
                          left_index += left->layout.strides[dimnum];
                        }
                        if (!right_broadcast[dimnum]) {
                          right_index+= right->layout.strides[dimnum];
                        }
                        result_pos[dimnum]++;
                        result_index += result_ref->layout.strides[dimnum];
                        //check if this dimension has reached its limit
                        if (result_pos[dimnum] >= outdims[dimnum]) {
                          //reset this position to 0
                          result_pos[dimnum] = 0;
                          result_index -= result_ref->layout.strides[dimnum]*outdims[dimnum];
                          if (!left_broadcast[dimnum]) {
                          
                            left_index -= left->layout.strides[dimnum]*outdims[dimnum];
                          }
                          if (!right_broadcast[dimnum]) {
                            right_index -= right->layout.strides[dimnum]*outdims[dimnum];
                          }
                          //allow us to increment the preceding dimension
                        } else {
                          //we have reached a valid new position
                          break;

                        }

                      }
                        

                    }
                  }
                  unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready()
                    
                  result_ref->rec->mark_data_ready();
                    //snde_warning("avg: Generated new result (rev %llu)",(unsigned long long)result_ref->rec->info->revision);
                  
                });
              });
	      
            });
      }));
      
    };

  
  
    
  };
  template <typename Type1, typename Type2>
  using addition_op = arithmetic_binary_op<addition<Type1,Type2>,Type1,Type2>;


  std::shared_ptr<math_function> define_addition_function()
  {
    std::shared_ptr<math_function> newfunc = std::make_shared<cpp_math_function>("snde.addition",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) -> std::shared_ptr<executing_math_function>  {
      std::shared_ptr<executing_math_function> executing;

      if (!inst) {
        // initial call with no instantiation to probe parameters; just use int32 case
        return std::make_shared<arithmetic_binary_op<addition<snde_float32,snde_float32>,snde_float32,snde_float32>>(rss,inst);
      }
      
      std::shared_ptr<ndarray_recording_ref> param0_ref = math_param_ref(rss,inst,0);
      std::shared_ptr<ndarray_recording_ref> param1_ref = math_param_ref(rss,inst,1);
      
      ref_real_var param0 = math_param_ref_real(param0_ref);
      ref_real_var param1 = math_param_ref_real(param1_ref);

      
      if (param0.has_value() && param1.has_value()) {
        
        return make_cppfuncexec_twovariants<addition_op>(rss,inst,param0,param1);
      } else {
        throw math_parameter_mismatch("Recording parameters are not compatible with the addition operation");
      }
    
      return nullptr;
    });

    return newfunc;
    
  }

  SNDE_OCL_API std::shared_ptr<math_function> addition_function=define_addition_function();

  
  static int registered_addition_function = register_math_function(addition_function);
  







  template <typename Type1, typename Type2>
  using subtraction_op = arithmetic_binary_op<subtraction<Type1,Type2>,Type1,Type2>;


  std::shared_ptr<math_function> define_subtraction_function()
  {
    std::shared_ptr<math_function> newfunc = std::make_shared<cpp_math_function>("snde.subtraction",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) -> std::shared_ptr<executing_math_function>  {
      std::shared_ptr<executing_math_function> executing;

      if (!inst) {
        // initial call with no instantiation to probe parameters; just use int32 case
        return std::make_shared<arithmetic_binary_op<subtraction<snde_float32,snde_float32>,snde_float32,snde_float32>>(rss,inst);
      }
      
      std::shared_ptr<ndarray_recording_ref> param0_ref = math_param_ref(rss,inst,0);
      std::shared_ptr<ndarray_recording_ref> param1_ref = math_param_ref(rss,inst,1);
      
      ref_real_var param0 = math_param_ref_real(param0_ref);
      ref_real_var param1 = math_param_ref_real(param1_ref);

      
      if (param0.has_value() && param1.has_value()) {
        
        return make_cppfuncexec_twovariants<subtraction_op>(rss,inst,param0,param1);
      } else {
        throw math_parameter_mismatch("Recording parameters are not compatible with the subtraction operation");
      }
    
      return nullptr;
    });

    return newfunc;
    
  }

  SNDE_OCL_API std::shared_ptr<math_function> subtraction_function=define_subtraction_function();

  
  static int registered_subtraction_function = register_math_function(subtraction_function);
  
  
};
