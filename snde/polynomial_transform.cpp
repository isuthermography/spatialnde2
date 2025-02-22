#include <math.h>

#include "snde/recmath_cppfunction.hpp"

#ifdef SNDE_OPENCL
#include "snde/opencl_utils.hpp"
#include "snde/openclcachemanager.hpp"
#include "snde/recmath_compute_resource_opencl.hpp"
#endif

#include "snde/snde_types_h.h"


#include "snde/polynomial_transform.hpp"

#include "snde/polynomial_transform_c.h"




namespace snde {

  template <typename InputType, typename OutputType, typename PolyType>
  void polynomial_transform_math(OCL_GLOBAL_ADDR OutputType* output, // output array
    OCL_GLOBAL_ADDR InputType* data, // input data -- must be same dimensions as output array
    OCL_GLOBAL_ADDR PolyType* poly, // polynomial -- must be same dimensions as data in all but one axis
    snde_index datandim, // number of dimensions of data array
    snde_index* datadims, // array of data dimensions
    snde_index* datacalc, // Empty array of length datandim used during calculation to determine where we are in the data array
    snde_index polyndim, // number of dimensions of data array
    snde_index* polydims, // array of data dimensions
    snde_index* polycalc, // Empty array of length datandim used during calculation to determine where we are in the data array
    snde_index polyaxis, // Axis along which polynomial coefficients exist
    snde_index idx)  // index to calculate
  {
    throw snde_error("polynomial_transform_math not implemented for <%s, %s, %s>", typeid(InputType).name(), typeid(OutputType).name(), typeid(PolyType).name());
  }


  template <typename InputType, typename OutputType, typename PolyType>  // template for different floating point number classes 
  class polynomial_transform : public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<InputType>>, std::shared_ptr<ndtyped_recording_ref<PolyType>>, snde_index>
  {
  public:
    polynomial_transform(std::shared_ptr<recording_set_state> rss, std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<InputType>>, std::shared_ptr<ndtyped_recording_ref<PolyType>>, snde_index>(rss, inst)
    {

    }

    // Specializations implemented by including C file with an appropriate define
    typedef OutputType function_outtype;
    typedef InputType function_inputtype;
    typedef PolyType function_polytype;

#include "polynomial_transform.c"  
#undef polynomial_transform_math  


    // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<InputType>>, std::shared_ptr<ndtyped_recording_ref<PolyType>>, snde_index>::define_recs_function_override_type define_recs_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<InputType>>, std::shared_ptr<ndtyped_recording_ref<PolyType>>, snde_index>::metadata_function_override_type metadata_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<InputType>>, std::shared_ptr<ndtyped_recording_ref<PolyType>>, snde_index>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<InputType>>, std::shared_ptr<ndtyped_recording_ref<PolyType>>, snde_index>::exec_function_override_type exec_function_override_type;

    // just using the default for decide_new_revision

    std::pair<std::vector<std::shared_ptr<compute_resource_option>>, std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<ndtyped_recording_ref<InputType>> input, std::shared_ptr<ndtyped_recording_ref<PolyType>> poly, snde_index polyaxis)
    {

      if (!input->layout.is_c_contiguous() || !poly->layout.is_c_contiguous()) {
	throw snde_error("polynomial_transform:  C Layout is required");
      }

      InputType injunk = 0.0;
      OutputType outjunk = 0.0;
      PolyType polyjunk = 0.0;
      bool doubleprec = false;
      if (sizeof(injunk) > 4 || sizeof(outjunk) > 4 || sizeof(polyjunk) > 4) {
	doubleprec = true;
      }

      unsigned long nelem = 1;
      for (const auto& e : input->layout.dimlen) {
	nelem *= e;
      }

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
      {	

	std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
						      0, //metadata_bytes 
						      nelem * sizeof(OutputType), // data_bytes for transfer
						      0.0, // flops
						      1, // max effective cpu cores
						      1), // useful_cpu_cores (min # of cores to supply

#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
							   0, //metadata_bytes
							   nelem * sizeof(OutputType),
							   0.0, // cpu_flops
							   0.0, // gpuflops
							   1, // max effective cpu cores
							   1, // useful_cpu_cores (min # of cores to supply
							   doubleprec), // requires_doubleprec 
#endif // SNDE_OPENCL
      };
      return std::make_pair(option_list, std::make_shared<define_recs_function_override_type>([this, input, poly, polyaxis, nelem]() {

	// define_recs code
	//snde_debug(SNDE_DC_APP,"define_recs()");
	// Use of "this" in the next line for the same reason as the typedefs, above
	//std::shared_ptr<multi_ndarray_recording> result_rec = create_recording_math<multi_ndarray_recording>(this->get_result_channel_path(0), this->rss, 1);
	input->assert_no_scale_or_offset(this->inst->definition->definition_command);

	std::shared_ptr<ndtyped_recording_ref<OutputType>> result_ref;
	result_ref = create_typed_ndarray_ref_math<OutputType>(this->get_result_channel_path(0), this->rss);

	return std::make_shared<metadata_function_override_type>([this, result_ref, input, poly, polyaxis, nelem]() {
	  // metadata code
	  //std::unordered_map<std::string,metadatum> metadata;
	  //snde_debug(SNDE_DC_APP,"metadata()");
	  //metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));

	  result_ref->rec->metadata = std::shared_ptr<snde::constructible_metadata>();
	  result_ref->rec->metadata = snde::MergeMetadata(result_ref->rec->metadata, input->rec->metadata);

	  std::shared_ptr<snde::constructible_metadata> mergemdata = std::make_shared<snde::constructible_metadata>();
	  mergemdata->AddMetaDatum(snde::metadatum("ande_array-ampl_units", poly->rec->metadata->GetMetaDatumStr("ande_array-ampl_units", "Arb")));
	  mergemdata->AddMetaDatum(snde::metadatum("ande_array-ampl_coord", poly->rec->metadata->GetMetaDatumStr("ande_array-ampl_coord", "Intensity")));
	  result_ref->rec->metadata = snde::MergeMetadata(result_ref->rec->metadata, mergemdata);
	  
	  result_ref->rec->mark_metadata_done();

	  return std::make_shared<lock_alloc_function_override_type>([this, result_ref, input, poly, polyaxis, nelem]() {
	    // lock_alloc code

	    result_ref->allocate_storage(input->layout.dimlen, false);

	    // locking is only required for certain recordings
	    // with special storage under certain conditions,
	    // however it is always good to explicitly request
	    // the locks, as the locking is a no-op if
	    // locking is not actually required.

	    // lock our output arrays
	    std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>, std::pair<size_t, bool>>> recrefs_to_lock = {
	      { result_ref->rec, { 0, true }},
	      { input->rec, {0, false}},
	      { poly->rec, {0, false}}
	    };

	    rwlock_token_set locktokens = this->lockmgr->lock_recording_arrays(recrefs_to_lock,
#ifdef SNDE_OPENCL
	      true
#else
	      false
#endif
	    );

	    return std::make_shared<exec_function_override_type>([this, locktokens, result_ref, input, poly, polyaxis, nelem]() {
	      // exec code
	      //snde_index flattened_length = recording->layout.flattened_length();
	      //for (snde_index pos=0;pos < flattened_length;pos++){
	      //  result_rec->element(pos) = (recording->element(pos)-offset)/unitsperintensity;
	      //}



#ifdef SNDE_OPENCL
	      std::shared_ptr<assigned_compute_resource_opencl> opencl_resource = std::dynamic_pointer_cast<assigned_compute_resource_opencl>(this->compute_resource);
	      if (opencl_resource && nelem > 0) {

		try {

		  //fprintf(stderr,"Executing in OpenCL!\n");
		  InputType injunk = 0.0;
		  OutputType outjunk = 0.0;
		  PolyType polyjunk = 0.0;

		  cl::Kernel polynomial_transform_kern = build_typed_opencl_program<InputType, OutputType, PolyType>("snde.polynomial_transform", (std::function<std::shared_ptr<opencl_program>(std::string,std::string,std::string)>)[](std::string oclintypename,std::string oclouttypename, std::string oclpolytypename) {
		    // OpenCL templating via a typedef....
		    return std::make_shared<opencl_program>("polynomial_transform_math_ocl", std::vector<std::string>({ snde_types_h,
													       "\ntypedef " + oclouttypename + " function_outtype;\n",
													       "\ntypedef " + oclintypename + " function_inputtype;\n",
													       "\ntypedef " + oclpolytypename + " function_polytype;\n",
													       polynomial_transform_c }));
		    })->get_kernel(opencl_resource->context, opencl_resource->devices.at(0));

		    OpenCLBuffers Buffers(opencl_resource->oclcache, opencl_resource->context, opencl_resource->devices.at(0), locktokens);

		    std::vector<cl::Event> kerndoneevents;

		    snde_index polyelem = 1;
		    for (const auto& e : poly->layout.dimlen) {
		      polyelem *= e;
		    }


		    snde_index datandim = input->ndinfo()->ndim; // number of dimensions of data array
		    snde_index polyndim = poly->ndinfo()->ndim; // number of dimensions of data array

		    snde_index polyax = polyaxis;

		    cl_int err = 0;

		    //snde_warning("%llu, %llu, %llu\n\n", input->layout.dimlen.at(0), input->layout.dimlen.at(1), input->layout.dimlen.at(2));

		    void* datadimsvoid = static_cast<void*>(input->layout.dimlen.data());
		    void* polydimsvoid = static_cast<void*>(poly->layout.dimlen.data());

		    //CL_MEM_COPY_HOST_POINTER
		    //cl::EnqueueCopy -- will produce event

		    std::vector<cl::Event> Events = {};

		    cl::Buffer datadimsmem(opencl_resource->context, CL_MEM_READ_ONLY, sizeof(snde_index) * datandim);
		    cl::Buffer polydimsmem(opencl_resource->context, CL_MEM_READ_ONLY, sizeof(snde_index) * polyndim);
		    //cl_mem datadimsmem = clCreateBuffer(opencl_resource->context.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, , &datadimsvoid, &ret);
		    //cl_mem polydimsmem = clCreateBuffer(opencl_resource->context.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(snde_index) * polyndim, &polydimsvoid, &ret);

		    err = opencl_resource->queues.at(0).enqueueWriteBuffer(datadimsmem, CL_TRUE, 0, sizeof(snde_index) * datandim, datadimsvoid, &Events);
		    if (err != CL_SUCCESS) {
		      throw openclerror(err, "Error enqueueing kernel");
		    }
		    err = opencl_resource->queues.at(0).enqueueWriteBuffer(polydimsmem, CL_TRUE, 0, sizeof(snde_index) * polyndim, polydimsvoid, &Events);
		    if (err != CL_SUCCESS) {
		      throw openclerror(err, "Error enqueueing kernel");
		    }

		    std::vector<snde_index> datacalc(input->ndinfo()->ndim, 0);
		    std::vector<snde_index> polycalc(input->ndinfo()->ndim, 0);

		    size_t datacalc_octwords_per_workitem = (datandim * sizeof(snde_index) + 7) / 8;
		    size_t polycalc_octwords_per_workitem = (polyndim * sizeof(snde_index) + 7) / 8;
		    size_t local_memory_octowrds_per_workitem = datacalc_octwords_per_workitem + polycalc_octwords_per_workitem;
		    size_t kern_work_group_size, kernel_global_work_items;

		    std::tie(kern_work_group_size, kernel_global_work_items) = opencl_layout_workgroups_for_localmemory_1D(opencl_resource->devices.at(0), polynomial_transform_kern, local_memory_octowrds_per_workitem, nelem);

		    Buffers.AddBufferPortionAsKernelArg(result_ref, 0, nelem, polynomial_transform_kern, 0, true, false);
		    Buffers.AddBufferPortionAsKernelArg(input, 0, nelem, polynomial_transform_kern, 1, false, false);
		    Buffers.AddBufferPortionAsKernelArg(poly, 0, polyelem, polynomial_transform_kern, 2, false, false);
		    polynomial_transform_kern.setArg(3, sizeof(datandim), &datandim);
		    polynomial_transform_kern.setArg(4, datadimsmem);
		    polynomial_transform_kern.setArg(5, datacalc_octwords_per_workitem * 8 * kern_work_group_size, nullptr);
		    polynomial_transform_kern.setArg(6, sizeof(polyndim), &polyndim);
		    polynomial_transform_kern.setArg(7, polydimsmem);
		    polynomial_transform_kern.setArg(8, polycalc_octwords_per_workitem * 8 * kern_work_group_size, nullptr);
		    polynomial_transform_kern.setArg(9, sizeof(polyax), &polyax);


		    cl::Event kerndone;

		    std::vector<cl::Event> FillEvents = Buffers.FillEvents();

		    FillEvents.insert(FillEvents.end(), Events.begin(), Events.end());


		    err = opencl_resource->queues.at(0).enqueueNDRangeKernel(polynomial_transform_kern, {}, { kernel_global_work_items }, { kern_work_group_size}, &FillEvents, &kerndone);
		    if (err != CL_SUCCESS) {
		      throw openclerror(err, "Error enqueueing kernel");
		    }

		    Buffers.BufferPortionDirty(result_ref, 0, nelem);
		    kerndoneevents.push_back(kerndone);


		    opencl_resource->queues.at(0).flush(); /* trigger execution */
		    // mark that the kernel has modified result_rec
		    // wait for kernel execution and transfers to complete

		    cl::Event::waitForEvents(kerndoneevents);
		    Buffers.RemBuffers(*(kerndoneevents.end() - 1), *(kerndoneevents.end() - 1), true);

		} catch (const cl::Error& exc) {
		  // Only consider exceptions derived from std::exception because there's no general way to print anything else, so we might as well just crash in that case. 
		  // func is our math_function_execution
		  throw snde_error("Exception class %s caught in math thread pool: [%d] %s (function %s)", typeid(exc).name(), exc.err(), exc.what(), "polynomial_transform");
		}

	      }
	      else {
#endif // SNDE_OPENCL
		snde_warning("Performing waveform vertex calculation on CPU. ");

		

		for (snde_index cnt = 0; cnt < nelem; cnt++) {

		  std::vector<snde_index> datacalc(input->ndinfo()->ndim, 0);
		  std::vector<snde_index> polycalc(input->ndinfo()->ndim, 0);

		  polynomial_transform_math(result_ref->shifted_arrayptr(),
		    input->shifted_arrayptr(),
		    poly->shifted_arrayptr(),
		    input->ndinfo()->ndim,
		    input->layout.dimlen.data(),
		    datacalc.data(),
		    poly->ndinfo()->ndim,
		    poly->layout.dimlen.data(),
		    polycalc.data(),
		    polyaxis,
		    cnt);
		}
#ifdef SNDE_OPENCL
	      }
#endif // SNDE_OPENCL

	      unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	      result_ref->rec->mark_data_ready();

	      });
	    });
	  });
	}));
    };
  };



  std::shared_ptr<math_function> define_polynomial_transform_function_float32()
  {
    return std::make_shared<cpp_math_function>("snde.polynomial_transform_float32",1,[](std::shared_ptr<recording_set_state> rss, std::shared_ptr<instantiated_math_function> inst) {

      if (!inst) {
	// initial call with no instantiation to probe parameters; just use float32 case
	return make_cppfuncexec_floatingtypes<polynomial_transform, snde_float32, snde_float32>(rss, inst);
      }

      std::shared_ptr<math_parameter> secondparam = inst->parameters.at(1);

      assert(secondparam->paramtype == SNDE_MFPT_RECORDING);

      std::shared_ptr<math_parameter_recording> secondparam_rec = std::dynamic_pointer_cast<math_parameter_recording>(secondparam);

      assert(secondparam_rec);

      std::shared_ptr<ndarray_recording_ref> secondparam_rec_val = secondparam_rec->get_ndarray_recording_ref(rss, inst->channel_path_context, inst->definition, 1);

      if (!secondparam_rec_val) { // Won't ever happen because get_ndarray_recording_ref() now throws the exception itself
	throw snde_error("In attempting to call math function %s, first parameter %s is not an ndarray recording", inst->definition->definition_command.c_str(), secondparam_rec->channel_name.c_str());
      }

      std::shared_ptr<executing_math_function> executing;

      switch (secondparam_rec_val->ndinfo()->typenum) {
      case SNDE_RTN_FLOAT32:
	executing = make_cppfuncexec_floatingtypes<polynomial_transform, snde_float32, snde_float32>(rss, inst);
	if (!executing) {
	  executing = make_cppfuncexec_integertypes<polynomial_transform, snde_float32, snde_float32>(rss, inst);
	}
	if (!executing) {
	  throw snde_error("In attempting to call math function %s, first parameter has unsupported data type.", inst->definition->definition_command.c_str());
	}
	return executing;

      case SNDE_RTN_FLOAT64:
	executing = make_cppfuncexec_floatingtypes<polynomial_transform, snde_float32, snde_float64>(rss, inst);
	if (!executing) {
	  executing = make_cppfuncexec_integertypes<polynomial_transform, snde_float32, snde_float64>(rss, inst);
	}
	if (!executing) {
	  throw snde_error("In attempting to call math function %s, first parameter has unsupported data type.", inst->definition->definition_command.c_str());
	}
	return executing;

#ifdef SNDE_HAVE_FLOAT16
      case SNDE_RTN_FLOAT16:
	executing = make_cppfuncexec_floatingtypes<polynomial_transform, snde_float32, snde_float16>(rss, inst);
	if (!executing) {
	  executing = make_cppfuncexec_integertypes<polynomial_transform, snde_float32, snde_float16>(rss, inst);
	}
	if (!executing) {
	  throw snde_error("In attempting to call math function %s, first parameter has unsupported data type.", inst->definition->definition_command.c_str());
	}
	return executing;
#endif

      default:
	throw snde_error("In attempting to call math function %s, second parameter %s has non-floating point type %s",inst->definition->definition_command.c_str(),secondparam_rec->channel_name.c_str(),rtn_typenamemap.at(secondparam_rec_val->ndinfo()->typenum).c_str());
      }

      }
    );

  }

  std::shared_ptr<math_function> define_polynomial_transform_function_float64()
  {
    return std::make_shared<cpp_math_function>("snde.polynomial_transform_float64",1,[](std::shared_ptr<recording_set_state> rss, std::shared_ptr<instantiated_math_function> inst) {

      if (!inst) {
	// initial call with no instantiation to probe parameters; just use float32 case
	return make_cppfuncexec_floatingtypes<polynomial_transform, snde_float64, snde_float32>(rss, inst);
      }

      std::shared_ptr<math_parameter> secondparam = inst->parameters.at(1);

      assert(secondparam->paramtype == SNDE_MFPT_RECORDING);

      std::shared_ptr<math_parameter_recording> secondparam_rec = std::dynamic_pointer_cast<math_parameter_recording>(secondparam);

      assert(secondparam_rec);

      std::shared_ptr<ndarray_recording_ref> secondparam_rec_val = secondparam_rec->get_ndarray_recording_ref(rss, inst->channel_path_context, inst->definition, 1);

      if (!secondparam_rec_val) { // Won't ever happen because get_ndarray_recording_ref() now throws the exception itself
	throw snde_error("In attempting to call math function %s, first parameter %s is not an ndarray recording", inst->definition->definition_command.c_str(), secondparam_rec->channel_name.c_str());
      }

      std::shared_ptr<executing_math_function> executing;

      switch (secondparam_rec_val->ndinfo()->typenum) {
      case SNDE_RTN_FLOAT32:
	executing = make_cppfuncexec_floatingtypes<polynomial_transform, snde_float64, snde_float32>(rss, inst);
	if (!executing) {
	  executing = make_cppfuncexec_integertypes<polynomial_transform, snde_float64, snde_float32>(rss, inst);
	}
	if (!executing) {
	  throw snde_error("In attempting to call math function %s, first parameter has unsupported data type.", inst->definition->definition_command.c_str());
	}
	return executing;

      case SNDE_RTN_FLOAT64:
	executing = make_cppfuncexec_floatingtypes<polynomial_transform, snde_float64, snde_float64>(rss, inst);
	if (!executing) {
	  executing = make_cppfuncexec_integertypes<polynomial_transform, snde_float64, snde_float64>(rss, inst);
	}
	if (!executing) {
	  throw snde_error("In attempting to call math function %s, first parameter has unsupported data type.", inst->definition->definition_command.c_str());
	}
	return executing;

#ifdef SNDE_HAVE_FLOAT16
      case SNDE_RTN_FLOAT16:
	executing = make_cppfuncexec_floatingtypes<polynomial_transform, snde_float64, snde_float16>(rss, inst);
	if (!executing) {
	  executing = make_cppfuncexec_integertypes<polynomial_transform, snde_float64, snde_float16>(rss, inst);
	}
	if (!executing) {
	  throw snde_error("In attempting to call math function %s, first parameter has unsupported data type.", inst->definition->definition_command.c_str());
	}
	return executing;
#endif

      default:
	throw snde_error("In attempting to call math function %s, second parameter %s has non-floating point type %s", inst->definition->definition_command.c_str(), secondparam_rec->channel_name.c_str(), rtn_typenamemap.at(secondparam_rec_val->ndinfo()->typenum).c_str());
      }

      }
    );

  }

  SNDE_OCL_API std::shared_ptr<math_function> polynomial_transform_function_float32 = define_polynomial_transform_function_float32();

  static int registered_polynomial_transform_function_float32 = register_math_function( polynomial_transform_function_float32);


  SNDE_OCL_API std::shared_ptr<math_function> polynomial_transform_function_float64 = define_polynomial_transform_function_float64();

  static int registered_polynomial_transform_function_float64 = register_math_function(polynomial_transform_function_float64);




};





