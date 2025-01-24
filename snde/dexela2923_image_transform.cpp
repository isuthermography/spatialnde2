#include "snde/rec_display_colormap.hpp"
#include "snde/recmath_cppfunction.hpp"
#include "snde/dexela2923_image_transform_kernel_h.h"
#include "snde/snde_types_h.h"
#include "snde/geometry_types_h.h"
#include "snde/dexela2923_image_transform_kernel.h"
#include "snde/dexela2923_image_transform.hpp"

#ifdef SNDE_OPENCL
#include "snde/opencl_utils.hpp"
#include "snde/openclcachemanager.hpp"
#include "snde/recmath_compute_resource_opencl.hpp"
#endif

namespace snde {

#ifdef SNDE_OPENCL


  
  static opencl_program dexela2923_image_transform_function_opencl("dexela2923_image_transform_kernel", { snde_types_h, geometry_types_h,  dexela2923_image_transform_kernel_h });
#endif // SNDE_OPENCL

  
  
  class dexela2923_image_transform: public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<uint16_t>>>
  {
  public:
    dexela2923_image_transform(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<uint16_t>>>(rss,inst)
    {
      
    }
    
    
    // just using the default for decide_new_revision

    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<ndtyped_recording_ref<uint16_t>> rawimage) 
    {
      snde_index numdatapoints = rawimage->layout.dimlen.at(0)*rawimage->layout.dimlen.at(1);

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numdatapoints*2*sizeof(uint16_t), // data_bytes for transfer
							numdatapoints*(10), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
							   0, //metadata_bytes
							   numdatapoints*2*sizeof(uint16_t),
							   0, // cpu_flops
							   numdatapoints*(10), // gpuflops
							   1, // max effective cpu cores
							   1, // useful_cpu_cores (min # of cores to supply
							   false), // requires_doubleprec 
#endif // SNDE_OPENCL
	};
      return std::make_pair(option_list,nullptr);
    }


 
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<ndtyped_recording_ref<uint16_t>> rawimage) 
    {
      // define_recs code
      //snde_debug(SNDE_DC_APP,"define_recs()");
      // Use of "this" in the next line for the same reason as the typedefs, above
      rawimage->assert_no_scale_or_offset(this->inst->definition->definition_command);

      std::shared_ptr<ndtyped_recording_ref<uint16_t>> result_rec = create_typed_ndarray_ref_math<uint16_t>(this->get_result_channel_path(0),this->rss);
      
      return std::make_shared<metadata_function_override_type>([ this,result_rec,rawimage ] () {
	// metadata code
	//std::unordered_map<std::string,metadatum> metadata;
	//snde_debug(SNDE_DC_APP,"metadata()");
	//metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));
		std::shared_ptr<constructible_metadata> new_metadata=std::make_shared<constructible_metadata>(rawimage->rec->metadata);
		new_metadata->AddMetaDatum(metadatum("ande_array-axis0_offset",-3072/2,"pixels"));
		new_metadata->AddMetaDatum(metadatum("ande_array-axis1_offset",3888/2,"pixels"));
	result_rec->rec->metadata=new_metadata;
	result_rec->rec->mark_metadata_done();
	
	return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,rawimage ]() {
	  // lock_alloc code
	  result_rec->rec->assign_storage_manager(this->recdb->default_storage_manager); // Force default storage manager so that we DON'T go to the graphics storage (which is unnecessary for temporary output such as this)
	  
	  //# Dexela 2923 detector is 3072x3888
	  //# But it is really four 1536x1944 detectors
	  //# Read back in parallel.
	  //# Therefore the capture card should be configured
	  //# for a 16 bit 6144x1944 detector

	  if (rawimage->layout.dimlen.size() != 2 ||
	      rawimage->layout.dimlen.at(0) != 6144 ||
	      rawimage->layout.dimlen.at(1) != 1944 ||
	      !rawimage->layout.is_f_contiguous() ) {
	    throw snde_error("dexela2923_image_transform:invalid input array size or layout");
	  }
	  
	  result_rec->allocate_storage({3072,3888},true); // Note fortran order flag 
      
	  
#ifdef SNDE_OPENCL
	  std::shared_ptr<assigned_compute_resource_opencl> opencl_resource = std::dynamic_pointer_cast<assigned_compute_resource_opencl>(this->compute_resource);
	  bool using_gpu = opencl_resource != nullptr;
#else
	  bool using_gpu = false;
#endif

	  // locking is only required for certain recordings
	  // with special storage under certain conditions,
	  // however it is always good to explicitly request
	  // the locks, as the locking is a no-op if
	  // locking is not actually required. 
	  rwlock_token_set locktokens = this->lockmgr->lock_recording_refs({
	      { rawimage, false }, // first element is recording_ref, 2nd parameter is false for read, true for write 
	      { result_rec, true },
	    },using_gpu);
	  
	  
	  return std::make_shared<exec_function_override_type>([ this, locktokens,result_rec,rawimage ]() {
	    // exec code
	   

#ifdef SNDE_OPENCL
	    std::shared_ptr<assigned_compute_resource_opencl> opencl_resource = std::dynamic_pointer_cast<assigned_compute_resource_opencl>(this->compute_resource);
	    if (opencl_resource) {

	      cl::Kernel dexela2923_kern = dexela2923_image_transform_function_opencl.get_kernel(opencl_resource->context,opencl_resource->devices.at(0));
	      
	      
	      OpenCLBuffers Buffers(opencl_resource->oclcache,opencl_resource->context,opencl_resource->devices.at(0),locktokens);
	      
		  //assert(recording->ndinfo()->base_index==0); // we don't support a shift (at least not currently)
	      Buffers.AddBufferAsKernelArg(rawimage, dexela2923_kern, 0, false, false); 
	      Buffers.AddBufferAsKernelArg(result_rec,dexela2923_kern,1,true,true);

	      
	      cl::Event kerndone;
	      std::vector<cl::Event> FillEvents=Buffers.FillEvents();
	      
	      cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(dexela2923_kern,{},{1944,256},{},&FillEvents,&kerndone);	      
	      if (err != CL_SUCCESS) {
		throw openclerror(err,"Error enqueueing kernel");
	      }
	      opencl_resource->queues.at(0).flush(); /* trigger execution */
	      // mark that the kernel has modified result_rec
	      Buffers.BufferDirty(result_rec);
	      // wait for kernel execution and transfers to complete
	      Buffers.RemBuffers(kerndone,kerndone,true);
	      
	    } else {	    
#endif // SNDE_OPENCL
	      //snde_warning("Performing colormapping on CPU. This will be slow.");
	      uint16_t *raw_pointer = rawimage->shifted_arrayptr();
	      uint16_t *result_pointer = result_rec->shifted_arrayptr();
	    
	      // !!!*** OpenCL version must generate fortran-ordered
          //recdb.latest.ref["/reorg"].data
	      // output
          memset(result_pointer,0,3888*3072*2);
	      for (snde_index row=0;row < 1944; row++){
		    for (snde_index strippos=0;strippos < 256; strippos++){		 
		      dexela2923_image_transform_row_strippos(raw_pointer,result_pointer,row,strippos);
              }
	      }
#ifdef SNDE_OPENCL
	    }
#endif // SNDE_OPENCL
	    
	    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	    result_rec->rec->mark_data_ready();
	  }); 
	});
      });
    }
    
    
  };
  

 

  std::shared_ptr<math_function> define_dexela2923_image_transform_function()
  {
    return std::make_shared<cpp_math_function>("snde.dexela2923_image_transform",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<dexela2923_image_transform>(rss,inst);

      
      
    }); 
  }

  SNDE_OCL_API std::shared_ptr<math_function> dexela2923_image_transform_function=define_dexela2923_image_transform_function();
  
  static int registered_dexela2923_image_transform_function = register_math_function(dexela2923_image_transform_function);
  

    
};
  
