#include <thread>
#include <cmath>

#include "snde/recstore.hpp"
#include "snde/recmath_cppfunction.hpp"

#include "snde/snde_types_h.h"
#include "snde/recstore_setup.hpp"

#ifdef SNDE_OPENCL
#include "snde/opencl_utils.hpp"
#include "snde/recmath_compute_resource_opencl.hpp"
#include "snde/openclcachemanager.hpp"
#include "snde/recstore_setup_opencl.hpp"
#endif // SNDE_OPENCL

using namespace snde;


const double scalefactor=4.5;

#ifdef SNDE_OPENCL
static opencl_program multiply_by_scalar_opencl("multiply_by_scalar", { snde_types_h, R"RAW(
__kernel void multiply_by_scalar(__global const snde_float32 *input,
                                 snde_float32 factor,
                                 __global snde_float32 *output)
{
  snde_index elemnum = get_global_id(0);
  output[elemnum] = input[elemnum]*factor;
}
)RAW"});
#endif // SNDE_OPENCL

class multiply_by_scalar: public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<snde_float32>>,snde_float64>
{
public:
  multiply_by_scalar(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
    recmath_cppfuncexec(rss,inst)
  {

  }
  
  std::pair<bool,std::shared_ptr<compute_options_function_override_type>> decide_new_revision(std::shared_ptr<ndtyped_recording_ref<snde_float32>> recording, snde_float64 multiplier)
  {
    // This is just a representation of the default
    return std::make_pair(true,nullptr);
  }

  std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<ndtyped_recording_ref<snde_float32>> recording, snde_float64 multiplier)
  {
    snde_index numentries = recording->layout.flattened_length();
    std::vector<std::shared_ptr<compute_resource_option>> option_list =
      {
	std::make_shared<compute_resource_option_cpu>(std::set<std::string>(),
						      0, //metadata_bytes 
						      numentries*sizeof(snde_float32)*2, // data_bytes for transfer
						      numentries, // flops
						      1, // max effective cpu cores
						      1), // useful_cpu_cores (min # of cores to supply
	
#ifdef SNDE_OPENCL
	std::make_shared<compute_resource_option_opencl>(std::set<std::string>(),
							 0, //metadata_bytes 
							 numentries*sizeof(snde_float32)*2, // data_bytes for transfer
							 0, // cpu_flops
							 numentries, // gpu_flops
							 1, // max effective cpu cores
							 1, // useful_cpu_cores (min # of cores to supply
							 false), // requires_doubleprec 
#endif // SNDE_OPENCL
      };
    return std::make_pair(option_list,nullptr);
  }
  
  
  std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<ndtyped_recording_ref<snde_float32>> recording, snde_float64 multiplier) 
  {
    // define_recs code
    printf("define_recs()\n");
    std::shared_ptr<ndtyped_recording_ref<snde_float32>> result_rec;
    result_rec = create_typed_ndarray_ref_math<snde_float32>(get_result_channel_path(0),rss);
    // ***!!! Should provide means to set allocation manager !!!***
    
    return std::make_shared<metadata_function_override_type>([ this,result_rec,recording,multiplier ]() {
      // metadata code
      std::unordered_map<std::string,metadatum> metadata;
      printf("metadata()\n");
      metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));
      
      result_rec->rec->metadata=std::make_shared<immutable_metadata>(metadata);
      result_rec->rec->mark_metadata_done();
      
      return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,recording,multiplier ]() {
	  // lock_alloc code
	  
	  result_rec->allocate_storage(recording->layout.dimlen);

	  // locking is only required for certain recordings
	  // with special storage under certain conditions,
	  // however it is always good to explicitly request
	  // the locks, as the locking is a no-op if
	  // locking is not actually required. 
	  rwlock_token_set locktokens = lockmgr->lock_recording_refs({
	      { recording, false }, // first element is recording_ref, 2nd parameter is false for read, true for write 
	      { result_rec, true },
	    });
	  
	  return std::make_shared<exec_function_override_type>([ this,locktokens, result_rec,recording,multiplier ]() {
	    // exec code
#ifdef SNDE_OPENCL
	    std::shared_ptr<assigned_compute_resource_opencl> opencl_resource=std::dynamic_pointer_cast<assigned_compute_resource_opencl>(compute_resource);
	    if (opencl_resource) {

	      fprintf(stderr,"Executing in OpenCL!\n");
	      cl::Kernel mbs_kern = multiply_by_scalar_opencl.get_kernel(opencl_resource->context,opencl_resource->devices.at(0));
	      OpenCLBuffers Buffers(opencl_resource->oclcache,opencl_resource->context,opencl_resource->devices.at(0),locktokens);
	      
	      Buffers.AddBufferAsKernelArg(recording,mbs_kern,0,false);
	      snde_float32 factor = multiplier;
	      mbs_kern.setArg(1,sizeof(factor),&factor);
	      Buffers.AddBufferAsKernelArg(result_rec,mbs_kern,2,true,true);
	      snde_index numelem = recording->layout.flattened_length();
	      //mbs_kern.setArg(3,sizeof(numelem),&numelem);
	      cl::Event kerndone;
	      std::vector<cl::Event> FillEvents=Buffers.FillEvents();
	      cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(mbs_kern,{},{ numelem },{},&FillEvents,&kerndone);
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
	      fprintf(stderr,"Not executing in OpenCL\n");
	      
	      for (snde_index pos=0;pos < recording->layout.dimlen.at(0);pos++){
		result_rec->element({pos}) = recording->element({pos}) * multiplier;
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




int main(int argc, char *argv[])
{
  size_t len=100;
  
  std::shared_ptr<snde::recdatabase> recdb=std::make_shared<snde::recdatabase>();
  setup_cpu(recdb,{},std::thread::hardware_concurrency());
#ifdef SNDE_OPENCL
  setup_opencl(recdb,{},false,8,nullptr); // limit to 8 parallel jobs. Could replace nullptr with OpenCL platform name
#endif // SNDE_OPENCL
  setup_storage_manager(recdb);
  setup_math_functions(recdb,{});
  recdb->startup();

  
  std::shared_ptr<snde::ndarray_recording_ref> test_rec;  
  std::shared_ptr<math_function> multiply_by_scalar_function = std::make_shared<cpp_math_function>("multiply_by_scalar",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
    return std::make_shared<multiply_by_scalar>(rss,inst);
  });
  
  std::shared_ptr<instantiated_math_function> scaled_channel_function = multiply_by_scalar_function->instantiate({
      std::make_shared<math_parameter_recording>("/test_channel"),
      std::make_shared<math_parameter_double_const>(scalefactor),
    },
    { std::make_shared<std::string>("/scaled channel") },
    "",
    false,
    false,
    false,
    std::make_shared<math_definition>("c++ definition"),
    {},
    nullptr);
  
  
  
  std::shared_ptr<snde::active_transaction> transact=recdb->start_transaction(); // Transaction RAII holder
  
  recdb->add_math_function(transact,scaled_channel_function,false);
  
  std::shared_ptr<snde::channelconfig> testchan_config=std::make_shared<snde::channelconfig>("/test_channel", "main",false);
  
  std::shared_ptr<snde::reserved_channel> testchan = recdb->reserve_channel(transact,testchan_config);
  test_rec = create_ndarray_ref(transact,testchan,SNDE_RTN_FLOAT32);
  std::shared_ptr<snde::globalrevision> globalrev = transact->end_transaction()->globalrev_available();

  test_rec->rec->metadata=std::make_shared<snde::immutable_metadata>();
  test_rec->rec->mark_metadata_done();
  test_rec->allocate_storage(std::vector<snde_index>{len});

  // locking is only required for certain recordings
  // with special storage under certain conditions,
  // however it is always good to explicitly request
  // the locks, as the locking is a no-op if
  // locking is not actually required.
  // Note that requiring locking for read is extremely rare
  // and won't apply to normal channels. Requiring locking
  // for write is relatively common. 
  {
    rwlock_token_set locktokens = recdb->lockmgr->lock_recording_refs({
	{ test_rec, true }, // first element is recording_ref, 2nd parameter is false for read, true for write 
      });
    for (size_t cnt=0;cnt < len; cnt++) {
      test_rec->assign_double({cnt},100.0*sin(cnt));
      
    }
    // locktokens automatically dropped as it goes out of scope
    // must drop before mark_data_ready()

  }
  test_rec->rec->mark_data_ready();

  printf("About to wait_complete()\n");
  fflush(stdout);
  globalrev->wait_complete();

  printf("wait_complete() done\n");
  fflush(stdout);
  std::shared_ptr<ndarray_recording_ref> scaled_rec = globalrev->get_ndarray_ref("/scaled channel");
    
  // verify it is OK to read these channels without locking
  assert(!scaled_rec->ndinfo()->requires_locking_read);
  assert(!test_rec->ndinfo()->requires_locking_read);
  for (size_t cnt=0;cnt < len; cnt++) {
    double math_function_value = scaled_rec->element_double({cnt});
    double recalc_value = (float)(test_rec->element_double({cnt})*scalefactor);
    printf(" %f \t \t %f\n",recalc_value,math_function_value);
    assert(math_function_value == recalc_value);
  }
  
  printf("Exiting.\n");
  return 0;
}
