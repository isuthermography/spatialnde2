#include <math.h>

#include "snde/recmath_cppfunction.hpp"
#include "snde/batched_live_accumulator.hpp"

#ifdef SNDE_OPENCL
#include "snde/opencl_utils.hpp"
#include "snde/openclcachemanager.hpp"
#include "snde/recmath_compute_resource_opencl.hpp"
#endif

#include "snde/vecops.h"

#include "snde/snde_types_h.h"
#include "snde/geometry_types_h.h"
#include "snde/vecops_h.h"


#include "snde/phase_plane_vertex_functions.hpp"

#include "snde/phase_plane_vertex_calcs_c.h"

  


namespace snde {

  template <typename T>
  void phase_plane_vertices_alphas_one(OCL_GLOBAL_ADDR T *complex_inputs,
				       OCL_GLOBAL_ADDR snde_coord3 *tri_vertices,
				       OCL_GLOBAL_ADDR snde_float32 *trivert_colors,
				       T previous_coords,
				       snde_index pos, // within these inputs and these outputs,
				       snde_index totalpos, // for historical_fade, with 0 representing the previous_coords for the first call
				       snde_index totallen, // for historical_fade -- generally the number of end points, or 1 more than the total number of calls to this function
				       snde_float32 linewidth_horiz,
				       snde_float32 linewidth_vert,
				       snde_float32 R,
				       snde_float32 G,
				       snde_float32 B,
				       snde_float32 A,
				       snde_bool historical_fade)
  {
    throw snde_error("phase_plane_vertices_alphas_one not implemented for type %s",typeid(T).name());
  }

  // Specializations implemented by including C file with an appropriate define
#define PPVAO_DECL template<>
  
#define ppvao_intype snde_complexfloat32
#define phase_plane_vertices_alphas_one phase_plane_vertices_alphas_one<snde_complexfloat32>
#include "phase_plane_vertex_calcs.c"  
#undef ppvao_intype
#undef phase_plane_vertices_alphas_one  
  

#define ppvao_intype snde_complexfloat64
#define phase_plane_vertices_alphas_one phase_plane_vertices_alphas_one<snde_complexfloat64>
#include "phase_plane_vertex_calcs.c"
#undef ppvao_intype 
#undef phase_plane_vertices_alphas_one  
  
  
  

  template <typename T>  // template for different floating point complex number classes 
  class phase_plane_line_triangle_vertices_alphas: public recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>,double,double,double,double,double,double>
  {
  public:
    phase_plane_line_triangle_vertices_alphas(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>,double,double,double,double,double,double>(rss,inst)
    {
      
    }
    

    // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
    typedef typename recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>,double,double,double,double>::define_recs_function_override_type define_recs_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>,double,double,double,double>::metadata_function_override_type metadata_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>,double,double,double,double>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>,double,double,double,double>::exec_function_override_type exec_function_override_type;

    // just using the default for decide_new_revision


    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<multi_ndarray_recording> recording,double R,double G,double B,double A,double linewidth_horiz,double linewidth_vert)
    {

    
      std::vector<snde_index> layout_dims;
      snde_index layout_length=0;
      
      bool consistent_layout_c=true; // consistent_layout_c means multiple arrays but all with the same layout except for the first axis which is implicitly concatenated
      bool consistent_layout_f=true; // consistent_layout_f means multiple arrays but all with the same layout except for the last axis which is implicitly concatenated
      bool consistent_ndim=true; 
      
      std::tie(consistent_ndim,consistent_layout_c,consistent_layout_f,layout_dims,layout_length)
	= analyze_potentially_batched_multi_ndarray_layout(recording);
      
      
      assert(consistent_ndim==1 && layout_dims.size()==0);  // 1D only

      T junk={0,0};
      
      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							layout_length * sizeof(snde_coord)*3*6*2, // data_bytes for transfer
							0.0, // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
							   0, //metadata_bytes
							   layout_length * sizeof(snde_coord)*3*6*2,
							   0.0, // cpu_flops
							   0.0, // gpuflops
							   1, // max effective cpu cores
							   1, // useful_cpu_cores (min # of cores to supply
							   sizeof(junk.real) > 4), // requires_doubleprec 
#endif // SNDE_OPENCL
	};
      return std::make_pair(option_list,nullptr);
    }
    
    
    
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<multi_ndarray_recording> recording,double R,double G,double B,double A,double linewidth_horiz,double linewidth_vert)
    {

      std::vector<snde_index> layout_dims;
      snde_index layout_length=0;
      
      bool consistent_layout_c=true; // consistent_layout_c means multiple arrays but all with the same layout except for the first axis which is implicitly concatenated
      bool consistent_layout_f=true; // consistent_layout_f means multiple arrays but all with the same layout except for the last axis which is implicitly concatenated
      bool consistent_ndim=true; 
      recording->assert_no_scale_or_offset(this->inst->definition->definition_command);

      std::tie(consistent_ndim,consistent_layout_c,consistent_layout_f,layout_dims,layout_length)
	= analyze_potentially_batched_multi_ndarray_layout(recording);
 
      // define_recs code
      //snde_debug(SNDE_DC_APP,"define_recs()");
      // Use of "this" in the next line for the same reason as the typedefs, above

      std::shared_ptr<multi_ndarray_recording> result_rec = create_recording_math<multi_ndarray_recording>(this->get_result_channel_path(0),this->rss,2);
      result_rec->define_array(0,SNDE_RTN_SNDE_COORD3,"vertcoord");
      result_rec->define_array(1,SNDE_RTN_FLOAT32,"vertcoord_color");
      
      return std::make_shared<metadata_function_override_type>([ this,result_rec,recording,R,G,B,A,layout_length,linewidth_horiz,linewidth_vert ]() {
	// metadata code
	//std::unordered_map<std::string,metadatum> metadata;
	//snde_debug(SNDE_DC_APP,"metadata()");
	//metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));
	
	result_rec->metadata=std::make_shared<immutable_metadata>();
	result_rec->mark_metadata_done();
	
	return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,recording,R,G,B,A,layout_length,linewidth_horiz,linewidth_vert ]() {
	  // lock_alloc code
	  
	  result_rec->allocate_storage("vertcoord",{(layout_length-1)*6},false);
	  result_rec->allocate_storage("vertcoord_color",{(layout_length-1)*6*4},false);
	  

	  // locking is only required for certain recordings
	  // with special storage under certain conditions,
	  // however it is always good to explicitly request
	  // the locks, as the locking is a no-op if
	  // locking is not actually required.

	  // lock our output arrays
	  std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>> recrefs_to_lock = {
	    { result_rec, { 0, true } }, // vertcoord
	    { result_rec, { 1, true } }, // vertcoord_color
	    
	  };

	  // ... and all the input arrays. 
	  for (size_t arraynum=0;arraynum < recording->mndinfo()->num_arrays;arraynum++) {
	    recrefs_to_lock.emplace_back(std::make_pair(recording,std::make_pair(arraynum,false)));
	  }
	  
	  rwlock_token_set locktokens = this->lockmgr->lock_recording_arrays(recrefs_to_lock,
#ifdef SNDE_OPENCL
									     true
#else
									     false
#endif
									     );
	  
	  return std::make_shared<exec_function_override_type>([ this,locktokens,result_rec,recording, R, G, B, A, layout_length,linewidth_horiz,linewidth_vert ]() {
	    // exec code
	    //snde_index flattened_length = recording->layout.flattened_length();
	    //for (snde_index pos=0;pos < flattened_length;pos++){
	    //  result_rec->element(pos) = (recording->element(pos)-offset)/unitsperintensity;
	    //}

	    snde_bool phase_plane_historical_fade=recording->metadata->GetMetaDatumBool("snde_phase_plane_historical_fade",false);
	    
#ifdef SNDE_OPENCL
	    std::shared_ptr<assigned_compute_resource_opencl> opencl_resource=std::dynamic_pointer_cast<assigned_compute_resource_opencl>(compute_resource);
	    if (opencl_resource && recording->mndinfo()->num_arrays > 0) {
	      
	      //fprintf(stderr,"Executing in OpenCL!\n");

	      cl::Kernel phase_plane_vert_kern = build_typed_opencl_program<T>("snde.colormap",(std::function<std::shared_ptr<opencl_program>(std::string)>)[] (std::string ocltypename) {
		// OpenCL templating via a typedef....
		return std::make_shared<opencl_program>("phase_plane_vertices_alphas", std::vector<std::string>({ snde_types_h, geometry_types_h,vecops_h,"\ntypedef " + ocltypename + " ppvao_intype;\n", phase_plane_vertex_calcs_c }));
	      })->get_kernel(opencl_resource->context,opencl_resource->devices.at(0));
	      
	      OpenCLBuffers Buffers(opencl_resource->oclcache,opencl_resource->context,opencl_resource->devices.at(0),locktokens);

	      snde_index output_pos=0;
	      T previous_coords = { 0,0 };
	      snde_float32 R_fl=(snde_float32)R;
	      snde_float32 G_fl=(snde_float32)G;
	      snde_float32 B_fl=(snde_float32)B;
	      snde_float32 A_fl=(snde_float32)A;
	      snde_float32 linewidth_horiz_fl=(snde_float32)linewidth_horiz;
	      snde_float32 linewidth_vert_fl=(snde_float32)linewidth_vert;
	      
	      std::vector<cl::Event> kerndoneevents;
	      
	      for (size_t arraynum=0;arraynum < recording->mndinfo()->num_arrays;arraynum++) {
		snde_index input_pos=0;
		snde_index input_length = recording->layouts.at(arraynum).dimlen.at(0);
		snde_index output_length = input_length*6;
		snde_index totalpos = output_pos+1;
		snde_index totallen = layout_length*6;

		if (input_length < 2) {
		  continue; // need at least two points to plot.
		}
		if (!output_pos) {
		  // first iteration: Use first element as previous value
		  input_length -= 1;
		  output_length -= 6;
		  input_pos += 1;

		  previous_coords = recording->reference_typed_ndarray<T>(arraynum)->element(0);
		}
		Buffers.AddBufferPortionAsKernelArg(recording,arraynum,input_pos,input_length,phase_plane_vert_kern,0,false,false);
		Buffers.AddBufferPortionAsKernelArg(result_rec,"vertcoord",output_pos,output_length,phase_plane_vert_kern,1,true,true);
		Buffers.AddBufferPortionAsKernelArg(result_rec,"vertcoord_color",output_pos*4,output_length*4,phase_plane_vert_kern,2,true,true);
		phase_plane_vert_kern.setArg(3,sizeof(previous_coords),&previous_coords);
		phase_plane_vert_kern.setArg(4,sizeof(totalpos),&totalpos);
		phase_plane_vert_kern.setArg(5,sizeof(totallen),&totallen);
		phase_plane_vert_kern.setArg(6,sizeof(linewidth_horiz_fl),&linewidth_horiz_fl);
		phase_plane_vert_kern.setArg(7,sizeof(linewidth_vert_fl),&linewidth_vert_fl);
		phase_plane_vert_kern.setArg(8,sizeof(R_fl),&R_fl);
		phase_plane_vert_kern.setArg(9,sizeof(G_fl),&G_fl);
		phase_plane_vert_kern.setArg(10,sizeof(B_fl),&B_fl);
		phase_plane_vert_kern.setArg(11,sizeof(A_fl),&A_fl);
		phase_plane_vert_kern.setArg(12,sizeof(phase_plane_historical_fade),&phase_plane_historical_fade);

		cl::Event kerndone;
		std::vector<cl::Event> FillEvents=Buffers.FillEvents();

		cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(phase_plane_vert_kern,{},{ input_length },{},&FillEvents,&kerndone);
		if (err != CL_SUCCESS) {
		  throw openclerror(err,"Error enqueueing kernel");
		}

		Buffers.BufferPortionDirty(result_rec,"vertcoord",output_pos,output_length);
		Buffers.BufferPortionDirty(result_rec,"vertcoord_color",output_pos*4,output_length*4);
		kerndoneevents.push_back(kerndone);

		
		previous_coords = recording->reference_typed_ndarray<T>(arraynum)->element(recording->layouts.at(arraynum).dimlen.at(0)-1);
		output_pos += output_length;
	      }
	      
	      opencl_resource->queues.at(0).flush(); /* trigger execution */
	      // mark that the kernel has modified result_rec
	      // wait for kernel execution and transfers to complete

	      cl::Event::waitForEvents(kerndoneevents);
	      Buffers.RemBuffers(*(kerndoneevents.end()-1),*(kerndoneevents.end()-1),true);
	    
	    } else {	    
#endif // SNDE_OPENCL
	      snde_warning("Performing phase plane vertex calculation on CPU. ");



	      snde_index output_pos=0;
	      T previous_coords = { 0,0 };

	      
	      
	      for (size_t arraynum=0;arraynum < recording->mndinfo()->num_arrays;arraynum++) {
		snde_index input_pos=0;
		snde_index input_length = recording->layouts.at(arraynum).dimlen.at(0);
		snde_index output_length = input_length*6;
		if (!output_pos) {
		  // first iteration: Use first element as previous value
		  input_length -= 1;
		  output_length -= 6;
		  input_pos += 1;
		  previous_coords = recording->reference_typed_ndarray<T>(arraynum)->element(0);


		}


		for (snde_index cnt=0;cnt < input_length; cnt++) {
		  phase_plane_vertices_alphas_one<T>(((T *)recording->void_shifted_arrayptr(arraynum))+input_pos,
						     ((snde_coord3 *)result_rec->void_shifted_arrayptr("vertcoord")) + output_pos,
						     ((snde_float32 *)result_rec->void_shifted_arrayptr("vertcoord_color")) + output_pos*4,
						     previous_coords,
						     cnt,
						     output_pos+cnt+1,
						     layout_length*6,
						     linewidth_horiz,
						     linewidth_vert,
						     R,
						     G,
						     B,
						     A,
						     phase_plane_historical_fade);
		}
		
		previous_coords = recording->reference_typed_ndarray<T>(arraynum)->element(recording->layouts.at(arraynum).dimlen.at(0)-1);
		output_pos += output_length;

	      }
	      
	      
#ifdef SNDE_OPENCL
	    }
#endif // SNDE_OPENCL
	    
	    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	    result_rec->mark_data_ready();
	    
	  }); 
	}); 
      });
    }
  };
  
  
  std::shared_ptr<math_function> define_phase_plane_line_triangle_vertices_alphas_function()
  {
    return std::make_shared<cpp_math_function>("snde.phase_plane_line_triangle_vertices_alphas",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return make_cppfuncexec_complextypes<phase_plane_line_triangle_vertices_alphas>(rss,inst);
    });
    
  }

  SNDE_OCL_API std::shared_ptr<math_function> phase_plane_line_triangle_vertices_alphas_function = define_phase_plane_line_triangle_vertices_alphas_function();
  
  static int registered_phase_plane_line_triangle_vertices_alphas_function = register_math_function(phase_plane_line_triangle_vertices_alphas_function);
  
  


  
  class phase_plane_endpoint_octagon_vertices: public recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>,double,double> {
  public:
    phase_plane_endpoint_octagon_vertices(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec(rss,inst)
    {
      
    }
    
    // use default for decide_new_revision
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<multi_ndarray_recording> rec, double radius_horiz, double radius_vert)
    {
      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							0, // data_bytes for transfer
							0.0f, // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
	};
      return std::make_pair(option_list,nullptr);
    }
    
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<multi_ndarray_recording> rec,double radius_horiz,double radius_vert) 
    {
      // define_recs code
      //printf("define_recs()\n");
      rec->assert_no_scale_or_offset(this->inst->definition->definition_command);

      std::shared_ptr<multi_ndarray_recording> result_rec = create_recording_math<multi_ndarray_recording>(this->get_result_channel_path(0),this->rss,1);
      result_rec->define_array(0,SNDE_RTN_SNDE_COORD3,"vertcoord");
      //result_rec->define_array(0,SNDE_RTN_SNDE_FLOAT32,"vertcoord_color");

    
      return std::make_shared<metadata_function_override_type>([ this,result_rec,rec,radius_horiz,radius_vert ]() {
	// metadata code
	std::unordered_map<std::string,metadatum> metadata;
	
	result_rec->metadata=std::make_shared<immutable_metadata>(metadata);
	result_rec->mark_metadata_done();
	
	return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,rec,radius_horiz,radius_vert ]() {
	  // lock_alloc code 
	  size_t num_arrays = rec->mndinfo()->num_arrays; 

	  if (num_arrays > 0) {
	    result_rec->allocate_storage("vertcoord",{8*3},false);
	  } else {
	    result_rec->allocate_storage("vertcoord",{0},false);
	  }
	  
	  std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>> recrefs_to_lock = {
	    { result_rec, { 0, true } }, // vertcoord
	    //{ result_rec, { 1, true } }, // vertcoord_color
	    
	  };

	  // ... and all the last input arrays.
	  if (num_arrays > 0) { 
	    size_t arraynum = num_arrays-1; 
	    recrefs_to_lock.emplace_back(std::make_pair(rec,std::make_pair(arraynum,false)));
	  }

	  rwlock_token_set locktokens = this->lockmgr->lock_recording_arrays(recrefs_to_lock,false);

	  
	  return std::make_shared<exec_function_override_type>([ this,locktokens, result_rec, rec, num_arrays,radius_horiz,radius_vert ]() {
	    if (num_arrays > 0) {
	      snde_index final_length = rec->layouts.at(num_arrays-1).dimlen.at(0);
	      assert(final_length > 0);
	      snde_index idx = final_length-1; 
	      
	      snde_complexfloat64 centerpos = rec->element_complexfloat64(num_arrays-1,idx);
	      
	      std::shared_ptr<ndtyped_recording_ref<snde_coord3>> vertcoord_result_ref=result_rec->reference_typed_ndarray<snde_coord3>(0);
	    
	      for (unsigned trinum=0; trinum < 8; trinum++) {
	      // vertices in CCW order for this pie-slice
		vertcoord_result_ref->element(trinum*3 + 0).coord[0] = centerpos.real;
		vertcoord_result_ref->element(trinum*3 + 0).coord[1] = centerpos.imag;
		vertcoord_result_ref->element(trinum*3 + 0).coord[2] = 0.0f;
		
		vertcoord_result_ref->element(trinum*3 + 1).coord[0] = centerpos.real + radius_horiz*cos(M_PI/4.0*trinum);
		vertcoord_result_ref->element(trinum*3 + 1).coord[1] = centerpos.imag + radius_vert*sin(M_PI/4.0*trinum);
		vertcoord_result_ref->element(trinum*3 + 1).coord[2] = 0.0f;
		
		vertcoord_result_ref->element(trinum*3 + 2).coord[0] = centerpos.real + radius_horiz*cos(M_PI/4.0*(trinum+1));
		vertcoord_result_ref->element(trinum*3 + 2).coord[1] = centerpos.imag + radius_vert*sin(M_PI/4.0*(trinum+1));
		vertcoord_result_ref->element(trinum*3 + 2).coord[2] = 0.0f;
		
	      }
	    }
	    
	    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	    result_rec->mark_data_ready();
	    
	  }); 
	});
      });
    };
    
  };
  
  
  std::shared_ptr<math_function> define_spatialnde2_phase_plane_endpoint_octagon_vertices_function()
  {
    return std::make_shared<cpp_math_function>("snde.phase_plane_endpoint_octagon_vertices",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<phase_plane_endpoint_octagon_vertices>(rss,inst);
    }); 
  }

  SNDE_OCL_API std::shared_ptr<math_function> phase_plane_endpoint_octagon_vertices_function = define_spatialnde2_phase_plane_endpoint_octagon_vertices_function();
  
  static int registered_phase_plane_endpoint_octagon_vertices_function = register_math_function(phase_plane_endpoint_octagon_vertices_function);
  
  

  
  
  
};


