#include "snde/rec_display_colormap.hpp"
#include "snde/recmath_cppfunction.hpp"
#include "snde/colormap.h"
#include "snde/fusion_colormap.h"


#include "snde/snde_types_h.h"
#include "snde/geometry_types_h.h"
#include "snde/colormap_h.h"
#include "snde/scale_colormap_c.h"
#include "snde/fusion_colormap_c.h"


#ifdef SNDE_OPENCL
#include "snde/opencl_utils.hpp"
#include "snde/openclcachemanager.hpp"
#include "snde/recmath_compute_resource_opencl.hpp"
#endif

// NOTE:
// It would be significantly more efficient to implement all
// of this colormapping as OpenGL shaders. The reason it isn't
// implemented that way is primarily historical. It would both
// save the latency of the calculation math step as well as
// save the CPU and graphics memory used to store the output. 

namespace snde {

#ifdef SNDE_OPENCL


  
  static opencl_program pointcloud_colormap_function_opencl("scale_pointcloud_colormap", { snde_types_h, geometry_types_h, colormap_h, "\ntypedef " + rtn_ocltypemap.at(SNDE_RTN_SNDE_COORD) + " sc_intype;\n",  scale_colormap_c });
#endif // SNDE_OPENCL

  
  
  template <typename T>
  class colormap_recording: public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,int,snde_float64,snde_float64,std::vector<snde_index>,unsigned,unsigned>
  {
  public:
    colormap_recording(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,int,snde_float64,snde_float64,std::vector<snde_index>,unsigned,unsigned>(rss,inst)
    {
      
    }
    
    // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,int,snde_float64,snde_float64,std::vector<snde_index>,unsigned,unsigned>::define_recs_function_override_type define_recs_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,int,snde_float64,snde_float64,std::vector<snde_index>,unsigned,unsigned>::metadata_function_override_type metadata_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,int,snde_float64,snde_float64,std::vector<snde_index>,unsigned,unsigned>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,int,snde_float64,snde_float64,std::vector<snde_index>,unsigned,unsigned>::exec_function_override_type exec_function_override_type;
    
    // just using the default for decide_new_revision

    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<ndtyped_recording_ref<T>> recording, int colormap_type,snde_float64 offset, snde_float64 unitsperintensity,std::vector<snde_index> base_position,unsigned u_dim,unsigned v_dim) 
    {
      snde_index numdatapoints = recording->layout.dimlen.at(u_dim)*recording->layout.dimlen.at(v_dim);

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numdatapoints*(sizeof(T)+sizeof(snde_rgba)), // data_bytes for transfer
							numdatapoints*(10), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
							   0, //metadata_bytes
							   numdatapoints*(sizeof(T)+sizeof(snde_rgba)),
							   0, // cpu_flops
							   numdatapoints*(10), // gpuflops
							   1, // max effective cpu cores
							   1, // useful_cpu_cores (min # of cores to supply
							   std::is_floating_point<T>::value && (sizeof(T) > sizeof(float))), // requires_doubleprec 
#endif // SNDE_OPENCL
	};
      return std::make_pair(option_list,nullptr);
    }


 
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<ndtyped_recording_ref<T>> recording, int colormap_type,snde_float64 offset, snde_float64 unitsperintensity,std::vector<snde_index> base_position,unsigned u_dim,unsigned v_dim) 
    {
      // define_recs code
      //snde_debug(SNDE_DC_APP,"define_recs()");
      // Use of "this" in the next line for the same reason as the typedefs, above
      recording->assert_no_scale_or_offset(this->inst->definition->definition_command);

      std::shared_ptr<ndtyped_recording_ref<snde_rgba>> result_rec = create_typed_ndarray_ref_math<snde_rgba>(this->get_result_channel_path(0),this->rss);
      
      return std::make_shared<metadata_function_override_type>([ this,result_rec,recording,colormap_type,offset,unitsperintensity,base_position,u_dim,v_dim ]() {
	// metadata code
	//std::unordered_map<std::string,metadatum> metadata;
	//snde_debug(SNDE_DC_APP,"metadata()");
	//metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));
	
	result_rec->rec->metadata=recording->rec->metadata;
	result_rec->rec->mark_metadata_done();
	
	return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,recording,colormap_type,offset,unitsperintensity,base_position,u_dim,v_dim ]() {
	  // lock_alloc code
	  result_rec->rec->assign_storage_manager(this->recdb->default_storage_manager); // Force default storage manager so that we DON'T go to the graphics storage (which is unnecessary for temporary output such as this)
	  result_rec->allocate_storage(recording->layout.dimlen,true); // Note fortran order flag -- required by renderer  
	  
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
	      { recording, false }, // first element is recording_ref, 2nd parameter is false for read, true for write 
	      { result_rec, true },
	    },using_gpu);
	  
	  
	  return std::make_shared<exec_function_override_type>([ this, locktokens,result_rec,recording,colormap_type,offset,unitsperintensity,base_position,u_dim,v_dim ]() {
	    // exec code
	    //snde_index flattened_length = recording->layout.flattened_length();
	    //for (snde_index pos=0;pos < flattened_length;pos++){
	    //  result_rec->element(pos) = (recording->element(pos)-offset)/unitsperintensity;
	    //}

#ifdef SNDE_OPENCL
	    std::shared_ptr<assigned_compute_resource_opencl> opencl_resource = std::dynamic_pointer_cast<assigned_compute_resource_opencl>(this->compute_resource);
	    if (opencl_resource) {

	      //cl::Kernel colormap_kern = get_opencl_colormap_program<T>()->get_kernel(opencl_resource->context,opencl_resource->devices.at(0));
	      cl::Kernel colormap_kern = build_typed_opencl_program<T>("snde2.colormap",(std::function<std::shared_ptr<opencl_program>(std::string)>)[] (std::string ocltypename) {
		// OpenCL templating via a typedef....
		return std::make_shared<opencl_program>("scale_colormap", std::vector<std::string>({ snde_types_h, geometry_types_h, colormap_h, "\ntypedef " + ocltypename + " sc_intype;\n", scale_colormap_c }));
	      })->get_kernel(opencl_resource->context,opencl_resource->devices.at(0));
	      
	      OpenCLBuffers Buffers(opencl_resource->oclcache,opencl_resource->context,opencl_resource->devices.at(0),locktokens);
	      
		  //assert(recording->ndinfo()->base_index==0); // we don't support a shift (at least not currently)
		  Buffers.AddBufferAsKernelArg(recording, colormap_kern, 0, false, false);      
	      Buffers.AddBufferAsKernelArg(result_rec,colormap_kern,1,true,true);

	      snde_index stride_u=recording->layout.strides.at(u_dim);
	      snde_index stride_v=recording->layout.strides.at(v_dim);	
		  snde_index stride_w = 0;
		  snde_index DisplayFrame = 0;
	      colormap_kern.setArg(2,sizeof(stride_u),&stride_u);
	      colormap_kern.setArg(3,sizeof(stride_v),&stride_v);
		  if (base_position.size() >= 3) {
			  DisplayFrame = base_position.at(2);
			  stride_w = recording->layout.strides.at(2);
		  }
		  colormap_kern.setArg(4, sizeof(stride_w), &stride_w);
		  colormap_kern.setArg(5, sizeof(DisplayFrame), &DisplayFrame);

	      snde_float32 ocl_offset = (snde_float32)offset;
	      colormap_kern.setArg(6,sizeof(ocl_offset),&ocl_offset);
	      uint8_t ocl_alpha = 255;
	      colormap_kern.setArg(7,sizeof(ocl_alpha),&ocl_alpha);
	      uint32_t ocl_colormap_type = colormap_type;
	      colormap_kern.setArg(8,sizeof(ocl_colormap_type),&ocl_colormap_type);
	      snde_float32 ocl_intensityperunits = (snde_float32)(1.0/unitsperintensity);
	      colormap_kern.setArg(9,sizeof(ocl_intensityperunits),&ocl_intensityperunits);
	      
	      cl::Event kerndone;
	      std::vector<cl::Event> FillEvents=Buffers.FillEvents();
	      
	      cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(colormap_kern,{},{
		  recording->layout.dimlen.at(u_dim),
		  recording->layout.dimlen.at(v_dim)	      
		},{},&FillEvents,&kerndone);	      
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

	      std::vector<snde_index> pos(base_position);
	    
	      // !!!*** OpenCL version must generate fortran-ordered
	      // output
	      for (snde_index vpos=0;vpos < recording->layout.dimlen.at(v_dim);vpos++){
		for (snde_index upos=0;upos < recording->layout.dimlen.at(u_dim);upos++){
		  pos.at(u_dim)=upos;
		  pos.at(v_dim)=vpos;
		  //result_rec->element(pos) = do_colormap(colormap_type,recording->element(pos)-offset)/unitsperintensity;
		  result_rec->element(pos) = snde_colormap(colormap_type,(float)((recording->element(pos)-offset)/unitsperintensity),255);
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
  

  std::shared_ptr<math_function> define_colormap_function()
  {
    return std::make_shared<cpp_math_function>("snde.colormap",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      std::shared_ptr<executing_math_function> executing;

      executing = make_cppfuncexec_floatingtypes<colormap_recording>(rss,inst);
      if (!executing) {
	executing = make_cppfuncexec_integertypes<colormap_recording>(rss,inst);
      }
      if (!executing) {
	throw snde_error("In attempting to call math function %s, first parameter has unsupported data type.",inst->definition->definition_command.c_str());
      }
      return executing;
      
      
      
    }); 
  }


  static int registered_colormap_function = register_math_function(define_colormap_function());
  







    
  class pointcloud_colormap_recording: public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<snde_coord3>>,int,snde_float64,snde_float64>
  {
  public:
    pointcloud_colormap_recording(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<snde_coord3>>,int,snde_float64,snde_float64>(rss,inst)
    {
      
    }
    
    
    // just using the default for decide_new_revision

    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<ndtyped_recording_ref<snde_coord3>> recording, int colormap_type,snde_float64 offset, snde_float64 unitsperintensity) 
    {
      snde_index numdatapoints=1;

      //if (recording->layout.dimlen.at(0) != 3) {
      //throw snde_error("pointcloud_colormap_recording: point cloud first dimension must be 3 (and layout must be Fortran contiguous)");
      //}
      for (size_t dimnum=0;dimnum < recording->layout.dimlen.size();dimnum++) {
	numdatapoints *= recording->layout.dimlen.at(dimnum);
      }
      
      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numdatapoints*(3*sizeof(snde_coord)+sizeof(snde_rgba)), // data_bytes for transfer
							numdatapoints*(10.0), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
							   0, //metadata_bytes
							   numdatapoints*(3*sizeof(snde_coord)+sizeof(snde_rgba)),
							   0, // cpu_flops
							   numdatapoints*(10), // gpuflops
							   1, // max effective cpu cores
							   1, // useful_cpu_cores (min # of cores to supply
							   false), // requires_doubleprec 
#endif // SNDE_OPENCL
	};
      return std::make_pair(option_list,nullptr);
    };
    
    
      
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<ndtyped_recording_ref<snde_coord3>> recording, int colormap_type,snde_float64 offset, snde_float64 unitsperintensity) 
    {
      // define_recs code
      //snde_debug(SNDE_DC_APP,"define_recs()");
      // Use of "this" in the next line for the same reason as the typedefs, above
      recording->assert_no_scale_or_offset(this->inst->definition->definition_command);

      std::shared_ptr<ndtyped_recording_ref<snde_float32>> result_rec = create_typed_ndarray_ref_math<snde_float32>(this->get_result_channel_path(0),this->rss);
      
      return std::make_shared<metadata_function_override_type>([ this,result_rec,recording,colormap_type,offset,unitsperintensity ]() {
	// metadata code
	//std::unordered_map<std::string,metadatum> metadata;
	//snde_debug(SNDE_DC_APP,"metadata()");
	//metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));
	
	result_rec->rec->metadata=recording->rec->metadata;
	result_rec->rec->mark_metadata_done();
	
	return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,recording,colormap_type,offset,unitsperintensity ]() {
	  // lock_alloc code

	  std::vector<snde_index> result_dimlen=recording->layout.dimlen;

	  //result_dimlen[0] = 4; // OSG expects a floating point RGBA array (e.g. osg::Vec4Array) for color

	  // But shouldn't we just to RGBA in uchar format???
	  // right now we just do floating point
	  result_dimlen.insert(result_dimlen.begin(),4);
	  result_rec->rec->assign_storage_manager(this->recdb->default_storage_manager); // Force default storage manager so that we DON'T go to the graphics storage (which is unnecessary for temporary output such as this)

	  result_rec->allocate_storage(result_dimlen,true); // Note fortran order flag -- required by renderer
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
	      { recording, false }, // first element is recording_ref, 2nd parameter is false for read, true for write 
	      { result_rec, true },
	    },using_gpu);
	  
	  
	  return std::make_shared<exec_function_override_type>([ this, locktokens,result_rec,recording,colormap_type,offset,unitsperintensity ]() {
	    // exec code
	    //snde_index flattened_length = recording->layout.flattened_length();
	    //for (snde_index pos=0;pos < flattened_length;pos++){
	    //  result_rec->element(pos) = (recording->element(pos)-offset)/unitsperintensity;
	    //}

	    if (!recording->layout.is_f_contiguous()) {
	      throw snde_error("pointcloud_colormap_recording: input array must be Fortran contiguous");
	    }

	    snde_index numdatapoints=1;
	    
	    for (size_t dimnum=0;dimnum < recording->layout.dimlen.size();dimnum++) {
	      numdatapoints *= recording->layout.dimlen.at(dimnum);
	    }

	    
#ifdef SNDE_OPENCL
	    std::shared_ptr<assigned_compute_resource_opencl> opencl_resource = std::dynamic_pointer_cast<assigned_compute_resource_opencl>(this->compute_resource);
	    if (opencl_resource) {
	      
	      cl::Kernel colormap_kern = pointcloud_colormap_function_opencl.get_kernel(opencl_resource->context,opencl_resource->devices.at(0));
	      OpenCLBuffers Buffers(opencl_resource->oclcache,opencl_resource->context,opencl_resource->devices.at(0),locktokens);
	      assert(recording->ndinfo()->base_index==0); // we don't support a shift (at least not currently)
	      
	      Buffers.AddBufferAsKernelArg(recording,colormap_kern,0,false,false);
	      Buffers.AddBufferAsKernelArg(result_rec,colormap_kern,1,true,true);
	      
	      snde_index stride=1; // number of coord3's  per point
	      colormap_kern.setArg(2,sizeof(stride),&stride);

	      snde_float32 ocl_offset = (snde_float32)offset;
	      colormap_kern.setArg(3,sizeof(ocl_offset),&ocl_offset);
	      snde_float32 ocl_alpha = 1.0f;
	      colormap_kern.setArg(4,sizeof(ocl_alpha),&ocl_alpha);
	      uint32_t ocl_colormap_type = colormap_type;
	      colormap_kern.setArg(5,sizeof(ocl_colormap_type),&ocl_colormap_type);
	      snde_float32 ocl_intensityperunits = (snde_float32)(1.0f/unitsperintensity);
	      colormap_kern.setArg(6,sizeof(ocl_intensityperunits),&ocl_intensityperunits);
	      
	      cl::Event kerndone;
	      std::vector<cl::Event> FillEvents=Buffers.FillEvents();
	      
	      cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(colormap_kern,{},{
		  numdatapoints	      
		},{},&FillEvents,&kerndone);	      
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
	      snde_warning("Performing colormapping on CPU. This will be slow.");
	      snde_index pos=0;// (base_position);
	      for (pos=0;pos < numdatapoints;pos++){
		//result_rec->element(pos*4,true)=0.0;
		//result_rec->element(pos*4+1,true)=1.0;
		//result_rec->element(pos*4+2,true)=1.0;
		//result_rec->element(pos*4+3,true)=1.0;
		snde_colormap_float(colormap_type,(recording->element(pos,true).coord[2]-offset)/unitsperintensity,1.0,&result_rec->element(pos*4,true)); // currently hardwired to colormap coord[2] (z position)
		//snde_warning("val: %f element color: %.2f %.2f %.2f %.2f",(recording->element(pos,true).coord[2]-offset)/unitsperintensity,result_rec->element(pos*4,true),result_rec->element(pos*4+1,true),result_rec->element(pos*4+2,true),result_rec->element(pos*4+3,true));
	      
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
    

  

  std::shared_ptr<math_function> define_pointcloud_colormap_function()
  {
    return std::make_shared<cpp_math_function>("snde.pointcloud_colormap",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<pointcloud_colormap_recording>(rss,inst);
      
    }); 
  }


  static int registered_pointcloud_colormap_function = register_math_function(define_pointcloud_colormap_function());
  



  template <typename T,typename Enable = void>
  struct fusion_colormap {
    
    snde_rgba operator()(int colormap_type,T accumulated,snde_float32 total,
			 snde_float32 offset,snde_float32 unitsperintensity,
			 snde_float32 maxtotal,uint8_t alphaval)
    {
      throw snde_error("fusion_colormap on unsupported type %s",typeid(T).name());
    }
    
  };



  template <typename T>
  struct fusion_colormap<T,typename std::enable_if<std::is_floating_point<T>::value>::type> {
    
    snde_rgba operator()(int colormap_type,T accumulated,snde_float32 total,
			 snde_float32 offset,snde_float32 unitsperintensity,
			 snde_float32 maxtotal,uint8_t alphaval)
    {

      return snde_fusion_colormap_real((snde_float32)accumulated,total,colormap_type,offset,unitsperintensity,maxtotal,alphaval);
				       
    }
    
  };




  template <typename T>
  struct fusion_colormap<T,typename std::enable_if<is_complex<T>::value>::type> {
    
    snde_rgba operator()(int colormap_type,T accumulated,snde_float32 total,
			 snde_float32 offset,snde_float32 unitsperintensity,
			 snde_float32 maxtotal,
			 uint8_t alphaval)
    {

      return snde_fusion_colormap_complex(accumulated,total,colormap_type,offset,unitsperintensity,maxtotal,alphaval);
    }

    
  };

  
  template <>
  struct fusion_colormap<snde_complexfloat64> {
    
    snde_rgba operator()(int colormap_type,snde_complexfloat64 accumulated,snde_float32 total,
			 snde_float32 offset,snde_float32 unitsperintensity,
			 snde_float32 maxtotal,
			 uint8_t alphaval)
    {
      snde_complexfloat32 cf32;
      cf32.real=(snde_float32)accumulated.real;
      cf32.imag=(snde_float32)accumulated.imag;

      return snde_fusion_colormap_complex(cf32,total,colormap_type,offset,unitsperintensity,maxtotal,alphaval);
    }

    
  };



  
  
  template <typename T>
  class fusion_colormapping: public recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,std::shared_ptr<fusion_ndarray_recording>,int,snde_float64,snde_float64,std::vector<snde_index>,unsigned,unsigned>
  {
  public:
    fusion_colormapping(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,std::shared_ptr<fusion_ndarray_recording>,int,snde_float64,snde_float64,std::vector<snde_index>,unsigned,unsigned>(rss,inst)
    {
      
    }
    
    // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,std::shared_ptr<fusion_ndarray_recording>,int,snde_float64,snde_float64,std::vector<snde_index>,unsigned,unsigned>::define_recs_function_override_type define_recs_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,std::shared_ptr<fusion_ndarray_recording>,int,snde_float64,snde_float64,std::vector<snde_index>,unsigned,unsigned>::metadata_function_override_type metadata_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,std::shared_ptr<fusion_ndarray_recording>,int,snde_float64,snde_float64,std::vector<snde_index>,unsigned,unsigned>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<ndtyped_recording_ref<T>>,std::shared_ptr<fusion_ndarray_recording>,int,snde_float64,snde_float64,std::vector<snde_index>,unsigned,unsigned>::exec_function_override_type exec_function_override_type;
    
    // just using the default for decide_new_revision

    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<ndtyped_recording_ref<T>> accumulator, std::shared_ptr<fusion_ndarray_recording> fusion,int colormap_type,snde_float64 offset, snde_float64 unitsperintensity,std::vector<snde_index> base_position,unsigned u_dim,unsigned v_dim) 
    {
      snde_index numdatapoints = accumulator->layout.dimlen.at(u_dim)*accumulator->layout.dimlen.at(v_dim);

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numdatapoints*(sizeof(T)+sizeof(snde_rgba)), // data_bytes for transfer
							numdatapoints*(10), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
							   0, //metadata_bytes
							   numdatapoints*(sizeof(T)+sizeof(snde_rgba)),
							   0, // cpu_flops
							   numdatapoints*(10), // gpuflops
							   1, // max effective cpu cores
							   1, // useful_cpu_cores (min # of cores to supply
							   std::is_floating_point<T>::value && (sizeof(T) > sizeof(float))), // requires_doubleprec 
#endif // SNDE_OPENCL
	};
      return std::make_pair(option_list,nullptr);
    }


 
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<ndtyped_recording_ref<T>> accumulator, std::shared_ptr<fusion_ndarray_recording> fusion, int colormap_type,snde_float64 offset, snde_float64 unitsperintensity,std::vector<snde_index> base_position,unsigned u_dim,unsigned v_dim) 
    {
      // define_recs code
      //snde_debug(SNDE_DC_APP,"define_recs()");
      // Use of "this" in the next line for the same reason as the typedefs, above
      accumulator->assert_no_scale_or_offset(this->inst->definition->definition_command);

      std::shared_ptr<ndtyped_recording_ref<snde_rgba>> result_rec = create_typed_ndarray_ref_math<snde_rgba>(this->get_result_channel_path(0),this->rss);
      
      return std::make_shared<metadata_function_override_type>([ this,result_rec,accumulator,fusion,colormap_type,offset,unitsperintensity,base_position,u_dim,v_dim ]() {
	// metadata code
	//std::unordered_map<std::string,metadatum> metadata;
	//snde_debug(SNDE_DC_APP,"metadata()");
	//metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));
	
	result_rec->rec->metadata=accumulator->rec->metadata;
	result_rec->rec->mark_metadata_done();
	
	return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,accumulator,fusion,colormap_type,offset,unitsperintensity,base_position,u_dim,v_dim ]() {
	  // lock_alloc code
	  
	  result_rec->rec->assign_storage_manager(this->recdb->default_storage_manager); // Force default storage manager so that we DON'T go to the graphics storage (which is unnecessary for temporary output such as this)
	  result_rec->allocate_storage(accumulator->layout.dimlen,true); // Note fortran order flag -- required by renderer
	   

	  std::shared_ptr<ndtyped_recording_ref<snde_float32>> totals = fusion->reference_typed_ndarray<snde_float32>("totals");

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
	      { accumulator, false }, // first element is recording_ref, 2nd parameter is false for read, true for write 
	      { totals, false }, // first element is recording_ref, 2nd parameter is false for read, true for write 
	      { result_rec, true },
	    }, using_gpu);
	  
	  
	  return std::make_shared<exec_function_override_type>([ this, locktokens,result_rec,accumulator,fusion,totals,colormap_type,offset,unitsperintensity,base_position,u_dim,v_dim ]() {
	    // exec code
	    //snde_index flattened_length = recording->layout.flattened_length();
	    //for (snde_index pos=0;pos < flattened_length;pos++){
	    //  result_rec->element(pos) = (recording->element(pos)-offset)/unitsperintensity;
	    //}

	    snde_coord maxtotal=0.0;
	    {
	      snde_float32 *totals_data=totals->shifted_arrayptr();
	      assert(totals->layout.is_contiguous());
	      
	      snde_index totals_size=totals->layout.flattened_size();
	      snde_index pos; 

	      for (pos=0;pos < totals_size;pos++) {
		if (totals_data[pos] > maxtotal) {
		  maxtotal=totals_data[pos];
		}
	      }
	    }

	    
	    bool complexflag=false;
	    if (accumulator->ndinfo()->typenum==SNDE_RTN_COMPLEXFLOAT32 || accumulator->ndinfo()->typenum==SNDE_RTN_COMPLEXFLOAT64 || accumulator->ndinfo()->typenum==SNDE_RTN_COMPLEXFLOAT16 || accumulator->ndinfo()->typenum==SNDE_RTN_SNDE_COMPLEXIMAGEDATA) {
	      complexflag=true; 
	    }
	    
#ifdef SNDE_OPENCL
	    std::shared_ptr<assigned_compute_resource_opencl> opencl_resource = std::dynamic_pointer_cast<assigned_compute_resource_opencl>(this->compute_resource);
	    if (opencl_resource) {

	      //cl::Kernel colormap_kern = get_opencl_colormap_program<T>()->get_kernel(opencl_resource->context,opencl_resource->devices.at(0));
	      std::shared_ptr<opencl_program> colormap_prog = build_typed_opencl_program<T>("snde.fusion_colormap",(std::function<std::shared_ptr<opencl_program>(std::string)>)[ complexflag ] (std::string ocltypename) -> std::shared_ptr<opencl_program> {
		// OpenCL templating via a typedef....
		return std::make_shared<opencl_program>("fusion_colormap_kern", std::vector<std::string>({
		      snde_types_h,
		      geometry_types_h,
		      colormap_h,
		      "\ntypedef " + ocltypename + " fc_intype;\n",
		      complexflag ? ("\ntypedef " + ocltypename + " fc_complex_intype;\n"):("\ntypedef " + ocltypename + "fc_real_intype;\n"),
		      complexflag ? ("\ntypedef snde_float32 fc_real_intype;\n"):("\ntypedef snde_complexfloat32 fc_complex_intype;\n"),
		      complexflag ? ("\n#define snde_fusion_colormap snde_fusion_colormap_complex\n"):("\n#define snde_fusion_colormap snde_fusion_colormap_real\n"),
		      fusion_colormap_c
		    }));
	      });
	      
	      cl::Kernel colormap_kern = colormap_prog->get_kernel(opencl_resource->context,opencl_resource->devices.at(0));
	      
	      OpenCLBuffers Buffers(opencl_resource->oclcache,opencl_resource->context,opencl_resource->devices.at(0),locktokens);
	      
	      //assert(accumulator->ndinfo()->base_index==0); // we don't support a shift (at least not currently) (yes we do)
	      Buffers.AddBufferAsKernelArg(accumulator,colormap_kern,0,false,false);
	      Buffers.AddBufferAsKernelArg(totals,colormap_kern,1,false,false);	      
	      Buffers.AddBufferAsKernelArg(result_rec,colormap_kern,2,true,true);
	      
	      snde_index stride_u=accumulator->layout.strides.at(u_dim);
	      snde_index stride_v=accumulator->layout.strides.at(v_dim);	      
	      colormap_kern.setArg(3,sizeof(stride_u),&stride_u);
	      colormap_kern.setArg(4,sizeof(stride_v),&stride_v);

	      uint32_t ocl_colormap_type = colormap_type;
	      colormap_kern.setArg(5,sizeof(ocl_colormap_type),&ocl_colormap_type);

	      snde_float32 ocl_offset = (snde_float32)offset;
	      colormap_kern.setArg(6,sizeof(ocl_offset),&ocl_offset);
	      
	      snde_float32 ocl_intensityperunits = (snde_float32)(1.0/unitsperintensity);
	      colormap_kern.setArg(7,sizeof(ocl_intensityperunits),&ocl_intensityperunits);

	      snde_float32 maxtotal_float32 = maxtotal;
	      colormap_kern.setArg(8,sizeof(maxtotal_float32),&maxtotal_float32);
	      
	      uint8_t ocl_alpha = 255;
	      colormap_kern.setArg(9,sizeof(ocl_alpha),&ocl_alpha);
	      
	      cl::Event kerndone;
	      std::vector<cl::Event> FillEvents=Buffers.FillEvents();

	      //snde_warning("Performing fusion colormapping");
	      cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(colormap_kern,{},{
		  accumulator->layout.dimlen.at(u_dim),
		  accumulator->layout.dimlen.at(v_dim)	      
		},{},&FillEvents,&kerndone);	      
	      if (err != CL_SUCCESS) {
		throw openclerror(err,"Error enqueueing kernel");
	      }
	      opencl_resource->queues.at(0).flush(); /* trigger execution */
	      // mark that the kernel has modified result_rec
	      Buffers.BufferDirty(result_rec);
	      // wait for kernel execution and transfers to complete
	      Buffers.RemBuffers(kerndone,kerndone,true);

	      /*
	      int num_nonzero=0;
	      for (snde_index vpos=0;vpos < accumulator->layout.dimlen.at(v_dim);vpos++){
		for (snde_index upos=0;upos < accumulator->layout.dimlen.at(u_dim);upos++){
		  if (accumulator->element_complexfloat64({upos,vpos}).real != 0.0) {
		    num_nonzero++;
		  }
		}
	      }
	      snde_warning("Fusion_colormap: num_nonzero=%d",num_nonzero);
	      */
	    } else {	    
#endif // SNDE_OPENCL
	      //snde_warning("Performing colormapping on CPU. This will be slow.");

	      std::vector<snde_index> pos(base_position);
	    
	      // !!!*** OpenCL version must generate fortran-ordered
	      // output
	      for (snde_index vpos=0;vpos < accumulator->layout.dimlen.at(v_dim);vpos++){
		for (snde_index upos=0;upos < accumulator->layout.dimlen.at(u_dim);upos++){
		  pos.at(u_dim)=upos;
		  pos.at(v_dim)=vpos;
		  //result_rec->element(pos) = do_colormap(colormap_type,recording->element(pos)-offset)/unitsperintensity;
		  //result_rec->element(pos) = fusion_colormap(colormap_type,(float)((recording->element(pos)-offset)/unitsperintensity),255);
		  result_rec->element(pos) = fusion_colormap<T>{}(colormap_type,accumulator->element(pos),totals->element(pos),
								  offset,unitsperintensity,maxtotal,255);
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
  




  std::shared_ptr<math_function> define_fusion_colormapping_function()
  {
    return std::make_shared<cpp_math_function>("snde.fusion_colormapping",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      std::shared_ptr<executing_math_function> executing;
      
      executing = make_cppfuncexec_floatingtypes<fusion_colormapping>(rss,inst);
      if (!executing) {
	executing = make_cppfuncexec_complextypes<fusion_colormapping>(rss,inst);
      }
      if (!executing) {
	throw snde_error("In attempting to call math function %s, first parameter has unsupported data type.",inst->definition->definition_command.c_str());
      }
      
      return executing;      
    });
    
  }

  SNDE_OCL_API std::shared_ptr<math_function> fusion_colormapping_function = define_fusion_colormapping_function();


  static int registered_fusion_colormapping_function = register_math_function(fusion_colormapping_function);
  




};


