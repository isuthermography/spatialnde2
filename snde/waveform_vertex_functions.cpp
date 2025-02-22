#include <math.h>

#include "snde/recmath_cppfunction.hpp"

#ifdef SNDE_OPENCL
#include "snde/opencl_utils.hpp"
#include "snde/openclcachemanager.hpp"
#include "snde/recmath_compute_resource_opencl.hpp"
#endif

#include "snde/snde_types_h.h"
#include "snde/geometry_types_h.h"


#include "snde/waveform_vertex_functions.hpp"

#include "snde/waveform_vertex_calcs_c.h"




namespace snde {

  template <typename T>
  void waveform_as_interplines(OCL_GLOBAL_ADDR T* inputs,
    OCL_GLOBAL_ADDR snde_coord3* tri_vertices,
    OCL_GLOBAL_ADDR snde_float32* trivert_colors,
    snde_index cnt, // within these inputs and these outputs,
    snde_index pos,
    float inival,
    float step,
    snde_float32 linewidth_horiz,
    snde_float32 linewidth_vert,
    snde_float32 R,
    snde_float32 G,
    snde_float32 B,
    snde_float32 A)
  {
    throw snde_error("waveform_as_interplines not implemented for type %s", typeid(T).name());
  }

  template <typename T>
  void waveform_as_vertlines(OCL_GLOBAL_ADDR T* inputs,
    OCL_GLOBAL_ADDR snde_coord3* tri_vertices,
    OCL_GLOBAL_ADDR snde_float32* trivert_colors,
    snde_index cnt, // within these inputs and these outputs,
    snde_index startidx,
    snde_index endidx,
    float inival,
    float step,
    snde_index pxstep,
    snde_float32 linewidth_horiz,
    snde_float32 linewidth_vert,
    snde_float32 R,
    snde_float32 G,
    snde_float32 B,
    snde_float32 A)
  {
    throw snde_error("waveform_as_vertlines not implemented for type %s", typeid(T).name());
  }

  template <typename T>
  void waveform_as_points(OCL_GLOBAL_ADDR T* inputs,
    OCL_GLOBAL_ADDR snde_coord3* tri_vertices,
    OCL_GLOBAL_ADDR snde_float32* trivert_colors,
    snde_index cnt, // within these inputs and these outputs,
    snde_index pos,
    float inival,
    float step,
    snde_float32 R,
    snde_float32 G,
    snde_float32 B,
    snde_float32 A)
  {
    throw snde_error("waveform_as_points not implemented for type %s", typeid(T).name());
  }


  // Specializations implemented by including C file with an appropriate define
#define WAVEFORM_DECL template<>

#define waveform_intype snde_float32
#define waveform_as_interplines waveform_as_interplines<snde_float32>
#define waveform_as_vertlines waveform_as_vertlines<snde_float32>
#define waveform_as_points waveform_as_points<snde_float32>
#include "waveform_vertex_calcs.c"  
#undef waveform_intype
#undef waveform_as_interplines  
#undef waveform_as_vertlines
#undef waveform_as_points


#define waveform_intype snde_float64
#define waveform_as_interplines waveform_as_interplines<snde_float64>
#define waveform_as_vertlines waveform_as_vertlines<snde_float64>
#define waveform_as_points waveform_as_points<snde_float64>
#include "waveform_vertex_calcs.c"
#undef waveform_intype
#undef waveform_vertices_alphas_one  
#undef waveform_as_interplines  
#undef waveform_as_vertlines
#undef waveform_as_points

  template <typename T>  // template for different floating point number classes 
  class waveform_interplines : public recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>, double, double, double, double, double, double, snde_index, snde_index, snde_index, double, double>
  {
  public:
    waveform_interplines(std::shared_ptr<recording_set_state> rss, std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>, double, double, double, double, double, double, snde_index, snde_index, snde_index, double, double>(rss, inst)
    {

    }


    // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
    typedef typename recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>, double, double, double, double, double, double, snde_index, snde_index, snde_index, double, double>::define_recs_function_override_type define_recs_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>, double, double, double, double, double, double, snde_index, snde_index, snde_index, double, double>::metadata_function_override_type metadata_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>, double, double, double, double, double, double, snde_index, snde_index, snde_index, double, double>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>, double, double, double, double, double, double, snde_index, snde_index, snde_index, double, double>::exec_function_override_type exec_function_override_type;

    // just using the default for decide_new_revision

    std::pair<std::vector<std::shared_ptr<compute_resource_option>>, std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<multi_ndarray_recording> recording, double R, double G, double B, double A, double linewidth_horiz, double linewidth_vert, snde_index startidx, snde_index endidx, snde_index idxstep, double datainival, double datastep)
    {
      bool consistent_layout_c = true; // consistent_layout_c means multiple arrays but all with the same layout except for the first axis which is implicitly concatenated
      bool consistent_layout_f = true; // consistent_layout_f means multiple arrays but all with the same layout except for the last axis which is implicitly concatenated
      bool consistent_ndim = true;

      assert(recording->layouts.size() == 1 && recording->layouts.at(0).dimlen.size() == 1);

      T junk = 0.0;

      snde_index outlen = (endidx - startidx);  // Double check this once the code below is written

	  if (outlen < 1)
		  outlen = 1;

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
      {
	std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
						      0, //metadata_bytes 
						      outlen * sizeof(snde_coord) * 3 * 6 + outlen * sizeof(snde_coord) * 3 * 6 * 4, // data_bytes for transfer
						      0.0, // flops
						      1, // max effective cpu cores
						      1), // useful_cpu_cores (min # of cores to supply

#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
							   0, //metadata_bytes
							   outlen * sizeof(snde_coord) * 3 * 6 + outlen * sizeof(snde_coord) * 3 * 6 * 4,
							   0.0, // cpu_flops
							   0.0, // gpuflops
							   1, // max effective cpu cores
							   1, // useful_cpu_cores (min # of cores to supply
							   sizeof(junk) > 4), // requires_doubleprec 
#endif // SNDE_OPENCL
      };
      return std::make_pair(option_list, std::make_shared<define_recs_function_override_type>([this, recording, R, G, B, A, linewidth_horiz, linewidth_vert, startidx, endidx, idxstep, datainival, datastep, outlen]() {

	// define_recs code
	//snde_debug(SNDE_DC_APP,"define_recs()");
	// Use of "this" in the next line for the same reason as the typedefs, above
	std::shared_ptr<multi_ndarray_recording> result_rec = create_recording_math<multi_ndarray_recording>(this->get_result_channel_path(0), this->rss, 2);
	result_rec->define_array(0, SNDE_RTN_SNDE_COORD3, "vertcoord");
	result_rec->define_array(1, SNDE_RTN_FLOAT32, "vertcoord_color");

	return std::make_shared<metadata_function_override_type>([this, result_rec, recording, R, G, B, A, linewidth_horiz, linewidth_vert, startidx, endidx, idxstep, datainival, datastep, outlen]() {
	  // metadata code
	  //std::unordered_map<std::string,metadatum> metadata;
	  //snde_debug(SNDE_DC_APP,"metadata()");
	  //metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));

	  result_rec->metadata = std::make_shared<immutable_metadata>();
	  result_rec->mark_metadata_done();

	  return std::make_shared<lock_alloc_function_override_type>([this, result_rec, recording, R, G, B, A, linewidth_horiz, linewidth_vert, startidx, endidx, idxstep, datainival, datastep, outlen]() {
	    // lock_alloc code

	    result_rec->allocate_storage("vertcoord", { (outlen) * 6 }, false);
	    result_rec->allocate_storage("vertcoord_color", { (outlen) * 6 * 4 }, false);

	    // locking is only required for certain recordings
	    // with special storage under certain conditions,
	    // however it is always good to explicitly request
	    // the locks, as the locking is a no-op if
	    // locking is not actually required.

	    // lock our output arrays
	    std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>, std::pair<size_t, bool>>> recrefs_to_lock = {
	      { result_rec, { 0, true } }, // vertcoord
	      { result_rec, { 1, true } }, // vertcoord_color
	    };

	    // ... and all the input arrays. 
	    for (size_t arraynum = 0; arraynum < recording->mndinfo()->num_arrays; arraynum++) {
	      recrefs_to_lock.emplace_back(std::make_pair(recording, std::make_pair(arraynum, false)));
	    }

	    rwlock_token_set locktokens = this->lockmgr->lock_recording_arrays(recrefs_to_lock,
#ifdef SNDE_OPENCL
	      true
#else
	      false
#endif
	    );

	    return std::make_shared<exec_function_override_type>([this, locktokens, result_rec, recording, R, G, B, A, linewidth_horiz, linewidth_vert, startidx, endidx, idxstep, datainival, datastep, outlen]() {
	      // exec code
	      //snde_index flattened_length = recording->layout.flattened_length();
	      //for (snde_index pos=0;pos < flattened_length;pos++){
	      //  result_rec->element(pos) = (recording->element(pos)-offset)/unitsperintensity;
	      //}



#ifdef SNDE_OPENCL
	      std::shared_ptr<assigned_compute_resource_opencl> opencl_resource = std::dynamic_pointer_cast<assigned_compute_resource_opencl>(compute_resource);
	      if (opencl_resource && recording->mndinfo()->num_arrays > 0) {

		//fprintf(stderr,"Executing in OpenCL!\n");

		cl::Kernel waveform_interplines_vert_kern = build_typed_opencl_program<T>("snde.waveform_interplines", (std::function<std::shared_ptr<opencl_program>(std::string)>)[](std::string ocltypename) {
		  // OpenCL templating via a typedef....
		  return std::make_shared<opencl_program>("waveform_interplines", std::vector<std::string>({ snde_types_h, geometry_types_h, "\ntypedef " + ocltypename + " waveform_intype;\n", waveform_vertex_calcs_c }));
		  })->get_kernel(opencl_resource->context, opencl_resource->devices.at(0));

		  OpenCLBuffers Buffers(opencl_resource->oclcache, opencl_resource->context, opencl_resource->devices.at(0), locktokens);

		  snde_index curpos = (snde_index)(startidx + 1);
		  snde_float32 R_fl = (snde_float32)R;
		  snde_float32 G_fl = (snde_float32)G;
		  snde_float32 B_fl = (snde_float32)B;
		  snde_float32 A_fl = (snde_float32)A;
		  snde_float32 linewidth_horiz_fl = (snde_float32)linewidth_horiz;
		  snde_float32 linewidth_vert_fl = (snde_float32)linewidth_vert;

		  std::vector<cl::Event> kerndoneevents;

		  
		  Buffers.AddBufferPortionAsKernelArg(recording, 0, 0, recording->layouts.at(0).dimlen.at(0), waveform_interplines_vert_kern, 0, false, false);
		  Buffers.AddBufferPortionAsKernelArg(result_rec, "vertcoord", 0, outlen * 6, waveform_interplines_vert_kern, 1, true, true);
		  Buffers.AddBufferPortionAsKernelArg(result_rec, "vertcoord_color", 0, outlen * 6 * 4, waveform_interplines_vert_kern, 2, true, true);
		  //waveform_interplines_vert_kern.setArg(3, sizeof(cnt), &cnt);
		  waveform_interplines_vert_kern.setArg(3, sizeof(curpos), &curpos);
		  float datainival_float = datainival;
		  waveform_interplines_vert_kern.setArg(4, sizeof(datainival_float), &datainival_float);
		  float datastep_float = datastep;
		  waveform_interplines_vert_kern.setArg(5, sizeof(datastep_float), &datastep_float);
		  waveform_interplines_vert_kern.setArg(6, sizeof(linewidth_horiz_fl), &linewidth_horiz_fl);
		  waveform_interplines_vert_kern.setArg(7, sizeof(linewidth_vert_fl), &linewidth_vert_fl);
		  waveform_interplines_vert_kern.setArg(8, sizeof(R_fl), &R_fl);
		  waveform_interplines_vert_kern.setArg(9, sizeof(G_fl), &G_fl);
		  waveform_interplines_vert_kern.setArg(10, sizeof(B_fl), &B_fl);
		  waveform_interplines_vert_kern.setArg(11, sizeof(A_fl), &A_fl);

		  cl::Event kerndone;
		  std::vector<cl::Event> FillEvents = Buffers.FillEvents();

		  cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(waveform_interplines_vert_kern, {}, { outlen }, {}, &FillEvents, &kerndone);
		  if (err != CL_SUCCESS) {
		    throw openclerror(err, "Error enqueueing kernel");
		  }

		  Buffers.BufferPortionDirty(result_rec, "vertcoord", 0, outlen * 6);
		  Buffers.BufferPortionDirty(result_rec, "vertcoord_color", 0, outlen * 6 * 4);
		  kerndoneevents.push_back(kerndone);


		  opencl_resource->queues.at(0).flush(); /* trigger execution */
		  // mark that the kernel has modified result_rec
		  // wait for kernel execution and transfers to complete

		  cl::Event::waitForEvents(kerndoneevents);
		  Buffers.RemBuffers(*(kerndoneevents.end() - 1), *(kerndoneevents.end() - 1), true);

	      }
	      else {
#endif // SNDE_OPENCL
		//snde_warning("Performing waveform vertex calculation on CPU. ");

		T junk = 0.0;

		// Plot Normal Lines
		for (snde_index cnt = 0; cnt < outlen; cnt++) {
		  waveform_as_interplines<T>(((T*)recording->void_shifted_arrayptr(0)),
		    ((snde_coord3*)result_rec->void_shifted_arrayptr("vertcoord")),
		    ((snde_float32*)result_rec->void_shifted_arrayptr("vertcoord_color")),
		    cnt,
		    startidx + 1,
		    datainival,
		    datastep,
		    linewidth_horiz,
		    linewidth_vert,
		    R,
		    G,
		    B,
		    A);
		}
#ifdef SNDE_OPENCL
	      }
#endif // SNDE_OPENCL

	      unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	      result_rec->mark_data_ready();

	      });
	    });
	  });
	}));
    };
  };



  std::shared_ptr<math_function> define_waveform_interplines_function()
  {
    return std::make_shared<cpp_math_function>("snde.waveform_interplines",1,[](std::shared_ptr<recording_set_state> rss, std::shared_ptr<instantiated_math_function> inst) {
      return make_cppfuncexec_floatingtypes<waveform_interplines>(rss, inst);
      }
    );

  }

  SNDE_OCL_API std::shared_ptr<math_function> waveform_interplines_function = define_waveform_interplines_function();

  static int registered_waveform_interplines_function = register_math_function( waveform_interplines_function);




  template <typename T>  // template for different floating point number classes 
  class waveform_vertlines : public recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>, double, double, double, double, double, double, snde_index, snde_index, snde_index, double, double>
  {
  public:
    waveform_vertlines(std::shared_ptr<recording_set_state> rss, std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>, double, double, double, double, double, double, snde_index, snde_index, snde_index, double, double>(rss, inst)
    {

    }


    // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
    typedef typename recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>, double, double, double, double, double, double, snde_index, snde_index, snde_index, double, double>::define_recs_function_override_type define_recs_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>, double, double, double, double, double, double, snde_index, snde_index, snde_index, double, double>::metadata_function_override_type metadata_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>, double, double, double, double, double, double, snde_index, snde_index, snde_index, double, double>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>, double, double, double, double, double, double, snde_index, snde_index, snde_index, double, double>::exec_function_override_type exec_function_override_type;

    // just using the default for decide_new_revision

    std::pair<std::vector<std::shared_ptr<compute_resource_option>>, std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<multi_ndarray_recording> recording, double R, double G, double B, double A, double linewidth_horiz, double linewidth_vert, snde_index startidx, snde_index endidx, snde_index idxstep, double datainival, double datastep)
    {
      bool consistent_layout_c = true; // consistent_layout_c means multiple arrays but all with the same layout except for the first axis which is implicitly concatenated
      bool consistent_layout_f = true; // consistent_layout_f means multiple arrays but all with the same layout except for the last axis which is implicitly concatenated
      bool consistent_ndim = true;

      assert(recording->layouts.size() == 1 && recording->layouts.at(0).dimlen.size() == 1);

      T junk = 0.0;

      snde_index outlen = (endidx - startidx) / idxstep;  // Double check this once the code below is written
	  
	  if (outlen < 1)
		  outlen = 1;


      std::vector<std::shared_ptr<compute_resource_option>> option_list =
      {
	std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
						      0, //metadata_bytes 
						      outlen * sizeof(snde_coord) * 3 * 6 + outlen * sizeof(snde_coord) * 3 * 6 * 4, // data_bytes for transfer
						      0.0, // flops
						      1, // max effective cpu cores
						      1), // useful_cpu_cores (min # of cores to supply

#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
							   0, //metadata_bytes
							   outlen * sizeof(snde_coord) * 3 * 6 + outlen * sizeof(snde_coord) * 3 * 6 * 4,
							   0.0, // cpu_flops
							   0.0, // gpuflops
							   1, // max effective cpu cores
							   1, // useful_cpu_cores (min # of cores to supply
							   sizeof(junk) > 4), // requires_doubleprec 
#endif // SNDE_OPENCL
      };
      return std::make_pair(option_list, std::make_shared<define_recs_function_override_type>([this, recording, R, G, B, A, linewidth_horiz, linewidth_vert, startidx, endidx, idxstep, datainival, datastep, outlen]() {

	// define_recs code
	//snde_debug(SNDE_DC_APP,"define_recs()");
	// Use of "this" in the next line for the same reason as the typedefs, above
	std::shared_ptr<multi_ndarray_recording> result_rec = create_recording_math<multi_ndarray_recording>(this->get_result_channel_path(0), this->rss, 2);
	result_rec->define_array(0, SNDE_RTN_SNDE_COORD3, "vertcoord");
	result_rec->define_array(1, SNDE_RTN_FLOAT32, "vertcoord_color");

	return std::make_shared<metadata_function_override_type>([this, result_rec, recording, R, G, B, A, linewidth_horiz, linewidth_vert, startidx, endidx, idxstep, datainival, datastep, outlen]() {
	  // metadata code
	  //std::unordered_map<std::string,metadatum> metadata;
	  //snde_debug(SNDE_DC_APP,"metadata()");
	  //metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));

	  result_rec->metadata = std::make_shared<immutable_metadata>();
	  result_rec->mark_metadata_done();

	  return std::make_shared<lock_alloc_function_override_type>([this, result_rec, recording, R, G, B, A, linewidth_horiz, linewidth_vert, startidx, endidx, idxstep, datainival, datastep, outlen]() {
	    // lock_alloc code

	    result_rec->allocate_storage("vertcoord", { (outlen) * 6 }, false);
	    result_rec->allocate_storage("vertcoord_color", { (outlen) * 6 * 4 }, false);

	    // locking is only required for certain recordings
	    // with special storage under certain conditions,
	    // however it is always good to explicitly request
	    // the locks, as the locking is a no-op if
	    // locking is not actually required.

	    // lock our output arrays
	    std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>, std::pair<size_t, bool>>> recrefs_to_lock = {
	      { result_rec, { 0, true } }, // vertcoord
	      { result_rec, { 1, true } }, // vertcoord_color
	    };

	    // ... and all the input arrays. 
	    for (size_t arraynum = 0; arraynum < recording->mndinfo()->num_arrays; arraynum++) {
	      recrefs_to_lock.emplace_back(std::make_pair(recording, std::make_pair(arraynum, false)));
	    }

	    rwlock_token_set locktokens = this->lockmgr->lock_recording_arrays(recrefs_to_lock,
#ifdef SNDE_OPENCL
	      true
#else
	      false
#endif
	    );

	    return std::make_shared<exec_function_override_type>([this, locktokens, result_rec, recording, R, G, B, A, linewidth_horiz, linewidth_vert, startidx, endidx, idxstep, datainival, datastep, outlen]() {
	      // exec code
	      //snde_index flattened_length = recording->layout.flattened_length();
	      //for (snde_index pos=0;pos < flattened_length;pos++){
	      //  result_rec->element(pos) = (recording->element(pos)-offset)/unitsperintensity;
	      //}



#ifdef SNDE_OPENCL
	      std::shared_ptr<assigned_compute_resource_opencl> opencl_resource = std::dynamic_pointer_cast<assigned_compute_resource_opencl>(compute_resource);
	      if (opencl_resource && recording->mndinfo()->num_arrays > 0) {

		//fprintf(stderr,"Executing in OpenCL!\n");

		cl::Kernel waveform_vertlines_vert_kern = build_typed_opencl_program<T>("snde.waveform_vertlines", (std::function<std::shared_ptr<opencl_program>(std::string)>)[](std::string ocltypename) {
		  // OpenCL templating via a typedef....
		  return std::make_shared<opencl_program>("waveform_vertlines", std::vector<std::string>({ snde_types_h, geometry_types_h, "\ntypedef " + ocltypename + " waveform_intype;\n", waveform_vertex_calcs_c }));
		  })->get_kernel(opencl_resource->context, opencl_resource->devices.at(0));

		  OpenCLBuffers Buffers(opencl_resource->oclcache, opencl_resource->context, opencl_resource->devices.at(0), locktokens);

		  snde_index curpos = (snde_index)(startidx + 1);
		  snde_float32 R_fl = (snde_float32)R;
		  snde_float32 G_fl = (snde_float32)G;
		  snde_float32 B_fl = (snde_float32)B;
		  snde_float32 A_fl = (snde_float32)A;
		  snde_float32 linewidth_horiz_fl = (snde_float32)linewidth_horiz;
		  snde_float32 linewidth_vert_fl = (snde_float32)linewidth_vert;

		  std::vector<cl::Event> kerndoneevents;


		  Buffers.AddBufferPortionAsKernelArg(recording, 0, 0, recording->layouts.at(0).dimlen.at(0), waveform_vertlines_vert_kern, 0, false, false);
		  Buffers.AddBufferPortionAsKernelArg(result_rec, "vertcoord", 0, outlen * 6, waveform_vertlines_vert_kern, 1, true, true);
		  Buffers.AddBufferPortionAsKernelArg(result_rec, "vertcoord_color", 0, outlen * 6 * 4, waveform_vertlines_vert_kern, 2, true, true);
		  //waveform_interplines_vert_kern.setArg(3, sizeof(cnt), &cnt);
		  waveform_vertlines_vert_kern.setArg(3, sizeof(curpos), &curpos);
		  waveform_vertlines_vert_kern.setArg(4, sizeof(endidx), &endidx);
		  float datainival_float = datainival;
		  waveform_vertlines_vert_kern.setArg(5, sizeof(datainival_float), &datainival_float);
		  float datastep_float = datastep;  
		  waveform_vertlines_vert_kern.setArg(6, sizeof(datastep_float), &datastep_float);
		  waveform_vertlines_vert_kern.setArg(7, sizeof(idxstep), &idxstep);
		  waveform_vertlines_vert_kern.setArg(8, sizeof(linewidth_horiz_fl), &linewidth_horiz_fl);
		  waveform_vertlines_vert_kern.setArg(9, sizeof(linewidth_vert_fl), &linewidth_vert_fl);
		  waveform_vertlines_vert_kern.setArg(10, sizeof(R_fl), &R_fl);
		  waveform_vertlines_vert_kern.setArg(11, sizeof(G_fl), &G_fl);
		  waveform_vertlines_vert_kern.setArg(12, sizeof(B_fl), &B_fl);
		  waveform_vertlines_vert_kern.setArg(13, sizeof(A_fl), &A_fl);

		  cl::Event kerndone;
		  std::vector<cl::Event> FillEvents = Buffers.FillEvents();

		  cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(waveform_vertlines_vert_kern, {}, { outlen }, {}, &FillEvents, &kerndone);
		  if (err != CL_SUCCESS) {
		    throw openclerror(err, "Error enqueueing kernel");
		  }

		  Buffers.BufferPortionDirty(result_rec, "vertcoord", 0, outlen * 6);
		  Buffers.BufferPortionDirty(result_rec, "vertcoord_color", 0, outlen * 6 * 4);
		  kerndoneevents.push_back(kerndone);


		  opencl_resource->queues.at(0).flush(); /* trigger execution */
		  // mark that the kernel has modified result_rec
		  // wait for kernel execution and transfers to complete

		  cl::Event::waitForEvents(kerndoneevents);
		  Buffers.RemBuffers(*(kerndoneevents.end() - 1), *(kerndoneevents.end() - 1), true);
	      }
	      else {
#endif // SNDE_OPENCL
		//snde_warning("Performing waveform vertex calculation on CPU. ");

		T junk = 0.0;

		// Plot Vertical Lines
		for (snde_index cnt = 0; cnt < outlen; cnt++) {
		  waveform_as_vertlines<T>(((T*)recording->void_shifted_arrayptr(0)),
		    ((snde_coord3*)result_rec->void_shifted_arrayptr("vertcoord")),
		    ((snde_float32*)result_rec->void_shifted_arrayptr("vertcoord_color")),
		    cnt,
		    startidx,
		    endidx,
		    datainival,
		    datastep,
		    idxstep,
		    linewidth_horiz,
		    linewidth_vert,
		    R,
		    G,
		    B,
		    A);
		}
#ifdef SNDE_OPENCL
	      }
#endif // SNDE_OPENCL

	      unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	      result_rec->mark_data_ready();

	      });
	    });
	  });
	}));
    };
  };



  std::shared_ptr<math_function> define_waveform_vertlines_function()
  {
    return std::make_shared<cpp_math_function>("snde.waveform_vertlines",1,[](std::shared_ptr<recording_set_state> rss, std::shared_ptr<instantiated_math_function> inst) {
      return make_cppfuncexec_floatingtypes<waveform_vertlines>(rss, inst);
      }
    );

  }

  SNDE_OCL_API std::shared_ptr<math_function> waveform_vertlines_function = define_waveform_vertlines_function();

  static int registered_waveform_vertlines_function = register_math_function( waveform_vertlines_function);







  template <typename T>  // template for different floating point number classes 
  class waveform_points : public recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>, double, double, double, double, double, double, snde_index, snde_index, snde_index, double, double>
  {
  public:
    waveform_points(std::shared_ptr<recording_set_state> rss, std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>, double, double, double, double, double, double, snde_index, snde_index, snde_index, double, double>(rss, inst)
    {

    }


    // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
    typedef typename recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>, double, double, double, double, double, double, snde_index, snde_index, snde_index, double, double>::define_recs_function_override_type define_recs_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>, double, double, double, double, double, double, snde_index, snde_index, snde_index, double, double>::metadata_function_override_type metadata_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>, double, double, double, double, double, double, snde_index, snde_index, snde_index, double, double>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename recmath_cppfuncexec<std::shared_ptr<multi_ndarray_recording>, double, double, double, double, double, double, snde_index, snde_index, snde_index, double, double>::exec_function_override_type exec_function_override_type;

    // just using the default for decide_new_revision

    std::pair<std::vector<std::shared_ptr<compute_resource_option>>, std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<multi_ndarray_recording> recording, double R, double G, double B, double A, double linewidth_horiz, double linewidth_vert, snde_index startidx, snde_index endidx, snde_index idxstep, double datainival, double datastep)
    {
      bool consistent_layout_c = true; // consistent_layout_c means multiple arrays but all with the same layout except for the first axis which is implicitly concatenated
      bool consistent_layout_f = true; // consistent_layout_f means multiple arrays but all with the same layout except for the last axis which is implicitly concatenated
      bool consistent_ndim = true;

      assert(recording->layouts.size() == 1 && recording->layouts.at(0).dimlen.size() == 1);

      T junk = 0.0;

      snde_index outlen = (endidx - startidx);  // Double check this once the code below is written

	  if (outlen < 1)
		  outlen = 1;


      std::vector<std::shared_ptr<compute_resource_option>> option_list =
      {
	std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
						      0, //metadata_bytes 
						      outlen * sizeof(snde_coord) * 3 + outlen * sizeof(snde_coord) * 4, // data_bytes for transfer
						      0.0, // flops
						      1, // max effective cpu cores
						      1), // useful_cpu_cores (min # of cores to supply

#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
							   0, //metadata_bytes
							   outlen * sizeof(snde_coord) * 3 + outlen * sizeof(snde_coord) * 4,
							   0.0, // cpu_flops
							   0.0, // gpuflops
							   1, // max effective cpu cores
							   1, // useful_cpu_cores (min # of cores to supply
							   sizeof(junk) > 4), // requires_doubleprec 
#endif // SNDE_OPENCL
      };
      return std::make_pair(option_list, std::make_shared<define_recs_function_override_type>([this, recording, R, G, B, A, linewidth_horiz, linewidth_vert, startidx, endidx, idxstep, datainival, datastep, outlen]() {

	// define_recs code
	//snde_debug(SNDE_DC_APP,"define_recs()");
	// Use of "this" in the next line for the same reason as the typedefs, above
	std::shared_ptr<multi_ndarray_recording> result_rec = create_recording_math<multi_ndarray_recording>(this->get_result_channel_path(0), this->rss, 2);
	assert(result_rec->originating_state);
	result_rec->define_array(0, SNDE_RTN_SNDE_COORD3, "pointcoord");
	result_rec->define_array(1, SNDE_RTN_FLOAT32, "pointcoord_color");

	return std::make_shared<metadata_function_override_type>([this, result_rec, recording, R, G, B, A, linewidth_horiz, linewidth_vert, startidx, endidx, idxstep, datainival, datastep, outlen]() {
	  // metadata code
	  //std::unordered_map<std::string,metadatum> metadata;
	  //snde_debug(SNDE_DC_APP,"metadata()");
	  //metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));

	  result_rec->metadata = std::make_shared<immutable_metadata>();
	  result_rec->mark_metadata_done();

	  return std::make_shared<lock_alloc_function_override_type>([this, result_rec, recording, R, G, B, A, linewidth_horiz, linewidth_vert, startidx, endidx, idxstep, datainival, datastep, outlen]() {
	    // lock_alloc code

	    result_rec->allocate_storage("pointcoord", { (outlen) }, false);
	    result_rec->allocate_storage("pointcoord_color", { (outlen) * 4 }, false);

	    // locking is only required for certain recordings
	    // with special storage under certain conditions,
	    // however it is always good to explicitly request
	    // the locks, as the locking is a no-op if
	    // locking is not actually required.

	    // lock our output arrays
	    std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>, std::pair<size_t, bool>>> recrefs_to_lock = {
	      { result_rec, { 0, true } }, // pointcoord
	      { result_rec, { 1, true } }, // pointcoord_color
	    };

	    // ... and all the input arrays. 
	    for (size_t arraynum = 0; arraynum < recording->mndinfo()->num_arrays; arraynum++) {
	      recrefs_to_lock.emplace_back(std::make_pair(recording, std::make_pair(arraynum, false)));
	    }

	    rwlock_token_set locktokens = this->lockmgr->lock_recording_arrays(recrefs_to_lock,
#ifdef SNDE_OPENCL
	      true
#else
	      false
#endif
	    );

	    return std::make_shared<exec_function_override_type>([this, locktokens, result_rec, recording, R, G, B, A, linewidth_horiz, linewidth_vert, startidx, endidx, idxstep, datainival, datastep, outlen]() {
	      // exec code
	      //snde_index flattened_length = recording->layout.flattened_length();
	      //for (snde_index pos=0;pos < flattened_length;pos++){
	      //  result_rec->element(pos) = (recording->element(pos)-offset)/unitsperintensity;
	      //}



#ifdef SNDE_OPENCL
	      std::shared_ptr<assigned_compute_resource_opencl> opencl_resource = std::dynamic_pointer_cast<assigned_compute_resource_opencl>(compute_resource);
	      if (opencl_resource && recording->mndinfo()->num_arrays > 0) {

		//fprintf(stderr,"Executing in OpenCL!\n");

		cl::Kernel waveform_points_vert_kern = build_typed_opencl_program<T>("snde.waveform_points", (std::function<std::shared_ptr<opencl_program>(std::string)>)[](std::string ocltypename) {
		  // OpenCL templating via a typedef....
		  return std::make_shared<opencl_program>("waveform_points", std::vector<std::string>({ snde_types_h, geometry_types_h, "\ntypedef " + ocltypename + " waveform_intype;\n", waveform_vertex_calcs_c }));
		  })->get_kernel(opencl_resource->context, opencl_resource->devices.at(0));

		  OpenCLBuffers Buffers(opencl_resource->oclcache, opencl_resource->context, opencl_resource->devices.at(0), locktokens);

		  snde_float32 R_fl = (snde_float32)R;
		  snde_float32 G_fl = (snde_float32)G;
		  snde_float32 B_fl = (snde_float32)B;
		  snde_float32 A_fl = (snde_float32)A;

		  std::vector<cl::Event> kerndoneevents;


		  Buffers.AddBufferPortionAsKernelArg(recording, 0, 0, recording->layouts.at(0).dimlen.at(0), waveform_points_vert_kern, 0, false, false);
		  Buffers.AddBufferPortionAsKernelArg(result_rec, "pointcoord", 0, outlen, waveform_points_vert_kern, 1, true, true);
		  Buffers.AddBufferPortionAsKernelArg(result_rec, "pointcoord_color", 0, outlen * 4, waveform_points_vert_kern, 2, true, true);
		  //waveform_interplines_vert_kern.setArg(3, sizeof(cnt), &cnt);
		  waveform_points_vert_kern.setArg(3, sizeof(startidx), &startidx);
		  float datainival_float = datainival;
		  waveform_points_vert_kern.setArg(4, sizeof(datainival_float), &datainival_float);
		  float datastep_float = datastep;
		  waveform_points_vert_kern.setArg(5, sizeof(datastep_float), &datastep_float);
		  waveform_points_vert_kern.setArg(6, sizeof(R_fl), &R_fl);
		  waveform_points_vert_kern.setArg(7, sizeof(G_fl), &G_fl);
		  waveform_points_vert_kern.setArg(8, sizeof(B_fl), &B_fl);
		  waveform_points_vert_kern.setArg(9, sizeof(A_fl), &A_fl);

		  cl::Event kerndone;
		  std::vector<cl::Event> FillEvents = Buffers.FillEvents();

		  cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(waveform_points_vert_kern, {}, { outlen }, {}, &FillEvents, &kerndone);
		  if (err != CL_SUCCESS) {
		    throw openclerror(err, "Error enqueueing kernel");
		  }

		  Buffers.BufferPortionDirty(result_rec, "pointcoord", 0, outlen);
		  Buffers.BufferPortionDirty(result_rec, "pointcoord_color", 0, outlen * 4);
		  kerndoneevents.push_back(kerndone);


		  opencl_resource->queues.at(0).flush(); /* trigger execution */
		  // mark that the kernel has modified result_rec
		  // wait for kernel execution and transfers to complete

		  cl::Event::waitForEvents(kerndoneevents);
		  Buffers.RemBuffers(*(kerndoneevents.end() - 1), *(kerndoneevents.end() - 1), true);

	      }
	      else {
#endif // SNDE_OPENCL
		//snde_warning("Performing waveform vertex calculation on CPU. ");

		T junk = 0.0;

		// Plot Vertical Lines
		for (snde_index cnt = 0; cnt < outlen; cnt++) {
		  waveform_as_points<T>(((T*)recording->void_shifted_arrayptr(0)),
		    ((snde_coord3*)result_rec->void_shifted_arrayptr("pointcoord")),
		    ((snde_float32*)result_rec->void_shifted_arrayptr("pointcoord_color")),
		    cnt,
		    startidx,
		    datainival,
		    datastep,
		    R,
		    G,
		    B,
		    A);
		}
#ifdef SNDE_OPENCL
	      }
#endif // SNDE_OPENCL

	      unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	      result_rec->mark_data_ready();

	      });
	    });
	  });
	}));
    };
  };



  std::shared_ptr<math_function> define_waveform_points_function()
  {
    return std::make_shared<cpp_math_function>("snde.waveform_points",1,[](std::shared_ptr<recording_set_state> rss, std::shared_ptr<instantiated_math_function> inst) {
      return make_cppfuncexec_floatingtypes<waveform_points>(rss, inst);
      }
    );

  }

  SNDE_OCL_API std::shared_ptr<math_function> waveform_points_function = define_waveform_points_function();

  static int registered_waveform_points_function = register_math_function( waveform_points_function);






};





