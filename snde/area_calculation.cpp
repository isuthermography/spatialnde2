#include <array>

#ifdef SNDE_OPENCL
#include "snde/snde_types_h.h"
#include "snde/geometry_types_h.h"
#include "snde/vecops_h.h"
#include "snde/geometry_ops_h.h"
#include "snde/area_calc_c.h"
#endif // SNDE_OPENCL


#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/vecops.h"
#include "snde/geometry_ops.h"
#include "snde/geometrydata.h"

#include "snde/recmath_cppfunction.hpp"
#include "snde/graphics_recording.hpp"
#include "snde/graphics_storage.hpp"
#include "snde/geometry_processing.hpp"

#ifdef SNDE_OPENCL
#include "snde/opencl_utils.hpp"
#include "snde/openclcachemanager.hpp"
#include "snde/recmath_compute_resource_opencl.hpp"
#endif // SNDE_OPENCL

#include "snde/area_calc.h"
#include "snde/area_calculation.hpp"

namespace snde {


#ifdef SNDE_OPENCL
  static opencl_program areacalc_triangleareas_opencl("snde_areacalc_triangleareas", { snde_types_h, geometry_types_h, vecops_h, geometry_ops_h, area_calc_c });
#endif // SNDE_OPENCL


  
  class trianglearea_calculation: public recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>> {
  public:
    trianglearea_calculation(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec(rss,inst)
    {
      
    }
    
    // use default for decide_new_revision
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<meshed_part_recording> recording)
    {
      snde_ndarray_info *rec_tri_info = recording->ndinfo(recording->name_mapping.at("triangles"));
      if (rec_tri_info->ndim != 1) {
	throw snde_error("area_calculation: triangles dimensionality must be 1");
      }
      snde_index numtris = rec_tri_info->dimlen[0];

      snde_ndarray_info *rec_edge_info = recording->ndinfo(recording->name_mapping.at("edges"));
      if (rec_edge_info->ndim != 1) {
	throw snde_error("area_calculation: edges dimensionality must be 1");
      }
      snde_index numedges = rec_edge_info->dimlen[0];

      snde_ndarray_info *rec_vert_info = recording->ndinfo(recording->name_mapping.at("vertices"));
      if (rec_vert_info->ndim != 1) {
	throw snde_error("area_calculation: vertices dimensionality must be 1");
      }
      snde_index numverts = rec_vert_info->dimlen[0];


      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numtris*sizeof(snde_triangle) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord3) +  numtris*sizeof(snde_coord), // data_bytes for transfer
							numtris*(50), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
							   0, //metadata_bytes
							   numtris*sizeof(snde_triangle) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord3) + numtris*sizeof(snde_coord),
							   0, // cpu_flops
							   numtris*(50), // gpuflops
							   1, // max effective cpu cores
							   1, // useful_cpu_cores (min # of cores to supply
							   snde_doubleprec_coords()), // requires_doubleprec 
#endif // SNDE_OPENCL
	  
	};
      
      return std::make_pair(option_list,nullptr);
    }
    
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<meshed_part_recording> recording) 
    {
      // define_recs code
      //printf("define_recs()\n");
      std::shared_ptr<ndtyped_recording_ref<snde_coord>> result_ref;
      
      recording->assert_no_scale_or_offset(this->inst->definition->definition_command);
	    

      result_ref = create_typed_named_ndarray_ref_math<snde_coord>(get_result_channel_path(0),rss,"trianglearea");
      
      return std::make_shared<metadata_function_override_type>([ this,result_ref,recording ]() {
	// metadata code moved to exec function so we can get max depth info 
	constructible_metadata metadata;
	
	result_ref->rec->metadata=std::make_shared<immutable_metadata>(metadata);
	result_ref->rec->mark_metadata_done();
      
	return std::make_shared<lock_alloc_function_override_type>([ this,result_ref,recording ]() {
	  // lock_alloc code

	  snde_ndarray_info *rec_tri_info = recording->ndinfo("triangles");
	  snde_index numtris = rec_tri_info->dimlen[0];
	  
	  
	  std::shared_ptr<graphics_storage_manager> graphman = std::dynamic_pointer_cast<graphics_storage_manager>(result_ref->rec->assign_storage_manager());
	  
	  if (!graphman) {
	    throw snde_error("area_calculation: Output arrays must be managed by a graphics storage manager");
	  }
	
	  
	  result_ref->allocate_storage({numtris},false);
	  
	  
	  rwlock_token_set locktokens = lockmgr->lock_recording_arrays({
	      { recording, { "parts", true }}, // first element is recording_ref, 2nd parameter is false for read, true for write
	      { recording, { "triangles", false }},
	      { recording, { "edges", false }},
	      { recording, { "vertices", false}},
	      { result_ref->rec, { "trianglearea", true }},
	    },
#ifdef SNDE_OPENCL
	    true
#else
	    false
#endif
	    );
	  
	  return std::make_shared<exec_function_override_type>([ this,locktokens, result_ref, recording,numtris  ]() {
	    // exec code
	    
	    
	    snde_part *parts = (struct snde_part *)recording->void_shifted_arrayptr("parts");

	    parts->first_triarea = result_ref->storage->base_index;
	    recording->reference_ndarray("parts")->storage->mark_as_modified(nullptr,0,1,true); // indicate that we have modified this first element of "parts", invalidating caches. 
	    
#ifdef SNDE_OPENCL
	    std::shared_ptr<assigned_compute_resource_opencl> opencl_resource=std::dynamic_pointer_cast<assigned_compute_resource_opencl>(compute_resource);
	    if (opencl_resource) {
	      
	      //fprintf(stderr,"Executing in OpenCL!\n");
	      cl::Kernel triarea_kern = areacalc_triangleareas_opencl.get_kernel(opencl_resource->context,opencl_resource->devices.at(0));
	      OpenCLBuffers Buffers(opencl_resource->oclcache,opencl_resource->context,opencl_resource->devices.at(0),locktokens);
	      
	      //Buffers.AddBufferAsKernelArg(recording,"parts",triarea_kern,0,false,false);
	      Buffers.AddBufferAsKernelArg(recording,"triangles",triarea_kern,0,false,false);
	      Buffers.AddBufferAsKernelArg(recording,"edges",triarea_kern,1,false,false);
	      Buffers.AddBufferAsKernelArg(recording,"vertices",triarea_kern,2,false,false);
	      Buffers.AddBufferAsKernelArg(result_ref->rec,"trianglearea",triarea_kern,3,true,true);
	      
	      
	      cl::Event kerndone;
	      std::vector<cl::Event> FillEvents=Buffers.FillEvents();
	      
	      cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(triarea_kern,{},{ numtris },{},&FillEvents,&kerndone);
	      if (err != CL_SUCCESS) {
		throw openclerror(err,"Error enqueueing kernel");
	      }
	      opencl_resource->queues.at(0).flush(); /* trigger execution */
	      // mark that the kernel has modified result_rec
	      Buffers.BufferDirty(result_ref->rec,"trianglearea");
	      // wait for kernel execution and transfers to complete
	      Buffers.RemBuffers(kerndone,kerndone,true);
	      
	    } else {	    
#endif // SNDE_OPENCL
	      snde_warning("Performing area calculation on CPU. This will be slow.");
	      //fprintf(stderr,"Not executing in OpenCL\n");
	      // Should OpenMP this (!)
	      const snde_triangle *triangles=(snde_triangle *)recording->void_shifted_arrayptr("triangles");
	      const snde_edge *edges=(snde_edge *)recording->void_shifted_arrayptr("edges");
	      const snde_coord3 *vertices=(snde_coord3 *)recording->void_shifted_arrayptr("vertices");
	      //snde_trivertnormals *vertnormals=(snde_trivertnormals *)result_rec->void_shifted_arrayptr("vertnormals");
	      snde_coord *trianglearea=(snde_coord *)result_ref->void_shifted_arrayptr();
	      
	      
	      for (snde_index pos=0;pos < numtris;pos++){
		trianglearea[pos]=snde_areacalc_trianglearea(triangles,
							     edges,
							     vertices,
							     pos);
	      }
	      
#ifdef SNDE_OPENCL
	    }
#endif // SNDE_OPENCL
	    
	    
	    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	    
	    result_ref->rec->mark_data_ready();
	    
	    
	  });
	});
      });
    };

  };
  
  
  
  
  
  std::shared_ptr<math_function> define_spatialnde2_trianglearea_calculation_function()
  {
    return std::make_shared<cpp_math_function>("snde.trianglearea_calculation",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<trianglearea_calculation>(rss,inst);
    }); 
    
  }

  // NOTE: Change to SNDE_OCL_API if/when we add GPU acceleration support, and
  // (in CMakeLists.txt) make it move into the _ocl.so library)
  SNDE_OCL_API std::shared_ptr<math_function> trianglearea_calculation_function = define_spatialnde2_trianglearea_calculation_function();
  
  static int registered_trianglearea_calculation_function = register_math_function(trianglearea_calculation_function);


  void instantiate_trianglearea(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags)
  {
    
    std::string context = recdb_path_context(loaded_geom->info->name);
    std::shared_ptr<instantiated_math_function> instantiated = trianglearea_calculation_function->instantiate( {
	std::make_shared<math_parameter_recording>("meshed"),
      },
      {
	std::make_shared<std::string>("trianglearea")
      },
      context,
      false, // is_mutable
      false, // ondemand
      false, // mdonly
      std::make_shared<math_definition>("instantiate_trianglearea()"),
      {},
      nullptr);

    
    trans->recdb->add_math_function(trans,instantiated,true); // trinormals are generally hidden by default
    loaded_geom->processed_relpaths.emplace("trianglearea","trianglearea");
  }
  
  static int registered_trianglearea_processor = register_geomproc_math_function("trianglearea",instantiate_trianglearea);







#ifdef SNDE_OPENCL
  static opencl_program areacalc_vertexareas_opencl("snde_areacalc_vertexareas", { snde_types_h, geometry_types_h, vecops_h, geometry_ops_h, area_calc_c });
#endif // SNDE_OPENCL


  
  class vertexarea_calculation: public recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>,std::shared_ptr<ndtyped_recording_ref<snde_coord>>> {
  public:
    vertexarea_calculation(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec(rss,inst)
    {
      
    }
    
    // use default for decide_new_revision
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<meshed_part_recording> recording,std::shared_ptr<ndtyped_recording_ref<snde_coord>> ref_trianglearea)
    {
      snde_ndarray_info *rec_tri_info = recording->ndinfo(recording->name_mapping.at("triangles"));
      if (rec_tri_info->ndim != 1) {
	throw snde_error("area_calculation: triangles dimensionality must be 1");
      }
      snde_index numtris = rec_tri_info->dimlen[0];

      snde_ndarray_info *rec_edge_info = recording->ndinfo(recording->name_mapping.at("edges"));
      if (rec_edge_info->ndim != 1) {
	throw snde_error("area_calculation: edges dimensionality must be 1");
      }
      snde_index numedges = rec_edge_info->dimlen[0];

      snde_ndarray_info *rec_vert_info = recording->ndinfo(recording->name_mapping.at("vertices"));
      if (rec_vert_info->ndim != 1) {
	throw snde_error("area_calculation: vertices dimensionality must be 1");
      }
      snde_index numverts = rec_vert_info->dimlen[0];


      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numtris*sizeof(snde_triangle) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord3) +  numtris*sizeof(snde_coord), // data_bytes for transfer
							numtris*(50), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
							   0, //metadata_bytes
							   numtris*sizeof(snde_triangle) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord3) + numtris*sizeof(snde_coord),
							   0, // cpu_flops
							   numtris*(50), // gpuflops
							   1, // max effective cpu cores
							   1, // useful_cpu_cores (min # of cores to supply
							   snde_doubleprec_coords()), // requires_doubleprec 
#endif // SNDE_OPENCL
	  
	};
      
      return std::make_pair(option_list,nullptr);
    }
    
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<meshed_part_recording> recording,std::shared_ptr<ndtyped_recording_ref<snde_coord>> ref_trianglearea) 
    {
      // define_recs code
      //printf("define_recs()\n");
      std::shared_ptr<ndtyped_recording_ref<snde_coord>> result_ref;
      
      recording->assert_no_scale_or_offset(this->inst->definition->definition_command);
	    

      result_ref = create_typed_named_ndarray_ref_math<snde_coord>(get_result_channel_path(0),rss,"vertexarea");
      
      return std::make_shared<metadata_function_override_type>([ this,result_ref,recording,ref_trianglearea ]() {
	// metadata code moved to exec function so we can get max depth info 
	constructible_metadata metadata;
	
	result_ref->rec->metadata=std::make_shared<immutable_metadata>(metadata);
	result_ref->rec->mark_metadata_done();
      
	return std::make_shared<lock_alloc_function_override_type>([ this,result_ref,recording,ref_trianglearea ]() {
	  // lock_alloc code

	  snde_ndarray_info *rec_vert_info = recording->ndinfo("vertices");
	  snde_index numverts = rec_vert_info->dimlen[0];
	  
	  
	  std::shared_ptr<graphics_storage_manager> graphman = std::dynamic_pointer_cast<graphics_storage_manager>(result_ref->rec->assign_storage_manager());
	  
	  if (!graphman) {
	    throw snde_error("area_calculation: Output arrays must be managed by a graphics storage manager");
	  }
	
	  
	  result_ref->allocate_storage({numverts},false);
	  
	  
	  rwlock_token_set locktokens = lockmgr->lock_recording_arrays({
	      { recording, { "parts", true }}, // first element is recording_ref, 2nd parameter is false for read, true for write
	      { recording, { "triangles", false }},
	      { recording, { "edges", false }},
	      { recording, { "vertices", false}},
	      { recording, { "vertex_edgelist_indices", false}},
	      { recording, { "vertex_edgelist", false}},
	      { ref_trianglearea->rec, { "trianglearea", false}},
	      { result_ref->rec, { "vertexarea", true }},
	    },
#ifdef SNDE_OPENCL
	    true
#else
	    false
#endif
	    );
	  
	  return std::make_shared<exec_function_override_type>([ this,locktokens, result_ref, recording, ref_trianglearea, numverts  ]() {
	    // exec code
	    
	    
	    snde_part *parts = (struct snde_part *)recording->void_shifted_arrayptr("parts");

	    parts->first_vertarea = result_ref->storage->base_index;
	    recording->reference_ndarray("parts")->storage->mark_as_modified(nullptr,0,1,true); // indicate that we have modified this first element of "parts", invalidating caches. 
	    
#ifdef SNDE_OPENCL
	    std::shared_ptr<assigned_compute_resource_opencl> opencl_resource=std::dynamic_pointer_cast<assigned_compute_resource_opencl>(compute_resource);
	    if (opencl_resource) {
	      
	      //fprintf(stderr,"Executing in OpenCL!\n");
	      cl::Kernel vertarea_kern = areacalc_vertexareas_opencl.get_kernel(opencl_resource->context,opencl_resource->devices.at(0));
	      OpenCLBuffers Buffers(opencl_resource->oclcache,opencl_resource->context,opencl_resource->devices.at(0),locktokens);
	      
	      //Buffers.AddBufferAsKernelArg(recording,"parts",triarea_kern,0,false,false);
	      Buffers.AddBufferAsKernelArg(recording,"triangles",vertarea_kern,0,false,false);
	      Buffers.AddBufferAsKernelArg(recording,"edges",vertarea_kern,1,false,false);
	      Buffers.AddBufferAsKernelArg(recording,"vertices",vertarea_kern,2,false,false);
	      Buffers.AddBufferAsKernelArg(recording,"vertex_edgelist_indices",vertarea_kern,3,false,false);
	      Buffers.AddBufferAsKernelArg(recording,"vertex_edgelist",vertarea_kern,4,false,false);
	      Buffers.AddBufferAsKernelArg(ref_trianglearea->rec,"trianglearea",vertarea_kern,5,false,false);
	      Buffers.AddBufferAsKernelArg(result_ref->rec,"vertexarea",vertarea_kern,6,true,true);
	      
	      
	      cl::Event kerndone;
	      std::vector<cl::Event> FillEvents=Buffers.FillEvents();
	      
	      cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(vertarea_kern,{},{ numverts },{},&FillEvents,&kerndone);
	      if (err != CL_SUCCESS) {
		throw openclerror(err,"Error enqueueing kernel");
	      }
	      opencl_resource->queues.at(0).flush(); /* trigger execution */
	      // mark that the kernel has modified result_rec
	      Buffers.BufferDirty(result_ref->rec,"vertexarea");
	      // wait for kernel execution and transfers to complete
	      Buffers.RemBuffers(kerndone,kerndone,true);
	      
	    } else {	    
#endif // SNDE_OPENCL
	      snde_warning("Performing area calculation on CPU. This will be slow.");
	      //fprintf(stderr,"Not executing in OpenCL\n");
	      // Should OpenMP this (!)
	      const snde_triangle *triangles=(snde_triangle *)recording->void_shifted_arrayptr("triangles");
	      const snde_edge *edges=(snde_edge *)recording->void_shifted_arrayptr("edges");
	      const snde_coord3 *vertices=(snde_coord3 *)recording->void_shifted_arrayptr("vertices");
	      const snde_vertex_edgelist_index *vertex_edgelist_indices=(snde_vertex_edgelist_index *)recording->void_shifted_arrayptr("vertex_edgelist_indices");
	      const snde_index *vertex_edgelist=(snde_index *)recording->void_shifted_arrayptr("vertex_edgelist");
	      //snde_trivertnormals *vertnormals=(snde_trivertnormals *)result_rec->void_shifted_arrayptr("vertnormals");
	      snde_coord *trianglearea=(snde_coord *)ref_trianglearea->void_shifted_arrayptr();
	      snde_coord *vertarea=(snde_coord *)result_ref->void_shifted_arrayptr();
	      
	      
	      for (snde_index pos=0;pos < numverts;pos++){
		vertarea[pos]=snde_areacalc_vertexarea(triangles,
							     edges,
							     vertices,
							     vertex_edgelist_indices,
							     vertex_edgelist,
							     trianglearea,
							     pos);
	      }
	      
#ifdef SNDE_OPENCL
	    }
#endif // SNDE_OPENCL
	    
	    
	    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	    
	    result_ref->rec->mark_data_ready();
	    
	    
	  });
	});
      });
    };

  };
  
  
  
  
  
  std::shared_ptr<math_function> define_spatialnde2_vertexarea_calculation_function()
  {
    return std::make_shared<cpp_math_function>("snde.vertexarea_calculation",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<vertexarea_calculation>(rss,inst);
    }); 
    
  }

  // NOTE: Change to SNDE_OCL_API if/when we add GPU acceleration support, and
  // (in CMakeLists.txt) make it move into the _ocl.so library)
  SNDE_OCL_API std::shared_ptr<math_function> vertexarea_calculation_function = define_spatialnde2_vertexarea_calculation_function();
  
  static int registered_vertexarea_calculation_function = register_math_function(vertexarea_calculation_function);


  void instantiate_vertexarea(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags)
  {
    
    std::string context = recdb_path_context(loaded_geom->info->name);

    geomproc_specify_dependency(remaining_processing_tags,all_processing_tags,"trianglearea"); // we require trianglearea

    
    std::shared_ptr<instantiated_math_function> instantiated = vertexarea_calculation_function->instantiate( {
	std::make_shared<math_parameter_recording>("meshed"),
	std::make_shared<math_parameter_recording>("trianglearea"),
      },
      {
	std::make_shared<std::string>("vertexarea")
      },
      context,
      false, // is_mutable
      false, // ondemand
      false, // mdonly
      std::make_shared<math_definition>("instantiate_vertexarea()"),
      {},
      nullptr);

    
    trans->recdb->add_math_function(trans,instantiated,true); // vertexareas are generally hidden by default
    loaded_geom->processed_relpaths.emplace("vertexarea","vertexarea");
  }
  
  static int registered_vertexarea_processor = register_geomproc_math_function("vertexarea",instantiate_vertexarea);
  
  
  
};

