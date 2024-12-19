#include "snde/rec_display_vertex_functions.hpp"
#include "snde/recmath_cppfunction.hpp"
#include "snde/rec_display_vertexarray.h"
#include "snde/rec_display_texvertexarray.h"
#include "snde/graphics_recording.hpp"
#include "snde/graphics_storage.hpp"

#ifdef SNDE_OPENCL
#include "snde/opencl_utils.hpp"
#include "snde/openclcachemanager.hpp"
#include "snde/recmath_compute_resource_opencl.hpp"
#endif

#include "snde/snde_types_h.h"
#include "snde/geometry_types_h.h"
#include "snde/vecops_h.h"
#include "snde/geometry_ops_h.h"
#include "snde/rec_display_vertexarray_c.h"
#include "snde/rec_display_texvertexarray_c.h"



namespace snde {

  // vertexarray 
  
#ifdef SNDE_OPENCL
  static opencl_program vertexarray_function_opencl("rec_display_vertexarray", { snde_types_h, geometry_types_h, vecops_h, geometry_ops_h, rec_display_vertexarray_c  });
#endif // SNDE_OPENCL

  
  class vertexarray_function: public recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>>
  {
  public:
    vertexarray_function(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>>(rss,inst)
    {
      
    }
    
    
    // just using the default for decide_new_revision


    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<meshed_part_recording> recording)
    {
      snde_ndarray_info *rec_tri_info = recording->ndinfo(recording->name_mapping.at("triangles"));
      if (rec_tri_info->ndim != 1) {
	throw snde_error("vertexarray_function: triangle dimensionality must be 1");
      }
      snde_index numtris = rec_tri_info->dimlen[0];

      snde_ndarray_info *rec_edge_info = recording->ndinfo(recording->name_mapping.at("edges"));
      if (rec_edge_info->ndim != 1) {
	throw snde_error("vertexarray_function: edge dimensionality must be 1");
      }
      snde_index numedges = rec_edge_info->dimlen[0];
      
      
      snde_ndarray_info *rec_vert_info = recording->ndinfo(recording->name_mapping.at("vertices"));
      if (rec_vert_info->ndim != 1) {
	throw snde_error("vertexarray_function: vertices dimensionality must be 1");
      }
      snde_index numverts = rec_vert_info->dimlen[0];

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numtris*sizeof(snde_triangle) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord3) + numtris*sizeof(snde_trivertnormals), // data_bytes for transfer
							numtris*(10), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
							   0, //metadata_bytes
							   numtris*sizeof(snde_triangle) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord3) + numtris*sizeof(snde_trivertnormals),
							   0, // cpu_flops
							   numtris*(10), // gpuflops
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
      //snde_debug(SNDE_DC_APP,"define_recs()");
      // Use of "this" in the next line for the same reason as the typedefs, above
      std::shared_ptr<meshed_vertexarray_recording> result_rec = create_recording_math<meshed_vertexarray_recording>(this->get_result_channel_path(0),this->rss);
      
      return std::make_shared<metadata_function_override_type>([ this,result_rec,recording ]() {
	// metadata code
	//std::unordered_map<std::string,metadatum> metadata;
	//snde_debug(SNDE_DC_APP,"metadata()");
	//metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));
	
	result_rec->metadata=std::make_shared<immutable_metadata>();
	result_rec->mark_metadata_done();
	
	return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,recording ]() {
	  // lock_alloc code
	  
	  snde_ndarray_info *rec_tri_info = recording->ndinfo(recording->name_mapping.at("triangles"));
	  snde_index numtris = rec_tri_info->dimlen[0];
	  
	  //std::shared_ptr<graphics_storage_manager> graphman = std::dynamic_pointer_cast<graphics_storage_manager>(result_rec->assign_storage_manager());
	  //// Don't think special storage manager is really necessary for this as it's now just a rendering intermediate!!!***
	  //if (!graphman) {
	  //  throw snde_error("vertexarray_function: Output arrays must be managed by a graphics storage manager");
	  //}

	  //std::shared_ptr<graphics_storage> vertarrays_storage = std::dynamic_pointer_cast<graphics_storage>(graphman->allocate_recording(result_rec->info->name,"vertex_arrays",result_rec->info->revision,sizeof(*graphman->geom.vertex_arrays),rtn_typemap.at(typeid(*graphman->geom.vertex_arrays)),numtris*9,false));
	  //result_rec->assign_storage(vertarrays_storage,"vertex_arrays",{numtris*9});
	  result_rec->assign_storage_manager(recdb->default_storage_manager); // Force default storage manager so that we DON'T go to the graphics storage (which is unnecessary for temporary output such as this)

	  result_rec->allocate_storage("vertex_arrays",{numtris*9},false);
	  

	  // locking is only required for certain recordings
	  // with special storage under certain conditions,
	  // however it is always good to explicitly request
	  // the locks, as the locking is a no-op if
	  // locking is not actually required. 
	  rwlock_token_set locktokens = this->lockmgr->lock_recording_arrays({
	      { recording, { "parts",false }}, // first element is recording_ref, 2nd parameter is false for read, true for write 
	      { recording, { "triangles",false }}, 
	      { recording, { "edges",false }}, 
	      { recording, { "vertices",false }}, 
	      { result_rec, { "vertex_arrays", true }},
	    },
#ifdef SNDE_OPENCL
	    true
#else
	    false
#endif
	    );
	  
	  
	  return std::make_shared<exec_function_override_type>([ this,locktokens,result_rec,recording ]() {
	    // exec code
	    //snde_index flattened_length = recording->layout.flattened_length();
	    //for (snde_index pos=0;pos < flattened_length;pos++){
	    //  result_rec->element(pos) = (recording->element(pos)-offset)/unitsperintensity;
	    //}

	    snde_ndarray_info *rec_tri_info = recording->ndinfo(recording->name_mapping.at("triangles"));
	    snde_index numtris = rec_tri_info->dimlen[0];
	    
	    
#ifdef SNDE_OPENCL
	    std::shared_ptr<assigned_compute_resource_opencl> opencl_resource=std::dynamic_pointer_cast<assigned_compute_resource_opencl>(compute_resource);
	    if (opencl_resource) {
	      
	      //fprintf(stderr,"Executing in OpenCL!\n");
	      cl::Kernel  vertexarray_kern = vertexarray_function_opencl.get_kernel(opencl_resource->context,opencl_resource->devices.at(0));
	      OpenCLBuffers Buffers(opencl_resource->oclcache,opencl_resource->context,opencl_resource->devices.at(0),locktokens);
	      
	      Buffers.AddBufferAsKernelArg(recording,"parts",vertexarray_kern,0,false,false);
	      Buffers.AddBufferAsKernelArg(recording,"triangles",vertexarray_kern,1,false,false);
	      Buffers.AddBufferAsKernelArg(recording,"edges",vertexarray_kern,2,false,false);
	      Buffers.AddBufferAsKernelArg(recording,"vertices",vertexarray_kern,3,false,false);
	      Buffers.AddBufferAsKernelArg(result_rec,"vertex_arrays",vertexarray_kern,4,true,true);
	    
	      
	      cl::Event kerndone;
	      std::vector<cl::Event> FillEvents=Buffers.FillEvents();
	      
	      cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(vertexarray_kern,{},{ numtris },{},&FillEvents,&kerndone);
	      if (err != CL_SUCCESS) {
		throw openclerror(err,"Error enqueueing kernel");
	      }
	      opencl_resource->queues.at(0).flush(); /* trigger execution */
	      // mark that the kernel has modified result_rec
	      //snde_warning("BufferDirty() on vertex_arrays... start_elem=%u; size=%u",(unsigned)result_rec->ndinfo(result_rec->name_mapping.at("vertex_arrays"))->base_index -  result_rec->storage.at(result_rec->name_mapping.at("vertex_arrays"))->base_index,(unsigned)result_rec->layouts.at(result_rec->name_mapping.at("vertex_arrays")).flattened_size());
	      Buffers.BufferDirty(result_rec,"vertex_arrays");
	      // wait for kernel execution and transfers to complete
	      Buffers.RemBuffers(kerndone,kerndone,true);
	    
	    } else {	    
#endif // SNDE_OPENCL
	      snde_warning("Performing vertex_arrays calculation on CPU. This will be slow.");
	      //fprintf(stderr,"Not executing in OpenCL\n");
	      // Should OpenMP this (!)
	      const struct snde_part *parts =(snde_part *)recording->void_shifted_arrayptr("parts");
	      const snde_triangle *triangles=(snde_triangle *)recording->void_shifted_arrayptr("triangles");
	      const snde_edge *edges=(snde_edge *)recording->void_shifted_arrayptr("edges");
	      const snde_coord3 *vertices=(snde_coord3 *)recording->void_shifted_arrayptr("vertices");
	      snde_rendercoord *vertex_arrays=(snde_rendercoord *)result_rec->void_shifted_arrayptr("vertex_arrays");
	    	    
	      
	      for (snde_index pos=0;pos < numtris;pos++){
		snde_rec_display_vertexarray_onetri(parts,
						    triangles,
						    edges,
						    vertices,
						    vertex_arrays,
						    pos);
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

  
  std::shared_ptr<math_function> define_meshedpart_vertexarray_function()
  {
    return std::make_shared<cpp_math_function>("snde.meshedpart_vertexarray",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<vertexarray_function>(rss,inst);
    });
    
  }
  
  static int registered_vertexarray_function = register_math_function(define_meshedpart_vertexarray_function());
  





  // texvertexarray

#ifdef SNDE_OPENCL
  static opencl_program texvertexarray_function_opencl("rec_display_texvertexarray", { snde_types_h, geometry_types_h, vecops_h, geometry_ops_h, rec_display_texvertexarray_c  });
#endif // SNDE_OPENCL

  
  class texvertexarray_function: public recmath_cppfuncexec<std::shared_ptr<meshed_parameterization_recording>>
  {
  public:
    texvertexarray_function(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<meshed_parameterization_recording>>(rss,inst)
    {
      
    }
    
    
    // just using the default for decide_new_revision


    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<meshed_parameterization_recording> recording)
    {
      snde_ndarray_info *rec_tri_info = recording->ndinfo(recording->name_mapping.at("uv_triangles"));
      if (rec_tri_info->ndim != 1) {
	throw snde_error("texvertexarray_function: triangle dimensionality must be 1");
      }
      snde_index numtris = rec_tri_info->dimlen[0];

      snde_ndarray_info *rec_edge_info = recording->ndinfo(recording->name_mapping.at("uv_edges"));
      if (rec_edge_info->ndim != 1) {
	throw snde_error("texvertexarray_function: edge dimensionality must be 1");
      }
      snde_index numedges = rec_edge_info->dimlen[0];
      
      
      snde_ndarray_info *rec_vert_info = recording->ndinfo(recording->name_mapping.at("uv_vertices"));
      if (rec_vert_info->ndim != 1) {
	throw snde_error("texvertexarray_function: vertices dimensionality must be 1");
      }
      snde_index numverts = rec_vert_info->dimlen[0];

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numtris*sizeof(snde_triangle) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord2), // data_bytes for transfer
							numtris*(10), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
							   0, //metadata_bytes
							   numtris*sizeof(snde_triangle) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord2) ,
							   0, // cpu_flops
							   numtris*(10), // gpuflops
							   1, // max effective cpu cores
							   1, // useful_cpu_cores (min # of cores to supply
							   snde_doubleprec_coords()), // requires_doubleprec 
#endif // SNDE_OPENCL
	};
      return std::make_pair(option_list,nullptr);
    }

      
    
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<meshed_parameterization_recording> recording) 
    {
      // define_recs code
      //snde_debug(SNDE_DC_APP,"define_recs()");
      // Use of "this" in the next line for the same reason as the typedefs, above
      std::shared_ptr<meshed_texvertex_recording> result_rec = create_recording_math<meshed_texvertex_recording>(this->get_result_channel_path(0),this->rss);
      
      return std::make_shared<metadata_function_override_type>([ this,result_rec,recording ]() {
	// metadata code
	//std::unordered_map<std::string,metadatum> metadata;
	//snde_debug(SNDE_DC_APP,"metadata()");
	//metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));
	
	result_rec->metadata=std::make_shared<immutable_metadata>();
	result_rec->mark_metadata_done();
	
	return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,recording ]() {
	  // lock_alloc code
	  

	  snde_ndarray_info *rec_tri_info = recording->ndinfo(recording->name_mapping.at("uv_triangles"));
	  snde_index numtris = rec_tri_info->dimlen[0];
	  
	  //std::shared_ptr<graphics_storage_manager> graphman = std::dynamic_pointer_cast<graphics_storage_manager>(result_rec->assign_storage_manager());
	  //
	  //if (!graphman) {
	  //  throw snde_error("texvertexarray_function: Output arrays must be managed by a graphics storage manager");
	  //}

	  //std::shared_ptr<graphics_storage> vertarrays_storage = std::dynamic_pointer_cast<graphics_storage>(graphman->allocate_recording(result_rec->info->name,"texvertex_arrays",result_rec->info->revision,sizeof(*graphman->geom.texvertex_arrays),rtn_typemap.at(typeid(*graphman->geom.texvertex_arrays)),numtris*6,false));
	  //result_rec->assign_storage(vertarrays_storage,"texvertex_arrays",{numtris*6});
	  result_rec->assign_storage_manager(this->recdb->default_storage_manager); // Force default storage manager so that we DON'T go to the graphics storage (which is unnecessary for temporary output such as this)

	  result_rec->allocate_storage("texvertex_arrays",{numtris*6},false);


	  // locking is only required for certain recordings
	  // with special storage under certain conditions,
	  // however it is always good to explicitly request
	  // the locks, as the locking is a no-op if
	  // locking is not actually required. 
	  rwlock_token_set locktokens = this->lockmgr->lock_recording_arrays({
	      { recording, { "uvs",false }}, // first element is recording_ref, 2nd parameter is false for read, true for write 
	      { recording, { "uv_triangles",false }}, 
	      { recording, { "uv_edges",false }}, 
	      { recording, { "uv_vertices",false }}, 
	      { result_rec, { "texvertex_arrays", true }},
	    },
#ifdef SNDE_OPENCL
	    true
#else
	    false
#endif
	    );
	  
	  
	  return std::make_shared<exec_function_override_type>([ this,locktokens,result_rec,recording ]() {
	    // exec code
	    //snde_index flattened_length = recording->layout.flattened_length();
	    //for (snde_index pos=0;pos < flattened_length;pos++){
	    //  result_rec->element(pos) = (recording->element(pos)-offset)/unitsperintensity;
	    //}

	    snde_ndarray_info *rec_tri_info = recording->ndinfo(recording->name_mapping.at("uv_triangles"));
	    snde_index numtris = rec_tri_info->dimlen[0];
	    
	    
#ifdef SNDE_OPENCL
	    std::shared_ptr<assigned_compute_resource_opencl> opencl_resource=std::dynamic_pointer_cast<assigned_compute_resource_opencl>(compute_resource);
	    if (opencl_resource) {
	      
	      //fprintf(stderr,"Executing in OpenCL!\n");
	      cl::Kernel  texvertexarray_kern = texvertexarray_function_opencl.get_kernel(opencl_resource->context,opencl_resource->devices.at(0));
	      OpenCLBuffers Buffers(opencl_resource->oclcache,opencl_resource->context,opencl_resource->devices.at(0),locktokens);
	      
	      Buffers.AddBufferAsKernelArg(recording,"uvs",texvertexarray_kern,0,false,false);
	      Buffers.AddBufferAsKernelArg(recording,"uv_triangles",texvertexarray_kern,1,false,false);
	      Buffers.AddBufferAsKernelArg(recording,"uv_edges",texvertexarray_kern,2,false,false);
	      Buffers.AddBufferAsKernelArg(recording,"uv_vertices",texvertexarray_kern,3,false,false);
	      Buffers.AddBufferAsKernelArg(result_rec,"texvertex_arrays",texvertexarray_kern,4,true,true);
	    
	      
	      cl::Event kerndone;
	      std::vector<cl::Event> FillEvents=Buffers.FillEvents();
	      
	      cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(texvertexarray_kern,{},{ numtris },{},&FillEvents,&kerndone);
	      if (err != CL_SUCCESS) {
		throw openclerror(err,"Error enqueueing kernel");
	      }
	      opencl_resource->queues.at(0).flush(); /* trigger execution */
	      // mark that the kernel has modified result_rec
	      Buffers.BufferDirty(result_rec,"texvertex_arrays");
	      // wait for kernel execution and transfers to complete
	      Buffers.RemBuffers(kerndone,kerndone,true);
	    
	    } else {	    
#endif // SNDE_OPENCL
	      snde_warning("Performing texvertex_arrays calculation on CPU. This will be slow.");
	      //fprintf(stderr,"Not executing in OpenCL\n");
	      // Should OpenMP this (!)
	      const struct snde_parameterization *uvs =(snde_parameterization *)recording->void_shifted_arrayptr("uvs");
	      const snde_triangle *uv_triangles=(snde_triangle *)recording->void_shifted_arrayptr("uv_triangles");
	      const snde_edge *uv_edges=(snde_edge *)recording->void_shifted_arrayptr("uv_edges");
	      const snde_coord2 *uv_vertices=(snde_coord2 *)recording->void_shifted_arrayptr("uv_vertices");
	      snde_rendercoord *texvertex_arrays=(snde_rendercoord *)result_rec->void_shifted_arrayptr("texvertex_arrays");
	    	    
	      
	      for (snde_index pos=0;pos < numtris;pos++){
		snde_rec_display_texvertexarray_onetri(uvs,
						       uv_triangles,
						       uv_edges,
						       uv_vertices,
						       texvertex_arrays,
						       pos);
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


  std::shared_ptr<math_function> define_meshedparameterization_texvertexarray_function()
  {

    return std::make_shared<cpp_math_function>("snde.meshedparameterization_texvertexarray",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<texvertexarray_function>(rss,inst);
    });
  }
  
  static int registered_texvertexarray_function = register_math_function(define_meshedparameterization_texvertexarray_function());




  
};


