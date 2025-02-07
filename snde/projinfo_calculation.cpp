#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/vecops.h"
#include "snde/geometry_ops.h"
#include "snde/projinfo_calc.h"
#include "snde/geometrydata.h"
#include "snde/recmath_cppfunction.hpp"
#include "snde/graphics_recording.hpp"
#include "snde/graphics_storage.hpp"
#include "snde/geometry_processing.hpp"

#ifdef SNDE_OPENCL
#include "snde/snde_types_h.h"
#include "snde/geometry_types_h.h"
#include "snde/vecops_h.h"
#include "snde/geometry_ops_h.h"
#include "snde/projinfo_calc_c.h"
#endif // SNDE_OPENCL



#ifdef SNDE_OPENCL
#include "snde/openclcachemanager.hpp"
#include "snde/opencl_utils.hpp"
#include "snde/recmath_compute_resource_opencl.hpp"
#endif // SNDE_OPENCL


#include "snde/projinfo_calculation.hpp"

namespace snde {

#ifdef SNDE_OPENCL

  static opencl_program projinfo_opencl_program("snde_projinfo_calc", { snde_types_h, geometry_types_h, vecops_h, geometry_ops_h, projinfo_calc_c });

#endif //SNDE_OPENCL



  class projinfo_calculation: public recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>,std::shared_ptr<meshed_inplanemat_recording>,std::shared_ptr<meshed_parameterization_recording>> {
  public:
    projinfo_calculation(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec(rss,inst)
    {
      
    }
    
    // use default for decide_new_revision
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<meshed_part_recording> part,std::shared_ptr<meshed_inplanemat_recording> inplanemat,std::shared_ptr<meshed_parameterization_recording> parameterization)
    {
      snde_ndarray_info *rec_tri_info = part->ndinfo(part->name_mapping.at("triangles"));
      if (rec_tri_info->ndim != 1) {
	throw snde_error("projinfo_calculation: triangle dimensionality must be 1");
      }
      snde_index numtris = rec_tri_info->dimlen[0];

      snde_ndarray_info *rec_edge_info = part->ndinfo(part->name_mapping.at("edges"));
      if (rec_edge_info->ndim != 1) {
	throw snde_error("projinfo_calculation: edge dimensionality must be 1");
      }
      snde_index numedges = rec_edge_info->dimlen[0];
      
      
      snde_ndarray_info *rec_vert_info = part->ndinfo(part->name_mapping.at("vertices"));
      if (rec_vert_info->ndim != 1) {
	throw snde_error("projinfo_calculation: vertices dimensionality must be 1");
      }
      snde_index numverts = rec_vert_info->dimlen[0];

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numtris*(sizeof(snde_triangle) + 2*sizeof(snde_cmat23)) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord3), // data_bytes for transfer
							numtris*(200.0), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
							   0, //metadata_bytes
							   numtris*(sizeof(snde_triangle)+2*sizeof(snde_cmat23)) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord3),
							   0, // cpu_flops
							   numtris*(200.0), // gpuflops
							   1, // max effective cpu cores
							   1, // useful_cpu_cores (min # of cores to supply
							   snde_doubleprec_coords()), // requires_doubleprec 
#endif // SNDE_OPENCL
	};
      return std::make_pair(option_list,nullptr);
    }
    
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<meshed_part_recording> part,std::shared_ptr<meshed_inplanemat_recording> inplanemat,std::shared_ptr<meshed_parameterization_recording> parameterization) 
  {
    // define_recs code
    //printf("define_recs()\n");
    std::shared_ptr<meshed_projinfo_recording> result_rec;
    result_rec = create_recording_math<meshed_projinfo_recording>(get_result_channel_path(0),rss);
    
    return std::make_shared<metadata_function_override_type>([ this,result_rec,part,inplanemat,parameterization ]() {
      // metadata code
      std::unordered_map<std::string,metadatum> metadata;
      //printf("metadata()\n");
      //metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));
      
      result_rec->metadata=std::make_shared<immutable_metadata>(metadata);
      result_rec->mark_metadata_done();
      
      return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,part,inplanemat,parameterization ]() {
	// lock_alloc code
	
	std::shared_ptr<graphics_storage_manager> graphman = std::dynamic_pointer_cast<graphics_storage_manager>(result_rec->assign_storage_manager());
	
	if (!graphman) {
	  throw snde_error("projinfo_calculation: Output arrays must be managed by a graphics storage manager");
	}
	
	std::shared_ptr<graphics_storage> leader_storage = std::dynamic_pointer_cast<graphics_storage>(parameterization->storage.at(parameterization->name_mapping.at("uv_triangles")));
	
	snde_index addr = leader_storage->base_index;
	snde_index nmemb = leader_storage->nelem;
	
	
	std::shared_ptr<graphics_storage> inplane2uv_storage = graphman->storage_from_allocation(result_rec->info->name,leader_storage,"inplane2uvcoords",result_rec->info->revision,rss->unique_index,addr,sizeof(*graphman->geom.inplane2uvcoords),rtn_typemap.at(typeid(*graphman->geom.inplane2uvcoords)),nmemb);
	std::shared_ptr<graphics_storage> uv2inplane_storage = graphman->storage_from_allocation(result_rec->info->name,leader_storage,"uvcoords2inplane",result_rec->info->revision,rss->unique_index,addr,sizeof(*graphman->geom.uvcoords2inplane),rtn_typemap.at(typeid(*graphman->geom.uvcoords2inplane)),nmemb);
	
	result_rec->assign_storage(inplane2uv_storage,"inplane2uvcoords",{nmemb});
	result_rec->assign_storage(uv2inplane_storage,"uvcoords2inplane",{nmemb});
	

	//parts_ref = recording->reference_ndarray("parts")
	
	
	// locking is only required for certain recordings
	// with special storage under certain conditions,
	// however it is always good to explicitly request
	// the locks, as the locking is a no-op if
	// locking is not actually required.
	rwlock_token_set locktokens = lockmgr->lock_recording_arrays({
	    //{ part, { "parts", false }}, // first element is recording_ref, 2nd parameter is false for read, true for write
	    { part, { "triangles", false }},
	    { part, {"edges", false }},
	    { part, {"vertices", false}},
	    { inplanemat, { "inplanemats", false }},
	    //{ parameterization, { "uvs", false }},
	    { parameterization, { "uv_triangles", false }},
	    { parameterization, { "uv_edges", false }},
	    { parameterization, { "uv_vertices", false }},	    
	    { result_rec,{"uvcoords2inplane", true }},
	    { result_rec,{"inplane2uvcoords", true }}
	  },
#ifdef SNDE_OPENCL
	    true
#else
	    false
#endif
	  );
	
	return std::make_shared<exec_function_override_type>([ this,locktokens, result_rec, part, inplanemat,parameterization ]() {
	  // exec code
	  snde_ndarray_info *rec_tri_info = part->ndinfo(part->name_mapping.at("triangles"));
	  snde_index numtris = rec_tri_info->dimlen[0];
	  
#ifdef SNDE_OPENCL
	  std::shared_ptr<assigned_compute_resource_opencl> opencl_resource=std::dynamic_pointer_cast<assigned_compute_resource_opencl>(compute_resource);
	  if (opencl_resource) {
	    
	    //fprintf(stderr,"Executing in OpenCL!\n");
	    cl::Kernel projinfo_kern = projinfo_opencl_program.get_kernel(opencl_resource->context,opencl_resource->devices.at(0));
	    OpenCLBuffers Buffers(opencl_resource->oclcache,opencl_resource->context,opencl_resource->devices.at(0),locktokens);
	    
	    //Buffers.AddBufferAsKernelArg(recording,"parts",normal_kern,0,false,false);
	    Buffers.AddBufferAsKernelArg(part,"triangles",projinfo_kern,0,false,false);
	    Buffers.AddBufferAsKernelArg(part,"edges",projinfo_kern,1,false,false);
	    Buffers.AddBufferAsKernelArg(part,"vertices",projinfo_kern,2,false,false);
	    Buffers.AddBufferAsKernelArg(inplanemat,"inplanemats",projinfo_kern,3,false,false);
	    Buffers.AddBufferAsKernelArg(parameterization,"uv_triangles",projinfo_kern,4,false,false);
	    Buffers.AddBufferAsKernelArg(parameterization,"uv_edges",projinfo_kern,5,false,false);
	    Buffers.AddBufferAsKernelArg(parameterization,"uv_vertices",projinfo_kern,6,false,false);
	    Buffers.AddBufferAsKernelArg(result_rec,"inplane2uvcoords",projinfo_kern,7,true,true);
	    Buffers.AddBufferAsKernelArg(result_rec,"uvcoords2inplane",projinfo_kern,8,true,true);

      
	    cl::Event kerndone;
	    std::vector<cl::Event> FillEvents=Buffers.FillEvents();
	     
	    cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(projinfo_kern,{},{ numtris },{},&FillEvents,&kerndone);
	    if (err != CL_SUCCESS) {
	      throw openclerror(err,"Error enqueueing kernel");
	    }
	    opencl_resource->queues.at(0).flush(); /* trigger execution */
	    // mark that the kernel has modified result_rec
	    Buffers.BufferDirty(result_rec,"inplane2uvcoords");
	    Buffers.BufferDirty(result_rec,"uvcoords2inplane");
	    // wait for kernel execution and transfers to complete
	    Buffers.RemBuffers(kerndone,kerndone,true);
	    
	  } else {	    
#endif // SNDE_OPENCL
	    snde_warning("Performing projinfo calculation on CPU. This will be slow.");
	    //fprintf(stderr,"Not executing in OpenCL\n");
	    // Should OpenMP this (!)
	    const snde_triangle *triangles=(snde_triangle *)part->void_shifted_arrayptr("triangles");
	    const snde_edge *edges=(snde_edge *)part->void_shifted_arrayptr("edges");
	    const snde_coord3 *vertices=(snde_coord3 *)part->void_shifted_arrayptr("vertices");
	    const snde_cmat23 *inplanemats=(snde_cmat23 *)inplanemat->void_shifted_arrayptr("inplanemats");
	    const snde_triangle *uv_triangles=(snde_triangle *)parameterization->void_shifted_arrayptr("uv_triangles");
	    const snde_edge *uv_edges=(snde_edge *)parameterization->void_shifted_arrayptr("uv_edges");
	    const snde_coord2 *uv_vertices=(snde_coord2 *)parameterization->void_shifted_arrayptr("uv_vertices");
	    snde_cmat23 *inplane2uvcoords=(snde_cmat23 *)result_rec->void_shifted_arrayptr("inplane2uvcoords");
	    snde_cmat23 *uvcoords2inplane=(snde_cmat23 *)result_rec->void_shifted_arrayptr("uvcoords2inplane");
	    
	      
	    for (snde_index pos=0;pos < numtris;pos++){
	      snde_projinfo_calc_one(triangles,
				     edges,
				     vertices,
				     inplanemats,
				     uv_triangles,
				     uv_edges,
				     uv_vertices,
				     inplane2uvcoords,
				     uvcoords2inplane,  
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
  };
    
  };
    
  

  std::shared_ptr<math_function> define_spatialnde2_projinfo_calculation_function()
  {
    return std::make_shared<cpp_math_function>("snde.projinfo_calculation",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<projinfo_calculation>(rss,inst);
    }); 
  }
  
  SNDE_OCL_API std::shared_ptr<math_function> projinfo_calculation_function = define_spatialnde2_projinfo_calculation_function();
  
  static int registered_projinfo_calculation_function = register_math_function(projinfo_calculation_function);
  
  
  void instantiate_projinfo(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags)
  {
    std::string context = recdb_path_context(loaded_geom->info->name);

    geomproc_specify_dependency(remaining_processing_tags,all_processing_tags,"inplanemat"); // we require inplanemat 

    std::shared_ptr<instantiated_math_function> instantiated = projinfo_calculation_function->instantiate( {
	std::make_shared<math_parameter_recording>("meshed"),
	std::make_shared<math_parameter_recording>("inplanemat"),
	std::make_shared<math_parameter_recording>("uv")
      },
      {
	std::make_shared<std::string>("projinfo")
      },
      context,
      false, // is_mutable
      false, // ondemand
      false, // mdonly
      std::make_shared<math_definition>("instantiate_projinfo()"),
      {},
      nullptr);


    trans->recdb->add_math_function(trans,instantiated,true); // trinormals are generally hidden by default
    loaded_geom->processed_relpaths.emplace("projinfo","projinfo");

  }

  static int registered_projinfo_processor = register_geomproc_math_function("projinfo",instantiate_projinfo);

  

  
};
