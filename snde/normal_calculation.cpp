#ifdef SNDE_OPENCL
#include "snde/snde_types_h.h"
#include "snde/geometry_types_h.h"
#include "snde/vecops_h.h"
#include "snde/geometry_ops_h.h"
#include "snde/normal_calc_c.h"
#endif // SNDE_OPENCL


#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/vecops.h"
#include "snde/geometry_ops.h"
#include "snde/geometrydata.h"
#include "snde/normal_calc.h"

#include "snde/recstore.hpp"
#include "snde/recmath_cppfunction.hpp"
#include "snde/graphics_recording.hpp"
#include "snde/graphics_storage.hpp"
#include "snde/geometry_processing.hpp"

#ifdef SNDE_OPENCL
#include "snde/opencl_utils.hpp"
#include "snde/openclcachemanager.hpp"
#include "snde/recmath_compute_resource_opencl.hpp"
#endif // SNDE_OPENCL

#include "snde/normal_calculation.hpp"

namespace snde {

  // trinormals starts here
  
#ifdef SNDE_OPENCL
  static opencl_program normalcalc_trinormals_opencl("snde_normalcalc_trinormals", { snde_types_h, geometry_types_h, vecops_h, geometry_ops_h, normal_calc_c });
#endif // SNDE_OPENCL
  
  class normal_calculation_trinormals: public recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>> {
  public:
    normal_calculation_trinormals(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec(rss,inst)
    {
      
    }
    
    // use default for decide_new_revision
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<meshed_part_recording> recording)
    {
      snde_ndarray_info *rec_tri_info = recording->ndinfo(recording->name_mapping.at("triangles"));
      if (rec_tri_info->ndim != 1) {
	throw snde_error("normal_calculation: triangle dimensionality must be 1");
      }
      snde_index numtris = rec_tri_info->dimlen[0];

      snde_ndarray_info *rec_edge_info = recording->ndinfo(recording->name_mapping.at("edges"));
      if (rec_edge_info->ndim != 1) {
	throw snde_error("normal_calculation: edge dimensionality must be 1");
      }
      snde_index numedges = rec_edge_info->dimlen[0];
      
      
      snde_ndarray_info *rec_vert_info = recording->ndinfo(recording->name_mapping.at("vertices"));
      if (rec_vert_info->ndim != 1) {
	throw snde_error("normal_calculation: vertices dimensionality must be 1");
      }
      snde_index numverts = rec_vert_info->dimlen[0];

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numtris*sizeof(snde_triangle) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord3) + numtris*sizeof(snde_trivertnormals) + numtris*sizeof(snde_coord3), // data_bytes for transfer
							numtris*(3+5+3+5+9+5), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
							   0, //metadata_bytes
							   numtris*sizeof(snde_triangle) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord3) + numtris*sizeof(snde_trivertnormals) + numtris*sizeof(snde_coord3),
							   0, // cpu_flops
							   numtris*(3+5+3+5+9+5), // gpuflops
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
    std::shared_ptr<meshed_trinormals_recording> result_rec;
    result_rec = create_recording_math<meshed_trinormals_recording>(get_result_channel_path(0),rss);
    
    return std::make_shared<metadata_function_override_type>([ this,result_rec,recording ]() {
      // metadata code
      std::unordered_map<std::string,metadatum> metadata;
      //printf("metadata()\n");
      //metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));
      
      result_rec->metadata=std::make_shared<immutable_metadata>(metadata);
      result_rec->mark_metadata_done();
      
      return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,recording ]() {
	// lock_alloc code
	
	std::shared_ptr<graphics_storage_manager> graphman = std::dynamic_pointer_cast<graphics_storage_manager>(result_rec->assign_storage_manager());

	if (!graphman) {
	  throw snde_error("Normal_calculation: Output arrays must be managed by a graphics storage manager");
	}

	std::shared_ptr<graphics_storage> leader_storage = std::dynamic_pointer_cast<graphics_storage>(recording->storage.at(recording->name_mapping.at("triangles")));

	snde_index addr = leader_storage->base_index;
	snde_index nmemb = leader_storage->nelem;


	std::shared_ptr<graphics_storage> trinormals_storage = graphman->storage_from_allocation(result_rec->info->name,leader_storage,"trinormals",result_rec->info->revision,rss->unique_index,addr,sizeof(*graphman->geom.trinormals),rtn_typemap.at(typeid(*graphman->geom.trinormals)),nmemb);
	result_rec->assign_storage(trinormals_storage,"trinormals",{nmemb});


	//parts_ref = recording->reference_ndarray("parts")
	
	
	// locking is only required for certain recordings
	// with special storage under certain conditions,
	// however it is always good to explicitly request
	// the locks, as the locking is a no-op if
	// locking is not actually required.
	rwlock_token_set locktokens = lockmgr->lock_recording_arrays({
	    { recording, { "parts", false }}, // first element is recording_ref, 2nd parameter is false for read, true for write
	    { recording, { "triangles", false }},
	    { recording, {"edges", false }},
	    { recording, {"vertices", false}},
	    { result_rec,{"trinormals", true }}
	  },
#ifdef SNDE_OPENCL
	    true
#else
	    false
#endif
	  );
	
	return std::make_shared<exec_function_override_type>([ this,locktokens, result_rec,recording ]() {
	  // exec code
	  snde_ndarray_info *rec_tri_info = recording->ndinfo(recording->name_mapping.at("triangles"));
	  snde_index numtris = rec_tri_info->dimlen[0];
	  
#ifdef SNDE_OPENCL
	  std::shared_ptr<assigned_compute_resource_opencl> opencl_resource=std::dynamic_pointer_cast<assigned_compute_resource_opencl>(compute_resource);
	  if (opencl_resource) {
	    
	    //fprintf(stderr,"Executing in OpenCL!\n");
	    cl::Kernel normal_kern = normalcalc_trinormals_opencl.get_kernel(opencl_resource->context,opencl_resource->devices.at(0));
	    OpenCLBuffers Buffers(opencl_resource->oclcache,opencl_resource->context,opencl_resource->devices.at(0),locktokens);
	    
	    Buffers.AddBufferAsKernelArg(recording,"parts",normal_kern,0,false,false);
	    Buffers.AddBufferAsKernelArg(recording,"triangles",normal_kern,1,false,false);
	    Buffers.AddBufferAsKernelArg(recording,"edges",normal_kern,2,false,false);
	    Buffers.AddBufferAsKernelArg(recording,"vertices",normal_kern,3,false,false);
	    Buffers.AddBufferAsKernelArg(result_rec,"trinormals",normal_kern,4,true,true);

      
	    cl::Event kerndone;
	    std::vector<cl::Event> FillEvents=Buffers.FillEvents();
	    
	    cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(normal_kern,{},{ numtris },{},&FillEvents,&kerndone);
	    if (err != CL_SUCCESS) {
	      throw openclerror(err,"Error enqueueing kernel");
	    }
	    opencl_resource->queues.at(0).flush(); /* trigger execution */
	    // mark that the kernel has modified result_rec
	    Buffers.BufferDirty(result_rec,"trinormals");
	    // wait for kernel execution and transfers to complete
	    Buffers.RemBuffers(kerndone,kerndone,true);
	    
	  } else {	    
#endif // SNDE_OPENCL
	    snde_warning("Performing normal calculation on CPU. This will be slow.");
	    //fprintf(stderr,"Not executing in OpenCL\n");
	    // Should OpenMP this (!)
	    const struct snde_part *parts =(snde_part *)recording->void_shifted_arrayptr("parts");
	    const snde_triangle *triangles=(snde_triangle *)recording->void_shifted_arrayptr("triangles");
	    const snde_edge *edges=(snde_edge *)recording->void_shifted_arrayptr("edges");
	    const snde_coord3 *vertices=(snde_coord3 *)recording->void_shifted_arrayptr("vertices");
	    //snde_trivertnormals *vertnormals=(snde_trivertnormals *)result_rec->void_shifted_arrayptr("vertnormals");
	    snde_coord3 *trinormals=(snde_coord3 *)result_rec->void_shifted_arrayptr("trinormals");
	    	    
	      
	    for (snde_index pos=0;pos < numtris;pos++){
	      trinormals[pos]=snde_normalcalc_triangle(parts,
						       triangles,
						       edges,
						       vertices,
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
    
  
  // !!!*** Need to fix initialization order ***!!!
  //std::shared_ptr<math_function> normal_calculation_trinormals_function = std::make_shared<cpp_math_function>([] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
  //  return std::make_shared<normal_calculation_trinormals>(rss,inst);
  //});

  std::shared_ptr<math_function> define_spatialnde2_trinormals_function()
  {
    return std::make_shared<cpp_math_function>("snde.trinormals",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<normal_calculation_trinormals>(rss,inst);
    }); 
  }

  SNDE_OCL_API std::shared_ptr<math_function> trinormals_function = define_spatialnde2_trinormals_function();
  
  static int registered_trinormals_function = register_math_function(trinormals_function);

  void instantiate_trinormals(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags)
  {
    std::string context = recdb_path_context(loaded_geom->info->name);
	
    std::shared_ptr<instantiated_math_function> instantiated = trinormals_function->instantiate( {
	std::make_shared<math_parameter_recording>("meshed")
      },
      {
	std::make_shared<std::string>("trinormals")
      },
      context,
      false, // is_mutable
      false, // ondemand
      false, // mdonly
      std::make_shared<math_definition>("instantiate_trinormals()"),
      {},
      nullptr);


    trans->recdb->add_math_function(trans,instantiated,true); // trinormals are generally hidden by default
    loaded_geom->processed_relpaths.emplace("trinormals","trinormals");

  }

  static int registered_trinormals_processor = register_geomproc_math_function("trinormals",instantiate_trinormals);



  // ... vertnormalarray starts here


  
#ifdef SNDE_OPENCL
  static opencl_program normalcalc_vertnormalarray_opencl("snde_normalcalc_vertnormalarray", { snde_types_h, geometry_types_h, vecops_h, geometry_ops_h, normal_calc_c });
#endif // SNDE_OPENCL
  
  class normal_calculation_vertnormalarray: public recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>> {
  public:
    normal_calculation_vertnormalarray(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec(rss,inst)
    {
      
    }
    
    // use default for decide_new_revision
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<meshed_part_recording> recording)
    {
      snde_ndarray_info *rec_tri_info = recording->ndinfo(recording->name_mapping.at("triangles"));
      if (rec_tri_info->ndim != 1) {
	throw snde_error("normal_calculation: triangle dimensionality must be 1");
      }
      snde_index numtris = rec_tri_info->dimlen[0];

      snde_ndarray_info *rec_edge_info = recording->ndinfo(recording->name_mapping.at("edges"));
      if (rec_edge_info->ndim != 1) {
	throw snde_error("normal_calculation: edge dimensionality must be 1");
      }
      snde_index numedges = rec_edge_info->dimlen[0];
      
      
      snde_ndarray_info *rec_vert_info = recording->ndinfo(recording->name_mapping.at("vertices"));
      if (rec_vert_info->ndim != 1) {
	throw snde_error("normal_calculation: vertices dimensionality must be 1");
      }
      snde_index numverts = rec_vert_info->dimlen[0];

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numtris*sizeof(snde_triangle) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord3) + numtris*sizeof(snde_trivertnormals) + numtris*sizeof(snde_coord3), // data_bytes for transfer
							numtris*(3+5+3+5+9+5), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
							   0, //metadata_bytes
							   numtris*sizeof(snde_triangle) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord3) + numtris*sizeof(snde_trivertnormals) + numtris*sizeof(snde_coord3),
							   0, // cpu_flops
							   numtris*(3+5+3+5+9+5), // gpuflops
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
    std::shared_ptr<meshed_vertnormalarrays_recording> result_rec;
    result_rec = create_recording_math<meshed_vertnormalarrays_recording>(get_result_channel_path(0),rss);
    
    return std::make_shared<metadata_function_override_type>([ this,result_rec,recording ]() {
      // metadata code
      std::unordered_map<std::string,metadatum> metadata;
      //printf("metadata()\n");
      //metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));
      
      result_rec->metadata=std::make_shared<immutable_metadata>(metadata);
      result_rec->mark_metadata_done();
      
      return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,recording ]() {
	// lock_alloc code
	
	std::shared_ptr<graphics_storage_manager> graphman = std::dynamic_pointer_cast<graphics_storage_manager>(result_rec->assign_storage_manager());

	if (!graphman) {
	  // Don't think special storage manager is really necessary for this as it's now just a rendering intermediate!!!***
	  throw snde_error("Normal_calculation: Output arrays must be managed by a graphics storage manager");
	}

	//std::shared_ptr<graphics_storage> leader_storage = recording->storage->at(recording->name_mapping.at("triangles"));

	//snde_index addr = leader_storage->base_index;
	//snde_index nmemb = leader_storage->nelem;
	
	snde_ndarray_info *rec_tri_info = recording->ndinfo(recording->name_mapping.at("triangles"));
	snde_index numtris = rec_tri_info->dimlen[0];
	
	
	//std::shared_ptr<graphics_storage> vertnormals_storage = std::dynamic_pointer_cast<graphics_storage>(graphman->allocate_recording(result_rec->info->name,"vertnormals",result_rec->info->revision,sizeof(*graphman->geom.vertnormals),rtn_typemap.at(typeid(*graphman->geom.vertnormals)),numtris,false));
	//result_rec->assign_storage(vertnormals_storage,"vertnormals",{numtris});
	result_rec->assign_storage_manager(this->recdb->default_storage_manager); // Force default storage manager so that we DON'T go to the graphics storage (which is unnecessary for temporary output such as this)

	result_rec->allocate_storage("vertnormal_arrays",{numtris},false);


	//parts_ref = recording->reference_ndarray("parts")
	
	
	// locking is only required for certain recordings
	// with special storage under certain conditions,
	// however it is always good to explicitly request
	// the locks, as the locking is a no-op if
	// locking is not actually required.
	rwlock_token_set locktokens = lockmgr->lock_recording_arrays({
	    { recording, { "parts", false }}, // first element is recording_ref, 2nd parameter is false for read, true for write
	    { recording, { "triangles", false }},
	    { recording, {"edges", false }},
	    { recording, {"vertices", false}},
	    { result_rec,{"vertnormal_arrays", true }}
	  },
#ifdef SNDE_OPENCL
	  true
#else
	  false
#endif
	  );
	
	return std::make_shared<exec_function_override_type>([ this,locktokens, result_rec,recording ]() {
	  // exec code
	  snde_ndarray_info *rec_tri_info = recording->ndinfo(recording->name_mapping.at("triangles"));
	  snde_index numtris = rec_tri_info->dimlen[0];
	  
#ifdef SNDE_OPENCL
	  std::shared_ptr<assigned_compute_resource_opencl> opencl_resource=std::dynamic_pointer_cast<assigned_compute_resource_opencl>(compute_resource);
	  if (opencl_resource) {
	    
	    //fprintf(stderr,"Executing in OpenCL!\n");
	    cl::Kernel normal_kern = normalcalc_vertnormalarray_opencl.get_kernel(opencl_resource->context,opencl_resource->devices.at(0));
	    OpenCLBuffers Buffers(opencl_resource->oclcache,opencl_resource->context,opencl_resource->devices.at(0),locktokens);
	    
	    Buffers.AddBufferAsKernelArg(recording,"parts",normal_kern,0,false,false);
	    Buffers.AddBufferAsKernelArg(recording,"triangles",normal_kern,1,false,false);
	    Buffers.AddBufferAsKernelArg(recording,"edges",normal_kern,2,false,false);
	    Buffers.AddBufferAsKernelArg(recording,"vertices",normal_kern,3,false,false);
	    Buffers.AddBufferAsKernelArg(result_rec,"vertnormal_arrays",normal_kern,4,true,true);

      
	    cl::Event kerndone;
	    std::vector<cl::Event> FillEvents=Buffers.FillEvents();
	    
	    cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(normal_kern,{},{ numtris },{},&FillEvents,&kerndone);
	    if (err != CL_SUCCESS) {
	      throw openclerror(err,"Error enqueueing kernel");
	    }
	    opencl_resource->queues.at(0).flush(); /* trigger execution */
	    // mark that the kernel has modified result_rec
	    Buffers.BufferDirty(result_rec,"vertnormal_arrays");
	    // wait for kernel execution and transfers to complete
	    Buffers.RemBuffers(kerndone,kerndone,true);
	    
	  } else {	    
#endif // SNDE_OPENCL
	    snde_warning("Performing normal calculation on CPU. This will be slow.");
	    //fprintf(stderr,"Not executing in OpenCL\n");
	    // Should OpenMP this (!)
	    const struct snde_part *parts =(snde_part *)recording->void_shifted_arrayptr("parts");
	    const snde_triangle *triangles=(snde_triangle *)recording->void_shifted_arrayptr("triangles");
	    const snde_edge *edges=(snde_edge *)recording->void_shifted_arrayptr("edges");
	    const snde_coord3 *vertices=(snde_coord3 *)recording->void_shifted_arrayptr("vertices");
	    snde_trivertnormals *vertnormalarray=(snde_trivertnormals *)result_rec->void_shifted_arrayptr("vertnormal_arrays");
	    //snde_coord3 *trinormals=(snde_coord3 *)result_rec->void_shifted_arrayptr("trinormals");
	    	    
	      
	    for (snde_index pos=0;pos < numtris;pos++){
	      snde_coord3 trinormal = snde_normalcalc_triangle(parts,
								triangles,
								edges,
								vertices,
								pos);
	      vertnormalarray[pos].vertnorms[0]=trinormal;
	      vertnormalarray[pos].vertnorms[1]=trinormal;
	      vertnormalarray[pos].vertnorms[2]=trinormal;
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
    
  std::shared_ptr<math_function> define_vertnormalarray_recording_function()
  {
    return std::make_shared<cpp_math_function>("snde.vertnormalarray",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<normal_calculation_vertnormalarray>(rss,inst);
    });
    
  }

  SNDE_OCL_API std::shared_ptr<math_function> vertnormalarray_recording_function=define_vertnormalarray_recording_function();
  
  static int registered_vertnormalarray_function = register_math_function(vertnormalarray_recording_function);
  


  
  // ... vertnormals starts here


  
#ifdef SNDE_OPENCL
  static opencl_program normalcalc_vertnormals_opencl("snde_normalcalc_vertnormals", { snde_types_h, geometry_types_h, vecops_h, geometry_ops_h, normal_calc_c });
#endif // SNDE_OPENCL
  
  class normal_calculation_vertnormals: public recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>,std::shared_ptr<meshed_trinormals_recording>> {
  public:
    normal_calculation_vertnormals(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec(rss,inst)
    {
      
    }
    
    // use default for decide_new_revision
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<meshed_part_recording> recording,std::shared_ptr<meshed_trinormals_recording> trinormals_rec)
    {
      snde_ndarray_info *rec_tri_info = recording->ndinfo(recording->name_mapping.at("triangles"));
      if (rec_tri_info->ndim != 1) {
	throw snde_error("normal_calculation: triangle dimensionality must be 1");
      }
      snde_index numtris = rec_tri_info->dimlen[0];

      snde_ndarray_info *rec_edge_info = recording->ndinfo(recording->name_mapping.at("edges"));
      if (rec_edge_info->ndim != 1) {
	throw snde_error("normal_calculation: edge dimensionality must be 1");
      }
      snde_index numedges = rec_edge_info->dimlen[0];
      
      
      snde_ndarray_info *rec_vert_info = recording->ndinfo(recording->name_mapping.at("vertices"));
      if (rec_vert_info->ndim != 1) {
	throw snde_error("normal_calculation: vertices dimensionality must be 1");
      }
      snde_index numverts = rec_vert_info->dimlen[0];

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numtris*sizeof(snde_triangle) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord3) + numtris*sizeof(snde_trivertnormals) + numtris*sizeof(snde_coord3), // data_bytes for transfer
							numtris*(3+5+3+5+9+5), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
#ifdef SNDE_OPENCL
	  std::make_shared<compute_resource_option_opencl>(std::set<std::string>(), // no tags
							   0, //metadata_bytes
							   numtris*sizeof(snde_triangle) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord3) + numtris*sizeof(snde_trivertnormals) + numtris*sizeof(snde_coord3),
							   0, // cpu_flops
							   numtris*(3+5+3+5+9+5), // gpuflops
							   1, // max effective cpu cores
							   1, // useful_cpu_cores (min # of cores to supply
							   snde_doubleprec_coords()), // requires_doubleprec 
#endif // SNDE_OPENCL
	};
      return std::make_pair(option_list,nullptr);
    }
  
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<meshed_part_recording> recording,std::shared_ptr<meshed_trinormals_recording> trinormals_rec) 
  {
    // define_recs code
    //printf("define_recs()\n");
    std::shared_ptr<meshed_vertnormals_recording> result_rec;
    result_rec = create_recording_math<meshed_vertnormals_recording>(get_result_channel_path(0),rss);
    
    return std::make_shared<metadata_function_override_type>([ this,result_rec,recording,trinormals_rec ]() {
      // metadata code
      std::unordered_map<std::string,metadatum> metadata;
      //printf("metadata()\n");
      //metadata.emplace("Test_metadata_entry",metadatum("Test_metadata_entry",3.14));
      
      result_rec->metadata=std::make_shared<immutable_metadata>(metadata);
      result_rec->mark_metadata_done();
      
      return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,recording,trinormals_rec ]() {
	// lock_alloc code
	
	std::shared_ptr<graphics_storage_manager> graphman = std::dynamic_pointer_cast<graphics_storage_manager>(result_rec->assign_storage_manager());

	if (!graphman) {
	  throw snde_error("Normal_calculation: Output arrays must be managed by a graphics storage manager");
	}

	//std::shared_ptr<graphics_storage> leader_storage = recording->storage->at(recording->name_mapping.at("triangles"));

	//snde_index addr = leader_storage->base_index;
	//snde_index nmemb = leader_storage->nelem;
	
	snde_ndarray_info *rec_vert_info = recording->ndinfo(recording->name_mapping.at("vertices"));
	if (rec_vert_info->ndim != 1) {
	  throw snde_error("normal_calculation: vertices dimensionality must be 1");
	}
	snde_index numverts = rec_vert_info->dimlen[0];
	//snde_ndarray_info *rec_tri_info = recording->ndinfo(recording->name_mapping.at("triangles"));
	//snde_index numtris = rec_tri_info->dimlen[0];
	
	
	//std::shared_ptr<graphics_storage> vertnormals_storage = std::dynamic_pointer_cast<graphics_storage>(graphman->allocate_recording(result_rec->info->name,"vertnormals",result_rec->info->revision,sizeof(*graphman->geom.vertnormals),rtn_typemap.at(typeid(*graphman->geom.vertnormals)),numtris,false));
	//result_rec->assign_storage(vertnormals_storage,"vertnormals",{numtris});


	
	//result_rec->allocate_storage("vertnormals",{numverts},false);
	std::shared_ptr<graphics_storage> leader_storage = std::dynamic_pointer_cast<graphics_storage>(recording->storage.at(recording->name_mapping.at("vertices")));

	snde_index addr = leader_storage->base_index;
	snde_index nmemb = leader_storage->nelem;
	
	std::shared_ptr<graphics_storage> vertnormals_storage = graphman->storage_from_allocation(result_rec->info->name,leader_storage,"vertnormals",result_rec->info->revision,rss->unique_index,addr,sizeof(*graphman->geom.vertnormals),rtn_typemap.at(typeid(*graphman->geom.vertnormals)),nmemb);
	result_rec->assign_storage(vertnormals_storage,"vertnormals",{nmemb});
	assert(nmemb == numverts);
	

	//parts_ref = recording->reference_ndarray("parts")
	
	
	// locking is only required for certain recordings
	// with special storage under certain conditions,
	// however it is always good to explicitly request
	// the locks, as the locking is a no-op if
	// locking is not actually required.
	rwlock_token_set locktokens = lockmgr->lock_recording_arrays({
	    { recording, { "parts", false }}, // first element is recording_ref, 2nd parameter is false for read, true for write
	    { recording, { "triangles", false }},
	    { recording, {"edges", false }},
	    { recording, {"vertices", false}}, 
	    { recording, {"vertex_edgelist_indices", false}}, 
	    { recording, {"vertex_edgelist", false}}, 
	    { trinormals_rec, {"trinormals", false}}, 
	    { result_rec,{"vertnormals", true }}
	  },
#ifdef SNDE_OPENCL
	  true
#else
	  false
#endif
	  );
	
	return std::make_shared<exec_function_override_type>([ this,locktokens, result_rec,recording,trinormals_rec,numverts ]() {
	  // exec code
	  snde_ndarray_info *rec_tri_info = recording->ndinfo(recording->name_mapping.at("triangles"));
	  
#ifdef SNDE_OPENCL
	  std::shared_ptr<assigned_compute_resource_opencl> opencl_resource=std::dynamic_pointer_cast<assigned_compute_resource_opencl>(compute_resource);
	  if (opencl_resource) {
	    
	    //fprintf(stderr,"Executing in OpenCL!\n");
	    cl::Kernel normal_kern = normalcalc_vertnormals_opencl.get_kernel(opencl_resource->context,opencl_resource->devices.at(0));
	    OpenCLBuffers Buffers(opencl_resource->oclcache,opencl_resource->context,opencl_resource->devices.at(0),locktokens);
	    
	    Buffers.AddBufferAsKernelArg(recording,"parts",normal_kern,0,false,false);
	    Buffers.AddBufferAsKernelArg(recording,"triangles",normal_kern,1,false,false);
	    Buffers.AddBufferAsKernelArg(recording,"edges",normal_kern,2,false,false);
	    Buffers.AddBufferAsKernelArg(recording,"vertices",normal_kern,3,false,false);
	    Buffers.AddBufferAsKernelArg(recording,"vertex_edgelist_indices",normal_kern,4,false,false);
	    Buffers.AddBufferAsKernelArg(recording,"vertex_edgelist",normal_kern,5,false,false);
	    Buffers.AddBufferAsKernelArg(trinormals_rec,"trinormals",normal_kern,6,false,false);
	    Buffers.AddBufferAsKernelArg(result_rec,"vertnormals",normal_kern,7,true,true);

      
	    cl::Event kerndone;
	    std::vector<cl::Event> FillEvents=Buffers.FillEvents();
	    
	    cl_int err = opencl_resource->queues.at(0).enqueueNDRangeKernel(normal_kern,{},{ numverts },{},&FillEvents,&kerndone);
	    if (err != CL_SUCCESS) {
	      throw openclerror(err,"Error enqueueing kernel");
	    }
	    opencl_resource->queues.at(0).flush(); /* trigger execution */
	    // mark that the kernel has modified result_rec
	    Buffers.BufferDirty(result_rec,"vertnormals");
	    // wait for kernel execution and transfers to complete
	    Buffers.RemBuffers(kerndone,kerndone,true);
	    
	  } else {	    
#endif // SNDE_OPENCL
	    snde_warning("Performing normal calculation on CPU. This will be slow.");
	    //fprintf(stderr,"Not executing in OpenCL\n");
	    // Should OpenMP this (!)
	    const struct snde_part *parts =(snde_part *)recording->void_shifted_arrayptr("parts");
	    const snde_triangle *triangles=(snde_triangle *)recording->void_shifted_arrayptr("triangles");
	    const snde_edge *edges=(snde_edge *)recording->void_shifted_arrayptr("edges");
	    const snde_coord3 *vertices=(snde_coord3 *)recording->void_shifted_arrayptr("vertices");
	    const snde_vertex_edgelist_index *vertex_edgelist_indices=(snde_vertex_edgelist_index *)recording->void_shifted_arrayptr("vertex_edgelist_indices");
	    const snde_index *vertex_edgelist = (snde_index *)recording->void_shifted_arrayptr("vertex_edgelist");
	    const snde_coord3 *trinormals=(snde_coord3 *)trinormals_rec->void_shifted_arrayptr("trinormals");
	    snde_coord3 *vertnormals=(snde_coord3 *)result_rec->void_shifted_arrayptr("vertnormals");
	    //snde_coord3 *trinormals=(snde_coord3 *)result_rec->void_shifted_arrayptr("trinormals");
	    	    
	      
	    for (snde_index pos=0;pos < numverts;pos++){
	      vertnormals[pos] = snde_normalcalc_vertex(parts,
							triangles,
							edges,
							vertices,
							vertex_edgelist_indices,
							vertex_edgelist,
							trinormals,
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
    
  std::shared_ptr<math_function> define_vertnormals_recording_function()
  {
    return std::make_shared<cpp_math_function>("snde.vertnormals_recording",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<normal_calculation_vertnormals>(rss,inst);
    });
    
  }

  SNDE_OCL_API std::shared_ptr<math_function> vertnormals_recording_function=define_vertnormals_recording_function();
  
  static int registered_vertnormals_function = register_math_function(vertnormals_recording_function);
  

  void instantiate_vertnormals(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags)
  {
    std::string context = recdb_path_context(loaded_geom->info->name);

    geomproc_specify_dependency(remaining_processing_tags,all_processing_tags,"trinormals"); // we require trinormals 
    std::shared_ptr<instantiated_math_function> instantiated = vertnormals_recording_function->instantiate( {
	std::make_shared<math_parameter_recording>("meshed"),
	std::make_shared<math_parameter_recording>("trinormals")
      },
      {
	std::make_shared<std::string>("vertnormals")
      },
      context,
      false, // is_mutable
      false, // ondemand
      false, // mdonly
      std::make_shared<math_definition>("instantiate_vertnormals()"),
      {},
      nullptr);

    trans->recdb->add_math_function(trans,instantiated,true); // vertnormals are generally hidden by default
    loaded_geom->processed_relpaths.emplace("vertnormals","vertnormals");
    
    //recdb_path_join(context,loaded_geom->processed_relpaths.at("meshed"))
    
  }

  static int registered_vertnormals_processor = register_geomproc_math_function("vertnormals",instantiate_vertnormals);

  
};
