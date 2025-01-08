#include <Eigen/Dense>


#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/vecops.h"
#include "snde/geometry_ops.h"
#include "snde/projinfo_calc.h"
#include "snde/geometrydata.h"
#include "snde/quaternion.h"
#include "snde/raytrace.h"
#include "snde/recstore.hpp"
#include "snde/recmath_cppfunction.hpp"
#include "snde/graphics_recording.hpp"
#include "snde/graphics_storage.hpp"

#include "snde/project_onto_parameterization.hpp"

namespace snde {

  template <typename T>
  class project_point_onto_parameterization: public recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>,std::shared_ptr<meshed_parameterization_recording>,std::shared_ptr<meshed_trinormals_recording>,std::shared_ptr<boxes3d_recording>,std::shared_ptr<meshed_projinfo_recording>,std::shared_ptr<meshed_inplanemat_recording>,std::shared_ptr<recording_base>,std::shared_ptr<recording_base>,std::shared_ptr<ndtyped_recording_ref<T>>,double,double,double,snde_index,snde_index,snde_bool> {
  public:
    project_point_onto_parameterization(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>,std::shared_ptr<meshed_parameterization_recording>,std::shared_ptr<meshed_trinormals_recording>,std::shared_ptr<boxes3d_recording>,std::shared_ptr<meshed_projinfo_recording>,std::shared_ptr<meshed_inplanemat_recording>,std::shared_ptr<recording_base>,std::shared_ptr<recording_base>,std::shared_ptr<ndtyped_recording_ref<T>>,double,double,double,snde_index,snde_index,snde_bool>(rss,inst)
    {
      
    }


        // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
    typedef typename snde::recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>,std::shared_ptr<meshed_parameterization_recording>,std::shared_ptr<meshed_trinormals_recording>,std::shared_ptr<boxes3d_recording>,std::shared_ptr<meshed_projinfo_recording>,std::shared_ptr<meshed_inplanemat_recording>,std::shared_ptr<recording_base>,std::shared_ptr<recording_base>,std::shared_ptr<ndtyped_recording_ref<T>>,double,double,double,snde_index,snde_index,snde_bool>::compute_options_function_override_type compute_options_function_override_type;
    typedef typename snde::recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>,std::shared_ptr<meshed_parameterization_recording>,std::shared_ptr<meshed_trinormals_recording>,std::shared_ptr<boxes3d_recording>,std::shared_ptr<meshed_projinfo_recording>,std::shared_ptr<meshed_inplanemat_recording>,std::shared_ptr<recording_base>,std::shared_ptr<recording_base>,std::shared_ptr<ndtyped_recording_ref<T>>,double,double,double,snde_index,snde_index,snde_bool>::define_recs_function_override_type define_recs_function_override_type;
    typedef typename snde::recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>,std::shared_ptr<meshed_parameterization_recording>,std::shared_ptr<meshed_trinormals_recording>,std::shared_ptr<boxes3d_recording>,std::shared_ptr<meshed_projinfo_recording>,std::shared_ptr<meshed_inplanemat_recording>,std::shared_ptr<recording_base>,std::shared_ptr<recording_base>,std::shared_ptr<ndtyped_recording_ref<T>>,double,double,double,snde_index,snde_index,snde_bool>::metadata_function_override_type metadata_function_override_type;
    typedef typename snde::recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>,std::shared_ptr<meshed_parameterization_recording>,std::shared_ptr<meshed_trinormals_recording>,std::shared_ptr<boxes3d_recording>,std::shared_ptr<meshed_projinfo_recording>,std::shared_ptr<meshed_inplanemat_recording>,std::shared_ptr<recording_base>,std::shared_ptr<recording_base>,std::shared_ptr<ndtyped_recording_ref<T>>,double,double,double,snde_index,snde_index,snde_bool>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename snde::recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>,std::shared_ptr<meshed_parameterization_recording>,std::shared_ptr<meshed_trinormals_recording>,std::shared_ptr<boxes3d_recording>,std::shared_ptr<meshed_projinfo_recording>,std::shared_ptr<meshed_inplanemat_recording>,std::shared_ptr<recording_base>,std::shared_ptr<recording_base>,std::shared_ptr<ndtyped_recording_ref<T>>,double,double,double,snde_index,snde_index,snde_bool>::exec_function_override_type exec_function_override_type;

    
    // use default for decide_execution
    std::pair<bool,std::shared_ptr<compute_options_function_override_type>>
    decide_execution(std::shared_ptr<meshed_part_recording> part,
		std::shared_ptr<meshed_parameterization_recording> param,
		std::shared_ptr<meshed_trinormals_recording> trinormals,
		std::shared_ptr<boxes3d_recording> boxes3d,
		std::shared_ptr<meshed_projinfo_recording> projinfo,
		std::shared_ptr<meshed_inplanemat_recording> inplanemat,
		std::shared_ptr<recording_base> part_orientation,
		std::shared_ptr<recording_base> source_orientation,
		std::shared_ptr<ndtyped_recording_ref<T>> to_project,
		double min_dist,
		double max_dist,
		double radius,
		snde_index horizontal_pixels,
		snde_index vertical_pixels, 
		snde_bool use_surface_normal)
    {
      
      
      std::shared_ptr<multi_ndarray_recording> part_ndarray=std::dynamic_pointer_cast<multi_ndarray_recording>(part_orientation);
      if (!part_ndarray) {
	return std::make_pair(false,nullptr);
      }
      std::shared_ptr<ndarray_recording_ref> part_orientation_ref=part_ndarray->reference_ndarray(0);
      if (!part_orientation_ref) {
	return std::make_pair(false,nullptr);
      }
      std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> part_orientation_tref=std::dynamic_pointer_cast<ndtyped_recording_ref<snde_orientation3>>(part_orientation_ref);

      
      
      std::shared_ptr<multi_ndarray_recording> source_ndarray=std::dynamic_pointer_cast<multi_ndarray_recording>(source_orientation);
      if (!source_ndarray) {
	return std::make_pair(false,nullptr);
      }
      std::shared_ptr<ndarray_recording_ref> source_orientation_ref=source_ndarray->reference_ndarray(0);
      if (!source_orientation_ref) {
	return std::make_pair(false,nullptr);
      }
      std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> source_orientation_tref=std::dynamic_pointer_cast<ndtyped_recording_ref<snde_orientation3>>(source_orientation_ref);



      if (!part_orientation_tref) {
	return std::make_pair(false,nullptr);
      }

      if (!source_orientation_tref) {
	return std::make_pair(false,nullptr);
      }
      return std::make_pair(true,nullptr);
    }
    
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>>
      compute_options(std::shared_ptr<meshed_part_recording> part,
		      std::shared_ptr<meshed_parameterization_recording> param,
		      std::shared_ptr<meshed_trinormals_recording> trinormals,
		      std::shared_ptr<boxes3d_recording> boxes3d,
		      std::shared_ptr<meshed_projinfo_recording> projinfo,
		      std::shared_ptr<meshed_inplanemat_recording> inplanemat,
		      std::shared_ptr<recording_base> part_orientation,
		      std::shared_ptr<recording_base> source_orientation,
		      std::shared_ptr<ndtyped_recording_ref<T>> to_project,
		      double min_dist,
		      double max_dist,
		      double radius,
		      snde_index horizontal_pixels,
		      snde_index vertical_pixels, 
		      snde_bool use_surface_normal)
    {

      /*
      {
	// debugging
	snde_warning("project_onto_parameterization(): compute_options on globalrev %llu",(unsigned long long)std::dynamic_pointer_cast<globalrevision>(this->rss)->globalrev);
      }
      */
      
      snde_ndarray_info *rec_tri_info = part->ndinfo(part->name_mapping.at("triangles"));
      if (rec_tri_info->ndim != 1) {
	throw snde_error("project_onto_parameterization: triangle dimensionality must be 1");
      }
      snde_index numtris = rec_tri_info->dimlen[0];

      snde_ndarray_info *rec_edge_info = part->ndinfo(part->name_mapping.at("edges"));
      if (rec_edge_info->ndim != 1) {
	throw snde_error("project_onto_parameterization: edge dimensionality must be 1");
      }
      snde_index numedges = rec_edge_info->dimlen[0];
      
      
      snde_ndarray_info *rec_vert_info = part->ndinfo(part->name_mapping.at("vertices"));
      if (rec_vert_info->ndim != 1) {
	throw snde_error("project_onto_parameterization: vertices dimensionality must be 1");
      }
      snde_index numverts = rec_vert_info->dimlen[0];

      
      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numtris*sizeof(snde_triangle) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord3) + numtris*sizeof(snde_trivertnormals) + numtris*sizeof(snde_cmat23), // data_bytes for transfer
							0., // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
	};
      return std::make_pair(option_list,nullptr);
    }
  
    std::shared_ptr<metadata_function_override_type>
    define_recs(std::shared_ptr<meshed_part_recording> part,
		std::shared_ptr<meshed_parameterization_recording> param,
		std::shared_ptr<meshed_trinormals_recording> trinormals,
		std::shared_ptr<boxes3d_recording> boxes3d,
		std::shared_ptr<meshed_projinfo_recording> projinfo,
		std::shared_ptr<meshed_inplanemat_recording> inplanemat,
		std::shared_ptr<recording_base> part_orientation,
		std::shared_ptr<recording_base> source_orientation,
		std::shared_ptr<ndtyped_recording_ref<T>> to_project,
		double min_dist,
		double max_dist,
		double radius,
		snde_index horizontal_pixels,
		snde_index vertical_pixels, 
		snde_bool use_surface_normal)
    {
      // define_recs code
      /*
      {
	// debugging
	snde_warning("project_onto_parameterization(): define_recs on globalrev %llu",(unsigned long long)std::dynamic_pointer_cast<globalrevision>(this->rss)->globalrev);
      }
      */
      to_project->assert_no_scale_or_offset(this->inst->definition->definition_command);

      // determine real vs. complex
      bool is_complex; 
      const std::set<unsigned> &compatible_with_imagedata = rtn_compatible_types.at(SNDE_RTN_SNDE_IMAGEDATA);
      const std::set<unsigned> &compatible_with_complex_imagedata = rtn_compatible_types.at(SNDE_RTN_SNDE_COMPLEXIMAGEDATA);
      
      if (to_project->typenum==SNDE_RTN_SNDE_IMAGEDATA || compatible_with_imagedata.find(to_project->typenum) != compatible_with_imagedata.end()) {
	is_complex=false; 
      } else if (to_project->typenum==SNDE_RTN_SNDE_COMPLEXIMAGEDATA || compatible_with_complex_imagedata.find(to_project->typenum) != compatible_with_complex_imagedata.end()) {
	is_complex=true;	
      } else {
	assert(0); // if tihs triggers then the typechecking here must have diverged from the code in define_spatialnde2_project_point_onto_parameterization_function(), below. 
      }


      
      std::shared_ptr<fusion_ndarray_recording> result_rec;
      result_rec = create_recording_math<fusion_ndarray_recording>(this->get_result_channel_path(0),this->rss,is_complex ? SNDE_RTN_SNDE_COMPLEXIMAGEDATA:SNDE_RTN_SNDE_IMAGEDATA); // defines two ndarrays: "accumulator" and "totals"
      result_rec->info->immutable = false;
      
      
      return std::make_shared<metadata_function_override_type>([ this,
								 part,
								 param,
								 trinormals,
								 boxes3d,
								 projinfo,
								 inplanemat,
								 part_orientation,
								 source_orientation,
								 to_project,
								 min_dist,
								 max_dist,
								 radius,
								 horizontal_pixels,
								 vertical_pixels,
								 use_surface_normal,
								 result_rec,
								 is_complex ]() {
	/*
	{
	  // debugging
	  snde_warning("project_onto_parameterization(): metadata on globalrev %llu",(unsigned long long)std::dynamic_pointer_cast<globalrevision>(this->rss)->globalrev);
	}
	*/
	
	// metadata code
	std::shared_ptr<constructible_metadata> metadata=std::make_shared<constructible_metadata>(*to_project->rec->metadata);

	// override any previous render goal with default
	metadata->AddMetaDatum(metadatum("snde_render_goal","SNDE_SRG_RENDERING"));


	snde_coord min_u = 0.0;
	snde_coord max_u = 0.0;
	snde_coord min_v = 1.0;
	snde_coord max_v = 1.0;
	
	snde_index numuvpatches=0;
	{
	  // pull out critical information from our parameter recordings
	  rwlock_token_set initialization_locktokens = this->lockmgr->lock_recording_arrays({
	      //{ part, { "parts", false }}, // first element is recording_ref, 2nd parameter is false for read, true for write
	      //{ param, { "uvs", false }},
	      { param, { "uv_patches", false }},
	    },
	    false
	    );

	  std::shared_ptr<ndtyped_recording_ref<snde_parameterization>> param_ref = param->reference_typed_ndarray<snde_parameterization>("uvs");
	  numuvpatches = param_ref->element(0).numuvpatches;
	  assert(numuvpatches==1); // only support a single patch for now
	  
	  // Now lock the boxes; should be OK because boxes are after the parts, uvs, and uv_patches
	  // in the structure, thus they are later in the locking order
	  
	  //rwlock_token_set initialization_locktokens2 = lockmgr->lock_recording_arrays({
	  //    //{ param, { "uv_boxes0", false }}, // NOTE: To support multiple patches we would need to lock uv_boxes1, etc. 
	  //    { param, { "uv_boxcoord0", false }}, 
	  //  },
	  //  false
	  //  );
	  
	  std::shared_ptr<ndtyped_recording_ref<snde_parameterization_patch>> patch_ref = param->reference_typed_ndarray<snde_parameterization_patch>("uv_patches");
	  snde_boxcoord2 uv_domain = patch_ref->element(0).domain;
	  
	  min_u = uv_domain.min.coord[0]; 
	  max_u = uv_domain.max.coord[0]; 
	  min_v = uv_domain.min.coord[1]; 
	  max_v = uv_domain.max.coord[1]; 
	}
	
	
	std::string coord0 = "U Position"; 
	std::string units0 = "meters";   
	std::string coord1 = "V Position";
	std::string units1 = "meters";   
	std::string ampl_units = to_project->rec->metadata->GetMetaDatumStr("ande_array-ampl_units","Volts");
	std::string ampl_coord = to_project->rec->metadata->GetMetaDatumStr("ande_array-ampl_coord","Volts");

	double step0 = (max_u-min_u)/horizontal_pixels;
	double step1 = (max_v-min_v)/vertical_pixels;
	
	double inival0 = min_u + step0/2.0;
	double inival1 = min_v + step1/2.0;

	metadata->AddMetaDatum(metadatum("ande_array-axis0_scale",step0,units0));
	metadata->AddMetaDatum(metadatum("ande_array-axis0_offset",inival0,units0));
	metadata->AddMetaDatum(metadatum("ande_array-axis0_coord",coord0));
	//metadata->AddMetaDatum(metadatum("ande_array-axis0_units",units0));

	metadata->AddMetaDatum(metadatum("ande_array-axis1_scale",step1,units1));
	metadata->AddMetaDatum(metadatum("ande_array-axis1_offset",inival1,units1));
	metadata->AddMetaDatum(metadatum("ande_array-axis1_coord",coord1));
	//metadata->AddMetaDatum(metadatum("ande_array-axis1_units",units1));

	metadata->AddMetaDatum(metadatum("ande_array-ampl_coord",ampl_coord));
	metadata->AddMetaDatum(metadatum("ande_array-ampl_units",ampl_units));

	
	
	result_rec->metadata=metadata;
	result_rec->mark_metadata_done();
      
	return std::make_shared<lock_alloc_function_override_type>([ this,
								     part,
								     param,
								     trinormals,
								     boxes3d,
								     projinfo,
								     inplanemat,
								     part_orientation,
								     source_orientation,
								     to_project,
								     min_dist,
								     max_dist,
								     radius,
								     horizontal_pixels,
								     vertical_pixels,
								     use_surface_normal,
								     result_rec,
								     is_complex,
								     step0,inival0,coord0,units0,
								     step1,inival1,coord1,units1,
								     ampl_coord,ampl_units ]() {
	  // lock_alloc code
	  /*
	  {
	    // debugging
	    snde_warning("project_onto_parameterization(): lock_alloc on globalrev %llu",(unsigned long long)std::dynamic_pointer_cast<globalrevision>(this->rss)->globalrev);
	  }
	  */
	  
	  std::shared_ptr<graphics_storage_manager> graphman = std::dynamic_pointer_cast<graphics_storage_manager>(result_rec->assign_storage_manager());
	  
	  if (!graphman) {
	    throw snde_error("project_onto_parameterization: Output arrays must be managed by a graphics storage manager");
	  }
	  

	  bool build_on_previous = false; 
	  std::shared_ptr<multi_ndarray_recording> previous_recording_ndarray;
	  std::shared_ptr<recording_base> previous_recording = this->self_dependent_recordings.at(0);
	  std::shared_ptr<multi_ndarray_recording> previous_ndarray = std::dynamic_pointer_cast<multi_ndarray_recording>(previous_recording);
	  
	

	  unsigned typenum = rtn_typemap.at(typeid(T));

	  if (typenum==SNDE_RTN_FLOAT32 && sizeof(snde_float32)==sizeof(snde_imagedata)) {
	    typenum=SNDE_RTN_SNDE_IMAGEDATA;
	  }
	  
	  if (typenum==SNDE_RTN_COMPLEXFLOAT32 && sizeof(snde_complexfloat32)==sizeof(snde_compleximagedata)) {
	    typenum=SNDE_RTN_SNDE_COMPLEXIMAGEDATA;
	  }
	  
	  std::vector<snde_index> dimlen = { horizontal_pixels, vertical_pixels };

	  if (previous_ndarray && this->inst->is_mutable && !previous_ndarray->info->immutable) {
	    // check size compatibility, etc.

	    
	    if (previous_ndarray->mndinfo()->num_arrays==2) {
	      if (previous_ndarray->layouts.at(0).dimlen == dimlen &&
		  previous_ndarray->layouts.at(1).dimlen == dimlen &&
		  previous_ndarray->layouts.at(0).is_f_contiguous() &&
		  previous_ndarray->layouts.at(1).is_f_contiguous() &&
		  previous_ndarray->storage_manager == graphman &&
		  !previous_ndarray->info->immutable) {
		
		if (previous_ndarray->storage.at(0)->typenum == typenum && previous_ndarray->storage.at(1)->typenum == SNDE_RTN_SNDE_IMAGEDATA) {

		  
		  double previous_step0;
		  std::string previous_step0_units;
		  std::tie(previous_step0,previous_step0_units)=previous_ndarray->metadata->GetMetaDatumDblUnits("ande_array-axis0_scale",1.0,"meters");
		  
		  double previous_step1;
		  std::string previous_step1_units;
		  std::tie(previous_step1,previous_step1_units)=previous_ndarray->metadata->GetMetaDatumDblUnits("ande_array-axis1_scale",1.0,"meters");
		  
		  double previous_inival0;
		  std::string previous_inival0_units;
		  std::tie(previous_inival0,previous_inival0_units)=previous_ndarray->metadata->GetMetaDatumDblUnits("ande_array-axis0_offset",0.0,"meters");

		  double previous_inival1;
		  std::string previous_inival1_units;		  
		  std::tie(previous_inival1,previous_inival1_units)=previous_ndarray->metadata->GetMetaDatumDblUnits("ande_array-axis1_offset",0.0,"meters");
		  
		  std::string previous_coord0 = previous_ndarray->metadata->GetMetaDatumStr("ande_array-axis0_coord","X Position");
		  std::string previous_coord1 = previous_ndarray->metadata->GetMetaDatumStr("ande_array-axis1_coord","Y Position");
		  
		  
		  std::string previous_ampl_units = previous_ndarray->metadata->GetMetaDatumStr("ande_array-ampl_units","Volts");
		  std::string previous_ampl_coord = previous_ndarray->metadata->GetMetaDatumStr("ande_array-ampl_coord","Voltage");
		  
		  if (previous_step0 == step0 && previous_step1==step1 && previous_inival0==inival0 && previous_inival1==inival1 &&
		      previous_coord0 == coord0 && previous_coord1 == coord1 && previous_step0_units==units0 && previous_inival0_units==units0 && previous_step1_units==units1 && previous_inival1_units==units1 && 
		      previous_ampl_units == ampl_units && previous_ampl_coord == ampl_coord) {
		    
		    build_on_previous = true;
		  }
		}
	      }
	    }
	  }
	

	  //snde_warning("build_on_previous=%d",(int)build_on_previous);
      
	  if (!build_on_previous) {
	    result_rec->allocate_storage_in_named_array(0,is_complex ? "compleximagebuf":"imagebuf",dimlen,true); // storage for image
	    result_rec->allocate_storage_in_named_array(1,"imagebuf",dimlen,true); // storage for validity mask 
	  } else {
	    // accumulate on top of previous recording -- it is mutable storage!
	    result_rec->assign_storage_strides(previous_ndarray->storage.at(0),0,previous_ndarray->layouts.at(0).dimlen,previous_ndarray->layouts.at(0).strides);
	    result_rec->assign_storage_strides(previous_ndarray->storage.at(1),1,previous_ndarray->layouts.at(1).dimlen,previous_ndarray->layouts.at(1).strides);
	  }
	  
	  
	  std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> part_orientation_ref=std::dynamic_pointer_cast<multi_ndarray_recording>(part_orientation)->reference_typed_ndarray<snde_orientation3>(0);
	  std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> source_orientation_ref=std::dynamic_pointer_cast<multi_ndarray_recording>(source_orientation)->reference_typed_ndarray<snde_orientation3>(0);
	  
	  rwlock_token_set locktokens = this->lockmgr->lock_recording_refs({
	      { part->reference_ndarray("parts"), false }, // first element is recording_ref, 2nd parameter is false for read, true for write
	      { part->reference_ndarray("topos"), false }, // first element is recording_ref, 2nd parameter is false for read, true for write
		{ part->reference_ndarray("triangles"), false },
		{ part->reference_ndarray("edges"), false },
		{ part->reference_ndarray("vertices"), false},
		{ trinormals->reference_ndarray("trinormals"), false },
		{ inplanemat->reference_ndarray("inplanemats"),false},
		{ boxes3d->reference_ndarray("boxes"), false},
		{ boxes3d->reference_ndarray("boxcoord"), false},
		{ boxes3d->reference_ndarray("boxpolys"), false},
		{ param->reference_ndarray("uvs"),false},
		{ param->reference_ndarray("uv_topos"),false},
		{ param->reference_ndarray("uv_triangles"),false},
		{ projinfo->reference_ndarray("inplane2uvcoords"), false},
		{ part_orientation_ref, false },
		{ source_orientation_ref, false },
	    
		// projectionarray_image
		
		{ result_rec->reference_ndarray("accumulator"), true},
		{ result_rec->reference_ndarray("totals"), true }
	    },
	    false
	    );
	  
	  return std::make_shared<exec_function_override_type>([ this,
								 part,
								 param,
								 trinormals,
								 boxes3d,
								 projinfo,
								 inplanemat,
								 part_orientation, part_orientation_ref,
								 source_orientation, source_orientation_ref,
								 to_project,
								 min_dist,
								 max_dist,
								 radius,
								 horizontal_pixels,
								 vertical_pixels,
								 use_surface_normal,
								 result_rec,
								 is_complex,
								 step0,inival0,coord0,units0,
								 step1,inival1,coord1,units1,
								 ampl_coord,ampl_units,
								 build_on_previous, dimlen, locktokens ]() {
	    // exec code
	    /*
	    {
	      // debugging
	      snde_warning("project_onto_parameterization(): exec on globalrev %llu",(unsigned long long)std::dynamic_pointer_cast<globalrevision>(this->rss)->globalrev);
	    }
	    */
	    
	    if (!build_on_previous) {
	      // fill new buffer with all zeros. 
	      
	      snde_index nu = dimlen.at(0);
	      snde_index nv = dimlen.at(1);
	      
	      if (is_complex) {
		std::shared_ptr<ndtyped_recording_ref<snde_compleximagedata>> result_imagebuf = result_rec->reference_typed_ndarray<snde_compleximagedata>("accumulator");
		snde_compleximagedata *buf = result_imagebuf->shifted_arrayptr();
		
		snde_index su = result_imagebuf->layout.strides.at(0);
		snde_index sv = result_imagebuf->layout.strides.at(1);
		
		
		for (snde_index vcnt=0;vcnt < nv; vcnt++) {
		  for (snde_index ucnt=0;ucnt < nu; ucnt++) {
		    buf[ucnt*su + vcnt*sv].real=0.0;
		    buf[ucnt*su + vcnt*sv].imag=0.0;
		  }
		}

		

	      } else {
		std::shared_ptr<ndtyped_recording_ref<snde_imagedata>> result_imagebuf = result_rec->reference_typed_ndarray<snde_imagedata>("accumulator");
		snde_imagedata *buf = result_imagebuf->shifted_arrayptr();
		
		snde_index su = result_imagebuf->layout.strides.at(0);
		snde_index sv = result_imagebuf->layout.strides.at(1);
		
		
		for (snde_index vcnt=0;vcnt < nv; vcnt++) {
		  for (snde_index ucnt=0;ucnt < nu; ucnt++) {
		    buf[ucnt*su + vcnt*sv]=0.0;
		  }
		}


	      }



	      std::shared_ptr<ndtyped_recording_ref<snde_imagedata>> result_validitybuf = result_rec->reference_typed_ndarray<snde_imagedata>("totals");
	      snde_imagedata *buf = result_validitybuf->shifted_arrayptr();
	      
	      snde_index su = result_validitybuf->layout.strides.at(0);
	      snde_index sv = result_validitybuf->layout.strides.at(1);
		
	      
	      for (snde_index vcnt=0;vcnt < nv; vcnt++) {
		for (snde_index ucnt=0;ucnt < nu; ucnt++) {
		  buf[ucnt*su + vcnt*sv]=0.0;
		}
	      }
	      
	      
	      
	    }
	    
	    // call raytrace_camera_evaluate_zdist() or similar, then
	    // project_to_uv_arrays()

	    // assemble a struct snde_partinstance

	    snde_index partnum = part->ndinfo("parts")->base_index;
	    snde_index paramnum = param->ndinfo("uvs")->base_index;
	    
	    std::vector<snde_partinstance> instances;
	    snde_orientation3 orient_inv;
	    orientation_inverse(part_orientation_ref->element(0),&orient_inv);
	    
	    instances.push_back(snde_partinstance{
		.orientation = part_orientation_ref->element(0),
		.orientation_inverse = orient_inv,
		.partnum=partnum,  
		.firstuvpatch=0, // only support single patch for now
		.uvnum=paramnum,
	      });
	    
	    
	    snde_orientation3 sensorcoords_to_wrlcoords=source_orientation_ref->element(0);
	    if (!isnan(sensorcoords_to_wrlcoords.quat.coord[0])) {
	      
	      snde_orientation3 wrlcoords_to_sensorcoords;
	      orientation_inverse(sensorcoords_to_wrlcoords,&wrlcoords_to_sensorcoords);
	      
	      
	      snde_image projectionarray_info={
		.projectionbufoffset=0,
		.weightingbufoffset=0,
		.validitybufoffset=0,
		.nx=horizontal_pixels,
		.ny=vertical_pixels,
		.inival={{ (snde_coord)inival0,(snde_coord)inival1 }},
		.step={{ (snde_coord)step0,(snde_coord)step1 }},
		.projection_strides={ is_complex ? 2u:1u,is_complex ? (2*horizontal_pixels):horizontal_pixels },  // need to multiply by 2 if complex
		.weighting_strides={ 0,0 }, // don't use weighting
		.validity_strides={ 1,horizontal_pixels },
	      }; // !!!*** will need an array here if we start supporting multiple (u,v) patches ***!!!
	      struct rayintersection_properties imagedata_intersectprops;
	      snde_index *boxnum_stack;
	      snde_index frin_stacksize=boxes3d->metadata->GetMetaDatumUnsigned("snde_boxes3d_max_depth",10)*8+2;
	      boxnum_stack = (snde_index *)malloc(frin_stacksize*sizeof(*boxnum_stack));
	      
	      raytrace_sensor_evaluate_zdist(
					     sensorcoords_to_wrlcoords,
					     wrlcoords_to_sensorcoords,
					     min_dist,max_dist,
					     instances.data(),
					     instances.size(),
					     (snde_part *)part->void_shifted_arrayptr("parts"),part->ndinfo("parts")->base_index,
					     (snde_topological *)part->void_shifted_arrayptr("topos"),part->ndinfo("topos")->base_index,
					     (snde_triangle *)part->void_shifted_arrayptr("triangles"),part->ndinfo("triangles")->base_index,
					     (snde_coord3 *)trinormals->void_shifted_arrayptr("trinormals"),
					     (snde_cmat23 *)inplanemat->void_shifted_arrayptr("inplanemats"),
					     (snde_edge *)part->void_shifted_arrayptr("edges"),part->ndinfo("edges")->base_index,
					     (snde_coord3 *)part->void_shifted_arrayptr("vertices"),part->ndinfo("vertices")->base_index,
					     (snde_box3 *)boxes3d->void_shifted_arrayptr("boxes"),boxes3d->ndinfo("boxes")->base_index,
					     (snde_boxcoord3 *)boxes3d->void_shifted_arrayptr("boxcoord"),
					     (snde_index *)boxes3d->void_shifted_arrayptr("boxpolys"),boxes3d->ndinfo("boxpolys")->base_index,
					     (snde_parameterization *)param->void_shifted_arrayptr("uvs"),param->ndinfo("uvs")->base_index,
					     (snde_triangle *)param->void_shifted_arrayptr("uv_triangles"),param->ndinfo("uv_triangles")->base_index,
					     (snde_cmat23 *)projinfo->void_shifted_arrayptr("inplane2uvcoords"),
					     &projectionarray_info, // projectionarray_info, indexed according to the firstuvpatches of the partinstance, defines the layout of uvdata_angleofincidence_weighting and uvdata_angleofincidence_weighting_validity uv imagedata arrays
					     frin_stacksize,
					     boxnum_stack,
					     &imagedata_intersectprops); // JUST the structure for this pixel... we don't index it
	      
	      free(boxnum_stack);
	      
	      //snde_warning("project_onto_parameterization: intersects at u=%f, v=%f",imagedata_intersectprops.uvcoords.coord[0],imagedata_intersectprops.uvcoords.coord[1]);
	      
	      snde_imagedata real_pixelval=to_project->element_complexfloat64(0).real;
	      snde_imagedata pixelweighting=1.0;
	      
	      //snde_coord3 uvcoords = { imagedata_intersectprops.uvcoords.coord[0],imagedata_intersectprops.uvcoords.coord[1],1.0 };
	      //snde_coord min_radius_uv_pixels = 2.0; // external parameter?
	      std::shared_ptr<ndtyped_recording_ref<snde_parameterization>> uvs_ref = param->reference_typed_ndarray<snde_parameterization>("uvs");
	      
	      
	      //// debugging
	      //std::shared_ptr<ndtyped_recording_ref<snde_cmat23>> inplanemat_ref = inplanemat->reference_typed_ndarray<snde_cmat23>("inplanemats");
	      //const snde_cmat23 &inplanemat = inplanemat_ref->element(part->reference_typed_ndarray<snde_part>("parts")->element(0).firsttri + imagedata_intersectprops.trinum - part->ndinfo("triangles")->base_index);
	      
	      // end debugging
	      
	      std::shared_ptr<ndtyped_recording_ref<snde_cmat23>> inplane2uvcoords_ref = projinfo->reference_typed_ndarray<snde_cmat23>("inplane2uvcoords");
	      const snde_cmat23 &inplane2uvcoords = inplane2uvcoords_ref->element(uvs_ref->element(0).firstuvtri + imagedata_intersectprops.trinum - param->ndinfo("uv_triangles")->base_index);
	      
	      // Use the average of the magnitudes of the eigenvalues of the left 2x2 of inplane2uvcoords as an estimate of the scaling between physical in plane coordinates and "meaningful" uv coordinates
	      snde_coord i2uv_trace = inplane2uvcoords.row[0].coord[0] + inplane2uvcoords.row[1].coord[1];
	      snde_coord i2uv_det = inplane2uvcoords.row[0].coord[0]*inplane2uvcoords.row[1].coord[1] - inplane2uvcoords.row[1].coord[0]*inplane2uvcoords.row[0].coord[1];

	      // e-vals solutions of: lambda^2 - trace*lambda + det = 0
	      // lambda = trace/2 +/- (1/2)sqrt(trace^2 - 4*det)
	      
	      // magnitude = sqrt( (trace/2)^2 + (4*det - trace^2)/4
	      // magnitude = sqrt( (trace)^2/4 + (det - trace^2/4 )
	      // magnitude = sqrt( det)
	      snde_coord lambda_magnitude = sqrt(i2uv_det);
	      
	      assert(lambda_magnitude); // if this fails then inplane2uvcoords could be doing some kind of weird mirroring maybe (!?)
	      
	      snde_coord duvcoords_dinplane = lambda_magnitude;
	      snde_coord dpixels_duvcoords = 2.0/(step0+step1); // 1/average step size 
	      
	      
	      
	      snde_coord min_radius_uv_pixels = dpixels_duvcoords * duvcoords_dinplane * radius;
	      snde_coord min_radius_src_pixels = 0.0; // (has no effect)
	      snde_coord bandwidth_fraction = .4; // should this be an external parameter? 
	      
	      project_to_uv_arrays(real_pixelval,pixelweighting,
				   imagedata_intersectprops.uvcoords,nullptr,nullptr,
				   projectionarray_info,
				   (snde_atomicimagedata *)result_rec->void_shifted_arrayptr("accumulator"),
				   projectionarray_info.projection_strides,
				   nullptr, // OCL_GLOBAL_ADDR snde_imagedata *uvdata_weighting_arrays,
				   projectionarray_info.weighting_strides,
				   (snde_atomicimagedata *)result_rec->void_shifted_arrayptr("totals"), // OCL_GLOBAL_ADDR snde_atomicimagedata *uvdata_validity_arrays,
				   projectionarray_info.validity_strides,
				   min_radius_uv_pixels,min_radius_src_pixels,bandwidth_fraction);
	      
	      
	      if (is_complex) {
		// project imaginary part
		snde_imagedata imag_pixelval=to_project->element_complexfloat64(0).imag;
		
		project_to_uv_arrays(imag_pixelval,pixelweighting,
				     imagedata_intersectprops.uvcoords,nullptr,nullptr,
				     projectionarray_info,
				     ((snde_atomicimagedata *)result_rec->void_shifted_arrayptr("accumulator"))+1, // the +1 switches us from the real part to the imaginary part
				     projectionarray_info.projection_strides,
				     nullptr, // OCL_GLOBAL_ADDR snde_imagedata *uvdata_weighting_arrays,
				     projectionarray_info.weighting_strides,
				     nullptr,
				     projectionarray_info.validity_strides,
				     min_radius_uv_pixels,min_radius_src_pixels,bandwidth_fraction);
		
	      }
	      
	    
	      //snde_warning("Project_onto_parameterization calculation complete.");
	      // Need to mark our modification zone
	      // !!!*** Would be nice to be able to specify a rectangle here ***!!!
	      result_rec->storage.at(0)->mark_as_modified(nullptr,0,result_rec->storage.at(0)->nelem);
	      result_rec->storage.at(1)->mark_as_modified(nullptr,0,result_rec->storage.at(1)->nelem);
	    }
	    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	    result_rec->mark_data_ready();
	    
	  }); 
	});
      });
    };
    
  };
  
  
  std::shared_ptr<math_function> define_spatialnde2_project_point_onto_parameterization_function()
  {
    std::shared_ptr<math_function> newfunc = std::make_shared<cpp_math_function>("snde.project_point_onto_parameterization",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) -> std::shared_ptr<executing_math_function> {
      if (!inst) {
	// initial call with no instantiation to probe parameters; just use snde_imagedata case
	return std::make_shared<project_point_onto_parameterization<snde_imagedata>>(rss,inst);
      }

      std::shared_ptr<math_parameter> to_project = inst->parameters.at(8); // to_project is our 8th parameter: Use it for the type hint
      
      assert(to_project->paramtype==SNDE_MFPT_RECORDING);
      
      std::shared_ptr<math_parameter_recording> to_project_rec = std::dynamic_pointer_cast<math_parameter_recording>(to_project);
      
      assert(to_project);
      
      std::shared_ptr<ndarray_recording_ref> to_project_rec_ref = to_project_rec->get_ndarray_recording_ref(rss,inst->channel_path_context,inst->definition,1);

      // ***!!! Important: keep this typechecking consistent with the code in define_recs(), above
      const std::set<unsigned> &compatible_with_imagedata = rtn_compatible_types.at(SNDE_RTN_SNDE_IMAGEDATA);
      if (to_project_rec_ref->typenum==SNDE_RTN_SNDE_IMAGEDATA || compatible_with_imagedata.find(to_project_rec_ref->typenum) != compatible_with_imagedata.end()) {
	return std::make_shared<project_point_onto_parameterization<snde_imagedata>>(rss,inst);
	
      }


      const std::set<unsigned> &compatible_with_complex_imagedata = rtn_compatible_types.at(SNDE_RTN_SNDE_COMPLEXIMAGEDATA);
      if (to_project_rec_ref->typenum==SNDE_RTN_SNDE_COMPLEXIMAGEDATA || compatible_with_complex_imagedata.find(to_project_rec_ref->typenum) != compatible_with_complex_imagedata.end()) {
	return std::make_shared<project_point_onto_parameterization<snde_compleximagedata>>(rss,inst);
	
      }

      throw snde_error("Projection only supports real or complex imagedata: Can not project onto array of type %s",rtn_typenamemap.at(to_project_rec_ref->typenum).c_str());
      
    });

    newfunc->self_dependent=true;
    newfunc->mandatory_mutable=true;
    newfunc->new_revision_optional=true;
    return newfunc;
  }
  
  // NOTE: Change to SNDE_OCL_API if/when we add GPU acceleration support, and
  // (in CMakeLists.txt) make it move into the _ocl.so library)
  SNDE_API std::shared_ptr<math_function> project_point_onto_parameterization_function = define_spatialnde2_project_point_onto_parameterization_function();
  
  static int registered_project_point_onto_parameterization_function = register_math_function(project_point_onto_parameterization_function);
  
  



};



