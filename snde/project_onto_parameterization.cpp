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
  class project_point_onto_parameterization: public recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>,std::shared_ptr<meshed_parameterization_recording>,std::shared_ptr<meshed_trinormals_recording>,std::shared_ptr<boxes3d_recording>,std::shared_ptr<meshed_projinfo_recording>,std::shared_ptr<meshed_inplanemat_recording>,std::shared_ptr<ndtyped_recording_ref<snde_orientation3>>,std::shared_ptr<ndtyped_recording_ref<snde_orientation3>>,std::shared_ptr<ndtyped_recording_ref<T>>,double,double,double,snde_index,snde_index,snde_bool> {
  public:
    project_point_onto_parameterization(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>,std::shared_ptr<meshed_parameterization_recording>,std::shared_ptr<meshed_trinormals_recording>,std::shared_ptr<boxes3d_recording>,std::shared_ptr<meshed_projinfo_recording>,std::shared_ptr<meshed_inplanemat_recording>,std::shared_ptr<ndtyped_recording_ref<snde_orientation3>>,std::shared_ptr<ndtyped_recording_ref<snde_orientation3>>,std::shared_ptr<ndtyped_recording_ref<T>>,double,double,double,snde_index,snde_index,snde_bool>(rss,inst)
    {
      
    }


        // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
    typedef typename snde::recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>,std::shared_ptr<meshed_parameterization_recording>,std::shared_ptr<meshed_trinormals_recording>,std::shared_ptr<boxes3d_recording>,std::shared_ptr<meshed_projinfo_recording>,std::shared_ptr<meshed_inplanemat_recording>,std::shared_ptr<ndtyped_recording_ref<snde_orientation3>>,std::shared_ptr<ndtyped_recording_ref<snde_orientation3>>,std::shared_ptr<ndtyped_recording_ref<T>>,double,double,double,snde_index,snde_index,snde_bool>::define_recs_function_override_type define_recs_function_override_type;
    typedef typename snde::recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>,std::shared_ptr<meshed_parameterization_recording>,std::shared_ptr<meshed_trinormals_recording>,std::shared_ptr<boxes3d_recording>,std::shared_ptr<meshed_projinfo_recording>,std::shared_ptr<meshed_inplanemat_recording>,std::shared_ptr<ndtyped_recording_ref<snde_orientation3>>,std::shared_ptr<ndtyped_recording_ref<snde_orientation3>>,std::shared_ptr<ndtyped_recording_ref<T>>,double,double,double,snde_index,snde_index,snde_bool>::metadata_function_override_type metadata_function_override_type;
    typedef typename snde::recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>,std::shared_ptr<meshed_parameterization_recording>,std::shared_ptr<meshed_trinormals_recording>,std::shared_ptr<boxes3d_recording>,std::shared_ptr<meshed_projinfo_recording>,std::shared_ptr<meshed_inplanemat_recording>,std::shared_ptr<ndtyped_recording_ref<snde_orientation3>>,std::shared_ptr<ndtyped_recording_ref<snde_orientation3>>,std::shared_ptr<ndtyped_recording_ref<T>>,double,double,double,snde_index,snde_index,snde_bool>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename snde::recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>,std::shared_ptr<meshed_parameterization_recording>,std::shared_ptr<meshed_trinormals_recording>,std::shared_ptr<boxes3d_recording>,std::shared_ptr<meshed_projinfo_recording>,std::shared_ptr<meshed_inplanemat_recording>,std::shared_ptr<ndtyped_recording_ref<snde_orientation3>>,std::shared_ptr<ndtyped_recording_ref<snde_orientation3>>,std::shared_ptr<ndtyped_recording_ref<T>>,double,double,double,snde_index,snde_index,snde_bool>::exec_function_override_type exec_function_override_type;

    
    // use default for decide_new_revision
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>>
      compute_options(std::shared_ptr<meshed_part_recording> part,
		      std::shared_ptr<meshed_parameterization_recording> param,
		      std::shared_ptr<meshed_trinormals_recording> trinormals,
		      std::shared_ptr<boxes3d_recording> boxes3d,
		      std::shared_ptr<meshed_projinfo_recording> projinfo,
		      std::shared_ptr<meshed_inplanemat_recording> inplanemat,
		      std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> part_orientation,
		      std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> source_orientation,
		      std::shared_ptr<ndtyped_recording_ref<T>> to_project,
		      double min_dist,
		      double max_dist,
		      double radius,
		      snde_index horizontal_pixels,
		      snde_index vertical_pixels, 
		      snde_bool use_surface_normal)
    {
      
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
	  std::make_shared<compute_resource_option_cpu>(0, //metadata_bytes 
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
		std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> part_orientation,
		std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> source_orientation,
		std::shared_ptr<ndtyped_recording_ref<T>> to_project,
		double min_dist,
		double max_dist,
		double radius,
		snde_index horizontal_pixels,
		snde_index vertical_pixels, 
		snde_bool use_surface_normal)
    {
      // define_recs code

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
	// metadata code
	std::shared_ptr<constructible_metadata> metadata=std::make_shared<constructible_metadata>(*to_project->rec->metadata);


	snde_index min_u = 0.0;
	snde_index max_u = 0.0;
	snde_index min_v = 1.0;
	snde_index max_v = 1.0;
	
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
	  
	  std::shared_ptr<ndtyped_recording_ref<snde_parameterization_patch>> patch_ref = param->reference_typed_ndarray<snde_parameterization_patch>("uv_patchess");
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
	std::string ampl_units = to_project->rec->metadata->GetMetaDatumStr("nde_array-ampl_units","Volts");
	std::string ampl_coord = to_project->rec->metadata->GetMetaDatumStr("nde_array-ampl_coord","Volts");

	double step0 = (max_u-min_u)/horizontal_pixels;
	double step1 = (max_v-min_v)/vertical_pixels;
	
	double inival0 = min_u + step0/2.0;
	double inival1 = min_v + step1/2.0;

	metadata->AddMetaDatum(metadatum("nde_array-axis0_step",step0));
	metadata->AddMetaDatum(metadatum("nde_array-axis0_inival",inival0));
	metadata->AddMetaDatum(metadatum("nde_array-axis0_coord",coord0));
	metadata->AddMetaDatum(metadatum("nde_array-axis0_units",units0));

	metadata->AddMetaDatum(metadatum("nde_array-axis1_step",step1));
	metadata->AddMetaDatum(metadatum("nde_array-axis1_inival",inival1));
	metadata->AddMetaDatum(metadatum("nde_array-axis1_coord",coord1));
	metadata->AddMetaDatum(metadatum("nde_array-axis1_units",units1));

	metadata->AddMetaDatum(metadatum("nde_array-ampl_coord",ampl_coord));
	metadata->AddMetaDatum(metadatum("nde_array-ampl_units",ampl_units));

	
	
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
	  
	  std::shared_ptr<graphics_storage_manager> graphman = std::dynamic_pointer_cast<graphics_storage_manager>(result_rec->assign_storage_manager());
	  
	  if (!graphman) {
	    throw snde_error("inplanemat_calculation: Output arrays must be managed by a graphics storage manager");
	  }
	  

	  bool build_on_previous = false; 
	  std::shared_ptr<multi_ndarray_recording> previous_recording_ndarray;
	  std::shared_ptr<recording_base> previous_recording = this->self_dependent_recordings.at(0);
	  std::shared_ptr<multi_ndarray_recording> previous_ndarray = std::dynamic_pointer_cast<multi_ndarray_recording>(previous_recording);
	  
	

	  unsigned typenum = rtn_typemap.at(typeid(T));
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
		  
		  double previous_step0=previous_ndarray->metadata->GetMetaDatumDbl("nde_array-axis0_step",1.0);
		  double previous_step1=previous_ndarray->metadata->GetMetaDatumDbl("nde_array-axis1_step",1.0);
		  
		  double previous_inival0=previous_ndarray->metadata->GetMetaDatumDbl("nde_array-axis0_inival",0.0);
		  double previous_inival1=previous_ndarray->metadata->GetMetaDatumDbl("nde_array-axis1_inival",0.0);
		  
		  std::string previous_coord0 = previous_ndarray->metadata->GetMetaDatumStr("nde_array-axis0_coord","X Position");
		  std::string previous_coord1 = previous_ndarray->metadata->GetMetaDatumStr("nde_array-axis1_coord","Y Position");
		  
		  std::string previous_units0 = previous_ndarray->metadata->GetMetaDatumStr("nde_array-axis0_units","meters");
		  std::string previous_units1 = previous_ndarray->metadata->GetMetaDatumStr("nde_array-axis1_units","meters");
		  
		  std::string previous_ampl_units = previous_ndarray->metadata->GetMetaDatumStr("nde_array-ampl_units","Volts");
		  std::string previous_ampl_coord = previous_ndarray->metadata->GetMetaDatumStr("nde_array-ampl_coord","Voltage");
		  
		  if (previous_step0 == step0 && previous_step1==step1 && previous_inival0==inival0 && previous_inival1==inival1 &&
		      previous_coord0 == coord0 && previous_coord1 == coord1 && previous_units0==units0 && previous_units1==units1 &&
		      previous_ampl_units == ampl_units && previous_ampl_coord == ampl_coord) {
		    
		    build_on_previous = true;
		  }
		}
	      }
	    }
	  }
	

	  
      
	  if (!build_on_previous) {
	    result_rec->allocate_storage(0,dimlen,true); // storage for image
	    result_rec->allocate_storage(1,dimlen,true); // storage for validity mask 
	  } else {
	    // accumulate on top of previous recording -- it is mutable storage!
	    result_rec->storage.at(0) = previous_ndarray->storage.at(0);
	    result_rec->storage.at(0) = previous_ndarray->storage.at(1);
	  }
	  
	  
	  
	  
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
		{ part_orientation, false },
		{ source_orientation, false },
	    
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
								 ampl_coord,ampl_units,
								 build_on_previous, dimlen, locktokens ]() {
	    // exec code
	    
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
	    instances.push_back(snde_partinstance{
		.orientation = part_orientation->element(0),
		.partnum=partnum,  
		.firstuvpatch=0, // only support single patch for now
		.uvnum=paramnum,
	      });
	    
	    
	    snde_orientation3 sensorcoords_to_wrlcoords=source_orientation->element(0);
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
	    snde_index frin_stacksize=boxes3d->metadata->GetMetaDatumUnsigned("snde_boxes3d_max_depth",10);
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
	    
	    snde_imagedata real_pixelval=to_project->element_complexfloat64(0).real;
	    snde_imagedata pixelweighting=1.0;

	    //snde_coord3 uvcoords = { imagedata_intersectprops.uvcoords.coord[0],imagedata_intersectprops.uvcoords.coord[1],1.0 };
	    snde_coord min_radius_uv_pixels = 2.0; // external parameter? 
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
	    
	    
	    snde_warning("Project_onto_parameterization calculation complete.");
	    
	    
	    unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_as_ready() 
	    result_rec->mark_as_ready();
	    
	  }); 
	});
      });
    };
    
  };
  
  
  std::shared_ptr<math_function> define_spatialnde2_project_point_onto_parameterization_function()
  {
    std::shared_ptr<math_function> newfunc = std::make_shared<cpp_math_function>([] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) -> std::shared_ptr<executing_math_function> {
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

      throw snde_error("Projection only supports real or complex imagedata: Can not project onto array of type %s",rtn_typenamemap.at(to_project_rec_ref->typenum));
      
    });

    newfunc->self_dependent=true;
    newfunc->mandatory_mutable=true;
    return newfunc;
  }
  
  // NOTE: Change to SNDE_OCL_API if/when we add GPU acceleration support, and
  // (in CMakeLists.txt) make it move into the _ocl.so library)
  SNDE_API std::shared_ptr<math_function> project_point_onto_parameterization_function = define_spatialnde2_project_point_onto_parameterization_function();
  
  static int registered_project_point_onto_parameterization_function = register_math_function("spatialnde2.project_point_onto_parameterization",project_point_onto_parameterization_function);
  
  



};



