#include <Eigen/Dense>


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

#include "snde/inplanemat_calculation.hpp"

namespace snde {


  class inplanemat_calculation: public recmath_cppfuncexec<std::shared_ptr<meshed_part_recording>,std::shared_ptr<meshed_trinormals_recording>> {
  public:
    inplanemat_calculation(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) :
      recmath_cppfuncexec(rss,inst)
    {
      
    }
    
    // use default for decide_new_revision
    
    std::pair<std::vector<std::shared_ptr<compute_resource_option>>,std::shared_ptr<define_recs_function_override_type>> compute_options(std::shared_ptr<meshed_part_recording> part,std::shared_ptr<meshed_trinormals_recording> part_trinormals)
    {
      snde_ndarray_info *rec_tri_info = part->ndinfo(part->name_mapping.at("triangles"));
      if (rec_tri_info->ndim != 1) {
	throw snde_error("inplanemat_calculation: triangle dimensionality must be 1");
      }
      snde_index numtris = rec_tri_info->dimlen[0];

      snde_ndarray_info *rec_edge_info = part->ndinfo(part->name_mapping.at("edges"));
      if (rec_edge_info->ndim != 1) {
	throw snde_error("inplanemat_calculation: edge dimensionality must be 1");
      }
      snde_index numedges = rec_edge_info->dimlen[0];
      
      
      snde_ndarray_info *rec_vert_info = part->ndinfo(part->name_mapping.at("vertices"));
      if (rec_vert_info->ndim != 1) {
	throw snde_error("inplanemat_calculation: vertices dimensionality must be 1");
      }
      snde_index numverts = rec_vert_info->dimlen[0];

      std::vector<std::shared_ptr<compute_resource_option>> option_list =
	{
	  std::make_shared<compute_resource_option_cpu>(std::set<std::string>(), // no tags
							0, //metadata_bytes 
							numtris*sizeof(snde_triangle) + numedges*sizeof(snde_edge) + numverts*sizeof(snde_coord3) + numtris*sizeof(snde_trivertnormals) + numtris*sizeof(snde_cmat23), // data_bytes for transfer
							numtris*(200), // flops
							1, // max effective cpu cores
							1), // useful_cpu_cores (min # of cores to supply
	  
	};
      return std::make_pair(option_list,nullptr);
    }
  
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<meshed_part_recording> part,std::shared_ptr<meshed_trinormals_recording> part_trinormals) 
  {
    // define_recs code
    //printf("define_recs()\n");
    std::shared_ptr<meshed_inplanemat_recording> result_rec;
    result_rec = create_recording_math<meshed_inplanemat_recording>(get_result_channel_path(0),rss);
    
    return std::make_shared<metadata_function_override_type>([ this,result_rec,part, part_trinormals ]() {
      // metadata code
      std::unordered_map<std::string,metadatum> metadata;
      
      result_rec->metadata=std::make_shared<immutable_metadata>(metadata);
      result_rec->mark_metadata_done();
      
      return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,part, part_trinormals ]() {
	// lock_alloc code
	
	std::shared_ptr<graphics_storage_manager> graphman = std::dynamic_pointer_cast<graphics_storage_manager>(result_rec->assign_storage_manager());

	if (!graphman) {
	  throw snde_error("inplanemat_calculation: Output arrays must be managed by a graphics storage manager");
	}
	
	std::shared_ptr<graphics_storage> leader_storage = std::dynamic_pointer_cast<graphics_storage>(part->storage.at(part->name_mapping.at("triangles")));
	
	snde_index addr = leader_storage->base_index;
	snde_index nmemb = leader_storage->nelem;

	
	std::shared_ptr<graphics_storage> inplanemats_storage = graphman->storage_from_allocation(result_rec->info->name,leader_storage,"inplanemats",result_rec->info->revision,rss->unique_index,addr,sizeof(*graphman->geom.inplanemats),rtn_typemap.at(typeid(*graphman->geom.inplanemats)),nmemb);
	result_rec->assign_storage(inplanemats_storage,"inplanemats",{nmemb});
	
	
	rwlock_token_set locktokens = lockmgr->lock_recording_arrays({
	    //{ part, { "parts", false }}, // first element is recording_ref, 2nd parameter is false for read, true for write
	    { part, { "triangles", false }},
	    { part, {"edges", false }},
	    { part, {"vertices", false}},
	    { part_trinormals, {"trinormals", false}},
	    { result_rec,{"inplanemats", true }}
	  },
	  false
	  );
	
	return std::make_shared<exec_function_override_type>([ this,locktokens, result_rec, part, part_trinormals ]() {
	  // exec code
	  snde_ndarray_info *rec_tri_info = part->ndinfo(part->name_mapping.at("triangles"));
	  snde_index numtris = rec_tri_info->dimlen[0];
	  
	  //fprintf(stderr,"Not executing in OpenCL\n");
	  // Should OpenMP this (!)
	  //const struct snde_part *parts =(snde_part *)recording->void_shifted_arrayptr("parts");
	  const snde_triangle *triangles=(snde_triangle *)part->void_shifted_arrayptr("triangles");
	  const snde_edge *edges=(snde_edge *)part->void_shifted_arrayptr("edges");
	  const snde_coord3 *vertices=(snde_coord3 *)part->void_shifted_arrayptr("vertices");
	  const snde_coord3 *trinormals=(snde_coord3 *)part_trinormals->void_shifted_arrayptr("trinormals");
	  //snde_trivertnormals *vertnormals=(snde_trivertnormals *)result_rec->void_shifted_arrayptr("vertnormals");
	  snde_cmat23 *inplanemats=(snde_cmat23 *)result_rec->void_shifted_arrayptr("inplanemats");
	  
	  
	  for (snde_index cnt=0;cnt < numtris;cnt++){
	      
	    snde_coord3 tri_vertices[3];
	    get_we_triverts_3d(triangles,cnt,
			       edges,
			       vertices,
			       tri_vertices);
	    // The Eigen::Map points to the underlying data in tri_vertices
	    Eigen::Map<Eigen::Matrix<snde_coord,3,3,Eigen::ColMajor>> coords(tri_vertices[0].coord); // vertex coords, indexes: axis (x,y, or z) index by vertex index
	    
	    // now we got the vertex locations in coords
	    // subtract out the centroid
	    for (unsigned axiscnt=0;axiscnt < 3;axiscnt++) {
	      double mean=(coords(axiscnt,0)+coords(axiscnt,1)+coords(axiscnt,2))/3.0;
	      coords(axiscnt,0) -= mean;
	      coords(axiscnt,1) -= mean;
	      coords(axiscnt,2) -= mean;
	    }
	    
	    // calculate SVD
	    Eigen::JacobiSVD<Eigen::Matrix<double,3,3>> svd(coords.cast<double>(),Eigen::ComputeFullV | Eigen::ComputeFullU);
	    Eigen::Matrix<double,3,3> U=svd.matrixU();
	    Eigen::Vector3d s=svd.singularValues();
	    Eigen::Matrix<double,3,3> V=svd.matrixV();
	    // extract columns for 2d coordinate basis vectors
	    // want columns x and y that correspond to the largest two
	    // singular values and z that corresponds to the smallest singular value
	    
	    // We also want the x column cross the y column to give
	    // the outward normal
	    snde_index xcolindex=0;
	    snde_index ycolindex=1;
	    snde_index zcolindex=2;
	    
	    // First, select colums to ensure zcolindex
	    // corresponds to minimum singular value (normal direction)
	    
	    // per Eigen manual singular values are always positive and sorted in decreasing order
	    // therefore element 2 is smallest singular value as
	    //// initialized above and these conditionals are unnecessary
	    //if (fabs(s(0)) < fabs(s(1)) and fabs(s(0)) < fabs(s(2))) {
	    //  // element 0 is smallest s.v.
	    //  xcolindex=2;
	    //  zcolindex=0;
	    //}
	    //
	    //if (fabs(s(1)) < fabs(s(2)) and fabs(s(1)) < fabs(s(0))) {
	    //  // element 1 is smallest s.v.
	    //  ycolindex=2;
	    //  zcolindex=1;
	    //}
	    
	    // Second, check to see if xcol cross ycol is in the
	    // normal direction
	    Eigen::Vector3d normal;
	    normal(0)=trinormals[cnt].coord[0];
	    normal(1)=trinormals[cnt].coord[1];
	    normal(2)=trinormals[cnt].coord[2];
	    
	    if (U.col(xcolindex).cross(U.col(ycolindex)).dot(normal) < 0.0) {
	      // x cross y is in wrong direction
	      snde_index temp=xcolindex;
	      xcolindex=ycolindex;
	      ycolindex=temp;
	    }
	    // To2D=U[:,np.array((xcolindex,ycolindex))].T # 2x3... Rows of To2D are x and y basis vectors, respectively
	    Eigen::Matrix<double,2,3> inplanemats_eigen;
	    inplanemats_eigen.row(0)=U.col(xcolindex);
	    inplanemats_eigen.row(1)=U.col(ycolindex);
	    inplanemats[cnt].row[0].coord[0]=inplanemats_eigen(0,0);  
	    inplanemats[cnt].row[0].coord[1]=inplanemats_eigen(0,1);  
	    inplanemats[cnt].row[0].coord[2]=inplanemats_eigen(0,2);  
	    inplanemats[cnt].row[1].coord[0]=inplanemats_eigen(1,0);  
	    inplanemats[cnt].row[1].coord[1]=inplanemats_eigen(1,1);  
	    inplanemats[cnt].row[1].coord[2]=inplanemats_eigen(1,2);  
						      
	  }
	  
	  
	  snde_warning("Inplanemat calculation complete; numtris=%llu",(unsigned long long)numtris);
	  
	  
	  unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 
	  result_rec->mark_data_ready();
	  
	}); 
      });
    });
  };
    
  };
  

  std::shared_ptr<math_function> define_spatialnde2_inplanemat_calculation_function()
  {
    return std::make_shared<cpp_math_function>("snde.inplanemat_calculation",1,[] (std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst) {
      return std::make_shared<inplanemat_calculation>(rss,inst);
    }); 
  }

  // NOTE: Change to SNDE_OCL_API if/when we add GPU acceleration support, and
  // (in CMakeLists.txt) make it move into the _ocl.so library)
  SNDE_API std::shared_ptr<math_function> inplanemat_calculation_function = define_spatialnde2_inplanemat_calculation_function();
  
  static int registered_inplanemat_calculation_function = register_math_function(inplanemat_calculation_function);
  
  
  void instantiate_inplanemat(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags)
  {
    std::string context = recdb_path_context(loaded_geom->info->name);

    // require trinormals
    geomproc_specify_dependency(remaining_processing_tags,all_processing_tags,"trinormals");
    
    std::shared_ptr<instantiated_math_function> instantiated = inplanemat_calculation_function->instantiate( {
	std::make_shared<math_parameter_recording>("meshed"),
	std::make_shared<math_parameter_recording>("trinormals")
      },
      {
	std::make_shared<std::string>("inplanemat")
      },
      context,
      false, // is_mutable
      false, // ondemand
      false, // mdonly
      std::make_shared<math_definition>("instantiate_inplanemat()"),
      {},
      nullptr);


    trans->recdb->add_math_function(trans,instantiated,true); // trinormals are generally hidden by default
    loaded_geom->processed_relpaths.emplace("inplanemat","inplanemat");
    
  }

  static int registered_inplanemat_processor = register_geomproc_math_function("inplanemat",instantiate_inplanemat);




};


