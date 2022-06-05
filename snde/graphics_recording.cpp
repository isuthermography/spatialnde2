
#include "snde/graphics_recording.hpp"
#include "snde/geometrydata.h"
#include "snde/display_requirements.hpp"

namespace snde {

  /*
  // pointcloud_recording is only compatible with the
  // graphics_storage_manager that defines special storage for
  // certain arrays, including "vertices"
  pointcloud_recording::pointcloud_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,size_t num_ndarrays) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,1)
  {
    snde_geometrydata dummy={0};
    
    define_array(0,rtn_typemap.at(typeid(*dummy.vertices)),"vertices");

  }
  */

  
  meshed_part_recording::meshed_part_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,8)
  {
    snde_geometrydata dummy={0};
    
    define_array(0,rtn_typemap.at(typeid(*dummy.parts)),"parts");

    define_array(1,rtn_typemap.at(typeid(*dummy.topos)),"topos");

    define_array(2,rtn_typemap.at(typeid(*dummy.topo_indices)),"topo_indices");

    define_array(3,rtn_typemap.at(typeid(*dummy.triangles)),"triangles");

    define_array(4,rtn_typemap.at(typeid(*dummy.edges)),"edges");

    define_array(5,rtn_typemap.at(typeid(*dummy.vertices)),"vertices");

    define_array(6,rtn_typemap.at(typeid(*dummy.vertex_edgelist_indices)),"vertex_edgelist_indices");
    
    define_array(7,rtn_typemap.at(typeid(*dummy.vertex_edgelist)),"vertex_edgelist");

    // NOTE: Final parameter to multi_ndarray_recording() above is number of mapping entries ***!!! 
  }

  meshed_vertexarray_recording::meshed_vertexarray_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,1)
  {
    //snde_geometrydata dummy={0};
    

    define_array(0,rtn_typemap.at(typeid(snde_rendercoord /**dummy.vertex_arrays*/)),"vertex_arrays");
  }


  meshed_inplanemat_recording::meshed_inplanemat_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,1)
  {
    snde_geometrydata dummy={0};
    
    define_array(0,rtn_typemap.at(typeid(*dummy.inplanemats)),"inplanemats");
  }

  
  meshed_texvertex_recording::meshed_texvertex_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,1)
  {
    snde_geometrydata dummy={0};
    
    define_array(0,rtn_typemap.at(typeid(snde_rendercoord /* *dummy.texvertex_arrays */)),"texvertex_arrays");

  }
    
  meshed_vertnormals_recording::meshed_vertnormals_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,1)
  {
    snde_geometrydata dummy={0};
    
    define_array(0,rtn_typemap.at(typeid(*dummy.vertnormals)),"vertnormals");

  }

  meshed_vertnormalarrays_recording::meshed_vertnormalarrays_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,1)
  {
    snde_geometrydata dummy={0};
    
    define_array(0,rtn_typemap.at(typeid(snde_trivertnormals /* *dummy.vertnormal_arrays */)),"vertnormal_arrays");

  }

  
  meshed_trinormals_recording::meshed_trinormals_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
   multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,1)
  {
    snde_geometrydata dummy={0};

    define_array(0,rtn_typemap.at(typeid(*dummy.trinormals)),"trinormals");

  }

  meshed_parameterization_recording::meshed_parameterization_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
   multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,9)
  {
    snde_geometrydata dummy={0};

    define_array(0,rtn_typemap.at(typeid(*dummy.uvs)),"uvs");

    define_array(1,rtn_typemap.at(typeid(*dummy.uv_patches)),"uv_patches");

    define_array(2,rtn_typemap.at(typeid(*dummy.uv_topos)),"uv_topos");

    define_array(3,rtn_typemap.at(typeid(*dummy.uv_topo_indices)),"uv_topo_indices");

    define_array(4,rtn_typemap.at(typeid(*dummy.uv_triangles)),"uv_triangles");

    define_array(5,rtn_typemap.at(typeid(*dummy.uv_edges)),"uv_edges");

    define_array(6,rtn_typemap.at(typeid(*dummy.uv_vertices)),"uv_vertices");

    define_array(7,rtn_typemap.at(typeid(*dummy.uv_vertex_edgelist_indices)),"uv_vertex_edgelist_indices");

    define_array(8,rtn_typemap.at(typeid(*dummy.uv_vertex_edgelist)),"uv_vertex_edgelist");

    // NOTE: Final parameter to multi_ndarray_recording() above is number of mapping entries ***!!! 
  }


  meshed_projinfo_recording::meshed_projinfo_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
   multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,2)
  {
    snde_geometrydata dummy={0};

    define_array(0,rtn_typemap.at(typeid(*dummy.inplane2uvcoords)),"inplane2uvcoords");

    define_array(1,rtn_typemap.at(typeid(*dummy.uvcoords2inplane)),"uvcoords2inplane");
    
    
    // NOTE: Final parameter to multi_ndarray_recording() above is number of mapping entries ***!!! 
  }


  boxes3d_recording::boxes3d_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
   multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,3)
  {
    snde_geometrydata dummy={0};

    define_array(0,rtn_typemap.at(typeid(*dummy.boxes)),"boxes");

    define_array(1,rtn_typemap.at(typeid(*dummy.boxcoord)),"boxcoord");

    define_array(2,rtn_typemap.at(typeid(*dummy.boxpolys)),"boxpolys");

    
    // NOTE: Final parameter to multi_ndarray_recording() above is number of mapping entries ***!!! 
  }



  boxes2d_recording::boxes2d_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
   multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,0)
  {
    // ***!!!! NOTE: Must call set_num_patches after construction and before assigning storage ***!!!

    // NOTE: Final parameter to multi_ndarray_recording() above is number of mapping entries, which we initialize to 0
    // and is updated by set_num_patches()

  }

  void boxes2d_recording::set_num_patches(snde_index num_patches)
  {
    snde_geometrydata dummy={0};
    
    // call superclass
    set_num_ndarrays(3*num_patches);
    
    for (snde_index patchnum=0;patchnum < num_patches;patchnum++) {
      define_array(patchnum*3+0,rtn_typemap.at(typeid(*dummy.uv_boxes)),"uv_boxes"+std::to_string(patchnum));

      
      define_array(patchnum*3+1,rtn_typemap.at(typeid(*dummy.uv_boxcoord)),"uv_boxcoord"+std::to_string(patchnum));

      
      define_array(patchnum*3+2,rtn_typemap.at(typeid(*dummy.uv_boxpolys)),"uv_boxpolys"+std::to_string(patchnum));
      
    }
    
  }


  
  
  texture_recording::texture_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,1)
    
  {
    snde_geometrydata dummy={0};
    
    define_array(0,rtn_typemap.at(typeid(*dummy.texbuffer)),"texbuffer");
    
    
  }

  
  image_reference::image_reference(std::string image_path, snde_index u_dimnum, snde_index v_dimnum, const std::vector<snde_index> &other_indices) :
    image_path(image_path),
    u_dimnum(u_dimnum),
    v_dimnum(v_dimnum),
    other_indices(other_indices)
  {

  }

  textured_part_recording::textured_part_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,std::string part_name, std::shared_ptr<std::string> parameterization_name, const std::map<snde_index,std::shared_ptr<image_reference>> &texture_refs) :
    recording_base(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize),
    part_name(part_name),
    parameterization_name(parameterization_name),
    texture_refs(texture_refs)
  {

  }

  
  // This version primarily for Python wrapping
  textured_part_recording::textured_part_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,std::string part_name, std::shared_ptr<std::string> parameterization_name, std::vector<std::pair<snde_index,std::shared_ptr<image_reference>>> texture_refs_vec) :
    recording_base(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize),
    part_name(part_name),
    parameterization_name(parameterization_name)
  {
    for (auto && texref: texture_refs_vec) {
      texture_refs.emplace(texref.first,texref.second);
    }
    
  }


  assembly_recording::assembly_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,const std::vector<std::pair<std::string,snde_orientation3>> &pieces) :
    recording_base(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize),
    pieces(pieces)
  {
    
  }


  loaded_part_geometry_recording::loaded_part_geometry_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,const std::unordered_set<std::string> &processing_tags)
 :
    recording_group(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,nullptr),
    processing_tags(processing_tags)
  {
    
  }

  
  tracking_pose_recording::tracking_pose_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,std::string channel_to_reorient,std::string component_name):
    recording_base(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize),
    channel_to_reorient(channel_to_reorient),
    component_name(component_name)
  {
    
  }


  // Register the pre-existing tracking_pose_recording_display_handler in display_requirement.cpp/hpp as the display handler for pose_channel_tracking_pose_recording
  static int register_pctpr_display_handler = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING,typeid(pose_channel_tracking_pose_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
	return std::make_shared<tracking_pose_recording_display_handler>(display,displaychan,base_rss);
      }));

  pose_channel_tracking_pose_recording::pose_channel_tracking_pose_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,std::string channel_to_reorient,std::string component_name,std::string pose_channel_name):
    tracking_pose_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,channel_to_reorient,component_name),
    pose_channel_name(pose_channel_name)
  {
    
  }

  snde_orientation3 pose_channel_tracking_pose_recording::get_channel_to_reorient_pose(std::shared_ptr<recording_set_state> rss) const
  {
    snde_orientation3 retval;

    snde_invalid_orientation3(&retval); // invalid orientation

    std::string chanpath = info->name;
    std::string pose_recording_fullpath = recdb_path_join(chanpath,pose_channel_name);
    std::shared_ptr<recording_base> pose_recording = rss->get_recording(pose_recording_fullpath);

    if (!pose_recording)  {
      return retval;
    }

    std::shared_ptr<multi_ndarray_recording> pose_rec_ndarray = pose_recording->cast_to_multi_ndarray();
    if (!pose_rec_ndarray) {
      return retval;
    }

    std::shared_ptr<ndtyped_recording_ref<snde_orientation3>> pose_ref = pose_rec_ndarray->reference_typed_ndarray<snde_orientation3>();
    if (!pose_ref) {
      return retval;
    }

    if (pose_ref->storage->requires_locking_read) {
      throw snde_error("pose_channel_tracking_pose_recording::get_channel_to_reorient_pose(), channel %s: Pose channel %s requires locking for read, which may be unsafe in this context. Switch it to a storage manager that does not require locking.",chanpath.c_str(),pose_recording_fullpath.c_str());
    }
    return pose_ref->element(0);
    
  }


  pose_channel_recording::pose_channel_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,size_t num_ndarrays,std::string channel_to_reorient) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,num_ndarrays),
    channel_to_reorient(channel_to_reorient)
    
  {

    if (num_ndarrays != 1) {
      throw snde_error("pose_channel_recording::pose_channel_recording(%s): Error only single ndarray supported",chanpath.c_str());
    }
    
    define_array(0,rtn_typemap.at(typeid(snde_orientation3)),"pose");
  }

  
  // only call during initialization
  void pose_channel_recording::set_untransformed_render_channel(std::string component_name_str)
  {
    component_name = std::make_shared<std::string>(component_name_str);
  }

  /* static */ std::shared_ptr<pose_channel_recording> pose_channel_recording::from_ndarray_recording(std::shared_ptr<multi_ndarray_recording> rec)
  {
    return std::dynamic_pointer_cast<pose_channel_recording>(rec);
  }

  std::shared_ptr<ndarray_recording_ref> create_pose_channel_recording_ref(std::shared_ptr<recdatabase> recdb,std::shared_ptr<channel> chan,void *owner_id,std::string channel_to_reorient_name)
  {
    return create_subclass_recording_ref<pose_channel_recording>(recdb,chan,owner_id,SNDE_RTN_SNDE_ORIENTATION3,channel_to_reorient_name);
  }

  
};
