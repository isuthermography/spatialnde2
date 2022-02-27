
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
    
    name_mapping.emplace(std::make_pair("vertices",0));
    name_reverse_mapping.emplace(std::make_pair(0,"vertices"));
    define_array(0,rtn_typemap.at(typeid(*dummy.vertices)));

  }
  */

  
  meshed_part_recording::meshed_part_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,8)
  {
    snde_geometrydata dummy={0};
    
    name_mapping.emplace(std::make_pair("parts",0));
    name_reverse_mapping.emplace(std::make_pair(0,"parts"));
    define_array(0,rtn_typemap.at(typeid(*dummy.parts)));

    name_mapping.emplace(std::make_pair("topos",1));
    name_reverse_mapping.emplace(std::make_pair(1,"topos"));
    define_array(1,rtn_typemap.at(typeid(*dummy.topos)));

    name_mapping.emplace(std::make_pair("topo_indices",2));
    name_reverse_mapping.emplace(std::make_pair(2,"topo_indices"));
    define_array(2,rtn_typemap.at(typeid(*dummy.topo_indices)));

    name_mapping.emplace(std::make_pair("triangles",3));
    name_reverse_mapping.emplace(std::make_pair(3,"triangles"));
    define_array(3,rtn_typemap.at(typeid(*dummy.triangles)));

    name_mapping.emplace(std::make_pair("edges",4));
    name_reverse_mapping.emplace(std::make_pair(4,"edges"));
    define_array(4,rtn_typemap.at(typeid(*dummy.edges)));

    name_mapping.emplace(std::make_pair("vertices",5));
    name_reverse_mapping.emplace(std::make_pair(5,"vertices"));
    define_array(5,rtn_typemap.at(typeid(*dummy.vertices)));

    name_mapping.emplace(std::make_pair("vertex_edgelist_indices",6));
    name_reverse_mapping.emplace(std::make_pair(6,"vertex_edgelist_indices"));
    define_array(6,rtn_typemap.at(typeid(*dummy.vertex_edgelist_indices)));
    
    name_mapping.emplace(std::make_pair("vertex_edgelist",7));
    name_reverse_mapping.emplace(std::make_pair(7,"vertex_edgelist"));
    define_array(7,rtn_typemap.at(typeid(*dummy.vertex_edgelist)));

    // NOTE: Final parameter to multi_ndarray_recording() above is number of mapping entries ***!!! 
  }

  meshed_vertexarray_recording::meshed_vertexarray_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,1)
  {
    snde_geometrydata dummy={0};
    
    name_mapping.emplace(std::make_pair("vertex_arrays",0));
    name_reverse_mapping.emplace(std::make_pair(0,"vertex_arrays"));

    define_array(0,rtn_typemap.at(typeid(*dummy.vertex_arrays)));
  }

  meshed_texvertex_recording::meshed_texvertex_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,1)
  {
    snde_geometrydata dummy={0};
    
    name_mapping.emplace(std::make_pair("texvertex_arrays",0));
    name_reverse_mapping.emplace(std::make_pair(0,"texvertex_arrays"));
    define_array(0,rtn_typemap.at(typeid(*dummy.texvertex_arrays)));

  }
    
  meshed_vertnormals_recording::meshed_vertnormals_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,1)
  {
    snde_geometrydata dummy={0};
    
    name_mapping.emplace(std::make_pair("vertnormals",0));
    name_reverse_mapping.emplace(std::make_pair(0,"vertnormals"));
    define_array(0,rtn_typemap.at(typeid(*dummy.vertnormals)));

  }

  meshed_trinormals_recording::meshed_trinormals_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
   multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,1)
  {
    snde_geometrydata dummy={0};

    name_mapping.emplace(std::make_pair("trinormals",0));
    name_reverse_mapping.emplace(std::make_pair(0,"trinormals"));
    define_array(0,rtn_typemap.at(typeid(*dummy.trinormals)));

  }

  meshed_parameterization_recording::meshed_parameterization_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
   multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,9)
  {
    snde_geometrydata dummy={0};

    name_mapping.emplace(std::make_pair("uvs",0));
    name_reverse_mapping.emplace(std::make_pair(0,"uvs"));
    define_array(0,rtn_typemap.at(typeid(*dummy.uvs)));

    name_mapping.emplace(std::make_pair("uv_patches",1));
    name_reverse_mapping.emplace(std::make_pair(1,"uv_patches"));
    define_array(1,rtn_typemap.at(typeid(*dummy.uv_patches)));

    name_mapping.emplace(std::make_pair("uv_topos",2));
    name_reverse_mapping.emplace(std::make_pair(2,"uv_topos"));
    define_array(2,rtn_typemap.at(typeid(*dummy.uv_topos)));

    name_mapping.emplace(std::make_pair("uv_topo_indices",3));
    name_reverse_mapping.emplace(std::make_pair(3,"uv_topo_indices"));
    define_array(3,rtn_typemap.at(typeid(*dummy.uv_topo_indices)));

    name_mapping.emplace(std::make_pair("uv_triangles",4));
    name_reverse_mapping.emplace(std::make_pair(4,"uv_triangles"));
    define_array(4,rtn_typemap.at(typeid(*dummy.uv_triangles)));

    name_mapping.emplace(std::make_pair("uv_edges",5));
    name_reverse_mapping.emplace(std::make_pair(5,"uv_edges"));
    define_array(5,rtn_typemap.at(typeid(*dummy.uv_edges)));

    name_mapping.emplace(std::make_pair("uv_vertices",6));
    name_reverse_mapping.emplace(std::make_pair(6,"uv_vertices"));
    define_array(6,rtn_typemap.at(typeid(*dummy.uv_vertices)));

    name_mapping.emplace(std::make_pair("uv_vertex_edgelist_indices",7));
    name_reverse_mapping.emplace(std::make_pair(7,"uv_vertex_edgelist_indices"));
    define_array(7,rtn_typemap.at(typeid(*dummy.uv_vertex_edgelist_indices)));

    name_mapping.emplace(std::make_pair("uv_vertex_edgelist",8));
    name_reverse_mapping.emplace(std::make_pair(8,"uv_vertex_edgelist"));
    define_array(8,rtn_typemap.at(typeid(*dummy.uv_vertex_edgelist)));

    // NOTE: Final parameter to multi_ndarray_recording() above is number of mapping entries ***!!! 
  }

  texture_recording::texture_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,1)
    
  {
    snde_geometrydata dummy={0};
    
    name_mapping.emplace(std::make_pair("texbuffer",0));
    name_reverse_mapping.emplace(std::make_pair(0,"texbuffer"));
    define_array(0,rtn_typemap.at(typeid(*dummy.texbuffer)));
    
    
  }

  
  image_reference::image_reference(std::string image_path, snde_index u_dimnum, snde_index v_dimnum, const std::vector<snde_index> &other_indices) :
    image_path(image_path),
    u_dimnum(u_dimnum),
    v_dimnum(v_dimnum),
    other_indices(other_indices)
  {

  }

  textured_part_recording::textured_part_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,std::string part_name, std::shared_ptr<std::string> parameterization_name, const std::map<snde_index,std::shared_ptr<image_reference>> &texture_refs) :
    recording_group(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,nullptr),
    part_name(part_name),
    parameterization_name(parameterization_name),
    texture_refs(texture_refs)
  {

  }

  assembly_recording::assembly_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,const std::vector<std::pair<std::string,snde_orientation3>> &pieces) :
    recording_group(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,nullptr),
    pieces(pieces)
  {
    
  }

  
  tracking_pose_recording::tracking_pose_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,std::string channel_to_reorient,std::string component_name):
    recording_group(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,nullptr),
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
    snde_orientation3 retval = { {{ 0.0, 0.0, 0.0, 0.0 } }, {{ 0.0, 0.0, 0.0, 0.0 } } }; // invalid orientation

    std::string chanpath = info->name;
    std::string pose_recording_fullpath = recdb_join_assembly_and_component_names(chanpath,pose_channel_name);
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


  
  
};
