
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
    rec_classes.push_back(recording_class_info("snde::pointcloud_recording",typeid(pointcloud_recording),ptr_to_new_shared_impl<pointcloud_recording));

  }
  */

  
  meshed_part_recording::meshed_part_recording(struct recording_params params,size_t info_structsize) :
    multi_ndarray_recording(params,info_structsize,8)
  {
    snde_geometrydata dummy={0};

    rec_classes.push_back(recording_class_info("snde::meshed_part_recording",typeid(meshed_part_recording),ptr_to_new_shared_impl<meshed_part_recording>));

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

  std::shared_ptr<std::set<std::string>> meshed_part_recording::graphics_componentpart_channels(std::shared_ptr<recording_set_state> rss,std::vector<std::string> processing_tags)
  {
    std::shared_ptr<std::set<std::string>> componentpart_chans = std::make_shared<std::set<std::string>>();

    std::unordered_set<std::string> processing_tag_set;
    for (auto && tag: processing_tags) {
      processing_tag_set.emplace(tag);
    }

    if (processing_tag_set.find("trianglearea") != processing_tag_set.end()){
      componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),processed_relpaths.at("trianglearea"))); 
    }
    if (processing_tag_set.find("vertexarea") != processing_tag_set.end()){
      componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),processed_relpaths.at("vertexarea"))); 
    }

    if (processing_tag_set.find("boxes3d") != processing_tag_set.end()){
      componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),processed_relpaths.at("boxes3d"))); 
    }
    if (processing_tag_set.find("inplanemat") != processing_tag_set.end()){
      componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),processed_relpaths.at("inplanemat"))); 
    }
    if (processing_tag_set.find("vertex_kdtree") != processing_tag_set.end()){
      componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),processed_relpaths.at("vertex_kdtree"))); 
    }
    if (processing_tag_set.find("trinormals") != processing_tag_set.end()){
      componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),processed_relpaths.at("trinormals"))); 
    }
    if (processing_tag_set.find("vertnormals") != processing_tag_set.end()){
      componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),processed_relpaths.at("vertnormals"))); 
    }
    return componentpart_chans;
  }
  
  meshed_vertexarray_recording::meshed_vertexarray_recording(struct recording_params params,size_t info_structsize) :
    multi_ndarray_recording(params,info_structsize,1)
  {
    //snde_geometrydata dummy={0};
    
    rec_classes.push_back(recording_class_info("snde::meshed_vertexarray_recording",typeid(meshed_vertexarray_recording),ptr_to_new_shared_impl<meshed_vertexarray_recording>));

    define_array(0,rtn_typemap.at(typeid(snde_rendercoord /**dummy.vertex_arrays*/)),"vertex_arrays");
  }


  meshed_inplanemat_recording::meshed_inplanemat_recording(struct recording_params params,size_t info_structsize) :
    multi_ndarray_recording(params,info_structsize,1)
  {
    snde_geometrydata dummy={0};

    rec_classes.push_back(recording_class_info("snde::meshed_inplanemat_recording",typeid(meshed_inplanemat_recording),ptr_to_new_shared_impl<meshed_inplanemat_recording>));

    define_array(0,rtn_typemap.at(typeid(*dummy.inplanemats)),"inplanemats");
  }

  
  meshed_texvertex_recording::meshed_texvertex_recording(struct recording_params params,size_t info_structsize) :
    multi_ndarray_recording(params,info_structsize,1)
  {
    snde_geometrydata dummy={0};

    rec_classes.push_back(recording_class_info("snde::meshed_texvertex_recording",typeid(meshed_texvertex_recording),ptr_to_new_shared_impl<meshed_texvertex_recording>));

    define_array(0,rtn_typemap.at(typeid(snde_rendercoord /* *dummy.texvertex_arrays */)),"texvertex_arrays");

  }
    
  meshed_vertnormals_recording::meshed_vertnormals_recording(struct recording_params params,size_t info_structsize) :
    multi_ndarray_recording(params,info_structsize,1)
  {
    snde_geometrydata dummy={0};
    
    rec_classes.push_back(recording_class_info("snde::meshed_vertnormals_recording",typeid(meshed_vertnormals_recording),ptr_to_new_shared_impl<meshed_vertnormals_recording>));

    define_array(0,rtn_typemap.at(typeid(*dummy.vertnormals)),"vertnormals");

  }

  meshed_vertnormalarrays_recording::meshed_vertnormalarrays_recording(struct recording_params params,size_t info_structsize) :
    multi_ndarray_recording(params,info_structsize,1)
  {
    snde_geometrydata dummy={0};

    rec_classes.push_back(recording_class_info("snde::meshed_vertnormalarrays_recording",typeid(meshed_vertnormalarrays_recording),ptr_to_new_shared_impl<meshed_vertnormalarrays_recording>));

    define_array(0,rtn_typemap.at(typeid(snde_trivertnormals /* *dummy.vertnormal_arrays */)),"vertnormal_arrays");

  }

  
  meshed_trinormals_recording::meshed_trinormals_recording(struct recording_params params,size_t info_structsize) :
    multi_ndarray_recording(params,info_structsize,1)
  {
    snde_geometrydata dummy={0};
    
    rec_classes.push_back(recording_class_info("snde::meshed_trinormals_recording",typeid(meshed_trinormals_recording),ptr_to_new_shared_impl<meshed_trinormals_recording>));

    define_array(0,rtn_typemap.at(typeid(*dummy.trinormals)),"trinormals");

  }

  meshed_parameterization_recording::meshed_parameterization_recording(struct recording_params params,size_t info_structsize) :
    multi_ndarray_recording(params,info_structsize,9)
  {
    snde_geometrydata dummy={0};


    rec_classes.push_back(recording_class_info("snde::meshed_parameterization_recording",typeid(meshed_parameterization_recording),ptr_to_new_shared_impl<meshed_parameterization_recording>));

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

  std::shared_ptr<std::set<std::string>> meshed_parameterization_recording::graphics_componentpart_channels(std::shared_ptr<recording_set_state> rss,std::vector<std::string> processing_tags)
  {
    std::shared_ptr<std::set<std::string>> componentpart_chans = std::make_shared<std::set<std::string>>();

    std::unordered_set<std::string> processing_tag_set;
    for (auto && tag: processing_tags) {
      processing_tag_set.emplace(tag);
    }

    if (processing_tag_set.find("trianglearea") != processing_tag_set.end()){
      componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),processed_relpaths.at("trianglearea"))); 
    }
    if (processing_tag_set.find("vertexarea") != processing_tag_set.end()){
      componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),processed_relpaths.at("vertexarea"))); 
    }
    if (processing_tag_set.find("boxes2d") != processing_tag_set.end()){
      componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),processed_relpaths.at("boxes2d"))); 
    }
    return componentpart_chans;
  }
  meshed_projinfo_recording::meshed_projinfo_recording(struct recording_params params,size_t info_structsize) :
    multi_ndarray_recording(params,info_structsize,2)
  {
    snde_geometrydata dummy={0};

    rec_classes.push_back(recording_class_info("snde::meshed_projinfo_recording",typeid(meshed_projinfo_recording),ptr_to_new_shared_impl<meshed_projinfo_recording>));

    define_array(0,rtn_typemap.at(typeid(*dummy.inplane2uvcoords)),"inplane2uvcoords");

    define_array(1,rtn_typemap.at(typeid(*dummy.uvcoords2inplane)),"uvcoords2inplane");
    
    
    // NOTE: Final parameter to multi_ndarray_recording() above is number of mapping entries ***!!! 
  }


  boxes3d_recording::boxes3d_recording(struct recording_params params,size_t info_structsize) :
    multi_ndarray_recording(params,info_structsize,3)
  {
    snde_geometrydata dummy={0};

    rec_classes.push_back(recording_class_info("snde::boxes3d_recording",typeid(boxes3d_recording),ptr_to_new_shared_impl<boxes3d_recording>));

    
    define_array(0,rtn_typemap.at(typeid(*dummy.boxes)),"boxes");

    define_array(1,rtn_typemap.at(typeid(*dummy.boxcoord)),"boxcoord");

    define_array(2,rtn_typemap.at(typeid(*dummy.boxpolys)),"boxpolys");

    
    // NOTE: Final parameter to multi_ndarray_recording() above is number of mapping entries ***!!! 
  }



  boxes2d_recording::boxes2d_recording(struct recording_params params,size_t info_structsize) :
    multi_ndarray_recording(params,info_structsize,0)
  {
    // ***!!!! NOTE: Must call set_num_patches after construction and before assigning storage ***!!!
    rec_classes.push_back(recording_class_info("snde::boxes2d_recording",typeid(boxes2d_recording),ptr_to_new_shared_impl<boxes2d_recording>));

    
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


  
  
  texture_recording::texture_recording(struct recording_params params,size_t info_structsize) :
    multi_ndarray_recording(params,info_structsize,1)
    
  {
    snde_geometrydata dummy={0};

    rec_classes.push_back(recording_class_info("snde::texture_recording",typeid(texture_recording),ptr_to_new_shared_impl<texture_recording>));

    define_array(0,rtn_typemap.at(typeid(*dummy.texbuffer)),"texbuffer");
    
    
  }

  
  image_reference::image_reference(std::string image_path, snde_index u_dimnum, snde_index v_dimnum, const std::vector<snde_index> &other_indices) :
    image_path(image_path),
    u_dimnum(u_dimnum),
    v_dimnum(v_dimnum),
    other_indices(other_indices)
  {

  }

  textured_part_recording::textured_part_recording(struct recording_params params,size_t info_structsize,std::string part_name, std::shared_ptr<std::string> parameterization_name, const std::map<snde_index,std::shared_ptr<image_reference>> &texture_refs) :
    recording_base(params,info_structsize),
    part_name(part_name),
    parameterization_name(parameterization_name),
    texture_refs(texture_refs)
  {
    rec_classes.push_back(recording_class_info("snde::textured_part_recording",typeid(textured_part_recording),ptr_to_new_shared_impl<textured_part_recording>));

  }

  
  // This version primarily for Python wrapping
  textured_part_recording::textured_part_recording(struct recording_params params,size_t info_structsize,std::string part_name, std::shared_ptr<std::string> parameterization_name, std::vector<std::pair<snde_index,std::shared_ptr<image_reference>>> texture_refs_vec) :
    recording_base(params,info_structsize),
    part_name(part_name),
    parameterization_name(parameterization_name)
  {
    for (auto && texref: texture_refs_vec) {
      texture_refs.emplace(texref.first,texref.second);
    }
    
  }

  std::shared_ptr<std::set<std::string>> textured_part_recording::graphics_componentpart_channels(std::shared_ptr<recording_set_state> rss,std::vector<std::string> processing_tags)
  {
    std::shared_ptr<std::set<std::string>> componentpart_chans = std::make_shared<std::set<std::string>>();

    std::unordered_set<std::string> processing_tag_set;
    for (auto && tag: processing_tags) {
      processing_tag_set.emplace(tag);
    }

    if (processing_tag_set.find("projinfo") != processing_tag_set.end()){
      componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),processed_relpaths.at("projinfo"))); 
    }
    componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),part_name));

    if (parameterization_name){
      componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),*parameterization_name));
    }

    for (auto && facenum_imageref: texture_refs) {
      componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),facenum_imageref.second->image_path));
    }
    return componentpart_chans;
  }

  
  assembly_recording::assembly_recording(struct recording_params params,size_t info_structsize,const std::vector<std::pair<std::string,snde_orientation3>> &pieces) :
    recording_base(params,info_structsize),
    pieces(pieces)
  {
    rec_classes.push_back(recording_class_info("snde::assembly_recording",typeid(assembly_recording),ptr_to_new_shared_impl<assembly_recording>));

  }

  const std::shared_ptr<std::map<std::string,std::pair<std::string,std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>>>> assembly_recording::graphics_subcomponents_orientation_lockinfo(std::shared_ptr<recording_set_state> rss)
  {
    std::shared_ptr<std::map<std::string,std::pair<std::string,std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>>>> subcomponents = std::make_shared<std::map<std::string,std::pair<std::string,std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>>>>();
    size_t cnt = 0;
    for (auto && piece: pieces) {
      std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>> empty_lockinfo(std::shared_ptr<multi_ndarray_recording>(),std::make_pair((size_t)0,false));
      subcomponents->emplace(std::make_pair(std::to_string(cnt),std::make_pair(piece.first,empty_lockinfo)));
      cnt++;
    }
    return subcomponents;
  }
  const std::shared_ptr<std::map<std::string,std::pair<std::string,snde_orientation3>>> assembly_recording::graphics_subcomponents_lockedorientations(std::shared_ptr<recording_set_state> rss)
  {
    std::shared_ptr<std::map<std::string,std::pair<std::string,snde_orientation3>>> subcomponents = std::make_shared<std::map<std::string,std::pair<std::string,snde_orientation3>>>();
    size_t cnt = 0;
    for (auto && piece: pieces) {
      subcomponents->emplace(std::to_string(cnt),piece);
      cnt++;
    }
    return subcomponents;
  }

  
  std::shared_ptr<std::set<std::string>> assembly_recording::graphics_componentpart_channels(std::shared_ptr<recording_set_state> rss,std::vector<std::string> processing_tags)
  {
    std::shared_ptr<std::set<std::string>> componentpart_chans = std::make_shared<std::set<std::string>>();

    for (auto && channelpath_orientation: pieces) {
      componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),channelpath_orientation.first));
    }
    return componentpart_chans;
  }

  
  loaded_part_geometry_recording::loaded_part_geometry_recording(struct recording_params params,size_t info_structsize,const std::unordered_set<std::string> &processing_tags,bool loaded_landmarks,bool unchanged_since_load)
 :
    recording_group(params,info_structsize), //,nullptr),
    processing_tags(processing_tags),
    loaded_landmarks(loaded_landmarks),
    unchanged_since_load(unchanged_since_load)
  {
    rec_classes.push_back(recording_class_info("snde::loaded_part_geometry_recording",typeid(loaded_part_geometry_recording),ptr_to_new_shared_impl<loaded_part_geometry_recording>));
    
  }

  
  tracking_pose_recording::tracking_pose_recording(struct recording_params params,size_t info_structsize,std::string channel_to_reorient,std::string component_name):
    recording_base(params,info_structsize),
    channel_to_reorient(channel_to_reorient),
    component_name(component_name)
  {
    rec_classes.push_back(recording_class_info("snde::tracking_pose_recording",typeid(tracking_pose_recording),ptr_to_new_shared_impl<tracking_pose_recording>));

    
  }

  const std::shared_ptr<std::map<std::string,std::pair<std::string,std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>>>> tracking_pose_recording::graphics_subcomponents_orientation_lockinfo(std::shared_ptr<recording_set_state> rss)
  {
   std::shared_ptr<std::map<std::string,std::pair<std::string,std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>>>> subcomponents = std::make_shared<std::map<std::string,std::pair<std::string,std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>>>>();
   subcomponents->emplace("component_name",std::make_pair(component_name,std::make_pair(std::shared_ptr<multi_ndarray_recording>(),std::make_pair(0,false))));
    // Not including channel_to_reorient because its orientation is variable
    // and therefore can't be considered part of an immutable recording
    return subcomponents;
  }
   const std::shared_ptr<std::map<std::string,std::pair<std::string,snde_orientation3>>> tracking_pose_recording::graphics_subcomponents_lockedorientations(std::shared_ptr<recording_set_state> rss)
  {
   std::shared_ptr<std::map<std::string,std::pair<std::string,snde_orientation3>>> subcomponents = std::shared_ptr<std::map<std::string,std::pair<std::string,snde_orientation3>>>();
   snde_orientation3 null_orient;
   snde_null_orientation3(&null_orient);
   subcomponents->emplace("component_name",std::make_pair(component_name,null_orient));
    // Not including channel_to_reorient because its orientation is variable
    // and therefore can't be considered part of an immutable recording
   return subcomponents;
  }

  std::shared_ptr<std::set<std::string>> tracking_pose_recording::graphics_componentpart_channels(std::shared_ptr<recording_set_state> rss,std::vector<std::string> processing_tags)
  {
    std::shared_ptr<std::set<std::string>> componentpart_chans = std::make_shared<std::set<std::string>>();
    
    componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),channel_to_reorient));

    componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),component_name));

    return componentpart_chans;
  }

  
  // Register the pre-existing tracking_pose_recording_display_handler in display_requirement.cpp/hpp as the display handler for pose_channel_tracking_pose_recording
  static int register_pctpr_display_handler = register_recording_display_handler(rendergoal(SNDE_SRG_RENDERING,typeid(pose_channel_tracking_pose_recording)),std::make_shared<registered_recording_display_handler>([] (std::shared_ptr<display_info> display,std::shared_ptr<display_channel> displaychan,std::shared_ptr<recording_set_state> base_rss) -> std::shared_ptr<recording_display_handler_base> {
	return std::make_shared<tracking_pose_recording_display_handler>(display,displaychan,base_rss);
      }));

  pose_channel_tracking_pose_recording::pose_channel_tracking_pose_recording(struct recording_params params,size_t info_structsize,std::string channel_to_reorient,std::string component_name,std::string pose_channel_name):
    tracking_pose_recording(params,info_structsize,channel_to_reorient,component_name),
    pose_channel_name(pose_channel_name)
  {
    rec_classes.push_back(recording_class_info("snde::pose_channel_tracking_pose_recording",typeid(pose_channel_tracking_pose_recording),ptr_to_new_shared_impl<pose_channel_tracking_pose_recording>));

    
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


  pose_channel_recording::pose_channel_recording(struct recording_params params,size_t info_structsize,std::string channel_to_reorient) :
    multi_ndarray_recording(params,info_structsize,1),
    channel_to_reorient(channel_to_reorient)
    
  {
    rec_classes.push_back(recording_class_info("snde::pose_channel_recording",typeid(pose_channel_recording),ptr_to_new_shared_impl<pose_channel_recording>));

    // if (num_ndarrays != 1) {
    //  throw snde_error("pose_channel_recording::pose_channel_recording(%s): Error only single ndarray supported",chanpath.c_str());
    //  }
    
    define_array(0,rtn_typemap.at(typeid(snde_orientation3)),"pose");
  }

  const std::shared_ptr<std::map<std::string,std::pair<std::string,std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>>>> pose_channel_recording::graphics_subcomponents_orientation_lockinfo(std::shared_ptr<recording_set_state> rss)
  {
    std::shared_ptr<std::map<std::string,std::pair<std::string,std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>>>> subcomponents = std::make_shared<std::map<std::string,std::pair<std::string,std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>>>>();
    std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>> lockinfo(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),std::make_pair(0,false));
   
    subcomponents->emplace("channel_to_reorient",std::make_pair(channel_to_reorient,lockinfo));
    if (untransformed_channel) {
      
      
      subcomponents->emplace("untransformed_channel",std::make_pair(*untransformed_channel,std::make_pair(std::shared_ptr<multi_ndarray_recording>(),std::make_pair(0,false))));
    }
    return subcomponents;
  }
   const std::shared_ptr<std::map<std::string,std::pair<std::string,snde_orientation3>>> pose_channel_recording::graphics_subcomponents_lockedorientations(std::shared_ptr<recording_set_state> rss)
  {
    std::shared_ptr<std::map<std::string,std::pair<std::string,snde_orientation3>>> subcomponents = std::make_shared<std::map<std::string,std::pair<std::string,snde_orientation3>>>();
    subcomponents->emplace("channel_to_reorient",std::make_pair(channel_to_reorient,reference_typed_ndarray<snde_orientation3>(0)->element(0)));
    if (untransformed_channel) {
      snde_orientation3 null_orient;
      snde_null_orientation3(&null_orient);
      
      subcomponents->emplace("untransformed_channel",std::make_pair(*untransformed_channel,null_orient));
    }
    return subcomponents;
  }

    std::shared_ptr<std::set<std::string>> pose_channel_recording::graphics_componentpart_channels(std::shared_ptr<recording_set_state> rss,std::vector<std::string> processing_tags)
  {
    std::shared_ptr<std::set<std::string>> componentpart_chans = std::make_shared<std::set<std::string>>();
    
    componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),channel_to_reorient));

    if (untransformed_channel) {
      componentpart_chans->emplace(recdb_path_join(recdb_path_context(info->name),*untransformed_channel));
    }
    
    return componentpart_chans;
  }
  
  // only call during initialization
  void pose_channel_recording::set_untransformed_render_channel(std::string untransformed_channel_str)
  {
    untransformed_channel = std::make_shared<std::string>(untransformed_channel_str);
  }

  /* static */ std::shared_ptr<pose_channel_recording> pose_channel_recording::from_ndarray_recording(std::shared_ptr<multi_ndarray_recording> rec)
  {
    return std::dynamic_pointer_cast<pose_channel_recording>(rec);
  }

  std::shared_ptr<ndarray_recording_ref> create_pose_channel_ndarray_ref(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,std::string channel_to_reorient_name)
  {
    return create_subclass_ndarray_ref<pose_channel_recording>(trans,chan,SNDE_RTN_SNDE_ORIENTATION3,channel_to_reorient_name);
  }

  static void tso_lock_helper(std::shared_ptr<recording_set_state> rss,std::string channel_path,std::string component_path,std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>> this_lock,std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>> &locks,std::function<bool(std::string channel_path,std::string component_path)> recursion_approver)
  {
    std::shared_ptr<recording_base> graphicsrec=rss->check_for_recording(channel_path);
    if (graphicsrec) {
      if (this_lock.first) {
	locks.push_back(this_lock);
      }
      std::shared_ptr<meshed_part_recording> meshed;
      std::shared_ptr<textured_part_recording> texed;
      
      if ((meshed=std::dynamic_pointer_cast<meshed_part_recording>(graphicsrec))) {
	size_t parts_arrayindex=meshed->name_mapping.at("parts");
	//locks.push_back(std::make_pair(meshed,std::make_pair(parts_arrayindex,false)));

      } else if ((texed=std::dynamic_pointer_cast<textured_part_recording>(graphicsrec))) {
	std::shared_ptr<recording_base> meshedrec=rss->check_for_recording(recdb_path_join(recdb_path_context(channel_path),texed->part_name));
	if (meshedrec) {
	  if ((meshed=std::dynamic_pointer_cast<meshed_part_recording>(meshedrec))) {
	    size_t parts_arrayindex=meshed->name_mapping.at("parts");
	    // locks.push_back(std::make_pair(meshed,std::make_pair(parts_arrayindex,false)));  
	  }
	}

	if (texed->parameterization_name) {
	  
	  std::shared_ptr<recording_base> meshedparamrec=rss->check_for_recording(recdb_path_join(recdb_path_context(channel_path),*texed->parameterization_name));
	  if (meshedparamrec) {
	    std::shared_ptr<meshed_parameterization_recording> meshedparam; 
	    if ((meshedparam=std::dynamic_pointer_cast<meshed_parameterization_recording>(meshedparamrec))) {
	      size_t uvs_arrayindex=meshedparam->name_mapping.at("uvs");
	      // locks.push_back(std::make_pair(meshedparam,std::make_pair(uvs_arrayindex,false)));  
	    }
	  }
	}

	for (auto && facenum_imageref : texed->texture_refs) {
	  snde_index facenum;
	  std::shared_ptr<image_reference> imageref;
	  std::tie(facenum,imageref) = facenum_imageref;

	  if (imageref) {
	    std::shared_ptr<recording_base> imagerec = rss->check_for_recording(recdb_path_join(recdb_path_context(channel_path),imageref->image_path));
	    if (imagerec) {
	      std::shared_ptr<multi_ndarray_recording> image;
	      if ((image=std::dynamic_pointer_cast<multi_ndarray_recording>(imagerec))) {
		// locks.push_back(std::make_pair(image,std::make_pair(0,false))); // Assume the first array within the image is what needs to be locked. 
	      }
	    }
	  }
	}
      }
      
      const std::shared_ptr<std::map<std::string,std::pair<std::string,std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>>>> component_lock_map=graphicsrec->graphics_subcomponents_orientation_lockinfo(rss);

      for (auto && field_chanpath_lockinfo: *component_lock_map) {
	auto & lockinfo=field_chanpath_lockinfo.second.second;
	std::string fieldname = field_chanpath_lockinfo.first;
	std::string subcomponent=field_chanpath_lockinfo.second.first;
	std::string subcomponent_path = component_path + "/" + fieldname;
	if (!recdb_path_isabs(subcomponent)) {
	  subcomponent=recdb_path_join(recdb_path_context(channel_path),subcomponent);
	}
	if (recursion_approver(subcomponent,subcomponent_path)){
	  tso_lock_helper(rss,subcomponent,subcomponent_path,lockinfo,locks,recursion_approver);
	}
      }
    }
  }
   // This function collects the locks you need in order to traverse the scene graph and extract the orientations.
  // It does NOT collect the locks needed for the actual geometry or texture.
  // The returned vector is suitable for passing to lockmanager::lock_recording_arrays()
  // The given rss should be complete. 
  std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>> traverse_scenegraph_orientationlocks(std::shared_ptr<recording_set_state> rss,std::string channel_path)
  {
    if (!rss->check_complete()) {
      throw snde_error("traverse_scenegraph_orientationlocks: rss must be complete");
    }
    std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>> locks;
    std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>> this_lock = std::make_pair(nullptr,std::make_pair(0,false)); 
    tso_lock_helper(rss,channel_path,"",this_lock,locks,[] (std::string channel_path,std::string component_path) { return true; });
    return locks;
  }

  // This function is like traverse_scenegraph_orientationlocks except
  // that it will not recurse into any scenegraph node with a channel path
  // matching except_channelpath. In addition to returning the lock info
  // vector, it also returns a vector with recursion info with pairs of
  // (channel_path,component_path) for the instances matching the entries
  // in except_channelpaths. 
  std::pair<std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>>,std::vector<std::pair<std::string,std::string>>> traverse_scenegraph_orientationlocks_except_channelpaths(std::shared_ptr<recording_set_state> rss,std::string channel_path,const std::set<std::string> &except_channelpaths,std::string starting_component_path /* = "" */)
  {
    if (!rss->check_complete()) {
      throw snde_error("traverse_scenegraph_orientationlocks: rss must be complete");
    }
    std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>> locks;
    std::vector<std::pair<std::string,std::string>> except_channelpath_componentpaths;
    
    std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>> this_lock = std::make_pair(nullptr,std::make_pair(0,false)); 
    tso_lock_helper(rss,channel_path,"",this_lock,locks,[except_channelpaths,&except_channelpath_componentpaths] (std::string channel_path,std::string component_path) {
      auto except_channelpath_it=except_channelpaths.find(channel_path);
      if (except_channelpath_it != except_channelpaths.end()) {
	except_channelpath_componentpaths.push_back(std::make_pair(channel_path,component_path));
	return false;
      } else {
	return true;
      }});
    return std::make_pair(locks,except_channelpath_componentpaths);
  }

  
  static void tso_instance_helper(std::shared_ptr<recording_set_state> rss,std::string channel_path,std::string component_path,snde_orientation3 orientation,std::vector<std::tuple<std::string,std::string,snde_partinstance>> &channelpaths_componentpaths_instances,std::function<bool(std::string channel_path,std::string component_path,snde_orientation3 orientation)> recursion_approver)
  {
    std::shared_ptr<recording_base> graphicsrec=rss->check_for_recording(channel_path);
    if (graphicsrec) {
      const std::shared_ptr<std::map<std::string,std::pair<std::string,snde_orientation3>>> component_orientation_map=graphicsrec->graphics_subcomponents_lockedorientations(rss);

      snde_index partnum=SNDE_INDEX_INVALID;
      snde_index firstuvpatch=SNDE_INDEX_INVALID;
      snde_index uvnum=SNDE_INDEX_INVALID;
      std::shared_ptr<meshed_part_recording> meshed;
      std::shared_ptr<textured_part_recording> texed;
      
      if ((meshed=std::dynamic_pointer_cast<meshed_part_recording>(graphicsrec))) {
	size_t parts_arrayindex=meshed->name_mapping.at("parts");
	partnum=meshed->ndinfo(parts_arrayindex)->base_index;
      } else if ((texed=std::dynamic_pointer_cast<textured_part_recording>(graphicsrec))) {
	std::shared_ptr<recording_base> meshedrec=rss->check_for_recording(recdb_path_join(recdb_path_context(channel_path),texed->part_name));
	if (meshedrec) {
	  if ((meshed=std::dynamic_pointer_cast<meshed_part_recording>(meshedrec))) {
	    size_t parts_arrayindex=meshed->name_mapping.at("parts");
	    // locks.push_back(std::make_pair(meshed,std::make_pair(parts_arrayindex,false)));  
	  }
	}
	
	if (texed->parameterization_name) {
	  
	  std::shared_ptr<recording_base> meshedparamrec=rss->check_for_recording(recdb_path_join(recdb_path_context(channel_path),*texed->parameterization_name));
	  if (meshedparamrec) {
	    std::shared_ptr<meshed_parameterization_recording> meshedparam; 
	    if ((meshedparam=std::dynamic_pointer_cast<meshed_parameterization_recording>(meshedparamrec))) {
	      size_t uvs_arrayindex=meshedparam->name_mapping.at("uvs");
	      uvnum=meshedparam->ndinfo(uvs_arrayindex)->base_index;  
	    }
	  }
	}

	for (auto && facenum_imageref : texed->texture_refs) {
	  snde_index facenum;
	  std::shared_ptr<image_reference> imageref;
	  std::tie(facenum,imageref) = facenum_imageref;

	  if (imageref) {
	    std::shared_ptr<recording_base> imagerec = rss->check_for_recording(recdb_path_join(recdb_path_context(channel_path),imageref->image_path));
	    if (imagerec) {
	      std::shared_ptr<multi_ndarray_recording> image;
	      if ((image=std::dynamic_pointer_cast<multi_ndarray_recording>(imagerec))) {
		// ***!!! NEED TO FIGURE OUT HOW TO PROPERLY PASS THE UV PATCH LIST AND THEREFORE HOW TO PUT PATCH INFORMATION INTO THE INSTANCE !!!***
	      }
	    }
	  }
	}
      }

     

      if (partnum != SNDE_INDEX_INVALID) {
	snde_orientation3 inverse_orientation;
	orientation_inverse(orientation,&inverse_orientation);
	
	snde_partinstance instance{
	  orientation,
	  inverse_orientation,
	  partnum,
	  firstuvpatch,
	  uvnum,
	};

	channelpaths_componentpaths_instances.push_back(std::make_tuple(channel_path,component_path,instance));
      }
      for (auto && field_chanpath_orientation: *component_orientation_map) {
	std::string fieldname=field_chanpath_orientation.first;
	std::string subcomponent_channelpath=field_chanpath_orientation.second.first;
	snde_orientation3 subcomponent_relative_orientation=field_chanpath_orientation.second.second;
	if (!recdb_path_isabs(subcomponent_channelpath)) {
	  subcomponent_channelpath=recdb_path_join(recdb_path_context(channel_path),subcomponent_channelpath);
	}
	std::string subcomponent_path=component_path+"/"+fieldname;
	snde_orientation3 subcomponent_orientation;
	orientation_orientation_multiply(orientation,subcomponent_relative_orientation,&subcomponent_orientation);

	if (recursion_approver(subcomponent_channelpath,subcomponent_path,subcomponent_orientation)) {
	  tso_instance_helper(rss,subcomponent_channelpath,subcomponent_path,subcomponent_orientation,channelpaths_componentpaths_instances,recursion_approver);
	}
      }
    }
  }
   //This function traverses the scene graph and extracts the orientations into an array of snde_partinstance.
  //It requires that you have locked the arrays returned by traverse_scenegraph_orientationlocks()
  std::vector<std::tuple<std::string,std::string,snde_partinstance>> traverse_scenegraph_orientationlocked(std::shared_ptr<recording_set_state> rss,std::string channel_path)
  {
    if (!rss->check_complete()) {
      throw snde_error("traverse_scenegraph_orientationlocked: rss must be complete");
    }
    std::vector<std::tuple<std::string,std::string,snde_partinstance>> channelpaths_componentpaths_instances;

    snde_orientation3 null_orient;
    snde_null_orientation3(&null_orient);
    
    tso_instance_helper(rss,channel_path,"",null_orient,channelpaths_componentpaths_instances,[] (std::string channel_path,std::string component_path,snde_orientation3 orientation) { return true; });
    return channelpaths_componentpaths_instances; 
  }

  // This function is like traverse_scenegraph_orientationlocked, except
  // that it will not recurse into any scenegraph node with a channel
  // path matching except_channelpath. In addition it returns  a
  // vector containing the recursion info (channel_path,component_path,orientation) of the instances matching
  // the entries in except_channelpaths. 
  std::pair<std::vector<std::tuple<std::string,std::string,snde_partinstance>>,std::vector<std::tuple<std::string,std::string,snde_orientation3>>> traverse_scenegraph_orientationlocked_except_channelpaths(std::shared_ptr<recording_set_state> rss,std::string channel_path,const std::set<std::string> &except_channelpaths,std::string starting_componentpath /* ="" */,const snde_orientation3 *starting_orientation /* =nullptr */)
  {
    if (!rss->check_complete()) {
      throw snde_error("traverse_scenegraph_orientationlocked: rss must be complete");
    }
    std::vector<std::tuple<std::string,std::string,snde_partinstance>> channelpaths_componentpaths_instances;
    std::vector<std::tuple<std::string,std::string,snde_orientation3>> except_recursioninfo; 
    snde_orientation3 null_orient;

    if (!starting_orientation) {
      snde_null_orientation3(&null_orient);
      starting_orientation = &null_orient;
    }
    
    
    tso_instance_helper(rss,
			channel_path,
			starting_componentpath,
			*starting_orientation,
			channelpaths_componentpaths_instances,
			[ &except_channelpaths, &except_recursioninfo ] (std::string channel_path,std::string component_path,snde_orientation3 orientation) {
			  auto ecp_it = except_channelpaths.find(channel_path);
			  if (ecp_it != except_channelpaths.end()) {
			    except_recursioninfo.push_back(std::make_tuple(channel_path,component_path,orientation));
			    return false;
			  }
			  else {
			    return true;
			  }
			});
    return std::make_pair(channelpaths_componentpaths_instances,except_recursioninfo); 
  }
};
