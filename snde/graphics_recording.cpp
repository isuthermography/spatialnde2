
#include "snde/graphics_recording.hpp"

namespace snde {

  // pointcloud_recording is only compatible with the
  // graphics_storage_manager that defines special storage for
  // certain arrays, including "vertices"
  pointcloud_recording::pointcloud_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,size_t num_ndarrays) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,1)
  {
    name_mapping.emplace(std::make_pair("vertices",0));
    name_reverse_mapping.emplace(std::make_pair(0,"vertices"));
  }

  meshed_part_recording::meshed_part_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,8)
  {
    name_mapping.emplace(std::make_pair("parts",0));
    name_reverse_mapping.emplace(std::make_pair(0,"parts"));

    name_mapping.emplace(std::make_pair("topos",1));
    name_reverse_mapping.emplace(std::make_pair(1,"topos"));

    name_mapping.emplace(std::make_pair("topo_indices",2));
    name_reverse_mapping.emplace(std::make_pair(2,"topo_indices"));

    name_mapping.emplace(std::make_pair("triangles",3));
    name_reverse_mapping.emplace(std::make_pair(3,"triangles"));
    
    name_mapping.emplace(std::make_pair("edges",4));
    name_reverse_mapping.emplace(std::make_pair(4,"edges"));

    name_mapping.emplace(std::make_pair("vertices",5));
    name_reverse_mapping.emplace(std::make_pair(5,"vertices"));

    name_mapping.emplace(std::make_pair("vertex_edgelist_indices",6));
    name_reverse_mapping.emplace(std::make_pair(6,"vertex_edgelist_indices"));
    
    name_mapping.emplace(std::make_pair("vertex_edgelist",7));
    name_reverse_mapping.emplace(std::make_pair(7,"vertex_edgelist"));

    // NOTE: Final parameter to multi_ndarray_recording() above is number of mapping entries ***!!! 
  }

  meshed_vertexarray_recording::meshed_vertexarray_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,1)
  {
    name_mapping.emplace(std::make_pair("vertex_arrays",0));
    name_reverse_mapping.emplace(std::make_pair(0,"vertex_arrays"));

  }

  meshed_texvertex_recording::meshed_texvertex_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,1)
  {
    name_mapping.emplace(std::make_pair("texvertex_arrays",0));
    name_reverse_mapping.emplace(std::make_pair(0,"texvertex_arrays"));

  }
    
  meshed_vertnormals_recording::meshed_vertnormals_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,1)
  {
    name_mapping.emplace(std::make_pair("vertnormals",0));
    name_reverse_mapping.emplace(std::make_pair(0,"vertnormals"));

  }

  meshed_trinormals_recording::meshed_trinormals_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
   multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,1)
  {
    name_mapping.emplace(std::make_pair("trinormals",0));
    name_reverse_mapping.emplace(std::make_pair(0,"trinormals"));

  }

  meshed_parameterization_recording::meshed_parameterization_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
   multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,9)
  {

    name_mapping.emplace(std::make_pair("uvs",0));
    name_reverse_mapping.emplace(std::make_pair(0,"uvs"));

    name_mapping.emplace(std::make_pair("uv_patches",1));
    name_reverse_mapping.emplace(std::make_pair(1,"uv_patches"));

    name_mapping.emplace(std::make_pair("uv_topos",2));
    name_reverse_mapping.emplace(std::make_pair(2,"uv_topos"));

    name_mapping.emplace(std::make_pair("uv_topo_indices",3));
    name_reverse_mapping.emplace(std::make_pair(3,"uv_topo_indices"));

    name_mapping.emplace(std::make_pair("uv_triangles",4));
    name_reverse_mapping.emplace(std::make_pair(4,"uv_triangles"));

    name_mapping.emplace(std::make_pair("uv_edges",5));
    name_reverse_mapping.emplace(std::make_pair(5,"uv_edges"));

    name_mapping.emplace(std::make_pair("uv_vertices",6));
    name_reverse_mapping.emplace(std::make_pair(6,"uv_vertices"));

    name_mapping.emplace(std::make_pair("uv_vertex_edgelist_indices",7));
    name_reverse_mapping.emplace(std::make_pair(7,"uv_vertex_edgelist_indices"));

    name_mapping.emplace(std::make_pair("uv_vertex_edgelist",8));
    name_reverse_mapping.emplace(std::make_pair(8,"uv_vertex_edgelist"));

    // NOTE: Final parameter to multi_ndarray_recording() above is number of mapping entries ***!!! 
  }
  
  image_reference::image_reference(std::string image_path, snde_index u_dimnum, snde_index v_dimnum, const std::vector<snde_index> &other_indices) :
    image_path(image_path),
    u_dimnum(u_dimnum),
    v_dimnum(v_dimnum),
    other_indices(other_indices)
  {

  }

  textured_part_recording::textured_part_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,std::string part_name, std::shared_ptr<std::string> parameterization_name, const std::map<snde_index,std::shared_ptr<image_reference>> &texture_refs) :
    recording_group(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,nullptr,info_structsize),
    part_name(part_name),
    parameterization_name(parameterization_name),
    texture_refs(texture_refs)
  {

  }

  assembly_recording::assembly_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,const std::vector<std::tuple<std::string,snde_orientation3>> &pieces,size_t info_structsize) :
    recording_group(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,nullptr,info_structsize),
    pieces(pieces)
  {
    
  }

};
