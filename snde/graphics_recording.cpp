
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
  
  
  image_reference::image_reference(std::string image_path, snde_index u_dimnum, snde_index v_dimnum, const std::vector<snde_index> &other_indices) :
    image_path(image_path),
    u_dimnum(u_dimnum),
    v_dimnum(v_dimnum),
    other_indices(other_indices)
  {

  }


};
