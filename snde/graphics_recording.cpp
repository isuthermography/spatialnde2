
#include "snde/graphics_recording.hpp"

namespace snde {

  // pointcloud_recording is only compatible with the
  // graphics_storage_manager that defines special storage for
  // certain arrays, including "vertices"
  pointcloud_recording::pointcloud_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<channel> chan,void *owner_id,size_t num_ndarrays,size_t info_structsize/*=sizeof(struct snde_multi_ndarray_recording)*/) :
    multi_ndarray_recording(recdb,chan,owner_id,1,info_structsize)
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
