#ifndef SNDE_GRAPHICS_RECORDING_HPP
#define SNDE_GRAPHICS_RECORDING_HPP

#include "snde/recstore.hpp"

namespace snde {
  
  class pointcloud_recording: public multi_ndarray_recording {
    pointcloud_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<channel> chan,void *owner_id,size_t num_ndarrays,size_t info_structsize=sizeof(struct snde_multi_ndarray_recording));

  };


  class meshed_part_recording: public multi_ndarray_recording {
    meshed_part_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<channel> chan,void *owner_id,size_t num_ndarrays,size_t info_structsize=sizeof(struct snde_multi_ndarray_recording));

  };

  class meshed_parameterization_recording: public multi_ndarray_recording {
    meshed_parameterization_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<channel> chan,void *owner_id,size_t num_ndarrays,size_t info_structsize=sizeof(struct snde_multi_ndarray_recording));

  };

  class image_reference { // reference to an image or texture
  public:
    std::string image_path; // strings are path names, absolute or relative, treating the path of the textured_part_recording with a trailing slash as a group context
    snde_index u_dimnum; // dimnum of first image/texture coordinate
    snde_index v_dimnum; // dimnum of second image/texture coordinate

    std::vector<snde_index> other_indices; // the u_dimnum and v_dimnum elements should be zero. Should be the same length as the number of dimensions of the referenced texture ndarray

    image_reference(std::string image_path, snde_index u_dimnum, snde_index v_dimnum, const std::vector<snde_index> &other_indices);
    
  };
  
  class textured_part_recording: public recording_group {
  public:
    std::string part_name; // strings are path names, absolute or relative, treating the path of the assembly_recording with a trailing slash as a group context
    std::string parameterization_name;
    std::map<snde_index,std::shared_ptr<image_reference>> texture_refs; // indexed by parameterization face number
    
    
    textured_part_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<channel> chan,void *owner_id,size_t num_ndarrays,size_t info_structsize=sizeof(struct snde_recording_base));

  };

  
  class assembly_recording: public recording_group {
  public:
    const std::vector<std::tuple<std::string,snde_orientation3>> pieces; // strings are path names, absolute or relative, treating the path of the assembly_recording with a trailing slash as a group context
    
    assembly_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<channel> chan,void *owner_id,size_t num_ndarrays,size_t info_structsize=sizeof(struct snde_recording_base));

  };

};


#endif // SNDE_GRAPHICS_RECORDING_HPP
