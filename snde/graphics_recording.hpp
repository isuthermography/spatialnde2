#ifndef SNDE_GRAPHICS_RECORDING_HPP
#define SNDE_GRAPHICS_RECORDING_HPP

#include "snde/recstore.hpp"

namespace snde {
  
  class pointcloud_recording: public multi_ndarray_recording {
  public:
    pointcloud_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,size_t num_ndarrays);
    
  };


  class meshed_part_recording: public multi_ndarray_recording {
  public:
    //std::string part_name; // strings are path names, absolute or relative, treating the path of the assembly_recording with a trailing slash as a group context

    meshed_part_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize);

  };
  
  class meshed_vertexarray_recording: public multi_ndarray_recording {
  public:
    meshed_vertexarray_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize);

  };

  class meshed_texvertex_recording: public multi_ndarray_recording {
  public:
    meshed_texvertex_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize);

  };

  class meshed_vertnormals_recording: public multi_ndarray_recording {
  public:
    meshed_vertnormals_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize);
    // has vertnormals field. 
  };

  class meshed_trinormals_recording: public multi_ndarray_recording {
  public:
    meshed_trinormals_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize);
    // has trinormals field. 
  };

  class meshed_parameterization_recording: public multi_ndarray_recording {
  public:
    meshed_parameterization_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize);

  };

  // meshed_parameterization_recording -> meshed_texvertex_recording for rendering
  
  
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
    // NOTE: Texture may or may not be actually present (no texture indicated by nullptr parameterization_name and empty texture_refs
    std::string part_name; // strings are path names, absolute or relative, treating the path of the texured_part_recording with a trailing slash as a group context
    std::shared_ptr<std::string> parameterization_name;
    std::map<snde_index,std::shared_ptr<image_reference>> texture_refs; // indexed by parameterization face number
    
    
    textured_part_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,std::string part_name, std::shared_ptr<std::string> parameterization_name, const std::map<snde_index,std::shared_ptr<image_reference>> &texture_refs);

  };
  // textured_part_recording -> renderable_textured_part_recording for rendering, which points at the renderable_meshed_part recording, the meshed_texvertex recording, and an rgba_image_reference

  
  class assembly_recording: public recording_group {
  public:
    std::vector<std::tuple<std::string,snde_orientation3>> pieces; // strings are path names, absolute or relative, treating the path of the assembly_recording with a trailing slash as a group context
    
    assembly_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,const std::vector<std::tuple<std::string,snde_orientation3>> &pieces,size_t info_structsize);
  };

};


#endif // SNDE_GRAPHICS_RECORDING_HPP
