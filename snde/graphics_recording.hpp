#ifndef SNDE_GRAPHICS_RECORDING_HPP
#define SNDE_GRAPHICS_RECORDING_HPP

#include "snde/recstore.hpp"

namespace snde {

  /*
  class pointcloud_recording: public multi_ndarray_recording {
  public:
    pointcloud_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,size_t num_ndarrays);
    
    };*/


  class meshed_part_recording: public multi_ndarray_recording {
  public:
    //std::shared_ptr<std::string> vertnormals_name; // relative path based on recdb_path_context(recording_path)

    std::map<std::string,std::string> processed_relpaths; // indexed by "meshed", "uv", etc.; should be relative to the recdb_path_context(name), copied from the loaded_part_geometry_recording

    
    meshed_part_recording(struct recording_params params,size_t info_structsize);

    virtual std::shared_ptr<std::set<std::string>> graphics_componentpart_channels(std::shared_ptr<recording_set_state> rss,std::vector<std::string> processing_tags);
    
  };
  
  class meshed_vertexarray_recording: public multi_ndarray_recording {
  public:
    meshed_vertexarray_recording(struct recording_params params,size_t info_structsize);

  };


  class meshed_inplanemat_recording: public multi_ndarray_recording {
  public:
    meshed_inplanemat_recording(struct recording_params params,size_t info_structsize);

  };

  
  class meshed_texvertex_recording: public multi_ndarray_recording {
  public:
    meshed_texvertex_recording(struct recording_params params,size_t info_structsize);

  };

  class meshed_vertnormals_recording: public multi_ndarray_recording {
  public:
    meshed_vertnormals_recording(struct recording_params params,size_t info_structsize);
    // has vertnormals field. 
  };

    class meshed_vertnormalarrays_recording: public multi_ndarray_recording {
  public:
    meshed_vertnormalarrays_recording(struct recording_params params,size_t info_structsize);
    // has vertnormals field. 
  };

  class meshed_trinormals_recording: public multi_ndarray_recording {
  public:
    meshed_trinormals_recording(struct recording_params params,size_t info_structsize);
    // has trinormals field. 
  };

  class meshed_parameterization_recording: public multi_ndarray_recording {
  public:

    std::map<std::string,std::string> processed_relpaths; // indexed by "meshed", "uv", etc.; should be relative to the recdb_path_context(name), copied from the loaded_part_geometry_recording
    
    meshed_parameterization_recording(struct recording_params params,size_t info_structsize);

    virtual std::shared_ptr<std::set<std::string>> graphics_componentpart_channels(std::shared_ptr<recording_set_state> rss,std::vector<std::string> processing_tags);
    
  };

  // meshed_parameterization_recording -> meshed_texvertex_recording for rendering
  

  class meshed_projinfo_recording: public multi_ndarray_recording {
  public:
    meshed_projinfo_recording(struct recording_params params,size_t info_structsize);

  };




  class boxes3d_recording: public multi_ndarray_recording {
  public:
    boxes3d_recording(struct recording_params params,size_t info_structsize);

  };

  class boxes2d_recording: public multi_ndarray_recording {
  public:
    boxes2d_recording(struct recording_params params,size_t info_structsize);
    // ***!!!! NOTE: Must call set_num_patches after construction and before assigning storage ***!!!
    virtual void set_num_patches(snde_index num_patches); // may only be called once 
    
  };


  
  
  class texture_recording: public multi_ndarray_recording {
  public:
    texture_recording(struct recording_params params,size_t info_structsize);
    
  };

  
  class image_reference { // reference to an image or texture
  public:
    std::string image_path; // strings are path names, absolute or relative, treating the path of the textured_part_recording with a trailing slash as a group context
    snde_index u_dimnum; // dimnum of first image/texture coordinate
    snde_index v_dimnum; // dimnum of second image/texture coordinate

    std::vector<snde_index> other_indices; // the u_dimnum and v_dimnum elements should be zero. Should be the same length as the number of dimensions of the referenced texture ndarray

    image_reference(std::string image_path, snde_index u_dimnum, snde_index v_dimnum, const std::vector<snde_index> &other_indices);
     
  };

  
  class textured_part_recording: public recording_base {
  public:
    // NOTE: Texture may or may not be actually present (no texture indicated by nullptr parameterization_name and empty texture_refs
    std::string part_name; // strings are path names, usually relative, treating the recdb_path_context(path of the texured_part_recording) as context
    std::shared_ptr<std::string> parameterization_name;
    std::map<snde_index,std::shared_ptr<image_reference>> texture_refs; // indexed by parameterization face number

    std::map<std::string,std::string> processed_relpaths; // indexed by "meshed", "uv", etc.; should be relative to the recdb_path_context(recording name), copied from the loaded_part_geometry_recording

    
    textured_part_recording(struct recording_params params,size_t info_structsize,std::string part_name, std::shared_ptr<std::string> parameterization_name, const std::map<snde_index,std::shared_ptr<image_reference>> &texture_refs);

    // This version primarily for Python wrapping
    textured_part_recording(struct recording_params params,size_t info_structsize,std::string part_name, std::shared_ptr<std::string> parameterization_name, std::vector<std::pair<snde_index,std::shared_ptr<image_reference>>> texture_refs);

    virtual std::shared_ptr<std::set<std::string>> graphics_componentpart_channels(std::shared_ptr<recording_set_state> rss,std::vector<std::string> processing_tags);

  };
  // textured_part_recording -> renderable_textured_part_recording for rendering, which points at the renderable_meshed_part recording, the meshed_texvertex recording, and an rgba_image_reference

  
  class assembly_recording: public recording_base {
  public:
    std::vector<std::pair<std::string,snde_orientation3>> pieces; // strings are path names, hopefully relative, treating the recdb_path_context() of the assembly_recording's path as the relative context for pieces group context
    
    assembly_recording(struct recording_params params,size_t info_structsize,const std::vector<std::pair<std::string,snde_orientation3>> &pieces);

    virtual const std::shared_ptr<std::map<std::string,std::pair<std::string,std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>>>> graphics_subcomponents_orientation_lockinfo(std::shared_ptr<recording_set_state> rss);

    virtual const std::shared_ptr<std::map<std::string,std::pair<std::string,snde_orientation3>>> graphics_subcomponents_lockedorientations(std::shared_ptr<recording_set_state> rss);

    virtual std::shared_ptr<std::set<std::string>> graphics_componentpart_channels(std::shared_ptr<recording_set_state> rss,std::vector<std::string> processing_tags);
    
  };


  class loaded_part_geometry_recording: public recording_group {
    // represents loaded (or saveable) geometry -- usually a meshed part, a uv, perhaps a texed part and texture, etc. 
  public:
    std::unordered_set<std::string> processing_tags; // processing tags used on load

    std::map<std::string,std::string> processed_relpaths; // indexed by "meshed", "uv", etc.; should be relative to the group. Will also include "landmarks" if those are loaded.

    bool loaded_landmarks;

    bool unchanged_since_load;
    
    loaded_part_geometry_recording(struct recording_params params,size_t info_structsize,const std::unordered_set<std::string> &processing_tags,bool loaded_landmarks,bool unchanged_since_load);
    // Could (should) implement get_meshed_part(), get_texed_part(), get_parameterization(), etc. methods.
  };



  class tracking_pose_recording: public recording_base {
    // the tracking_pose_recording is like a two-component
    // assembly with one component (channel_to_reorient) having particular orientation that, when
    // rendered, tracks the pose of something else, as
    // determined by the behavior of the get_channel_to_reorient_pose() virtual method
    // The channel given by component_name is also rendered but isn't rotated. 
    
    // abstract class: Must subclass! ... Then register your class to use the tracking_pose_recording_display_handler (see display_requirements.cpp)
  public:
    std::string channel_to_reorient; // string is a path name, absolute or relative, treating the path of the tracking_pose_recording with a trailing slash as a group context
    // component name is the component that we can manipulate, etc. 
    std::string component_name; // string is a path name, absolute or relative, treating the path of the tracking_pose_recording with a trailing slash as a group context
    virtual snde_orientation3 get_channel_to_reorient_pose(std::shared_ptr<recording_set_state> rss) const = 0;
    
    tracking_pose_recording(struct recording_params params,size_t info_structsize,std::string channel_to_reorient,std::string component_name);


    virtual const std::shared_ptr<std::map<std::string,std::pair<std::string,std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>>>> graphics_subcomponents_orientation_lockinfo(std::shared_ptr<recording_set_state> rss);

    virtual const std::shared_ptr<std::map<std::string,std::pair<std::string,snde_orientation3>>> graphics_subcomponents_lockedorientations(std::shared_ptr<recording_set_state> rss);

    virtual std::shared_ptr<std::set<std::string>> graphics_componentpart_channels(std::shared_ptr<recording_set_state> rss,std::vector<std::string> processing_tags);
    
  };


  // Note: pose_channel_tracking_pose recording
  // is a renderable channel that refers to an external
  // ndarray pose recording.
  // It works, but is obsolete and (probably) no longer used.
  // Replaced by making the ndarray actually an ndarray_pose_recording
  // so you have one fewer channels. 
  class pose_channel_tracking_pose_recording: public tracking_pose_recording {
  public:
    // Get the orientation of the 
    std::string pose_channel_name;

    // channel_to_reorient will be rotated by the pose stored in the pose channel.
    // component_name will be included like in the assembly, but unrotated
    // pose_channel_name is the name of the channel containing the pose. 
    pose_channel_tracking_pose_recording(struct recording_params params,size_t info_structsize,std::string channel_to_reorient,std::string component_name,std::string pose_channel_name);
    
    virtual snde_orientation3 get_channel_to_reorient_pose(std::shared_ptr<recording_set_state> rss) const;

  };



  // THe pose_channel_recording is an ndarray with a single snde_orientation3
  // as its value. It renders the channel given as background_channel
  // untransformed alongside channel_to_reorient transformed by the
  // given orientation. 


  
  class pose_channel_recording: public multi_ndarray_recording {
  public:
    // should have a single 0D array of type snde_orientation3
    // with the value representing the orient_world_over_object. 
    
    std::string channel_to_reorient; // Name of the channel to render with the given pose, potentially relative to the parent of the pose_channel_recording
    std::shared_ptr<std::string> untransformed_channel; // nullptr, or name of the channel to render untransformed, potentially relative to the parent of the pose_channel_recording
    
    pose_channel_recording(struct recording_params params,size_t info_structsize,std::string channel_to_reorient); // must have num_ndarrays parameter for compatibility with create_subclass_ndarray_ref<S,T>...

    virtual const std::shared_ptr<std::map<std::string,std::pair<std::string,std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>>>> graphics_subcomponents_orientation_lockinfo(std::shared_ptr<recording_set_state> rss);

    virtual const std::shared_ptr<std::map<std::string,std::pair<std::string,snde_orientation3>>> graphics_subcomponents_lockedorientations(std::shared_ptr<recording_set_state> rss);

    virtual std::shared_ptr<std::set<std::string>> graphics_componentpart_channels(std::shared_ptr<recording_set_state> rss,std::vector<std::string> processing_tags);
    
    void set_untransformed_render_channel(std::string untransformed_channel_str); // only call during initialization

    // this static method is used by Python through the SWIG wrappers to get a pose_channel_recording from the .rec attribute of an ndarray_recording_ref
    static std::shared_ptr<pose_channel_recording> from_ndarray_recording(std::shared_ptr<multi_ndarray_recording> rec);

  };

  // convenience function for SWIG
  std::shared_ptr<ndarray_recording_ref> create_pose_channel_ndarray_ref(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,std::string channel_to_reorient_name);

  
  // This function collects the locks you need in order to traverse the scene graph and extract the orientations.
  // It does NOT collect the locks needed for the actual geometry or texture.
  // The returned vector is suitable for passing to lockmanager::lock_recording_arrays()
  std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>> traverse_scenegraph_orientationlocks(std::shared_ptr<recording_set_state> rss,std::string channel_path);

  // This function is like traverse_scenegraph_orientationlocks except
  // that it will not recurse into any scenegraph node with a channel path
  // matching except_channelpath. In addition to returning the lock info
  // vector, it also returns a vector with recursion info with pairs of
  // (channel_path,component_path) for the instances matching the entries
  // in except_channelpaths. 
  std::pair<std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>>,std::vector<std::pair<std::string,std::string>>> traverse_scenegraph_orientationlocks_except_channelpaths(std::shared_ptr<recording_set_state> rss,std::string channel_path,const std::set<std::string> &except_channelpaths,std::string starting_component_path = "");

  //This function traverses the scene graph and extracts the orientations into an array of snde_partinstance.
  //It requires that you have locked the arrays returned by traverse_scenegraph_orientationlocks()
  std::vector<std::tuple<std::string,std::string,snde_partinstance>> traverse_scenegraph_orientationlocked(std::shared_ptr<recording_set_state> rss,std::string channel_path);

 
  // This function is like traverse_scenegraph_orientationlocked, except
  // that it will not recurse into any scenegraph node with a channel
  // path matching except_channelpath. In addition it returns  a
  // vector containing the recursion info (channel_path,component_path,orientation) of the instances matching
  // the entries in except_channelpaths. 
  std::pair<std::vector<std::tuple<std::string,std::string,snde_partinstance>>,std::vector<std::tuple<std::string,std::string,snde_orientation3>>> traverse_scenegraph_orientationlocked_except_channelpaths(std::shared_ptr<recording_set_state> rss,std::string channel_path,const std::set<std::string> &except_channelpaths,std::string starting_componentpath="",const snde_orientation3 *starting_orientation=nullptr);


};
#endif // SNDE_GRAPHICS_RECORDING_HPP
