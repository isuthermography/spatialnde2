//%shared_ptr(snde::pointcloud_recording);
//snde_rawaccessible(snde::pointcloud_recording);

%shared_ptr(snde::meshed_part_recording);
snde_rawaccessible(snde::meshed_part_recording);


%shared_ptr(snde::meshed_vertexarray_recording);
snde_rawaccessible(snde::meshed_vertexarray_recording);

%shared_ptr(snde::meshed_inplanemat_recording);
snde_rawaccessible(snde::meshed_inplanemat_recording);

%shared_ptr(snde::meshed_texvertex_recording);
snde_rawaccessible(snde::meshed_texvertex_recording);

%shared_ptr(snde::meshed_vertnormals_recording);
snde_rawaccessible(snde::meshed_vertnormals_recording);

%shared_ptr(snde::meshed_vertnormalarrays_recording);
snde_rawaccessible(snde::meshed_vertnormalarrays_recording);

%shared_ptr(snde::meshed_trinormals_recording);
snde_rawaccessible(snde::meshed_trinormals_recording);

%shared_ptr(snde::meshed_parameterization_recording);
snde_rawaccessible(snde::meshed_parameterization_recording);

%shared_ptr(snde::meshed_projinfo_recording);
snde_rawaccessible(snde::meshed_projinfo_recording);

%shared_ptr(snde::boxes3d_recording);
snde_rawaccessible(snde::boxes3d_recording);

%shared_ptr(snde::boxes2d_recording);
snde_rawaccessible(snde::boxes2d_recording);

%shared_ptr(snde::texture_recording);
snde_rawaccessible(snde::texture_recording);

%shared_ptr(snde::image_reference);
snde_rawaccessible(snde::image_reference);

%shared_ptr(snde::textured_part_recording);
snde_rawaccessible(snde::textured_part_recording);

%shared_ptr(snde::assembly_recording);
snde_rawaccessible(snde::assembly_recording);

%shared_ptr(snde::loaded_part_geometry_recording);
snde_rawaccessible(snde::loaded_part_geometry_recording);

%shared_ptr(snde::tracking_pose_recording);
snde_rawaccessible(snde::tracking_pose_recording);

%shared_ptr(snde::pose_channel_tracking_pose_recording);
snde_rawaccessible(snde::pose_channel_tracking_pose_recording);


%shared_ptr(snde::pose_channel_recording);
snde_rawaccessible(snde::pose_channel_recording);


%{
#include "snde/graphics_recording.hpp"

%}


namespace snde {

  /*
  class pointcloud_recording: public multi_ndarray_recording {
  public:
    pointcloud_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,size_t num_ndarrays);
    
    };*/


  class meshed_part_recording: public multi_ndarray_recording {
  public:
    //std::string part_name; // strings are path names, absolute or relative, treating the path of the assembly_recording with a trailing slash as a group context

    meshed_part_recording(struct recording_params params,size_t info_structsize);

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
     meshed_parameterization_recording(struct recording_params params,size_t info_structsize);

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
};  

%template(Index_ImageReference_Pair) std::pair<snde_index,std::shared_ptr<snde::image_reference>>;
%template(Index_ImageReference_Pair_Vector) std::vector<std::pair<snde_index,std::shared_ptr<snde::image_reference>>>;

namespace snde {

  class textured_part_recording: public recording_base {
  public:
    // NOTE: Texture may or may not be actually present (no texture indicated by nullptr parameterization_name and empty texture_refs
    std::string part_name; // strings are path names, absolute or relative, treating the path of the texured_part_recording with a trailing slash as a group context
    std::shared_ptr<std::string> parameterization_name;
    std::map<snde_index,std::shared_ptr<image_reference>> texture_refs; // indexed by parameterization face number
    
    
    textured_part_recording(struct recording_params params,size_t info_structsize,std::string part_name, std::shared_ptr<std::string> parameterization_name, const std::map<snde_index,std::shared_ptr<image_reference>> &texture_refs);
    
    // This version primarily for Python wrapping
    textured_part_recording(struct recording_params params,size_t info_structsize,std::string part_name, std::shared_ptr<std::string> parameterization_name, std::vector<std::pair<snde_index,std::shared_ptr<image_reference>>> texture_refs);

  };
  // textured_part_recording -> renderable_textured_part_recording for rendering, which points at the renderable_meshed_part recording, the meshed_texvertex recording, and an rgba_image_reference

  
  class assembly_recording: public recording_base {
  public:
    std::vector<std::pair<std::string,snde_orientation3>> pieces; // strings are path names, absolute or relative, treating the path of the assembly_recording with a trailing slash as a group context
    
     assembly_recording(struct recording_params params,size_t info_structsize,const std::vector<std::pair<std::string,snde_orientation3>> &pieces);
     
  };


  class loaded_part_geometry_recording: public recording_group {
    // represents loaded geometry -- usually a meshed part, a uv, perhaps a texed part and texture, etc. 
  public:
    std::unordered_set<std::string> processing_tags;
    
   loaded_part_geometry_recording(struct recording_params params,size_t info_structsize,const std::unordered_set<std::string> &processing_tags,bool loaded_landmarks,bool unchanged_since_load);
     // Could (should) implement get_meshed_part(), get_texed_part(), get_parameterization(), etc. methods.
   
  };



  class tracking_pose_recording: public recording_base {
    // abstract class: Must subclass! ... Then register your class to use the tracking_pose_recording_display_handler (see display_requirements.cpp)
  public:
    std::string channel_to_reorient; // string is a path name, absolute or relative, treating the path of the tracking_pose_recording with a trailing slash as a group context
    // component name is the component that we can manipulate, etc.
    std::string component_name; // string is a path name, absolute or relative, treating the path of the tracking_pose_recording with a trailing slash as a group context
    virtual snde_orientation3 get_channel_to_reorient_pose(std::shared_ptr<recording_set_state> rss) const = 0;
    
    tracking_pose_recording(struct recording_params params,size_t info_structsize,std::string channel_to_reorient,std::string component_name);
    
  };

  class pose_channel_tracking_pose_recording: public tracking_pose_recording {
  public:
    // Get the orientation of the 
    std::string pose_channel_name;

    pose_channel_tracking_pose_recording(struct recording_params params,size_t info_structsize,std::string channel_to_reorient,std::string component_name,std::string pose_channel_name);
    
    virtual snde_orientation3 get_channel_to_reorient_pose(std::shared_ptr<recording_set_state> rss) const;

  };

  class pose_channel_recording: public multi_ndarray_recording {
  public:
    std::string channel_to_reorient; // Name of the channel to render with the given pose, potentially relative to the parent of the pose_channel_recording
    std::shared_ptr<std::string> untransformed_channel; // nullptr, or name of the channel to render untransformed. 

    
    pose_channel_recording(struct recording_params params,size_t info_structsize,std::string channel_to_reorient); // must have num_ndarrays parameter for compatibility with create_subclass_ndarray_ref<S,T>...

    void set_untransformed_render_channel(std::string component_name_str);

    // this static method is used by Python through the SWIG wrappers to get a pose_channel_recording from the .rec attribute of an ndarray_recording_ref
    static std::shared_ptr<pose_channel_recording> from_ndarray_recording(std::shared_ptr<multi_ndarray_recording> rec);

  };


  std::shared_ptr<ndarray_recording_ref> create_pose_channel_ndarray_ref(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,std::string channel_to_reorient_name);
  
  // moved from recstore.i...
  template <class T>
    std::shared_ptr<T> create_recording_textured_part_info(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,std::string part_name, std::shared_ptr<std::string> parameterization_name, std::vector<std::pair<snde_index,std::shared_ptr<image_reference>>> texture_refs);
  %{
#define create_recording_textured_part_info create_recording
   %}

  
  // create_recording templates
  %template(create_meshed_part_recording) create_recording_noargs<meshed_part_recording>;
  %template(create_meshed_vertexarray_recording) create_recording_noargs<meshed_vertexarray_recording>;
  %template(create_meshed_texvertex_recording) create_recording_noargs<meshed_texvertex_recording>;
  %template(create_meshed_vertnormals_recording) create_recording_noargs<meshed_vertnormals_recording>;
  %template(create_meshed_trinormals_recording) create_recording_noargs<meshed_trinormals_recording>;
  %template(create_meshed_parameterization_recording) create_recording_noargs<meshed_parameterization_recording>;
  %template(create_texture_recording) create_recording_noargs<texture_recording>;


  
  // Note customized pseudo-templates for extra create_recording arguments are defined at the
  // bottom of recstore.i
  %template(create_pose_channel_recording) create_recording_string<pose_channel_recording>;
  %template(create_textured_part_recording) create_recording_textured_part_info<textured_part_recording>;  
  %template(create_assembly_recording) create_recording_const_vector_of_string_orientation_pairs<assembly_recording>;
  
  // can't create a tracking_pose_recording because it is an abstract class
  %template(create_pose_channel_tracking_pose_recording) create_recording_three_strings<pose_channel_tracking_pose_recording>;

  // template for instantiation of a recording_ref to a pose_channel .... replaced by explict function, above
  //%template(create_pose_channel_ndarray_ref) snde::create_subclass_ndarray_ref_string<snde::pose_channel_recording>;

};
