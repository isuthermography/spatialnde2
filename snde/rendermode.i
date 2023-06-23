%shared_ptr(snde::renderparams_base);
snde_rawaccessible(snde::renderparams_base);
%shared_ptr(snde::waveform_params);
snde_rawaccessible(snde::waveform_params);
%shared_ptr(snde::color_linewidth_params);
snde_rawaccessible(snde::color_linewidth_params);
%shared_ptr(snde::rgbacolormapparams);
snde_rawaccessible(snde::rgbacolormapparams);
%shared_ptr(snde::assemblyparams);
snde_rawaccessible(snde::assemblyparams);
%shared_ptr(snde::poseparams);
snde_rawaccessible(snde::poseparams);

%{
#include "snde/rendermode.hpp"
%}

namespace snde {


  class rendergoal {
  public:
    // rendergoal is a hashable index indicating a goal for how
    // something should be rendered, indicating both the class
    // (recording_type, which should refer to a subclass of
    // snde::recording_base) and the rendering goal
    // (simple_goal)
    
    // For now it is rather trivial -- just wrapping a string, but it
    // will likely be extended in the future
    // to support add-on rendering code and guided intelligent choosing of modes.
    
    // Modes will be selected in rec_display.cpp:traverse_display_requirements()
    // and encoded in the display_requirement 
    std::string simple_goal; // see SNDE_SRG_XXXX
    //std::type_index recording_type; // typeindex corresponding to the subclass of snde::recording_base
    
    //rendergoal(std::string simple_goal,std::type_index recording_type);
    rendergoal(const rendergoal &orig) = default;
    //rendergoal & operator=(const rendergoal &) = default;
    ~rendergoal() = default;
    
    bool operator==(const rendergoal &b);

    bool operator<(const rendergoal &b) const;
    
    std::string str() const;
  };

#define SNDE_SRG_INVALID "SNDE_SRG_INVALID" // undetermined/invalid display mode
#define SNDE_SRG_DEFAULT "SNDE_SRG_DEFAULT" // Use default, which is to look up metadata entry "snde_render_goal" or use SNDE_SRG_RENDERING if not set. 
#define SNDE_SRG_DEFAULT_3D "SNDE_SRG_DEFAULT_3D" // Use 3D default, which is to look up metadata entry "snde_render_goal_3d" or use SNDE_SRG_RENDERING_3D if not set. 
#define SNDE_SRG_RENDERING "SNDE_SRG_RENDERING" // goal is to perform rendering of the underlying data in this recording
#define SNDE_SRG_RENDERING_3D "SNDE_SRG_RENDERING_3D" // goal is to perform 3D rendering of the underlying data in this recording
#define SNDE_SRG_RENDERING_2D "SNDE_SRG_RENDERING_2D" // goal is to perform 2D rendering of the underlying data in this recording
#define SNDE_SRG_TEXTURE "SNDE_SRG_TEXTURE" // goal is to create a texture representing the underlying data in this recording
#define SNDE_SRG_VERTEXARRAYS "SNDE_SRG_VERTEXARRAYS"  // goal is to create triangle vertex arrays
#define SNDE_SRG_VERTNORMALS "SNDE_SRG_VERTNORMALS" // goal is to create otriangle vertex arrays
#define SNDE_SRG_GEOMETRY "SNDE_SRG_GEOMETRY" // goal is to create bare geometry (vertices and parameterization, but no texture)
#define SNDE_SRG_POINTCLOUD "SNDE_SRG_POINTCLOUD" // goal is to render a point cloud
#define SNDE_SRG_POINTCLOUDCOLORMAP "SNDE_SRG_POINTCLOUDCOLORMAP" // goal is to render a point cloud colormap

#define SNDE_SRG_PHASEPLANE "SNDE_SRG_PHASEPLANE" // goal is to render a phase plane diagram
#define SNDE_SRG_PHASEPLANE_LINE_TRIANGLE_VERTICES_ALPHAS "SNDE_SRG_PHASEPLANE_LINE_TRIANGLE_VERTICES_ALPHAS"


  

  class rendermode {
  public:
    // rendermode is a hashable index indicating a choice of how
    // something is to be rendered. It is used to look up a renderer
    // class for performing the rendering in e.g the 
    // osg_renderer_registry.
    
    
    // Modes will be selected in rec_display.cpp:traverse_display_requirements()
    // and encoded in the display_requirement 
    std::string simple_mode; // see SNDE_SRM_XXXX
    //std::type_index handler_type; // typeindex corresponding to the subclass of snde::recording_display_handler_base
    
    //rendermode(std::string simple_mode,std::type_index handler_type);
    rendermode(const rendermode &orig) = default;
    //rendermode & operator=(const rendermode &) = default;
    ~rendermode() = default;

    
    bool operator==(const rendermode &b) const;

    std::string str() const;
  };
  
#define SNDE_SRM_INVALID "SNDE_SRM_INVALID" // undetermined/invalid display mode
#define SNDE_SRM_RAW "SNDE_SRM_RAW" // raw data OK (used for passing 1D waveforms to the renderer)   --- NOTE:  This may not be needed?  I don't think we want to pass raw data to the renderer, we need to generate a new set of arrays and dynamically choose which one to use depending on the zoom level.
#define SNDE_SRM_WAVEFORM "SNDE_SRM_WAVEFORM" //  1D waveform
#define SNDE_SRM_RGBAIMAGEDATA "SNDE_SRM_RGBAIMAGEDATA" // render as an RGBA texture
#define SNDE_SRM_RGBAIMAGE "SNDE_SRM_RGBAIMAGE" // render as an RGBA image
#define SNDE_SRM_MESHEDNORMALS "SNDE_SRM_MESHEDNORMALS" // collect array of meshed normals
#define SNDE_SRM_VERTEXARRAYS "SNDE_SRM_VERTEXARRAYS" // collect array of triangle vertices
  
#define SNDE_SRM_POINTCLOUD "SNDE_SRM_POINTCLOUD" // render a point cloud in 3D space
#define SNDE_SRM_POINTCLOUDCOLORMAP "SNDE_SRM_POINTCLOUDCOLORMAP" // colormapping for a point cloud in 3D space
#define SNDE_SRM_POINTCLOUDVERTICES "SNDE_SRM_POINTCLOUDVERTICES" // vertices for a point cloud in 3D space

#define SNDE_SRM_MESHED2DPARAMETERIZATION "SNDE_SRM_MESHED2DPARAMETERIZATION" // collect array of texture triangle vertices (parameterization
#define SNDE_SRM_MESHEDPARAMLESS3DPART "SNDE_SRM_MESHEDPARAMELESS3DPART" // render meshed 3D geometry part with no 2D parameterization or texture
#define SNDE_SRM_TEXEDMESHED3DGEOM "SNDE_SRM_TEXEDMESHED3DGEOM" // render meshed 3D geometry with texture
#define SNDE_SRM_TEXEDMESHEDPART "SNDE_SRM_TEXEDMESHEDPART" // render textured meshed 3D geometry part
#define SNDE_SRM_ASSEMBLY "SNDE_SRM_ASSEMBLY" // render a collection of objects (group) representing an assembly

#define SNDE_SRM_TRANSFORMEDCOMPONENT "SNDE_SRM_TRANSFORMEDCOMPONENT" // render a component with the pose transform defined in the rendermode_ext
  
#define SNDE_SRM_PHASE_PLANE_ENDPOINT_WITH_COLOREDTRANSPARENTLINES "SNDE_SRM_PHASE_PLANE_ENDPOINT_WITH_COLOREDTRANSPARENTLINES"

#define SNDE_SRM_COLOREDTRANSPARENTLINES "SNDE_SRM_COLOREDTRANSPARENTLINES"
  
  //#define SNDE_SRM_CLASSSPECIFIC 11 // render in a way that is specific to the particular recording_type indexed in the rendermode
    
  class renderparams_base {
    // derive specific cases of render parameters that affect
    // the graphic cache from this class.
    // implement the hash and equality operators (latter
    // using dynamic_cast to verify the type match)
    // Don't have 2nd generation descendent classes as
    // the appropriate behavior of operator==() becomes
    // somewhat ambiguous in that case 
  public:
    renderparams_base() = default;
    renderparams_base(const renderparams_base &orig) = default;
    renderparams_base & operator=(const renderparams_base &) = delete; // shouldn't need this...
    virtual ~renderparams_base() = default;
    virtual size_t hash()=0;
    virtual bool operator==(const renderparams_base &b)=0;
  };



  // rendermode_ext is used as the index in a renderer cache
  // to match the exact way something was rendered.

  // It consists of the rendermode, which uniquely identifies
  // the renderer, and an additional constraint based on
  // parameters that matter to the renderer.

  class rendermode_ext {
  public:
    rendermode mode;
    std::shared_ptr<renderparams_base> constraint; // constraint limits the validity of this cache entry

    //rendermode_ext(std::string simple_mode,std::type_index handler_type,std::shared_ptr<renderparams_base> constraint);
    rendermode_ext(const rendermode_ext &orig) = default;
    //rendermode_ext & operator=(const rendermode_ext &) = default; 
    ~rendermode_ext() = default;

    bool operator==(const rendermode_ext &b) const;
    
  };

  struct RecColor {
  double R,G,B;
    
    friend bool operator==(const RecColor &lhs, const RecColor &rhs);
    friend bool operator!=(const RecColor &lhs, const RecColor &rhs);
  };

  class waveform_params : public renderparams_base {
  public:
    RecColor color; // each element 0-1
    float overall_alpha; // 0-1
    double linewidth_horiz;
    double linewidth_vert;
    double pointsize;
    snde_index startidx;
    snde_index endidx;
    snde_index idxstep;
    double datainival;
    double datastep;

    waveform_params(RecColor color, float overall_alpha, double linewidth_horiz, double linewidth_vert, double pointsize, snde_index startidx, snde_index endidx, snde_index idxstep, double datainival, double datastep);

    virtual ~waveform_params() = default;
    virtual size_t hash();

    virtual bool operator==(const renderparams_base& b);

  };


  
  class color_linewidth_params: public renderparams_base {
  public:
    RecColor color; // each element 0-1
    float overall_alpha; // 0-1

    double linewidth_x;
    double linewidth_y;
    
    color_linewidth_params(RecColor color,float overall_alpha,double linewidth_x,double linewidth_y);

    virtual ~color_linewidth_params() = default;
    virtual size_t hash();
    
    virtual bool operator==(const renderparams_base &b);    
    
  };



  class rgbacolormapparams: public renderparams_base {
  public:
    const int ColorMap; // same as displaychan->ColorMap
    const double Offset;
    const double Scale;
    const std::vector<snde_index> other_indices;
    const snde_index u_dimnum;
    const snde_index v_dimnum;

    rgbacolormapparams(int ColorMap,double Offset, double Scale, const std::vector<snde_index> & other_indices,snde_index u_dimnum,snde_index v_dimnum);
    virtual ~rgbacolormapparams() = default;

    virtual size_t hash();
    
    virtual bool operator==(const renderparams_base &b);
    
  };

  
  class assemblyparams: public renderparams_base {
  public:
    std::vector<std::shared_ptr<renderparams_base>> component_params; // inner vector for each embedded part or sub assembly 
    // we don't worry about the orientation because that is part of our recording and
    // therefore orientation changes will get caught by the recording equality test of our attempt_reuse() function
    assemblyparams() = default;
    virtual ~assemblyparams() = default;
    
    virtual size_t hash();
    
    virtual bool operator==(const renderparams_base &b);

    virtual void push_back(std::shared_ptr<renderparams_base> value);
    
  };


  class poseparams: public renderparams_base { // used for tracking_pose_recording and subclasses
  public:
    std::shared_ptr<renderparams_base> channel_to_reorient_params; // channel we are tracking, with constant view regardless of our viewer orientation
    snde_orientation3 channel_to_reorient_orientation;
    
    
    std::shared_ptr<renderparams_base> untransformed_params; // channel we are observing 
    //snde_orientation3 untransformed_orientation;  // always the identity  
    
    
    poseparams() = default;
    virtual ~poseparams() = default;
    
    virtual size_t hash();
    
    virtual bool operator==(const renderparams_base &b);

    
  };



};

