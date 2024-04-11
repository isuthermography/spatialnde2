#ifndef SNDE_RENDERMODE_HPP
#define SNDE_RENDERMODE_HPP

#include <functional>

#include "snde/quaternion.h"
#include "snde/rec_display.hpp"

namespace snde {
  class renderparams_base;
  class rendergoal;





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
    std::type_index recording_type; // typeindex corresponding to the subclass of snde::recording_base
    
    rendergoal(std::string simple_goal,std::type_index recording_type) :
      simple_goal(simple_goal),
      recording_type(recording_type)
    {
      
    }

    rendergoal(const rendergoal &orig) = default;
    rendergoal & operator=(const rendergoal &) = default;
    ~rendergoal() = default;
    
    bool operator==(const rendergoal &b) const {
      return (simple_goal==b.simple_goal && recording_type==b.recording_type);
    }

    bool operator<(const rendergoal &b) const {
      // needed because we are used as a multimap key
      if (simple_goal < b.simple_goal) {
	return true;
      } else if (simple_goal > b.simple_goal) {
	return false;
      } else {
	return (recording_type < b.recording_type);
      }
    }
    
    std::string str() const {
      return ssprintf("SNDE_SRG: %s_%s",simple_goal.c_str(),recording_type.name());
    }
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
  
  //#define SNDE_SRG_CLASSSPECIFIC 1006 // render in a way that is specific to the particular recording_type indexed in the rendermode
  
  //#define SNDE_SRG_RAW 2 // raw data OK (used for passing 1D waveforms to the renderer)
  //#define SNDE_SRG_RGBAIMAGEDATA 3 // render as an RGBA texture
  //#define SNDE_SRG_RGBAIMAGE 4 // render as an RGBA image 
  //#define SNDE_SRG_GEOMETRY 5 // render as 3D geometry

  struct rendergoal_hash {
    size_t operator()(const rendergoal &x) const
    {
      return std::hash<std::string>{}(x.simple_goal) ^ std::hash<std::type_index>{}(x.recording_type);
    }
  };


  

  class rendermode {
  public:
    // rendermode is a hashable index indicating a choice of how
    // something is to be rendered. It is used to look up a renderer
    // class for performing the rendering in e.g the 
    // osg_renderer_registry.
    
    
    // Modes will be selected in rec_display.cpp:traverse_display_requirements()
    // and encoded in the display_requirement 
    std::string simple_mode; // see SNDE_SRM_XXXX
    std::type_index handler_type; // typeindex corresponding to the subclass of snde::recording_display_handler_base
    
    rendermode(std::string simple_mode,std::type_index handler_type) :
      simple_mode(simple_mode),
      handler_type(handler_type)
    {
      
    }
    rendermode(const rendermode &orig) = default;
    rendermode & operator=(const rendermode &) = default;
    ~rendermode() = default;
    
    bool operator==(const rendermode &b) const {
      return (simple_mode==b.simple_mode && handler_type==b.handler_type);
    }

    std::string str() const {
      return ssprintf("SNDE_SRM: %s_%s",simple_mode.c_str(),handler_type.name());
    }
  };
  
#define SNDE_SRM_INVALID "SNDE_SRM_INVALID" // undetermined/invalid display mode
#define SNDE_SRM_RAW "SNDE_SRM_RAW" // raw data OK (used for passing 1D waveforms to the renderer)   --- NOTE:  This may not be needed?  I don't think we want to pass raw data to the renderer, we need to generate a new set of arrays and dynamically choose which one to use depending on the zoom level.
#define SNDE_SRM_SCALAR "SNDE_SRM_SCALAR"  // maybe redundant with the above entry?
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
#define SNDE_SRM_COLOREDTRANSPARENTPOINTS "SNDE_SRM_COLOREDTRANSPARENTPOINTS"
  
  //#define SNDE_SRM_CLASSSPECIFIC 11 // render in a way that is specific to the particular recording_type indexed in the rendermode

  struct rendermode_hash {
    size_t operator()(const rendermode &x) const
    {
      return std::hash<std::string>{}(x.simple_mode) ^ std::hash<std::type_index>{}(x.handler_type);
    }
  };

    
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
    virtual size_t hash() {
      if (typeid(*this) == typeid(renderparams_base)) {
	return 0xF0F0F0F0F0;
      }

      else {
	throw snde_error("renderparams_base::hash(): hash function not overidden but it should be");
      }
    }
    virtual bool operator==(const renderparams_base& b) {
      if (typeid(*this) == typeid(renderparams_base)) {
	if (typeid(b) == typeid(renderparams_base)) {
	  return true;
	}
	else {
	  return false;
	}
      }

      else {
	throw snde_error("renderparams_base::operator==(): equality operator not overidden but it should be");
      }
    }
    virtual bool operator!=(const renderparams_base& b) {
      return !(*this == b);
    }

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

    rendermode_ext(std::string simple_mode,std::type_index handler_type,std::shared_ptr<renderparams_base> constraint) :
      mode(simple_mode,handler_type),
      constraint(constraint)
    {

    }
    rendermode_ext(const rendermode_ext &orig) = default;
    rendermode_ext & operator=(const rendermode_ext &) = default; 
    ~rendermode_ext() = default;

    bool operator==(const rendermode_ext &b) const {
      bool match = (mode==b.mode);
      if (!match) {
	return false;
      }
      if (constraint && !b.constraint) {
	return false;
      }
      if (!constraint && b.constraint) {
	return false;
      }
      if (!constraint && !b.constraint) {
	return true;
      }
      if (constraint && b.constraint) {
	return *constraint == *b.constraint;
      }
      return false; 
    }


    
  };


  struct rendermode_ext_hash {
    size_t operator()(const rendermode_ext &x) const
    {
      size_t hash = rendermode_hash{}(x.mode);
      if (x.constraint) {
	hash ^= x.constraint->hash();
      }
      return hash;
    }
  };


  class scalar_params : public renderparams_base {
  public:
    RecColor color; // each element 0-1
    float scale;

    scalar_params(RecColor color, float scale) : color(color), scale(scale)
    {

    }

    virtual ~scalar_params() = default;
    virtual size_t hash()
    {
      size_t hashv = std::hash<float>{}(color.R) ^ std::hash<float>{}(color.G) ^ std::hash<float>{}(color.B) ^ std::hash<float>{}(scale);

      return hashv;
    }

    virtual bool operator==(const renderparams_base& b)
    {
      const scalar_params* bptr = dynamic_cast<const scalar_params*>(&b);
      if (!bptr) return false;

      return color.R == bptr->color.R && color.G == bptr->color.G && color.B == bptr->color.B && scale == bptr->scale;

    }


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

    waveform_params(RecColor color, float overall_alpha, double linewidth_horiz, double linewidth_vert, double pointsize, snde_index startidx, snde_index endidx, snde_index idxstep, double datainival, double datastep) :
      color(color),
      overall_alpha(overall_alpha),
      linewidth_horiz(linewidth_horiz),
      linewidth_vert(linewidth_vert),
      pointsize(pointsize),
      startidx(startidx),
      endidx(endidx),
      idxstep(idxstep),
      datainival(datainival),
      datastep(datastep)
    {

    }

    virtual ~waveform_params() = default;
    virtual size_t hash()
    {
      size_t hashv = std::hash<float>{}(color.R) ^ std::hash<float>{}(color.G) ^ std::hash<float>{}(color.B) ^ std::hash<float>{}(overall_alpha) ^ std::hash<double>{}(linewidth_horiz) ^ std::hash<double>{}(linewidth_vert) ^ std::hash<double>{}(pointsize) ^ std::hash<snde_index>{}(startidx) ^ std::hash<snde_index>{}(endidx) ^ std::hash<snde_index>{}(idxstep) ^ std::hash<double>{}(datainival) ^ std::hash<double>{}(datastep);

      return hashv;
    }

    virtual bool operator==(const renderparams_base& b)
    {
      const waveform_params* bptr = dynamic_cast<const waveform_params*>(&b);
      if (!bptr) return false;

      return color.R == bptr->color.R && color.G == bptr->color.G && color.B == bptr->color.B && overall_alpha == bptr->overall_alpha && linewidth_horiz == bptr->linewidth_horiz && linewidth_vert == bptr->linewidth_vert && pointsize == bptr->pointsize && startidx == bptr->startidx && endidx == bptr->endidx && idxstep == bptr->idxstep && datainival == bptr->datainival && datastep == bptr->datastep;

    }


  };


  class color_linewidth_params: public renderparams_base {
  public:
    RecColor color; // each element 0-1
    float overall_alpha; // 0-1

    double linewidth_x;
    double linewidth_y;
    
    color_linewidth_params(RecColor color,float overall_alpha,double linewidth_x,double linewidth_y) :
      color(color),
      overall_alpha(overall_alpha),
      linewidth_x(linewidth_x),
      linewidth_y(linewidth_y)
    {

    }

    virtual ~color_linewidth_params() = default;
    virtual size_t hash()
    {
      size_t hashv = std::hash<float>{}(color.R) ^ std::hash<float>{}(color.G) ^ std::hash<float>{}(color.B) ^ std::hash<float>{}(overall_alpha) ^ std::hash<double>{}(linewidth_x) ^ std::hash<double>{}(linewidth_y);
      
      return hashv;
    }
    
    virtual bool operator==(const renderparams_base &b)
    {
      const color_linewidth_params *bptr = dynamic_cast<const color_linewidth_params*>(&b);
      if (!bptr) return false; 

      return color.R==bptr->color.R && color.G==bptr->color.G && color.B == bptr->color.B && overall_alpha==bptr->overall_alpha && linewidth_x==bptr->linewidth_x && linewidth_y == bptr->linewidth_y;
      
    }
    
    
  };


  class rgbacolormapparams: public renderparams_base {
  public:
    const int ColorMap; // same as displaychan->ColorMap
    const double Offset;
    const double Scale;
    const std::vector<snde_index> other_indices;
    const snde_index u_dimnum;
    const snde_index v_dimnum;

    rgbacolormapparams(int ColorMap,double Offset, double Scale, const std::vector<snde_index> & other_indices,snde_index u_dimnum,snde_index v_dimnum) :
      ColorMap(ColorMap),
      Offset(Offset),
      Scale(Scale),
      other_indices(other_indices),
      u_dimnum(u_dimnum),
      v_dimnum(v_dimnum)
    {

    }
    virtual ~rgbacolormapparams() = default;
    virtual size_t hash()
    {
      size_t hashv = std::hash<int>{}(ColorMap) ^ std::hash<double>{}(Offset) ^ std::hash<double>{}(Scale) ^ std::hash<snde_index>{}(u_dimnum) ^ std::hash<snde_index>{}(v_dimnum);

      for (auto && other_index: other_indices) {
	hashv ^= std::hash<snde_index>{}(other_index);
      }
      return hashv;
    }
    
    virtual bool operator==(const renderparams_base &b)
    {
      const rgbacolormapparams *bptr = dynamic_cast<const rgbacolormapparams*>(&b);
      if (!bptr) return false; 
      
      if (other_indices.size() != bptr->other_indices.size()) {
	return false; 
      }
      bool retval = (ColorMap == bptr->ColorMap && Offset == bptr->Offset && Scale == bptr->Scale && u_dimnum==bptr->u_dimnum && v_dimnum==bptr->v_dimnum);

      for (size_t cnt=0; cnt < other_indices.size(); cnt++) {
	retval = retval && (other_indices.at(cnt) == bptr->other_indices.at(cnt));
      }
      return retval;
      
    }
    
  };

  template <typename T> 
  class vector_renderparams: public renderparams_base {
  public:
    std::vector<T> vec;
    
    virtual size_t hash()
    {
      size_t hashv = 0;

      for (auto && entry: vec) {
	hashv ^= entry.hash();
      }
      return hashv;
    }

    
    virtual bool operator==(const renderparams_base &b)
    {
      const vector_renderparams *bptr = dynamic_cast<const vector_renderparams *>(&b);
      if (!bptr) return false;
      
      if (vec.size() != bptr->vec.size()) {
	return false; 
      }
      bool retval = true;

      for (size_t cnt=0; cnt < vec.size(); cnt++) {
	retval = retval && (vec.at(cnt) == bptr->vec.at(cnt));
      }
      return retval;
      
    }

    virtual void push_back(const T& value)
    {
      vec.push_back(value);
    }
  };
  
  struct chanpathmode_hash {
    size_t operator()(const std::pair<std::string,rendermode>&x) const
    {
      return std::hash<std::string>{}(x.first) + rendermode_hash{}(x.second);
    }
  };

  struct chanpathmodeext_hash {
    size_t operator()(const std::pair<std::string,rendermode_ext>&x) const
    {
      return std::hash<std::string>{}(x.first) + rendermode_ext_hash{}(x.second);
    }
  };

  
  class assemblyparams: public renderparams_base {
  public:
    std::vector<std::shared_ptr<renderparams_base>> component_params; // inner vector for each embedded part or sub assembly 
    // we don't worry about the orientation because that is part of our recording and
    // therefore orientation changes will get caught by the recording equality test of our attempt_reuse() function
    assemblyparams() = default;
    virtual ~assemblyparams() = default;
    
    virtual size_t hash()
    {
      size_t hashv = 0;
      
      for (auto && component_param: component_params) {
	if (component_param) {
	  hashv ^= component_param->hash();
	}
      }
      return hashv;
    }

    
    virtual bool operator==(const renderparams_base &b)
    {
      const assemblyparams *bptr = dynamic_cast<const assemblyparams *>(&b);
      if (!bptr) return false;
      
      if (component_params.size() != bptr->component_params.size()) {
	return false; 
      }
      bool retval = true;

      for (size_t cnt=0; cnt < component_params.size(); cnt++) {
	std::shared_ptr<renderparams_base> component_param = component_params.at(cnt);
	std::shared_ptr<renderparams_base> b_component_param = bptr->component_params.at(cnt);
	retval = retval && ( (!component_param && !b_component_param) || (*component_param == *b_component_param));
      }
      return retval;
      
    }

    virtual void push_back(std::shared_ptr<renderparams_base> value)
    {
      component_params.push_back(value);
    }
    
    
  };



  class poseparams: public renderparams_base { // used for tracking_pose_recording and subclasses
  public:
    std::shared_ptr<renderparams_base> channel_to_reorient_params; // channel we are tracking, with constant view regardless of our viewer orientation
    snde_orientation3 channel_to_reorient_orientation;
    
    
    std::shared_ptr<renderparams_base> untransformed_params; // channel we are observing 
    //snde_orientation3 component_orientation;  // always the identity  
    
    poseparams() = default;
    virtual ~poseparams() = default;
    
    virtual size_t hash()
    {
      size_t hashv = 0;
      
      hashv =
	std::hash<snde_coord>{}(channel_to_reorient_orientation.offset.coord[0]) ^
				 std::hash<snde_coord>{}(channel_to_reorient_orientation.offset.coord[1]) ^ 
				 std::hash<snde_coord>{}(channel_to_reorient_orientation.offset.coord[2]) ^ 
				 std::hash<snde_coord>{}(channel_to_reorient_orientation.offset.coord[3]) ^ 
				 std::hash<snde_coord>{}(channel_to_reorient_orientation.quat.coord[0]) ^ 
				 std::hash<snde_coord>{}(channel_to_reorient_orientation.quat.coord[1]) ^ 
				 std::hash<snde_coord>{}(channel_to_reorient_orientation.quat.coord[2]) ^ 
				 std::hash<snde_coord>{}(channel_to_reorient_orientation.quat.coord[3]);
      if (channel_to_reorient_params) {
	hashv = hashv ^ channel_to_reorient_params->hash();

      }
      if (untransformed_params) {
	hashv = hashv ^ untransformed_params->hash();

      }
							 
      return hashv;
    }

    
    virtual bool operator==(const renderparams_base &b)
    {
      const poseparams *bptr = dynamic_cast<const poseparams *>(&b);
      if (!bptr) return false; 

      if (untransformed_params && !bptr->untransformed_params) {
	return false;	  
      }

      if (!untransformed_params && bptr->untransformed_params) {
	return false;	  
      }

      
      if (untransformed_params && bptr->untransformed_params) {      
	if (!(*untransformed_params == *bptr->untransformed_params)) {
	  return false;
	}
      }

      if (channel_to_reorient_params && !bptr->channel_to_reorient_params) {
	return false;	  
      }

      if (!channel_to_reorient_params && bptr->channel_to_reorient_params) {
	return false;	  
      }

      
      if (channel_to_reorient_params && bptr->channel_to_reorient_params) {      
	if (!(*channel_to_reorient_params == *bptr->channel_to_reorient_params)) {
	  return false;
	}
      }

      
      if (!orientation3_equal(channel_to_reorient_orientation,bptr->channel_to_reorient_orientation)) {
	return false;
      }
      
      return true;
      
    }

    
  };



};

#endif // SNDE_RENDERMODE_HPP
