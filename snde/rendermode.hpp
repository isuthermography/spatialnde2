#ifndef SNDE_RENDERMODE_HPP
#define SNDE_RENDERMODE_HPP

#include <functional>

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
    
    // For now it is rather trivial -- just wrapping an int, but it
    // will likely be extended in the future
    // to support add-on rendering code and guided intelligent choosing of modes.
    
    // Modes will be selected in rec_display.cpp:traverse_display_requirements()
    // and encoded in the display_requirement 
    int simple_goal; // see SNDE_SRG_XXXX
    std::type_index recording_type; // typeindex corresponding to the subclass of snde::recording_base
    
    rendergoal(int simple_goal,std::type_index recording_type) :
      simple_goal(simple_goal),
      recording_type(recording_type)
    {
      
    }
    
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
      return ssprintf("SNDE_SRG_#%d_%s",simple_goal,recording_type.name());
    }
  };

  // Start at 1000 to provide an distinction between
  // these and the SNDE_SRM_... constants
#define SNDE_SRG_INVALID 1000 // undetermined/invalid display mode
#define SNDE_SRG_RENDERING 1001 // goal is to perform rendering of the underlying data in this recording
#define SNDE_SRG_TEXTURE 1002 // goal is to create a texture representing the underlying data in this recording
#define SNDE_SRG_VERTEXARRAYS 1003 // goal is to create otriangle vertex arrays
#define SNDE_SRG_VERTNORMALS 1004 // goal is to create otriangle vertex arrays
#define SNDE_SRG_GEOMETRY 1005 // goal is to create bare geometry (vertices and parameterization, but no texture)
#define SNDE_SRG_CLASSSPECIFIC 1006 // render in a way that is specific to the particular recording_type indexed in the rendermode
  
  //#define SNDE_SRG_RAW 2 // raw data OK (used for passing 1D waveforms to the renderer)
  //#define SNDE_SRG_RGBAIMAGEDATA 3 // render as an RGBA texture
  //#define SNDE_SRG_RGBAIMAGE 4 // render as an RGBA image 
  //#define SNDE_SRG_GEOMETRY 5 // render as 3D geometry

  struct rendergoal_hash {
    size_t operator()(const rendergoal &x) const
    {
      return std::hash<int>{}(x.simple_goal) ^ std::hash<std::type_index>{}(x.recording_type);
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
    int simple_mode; // see SNDE_SRM_XXXX
    std::type_index handler_type; // typeindex corresponding to the subclass of snde::recording_display_handler_base
    
    rendermode(int simple_mode,std::type_index handler_type) :
      simple_mode(simple_mode),
      handler_type(handler_type)
    {
      
    }
    
    bool operator==(const rendermode &b) const {
      return (simple_mode==b.simple_mode && handler_type==b.handler_type);
    }

    std::string str() const {
      return ssprintf("SNDE_SRM_#%d_%s",simple_mode,handler_type.name());
    }
  };
  
#define SNDE_SRM_INVALID 0 // undetermined/invalid display mode
#define SNDE_SRM_RAW 1 // raw data OK (used for passing 1D waveforms to the renderer)
#define SNDE_SRM_RGBAIMAGEDATA 2 // render as an RGBA texture
#define SNDE_SRM_RGBAIMAGE 3 // render as an RGBA image
#define SNDE_SRM_MESHEDNORMALS 4 // collect array of meshed normals
#define SNDE_SRM_VERTEXARRAYS 5 // collect array of triangle vertices
#define SNDE_SRM_MESHED2DPARAMETERIZATION 6 // collect array of texture triangle vertices (parameterization
#define SNDE_SRM_MESHEDPARAMLESS3DPART 7 // render meshed 3D geometry part with no 2D parameterization or texture
#define SNDE_SRM_TEXEDMESHED3DGEOM 8 // render meshed 3D geometry with texture
#define SNDE_SRM_TEXEDMESHEDPART 9 // render textured meshed 3D geometry part
#define SNDE_SRM_ASSEMBLY 10 // render a collection of objects (group) representing an assembly
  
#define SNDE_SRM_CLASSSPECIFIC 11 // render in a way that is specific to the particular recording_type indexed in the rendermode

  struct rendermode_hash {
    size_t operator()(const rendermode &x) const
    {
      return std::hash<int>{}(x.simple_mode) ^ std::hash<std::type_index>{}(x.handler_type);
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

    rendermode_ext(int simple_mode,std::type_index handler_type,std::shared_ptr<renderparams_base> constraint) :
      mode(simple_mode,handler_type),
      constraint(constraint)
    {

    }

    bool operator==(const rendermode_ext &b) const {
      return (mode==b.mode) && (*constraint == *b.constraint);
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
    
    virtual size_t hash()
    {
      size_t hashv = 0;
      
      for (auto && component_param: component_params) {
	hashv ^= component_param->hash();
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
	retval = retval && (*component_params.at(cnt) == *bptr->component_params.at(cnt));
      }
      return retval;
      
    }

    virtual void push_back(std::shared_ptr<renderparams_base> value)
    {
      component_params.push_back(value);
    }
    
    
  };
};

#endif // SNDE_RENDERMODE_HPP
