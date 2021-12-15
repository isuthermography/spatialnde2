#ifndef SNDE_OPENSCENEGRAPH_RENDERCACHE_HPP
#define SNDE_OPENSCENEGRAPH_RENDERCACHE_HPP

#include <memory>
#include <osg/LineWidth>
#include <osg/Group>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Texture2D>
#include <osg/Image>
#include <osg/StateSet>
#include <osg/TexMat>

#include "snde/recstore.hpp"
#include "snde/rendermode.hpp"
#include "snde/openscenegraph_array.hpp"
#include "snde/graphics_recording.hpp"

namespace snde {
  class display_requirement; // rec_display.hpp
  class display_channel; // rec_display.hpp
  class display_info; // rec_display.hpp
  
  class recstore_display_transforms; // recstore_display_transforms.hpp
  
  class osg_rendercacheentry;
  class osg_renderparams;

  typedef std::unordered_map<rendermode,std::function<std::shared_ptr<osg_rendercacheentry>(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)>,rendermode_hash> osg_renderer_map;
  std::shared_ptr<osg_renderer_map> osg_renderer_registry(); 

  int osg_register_renderer(rendermode mode,std::function<std::shared_ptr<osg_rendercacheentry>(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)> factory);


  class osg_rendercache {
  public:
    // Warning: Not thread safe -- presumed to be managed by a single thread    
    std::unordered_map<std::pair<std::string,rendermode_ext>,std::shared_ptr<osg_rendercacheentry>,chanpathmodeext_hash> cache; // indexed by channel name and mode

    osg_rendercache() = default;
    osg_rendercache & operator=(const osg_rendercache &) = delete; // shouldn't need copy assignment
    osg_rendercache(const osg_rendercache &) = delete; // shouldn't need copy constructor
    virtual ~osg_rendercache() = default; // subclassable (but not sure why)

    // GetEntry returns (entry, modified_flag). if not modified flag, then everything came out of the
    // cache unmodified so you may not need to rerender at all. 
    std::pair<std::shared_ptr<osg_rendercacheentry>,bool> GetEntry(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);  // mode from rendermode.hpp


    void mark_obsolete(); // mark the potentially_obsolete flag of all cache entries
    void erase_obsolete(); // remove all cache entries which have the potentially_obsolete flag
    
  };
  

  class osg_renderparams {
  public:
    // Do not keep pointers to any of these in your render cache entries
    // as it could create reference loops or keep data in memory unnecessarily
    //std::shared_ptr<recdatabase> recdb;
    std::shared_ptr<osg_rendercache> rendercache;
    std::shared_ptr<recording_set_state> with_display_transforms;
    //std::shared_ptr<display_info> display;

    //std::shared_ptr<display_channel> displaychan; -- get from display->lookup_channel(channel_path)

    
    //std::shared_ptr<recording_base> recording; -- get from with_display_transforms->check_for_recording(channel_path) may want to keep pointer to this in render cache to keep arrays valid (but in the future find way to eliminate so we don't keep recordings in memory unnecessarily???) -- may not be the biggest deal because in the render context we will mostly be looking at subarrays transformed to rgba.

    // Window boundaries -- these parameters are needed for the oscilloscope trace renderer (1D waveform recordings) and image renderer. Get them from display_req->spatial_bounds
    double left; // left edge of viewport in channel horizontal units
    double right; // right edge of viewport in channel horizontal units
    double bottom; // bottom edge of viewport in channel vertical units
    double top; // top edge of viewport in channel vertical units

    size_t width; // width of viewport in pixels
    size_t height; // height of viewport in pixels


  };
  
  
  class osg_rendercacheentry: public std::enable_shared_from_this<osg_rendercacheentry> {
  public:
    //osg_renderparams params;
    

    bool potentially_obsolete; // set by mark_obsolete()
    
    osg_rendercacheentry();
    osg_rendercacheentry & operator=(const osg_rendercacheentry &) = delete; // shouldn't need copy assignment
    osg_rendercacheentry(const osg_rendercacheentry &) = delete; // shouldn't need copy constructor
    virtual ~osg_rendercacheentry() = default; // subclassable

    // attempt_reuse returns two bools:
    //   1. Whether this cache entry can be reused
    //   2. Whether there were any modifications here or deeper in the tree. If false,
    //      then you may not actually have to rerender
    virtual std::pair<bool,bool> attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)=0;
    virtual void clear_potentially_obsolete(); // clear the obsolete flag
  };


  
  class osg_rendercachegroupentry: public osg_rendercacheentry {
  public:
    // inherited elements from osg_rendercacheentry:     
    //bool potentially_obsolete; // set by mark_obsolete()

    
    osg::ref_ptr<osg::Group> osg_group;

    
    osg_rendercachegroupentry()=default;
    osg_rendercachegroupentry & operator=(const osg_rendercachegroupentry &) = delete; // shouldn't need copy assignment
    osg_rendercachegroupentry(const osg_rendercachegroupentry &) = delete; // shouldn't need copy constructor
    virtual ~osg_rendercachegroupentry() = default; // subclassable
    
    virtual std::pair<bool,bool> attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)=0;

  };

  
  class osg_rendercachedrawableentry: public osg_rendercacheentry {
  public:
    // inherited elements from osg_rendercacheentry:     
    //bool potentially_obsolete; // set by mark_obsolete()
    
    osg::ref_ptr<osg::Drawable> osg_drawable;

    
    osg_rendercachedrawableentry()=default;
    osg_rendercachedrawableentry & operator=(const osg_rendercachedrawableentry &) = delete; // shouldn't need copy assignment
    osg_rendercachedrawableentry(const osg_rendercachedrawableentry &) = delete; // shouldn't need copy constructor
    virtual ~osg_rendercachedrawableentry() = default; // subclassable
    
    virtual std::pair<bool,bool> attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)=0;

  };



  class osg_rendercachetextureentry: public osg_rendercacheentry {
  public:
    // inherited elements from osg_rendercacheentry:     
    //bool potentially_obsolete; // set by mark_obsolete()

    osg::ref_ptr<osg::Texture> osg_texture;
    
    osg_rendercachetextureentry()=default;
    osg_rendercachetextureentry & operator=(const osg_rendercachetextureentry &) = delete; // shouldn't need copy assignment
    osg_rendercachetextureentry(const osg_rendercachetextureentry &) = delete; // shouldn't need copy constructor
    virtual ~osg_rendercachetextureentry() = default; // subclassable

    virtual std::pair<bool,bool> attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)=0;

  };

  class osg_rendercachearrayentry: public osg_rendercacheentry {
  public:
    // inherited elements from osg_rendercacheentry:     
    //bool potentially_obsolete; // set by mark_obsolete()

    osg::ref_ptr<OSGFPArray> osg_array;
    
    osg_rendercachearrayentry()=default;
    osg_rendercachearrayentry & operator=(const osg_rendercachearrayentry &) = delete; // shouldn't need copy assignment
    osg_rendercachearrayentry(const osg_rendercachearrayentry &) = delete; // shouldn't need copy constructor
    virtual ~osg_rendercachearrayentry() = default; // subclassable

    virtual std::pair<bool,bool> attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)=0;

  };
  

  

  class osg_cachedimagedata: public osg_rendercachetextureentry {
    // cachedimagedata is the underlying image data (texture)
  public:
    // inherited from osg_rendercachetextureentry/osg_rendercachentry:
    //bool potentially_obsolete; // set by mark_obsolete()
    //osg::ref_ptr<osg::Texture> osg_texture;

    std::shared_ptr<recording_base> cached_recording;
    snde_index dimlenx;
    snde_index dimleny;
    
    osg::ref_ptr<osg::Image> image;
    osg::ref_ptr<osg::PixelBufferObject> imagepbo;
    osg::ref_ptr<osg::TexMat> texture_transform;
    

    osg_cachedimagedata(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);
    ~osg_cachedimagedata() = default;
    
    virtual std::pair<bool,bool> attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);
    
  };

  
  class osg_cachedimage: public osg_rendercachegroupentry {
    // cachedimage is the 2D rendering of an image
  public:
    // inherited from osg_rendercachegroupentry/osg_rendercacheentry:
    // inherited elements from osg_rendercacheentry:     
    //bool potentially_obsolete; // set by mark_obsolete()
    //osg::ref_ptr<osg::Group> osg_group;

    std::shared_ptr<recording_base> cached_recording;

    osg::ref_ptr<osg::Geode> imagegeode;
    osg::ref_ptr<osg::Geometry> imagegeom;
    osg::ref_ptr<osg::DrawArrays> imagetris;
    osg::ref_ptr<osg::StateSet> imagestateset;
    std::shared_ptr<osg_rendercachetextureentry> texture;

    osg_cachedimage(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);
    ~osg_cachedimage() = default;
    
    virtual void clear_potentially_obsolete();

    virtual std::pair<bool,bool> attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);
    
  };

 
   class osg_cachedmeshednormals: public osg_rendercachearrayentry {
  public:
    // inherited from osg_rendercacheentry:
    //std::weak_ptr<display_info> display; // (or should these be passed every time?)
    //std::weak_ptr<display_channel> displaychan;
    
    //osg::ref_ptr<osg::Array> osg_array;
    
    std::shared_ptr<meshed_vertnormals_recording> cached_recording;
    

    osg_cachedmeshednormals(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);
    ~osg_cachedmeshednormals() = default;
    
    //void update(std::shared_ptr<recording_base> new_recording,size_t drawareawidth,size_t drawareaheight,size_t layer_index);
     virtual std::pair<bool,bool> attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);
    
  };


  class osg_cachedparameterizationdata: public osg_rendercachearrayentry {
    // cachedparameterizationdata is the parameterization data (texture coordinates)
  public:
    // inherited from osg_rendercachetextureentry/osg_rendercachentry:
    //bool potentially_obsolete; // set by mark_obsolete()
    //osg::ref_ptr<osg::Array> osg_array;

    std::shared_ptr<meshed_texvertex_recording> cached_recording;
    //snde_index num;

    osg_cachedparameterizationdata(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);
    ~osg_cachedparameterizationdata() = default;
    
    virtual std::pair<bool,bool> attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);
    
  };


  class osg_cachedmeshedvertexarray: public osg_rendercachearrayentry {
    // cachedmeshedvertexarray is the  vertex data (triangle vertex coordinates)
  public:
    // inherited from osg_rendercachearrayentry:
    //bool potentially_obsolete; // set by mark_obsolete()
    //osg::ref_ptr<osg::Array> osg_array;

    std::shared_ptr<meshed_vertexarray_recording> cached_recording;
    //snde_index num;
    

    osg_cachedmeshedvertexarray(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);
    ~osg_cachedmeshedvertexarray() = default;
    
    virtual std::pair<bool,bool> attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);
    
  };

  
  
  class osg_cachedmeshedpart: public osg_rendercachegroupentry {
  public:
    // inherited from osg_rendercacheentry:
    //std::weak_ptr<display_info> display; // (or should these be passed every time?)
    //std::weak_ptr<display_channel> displaychan;
    // osg::ref_ptr<osg::Group> osg_group;    
    std::shared_ptr<meshed_part_recording> cached_recording;
    
    osg::ref_ptr<osg::Geode> geode; 
    osg::ref_ptr<osg::Geometry> geom;
    osg::ref_ptr<osg::DrawArrays> drawarrays;
    osg::ref_ptr<osg::StateSet> stateset;
    
    osg::ref_ptr<snde::OSGFPArray> DataArray;
    osg::ref_ptr<snde::OSGFPArray> NormalArray;

    std::shared_ptr<osg_cachedmeshedvertexarray> vertexarrays_cache;
    std::shared_ptr<osg_cachedmeshednormals> normals_cache;
    
    osg_cachedmeshedpart(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);
    ~osg_cachedmeshedpart() = default;
    
    virtual std::pair<bool,bool> attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);
    //void update(std::shared_ptr<recording_base> new_recording,size_t drawareawidth,size_t drawareaheight,size_t layer_index);
    virtual void clear_potentially_obsolete();
    
  };


  class osg_cachedtexedmeshedgeom: public osg_rendercachedrawableentry {
  public:
    // inherited from osg_rendercacheentry:
    //std::weak_ptr<display_info> display; // (or should these be passed every time?)
    //std::weak_ptr<display_channel> displaychan;
    // osg::ref_ptr<osg::Drawable> osg_drawable;


    std::shared_ptr<textured_part_recording> cached_recording;

    
    //osg::ref_ptr<osg::Geode> geode; //Note: geode missing intentionally because that is how texture
    // is attached and we want the geometry data to be texture-agnostic
    osg::ref_ptr<osg::Geometry> geom; // same object as osg_drawable
    osg::ref_ptr<osg::DrawArrays> drawarrays;
    
    osg::ref_ptr<snde::OSGFPArray> DataArray;
    osg::ref_ptr<snde::OSGFPArray> NormalArray;
    osg::ref_ptr<snde::OSGFPArray> TexCoordArray;

    std::shared_ptr<osg_cachedmeshedvertexarray> vertexarrays_cache;
    std::shared_ptr<osg_cachedmeshednormals> normals_cache;
    std::shared_ptr<osg_cachedparameterizationdata> parameterization_cache;

    osg_cachedtexedmeshedgeom(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);
    ~osg_cachedtexedmeshedgeom() = default;
    
    virtual std::pair<bool,bool> attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);
    //void update(std::shared_ptr<recording_base> new_recording,size_t drawareawidth,size_t drawareaheight,size_t layer_index);
    virtual void clear_potentially_obsolete();
    
  };


  class osg_cachedtexedmeshedpart: public osg_rendercachegroupentry {
  public:
    // inherited from osg_rendercacheentry:
    //std::weak_ptr<display_info> display; // (or should these be passed every time?)
    //std::weak_ptr<display_channel> displaychan;
    // osg::ref_ptr<osg::Group> osg_group;    
    std::shared_ptr<textured_part_recording> cached_recording;
    
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::StateSet> stateset;

    std::shared_ptr<osg_cachedtexedmeshedgeom> geometry_cache;
    std::vector<std::shared_ptr<osg_cachedimagedata>> texture_caches;


    osg_cachedtexedmeshedpart(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);
    ~osg_cachedtexedmeshedpart() = default;
    
    virtual std::pair<bool,bool> attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);
    
    //void update(std::shared_ptr<recording_base> new_recording,size_t drawareawidth,size_t drawareaheight,size_t layer_index);

    virtual void clear_potentially_obsolete();

  };




  class osg_cachedassembly: public osg_rendercachegroupentry {
  public:
    // inherited from osg_rendercacheentry:
    //std::weak_ptr<display_info> display; // (or should these be passed every time?)
    //std::weak_ptr<display_channel> displaychan;
    // osg::ref_ptr<osg::Group> osg_group;    
    
    std::shared_ptr<assembly_recording> cached_recording;

    std::vector<std::shared_ptr<osg_rendercachegroupentry>> sub_components;

    osg_cachedassembly(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);
    ~osg_cachedassembly() = default;
    
    virtual std::pair<bool,bool> attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req);
    
    //void update(std::shared_ptr<recording_base> new_recording,size_t drawareawidth,size_t drawareaheight,size_t layer_index);

    virtual void clear_potentially_obsolete();

  };



};

#endif // SNDE_OPENSCENEGRAPH_RENDERCACHE_HPP
