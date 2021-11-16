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

#include "snde/recstore.hpp"

namespace snde {
  class display_requirement; // rec_display.hpp
  class display_channel; // rec_display.hpp
  class display_info; // rec_display.hpp
  
  class recstore_display_transforms; // recstore_display_transforms.hpp
  
  class osg_rendercacheentry;
  
  
  class osg_rendercache {
  public:
    // Warning: Not thread safe -- presumed to be managed by a single thread    
    std::map<std::pair<std::string,int>,std::shared_ptr<osg_rendercacheentry>> cache; // indexed by channel name

    osg_rendercache() = default;
    osg_rendercache & operator=(const osg_rendercache &) = delete; // shouldn't need copy assignment
    osg_rendercache(const osg_rendercache &) = delete; // shouldn't need copy constructor
    virtual ~osg_rendercache() = default; // subclassable (but not sure why)


    osg::ref_ptr<osg::Group> GetEntry(std::shared_ptr<recording_set_state> with_display_transforms,const std::string &channel_path,int mode,double left,double right,double bottom,double top); // mode is SNDE_DRM_xxxx from rec_display.hpp
    
    //void update_cache(std::shared_ptr<recdatabase> recdb,const std::vector<display_requirement> &display_reqs,std::shared_ptr<recstore_display_transforms> viewable,std::vector<std::shared_ptr<display_channel>> channels, const std::string &selected, bool selected_only); // NOTE: this routine locks mutable recordings

    void mark_obsolete(); // mark the potentially_obsolete flag of all cache entries
    void clear_obsolete(); // remove all cache entries which have the potentially_obsolete flag
    
  };
  
  
  class osg_rendercacheentry: public std::enable_shared_from_this<osg_rendercacheentry> {
  public:
    //std::weak_ptr<display_info> display; // (or should these be passed every time?)
    //std::weak_ptr<display_channel> displaychan;
    
    std::shared_ptr<recording_base> cached_recording; // shared because we don't really want to let the recording go away while we have a pointer to the data in the cache
    osg::ref_ptr<osg::Group> osg_group;

    double left;
    double right;
    double bottom;
    double top;

    bool potentially_obsolete; // set by mark_obsolete()
    
    //bool touched;  // this is a flag used by the update process to indicate this entry is still valid
    // (alternative approach: add it to a new map, then swap out the maps.)
    
    osg_rendercacheentry(std::shared_ptr<recording_base> cached_recording,double left,double right,double bottom,double top);
    osg_rendercacheentry & operator=(const osg_rendercacheentry &) = delete; // shouldn't need copy assignment
    osg_rendercacheentry(const osg_rendercacheentry &) = delete; // shouldn't need copy constructor
    virtual ~osg_rendercacheentry() = default; // subclassable
    
  };
  
  class osg_cachedimagedata: public osg_rendercacheentry {
  public:
    // inherited from osg_rendercacheentry:
    //std::weak_ptr<display_info> display; // (or should these be passed every time?)
    //std::weak_ptr<display_channel> displaychan;
    // std::shared_ptr<recording_base> cached_recording;
    // osg::ref_ptr<osg::Group> osg_group;    
    
    osg::ref_ptr<osg::Geode> imagegeode;
    osg::ref_ptr<osg::Geometry> imagegeom;
    osg::ref_ptr<osg::DrawArrays> imagetris;
    osg::ref_ptr<osg::Texture2D> imagetexture;
    osg::ref_ptr<osg::Image> image;
    osg::ref_ptr<osg::PixelBufferObject> imagepbo;
    
    osg::ref_ptr<osg::StateSet> imagestateset;

    osg_cachedimagedata(std::shared_ptr<recording_base> cached_recording,double left,double right,double bottom,double top);
    ~osg_cachedimagedata() = default;
    
    //void update(std::shared_ptr<recording_base> new_recording,size_t drawareawidth,size_t drawareaheight,size_t layer_index);
    
  };

  // need to define osg_cachedgeomdata

};

#endif // SNDE_OPENSCENEGRAPH_RENDERCACHE_HPP
