#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/BlendFunc>
#include <osg/Point>
#include <osgText/Font>
#include <osgText/Text>
#include <iostream>


#include "snde/snde_types.h"
#include "snde/quaternion.h"
#include "snde/openscenegraph_rendercache.hpp"
#include "snde/rec_display.hpp"
#include "snde/display_requirements.hpp"

namespace snde {
  
  // Lookups in the renderer registry are done per the indexes assigned by the registered recording display handlers defined in rec_display.cpp

  static int osg_registered_scalar = osg_register_renderer(rendermode(SNDE_SRM_SCALAR, typeid(multi_ndarray_recording_display_handler)), [](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>, bool>>* all_locks_required, const osg_renderparams& params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry> {
    return std::make_shared<osg_cachedscalar>(all_locks_required, params, display_req);
    });

  static int osg_registered_imagedata = osg_register_renderer(rendermode(SNDE_SRM_RGBAIMAGEDATA,typeid(multi_ndarray_recording_display_handler)),[](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry> {
      return std::make_shared<osg_cachedimagedata>(all_locks_required,params,display_req);
    });
  
  static int osg_registered_image = osg_register_renderer(rendermode(SNDE_SRM_RGBAIMAGE,typeid(multi_ndarray_recording_display_handler)),[](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedimage>(all_locks_required,params,display_req);
    });
  
  static int osg_registered_pointcloudvertices = osg_register_renderer(rendermode(SNDE_SRM_POINTCLOUDVERTICES,typeid(multi_ndarray_recording_display_handler)),[](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedpointcloudvertices>(all_locks_required,params,display_req);
    });

  static int osg_registered_pointcloudcolormap = osg_register_renderer(rendermode(SNDE_SRM_POINTCLOUDCOLORMAP,typeid(multi_ndarray_recording_display_handler)),[](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedpointcloudcolormap>(all_locks_required,params,display_req);
    });

  static int osg_registered_pointcloud = osg_register_renderer(rendermode(SNDE_SRM_POINTCLOUD,typeid(multi_ndarray_recording_display_handler)),[](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedpointcloud>(all_locks_required,params,display_req);
    });

  static int osg_registered_cachedcoloredtransparentlines = osg_register_renderer(rendermode(SNDE_SRM_COLOREDTRANSPARENTLINES,typeid(multi_ndarray_recording_display_handler)),[](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedcoloredtransparentlines>(all_locks_required,params,display_req);
    });

  static int osg_registered_cachedcoloredtransparentpoints = osg_register_renderer(rendermode(SNDE_SRM_COLOREDTRANSPARENTPOINTS, typeid(multi_ndarray_recording_display_handler)), [](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams& params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry> {
      return std::make_shared<osg_cachedcoloredtransparentpoints>(all_locks_required,params, display_req);
    });

  static int osg_registered_waveform = osg_register_renderer(rendermode(SNDE_SRM_WAVEFORM, typeid(multi_ndarray_recording_display_handler)), [](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams& params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry> {
      return std::make_shared<osg_cachedwaveform>(all_locks_required,params, display_req);
      });

  static int osg_registered_cachedphaseplaneendpointwithcoloredtransparentlines = osg_register_renderer(rendermode(SNDE_SRM_PHASE_PLANE_ENDPOINT_WITH_COLOREDTRANSPARENTLINES,typeid(multi_ndarray_recording_display_handler)),[](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedphaseplaneendpointwithcoloredtransparentlines>(all_locks_required,params,display_req);
    });


  
  static int osg_registered_meshednormals = osg_register_renderer(rendermode(SNDE_SRM_MESHEDNORMALS,typeid(meshed_part_recording_display_handler)),[](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedmeshednormals>(all_locks_required,params,display_req);
    });
  static int osg_registered_meshedvertexarray = osg_register_renderer(rendermode(SNDE_SRM_VERTEXARRAYS,typeid(meshed_part_recording_display_handler)),[](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedmeshedvertexarray>(all_locks_required,params,display_req);
    });


  static int osg_registered_parameterizationdata = osg_register_renderer(rendermode(SNDE_SRM_MESHED2DPARAMETERIZATION,typeid(meshed_parameterization_recording_display_handler)),[](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedparameterizationdata>(all_locks_required,params,display_req);
    });
  
  static int osg_registered_meshedpart = osg_register_renderer(rendermode(SNDE_SRM_MESHEDPARAMLESS3DPART,typeid(meshed_part_recording_display_handler)),[](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedmeshedpart>(all_locks_required,params,display_req);
    });

  static int osg_registered_texedmeshedgeom = osg_register_renderer(rendermode(SNDE_SRM_TEXEDMESHED3DGEOM,typeid(textured_part_recording_display_handler)),[](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedtexedmeshedgeom>(all_locks_required,params,display_req);
    });
  
  static int osg_registered_texedmeshedpart = osg_register_renderer(rendermode(SNDE_SRM_TEXEDMESHEDPART,typeid(textured_part_recording_display_handler)),[](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedtexedmeshedpart>(all_locks_required,params,display_req);
    });
  

  static int osg_registered_assembly = osg_register_renderer(rendermode(SNDE_SRM_ASSEMBLY,typeid(assembly_recording_display_handler)),[](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedassembly>(all_locks_required,params,display_req);
    });
  

  // register our cachedtransformedcomponent as accommodating the tracking_pose_recording_display_handler
  static int osg_registered_transformedcomponent = osg_register_renderer(rendermode(SNDE_SRM_TRANSFORMEDCOMPONENT,typeid(tracking_pose_recording_display_handler)),[](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedtransformedcomponent>(all_locks_required,params,display_req);
    });

  // register our cachedtransformedcomponent as accommodating the pose_channel_recording_display_handler
  static int osg_registered_transformedcomponent_posechannel = osg_register_renderer(rendermode(SNDE_SRM_TRANSFORMEDCOMPONENT,typeid(pose_channel_recording_display_handler)),[](std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedtransformedcomponent>(all_locks_required,params,display_req);
    });

  
  
  static std::shared_ptr<osg_renderer_map> *_osg_renderer_registry; // default-initialized to nullptr

  static std::mutex &osg_renderer_registry_mutex()
  {
    // take advantage of the fact that since C++11 initialization of function statics
    // happens on first execution and is guaranteed thread-safe. This lets us
    // work around the "static initialization order fiasco" using the
    // "construct on first use idiom".
    // We just use regular pointers, which are safe from the order fiasco,
    // but we need some way to bootstrap thread-safety, and this mutex
    // is it. 
    static std::mutex regmutex; 
    return regmutex; 
  }
  

  
  std::shared_ptr<osg_renderer_map> osg_renderer_registry()
  {
    std::mutex &regmutex = osg_renderer_registry_mutex();
    std::lock_guard<std::mutex> reglock(regmutex);

    if (!_osg_renderer_registry) {
      _osg_renderer_registry = new std::shared_ptr<osg_renderer_map>(std::make_shared<osg_renderer_map>());
    }
    return *_osg_renderer_registry;
  }


    

  int osg_register_renderer(rendermode mode,std::function<std::shared_ptr<osg_rendercacheentry>(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)> factory)
  {

    osg_renderer_registry(); // Ensure that the registry poiter exists
      
    std::mutex &regmutex = osg_renderer_registry_mutex();
    std::lock_guard<std::mutex> reglock(regmutex);
    
    // copy map and update then publish the copy
    std::shared_ptr<osg_renderer_map> new_map = std::make_shared<osg_renderer_map>(**_osg_renderer_registry);
    
    new_map->emplace(mode,factory);

    *_osg_renderer_registry = new_map;
    return 0;
  }
  


  
  static inline bool GetGeom(std::shared_ptr<recording_base> rec,size_t *ndim,
			     double *IniValX,double *StepSzX,snde_index *dimlenx,
			     double *IniValY,double *StepSzY,snde_index *dimleny,
			     double *IniValZ,double *StepSzZ,snde_index *dimlenz, /* Z optional */
			     double *IniValW,double *StepSzW,snde_index *dimlenw) /* W optional */
  {
    double Junk=0.0;
    snde_index Junk2=0;
    size_t junk3=0;
    std::shared_ptr<ndarray_recording_ref> datastore = rec->cast_to_multi_ndarray()->reference_ndarray();

    if (!ndim) ndim=&junk3;
    
    if (!IniValX) IniValX=&Junk;
    if (!StepSzX) StepSzX=&Junk;
    if (!dimlenx) dimlenx=&Junk2;

    if (!IniValY) IniValY=&Junk;
    if (!StepSzY) StepSzY=&Junk;
    if (!dimleny) dimleny=&Junk2;

    
    if (!IniValZ) IniValZ=&Junk;
    if (!StepSzZ) StepSzZ=&Junk;
    if (!dimlenz) dimlenz=&Junk2;
    
    if (!IniValW) IniValW=&Junk;
    if (!StepSzW) StepSzW=&Junk;
    if (!dimlenw) dimlenw=&Junk2;

    if (!datastore) {
      return false; // cast failed; return all zeros
    }
  
    *ndim=datastore->layout.dimlen.size();

    std::string IniValXUnits;
    std::string StepSzXUnits;
    
    std::tie(*IniValX,IniValXUnits)=datastore->rec->metadata->GetMetaDatumDblUnits("ande_array-axis0_offset",0.0,"pixels"); /* in units  */
    std::tie(*StepSzX,StepSzXUnits)=datastore->rec->metadata->GetMetaDatumDblUnits("ande_array-axis0_scale",1.0,"pixels");  /* in units/index */
    
    if (datastore->layout.dimlen.size() >= 1) {
      *dimlenx=datastore->layout.dimlen.at(0);
    } else {
      *dimlenx=1;
    }
    
    std::string IniValYUnits;
    std::string StepSzYUnits;
    
    std::tie(*IniValY,IniValYUnits)=datastore->rec->metadata->GetMetaDatumDblUnits("ande_array-axis1_offset",0.0,"pixels"); /* in units */
    std::tie(*StepSzY,IniValYUnits)=datastore->rec->metadata->GetMetaDatumDblUnits("ande_array-axis1_scale",1.0,"pixels"); /* in units/index */
    
    if (datastore->layout.dimlen.size() >= 2) {
      *dimleny=datastore->layout.dimlen.at(1);
    } else {
      *dimleny=1;
    }

    std::string IniValZUnits;
    std::string StepSzZUnits;

    std::tie(*IniValZ,IniValZUnits)=datastore->rec->metadata->GetMetaDatumDblUnits("ande_array-axis2_offset",0.0,"pixels"); /* in units */
    std::tie(*StepSzZ,IniValZUnits)=datastore->rec->metadata->GetMetaDatumDblUnits("ande_array-axis2_scale",1.0,"pixels"); /* in units/index */
    if (datastore->layout.dimlen.size() >= 3) {
      *dimlenz=datastore->layout.dimlen.at(2);
    } else {
      *dimlenz=1;
    }
    
    std::string IniValWUnits;
    std::string StepSzWUnits;
    
    std::tie(*IniValW,IniValWUnits)=datastore->rec->metadata->GetMetaDatumDblUnits("ande_array-axis3_offset",0.0,"pixels"); /* in units */
    std::tie(*StepSzW,StepSzWUnits)=datastore->rec->metadata->GetMetaDatumDblUnits("ande_array-axis3_scale",1.0,"pixels"); /* in units/index */
    if (datastore->layout.dimlen.size() >= 4) {
      *dimlenw=datastore->layout.dimlen.at(3);
    } else {
      *dimlenw=1;
    }
    
    return true;
}
  

  
  std::pair<std::shared_ptr<osg_rendercacheentry>,bool> osg_rendercache::GetEntry(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)   // mode from rendermode.hpp
  // how all_locks_required works: each cache entry adds its own locks
  // to the all_locks_required vector as it is traversed in the tree
  // of calls either to attempt_reuse() or recursive GetEntry().
  // each entry in all_locks_required consists of the ndarray_recording_ref
  // that needs to be locked, alongside a bool that represents whether a
  // write lock (true) or read lock (false) is required (i.e. almost always false)
  {
    const std::string &channel_path = display_req->channelpath;
    const rendermode_ext &mode = display_req->mode;

    // channel_path contents is often an rgba image transformed by the
    // colormapper, so the manual selection of how to view 
    // (what frame, etc.) is already included in such situations
    
    auto cache_it = cache.find(std::make_pair(channel_path,mode));
    
    if (cache_it != cache.end()) {
      bool reusable,modified;
      std::tie(reusable,modified) = cache_it->second->attempt_reuse(params,display_req,all_locks_required);
      if (reusable) {

	// don't need to modify all_locks_required because attempt_reuse is
	// required to add any locks to it.
	/*for (auto && required_lock: cache_it->second->locks_required) {
	  locks_required->push_back(required_lock);
	  }*/
	
	cache_it->second->clear_potentially_obsolete(); // not an obsolete entry
	return std::make_pair(cache_it->second,modified);	
      }      
      
    }


    // If we got here, then no suitable entry was found.
    // Create a new one

    std::shared_ptr<osg_renderer_map> reg = osg_renderer_registry();
    auto renderer_it = reg->find(mode.mode);
    if (renderer_it == reg->end()) {
      throw snde_error("Unable to find an OpenSceneGraph renderer for %s mode %s",channel_path.c_str(),mode.mode.str().c_str());
    }
    
    //std::shared_ptr<recording_base> new_recording = with_display_transforms->check_for_recording(channel_path);

    //if (!new_recording) {
    //  // recording not present (!)
    //  throw snde_error("Trying to display missing recording %s",channel_path);
    //}
    
    
    std::shared_ptr<osg_rendercacheentry> retval = renderer_it->second(all_locks_required,params,display_req);
    cache.erase(std::make_pair(channel_path,mode));
    cache.emplace(std::make_pair(channel_path,mode),retval);


    // constructor now required to add locks to all_locks_required
    //for (auto && required_lock: retval->locks_required) {
    //  locks_required->push_back(required_lock);
    //}
    
    return std::make_pair(retval,true);
    
    //std::shared_ptr<osg_cachedimage> imgentry = std::make_shared<osg_cachedimage>(new_recording,texture);
      
  }


  void osg_rendercache::mark_obsolete()
  {
    for (auto && chanpathmode_cacheentry: cache) {
      chanpathmode_cacheentry.second->potentially_obsolete=true;
    }
  }


  void osg_rendercache::erase_obsolete()
  {
    std::unordered_map<std::pair<std::string,rendermode_ext>,std::shared_ptr<osg_rendercacheentry>,chanpathmodeext_hash>::iterator cache_it, next_it;
    
    for (cache_it = cache.begin();cache_it != cache.end();cache_it = next_it) {
      next_it = cache_it;
      ++next_it;
      
      if (cache_it->second->potentially_obsolete) {
	// obsolete entry; remove it
	cache.erase(cache_it);
      }
    }
  }

  osg_rendercacheentry::osg_rendercacheentry(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required) :
    potentially_obsolete(false)
  {
    
  }

  void osg_rendercacheentry::accumulate_locks_required(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)
  {
    // accumulate all entries from this instance's locks_required element
    // into the all_locks_required vector.
    for (auto && required_lock: locks_required) {
      all_locks_required->push_back(required_lock);
    }
  }
  //bool osg_rendercacheentry::attempt_reuse(const osg_renderparams &params,const std::string &channel_path,const rendermode & mode)
  //{
  //  return false;
  //}

  void osg_rendercacheentry::clear_potentially_obsolete()
  {
    potentially_obsolete=false;
  }

  osg_rendercachegroupentry::osg_rendercachegroupentry(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required) :
    osg_rendercacheentry(all_locks_required)
  {
    
  }

  osg_rendercachedrawableentry::osg_rendercachedrawableentry(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required) :
    osg_rendercacheentry(all_locks_required)
  {
    
  }

  osg_rendercachetextureentry::osg_rendercachetextureentry(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required) :
    osg_rendercacheentry(all_locks_required)
  {

  }

  osg_rendercachearrayentry::osg_rendercachearrayentry(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required) :
    osg_rendercacheentry(all_locks_required)
  {
    
  }

  osg_rendercachearraysentry::osg_rendercachearraysentry(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required) :
    osg_rendercacheentry(all_locks_required)
  {

  }
      

  template<typename ... Args>
  std::string string_format(const std::string& format, Args ... args)
  {
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
    if (size_s <= 0) { throw std::runtime_error("Error during formatting."); }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
  }

  osg_cachedscalar::osg_cachedscalar(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>, bool>>* all_locks_required, const osg_renderparams& params, std::shared_ptr<display_requirement> display_req) :
    osg_rendercachegroupentry(all_locks_required)
  {
    cached_recording = std::dynamic_pointer_cast<multi_ndarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));

    if (!cached_recording) {
      throw snde_error("osg_cachedscalar: Could not get recording for %s", display_req->renderable_channelpath->c_str());
    }

    locks_required.push_back({ cached_recording->reference_ndarray(0) ,false }); // accmulate locks needed for lockmanager::lock_recording_refs()


    osg_group = new osg::Group;
    osg::ref_ptr<osgText::Font> font = osgText::readRefFontFile("fonts/arial.ttf");
    osg::ref_ptr<osg::Geode> geode = new osg::Geode;
    osg_group->addChild(geode);
    osg::ref_ptr<osgText::Text> text = new osgText::Text;
    text->setFont(font);
    std::shared_ptr<scalar_params> renderparams = std::dynamic_pointer_cast<scalar_params>(display_req->mode.constraint);
    text->setColor({ (float)renderparams->color.R, (float)renderparams->color.G, (float)renderparams->color.B, 1.0f });
    text->setFontResolution(60, 60);
    text->setCharacterSize(renderparams->scale);
    text->setPosition({ /*(float)(params.left)*/0.0f, /*(float)(params.top)*/0.0f, 0.0f});
    text->setLayout(osgText::Text::LEFT_TO_RIGHT);
    std::string displaytext = *display_req->renderable_channelpath + " = ";
    switch (cached_recording->ndinfo(0)->typenum) {
    case SNDE_RTN_COMPLEXFLOAT16:
    case SNDE_RTN_COMPLEXFLOAT32:
    case SNDE_RTN_COMPLEXFLOAT64:
      displaytext += string_format("%.3f + j%.3f", cached_recording->element_complexfloat64(0, 0).real, cached_recording->element_complexfloat64(0, 0).imag);
      break;
    case SNDE_RTN_FLOAT16:
    case SNDE_RTN_FLOAT32:
    case SNDE_RTN_FLOAT64:
      displaytext += string_format("%.3f", cached_recording->element_double(0,0));
      break;
    case SNDE_RTN_UINT8:
    case SNDE_RTN_UINT16:
    case SNDE_RTN_UINT32:
    case SNDE_RTN_UINT64:
      displaytext += string_format("%d", cached_recording->element_unsigned(0,0));
      break;
    case SNDE_RTN_INT8:
    case SNDE_RTN_INT16:
    case SNDE_RTN_INT32:
    case SNDE_RTN_INT64:
      displaytext += string_format("%d", cached_recording->element_int(0,0));
      break;
    case SNDE_RTN_STRING:
      displaytext += static_cast<std::string*>(cached_recording->void_shifted_arrayptr(0))->c_str();
      break;
    default:
      displaytext += "ERROR";
      break;
    }

    displaytext += " ";
    displaytext += cached_recording->metadata->GetMetaDatumStr("ande_array-ampl_units", "");

    text->setText(displaytext);

    geode->addDrawable(text);

    accumulate_locks_required(all_locks_required);
  }


  std::pair<bool, bool> osg_cachedscalar::attempt_reuse(const osg_renderparams& params, std::shared_ptr<display_requirement> display_req, std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>, bool>>* all_locks_required)
  {
    std::shared_ptr<multi_ndarray_recording> new_recording = std::dynamic_pointer_cast<multi_ndarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedscalar::attempt_reuse: Could not get recording for %s", display_req->renderable_channelpath->c_str());
    }

    if (new_recording == cached_recording && new_recording->info->immutable) {
      accumulate_locks_required(all_locks_required);
      std::make_pair(true, false);
    }

    return std::make_pair(false, false); // (reusable,modified)

  }
  

  
  osg_cachedimagedata::osg_cachedimagedata(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachetextureentry(all_locks_required)
  {
    size_t ndim;
    double IniValX,IniValY,IniValZ,IniValW;
    double StepX,StepY,StepZ,StepW;
    snde_index dimlenz,dimlenw; // Note: dimlenx, dimleny are class members

    cached_recording = params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath);
    if (!cached_recording) {
      throw snde_error("osg_cachedimagedata: Could not get recording for %s",display_req->renderable_channelpath->c_str()); 
      
    }

    //snde_warning("osg_cachedimagedata on %s rev %llu rec=0x%llx this=0x%llx",display_req->renderable_channelpath->c_str(),(unsigned long long)cached_recording->info->revision,(unsigned long long)cached_recording.get(),(unsigned long long)this);
    
    if (!GetGeom(cached_recording,&ndim,
		 &IniValX,&StepX,&dimlenx,
		 &IniValY,&StepY,&dimleny,
		 &IniValZ,&StepZ,&dimlenz,
		 &IniValW,&StepW,&dimlenw)) {
      throw snde_error("osg_cachedimagedata: Could not get geometry for %s",display_req->renderable_channelpath->c_str()); 
    }

    
    osg::ref_ptr<osg::Texture2D> imagetexture=new osg::Texture2D();
    osg_texture = imagetexture;
    
    image=new osg::Image();
    //imagepbo=new osg::PixelBufferObject();
    

    // Set up scene graph
    //imagepbo->setImage(image);
    //image->setPixelBufferObject(imagepbo);
    imagetexture->setResizeNonPowerOfTwoHint(false);
    
    if (cached_recording->info->immutable) {
      image->setDataVariance(osg::Object::STATIC); 

    } else {
      image->setDataVariance(osg::Object::DYNAMIC); 
    }


    std::shared_ptr<multi_ndarray_recording> cached_ndarray_rec = cached_recording->cast_to_multi_ndarray();

    if (!cached_ndarray_rec->layouts.at(0).is_f_contiguous()) {
      throw snde_error("Error displaying image from channel %s as texture: Textures must be indexed fortran-order",cached_ndarray_rec->info->name);
    }
    
    locks_required.push_back({cached_ndarray_rec->reference_ndarray(),false});
    image->setImage(dimlenx,dimleny,1,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,(unsigned char *)cached_ndarray_rec->void_shifted_arrayptr(0),osg::Image::AllocationMode::NO_DELETE);
    imagetexture->setInternalFormat(GL_RGBA);
    imagetexture->setInternalFormatMode(osg::Texture::USE_IMAGE_DATA_FORMAT);
    imagetexture->setImage(image);    



    // Implement proper scaling of points in parameterization
    // space


    
    // TexMat should transform lower-left image corner
    // (at 0.5,0.5 pixel offset) to (0,0) in texture coordinates
    // and upper-right image corner (at 0.5,0,5 pixe offset)
    // to (1,1) in texture coordinates


  /* Should texture_transform be part of instancecache? 
   * geometry->parameterization->image does not need startcorner and
   * step, but perhaps could use definition of the desired corners 
   * of the parameterization space
   *  
   * The texture transform converts meaningful units from 
   * parameterization coordinates  to the range 0:1 for rendering.
   * The channel with parameterization data (texture, i.e. this)
   * provides a rectangular block representing a portion or 
   * superset of the parameterization space. 
   *  i.e.
   * posn_within_image <- 0...1 <- texture_transform <- meaningful coordinate
   * So the texture transform is really dependent on both the coordinate 
   * interpretation for the uv coordinates AND the coordinate interpretation
   * for the texture image. 
   * 
   * Equations (for positive Step1): 
   *   Meaningful U coordinate of IniVal1-0.5*Step1 should map to 0.0
   *   Meaningful U coordinate of IniVal1+(Dimlenx-1+0.5)*Step1 should map to 1.0
   * Equations (for negative Step1): 
   *   Meaningful U coordinate of IniVal1+(Dimlenx-1+0.5)*Step1 should map to 0.0
   *   Meaningful U coordinate of IniVal1+(-0.5)*Step1 should map to 1.0

   * So the transform is strictly defined by the positioning and size of 
   * the parameterization channel.
   * Therefore it should be kept here, in the texture cache 
   * (Positive Step1):
   * The TexMat scaling will be 1.0/(Step1*Dimlenx) and the offset will be:
   *      *      scaling*(IniVal1 - 0.5*Step1) + offset = 0.0
   *      *       offset = -scaling*(IniVal1-0.5*Step1)
   * (Negative Step1):
   * The TexMat scaling will be -1.0/(Step1*Dimlenx) and the offset will be:
   *      *      scaling*(IniVal1 + (dimlenx-1+0.5)*Step1) + offset = 0.0
   *      *       offset = -scaling*(IniVal1+ (dimlenx-1+0.5)*Step1)



   */

    
    double Xoffset,Yoffset;
    
    if (StepX > 0.0) {
      //Xoffset = -IniValX/(fabs(StepX)*DimLenX) + StepX/(fabs(StepX)*DimLenX)/2.0; 
      Xoffset = -IniValX/(StepX*dimlenx) + 1.0/(2.0*dimlenx); 
    } else {
      //Xoffset = -IniValX/(fabs(StepX)*DimLenX) - StepX*DimLenX/(fabs(StepX)*DimLenX) + StepX/(fabs(StepX)*DimLenX)/2.0;
      // remember StepX negative
      //Xoffset = IniValX/(StepX*dimlenx) + 1.0 - 1.0/(2.0*dimlenx);
      //Xoffset = -IniValX/(StepX*DimLenX) - StepX*DimLenX/(StepX*DimLenX) + StepX/(StepX*DimLenX)/2.0;
      //Xoffset = -IniValX/(StepX*dimlenx) - 1.0 + 1.0/(dimlenx*2.0);
      Xoffset = -IniValX/(StepX*dimlenx) + 1.0/(dimlenx*2.0);
    }

    if (StepY > 0.0) {
      //YOffset = -IniValY/(fabs(StepY)*DimLenY) + StepY/(fabs(StepY)*DimLenY)/2.0; 
      Yoffset = -IniValY/(StepY*dimleny) + 1.0/(dimleny*2.0); 
      
    } else {
      //YOffset = -IniValY/(fabs(StepY)*DimLenY) - StepY*DimLenY/(fabs(StepY)*DimLenY) + Step2/(2.0*fabs(StepY)*DimLenY);
      //Yoffset = IniValY/(StepY*dimleny) + 1.0 - 1.0/(2.0*dimleny);
      //YOffset = -IniValY/(StepY*DimLenY) - StepY*DimLenY/(StepY*DimLenY) + Step2/(2.0*StepY*DimLenY);
      //Yoffset = -IniValY/(StepY*dimleny) - 1.0 + 1.0/(2.0*dimleny);
      Yoffset = -IniValY/(StepY*dimleny) + 1.0/(2.0*dimleny);
      }
    /*
    double ScalingX = 1.0/(StepX*dimlenx);
    double ScalingY = 1.0/(StepY*dimleny);

    if (StepX > 0.0) {
      Xoffset = -ScalingX*(IniValX-0.5*StepX);
    } else {
      Xoffset = ScalingX*(IniValX +(dimlenx-1+0.5)*StepX);
    }

    if (StepY > 0.0) {
      Yoffset = -ScalingY*(IniValY-0.5*StepY);
    } else {
      Yoffset = ScalingY*(IniValY +(dimleny-1+0.5)*StepY);
    }
*/
#ifdef SNDE_DOUBLEPREC_COORDS
#define TEXMATMATRIX osg::Matrixd
#else
#define TEXMATMATRIX osg::Matrixf
#endif
    //fprintf(stderr,"StepY=%f\n",StepY);
    texture_transform = new osg::TexMat(TEXMATMATRIX{
	(snde_coord)(1.0/(StepX*dimlenx)),0.0,0.0,0.0,
	0.0,(snde_coord)(1.0/((StepY)*dimleny)),0.0,0.0,
	0.0,0.0,1.0,0.0,
	(snde_coord)Xoffset,(snde_coord)Yoffset,0.0,1.0, 
	});;
    
    
    /*
    unsigned char *arrayptr = (unsigned char *)cached_recording->cast_to_multi_ndarray()->void_shifted_arrayptr(0);
    snde_index i,j;
    for (j=0;j < dimleny;j++) {
      for (i=0;i < dimlenx;i++) {
	
	printf("%d ",arrayptr[4*(i+dimlenx*j)]);
      }
      printf("\n");
    }
    fflush(stdout);
    */
    accumulate_locks_required(all_locks_required);
  }
  
  std::pair<bool,bool> osg_cachedimagedata::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)
  {
    // only reuse if the recording pointer is the same; everything else here is
    // trivial enough it's pointless to try to reuse.
    
    std::shared_ptr<recording_base> new_recording = params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath);
    
    if (!new_recording) {
      throw snde_error("osg_cachedimagedata::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    bool reusable = (new_recording == cached_recording && new_recording->info->immutable); // ***!!! For mutable recordings if we wanted we could verify that the pointer remains the same and just mark the array as dirty in OSG

    if (reusable) {
      accumulate_locks_required(all_locks_required);
    }
    
    return std::make_pair(reusable,false); // second element is modified: if it is resuable, it is not modified.
    
  }



  osg_cachedimage::osg_cachedimage(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params,std::shared_ptr<display_requirement> display_req):
    osg_rendercachegroupentry(all_locks_required)
  {
    size_t ndim;
    double IniValX,IniValY,IniValZ,IniValW;
    double StepSzX,StepSzY,StepSzZ,StepSzW;
    snde_index dimlenx,dimleny,dimlenz,dimlenw;


    cached_recording = params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath);
    if (!cached_recording) {
      throw snde_error("osg_cachedimage: Could not get recording for %s",display_req->renderable_channelpath->c_str()); 
      
    }

    if (!GetGeom(cached_recording,&ndim, // doesn't count as a parameter because dependent solely on the underlying recording
		 &IniValX,&StepSzX,&dimlenx,
		 &IniValY,&StepSzY,&dimleny,
		 &IniValZ,&StepSzZ,&dimlenz,
		 &IniValW,&StepSzW,&dimlenw)) {
      // cast failed; return empty group
      throw snde_error("osg_cachedimage: Could not get geometry for %s",display_req->renderable_channelpath->c_str()); 
    }
    
    // Get texture correpsonding to this same channel
    bool modified;
    std::shared_ptr<osg_rendercacheentry> raw_entry;

    std::shared_ptr<display_requirement> subreq = display_req->sub_requirements.at(0);

    if (!subreq) {
        throw snde_error("osg_cachedimage: Unable to get subrequirement for %s", display_req->renderable_channelpath->c_str());
    }

    std::tie(raw_entry,modified) = params.rendercache->GetEntry(params,subreq,all_locks_required);

    texture = std::dynamic_pointer_cast<osg_rendercachetextureentry>(raw_entry);

    if (!texture) {
      throw snde_error("osg_cachedimage: Unable to get texture cache entry for %s",display_req->sub_requirements.at(0)->renderable_channelpath->c_str());
    }

    //snde_warning("osg_cachedimage on %s rev %llu 0x%llx data 0x%llx",display_req->renderable_channelpath->c_str(),(unsigned long long)cached_recording->info->revision,(unsigned long long)cached_recording.get(),(unsigned long long)texture.get());

    osg::ref_ptr<osg::Texture> imagetexture = texture->osg_texture;
    
    
    //transform=new osg::MatrixTransform();
    imagegeode=new osg::Geode();
    imagegeom=new osg::Geometry();
    imagetris=new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,0); // # is number of triangles * number of coordinates per triangle
    imagestateset=nullptr;
    

    // Set up scene graph
    osg_group = new osg::Group();
    osg_group->addChild(imagegeode);
    imagegeom->setUseVertexBufferObjects(true);
    // At least on Linux/Intel graphics we get nasty messages
    // from the driver if we dont set the VBO in DYNAMIC_DRAW mode
    imagegeom->getOrCreateVertexBufferObject()->setUsage(GL_DYNAMIC_DRAW);
    imagegeom->addPrimitiveSet(imagetris);
    imagestateset=imagegeode->getOrCreateStateSet();
    imagestateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    imagestateset->setTextureAttributeAndModes(0,imagetexture,osg::StateAttribute::ON);
    imagestateset->setTextureAttributeAndModes(0,texture->texture_transform,osg::StateAttribute::ON);
    
    osg::ref_ptr<osg::Vec4Array> ColorArray=new osg::Vec4Array();
    ColorArray->push_back(osg::Vec4(1.0,1.0,1.0,1.0)); // Setting the first 3 to less than 1.0 will dim the output. Setting the last one would probably add alpha transparency (?)
    imagegeom->setColorArray(ColorArray,osg::Array::BIND_OVERALL);
    imagegeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    
    imagegeom->setStateSet(imagestateset); // I think this is redundant because state should be inherited from the geode
    imagegeode->addDrawable(imagegeom);


    // Image coordinates, from actual corners, counterclockwise,
    // Two triangles    
    osg::ref_ptr<osg::Vec3dArray> ImageCoords=new osg::Vec3dArray(6);
    osg::ref_ptr<osg::Vec2dArray> ImageTexCoords=new osg::Vec2dArray(6);

    
    if ((StepSzX >= 0 && StepSzY >= 0) || (StepSzX < 0 && StepSzY < 0)) {
      snde_debug(SNDE_DC_RENDERING,"Channel %s image regular",display_req->renderable_channelpath->c_str());
      // lower-left triangle (if both StepSzX and StepSzY positive)
      (*ImageCoords)[0]=osg::Vec3d(IniValX-0.5*StepSzX,
				   IniValY-0.5*StepSzY,
				   0.0);
      (*ImageCoords)[1]=osg::Vec3d(IniValX+dimlenx*StepSzX-0.5*StepSzX,
				   IniValY-0.5*StepSzY,
				   0.0);
      (*ImageCoords)[2]=osg::Vec3d(IniValX-0.5*StepSzX,
				   IniValY+dimleny*StepSzY-0.5*StepSzY,
				   0.0);

      /*
      (*ImageTexCoords)[0]=osg::Vec2d(0,0);
      (*ImageTexCoords)[1]=osg::Vec2d(1,0);
      (*ImageTexCoords)[2]=osg::Vec2d(0,1);*/
      (*ImageTexCoords)[0]=osg::Vec2d(IniValX-0.5*StepSzX,
				      IniValY-0.5*StepSzY);
      (*ImageTexCoords)[1]=osg::Vec2d(IniValX+dimlenx*StepSzX-0.5*StepSzX,
				      IniValY-0.5*StepSzY);
      (*ImageTexCoords)[2]=osg::Vec2d(IniValX-0.5*StepSzX,
				      IniValY+dimleny*StepSzY-0.5*StepSzY);
      
      
      // upper-right triangle (if both StepSzX and StepSzY positive)
      (*ImageCoords)[3]=osg::Vec3d(IniValX+dimlenx*StepSzX-0.5*StepSzX,
				   IniValY+dimleny*StepSzY-0.5*StepSzY,
				   0.0);
      (*ImageCoords)[4]=osg::Vec3d(IniValX-0.5*StepSzX,
				   IniValY+dimleny*StepSzY-0.5*StepSzY,
				   0.0);
      (*ImageCoords)[5]=osg::Vec3d(IniValX+dimlenx*StepSzX-0.5*StepSzX,
				   IniValY-0.5*StepSzY,
				   0.0);
      /*
      (*ImageTexCoords)[3]=osg::Vec2d(1,1);
      (*ImageTexCoords)[4]=osg::Vec2d(0,1);
      (*ImageTexCoords)[5]=osg::Vec2d(1,0);
      */
      (*ImageTexCoords)[3]=osg::Vec2d(IniValX+dimlenx*StepSzX-0.5*StepSzX,
				      IniValY+dimleny*StepSzY-0.5*StepSzY);
      (*ImageTexCoords)[4]=osg::Vec2d(IniValX-0.5*StepSzX,
				      IniValY+dimleny*StepSzY-0.5*StepSzY);
      (*ImageTexCoords)[5]=osg::Vec2d(IniValX+dimlenx*StepSzX-0.5*StepSzX,
				      IniValY-0.5*StepSzY);
      
    } else {
      snde_debug(SNDE_DC_RENDERING,"Channel %s image flipped",display_req->renderable_channelpath->c_str());
      // One of StepSzX or StepSzY is positive, one is negative
      // work as raster coordinates (StepSzY negative)
      // lower-left triangle
      
      (*ImageCoords)[0]=osg::Vec3d(IniValX-0.5*StepSzX,
				   IniValY+dimleny*StepSzY-0.5*StepSzY,
				   0.0);
      (*ImageCoords)[1]=osg::Vec3d(IniValX+dimlenx*StepSzX-0.5*StepSzX,
				   IniValY+dimleny*StepSzY-0.5*StepSzY,
				   0.0);
      (*ImageCoords)[2]=osg::Vec3d(IniValX-0.5*StepSzX,
				   IniValY-0.5*StepSzY,
				   0.0);
      /*
      (*ImageTexCoords)[0]=osg::Vec2d(0,1);
      (*ImageTexCoords)[1]=osg::Vec2d(1,1);
      (*ImageTexCoords)[2]=osg::Vec2d(0,0);
      */
      (*ImageTexCoords)[0]=osg::Vec2d(IniValX-0.5*StepSzX,
				      IniValY+dimleny*StepSzY-0.5*StepSzY);
      (*ImageTexCoords)[1]=osg::Vec2d(IniValX+dimlenx*StepSzX-0.5*StepSzX,
				      IniValY+dimleny*StepSzY-0.5*StepSzY);
      (*ImageTexCoords)[2]=osg::Vec2d(IniValX-0.5*StepSzX,
				      IniValY-0.5*StepSzY);
      // upper-right triangle 
      (*ImageCoords)[3]=osg::Vec3d(IniValX+dimlenx*StepSzX-0.5*StepSzX,
				   IniValY-0.5*StepSzY,
				   0.0);
      (*ImageCoords)[4]=osg::Vec3d(IniValX-0.5*StepSzX,
				   IniValY-0.5*StepSzY,
				   0.0);
      (*ImageCoords)[5]=osg::Vec3d(IniValX+dimlenx*StepSzX-0.5*StepSzX,
				   IniValY+dimleny*StepSzY-0.5*StepSzY,
				   0.0);
      /*
      (*ImageTexCoords)[3]=osg::Vec2d(1,0);
      (*ImageTexCoords)[4]=osg::Vec2d(0,0);
      (*ImageTexCoords)[5]=osg::Vec2d(1,1);
      */
      (*ImageTexCoords)[3]=osg::Vec2d(IniValX+dimlenx*StepSzX-0.5*StepSzX,
				      IniValY-0.5*StepSzY);
      (*ImageTexCoords)[4]=osg::Vec2d(IniValX-0.5*StepSzX,
				      IniValY-0.5*StepSzY);
      (*ImageTexCoords)[5]=osg::Vec2d(IniValX+dimlenx*StepSzX-0.5*StepSzX,
				      IniValY+dimleny*StepSzY-0.5*StepSzY);
    }
    

    imagegeom->setVertexArray(ImageCoords);
    imagegeom->setTexCoordArray(0,ImageTexCoords);
    imagetris->setCount(6);
    
    accumulate_locks_required(all_locks_required);
    
  }

  void osg_cachedimage::clear_potentially_obsolete()
  {
    potentially_obsolete=false;

    // also clear potentially_obsolete flag of our texture
    texture->clear_potentially_obsolete();
  }


  std::pair<bool,bool> osg_cachedimage::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)
  {
    // only reuse if the recording pointer is the same; everything else here is
    // trivial enough it's pointless to try to reuse.
    
    std::shared_ptr<recording_base> new_recording = params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath);
    
    if (!new_recording) {
      throw snde_error("osg_cachedimage::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    std::shared_ptr<display_requirement> subreq = display_req->sub_requirements.at(0);

    if (!subreq) {
        throw snde_error("osg_cachedimage: Unable to get subrequirement for %s", display_req->renderable_channelpath->c_str());
    }

    bool texture_reusable, texture_modified;
    

    std::tie(texture_reusable,texture_modified) = texture->attempt_reuse(params,subreq,all_locks_required);
    
    if (texture_reusable && new_recording == cached_recording && new_recording->info->immutable) {
      accumulate_locks_required(all_locks_required);
      return std::make_pair(true,texture_modified); // reusable, modified
    }

    return std::make_pair(false,false); // (reusable,modified)
  }


  osg_cachedpointcloudvertices::osg_cachedpointcloudvertices(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachearrayentry(all_locks_required)
  {
    cached_recording = std::dynamic_pointer_cast<multi_ndarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));

    
    if (!cached_recording) {
      throw snde_error("osg_cachedpointcloudvertices: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }
    std::shared_ptr<ndarray_recording_ref> point_cloud_vertices_array = cached_recording->reference_ndarray();
    
    locks_required.push_back( { point_cloud_vertices_array,false } ); // accmulate locks needed for lockmanager::lock_recording_refs()
    osg_array = new OSGFPArray(point_cloud_vertices_array,3,3); // 3 for 3d point coordinates
    accumulate_locks_required(all_locks_required);
  }


  std::pair<bool,bool> osg_cachedpointcloudvertices::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)
  {
    std::shared_ptr<multi_ndarray_recording> new_recording = std::dynamic_pointer_cast<multi_ndarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedpointcloudvertices::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }
    
    if (new_recording==cached_recording && new_recording->info->immutable) {
      accumulate_locks_required(all_locks_required);
      return std::make_pair(true,false);
    }
    
    return std::make_pair(false,false); // (reusable,modified)
  }



  osg_cachedpointcloudcolormap::osg_cachedpointcloudcolormap(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachearrayentry(all_locks_required)
  {
    cached_recording = std::dynamic_pointer_cast<multi_ndarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));

    
    if (!cached_recording) {
      throw snde_error("osg_cachedpointcloudcolormap: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    std::shared_ptr<ndarray_recording_ref> point_cloud_colormap_array = cached_recording->reference_ndarray();

    locks_required.push_back( { point_cloud_colormap_array,false } ); // accmulate locks needed for lockmanager::lock_recording_refs()
    osg_array = new OSGFPArray(point_cloud_colormap_array,1,4); // 4 for RGB&A components
    osg_array->setBinding(osg::Array::BIND_PER_VERTEX);

    accumulate_locks_required(all_locks_required);
  }


  std::pair<bool,bool> osg_cachedpointcloudcolormap::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)
  {
    std::shared_ptr<multi_ndarray_recording> new_recording = std::dynamic_pointer_cast<multi_ndarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedpointcloudcolormap::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    if (new_recording==cached_recording && new_recording->info->immutable){
      accumulate_locks_required(all_locks_required);
      return std::make_pair(true,false);
    }
    
    return std::make_pair(false,false); // (reusable,modified)
  }




  osg_cachedpointcloud::osg_cachedpointcloud(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachegroupentry(all_locks_required)
  {
    cached_recording = std::dynamic_pointer_cast<multi_ndarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    
    
    if (!cached_recording) {
      throw snde_error("osg_cachedpointcloud: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    bool modified;

    // get sub-requirement #0: SNDE_SRM_POINTCLOUDCOLORMAP
    std::shared_ptr<osg_rendercacheentry> raw_entry;
    std::tie(raw_entry,modified) = params.rendercache->GetEntry(params,display_req->sub_requirements.at(0),all_locks_required);
    
    // std::shared_ptr<osg_cachedpointcloudcolormap> colormap; (included in class definition)
    colormap = std::dynamic_pointer_cast<osg_cachedpointcloudcolormap>(raw_entry);
    if (!colormap) {
      throw snde_error("osg_cachedpointcloud: Unable to get colormap cache entry for %s",display_req->sub_requirements.at(0)->renderable_channelpath->c_str());
    }

    
    
    // get sub-requirement #1: SNDE_SRM_POINTCLOUDVERTICES
    std::tie(raw_entry,modified) = params.rendercache->GetEntry(params,display_req->sub_requirements.at(1),all_locks_required);
    //std::shared_ptr<osg_cachedpointcloudvertices> vertices; (included in class definition)
    vertices = std::dynamic_pointer_cast<osg_cachedpointcloudvertices>(raw_entry);
    if (!vertices) {
      throw snde_error("osg_cachedpointcloud: Unable to get vertex cache entry for %s",display_req->sub_requirements.at(0)->renderable_channelpath->c_str());
    }
    
    pc_geode = new osg::Geode();
    pc_geom = new osg::Geometry();
    pc_points = new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,0);

    snde_index numpoints=1;
    std::shared_ptr<ndarray_recording_ref> ref=cached_recording->reference_ndarray();
    for (size_t dimnum=0;dimnum < ref->layout.dimlen.size();dimnum++) {
      numpoints *= ref->layout.dimlen.at(dimnum);
    }

    /*
    //snde_warning("pc_points: %d",numpoints);

    snde_float32 min=1e9;
    snde_float32 max=-1e9;
    for (size_t ycnt=0;ycnt < ref->layout.dimlen.at(1);ycnt++) {
      for (size_t xcnt=0;xcnt < ref->layout.dimlen.at(0);xcnt++) {
        snde_coord3 val = ((snde_coord3*)ref->void_shifted_arrayptr())[ref->element_offset(xcnt,ycnt)];
	for (int ax=0;ax < 3;ax++) {
	  snde_float32 num = val.coord[ax];
	  if (num < min) min=num;
	  if (num > max) max=num;
	}
      }
    }

    snde_warning("pc_points: %d; min=%f max=%f",numpoints,min,max);
    */

    
    pc_points->setCount(numpoints);
    
    osg_group = new osg::Group();
    osg_group->addChild(pc_geode);
    pc_geom->setUseVertexBufferObjects(true);
    // At least on Linux/Intel graphics we get nasty messages
    // from the driver if we dont set the VBO in DYNAMIC_DRAW mode
    pc_geom->getOrCreateVertexBufferObject()->setUsage(GL_DYNAMIC_DRAW);
    pc_geom->addPrimitiveSet(pc_points);
    
    pc_stateset = pc_geode->getOrCreateStateSet();
    pc_stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    
    //pc_geom->setColorArray(testarray,osg::Array::BIND_PER_VERTEX);
    pc_geom->setColorArray(colormap->osg_array,osg::Array::BIND_PER_VERTEX);
    pc_geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    pc_geom->setStateSet(pc_stateset); // probably redundant 
    pc_geom->setVertexArray(vertices->osg_array);
    pc_geode->addDrawable(pc_geom);
    
    /*
osg::BoundingBox bbox = pc_geom->getBoundingBox();

    snde_warning("bbox minx=%f miny=%f minz=%f bbox maxx=%f bbox maxy=%f bbox maxz=%f",bbox._min.x(),bbox._min.y(),bbox._min.z(),bbox._max.x(),bbox._max.y(),bbox._max.z());
    */

    accumulate_locks_required(all_locks_required);
  }

  void osg_cachedpointcloud::clear_potentially_obsolete()
  {
    potentially_obsolete=false;

    // also clear potentially_obsolete flag of our colormap and vertices
    colormap->clear_potentially_obsolete();
    vertices->clear_potentially_obsolete();
  }



  std::pair<bool,bool> osg_cachedpointcloud::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)
  {
    std::shared_ptr<multi_ndarray_recording> new_recording = std::dynamic_pointer_cast<multi_ndarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedpointcloudcolormap::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    bool colormap_reusable,colormap_modified,vertices_reusable,vertices_modified;
    
    std::tie(colormap_reusable,colormap_modified) = colormap->attempt_reuse(params,display_req->sub_requirements.at(0),all_locks_required);
    std::tie(vertices_reusable,vertices_modified) = vertices->attempt_reuse(params,display_req->sub_requirements.at(1),all_locks_required);

    if (colormap_reusable && vertices_reusable && new_recording==cached_recording && new_recording->info->immutable) {
      return std::make_pair(true,colormap_modified || vertices_modified);
    }
    
    return std::make_pair(false,false); // (reusable,modified)
  }



  osg_cachedcoloredtransparentlines::osg_cachedcoloredtransparentlines(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachearraysentry(all_locks_required)
  {
    cached_recording = std::dynamic_pointer_cast<multi_ndarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));

    if (!cached_recording) {
      throw snde_error("osg_cachedcoloredtransparentlines: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }
    
    std::shared_ptr<ndarray_recording_ref> vertcoord_array = cached_recording->reference_ndarray("vertcoord");
    std::shared_ptr<ndarray_recording_ref> vertcoordcolor_array = cached_recording->reference_ndarray("vertcoord_color");
    
    locks_required.push_back( { vertcoord_array,false } ); // accmulate locks needed for lockmanager::lock_recording_refs()
    locks_required.push_back( { vertcoordcolor_array,false } ); // accmulate locks needed for lockmanager::lock_recording_refs()

    vertcoord_osg_array = new OSGFPArray(vertcoord_array,3,3); // 3 for 3d point coordinates    
    vertcoordcolor_osg_array = new OSGFPArray(vertcoordcolor_array,1,4); // 4 for RGBA entries    

    osg_arrays.push_back(vertcoord_osg_array);
    osg_arrays.push_back(vertcoordcolor_osg_array);

    accumulate_locks_required(all_locks_required);
  }


  std::pair<bool,bool> osg_cachedcoloredtransparentlines::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)
  {
    std::shared_ptr<multi_ndarray_recording> new_recording = std::dynamic_pointer_cast<multi_ndarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedcoloredtransparentlines::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    if (new_recording==cached_recording && new_recording->info->immutable) {
      accumulate_locks_required(all_locks_required);
      std::make_pair(true,false);
    }

    return std::make_pair(false,false); // (reusable,modified)
    
  }


  osg_cachedcoloredtransparentpoints::osg_cachedcoloredtransparentpoints(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams& params, std::shared_ptr<display_requirement> display_req) :
    osg_rendercachearraysentry(all_locks_required)
  {
    cached_recording = std::dynamic_pointer_cast<multi_ndarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));

    if (!cached_recording) {
      throw snde_error("osg_cachedcoloredtransparentpoints: Could not get recording for %s", display_req->renderable_channelpath->c_str());
    }

    std::shared_ptr<ndarray_recording_ref> pointcoord_array = cached_recording->reference_ndarray("pointcoord");
    std::shared_ptr<ndarray_recording_ref> pointcoordcolor_array = cached_recording->reference_ndarray("pointcoord_color");

    locks_required.push_back({ pointcoord_array,false }); // accmulate locks needed for lockmanager::lock_recording_refs()
    locks_required.push_back({ pointcoordcolor_array,false }); // accmulate locks needed for lockmanager::lock_recording_refs()

    pointcoord_osg_array = new OSGFPArray(pointcoord_array, 3, 3); // 3 for 3d point coordinates    
    pointcoordcolor_osg_array = new OSGFPArray(pointcoordcolor_array, 1, 4); // 4 for RGBA entries    

    osg_arrays.push_back(pointcoord_osg_array);
    osg_arrays.push_back(pointcoordcolor_osg_array);

    accumulate_locks_required(all_locks_required);
  }


  std::pair<bool, bool> osg_cachedcoloredtransparentpoints::attempt_reuse(const osg_renderparams& params, std::shared_ptr<display_requirement> display_req,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)
  {
    std::shared_ptr<multi_ndarray_recording> new_recording = std::dynamic_pointer_cast<multi_ndarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedcoloredtransparentpoints::attempt_reuse: Could not get recording for %s", display_req->renderable_channelpath->c_str());
    }

    if (new_recording == cached_recording && new_recording->info->immutable) {
      accumulate_locks_required(all_locks_required);
      return std::make_pair(true,false);
    }

    return std::make_pair(false, false); // (reusable,modified)

  }




  osg_cachedwaveform::osg_cachedwaveform(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams& params, std::shared_ptr<display_requirement> display_req) :
    osg_rendercachegroupentry(all_locks_required)
  {


      cached_recording = std::dynamic_pointer_cast<multi_ndarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));

      if (!cached_recording) {
          throw snde_error("osg_cachedwaveform: Could not get recording for %s", display_req->renderable_channelpath->c_str());
      }

      /* std::shared_ptr<color_linewidth_params> cached_params */
      cached_params = std::dynamic_pointer_cast<waveform_params>(display_req->mode.constraint);
      if (!cached_params) {
          throw snde_error("osg_cachedwaveform: Could not get waveform_params");

      }


      /*
      std::shared_ptr<ndarray_recording_ref> endpoint_vertcoord_array = cached_recording->reference_ndarray("vertcoord");

      locks_required.push_back({ endpoint_vertcoord_array,false }); // accmulate locks needed for lockmanager::lock_recording_refs()

      endpoint_vertcoord_osg_array = new OSGFPArray(endpoint_vertcoord_array, 3, 3); // 3 for 3d point coordinates

      */


      bool modified;

      // get sub-requirement #0: SNDE_SRM_COLOREDTRANSPARENTLINES
      std::shared_ptr<osg_rendercacheentry> raw_entry;
      std::tie(raw_entry, modified) = params.rendercache->GetEntry(params, display_req->sub_requirements.at(0), all_locks_required);

      // std::shared_ptr<osg_cachedcoloredtransparentlines> coloredtransparentlines; (included in class definition)
      coloredtransparentlines = std::dynamic_pointer_cast<osg_cachedcoloredtransparentlines>(raw_entry);
      if (!coloredtransparentlines) {
          throw snde_error("osg_cachedwaveform: Unable to get colored transparent lines cache entry for %s", display_req->renderable_channelpath->c_str());
      }


      if (display_req->sub_requirements.size() > 1)
      {
	// get sub-requirement #1: SNDE_SRM_COLOREDTRANSPARENTPOINTS
	std::shared_ptr<osg_rendercacheentry> raw_entry2;
	std::tie(raw_entry2, modified) = params.rendercache->GetEntry(params, display_req->sub_requirements.at(1), all_locks_required);

	// std::shared_ptr<osg_cachedcoloredtransparentlines> coloredtransparentlines; (included in class definition)
	coloredtransparentpoints = std::dynamic_pointer_cast<osg_cachedcoloredtransparentpoints>(raw_entry2);
      }



      pp_geode = new osg::Geode();
      pp_lines_geom = new osg::Geometry();
      pp_lines_tris = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES, 0, 0);

      pp_lines_geom->addPrimitiveSet(pp_lines_tris);
      if (!coloredtransparentlines->cached_recording->info->immutable) {
          pp_lines_geom->setDataVariance(osg::Object::DYNAMIC);
          pp_lines_tris->setDataVariance(osg::Object::DYNAMIC);
      }
      else {
          pp_lines_geom->setDataVariance(osg::Object::STATIC);
          pp_lines_tris->setDataVariance(osg::Object::STATIC);
      }

      pp_lines_geom->setUseVertexBufferObjects(true);
      // At least on Linux/Intel graphics we get nasty messages
  // from the driver if we dont set the VBO in DYNAMIC_DRAW mode
      pp_lines_geom->getOrCreateVertexBufferObject()->setUsage(GL_DYNAMIC_DRAW);
      pp_lines_tris->setCount(coloredtransparentlines->vertcoord_osg_array->nvec);
      pp_lines_geom->setVertexArray(coloredtransparentlines->vertcoord_osg_array); // (vertex coordinates)

      pp_lines_geom->setColorArray(coloredtransparentlines->vertcoordcolor_osg_array, osg::Array::BIND_PER_VERTEX);
      pp_lines_geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
      osg::ref_ptr<osg::StateSet> pp_lines_ss = pp_lines_geom->getOrCreateStateSet();
      pp_lines_ss->setMode(GL_BLEND, osg::StateAttribute::ON);
      //pp_lines_ss->setAttribute(new osg::Point(3.0f), osg::StateAttribute::ON);
      osg::ref_ptr<osg::BlendFunc> pp_lines_bf = new osg::BlendFunc(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
      pp_lines_ss->setAttributeAndModes(pp_lines_bf.get());
      pp_lines_ss->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
      pp_geode->addDrawable(pp_lines_geom);


      if (coloredtransparentpoints) {
	pp_points_geom = new osg::Geometry();
	pp_points_points = new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, 0);

	pp_points_geom->addPrimitiveSet(pp_points_points);
	if (!coloredtransparentlines->cached_recording->info->immutable) {
	  pp_points_geom->setDataVariance(osg::Object::DYNAMIC);
	  pp_points_points->setDataVariance(osg::Object::DYNAMIC);
	}
	else {
	  pp_points_geom->setDataVariance(osg::Object::STATIC);
	  pp_points_points->setDataVariance(osg::Object::STATIC);
	}

	pp_points_geom->setUseVertexBufferObjects(true);
	// At least on Linux/Intel graphics we get nasty messages
    // from the driver if we dont set the VBO in DYNAMIC_DRAW mode
	pp_points_geom->getOrCreateVertexBufferObject()->setUsage(GL_DYNAMIC_DRAW);
	pp_points_points->setCount(coloredtransparentpoints->pointcoord_osg_array->nvec);
	pp_points_geom->setVertexArray(coloredtransparentpoints->pointcoord_osg_array); // (vertex coordinates)

	pp_points_geom->setColorArray(coloredtransparentpoints->pointcoordcolor_osg_array, osg::Array::BIND_PER_VERTEX);
	pp_points_geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
	pp_points_stateset = pp_points_geom->getOrCreateStateSet();
	pp_points_stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
	pp_points_stateset->setAttribute(new osg::Point(cached_params->pointsize), osg::StateAttribute::ON);
	osg::ref_ptr<osg::BlendFunc> pp_points_bf = new osg::BlendFunc(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
	pp_points_stateset->setAttributeAndModes(pp_points_bf.get());
	pp_points_stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
	pp_geode->addDrawable(pp_points_geom);

      }



      //osg::DisplaySettings::instance()->setNumMultiSamples(4);

      /*

      pp_endpoint_geom = new osg::Geometry();
      pp_endpoint_tris = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES, 0, 0);
      pp_endpoint_geom->addPrimitiveSet(pp_endpoint_tris);
      if (!cached_recording->info->immutable) {
          pp_endpoint_geom->setDataVariance(osg::Object::DYNAMIC);
          pp_endpoint_tris->setDataVariance(osg::Object::DYNAMIC);
      }
      else {
          pp_endpoint_geom->setDataVariance(osg::Object::STATIC);
          pp_endpoint_tris->setDataVariance(osg::Object::STATIC);
      }

      pp_endpoint_geom->setUseVertexBufferObjects(true);
      // At least on Linux/Intel graphics we get nasty messages
  // from the driver if we dont set the VBO in DYNAMIC_DRAW mode
      pp_endpoint_geom->getOrCreateVertexBufferObject()->setUsage(GL_DYNAMIC_DRAW);
      pp_endpoint_tris->setCount(endpoint_vertcoord_osg_array->nvec);
      pp_endpoint_geom->setVertexArray(endpoint_vertcoord_osg_array); // (vertex coordinates)

      osg::ref_ptr<osg::Vec4Array> EndpointColorArray = new osg::Vec4Array();

      EndpointColorArray->push_back(osg::Vec4(cached_params->color.R * 1.2, cached_params->color.G * 1.2, cached_params->color.B * 1.2, cached_params->overall_alpha)); // Setting the first 3 to less than 1.0 will dim the output. Setting the last one would probably add alpha transparency (?)

      pp_endpoint_geom->setColorArray(EndpointColorArray, osg::Array::BIND_OVERALL);
      pp_endpoint_geom->setColorBinding(osg::Geometry::BIND_OVERALL);
      pp_geode->addDrawable(pp_endpoint_geom);
      */

      osg_group = pp_geode;
      accumulate_locks_required(all_locks_required);
  }

  std::pair<bool, bool> osg_cachedwaveform::attempt_reuse(const osg_renderparams& params, std::shared_ptr<display_requirement> display_req,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)
  {
      std::shared_ptr<multi_ndarray_recording> new_recording = std::dynamic_pointer_cast<multi_ndarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
      if (!new_recording) {
          throw snde_error("osg_cachedwaveform::attempt_reuse: Could not get recording for %s", display_req->renderable_channelpath->c_str());
      }
      bool coloredtransparentlines_reusable, coloredtransparentlines_modified;
      bool coloredtransparentpoints_reusable=true, coloredtransparentpoints_modified=false;
      std::tie(coloredtransparentlines_reusable,
	       coloredtransparentlines_modified) =
	coloredtransparentlines->attempt_reuse(params,display_req->sub_requirements.at(0),all_locks_required);
      if (display_req->sub_requirements.size() > 1) {
	if (coloredtransparentpoints) {

	
	  std::tie(coloredtransparentpoints_reusable, coloredtransparentpoints_modified) = coloredtransparentpoints->attempt_reuse(params,display_req->sub_requirements.at(1),all_locks_required);
	}
	else {
	  coloredtransparentpoints_reusable = false;
	}
      }

      if (coloredtransparentlines_reusable && coloredtransparentpoints_reusable && new_recording == cached_recording && new_recording->info->immutable && display_req->mode.constraint == cached_params) {

	accumulate_locks_required(all_locks_required);
	return std::make_pair(true,coloredtransparentlines_modified || coloredtransparentpoints_modified);
      }
      
      return std::make_pair(false, false); // (reusable,modified)

  }


  void osg_cachedwaveform::clear_potentially_obsolete()
  {
      potentially_obsolete = false;
      coloredtransparentlines->clear_potentially_obsolete();
  }







  osg_cachedphaseplaneendpointwithcoloredtransparentlines::osg_cachedphaseplaneendpointwithcoloredtransparentlines(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachegroupentry(all_locks_required)
  {

    
    cached_recording = std::dynamic_pointer_cast<multi_ndarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));

    if (!cached_recording) {
      throw snde_error("osg_cachedphaseplaneendpointwithcoloredtransparentlines: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    /* std::shared_ptr<color_linewidth_params> cached_params */
    cached_params = std::dynamic_pointer_cast<color_linewidth_params>(display_req->mode.constraint);
    if (!cached_params) {
      throw snde_error("osg_cachedphaseplaneendpointwithcoloredtransparentlines: Could not get color_linewidth_params");       

    }
    
    std::shared_ptr<ndarray_recording_ref> endpoint_vertcoord_array = cached_recording->reference_ndarray("vertcoord");

    locks_required.push_back( { endpoint_vertcoord_array,false } ); // accmulate locks needed for lockmanager::lock_recording_refs()

    endpoint_vertcoord_osg_array = new OSGFPArray(endpoint_vertcoord_array,3,3); // 3 for 3d point coordinates

    
    

    
    bool modified;

    // get sub-requirement #0: SNDE_SRM_COLOREDTRANSPARENTLINES
    std::shared_ptr<osg_rendercacheentry> raw_entry;
    std::tie(raw_entry,modified) = params.rendercache->GetEntry(params,display_req->sub_requirements.at(0),all_locks_required);
    
    // std::shared_ptr<osg_cachedcoloredtransparentlines> coloredtransparentlines; (included in class definition)
    coloredtransparentlines = std::dynamic_pointer_cast<osg_cachedcoloredtransparentlines>(raw_entry);
    if (!coloredtransparentlines) {
      throw snde_error("osg_cachedphaseplaneendpointwithcoloredtransparentlines: Unable to get colored transparent lines cache entry for %s",display_req->sub_requirements.at(0)->renderable_channelpath->c_str());
    }

    
    
    pp_geode = new osg::Geode();
    pp_lines_geom = new osg::Geometry();
    pp_lines_tris = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,0);

    pp_lines_geom->addPrimitiveSet(pp_lines_tris);
    if (!coloredtransparentlines->cached_recording->info->immutable) {
      pp_lines_geom->setDataVariance(osg::Object::DYNAMIC);
      pp_lines_tris->setDataVariance(osg::Object::DYNAMIC);
    } else {
      pp_lines_geom->setDataVariance(osg::Object::STATIC);
      pp_lines_tris->setDataVariance(osg::Object::STATIC);
    }

    pp_lines_geom->setUseVertexBufferObjects(true);
        // At least on Linux/Intel graphics we get nasty messages
    // from the driver if we dont set the VBO in DYNAMIC_DRAW mode
    pp_lines_geom->getOrCreateVertexBufferObject()->setUsage(GL_DYNAMIC_DRAW);
    pp_lines_tris->setCount(coloredtransparentlines->vertcoord_osg_array->nvec); 
    pp_lines_geom->setVertexArray(coloredtransparentlines->vertcoord_osg_array); // (vertex coordinates)

    pp_lines_geom->setColorArray(coloredtransparentlines->vertcoordcolor_osg_array,osg::Array::BIND_PER_VERTEX);
    pp_lines_geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    osg::ref_ptr<osg::StateSet> pp_lines_ss = pp_lines_geom->getOrCreateStateSet();
    pp_lines_ss->setMode(GL_BLEND,osg::StateAttribute::ON);
    osg::ref_ptr<osg::BlendFunc> pp_lines_bf=new osg::BlendFunc(osg::BlendFunc::SRC_ALPHA,osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    pp_lines_ss->setAttributeAndModes(pp_lines_bf.get());
    pp_lines_ss->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    pp_geode->addDrawable(pp_lines_geom);
    


    pp_endpoint_geom = new osg::Geometry();
    pp_endpoint_tris = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,0);
    pp_endpoint_geom->addPrimitiveSet(pp_endpoint_tris);
    if (!cached_recording->info->immutable) {
      pp_endpoint_geom->setDataVariance(osg::Object::DYNAMIC);
      pp_endpoint_tris->setDataVariance(osg::Object::DYNAMIC);
    } else {
      pp_endpoint_geom->setDataVariance(osg::Object::STATIC);
      pp_endpoint_tris->setDataVariance(osg::Object::STATIC);
    }
    
    pp_endpoint_geom->setUseVertexBufferObjects(true);
        // At least on Linux/Intel graphics we get nasty messages
    // from the driver if we dont set the VBO in DYNAMIC_DRAW mode
    pp_endpoint_geom->getOrCreateVertexBufferObject()->setUsage(GL_DYNAMIC_DRAW);
    pp_endpoint_tris->setCount(endpoint_vertcoord_osg_array->nvec); 
    pp_endpoint_geom->setVertexArray(endpoint_vertcoord_osg_array); // (vertex coordinates)

    osg::ref_ptr<osg::Vec4Array> EndpointColorArray=new osg::Vec4Array();
    
    EndpointColorArray->push_back(osg::Vec4(cached_params->color.R*1.2,cached_params->color.G*1.2,cached_params->color.B*1.2,cached_params->overall_alpha)); // Setting the first 3 to less than 1.0 will dim the output. Setting the last one would probably add alpha transparency (?)
    
    pp_endpoint_geom->setColorArray(EndpointColorArray,osg::Array::BIND_OVERALL);
    pp_endpoint_geom->setColorBinding(osg::Geometry::BIND_OVERALL);    
    pp_geode->addDrawable(pp_endpoint_geom);
    
 
    osg_group = pp_geode;
    accumulate_locks_required(all_locks_required);
  }

  std::pair<bool,bool> osg_cachedphaseplaneendpointwithcoloredtransparentlines::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)
  {
    std::shared_ptr<multi_ndarray_recording> new_recording = std::dynamic_pointer_cast<multi_ndarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedphaseplaneendpointwithcoloredtransparentlines::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    bool coloredtransparentlines_reusable, coloredtransparentlines_modified;

    std::tie(coloredtransparentlines_reusable, coloredtransparentlines_modified) = coloredtransparentlines->attempt_reuse(params,display_req->sub_requirements.at(0),all_locks_required);

    if (coloredtransparentlines_reusable && new_recording==cached_recording && new_recording->info->immutable && display_req->mode.constraint == cached_params) {
      accumulate_locks_required(all_locks_required);
      return std::make_pair(true,coloredtransparentlines_modified);
    }

    return std::make_pair(false,false); // (reusable,modified)
    
  }


  void osg_cachedphaseplaneendpointwithcoloredtransparentlines::clear_potentially_obsolete()
  {
    potentially_obsolete = false;
    coloredtransparentlines->clear_potentially_obsolete();
  }



  
  osg_cachedmeshednormals::osg_cachedmeshednormals(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachearrayentry(all_locks_required)
  {
    
    cached_recording = std::dynamic_pointer_cast<meshed_vertnormalarrays_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!cached_recording) {
      throw snde_error("osg_cachedmeshednormals: Could not get recording for %s",display_req->renderable_channelpath->c_str()); 
      
    }
    
    std::shared_ptr<ndarray_recording_ref> vertnormal_array = cached_recording->reference_ndarray("vertnormal_arrays");
    
    locks_required.push_back( { vertnormal_array,false } ); // accmulate locks needed for lockmanager::lock_recording_refs()

    osg_array = new OSGFPArray(vertnormal_array,9,3); // SNDE groups them by 9 (per triangle), OSG by 3 (per vertex)for 3d coordinates

    accumulate_locks_required(all_locks_required);
    
  }


  std::pair<bool,bool> osg_cachedmeshednormals::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)
  {
    std::shared_ptr<meshed_vertnormalarrays_recording> new_recording = std::dynamic_pointer_cast<meshed_vertnormalarrays_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedmeshedvertexarray::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    if (new_recording==cached_recording && new_recording->info->immutable){
      accumulate_locks_required(all_locks_required);
      return std::make_pair(true,false);
    }
    return std::make_pair(false,false); // (reusable,modified)
  }


  
  
  osg_cachedparameterizationdata::osg_cachedparameterizationdata(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachearrayentry(all_locks_required)
  {

    cached_recording = std::dynamic_pointer_cast<meshed_texvertex_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!cached_recording) {
      throw snde_error("osg_cachedparameterizationdata: Could not get recording for %s",display_req->renderable_channelpath->c_str()); 
      
    }
    std::shared_ptr<ndarray_recording_ref> parameterization_texvertices_array = cached_recording->reference_ndarray("texvertex_arrays");

    locks_required.push_back( { parameterization_texvertices_array,false } ); // accmulate locks needed for lockmanager::lock_recording_refs()

    osg_array = new OSGFPArray(parameterization_texvertices_array,1,2); // 2 for 2d texture coordinates

    accumulate_locks_required(all_locks_required);
    
  }

  std::pair<bool,bool> osg_cachedparameterizationdata::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)
  {
    std::shared_ptr<meshed_texvertex_recording> new_recording = std::dynamic_pointer_cast<meshed_texvertex_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedparameterizationdata::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    if (new_recording==cached_recording && new_recording->info->immutable) {
      accumulate_locks_required(all_locks_required);
      return std::make_pair(true,false);
    }
    
    return std::make_pair(false,false); // (reusable,modified)
  }
  


  osg_cachedmeshedvertexarray::osg_cachedmeshedvertexarray(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachearrayentry(all_locks_required)
  {
    
    cached_recording = std::dynamic_pointer_cast<meshed_vertexarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!cached_recording) {
      throw snde_error("osg_cachedmeshedvertexarray: Could not get recording for %s",display_req->renderable_channelpath->c_str()); 
      
    }

    
    std::shared_ptr<ndarray_recording_ref> meshed_vertices_array = cached_recording->reference_ndarray("vertex_arrays");
    
    locks_required.push_back( { meshed_vertices_array,false } ); // accmulate locks needed for lockmanager::lock_recording_refs()
    osg_array = new OSGFPArray(meshed_vertices_array,1,3); // 3 for 3d coordinates

    accumulate_locks_required(all_locks_required);
  }

  std::pair<bool,bool> osg_cachedmeshedvertexarray::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)
  {
    std::shared_ptr<meshed_vertexarray_recording> new_recording = std::dynamic_pointer_cast<meshed_vertexarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedmeshedvertexarray::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    if (new_recording == cached_recording && new_recording->info->immutable){
      accumulate_locks_required(all_locks_required);
      return std::make_pair(true,false);
    }

    return std::make_pair(false,false); // (reusable,modified)
  }

  osg_cachedmeshedpart::osg_cachedmeshedpart(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachegroupentry(all_locks_required)
  {
    
    cached_recording = std::dynamic_pointer_cast<meshed_part_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!cached_recording) {
      throw snde_error("osg_cachedmeshedpart: Could not get recording for %s",display_req->renderable_channelpath->c_str()); 
      
    }
    // vertex_arrays are our first sub-requirement 
    std::shared_ptr<display_requirement> vertexarrays_requirement=display_req->sub_requirements.at(0);
    std::shared_ptr<osg_rendercacheentry> vertexarrays_entry;
    bool modified;
    
    std::tie(vertexarrays_entry,modified) = params.rendercache->GetEntry(params,vertexarrays_requirement,all_locks_required);
    if (!vertexarrays_entry) {
      throw snde_error("osg_cachedmeshedpart(): Could not get cache entry for vertex arrays channel %s",vertexarrays_requirement->renderable_channelpath->c_str());
    }
    vertexarrays_cache = std::dynamic_pointer_cast<osg_cachedmeshedvertexarray>(vertexarrays_entry);
    assert(vertexarrays_cache);
    
    
    // normals are our second sub-requirement
    std::shared_ptr<display_requirement> normals_requirement=display_req->sub_requirements.at(1);
    
    std::shared_ptr<osg_rendercacheentry> normals_entry;

    std::tie(normals_entry,modified) = params.rendercache->GetEntry(params,normals_requirement,all_locks_required);
    if (!normals_entry) {
      throw snde_error("osg_cachedmeshedpart(): Could not get cache entry for normals channel %s",normals_requirement->renderable_channelpath->c_str());
    }
    
    normals_cache = std::dynamic_pointer_cast<osg_cachedmeshednormals>(normals_entry);
    assert(normals_cache);

    
    
    // Get texture corresopnding to this same channel  ***!!! probably not appropriate here
    //texture = std::dynamic_pointer_cast<osg_rendercachetextureentry>(params.rendercache->GetEntry(params,channel_path,render_mode(SNDE_SRM_RGBATEXTURE)));
    //osg_array = new OSGFPArray(cached_recording->reference_ndarray("vertex_arrays"),3); // 3 for 3d coordinates

    geode = new osg::Geode();
    stateset=geode->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::ON);
    stateset->setMode(GL_DEPTH_TEST,osg::StateAttribute::ON);
    //stateset->setMode(GL_DEPTH_TEST,osg::StateAttribute::OFF);

    geom = new osg::Geometry();
    geode->addDrawable(geom);
    drawarrays = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,0);

    // Not entirely sure if ColorArray is necessary (?)
    osg::ref_ptr<osg::Vec4Array> ColorArray=new osg::Vec4Array();
    ColorArray->push_back(osg::Vec4(1.0,1.0,1.0,1.0)); // Setting the first 3 to less than 1.0 will dim the output. Setting the last one would probably add alpha transparency (?)
    geom->setColorArray(ColorArray,osg::Array::BIND_OVERALL);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    geom->addPrimitiveSet(drawarrays);
    if (!cached_recording->info->immutable) {
      geom->setDataVariance(osg::Object::DYNAMIC);
      drawarrays->setDataVariance(osg::Object::DYNAMIC);
    } else {
      geom->setDataVariance(osg::Object::STATIC);
      drawarrays->setDataVariance(osg::Object::STATIC);
    }
    geom->setUseVertexBufferObjects(true);
    // At least on Linux/Intel graphics we get nasty messages
    // from the driver if we dont set the VBO in DYNAMIC_DRAW mode
    geom->getOrCreateVertexBufferObject()->setUsage(GL_DYNAMIC_DRAW);
    drawarrays->setCount(vertexarrays_cache->osg_array->nvec); // add factor of two here to get an image !!!***
    geom->setVertexArray(vertexarrays_cache->osg_array); // (vertex coordinates)
    geom->setNormalArray(normals_cache->osg_array,osg::Array::BIND_PER_VERTEX);


    
    osg_group = geode;
    accumulate_locks_required(all_locks_required);
  }

  void osg_cachedmeshedpart::clear_potentially_obsolete()
  {
    potentially_obsolete=false;

    // also clear potentially_obsolete flag of our texture
    vertexarrays_cache->clear_potentially_obsolete();
    normals_cache->clear_potentially_obsolete();
  }


  
  std::pair<bool,bool> osg_cachedmeshedpart::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)
  {
    std::shared_ptr<meshed_part_recording> new_recording = std::dynamic_pointer_cast<meshed_part_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedmeshedpart::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    bool vertexarrays_reusable, vertexarrays_modified;
    bool normals_reusable, normals_modified;

    std::tie(vertexarrays_reusable,vertexarrays_modified)=vertexarrays_cache->attempt_reuse(params,display_req->sub_requirements.at(0),all_locks_required);
    std::tie(normals_reusable,normals_modified)=normals_cache->attempt_reuse(params,display_req->sub_requirements.at(1),all_locks_required);

    if (vertexarrays_reusable && normals_reusable && new_recording == cached_recording && new_recording->info->immutable) {
      accumulate_locks_required(all_locks_required);
      return std::make_pair(true,vertexarrays_modified || normals_modified);
    }

    return std::make_pair(false,false); // (reusable,modified)
  }

  osg_cachedtexedmeshedgeom::osg_cachedtexedmeshedgeom(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachedrawableentry(all_locks_required)
  {
    
    cached_recording = std::dynamic_pointer_cast<textured_part_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!cached_recording) {
      throw snde_error("osg_cachedtexedmeshedgeom: Could not get recording for %s",display_req->renderable_channelpath->c_str()); 
      
    }
    // vertex_arrays are our first sub-requirement 
    std::shared_ptr<display_requirement> vertexarrays_requirement=display_req->sub_requirements.at(0);
    std::shared_ptr<osg_rendercacheentry> vertexarrays_entry;
    bool modified;

    std::tie(vertexarrays_entry,modified) = params.rendercache->GetEntry(params,vertexarrays_requirement,all_locks_required);
    if (!vertexarrays_entry) {
      throw snde_error("osg_cachedtexedmeshedgeom(): Could not get cache entry for vertex arrays channel %s",vertexarrays_requirement->renderable_channelpath->c_str());
    }
    vertexarrays_cache = std::dynamic_pointer_cast<osg_cachedmeshedvertexarray>(vertexarrays_entry);
    assert(vertexarrays_cache);
    
    
    // normals are our second sub-requirement
    std::shared_ptr<display_requirement> normals_requirement=display_req->sub_requirements.at(1);

    std::shared_ptr<osg_rendercacheentry> normals_entry;

    std::tie(normals_entry,modified) = params.rendercache->GetEntry(params,normals_requirement,all_locks_required);
    if (!normals_entry) {
      throw snde_error("osg_cachedtexedmeshedgeom(): Could not get cache entry for normals channel %s",normals_requirement->renderable_channelpath->c_str());
    }
    
    normals_cache = std::dynamic_pointer_cast<osg_cachedmeshednormals>(normals_entry);
    assert(normals_cache);


    // parameterization is our third sub-requirement
    std::shared_ptr<display_requirement> parameterization_requirement=display_req->sub_requirements.at(2);
    std::shared_ptr<osg_rendercacheentry> parameterization_entry;
    
    std::tie(parameterization_entry,modified) = params.rendercache->GetEntry(params,parameterization_requirement,all_locks_required);
    if (!parameterization_entry) {
      throw snde_error("osg_cachedtexedmeshedgeom(): Could not get cache entry for parameterization channel %s",parameterization_requirement->renderable_channelpath->c_str());
    }

    parameterization_cache = std::dynamic_pointer_cast<osg_cachedparameterizationdata>(parameterization_entry);
    
    

    geom = new osg::Geometry();

    // Not entirely sure if ColorArray is necessary (?)
    osg::ref_ptr<osg::Vec4Array> ColorArray=new osg::Vec4Array();
    ColorArray->push_back(osg::Vec4(1.0,1.0,1.0,1.0)); // Setting the first 3 to less than 1.0 will dim the output. Setting the last one would probably add alpha transparency (?)
    geom->setColorArray(ColorArray,osg::Array::BIND_OVERALL);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    
    drawarrays = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,0);
    geom->addPrimitiveSet(drawarrays);
    if (!cached_recording->info->immutable || !vertexarrays_cache->cached_recording->info->immutable || !parameterization_cache->cached_recording->info->immutable) {  
      drawarrays->setDataVariance(osg::Object::DYNAMIC);
      geom->setDataVariance(osg::Object::DYNAMIC);
    } else {
      drawarrays->setDataVariance(osg::Object::STATIC);
      geom->setDataVariance(osg::Object::STATIC);

    }
    geom->setUseVertexBufferObjects(true);

    // At least on Linux/Intel graphics we get nasty messages
    // from the driver if we dont set the VBO in DYNAMIC_DRAW mode
    geom->getOrCreateVertexBufferObject()->setUsage(GL_DYNAMIC_DRAW);

    DataArray = vertexarrays_cache->osg_array;
    drawarrays->setCount(DataArray->nvec);
    geom->setVertexArray(DataArray); // (vertex coordinates)
    NormalArray = normals_cache->osg_array;
    geom->setNormalArray(NormalArray,osg::Array::BIND_PER_VERTEX);

    TexCoordArray = parameterization_cache->osg_array;
    geom->setTexCoordArray(0,TexCoordArray,osg::Array::BIND_PER_VERTEX); // !!!*** Do we need to mark multiple texture units to support multiple images in the parameterization space? probably...

    osg_drawable = geom; // the osg::Geometry IS our drawable. 
    accumulate_locks_required(all_locks_required);
  }
  

  void osg_cachedtexedmeshedgeom::clear_potentially_obsolete()
  {
    potentially_obsolete=false;

    // also clear potentially_obsolete flag of our subcomponents
    vertexarrays_cache->clear_potentially_obsolete();
    normals_cache->clear_potentially_obsolete();
    parameterization_cache->clear_potentially_obsolete();
  }

  
  std::pair<bool,bool> osg_cachedtexedmeshedgeom::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)
  {
    std::shared_ptr<textured_part_recording> new_recording = std::dynamic_pointer_cast<textured_part_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedmeshedpart::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    bool vertexarrays_reusable, vertexarrays_modified;
    bool normals_reusable, normals_modified;
    bool parameterization_reusable, parameterization_modified;

    std::tie(vertexarrays_reusable,vertexarrays_modified) = vertexarrays_cache->attempt_reuse(params,display_req->sub_requirements.at(0),all_locks_required);
    std::tie(normals_reusable,normals_modified) = normals_cache->attempt_reuse(params,display_req->sub_requirements.at(1),all_locks_required);
    std::tie(parameterization_reusable,parameterization_modified) = parameterization_cache->attempt_reuse(params,display_req->sub_requirements.at(2),all_locks_required);

    if (vertexarrays_reusable && normals_reusable && parameterization_reusable && new_recording==cached_recording && new_recording->info->immutable) {
      accumulate_locks_required(all_locks_required);
      return std::make_pair(true,vertexarrays_modified || normals_modified || parameterization_modified);
    }
    
      return std::make_pair(false,false); // (reusable,modified)
  }



  osg_cachedtexedmeshedpart::osg_cachedtexedmeshedpart(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachegroupentry(all_locks_required)
  {
    
    cached_recording = std::dynamic_pointer_cast<textured_part_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!cached_recording) {
      throw snde_error("osg_cachedmeshedpart: Could not get recording for %s",display_req->renderable_channelpath->c_str()); 
      
    }
    // geometry is our first sub-requirement 
    std::shared_ptr<display_requirement> geometry_requirement=display_req->sub_requirements.at(0);
    std::shared_ptr<osg_rendercacheentry> geometry_entry;
    bool modified;

    std::tie(geometry_entry,modified) = params.rendercache->GetEntry(params,geometry_requirement,all_locks_required);
    if (!geometry_entry) {
      throw snde_error("osg_cachedtexedmeshedpart(): Could not get cache entry for geometry channel %s",geometry_requirement->renderable_channelpath->c_str());
    }
    geometry_cache = std::dynamic_pointer_cast<osg_cachedtexedmeshedgeom>(geometry_entry);
    assert(geometry_cache);
    
    
    
    // Textures are our remaining sub-requirements
    size_t reqnum;
    for (reqnum=1;reqnum < display_req->sub_requirements.size();reqnum++) {
      std::shared_ptr<display_requirement> texture_requirement=display_req->sub_requirements.at(reqnum);
      std::shared_ptr<osg_rendercacheentry> texture_entry;
      bool modified;

      std::tie(texture_entry,modified) = params.rendercache->GetEntry(params,texture_requirement,all_locks_required);
      if (!texture_entry) {
	throw snde_error("osg_cachedtexedmeshedpart(): Could not get cache entry for texture channel %s",texture_requirement->renderable_channelpath->c_str());
      }
      std::shared_ptr<osg_cachedimagedata> texture_cache = std::dynamic_pointer_cast<osg_cachedimagedata>(texture_entry);
      
      texture_caches.push_back(texture_cache);      
    }
    //osg_array = new OSGFPArray(cached_recording->reference_ndarray("vertex_arrays"),3); // 3 for 3d coordinates


    geode = new osg::Geode();
    stateset=geode->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::ON);
    //stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    stateset->setMode(GL_DEPTH_TEST,osg::StateAttribute::ON);

    geode->addDrawable(geometry_cache->osg_drawable);
    


    // !!!*** Should handle multiple textures
    if (texture_caches.size() > 0) {
      stateset->setTextureAttributeAndModes(0,texture_caches.at(0)->osg_texture,osg::StateAttribute::ON);
      stateset->setTextureAttributeAndModes(0,texture_caches.at(0)->texture_transform,osg::StateAttribute::ON);
    }
    
    //geode->setStateSet(texture_state_set)

    osg_group = geode;
    accumulate_locks_required(all_locks_required);
  }


  void osg_cachedtexedmeshedpart::clear_potentially_obsolete()
  {
    potentially_obsolete=false;

    // also clear potentially_obsolete flag of our subcomponents
    geometry_cache->clear_potentially_obsolete();

    for (auto && texcache: texture_caches) {
      texcache->clear_potentially_obsolete();
    }
    
  }

  std::pair<bool,bool> osg_cachedtexedmeshedpart::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)
  {
    std::shared_ptr<textured_part_recording> new_recording = std::dynamic_pointer_cast<textured_part_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedtexedmeshedpart::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    bool geometry_reusable, geometry_modified;
    bool all_textures_reusable=true, any_texture_modified=false;

    std::tie(geometry_reusable,geometry_modified) = geometry_cache->attempt_reuse(params,display_req->sub_requirements.at(0),all_locks_required);
    for (size_t req_num=1; req_num < display_req->sub_requirements.size(); req_num++) {
      bool this_texture_reusable, this_texture_modified;
      if (req_num-1 >= texture_caches.size()) {
	all_textures_reusable = false;
	break;
      }
      std::tie(this_texture_reusable,this_texture_modified)=texture_caches.at(req_num-1)->attempt_reuse(params,display_req->sub_requirements.at(req_num),all_locks_required);
      all_textures_reusable = all_textures_reusable && this_texture_reusable;
      any_texture_modified = any_texture_modified || this_texture_modified;
    }

    if (geometry_reusable && all_textures_reusable && new_recording == cached_recording && new_recording->info->immutable) {
      accumulate_locks_required(all_locks_required);
      return std::make_pair(true,geometry_modified || any_texture_modified);
    }
    
    return std::make_pair(false,false); // (reusable,modified)
  }






  osg_cachedassembly::osg_cachedassembly(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachegroupentry(all_locks_required)
  {
    
    cached_recording = std::dynamic_pointer_cast<assembly_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!cached_recording) {
      throw snde_error("osg_cachedassembly: Could not get recording for %s",display_req->renderable_channelpath->c_str()); 
      
    }

    //std::vector<std::shared_ptr<recording_base>> sub_component_recordings;
    // Sub-components are our sub-requirements
    size_t reqnum;
    for (reqnum=0;reqnum < display_req->sub_requirements.size();reqnum++) {
      std::shared_ptr<display_requirement> component_requirement=display_req->sub_requirements.at(reqnum);
      std::shared_ptr<osg_rendercacheentry> component_entry;
      bool modified;

      std::tie(component_entry,modified) = params.rendercache->GetEntry(params,component_requirement,all_locks_required);
      if (!component_entry) {
	throw snde_error("osg_cachedassembly(): Could not get cache entry for sub-component %s",component_requirement->renderable_channelpath->c_str());
      }
      std::shared_ptr<osg_rendercachegroupentry> component_cache = std::dynamic_pointer_cast<osg_rendercachegroupentry>(component_entry);
      if (!component_cache) {
	throw snde_error("osg:cachedassembly(): Cache entry for sub-component %s not convertible to a group",component_requirement->renderable_channelpath->c_str());	
      }
      
      sub_components.push_back(component_cache);      
      //sub_component_recordings.push_back(component_requirement->original_recording);      
    }
    //osg_array = new OSGFPArray(cached_recording->reference_ndarray("vertex_arrays"),3); // 3 for 3d coordinates

    osg_group = new osg::Group();

    for (size_t component_index=0; component_index < sub_components.size(); component_index++) {
      const snde_orientation3 & piece_orientation = std::get<1>(cached_recording->pieces.at(component_index));
      snde_coord4 rotmtx[4]; // index identifies which column (data stored column-major)
      if (!orientation_valid(piece_orientation)) {
	snde_warning("osg_cachedassembly: invalid orientation for assembly component %s of channel %s", display_req->sub_requirements.at(component_index)->renderable_channelpath->c_str() ,display_req->renderable_channelpath->c_str());
	continue;
      }
      orientation_build_rotmtx(piece_orientation,rotmtx);
      
      osg::ref_ptr<osg::MatrixTransform> xform  = new osg::MatrixTransform(osg::Matrixd(&rotmtx[0].coord[0])); // remember osg::MatrixTransform also wants the matrix column-major
      xform->addChild(sub_components.at(component_index)->osg_group);
      osg_group->addChild(xform);
    }

    accumulate_locks_required(all_locks_required);
  }


  std::pair<bool,bool> osg_cachedassembly::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)
  {
    std::shared_ptr<assembly_recording> new_recording = std::dynamic_pointer_cast<assembly_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedassembly::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    bool all_sub_components_reusable=true,any_sub_components_modified = false;
    for (size_t reqnum = 0; reqnum < display_req->sub_requirements.size(); reqnum++) {
      bool this_sub_component_reusable, this_sub_component_modified;
      if (reqnum >= sub_components.size()) {
	all_sub_components_reusable=false;
	break;
      }
      std::tie(this_sub_component_reusable, this_sub_component_modified) = sub_components.at(reqnum)->attempt_reuse(params,display_req->sub_requirements.at(reqnum),all_locks_required);
      all_sub_components_reusable = all_sub_components_reusable && this_sub_component_reusable;
      any_sub_components_modified = any_sub_components_modified || this_sub_component_modified;
    }

    if (all_sub_components_reusable && new_recording == cached_recording && new_recording->info->immutable) {
      accumulate_locks_required(all_locks_required);
      return std::make_pair(true,any_sub_components_modified);
    }
    
    return std::make_pair(false,false); // (reusable,modified)
  }
  

  void osg_cachedassembly::clear_potentially_obsolete()
  {
    potentially_obsolete=false;

    for (auto && sub_component: sub_components) {
      sub_component->clear_potentially_obsolete();
    }
  }






  

  osg_cachedtransformedcomponent::osg_cachedtransformedcomponent(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required,const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachegroupentry(all_locks_required)
  {
    
    cached_recording = params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath);
    if (!cached_recording) {
      throw snde_error("osg_cachedtransformedcomponent: Could not get recording for %s",display_req->renderable_channelpath->c_str()); 
      
    }
    
    // sub-requirement 1 is our channel_to_reorient in rendering mode
    std::shared_ptr<display_requirement> channeltotrack_requirement=display_req->sub_requirements.at(0);
    std::shared_ptr<osg_rendercacheentry> channeltotrack_entry;
    
    bool ctt_modified;

    std::tie(channeltotrack_entry,ctt_modified) = params.rendercache->GetEntry(params,channeltotrack_requirement,all_locks_required);
    if (!channeltotrack_entry) {
      throw snde_error("osg_cachedtransformedcomponent(): Could not get cache entry for channeltotrack %s",channeltotrack_requirement->renderable_channelpath->c_str());
    }
    channel_to_reorient = std::dynamic_pointer_cast<osg_rendercachegroupentry>(channeltotrack_entry);
    if (!channel_to_reorient) {
      throw snde_error("osg:cachedtransformedcomponent(): Cache entry for channeltotrack %s not convertible to a group",channeltotrack_requirement->renderable_channelpath->c_str());	
    }   
    // sub-requirement 2, if present, is our component in rendering mode
    if (display_req->sub_requirements.size() > 1) {
      std::shared_ptr<display_requirement> subcomponent_requirement=display_req->sub_requirements.at(1);
      std::shared_ptr<osg_rendercacheentry> subcomponent_entry;
    
      bool sc_modified;

      std::tie(subcomponent_entry,sc_modified) = params.rendercache->GetEntry(params,subcomponent_requirement,all_locks_required);
      if (!subcomponent_entry) {
	throw snde_error("osg_cachedtransformedcomponent(): Could not get cache entry for sub-component %s",subcomponent_requirement->renderable_channelpath->c_str());
      }

      // sub_component is a class member
      sub_component = std::dynamic_pointer_cast<osg_rendercachegroupentry>(subcomponent_entry);
      if (!sub_component) {
	throw snde_error("osg:cachedtransformedcomponent(): Cache entry for sub-component %s not convertible to a group",subcomponent_requirement->renderable_channelpath->c_str());	
      }   
    }
    
    osg_group = new osg::Group();

    std::shared_ptr<poseparams> pose_params = std::dynamic_pointer_cast<poseparams>(display_req->mode.constraint);
    assert(pose_params);
    

    if (orientation_valid(pose_params->channel_to_reorient_orientation)) {
      // (otherwise just ignore NaN poses)
      
      snde_coord4 rotmtx[4]; // index identifies which column (data stored column-major)
      orientation_build_rotmtx(pose_params->channel_to_reorient_orientation,rotmtx);
      osg::ref_ptr<osg::MatrixTransform> xform  = new osg::MatrixTransform(osg::Matrixd(&rotmtx[0].coord[0])); // remember osg::MatrixTransform also wants the matrix column-major (as we interpret it; osg interprets it as row major, with left multiplication rather than right multiplication)
      //std::cout << "ChannelToTrackTransform:\n " << Eigen::Map<const Eigen::Matrix4d>(xform->getMatrix().ptr()) << "\n";
      xform->addChild(channel_to_reorient->osg_group);
      osg_group->addChild(xform);

      if (sub_component) {
	osg_group->addChild(sub_component->osg_group);
      }
    } else {
      snde_warning("osg_cachedtransformedcomponent: invalid orientation of channel %s within channel %s", channeltotrack_requirement->renderable_channelpath->c_str() ,display_req->renderable_channelpath->c_str());
    }
    
    accumulate_locks_required(all_locks_required);
  }
  
  
  std::pair<bool,bool> osg_cachedtransformedcomponent::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req,std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> *all_locks_required)
  {
    std::shared_ptr<recording_base> new_recording = params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath);
    if (!new_recording) {
      throw snde_error("osg_cachedtransformedcomponent::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }
    bool channel_to_reorient_reusable, channel_to_reorient_modified;
    bool sub_component_reusable=true, sub_component_modified=false;

    std::tie(channel_to_reorient_reusable,channel_to_reorient_modified) = channel_to_reorient->attempt_reuse(params,display_req->sub_requirements.at(0),all_locks_required);

    if (display_req->sub_requirements.size() > 1) {
      if (sub_component) {
	std::tie(sub_component_reusable,sub_component_modified)=sub_component->attempt_reuse(params,display_req->sub_requirements.at(1),all_locks_required);
      }
      else {
	sub_component_reusable = false;
      }
    }

    if (sub_component_reusable && new_recording == cached_recording && new_recording->info->immutable) {
      accumulate_locks_required(all_locks_required);
      return std::make_pair(true,sub_component_modified);
    }
    
    return std::make_pair(false,false); // (reusable,modified)
  }
  

  void osg_cachedtransformedcomponent::clear_potentially_obsolete()
  {
    potentially_obsolete=false;

    if (sub_component) {
      sub_component->clear_potentially_obsolete();
    }
  }
  

  
#if 0 // obsolste code, at least for now
  
  static std::tuple<double,double> GetPadding(std::shared_ptr<display_info> display,size_t drawareawidth,size_t drawareaheight)
  {
    double horizontal_padding = (drawareawidth-display->horizontal_divisions*display->pixelsperdiv)/2.0;
    double vertical_padding = (drawareaheight-display->vertical_divisions*display->pixelsperdiv)/2.0;

    return std::make_tuple(horizontal_padding,vertical_padding);
  }
  

  static std::tuple<double,double> GetScalefactors(std::shared_ptr<display_info> display,std::string recname)
  {
    double horizscalefactor,vertscalefactor;
    
    std::shared_ptr<display_axis> a = display->GetFirstAxis(recname);
    std::shared_ptr<display_axis> b = display->GetSecondAxis(recname);

    std::shared_ptr<display_unit> u = a->unit;
    std::shared_ptr<display_unit> v = b->unit;
    

    {
      std::lock_guard<std::mutex> adminlock(u->admin);
      if (u->pixelflag) {
	horizscalefactor=u->scale*display->pixelsperdiv;
	//fprintf(stderr,"%f units/pixel\n",u->scale);
      }
      else {
	horizscalefactor=u->scale;
      //fprintf(stderr,"%f units/div",horizscalefactor);
      }
    }

    
    {
      std::lock_guard<std::mutex> adminlock(v->admin);
      if (v->pixelflag)
	vertscalefactor=v->scale*display->pixelsperdiv;
      else
	vertscalefactor=v->scale;
    }

    return std::make_tuple(horizscalefactor,vertscalefactor);
  }
  


  static osg::Matrixd GetChannelTransform(std::shared_ptr<display_info> display,std::string recname,std::shared_ptr<display_channel> displaychan,size_t drawareawidth,size_t drawareaheight,size_t layer_index)
  {
    
    
    double horizontal_padding;
    double vertical_padding;

    double horizscalefactor,vertscalefactor;
    
    std::tie(horizontal_padding,vertical_padding) = GetPadding(drawareawidth,drawareaheight);
    
    std::shared_ptr<display_axis> a = display->GetFirstAxis(recname);
    std::shared_ptr<display_axis> b = display->GetSecondAxis(recname);

    // we assume a drawing area that goes from (-0.5,-0.5) in the lower-left corner
    // to (drawareawidth-0.5,drawareaheight-0.5) in the upper-right.

    // pixel centers are at (0,0)..(drawareawidth-1,drawareaheight-1)

    double xcenter;
    
    {
      std::lock_guard<std::mutex> adminlock(a->admin);
      xcenter=a->CenterCoord; /* in units */
    }
    //fprintf(stderr,"Got Centercoord=%f\n",xcenter);

    double ycenter;
    double VertUnitsPerDiv=display->GetVertUnitsPerDiv(displaychan);
    
    {
      std::lock_guard<std::mutex> adminlock(displaychan->admin);
      
      if (displaychan->VertZoomAroundAxis) {
	ycenter=-displaychan->Position*VertUnitsPerDiv;/**pixelsperdiv*scalefactor;*/ /* in units */
      } else {
	ycenter=displaychan->VertCenterCoord;/**pixelsperdiv*scalefactor;*/ /* in units */
      }
    }

    std::tie(horizscalefactor,vertscalefactor)=GetScalefactors(recname);


    
    
    // NOTE: transform includes z shift (away from viewer) of layer_index
    // OSG transformation matrices are transposed (!)
    //fprintf(stderr,"-xcenter/horizscalefactor = %f\n",-xcenter/horizscalefactor);
    osg::Matrixd transformmtx(display->pixelsperdiv/horizscalefactor,0,0,0, 
			      0,display->pixelsperdiv/vertscalefactor,0,0,
			      0,0,1,0,
			      -xcenter*display->pixelsperdiv/horizscalefactor+horizontal_padding+display->pixelsperdiv*display->horizontal_divisions/2.0-0.5,-ycenter*display->pixelsperdiv/vertscalefactor+vertical_padding+display->pixelsperdiv*display->vertical_divisions/2.0-0.5,-1.0*layer_index,1);// ***!!! are -0.5's and negative sign in front of layer_index correct?  .... fix here and in GraticuleTransform->setMatrix

    return transformmtx;
  }
    
#endif // obsolete code
  
};
