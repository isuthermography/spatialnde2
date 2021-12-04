#include <osg/Group>
#include <osg/MatrixTransform>

#include "snde/snde_types.h"
#include "snde/quaternion.h"
#include "snde/openscenegraph_rendercache.hpp"
#include "snde/rec_display.hpp"


namespace snde {
  
  // Lookups in the renderer registry are done per the indexes assigned by the registered recording display handlers defined in rec_display.cpp
  
  static int osg_registered_imagedata = osg_register_renderer(rendermode(SNDE_SRM_RGBAIMAGEDATA,typeid(multi_ndarray_recording_display_handler)),[](const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry> {
      return std::make_shared<osg_cachedimagedata>(params,display_req);
    });
  
  static int osg_registered_image = osg_register_renderer(rendermode(SNDE_SRM_RGBAIMAGE,typeid(multi_ndarray_recording_display_handler)),[](const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedimage>(params,display_req);
    });
  
  static int osg_registered_meshednormals = osg_register_renderer(rendermode(SNDE_SRM_MESHEDNORMALS,typeid(meshed_part_recording_display_handler)),[](const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedmeshednormals>(params,display_req);
    });
  static int osg_registered_meshedvertexarray = osg_register_renderer(rendermode(SNDE_SRM_VERTEXARRAYS,typeid(meshed_part_recording_display_handler)),[](const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedmeshedvertexarray>(params,display_req);
    });
  static int osg_registered_parameterizationdata = osg_register_renderer(rendermode(SNDE_SRM_MESHED2DPARAMETERIZATION,typeid(meshed_parameterization_recording_display_handler)),[](const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedparameterizationdata>(params,display_req);
    });
  
  static int osg_registered_meshedpart = osg_register_renderer(rendermode(SNDE_SRM_MESHEDPARAMLESS3DPART,typeid(meshed_part_recording_display_handler)),[](const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedmeshedpart>(params,display_req);
    });

  static int osg_registered_texedmeshedgeom = osg_register_renderer(rendermode(SNDE_SRM_TEXEDMESHED3DGEOM,typeid(textured_part_recording_display_handler)),[](const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedtexedmeshedgeom>(params,display_req);
    });
  
  static int osg_registered_texedmeshedpart = osg_register_renderer(rendermode(SNDE_SRM_TEXEDMESHEDPART,typeid(textured_part_recording_display_handler)),[](const osg_renderparams &params, std::shared_ptr<display_requirement> display_req) -> std::shared_ptr<osg_rendercacheentry>  {
      return std::make_shared<osg_cachedtexedmeshedpart>(params,display_req);
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


    

  int osg_register_renderer(rendermode mode,std::function<std::shared_ptr<osg_rendercacheentry>(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)> factory)
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
    
    
    *IniValX=datastore->rec->metadata->GetMetaDatumDbl("IniVal1",0.0); /* in units  */
    *StepSzX=datastore->rec->metadata->GetMetaDatumDbl("Step1",1.0);  /* in units/index */
    
    if (datastore->layout.dimlen.size() >= 1) {
      *dimlenx=datastore->layout.dimlen.at(0);
    } else {
      *dimlenx=1;
    }
    
    
    *IniValY=datastore->rec->metadata->GetMetaDatumDbl("IniVal2",0.0); /* in units */
    *StepSzY=datastore->rec->metadata->GetMetaDatumDbl("Step2",1.0); /* in units/index */
    
    if (datastore->layout.dimlen.size() >= 2) {
      *dimleny=datastore->layout.dimlen.at(1);
    } else {
      *dimleny=1;
    }
    
    *IniValZ=datastore->rec->metadata->GetMetaDatumDbl("IniVal3",0.0); /* in units */
    *StepSzZ=datastore->rec->metadata->GetMetaDatumDbl("Step3",1.0); /* in units/index */
    if (datastore->layout.dimlen.size() >= 3) {
      *dimlenz=datastore->layout.dimlen.at(2);
    } else {
      *dimlenz=1;
    }
    
    
    *IniValW=datastore->rec->metadata->GetMetaDatumDbl("IniVal4",0.0); /* in units */
    *StepSzW=datastore->rec->metadata->GetMetaDatumDbl("Step4",1.0); /* in units/index */
    if (datastore->layout.dimlen.size() >= 4) {
      *dimlenw=datastore->layout.dimlen.at(3);
    } else {
      *dimlenw=1;
    }
    
    return true;
}
  

  
  std::shared_ptr<osg_rendercacheentry> osg_rendercache::GetEntry(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)  // mode from rendermode.hpp
  // mode from rendermode.hpp
  {
    const std::string &channel_path = display_req->channelpath;
    const rendermode_ext &mode = display_req->mode;

    // channel_path contents is often an rgba image transformed by the
    // colormapper, so the manual selection of how to view 
    // (what frame, etc.) is already included in such situations
    
    auto cache_it = cache.find(std::make_pair(channel_path,mode));
    
    if (cache_it != cache.end()) {

      if (cache_it->second->attempt_reuse(params,display_req)) {
	
	cache_it->second->clear_potentially_obsolete(); // not an obsolete entry
	return cache_it->second;	
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
    
    
    std::shared_ptr<osg_rendercacheentry> retval = renderer_it->second(params,display_req);
    cache.emplace(std::make_pair(channel_path,mode),retval);
      
    return retval;
    
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

  osg_rendercacheentry::osg_rendercacheentry() :
    potentially_obsolete(false)
  {
    
  }

  //bool osg_rendercacheentry::attempt_reuse(const osg_renderparams &params,const std::string &channel_path,const rendermode & mode)
  //{
  //  return false;
  //}

  void osg_rendercacheentry::clear_potentially_obsolete()
  {
    potentially_obsolete=false;
  }

  

  
  osg_cachedimagedata::osg_cachedimagedata(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachetextureentry()
  {
    size_t ndim;
    double IniValX,IniValY,IniValZ,IniValW;
    double StepSzX,StepSzY,StepSzZ,StepSzW;
    snde_index dimlen3,dimlen4; // Note: dimlen1, dimlen2 are class members

    cached_recording = params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath);
    if (!cached_recording) {
      throw snde_error("osg_cachedimagedata: Could not get recording for %s",display_req->renderable_channelpath->c_str()); 
      
    }
    
    if (!GetGeom(cached_recording,&ndim,
		 &IniValX,&StepSzX,&dimlen1,
		 &IniValY,&StepSzY,&dimlen2,
		 &IniValZ,&StepSzZ,&dimlen3,
		 &IniValW,&StepSzW,&dimlen4)) {
      throw snde_error("osg_cachedimagedata: Could not get geometry for %s",display_req->renderable_channelpath->c_str()); 
    }

    
    osg::ref_ptr<osg::Texture2D> imagetexture=new osg::Texture2D();
    osg_texture = imagetexture;
    
    image=new osg::Image();
    imagepbo=new osg::PixelBufferObject();
    

    // Set up scene graph
    imagepbo->setImage(image);
    image->setPixelBufferObject(imagepbo);
    imagetexture->setResizeNonPowerOfTwoHint(false);
    
    if (cached_recording->info->immutable) {
      image->setDataVariance(osg::Object::STATIC); 

    } else {
      image->setDataVariance(osg::Object::DYNAMIC); 
    }

    

    image->setImage(dimlen1,dimlen2,1,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,(unsigned char *)cached_recording->cast_to_multi_ndarray()->void_shifted_arrayptr(0),osg::Image::AllocationMode::NO_DELETE);
    imagetexture->setInternalFormat(GL_RGBA);
    imagetexture->setImage(image);    
    
  }
  
  bool osg_cachedimagedata::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)
  {
    // only reuse if the recording pointer is the same; everything else here is
    // trivial enough it's pointless to try to reuse.
    
    std::shared_ptr<recording_base> new_recording = params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath);
    
    if (!new_recording) {
      throw snde_error("osg_cachedimagedata::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    return (new_recording == cached_recording && new_recording->info->immutable); // ***!!! For mutable recordings if we wanted we could verify that the pointer remains the same and just mark the array as dirty in OSG
  }



  osg_cachedimage::osg_cachedimage(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req):
    osg_rendercachegroupentry()
  {
    size_t ndim;
    double IniValX,IniValY,IniValZ,IniValW;
    double StepSzX,StepSzY,StepSzZ,StepSzW;
    snde_index dimlen1,dimlen2,dimlen3,dimlen4;


    cached_recording = params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath);
    if (!cached_recording) {
      throw snde_error("osg_cachedimagedata: Could not get recording for %s",display_req->renderable_channelpath->c_str()); 
      
    }
    if (!GetGeom(cached_recording,&ndim, // doesn't count as a parameter because dependent solely on the underlying recording
		 &IniValX,&StepSzX,&dimlen1,
		 &IniValY,&StepSzY,&dimlen2,
		 &IniValZ,&StepSzZ,&dimlen3,
		 &IniValW,&StepSzW,&dimlen4)) {
      // cast failed; return empty group
      throw snde_error("osg_cachedimage: Could not get geometry for %s",display_req->renderable_channelpath->c_str()); 
    }
    
    // Get texture correpsonding to this same channel
    texture = std::dynamic_pointer_cast<osg_rendercachetextureentry>(params.rendercache->GetEntry(params,display_req->sub_requirements.at(0)));

    if (!texture) {
      throw snde_error("osg_cachedimage: Unable to get texture cache entry for %s",display_req->sub_requirements.at(0)->renderable_channelpath->c_str());
    }
    
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
    imagegeom->addPrimitiveSet(imagetris);
    imagestateset=imagegeode->getOrCreateStateSet();
    imagestateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    imagestateset->setTextureAttributeAndModes(0,imagetexture,osg::StateAttribute::ON);
    
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
      // lower-left triangle (if both StepSzX and StepSzY positive)
      (*ImageCoords)[0]=osg::Vec3d(IniValX-0.5*StepSzX,
				   IniValY-0.5*StepSzY,
				   0.0);
      (*ImageCoords)[1]=osg::Vec3d(IniValX+dimlen1*StepSzX-0.5*StepSzX,
				   IniValY-0.5*StepSzY,
				   0.0);
      (*ImageCoords)[2]=osg::Vec3d(IniValX-0.5*StepSzX,
				   IniValY+dimlen2*StepSzY-0.5*StepSzY,
				   0.0);
      (*ImageTexCoords)[0]=osg::Vec2d(0,0);
      (*ImageTexCoords)[1]=osg::Vec2d(1,0);
      (*ImageTexCoords)[2]=osg::Vec2d(0,1);
      
      // upper-right triangle (if both StepSzX and StepSzY positive)
      (*ImageCoords)[3]=osg::Vec3d(IniValX+dimlen1*StepSzX-0.5*StepSzX,
				   IniValY+dimlen2*StepSzY-0.5*StepSzY,
				   0.0);
      (*ImageCoords)[4]=osg::Vec3d(IniValX-0.5*StepSzX,
				   IniValY+dimlen2*StepSzY-0.5*StepSzY,
				   0.0);
      (*ImageCoords)[5]=osg::Vec3d(IniValX+dimlen1*StepSzX-0.5*StepSzX,
				   IniValY-0.5*StepSzY,
				   0.0);
      (*ImageTexCoords)[3]=osg::Vec2d(1,1);
      (*ImageTexCoords)[4]=osg::Vec2d(0,1);
      (*ImageTexCoords)[5]=osg::Vec2d(1,0);
    } else {
      // One of StepSzX or StepSzY is positive, one is negative
      // work as raster coordinates (StepSzY negative)
      // lower-left triangle
      (*ImageCoords)[0]=osg::Vec3d(IniValX-0.5*StepSzX,
				   IniValY+dimlen2*StepSzY-0.5*StepSzY,
				   0.0);
      (*ImageCoords)[1]=osg::Vec3d(IniValX+dimlen1*StepSzX-0.5*StepSzX,
				   IniValY+dimlen2*StepSzY-0.5*StepSzY,
				   0.0);
      (*ImageCoords)[2]=osg::Vec3d(IniValX-0.5*StepSzX,
				   IniValY-0.5*StepSzY,
				   0.0);
      (*ImageTexCoords)[0]=osg::Vec2d(0,1);
      (*ImageTexCoords)[1]=osg::Vec2d(1,1);
      (*ImageTexCoords)[2]=osg::Vec2d(0,0);
      
      // upper-right triangle 
      (*ImageCoords)[3]=osg::Vec3d(IniValX+dimlen1*StepSzX-0.5*StepSzX,
				   IniValY-0.5*StepSzY,
				   0.0);
      (*ImageCoords)[4]=osg::Vec3d(IniValX-0.5*StepSzX,
				   IniValY-0.5*StepSzY,
				   0.0);
      (*ImageCoords)[5]=osg::Vec3d(IniValX+dimlen1*StepSzX-0.5*StepSzX,
				   IniValY+dimlen2*StepSzY-0.5*StepSzY,
				   0.0);
      (*ImageTexCoords)[3]=osg::Vec2d(1,0);
      (*ImageTexCoords)[4]=osg::Vec2d(0,0);
      (*ImageTexCoords)[5]=osg::Vec2d(1,1);
      
    }
    

    imagegeom->setVertexArray(ImageCoords);
    imagegeom->setTexCoordArray(0,ImageTexCoords);
    imagetris->setCount(6);
    
    
    
  }

  void osg_cachedimage::clear_potentially_obsolete()
  {
    potentially_obsolete=false;

    // also clear potentially_obsolete flag of our texture
    texture->clear_potentially_obsolete();
  }


  bool osg_cachedimage::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)
  {
    // only reuse if the recording pointer is the same; everything else here is
    // trivial enough it's pointless to try to reuse.
    
    std::shared_ptr<recording_base> new_recording = params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath);
    
    if (!new_recording) {
      throw snde_error("osg_cachedimage::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    return (new_recording == cached_recording && new_recording->info->immutable);
  }


  osg_cachedparameterizationdata::osg_cachedparameterizationdata(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachearrayentry()
  {

    cached_recording = std::dynamic_pointer_cast<meshed_texvertex_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!cached_recording) {
      throw snde_error("osg_cachedparameterizationdata: Could not get recording for %s",display_req->renderable_channelpath->c_str()); 
      
    }

    osg_array = new OSGFPArray(cached_recording->reference_ndarray("texvertex_arrays"),2); // 2 for 2d texture coordinates
    
  }

  bool osg_cachedparameterizationdata::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)
  {
    std::shared_ptr<meshed_texvertex_recording> new_recording = std::dynamic_pointer_cast<meshed_texvertex_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedparameterizationdata::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    return (new_recording==cached_recording && new_recording->info->immutable);
  }
  


  osg_cachedmeshedvertexarray::osg_cachedmeshedvertexarray(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachearrayentry()
  {
    
    cached_recording = std::dynamic_pointer_cast<meshed_vertexarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!cached_recording) {
      throw snde_error("osg_cachedmeshedvertexarray: Could not get recording for %s",display_req->renderable_channelpath->c_str()); 
      
    }

    
    osg_array = new OSGFPArray(cached_recording->reference_ndarray("vertex_arrays"),3); // 3 for 3d coordinates
    
  }

  bool osg_cachedmeshedvertexarray::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)
  {
    std::shared_ptr<meshed_vertexarray_recording> new_recording = std::dynamic_pointer_cast<meshed_vertexarray_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedmeshedvertexarray::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    return (new_recording==cached_recording && new_recording->info->immutable);
  }
  

  

  osg_cachedmeshednormals::osg_cachedmeshednormals(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachearrayentry()
  {
    
    cached_recording = std::dynamic_pointer_cast<meshed_vertnormals_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!cached_recording) {
      throw snde_error("osg_cachedmeshednormals: Could not get recording for %s",display_req->renderable_channelpath->c_str()); 
      
    }

    
    osg_array = new OSGFPArray(cached_recording->reference_ndarray("vertnormals"),3); // 3 for 3d coordinates
    
  }


  bool osg_cachedmeshednormals::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)
  {
    std::shared_ptr<meshed_vertnormals_recording> new_recording = std::dynamic_pointer_cast<meshed_vertnormals_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedmeshedvertexarray::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    return (new_recording==cached_recording && new_recording->info->immutable);
  }

  

  osg_cachedmeshedpart::osg_cachedmeshedpart(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachegroupentry()
  {
    
    cached_recording = std::dynamic_pointer_cast<meshed_part_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!cached_recording) {
      throw snde_error("osg_cachedmeshedpart: Could not get recording for %s",display_req->renderable_channelpath->c_str()); 
      
    }
    // vertex_arrays are our first sub-requirement 
    std::shared_ptr<display_requirement> vertexarrays_requirement=display_req->sub_requirements.at(0);
    std::shared_ptr<osg_rendercacheentry> vertexarrays_entry = params.rendercache->GetEntry(params,vertexarrays_requirement);
    if (!vertexarrays_entry) {
      throw snde_error("osg_cachedmeshedpart(): Could not get cache entry for vertex arrays channel %s",vertexarrays_requirement->renderable_channelpath->c_str());
    }
    vertexarrays_cache = std::dynamic_pointer_cast<osg_cachedmeshedvertexarray>(vertexarrays_entry);
    assert(vertexarrays_cache);
    
    
    // normals are our second sub-requirement
    std::shared_ptr<display_requirement> normals_requirement=display_req->sub_requirements.at(1);

    std::shared_ptr<osg_rendercacheentry> normals_entry = params.rendercache->GetEntry(params,normals_requirement);
    if (!normals_entry) {
      throw snde_error("osg_cachedmeshedpart(): Could not get cache entry for normals channel %s",normals_requirement->renderable_channelpath->c_str());
    }
    
    normals_cache = std::dynamic_pointer_cast<osg_cachedmeshednormals>(normals_entry);
    assert(normals_cache);

    
    
    // Get texture corresopnding to this same channel  ***!!! probably not appropriate here
    //texture = std::dynamic_pointer_cast<osg_rendercachetextureentry>(params.rendercache->GetEntry(params,channel_path,render_mode(SNDE_SRM_RGBATEXTURE)));
    //osg_array = new OSGFPArray(cached_recording->reference_ndarray("vertex_arrays"),3); // 3 for 3d coordinates


    geom = new osg::Geometry();
    drawarrays = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,0);
    geom->addPrimitiveSet(drawarrays);
    if (!cached_recording->info->immutable) {
      geom->setDataVariance(osg::Object::DYNAMIC);
    } else {
      geom->setDataVariance(osg::Object::STATIC);
    }
    geom->setUseVertexBufferObjects(true);
    geom->setVertexArray(vertexarrays_cache->osg_array); // (vertex coordinates)
    geom->setNormalArray(normals_cache->osg_array,osg::Array::BIND_PER_VERTEX);
    
    geode = new osg::Geode();
    geode->addDrawable(geom);

    osg_group = geode;
  }

  void osg_cachedmeshedpart::clear_potentially_obsolete()
  {
    potentially_obsolete=false;

    // also clear potentially_obsolete flag of our texture
    vertexarrays_cache->clear_potentially_obsolete();
    normals_cache->clear_potentially_obsolete();
  }


  
  bool osg_cachedmeshedpart::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)
  {
    std::shared_ptr<meshed_part_recording> new_recording = std::dynamic_pointer_cast<meshed_part_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedmeshedpart::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    return (new_recording==cached_recording && new_recording->info->immutable);
  }

  osg_cachedtexedmeshedgeom::osg_cachedtexedmeshedgeom(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachedrawableentry()
  {
    
    cached_recording = std::dynamic_pointer_cast<textured_part_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!cached_recording) {
      throw snde_error("osg_cachedtexedmeshedgeom: Could not get recording for %s",display_req->renderable_channelpath->c_str()); 
      
    }
    // vertex_arrays are our first sub-requirement 
    std::shared_ptr<display_requirement> vertexarrays_requirement=display_req->sub_requirements.at(0);
    std::shared_ptr<osg_rendercacheentry> vertexarrays_entry = params.rendercache->GetEntry(params,vertexarrays_requirement);
    if (!vertexarrays_entry) {
      throw snde_error("osg_cachedtexedmeshedgeom(): Could not get cache entry for vertex arrays channel %s",vertexarrays_requirement->renderable_channelpath->c_str());
    }
    vertexarrays_cache = std::dynamic_pointer_cast<osg_cachedmeshedvertexarray>(vertexarrays_entry);
    assert(vertexarrays_cache);
    
    
    // normals are our second sub-requirement
    std::shared_ptr<display_requirement> normals_requirement=display_req->sub_requirements.at(1);

    std::shared_ptr<osg_rendercacheentry> normals_entry = params.rendercache->GetEntry(params,normals_requirement);
    if (!normals_entry) {
      throw snde_error("osg_cachedtexedmeshedgeom(): Could not get cache entry for normals channel %s",normals_requirement->renderable_channelpath->c_str());
    }
    
    normals_cache = std::dynamic_pointer_cast<osg_cachedmeshednormals>(normals_entry);
    assert(normals_cache);


    // parameterization is our third sub-requirement
    std::shared_ptr<display_requirement> parameterization_requirement=display_req->sub_requirements.at(1);
    std::shared_ptr<osg_rendercacheentry> parameterization_entry = params.rendercache->GetEntry(params,parameterization_requirement);
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
      geom->setDataVariance(osg::Object::DYNAMIC);
    } else {
      geom->setDataVariance(osg::Object::STATIC);
    }
    geom->setUseVertexBufferObjects(true);

    DataArray = vertexarrays_cache->osg_array;
    geom->setVertexArray(DataArray); // (vertex coordinates)

    NormalArray = normals_cache->osg_array;
    geom->setNormalArray(NormalArray,osg::Array::BIND_PER_VERTEX);

    TexCoordArray = parameterization_cache->osg_array;
    geom->setTexCoordArray(0,TexCoordArray,osg::Array::BIND_PER_VERTEX); // !!!*** Do we need to mark multiple texture units to support multiple images in the parameterization space? probably...

    osg_drawable = geom; // the osg::Geometry IS our drawable. 
    
  }
  

  void osg_cachedtexedmeshedgeom::clear_potentially_obsolete()
  {
    potentially_obsolete=false;

    // also clear potentially_obsolete flag of our subcomponents
    vertexarrays_cache->clear_potentially_obsolete();
    normals_cache->clear_potentially_obsolete();
    parameterization_cache->clear_potentially_obsolete();
  }

  
  bool osg_cachedtexedmeshedgeom::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)
  {
    std::shared_ptr<textured_part_recording> new_recording = std::dynamic_pointer_cast<textured_part_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedmeshedpart::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    return (new_recording==cached_recording && new_recording->info->immutable);
  }



  osg_cachedtexedmeshedpart::osg_cachedtexedmeshedpart(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachegroupentry()
  {
    
    cached_recording = std::dynamic_pointer_cast<textured_part_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!cached_recording) {
      throw snde_error("osg_cachedmeshedpart: Could not get recording for %s",display_req->renderable_channelpath->c_str()); 
      
    }
    // geometry is our first sub-requirement 
    std::shared_ptr<display_requirement> geometry_requirement=display_req->sub_requirements.at(0);
    std::shared_ptr<osg_rendercacheentry> geometry_entry = params.rendercache->GetEntry(params,geometry_requirement);
    if (!geometry_entry) {
      throw snde_error("osg_cachedtexedmeshedpart(): Could not get cache entry for geometry channel %s",geometry_requirement->renderable_channelpath->c_str());
    }
    geometry_cache = std::dynamic_pointer_cast<osg_cachedtexedmeshedgeom>(geometry_entry);
    assert(geometry_cache);
    
    
    
    // Textures are our remaining sub-requirements
    size_t reqnum;
    for (reqnum=1;reqnum < display_req->sub_requirements.size();reqnum++) {
      std::shared_ptr<display_requirement> texture_requirement=display_req->sub_requirements.at(reqnum);
      std::shared_ptr<osg_rendercacheentry> texture_entry = params.rendercache->GetEntry(params,texture_requirement);
      if (!texture_entry) {
	throw snde_error("osg_cachedtexedmeshedpart(): Could not get cache entry for texture channel %s",texture_requirement->renderable_channelpath->c_str());
      }
      std::shared_ptr<osg_cachedimagedata> texture_cache = std::dynamic_pointer_cast<osg_cachedimagedata>(texture_entry);
      
      texture_caches.push_back(texture_cache);      
    }
    //osg_array = new OSGFPArray(cached_recording->reference_ndarray("vertex_arrays"),3); // 3 for 3d coordinates


    geode = new osg::Geode();
    geode->addDrawable(geometry_cache->osg_drawable);
    

    stateset=geode->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::ON);
    stateset->setMode(GL_DEPTH_TEST,osg::StateAttribute::ON);

    // !!!*** Should handle multiple textures
    if (texture_caches.size() > 0) {
      stateset->setTextureAttributeAndModes(0,texture_caches.at(0)->osg_texture,osg::StateAttribute::ON);
    }
    
    //geode->setStateSet(texture_state_set)

    osg_group = geode;
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

  bool osg_cachedtexedmeshedpart::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)
  {
    std::shared_ptr<textured_part_recording> new_recording = std::dynamic_pointer_cast<textured_part_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedmeshedpart::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    return (new_recording==cached_recording && new_recording->info->immutable);
  }






  osg_cachedassembly::osg_cachedassembly(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req) :
    osg_rendercachegroupentry()
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
      std::shared_ptr<osg_rendercacheentry> component_entry = params.rendercache->GetEntry(params,component_requirement);
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
      orientation_build_rotmtx(piece_orientation,rotmtx);
      
      osg::ref_ptr<osg::MatrixTransform> xform  = new osg::MatrixTransform(osg::Matrixd(&rotmtx[0].coord[0])); // remember osg::MatrixTransform also wants the matrix column-major
      xform->addChild(sub_components.at(component_index)->osg_group);
      osg_group->addChild(xform);
    }
  }


  bool osg_cachedassembly::attempt_reuse(const osg_renderparams &params,std::shared_ptr<display_requirement> display_req)
  {
    std::shared_ptr<assembly_recording> new_recording = std::dynamic_pointer_cast<assembly_recording>(params.with_display_transforms->check_for_recording(*display_req->renderable_channelpath));
    if (!new_recording) {
      throw snde_error("osg_cachedassembly::attempt_reuse: Could not get recording for %s",display_req->renderable_channelpath->c_str());       
    }

    return (new_recording==cached_recording && new_recording->info->immutable);
  }
  

  void osg_cachedassembly::clear_potentially_obsolete()
  {
    potentially_obsolete=false;

    for (auto && sub_component: sub_components) {
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
