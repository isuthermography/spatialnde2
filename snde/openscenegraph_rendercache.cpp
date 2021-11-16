

#include "snde/openscenegraph_rendercache.hpp"
#include "snde/rec_display.hpp"

namespace snde {

  
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
  

  
  osg::ref_ptr<osg::Group> osg_rendercache::GetEntry(std::shared_ptr<recording_set_state> with_display_transforms,const std::string &channel_path,int mode,double left,double right,double bottom,double top)
  // mode is SNDE_DRM_xxxx from rec_display.hpp
  {

    bool found_suitable_entry=false;
    
    auto cache_it = cache.find(std::make_pair(channel_path,mode));

    if (cache_it != cache.end()) {
      if (cache_it->second->cached_recording) {
	std::shared_ptr<recording_base> new_recording = with_display_transforms->check_for_recording(channel_path);
	if (new_recording==cache_it->second->cached_recording) {
	  if (mode==SNDE_DRM_GEOMETRY) {
	    found_suitable_entry = true;
	  } else if (mode==SNDE_DRM_RGBAIMAGE || mode==SNDE_DRM_RAW) {
	    // for these, the (left,right,bottom,top) matter
	    if (left >= cache_it->second->left && right <= cache_it->second->right &&
		bottom >= cache_it->second->bottom && top <= cache_it->second->top) {
	      // new requirements are strictly inside old requirements
	      found_suitable_entry = true; 
	    }
	  }
	}
      }
      
    }
    if (found_suitable_entry) {
      cache_it->second->potentially_obsolete = false; // not an obsolete entry
      return cache_it->second->osg_group;
    }


    // If we got here, then no suitable entry was found.
    // Create a new one
    if (mode==SNDE_DRM_RGBAIMAGE) {
      std::shared_ptr<recording_base> new_recording = with_display_transforms->check_for_recording(channel_path);

      if (!new_recording) {
	// recording not present (!)
	throw snde_error("Trying to display missing recording %s",channel_path);
      }
      std::shared_ptr<osg_cachedimagedata> imgentry = std::make_shared<osg_cachedimagedata>(new_recording,left,right,bottom,top);

      // a bit of a question whether we should cache mutable recordings
      // at all. For now they go in here...
      cache.emplace(std::make_pair(channel_path,mode),imgentry);
      
      return imgentry->osg_group;
    } else {
      snde_warning("openscenegraph_rendercache: No caching process for mode #%d",mode);
    }
    return nullptr;
  }


  void osg_rendercache::mark_obsolete()
  {
    for (auto && chanpathmode_cacheentry: cache) {
      chanpathmode_cacheentry.second->potentially_obsolete=true;
    }
  }


  void osg_rendercache::clear_obsolete()
  {
    std::map<std::pair<std::string,int>,std::shared_ptr<osg_rendercacheentry>>::iterator cache_it, next_it;
    
    for (cache_it = cache.begin();cache_it != cache.end();cache_it = next_it) {
      next_it = cache_it;
      ++next_it;
      
      if (cache_it->second->potentially_obsolete) {
	// obsolete entry; remove it
	cache.erase(cache_it);
      }
    }
  }
  
  osg_rendercacheentry::osg_rendercacheentry(std::shared_ptr<recording_base> cached_recording,double left,double right,double bottom,double top) :
    //display(display),
    //displaychan(displaychan),
    cached_recording(cached_recording),
    left(left),
    right(right),
    bottom(bottom),
    top(top),
    potentially_obsolete(false)
  {

  }

  osg_cachedimagedata::osg_cachedimagedata(std::shared_ptr<recording_base> cached_recording,double left,double right,double bottom,double top):
    osg_rendercacheentry(cached_recording,left,right,bottom,top)
  {
    size_t ndim;
    double IniValX,IniValY,IniValZ,IniValW;
    double StepSzX,StepSzY,StepSzZ,StepSzW;
    snde_index dimlen1,dimlen2,dimlen3,dimlen4;

    if (!GetGeom(cached_recording,&ndim,
		 &IniValX,&StepSzX,&dimlen1,
		 &IniValY,&StepSzY,&dimlen2,
		 &IniValZ,&StepSzZ,&dimlen3,
		 &IniValW,&StepSzW,&dimlen4)) {
      // cast failed; return empty group
      osg_group = new osg::Group();
      return;
    }

    
    //transform=new osg::MatrixTransform();
    imagegeode=new osg::Geode();
    imagegeom=new osg::Geometry();
    imagetris=new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,0); // # is number of triangles * number of coordinates per triangle
    imagetexture=new osg::Texture2D();
    image=new osg::Image();
    imagepbo=new osg::PixelBufferObject();
    imagestateset=nullptr;
    

    // Set up scene graph
    osg_group = new osg::Group();
    osg_group->addChild(imagegeode);
    imagegeom->setUseVertexBufferObjects(true);
    imagegeom->addPrimitiveSet(imagetris);
    imagepbo->setImage(image);
    image->setPixelBufferObject(imagepbo);
    imagetexture->setResizeNonPowerOfTwoHint(false);
    imagestateset=imagegeode->getOrCreateStateSet();
    imagestateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    imagestateset->setTextureAttributeAndModes(0,imagetexture,osg::StateAttribute::ON);
    
    osg::ref_ptr<osg::Vec4Array> ColorArray=new osg::Vec4Array();
    ColorArray->push_back(osg::Vec4(1.0,1.0,1.0,1.0)); // Setting the first 3 to less than 1.0 will dim the output. Setting the last one would probably add alpha transparency (?)
    imagegeom->setColorArray(ColorArray,osg::Array::BIND_OVERALL);
    imagegeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    
    imagegeom->setStateSet(imagestateset);
    imagegeode->addDrawable(imagegeom);

    if (cached_recording->info->immutable) {
      image->setDataVariance(osg::Object::STATIC); 

    } else {
      image->setDataVariance(osg::Object::DYNAMIC); 
    }

    

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

    image->setImage(dimlen1,dimlen2,1,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,(unsigned char *)cached_recording->cast_to_multi_ndarray()->void_shifted_arrayptr(0),osg::Image::AllocationMode::NO_DELETE);
    imagetexture->setInternalFormat(GL_RGBA);
    imagetexture->setImage(image);

    imagegeom->setVertexArray(ImageCoords);
    imagegeom->setTexCoordArray(0,ImageTexCoords);
    imagetris->setCount(6);
    
    
    
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
