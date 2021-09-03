#include <osg/Geode>
#include <osg/Geometry>
#include <osg/PrimitiveSet>
#include <osg/Texture>
#include <osg/Texture2D>
#include <osg/Image>
#include <osg/StateSet>
#include <osg/LineWidth>

#include <algorithm>


#include "snde/revision_manager.hpp"
#include "snde/data_to_rgba.hpp"
#include "snde/rec_display.hpp"

#ifndef SNDE_OPENSCENEGRAPH_DATA_HPP
#define SNDE_OPENSCENEGRAPH_DATA_HPP

namespace snde {
class osg_datacachebase; // forward reference


class osg_datacacheptr: public osg::Referenced {
public:
  osg_datacachebase *ptr;
  osg_datacacheptr(osg_datacachebase *ptr) :
    ptr(ptr)
  {
    
  }
};
  
class osg_datacachebase  { // base class for datacache entries
public:
  osg::ref_ptr<osg::Group> group; 
  bool touched; // used in our latest update() pass

  osg_datacachebase()
  {
    group=new osg::Group();
    group->setUserData(new osg_datacacheptr(this));
    touched=true;
  }
  
  virtual ~osg_datacachebase() {}  // This is a polymorphic class
};

class osg_dataimagecacheentry: public osg_datacachebase {
public:
  // transform element is in the base class (osg_datacachebase)
  //osg::ref_ptr<osg::MatrixTransform> transform; /* contains geode */
  osg::ref_ptr<osg::MatrixTransform> transform; /* contains geode */
  osg::ref_ptr<osg::Geode> bordergeode; /* contains bordergeom */
  osg::ref_ptr<osg::Geometry> bordergeom;
  osg::ref_ptr<osg::DrawArrays> borderlines; // add rest of stuff here
  osg::ref_ptr<osg::StateSet> borderstateset;
  osg::ref_ptr<osg::LineWidth> borderlinewidth;
  
  osg::ref_ptr<osg::Geode> imagegeode;
  osg::ref_ptr<osg::Geometry> imagegeom;
  osg::ref_ptr<osg::DrawArrays> imagetris;
  osg::ref_ptr<osg::Texture2D> imagetexture;
  osg::ref_ptr<osg::Image> image;
  osg::ref_ptr<osg::PixelBufferObject> imagepbo;
  
  osg::ref_ptr<osg::StateSet> imagestateset;
  
  std::shared_ptr<trm_dependency> rgba_dep;

  osg_dataimagecacheentry() :
    osg_datacachebase()
  {
    transform=new osg::MatrixTransform();
    bordergeode=new osg::Geode();
    bordergeom=new osg::Geometry();
    borderlines=new osg::DrawArrays(osg::PrimitiveSet::LINES,0,0); // # is number of lines * number of coordinates per line
    borderstateset=nullptr;
    borderlinewidth=new osg::LineWidth();
    imagegeode=new osg::Geode();
    imagegeom=new osg::Geometry();
    imagetris=new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,0); // # is number of triangles * number of coordinates per triangle
    imagetexture=new osg::Texture2D();
    image=new osg::Image();
    imagepbo=new osg::PixelBufferObject();
    imagestateset=nullptr;
    rgba_dep=nullptr;
    
  }
};


/* I think this is overcomplicated, making the map index be weak_ptrs...
   because we really want to destroy any cache elements not needed NOW.. 
   because they demand keeping the renderable textures updated --
   which isn't necessary if they aren't being used! 
*/

template <typename T>
class osg_datacache {
  // single-threaded -- not thread safe at this time
public:
  std::map<std::weak_ptr<display_channel>,std::shared_ptr<T>,std::owner_less<std::weak_ptr<display_channel>>> cache;

  std::shared_ptr<T> lookup(std::shared_ptr<display_channel> chan)
  {
    auto cache_iter=cache.find(std::weak_ptr<display_channel>(chan));
    if (cache_iter==cache.end()) {
      return nullptr;
    }
    return cache_iter->second;
  }
  
  void cache_cleanup() /// I don't think this is currently used... the cacheiter cleanup way below is quite a bit more agressive
  // Call this regularly! ... perhaps after each frame... */
  {
    typename std::map<std::weak_ptr<display_channel>,std::shared_ptr<T>>::iterator cur,next;

    for (cur=cache.begin();cur != cache.end();cur=next) {
      next=cur;
      next++;

      if (cur.first.expired()) {
	cache.erase(cur);
      }
    }
  }
};


template <typename test, typename instance>
static inline bool instanceof(const instance &inst) {
  return dynamic_cast<const test *>(&inst) != nullptr;
}


static inline void GetGeom(std::shared_ptr<mutabledatastore> datastore,size_t *ndim,
	     double *IniValX,double *StepSzX,snde_index *dimlenx,
	     double *IniValY,double *StepSzY,snde_index *dimleny,
	     double *IniValZ,double *StepSzZ,snde_index *dimlenz, /* Z optional */
	     double *IniValW,double *StepSzW,snde_index *dimlenw) /* W optional */
{
  double Junk=0.0;
  snde_index Junk2=0;
  
  if (!IniValZ) IniValZ=&Junk;
  if (!StepSzZ) StepSzZ=&Junk;
  if (!dimlenz) dimlenz=&Junk2;

  if (!IniValW) IniValW=&Junk;
  if (!StepSzW) StepSzW=&Junk;
  if (!dimlenw) dimlenw=&Junk2;
  
  
  *ndim=datastore->dimlen.size();


  *IniValX=datastore->metadata.GetMetaDatumDbl("IniVal1",0.0); /* in units  */
  *StepSzX=datastore->metadata.GetMetaDatumDbl("Step1",1.0);  /* in units/index */

  if (datastore->dimlen.size() >= 1) {
    *dimlenx=datastore->dimlen[0];
  } else {
    *dimlenx=1;
  }
  
  
  *IniValY=datastore->metadata.GetMetaDatumDbl("IniVal2",0.0); /* in units */
  *StepSzY=datastore->metadata.GetMetaDatumDbl("Step2",1.0); /* in units/index */

  if (datastore->dimlen.size() >= 2) {
      *dimleny=datastore->dimlen[1];
  } else {
    *dimleny=1;
  }
  
  *IniValZ=datastore->metadata.GetMetaDatumDbl("IniVal3",0.0); /* in units */
  *StepSzZ=datastore->metadata.GetMetaDatumDbl("Step3",1.0); /* in units/index */
  if (datastore->dimlen.size() >= 3) {
    *dimlenz=datastore->dimlen[2];
  } else {
    *dimlenz=1;
  }
  

  *IniValW=datastore->metadata.GetMetaDatumDbl("IniVal4",0.0); /* in units */
  *StepSzW=datastore->metadata.GetMetaDatumDbl("Step4",1.0); /* in units/index */
  if (datastore->dimlen.size() >= 4) {
    *dimlenw=datastore->dimlen[3];
  } else {
    *dimlenw=1;
  }
  
  
}


class OSGData: public osg::Group {
public:
  std::shared_ptr<osg_datacache<osg_dataimagecacheentry>> imagecache;
  std::shared_ptr<display_info> display;
  std::shared_ptr<trm> rendering_revman; // on-demand revision manager used for rendering
  osg::ref_ptr<osg::MatrixTransform> PickerCrossHairs;

  osg::ref_ptr<osg::MatrixTransform> GraticuleTransform; // entire graticule hangs off of this!
  //osg::ref_ptr<osg::Geode> GraticuleGeode;
  // osg::ref_ptr<osg::Geometry> GraticuleGeometry;

 
  // osg::StateSet *ss = getOrCreateStateSet(); 
  // Geode->getStateSet()->setRenderBinDetails(1, "transparent");
  // ss->setRenderingHint( osg::StateSet::TRANSPARENT_BIN ); 
  // // ... which is apparently a front end to  ss->setRenderBinDetails(10, "DepthSortedBin"); 
  //  ss->setMode( GL_DEPTH_TEST, osg::StateAttribute::ON ); 

  ////osg::BlendFunc *fuct = new osg::BlendFunc(); 
  ////func->setFunction(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
  //// Geode->getStateSet()->setAttributeAndModes(func); 


  // alternatively: use scenegraphorderrenderbin
  
  //Geode->getStateSet()->setMode( GL_BLEND, osg::StateAttribute::ON ); 
  //Geode->getStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
  
  OSGData(//std::shared_ptr<osg_datacache<osg_dataimagecacheentry>> imagecache,
	  std::shared_ptr<display_info> display,
	  std::shared_ptr<trm> rendering_revman) :
    //imagecache(imagecache),
    display(display),
    rendering_revman(rendering_revman)
    
  {
    imagecache=std::make_shared<osg_datacache<osg_dataimagecacheentry>>();

    GraticuleTransform = new osg::MatrixTransform();
    osg::ref_ptr<osg::Geode> GraticuleThickGeode = new osg::Geode();
    GraticuleTransform->addChild(GraticuleThickGeode);
    osg::ref_ptr<osg::Geometry> GraticuleThickGeom = new osg::Geometry();
    GraticuleThickGeode->addDrawable(GraticuleThickGeom);
    osg::ref_ptr<osg::Geode> GraticuleThinGeode = new osg::Geode();
    GraticuleTransform->addChild(GraticuleThinGeode);
    osg::ref_ptr<osg::Geometry> GraticuleThinGeom = new osg::Geometry();
    GraticuleThinGeode->addDrawable(GraticuleThinGeom);
    
    osg::ref_ptr<osg::StateSet> GraticuleThinStateSet=GraticuleThinGeode->getOrCreateStateSet();
    GraticuleThinStateSet->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    osg::ref_ptr<osg::LineWidth> GraticuleThinLineWidth=new osg::LineWidth();
    GraticuleThinLineWidth->setWidth(display->borderwidthpixels);
    GraticuleThinStateSet->setAttributeAndModes(GraticuleThinLineWidth,osg::StateAttribute::ON);
    GraticuleThinGeom->setStateSet(GraticuleThinStateSet);
    
    osg::ref_ptr<osg::StateSet> GraticuleThickStateSet=GraticuleThickGeode->getOrCreateStateSet();
    GraticuleThickStateSet->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    osg::ref_ptr<osg::LineWidth> GraticuleThickLineWidth=new osg::LineWidth();
    GraticuleThickLineWidth->setWidth(display->borderwidthpixels*2);
    GraticuleThickStateSet->setAttributeAndModes(GraticuleThickLineWidth,osg::StateAttribute::ON);
    GraticuleThickGeom->setStateSet(GraticuleThickStateSet);

    osg::ref_ptr<osg::Vec4Array> GraticuleColorArray=new osg::Vec4Array();
    GraticuleColorArray->push_back(osg::Vec4(1.0,1.0,1.0,1.0));
    GraticuleThinGeom->setColorArray(GraticuleColorArray,osg::Array::BIND_OVERALL);
    GraticuleThinGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    GraticuleThickGeom->setColorArray(GraticuleColorArray,osg::Array::BIND_OVERALL);
    GraticuleThickGeom->setColorBinding(osg::Geometry::BIND_OVERALL);

    // Units in these coordinates are 5 per division
    osg::ref_ptr<osg::Vec3dArray> ThinGridLineCoords=new osg::Vec3dArray();
    // horizontal thin grid lines
    for (size_t cnt=0; cnt <= display->vertical_divisions;cnt++) {
      double Pos;
      Pos = -1.0*display->vertical_divisions*5.0/2.0 + cnt*5.0;
      ThinGridLineCoords->push_back(osg::Vec3d(-1.0*display->horizontal_divisions*5.0/2.0,Pos,0));
      ThinGridLineCoords->push_back(osg::Vec3d(display->horizontal_divisions*5.0/2.0,Pos,0));
    }
    // vertical thin grid lines
    for (size_t cnt=0; cnt <= display->horizontal_divisions;cnt++) {
      double Pos;
      Pos = -1.0*display->horizontal_divisions*5.0/2.0 + cnt*5.0;
      ThinGridLineCoords->push_back(osg::Vec3d(Pos,-1.0*display->vertical_divisions*5.0/2.0,0));
      ThinGridLineCoords->push_back(osg::Vec3d(Pos,display->vertical_divisions*5.0/2.0,0));
    }

    // horizontal thin minidiv lines
    for (size_t cnt=0; cnt <= display->vertical_divisions*5;cnt++) {
      double Pos;
      Pos = -1.0*display->vertical_divisions*5.0/2.0 + cnt;
      ThinGridLineCoords->push_back(osg::Vec3d(-0.5,Pos,0));
      ThinGridLineCoords->push_back(osg::Vec3d(0.5,Pos,0));
    }
    // vertical thin minidiv lines
    for (size_t cnt=0; cnt <= display->horizontal_divisions*5;cnt++) {
      double Pos;
      Pos = -1.0*display->horizontal_divisions*5.0/2.0 + cnt;
      ThinGridLineCoords->push_back(osg::Vec3d(Pos,-0.5,0));
      ThinGridLineCoords->push_back(osg::Vec3d(Pos,0.5,0));
    }

    osg::ref_ptr<osg::Vec3dArray> ThickGridLineCoords=new osg::Vec3dArray();
    // horizontal main cross line
    ThickGridLineCoords->push_back(osg::Vec3d(-1.0*display->horizontal_divisions*5.0/2.0,0.0,0.0));
    ThickGridLineCoords->push_back(osg::Vec3d(display->horizontal_divisions*5.0/2.0,0.0,0.0));

    // vertical main cross line
    ThickGridLineCoords->push_back(osg::Vec3d(0.0,-1.0*display->vertical_divisions*5.0/2.0,0.0));
    ThickGridLineCoords->push_back(osg::Vec3d(0.0,display->vertical_divisions*5.0/2.0,0.0));


    
    osg::ref_ptr<osg::DrawArrays> GraticuleThinLines = new osg::DrawArrays(osg::PrimitiveSet::LINES,0,ThinGridLineCoords->size());
    osg::ref_ptr<osg::DrawArrays> GraticuleThickLines = new osg::DrawArrays(osg::PrimitiveSet::LINES,0,ThickGridLineCoords->size());
    
    GraticuleThinGeom->addPrimitiveSet(GraticuleThinLines);
    GraticuleThickGeom->addPrimitiveSet(GraticuleThickLines);
    
    GraticuleThinGeom->setVertexArray(ThinGridLineCoords);
    GraticuleThickGeom->setVertexArray(ThickGridLineCoords);
    SetPickerCrossHairs();

  }

  std::tuple<double,double> GetPadding(size_t drawareawidth,size_t drawareaheight)
  {
    double horizontal_padding = (drawareawidth-display->horizontal_divisions*display->pixelsperdiv)/2.0;
    double vertical_padding = (drawareaheight-display->vertical_divisions*display->pixelsperdiv)/2.0;

    return std::make_tuple(horizontal_padding,vertical_padding);
  }
  

  std::tuple<double,double> GetScalefactors(std::string recname)
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
  
  osg::Matrixd GetChannelTransform(std::string recname,std::shared_ptr<display_channel> displaychan,size_t drawareawidth,size_t drawareaheight,size_t layer_index)
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
    
  

  std::shared_ptr<osg_datacachebase> update_datastore_image(std::shared_ptr<geometry> geom,std::shared_ptr<mutablerecdb> recdb, std::shared_ptr<mutabledatastore> datastore,std::shared_ptr<display_channel> displaychan,size_t drawareawidth,size_t drawareaheight,size_t layer_index,cl_context context, cl_device_id device, cl_command_queue queue)
  /****!!! Can only be called during a rendering_revman transaction (but 
       nothing shoudl be locked) */ 
  {
    double IniValX,IniValY,StepSzX,StepSzY,IniValZ=0.0,StepSzZ=0.0,IniValW=0.0,StepSzW=0.0;
    size_t ndim;
    snde_index dimlen1,dimlen2,dimlen3,dimlen4;

    double vertscalefactor,horizscalefactor;
    
    std::shared_ptr<display_axis> third_axis,fourth_axis;
    std::shared_ptr<display_unit> third_unit,fourth_unit;
    double thirdunitperdiv,fourthunitperdiv;

    double borderbox_xleft,borderbox_xright;
    double borderbox_ybot,borderbox_ytop;

    
    GetGeom(datastore,&ndim,
	    &IniValX,&StepSzX,&dimlen1,
	    &IniValY,&StepSzY,&dimlen2,
	    &IniValZ,&StepSzZ,&dimlen3,
	    &IniValW,&StepSzW,&dimlen4);

    assert(ndim <= 4);

    // Note: all of this axis data not used yet... but it will
    // need to be for labels, etc. 
    
    if (ndim==4) {
      if (displaychan->DisplaySeq >= dimlen4) {
	displaychan->DisplaySeq=dimlen4-1;
      }
    } else {
      displaychan->DisplaySeq=0;
    }
    
    if (ndim>=3) {
      if (displaychan->DisplayFrame >= dimlen3) {
	displaychan->DisplayFrame=dimlen3-1;
      }
    } else {
      displaychan->DisplayFrame=0;
    }
    assert(ndim >= 2);
    
    std::shared_ptr<display_axis> a = display->GetFirstAxis(displaychan->FullName());
    std::shared_ptr<display_axis> b = display->GetSecondAxis(displaychan->FullName());

    std::shared_ptr<display_unit> u = a->unit;
    std::shared_ptr<display_unit> v = b->unit;
    
    
    if (ndim >= 3) {
      third_axis=display->GetThirdAxis(displaychan->FullName());
      third_unit=third_axis->unit;

      {
	std::lock_guard<std::mutex> adminlock(third_unit->admin);
	if (third_unit->pixelflag)
	  thirdunitperdiv=third_unit->scale*display->pixelsperdiv;
	else
	  thirdunitperdiv=third_unit->scale;
      }
    }
    
    if (ndim >= 4) {
      fourth_axis=display->GetFourthAxis(displaychan->FullName());
      fourth_unit=fourth_axis->unit;

      {
	std::lock_guard<std::mutex> adminlock(fourth_unit->admin);
	if (fourth_unit->pixelflag)
	  fourthunitperdiv=fourth_unit->scale*display->pixelsperdiv;
	else
	  fourthunitperdiv=fourth_unit->scale;
      }
    }

    std::tie(horizscalefactor,vertscalefactor)=GetScalefactors(displaychan->FullName());

    

    osg::Matrixd transformmtx = GetChannelTransform(displaychan->FullName(),displaychan,drawareawidth,drawareaheight,layer_index);

    
    if (StepSzX > 0) {
      borderbox_xleft = std::max(IniValX-StepSzX*0.5-display->borderwidthpixels*horizscalefactor/display->pixelsperdiv/2.0,
				   (display->borderwidthpixels/2.0-0.5 - transformmtx(3,0))/transformmtx(0,0));
      borderbox_xright = std::min(IniValX+StepSzX*(dimlen1-1)+StepSzX*0.5+display->borderwidthpixels*horizscalefactor/display->pixelsperdiv/2.0,
				    (drawareawidth-display->borderwidthpixels/2.0-0.5 - transformmtx(3,0))/transformmtx(0,0));
    } else {
      borderbox_xleft = std::max(IniValX+StepSzX*(dimlen1-1)+StepSzX*0.5-display->borderwidthpixels*horizscalefactor/display->pixelsperdiv/2.0,
				   (display->borderwidthpixels/2.0-0.5 - transformmtx(3,0))/transformmtx(0,0));
      borderbox_xright = std::min(IniValX-StepSzX*0.5+display->borderwidthpixels*horizscalefactor/display->pixelsperdiv/2.0,
				    (drawareawidth-display->borderwidthpixels/2.0-0.5 - transformmtx(3,0))/transformmtx(0,0));
      
    }

    if (StepSzY > 0) {
      borderbox_ybot = std::max(IniValY-StepSzY*0.5-display->borderwidthpixels*vertscalefactor/display->pixelsperdiv/2.0,
			   (display->borderwidthpixels/2.0-0.5 - transformmtx(3,1))/transformmtx(1,1));
      borderbox_ytop = std::min(IniValY+StepSzY*(dimlen2-1)+StepSzY*0.5+display->borderwidthpixels*vertscalefactor/display->pixelsperdiv/2.0,
			   (drawareaheight-display->borderwidthpixels/2.0-0.5 - transformmtx(3,1))/transformmtx(1,1));
    } else {
      borderbox_ybot = std::max(IniValY+StepSzY*(dimlen2-1)+StepSzY*0.5-display->borderwidthpixels*vertscalefactor/display->pixelsperdiv/2.0,
			   (display->borderwidthpixels/2.0-0.5 - transformmtx(3,1))/transformmtx(1,1));
      borderbox_ytop = std::min(IniValY-StepSzY*0.5+display->borderwidthpixels*vertscalefactor/display->pixelsperdiv/2.0,
			   (drawareaheight-display->borderwidthpixels/2.0-0.5 - transformmtx(3,1))/transformmtx(1,1));
      
    }


    /* !!!*** NOTE: These sets of coordinates (esp. ImageCoords) SHOULD 
       be in double precision, but that seems to have triggered an 
       OSG bug related to the bounding box size (?) */
    
    // Z position of border is -0.5 relative to image, so it appears on top
    // around edge
    osg::ref_ptr<osg::Vec3dArray> BorderCoords=new osg::Vec3dArray(8);
    (*BorderCoords)[0]=osg::Vec3d(borderbox_xleft,borderbox_ybot,-0.5);
    (*BorderCoords)[1]=osg::Vec3d(borderbox_xright,borderbox_ybot,-0.5);
    
    (*BorderCoords)[2]=osg::Vec3d(borderbox_xright,borderbox_ybot,-0.5);
    (*BorderCoords)[3]=osg::Vec3d(borderbox_xright,borderbox_ytop,-0.5);
    
    (*BorderCoords)[4]=osg::Vec3d(borderbox_xright,borderbox_ytop,-0.5);
    (*BorderCoords)[5]=osg::Vec3d(borderbox_xleft,borderbox_ytop,-0.5);

    (*BorderCoords)[6]=osg::Vec3d(borderbox_xleft,borderbox_ytop,-0.5);
    (*BorderCoords)[7]=osg::Vec3d(borderbox_xleft,borderbox_ybot,-0.5);


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

    
    
    std::shared_ptr<osg_dataimagecacheentry> cacheentry=imagecache->lookup(displaychan);
    if (!cacheentry) {
      cacheentry=std::make_shared<osg_dataimagecacheentry>();
      cacheentry->group->addChild(cacheentry->transform);
      cacheentry->transform->addChild(cacheentry->bordergeode);
      cacheentry->bordergeode->addDrawable(cacheentry->bordergeom);
      cacheentry->bordergeom->setUseVertexBufferObjects(true);
      cacheentry->bordergeom->addPrimitiveSet(cacheentry->borderlines);
      cacheentry->borderstateset=cacheentry->bordergeode->getOrCreateStateSet();
      cacheentry->borderstateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
      
      cacheentry->borderlinewidth->setWidth(display->borderwidthpixels);
      cacheentry->borderstateset->setAttributeAndModes(cacheentry->borderlinewidth,osg::StateAttribute::ON);
      cacheentry->bordergeom->setStateSet(cacheentry->borderstateset);
      osg::ref_ptr<osg::Vec4Array> BorderColorArray=new osg::Vec4Array();
      BorderColorArray->push_back(osg::Vec4(RecColorTable[displaychan->ColorIdx].R,RecColorTable[displaychan->ColorIdx].G,RecColorTable[displaychan->ColorIdx].B,1.0));
      cacheentry->bordergeom->setColorArray(BorderColorArray,osg::Array::BIND_OVERALL);
      cacheentry->bordergeom->setColorBinding(osg::Geometry::BIND_OVERALL);
      

      
      cacheentry->transform->addChild(cacheentry->imagegeode);
      cacheentry->imagegeom->setUseVertexBufferObjects(true);
      cacheentry->imagegeom->addPrimitiveSet(cacheentry->imagetris);
      cacheentry->imagepbo->setImage(cacheentry->image);
      cacheentry->image->setPixelBufferObject(cacheentry->imagepbo);
      cacheentry->imagetexture->setResizeNonPowerOfTwoHint(false);
      cacheentry->imagestateset=cacheentry->imagegeode->getOrCreateStateSet();
      cacheentry->imagestateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
      cacheentry->imagestateset->setTextureAttributeAndModes(0,cacheentry->imagetexture,osg::StateAttribute::ON);

      osg::ref_ptr<osg::Vec4Array> ColorArray=new osg::Vec4Array();
      ColorArray->push_back(osg::Vec4(.8,.8,.8,1.0));
      cacheentry->imagegeom->setColorArray(ColorArray,osg::Array::BIND_OVERALL);
      cacheentry->imagegeom->setColorBinding(osg::Geometry::BIND_OVERALL);

      cacheentry->imagegeom->setStateSet(cacheentry->imagestateset);
      cacheentry->imagegeode->addDrawable(cacheentry->imagegeom);
    
      cacheentry->image->setDataVariance(osg::Object::DYNAMIC);

      
      
      std::weak_ptr<osg_dataimagecacheentry> cacheentryweak=std::weak_ptr<osg_dataimagecacheentry>(cacheentry);
      
      cacheentry->rgba_dep=CreateRGBADependency(rendering_revman,
						recdb,
						datastore->fullname,
						//datastore->typenum,
						geom->manager,
						(void **)&geom->geom.texbuffer,
						displaychan,
						context,
						device,
						queue,
						[ cacheentryweak, geom ] (std::shared_ptr<lockholder> input_and_array_locks,rwlock_token_set all_locks,trm_arrayregion input,trm_arrayregion output,snde_rgba **imagearray,snde_index start,size_t xsize,size_t ysize,snde_coord2 inival, snde_coord2 step) {
						  rwlock_token_set input_array_lock = input_and_array_locks->get(input.array,false,input.start,input.len);
						  unlock_rwlock_token_set(input_array_lock); // free up input lock while we process output array
						  assert(output.array==(void**)imagearray);
						  assert(output.start==start);
						  
						  rwlock_token_set output_lock = input_and_array_locks->get(output.array,true,output.start,output.len);

						  // release all_locks by moving them into a new context that we then release. 
						  {
						    rwlock_token_set release_locks=std::move(all_locks);
						  }
						  // !!!*** downgrade_to_read not currently working
						  // probably not a problem because I don't think
						  // setImage() and dirty() are time consuming --
						  // the work is done later, during rendering. 
						  //geom->manager->locker->downgrade_to_read(output_lock); // free up output write-lock while we process
						  
						  
						  std::shared_ptr<osg_dataimagecacheentry> cacheentrystrong(cacheentryweak);
						  // release lock on input control data structure
						  //inputlock.unlock();
						  // 
						  
						  
						  if (cacheentrystrong) {
						    //cacheentrystrong->imagetexture->setTextureWidth(xsize);
						    //cacheentrystrong->imagetexture->setTextureHeight(ysize);

						    /* ****!!!! WHY does OSG report (in Texture.cpp) scaling the image down from 300x300->256x256 ***???? */
						    cacheentrystrong->image->setImage(xsize,ysize,1,GL_RGBA,GL_RGBA,GL_UNSIGNED_BYTE,(unsigned char *)((*imagearray)+start),osg::Image::AllocationMode::NO_DELETE);
						    cacheentrystrong->imagetexture->setImage(cacheentrystrong->image);
						    fprintf(stderr,"First texture pixel: 0x%x 0x%x 0x%x 0x%x\n",
							    (unsigned)*((unsigned char *)((*imagearray)+start)+0),
							    (unsigned)*((unsigned char *)((*imagearray)+start)+1),
							    (unsigned)*((unsigned char *)((*imagearray)+start)+2),
							    (unsigned)*((unsigned char *)((*imagearray)+start)+3));
						      
						    //cacheentrystrong->image->setFileName("Foo.png");
						    //cacheentrystrong->imagestateset->setTextureAttributeAndModes(0,cacheentrystrong->imagetexture,osg::StateAttribute::ON);
						    //cacheentrystrong->imagepbo->setImage(cacheentrystrong->image);
						    //cacheentrystrong->image->setPixelBufferObject(cacheentrystrong->imagepbo);

						    cacheentrystrong->image->dirty();
						  }
						},
						[ cacheentryweak ] (void) { //cleanup
						  std::shared_ptr<osg_dataimagecacheentry> cacheentrystrong=cacheentryweak.lock();
						  if (cacheentrystrong) {
						    cacheentrystrong->image->setImage(0,0,1,GL_RGBA,GL_RGBA,GL_UNSIGNED_BYTE,nullptr,osg::Image::AllocationMode::NO_DELETE);
						    
						  }
						});
      
      imagecache->cache.emplace(displaychan,cacheentry);
    }
    cacheentry->transform->setMatrix(transformmtx);
    cacheentry->bordergeom->setVertexArray(BorderCoords);
    //cacheentry->bordergeom->dirty();
    cacheentry->borderlines->setCount(8);
    
    cacheentry->imagegeom->setVertexArray(ImageCoords);
    cacheentry->imagegeom->setTexCoordArray(0,ImageTexCoords);
    //cacheentry->imagegeom->dirty();
    cacheentry->imagetris->setCount(6);
    
    return cacheentry;
  }
  
  /* Update our cache/OSG tree entries for a mutabledatastore */

  std::shared_ptr<osg_datacachebase> update_datastore(std::shared_ptr<geometry> geom,std::shared_ptr<mutablerecdb> recdb, std::shared_ptr<mutabledatastore> datastore,std::shared_ptr<display_channel> displaychan,size_t drawareawidth,size_t drawareaheight,size_t layer_index,cl_context context, cl_device_id device, cl_command_queue queue)
  /* Update our cache/OSG tree entries for a mutabledatastore */
  // geom needed because that is teh current location for the texture RGBA... it also references the array manager and lock manager ....
  /****!!! Can only be called during a rendering_revman transaction (but 
       nothing should be locked) */ 
  {
    double horizunitsperdiv=1.0;
    std::shared_ptr<osg_datacachebase> cacheentry=nullptr;

      
    /* Figure out type of rendering... */
    std::shared_ptr<display_axis> axis=display->GetFirstAxis(displaychan->FullName());
    
    {
      std::lock_guard<std::mutex> adminlock(axis->unit->admin);
      if (axis->unit->pixelflag) {
	horizunitsperdiv=axis->unit->scale*display->pixelsperdiv;
      } else {
	horizunitsperdiv=axis->unit->scale;
      }
    }
    // Perhaps evaluate/render Max and Min levels here (see scope_drawrec.c)

    snde_index NDim = datastore->dimlen.size();
    snde_index DimLen1=1;
    if (NDim > 0) {
      DimLen1 = datastore->dimlen[0];
    }

    //if (!displaychan->Enabled) {
    //  return nullptr; /* no point in update for a disabled recording */
    //}
    
    if (NDim<=1 && DimLen1==1) {
      /* "single point" recording */
      fprintf(stderr,"openscenegraph_data: Single point recording rendering not yet implemented\n");
    } else if (NDim==1) {
      // 1D recording
      fprintf(stderr,"openscenegraph_data: 1D recording rendering not yet implemented\n");
    } else if (NDim > 1 && NDim <= 4) {
      // image data
      cacheentry=update_datastore_image(geom,recdb,datastore,displaychan,drawareawidth,drawareaheight,layer_index,context,device,queue);
    }
    
    
    return cacheentry;
  }

  void clearcache()
  {
    /* clear out our cache, if we aren't going to be used for a while */
    std::map<std::weak_ptr<display_channel>,std::shared_ptr<osg_dataimagecacheentry>>::iterator cacheiter;
    std::map<std::weak_ptr<display_channel>,std::shared_ptr<osg_dataimagecacheentry>>::iterator nextcacheiter;
    for (cacheiter=imagecache->cache.begin(); cacheiter != imagecache->cache.end(); cacheiter=nextcacheiter) {
      nextcacheiter=cacheiter;
      nextcacheiter++;
      
      imagecache->cache.erase(cacheiter);
      

    }
    
    
  }
  
  // update operates on a flattened list (from display_info::update()) instead of recdb directly!
  // displaychans list should generally not include disabled recordings 
  void update(std::shared_ptr<geometry> geom,std::shared_ptr<mutablerecdb> recdb,std::string selected,const std::vector<std::shared_ptr<display_channel>> & displaychans,size_t drawareawidth,size_t drawareaheight,cl_context context,cl_device_id device,cl_command_queue queue)
  // geom needed because that is the current location for the texture RGBA... it also references the array manager and lock manager ....
  /****!!! Can only be called during a rendering_revman transaction (but 
       nothing should be locked) */ 
  {
    /* NOTE: NOT THREAD SAFE... should probably be called from GUI rendering thread only! */
    size_t child_num=0;
    size_t layer_index=0;


    // Insert selected cross hairs first, if present
    display_posn selected_posn=display->get_selected_posn();
    if (selected.size() > 0 && display->GetFirstAxis(selected)==selected_posn.Horiz &&
	display->GetSecondAxis(selected)==selected_posn.Vert) {
      
      
      std::shared_ptr<mutabledatastore> selected_datastore=std::dynamic_pointer_cast<mutabledatastore>(recdb->lookup(selected));
      
      
      if (selected_datastore) {
	double horizscalefactor, vertscalefactor;
	
	std::tie(horizscalefactor,vertscalefactor)=GetScalefactors(selected);
	std::shared_ptr<display_channel> displaychan = display->lookup_channel(selected);
	
	osg::Matrixd channeltransform = GetChannelTransform(selected,displaychan,drawareawidth,drawareaheight,layer_index);
	osg::Matrixd PickerTransform(1.0,0.0,0.0,0.0,
				     0.0,1.0,0.0,0.0,
				     0.0,0.0,1.0,0.0,
				     channeltransform(3,0)+display->pixelsperdiv/horizscalefactor*selected_posn.HorizPosn,
				     channeltransform(3,1)+display->pixelsperdiv/vertscalefactor*selected_posn.VertPosn,
				     channeltransform(3,2),
				     1.0);
	
	fprintf(stderr,"Rendering crosshairs at (%f,%f)\n",channeltransform(3,0)+display->pixelsperdiv/horizscalefactor*selected_posn.HorizPosn,
		channeltransform(3,1)+display->pixelsperdiv/vertscalefactor*selected_posn.VertPosn);
	
	
	PickerCrossHairs->setMatrix(PickerTransform);
	
	if (child_num >= getNumChildren() || getChild(child_num)!=PickerCrossHairs) {
	  // picker cross hairs missing or out-of-place
	  if (containsNode(PickerCrossHairs)) {
	    removeChild(PickerCrossHairs);
	  }
	  insertChild(child_num,PickerCrossHairs);
	}
	
	child_num++; // Graticule child/layer got added either way
	layer_index++;
      }
    }
    
    
    

    // iterate over cache, setting touched flag
    for (auto & cacheentry : imagecache->cache) {
      cacheentry.second->touched=true; 
    }
    
    /* alternate approach (not used):  use a std::map on an osg::observer_ptr to the cacheentry->transform...
     * Build map to boolean by iterating over existing children in OSG tree. 
     * Set booleans to true as they come back from updatedatastore(), etc. 
     * Remove any children not marked as true from OSG tree */
    
    std::shared_ptr<osg_datacachebase> cacheentry; 
    for (auto & displaychan : displaychans) {
      //if std::type_index(typeid(*(*displaychan)->chan_data))==std::type_index(typeid())
      std::shared_ptr<mutableinfostore> chan_data = recdb->lookup(displaychan->FullName());
      if (chan_data && instanceof<mutabledatastore>(*chan_data)) {
	cacheentry = update_datastore(geom,recdb,std::dynamic_pointer_cast<mutabledatastore>(chan_data),displaychan,drawareawidth,drawareaheight,layer_index,context,device,queue);
	cacheentry->touched=true; 
	if (child_num >= getNumChildren() || getChild(child_num)!=cacheentry->group.get()) {
	  // not in correct position in our osg::Group (superclass)
	  if (containsNode(cacheentry->group)) {
	    removeChild(cacheentry->group);
	  }
	  insertChild(child_num,cacheentry->group);
	  
	}
	child_num++;
      } else if (chan_data && instanceof<mutablegeomstore>(*chan_data)) {
	
	// geomstores handled separately by openscenegraph_geom.hpp...
      } else if (chan_data) {
	fprintf(stderr,"Warning: Unknown mutableinfostore subclass %s is not renderable.\n",typeid(*chan_data).name());
      }
      layer_index++;
      
    }

    // Update graticule matrix
    // transform from the 5/div scaling of the graticule, onto the screen
    // placing it at a z distance of layerindex
    double horizontal_padding;
    double vertical_padding;
    
    std::tie(horizontal_padding,vertical_padding) = GetPadding(drawareawidth,drawareaheight);
    

    GraticuleTransform->setMatrix(osg::Matrixd(display->pixelsperdiv/5.0,0,0,0,
					       0,display->pixelsperdiv/5.0,0,0,
					       0,0,1,0,
					       horizontal_padding+display->pixelsperdiv*display->horizontal_divisions/2.0-0.5,vertical_padding+display->pixelsperdiv*display->vertical_divisions/2.0-0.5,-1.0*layer_index,1)); // ***!!! are -0.5's and negative sign in front of layer_index correct?  .... fix here and in transformmtx, above. 
    
    // Check for graticule in our group
    if (child_num >= getNumChildren() || getChild(child_num)!=GraticuleTransform) {
      // graticule missing or out-of-place
      if (containsNode(GraticuleTransform)) {
	removeChild(GraticuleTransform);
      }
      insertChild(child_num,GraticuleTransform);
    } 
    child_num++; // Graticule child/layer got added either way
    layer_index++;


    
    // remove any remaining children...
    while (child_num < getNumChildren()) {
      removeChild(getChild(child_num));
    }
    
    // iterate again over cache, checking 'touched' flag and removing anything not touched.
    std::map<std::weak_ptr<display_channel>,std::shared_ptr<osg_dataimagecacheentry>>::iterator cacheiter;
    std::map<std::weak_ptr<display_channel>,std::shared_ptr<osg_dataimagecacheentry>>::iterator nextcacheiter;
    for (cacheiter=imagecache->cache.begin(); cacheiter != imagecache->cache.end(); cacheiter=nextcacheiter) {
      nextcacheiter=cacheiter;
      nextcacheiter++;

      if (!(*cacheiter).second->touched) {
	imagecache->cache.erase(cacheiter);
      }

    }
  }

  void LockGraphics(std::shared_ptr<lockingprocess_threaded> lockprocess)
  {
    // the children of this group are mostly the group members of osg_dataimagecacheentries
    unsigned childnum;

    std::vector<trm_arrayregion> to_lock;

    for (childnum=0;childnum < getNumChildren();childnum++) {
      osg::Referenced *userdata = getChild(childnum)->getUserData();

      if (userdata) {
	osg_dataimagecacheentry *cacheentry = dynamic_cast<osg_dataimagecacheentry *>(userdata);
	if (cacheentry) {
	  to_lock.push_back(cacheentry->rgba_dep->outputs.at(0));
	}
      }
    }
    trm_lock_arrayregions(lockprocess,to_lock);

  }

  void SetPickerCrossHairs()
  {
    
    PickerCrossHairs = new osg::MatrixTransform();
    osg::ref_ptr<osg::Geode> CrossHairsGeode = new osg::Geode();
    osg::ref_ptr<osg::Geometry> CrossHairsGeom = new osg::Geometry();
    osg::ref_ptr<osg::StateSet> CrossHairsStateSet = CrossHairsGeode->getOrCreateStateSet();
    PickerCrossHairs->addChild(CrossHairsGeode);
    CrossHairsGeode->addDrawable(CrossHairsGeom);
    CrossHairsStateSet->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    osg::ref_ptr<osg::LineWidth> CrossHairsLineWidth=new osg::LineWidth();
    CrossHairsLineWidth->setWidth(4);
    CrossHairsStateSet->setAttributeAndModes(CrossHairsLineWidth,osg::StateAttribute::ON);
    CrossHairsGeom->setStateSet(CrossHairsStateSet);
    osg::ref_ptr<osg::Vec4Array> CrossHairsColorArray=new osg::Vec4Array();
    CrossHairsColorArray->push_back(osg::Vec4(1.0,1.0,1.0,1.0)); // R, G, B, A
    CrossHairsGeom->setColorArray(CrossHairsColorArray,osg::Array::BIND_OVERALL);
    CrossHairsGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    
    
    osg::ref_ptr<osg::Vec3Array> CrossHairsLinesCoords=new osg::Vec3Array();
    CrossHairsLinesCoords->push_back(osg::Vec3(-10.0,-10.0,0.0));
    CrossHairsLinesCoords->push_back(osg::Vec3(10.0,10.0,0.0));
    CrossHairsLinesCoords->push_back(osg::Vec3(-10.0,10.0,0.0));
    CrossHairsLinesCoords->push_back(osg::Vec3(10.0,-10.0,0.0));
    
    osg::ref_ptr<osg::DrawArrays> CrossHairsLines = new osg::DrawArrays(osg::PrimitiveSet::LINES,0,CrossHairsLinesCoords->size());
    
    CrossHairsGeom->addPrimitiveSet(CrossHairsLines);
    CrossHairsGeom->setVertexArray(CrossHairsLinesCoords);
    
  }

};

}

#endif // SNDE_OPENSCENEGRAPH_DATA_HPP


