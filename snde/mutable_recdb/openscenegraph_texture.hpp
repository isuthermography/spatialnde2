#include <vector>

#include <osg/Array>
#include <osg/Geode>
#include <osg/Group>
#include <osg/Geometry>
#include <osg/TexMat>
#include <osg/Texture2D>
#include <osg/MatrixTransform>

#include "snde/geometry_types.h"
#include "snde/geometrydata.h"
#include "snde/lockmanager.hpp"
#include "snde/geometry.hpp"

#include "snde/openclcachemanager.hpp"
#include "snde/opencl_utils.hpp"

#include "snde/mutablerecstore.hpp"

#include "snde/rec_display.hpp"

#include "snde/data_to_rgba.hpp"

#ifndef SNDE_OPENSCENEGRAPH_TEXTURE_HPP
#define SNDE_OPENSCENEGRAPH_TEXTURE_HPP

namespace snde {
  
class osg_texturecachekey
{
  // as a key it must be immutable...
public:
  std::weak_ptr<mutabledatastore> datastore;
  std::weak_ptr<display_channel> displaychan;  // displaychan structure with channel params to use

  osg_texturecachekey(std::weak_ptr<mutabledatastore> datastore, std::weak_ptr<display_channel> displaychan) :
    datastore(datastore),
    displaychan(displaychan)
  {

  }

  friend bool operator<(const osg_texturecachekey &l, const osg_texturecachekey &r)
  {
    if (l.datastore.owner_before(r.datastore)) return true;
    if (r.datastore.owner_before(l.datastore)) return false;

    return l.displaychan.owner_before(r.displaychan);
  }
};


  // !!!*** osg_texturecacheentry uses the image_data abstraction
  // with a method get_texture_image() to be able to get the
  // image data ... but nothing currently takes advantage of this,
  // because in openscenegraph_geom.hpp we need to extract the
  // texturestateset instead.... is the image_data abstraction
  // really necessary???

  
class osg_texturecacheentry : public std::enable_shared_from_this<osg_texturecacheentry>, public image_data {
  // The osg_texturecacheentry represents an entry in the osg_instancecache that will
  // automatically remove itself once all references are gone
  // (unless marked as "persistent" -- not yet implemented)
  // It does this by being referenced through shared_ptrs created with a custom deleter
  // When the shared_ptr reference count drops to zero, the custom deleter gets called,
  // thus removing us from the cache
public:
  std::weak_ptr<geometry> geom;
  std::weak_ptr<trm> rendering_revman;
  std::weak_ptr<mutablerecdb> recdb;

  cl_context context;
  cl_device_id device;
  cl_command_queue queue;

  std::weak_ptr<osg_texturecacheentry> thisptr; /* Store this pointer so we can return it on demand... must be created by osg_texturecache */
  std::shared_ptr<osg_texturecacheentry> persistentptr; /* Store pointer here if we want persistence (otherwise leave as nullptr */

  osg_texturecachekey key;
  
  osg::ref_ptr<osg::Texture2D> texture;
  osg::ref_ptr<osg::Image> image;
  osg::ref_ptr<osg::TexMat> texture_transform; // will be applied to the MatrixTransform 
  osg::ref_ptr<osg::StateSet> texture_state_set;
  osg::ref_ptr<osg::PixelBufferObject> image_pbo;
  std::shared_ptr<trm_dependency> rgba_dep;

  std::mutex texture_image_lock; // texture_image should probably be moved to be part of the geometry array data and its
  //index part of the image_data superclass. Then it can be locked like other data structures 
  std::shared_ptr<snde_image> texture_image;
  
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
   *   Meaningful U coordinate of IniVal1+(DimLen1-1+0.5)*Step1 should map to 1.0
   * Equations (for negative Step1): 
   *   Meaningful U coordinate of IniVal1+(Dimlen1-1+0.5)*Step1 should map to 0.0
   *   Meaningful U coordinate of IniVal1+(-0.5)*Step1 should map to 1.0

   * So the transform is strictly defined by the positioning and size of 
   * the parameterization channel.
   * Therefore it should be kept here, in the texture cache 
   * (Positive Step1):
   * The TexMat scaling will be 1.0/(Step1*DimLen1) and the offset will be:
   *      *      scaling*(IniVal1 - 0.5*Step1) + offset = 0.0
   *      *       offset = -scaling*(IniVal1-0.5*Step1)
   * (Negative Step1):
   * The TexMat scaling will be -1.0/(Step1*DimLen1) and the offset will be:
   *      *      scaling*(IniVal1 + (dimLen1-1+0.5)*Step1) + offset = 0.0
   *      *       offset = -scaling*(IniVal1+ (dimlen1-1+0.5)*Step1)



   */

  osg_texturecacheentry(const osg_texturecacheentry &)=delete; // no copy constructor
  osg_texturecacheentry & operator=(const osg_texturecacheentry &)=delete; // no copy assignment

  
  osg_texturecacheentry(std::shared_ptr<geometry> geom,std::shared_ptr<trm> rendering_revman,std::shared_ptr<mutablerecdb> recdb, cl_context context, cl_device_id device, cl_command_queue queue, osg_texturecachekey &key) :
    geom(geom),
    rendering_revman(rendering_revman),
    recdb(recdb),
    context(context),
    device(device),
    queue(queue),
    key(key)
  {
    texture = new osg::Texture2D();
    texture->setResizeNonPowerOfTwoHint(false);
    
    image = new osg::Image();
    texture_transform = new osg::TexMat();
    texture_state_set = new osg::StateSet();
    //texture_state_set->setMode(GL_DEPTH_TEST,osg::StateAttribute::ON);
    texture_state_set->setTextureAttributeAndModes(0,texture,osg::StateAttribute::ON);
    texture_state_set->setTextureAttributeAndModes(0,texture_transform,osg::StateAttribute::ON);
    texture->setImage(image);

    texture_image = std::make_shared<snde_image>(snde_image{.imgbufoffset=SNDE_INDEX_INVALID});
  
    image_pbo = new osg::PixelBufferObject();
    image_pbo->setImage(image);
    image->setPixelBufferObject(image_pbo);
    image->setDataVariance(osg::Object::DYNAMIC);

    // In general should call the Update() method immediately after creation and after the shared_ptr is created, but before it is
    // inserted in the actual cache itself
  }

  // get_texture_image returns read-only copy
  std::shared_ptr<snde_image> get_texture_image()
  {
    std::lock_guard<std::mutex> teximglock(texture_image_lock);
    return std::make_shared<snde_image>(*texture_image); 
  }

  std::shared_ptr<osg_texturecacheentry> lock()
  {
    return shared_from_this();
  }

  void Update()
  {
    // HOW SHOULD THIS BE LOCKED???
    std::shared_ptr<mutabledatastore> datastore;
    std::shared_ptr<display_channel> displaychan;  // displaychan structure with channel params to use
    std::shared_ptr<geometry> geom_strong(geom);

    if (!geom_strong) return;
    
    datastore = key.datastore.lock();
    displaychan = key.displaychan.lock();


    if (!datastore || !displaychan) return; // nothing to do if our "key" inputs have evaporated

    // Note: osg::Matrixd reads elements in column-major (Fortran) order

    double Step1,Step2,IniVal1,IniVal2;
    snde_index dimlen1,dimlen2;
    {
      rwlock_token_set datastore_tokens=empty_rwlock_token_set();
      geom_strong->manager->locker->get_locks_read_lockable(datastore_tokens,datastore);

      if (datastore->dimlen.size() < 2) return;
      Step1 = datastore->metadata.GetMetaDatumDbl("Step1",1.0);
      Step2 = datastore->metadata.GetMetaDatumDbl("Step2",-1.0);

      dimlen1 = datastore->dimlen[0];
      dimlen2 = datastore->dimlen[1];

      IniVal1 = datastore->metadata.GetMetaDatumDbl("IniVal1",-Step1*dimlen1/2.0);
      IniVal2 = datastore->metadata.GetMetaDatumDbl("IniVal2",Step2*dimlen2/2.0); 
    }

    
    // See comment above osg_texturecacheentry constructor for more
    // information on the texture transform and offsets
    double Offset1,Offset2;
    double Scaling1,Scaling2;

    Scaling1 = 1.0/(Step1*dimlen1);
    if (Step1 > 0.0) {
      Offset1 = -Scaling1*(IniVal1-0.5*Step1);
    } else {
      Offset1 = Scaling1*(IniVal1 +(dimlen1-1+0.5)*Step1);
    }

    Scaling2 = 1.0/(Step2*dimlen2);
    if (Step2 > 0.0) {
      Offset2 = -Scaling2*(IniVal2-0.5*Step2);
    } else {
      Offset2 = Scaling2*(IniVal2 +(dimlen2-1+0.5)*Step2);
    }

    texture_transform->setMatrix(osg::Matrixd(Scaling1, 0,0,0,
					      0,Scaling2, 0,0,
					      0,0,1,0,
					      Offset1,Offset2,0,1));
    
    //image->setImage(xsize,ysize,1,GL_RGBA,GL_RGBA,GL_UNSIGNED_BYTE,(unsigned char *)((*imagearray)+start),osg::Image::AllocationMode::NO_DELETE);
    //image->dirty();

    std::weak_ptr<osg_texturecacheentry> cacheentryweak = shared_from_this();

    std::shared_ptr<mutablerecdb> recdb_strong(recdb);
    std::shared_ptr<trm> revman_strong(rendering_revman);

    if (!recdb_strong || !revman_strong) return;
    
    if (!rgba_dep) {
      rgba_dep=CreateRGBADependency(revman_strong,
				    recdb_strong,
				    datastore->fullname,
				    geom_strong->manager,
				    (void **)&geom_strong->geom.texbuffer,
				    displaychan,
				    context,
				    device,
				    queue,
				    [ cacheentryweak ] (std::shared_ptr<lockholder> input_and_array_locks,rwlock_token_set all_locks,trm_arrayregion input,trm_arrayregion output,snde_rgba **imagearray,snde_index start,size_t xsize,size_t ysize,snde_coord2 inival, snde_coord2 step) {
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
				      
				      
				      std::shared_ptr<osg_texturecacheentry> cacheentrystrong=cacheentryweak.lock();
				      // release lock on input control data structure
				      //inputlock.unlock();
				      // 
				      
				      
				      if (cacheentrystrong) {
					//cacheentrystrong->imagetexture->setTextureWidth(xsize);
					//cacheentrystrong->imagetexture->setTextureHeight(ysize);
					
					/* ****!!!! WHY does OSG report (in Texture.cpp) scaling the image down from 300x300->256x256 ***???? */
					cacheentrystrong->image->setImage(xsize,ysize,1,GL_RGBA,GL_RGBA,GL_UNSIGNED_BYTE,(unsigned char *)((*imagearray)+start),osg::Image::AllocationMode::NO_DELETE);
					cacheentrystrong->texture->setImage(cacheentrystrong->image);
					fprintf(stderr,"First texture pixel: 0x%x 0x%x 0x%x 0x%x\n",
						(unsigned)*((unsigned char *)((*imagearray)+start)+0),
						(unsigned)*((unsigned char *)((*imagearray)+start)+1),
						(unsigned)*((unsigned char *)((*imagearray)+start)+2),
						(unsigned)*((unsigned char *)((*imagearray)+start)+3));

					{
					  std::lock_guard<std::mutex> teximg(cacheentrystrong->texture_image_lock);
					  cacheentrystrong->texture_image->imgbufoffset=start;
					  cacheentrystrong->texture_image->nx=xsize;
					  cacheentrystrong->texture_image->ny=ysize;
					  cacheentrystrong->texture_image->inival=inival;
					  cacheentrystrong->texture_image->step=step;
					}
					//cacheentrystrong->image->setFileName("Foo.png");
					//cacheentrystrong->imagestateset->setTextureAttributeAndModes(0,cacheentrystrong->imagetexture,osg::StateAttribute::ON);
					//cacheentrystrong->imagepbo->setImage(cacheentrystrong->image);
					//cacheentrystrong->image->setPixelBufferObject(cacheentrystrong->imagepbo);
					
					cacheentrystrong->image->dirty();
					fprintf(stderr,"Texture RGBA generated\n");
				      }
				    },
				    [ cacheentryweak ] (void) { //cleanup
				      std::shared_ptr<osg_texturecacheentry> cacheentrystrong=cacheentryweak.lock();
				      if (cacheentrystrong) {
					cacheentrystrong->image->setImage(0,0,1,GL_RGBA,GL_RGBA,GL_UNSIGNED_BYTE,nullptr,osg::Image::AllocationMode::NO_DELETE);
					
				      }
				    });
      
    }
  }

  virtual ~osg_texturecacheentry() { }
};




class osg_texturecache: public std::enable_shared_from_this<osg_texturecache> {
public:

  /* use an map because we can use it with a key that is really a std::weak_ptr  under the hood */
  std::map<osg_texturecachekey,osg_texturecacheentry> texture_cachedata;
  std::shared_ptr<geometry> snde_geom;
  std::shared_ptr<trm> rendering_revman;
  std::shared_ptr<mutablerecdb> recdb;
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  
  std::mutex admin; // serialize references to texture_cachedata because that could be used from any thread that drops the last reference to an texturecacheentry ... Need to think thread-safety of the instancecache through more carefully 


  osg_texturecache(std::shared_ptr<geometry> snde_geom,
		   std::shared_ptr<trm> rendering_revman,
		   std::shared_ptr<mutablerecdb> recdb,
		   cl_context context,
		   cl_device_id device,
		   cl_command_queue queue) :
    snde_geom(snde_geom),
    rendering_revman(rendering_revman),
    recdb(recdb),
    context(context),
    device(device),
    queue(queue)
  {
    
  }


  std::shared_ptr<osg_texturecacheentry> lookup(std::shared_ptr<mutabledatastore> texturedata,std::shared_ptr<display_channel> displaychan) 
  {
    std::map<osg_texturecachekey,osg_texturecacheentry>::iterator cache_entry;

    std::unique_lock<std::mutex> adminlock(admin);
    
    
    cache_entry = texture_cachedata.find(osg_texturecachekey(texturedata,displaychan));
    if (cache_entry==texture_cachedata.end()) {
      bool junk;
      osg_texturecachekey key(texturedata,displaychan);
      std::tie(cache_entry,junk) = texture_cachedata.emplace(std::piecewise_construct,
							     std::forward_as_tuple(key),
							     std::forward_as_tuple(snde_geom,rendering_revman,recdb,context,device,queue,key));

      std::shared_ptr<osg_texturecache> shared_cache = shared_from_this();
      
      // create shared pointer with custom deleter such that when
      // all references to this entry go away, we get called and can remove it
      // from the cache
	
      std::shared_ptr<osg_texturecacheentry> entry_ptr(&(cache_entry->second),
						       [ shared_cache ](osg_texturecacheentry *ent) { /* custom deleter... this is a parameter to the shared_ptr constructor, ... the osg_instancecachentry was created in emplace(), above.  */ 
							 std::map<osg_texturecachekey,osg_texturecacheentry>::iterator foundent;
							 
							 std::lock_guard<std::mutex> adminlock(shared_cache->admin);
							 
							 foundent = shared_cache->texture_cachedata.find(ent->key);
							 assert(foundent != shared_cache->texture_cachedata.end()); /* cache entry should be in cache */
							 assert(ent == &foundent->second); /* should match what we are trying to delete */
							  // Note: cacheentry destructor being called while holding adminlock!
							 shared_cache->texture_cachedata.erase(foundent); /* remove the element */ 
							 
						       } );
      // cacheentry requires Update() to be called once shared_ptr has been created
      entry_ptr->Update();
      
      return entry_ptr;
    } else {
      std::shared_ptr<osg_texturecacheentry> entry_ptr = cache_entry->second.lock();
      if (entry_ptr) {
	return entry_ptr;
      }
      else {
	// obsolete cache entry 
	texture_cachedata.erase(cache_entry);
	adminlock.unlock();
	// recursive call to make a new cache entry
	return lookup(texturedata,displaychan); 
	
      }
    }
  }

};

}

#endif // SNDE_OPENSCENEGRAPH_TEXTURE_HPP
