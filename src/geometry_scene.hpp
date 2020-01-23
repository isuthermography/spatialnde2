#include "wfm_display.hpp"

#ifndef SNDE_GEOMETRY_SCENE_HPP
#define SNDE_GEOMETRY_SCENE_HPP



namespace snde {

  
  class geometry_scene {
  public:
    
    std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>> instances; // The part indexes the 3D geometry, the parameterization indexes the 2D surface parameterization of that geometry, and the image_data provides the parameterized 2D data
    rwlock_token_set scene_lock; // the locks that hold this scene fixed while we use it

    std::set<std::string> channels_locked;

    geometry_scene()
    {
      
    }

    geometry_scene(const std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>> &instances,
		   rwlock_token_set scene_lock,
		   const std::set<std::string> & channels_locked) :
      instances(instances),
      scene_lock(scene_lock),
      channels_locked(channels_locked)
    {

    }


    void drop_locks()
    {
      scene_lock->clear(); 
    }

    static geometry_scene lock_scene(std::shared_ptr<lockmanager> locker,
				     std::shared_ptr<mutablewfmdb> wfmdb,
				     std::function<std::tuple<std::shared_ptr<component>,std::shared_ptr<immutable_metadata>,std::set<std::string>>()> get_component_metadata_and_extra_channels,
				     std::function<std::shared_ptr<image_data>(std::shared_ptr<mutabledatastore>,std::string)> get_image_data, // return pointers to image_data structures that will be included in the instances member of the returned geometry scene. The actual content of these image_data structures does not need to be ready. 
				     const std::set<std::string> &last_channels_to_lock=std::set<std::string>())
    // There's a bit of weirdness here because of the locking order:
    //  * The locking order of the channels in the wfmdb is arbitrary and
    //    __precedes__ comp in the locking order.
    //  * That means comp can NOT be locked going into this, because in locking
    //    the scene you need to lock textures, which are stored in channels
    //  * Also the caller might be dependent on information from other channels
    //    that need to be locked as well.
    //  * So the approach here is that the caller can provide a function
    //    that returns a set 9by name)
    //    of extra channels that need locked. The provided function must be callable
    //    whether or not the channels are locked. 
    //  * lock_scene()  will use the provided function to identify what needs to be
    //    locked, and then once they are locked call it again to verify that
    //    the set hasn't changed. If there is anything new in the set, then
    //    all locks will be dropped and lock_scene will try again to do the locking.
    // Separately last_channels_to_lock is a hint of which channels needed to be locked
    // the last time around. Any channel specified in that set may be locked in
    // addition to those actually required for the scene.
    //
    // Note also that this locks a single component. If you need multiple components, create
    // a private assembly that contains them. 
    //
    // This locks the wfmdb entries and channels, but NOT the underlying data. The underlying
    // data (geometrydata.h) is later in the locking order than the wfmdb entries and channels,
    // so it can still be locked after this returns. 
    {

      std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>> instances; // The part indexes the 3D geometry, the parameterization indexes the 2D surface parameterization of that geometry, and the image_data provides the parameterized 2D data 
      
      std::set<std::string> channels_to_lock;  // use std::set instead of std::unordered_set so we can compare them with operator==()
      std::set<std::string> new_channels_to_lock;

      std::set<std::string> extra_channels;
      std::shared_ptr<component> comp,newcomp;
      rwlock_token_set all_locks;

      std::shared_ptr<immutable_metadata> metadata,newmetadata;
      
      std::tie(newcomp,newmetadata,extra_channels)=get_component_metadata_and_extra_channels();

      new_channels_to_lock = last_channels_to_lock;

      // merge in extra_channels to new_channels_to_lock 
      new_channels_to_lock.insert(extra_channels.begin(),extra_channels.end());
      
      // repeatedly try to get instance data until channels_to_lock and new_channels_to_lock converge
      // with at least the right channels' metadata locked
      do {

	// clear out all_locks
	all_locks.reset();
	
	// Update what we are supposed to lock
	channels_to_lock=new_channels_to_lock;
	metadata=newmetadata;
	comp=newcomp;

	// start rebuild new_channels_to_lock (will continue below)
	new_channels_to_lock.clear();
	

	// clear out any old instances
	instances.clear();
	
	std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(locker); // new locking process
	
	
	// read lock on all these infostores
	// ***!!! We should probably handle the exception where
	// one of these channels_to_lock has just disappeared!
	lockprocess->lock_infostores(wfmdb,channels_to_lock,false);
	//// lock access to comp mutablegeomstore
	//rwlock_token_set all_locks=empty_rwlock_token_set();
	//get_locks_read_infostore(all_locks,std::static_pointer_cast<mutableinfostore>(comp));
	
	if (comp) {
	  
	  comp->obtain_lock(lockprocess,SNDE_INFOSTORE_COMPONENTS);
	}
	all_locks = lockprocess->finish();

	if (comp) {
	  snde_orientation3 null_orientation;
	  snde_null_orientation3(&null_orientation);
	  
	  instances = comp->get_instances(null_orientation,metadata,[ &new_channels_to_lock, wfmdb, get_image_data ] (std::shared_ptr<part> partdata,std::vector<std::string> parameterization_data_names) -> std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>> {
	      
	      std::map<snde_index,std::shared_ptr<image_data>> images_out;
	      
	      
	      // NOTE: Parameterization_data_names is unordered... need to get the face number(s) from the uv_parameterization_facenum metadata
	      std::shared_ptr<parameterization> use_param;
	      
	      // parameterization data are stored in the channels given by the parameterization_data_names parameter to this lambda
	      
	      for (auto & wfmname: parameterization_data_names) {	  
		std::shared_ptr<mutableinfostore> paraminfo = wfmdb->lookup(wfmname);
		
		// Add this parameterizatin channel to the list of what needs locked
		new_channels_to_lock.emplace(std::string(wfmname));
		
		if (!paraminfo) {
		  continue;
		}
		std::shared_ptr<mutabledatastore> paramdata=std::dynamic_pointer_cast<mutabledatastore>(paraminfo);
		if (!paramdata) {
		  // must be a data channel
		  continue;
		}
		
		// look up parameterization
		std::string parameterization_name = paramdata->metadata.GetMetaDatumStr("uv_parameterization","intrinsic");
		
		std::map<std::string,std::shared_ptr<parameterization>>::iterator gotparam=partdata->parameterizations.find(parameterization_name);
		if (gotparam == partdata->parameterizations.end()) {
		  fprintf(stderr,"lock_scene(): Unknown parameterization %s specified in channel %s\n",parameterization_name,paramdata->fullname);
		  continue; 
		}
		if (!use_param) {
		  use_param=gotparam->second;
		} else {
		  if (gotparam->second != use_param) {
		    fprintf(stderr,"lock_scene(): Warning: inconsistent parameterizations specified (including %s) in channel %s\n",parameterization_name,paramdata->fullname);
		    continue; 
		  }
		}
		

		std::shared_ptr<image_data> texinfo = get_image_data(paramdata,wfmname);
		
		//std::shared_ptr<snde_image> teximage = texinfo->get_texture_image();
		
		// ***!!!! Should really accept a comma separated array of facenums here. right now we have it hotwired so that
		// if uv_parameterization_imagenum is unset it will be interpreted as matching every face OK!
		std::string parameterization_imagenums_str = paramdata->metadata.GetMetaDatumStr("uv_parameterization_imagenums","");
		if (parameterization_imagenums_str=="") {
		  // interpret blank as 0
		  images_out.emplace(0,texinfo);
		} else {
		  char *parameterization_imagenums_tokenized=strdup(parameterization_imagenums_str.c_str());
		  char *saveptr=NULL;
		  
		  for (char *tok=strtok_r(parameterization_imagenums_tokenized,",",&saveptr);tok;tok=strtok_r(NULL,",",&saveptr)) {
		    snde_index parameterization_imagenum = strtoul(stripstr(tok).c_str(),NULL,10);
		    images_out.emplace(parameterization_imagenum,texinfo);
		  
		  }
		  free(parameterization_imagenums_tokenized);
		  
		  
		}
		
		// ***!!!!! NOT CURRENTLY DOING ANYTHING WITH teximage
		
		
	      }
	      //std::vector<std::tuple<std::shared_ptr<image_data>,std::shared_ptr<snde_image>>> images_outvec;
	      return std::make_tuple(use_param,images_out);
	    });
	}
	
	

	// re-extract component, metadata, and extra_channels
	// now that stuff is locked. If there is a mismatch,
	// we will have to loop back. 
	
	std::tie(newcomp,newmetadata,extra_channels)=get_component_metadata_and_extra_channels();
	// merge in extra_channels to new_channels_to_lock 
	new_channels_to_lock.insert(extra_channels.begin(),extra_channels.end());
	
	
	
      } while (newcomp != comp || newmetadata != metadata || !std::includes(channels_to_lock.begin(),channels_to_lock.end(),new_channels_to_lock.begin(),new_channels_to_lock.end()));  // while comp is wrong or channels_to_lock does not include everything in new_channels_to_lock
 
      
    
      return geometry_scene(instances,all_locks,channels_to_lock);
    }
    
    static geometry_scene lock_scene(std::shared_ptr<lockmanager> locker,
				     std::shared_ptr<mutablewfmdb> wfmdb,
				     std::function<std::shared_ptr<image_data>(std::shared_ptr<mutabledatastore>,std::string)> get_image_data,
				     std::string chan_fullname,const std::set<std::string> &last_channels_to_lock=std::set<std::string>())
    {
      
      return lock_scene(locker,wfmdb,[ wfmdb, chan_fullname ] () -> std::tuple<std::shared_ptr<component>,std::shared_ptr<immutable_metadata>,std::set<std::string>> {
	  std::shared_ptr<mutableinfostore> infostore;
	  std::shared_ptr<mutablegeomstore> geomstore;
	  std::set<std::string> chan_names;
	  std::shared_ptr<immutable_metadata> metadata;
	  
	  chan_names.emplace(chan_fullname);
	  
	  infostore=wfmdb->lookup(chan_fullname);
	  if (infostore) {
	    geomstore=std::dynamic_pointer_cast<mutablegeomstore>(infostore);
	    metadata=infostore->metadata.metadata();
	    if (geomstore) {
	      return std::make_tuple(geomstore->comp,metadata,chan_names);
	    }
	  }

	  return std::make_tuple(std::shared_ptr<component>(),std::shared_ptr<immutable_metadata>(),chan_names);
	},get_image_data,last_channels_to_lock);
    }
    
  };
  
  
};




#endif // SNDE_GEOMETRY_SCENE_HPP
