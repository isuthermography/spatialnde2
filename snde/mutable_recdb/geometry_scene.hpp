#include "snde/rec_display.hpp"

#ifndef SNDE_GEOMETRY_SCENE_HPP
#define SNDE_GEOMETRY_SCENE_HPP



namespace snde {

  
  class geometry_scene {
  public:
    
    std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>> instances; // The part indexes the 3D geometry, the parameterization indexes the 2D surface parameterization of that geometry, and the image_data provides the parameterized 2D data
    rwlock_token_set scene_lock; // the locks that hold this scene fixed while we use it

    std::set<std::string> channels_locked;

    geometry_scene()
    {
      
    }

    geometry_scene(const std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>> &instances,
		   rwlock_token_set scene_lock) :
		   //const std::set<std::string> & channels_locked) :
      instances(instances),
      scene_lock(scene_lock)
      //channels_locked(channels_locked)
    {

    }


    void drop_locks()
    {
      scene_lock->clear(); 
    }

    static geometry_scene lock_scene(std::shared_ptr<lockmanager> locker,
				     std::shared_ptr<mutablerecdb> recdb,
				     std::string recdb_context,
				     std::string recdb_chan_name,
				     std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> extra_components,
				     //std::function<std::tuple<std::shared_ptr<component>,std::shared_ptr<immutable_metadata>,std::set<std::string>>()> get_component_metadata_and_extra_channels,
				     std::function<std::shared_ptr<image_data>(std::shared_ptr<mutabledatastore>,std::string)> get_image_data) // return pointers to image_data structures that will be included in the instances member of the returned geometry scene. The actual content of these image_data structures does not need to be ready. 

    // lock_scene() locks the geometry scene data structures (assembly, part, etc.) of the component returned from
    // get_component_metadata_and_extra_channels() as well as the other channels (but not including any geometry)
    // specified in the string set returned by get_component_metadata_and_extra_channels().
    //
    // Does NOT lock the underlying geometry data... just the C++ data structures. 
      
    //  * So in general this locks all of the specified channels, then all the
    //    geometry data within them, following the locking order. 
    //
    // Note also that this locks a single component. If you need multiple components, create
    // a private assembly that contains them. 
    //
    // This locks the recdb entries and channels, but NOT the underlying data. The underlying
    // data (geometrydata.h) is later in the locking order than the recdb entries and channels,
    // so it can still be locked after this returns. 
    {


      
      std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>> instances; // The part indexes the 3D geometry, the parameterization indexes the 2D surface parameterization of that geometry, and the image_data provides the parameterized 2D data 
      
      std::set<std::string> channels_to_lock;  // use std::set instead of std::unordered_set so we can compare them with operator==()
      std::set<std::string> new_channels_to_lock;

      //std::set<std::string> extra_channels;
      //std::shared_ptr<component> comp;
      rwlock_token_set all_locks;

      std::shared_ptr<immutable_metadata> metadata;
      
      //std::tie(comp,metadata,extra_channels)=get_component_metadata_and_extra_channels();

      //std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> roots;
      //roots.emplace(comp);

	
      std::shared_ptr<lockingprocess_threaded> lockprocess=std::make_shared<lockingprocess_threaded>(locker); // new locking process
      snde_orientation3 null_orientation;
      snde_null_orientation3(&null_orientation);
      
      std::shared_ptr<iterablerecrefs> recdb_reclist;
	
      std::tie(recdb_reclist,instances) =
	obtain_graph_lock_instances(lockprocess,
				    std::make_tuple(null_orientation,recdb_chan_name),
				    std::vector<std::string>(), // extra_channels
				    extra_components,
				    std::shared_ptr<immutable_metadata>(),
				    [ recdb_context, get_image_data ] (std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names) -> std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>> {
				      std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>> images_out;
				      
				    
				      // NOTE: uv_imagedata_names is unordered... need to get the face number(s) from the uv_parameterization_facenum metadata
				      std::shared_ptr<parameterization> use_param;
				      std::string use_param_name;
				      // uv image data are stored in the channels given by the uv_imagedata_names parameter to this lambda
				      
				      for (auto & recname: uv_imagedata_names) {
					fprintf(stderr,"Got uv_imagedata name: \"%s\"\n",recname.c_str());
					std::string full_recname=recdb_path_join(recdb_context,recname);;
					
					
					std::shared_ptr<mutableinfostore> uv_imagedata_info = recdb_reclist->lookup(full_recname);
					
					
					if (!uv_imagedata_info) {
					  continue;
					}
					std::shared_ptr<mutabledatastore> uv_image_data=std::dynamic_pointer_cast<mutabledatastore>(uv_imagedata_info);
					if (!uv_image_data) {
					  // must be a data channel
					  continue;
					}
					
					// look up parameterization
					std::string parameterization_name = uv_image_data->metadata.GetMetaDatumStr("uv_parameterization","intrinsic");
					std::shared_ptr<const std::set<std::string>> parameterizations=partdata->parameterizations();
					std::set<std::string>::const_iterator gotparam=parameterizations->find(parameterization_name);
					if (gotparam == parameterizations->end()) {
					  fprintf(stderr,"lock_scene(): Unknown parameterization %s specified in channel %s\n",parameterization_name,uv_image_data->fullname);
					  continue; 
					}
					std::shared_ptr<mutableinfostore> paraminfostore=recdb_reclist->lookup(recdb_path_join(recdb_context,parameterization_name));
					std::shared_ptr<mutableparameterizationstore> paramstore;
					if (paraminfostore) {					  
					  paramstore=std::dynamic_pointer_cast<mutableparameterizationstore>(paraminfostore);
					}
					
					if (!paramstore) {
					  fprintf(stderr,"lock_scene(): Unable to load parameterization %s specified in channel %s\n",parameterization_name,uv_image_data->fullname);
					  continue;
					}
					
					
					if (!use_param) {
					  use_param=paramstore->param();
					  use_param_name = recdb_path_join(recdb_context,parameterization_name);
					} else {
					  if (recdb_path_join(recdb_context,parameterization_name) != use_param_name) {
					    fprintf(stderr,"lock_scene(): Warning: inconsistent parameterizations specified (including %s) in channel %s\n",parameterization_name,uv_image_data->fullname);
					    continue; 
					  }
					}
					
					
					std::shared_ptr<image_data> texinfo = get_image_data(uv_image_data,recname);
					
					//std::shared_ptr<snde_image> teximage = texinfo->get_texture_image();
					
					// ***!!!! Should really accept a comma separated array of facenums here. right now we have it hotwired so that
					// if uv_imagedata_imagenum is unset it will be interpreted as matching every face OK!
					std::string uv_imagedata_imagenums_str = uv_image_data->metadata.GetMetaDatumStr("uv_imagedata_imagenums","");
					if (uv_imagedata_imagenums_str=="") {
					  // interpret blank as 0
					  images_out.emplace(0,std::make_tuple(uv_image_data,texinfo));
					} else {
					  char *uv_imagedata_imagenums_tokenized=strdup(uv_imagedata_imagenums_str.c_str());
					  char *saveptr=NULL;
					  
					  for (char *tok=strtok_r(uv_imagedata_imagenums_tokenized,",",&saveptr);tok;tok=strtok_r(NULL,",",&saveptr)) {
					    snde_index uv_imagedata_imagenum = strtoul(stripstr(tok).c_str(),NULL,10);
					    images_out.emplace(uv_imagedata_imagenum,std::make_tuple(uv_image_data,texinfo));
					    
					  }
					  free(uv_imagedata_imagenums_tokenized);
					  
					  
					}
					
					// ***!!!!! NOT CURRENTLY DOING ANYTHING WITH teximage
					
					
				      }
				      //std::vector<std::tuple<std::shared_ptr<image_data>,std::shared_ptr<snde_image>>> images_outvec;
				      return std::make_tuple(use_param,images_out);
				      
				    },
				    recdb,
				    recdb_context,
				    SNDE_INFOSTORE_ALL,
				    0);
      
      all_locks = lockprocess->finish();

	

	// re-extract component, metadata, and extra_channels
	// now that stuff is locked. If there is a mismatch,
	// we will have to loop back. 
	
      //std::tie(newcomp,newmetadata,extra_channels)=get_component_metadata_and_extra_channels();
      // merge in extra_channels to new_channels_to_lock 
      //new_channels_to_lock.insert(extra_channels.begin(),extra_channels.end());
	
	
    
      
	
      return geometry_scene(instances,all_locks);
    }
    
    /*
    static geometry_scene lock_scene(std::shared_ptr<lockmanager> locker,
				     std::shared_ptr<mutablerecdb> recdb,
				     std::function<std::shared_ptr<image_data>(std::shared_ptr<mutabledatastore>,std::string)> get_image_data,
				     std::string chan_fullname,const std::set<std::string> &last_channels_to_lock=std::set<std::string>())
    {
      
      return lock_scene(locker,recdb,[ recdb, chan_fullname ] () -> std::tuple<std::shared_ptr<component>,std::shared_ptr<immutable_metadata>,std::set<std::string>> {
	  std::shared_ptr<mutableinfostore> infostore;
	  std::shared_ptr<mutablegeomstore> geomstore;
	  std::set<std::string> chan_names;
	  std::shared_ptr<immutable_metadata> metadata;
	  
	  chan_names.emplace(chan_fullname);
	  
	  infostore=recdb->lookup(chan_fullname);
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
    */ 
  };
  
  
};




#endif // SNDE_GEOMETRY_SCENE_HPP
