#include <cstdint>
#include <memory>


#include "snde/geometry_types.h"
#include "snde/arraymanager.hpp"
#include "snde/geometrydata.h"
#include "snde/geometry.hpp"
#include "snde/mutablerecstore.hpp"

namespace snde {
  //  component::component() {};
  
  component::~component()
#if !defined(_MSC_VER) || _MSC_VER > 1800 // except for MSVC2013 and earlier
    noexcept(false)
#endif
  {

  }

  //uv_image::uv_image(std::shared_ptr<uv_images> images,
  //snde_index imageidx) :
  //  geom(images->geom),
  //  firstuvimage(images->firstuvimage),
  //  imageidx(imageidx)
  //{
  // 
  //}



    std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>
    part::explore_component_get_instances(std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> &component_set,
				    std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,
				    snde_orientation3 orientation,
				    std::shared_ptr<immutable_metadata> metadata,
				    std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names)> get_uv_imagedata)
    {
      
      std::shared_ptr<lockable_infostore_or_component> our_ptr=shared_from_this();

      if (component_set.find(our_ptr)==component_set.end()) {
	component_set.emplace(our_ptr);
	
      }

      //// also explore parameterizations */
      //for (auto && paramname_parameterization: *parameterizations()) {
      //  paramname_parameterization._explore_component(component_set,recdb_reclist,recdb_context);
      //  //component_set.emplace(paramname_parameterization.second);
      //}


      struct snde_partinstance ret=snde_partinstance{ .orientation=orientation,
	                                              .partnum = idx(),
						      .firstuvimage=SNDE_INDEX_INVALID,
						      .uvnum=SNDE_INDEX_INVALID,};
      std::shared_ptr<part> ret_ptr;

      ret_ptr = std::dynamic_pointer_cast<part>(our_ptr);

      std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>> ret_vec;
      

      // Look up possible uv_imagedata sources
      std::vector<std::string> uv_imagedata_names;
      if (metadata) {
	fprintf(stderr,"Got non-null metadata (%d entries)\n",(int)metadata->metadata.size());
	
	std::string uv_imagedata_channels=metadata->GetMetaDatumStr("uv_imagedata_channels","");
	// split comma-separated list of uv_imagedata_names
	
	char *uv_imagedata_channels_c=strdup(uv_imagedata_channels.c_str());
	char *saveptr=NULL;
	for (char *tok=strtok_r(uv_imagedata_channels_c,",",&saveptr);tok;tok=strtok_r(NULL,",",&saveptr)) {
	  fprintf(stderr,"part::explore_component_get_instances: Got uv_imagedata_name %s\n",tok);
	  uv_imagedata_names.push_back(stripstr(tok));
	}
	
	::free(uv_imagedata_channels_c); // :: means search in the global namespace for cstdlib free
      }
      std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>> uv_imagedata = get_uv_imagedata(recdb_reclist,std::dynamic_pointer_cast<part>(shared_from_this()),uv_imagedata_names);
      
      std::shared_ptr<parameterization> param;
      std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>> uv_imagedata_map;
      std::tie(param,uv_imagedata_map) = uv_imagedata;
      
      if (param) {
	// should we pre-process (reduce) metadata?
	fprintf(stderr,"got param...\n");
	// we ignore the return because a parameterization can't link to components with instances
	param->explore_component_get_instances(component_set,recdb_reclist,recdb_context,orientation,metadata,get_uv_imagedata);
      } else {
	fprintf(stderr,"got no param.\n");
      }
      fprintf(stderr,"%d entries in uv_imagedata_map\n",(int)uv_imagedata_map.size());
      
      for (auto & index_datastore_imagedata : uv_imagedata_map) {
	snde_index uv_index;
	std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>> datastore_imagedata;

	std::tie(uv_index,datastore_imagedata) = index_datastore_imagedata;

	std::shared_ptr<mutabledatastore> datastore;
	std::shared_ptr<image_data> imagedata;

	std::tie(datastore,imagedata) = datastore_imagedata;

	// should we pre-process (reduce) metadata?
	// we ignore the return because uv imagedata can't link to components with instances
	datastore->explore_component_get_instances(component_set,recdb_reclist,recdb_context,orientation,metadata,get_uv_imagedata);
      }
      
      ret_vec.push_back(std::tuple_cat(std::make_tuple(ret,ret_ptr),uv_imagedata));
      return ret_vec;

      
    }

  std::shared_ptr<mutableparameterizationstore> part::addparameterization(std::shared_ptr<mutablerecdb> recdb,std::string recdb_context,std::shared_ptr<snde::parameterization> parameterization,std::string name,const recmetadata &metadata)
    // Component must be locked for write
  // Given the parameterization object, this creates a named recording in the given context, adds the parameterization to the recording dictionary, and  references this recording
  // in this part.
    {
      std::shared_ptr<mutableparameterizationstore> infostore=std::make_shared<mutableparameterizationstore>(name,recdb_path_join(recdb_context,name),metadata,geom,parameterization);
      recdb->addinfostore(infostore);
      std::shared_ptr<std::set<std::string>> updated_parameterizations = _begin_atomic_parameterizations_update();
      updated_parameterizations->emplace(name);
      _end_atomic_parameterizations_update(updated_parameterizations);
      return infostore; 
    }

  
  /*  std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>> reccomponent::get_instances(std::shared_ptr<mutablerecdb> recdb,std::string recdb_context,snde_orientation3 orientation, std::shared_ptr<immutable_metadata> metadata, std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>(std::shared_ptr<part> partdata,std::vector<std::string> parameterization_data_names)> get_param_data)
    {

      std::string full_path=recdb_path_join(recdb_context,path);
      std::shared_ptr<mutablegeomstore> geomstore = std::dynamic_pointer_cast<mutablegeomstore>(recdb->lookup(full_path));
      if (!geomstore) {
	return std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::shared_ptr<image_data>>>>();
      }
      return geomstore->comp->get_instances(recdb,full_path,orientation,metadata,get_param_data);
    }
  */
  
  void reccomponent::obtain_geom_lock(std::shared_ptr<lockingprocess> process, std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,snde_infostore_lock_mask_t readmask,snde_infostore_lock_mask_t writemask,snde_infostore_lock_mask_t resizemask) /* writemask contains OR'd SNDE_COMPONENT_GEOM_xxx bits */
    {
      // readmask and writemask contain OR'd SNDE_COMPONENT_GEOM_xxx bits 

      //
      //	 obtain locks from all our components... 
      //	 These have to be spawned so they can all obtain in parallel, 
      //	 following the locking order. 

      //	 NOTE: You must have at least read locks on  all the components OR 
      //	 readlocks on the object trees lock while this is executing!

      // Nothing to do here as the traversal will have reached the underlying component geometry separately
      
      /*
      std::string full_path=recdb_path_join(recdb_context,path);
      std::shared_ptr<mutablegeomstore> geomstore = std::dynamic_pointer_cast<mutablegeomstore>(recdb->lookup(full_path));
      if (geomstore) {
	geomstore->comp->obtain_geom_lock(process,recdb,full_path,readmask,writemask,resizemask);
      } else {
	fprintf(stderr,"Warning: reccomponent:obtain_geom_lock(): geometry store %s not found\n",(char *)full_path.c_str());
      }
      */
    }
 
  
  std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>
  reccomponent::explore_component_get_instances(std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> &component_set,
				     std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,
				     snde_orientation3 orientation,
				     std::shared_ptr<immutable_metadata> metadata,
						std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> parameterization_data_names)> get_uv_imagedata)  
    {
      // should be holding SNDE_INFOSTORE_OBJECT_TREES as at least read in order to do the _explore()
      std::shared_ptr<component> our_ptr=std::dynamic_pointer_cast<component>(shared_from_this());
      
      component_set.emplace(our_ptr);      

      std::string full_path=recdb_path_join(recdb_context,path);
      std::shared_ptr<mutablegeomstore> geomstore = std::dynamic_pointer_cast<mutablegeomstore>(recdb_reclist->lookup(full_path));
     
      if (geomstore) {
	return geomstore->explore_component_get_instances(component_set,recdb_reclist,full_path,
							  orientation,
							  metadata,
							  get_uv_imagedata);
      } else {
	fprintf(stderr,"Warning: reccomponent:_explore_component: geometry store %s not found\n",(char *)full_path.c_str());
	return std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>();
      }

    }



  std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>
  assembly::explore_component_get_instances(std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> &component_set,
				    std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,
				    snde_orientation3 orientation,
				    std::shared_ptr<immutable_metadata> metadata,
				    std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names)> get_uv_imagedata)
    {
      std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>> instances;
      std::shared_ptr<component> our_ptr=std::dynamic_pointer_cast<component>(shared_from_this());
      
      component_set.emplace(our_ptr);
      std::unordered_map<std::string,int> name_counts;
      
      for (auto & piece: *pieces()) {
	// let our sub-components add themselves
	std::string piece_name;
	snde_orientation3 piece_orient;
	  
	std::tie(piece_name,piece_orient) = piece;

	std::unordered_map<std::string,int>::iterator name_counts_entry;
	bool junkbool;
	std::tie(name_counts_entry,junkbool) = name_counts.emplace(piece_name,0);

	name_counts_entry->second++;

	std::string suffix="";
	if (name_counts_entry->second > 1) {
	  suffix=std::to_string(name_counts_entry->second);
	}
	
	std::shared_ptr<immutable_metadata> reduced_metadata=reduce_partspecific_metadata(metadata,piece_name+suffix);
	
	snde_orientation3 neworientation;
	orientation_orientation_multiply(orientation,piece_orient,&neworientation);


	std::shared_ptr<mutableinfostore> piece_infostore = recdb_reclist->lookup(recdb_path_join(recdb_context,piece_name));
	if (piece_infostore) {
	  
	
	  std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>  newinstances 
	    = piece_infostore->explore_component_get_instances(component_set,recdb_reclist,recdb_context,
							      neworientation,
							      reduced_metadata,
							      get_uv_imagedata);
	  instances.insert(instances.end(),newinstances.begin(),newinstances.end());
	} else {
	  fprintf(stderr,"WARNING: geometry.hpp/snde::assembly::explore_component_get_instances: No infostore for \"%s\" in context \"%s\" found.\n",piece_name.c_str(),recdb_context.c_str());
	}
      }
      
      return instances;
       
    }

  void assembly::obtain_geom_lock(std::shared_ptr<lockingprocess> process, std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,snde_infostore_lock_mask_t readmask,snde_infostore_lock_mask_t writemask,snde_infostore_lock_mask_t resizemask)
    {
      /* readmask and writemask contain OR'd SNDE_COMPONENT_GEOM_xxx bits */

      /* 
	 obtain locks from all our components... 
	 These have to be spawned so they can all obtain in parallel, 
	 following the locking order. 

	 NOTE: You must have at least read locks on  all the components OR 
	 readlocks on the object trees lock while this is executing!
      */
      std::shared_ptr<const std::vector<std::tuple<std::string,snde_orientation3>>> pieces_ptr=pieces();
	
      for (auto piece=pieces_ptr->begin();piece != pieces_ptr->end(); piece++) {
	std::string piece_name;
	snde_orientation3 piece_orientation;
	
	std::tie(piece_name,piece_orientation) = *piece;

	std::shared_ptr<mutableinfostore> piece_infostore = recdb_reclist->lookup(recdb_path_join(recdb_context,piece_name));
	std::shared_ptr<mutablegeomstore> piece_geomstore;
	std::shared_ptr<component> pieceptr;
	
	if (piece_infostore) {
	  piece_geomstore=std::dynamic_pointer_cast<mutablegeomstore>(piece_infostore);
	  if (piece_geomstore) {
	    pieceptr=piece_geomstore->comp(); // get snde::component pointer
	  } else {
	    fprintf(stderr,"Warning: Assembly geometry entry \"%s\" within context \"%s\" not a geometry store.\n",piece_name.c_str(),recdb_context.c_str());
	  }
	} else {
	  fprintf(stderr,"Warning: Assembly geometry entry \"%s\" within context \"%s\" not found.\n",piece_name.c_str(),recdb_context.c_str());
	}

	if (pieceptr) {
	  process->spawn([ process, recdb_reclist, recdb_context, pieceptr, readmask, writemask, resizemask ]() { pieceptr->obtain_geom_lock(process,recdb_reclist,recdb_context,readmask,writemask,resizemask); } );
	}
      }
      
    }


  
};
