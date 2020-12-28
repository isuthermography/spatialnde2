#include <vector>
#include <set>
#include <string>

#include "lockmanager.hpp"
#include "infostore_or_component.hpp"
#include "wfmdb_paths.hpp"
#include "mutablewfmstore.hpp"

namespace snde {


  std::tuple<std::shared_ptr<iterablewfmrefs>,std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>>
  obtain_graph_lock_instances_multiple(std::shared_ptr<lockingprocess> process,
					 std::vector<std::tuple<snde_orientation3,std::string>> named_roots,
					 std::vector<std::tuple<snde_orientation3,std::shared_ptr<lockable_infostore_or_component>>> pointer_roots,
					 std::vector<std::string> extra_channels,
					 std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> extra_components,  // NOTE: Does NOT traverse the graph of extra_components (unless reached by other means)
					 std::shared_ptr<immutable_metadata> metadata,
				       std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablewfmrefs> wfmdb_wfmlist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names)> get_uv_imagedata,
					 std::shared_ptr<mutablewfmdb> wfmdb,
					 std::string wfmdb_context,
					 snde_infostore_lock_mask_t readmask,
					 snde_infostore_lock_mask_t writemask)
  {
    
    std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> component_set=extra_components;
    rwlock_token_set temporary_lock_pool;
    std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>> instancearray;
    std::shared_ptr<iterablewfmrefs> wfmdb_wfmlist;

    if (wfmdb) {
      wfmdb_wfmlist=wfmdb->wfmlist();
    }
      
    std::vector<std::tuple<snde_orientation3,std::shared_ptr<lockable_infostore_or_component>>> pointer_allroots=pointer_roots;
    for (auto & root: named_roots) {
      snde_orientation3 root_orient;
      std::string root_name;

      std::tie(root_orient,root_name) = root;
      std::string root_fullname = wfmdb_path_join(wfmdb_context,root_name);
      
      std::shared_ptr<lockable_infostore_or_component> root_comp = wfmdb_wfmlist->lookup(root_fullname);

      if (root_comp) {
	pointer_allroots.push_back(std::make_tuple(root_orient,root_comp));
      } else {
	fprintf(stderr,"infostore_or_component.cpp:obtain_graph_lock_instances_multiple(): locking root fullname \"%s\" not found in wfmdb\n",root_fullname);
      }
    }
    for (auto & root: pointer_allroots) {
      snde_orientation3 root_orient;
      std::shared_ptr<lockable_infostore_or_component> root_comp;

      std::tie(root_orient,root_comp) = root; 
      
      std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>> newinstancearray;

      // thisroot_metadata is private so is mutable until we are finished creating it 
      std::shared_ptr<immutable_metadata> thisroot_metadata;
      if (metadata) {
	thisroot_metadata = std::make_shared<immutable_metadata>(*metadata);
      } else {
	thisroot_metadata = std::make_shared<immutable_metadata>();
      }

      
      std::shared_ptr<mutableinfostore> root_infostore=std::dynamic_pointer_cast<mutableinfostore>(root_comp);
      //if (root_infostore) { // if we are actually an infostore...
	/* merge in metadata from this root and pass on. Entries in the preexisting metadata will override */
	/*  **** This is now done in mutablewfmstore.hpp/explore_component_get_instances() ****
	std::shared_ptr<immutable_metadata> root_metadata = root_infostore->metadata.metadata();
	
	for (auto & string_metadatum: root_metadata->metadata) {
	  if (thisroot_metadata->metadata.find(string_metadatum.first)==thisroot_metadata->metadata.end()) {
	    // if entry of this name not already present
	    fprintf(stderr,"Adding metadatum: %s\n",string_metadatum.first.c_str());
	    thisroot_metadata->metadata.emplace(string_metadatum);
	  }
	}
	*/
      //}
      
      
      newinstancearray = root_comp->explore_component_get_instances(component_set,                 // accumulate all subcomponents into component_set, which is sorted
								wfmdb_wfmlist,wfmdb_context,   // via owner_less, i.e. corresponding to the locking order
								root_orient,
								metadata,
								get_uv_imagedata);
      instancearray.insert(instancearray.end(),newinstancearray.begin(),newinstancearray.end());
      
    }
    
    for (auto & chan: extra_channels) {
      std::string chan_fullname = wfmdb_path_join(wfmdb_context,chan);
      component_set.emplace(wfmdb_wfmlist->lookup(chan_fullname));
    }
    
    
    temporary_lock_pool=process->begin_temporary_locking(lockingposition::lockingposition_before_lic());
    
    // First the components/infostores
    for (auto & comp: component_set) {
      //rwlock_token_set tokens_from_this_component=comp->obtain_lock(process,wfmdb,wfmdb_context,readmask & SNDE_INFOSTORE_ALL,writemask & SNDE_INFOSTORE_ALL,temporary=true);

      process->get_locks_lockable_mask_temporary(temporary_lock_pool,comp,comp->lic_mask,readmask,writemask);
    }
    
    // check for consistent geometry if we are locking SNDE_INFOSTORE_COMPONENTS|SNDE_INFOSTORE_INFOSTORES
    if (((readmask|writemask) & SNDE_INFOSTORE_COMPONENTS)  && ((readmask|writemask) & SNDE_INFOSTORE_INFOSTORES)) {
      bool consistent=false;
      
      do {
	std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> updated_component_set=extra_components;
	std::shared_ptr<iterablewfmrefs> new_wfmdb_wfmlist;

	if (wfmdb) {
	  new_wfmdb_wfmlist=wfmdb->wfmlist();
	}

	
	pointer_allroots=pointer_roots;
	
	for (auto & root: named_roots) {
	  snde_orientation3 root_orient;
	  std::string root_name;
	  
	  std::tie(root_orient,root_name) = root;
	  std::string root_fullname = wfmdb_path_join(wfmdb_context,root_name);
	  
	  std::shared_ptr<lockable_infostore_or_component> root_comp = new_wfmdb_wfmlist->lookup(root_fullname);
	  
	  if (root_comp) {	  
	    pointer_allroots.push_back(std::make_tuple(root_orient,root_comp));
	  } else {
	    fprintf(stderr,"infostore_or_component.cpp:obtain_graph_lock_instances_multiple(): locking root fullname \"%s\" not found in wfmdb\n",root_fullname);
	  }
	}
	
	
	instancearray.clear();
	
	for (auto & root: pointer_allroots) {
	  snde_orientation3 root_orient;
	  std::shared_ptr<lockable_infostore_or_component> root_comp;
	  
	  std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>> newinstancearray;
	  
	  std::tie(root_orient,root_comp) = root; 
	  
	  newinstancearray = root_comp->explore_component_get_instances(updated_component_set,                 // accumulate all subcomponents into component_set, which is sorted
								    new_wfmdb_wfmlist,wfmdb_context,   // via owner_less, i.e. corresponding to the locking order
								    root_orient,
								    metadata,
								    get_uv_imagedata);
	  instancearray.insert(instancearray.end(),newinstancearray.begin(),newinstancearray.end());
	  
	}
	for (auto & chan: extra_channels) {
	  std::string chan_fullname = wfmdb_path_join(wfmdb_context,chan);
	  updated_component_set.emplace(new_wfmdb_wfmlist->lookup(chan_fullname));
	}
	
	if (updated_component_set != component_set) {
	  // inconstent... unlock and re-lock
	  process->abort_temporary_locking(temporary_lock_pool);	    
	  temporary_lock_pool=process->begin_temporary_locking(lockingposition::lockingposition_before_lic());
	  
	  component_set = updated_component_set;
	  
	  for (auto & comp: component_set) {
	    process->get_locks_lockable_mask_temporary(temporary_lock_pool,comp,comp->lic_mask,readmask,writemask);
	  }
	  
	  
	} else {
	  consistent=true;
	  wfmdb_wfmlist = new_wfmdb_wfmlist;
	}
	  
      } while (!consistent);
    }
    process->finish_temporary_locking(lockingposition::lockingposition_before_arrays(),temporary_lock_pool);
    
    // Now the geometry (if applicable)
    if (readmask & SNDE_COMPONENT_GEOM_ALL || writemask & SNDE_COMPONENT_GEOM_ALL) { // if ANY geometry requested...
      for (auto & comp: component_set) {
	comp->obtain_geom_lock(process,wfmdb_wfmlist,wfmdb_context,readmask & SNDE_COMPONENT_GEOM_ALL,writemask & SNDE_COMPONENT_GEOM_ALL);
      }
    }
    
    // Now the uv parameterization data (if applicable)... all after the geometry in the data structures
    if (readmask & SNDE_UV_GEOM_ALL || writemask & SNDE_UV_GEOM_ALL) {
      // if ANY uv parameterization data requested
      for (auto & comp: component_set) {
	comp->obtain_uv_lock(process,wfmdb_wfmlist,wfmdb_context,readmask & SNDE_UV_GEOM_ALL,writemask & SNDE_UV_GEOM_ALL);
      }
    }
    
    return std::make_tuple(wfmdb_wfmlist,instancearray);
    
  }
  

  

};
