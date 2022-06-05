#include <functional>
#include <string>
#include <unordered_set>
#include <memory>
#include <map>

#include "snde/recstore.hpp"
#include "snde/recmath.hpp"

#include "snde/graphics_recording.hpp"

#include "snde/geometry_processing.hpp"

namespace snde {

  typedef std::map<std::string,geomproc_instantiator> geomproc_instantiator_map;

  static std::shared_ptr<geomproc_instantiator_map> *_geomproc_instantiator_registry; // default-initialized to nullptr


  static std::mutex &geomproc_instantiator_registry_mutex()
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
  
  static std::shared_ptr<geomproc_instantiator_map> geomproc_instantiator_registry()
  {
    std::mutex &regmutex = geomproc_instantiator_registry_mutex();
    std::lock_guard<std::mutex> reglock(regmutex);

    if (!_geomproc_instantiator_registry) {
      _geomproc_instantiator_registry = new std::shared_ptr<geomproc_instantiator_map>(std::make_shared<geomproc_instantiator_map>());
    }
    return *_geomproc_instantiator_registry;
  }
  
  int register_geomproc_math_function(std::string tagname,geomproc_instantiator instantiator)
  {
    geomproc_instantiator_registry(); // ensure that the registry pointer exists

    std::mutex &regmutex = geomproc_instantiator_registry_mutex();
    std::lock_guard<std::mutex> reglock(regmutex);
    
    // copy map and update then publish the copy
    std::shared_ptr<geomproc_instantiator_map> new_map = std::make_shared<geomproc_instantiator_map>(**_geomproc_instantiator_registry);
    
    new_map->emplace(tagname,instantiator);

    *_geomproc_instantiator_registry = new_map;
    return 0;
    
    
  }

  

  bool extract_geomproc_option(std::unordered_set<std::string> *processing_tags,const std::string &tag_name)
  {
    // check for presence of tag_name in processing tags.
    // if present, remove it, and return true; if not present return false

    auto tag_it = processing_tags->find(tag_name);
    if (tag_it != processing_tags->end()) {
      processing_tags->erase(tag_it);
      return true;
    }

    return false; 
  }
  
  std::unordered_set<std::string> geomproc_vector_to_set(std::vector<std::string> vec)
  {
    std::unordered_set<std::string> retval;

    for (auto && tag: vec) {
      retval.emplace(tag);
    }

    return retval;
    
  }

  void geomproc_specify_dependency(std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags,std::string needed_tag)
  // Specify from within an instantiation routine that the current routine is dependent on some other tag,
  // which may or may not have already been specified. 
  {
    if (all_processing_tags->find(needed_tag)==all_processing_tags->end()) {
      // not already specified
      all_processing_tags->emplace(needed_tag);
      remaining_processing_tags->emplace(needed_tag);
    }
  }
  
  // Instantiate the relevant geometry processing math functions according to the specified processing
  // tags (which are removed from the set). NOTE: Must be called while still in the transaction
  // in which the geometry is defined and loaded, and before meshedcurpart/texedcurpart are marked
  // as "data ready"
  void instantiate_geomproc_math_functions(std::shared_ptr<recdatabase> recdb,std::shared_ptr<loaded_part_geometry_recording> loaded_geom, std::shared_ptr<meshed_part_recording> meshedcurpart,std::shared_ptr<textured_part_recording> texedcurpart, std::unordered_set<std::string> *processing_tags)
  {
    std::shared_ptr<geomproc_instantiator_map> instantiator_map = geomproc_instantiator_registry();

    if (meshedcurpart) {
      loaded_geom->processed_relpaths.emplace("meshed",recdb_relative_path_to(recdb_path_context(loaded_geom->info->name),meshedcurpart->info->name));
    }

    if (texedcurpart) {
      loaded_geom->processed_relpaths.emplace("texed",recdb_relative_path_to(recdb_path_context(loaded_geom->info->name),texedcurpart->info->name));
    }
    
    std::unordered_set<std::string>::iterator thistag,nexttag;

    std::unordered_set<std::string> remaining_processing_tags = *processing_tags;
    std::unordered_set<std::string> all_processing_tags = *processing_tags; // copy the list we were provided
    std::unordered_set<std::string> missing_processing_tags;
    
    for (thistag=remaining_processing_tags.begin();thistag != remaining_processing_tags.end();thistag=remaining_processing_tags.begin()) {

      std::string thistag_str = *thistag; 
      geomproc_instantiator_map::iterator map_entry = instantiator_map->find(thistag_str);

      if (map_entry != instantiator_map->end()) {
	// Found this tag in the instantiator map

	// ... instantiate.
	map_entry->second(recdb,loaded_geom,&remaining_processing_tags,&all_processing_tags);

	// Remove tag if still present from remaining_processing_tags
	remaining_processing_tags.erase(thistag_str);
	
      } else {
	// did not find: Move to missing_processing_tags
	missing_processing_tags.emplace(thistag_str);
	remaining_processing_tags.erase(thistag_str);
      }
      
    }

    if (meshedcurpart) {
      meshedcurpart->processed_relpaths = loaded_geom->processed_relpaths;
    }
    if (texedcurpart) {
      texedcurpart->processed_relpaths = loaded_geom->processed_relpaths;
    }

    // return just the missing processing tags
    *processing_tags = missing_processing_tags; 
      
  }

    
  
  
};
