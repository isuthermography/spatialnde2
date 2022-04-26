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

  
  // Instantiate the relevant geometry processing math functions according to the specified processing
  // tags (which are removed from the set)
  void instantiate_geomproc_math_functions(std::shared_ptr<recdatabase> recdb,std::shared_ptr<loaded_part_geometry_recording> loaded_geom, std::unordered_set<std::string> *processing_tags)
  {
    std::shared_ptr<geomproc_instantiator_map> instantiator_map = geomproc_instantiator_registry();

    std::unordered_set<std::string>::iterator thistag,nexttag;

    for (thistag=processing_tags->begin();thistag != processing_tags->end();thistag=nexttag) {
      nexttag = thistag;
      ++nexttag;
      
      geomproc_instantiator_map::iterator map_entry = instantiator_map->find(*thistag);

      if (map_entry != instantiator_map->end()) {
	// Found this tag in the instantiator map

	// ... instantiate.
	map_entry->second(recdb,loaded_geom);

	// Remove tag
	processing_tags->erase(thistag);
	
      }
      
    }
    
  }

    
  
  
};
