#ifndef SNDE_GEOMETRY_PROCESSING_HPP
#define SNDE_GEOMETRY_PROCESSING_HPP


namespace snde {


  
  typedef std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<loaded_part_geometry_recording> loaded_geom)> geomproc_instantiator;

  
  int register_geomproc_math_function(std::string tagname,geomproc_instantiator instantiator);

  
  // check for presence of tag_name in processing tags.
  // if present, remove it, and return true; if not present return false
  bool extract_geomproc_option(std::unordered_set<std::string> *processing_tags,const std::string &tag_name);

  std::unordered_set<std::string> geomproc_vector_to_set(std::vector<std::string> vec);

  // Instantiate the relevant geometry processing math functions according to the specified processing
  // tags (which are removed from the set)
  void instantiate_geomproc_math_functions(std::shared_ptr<recdatabase> recdb,std::shared_ptr<loaded_part_geometry_recording> loaded_geom, std::unordered_set<std::string> *processing_tags);


  
};

#endif // SNDE_GEOMETRY_PROCESSING_HPP
