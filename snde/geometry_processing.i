%{
#include "snde/geometry_processing.hpp"
%}


namespace snde {


  
  //typedef std::function<void(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags)> geomproc_instantiator;

  
  //int register_geomproc_math_function(std::string tagname,geomproc_instantiator instantiator);

  
  // check for presence of tag_name in processing tags.
  // if present, remove it, and return true; if not present return false
  //bool extract_geomproc_option(std::unordered_set<std::string> *processing_tags,const std::string &tag_name);

  //std::unordered_set<std::string> geomproc_vector_to_set(std::vector<std::string> vec);


  // Specify from within an instantiation routine that the current routine is dependent on some other tag,
  // which may or may not have already been specified. 
  //void geomproc_specify_dependency(std::unordered_set<std::string> *remaining_processing_tags,std::unordered_set<std::string> *all_processing_tags,std::string needed_tag);

  
  // Instantiate the relevant geometry processing math functions according to the specified processing
  // tags (which are removed from the set). NOTE: Must be called while still in the transaction
  // in which the geometry is defined and loaded, and before meshedcurpart/texedcurpart are marked
  // as "data ready"
  //void instantiate_geomproc_math_functions(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom, std::shared_ptr<meshed_part_recording> meshedcurpart,std::shared_ptr<meshed_parameterization_recording> meshedcurparam,std::shared_ptr<textured_part_recording> texedcurpart,std::unordered_set<std::string> *processing_tags);

  void load_geom_landmarks(std::shared_ptr<recdatabase> recdb,std::shared_ptr<active_transaction> trans, std::string landmarks_filename,std::shared_ptr<loaded_part_geometry_recording> loaded_geom, std::string ownername);

  
};

