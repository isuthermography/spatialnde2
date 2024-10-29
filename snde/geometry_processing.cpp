#include <functional>
#include <string>
#include <unordered_set>
#include <memory>
#include <map>
#include <fstream>

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
  void instantiate_geomproc_math_functions(std::shared_ptr<active_transaction> trans,std::shared_ptr<loaded_part_geometry_recording> loaded_geom, std::shared_ptr<meshed_part_recording> meshedcurpart,std::shared_ptr<meshed_parameterization_recording> meshedcurparam,std::shared_ptr<textured_part_recording> texedcurpart, std::unordered_set<std::string> *processing_tags)
  {
    std::shared_ptr<geomproc_instantiator_map> instantiator_map = geomproc_instantiator_registry();

    if (meshedcurpart) {
      loaded_geom->processed_relpaths.emplace("meshed",recdb_relative_path_to(recdb_path_context(loaded_geom->info->name),meshedcurpart->info->name));
    }

    if (meshedcurparam) {
      loaded_geom->processed_relpaths.emplace("uv",recdb_relative_path_to(recdb_path_context(loaded_geom->info->name),meshedcurpart->info->name));
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
	map_entry->second(trans,loaded_geom,&remaining_processing_tags,&all_processing_tags);

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
    
    if (meshedcurparam) {
      meshedcurparam->processed_relpaths = loaded_geom->processed_relpaths;
    }
    
    if (texedcurpart) {
      texedcurpart->processed_relpaths = loaded_geom->processed_relpaths;
    }

    // return just the missing processing tags
    *processing_tags = missing_processing_tags; 
      
  }

  static std::string simplecsv_strip_spaces(const char* lineptr,size_t startpos,size_t endpos) {

    std::string line(lineptr);
    
    for (;lineptr[startpos] && (lineptr[startpos] == ' ' || lineptr[startpos] == '\t'); startpos++);
    
    for (endpos = startpos; endpos >= 1 && (lineptr[endpos - 1] == ' ' || lineptr[endpos - 1] == '\t' || lineptr[endpos -1] == '\r' || lineptr[endpos - 1] == '\n'); endpos--);
    
    assert(endpos >= startpos);
	
    return line.substr(startpos, endpos - startpos);
  }

  static std::vector<std::string> simplecsv_read_row(std::istream &src)
  {
    std::vector<std::string> ret;
    std::string line;
    size_t pos = 0;
    size_t lastpos = 0;
    std::getline(src,line);
    for (pos = 0; pos != std::string::npos && pos < line.size(); lastpos = pos, pos = line.find(',',pos)) {
      if (pos != 0) {
	// not the first iteration
	const char* lineptr = line.c_str();
	ret.push_back(simplecsv_strip_spaces(lineptr,lastpos,pos));
	
	     
      }
    }
    if (pos == std::string::npos) {
      ret.push_back(simplecsv_strip_spaces(line.c_str(),lastpos,line.size()));
    }
    return ret;
  }
  
  void load_geom_landmarks(std::shared_ptr<recdatabase> recdb,std::shared_ptr<active_transaction> trans,std::string landmarks_filename,std::shared_ptr<loaded_part_geometry_recording> loaded_geom,std::string ownername)
  // This gets called within a transaction with an incomplete
  // loaded_part_geometry_recording that we can add to.
  {
    std::vector<std::tuple<std::string,double,double>> landmarks_2D;
    std::vector<std::tuple<std::string,double,double,double>> landmarks_3D;
    
    std::ifstream landmarks_file(landmarks_filename);
    std::vector<std::string> firstline = simplecsv_read_row(landmarks_file);
    if (firstline.size() != 4) {
      throw snde_error("load_geom_landmarks(): first row of file \"%s\" does not contain exactly 4 comma-separated headings",landmarks_filename.c_str());
    }

    if (firstline.at(0) != "Landmark name") {
      throw snde_error("load_geom_landmarks(): first column heading in file \"%s\" is not \"Landmark name\". ",landmarks_filename.c_str());
    }

    
    if (firstline.at(1) != "2D or 3D") {
      throw snde_error("load_geom_landmarks(): second column heading in file \"%s\" is not \"2D or 3D\". ",landmarks_filename.c_str());
    }

    if (firstline.at(2) != "x or u (meters)") {
      throw snde_error("load_geom_landmarks(): third column heading in file \"%s\" is not \"x or u (meters)\". ",landmarks_filename.c_str());
    }

    if (firstline.at(3) != "y or v (meters)") {
      throw snde_error("load_geom_landmarks(): fourth column heading in file \"%s\" is not \"y or v (meters)\". ",landmarks_filename.c_str());
    }

    if (firstline.at(4) != "z or 0 (meters)") {
      throw snde_error("load_geom_landmarks(): fifth column heading in file \"%s\" is not \"z or 0 (meters)\". ",landmarks_filename.c_str());
    }

    size_t line_num = 2;
    do {
      std::vector<std::string> row = simplecsv_read_row(landmarks_file);
      if (row.size() == 0) {
	line_num++;
	continue;
      }
      if (row.size() != 5) {
	throw snde_error("load_geom_landmarks(): line %d does not have five entries",(int)line_num);
      }
      char* endptr;
      
      double x_or_u;
      std::string x_or_u_str = row.at(2);
      x_or_u = strtod(x_or_u_str.c_str(),&endptr);
      if (*endptr != 0) {
	throw snde_error("load_geom_landmarks(): x or u value %s on line %d is not parseable as a number",x_or_u_str.c_str(),(int)line_num);
      }
      
      double y_or_v;
      std::string y_or_v_str = row.at(3);
      y_or_v = strtod(y_or_v_str.c_str(),&endptr);
      if (*endptr != 0) {
	throw snde_error("load_geom_landmarks(): y or v value %s on line %d is not parseable as a number",y_or_v_str.c_str(),(int)line_num);
      }
      
      double z_or_0;
      std::string z_or_0_str = row.at(4);
      z_or_0 = strtod(z_or_0_str.c_str(),&endptr);
      if (*endptr != 0) {
	throw snde_error("load_geom_landmarks(): z or 0 value %s on line %d is not parseable as a number",z_or_0_str.c_str(),(int)line_num);
      }
      
      std::string landmark_name = row.at(0);
      if (row.at(1) == "2D") {
	landmarks_2D.push_back(std::make_tuple(landmark_name,x_or_u,y_or_v));
      } else if (row.at(1) == "3D") {
	landmarks_3D.push_back(std::make_tuple(landmark_name,x_or_u,y_or_v,z_or_0));
      } else {
	throw snde_error("load_geom_landmarks(): line %d does not specify 2D or 3D",(int)line_num);
      }
      line_num++;
    } while (!landmarks_file.eofbit);

    std::string landmarks_chanpath = recdb_path_join(loaded_geom->info->name,"landmarks");
    std::shared_ptr<channelconfig> landmarks_config = std::make_shared<snde::channelconfig>(landmarks_chanpath,ownername,false);

    std::shared_ptr<reserved_channel> landmarks_chan = recdb->reserve_channel(trans,landmarks_config);
    
    std::shared_ptr<recording_base> landmarks_recording = create_recording<recording_base>(trans,landmarks_chan);

    std::shared_ptr<constructible_metadata> metadata = std::make_shared<constructible_metadata>();
    for (auto lmname_u_v: landmarks_2D) {
      std::string lm_name;
      double u,v;

      std::tie(lm_name,u,v) = lmname_u_v;
      metadata->AddMetaDatum(metadatum_str(ssprintf("ande_landmarks_lm_%s_type",lm_name.c_str()),"2D"));
      
      metadata->AddMetaDatum(metadatum_dblunits(ssprintf("ande_landmarks_lm_%s_u",lm_name.c_str()),u,"meters"));

      metadata->AddMetaDatum(metadatum_dblunits(ssprintf("ande_landmarks_lm_%s_v",lm_name.c_str()),v,"meters"));
    }
    for (auto lmname_x_y_z: landmarks_3D) {
      std::string lm_name;
      double x,y,z;

      std::tie(lm_name,x,y,z) = lmname_x_y_z;
      metadata->AddMetaDatum(metadatum_str(ssprintf("ande_landmarks_lm_%s_type",lm_name.c_str()),"3D"));
      
      metadata->AddMetaDatum(metadatum_dblunits(ssprintf("ande_landmarks_lm_%s_x",lm_name.c_str()),x,"meters"));

      metadata->AddMetaDatum(metadatum_dblunits(ssprintf("ande_landmarks_lm_%s_y",lm_name.c_str()),y,"meters"));

      metadata->AddMetaDatum(metadatum_dblunits(ssprintf("ande_landmarks_lm_%s_z",lm_name.c_str()),z,"meters"));
    }
    landmarks_recording->metadata = metadata;
    landmarks_recording->mark_metadata_done();
    landmarks_recording->mark_data_ready();
  }
  
  
};
