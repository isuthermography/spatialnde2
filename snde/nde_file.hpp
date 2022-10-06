#ifndef SNDE_NDE_FILE_HPP
#define SNDE_NDE_FILE_HPP

#include <string>
#include <unordered_map>
#include <vector>
#include <set>
#include <memory>
#include <H5Cpp.h>


#include "snde/snde_types.h"
#include "snde/metadata.hpp"
#include "snde/recstore.hpp"

// The ndefile_readxxxx class
// structure parallels the nde file spec.
// Basically lambdas that call the constructors
// should be registered into the ndefile_reader_registry
//
// Then a rather dumb algorithm looks up the nde_classes
// and tries to find a registry key that covers the
// largest number and uses that to select which
// reader to call. Instantiating the reader pulls
// basic info out of the file being read but not the data
//
// Once the reader is instantiated, we call the define()
// virtual method to construct the proper subclass of
// recording_base and then the read() method to actually
// load the data 

namespace snde {

  class ndefile_readrecording_base; // forward reference


  typedef std::map<std::string,std::pair<std::shared_ptr<ndefile_readrecording_base>,std::shared_ptr<recording_base>>> nde_loadrecording_map;

  typedef std::function<std::shared_ptr<ndefile_readrecording_base>(const std::set<std::string> &nde_classes,std::string h5path, H5::Group &group, std::string recpath,std::shared_ptr<nde_loadrecording_map> filemap)> ndefile_loaderfunc;
  typedef std::map<std::string,std::pair<unsigned,ndefile_loaderfunc>> ndefile_loader_map;



  // The saverfunc is responsible for the highlevel operation of saving a recording.
  // it usually calls the lower level implementation of the writerfunc.
  // For each saveable class, the saverfunc will usually be generic
  // (ndefile_save_generic_group() or ndefile_save_generic_recording())
  // and the details will be in the writerfunc, which usually calls the
  // baseclass writerfunc.
  
  typedef std::function<std::map<std::string,channel_state>::iterator(std::shared_ptr<std::map<std::string,channel_state>> *channel_map,std::map<std::string,channel_state>::iterator starting_iterator,std::shared_ptr<recording_base> recording_to_save,std::string saveclass,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath)> ndefile_saverfunc;

  typedef std::function<void(std::shared_ptr<recording_base> recording_to_save,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath,H5::Group recgroup)> ndefile_writerfunc;
  
  typedef std::map<std::string,std::pair<ndefile_saverfunc,ndefile_writerfunc>> ndefile_saver_map;

  std::shared_ptr<std::unordered_map<std::string,std::pair<std::function<H5::DataType()>,unsigned>>> nde_file_map_by_nativetype();
  std::shared_ptr<std::unordered_map<unsigned,std::pair<std::function<H5::DataType()>,std::string>>> nde_file_map_by_typenum();

  int add_nde_file_nativetype_mapping(std::string nde_nativetype_str,std::function<H5::DataType()> h5_datatype_builder,unsigned snde_rtn_typenum);



  std::shared_ptr<ndefile_loader_map> ndefile_loader_registry();
  std::shared_ptr<ndefile_saver_map> ndefile_saver_registry();




  
  //extern SNDE_API std::unordered_map<std::string,std::pair<H5::DataType,unsigned>> nde_file_nativetype_mappings;


  
  int register_ndefile_loader(std::string classname,unsigned depth,ndefile_loaderfunc loaderfunc); // depth of 1 = recording_base, depth of 2 = immediate subclass of recording_base, etc. 


  template <typename T>
  int register_ndefile_loader_class(std::string classname,unsigned depth)
  {
    int value = register_ndefile_loader(classname,depth,
					[] (const std::set<std::string> &nde_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<nde_loadrecording_map> filemap) {
					  return std::make_shared<T>(nde_classes,h5path,group,recpath,filemap);
					}); 
    return value;
  }
  
  
  int register_ndefile_saver_function(std::string classname,ndefile_saverfunc function);
  
  
  class ndefile_readrecording_base {
  public:
    
    std::set<std::string> nde_classes;
    std::set<std::string> nde_class_tags;
    H5::H5File file;
    std::string h5path;
    H5::Group group;
    std::string recpath;
    H5std_string nde_recording_version; // actually a std::string
    H5std_string nde_recording_label; // actually a std::string
    std::shared_ptr<immutable_metadata> metadata;
    
    ndefile_readrecording_base(const std::set<std::string> &nde_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<nde_loadrecording_map> filemap);
    
    virtual std::shared_ptr<recording_base> define_rec(std::shared_ptr<recdatabase> recdb,std::string ownername,void *owner_id)=0;
    
    virtual void read(std::shared_ptr<recording_base> rec) // read actual data into rec and any sub-recordings into filemap
    {
      // no-op
      rec->mark_metadata_done();
      rec->mark_data_ready();
    }
    virtual ~ndefile_readrecording_base() = default;
  };
  
  
  class ndefile_readarray: public ndefile_readrecording_base {
  public:
    // From superclass
    //std::set<std::string> nde_classes;
    //H5::H5File file;
    //std::string h5path;
    //H5::Group group;
    //std::string recpath;
    
    bool hidden;

    int64_t numarrays; 

    // should have metadata here
    
    ndefile_readarray(const std::set<std::string> &nde_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<nde_loadrecording_map> filemap);
    
    virtual std::shared_ptr<recording_base> define_rec(std::shared_ptr<recdatabase> recdb,std::string ownername,void *owner_id);    
    virtual void read(std::shared_ptr<recording_base> rec); // read actual data

    virtual ~ndefile_readarray() = default;
  };
  


  class ndefile_readgroup: public ndefile_readrecording_base {
  public:
    // From superclass
    //std::set<std::string> nde_classes;
    // H5::File file
    //std::string h5path;
    //H5::Group group;
    //std::string recpath;
    
    std::string group_version;

    std::vector<std::tuple<std::string,std::shared_ptr<ndefile_readrecording_base>>> group_subloaders;
    
    
    ndefile_readgroup(const std::set<std::string> &nde_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<nde_loadrecording_map> filemap);
    
    virtual std::shared_ptr<recording_base> define_rec(std::shared_ptr<recdatabase> recdb,std::string ownername,void *owner_id);    
    virtual void read(std::shared_ptr<recording_base> rec) // read actual data
    {
      // nothing to actually read for a group
      rec->mark_metadata_done();
      rec->mark_data_ready();
    }
    virtual ~ndefile_readgroup() = default;
    
  };

  ndefile_writerfunc ndefile_lookup_writer_function(std::shared_ptr<recording_base> rec_to_write);
  ndefile_writerfunc ndefile_lookup_writer_function_by_class(std::string classname);
  std::pair<ndefile_saverfunc,std::string> ndefile_lookup_saver_function(std::shared_ptr<recording_base> rec_to_write);
  void ndefile_write_superclass(std::shared_ptr<recording_base> recording_to_save,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath,H5::Group recgroup, std::string classname);

  void ndefile_write_recording_base(std::shared_ptr<recording_base> recording_to_save,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath,H5::Group recgroup);
  void ndefile_write_recording_group(std::shared_ptr<recording_base> recording_to_save,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath,H5::Group recgroup);
  void ndefile_write_multi_ndarray_recording(std::shared_ptr<recording_base> recording_to_save,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath,H5::Group recgroup);

  std::map<std::string,channel_state>::iterator ndefile_save_generic_recording(std::shared_ptr<std::map<std::string,channel_state>> *channel_map,std::map<std::string,channel_state>::iterator starting_iterator,std::shared_ptr<recording_base> recording_to_save,std::string saveclass,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath);
  std::map<std::string,channel_state>::iterator ndefile_save_generic_group(std::shared_ptr<std::map<std::string,channel_state>> *channel_map,std::map<std::string,channel_state>::iterator starting_iterator,std::shared_ptr<recording_base> recording_to_save,std::string saveclass,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath);




  
  std::shared_ptr<ndefile_readrecording_base> ndefile_loadrecording(std::string h5path,H5::Group group,std::string recpath,std::shared_ptr<nde_loadrecording_map> filemap);

  std::shared_ptr<nde_loadrecording_map> ndefile_loadfile(std::shared_ptr<recdatabase> recdb,std::string ownername,void *owner_id,std::string filename,std::string recpath="/"); // add filter function parameter or specific recording to request to limit what is loaded?

  bool ndefile_savefile_pathstack_top_is_start_of(std::vector<std::string> *pathstack,const std::string &writepath_group);
  
  void ndefile_savefile_pop_to_common(std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,const std::string &writepath);

  void ndefile_savefile_push_to_group(H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,const std::string &writepath);



  void ndefile_savefile(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_set_state> rss_or_globalrev,std::string filename,std::string grouppath="/");

};

#endif // SNDE_NDE_FILE_HPP

