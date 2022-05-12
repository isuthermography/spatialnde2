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
  
  extern SNDE_API std::unordered_map<std::string,std::pair<H5::DataType,unsigned>> nde_file_nativetype_mappings;


  typedef std::map<std::string,std::pair<std::shared_ptr<ndefile_readrecording_base>,std::shared_ptr<recording_base>>> nde_recording_map;

  typedef std::function<std::shared_ptr<ndefile_readrecording_base>(const std::set<std::string> &nde_classes,std::string h5path, H5::Group &group, std::string recpath,std::shared_ptr<nde_recording_map> filemap)> ndefile_loaderfunc;

  int register_ndefile_loader(std::string classname,unsigned depth,ndefile_loaderfunc loaderfunc); // depth of 1 = recording_base, depth of 2 = immediate subclass of recording_base, etc. 


  template <typename T>
  int register_ndefile_loader_class(std::string classname,unsigned depth)
  {
    int value = register_ndefile_loader(classname,depth,
					[] (const std::set<std::string> &nde_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<nde_recording_map> filemap) {
					  return std::make_shared<T>(nde_classes,h5path,group,recpath,filemap);
					}); 
    return value;
  }
  
  
  
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
    
    ndefile_readrecording_base(const std::set<std::string> &nde_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<nde_recording_map> filemap);
    
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
    
    ndefile_readarray(const std::set<std::string> &nde_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<nde_recording_map> filemap);
    
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
    
    
    ndefile_readgroup(const std::set<std::string> &nde_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<nde_recording_map> filemap);
    
    virtual std::shared_ptr<recording_base> define_rec(std::shared_ptr<recdatabase> recdb,std::string ownername,void *owner_id);    
    virtual void read(std::shared_ptr<recording_base> rec) // read actual data
    {
      // nothing to actually read for a group
      rec->mark_metadata_done();
      rec->mark_data_ready();
    }
    virtual ~ndefile_readgroup() = default;
    
  };

  
  std::shared_ptr<ndefile_readrecording_base> ndefile_loadrecording(std::string h5path,H5::Group group,std::string recpath,std::shared_ptr<nde_recording_map> filemap);

  std::shared_ptr<nde_recording_map> ndefile_loadfile(std::shared_ptr<recdatabase> recdb,std::string ownername,void *owner_id,std::string filename,std::string recpath="/"); // add filter function parameter or specific recording to request to limit what is loaded? 

};

#endif // SNDE_NDE_FILE_HPP

