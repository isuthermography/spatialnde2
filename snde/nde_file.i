%shared_ptr(snde::ndefile_readrecording_base);
snde_rawaccessible(snde::ndefile_readrecording_base);

%shared_ptr(snde::ndefile_readarray);
snde_rawaccessible(snde::ndefile_readarray);

%shared_ptr(snde::ndefile_readgroup);
snde_rawaccessible(snde::ndefile_readgroup);

%{
#include "snde/nde_file.hpp"
%}



namespace snde {

  class ndefile_readrecording_base; // forward reference
  
  //extern SNDE_API std::unordered_map<std::string,std::pair<H5::DataType,unsigned>> nde_file_nativetype_mappings;


  typedef std::map<std::string,std::pair<std::shared_ptr<ndefile_readrecording_base>,std::shared_ptr<recording_base>>> nde_recording_map;

  typedef std::function<std::shared_ptr<ndefile_readrecording_base>(const std::set<std::string> &nde_classes,std::string h5path, H5::Group &group, std::string recpath,std::shared_ptr<nde_recording_map> filemap)> ndefile_loaderfunc;
  
  
  class ndefile_readrecording_base {
  public:
    
    std::set<std::string> nde_classes;
    std::set<std::string> nde_class_tags;
    //H5::H5File file;
    std::string h5path;
    //H5::Group group;
    std::string recpath;
    std::string nde_recording_version; // actually a std::string
    std::string nde_recording_label; // actually a std::string
    std::shared_ptr<immutable_metadata> metadata;
    
    //ndefile_readrecording_base(const std::set<std::string> &nde_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<nde_recording_map> filemap);
    
    virtual std::shared_ptr<recording_base> define_rec(std::shared_ptr<recdatabase> recdb,std::string ownername,void *owner_id)=0;
    
    virtual void read(std::shared_ptr<recording_base> rec); // read actual data into rec and any sub-recordings into filemap
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
    
    //ndefile_readarray(const std::set<std::string> &nde_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<nde_recording_map> filemap);

   
    ndefile_readarray() = delete; // tell SWIG we don't have a constructor.
    
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
    
    
    //ndefile_readgroup(const std::set<std::string> &nde_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<nde_recording_map> filemap);
    ndefile_readgroup() = delete; // tell SWIG we don't have a constructor.
    
    virtual std::shared_ptr<recording_base> define_rec(std::shared_ptr<recdatabase> recdb,std::string ownername,void *owner_id);    
    virtual void read(std::shared_ptr<recording_base> rec); // read actual data
    virtual ~ndefile_readgroup() = default;
    
  };

  
  //std::shared_ptr<ndefile_readrecording_base> ndefile_loadrecording(std::string h5path,H5::Group group,std::string recpath,std::shared_ptr<nde_recording_map> filemap);
  
  std::shared_ptr<nde_recording_map> ndefile_loadfile(std::shared_ptr<recdatabase> recdb,std::string ownername,void *owner_id,std::string filename,std::string recpath="/"); // add filter function parameter or specific recording to request to limit what is loaded? 
  
};

