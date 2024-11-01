%shared_ptr(snde::andefile_readrecording_base);
snde_rawaccessible(snde::andefile_readrecording_base);

%shared_ptr(snde::ande_loadrecording_map);
//snde_rawaccessible(snde::ande_loadrecording_map);

%shared_ptr(snde::andefile_readarray);
snde_rawaccessible(snde::andefile_readarray);

%shared_ptr(snde::andefile_readgroup);
snde_rawaccessible(snde::andefile_readgroup);

%{
#include "snde/ande_file.hpp"
%}



namespace snde {

  class andefile_readrecording_base; // forward reference
  
  //extern SNDE_API std::unordered_map<std::string,std::pair<H5::DataType,unsigned>> ande_file_nativetype_mappings;


  typedef std::map<std::string,std::pair<std::shared_ptr<andefile_readrecording_base>,std::shared_ptr<recording_base>>> ande_loadrecording_map;

  typedef std::function<std::shared_ptr<andefile_readrecording_base>(const std::set<std::string> &ande_classes,std::string h5path, H5::Group &group, std::string recpath,std::shared_ptr<ande_loadrecording_map> filemap)> andefile_loaderfunc;
  typedef std::map<std::string,std::pair<unsigned,andefile_loaderfunc>> andefile_loader_map;

    typedef std::function<std::map<std::string,channel_state>::iterator(std::shared_ptr<std::map<std::string,channel_state>> *channel_map,std::map<std::string,channel_state>::iterator starting_iterator,std::shared_ptr<recording_base> recording_to_save,std::string saveclass,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath)> andefile_saverfunc;

  typedef std::function<void(std::shared_ptr<recording_base> recording_to_save,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath,H5::Group recgroup)> andefile_writerfunc;
  
  typedef std::map<std::string,std::pair<andefile_saverfunc,andefile_writerfunc>> andefile_saver_map;

  //std::shared_ptr<std::unordered_map<std::string,std::pair<std::function<H5::DataType()>,unsigned>>> ande_file_map_by_nativetype();
  //std::shared_ptr<std::unordered_map<unsigned,std::pair<std::function<H5::DataType()>,std::string>>> ande_file_map_by_typenum();

  //std::shared_ptr<andefile_loader_map> andefile_loader_registry();
  //std::shared_ptr<andefile_saver_map> andefile_saver_registry();

  //int register_andefile_loader(std::string classname,unsigned depth,andefile_loaderfunc loaderfunc); // depth of 1 = recording_base, depth of 2 = immediate subclass of recording_base, etc. 

  //int register_andefile_saver_function(std::string classname,andefile_saverfunc function);

  class andefile_readrecording_base {
  public:
    
    std::set<std::string> ande_classes;
    std::set<std::string> ande_class_tags;
    //H5::H5File file;
    std::string h5path;
    //H5::Group group;
    std::string recpath;
    std::string ande_recording_version; // actually a std::string
    std::string ande_recording_label; // actually a std::string
    std::shared_ptr<immutable_metadata> metadata;
    
    //andefile_readrecording_base(const std::set<std::string> &ande_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<ande_recording_map> filemap);
    
    virtual std::shared_ptr<recording_base> define_rec(std::shared_ptr<active_transaction> trans,std::string ownername)=0;
    
    virtual void read(std::shared_ptr<recording_base> rec); // read actual data into rec and any sub-recordings into filemap
    virtual ~andefile_readrecording_base() = default;
  };
  
  
  class andefile_readarray: public andefile_readrecording_base {
  public:
    // From superclass
    //std::set<std::string> ande_classes;
    //H5::H5File file;
    //std::string h5path;
    //H5::Group group;
    //std::string recpath;
    
    bool hidden;

    int64_t numarrays; 

    // should have metadata here
    
    //andefile_readarray(const std::set<std::string> &ande_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<ande_recording_map> filemap);

   
    andefile_readarray() = delete; // tell SWIG we don't have a constructor.
    
    virtual std::shared_ptr<recording_base> define_rec(std::shared_ptr<active_transaction> trans,std::string ownername);     
    virtual void read(std::shared_ptr<recording_base> rec); // read actual data

    virtual ~andefile_readarray() = default;
  };
  


  class andefile_readgroup: public andefile_readrecording_base {
  public:
    // From superclass
    //std::set<std::string> ande_classes;
    // H5::File file
    //std::string h5path;
    //H5::Group group;
    //std::string recpath;
    
    std::string group_version;

    std::vector<std::tuple<std::string,std::shared_ptr<andefile_readrecording_base>>> group_subloaders;
    
    
    //andefile_readgroup(const std::set<std::string> &ande_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<ande_recording_map> filemap);
    andefile_readgroup() = delete; // tell SWIG we don't have a constructor.
    
    virtual std::shared_ptr<recording_base> define_rec(std::shared_ptr<active_transaction> trans,std::string ownername);      
    virtual void read(std::shared_ptr<recording_base> rec); // read actual data
    virtual ~andefile_readgroup() = default;
    
  };

  //andefile_writerfunc andefile_lookup_writer_function(std::shared_ptr<recording_base> rec_to_write);
  //andefile_writerfunc andefile_lookup_writer_function_by_class(std::string classname);
  //std::pair<andefile_saverfunc,std::string> andefile_lookup_saver_function(std::shared_ptr<recording_base> rec_to_write);
  //void andefile_write_superclass(std::shared_ptr<recording_base> recording_to_save,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath,H5::Group recgroup, std::string classname);

  //void andefile_write_recording_base(std::shared_ptr<recording_base> recording_to_save,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath,H5::Group recgroup);
  //void andefile_write_recording_group(std::shared_ptr<recording_base> recording_to_save,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath,H5::Group recgroup);
  //void andefile_write_multi_ndarray_recording(std::shared_ptr<recording_base> recording_to_save,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath,H5::Group recgroup);

  //std::map<std::string,channel_state>::iterator andefile_save_generic_recording(std::shared_ptr<std::map<std::string,channel_state>> *channel_map,std::map<std::string,channel_state>::iterator starting_iterator,std::shared_ptr<recording_base> recording_to_save,std::string saveclass,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath);
  //std::map<std::string,channel_state>::iterator andefile_save_generic_group(std::shared_ptr<std::map<std::string,channel_state>> *channel_map,std::map<std::string,channel_state>::iterator starting_iterator,std::shared_ptr<recording_base> recording_to_save,std::string saveclass,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath);


  
  //std::shared_ptr<andefile_readrecording_base> andefile_loadrecording(std::string h5path,H5::Group group,std::string recpath,std::shared_ptr<ande_recording_map> filemap);
  
  std::shared_ptr<ande_loadrecording_map> andefile_loadfile(std::shared_ptr<active_transaction> trans,std::string ownername,std::string filename,std::string recpath="/"); // add filter function parameter or specific recording to request to limit what is loaded?


  //bool andefile_savefile_pathstack_top_is_start_of(std::vector<std::string> *pathstack,const std::string &writepath_group);
  
  //void andefile_savefile_pop_to_common(std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,const std::string &writepath);

  //void andefile_savefile_push_to_group(H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,const std::string &writepath);

  void andefile_savefile(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_set_state> rss_or_globalrev,std::string filename,std::string grouppath="/");
  
};

