
#include "nde_file.hpp"

namespace snde {


  //
  //class nde_file {
  //  // Do we actually need this abstraction???
  //  H5::H5File H5Obj;


  //  nde_file(std::string filename):
  //   H5Obj(filename.get(),HF5ACC_RDONLY)
  // {
  //    
  // }

    /*
    H5Group open_group(std::string path)
    {
      return std::make_shared<H5Group>(
      }*/
    //
  //};


  // Architecture: For loading: 
  // Creating an object we need to instantiate the deepest subclass
  // But we want to share code for building the parent class,
  // which can be easily handled by the constructor chain.
  //
  // We can't actually identify where to go until we have at least
  // read the class list. 

  // So we have a plugin database of loader class handlers with depth indices
  // and instantiate the deepest class we can

  // pointed registries are immutable once created. However, the regular and shared
  // pointers are mutable and locked by the mutexes accessible by the getter functions,
  // below.  Get your own shared_ptr with ndefile_loader_registry()
  // or ndefile_saver_registry() as appropriate. The returned map can be treated as
  // immutable.
  
  static std::shared_ptr<ndefile_loader_map> *_ndefile_loader_registry; // default-initialized to nullptr
  
  static std::shared_ptr<ndefile_saver_map> *_ndefile_saver_registry; // default-initialized to nullptr


  std::shared_ptr<std::unordered_map<std::string,std::pair<std::function<H5::DataType()>,unsigned>>> *_nde_file_map_by_nativetype;
  std::shared_ptr<std::unordered_map<unsigned,std::pair<std::function<H5::DataType()>,std::string>>> *_nde_file_map_by_typenum;

  static std::mutex &nde_file_map_mutex()
  {
    // take advantage of the fact that since C++11 initialization of function statics
    // happens on first execution and is guaranteed thread-safe. This lets us
    // work around the "static initialization order fiasco" using the
    // "construct on first use idiom".
    // We just use regular pointers, which are safe from the order fiasco,
    // but we need some way to bootstrap thread-safety, and this mutex
    // is it. 
    static std::mutex mapmutex; 
    return mapmutex; 
  }

  
  std::shared_ptr<std::unordered_map<std::string,std::pair<std::function<H5::DataType()>,unsigned>>> nde_file_map_by_nativetype()
  {
    std::mutex &mapmutex = nde_file_map_mutex();
    std::lock_guard<std::mutex> reglock(mapmutex);
    
    if (!_nde_file_map_by_nativetype) {
      _nde_file_map_by_nativetype = new std::shared_ptr<std::unordered_map<std::string,std::pair<std::function<H5::DataType()>,unsigned>>>(std::make_shared<std::unordered_map<std::string,std::pair<std::function<H5::DataType()>,unsigned>>>());
    }
    return *_nde_file_map_by_nativetype;

  }
  
  std::shared_ptr<std::unordered_map<unsigned,std::pair<std::function<H5::DataType()>,std::string>>> nde_file_map_by_typenum()
  {
    std::mutex &mapmutex = nde_file_map_mutex();
    std::lock_guard<std::mutex> reglock(mapmutex);
    
    if (!_nde_file_map_by_typenum) {
      _nde_file_map_by_typenum = new std::shared_ptr<std::unordered_map<unsigned,std::pair<std::function<H5::DataType()>,std::string>>>(std::make_shared<std::unordered_map<unsigned,std::pair<std::function<H5::DataType()>,std::string>>>());
    }
    return *_nde_file_map_by_typenum;
    
  }

  int add_nde_file_nativetype_mapping(std::string nde_nativetype_str,std::function<H5::DataType()> h5_datatype_builder,unsigned snde_rtn_typenum)
  {
    // ensure pointers initialized
      nde_file_map_by_typenum();
      nde_file_map_by_nativetype();

      
      std::mutex &mapmutex = nde_file_map_mutex();
      std::lock_guard<std::mutex> reglock(mapmutex);

      std::shared_ptr<std::unordered_map<unsigned,std::pair<std::function<H5::DataType()>,std::string>>> by_typenum = std::make_shared<std::unordered_map<unsigned,std::pair<std::function<H5::DataType()>,std::string>>>(**_nde_file_map_by_typenum);
      
      std::shared_ptr<std::unordered_map<std::string,std::pair<std::function<H5::DataType()>,unsigned>>> by_nativetype = std::make_shared<std::unordered_map<std::string,std::pair<std::function<H5::DataType()>,unsigned>>>(**_nde_file_map_by_nativetype);
      
      by_typenum->emplace(snde_rtn_typenum,std::make_pair(h5_datatype_builder,nde_nativetype_str));
      by_nativetype->emplace(nde_nativetype_str,std::make_pair(h5_datatype_builder,snde_rtn_typenum));


      *_nde_file_map_by_typenum = by_typenum;
      *_nde_file_map_by_nativetype = by_nativetype;

      return 1;
  }
  

  static std::mutex &ndefile_loader_registry_mutex()
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

  static std::mutex &ndefile_saver_registry_mutex()
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

  static void ndefile_metadata_reader_function(H5::H5Object &loc /*H5::H5Location &loc*/, const H5std_string attr_name, void *operator_data)
  {

    std::vector<std::string> *metadata_names = (std::vector<std::string> *)operator_data;

    metadata_names->push_back(attr_name);    
  }


  static herr_t ndefile_iterate_subgroup_names(hid_t group, const char *subgroup_name, void *operator_data)
  {

    std::vector<std::string> *subgroup_names = (std::vector<std::string> *)operator_data;

    subgroup_names->push_back(subgroup_name);

    return H5_ITER_CONT;
  }

  
  std::shared_ptr<ndefile_loader_map> ndefile_loader_registry()
  {
    std::mutex &regmutex = ndefile_loader_registry_mutex();
    std::lock_guard<std::mutex> reglock(regmutex);
    
    if (!_ndefile_loader_registry) {
      _ndefile_loader_registry = new std::shared_ptr<ndefile_loader_map>(std::make_shared<ndefile_loader_map>());
    }
    return *_ndefile_loader_registry;
  }
  
  
  std::shared_ptr<ndefile_saver_map> ndefile_saver_registry()
  {
    std::mutex &regmutex = ndefile_saver_registry_mutex();
    std::lock_guard<std::mutex> reglock(regmutex);

    if (!_ndefile_saver_registry) {
      _ndefile_saver_registry = new std::shared_ptr<ndefile_saver_map>(std::make_shared<ndefile_saver_map>());
    }
    return *_ndefile_saver_registry;
  }
  
  
  
  int register_ndefile_loader(std::string classname,unsigned depth,ndefile_loaderfunc loaderfunc) // depth of 1 = recording_base, depth of 2 = immediate subclass of recording_base, etc. 
  {
    ndefile_loader_registry(); // ensure that the registry pointer exists

    std::mutex &regmutex = ndefile_loader_registry_mutex();
    std::lock_guard<std::mutex> reglock(regmutex);

    // copy map and update then publish the copy
    std::shared_ptr<ndefile_loader_map> new_map = std::make_shared<ndefile_loader_map>(**_ndefile_loader_registry);

    auto new_map_it = new_map->find(classname);
    if (new_map_it != new_map->end()) {
      snde_warning("register_ndefile_loader: Overriding preexisting loader entry for class %s",classname.c_str());
      new_map->erase(new_map_it);
      
    }
    
    new_map->emplace(classname,std::make_pair(depth,loaderfunc));

    *_ndefile_loader_registry = new_map;
    return 0;

  }

  int register_ndefile_saver_function(std::string classname,ndefile_saverfunc saver_function,ndefile_writerfunc writer_function)
  {
    ndefile_saver_registry(); // ensure that the registry pointer exists
    
    std::mutex &regmutex = ndefile_saver_registry_mutex();
    std::lock_guard<std::mutex> reglock(regmutex);
    
    // copy map and update then publish the copy
    std::shared_ptr<ndefile_saver_map> new_map = std::make_shared<ndefile_saver_map>(**_ndefile_saver_registry);
    
    auto new_map_it = new_map->find(classname);
    if (new_map_it != new_map->end()) {
      snde_warning("register_ndefile_saver_function: Overriding preexisting saver entry for class %s",classname.c_str());
      new_map->erase(new_map_it);
      
    }
    
    new_map->emplace(classname,std::make_pair(saver_function,writer_function));
    
    *_ndefile_saver_registry = new_map;
    return 0;

    
  }
  
  ndefile_readrecording_base::ndefile_readrecording_base(const std::set<std::string> &nde_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<nde_loadrecording_map> filemap) :
    nde_classes(nde_classes),
    file(file),
    h5path(h5path),
    group(group),
    recpath(recpath)
  {
    // Should put code to read basic params here
    // Confirm required nde_classes
    
    if (nde_classes.find("nde_recording") == nde_classes.end()) {
      throw snde_error("HDF5 Group %s is not an nde_recording",h5path.c_str());
    }
    
    H5::Attribute nde_recording_version_attr = group.openAttribute("nde_recording-version");
    
    H5::DataType nde_rv_dtype = nde_recording_version_attr.getDataType();
    if (nde_rv_dtype.getClass() != H5T_STRING) {
      throw snde_error("nde_recording-version for hdf5 group %s should be a string",h5path.c_str());
    }
    nde_recording_version_attr.read(nde_rv_dtype,nde_recording_version);
    
    
    H5::Attribute nde_recording_label_attr = group.openAttribute("nde_recording-label");
    H5::DataType nde_rl_dtype = nde_recording_label_attr.getDataType();
    if (nde_rl_dtype.getClass() != H5T_STRING) {
      throw snde_error("nde_recording-label for hdf5 group %s should be a string",h5path.c_str());
    }
    nde_recording_label_attr.read(nde_rl_dtype,nde_recording_label);
    
    
    // Read the class tags here:
    H5::Attribute nde_class_tags_attr = group.openAttribute("nde_class-tags");
    H5::DataType nde_ct_dtype = nde_class_tags_attr.getDataType();
    H5::DataSpace nde_ct_dspace = nde_class_tags_attr.getSpace();
    
    if (nde_ct_dspace.getSimpleExtentNdims() != 1) {
      throw snde_error("nde_class-tags attribute for hdf5 group %s should have exactly one iterable dimension",h5path.c_str());
    }

    // number of classes
    hsize_t nde_ct_num=0;

    nde_ct_dspace.getSimpleExtentDims(&nde_ct_num);

    if (nde_ct_num > 0) {
      if (nde_ct_dtype.getClass() != H5T_STRING) {
	throw snde_error("nde_class-tags attribute for hdf5 group %s should be an array of strings",h5path.c_str());
      }
    
      // std::set<std::string> nde_class_tags; // actually a class member
      
      char **class_tag_strings = new char*[nde_ct_num];
      H5::StrType nde_ct_strtype(H5::PredType::C_S1,H5T_VARIABLE);
      nde_ct_strtype.setCset(H5T_CSET_UTF8);
      nde_ct_strtype.setStrpad(H5T_STR_NULLTERM);
      nde_class_tags_attr.read(nde_ct_strtype,(void *)class_tag_strings);
      
      size_t nde_class_tags_size=nde_ct_dtype.getSize();
      for (size_t class_idx=0;class_idx < nde_ct_num;class_idx++) {
	// Per https://stackoverflow.com/questions/43722194/reading-a-string-array-hdf5-attribute-in-c
	// we actually have to call delete[] on each string, which seems
	// odd  (maybe they really mean free()?), but....
	nde_class_tags.emplace(class_tag_strings[class_idx]);
	delete[] class_tag_strings[class_idx];
	
      }
      delete[] class_tag_strings;
    
    }
    
    // read the metadata here.
    //if (file.nameExists(h5path+"/nde_recording-metadata")) {
    if (group.nameExists("nde_recording-metadata")) {
      //H5::Group metadatagroup = file.openGroup(h5path+"/nde_recording-metadata");
      H5::Group metadatagroup = group.openGroup("nde_recording-metadata");
      
      constructible_metadata metadata_loader;
      std::vector<std::string> metadata_names;
      
      metadatagroup.iterateAttrs(&ndefile_metadata_reader_function,nullptr,(void *)&metadata_names);
      
      for (auto && attr_name: metadata_names) {

	if (attr_name.length() > 6 && !attr_name.compare(attr_name.length()-6,6,"-units")) {
	  // ending with -units means this is a units attribute of another metadata entry
	  continue;
	}
	
	H5::Attribute md_attr = metadatagroup.openAttribute(attr_name);
	H5::DataType md_dtype = md_attr.getDataType();
	
	switch (md_dtype.getClass()) {
	case H5T_INTEGER:
	  if (md_attr.getIntType().getSign() != H5T_SGN_NONE) {
	    // signed data
	    int64_t intval;
	    md_attr.read(H5::PredType::NATIVE_INT64,&intval);
	    metadata_loader.AddMetaDatum(metadatum(attr_name,intval));
	  } else {
	    // unsigned data
	    uint64_t uintval;
	    md_attr.read(H5::PredType::NATIVE_UINT64,&uintval);
	    metadata_loader.AddMetaDatum(metadatum(attr_name,uintval));
	  }
	  
	  break;
	  
	case H5T_FLOAT:
	  {
	    double dblval;
	    md_attr.read(H5::PredType::NATIVE_DOUBLE,&dblval);

	    // Floats can have units... check for them
	    std::string attr_units_name = attr_name+"-units";
	    if (metadatagroup.attrExists(attr_units_name)) {
	      H5::Attribute md_units_attr = metadatagroup.openAttribute(attr_units_name);
	      H5::DataType md_units_dtype = md_units_attr.getDataType();
	      
	      if (md_units_dtype.getClass() != H5T_STRING) {
		throw snde_error("Units attribute \"%s\" of metadata entry %s in hdf5 group %s/nde_recording-metadata should be of string type",attr_units_name.c_str(),attr_name.c_str(),h5path.c_str());
	      }
	      std::string md_units_val;
	      md_units_attr.read(md_units_dtype,md_units_val);
	      
	      metadata_loader.AddMetaDatum(metadatum(attr_name,dblval,md_units_val));
	      
	    } else {
	      metadata_loader.AddMetaDatum(metadatum(attr_name,dblval));
	    }
	  }
	  break;
	  
	case H5T_STRING:
	  { 
	    H5::StrType md_strtype(H5::PredType::C_S1,H5T_VARIABLE);
	    md_strtype.setCset(H5T_CSET_UTF8);
	    md_strtype.setStrpad(H5T_STR_NULLTERM);

	    std::string strval;
	    md_attr.read(md_strtype,strval);
	    metadata_loader.AddMetaDatum(metadatum(attr_name,strval));
	    
	  }
	  break;

	case H5T_ENUM: // Bools are stored as an H5T_ENUM which is a H5T_STD_I8LE with two values: "FALSE" (0)  and "TRUE" (1)
	  {
	    H5::EnumType md_enumtype = md_attr.getEnumType();

	    int n_enum_members = md_enumtype.getNmembers();
	    
	    if (n_enum_members != 2) {
	      throw snde_error("Enum metadata attribute \"%s\"  in hdf5 group %s/nde_recording-metadata should have exactly two members in order to be boolean",attr_name.c_str(),h5path.c_str());
	    }

	    for (int membnum=0;membnum < n_enum_members;membnum++) {
	      int membval;
	      md_enumtype.getMemberValue(membnum,&membval);
	      std::string name = md_enumtype.nameOf(&membval,100);

	      if (! ( (membval==0 && name=="FALSE") || (membval==1 && name=="TRUE"))) {
		throw snde_error("Enum metadata attribute \"%s\"  in hdf5 group %s/nde_recording-metadata values should be 0=FALSE and 1=TRUE",attr_name.c_str(),h5path.c_str());
		
	      }
	      
	    }

	    uint8_t uintval=0;
	    md_attr.read(H5::PredType::NATIVE_UINT8,&uintval);

	    metadata_loader.AddMetaDatum(metadatum(attr_name,(bool)uintval));
	    
	    
	  }
	  break;
	  
	default:
	  throw snde_error("Unsupported HDF5 data type class for metadata entry %s: %d",attr_name.c_str(),(int)md_dtype.getClass());
	}
	
      }
      
      metadata=std::make_shared<immutable_metadata>(metadata_loader);
    } else {
      metadata=std::make_shared<immutable_metadata>();
      
    }

    //snde_warning("Got metadata for %s: %s",recpath.c_str(),metadata->to_string().c_str());
    
    // basically anything that might be needed to decide how to instantiate
    // the final recording in define_rec needs to be read here. This
    // probably includes metadata and recursive traversal
    //
    // The actual large-quantity data should wait for the call to read()
    
  }

  ndefile_readarray::ndefile_readarray(const std::set<std::string> &nde_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<nde_loadrecording_map> filemap) :
    ndefile_readrecording_base(nde_classes,h5path,group,recpath,filemap),
      hidden(false)
  {
    // Should put code to read basic params here
    // Confirm required nde_classes
    numarrays=1;
    
    
    // read nde_array-numarrays entry
    if (group.attrExists("nde_array-numarrays")) {
      H5::Attribute na_attr = group.openAttribute("nde_array-numarrays");
      H5::DataType na_dtype = na_attr.getDataType();
      
      
      if (na_dtype.getClass() != H5T_INTEGER) {
	throw snde_error("nde_array-numarrays must be of integer type");
      }
      
      na_attr.read(H5::PredType::NATIVE_INT64,&numarrays);
      if (numarrays <= 0) {
	throw snde_error("nde_array-numarrays must be > 0");
      }
      
    }
    
    
  }
  
  std::shared_ptr<recording_base> ndefile_readarray::define_rec(std::shared_ptr<recdatabase> recdb,std::string ownername,void *owner_id)
  {
    std::shared_ptr<channel> loadchan;
    
    {
      std::lock_guard<std::mutex> recdb_admin(recdb->admin);
      auto channel_map_it = recdb->_channels.find(recpath);
      if (channel_map_it != recdb->_channels.end()) {
	loadchan = channel_map_it->second;
      }
    }
    if (!loadchan) {
      loadchan = recdb->define_channel(recpath,ownername,owner_id); // Note: no way (so far) to set hidden flag or storage manager
    }
    
    std::shared_ptr<multi_ndarray_recording> retval = create_recording<multi_ndarray_recording>(recdb,loadchan,owner_id,(size_t)numarrays);

    retval->metadata=metadata;
    
    return retval;
    
  }



  void ndefile_readarray::read(std::shared_ptr<recording_base> rec) // read actual data
  {
    std::shared_ptr<multi_ndarray_recording> mndarray = std::dynamic_pointer_cast<multi_ndarray_recording>(rec);
    assert(mndarray);
    
    assert(numarrays >= 0);
    
    for (size_t arraynum=0; arraynum < (size_t)numarrays; arraynum++) {
      std::string postfix;
      //if (!arraynum) {
      //postfix="";
      //} else {
      postfix=std::string("-")+std::to_string(arraynum);
	//}
      
      // nde_array-name   attribute
      // nde_array-array     dataset
      // nde_array-dimlenF   dataset
      // nde_array-dimlenC   dataset
      
      H5::Attribute nde_array_name_attr = group.openAttribute(ssprintf("nde_array-name%s",postfix.c_str()));
      
      H5::DataType nde_an_dtype = nde_array_name_attr.getDataType();
      if (nde_an_dtype.getClass() != H5T_STRING) {
	throw snde_error("nde_array-name%s for HDF5 group %s should be a string",postfix.c_str(),h5path.c_str());
      }
      H5std_string nde_array_name;
      nde_array_name_attr.read(nde_an_dtype,nde_array_name);

      //snde_warning("reading nde_array: %s",nde_array_name.c_str());
      
      std::vector<snde_index> dimlen;
      bool fortran_order=false;
      
      std::string dimlenCname = ssprintf("nde_array-dimlenC%s",postfix.c_str());
      std::string dimlenFname = ssprintf("nde_array-dimlenF%s",postfix.c_str());
      if (group.nameExists(dimlenCname)) {
	H5::DataSet nde_array_dimlenC_dataset = group.openDataSet(dimlenCname);
	H5::DataSpace nde_adC_dspace = nde_array_dimlenC_dataset.getSpace();
	H5::DataType nde_adC_dtype = nde_array_dimlenC_dataset.getDataType();
	if (nde_adC_dtype.getClass() != H5T_INTEGER) {
	  throw snde_error("nde_array-dimlenC%s for HDF5 group %s should be of integral type",postfix.c_str(),h5path.c_str());
	}
	if (nde_adC_dspace.getSimpleExtentNdims() != 1) {
	  throw snde_error("nde_array-dimlenC%s should have exactly one iterable dimension",postfix.c_str());
	}
	hsize_t dimlen_length=0;
	nde_adC_dspace.getSimpleExtentDims(&dimlen_length,NULL);
	
	std::vector<snde_index> dimlenC(dimlen_length,0);
	
	assert(sizeof(snde_index)==8);
	nde_array_dimlenC_dataset.read(dimlenC.data(),H5::PredType::NATIVE_UINT64,nde_adC_dspace);
	
	//std::vector<snde_index> strides;
	
	//for (size_t dimnum=0;dimnum < dimlenC.size();dimnum++) {
	// C order
	// strides.insert(strides.begin(),stride);
	// stride *= dimlenC.at(dimlenC.size()-dimnum-1);
	//}
	
	
	//mndarray->layouts.at(arraynum) = arraylayout(dimlen,strides);
	
	fortran_order = false;
	dimlen=dimlenC;
	
      } else if (group.nameExists(dimlenFname)) {
	H5::DataSet nde_array_dimlenF_dataset = group.openDataSet(dimlenFname);
	H5::DataSpace nde_adF_dspace = nde_array_dimlenF_dataset.getSpace();
	H5::DataType nde_adF_dtype = nde_array_dimlenF_dataset.getDataType();
	if (nde_adF_dtype.getClass() != H5T_INTEGER) {
	  throw snde_error("nde_array-dimlenF%s for HDF5 group %s should be of integral type",postfix.c_str(),h5path.c_str());
	}
	if (nde_adF_dspace.getSimpleExtentNdims() != 1) {
	  throw snde_error("nde_array-dimlenF%s should have exactly one iterable dimension",postfix.c_str());
	}
	hsize_t dimlen_length=0;
	nde_adF_dspace.getSimpleExtentDims(&dimlen_length,NULL);
	
	std::vector<snde_index> dimlenF(dimlen_length,0);
	
	assert(sizeof(snde_index)==8);
	nde_array_dimlenF_dataset.read(dimlenF.data(),H5::PredType::NATIVE_UINT64,nde_adF_dspace);
	
	//std::vector<snde_index> strides;
	
	//for (size_t dimnum=0;dimnum < dimlenF.size();dimnum++) {
	//  // Fortran order
	//  for (dimnum=0;dimnum < dimlenF.size();dimnum++) {
	//    strides.push_back(stride);
	//    stride *= dimlenF.at(dimnum);
	//  }
	//}
	
	//mndarray->layouts.at(arraynum) = arraylayout(dimlenF,strides);
	
	fortran_order = true;
	dimlen=dimlenF;
	
      } else {
	throw snde_error("Array dimensions (%s or %s) not found for HDF5 group %s",dimlenCname.c_str(),dimlenFname.c_str(),h5path.c_str());
      }
      
      
      
      
      std::string nde_array_array_name = ssprintf("nde_array-array%s",postfix.c_str());
      H5::DataSet nde_array_array_dataset = group.openDataSet(nde_array_array_name);


      H5::Attribute nde_array_nativetype_attr = nde_array_array_dataset.openAttribute(ssprintf("nde_array-nativetype"));
      
      H5::DataType nde_ant_dtype = nde_array_nativetype_attr.getDataType();
      if (nde_ant_dtype.getClass() != H5T_STRING) {
	throw snde_error("nde_array%s-nativetype for HDF5 group %s should be a string",postfix.c_str(),h5path.c_str());
      }
      H5std_string nde_array_nativetype;
      nde_array_nativetype_attr.read(nde_ant_dtype,nde_array_nativetype);

      std::shared_ptr<std::unordered_map<std::string,std::pair<std::function<H5::DataType()>,unsigned>>> by_nativetype = nde_file_map_by_nativetype();
      
      auto nt_mappings_it = by_nativetype->find(nde_array_nativetype);
      if (nt_mappings_it == by_nativetype->end()) {
	throw snde_error("No known native type mapping for type %s for array %d of HDF5 group %s",nde_array_nativetype.c_str(),arraynum,h5path.c_str());
      }


      
      H5::DataSpace nde_aa_dspace = nde_array_array_dataset.getSpace();
      H5::DataType nde_aa_dtype = nde_array_array_dataset.getDataType();
      
      
      
      mndarray->define_array(arraynum,nt_mappings_it->second.second);
      
      mndarray->name_mapping.emplace(nde_array_name,arraynum);
      mndarray->name_reverse_mapping.emplace(arraynum,nde_array_name);
      
      mndarray->allocate_storage(arraynum,dimlen,fortran_order);
      
      
      
      if (nde_aa_dspace.getSimpleExtentNdims() != 1) {
	throw snde_error("nde_array-array%s should have exactly one iterable dimension for HDF5 group %s",postfix.c_str(),h5path.c_str());
      }
      
      hsize_t nelements=0;
      nde_aa_dspace.getSimpleExtentDims(&nelements,NULL);
      if (nelements != mndarray->layouts.at(arraynum).flattened_length()) {
	throw snde_error("nde_array-array%s number of elements (%llu) does not exactly match product of dimlen dimensions (%llu) for hdf5 group %s",postfix.c_str(),(unsigned long long)nelements,(unsigned long long)mndarray->layouts.at(arraynum).flattened_length(),h5path.c_str());	  
      }
      
      
      // Notes potential on memory mapping
      //  * getOffset() method can get the location in the underlying file
      //  * Will need an alternate storage manager to accommodate a
      //    pre-existing memory mapped file. 
      //  * Need to make sure there is a single chunk and no filters
      //     * There is a get_num_chunks() function that is probably useful
      //     * There is a get_chunk_info() function with a filter_mask out
      //       parameter that will indicate filters
      // https://gist.github.com/maartenbreddels/09e1da79577151e5f7fec660c209f06e
      
      nde_array_array_dataset.read(mndarray->void_shifted_arrayptr(arraynum),nt_mappings_it->second.first(),nde_aa_dspace);
      
      
      
      
    }
    rec->mark_metadata_done();
    rec->mark_data_ready();
  }

  ndefile_readgroup::ndefile_readgroup(const std::set<std::string> &nde_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<nde_loadrecording_map> filemap) :
    ndefile_readrecording_base(nde_classes,h5path,group,recpath,filemap)
  {
    // Should put code to read basic params here
    // Confirm required nde_classes
    
    if (nde_classes.find("nde_group") == nde_classes.end()) {
      throw snde_error("HDF5 Group %s is not an nde_group",h5path.c_str());
    }
    
    
    H5::Attribute nde_group_version_attr = group.openAttribute("nde_group-version");
    H5::DataType nde_gv_dtype = nde_group_version_attr.getDataType();
    if (nde_gv_dtype.getClass() != H5T_STRING) {
      throw snde_error("nde_group-version for hdf5 group %s should be a string",h5path.c_str());
    }
    nde_group_version_attr.read(nde_gv_dtype,group_version);
    
    H5::Group subgroups_group = group.openGroup("nde_group-subgroups");
    int idx=0;
    std::vector<std::string> subgroup_names;
    subgroups_group.iterateElems(".",&idx,&ndefile_iterate_subgroup_names,(void *)&subgroup_names);
    
    for (auto && subgroup_name: subgroup_names) {
      std::string subgroup_recpath = recdb_path_join(recpath+"/",subgroup_name);
      std::shared_ptr<ndefile_readrecording_base> subgroup_loader = ndefile_loadrecording(h5path+"/nde_group-subgroups/"+subgroup_name,subgroups_group.openGroup(subgroup_name),subgroup_recpath,filemap);
      filemap->emplace(subgroup_recpath,std::make_pair(subgroup_loader,nullptr));
      
      group_subloaders.push_back(std::make_tuple(subgroup_name,subgroup_loader));
    }
    
  }
  
  std::shared_ptr<recording_base> ndefile_readgroup::define_rec(std::shared_ptr<recdatabase> recdb,std::string ownername,void *owner_id)
  {
    std::shared_ptr<channel> loadchan;
    
    {
      std::lock_guard<std::mutex> recdb_admin(recdb->admin);
      auto channel_map_it = recdb->_channels.find(recpath);
      if (channel_map_it != recdb->_channels.end()) {
	loadchan = channel_map_it->second;
      }
    }
    if (!loadchan) {
      loadchan = recdb->define_channel(recpath,ownername,owner_id); // Note: no way (so far) to set hidden flag or storage manager
    }
    
    // ***!!! Should we provide the group with an explicit order that matches the order in the file???
    std::shared_ptr<recording_group> retval = create_recording<recording_group>(recdb,loadchan,owner_id); //,nullptr);
    retval->metadata=metadata;

    return retval;
  }

  H5::Attribute AddStringToH5AttributeArray(H5::H5Object *parent,H5::Attribute arrayattr,std::string str)
  {
    H5::DataType dtype = arrayattr.getDataType();
    H5::DataSpace dspace = arrayattr.getSpace();
    std::string attrname = arrayattr.getName();

    if (dspace.getSimpleExtentNdims() != 1) {
      throw snde_error("AddStringToH5AttributeArray: Attribute %s should have exactly one iterable dimension",arrayattr.getName().c_str());
    }
    if (dtype.getClass() != H5T_STRING) {
      throw snde_error("AddStringToH5AttributeArray: Attribute %s should be an array of strings",arrayattr.getName().c_str());
    }

    std::vector<std::string> array;
    
    hsize_t existing_num=0;
    dspace.getSimpleExtentDims(&existing_num);

    char **array_strings = new char*[existing_num];
    
    arrayattr.read(dtype,(void *)array_strings);
    
    for (hsize_t cnt=0;cnt < existing_num;cnt++) {
      // Per https://stackoverflow.com/questions/43722194/reading-a-string-array-hdf5-attribute-in-c
      // we actually have to call delete[] on each string, which seems
      // odd  (maybe they really mean free()?), but....
      array.push_back(array_strings[cnt]);
      delete[] array_strings[cnt];
      
    }
    delete[] array_strings;

    hsize_t new_num = existing_num+1;

    // add new entry
    array.push_back(str);

    // create shadow array of char *
    std::vector<const char *> array_ptrs;

    for (hsize_t cnt=0;cnt < new_num;cnt++) {
      array_ptrs.push_back(array.at(cnt).c_str()); 
    }

    // resize dataspace
    dspace.setExtentSimple(1,&new_num);

    // remove/replace attribute
    parent->removeAttr(attrname);

    H5::Attribute new_arrayattr = parent->createAttribute(attrname,dtype,dspace);
    // rewrite data
    new_arrayattr.write(dtype,(void *)array_ptrs.data());

    return new_arrayattr;
  }

  ndefile_writerfunc ndefile_lookup_writer_function(std::shared_ptr<recording_base> rec_to_write)
  {
    std::shared_ptr<ndefile_saver_map> saver_reg = ndefile_saver_registry();

    for (size_t rec_classes_index = rec_to_write->rec_classes.size()-1;;rec_classes_index--) {
      
      std::string rec_class = rec_to_write->rec_classes.at(rec_classes_index).classname;
      
      auto rec_class_saver = saver_reg->find(rec_class);
      if (rec_class_saver != saver_reg->end()) {
	// got one!
	return rec_class_saver->second.second;
      }
      
      if (!rec_classes_index) {
	throw snde_error("No registered writer function for object of class %s or its base classes",rec_to_write->rec_classes.at(rec_to_write->rec_classes.size()-1).classname.c_str());
      }
    }
    
    
  }

  ndefile_writerfunc ndefile_lookup_writer_function_by_class(std::string classname)
  {
    std::shared_ptr<ndefile_saver_map> saver_reg = ndefile_saver_registry();
    
    
    auto rec_class_saver = saver_reg->find(classname);
    if (rec_class_saver != saver_reg->end()) {
      // got one!
      return rec_class_saver->second.second;
    }
    
    throw snde_error("No registered writer function for object of class %s",classname.c_str());
  }

  
  std::pair<ndefile_saverfunc,std::string> ndefile_lookup_saver_function(std::shared_ptr<recording_base> rec_to_write)
  {
    std::shared_ptr<ndefile_saver_map> saver_reg = ndefile_saver_registry();

    for (size_t rec_classes_index = rec_to_write->rec_classes.size()-1;;rec_classes_index--) {
      
      std::string rec_class = rec_to_write->rec_classes.at(rec_classes_index).classname;
      
      auto rec_class_saver = saver_reg->find(rec_class);
      if (rec_class_saver != saver_reg->end()) {
	// got one!
	return std::make_pair(rec_class_saver->second.first,rec_class);
      }
      
      if (!rec_classes_index) {
	throw snde_error("No registered saver function for object of class %s or its base classes",rec_to_write->rec_classes.at(rec_to_write->rec_classes.size()-1).classname.c_str());
      }
    }
    
    
  }

  void ndefile_write_superclass(std::shared_ptr<recording_base> recording_to_save,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath,H5::Group recgroup, std::string classname)
  {
    std::string superclass_name;
    if (!recording_to_save) {
      // nullptr recording only possible with snde::recording_group
      superclass_name="snde::recording_base";
    } else {
      size_t rec_classes_pos;
      for (rec_classes_pos = 0;rec_classes_pos < recording_to_save->rec_classes.size(); rec_classes_pos++) {
	if (recording_to_save->rec_classes.at(rec_classes_pos).classname==classname) {
	  superclass_name = recording_to_save->rec_classes.at(rec_classes_pos-1).classname;
	  break;
	}
      }
      if (rec_classes_pos == recording_to_save->rec_classes.size()) {
	throw snde_error("ndefile_write_superclass: current class %s not found in rec_classes for recording of class %s",classname.c_str(),typeid(recording_to_save).name());	
      }
      
    }

    ndefile_writerfunc superclass_writer = ndefile_lookup_writer_function_by_class(superclass_name);

    superclass_writer(recording_to_save,H5Obj,pathstack,groupstack,writepath,recgroup);
    
  }

  
  void ndefile_write_recording_base(std::shared_ptr<recording_base> recording_to_save,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath,H5::Group recgroup)
  {
    H5::StrType nde_str_dtype = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataType nde_int_file_dtype = H5::IntType(H5::PredType::STD_I64LE);
    H5::DataType nde_int_memory_dtype = H5::IntType(H5::PredType::NATIVE_INT64);
    H5::DataType nde_unsigned_file_dtype = H5::IntType(H5::PredType::STD_U64LE);
    H5::DataType nde_unsigned_memory_dtype = H5::IntType(H5::PredType::NATIVE_UINT64);
    H5::DataType nde_bool_file_dtype = H5::IntType(H5::PredType::STD_U8LE);
    H5::DataType nde_hbool_memory_dtype = H5::IntType(H5::PredType::NATIVE_HBOOL);

    H5::DataType nde_double_file_dtype = H5::FloatType(H5::PredType::IEEE_F64LE);
    H5::DataType nde_double_memory_dtype = H5::FloatType(H5::PredType::NATIVE_DOUBLE);

    nde_str_dtype.setCset(H5T_CSET_UTF8);
    nde_str_dtype.setStrpad(H5T_STR_NULLTERM);
    hsize_t dim = 0;
    H5::DataSpace nde_c_dspace = H5::DataSpace(1,&dim); // size of 0 for now (contains nde_recording) -- can expand later

    H5::Attribute nde_classes = recgroup.createAttribute("nde-classes",nde_str_dtype,nde_c_dspace);
    nde_classes = AddStringToH5AttributeArray(&recgroup,nde_classes,"nde_recording");

    H5::Attribute nde_class_tags = recgroup.createAttribute("nde_class-tags",nde_str_dtype,nde_c_dspace);

    H5::DataSpace nde_scalar_dspace = H5::DataSpace(H5S_SCALAR); // size of 0 for now (contains nde_recording) -- can expand later
    
    H5::Attribute nde_rec_label = recgroup.createAttribute("nde_recording-label",nde_str_dtype,nde_scalar_dspace);

    std::string pathseg;
    size_t slashpos = writepath.rfind('/');
    if (slashpos==std::string::npos) {
      throw snde_error("writepath contains no slashes (!)");
    }
    if (slashpos == writepath.size()-1) {
      // group: ends with a slash
      if (writepath.size() > 1) {
	slashpos = writepath.rfind('/',writepath.size()-2);
	if (slashpos==std::string::npos) {
	  throw snde_error("writepath contains leadng slash (!)");
	}
      }
    }

    std::string label = writepath.substr(slashpos+1); // last writepath segment
    
    nde_rec_label.write(nde_str_dtype,label);

    H5::Attribute nde_rec_ver = recgroup.createAttribute("nde_recording-version",nde_str_dtype,nde_scalar_dspace);
    nde_rec_ver.write(nde_str_dtype,std::string("0.0.0"));

    H5::Group md_group = recgroup.createGroup("nde_recording-metadata");

    if (recording_to_save) {
      for (auto && mdname_mdval: recording_to_save->metadata->metadata) {
	
	const metadatum &mdval=mdname_mdval.second;
	H5::Attribute md_attr,md_units_attr;
	hbool_t hboolval;
	
	switch (mdval.md_type) {
	case MWS_MDT_INT:
	  md_attr = md_group.createAttribute(mdval.Name,nde_int_file_dtype,nde_scalar_dspace);
	  md_attr.write(nde_int_memory_dtype,(void*)mdval.intval);
	  break;
	  
	case MWS_MDT_STR:
	  md_attr = md_group.createAttribute(mdval.Name,nde_str_dtype,nde_scalar_dspace);
	  md_attr.write(nde_str_dtype,mdval.strval);
	  break;
	case MWS_MDT_DBL:
	  md_attr = md_group.createAttribute(mdval.Name,nde_double_file_dtype,nde_scalar_dspace);
	  md_attr.write(nde_double_memory_dtype,(void*)&mdval.dblval);
	  break;
	case MWS_MDT_DBL_UNITS:
	  md_attr = md_group.createAttribute(mdval.Name,nde_double_file_dtype,nde_scalar_dspace);
	  md_attr.write(nde_double_memory_dtype,(void*)&mdval.dblval);
	  md_units_attr = md_group.createAttribute(mdval.Name+"-units",nde_str_dtype,nde_scalar_dspace);
	  md_units_attr.write(nde_str_dtype,mdval.dblunits);
	  break;
	  
	case MWS_MDT_UNSIGNED:
	  md_attr = md_group.createAttribute(mdval.Name,nde_unsigned_file_dtype,nde_scalar_dspace);
	  md_attr.write(nde_unsigned_memory_dtype,(void*)&mdval.unsignedval);
	  break;
	  
	case MWS_MDT_BOOL:
	  md_attr = md_group.createAttribute(mdval.Name,nde_bool_file_dtype,nde_scalar_dspace);
	  hboolval = mdval.boolval;
	  md_attr.write(nde_hbool_memory_dtype,(void*)&hboolval);
	  break;
	  
	case MWS_MDT_NONE:
	default:
	  throw snde_error("ndefile_write_recording_base: invalid MWS_MDT_ datatype %u for metadata entry %s",mdval.md_type,mdval.Name.c_str()); 
	  
	}
      }
    }
  }

  
  void ndefile_write_recording_group(std::shared_ptr<recording_base> recording_to_save,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath,H5::Group recgroup)
  {
    ndefile_write_superclass(recording_to_save,H5Obj,pathstack,groupstack,writepath,recgroup,"snde::recording_group");

    H5::Attribute nde_classes = recgroup.openAttribute("nde-classes");
    nde_classes = AddStringToH5AttributeArray(&recgroup,nde_classes,"nde_group");
    
    H5::StrType nde_str_dtype = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);
    nde_str_dtype.setCset(H5T_CSET_UTF8);
    nde_str_dtype.setStrpad(H5T_STR_NULLTERM);

    H5::DataSpace nde_scalar_dspace = H5::DataSpace(H5S_SCALAR); // size of 0 for now (contains nde_recording) -- can expand later
    
    
    H5::Attribute nde_group_ver = recgroup.createAttribute("nde_group-version",nde_str_dtype,nde_scalar_dspace);
    nde_group_ver.write(nde_str_dtype,std::string("0.0.0"));

    recgroup.createGroup("nde_group-subgroups");

    
  }


  void ndefile_write_multi_ndarray_recording(std::shared_ptr<recording_base> recording_to_save,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath,H5::Group recgroup)
  {
    ndefile_write_superclass(recording_to_save,H5Obj,pathstack,groupstack,writepath,recgroup,"snde::multi_ndarray_recording");

    H5::Attribute nde_classes = recgroup.openAttribute("nde-classes");
    nde_classes = AddStringToH5AttributeArray(&recgroup,nde_classes,"nde_array");

    std::shared_ptr<multi_ndarray_recording> mnr_to_save=std::dynamic_pointer_cast<multi_ndarray_recording>(recording_to_save);
    assert(mnr_to_save);

    H5::StrType nde_str_dtype = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);
    nde_str_dtype.setCset(H5T_CSET_UTF8);
    nde_str_dtype.setStrpad(H5T_STR_NULLTERM);

    H5::DataType nde_unsigned_file_dtype = H5::IntType(H5::PredType::STD_U64LE);
    H5::DataType nde_unsigned_memory_dtype = H5::IntType(H5::PredType::NATIVE_UINT64);

    H5::DataSpace nde_scalar_dspace = H5::DataSpace(H5S_SCALAR); // size of 0 for now (contains nde_recording) -- can expand later

    
    
    H5::Attribute nde_array_ver = recgroup.createAttribute("nde_array-version",nde_str_dtype,nde_scalar_dspace);
    nde_array_ver.write(nde_str_dtype,std::string("0.0.0"));


    H5::Attribute nde_array_numarrays = recgroup.createAttribute("nde_array-numarrays",nde_unsigned_file_dtype,nde_scalar_dspace);
    uint64_t numarrays = mnr_to_save->mndinfo()->num_arrays; 
    nde_array_numarrays.write(nde_unsigned_memory_dtype,&numarrays);
    
    for (uint64_t arraynum=0;arraynum < numarrays;arraynum++) {

      arraylayout &layout=mnr_to_save->layouts.at(arraynum);
      H5::Attribute nde_array_name = recgroup.createAttribute(ssprintf("nde_array-name-%u",(unsigned)arraynum),nde_str_dtype,nde_scalar_dspace);
      nde_array_name.write(nde_str_dtype,mnr_to_save->name_reverse_mapping.at(arraynum));

      hsize_t dim = layout.flattened_length();
      H5::DataSpace array_dspace = H5::DataSpace(1,&dim);

      std::shared_ptr<std::unordered_map<unsigned,std::pair<std::function<H5::DataType()>,std::string>>> by_typenum=nde_file_map_by_typenum();

      auto typenum_map_it = by_typenum->find(mnr_to_save->ndinfo(arraynum)->typenum);
      if (typenum_map_it==by_typenum->end()) {
	throw snde_error("ndefile_write_multi_ndarray_recording(): No HDF5 type found for channel %s type %s (%u)",mnr_to_save->info->name,rtn_typenamemap.at(mnr_to_save->ndinfo(arraynum)->typenum),mnr_to_save->ndinfo(arraynum)->typenum);
      }
      H5::DataType array_dtype = typenum_map_it->second.first();
      
      
      
      H5::DataSet array_dset = recgroup.createDataSet(ssprintf("nde_array-array-%u",(unsigned)arraynum),array_dtype,array_dspace);

      H5::Attribute nativetype_attr = array_dset.createAttribute("nde_array-nativetype",nde_str_dtype,nde_scalar_dspace);
      nativetype_attr.write(nde_str_dtype,typenum_map_it->second.second);

      hsize_t ndim = layout.dimlen.size();
      
      H5::DataSpace dimlen_dspace = H5::DataSpace(1,&ndim);

      if (layout.is_f_contiguous()) {
	H5::DataSet dimlenF_dset = recgroup.createDataSet(ssprintf("nde_array-dimlenF-%u",(unsigned)arraynum),nde_unsigned_file_dtype,dimlen_dspace);
#if (SIZEOF_SNDE_INDEX==8)
	dimlenF_dset.write((void *)layout.dimlen.data(),H5::IntType(H5::PredType::NATIVE_UINT64));
#else
	assert(SIZEOF_SNDE_INDEX==4);
	dimlenF_dset.write((void *)layout.dimlen.data(),H5::IntType(H5::PredType::NATIVE_UINT32));
#endif
	array_dset.write(mnr_to_save->void_shifted_arrayptr(arraynum),array_dtype);
      } else if (layout.is_c_contiguous()) {
	H5::DataSet dimlenC_dset = recgroup.createDataSet(ssprintf("nde_array-dimlenC-%u",(unsigned)arraynum),nde_unsigned_file_dtype,dimlen_dspace);
#if (SIZEOF_SNDE_INDEX==8)
	dimlenC_dset.write((void *)layout.dimlen.data(),H5::IntType(H5::PredType::NATIVE_UINT64));
#else
	assert(SIZEOF_SNDE_INDEX==4);
	dimlenC_dset.write((void *)layout.dimlen.data(),H5::IntType(H5::PredType::NATIVE_UINT32));
#endif
	array_dset.write(mnr_to_save->void_shifted_arrayptr(arraynum),array_dtype);
      } else {
	// existing storage not C or F contiguous
	// create temporary C contiguous copy
	H5::DataSet dimlenC_dset = recgroup.createDataSet(ssprintf("nde_array-dimlenC-%u",(unsigned)arraynum),nde_unsigned_file_dtype,dimlen_dspace);
#if (SIZEOF_SNDE_INDEX==8)
	dimlenC_dset.write((void *)layout.dimlen.data(),H5::IntType(H5::PredType::NATIVE_UINT64));
#else
	assert(SIZEOF_SNDE_INDEX==4);
	dimlenC_dset.write((void *)layout.dimlen.data(),H5::IntType(H5::PredType::NATIVE_UINT32));
#endif

	std::shared_ptr<ndarray_recording_ref> array_ref=mnr_to_save->reference_ndarray(arraynum);

	size_t elementsize = mnr_to_save->ndinfo(arraynum)->elementsize;
	char *c_indexed_array = (char*)malloc(dim*elementsize);
	char *orig_array=(char*)mnr_to_save->void_shifted_arrayptr(arraynum);
	for (size_t elemcnt=0;elemcnt < dim;elemcnt++) {
	  size_t off = array_ref->element_offset(elemcnt,false)*elementsize;
	  memcpy(c_indexed_array+elemcnt*elementsize,orig_array+off,elementsize);
	}
	array_dset.write(c_indexed_array,array_dtype);
	free(c_indexed_array);
      }
 

    }
    
  }


  std::map<std::string,channel_state>::iterator ndefile_save_generic_recording(std::shared_ptr<std::map<std::string,channel_state>> *channel_map,std::map<std::string,channel_state>::iterator starting_iterator,std::shared_ptr<recording_base> recording_to_save,std::string saveclass,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath)
  {

    ndefile_writerfunc writer_function = ndefile_lookup_writer_function_by_class(saveclass);

    assert(pathstack->size() > 0);
    assert(groupstack->size() == pathstack->size());
    
    assert(writepath != pathstack->at(pathstack->size()-1)); // if these match we are a group not a recording and you should be using ndefile_save_generic_group instead of this

    assert(channel_map);

    H5::Group subgroups = groupstack->at(groupstack->size()-1).openGroup("nde_group-subgroups");
    H5::Group recgroup = subgroups.createGroup(writepath.substr(pathstack->at(pathstack->size()-1).size()));

    writer_function(recording_to_save,H5Obj,pathstack,groupstack,writepath,recgroup);

    std::map<std::string,channel_state>::iterator ending_iterator = starting_iterator;
    ending_iterator++;
    return ending_iterator;
    
  }

  
  std::map<std::string,channel_state>::iterator ndefile_save_generic_group(std::shared_ptr<std::map<std::string,channel_state>> *channel_map,std::map<std::string,channel_state>::iterator starting_iterator,std::shared_ptr<recording_base> recording_to_save,std::string saveclass,H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,std::string writepath)
  {
    // Note for recording groups, because they can be implicit:
    //  * We can be called with a nullptr channel_map, which means starting_iterator is invalid,
    //    and should just be returned directly without incrementation
    //  * recording_to_save can be nullptr

    // Also for recording_group and any subclasses, the H5::Group to write is already created and at the top
    // of the groupstack; we just need to fill it out

    ndefile_writerfunc writer_function = ndefile_lookup_writer_function_by_class(saveclass);
    H5::Group recgroup = groupstack->at(groupstack->size()-1);
    writer_function(recording_to_save,H5Obj,pathstack,groupstack,writepath,recgroup);

    
    if (channel_map) {
      std::map<std::string,channel_state>::iterator ending_iterator=starting_iterator;
      ending_iterator++;
      return ending_iterator;
    } else {
      return starting_iterator;
    }
  }

  
  std::shared_ptr<ndefile_readrecording_base> ndefile_loadrecording(std::string h5path,H5::Group group,std::string recpath,std::shared_ptr<nde_loadrecording_map> filemap)
  {

  
    
    H5::Attribute nde_classes_attr = group.openAttribute("nde-classes");
    H5::DataType nde_c_dtype = nde_classes_attr.getDataType();
    H5::DataSpace nde_c_dspace = nde_classes_attr.getSpace();
    
    if (nde_c_dspace.getSimpleExtentNdims() != 1) {
      throw snde_error("nde-classes attribute for hdf5 group %s should have exactly one iterable dimension",h5path.c_str());
    }
    if (nde_c_dtype.getClass() != H5T_STRING) {
      throw snde_error("nde-classes attribute for hdf5 group %s should be an array of strings",h5path.c_str());
    }
    
    std::set<std::string> nde_classes;
    
    // number of classes
    hsize_t nde_c_num=0;
    nde_c_dspace.getSimpleExtentDims(&nde_c_num);
    char **class_strings = new char*[nde_c_num];
    H5::StrType nde_c_strtype(H5::PredType::C_S1,H5T_VARIABLE);
    nde_c_strtype.setCset(H5T_CSET_UTF8);
    nde_c_strtype.setStrpad(H5T_STR_NULLTERM);
    
    nde_classes_attr.read(nde_c_strtype,(void *)class_strings);
    
    size_t nde_classes_size=nde_c_dtype.getSize();
    for (size_t class_idx=0;class_idx < nde_c_num;class_idx++) {
      // Per https://stackoverflow.com/questions/43722194/reading-a-string-array-hdf5-attribute-in-c
      // we actually have to call delete[] on each string, which seems
      // odd  (maybe they really mean free()?), but....
      nde_classes.emplace(class_strings[class_idx]);
      delete[] class_strings[class_idx];
    }
    delete[] class_strings;

    std::shared_ptr<ndefile_loader_map> registry=ndefile_loader_registry();
    const ndefile_loaderfunc *deepest_loaderfunc=nullptr;
    unsigned deepest_depth=0;
    std::string deepest_class=""; 
    
    for (auto && classname: nde_classes) {
      
      ndefile_loader_map::iterator registry_it = registry->find(classname);
      
      if (registry_it != registry->end()) {
	// unpack registry entry
	const std::pair<unsigned,ndefile_loaderfunc> &depth_loaderfunc=registry_it->second;
	unsigned depth = depth_loaderfunc.first;
	const ndefile_loaderfunc &loaderfunc=depth_loaderfunc.second;
	
	if (depth > deepest_depth) {
	  deepest_loaderfunc = &loaderfunc;
	  deepest_depth = depth;
	  deepest_class = classname; 
	} else if (depth == deepest_depth) {
	  throw snde_error("ndefile_loadrecording: Error loading recording %s: Recording has two classes %s and %s which are at the same depth (%u) in the hierarchy, which is not allowed.", recpath.c_str(), deepest_class.c_str(), classname.c_str(),depth);
	}
      }
      
    }
    
    if (!deepest_loaderfunc) {
      throw snde_error("ndefile_loadrecording: Recording %s does not specify any known classes",recpath.c_str());
    }
    
    
    //snde_warning("Using class %s for hdf5 group %s",deepest_class.c_str(),h5path.c_str());
    std::shared_ptr<ndefile_readrecording_base> readerobj = (*deepest_loaderfunc)(nde_classes,h5path,group,recpath,filemap);  
    filemap->emplace(recpath,std::make_pair(readerobj,nullptr));
    
    
    
    
    
    return readerobj;
  }
  
  
  
  
  std::shared_ptr<nde_loadrecording_map> ndefile_loadfile(std::shared_ptr<recdatabase> recdb,std::string ownername,void *owner_id,std::string filename,std::string recpath /* ="/" */ ) // add filter function parameter or specific recording to request to limit what is loaded? 
  {
    //std::shared_ptr<nde_file> ndefile = std::make_shared<nde_file>(filename);
    H5::H5File H5Obj(filename,H5F_ACC_RDONLY);
    
    
    H5::Group rootgroup = H5Obj.openGroup("/");
    std::shared_ptr<nde_loadrecording_map> filemap = std::make_shared<nde_loadrecording_map>();

    std::shared_ptr<ndefile_readrecording_base> readerobj;
      
    readerobj = ndefile_loadrecording("/",rootgroup,recpath,filemap);
    //std::shared_ptr<recording_base> new_rec = readerobj->define_rec(recdb,ownername,owner_id);
  

    // iterate through all recordings and define them
    for (auto && recname_loaderptr_recordingptr: *filemap) {
      std::string recname = std::get<0>(recname_loaderptr_recordingptr);
      std::pair<std::shared_ptr<ndefile_readrecording_base>,std::shared_ptr<recording_base>> &loaderptr_recordingptr=std::get<1>(recname_loaderptr_recordingptr);
      // define recording and assign into filemap
      loaderptr_recordingptr.second = loaderptr_recordingptr.first->define_rec(recdb,ownername,owner_id);
    }
    
    // iterate through all recordings and load them
    for (auto && recname_loaderptr_recordingptr: *filemap) {
      std::string recname = std::get<0>(recname_loaderptr_recordingptr);
      std::pair<std::shared_ptr<ndefile_readrecording_base>,std::shared_ptr<recording_base>> &loaderptr_recordingptr=std::get<1>(recname_loaderptr_recordingptr);
      // define recording and assign into filemap
      loaderptr_recordingptr.first->read(loaderptr_recordingptr.second);
    }
    
    
    H5Obj.close();
    return filemap; 
  }

  bool ndefile_savefile_pathstack_top_is_start_of(std::vector<std::string> *pathstack,const std::string &writepath_group)
  {
    assert(pathstack->size() > 0); // should always be at least "/" on it
    
    auto top_it = pathstack->end()-1;

    size_t top_size = top_it->size();
    if (writepath_group.substr(0,top_size) == *top_it) {
      // pathstack top is the start of writepath_group
      return true;
    }
    return false;
  }

  void ndefile_savefile_pop_to_common(std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,const std::string &writepath)
  {

    std::string writepath_group = recdb_path_context(writepath);
    
    assert(pathstack->size() > 0); // should always be at least "/" on it

    while (!ndefile_savefile_pathstack_top_is_start_of(pathstack,writepath_group)) {
      // remove top stack element
      pathstack->pop_back();
      groupstack->pop_back();
    }
    
  }

  void ndefile_savefile_push_to_group(H5::H5File *H5Obj,std::vector<std::string> *pathstack,std::vector<H5::Group> *groupstack,const std::string &writepath)
  {
    bool writepath_is_group = writepath.at(writepath.size()-1)=='/';  // Name ends in slash: this should be a group or subclass of a group
    std::string writepath_group = recdb_path_context(writepath);

    std::shared_ptr<ndefile_saver_map> saver_reg = ndefile_saver_registry();


    // while the pathstack is empty or the top entry doesn't match the writepath group
    while (!pathstack->size() || *(pathstack->end()-1) != writepath_group) {
      std::string new_piece_fullname;
      std::string new_piece_name;
      
      if (pathstack->size()) {
	auto top_it = pathstack->end()-1;
	size_t top_size = top_it->size();

	assert(*top_it == writepath_group.substr(0,top_size)); // writepath_group must start with our current location

	std::string new_pieces=writepath_group.substr(top_size); // NOT including leading slash
	size_t new_pieces_slash_pos = new_pieces.find('/');
	assert(new_pieces_slash_pos != std::string::npos);

	// new_pieces_slash_pos equivalent to number of chars in first new piece, not including any slashes
	new_piece_name = new_pieces.substr(0,new_pieces_slash_pos+1); // including trailng slash
	new_piece_fullname = *top_it + new_piece_name;
      } else {
	new_piece_name = "/";
	new_piece_fullname = "/";
      }
      // push this onto the stack
      pathstack->push_back(new_piece_fullname);

      // Create corresponding group entry in file and on groupstack
      H5::Group newgroup;
      
      if (groupstack->size()) {
	H5::Group subgroups = (groupstack->end()-1)->openGroup("nde_group-subgroups");
	newgroup = subgroups.createGroup(new_piece_name);
      } else {
	newgroup = H5Obj->openGroup("/"); // root group
      }

      groupstack->push_back(newgroup);

      // lookup and call plug-in to initialize bare group
      // ... unless writepath_is_group and new_piece_fullname matches
      // writepath, in which case our caller will do it.
      if ( !(writepath_is_group && new_piece_fullname==writepath) ) {
	auto saver_reg_it = saver_reg->find("snde::recording_group");
	if (saver_reg_it==saver_reg->end()) {
	  throw snde_error("ndefile_savefile_push_to_group(): No saver plugin defined for snde::recording_group.");
	}

	ndefile_saverfunc saver_function = saver_reg_it->second.first;
	// Call the saver function, which increments our iterator (so that it can skip over
	// an entire group, if desired. It also might replace our channel_map in certain situations
	
	saver_function(nullptr,std::map<std::string,channel_state>::iterator(),nullptr,"snde::recording_group",H5Obj,pathstack,groupstack,new_piece_fullname);
	
      }
      
    }
  }

  
  void ndefile_savefile(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_set_state> rss_or_globalrev,std::string filename,std::string grouppath/*="/"*/)
  {

    if (!grouppath.size() || grouppath.at(grouppath.size()-1) != '/') {
      throw snde_error("ndefile_savefile: grouppath must have trailing slash");
    }

    if (grouppath.at(0) != '/') {
      throw snde_error("ndefile_savefile: grouppath must have leading slash");
    }

    
    std::vector<std::string> pathstack; // a stack of group paths used to
    // implicitly push/pop groups so as to create a hierarchical nde_file
    // structure from the flat rss->recstatus->channel_map

    std::vector<H5::Group> groupstack; // parallel stack of HDF5 groups

    H5::H5File H5Obj(filename,H5F_ACC_TRUNC); // Open a new output file

    //pathstack.push_back("/"); // start at the root
    
    // Obtain the root group
    //H5::Group rootgroup = H5Obj.openGroup("/");
    //groupstack.push_back(rootgroup);

    std::shared_ptr<std::map<std::string,channel_state>> channel_map = rss_or_globalrev->recstatus.channel_map;
    
    std::map<std::string,channel_state>::iterator group_it,next_it;
    for (
	 group_it = channel_map->lower_bound(grouppath);
	 group_it != channel_map->end() && group_it->first.substr(0,grouppath.size()) == grouppath; // valid entry that starts with grouppath
	 group_it=next_it) {

      std::string writepath = group_it->first.substr(grouppath.size()-1); // writepath has initial grouppath clipped (except for leading slash)

      assert(writepath.size() > 0);
      
      if (pathstack.size() > 0) {
	ndefile_savefile_pop_to_common(&pathstack,&groupstack,writepath);
	assert(pathstack.size() > 0);
      }
      ndefile_savefile_push_to_group(&H5Obj,&pathstack,&groupstack,writepath);

      // look up writer plugin
      std::shared_ptr<recording_base> rec_to_write = group_it->second.rec();
      if (!rec_to_write) {
	throw snde_error("Attempting to save recording for %s that has not been defined yet",group_it->first.c_str());
      }
      int info_state = rec_to_write->info_state;
      if (!((info_state & SNDE_RECS_FULLYREADY) == SNDE_RECS_FULLYREADY)) {
	throw snde_error("Attempting to save recording for %s that is not FULLYREADY (info->state=0x%x)",group_it->first.c_str(),(unsigned)info_state);
      }
      assert(rec_to_write->rec_classes.size() > 0);

      // look up the saver function to use in the saver registry
      ndefile_saverfunc saver_function;
      std::string saver_class;
      std::tie(saver_function,saver_class) = ndefile_lookup_saver_function(rec_to_write);
      
      // Call the saver function, which increments our iterator (so that it can skip over
      // an entire group, if desired. It also might replace our channel_map in certain situations

      next_it = saver_function(&channel_map,group_it,rec_to_write,saver_class,&H5Obj,&pathstack,&groupstack,writepath);
    }
    
  }



  // register nde_array and nde_group loaders

  int registered_nde_array_loader = register_ndefile_loader_class<ndefile_readarray>("nde_array",2); // depth of 1 = recording_base, depth of 2 = immediate subclass of recording_base, etc. 			    

  int registered_nde_group_loader = register_ndefile_loader_class<ndefile_readgroup>("nde_group",2); // depth of 1 = recording_base, depth of 2 = immediate subclass of recording_base, etc. 			    


  // register savers
  int registered_recording_base_saver = register_ndefile_saver_function("snde::recording_base",ndefile_save_generic_recording,ndefile_write_recording_base);
  int registered_recording_group_saver = register_ndefile_saver_function("snde::recording_group",ndefile_save_generic_group,ndefile_write_recording_group);
  int registered_multi_ndarray_recording_saver = register_ndefile_saver_function("snde::multi_ndarray_recording",ndefile_save_generic_recording,ndefile_write_multi_ndarray_recording);


  // register data types
  static int added_float = add_nde_file_nativetype_mapping("H5T_NATIVE_FLOAT",  [] { return H5::PredType::NATIVE_FLOAT; }, SNDE_RTN_FLOAT32);
  static int added_double = add_nde_file_nativetype_mapping("H5T_NATIVE_DOUBLE", [] { return H5::PredType::NATIVE_DOUBLE; }, SNDE_RTN_FLOAT64);
  static int added_int8 = add_nde_file_nativetype_mapping("H5T_NATIVE_INT8",  [] { return H5::PredType::NATIVE_INT8; }, SNDE_RTN_INT8);
  

};
