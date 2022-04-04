
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


  // Architecture:
  // Creating an object we need to instantiate the deepest subclass
  // But we want to share code for building the parent class,
  // which can be easily handled by the constructor chain.
  //
  // We can't actually identify where to go until we have at least
  // read the class list. 

  // So we have a plugin database of class handlers with depth indices
  // and instantiate the deepest class we can


  typedef std::map<std::string,std::pair<unsigned,ndefile_loaderfunc>> ndefile_loader_map;
  

  static std::shared_ptr<ndefile_loader_map> *_ndefile_loader_registry; // default-initialized to nullptr


  SNDE_API std::unordered_map<std::string,std::pair<H5::DataType,unsigned>> nde_file_nativetype_mappings({
      {"H5T_NATIVE_FLOAT", { H5::PredType::NATIVE_FLOAT, SNDE_RTN_FLOAT32 } },
      {"H5T_NATIVE_DOUBLE", { H5::PredType::NATIVE_DOUBLE, SNDE_RTN_FLOAT64 } },
      {"H5T_NATIVE_INT8", { H5::PredType::NATIVE_INT8, SNDE_RTN_INT8 } },
    });
  

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

  
  static std::shared_ptr<ndefile_loader_map> ndefile_loader_registry()
  {
    std::mutex &regmutex = ndefile_loader_registry_mutex();
    std::lock_guard<std::mutex> reglock(regmutex);

    if (!_ndefile_loader_registry) {
      _ndefile_loader_registry = new std::shared_ptr<ndefile_loader_map>(std::make_shared<ndefile_loader_map>());
    }
    return *_ndefile_loader_registry;
  }
  
  
  
  int register_ndefile_loader(std::string classname,unsigned depth,ndefile_loaderfunc loaderfunc) // depth of 1 = recording_base, depth of 2 = immediate subclass of recording_base, etc. 
  {
    ndefile_loader_registry(); // enure that the registry pointer exists

    std::mutex &regmutex = ndefile_loader_registry_mutex();
    std::lock_guard<std::mutex> reglock(regmutex);

    // copy map and update then publish the copy
    std::shared_ptr<ndefile_loader_map> new_map = std::make_shared<ndefile_loader_map>(**_ndefile_loader_registry);
    
    new_map->emplace(classname,std::make_pair(depth,loaderfunc));

    *_ndefile_loader_registry = new_map;
    return 0;

  }



  ndefile_readrecording_base::ndefile_readrecording_base(const std::set<std::string> &nde_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<nde_recording_map> filemap) :
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
    H5::Attribute nde_class_tags_attr = group.openAttribute("nde-class-tags");
    H5::DataType nde_ct_dtype = nde_class_tags_attr.getDataType();
    H5::DataSpace nde_ct_dspace = nde_class_tags_attr.getSpace();
    
    if (nde_ct_dspace.getSimpleExtentNdims() != 1) {
      throw snde_error("nde-class-tags attribute for hdf5 group %s should have exactly one iterable dimension",h5path.c_str());
    }

    // number of classes
    hsize_t nde_ct_num=0;

    nde_ct_dspace.getSimpleExtentDims(&nde_ct_num);

    if (nde_ct_num > 0) {
      if (nde_ct_dtype.getClass() != H5T_STRING) {
	throw snde_error("nde-class-tags attribute for hdf5 group %s should be an array of strings",h5path.c_str());
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
	    metadata_loader.AddMetaDatum(metadatum(attr_name,dblval));
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

  ndefile_readarray::ndefile_readarray(const std::set<std::string> &nde_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<nde_recording_map> filemap) :
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
      if (!arraynum) {
	postfix="";
      } else {
	postfix=std::to_string(arraynum);
      }
      
      // nde_array-name   attribute
      // nde_array-array     dataset
      // nde_array-dimlenF   dataset
      // nde_array-dimlenC   dataset
      
      H5::Attribute nde_array_name_attr = group.openAttribute(ssprintf("nde_array%s-name",postfix.c_str()));
      
      H5::DataType nde_an_dtype = nde_array_name_attr.getDataType();
      if (nde_an_dtype.getClass() != H5T_STRING) {
	throw snde_error("nde_array%s-name for HDF5 group %s should be a string",postfix.c_str(),h5path.c_str());
      }
      H5std_string nde_array_name;
      nde_array_name_attr.read(nde_an_dtype,nde_array_name);

      //snde_warning("reading nde_array: %s",nde_array_name.c_str());
      
      std::vector<snde_index> dimlen;
      bool fortran_order=false;
      
      std::string dimlenCname = ssprintf("nde_array%s-dimlenC",postfix.c_str());
      std::string dimlenFname = ssprintf("nde_array%s-dimlenF",postfix.c_str());
      if (group.nameExists(dimlenCname)) {
	H5::DataSet nde_array_dimlenC_dataset = group.openDataSet(dimlenCname);
	H5::DataSpace nde_adC_dspace = nde_array_dimlenC_dataset.getSpace();
	H5::DataType nde_adC_dtype = nde_array_dimlenC_dataset.getDataType();
	if (nde_adC_dtype.getClass() != H5T_INTEGER) {
	  throw snde_error("nde_array%s-dimlenC for HDF5 group %s should be of integral type",postfix.c_str(),h5path.c_str());
	}
	if (nde_adC_dspace.getSimpleExtentNdims() != 1) {
	  throw snde_error("nde_array%s-dimlenC should have exactly one iterable dimension",postfix.c_str());
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
	  throw snde_error("nde_array%s-dimlenF for HDF5 group %s should be of integral type",postfix.c_str(),h5path.c_str());
	}
	if (nde_adF_dspace.getSimpleExtentNdims() != 1) {
	  throw snde_error("nde_array%s-dimlenC should have exactly one iterable dimension",postfix.c_str());
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
      
      
      H5::Attribute nde_array_nativetype_attr = group.openAttribute(ssprintf("nde_array%s-nativetype",postfix.c_str()));
      
      H5::DataType nde_ant_dtype = nde_array_nativetype_attr.getDataType();
      if (nde_ant_dtype.getClass() != H5T_STRING) {
	throw snde_error("nde_array%s-nativetype for HDF5 group %s should be a string",postfix.c_str(),h5path.c_str());
      }
      H5std_string nde_array_nativetype;
      nde_array_nativetype_attr.read(nde_ant_dtype,nde_array_nativetype);
      
      
      auto nt_mappings_it = nde_file_nativetype_mappings.find(nde_array_nativetype);
      if (nt_mappings_it == nde_file_nativetype_mappings.end()) {
	throw snde_error("No known native type mapping for type %s for array %d of HDF5 group %s",nde_array_nativetype.c_str(),arraynum,h5path.c_str());
      }
      
      
      std::string nde_array_array_name = ssprintf("nde_array%s-array",postfix.c_str());
      H5::DataSet nde_array_array_dataset = group.openDataSet(nde_array_array_name);
      H5::DataSpace nde_aa_dspace = nde_array_array_dataset.getSpace();
      H5::DataType nde_aa_dtype = nde_array_array_dataset.getDataType();
      
      
      
      mndarray->define_array(arraynum,nt_mappings_it->second.second);
      
      mndarray->name_mapping.emplace(nde_array_name,arraynum);
      mndarray->name_reverse_mapping.emplace(arraynum,nde_array_name);
      
      mndarray->allocate_storage(arraynum,dimlen,fortran_order);
      
      
      
      if (nde_aa_dspace.getSimpleExtentNdims() != 1) {
	throw snde_error("nde_array%s-array should have exactly one iterable dimension for HDF5 group %s",postfix.c_str(),h5path.c_str());
      }
      
      hsize_t nelements=0;
      nde_aa_dspace.getSimpleExtentDims(&nelements,NULL);
      if (nelements != mndarray->layouts.at(arraynum).flattened_length()) {
	throw snde_error("nde_array%s-array number of elements (%llu) does not exactly match product of dimlen dimensions (%llu) for hdf5 group %s",postfix.c_str(),(unsigned long long)nelements,(unsigned long long)mndarray->layouts.at(arraynum).flattened_length(),h5path.c_str());	  
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
      
      nde_array_array_dataset.read(mndarray->void_shifted_arrayptr(arraynum),nt_mappings_it->second.first,nde_aa_dspace);
      
      
      
      
    }
    rec->mark_as_ready();
  }

  ndefile_readgroup::ndefile_readgroup(const std::set<std::string> &nde_classes,std::string h5path, H5::Group group, std::string recpath,std::shared_ptr<nde_recording_map> filemap) :
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
    std::shared_ptr<recording_group> retval = create_recording<recording_group>(recdb,loadchan,owner_id,nullptr);
    retval->metadata=metadata;

    return retval;
  }

  
  std::shared_ptr<ndefile_readrecording_base> ndefile_loadrecording(std::string h5path,H5::Group group,std::string recpath,std::shared_ptr<nde_recording_map> filemap)
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

  
  
  
  std::shared_ptr<nde_recording_map> ndefile_loadfile(std::shared_ptr<recdatabase> recdb,std::string ownername,void *owner_id,std::string filename,std::string recpath /* ="/" */ ) // add filter function parameter or specific recording to request to limit what is loaded? 
  {
    //std::shared_ptr<nde_file> ndefile = std::make_shared<nde_file>(filename);
    H5::H5File H5Obj(filename,H5F_ACC_RDONLY);
    
    
    H5::Group rootgroup = H5Obj.openGroup("/");
    std::shared_ptr<nde_recording_map> filemap = std::make_shared<nde_recording_map>();

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


  // register nde_array and nde_group loaders

  int registered_nde_array_loader = register_ndefile_loader_class<ndefile_readarray>("nde_array",2); // depth of 1 = recording_base, depth of 2 = immediate subclass of recording_base, etc. 			    

  int registered_nde_group_loader = register_ndefile_loader_class<ndefile_readgroup>("nde_group",2); // depth of 1 = recording_base, depth of 2 = immediate subclass of recording_base, etc. 			    


  
};
