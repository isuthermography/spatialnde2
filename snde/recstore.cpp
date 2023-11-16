#include "snde/recstore.hpp"
#include "snde/utils.hpp"
#include "snde/notify.hpp"
#include "snde/allocator.hpp"

#include "snde/graphics_recording.hpp"

//#ifndef _WIN32
//#include "shared_memory_allocator_posix.hpp"
//#endif // !_WIN32

namespace snde {


// rtn_typemap is indexed by typeid(type)
  SNDE_API const std::unordered_map<std::type_index,unsigned> rtn_typemap({ // look up typenum based on C++ typeid(type)
      {typeid(snde_float32), SNDE_RTN_FLOAT32},
      {typeid(snde_float64), SNDE_RTN_FLOAT64},
      // half-precision not generally
      // available
#ifdef SNDE_HAVE_FLOAT16
      {typeid(snde_float16), SNDE_RTN_FLOAT16},
#endif // SNDE_HAVE_FLOAT16
      {typeid(uint64_t),SNDE_RTN_UINT64},
      {typeid(int64_t),SNDE_RTN_INT64},
      {typeid(uint32_t),SNDE_RTN_UINT32},
      {typeid(int32_t),SNDE_RTN_INT32},
      {typeid(uint16_t),SNDE_RTN_UINT16},
      {typeid(int16_t),SNDE_RTN_INT16},
      {typeid(uint8_t),SNDE_RTN_UINT8},
      {typeid(int8_t),SNDE_RTN_INT8},
      {typeid(snde_rgba),SNDE_RTN_SNDE_RGBA},
      {typeid(snde_complexfloat32),SNDE_RTN_COMPLEXFLOAT32},
      {typeid(snde_complexfloat64),SNDE_RTN_COMPLEXFLOAT64},
      {typeid(snde_rgbd),SNDE_RTN_RGBD64},
      {typeid(std::string),SNDE_RTN_STRING},
      {typeid(std::shared_ptr<recording_base>),SNDE_RTN_RECORDING},      
      {typeid(std::shared_ptr<multi_ndarray_recording>),SNDE_RTN_RECORDING},      
      {typeid(std::shared_ptr<ndarray_recording_ref>),SNDE_RTN_RECORDING_REF},      
      {typeid(std::shared_ptr<constructible_metadata>),SNDE_RTN_CONSTRUCTIBLEMETADATA},
      {typeid(snde_coord3_int16),SNDE_RTN_SNDE_COORD3_INT16},
      {typeid(std::vector<snde_index>),SNDE_RTN_INDEXVEC},
      {typeid(std::shared_ptr<recording_group>),SNDE_RTN_RECORDING_GROUP},
      //{typeid(std::shared_ptr<pointcloud_recording>),SNDE_RTN_POINTCLOUD_RECORDING},
      {typeid(std::shared_ptr<meshed_part_recording>),SNDE_RTN_MESHED_PART_RECORDING},
      {typeid(std::shared_ptr<meshed_vertexarray_recording>),SNDE_RTN_MESHED_VERTEXARRAY_RECORDING},
      {typeid(std::shared_ptr<meshed_texvertex_recording>),SNDE_RTN_MESHED_TEXVERTEX_RECORDING},
      {typeid(std::shared_ptr<meshed_vertnormals_recording>),SNDE_RTN_MESHED_VERTNORMALS_RECORDING},
      {typeid(std::shared_ptr<meshed_trinormals_recording>),SNDE_RTN_MESHED_TRINORMALS_RECORDING},
      {typeid(std::shared_ptr<meshed_parameterization_recording>),SNDE_RTN_MESHED_PARAMETERIZATION_RECORDING},
      {typeid(std::shared_ptr<textured_part_recording>),SNDE_RTN_TEXTURED_PART_RECORDING},
      {typeid(std::shared_ptr<assembly_recording>),SNDE_RTN_ASSEMBLY_RECORDING},
      {typeid(snde_part),SNDE_RTN_SNDE_PART},
      {typeid(snde_topological),SNDE_RTN_SNDE_TOPOLOGICAL},
      {typeid(snde_triangle),SNDE_RTN_SNDE_TRIANGLE},
      {typeid(snde_coord3),SNDE_RTN_SNDE_COORD3},
      {typeid(snde_cmat23),SNDE_RTN_SNDE_CMAT23},
      {typeid(snde_edge),SNDE_RTN_SNDE_EDGE},
      {typeid(snde_coord2),SNDE_RTN_SNDE_COORD2},
      {typeid(snde_axis32),SNDE_RTN_SNDE_AXIS32},
      {typeid(snde_vertex_edgelist_index),SNDE_RTN_SNDE_VERTEX_EDGELIST_INDEX},
      {typeid(snde_box3),SNDE_RTN_SNDE_BOX3},
      {typeid(snde_boxcoord3),SNDE_RTN_SNDE_BOXCOORD3},
      {typeid(snde_parameterization),SNDE_RTN_SNDE_PARAMETERIZATION},
      {typeid(snde_parameterization_patch),SNDE_RTN_SNDE_PARAMETERIZATION_PATCH},
      {typeid(snde_trivertnormals),SNDE_RTN_SNDE_TRIVERTNORMALS},
      {typeid(snde_rendercoord),SNDE_RTN_SNDE_RENDERCOORD},
      {typeid(snde_box2),SNDE_RTN_SNDE_BOX2},
      {typeid(snde_boxcoord2),SNDE_RTN_SNDE_BOXCOORD2},
      {typeid(snde_imagedata),SNDE_RTN_SNDE_IMAGEDATA},
      //{typeid(snde_coord),SNDE_RTN_SNDE_COORD},// C++ type is indistinguishable from either float or double, depending on config
      {typeid(snde_coord4),SNDE_RTN_SNDE_COORD4},
      {typeid(snde_orientation2),SNDE_RTN_SNDE_ORIENTATION2},
      {typeid(snde_orientation3),SNDE_RTN_SNDE_ORIENTATION3},
      {typeid(snde_axis3),SNDE_RTN_SNDE_AXIS3},
      {typeid(snde_indexrange),SNDE_RTN_SNDE_INDEXRANGE},
      {typeid(snde_partinstance),SNDE_RTN_SNDE_PARTINSTANCE},
      {typeid(snde_image),SNDE_RTN_SNDE_IMAGE},
      {typeid(snde_kdnode),SNDE_RTN_SNDE_KDNODE},
      //{typeid(snde_bool),SNDE_RTN_SNDE_BOOL}, // C++ type is indistinguishable from UINT8
      //{typeid(snde_compleximagedata),SNDE_RTN_SNDE_COMPLEXIMAGEDATA}, // C++ type is indistinguishable from complexfloat32

  });
  
  // rtn_typesizemap is indexed by SNDE_RTN_xxx
  SNDE_API const std::unordered_map<unsigned,size_t> rtn_typesizemap({ // Look up element size bysed on typenum
      {SNDE_RTN_FLOAT32,sizeof(snde_float32)},
      {SNDE_RTN_FLOAT64,sizeof(snde_float64)},
      // half-precision not generally
      // available
#ifdef SNDE_HAVE_FLOAT16
      {SNDE_RTN_FLOAT16,sizeof(snde_float16)},
#else // SNDE_HAVE_FLOAT16
      {SNDE_RTN_FLOAT16,2},
#endif
      {SNDE_RTN_UINT64,sizeof(uint64_t)},
      {SNDE_RTN_INT64,sizeof(int64_t)},
      {SNDE_RTN_UINT32,sizeof(uint32_t)},
      {SNDE_RTN_INT32,sizeof(int32_t)},
      {SNDE_RTN_UINT16,sizeof(uint16_t)},
      {SNDE_RTN_INT16,sizeof(int16_t)},
      {SNDE_RTN_UINT8,sizeof(uint8_t)},
      {SNDE_RTN_INT8,sizeof(int8_t)},
      {SNDE_RTN_SNDE_RGBA,sizeof(snde_rgba)},
      {SNDE_RTN_COMPLEXFLOAT32,sizeof(snde_complexfloat32)},
      {SNDE_RTN_COMPLEXFLOAT64,sizeof(snde_complexfloat64)},
#ifdef SNDE_HAVE_FLOAT16
      {SNDE_RTN_COMPLEXFLOAT16,sizeof(snde_float16)},
#else // SNDE_HAVE_FLOAT16
      {SNDE_RTN_COMPLEXFLOAT16,4},
#endif
      {SNDE_RTN_RGBD64,sizeof(snde_rgbd)},
      {SNDE_RTN_SNDE_COORD3_INT16,sizeof(snde_coord3_int16)},
      // SNDE_RTN_INDEXVEC through SNDE_RTN_ASSEMBLY_RECORDING not applicable
      {SNDE_RTN_SNDE_PART,sizeof(snde_part)},
      {SNDE_RTN_SNDE_TOPOLOGICAL,sizeof(snde_topological)},
      {SNDE_RTN_SNDE_TRIANGLE,sizeof(snde_triangle)},
      {SNDE_RTN_SNDE_COORD3,sizeof(snde_coord3)},
      {SNDE_RTN_SNDE_CMAT23,sizeof(snde_cmat23)},
      {SNDE_RTN_SNDE_EDGE,sizeof(snde_edge)},
      {SNDE_RTN_SNDE_COORD2,sizeof(snde_coord2)},
      {SNDE_RTN_SNDE_AXIS32,sizeof(snde_axis32)},
      {SNDE_RTN_SNDE_VERTEX_EDGELIST_INDEX,sizeof(snde_vertex_edgelist_index)},
      {SNDE_RTN_SNDE_BOX3,sizeof(snde_box3)},
      {SNDE_RTN_SNDE_BOXCOORD3,sizeof(snde_boxcoord3)},
      {SNDE_RTN_SNDE_PARAMETERIZATION,sizeof(snde_parameterization)},
      {SNDE_RTN_SNDE_PARAMETERIZATION_PATCH,sizeof(snde_parameterization_patch)},
      {SNDE_RTN_SNDE_TRIVERTNORMALS,sizeof(snde_trivertnormals)},
      {SNDE_RTN_SNDE_RENDERCOORD,sizeof(snde_rendercoord)},
      {SNDE_RTN_SNDE_BOX2,sizeof(snde_box2)},
      {SNDE_RTN_SNDE_BOXCOORD2,sizeof(snde_boxcoord2)},
      {SNDE_RTN_SNDE_IMAGEDATA,sizeof(snde_imagedata)},
      {SNDE_RTN_SNDE_COORD,sizeof(snde_coord)},
      {SNDE_RTN_SNDE_COORD4,sizeof(snde_coord4)},
      {SNDE_RTN_SNDE_ORIENTATION2,sizeof(snde_orientation2)},
      {SNDE_RTN_SNDE_ORIENTATION3,sizeof(snde_orientation3)},
      {SNDE_RTN_SNDE_AXIS3,sizeof(snde_axis3)},
      {SNDE_RTN_SNDE_INDEXRANGE,sizeof(snde_indexrange)},
      {SNDE_RTN_SNDE_PARTINSTANCE,sizeof(snde_partinstance)},
      {SNDE_RTN_SNDE_IMAGE,sizeof(snde_image)},
      {SNDE_RTN_SNDE_KDNODE,sizeof(snde_kdnode)},
      {SNDE_RTN_SNDE_BOOL,sizeof(snde_bool)},
      {SNDE_RTN_SNDE_COMPLEXIMAGEDATA,sizeof(snde_compleximagedata)},
    });
  
  SNDE_API const std::unordered_map<unsigned,std::string> rtn_typenamemap({ // Look up type name based on typenum
      {SNDE_RTN_UNASSIGNED,"SNDE_RTN_UNASSIGNED"},
      {SNDE_RTN_FLOAT32,"SNDE_RTN_FLOAT32"},
      {SNDE_RTN_FLOAT64,"SNDE_RTN_FLOAT64"},
      // half-precision not generally
      // available
      {SNDE_RTN_FLOAT16,"SNDE_RTN_FLOAT16"},
      {SNDE_RTN_UINT64,"SNDE_RTN_UINT64"},
      {SNDE_RTN_INT64,"SNDE_RTN_INT64"},
      {SNDE_RTN_UINT32,"SNDE_RTN_UINT32"},
      {SNDE_RTN_INT32,"SNDE_RTN_INT32"},
      {SNDE_RTN_UINT16,"SNDE_RTN_UINT16"},
      {SNDE_RTN_INT16,"SNDE_RTN_INT16"},
      {SNDE_RTN_UINT8,"SNDE_RTN_UINT8"},
      {SNDE_RTN_INT8,"SNDE_RTN_INT8"},
      {SNDE_RTN_SNDE_RGBA,"SNDE_RTN_SNDE_RGBA"},
      {SNDE_RTN_COMPLEXFLOAT32,"SNDE_RTN_COMPLEXFLOAT32"},
      {SNDE_RTN_COMPLEXFLOAT64,"SNDE_RTN_COMPLEXFLOAT64"},
      {SNDE_RTN_COMPLEXFLOAT16,"SNDE_RTN_COMPLEXFLOAT16"},
      {SNDE_RTN_RGBD64,"SNDE_RTN_RGBD64"},
      {SNDE_RTN_STRING,"SNDE_RTN_STRING"},
      {SNDE_RTN_RECORDING,"SNDE_RTN_RECORDING"},      
      {SNDE_RTN_SNDE_COORD3_INT16,"SNDE_RTN_SNDE_COORD3_INT16"},   
      {SNDE_RTN_INDEXVEC,"SNDE_RTN_INDEXVEC"},

      {SNDE_RTN_RECORDING_GROUP,"SNDE_RTN_RECORDING_GROUP"},
      //{SNDE_RTN_POINTCLOUD_RECORDING,"SNDE_RTN_POINTCLOUD_RECORDING"},
      {SNDE_RTN_MESHED_PART_RECORDING,"SNDE_RTN_MESHED_PART_RECORDING"},
      {SNDE_RTN_MESHED_VERTEXARRAY_RECORDING, "SNDE_RTN_MESHED_VERTEXARRAY_RECORDING"},
      {SNDE_RTN_MESHED_TEXVERTEX_RECORDING,"SNDE_RTN_MESHED_TEXVERTEX_RECORDING"},
      {SNDE_RTN_MESHED_VERTNORMALS_RECORDING,"SNDE_RTN_MESHED_VERTNORMALS_RECORDING"},
      {SNDE_RTN_MESHED_TRINORMALS_RECORDING,"SNDE_RTN_MESHED_TRINORMALS_RECORDING"},
      {SNDE_RTN_MESHED_PARAMETERIZATION_RECORDING,"SNDE_RTN_MESHED_PARAMETERIZATION_RECORDING"},
      {SNDE_RTN_TEXTURED_PART_RECORDING,"SNDE_RTN_TEXTURED_PART_RECORDING"},
      {SNDE_RTN_ASSEMBLY_RECORDING,"SNDE_RTN_ASSEMBLY_RECORDING"},
      
      {SNDE_RTN_SNDE_PART,"SNDE_RTN_SNDE_PART"},
      {SNDE_RTN_SNDE_TOPOLOGICAL,"SNDE_RTN_SNDE_TOPOLOGICAL"},
      {SNDE_RTN_SNDE_TRIANGLE,"SNDE_RTN_SNDE_TRIANGLE"},
      {SNDE_RTN_SNDE_COORD3,"SNDE_RTN_SNDE_COORD3"},
      {SNDE_RTN_SNDE_CMAT23,"SNDE_RTN_SNDE_CMAT23"},
      {SNDE_RTN_SNDE_EDGE,"SNDE_RTN_SNDE_EDGE"},
      {SNDE_RTN_SNDE_COORD2,"SNDE_RTN_SNDE_COORD2"},
      {SNDE_RTN_SNDE_AXIS32,"SNDE_RTN_SNDE_AXIS32"},
      {SNDE_RTN_SNDE_VERTEX_EDGELIST_INDEX,"SNDE_RTN_SNDE_VERTEX_EDGELIST_INDEX"},
      {SNDE_RTN_SNDE_BOX3,"SNDE_RTN_SNDE_BOX3"},
      {SNDE_RTN_SNDE_BOXCOORD3,"SNDE_RTN_SNDE_BOXCOORD3"},
      {SNDE_RTN_SNDE_PARAMETERIZATION,"SNDE_RTN_SNDE_PARAMETERIZATION"},
      {SNDE_RTN_SNDE_PARAMETERIZATION_PATCH,"SNDE_RTN_SNDE_PARAMETERIZATION_PATCH"},
      {SNDE_RTN_SNDE_TRIVERTNORMALS,"SNDE_RTN_SNDE_TRIVERTNORMALS"},
      {SNDE_RTN_SNDE_RENDERCOORD,"SNDE_RTN_SNDE_RENDERCOORD"},

      {SNDE_RTN_SNDE_BOX2,"SNDE_RTN_SNDE_BOX2"},
      {SNDE_RTN_SNDE_BOXCOORD2,"SNDE_RTN_SNDE_BOXCOORD2"},
      {SNDE_RTN_SNDE_IMAGEDATA,"SNDE_RTN_SNDE_IMAGEDATA"},
      {SNDE_RTN_SNDE_COORD,"SNDE_RTN_SNDE_COORD"},
      {SNDE_RTN_SNDE_COORD4,"SNDE_RTN_SNDE_COORD4"},
      {SNDE_RTN_SNDE_ORIENTATION2,"SNDE_RTN_SNDE_ORIENTATION2"},
      {SNDE_RTN_SNDE_ORIENTATION3,"SNDE_RTN_SNDE_ORIENTATION3"},
      {SNDE_RTN_SNDE_AXIS3,"SNDE_RTN_SNDE_AXIS3"},
      {SNDE_RTN_SNDE_INDEXRANGE,"SNDE_RTN_SNDE_INDEXRANGE"},
      {SNDE_RTN_SNDE_PARTINSTANCE,"SNDE_RTN_SNDE_PARTINSTANCE"},
      {SNDE_RTN_SNDE_IMAGE,"SNDE_RTN_SNDE_IMAGE"},
      {SNDE_RTN_SNDE_KDNODE,"SNDE_RTN_SNDE_KDNODE"},
      {SNDE_RTN_SNDE_BOOL,"SNDE_RTN_SNDE_BOOL"},
      {SNDE_RTN_SNDE_COMPLEXIMAGEDATA,"SNDE_RTN_SNDE_COMPLEXIMAGEDATA"},
    });
  

  SNDE_API const std::unordered_map<unsigned,std::string> rtn_ocltypemap({ // Look up opencl type string based on typenum
      {SNDE_RTN_FLOAT32,"float"},
      {SNDE_RTN_FLOAT64,"double"},
      // half-precision not generally
      // available
      {SNDE_RTN_FLOAT16,"half"},
      {SNDE_RTN_UINT64,"unsigned long"},
      {SNDE_RTN_INT64,"long"},
      {SNDE_RTN_UINT32,"unsigned int"},
      {SNDE_RTN_INT32,"int"},
      {SNDE_RTN_UINT16,"unsigned short"},
      {SNDE_RTN_INT16,"short"},
      {SNDE_RTN_UINT8,"unsigned char"},
      {SNDE_RTN_INT8,"char"},
      {SNDE_RTN_SNDE_RGBA,"snde_rgba"},
      {SNDE_RTN_COMPLEXFLOAT32,"struct _snde_complexfloat32"},
      {SNDE_RTN_COMPLEXFLOAT64,"struct _snde_complexfloat64"},
      {SNDE_RTN_COMPLEXFLOAT16,"struct _snde_complexfloat16"},
      {SNDE_RTN_RGBD64,"snde_rgbd"},
      {SNDE_RTN_SNDE_COORD3_INT16,"snde_coord3_int16"},
      // SNDE_RTN_INDEXVEC not applicable

            
      
      //!!!**** CONTINUE HERE!!!***

      {SNDE_RTN_SNDE_COORD,"snde_coord"},
      {SNDE_RTN_SNDE_BOOL,"snde_bool"},
      {SNDE_RTN_SNDE_COMPLEXIMAGEDATA,"snde_compleximagedata"},
      
      
      
    });
  

  SNDE_API const std::unordered_map<unsigned,std::set<unsigned>> rtn_compatible_types({
      // These represent typedef (and conceptual) equivalencies
      // Note that we don't worry about transitive equivalencies as those generally wouldn't make sense
      // and therefore shouldn't come up. 
      {SNDE_RTN_FLOAT32, {
	  SNDE_RTN_SNDE_IMAGEDATA,
	  SNDE_RTN_SNDE_RENDERCOORD, 
#ifndef SNDE_DOUBLEPREC_COORDS
	  SNDE_RTN_SNDE_COORD,
#endif // SNDE_DOUBLEPREC_COORDS
	}
      },
      {SNDE_RTN_SNDE_IMAGEDATA, { SNDE_RTN_FLOAT32 }},      
      {SNDE_RTN_SNDE_RENDERCOORD, { SNDE_RTN_FLOAT32 }},
      
#ifdef SNDE_DOUBLEPREC_COORDS
      {SNDE_RTN_FLOAT64, { SNDE_RTN_SNDE_COORD } },
      {SNDE_RTN_SNDE_COORD, { SNDE_RTN_FLOAT64 } },
#else // SNDE_DOUBLEPREC_COORDS
      {SNDE_RTN_SNDE_COORD, { SNDE_RTN_FLOAT32 } },
#endif
      {SNDE_RTN_SNDE_BOOL, { SNDE_RTN_UINT8 } },
      {SNDE_RTN_UINT8, { SNDE_RTN_SNDE_BOOL } },

      {SNDE_RTN_COMPLEXFLOAT32, {SNDE_RTN_SNDE_COMPLEXIMAGEDATA} },
      {SNDE_RTN_SNDE_COMPLEXIMAGEDATA, {SNDE_RTN_COMPLEXFLOAT32} },
    });
  
  // see https://stackoverflow.com/questions/38644146/choose-template-based-on-run-time-string-in-c

  std::shared_ptr<recording_storage_manager> select_storage_manager_for_recording_during_transaction(std::shared_ptr<recdatabase> recdb,std::string chanpath)
  {
    // normally use select_storage_manager_for_recording (below) but if we create a recording during the transaction, the rss isn't created yet
    // so instead we walk the recdb's list of channels

    // Go up through hierarchy searching for a storage manager.
    // Drop down to the default if we hit the root
    std::string base=chanpath;
    std::string leaf;

    std::lock_guard<std::mutex> recdb_admin(recdb->admin);
    
    do {
      std::map<std::string,std::shared_ptr<channel>>::iterator channel_it=recdb->_channels.find(base);
      if (channel_it != recdb->_channels.end()) {
	// this folder/directory has an explicit channel entry
	
	std::shared_ptr<channelconfig> config = channel_it->second->config();
	if (config->storage_manager) {
	  return config->storage_manager;
	}
      }
      std::tie(base,leaf)=recdb_path_split(base);
    } while ((base.size() > 0) && !(base == "/" && leaf.size()==0));
    
    return recdb->default_storage_manager;
    
    
  }
  
  std::shared_ptr<recording_storage_manager> select_storage_manager_for_recording(std::shared_ptr<recdatabase> recdb,std::string chanpath,std::shared_ptr<recording_set_state> rss)
  {
    // Go up through hierarchy searching for a storage manager.
    // Drop down to the default if we hit the root
    std::string base=chanpath;
    std::string leaf="";

    do {
      std::map<std::string,channel_state>::iterator channel_map_it=rss->recstatus.channel_map->find(base);
      if (channel_map_it != rss->recstatus.channel_map->end()) {
	// this folder/directory has an explicit channel entry
	
	std::shared_ptr<channelconfig> config = channel_map_it->second.config;
	if (config->storage_manager) {
	  return config->storage_manager;
	}
      }
      if (base.size() > 0 && base.at(base.size()-1)=='/') {
	base = base.substr(0,base.size()-1); // if base ends with a '/' strip it so that recdb_path_split will remove the next segment.
      }
      std::tie(base,leaf)=recdb_path_split(base);
    } while ((base.size() > 0) && !(base == "/" && leaf.size()==0));
    
    return recdb->default_storage_manager;
  }

  
  recording_base::recording_base(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize) :
    // This constructor is to be called by everything except the math engine
    //  * Should be called by the owner of the given channel, as verified by owner_id
    //  * Should be called within a transaction (i.e. the recdb transaction_lock is held by the existance of the transaction)
    //  * Because within a transaction, recdb->current_transaction is valid
    // This constructor automatically adds the new recording to the current transaction
    info{nullptr},
    info_state(SNDE_RECF_DYNAMICMETADATAREADY), // downgrade this bit away prior to SNDE_RECS_STATICMETADATAREADY if we needs_dynamic_metadata
    metadata(nullptr),
    needs_dynamic_metadata(false),
    pending_dynamic_metadata(nullptr),
    storage_manager(storage_manager), // initially null but defined in end_transaction() for non-math recordings
    recdb_weak(recdb),
    defining_transact(defining_transact),
    originating_rss_unique_id(0), // must be assigned post-construction
    _originating_rss(_originating_rss)
  {
    //uint64_t new_revision = ++chan->latest_revision; // atomic variable so it is safe to pre-increment

    // validate info_structsize
    info_structsize = recording_default_info_structsize(info_structsize,sizeof(snde_recording_base));

    //if (!info_structsize) {
    //  info_structsize = sizeof(snde_recording_base);
    //}
    //assert(info_structsize >= sizeof(snde_recording_base));
    
    info = (snde_recording_base *)calloc(1,info_structsize);


    snde_recording_base info_prototype{
        .name = strdup(chanpath.c_str()),
        .revision = new_revision,
        .state = info_state,
        .metadata = nullptr,
        .metadata_valid = false,
        .deletable = false,
        .immutable = true,
    };
    *info = info_prototype;

    rec_classes.push_back(recording_class_info("snde::recording_base",typeid(recording_base),ptr_to_new_shared_impl<recording_base>));
    
  }

  

  recording_base::~recording_base()
  {
    free(info->name);
    info->name=nullptr;
    free(info);
    info=nullptr;
  }


  void recording_base::recording_needs_dynamic_metadata() // Call this before mark_metadata_done() to list this recording as needing dynamic metadata
  {
    std::lock_guard<std::mutex> rec_admin(admin);

    needs_dynamic_metadata=true;
    info->state &= ~SNDE_RECF_DYNAMICMETADATAREADY;
    info_state = info->state;
  }
  
  std::shared_ptr<multi_ndarray_recording> recording_base::cast_to_multi_ndarray()
  {
    return std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this());
  }
  
  const std::shared_ptr<std::map<std::string,std::pair<std::string,std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>>>> recording_base::graphics_subcomponents_orientation_lockinfo(std::shared_ptr<recording_set_state> rss)
  {
    return std::make_shared<std::map<std::string,std::pair<std::string,std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>>>>();
  }

  const std::shared_ptr<std::map<std::string,std::pair<std::string,snde_orientation3>>> recording_base::graphics_subcomponents_lockedorientations(std::shared_ptr<recording_set_state> rss)
  {
    return std::make_shared<std::map<std::string,std::pair<std::string,snde_orientation3>>>();
  }
  
  std::shared_ptr<std::set<std::string>> recording_base::graphics_componentpart_channels(std::shared_ptr<recording_set_state> rss,std::vector<std::string> processing_tags)
  {
    return std::make_shared<std::set<std::string>>();
  }
  
  std::shared_ptr<recording_set_state> recording_base::_get_originating_rss_rec_admin_prelocked() 
  // version of get_originating_rss() to use if you have (optionally the recdb and) the rec admin locks already locked.
  {
    std::shared_ptr<recording_set_state> originating_rss_strong;
    //std::shared_ptr<recdatabase> recdb_strong(recdb_weak);
    bool originating_rss_is_expired=false; // will assign to true if we get an invalid pointer and it turns out to be expired rather than null
    
    {
      originating_rss_strong = _originating_rss.lock();
      if (!originating_rss_strong) {
	originating_rss_is_expired = invalid_weak_ptr_is_expired(_originating_rss);
      }
      // get originating_rss from _originating_rss weak ptr in class and
      // if unavailable determine it is expired (see https://stackoverflow.com/questions/26913743/can-an-expired-weak-ptr-be-distinguished-from-an-uninitialized-one)
      //// check if merely unassigned vs. expired by testing with owner_before on a nullptr
      //std::weak_ptr<recording_set_state> null_weak_ptr;
      //if (null_weak_ptr.owner_before(_originating_rss) || _originating_rss.owner_before(null_weak_ptr)) {
      ///// this is distinct from the nullptr
      //originating_rss_is_expired=true; 
      //}
      
    }
    
    // OK; Now we have a strong ptr, which may be null, and
    // if so originating_rss_is_expired is true iff it was
    // once valid
    
    if (!originating_rss_strong && originating_rss_is_expired) {
      throw snde_error("Attempting to get expired originating recording set state (channel %s revision %llu", info->name,(unsigned long long)info->revision);
      
    }
    
    if (!originating_rss_strong) {
      // in this case originating_rss was never assigned. We need to extract it

      bool defining_transact_is_expired = false;
      std::shared_ptr<transaction> defining_transact_strong=defining_transact.lock();
      if (!defining_transact_strong) {
	defining_transact_is_expired = invalid_weak_ptr_is_expired(defining_transact);
	if (defining_transact_is_expired) {
	  throw snde_error("Attempting to get (expired transaction) originating recording set state (channel %s revision %llu", info->name,(unsigned long long)info->revision);
	  
	} else {
	  throw snde_error("Attempting to get originating recording set state (channel %s revision %llu for non-transaction and non-rss recording (?)", info->name,(unsigned long long)info->revision);
	  
	}

	
      }

      // defining_transact_strong is valid
      std::shared_ptr<globalrevision> originating_globalrev;
      bool originating_globalrev_expired;
      std::tie(originating_globalrev,originating_globalrev_expired) = defining_transact_strong->resulting_globalrevision();

      if (!originating_globalrev) {
	if (originating_globalrev_expired) {
	  throw snde_error("Attempting to get (expired globalrev) originating recording set state (channel %s revision %llu", info->name,(unsigned long long)info->revision);
	  
	} else {
	  // if originating_globalrev has never been set, then the transaction must still be in progress. 
	  throw snde_error("Attempting to get originating recording set state (channel %s revision %llu for recording while transaction still ongoing", info->name,(unsigned long long)info->revision);

	}
      }
      // originating_globalrev is valid
      originating_rss_strong = originating_globalrev;
    }
    // originating_rss_strong is valid; OK to return it
    
    return originating_rss_strong;
  }
  
  std::shared_ptr<recording_set_state> recording_base::_get_originating_rss_recdb_admin_prelocked()
  {
    std::shared_ptr<recording_set_state> originating_rss_strong = _originating_rss.lock();
    if (originating_rss_strong) return originating_rss_strong;
    
    std::lock_guard<std::mutex> recadmin(admin);
    return _get_originating_rss_rec_admin_prelocked();

  }


  
  std::shared_ptr<recording_set_state> recording_base::get_originating_rss()
  // Get the originating recording set state (often a globalrev)
  // You should only call this if you are sure that originating rss must still exist
  // (otherwise may generate a snde_error), such as before the creator has declared
  // the recording "ready". This will lock the rec admin locks,
  // so any locks currently held must precede that in the locking order
  {

    std::shared_ptr<recording_set_state> originating_rss_strong = _originating_rss.lock();
    if (originating_rss_strong) return originating_rss_strong;
    
    //std::shared_ptr<recdatabase> recdb_strong = recdb_weak.lock();
    //if (!recdb_strong) return nullptr; // shouldn't be possible in general
    //std::lock_guard<std::mutex> recdbadmin(recdb_strong->admin);
    std::lock_guard<std::mutex> recadmin(admin);
    return _get_originating_rss_rec_admin_prelocked();
  }

  bool recording_base::_transactionrec_transaction_still_in_progress_admin_prelocked() // with the recording admin locked,  return if this is a transaction recording where the transaction is still in progress and therefore we can't get the recording_set_state
  {
    // transaction recording if defining_transact is valid or expired
    std::shared_ptr<transaction> defining_transact_strong=defining_transact.lock();
    if (!defining_transact_strong) {
      if (invalid_weak_ptr_is_expired(defining_transact)) {
	throw snde_error("Attempting to check transaction status for an expired transaction (channel %s revision %llu", info->name,(unsigned long long)info->revision);
      } else {
	// not expired. Must not be a transaction recording
	// so we can just return false
	return false; 
      }
      
    }
    // defining_transact_strong is valid

    // is transaction still in progress?
    // when transaction ends, reulting_globalrevision() is assigned
    std::shared_ptr<globalrevision> resulting_globalrev;
    bool resulting_globalrev_expired;

    // resulting_globalrevision() locks the current_transaction's lock
    // thus ensuring synchronization with the call to
    // rss.update_recstatus__rss_admin_transaction_admin_locked()
    // in end_transaction()
    std::tie(resulting_globalrev,resulting_globalrev_expired) = defining_transact_strong->resulting_globalrevision();

    if (resulting_globalrev) {
      // got a globalrev; transaction must be done
      return false;
    }
    if (resulting_globalrev_expired) {
      throw snde_error("Attempting to check transaction status for an expired (globalrev) transaction (channel %s revision %llu", info->name,(unsigned long long)info->revision);
      
    } else {
      // resulting_globalrev is legitimately nullptr
      // i.e. transaction is still in progress
      return true; 
    }
  }


  void recording_base::_check_move_to_metadataonly__rss_and_recording_locked(std::shared_ptr<recording_set_state> rss,channel_state *chanstate)
  {

    if (info->state & SNDE_RECF_STATICMETADATAREADY && info->state & SNDE_RECF_DYNAMICMETADATAREADY) {
      // rss admin lock and recording admin lock should already be locked...  
      
      // if this is a metadataonly recording
      // Need to move this channel_state reference from rss->recstatus.instantiated_recordings into rss->recstatus->metadataonly_recordings
      std::map<std::string,std::shared_ptr<instantiated_math_function>>::iterator math_function_it;
      math_function_it = rss->mathstatus.math_functions->defined_math_functions.find(info->name);
      
      if (math_function_it != rss->mathstatus.math_functions->defined_math_functions.end()) {
	// channel is a math channel (only math channels can be mdonly)
	
	math_function_status &mathstatus = rss->mathstatus.function_status.at(math_function_it->second);
	
	if (mathstatus.mdonly) {
	  // yes, an mdonly channel... move it from instantiated recordings to metadataonly_recordings
	  //mdonly=true;
	  
	  std::unordered_map<std::shared_ptr<channelconfig>,channel_state *>::iterator instantiated_recording_it = rss->recstatus.instantiated_recordings.find(chanstate->config); 
	  assert(instantiated_recording_it != rss->recstatus.instantiated_recordings.end());
	  
	  rss->recstatus.instantiated_recordings.erase(instantiated_recording_it);
	  rss->recstatus.metadataonly_recordings.emplace(chanstate->config,chanstate);
	}
      }
      
    }
  }

  void recording_base::_merge_static_and_dynamic_metadata_admin_locked()
  {
    assert(info->state & SNDE_RECF_DYNAMICMETADATAREADY && info->state & SNDE_RECF_STATICMETADATAREADY);
    assert(!(info->state & SNDE_RECF_ALLMETADATAREADY));

    std::shared_ptr<constructible_metadata> all_metadata = MergeMetadata3(metadata,pending_metadata,pending_dynamic_metadata);

    metadata = all_metadata;
    info->metadata = metadata.get();
    
    info->state |= SNDE_RECF_ALLMETADATAREADY;
    info_state = info->state;
    
    pending_dynamic_metadata = nullptr;
    pending_metadata = nullptr;
  }

  void recording_base::_move_to_completed__rss_and_recording_locked(std::shared_ptr<recording_set_state> rss,channel_state *chanstate)
  {

    assert(info->state == SNDE_RECS_FULLYREADY);
    
    // move this state from recstatus.instantiated_recordings->recstatus.completed_recordings
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state *>::iterator instantiated_recording_it = rss->recstatus.instantiated_recordings.find(chanstate->config);
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state *>::iterator mdonly_recording_it = rss->recstatus.metadataonly_recordings.find(chanstate->config); 
	
    if (instantiated_recording_it != rss->recstatus.instantiated_recordings.end()) {
      rss->recstatus.instantiated_recordings.erase(instantiated_recording_it);
    } else if (mdonly_recording_it != rss->recstatus.metadataonly_recordings.end()) {
      rss->recstatus.metadataonly_recordings.erase(mdonly_recording_it);
    } else {
      throw snde_error("recording becomes fully ready with recording not found in instantiated or mdonly",(int)info_state);
    }
        
    rss->recstatus.completed_recordings.emplace(chanstate->config,chanstate);
    
  }


  void recording_base::mark_metadata_done()
  {
    // This should be called, not holding locks, (except perhaps dg_python context) after static metadata is finalized
    //bool mdonly=false; // set to true if we are an mdonly channel and therefore should send out mdonly notifications
    
    std::shared_ptr<recording_set_state> rss; // originating rss
    std::shared_ptr<recording_set_state> prerequisite_state; // prerequisite_state of originating rss
    
    std::shared_ptr<recdatabase> recdb = recdb_weak.lock();
    if (!recdb) return;

    bool all_metadata_done = false;
    bool becomes_fully_ready = false; 
    
    
    std::string channame;
    //std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_metadataonly;
    {
      std::lock_guard<std::mutex> adminlock(admin);
      if (_transactionrec_transaction_still_in_progress_admin_prelocked()) {
	// this transaction is still in progress; notifications will be handled by end_transaction so don't need to do notifications

	
	if (info_state & SNDE_RECF_STATICMETADATAREADY) {
	  return; // already ready (or beyond)
	}
	
	channame = info->name;
	info->state |= SNDE_RECF_STATICMETADATAREADY;
	info_state = info->state;

	if (info->state & SNDE_RECF_DYNAMICMETADATAREADY) {
	  _merge_static_and_dynamic_metadata_admin_locked();
	}
	return;
      }
    }
    channel_state *chanstate;
    {
      
      // with transaction complete, should be able to get an originating rss
      // (trying to make the state change atomically)
      rss = get_originating_rss();
      prerequisite_state = rss->prerequisite_state();
      std::lock_guard<std::mutex> rss_admin(rss->admin);
      std::lock_guard<std::mutex> adminlock(admin);

      
      if (info_state & SNDE_RECF_STATICMETADATAREADY) {
	return; // already ready (or beyond)
      }

      channame = info->name;
      
      chanstate = &rss->recstatus.channel_map->at(channame);
      
      info->state |= SNDE_RECF_STATICMETADATAREADY;
      info_state = info->state;

      if (info->state & SNDE_RECF_DYNAMICMETADATAREADY) {
	_merge_static_and_dynamic_metadata_admin_locked();
	all_metadata_done = true; 
	_check_move_to_metadataonly__rss_and_recording_locked(rss,chanstate);
	
	if (info->state == SNDE_RECS_FULLYREADY) {
	  becomes_fully_ready = true; 
	  _move_to_completed__rss_and_recording_locked(rss,chanstate);
	}
	
      }
      
    }

    //// perform notifications
    
    
    
    //for (auto && notify_ptr: *notify_about_this_channel_metadataonly) {
    //  notify_ptr->notify_metadataonly(channame);
    //}

    // Above replaced by chanstate.issue_nonmath_notifications

    //if (mdonly) {

    if (becomes_fully_ready || all_metadata_done) {
      chanstate->issue_math_notifications(recdb,rss);
    }
    chanstate->issue_nonmath_notifications(rss);
    //}
    

  }

    
    
  
  void recording_base::mark_dynamic_metadata_done()
  {
    // This should be called, not holding locks, (except perhaps dg_python context) after info->metadata is finalized
    //bool mdonly=false; // set to true if we are an mdonly channel and therefore should send out mdonly notifications
    
    std::shared_ptr<recording_set_state> rss; // originating rss
    std::shared_ptr<recording_set_state> prerequisite_state; // prerequisite_state of originating rss
    
    std::shared_ptr<recdatabase> recdb = recdb_weak.lock();
    if (!recdb) return;
    
    bool all_metadata_done = false;

    bool becomes_fully_ready = false; 
    
    std::string channame;
    {
      std::lock_guard<std::mutex> adminlock(admin);
      if (_transactionrec_transaction_still_in_progress_admin_prelocked()) {
	// this transaction is still in progress; notifications will be handled by end_transaction so don't need to do notifications
	
	
	if (info_state & SNDE_RECF_DYNAMICMETADATAREADY) {
	  return; // already ready (or beyond)
	}
	
	channame = info->name;
	info->state |= SNDE_RECF_DYNAMICMETADATAREADY;
	info_state = info->state;
	
	if (info->state & SNDE_RECF_STATICMETADATAREADY) {
	  _merge_static_and_dynamic_metadata_admin_locked();
	}
	return;
      }
    }
    channel_state *chanstate;
    {
      
      // with transaction complete, should be able to get an originating rss
      // (trying to make the state change atomically)
      rss = get_originating_rss();
      prerequisite_state = rss->prerequisite_state();
      std::lock_guard<std::mutex> rss_admin(rss->admin);
      std::lock_guard<std::mutex> adminlock(admin);
      
      if (info_state & SNDE_RECF_DYNAMICMETADATAREADY) {
	return; // already ready (or beyond)
      }

      channame = info->name;
      chanstate = &rss->recstatus.channel_map->at(channame);

      
      info->state |= SNDE_RECF_DYNAMICMETADATAREADY;
      info_state = info->state;

      if (info->state & SNDE_RECF_STATICMETADATAREADY) {
	_merge_static_and_dynamic_metadata_admin_locked();
	all_metadata_done=true; 
	_check_move_to_metadataonly__rss_and_recording_locked(rss,chanstate);
	if (info->state == SNDE_RECS_FULLYREADY) {
	  becomes_fully_ready = true;
	  _move_to_completed__rss_and_recording_locked(rss,chanstate);
	}
      }
      


    }

    
    
    //// perform notifications

    
    
    //for (auto && notify_ptr: *notify_about_this_channel_metadataonly) {
    //  notify_ptr->notify_metadataonly(channame);
    //}

    // Above replaced by chanstate.issue_nonmath_notifications

    //if (mdonly) {

    if (becomes_fully_ready || all_metadata_done) {
      chanstate->issue_math_notifications(recdb,rss);
    }
    chanstate->issue_nonmath_notifications(rss);
    //}
    
  }
  
  void recording_base::mark_data_ready()  
  {
    std::string channame;
    std::shared_ptr<recording_set_state> rss; // originating rss
    std::shared_ptr<recording_set_state> prerequisite_state; // prerequisite_state of originating rss
    //std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_ready;
    //std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_metadataonly;

    // This should be called, not holding locks, (except perhaps dg_python context) after info->metadata is finalized

    // NOTE: probably need to add call to storage manager at start here
    // to synchronously flush any data out e.g. to a GPU or similar. 
    
    std::shared_ptr<recdatabase> recdb = recdb_weak.lock();
    if (!recdb) return;

    bool becomes_fully_ready = false; 
    
    {
      std::unique_lock<std::mutex> rec_admin(admin);
      if (_transactionrec_transaction_still_in_progress_admin_prelocked()) {
	
	// this transaction is still in progress; notifications will be handled by end_transaction()
	
	
	if (info_state & SNDE_RECF_DATAREADY) {
	  return; // already ready (or beyond)
	}
	channame = info->name;
	
	
	info->state |= SNDE_RECF_DATAREADY;
	info_state = info->state;

	if (info->immutable) {
	  rec_admin.unlock();
	  _mark_storage_as_finalized_internal();
	}
	
	// These next few lines replaced by chanstate.issue_nonmath_notifications, below
	//channel_state &chanstate = rss->recstatus.channel_map->at(channame);
	//notify_about_this_channel_ready = chanstate.notify_about_this_channel_ready();
	//chanstate.end_atomic_notify_about_this_channel_ready_update(nullptr); // all notifications are now our responsibility
	
	return;  // no notifies because transaction still in progress
      }
    }
    
    channel_state *chanstate;
    {
      
      // with transaction complete, should be able to get an originating rss
      // (trying to make the state change atomically)
      rss = get_originating_rss();
      prerequisite_state = rss->prerequisite_state();
      std::unique_lock<std::mutex> rss_admin(rss->admin);
      std::unique_lock<std::mutex> adminlock(admin);


      if (info_state & SNDE_RECF_DATAREADY) {
	return; // already ready (or beyond)
      }
      channame = info->name;
      

      info->state |= SNDE_RECF_DATAREADY;
      info_state = info->state;


      chanstate = &rss->recstatus.channel_map->at(channame);

      if (info_state==SNDE_RECS_FULLYREADY) {

	// we are now fully ready
	
	_move_to_completed__rss_and_recording_locked(rss,chanstate);

	becomes_fully_ready = true; 
      }

      if (info->immutable) {
	adminlock.unlock();
	rss_admin.unlock();
	
	_mark_storage_as_finalized_internal();
      }
      
      
    // perform notifications (replaced by issue_nonmath_notifications())
    //for (auto && notify_ptr: *notify_about_this_channel_metadataonly) {
    //  notify_ptr->notify_metadataonly(channame);
    //}
    //for (auto && notify_ptr: *notify_about_this_channel_ready) {
    //  notify_ptr->notify_ready(channame);
    //}

    }


    //assert(chanstate.notify_about_this_channel_metadataonly());

    if (becomes_fully_ready) {
      chanstate->issue_math_notifications(recdb,rss);
    }
    
    chanstate->issue_nonmath_notifications(rss);
  }

  void recording_base::mark_data_and_metadata_ready()  
  {
    mark_metadata_done();
    mark_data_ready();
  }
  
  void recording_base::_mark_storage_as_finalized_internal()
  {
    // subclasses with storage should override this
  }


  std::shared_ptr<recording_storage_manager> recording_base::assign_storage_manager(std::shared_ptr<recording_storage_manager> storman)
  {
    std::lock_guard<std::mutex> rec_admin(admin); // lock recording
    storage_manager = storman;
    return storage_manager;
  }

  std::shared_ptr<recording_storage_manager> recording_base::assign_storage_manager()
  // may return nullptr if recdb not available
  {

    std::shared_ptr<recording_storage_manager> storman;

    {
      std::lock_guard<std::mutex> rec_admin(admin); // lock recording
      storman = storage_manager;
    }
    
    std::shared_ptr<recdatabase> recdb_strong=recdb_weak.lock();
    if (!recdb_strong) return nullptr; // if recdb is vanishing we are dead too

    
    if (!storman) {
      
      
      std::shared_ptr<recording_set_state> originating_rss_strong;

      {
	std::unique_lock<std::mutex> recdb_admin(recdb_strong->admin);
	std::unique_lock<std::mutex> rec_admin(admin); // lock recording

	originating_rss_strong = _originating_rss.lock();
	if (!originating_rss_strong && invalid_weak_ptr_is_expired(_originating_rss)) {
	  throw snde_error("Attempting to get expired originating recording set state (channel %s revision %llu", info->name,(unsigned long long)info->revision);
	
	}
	if (!originating_rss_strong) {
	  // in this case originating_rss was never assigned. We need to extract it
	  
	  bool defining_transact_is_expired = false;
	  std::shared_ptr<transaction> defining_transact_strong=defining_transact.lock();
	  if (!defining_transact_strong) {
	    defining_transact_is_expired = invalid_weak_ptr_is_expired(defining_transact);
	    if (defining_transact_is_expired) {
	      throw snde_error("Attempting to get (expired transaction) originating recording set state (channel %s revision %llu", info->name,(unsigned long long)info->revision);
	      
	    } else {
	      throw snde_error("Attempting to allocate storage for recording set state (channel %s revision %llu) for non-transaction and non-rss recording with no storage_manager set", info->name,(unsigned long long)info->revision);
	      
	    }
	    
	    
	  }
	  
	  // defining_transact_strong is valid
	  std::shared_ptr<globalrevision> originating_globalrev;
	  bool originating_globalrev_expired;
	  std::tie(originating_globalrev,originating_globalrev_expired) = defining_transact_strong->resulting_globalrevision();
	  
	  
	  if (!originating_globalrev) {
	    if (originating_globalrev_expired) {
	      throw snde_error("Attempting to get (expired globalrev) originating recording set state (channel %s revision %llu", info->name,(unsigned long long)info->revision);
	      
	    } else {
	      // if originating_globalrev has never been set, then the transaction must still be in progress. 
	      // Check the channel config
	      
	      // drop the locks, as select_storage_manager will need to reacquire them and they will not be needed from here on 
	      rec_admin.unlock();
	      recdb_admin.unlock();
	      storman=select_storage_manager_for_recording_during_transaction(recdb_strong,info->name);
	    
	    }
	  }
	  
	  originating_rss_strong = originating_globalrev;
	}
      } // locks dropped at this point

      
      if (!storman) {
	storman = select_storage_manager_for_recording(recdb_strong,info->name,originating_rss_strong);
      }
      
      
      {
	std::lock_guard<std::mutex> rec_admin(admin); // lock recording
	storage_manager=storman;
      }

      
    }
    
    
    return storman;
  }


  null_recording::null_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize/*=0*/) :
    recording_base(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,recording_default_info_structsize(info_structsize,sizeof(snde_recording_base)))
  {
    rec_classes.push_back(recording_class_info("snde::null_recording",typeid(null_recording),ptr_to_new_shared_impl<null_recording>));


  }
  

  recording_group::recording_group(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize/*,std::shared_ptr<std::string> path_to_primary*/) :
    recording_base(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,recording_default_info_structsize(info_structsize,sizeof(snde_recording_base))) //,
    //path_to_primary(path_to_primary)
  {
    rec_classes.push_back(recording_class_info("snde::recording_group",typeid(recording_group),ptr_to_new_shared_impl<recording_group>));


    //if (!chanpath.size() || chanpath.at(chanpath.size()-1) != '/') {
    //  throw snde_error("Recording group %s does not end with a trailing slash",chanpath.c_str());
    //}
  }

  
  // after construction, must call .define_array() exactly once for each ndarray
  multi_ndarray_recording::multi_ndarray_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,size_t num_ndarrays) :
    // after construction, must call .define_array() exactly once for each ndarray
    recording_base(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,recording_default_info_structsize(info_structsize,sizeof(snde_multi_ndarray_recording))),
    layouts(std::vector<arraylayout>(num_ndarrays)),
    storage(std::vector<std::shared_ptr<recording_storage>>(num_ndarrays))
  {
    rec_classes.push_back(recording_class_info("snde::multi_ndarray_recording",typeid(multi_ndarray_recording),ptr_to_new_shared_impl<multi_ndarray_recording>));


    mndinfo()->dims_valid=false;
    mndinfo()->data_valid=false;
    mndinfo()->num_arrays = num_ndarrays;
    if (num_ndarrays) {
      mndinfo()->arrays = (snde_ndarray_info *)calloc(sizeof(snde_ndarray_info)*num_ndarrays,1);
    }
    //ndinfo()->ndim=0;
    //ndinfo()->base_index=0;
    //ndinfo()->dimlen=nullptr;
    //ndinfo()->strides=nullptr;
    //ndinfo()->owns_dimlen_strides=false;
    //ndinfo()->typenum=typenum;
    //ndinfo()->elementsize=rtn_typesizemap.at(typenum);
    //ndinfo()->requires_locking_read=false;
    //ndinfo()->requires_locking_write=false;
    //ndinfo()->basearray = nullptr;
    //ndinfo()->basearray_holder = nullptr;
    
  }

  
  multi_ndarray_recording::~multi_ndarray_recording()
  {
    // c pointers get freed automatically because they point into the c++ structs.
    // except for info->arrays
    if (mndinfo()->arrays) {
      free(mndinfo()->arrays);
    }
    mndinfo()->arrays=nullptr; 
  }

  void multi_ndarray_recording::set_num_ndarrays(size_t num_ndarrays)
  // NOTE: This can only be called on a recording that was constructed with num_ndarrays
  // set to zero, and must be called before any arrays are defined, etc. etc.,
  // i.e. first thing after construction

  // Will still need to call define_array() on each array. 
  {
    assert(!mndinfo()->arrays);
    
    if (num_ndarrays > 0) {
      mndinfo()->arrays = (snde_ndarray_info *)calloc(sizeof(snde_ndarray_info)*num_ndarrays,1);
    }

    layouts = std::vector<arraylayout>(num_ndarrays);
    storage = std::vector<std::shared_ptr<recording_storage>>(num_ndarrays);

    mndinfo()->num_arrays = num_ndarrays;
    
  }

  
  void multi_ndarray_recording::mark_data_ready()
  {
    // Notify our storage that we are marked as ready, which may cause cache invalidation, etc.
    for (auto && recstorage: storage) {
      if (recstorage) {
	// recstorage might be nullptr if this recording
	// has legitimately no data for a particular array
	recstorage->ready_notification();
      }
    }
    
    // call superclass, which does the rest of the management
    recording_base::mark_data_ready();
    // (including a _mark_storage_as_finalized_internal() call at the end if we are supposed to be immutable)
  }
  
  void multi_ndarray_recording::_mark_storage_as_finalized_internal()
  {
    for (auto && recstorage: storage) {
      if (recstorage) {
	// recstorage might be nullptr if this recording
	// has legitimately no data for a particular array
	recstorage->mark_as_finalized();
      }
    }
  
  }

  void multi_ndarray_recording::define_array(size_t index,unsigned typenum)
  // should be called exactly once for each index < mndinfo()->num_arrays
  {
    std::lock_guard<std::mutex> rec_admin(admin);

    // ***!!!! Must keep content parallel to the explicit name define_array() (below)
    
    ndinfo(index)->ndim=0;
    ndinfo(index)->base_index=0;
    ndinfo(index)->dimlen=nullptr;
    ndinfo(index)->strides=nullptr;
    ndinfo(index)->owns_dimlen_strides=false;
    ndinfo(index)->typenum=typenum;

    if (typenum != SNDE_RTN_UNASSIGNED) {
      ndinfo(index)->elementsize=rtn_typesizemap.at(typenum);
    } else {
      ndinfo(index)->elementsize=0;
    }
    ndinfo(index)->requires_locking_read=false;
    ndinfo(index)->requires_locking_write=false;
    ndinfo(index)->basearray = nullptr;
    ndinfo(index)->shiftedarray = nullptr;
    //ndinfo()->basearray_holder = nullptr;

    std::string name=std::string("array")+std::to_string(index);
    
    name_mapping.emplace(name,index);
    name_reverse_mapping.emplace(index,name);

    
  }


  void multi_ndarray_recording::define_array(size_t index,unsigned typenum,std::string name)   // should be called exactly once for each index < mndinfo()->num_arrays
  {
    std::lock_guard<std::mutex> rec_admin(admin); 

    // ***!!!! Must keep content parallel to the implicit name define_array() (above)
    
    ndinfo(index)->ndim=0;
    ndinfo(index)->base_index=0;
    ndinfo(index)->dimlen=nullptr;
    ndinfo(index)->strides=nullptr;
    ndinfo(index)->owns_dimlen_strides=false;
    ndinfo(index)->typenum=typenum;

    if (typenum != SNDE_RTN_UNASSIGNED) {
      ndinfo(index)->elementsize=rtn_typesizemap.at(typenum);
    } else {
      ndinfo(index)->elementsize=0;
    }
    ndinfo(index)->requires_locking_read=false;
    ndinfo(index)->requires_locking_write=false;
    ndinfo(index)->basearray = nullptr;
    ndinfo(index)->shiftedarray = nullptr;
    //ndinfo()->basearray_holder = nullptr;


    if (isdigit(name.at(0))) {
      throw snde_error("Array names snould not begin with a digit");
    }

    if (name_mapping.find(name) != name_mapping.end()) {
      throw snde_error("Recording array %s already defined",name.c_str());
    }
    name_mapping.emplace(name,index);
    name_reverse_mapping.emplace(index,name);
    
  }

  std::shared_ptr<std::vector<std::string>> multi_ndarray_recording::list_arrays()
  {
    std::shared_ptr<std::vector<std::string>> retval = std::make_shared<std::vector<std::string>>();

    std::lock_guard<std::mutex> recadmin(admin);
    size_t num_refs = layouts.size();
    if (!num_refs) {
      return retval;
    }
    
    if (num_refs==1 && name_reverse_mapping.size()==0) {
      if (name_mapping.size() == 0) {
	snde_warning("ndarray %s does not provide a name mapping",info->name);
	return retval;
      }
      retval->push_back(name_mapping.begin()->first);
    } else {
      size_t refnum;
      if (name_mapping.size() == 0) {
	snde_warning("ndarray %s does not provide a name mapping",info->name);
	return retval;
      }
      
      for (refnum=0;refnum < num_refs;refnum++) {
	retval->push_back(name_reverse_mapping.at(refnum));
	
      }
    }
    

    return retval;
  }

  // Throw an snde_error exception if the recording specifies a scale
  // or offset. This should be used whenever interpreting array data
  // in a context that does not include multiplying by the scale and
  // then adding the offset (ande_array-ampl_scale and
  // ande_array-ampl_offset metadata entries, respectively)
  // The identifier parameter should identify the e.g. math function
  // or similar so that the source of the problem is readily
  // identifiable
  void multi_ndarray_recording::assert_no_scale_or_offset(std::string identifier)
  {
    snde_float64 scale=metadata->GetMetaDatumDbl("ande_array-ampl_scale", 1.0);
    snde_float64 offset=metadata->GetMetaDatumDbl("ande_array-ampl_offset", 0.0);
    if (scale != 1.0 || offset != 0.0) {

      throw snde_error("Attempting to use recording %s with a scale or offset in a context (%s) that does not support it.", info->name,identifier.c_str());
    }
  }

  // return (scale,offset) tuple from metadata
  std::tuple<snde_float64,snde_float64> multi_ndarray_recording::get_ampl_scale_offset()
  {
    snde_float64 scale=metadata->GetMetaDatumDbl("ande_array-ampl_scale", 1.0);
    snde_float64 offset=metadata->GetMetaDatumDbl("ande_array-ampl_offset", 0.0);
    return std::make_tuple(scale,offset);
  }
    
  std::shared_ptr<ndarray_recording_ref> multi_ndarray_recording::reference_ndarray(size_t index)
  {

    if (index >= layouts.size()) {
      throw snde_error("multi_ndarray_recording::reference_ndarray(): Array index %llu does not exist (size=%llu)",(unsigned long long)index,(unsigned long long)layouts.size());
    }
    
    // ***!!! Should look up maker method in a runtime-addable database ***!!!
    std::shared_ptr<ndarray_recording_ref> ref;
    unsigned typenum=mndinfo()->arrays[index].typenum;

    switch (typenum) {
    case SNDE_RTN_UNASSIGNED:
      ref=std::make_shared<ndarray_recording_ref>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),index,SNDE_RTN_UNASSIGNED);
      break;
	
    case SNDE_RTN_FLOAT32:
      ref = std::make_shared<ndtyped_recording_ref<snde_float32>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;
      
    case SNDE_RTN_FLOAT64: 
      ref = std::make_shared<ndtyped_recording_ref<snde_float64>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

#ifdef SNDE_HAVE_FLOAT16
    case SNDE_RTN_FLOAT16: 
      ref = std::make_shared<ndtyped_recording_ref<snde_float16>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;
#endif

    case SNDE_RTN_UINT64:
      ref = std::make_shared<ndtyped_recording_ref<uint64_t>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;
      
    case SNDE_RTN_INT64:
      ref = std::make_shared<ndtyped_recording_ref<int64_t>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_UINT32:
      ref = std::make_shared<ndtyped_recording_ref<uint32_t>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;
      
    case SNDE_RTN_INT32:
      ref = std::make_shared<ndtyped_recording_ref<int32_t>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_UINT16:
      ref = std::make_shared<ndtyped_recording_ref<uint16_t>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;
      
    case SNDE_RTN_INT16:
      ref = std::make_shared<ndtyped_recording_ref<int16_t>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;
      
    case SNDE_RTN_UINT8:
      ref = std::make_shared<ndtyped_recording_ref<uint8_t>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;
      
    case SNDE_RTN_INT8:
      ref = std::make_shared<ndtyped_recording_ref<int8_t>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_RGBA:
      ref = std::make_shared<ndtyped_recording_ref<snde_rgba>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;
      
    case SNDE_RTN_COMPLEXFLOAT32:
      ref = std::make_shared<ndtyped_recording_ref<snde_complexfloat32>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_COMPLEXFLOAT64:
      ref = std::make_shared<ndtyped_recording_ref<snde_complexfloat64>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;
      
#ifdef SNDE_HAVE_FLOAT16
    case SNDE_RTN_COMPLEXFLOAT16:
      ref = std::make_shared<ndtyped_recording_ref<snde_complexfloat16>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;
#endif
      
    case SNDE_RTN_RGBD64:
      ref = std::make_shared<ndtyped_recording_ref<snde_rgbd>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_COORD3_INT16:
      ref = std::make_shared<ndtyped_recording_ref<snde_coord3_int16>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

      // SNDE_RTN_INDEXVEC through SNDE_RTN_ASSEMBLY_RECORDING not applicable
      
      
    case SNDE_RTN_SNDE_PART:
      ref = std::make_shared<ndtyped_recording_ref<snde_part>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_TOPOLOGICAL:
      ref = std::make_shared<ndtyped_recording_ref<snde_topological>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_TRIANGLE:
      ref = std::make_shared<ndtyped_recording_ref<snde_triangle>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_COORD3:
      ref = std::make_shared<ndtyped_recording_ref<snde_coord3>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_CMAT23:
      ref = std::make_shared<ndtyped_recording_ref<snde_cmat23>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_EDGE:
      ref = std::make_shared<ndtyped_recording_ref<snde_edge>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_COORD2:
      ref = std::make_shared<ndtyped_recording_ref<snde_coord2>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_AXIS32:
      ref = std::make_shared<ndtyped_recording_ref<snde_axis32>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_VERTEX_EDGELIST_INDEX:
      ref = std::make_shared<ndtyped_recording_ref<snde_vertex_edgelist_index>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_BOX3:
      ref = std::make_shared<ndtyped_recording_ref<snde_box3>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_BOXCOORD3:
      ref = std::make_shared<ndtyped_recording_ref<snde_boxcoord3>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_PARAMETERIZATION:
      ref = std::make_shared<ndtyped_recording_ref<snde_parameterization>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_PARAMETERIZATION_PATCH:
      ref = std::make_shared<ndtyped_recording_ref<snde_parameterization_patch>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_TRIVERTNORMALS:
      ref = std::make_shared<ndtyped_recording_ref<snde_trivertnormals>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_RENDERCOORD:
      ref = std::make_shared<ndtyped_recording_ref<snde_rendercoord>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_BOX2:
      ref = std::make_shared<ndtyped_recording_ref<snde_box2>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_BOXCOORD2:
      ref = std::make_shared<ndtyped_recording_ref<snde_boxcoord2>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_IMAGEDATA:
      ref = std::make_shared<ndtyped_recording_ref<snde_imagedata>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_COORD:
      ref = std::make_shared<ndtyped_recording_ref<snde_coord>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_COORD4:
      ref = std::make_shared<ndtyped_recording_ref<snde_coord4>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_ORIENTATION2:
      ref = std::make_shared<ndtyped_recording_ref<snde_orientation2>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_ORIENTATION3:
      ref = std::make_shared<ndtyped_recording_ref<snde_orientation3>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_AXIS3:
      ref = std::make_shared<ndtyped_recording_ref<snde_axis3>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_INDEXRANGE:
      ref = std::make_shared<ndtyped_recording_ref<snde_indexrange>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_PARTINSTANCE:
      ref = std::make_shared<ndtyped_recording_ref<snde_partinstance>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_IMAGE:
      ref = std::make_shared<ndtyped_recording_ref<snde_image>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_KDNODE:
      ref = std::make_shared<ndtyped_recording_ref<snde_kdnode>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;


    case SNDE_RTN_SNDE_BOOL:
      ref = std::make_shared<ndtyped_recording_ref<snde_bool>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;

    case SNDE_RTN_SNDE_COMPLEXIMAGEDATA:
      ref = std::make_shared<ndtyped_recording_ref<snde_compleximagedata>>(std::dynamic_pointer_cast<multi_ndarray_recording>(shared_from_this()),typenum,index);
      break;
      
    default:
      throw snde_error("multi_ndarray_recording::reference_ndarray(): Unknown type number %u",typenum);
    }
    return ref;
  }

  std::shared_ptr<ndarray_recording_ref> multi_ndarray_recording::reference_ndarray(const std::string &array_name)
  {
    auto name_mapping_it = name_mapping.find(array_name);
    
    if (name_mapping_it == name_mapping.end()) {
      throw snde_error("multi_ndarray_recording::reference_ndarray(): Array name %s does not exist",array_name.c_str());
    }
    
    return reference_ndarray(name_mapping_it->second);
  }
  

  void multi_ndarray_recording::assign_storage(std::shared_ptr<recording_storage> stor,size_t array_index,const std::vector<snde_index> &dimlen, bool fortran_order/*=false*/)
  {
    size_t dimnum;
    snde_index stride=1;

    
    storage.at(array_index) = stor;
    

    std::vector<snde_index> strides;

    strides.reserve(dimlen.size());
    
    if (fortran_order) {
      for (dimnum=0;dimnum < dimlen.size();dimnum++) {
	strides.push_back(stride);
	stride *= dimlen.at(dimnum);
      }
    } else {
      // c order
      for (dimnum=0;dimnum < dimlen.size();dimnum++) {
	strides.insert(strides.begin(),stride);
	stride *= dimlen.at(dimlen.size()-dimnum-1);
      }
      
    }

    assert(stride==stor->nelem); // final stride (product of dimlen) had better equal number of elements!
    
    layouts.at(array_index)=arraylayout(dimlen,strides);
    ndinfo(array_index)->basearray = storage.at(array_index)->lockableaddr();
    ndinfo(array_index)->base_index=storage.at(array_index)->base_index;
    ndinfo(array_index)->ndim=layouts.at(array_index).dimlen.size();
    ndinfo(array_index)->dimlen=layouts.at(array_index).dimlen.data();
    ndinfo(array_index)->strides=layouts.at(array_index).strides.data();
    ndinfo(array_index)->requires_locking_read=storage.at(array_index)->requires_locking_read;
    ndinfo(array_index)->requires_locking_write=storage.at(array_index)->requires_locking_write;
    ndinfo(array_index)->shiftedarray = storage.at(array_index)->dataaddr_or_null();

  }


  
  void multi_ndarray_recording::assign_storage(std::shared_ptr<recording_storage> stor,std::string array_name,const std::vector<snde_index> &dimlen, bool fortran_order/*=false*/)
  {
    size_t array_index;

    
    array_index = name_mapping.at(array_name);
    assign_storage(stor,array_index,dimlen,fortran_order);
  }

  void multi_ndarray_recording::assign_storage_strides(std::shared_ptr<recording_storage> stor,size_t array_index,const std::vector<snde_index> &dimlen, const std::vector<snde_index> &strides)
  {
    size_t dimnum;

    
    storage.at(array_index) = stor;
    

    
    layouts.at(array_index)=arraylayout(dimlen,strides);
    ndinfo(array_index)->basearray = storage.at(array_index)->lockableaddr();
    ndinfo(array_index)->base_index=storage.at(array_index)->base_index;
    ndinfo(array_index)->ndim=layouts.at(array_index).dimlen.size();
    ndinfo(array_index)->dimlen=layouts.at(array_index).dimlen.data();
    ndinfo(array_index)->strides=layouts.at(array_index).strides.data();
    ndinfo(array_index)->requires_locking_read=storage.at(array_index)->requires_locking_read;
    ndinfo(array_index)->requires_locking_write=storage.at(array_index)->requires_locking_write;
    ndinfo(array_index)->shiftedarray = storage.at(array_index)->dataaddr_or_null();

  }



  void multi_ndarray_recording::assign_storage_portion(std::shared_ptr<recording_storage> stor,size_t array_index,const std::vector<snde_index> &new_dimlen, bool fortran_order,snde_index new_base_index)
  {
    size_t dimnum;
    snde_index stride=1;

    
    storage.at(array_index) = stor;
    

    std::vector<snde_index> strides;

    strides.reserve(new_dimlen.size());
    
    if (fortran_order) {
      for (dimnum=0;dimnum < new_dimlen.size();dimnum++) {
	strides.push_back(stride);
	stride *= new_dimlen.at(dimnum);
      }
    } else {
      // c order
      for (dimnum=0;dimnum < new_dimlen.size();dimnum++) {
	strides.insert(strides.begin(),stride);
	stride *= new_dimlen.at(new_dimlen.size()-dimnum-1);
      }
      
    }

    assert(new_base_index >= stor->base_index);
    
    snde_index base_index_shift = new_base_index-stor->base_index; 
    
    assert(stride + base_index_shift <= stor->nelem); // final stride (product of dimlen) + base_index difference had better be less than or equal to storage number of elements!
    
    layouts.at(array_index)=arraylayout(new_dimlen,strides);
    ndinfo(array_index)->basearray = storage.at(array_index)->lockableaddr();
    ndinfo(array_index)->base_index=new_base_index;
    ndinfo(array_index)->ndim=layouts.at(array_index).dimlen.size();
    ndinfo(array_index)->dimlen=layouts.at(array_index).dimlen.data();
    ndinfo(array_index)->strides=layouts.at(array_index).strides.data();
    ndinfo(array_index)->requires_locking_read=storage.at(array_index)->requires_locking_read;
    ndinfo(array_index)->requires_locking_write=storage.at(array_index)->requires_locking_write;
    void * dataaddr_or_null = storage.at(array_index)->dataaddr_or_null();
    if (dataaddr_or_null) {
      dataaddr_or_null = (void *)(((char *)dataaddr_or_null) + ndinfo(array_index)->elementsize*base_index_shift);
    }
    ndinfo(array_index)->shiftedarray = dataaddr_or_null;

  }

  
  std::shared_ptr<recording_storage> multi_ndarray_recording::allocate_storage(size_t array_index,const std::vector<snde_index> &dimlen, bool fortran_order) // fortran_order defaults to false
  // must assign info.elementsize and info.typenum before calling allocate_storage()
  // fortran_order only affects physical layout, not logical layout (interpretation of indices)
  {
    size_t dimnum;
    snde_index nelem=1;
    std::shared_ptr<recording_storage_manager> storman;
    std::shared_ptr<recording_storage> stor;

    for (dimnum=0;dimnum < dimlen.size();dimnum++) {
      nelem *= dimlen.at(dimnum);
    }

    std::string array_name="";
    if (name_mapping.size() > 0) {
      array_name = name_reverse_mapping.at(array_index);
    }

    storman = assign_storage_manager();
    
    // NOTE: Graphics storage manager allocate_recording() will create a nonmoving shadow if possible
    // to eliminate the need for locking
    stor = storman->allocate_recording(info->name,array_name,info->revision,originating_rss_unique_id,array_index,ndinfo(array_index)->elementsize,ndinfo(array_index)->typenum,nelem,!info->immutable);
    
    assign_storage(stor,array_index,dimlen,fortran_order);

    return stor;
    
  }


  std::shared_ptr<recording_storage> multi_ndarray_recording::allocate_storage_in_named_array(size_t array_index,std::string storage_array_name,const std::vector<snde_index> &dimlen, bool fortran_order) // fortran_order defaults to false
  // must assign info.elementsize and info.typenum before calling allocate_storage()
  // fortran_order only affects physical layout, not logical layout (interpretation of indices)
  {
    size_t dimnum;
    snde_index nelem=1;
    std::shared_ptr<recording_storage_manager> storman;
    std::shared_ptr<recording_storage> stor;

    for (dimnum=0;dimnum < dimlen.size();dimnum++) {
      nelem *= dimlen.at(dimnum);
    }


    storman = assign_storage_manager();
    
    // NOTE: Graphics storage manager allocate_recording() will create a nonmoving shadow if possible
    // to eliminate the need for locking
    stor = storman->allocate_recording(info->name,storage_array_name,info->revision,originating_rss_unique_id,array_index,ndinfo(array_index)->elementsize,ndinfo(array_index)->typenum,nelem,!info->immutable);
    
    assign_storage(stor,array_index,dimlen,fortran_order);

    return stor;
    
  }


  
  std::shared_ptr<recording_storage> multi_ndarray_recording::allocate_storage(std::string array_name,const std::vector<snde_index> &dimlen, bool fortran_order) // fortran_order defaults to false
  // must assign info.elementsize and info.typenum before calling allocate_storage()
  // fortran_order only affects physical layout, not logical layout (interpretation of indices)
  {
    
    size_t dimnum;
    snde_index nelem=1;
    std::shared_ptr<recording_storage_manager> storman;
    std::shared_ptr<recording_storage> stor;
    size_t array_index;

    for (dimnum=0;dimnum < dimlen.size();dimnum++) {
      nelem *= dimlen.at(dimnum);
    }

    array_index = name_mapping.at(array_name);



    storman = assign_storage_manager();
    
    // NOTE: Graphics storage manager allocate_recording() will create a nonmoving shadow if possible
    // to eliminate the need for locking
    stor = storman->allocate_recording(info->name,array_name,info->revision,originating_rss_unique_id,array_index,ndinfo(array_index)->elementsize,ndinfo(array_index)->typenum,nelem,!info->immutable);
    
    assign_storage(stor,array_index,dimlen,fortran_order);

    return stor; 
  }

  
  void multi_ndarray_recording::reference_immutable_recording(size_t array_index,std::shared_ptr<ndarray_recording_ref> rec,std::vector<snde_index> dimlen,std::vector<snde_index> strides)
  {
    snde_index first_index;
    snde_index last_index;
    size_t dimnum;

    if (!rec->rec->storage.at(array_index)->finalized) {
      throw snde_error("Recording %s trying to reference non-final data from recording %s",rec->rec->info->name,rec->rec->info->name);      
      
    }
    
    ndinfo(array_index)->typenum = rec->ndinfo()->typenum;

    if (ndinfo(array_index)->elementsize != 0 && ndinfo(array_index)->elementsize != rec->ndinfo()->elementsize) {
      throw snde_error("Element size mismatch in recording %s trying to reference data from recording %s",rec->rec->info->name,rec->rec->info->name);
    }
    ndinfo(array_index)->elementsize = rec->ndinfo()->elementsize;
    
    
    first_index=0;
    for (dimnum=0;dimnum < dimlen.size();dimnum++) {
      if (strides.at(dimnum) < 0) { // this is somewhat academic because strides is currently snde_index, which is currently unsigned so can't possibly be negative. This test is here in case we make snde_index signed some time in the future... 
	first_index += strides.at(dimnum)*(dimlen.at(dimnum))-1;
      }
    }
    if (first_index < 0) {
      throw snde_error("Referencing negative indices in recording %s trying to reference data from recording %s",rec->rec->info->name,rec->rec->info->name);
    }

    last_index=0;
    for (dimnum=0;dimnum < dimlen.size();dimnum++) {
      if (strides.at(dimnum) > 0) { 
	last_index += strides.at(dimnum)*(dimlen.at(dimnum))-1;
      }
    }
    if (last_index >= rec->storage->nelem) {
      throw snde_error("Referencing out-of-bounds indices in recording %s trying to reference data from recording %s",rec->rec->info->name,rec->rec->info->name);
    }

    
    storage.at(array_index) = rec->storage;
    layouts.at(array_index)=arraylayout(dimlen,strides);

    ndinfo(array_index)->basearray = storage.at(array_index)->lockableaddr();
    ndinfo(array_index)->base_index=storage.at(array_index)->base_index;
    ndinfo(array_index)->ndim=layouts.at(array_index).dimlen.size();
    ndinfo(array_index)->dimlen=layouts.at(array_index).dimlen.data();
    ndinfo(array_index)->strides=layouts.at(array_index).strides.data();
    ndinfo(array_index)->requires_locking_read=storage.at(array_index)->requires_locking_read;
    ndinfo(array_index)->requires_locking_write=storage.at(array_index)->requires_locking_write;
    ndinfo(array_index)->shiftedarray = storage.at(array_index)->dataaddr_or_null();
    
  }


  double multi_ndarray_recording::element_double(size_t array_index,const std::vector<snde_index> &idx)
  {
    return reference_ndarray(array_index)->element_double(idx);
  }
  double multi_ndarray_recording::element_double(size_t array_index,snde_index idx,bool fortran_order)
  {
    return reference_ndarray(array_index)->element_double(idx,fortran_order);
  }
  
  void multi_ndarray_recording::assign_double(size_t array_index,const std::vector<snde_index> &idx,double val)
  {
    reference_ndarray(array_index)->assign_double(idx,val);
  }

  void multi_ndarray_recording::assign_double(size_t array_index,snde_index idx,double val, bool fortran_order)
  {
    reference_ndarray(array_index)->assign_double(idx,val,fortran_order);
  }

  int64_t multi_ndarray_recording::element_int(size_t array_index,const std::vector<snde_index> &idx)
  {
    return reference_ndarray(array_index)->element_int(idx);
  }
  int64_t multi_ndarray_recording::element_int(size_t array_index,snde_index idx, bool fortran_order)
  {
    return reference_ndarray(array_index)->element_int(idx,fortran_order);
  }
  void multi_ndarray_recording::assign_int(size_t array_index,const std::vector<snde_index> &idx,int64_t val)
  {
    reference_ndarray(array_index)->assign_int(idx,val);
  }
  void multi_ndarray_recording::assign_int(size_t array_index,snde_index idx,int64_t val,bool fortran_order)
  {
    reference_ndarray(array_index)->assign_int(idx,val,fortran_order);
  }
  
  uint64_t multi_ndarray_recording::element_unsigned(size_t array_index,const std::vector<snde_index> &idx)
  {
    return reference_ndarray(array_index)->element_unsigned(idx);
  }
  uint64_t multi_ndarray_recording::element_unsigned(size_t array_index,snde_index idx, bool fortran_order)
  {
    return reference_ndarray(array_index)->element_unsigned(idx,fortran_order);
  }
  
  void multi_ndarray_recording::assign_unsigned(size_t array_index,const std::vector<snde_index> &idx,uint64_t val)
  {
    reference_ndarray(array_index)->assign_unsigned(idx,val);
  }
  void multi_ndarray_recording::assign_unsigned(size_t array_index,snde_index idx,uint64_t val,bool fortran_order)
  {
    reference_ndarray(array_index)->assign_unsigned(idx,val,fortran_order);
  }

  snde_complexfloat64  multi_ndarray_recording::element_complexfloat64(size_t array_index,const std::vector<snde_index> &idx)
  {
    return reference_ndarray(array_index)->element_complexfloat64(idx); 
  }
  snde_complexfloat64  multi_ndarray_recording::element_complexfloat64(size_t array_index,snde_index idx,bool fortran_order)
  {
    return reference_ndarray(array_index)->element_complexfloat64(idx,fortran_order); 
  }

  void multi_ndarray_recording::assign_complexfloat64(size_t array_index,const std::vector<snde_index> &idx,snde_complexfloat64 val)
  {
    reference_ndarray(array_index)->assign_complexfloat64(idx,val);
  }
  void multi_ndarray_recording::assign_complexfloat64(size_t array_index,snde_index idx,snde_complexfloat64 val,bool fortran_order)
  {
    reference_ndarray(array_index)->assign_complexfloat64(idx,val,fortran_order);
  }

  
  ndarray_recording_ref::ndarray_recording_ref(std::shared_ptr<multi_ndarray_recording> rec,size_t rec_index,unsigned typenum) :
    rec(rec),
    rec_index(rec_index),
    typenum(rec->ndinfo(rec_index)->typenum),
    info_state(rec->info_state),
    layout(rec->layouts.at(rec_index)),
    storage(rec->storage.at(rec_index))
  {
    
    bool found_compatible=false;
    if (this->typenum != typenum) {
      auto rct_it = rtn_compatible_types.find(this->typenum);
      if (rct_it != rtn_compatible_types.end()) {
	auto rct_compat_it = rct_it->second.find(typenum);
	if (rct_compat_it != rct_it->second.end()) {
	  found_compatible = true; 
	}
      }
    }
    
    assert(this->typenum==typenum || found_compatible);
    if (storage && storage->typenum != this->typenum) {
      throw snde_error("Type number mismatch between storage and reference: %d vs. %d",(int)storage->typenum,(int)typenum);
    }
  }

  ndarray_recording_ref::~ndarray_recording_ref()
  {
    
  }

  void ndarray_recording_ref::allocate_storage(std::vector<snde_index> dimlen, bool fortran_order)
  {
    rec->allocate_storage(rec_index,dimlen,fortran_order);
  }

  std::shared_ptr<recording_storage> ndarray_recording_ref::allocate_storage_in_named_array(std::string storage_array_name,const std::vector<snde_index> &dimlen, bool fortran_order)
  {
    return rec->allocate_storage_in_named_array(rec_index,storage_array_name,dimlen,fortran_order);
  }


  std::shared_ptr<ndarray_recording_ref> ndarray_recording_ref::assign_recording_type(unsigned typenum)
  // returns a new fully-typed reference. 
  {
    assert(!storage); // storage should not yet be defined
    this->typenum=typenum; // assign into ndinfo()->typenum via our reference
    assert(ndinfo()->typenum==typenum); // should always pass

    // assign elementsize
    ndinfo()->elementsize = rtn_typesizemap.at(typenum);

    return rec->reference_ndarray(rec_index); // Returned reference will now be fully typed and therefore can access the elements. 
  }
  
  double ndarray_recording_ref::element_double(const std::vector<snde_index> &idx)
  {
    throw snde_error("Cannot access elements of untyped recording reference. Create typed reference with .reference_ndarray() instead.");
  }

  double ndarray_recording_ref::element_double(snde_index idx,bool fortran_order)
  {
    throw snde_error("Cannot access elements of untyped recording reference. Create typed reference with .reference_ndarray() instead.");
  }

  void ndarray_recording_ref::assign_double(const std::vector<snde_index> &idx,double val)
  {
    throw snde_error("Cannot access elements of untyped recording reference. Create typed reference with .reference_ndarray()  instead.");
  }

  void ndarray_recording_ref::assign_double(snde_index idx,double val,bool fortran_order)
  {
    throw snde_error("Cannot access elements of untyped recording reference. Create typed reference with .reference_ndarray()  instead.");
  }

  int64_t ndarray_recording_ref::element_int(const std::vector<snde_index> &idx)
  {
    throw snde_error("Cannot access elements of untyped recording reference. Create typed reference with .reference_ndarray()  instead.");
  }

  int64_t ndarray_recording_ref::element_int(snde_index idx,bool fortran_order)
  {
    throw snde_error("Cannot access elements of untyped recording reference. Create typed reference with .reference_ndarray()  instead.");
  }

  void ndarray_recording_ref::assign_int(const std::vector<snde_index> &idx,int64_t val)
  {
    throw snde_error("Cannot access elements of untyped recording reference. Create typed reference with .reference_ndarray()  instead.");
  }

  void ndarray_recording_ref::assign_int(snde_index idx,int64_t val,bool fortran_order)
  {
    throw snde_error("Cannot access elements of untyped recording reference. Create typed reference with .reference_ndarray()  instead.");
  }

  uint64_t ndarray_recording_ref::element_unsigned(const std::vector<snde_index> &idx)
  {
    throw snde_error("Cannot access elements of untyped recording reference. Create typed reference with .reference_ndarray()  instead.");
  }

  uint64_t ndarray_recording_ref::element_unsigned(snde_index idx,bool fortran_order)
  {
    throw snde_error("Cannot access elements of untyped recording reference. Create typed reference with .reference_ndarray()  instead.");
  }

  
  void ndarray_recording_ref::assign_unsigned(const std::vector<snde_index> &idx,uint64_t val)
  {
    throw snde_error("Cannot access elements of untyped recording reference. Create typed reference with .reference_ndarray() instead.");
  }

  void ndarray_recording_ref::assign_unsigned(snde_index idx,uint64_t val,bool fortran_order)
  {
    throw snde_error("Cannot access elements of untyped recording reference. Create typed reference with .reference_ndarray() instead.");
  }

  snde_complexfloat64 ndarray_recording_ref::element_complexfloat64(const std::vector<snde_index> &idx)
  {
    throw snde_error("Cannot access elements of untyped recording reference. Create typed reference with .reference_ndarray() instead.");
  }
  snde_complexfloat64 ndarray_recording_ref::element_complexfloat64(snde_index idx,bool fortran_order)
  {
    throw snde_error("Cannot access elements of untyped recording reference. Create typed reference with .reference_ndarray() instead.");
  }
  
  void ndarray_recording_ref::assign_complexfloat64(const std::vector<snde_index> &idx,snde_complexfloat64 val)
  {
    throw snde_error("Cannot access elements of untyped recording reference. Create typed reference with .reference_ndarray() instead.");    
  }
  
  void ndarray_recording_ref::assign_complexfloat64(snde_index idx,snde_complexfloat64 val,bool fortran_order)
  {
    throw snde_error("Cannot access elements of untyped recording reference. Create typed reference with .reference_ndarray() instead.");    
  }

  
  
  fusion_ndarray_recording::fusion_ndarray_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_storage_manager> storage_manager,std::shared_ptr<transaction> defining_transact,std::string chanpath,std::shared_ptr<recording_set_state> _originating_rss,uint64_t new_revision,size_t info_structsize,unsigned typenum) :
    multi_ndarray_recording(recdb,storage_manager,defining_transact,chanpath,_originating_rss,new_revision,info_structsize,2 /* 2 ndarrays */)
    
  {
    rec_classes.push_back(recording_class_info("snde::fusion_ndarray_recording",typeid(fusion_ndarray_recording),ptr_to_new_shared_impl<fusion_ndarray_recording>));
    
    define_array(0,typenum,"accumulator");

    define_array(1,SNDE_RTN_SNDE_IMAGEDATA,"totals");
    
    
  }

  
  
#if 0
  // ***!!! This code is probably obsolete ***!!!
  rwlock_token_set recording::lock_storage_for_write()
  {
    bool immutable;
    {
      std::lock_guard<std::mutex> adminlock(admin);
      immutable=info->immutable;
    }
    if (immutable) {
      // storage locking not required for recording data for recordings that (will) be immutable
      return empty_rwlock_token_set();
    } else {
      assert(mutable_lock);
      rwlock_lockable *reader = &mutable_lock->reader;
      rwlock_token tok = std::make_shared<std::unique_lock<rwlock_lockable>>(*reader);
      rwlock_token_set ret = std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
      ret->emplace(reader,tok);
      return ret; 
    }
  }

  rwlock_token_set recording::lock_storage_for_read()
  {
    bool immutable;
    {
      std::lock_guard<std::mutex>(admin);
      immutable=info->immutable;
    }
    if (immutable) {
      // storage locking not required for recording data for recordings that (will) be immutable
      return empty_rwlock_token_set();
    } else {
      assert(mutable_lock);
      rwlock_lockable *writer = &mutable_lock->writer;
      rwlock_token tok = std::make_shared<std::unique_lock<rwlock_lockable>>(*writer);
      rwlock_token_set ret = std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
      ret->emplace(writer,tok);
      return ret; 
    }
    
  }
#endif // 0 (obsolete code)


  std::pair<std::shared_ptr<globalrevision>,bool> transaction::resulting_globalrevision()
  // returned bool true means null pointer indicates expired pointer, rather than in-progress transaction
  {
    bool expired_pointer=false;

    std::lock_guard<std::mutex> adminlock(admin);
    
    // NOTE: Very important not to try to turn this into something atomic.
    // This is used for synchronization of notification of transactions ending
    // vs recordings becoming ready. See also
    // _transactionrec_transaction_still_in_progress_admin_prelocked()
    // and rss._update_recstatus__rss_admin_transaction_admin_locked()
    
    std::shared_ptr<globalrevision> globalrev = _resulting_globalrevision.lock();
    if (!globalrev) {
      expired_pointer = invalid_weak_ptr_is_expired(_resulting_globalrevision);
    }
    return std::make_pair(globalrev,expired_pointer);
  }

  bool transaction::transaction_globalrev_is_complete()
  {
    std::shared_ptr<globalrevision> globalrev;
    bool expired_pointer;

    std::tie(globalrev,expired_pointer) = resulting_globalrevision();

    if (expired_pointer) {
      return true; 
    }
    
    if (!globalrev) {
      return false;
    }

    // check if all recordings are ready;
    {
      std::lock_guard<std::mutex> globalrev_admin(globalrev->admin);

      bool all_ready = !globalrev->recstatus.defined_recordings.size() && !globalrev->recstatus.instantiated_recordings.size();      

      return all_ready; 
    }
  }

  std::shared_ptr<promise_channel_notify> transaction::get_transaction_globalrev_complete_waiter()
  // The waiter is interruptable by a promise exception; some sort of interrupt() method.
  {
    std::shared_ptr<promise_channel_notify> retval;

    retval = std::make_shared<promise_channel_notify>(std::vector<std::string>(),std::vector<std::string>(),true);

    retval->apply_to_transaction(shared_from_this());

    return retval;
  }
  
  active_transaction::active_transaction(std::shared_ptr<recdatabase> recdb) :
    recdb(recdb),
    transaction_ended(false)
  {
    std::unique_lock<movable_mutex> tr_lock_acquire(recdb->transaction_lock);;

    tr_lock_acquire.swap(transaction_lock_holder); // transfer lock into holder
    //recdb->_transaction_raii_holder=shared_from_this();

    uint64_t previous_globalrev_index = 0;
    {
      std::lock_guard<std::mutex> recdb_lock(recdb->admin);

      assert(!recdb->current_transaction);
      
      recdb->current_transaction=std::make_shared<transaction>();
      
    

      if (recdb->_globalrevs.size()) {
	// if there are any globalrevs (otherwise we are starting the first!)
	std::map<uint64_t,std::shared_ptr<globalrevision>>::iterator last_globalrev_ptr = recdb->_globalrevs.end();
	--last_globalrev_ptr; // change from final+1 to final entry
	previous_globalrev_index = last_globalrev_ptr->first;
	previous_globalrev = last_globalrev_ptr->second;

      } else {
	// this is the first globalrev
	previous_globalrev = nullptr; 
      }
      recdb->current_transaction->globalrev = previous_globalrev_index+1;
      recdb->current_transaction->rss_unique_index = rss_get_unique(); // will be transferred into the rss during end_transaction()
    }     
    

    
  }

  static void _identify_changed_channels(const std::map<std::string,std::shared_ptr<channelconfig>> &all_channels_by_name,std::shared_ptr<instantiated_math_database> mathdb, std::unordered_set<std::shared_ptr<instantiated_math_function>> *unknownchanged_math_functions,std::unordered_set<std::shared_ptr<instantiated_math_function>> *changed_math_functions,std::unordered_set<std::shared_ptr<channelconfig>> *unknownchanged_channels,std::unordered_set<std::shared_ptr<channelconfig>> *changed_channels_dispatched,std::unordered_set<std::shared_ptr<channelconfig>> *changed_channels_need_dispatch,std::unordered_set<std::shared_ptr<instantiated_math_function>> *possiblychanged_math_functions,const std::unordered_set<std::shared_ptr<channelconfig>>::iterator &channel_to_dispatch_it,bool enable_ondemand) // channel_to_dispatch should be in changed_channels_need_dispatch
  // mathdb must be immutable; (i.e. it must already be copied into the global revision)
  {
    // std::unordered_set iterators remain valid so long as the iterated element itself is not erased, and the set does not need rehashing.
    // Therefore the changed_channels_dispatched and changed_channels_need_dispatch set needs to have reserve() called on it first with the maximum possible growth.
    
    // We do add to changed_channels_need dispatch
    //std::unordered_set<std::shared_ptr<channelconfig>>::iterator changed_chan;

    std::shared_ptr<channelconfig> channel_to_dispatch = *channel_to_dispatch_it;
    changed_channels_need_dispatch->erase(channel_to_dispatch_it);
    changed_channels_dispatched->emplace(channel_to_dispatch);

    // look up what is dependent on this channel
    //snde_debug(,"_identify_changed_channels(): Dispatching %s with %d math dependencies",channel_to_dispatch->channelpath.c_str(),);
    
    auto dep_it = mathdb->all_dependencies_of_channel.find(channel_to_dispatch->channelpath);
    if (dep_it != mathdb->all_dependencies_of_channel.end()) {
      
      std::unordered_set<std::shared_ptr<instantiated_math_function>> &dependent_math_functions=dep_it->second;
      
      for (auto && instantiated_math_ptr: dependent_math_functions) {
	if ((instantiated_math_ptr->ondemand && !enable_ondemand) || instantiated_math_ptr->disabled) {
	  // ignore disabled and ondemand dependent channels as appropriate
	  continue; 
	}

	std::unordered_set<std::shared_ptr<instantiated_math_function>>::iterator unknownchanged_it = unknownchanged_math_functions->find(instantiated_math_ptr);
	if (instantiated_math_ptr->fcn->new_revision_optional) {
	  // this math function is possiblychanged
	  
	  if (unknownchanged_it != unknownchanged_math_functions->end()) {
	    unknownchanged_math_functions->erase(unknownchanged_it);
	    possiblychanged_math_functions->emplace(instantiated_math_ptr);
	  }	
	} else {
	  // mark math function as changed if it isn't already
	  if (unknownchanged_it != unknownchanged_math_functions->end()) {
	    unknownchanged_math_functions->erase(unknownchanged_it);
	    changed_math_functions->emplace(instantiated_math_ptr);
	  } else {
	    // if listed as possibly-changed, bump up to definitely changed
	    std::unordered_set<std::shared_ptr<instantiated_math_function>>::iterator possiblychanged_it = possiblychanged_math_functions->find(instantiated_math_ptr);
	    if (possiblychanged_it != possiblychanged_math_functions->end()) {
	      possiblychanged_math_functions->erase(possiblychanged_it);
	      changed_math_functions->emplace(instantiated_math_ptr);
	    }
	  }
	  
	  for (auto && result_chanpath_name_ptr: instantiated_math_ptr->result_channel_paths) {
	    if (result_chanpath_name_ptr) {
	      // Found a dependent channel name
	      // Could probably speed this up by copying result_channel_paths into a form where it points directly at channelconfs. 
	      std::shared_ptr<channelconfig> channelconf = all_channels_by_name.at(recdb_path_join(instantiated_math_ptr->channel_path_context,*result_chanpath_name_ptr));
	      
	      std::unordered_set<std::shared_ptr<channelconfig>>::iterator unknownchanged_it = unknownchanged_channels->find(channelconf);
	      // Is the dependenent channel in unknownchanged_channels?... if so it is definitely changed, but not yet dispatched
	      if (unknownchanged_it != unknownchanged_channels->end()) {
		// put it in the changed dispatch set
		changed_channels_need_dispatch->emplace(*unknownchanged_it);
		// remove it from the unknown changed set
		unknownchanged_channels->erase(unknownchanged_it);
	      } 
	    }
	  }
	}
      }
    }
  

        
  }




  static void _identify_possiblychanged_channels(const std::map<std::string,std::shared_ptr<channelconfig>> &all_channels_by_name,std::shared_ptr<instantiated_math_database> mathdb, std::unordered_set<std::shared_ptr<instantiated_math_function>> *unknownchanged_math_functions,std::unordered_set<std::shared_ptr<instantiated_math_function>> *changed_math_functions,std::unordered_set<std::shared_ptr<channelconfig>> *unknownchanged_channels,std::unordered_set<std::shared_ptr<channelconfig>> *changed_channels_dispatched,std::unordered_set<std::shared_ptr<channelconfig>> *possiblychanged_channels_dispatched,std::unordered_set<std::shared_ptr<channelconfig>> *possiblychanged_channels_need_dispatch,std::unordered_set<std::shared_ptr<instantiated_math_function>> *possiblychanged_math_functions,const std::unordered_set<std::shared_ptr<channelconfig>>::iterator &channel_to_dispatch_it, bool enable_ondemand ) // channel_to_dispatch should be in changed_channels_need_dispatch
  // mathdb must be immutable; (i.e. it must already be copied into the global revision)
  {
    // std::unordered_set iterators remain valid so long as the iterated element itself is not erased, and the set does not need rehashing.
    // Therefore the possiblychanged_channels_dispatched and possiblychanged_channels_need_dispatch set needs to have reserve() called on it first with the maximum possible growth.
    
    // We do add to changed_channels_need dispatch
    //std::unordered_set<std::shared_ptr<channelconfig>>::iterator changed_chan;

    std::shared_ptr<channelconfig> channel_to_dispatch = *channel_to_dispatch_it;
    possiblychanged_channels_need_dispatch->erase(channel_to_dispatch_it);
    possiblychanged_channels_dispatched->emplace(channel_to_dispatch);

    // look up what is dependent on this channel
    
    auto dep_it = mathdb->all_dependencies_of_channel.find(channel_to_dispatch->channelpath);
    if (dep_it != mathdb->all_dependencies_of_channel.end()) {
      
      std::unordered_set<std::shared_ptr<instantiated_math_function>> &dependent_math_functions=dep_it->second;
      
      for (auto && instantiated_math_ptr: dependent_math_functions) {
	if ((instantiated_math_ptr->ondemand && !enable_ondemand) || instantiated_math_ptr->disabled) {
	  // ignore disabled and ondemand dependent channels as appropriate
	  continue; 
	}

	// mark math function as possibly changed if it isn't already
	std::unordered_set<std::shared_ptr<instantiated_math_function>>::iterator unknownchanged_it = unknownchanged_math_functions->find(instantiated_math_ptr);
	if (unknownchanged_it != unknownchanged_math_functions->end()) {
	  unknownchanged_math_functions->erase(unknownchanged_it);
	  possiblychanged_math_functions->emplace(instantiated_math_ptr);
	}	

	for (auto && result_chanpath_name_ptr: instantiated_math_ptr->result_channel_paths) {
	  if (result_chanpath_name_ptr) {
	    // Found a dependent channel name
	    // Could probably speed this up by copying result_channel_paths into a form where it points directly at channelconfs. 
	    std::shared_ptr<channelconfig> channelconf = all_channels_by_name.at(recdb_path_join(instantiated_math_ptr->channel_path_context,*result_chanpath_name_ptr));
	    
	    std::unordered_set<std::shared_ptr<channelconfig>>::iterator unknownchanged_it = unknownchanged_channels->find(channelconf);
	    // Is the dependenent channel in unknownchanged_channels?... if so it is possibly changed, but not yet dispatched
	    if (unknownchanged_it != unknownchanged_channels->end()) {
	      // put it in the possiblychanged dispatch set
	      possiblychanged_channels_need_dispatch->emplace(*unknownchanged_it);
	      // remove it from the unknown changed set
	      unknownchanged_channels->erase(unknownchanged_it);
	    } 
	    
	  }
	}
      }
    }
    
    
        
  }
  


  /*  
  static void mark_prospective_calculations(std::shared_ptr<recording_set_state> state,std::unordered_set<std::shared_ptr<channelconfig>> definitelychanged_channels_to_process)
  // marks execution_demanded flag in the math_function_status for all math functions directly
  // dependent on the definitively_changed_channels.
  // (should this really be in recmath.cpp?)
  {
    // Presumes any state admin lock already held or not needed because not yet published

    // Go through the all_dependencies_of_channel of the instantiated_math_database, accumulating
    // all of the instantiated_math_functions into a set
    // Then go through the set, look up the math_function_status from the math_status and set the execution_demanded flag

    //std::unordered_set<std::shared_ptr<instantiated_math_function>> dependent_functions;
    
    for (auto && changed_channelconfig_ptr : definitelychanged_channels_to_process) {
      
      //std::unordered_set<std::shared_ptr<instantiated_math_function>>  *dependent_functions_of_this_channel = nullptr;

      auto dep_it = state->mathstatus.math_functions->all_dependencies_of_channel.find(changed_channelconfig_ptr->channelpath);
      if (dep_it != state->mathstatus.math_functions->all_dependencies_of_channel.end()) {
	
	for (auto && affected_math_function: dep_it->second) {
	  // merge into set 
	  //dependent_functions.emplace(affected_math_function);
	  
	  // instead of merging into a set and doing it later we just go ahead and set the execution_demanded flag directly
	  if (!affected_math_function->disabled || !affected_math_function->ondemand) {
	    state->mathstatus.function_status.at(affected_math_function).execution_demanded = true; 
	  }
	}
      }
      

    }

    // go over all of the dependent functions and mark their execution_demanded flag
    //for (auto && affected_math_function: dependent_functions) {
    //  state->mathstatus.function_status.at(affected_math_function).execution_demanded = true; 
    //}
  }
  */
  
  static bool check_all_dependencies_mdonly(
					    std::shared_ptr<recording_set_state> state,
					    std::unordered_set<std::shared_ptr<instantiated_math_function>> *unchecked_functions,
					    std::unordered_set<std::shared_ptr<instantiated_math_function>> *passed_mdonly_functions,
					    std::unordered_set<std::shared_ptr<instantiated_math_function>> *failed_mdonly_functions,
					    std::unordered_set<std::shared_ptr<instantiated_math_function>> *checked_regular_functions,
					    const std::unordered_set<std::shared_ptr<instantiated_math_function>>::iterator &function_to_check_it)
  // returns true if function_to_check_it refers to an mdonly function with all dependencies also being mdonly
  // assumes state is admin-locked or still under construction and not needing locks. 
  {
    std::shared_ptr<instantiated_math_function> function_to_check = *function_to_check_it;
    unchecked_functions->erase(function_to_check_it);  // remove function_to_check from unchecked_functions
    
    if (!function_to_check->mdonly) {
      checked_regular_functions->emplace(function_to_check);
      return false;
    }

    bool all_mdonly = true;
    std::unordered_set<std::shared_ptr<instantiated_math_function>> &dependent_functions = state->mathstatus.math_functions->all_dependencies_of_function.at(function_to_check);

    for (auto && dependent_function : dependent_functions) {
      if (passed_mdonly_functions->find(dependent_function) != passed_mdonly_functions->end()) {
	// found previously passed function. all good
	continue; 
      }
      if (failed_mdonly_functions->find(dependent_function) != failed_mdonly_functions->end()) {
	// Found previously failed function. Failed.
	all_mdonly=false;
	break;
      }
      if (checked_regular_functions->find(dependent_function) != checked_regular_functions->end()) {
	// This dependent function is not mdonly so we are failed.
	all_mdonly=false;
	break;
      }

      std::unordered_set<std::shared_ptr<instantiated_math_function>>::iterator unchecked_it = unchecked_functions->find(dependent_function);
      if (unchecked_it != unchecked_functions->end()) {
	// unchecked function... check it with recursive call.
	bool unchecked_status = check_all_dependencies_mdonly(state,
							      unchecked_functions,
							      passed_mdonly_functions,
							      failed_mdonly_functions,
							      checked_regular_functions,
							      unchecked_it);
	if (unchecked_status) {
	  // passed
	  continue;
	} else {
	  // failed
	  all_mdonly=false;
	  break;
	}
      } else {
	// Should not be possible to get here -- dependency graph should be acyclic. 
	assert(0);
      }
    }
    
    return all_mdonly; 
  }


  void build_rss_from_functions_and_channels(std::shared_ptr<recdatabase> recdb,
					     std::shared_ptr<recording_set_state> previous_rss,
					     std::shared_ptr<recording_set_state> new_rss,
					     const std::map<std::string,std::shared_ptr<channelconfig>> &all_channels_by_name,
					     // set of channels definitely changed, according to whether we've dispatched them in our graph search
					     // for possibly dependent channels 
					     std::unordered_set<std::shared_ptr<channelconfig>> *changed_channels_need_dispatch,
					     std::unordered_set<std::shared_ptr<channelconfig>> *changed_channels_dispatched,
					     // Set of channels known to be definitely unchanged
					     std::unordered_set<std::shared_ptr<channelconfig>> *unchanged_channels,
					     // set of channels not yet known to be changed
					     std::unordered_set<std::shared_ptr<channelconfig>> *unknownchanged_channels,
					     // set of math functions not known to be changed or unchanged
					     std::unordered_set<std::shared_ptr<instantiated_math_function>> *unknownchanged_math_functions,
					     // set of math functions known to be (definitely) changed
					     std::unordered_set<std::shared_ptr<instantiated_math_function>> *changed_math_functions,
					     std::unordered_set<std::shared_ptr<channelconfig>> *explicitly_updated_channels,
    std::unordered_set<channel_state *> *ready_channels, // references into the new_rss->recstatus.channel_map
					     std::vector<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>> *ready_to_execute,
					     std::set<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>>,mncn_lessthan> *may_need_completion_notification,
					     bool *all_ready,
					     bool ondemand_only) // ondemand_only enables the ondemand mode, not worrying about math calculations that have don't have the ondemand flag set, and yes worrying about math calculations that do have the ondemand flag set
  
  {



    //std::set<std::shared_ptr<recording_base>> recordings_needing_finalization; // automatic recordings created here that need to be marked as ready
  	
    // set of math functions known to be potentially changed, i.e. dynamic dependencies)
    std::unordered_set<std::shared_ptr<instantiated_math_function>> possiblychanged_math_functions; 

    // set of unchanged incomplete channels
    std::unordered_set<channel_state *> unchanged_incomplete_channels; // references into the new_rss->recstatus.channel_map
    std::unordered_set<channel_state *> unchanged_incomplete_mdonly_channels; // references into the new_rss->recstatus.channel_map

    
    // Put all functions with dynamic_dependencies into possiblychanged_math_functions as they could well need to be run. 
    for (auto && instmath_function__function_deps: new_rss->mathstatus.math_functions->all_dependencies_of_function) {
      std::shared_ptr<instantiated_math_function> instmath_function = instmath_function__function_deps.first;

      if (instmath_function->fcn->dynamic_dependency) {

	std::unordered_set<std::shared_ptr<instantiated_math_function>>::iterator unknownchanged_it = unknownchanged_math_functions->find(instmath_function);
	if (unknownchanged_it != unknownchanged_math_functions->end()) {
	  unknownchanged_math_functions->erase(unknownchanged_it);
	  possiblychanged_math_functions.emplace(instmath_function);
	}
      }
    }


    
    // Now empty out changed_channels_need_dispatch, adding into it any dependencies of currently referenced changed channels
    // This progressively identifies all (possibly) changed channels and changed math functions

    // definitely changed channels (accumulated in changed_channels_dispatched) are channels that are either directly modified
    // or derive (perhaps indirectly) by non-new_revision_optional function(s) from a directly modified channel 
    std::unordered_set<std::shared_ptr<channelconfig>>::iterator channel_to_dispatch_it = changed_channels_need_dispatch->begin();
    while (channel_to_dispatch_it != changed_channels_need_dispatch->end()) {
      _identify_changed_channels(all_channels_by_name,new_rss->mathstatus.math_functions, unknownchanged_math_functions,changed_math_functions,unknownchanged_channels,changed_channels_dispatched,changed_channels_need_dispatch,&possiblychanged_math_functions,channel_to_dispatch_it, ondemand_only );

      channel_to_dispatch_it = changed_channels_need_dispatch->begin();
    }

    // place to store channels that are possibly, but not definitely, changed, but have not been fully processed
    std::unordered_set<std::shared_ptr<channelconfig>> possiblychanged_channels_need_dispatch;
    // place to store channels that are possibly, but not definitely, changed, but have been fully processed
    std::unordered_set<std::shared_ptr<channelconfig>> possiblychanged_channels_dispatched;

    // make sure hash tables won't rehash and screw up iterators or similar
    possiblychanged_channels_dispatched.reserve(unknownchanged_channels->size());
    possiblychanged_channels_need_dispatch.reserve(unknownchanged_channels->size());


    // Seed set of possibly changed channels from possibly changed math functions
    for (auto && possiblychanged_math_function: possiblychanged_math_functions) {
      for (auto && reschanpath_ptr: possiblychanged_math_function->result_channel_paths) {
	if (reschanpath_ptr) {
	  std::shared_ptr<channelconfig> possiblychanged_channel = all_channels_by_name.at(recdb_path_join(possiblychanged_math_function->channel_path_context,*reschanpath_ptr));
	  std::unordered_set<std::shared_ptr<channelconfig>>::iterator unknownchanged_it = unknownchanged_channels->find(possiblychanged_channel);

	  if (unknownchanged_it != unknownchanged_channels->end()) {
	    // this channel is in unknownchanged. Move it to possiblychanged_channeld_need_dispatch
	    possiblychanged_channels_need_dispatch.emplace(*unknownchanged_it);
	    unknownchanged_channels->erase(unknownchanged_it);
	  }
	  
	}
      }
    }

    // Now recursively seek out any other possiblychanged_channels 
    channel_to_dispatch_it = possiblychanged_channels_need_dispatch.begin();
    while (channel_to_dispatch_it != possiblychanged_channels_need_dispatch.end()) {
      _identify_possiblychanged_channels(all_channels_by_name,new_rss->mathstatus.math_functions, unknownchanged_math_functions,changed_math_functions,unknownchanged_channels,changed_channels_dispatched,&possiblychanged_channels_dispatched,&possiblychanged_channels_need_dispatch,&possiblychanged_math_functions,channel_to_dispatch_it, ondemand_only );

      channel_to_dispatch_it = possiblychanged_channels_need_dispatch.begin();
    }
    

    // possibly_changed_channels (accumulated in possiblychanged_channels_dispatched) are channels that:
    // (a) derive from a dynamic_dependency (listed in possiblychanged_math_functions)
    // (b) derive by a new_revision_optional function from a directly_modified or possibly_changed channel
    
    // ***!!!*** need to distinguish between definitively_changed_channels and possibly_changed_channels
    //   ***!!!*** Based on new_revision optional and dynamic_dependency.
    //   ***!!!*** But decision has to be made in the proper sequence so that it is know before any
    //   ***!!!*** outputs are used as inputs to other functions. 
    
    // now changed_channels_need_dispatch is empty, changed_channels_dispatched represents all changed or possibly-changed (i.e. math may decide presence of a new revision) channels,
    // and unknownchanged_channels represents channels which are known to be definitively unchanged (not dependent directly or indirectly on anything that may have changed)
    std::unordered_set<std::shared_ptr<channelconfig>> &definitelychanged_channels_to_process=*changed_channels_dispatched;
    std::unordered_set<std::shared_ptr<channelconfig>> &possiblychanged_channels_to_process=possiblychanged_channels_dispatched;
    //std::unordered_set<std::shared_ptr<channelconfig>> &unchanged_channels=*unknownchanged_channels;
    // Anything remaining in unknownchanged_channels is now definitively unchanged
    for (auto && unchanged_channel: *unknownchanged_channels) {
      unchanged_channels->emplace(unchanged_channel);
    }
    

    // have changed_math_functions, possiblychanged_math_functions. Anything unknown is now an unchanged_math_function
    std::unordered_set<std::shared_ptr<instantiated_math_function>> &unchanged_math_functions = *unknownchanged_math_functions; 


    std::unordered_set<std::shared_ptr<instantiated_math_function>> unchanged_complete_math_functions;
    std::unordered_set<std::shared_ptr<instantiated_math_function>> unchanged_complete_math_functions_mdonly;

    std::unordered_set<std::shared_ptr<instantiated_math_function>> unchanged_incomplete_math_functions;


    // Figure out which functions are actually mdonly...
    // Traverse graph from all mdonly_functions, checking that their dependencies are all mdonly. If not
    // they need to be moved from mdonly_pending_functions or completed_mdonly_functions into pending_functions
    std::unordered_set<std::shared_ptr<instantiated_math_function>> unchecked_functions = new_rss->mathstatus.math_functions->mdonly_functions;
    std::unordered_set<std::shared_ptr<instantiated_math_function>> passed_mdonly_functions;
    std::unordered_set<std::shared_ptr<instantiated_math_function>> failed_mdonly_functions;
    std::unordered_set<std::shared_ptr<instantiated_math_function>> checked_regular_functions;


    
    std::unordered_set<std::shared_ptr<instantiated_math_function>>::iterator unchecked_function = unchecked_functions.begin();
    while (unchecked_function != unchecked_functions.end()) {
      // check_all_dependencies_mdonly checks to see whether all of the direct or indirect results of an mdonly function are also
      // mdonly ("passed") or not ("failed") Only passed mdonly functions can be run as mdonly because otherwise we'd have a
      // non-mdonly function that depended on the output
      check_all_dependencies_mdonly(new_rss,&unchecked_functions,&passed_mdonly_functions,&failed_mdonly_functions,&checked_regular_functions,unchecked_function);
      
      unchecked_function = unchecked_functions.begin();
    }

    // any failed_mdonly_functions need to have their mdonly bool cleared in their math_function_status
    // and need to be moved from mdonly_pending_functions into pending_functions
    for (auto && failed_mdonly_function: failed_mdonly_functions) {
      math_function_status &funcstatus = new_rss->mathstatus.function_status.at(failed_mdonly_function);
      funcstatus.mdonly=false;
      if (funcstatus.execfunc) {
	funcstatus.execfunc->mdonly=false;
      }
      new_rss->mathstatus.mdonly_pending_functions.erase(new_rss->mathstatus.mdonly_pending_functions.find(failed_mdonly_function));
      new_rss->mathstatus.pending_functions.emplace(failed_mdonly_function);
    }


    // Now reference previous revision of all unchanged channels, inserting into the new new_rss's channel_map
    // and also marking any corresponding function_status as complete
    for (auto && unchanged_channel: *unchanged_channels) {
      // recall unchanged_channels are not changed or dependent directly or indirectly on anything changed
      
      std::shared_ptr<recording_base> channel_rec;
      std::shared_ptr<recording_base> channel_rec_is_complete;

      std::shared_ptr<instantiated_math_function> unchanged_channel_math_function;
      bool is_mdonly = false;
     
      if (unchanged_channel->math && !(ondemand_only ^ unchanged_channel->ondemand)) {
	// for math channels, ondemand math channels iff ondemand_only is set, non-ondemand channels iff ondemand_only is not set 
	unchanged_channel_math_function = new_rss->mathstatus.math_functions->defined_math_functions.at(unchanged_channel->channelpath);
	is_mdonly = unchanged_channel_math_function->mdonly; // new_rss->mathstatus.function_status.at(unchanged_channel_math_function).mdonly;
	
      }
      
      assert(previous_rss); // Should always be present, because if we are on the first revision, each channel should have had a new recording and therefore is changed!
      {
	std::lock_guard<std::mutex> previous_rss_admin(previous_rss->admin);
	
	std::map<std::string,channel_state>::iterator previous_rss_chanmap_entry = previous_rss->recstatus.channel_map->find(unchanged_channel->channelpath);
	if (previous_rss_chanmap_entry==previous_rss->recstatus.channel_map->end()) {
	  throw snde_error("Existing channel %s has no prior recording",unchanged_channel->channelpath.c_str());
	}
	channel_rec_is_complete = previous_rss_chanmap_entry->second.recording_is_complete(is_mdonly);
	if (channel_rec_is_complete) {
	  channel_rec = channel_rec_is_complete;
	} else {
	  channel_rec = previous_rss_chanmap_entry->second.rec();
	}
      }
      std::map<std::string,channel_state>::iterator channel_map_it;
      //bool added_successfully;
      //std::tie(channel_map_it,added_successfully) =
      //new_rss->recstatus.channel_map->emplace(std::piecewise_construct,
      //std::forward_as_tuple(unchanged_channel->channelpath),
      //						 std::forward_as_tuple(unchanged_channel,channel_rec,false)); // mark channel_state as not updated
      channel_map_it = new_rss->recstatus.channel_map->find(unchanged_channel->channelpath);
      assert(channel_map_it != new_rss->recstatus.channel_map->end());
      //channel_map_it->second._rec = channel_rec; // may be nullptr if recording hasn't been defined yet
      channel_map_it->second.end_atomic_rec_update(channel_rec);

      // check if added channel is complete
      if (channel_rec_is_complete) {
	// assign the channel_state revision field (now implicit in the above)
	//channel_map_it->second.revision = std::make_shared<uint64_t>(channel_rec->info->revision);

	
	// if it is math, queue it to mark the function_status as complete, because if we are looking at it here, all
	// direct or indirect prerequisites must also be unchanged
	if (unchanged_channel->math && !(ondemand_only ^ unchanged_channel->ondemand)) {
	  // for math channels, ondemand math channels iff ondemand_only is set, non-ondemand channels iff ondemand_only is not set
	  std::shared_ptr<instantiated_math_function> channel_math_function = new_rss->mathstatus.math_functions->defined_math_functions.at(unchanged_channel->channelpath);
	  if (is_mdonly) {
	    unchanged_complete_math_functions_mdonly.emplace(channel_math_function);
	  } else {
	    unchanged_complete_math_functions.emplace(channel_math_function);
	  }
	}
	// Queue notification that this channel is complete, except we are currently holding the admin lock, so we can't
	// do it now. Instead queue up a reference into the channel_map
	ready_channels->emplace(&channel_map_it->second);

	// place in new_rss->recstatus.completed_recordings
	new_rss->recstatus.defined_recordings.erase(unchanged_channel);
	new_rss->recstatus.completed_recordings.emplace(std::piecewise_construct,
							 std::forward_as_tuple((std::shared_ptr<channelconfig>)unchanged_channel),
							 std::forward_as_tuple((channel_state *)&channel_map_it->second));
      } else {
	// in this case channel_rec may or may not exist and is NOT complete
	if (unchanged_channel->math && !(ondemand_only ^ unchanged_channel->ondemand)) {
	// for math channels, ondemand math channels iff ondemand_only is set, non-ondemand channels iff ondemand_only is not set
	  
	  unchanged_incomplete_math_functions.emplace(unchanged_channel_math_function);
	  if (is_mdonly) {
	    unchanged_incomplete_mdonly_channels.emplace(&channel_map_it->second);
	  } else {
	    unchanged_incomplete_channels.emplace(&channel_map_it->second);
	  }

	} else {
	  // not a math function; cannot be mdonly
	  unchanged_incomplete_channels.emplace(&channel_map_it->second);
	}
	
      }
    }
    
    
    // All of the math functions were previously classified into changed_math_functions or unchanged_math_functions or possiblychanged_math_functions
    // The above should have further classified unchanged_math_functions into unchanged_complete(maybe with mdonly) or unchanged_incomplete
  
    for (auto && unchanged_complete_math_function: unchanged_complete_math_functions) {
      // For all fully ready math functions with no inputs changed, mark them as complete
      // and put them into the appropriate completed set in math_status.

      math_function_status &ucmf_status = new_rss->mathstatus.function_status.at(unchanged_complete_math_function);
      //ucmf_status.mdonly_executed=true;

      ucmf_status.complete = true; // is this correct if we are mdonly??? 

      // reference the prior math_function_execution, although it won't be of much use
      snde_debug(SNDE_DC_RECDB,"ucmf: assigning execfunc");
      
      std::shared_ptr<math_function_execution> execfunc;

      {
	std::lock_guard<std::mutex> prev_gr_admin(previous_rss->admin);
	execfunc = previous_rss->mathstatus.function_status.at(unchanged_complete_math_function).execfunc;
	ucmf_status.mdonly = execfunc->mdonly;
      }

      ucmf_status.execfunc = execfunc;
      // new new_rss needs to be added to execfunc->referencing_rss
      // but we're not ready to do this until we're ready to
      // publish this new_rss, so we do this later
      
      
      ucmf_status.complete = true;
      // remove from the appropriate set

      auto mpf_it = new_rss->mathstatus.mdonly_pending_functions.find(unchanged_complete_math_function);
      if (mpf_it != new_rss->mathstatus.mdonly_pending_functions.end()) {

	new_rss->mathstatus.mdonly_pending_functions.erase(mpf_it);

      } else {
	auto pf_it = new_rss->mathstatus.pending_functions.find(unchanged_complete_math_function);
	assert(pf_it != new_rss->mathstatus.pending_functions.end());

	
	new_rss->mathstatus.pending_functions.erase(pf_it);
	
      }
      // add to the complete fully ready set. 
      new_rss->mathstatus.completed_functions.emplace(unchanged_complete_math_function);
      
      
    }
    
    for (auto && unchanged_complete_math_function: unchanged_complete_math_functions_mdonly) {
      // For all fully ready mdonly math functions with no inputs changed, mark them as complete
      // and put them into the appropriate completed set in math_status.
      
      math_function_status &ucmf_status = new_rss->mathstatus.function_status.at(unchanged_complete_math_function);
      
      ucmf_status.complete = true; // is this correct if we are mdonly??? 

      // reference the prior math_function_execution; we may still need to execute past mdonly
      snde_debug(SNDE_DC_RECDB,"ucmfm: assigning execfunc");
      
      std::shared_ptr<math_function_execution> execfunc;
      {
	std::lock_guard<std::mutex> prev_gr_admin(previous_rss->admin);

	execfunc = previous_rss->mathstatus.function_status.at(unchanged_complete_math_function).execfunc;
	ucmf_status.mdonly = execfunc->mdonly;
      }
      ucmf_status.execfunc = execfunc;
      // new new_rss needs to be added to execfunc->referencing_rss
      // but we're not ready to do this until we're ready to
      // publish this new_rss, so we do this later

      assert(unchanged_complete_math_function->mdonly); // shouldn't be in this list if we aren't mdonly.
      
      auto mpf_it = new_rss->mathstatus.mdonly_pending_functions.find(unchanged_complete_math_function);
      assert(mpf_it != new_rss->mathstatus.mdonly_pending_functions.end());
	
      new_rss->mathstatus.mdonly_pending_functions.erase(mpf_it);

      new_rss->mathstatus.completed_mdonly_functions.emplace(unchanged_complete_math_function);
	
      
    }
  
  
    for (auto && unchanged_incomplete_math_function: unchanged_incomplete_math_functions) {
      // This math function had no inputs changed, but was not fully executed (at least as of when we checked above)
      // for unchanged incomplete math functions 
      // Need to import math progress: math_function_execution from previous_rss
            
      math_function_status &uimf_status = new_rss->mathstatus.function_status.at(unchanged_incomplete_math_function);
      snde_debug(SNDE_DC_RECDB,"ucimf: assigning execfunc");

      std::shared_ptr<math_function_execution> execfunc;
      {
	std::lock_guard<std::mutex> prev_gr_admin(previous_rss->admin);
	
	execfunc = previous_rss->mathstatus.function_status.at(unchanged_incomplete_math_function).execfunc;
	if (execfunc) {
	  uimf_status.mdonly = execfunc->mdonly;
	  uimf_status.execfunc = execfunc;
	} else {
	  // set implicit self-dependency we can get another try when prior execfunc is actually set
	  uimf_status.execfunc = nullptr;
	  uimf_status.self_dependent=true;
	  uimf_status.mdonly = unchanged_incomplete_math_function->mdonly; 
	}
      }
      // new new_rss needs to be added to execfunc->referencing_rss
      // but we're not ready to do this until we're ready to
      // publish this new_rss, so we do this later

    }


    for (auto && changed_math_function: *changed_math_functions) {
      // This math function was itself changed or one or more of its inputs changed
      // create a new math_function_execution
      
      math_function_status &cmf_status = new_rss->mathstatus.function_status.at(changed_math_function);
      cmf_status.execution_demanded=true; // function is changed, so execution is demanded
      bool mdonly = changed_math_function->mdonly;
      if (mdonly) {
	if (passed_mdonly_functions.find(changed_math_function)==passed_mdonly_functions.end()) {
	  // mdonly function did not pass -- something is dependent on it so it can't actually be mdonly
	  mdonly=false;
	}
      }
      cmf_status.mdonly = mdonly;
      std::shared_ptr<globalrevision> new_globalrev = std::dynamic_pointer_cast<globalrevision>(new_rss);
      if (new_globalrev) {
	snde_debug(SNDE_DC_RECMATH,"build_rss_from_functions_and_channels(): creating math_function_execution for globalrev %llu for %s",(unsigned long long)new_globalrev->globalrev,changed_math_function->definition->definition_command.c_str());
      }
      cmf_status.execfunc = std::make_shared<math_function_execution>(new_rss,changed_math_function,mdonly,changed_math_function->is_mutable);
      
      snde_debug(SNDE_DC_RECDB,"make execfunc=0x%llx for %s; new_rss=0x%lx",(unsigned long long)cmf_status.execfunc.get(),changed_math_function->definition->definition_command.c_str(),(unsigned long long )new_rss.get());

      // since execution is mandatory we define the new
      // revision numbers for each of the channels now, so that
      // they are are ordered properly and can execute in parallel

      for (auto && result_channel_path_ptr: changed_math_function->result_channel_paths) {
	
	if (result_channel_path_ptr) {
	  channel_state &state = new_rss->recstatus.channel_map->at(recdb_path_join(changed_math_function->channel_path_context,*result_channel_path_ptr));
	  
	  uint64_t new_revision = ++state._channel->latest_revision; // latest_revision is atomic; correct ordering because a new transaction cannot start until we are done
	  state.end_atomic_revision_update(new_revision);
	}
	
      }
      
    }


    // consider possibly changed math functions. These might be new_revision_optional with changed inputs, dynamic dependencies
    // with unchanged inputs, or functions dependent on the above
    for (auto && possiblychanged_math_function: possiblychanged_math_functions) {
      // This math function could have been changed
      
      math_function_status &pcmf_status = new_rss->mathstatus.function_status.at(possiblychanged_math_function);
      bool mdonly = possiblychanged_math_function->mdonly;
      if (mdonly) {
	if (passed_mdonly_functions.find(possiblychanged_math_function)==passed_mdonly_functions.end()) {
	  // mdonly function did not pass -- something is dependent on it so it can't actually be mdonly
	  mdonly=false;
	}
      }
      pcmf_status.mdonly = mdonly;
      // pcmf_status.execfunc is left as nullptr until such time as we know if we actually need to execute

      // because we don't know if we are even going to try to execute, we mark a (somewhat unnecessary)
      // self-dependency here so that when we go to decide, the prior revision will definitely have an
      // execfunc we can copy. This flag will get us added into the self_dependencies list (below) 
      
      pcmf_status.self_dependent = true; 
      
      
    }
    

    // We don't need to process explicitly updated channels, as we have already ensured
    // that these have new recordings defined.
    
    for (auto && config: *explicitly_updated_channels) {
      definitelychanged_channels_to_process.erase(config); // no further processing needed here
    }
    
  
    //for (auto && definitelychanged_channel: definitelychanged_channels_to_process) {
      // Go through the set of possibly_changed channels which were not processed above
      // (These must be math dependencies))
      // Since the math hasn't started, we don't have defined recordings yet
      // so the recording is just nullptr
      
      //auto cm_it = new_rss->recstatus.channel_map->emplace(std::piecewise_construct,
      //std::forward_as_tuple(possibly_changed_channel->channelpath),
      //							    std::forward_as_tuple(possibly_changed_channel,nullptr,false)).first; // mark updated as false (so far, at least)
    //auto cm_it = new_rss->recstatus.channel_map->find(possibly_changed_channel->channelpath);
    //assert(cm_it != new_rss->recstatus.channel_map->end());
      
      
      //// These recordings are defined but not instantiated
      // new_rss->recstatus.defined_recordings.emplace(std::piecewise_construct,
      //						     std::forward_as_tuple(possibly_changed_channel),
      //						     std::forward_as_tuple(&cm_it->second));      
      
    //}
  

    // definitelychanged channels drive the need for mandatory execution of math functions,
    //mark_prospective_calculations(new_rss,definitelychanged_channels_to_process);

    // mark all changed_math_functions as execution_demanded (now done above)
    //for (auto && inst_ptr : changed_math_functions) {
    //  math_function_status &inst_status = new_rss->mathstatus.functionstatus.at(inst_ptr);
    //  inst_status.execution_demanded=true;
    //}
    // How do we splice implicit and explicit self-dependencies into the calculation graph???
    // Simple: We need to add to the _external_dependencies_on_[channel|function] of the previous_rss and
    // add to the missing_external_[channel|function]_prerequisites of this new_rss.

    // Define variable to store all self-dependencies and fill it up
    std::vector<std::shared_ptr<instantiated_math_function>> self_dependencies;
    for (auto && mathfunction_alldeps: new_rss->mathstatus.math_functions->all_dependencies_of_function) {
      // Enumerate all self-dependencies here
      std::shared_ptr<instantiated_math_function> mathfunction = mathfunction_alldeps.first;
      math_function_status &mf_status = new_rss->mathstatus.function_status.at(mathfunction);
      
      if (!mf_status.complete) { // Complete marked above, if none of the direct or indirect inputs has changed. If complete, nothing else matters. 
	if (mf_status.self_dependent) {
	  // implicit or explicit self-dependency
	  self_dependencies.push_back(mathfunction);
	}
      }
      
    }
    
    
    //  Need to copy repetitive_notifies into place.
    {
      std::lock_guard<std::mutex> recdb_lock(recdb->admin);

      for (auto && repetitive_notify: recdb->repetitive_notifies) {
	// copy notify into this new_rss
	std::shared_ptr<channel_notify> chan_notify = repetitive_notify->create_notify_instance();

	// insert this notify into new_rss data structures.
	/// NOTE: ***!!! This could probably be simplified by leveraging apply_to_rss() method
	// but that would lose flexibility to ignore missing channels
	{
	  std::lock_guard<std::mutex> criteria_lock(chan_notify->criteria.admin);
	  for (auto && mdonly_channame : chan_notify->criteria.metadataonly_channels) {
	    auto channel_map_it = new_rss->recstatus.channel_map->find(mdonly_channame);
	    if (channel_map_it == new_rss->recstatus.channel_map->end()) {
	      std::string other_channels="";
	      for (auto && mdonly_channame2 : chan_notify->criteria.metadataonly_channels) {
		other_channels += mdonly_channame2+";";
	      }
	      snde_warning("MDOnly notification requested on non-existent channel %s; other MDOnly channels=%s. Ignoring.",mdonly_channame,other_channels);
	      continue;
	    }

	    channel_state &chanstate=channel_map_it->second;

	    // Add notification unless criterion already met
	    std::shared_ptr<recording_base> rec_is_complete = chanstate.recording_is_complete(true);

	    if (!rec_is_complete) {
	      // Criterion not met; add notification
	      chanstate._notify_about_this_channel_metadataonly->emplace(chan_notify);
	    }
	    
	  }


	  for (auto && fullyready_channame : chan_notify->criteria.fullyready_channels) {
	    auto channel_map_it = new_rss->recstatus.channel_map->find(fullyready_channame);
	    if (channel_map_it == new_rss->recstatus.channel_map->end()) {
	      std::string other_channels="";
	      for (auto && fullyready_channame2 : chan_notify->criteria.fullyready_channels) {
		other_channels += fullyready_channame2+";";
	      }
	      snde_warning("FullyReady notification requested on non-existent channel %s; other FullyReady channels=%s. Ignoring.",fullyready_channame,other_channels);
	      continue;
	    }
	    
	    channel_state &chanstate=channel_map_it->second;

	    // Add notification unless criterion already met
	    std::shared_ptr<recording_base> rec = chanstate.rec();
	    int rec_state = rec->info_state;
	    
	    if (rec && rec_state==SNDE_RECS_FULLYREADY) {
	      // Criterion met. Nothing to do 
	    } else {
	      // Criterion not met; add notification
	      chanstate._notify_about_this_channel_ready->emplace(chan_notify);
	    }
	    

	    
	  }

	  
	}

	
      }
      
    }
    
    // new_rss->recstatus should have exactly one entry entry in the _recordings maps
    // per channel_map entry
    //bool *all_ready;
    {
      // std::lock_guard<std::mutex> new_rss_admin(new_rss->admin);  // new_rss not yet published so holding lock is unnecessary
      assert(new_rss->recstatus.channel_map->size() == (new_rss->recstatus.defined_recordings.size() + new_rss->recstatus.instantiated_recordings.size() + new_rss->recstatus.metadataonly_recordings.size() + new_rss->recstatus.completed_recordings.size()));

      *all_ready = !new_rss->recstatus.defined_recordings.size() && !new_rss->recstatus.instantiated_recordings.size();
    }


    std::unordered_set<std::shared_ptr<instantiated_math_function>> need_to_check_if_ready;

    
    // Iterate through all non-complete math functions, filling out missing_prerequisites 
    for (auto && mathfunction_alldeps: new_rss->mathstatus.math_functions->all_dependencies_of_function) {
      std::shared_ptr<instantiated_math_function> mathfunction = mathfunction_alldeps.first;

      if (mathfunction->disabled || (mathfunction->ondemand ^ ondemand_only)) { // don't worry about disabled or on-demand functions unless we are in ondemand mode
	continue;
      }
      
      // If mathfunction is unchanged from prior rev, then we don't worry about prerequisites as we will
      // just be copying its value from the prior rev

      if (unchanged_incomplete_math_functions.find(mathfunction) != unchanged_incomplete_math_functions.end() && !mathfunction->self_dependent) {
	  snde_debug(SNDE_DC_RECMATH, "math function %s is unchanged and incomplete", mathfunction->definition->definition_command.c_str());
	  std::shared_ptr<globalrevision> new_globalrev = std::dynamic_pointer_cast<globalrevision>(new_rss);
	  if (new_globalrev) {
	    snde_debug(SNDE_DC_RECMATH, "unchanged and incomplete is for globalrev% llu", (unsigned long long)new_globalrev->globalrev);
	  }
	continue;
      }

      math_function_status &funcstatus = new_rss->mathstatus.function_status.at(mathfunction);
      if (!funcstatus.complete) {
	// This math function is not complete
	
	bool mathfunction_is_mdonly = false;
	if (mathfunction->mdonly) { //// to genuinely be mdonly our instantiated_math_function must be marked as such AND we should be in the mdonly_pending_functions set of the math_status (now there is a specific flag)
	  // mathfunction_is_mdonly = (new_rss->mathstatus.mdonly_pending_functions.find(mathfunction) != new_rss->mathstatus.mdonly_pending_functions.end());
	  mathfunction_is_mdonly = funcstatus.mdonly;
	}
	// iterate over all parameters that are dependent on other channels
	for (auto && parameter: mathfunction->parameters) {

	  // get the set of prerequisite channels
	  std::set<std::string> prereq_channels = parameter->get_prerequisites(/*new_rss,*/mathfunction->channel_path_context);
	  for (auto && prereq_channel: prereq_channels) {

	    // for each prerequisite channel, look at it's state.
	    snde_debug(SNDE_DC_RECDB,"chan_map_size=%d",(int)new_rss->recstatus.channel_map->size());
	    //snde_debug(SNDE_DC_RECDB,"prereq_channel=\"%s\"",prereq_channel.c_str());
	    //snde_debug(SNDE_DC_RECDB,"chan_map_begin()=\"%s\"",new_rss->recstatus.channel_map->begin()->first.c_str());
	    //snde_debug(SNDE_DC_RECDB,"chan_map_2nd=\"%s\"",(++new_rss->recstatus.channel_map->begin())->first.c_str());

	    std::string prereq_channel_fullpath=recdb_path_join(mathfunction->channel_path_context,prereq_channel);
	    

	    auto prereq_chan_it = new_rss->recstatus.channel_map->find(prereq_channel_fullpath);
	    if (prereq_chan_it==new_rss->recstatus.channel_map->end()) {
	      std::string result_name="(nullptr)";

	      std::shared_ptr<std::string> result_name_ptr = mathfunction->result_channel_paths.at(0);
	      if (result_name_ptr) {
		result_name = recdb_path_join(mathfunction->channel_path_context,*result_name_ptr);;
	      }
	      throw snde_error("Prerequisite channel %s of math function %s does not exist",prereq_channel_fullpath.c_str(),result_name.c_str());
	      
	    }
	    
	    channel_state &prereq_chanstate = prereq_chan_it->second;
	     
	    bool prereq_complete = false; 
	    
	    std::shared_ptr<recording_base> prereq_rec = prereq_chanstate.rec();
	    int prereq_rec_state = SNDE_RECS_INITIALIZING;

	    if (prereq_rec) {
	      prereq_rec_state = prereq_rec->info_state;
	    }
	    
	    if (prereq_chanstate.config->math && !(ondemand_only ^ prereq_chanstate.config->ondemand)) {
	      // for math channels, ondemand math channels iff ondemand_only is set, non-ondemand channels iff ondemand_only is not set
	      std::shared_ptr<instantiated_math_function> math_prereq = new_rss->mathstatus.math_functions->defined_math_functions.at(prereq_channel_fullpath);

	      if (mathfunction_is_mdonly && math_prereq->mdonly && prereq_rec && (prereq_rec_state & SNDE_RECF_ALLMETADATAREADY)) {
		// If we are mdonly, and the prereq is mdonly and the recording exists and is metadataready or fully ready,
		// then this prerequisite is complete and nothing is needed.
		
		prereq_complete = true; 
	      }
	    }
	    if (prereq_rec && prereq_rec_state == SNDE_RECS_FULLYREADY) {
	      // If the recording exists and is fully ready,
	      // then this prerequisite is complete and nothing is needed.
	      prereq_complete = true; 
	    }
	    if (!prereq_complete) {
	      // Prerequisite is not complete; Need to mark this in the missing_prerequisites of the math_function_status
	      new_rss->mathstatus.function_status.at(mathfunction).missing_prerequisites.emplace(prereq_chanstate.config);
	    }
	    if (prereq_complete) {
	      bool prereq_modified = false; 
	      std::shared_ptr<channelconfig> prereq_config = all_channels_by_name.at(prereq_channel_fullpath);

	      if (possiblychanged_channels_to_process.find(prereq_config) != possiblychanged_channels_to_process.end()) {
		throw snde_error("end_transaction(): possiblychanged channel turns out to be complete");
	      } else if (definitelychanged_channels_to_process.find(prereq_config) != definitelychanged_channels_to_process.end() || explicitly_updated_channels->find(prereq_config) != explicitly_updated_channels->end()) {
		prereq_modified = true; 
	      } else if (unchanged_channels->find(prereq_config) == unchanged_channels->end()) {
		throw snde_error("end_transaction(): complete prereq channel %s not in definitelychanged or or explicitly_updated or unchanged.",prereq_channel_fullpath.c_str());		
	      }

	      if (prereq_modified) {
		// Should not count self-dependencies here (we don't)
		funcstatus.num_modified_prerequisites++; // this is a complete prerequisite that is modified.
		need_to_check_if_ready.emplace(mathfunction);
	      }
	    }
	    
	    
	  }
	}
      }
    }
    
    // add self-dependencies to the missing_external_prerequisites of this new_rss
    // ***!!!! really we only know that we are actually dependent on our previous self
    // once we know that at least one of our parameters has actually changed... Which might not be
    // decided yet. This list should probably be possible_self_dependencies !!!***
    if (previous_rss) { // (at least if there was a previous_rss to be dependent on)
      std::lock_guard<std::mutex> previous_rss_admin(previous_rss->admin);
      for (auto && self_dep : self_dependencies) {
	// self_dep is an instantiated_math_function
	auto fs_it = previous_rss->mathstatus.function_status.find(self_dep);
	if (fs_it != previous_rss->mathstatus.function_status.end()) {
	  if (!fs_it->second.complete) {
	    new_rss->mathstatus.function_status.at(self_dep).missing_external_function_prerequisites.emplace(std::make_tuple(previous_rss,self_dep));
	  }
	}
      }
    }

    
    // Note from hereon we have published our new new_rss so we have to be a bit more careful about
    // locking it because we might get notification callbacks or similar if one of those external recordings becomes ready

    // Everything in missing_external_function_prerequisites also needs to be in the
    // external rss's external_dependencies_on_function
    if (previous_rss) { // (at least if there was a previous_rss to be dependent on) 
      std::lock_guard<std::mutex> previous_rss_admin(previous_rss->admin);
      
      std::shared_ptr<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> new_previous_rss_external_dependencies_on_function = previous_rss->mathstatus.begin_atomic_external_dependencies_on_function_update();
      for (auto && self_dep : self_dependencies) {
	auto extdep_it = new_previous_rss_external_dependencies_on_function->find(self_dep);
	if (extdep_it == new_previous_rss_external_dependencies_on_function->end()) {
	  new_previous_rss_external_dependencies_on_function->emplace(self_dep,
										  std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>());
	}
	std::shared_ptr<globalrevision> previous_globalrev = std::dynamic_pointer_cast<globalrevision>(previous_rss);
	std::shared_ptr<globalrevision> new_globalrev = std::dynamic_pointer_cast<globalrevision>(new_rss);
	if (previous_globalrev) {
	  snde_debug(SNDE_DC_RECMATH,"globalrev %d adding %s (0x%llx) in rev %d as dependent",
		     (unsigned long long)previous_globalrev->globalrev,
		     self_dep->definition->definition_command.c_str(),
		     (unsigned long long)self_dep.get(),
		     new_globalrev->globalrev);
	}
	new_previous_rss_external_dependencies_on_function->at(self_dep).push_back(std::make_tuple(new_rss,self_dep));
      }
      previous_rss->mathstatus.end_atomic_external_dependencies_on_function_update(new_previous_rss_external_dependencies_on_function);
    }
    
    // Go through all our math functions and make sure we are
    // in the referencing_rss set of their execfunc's
    
    // iterate through all math functions, putting us in their referencing_rss
    {
      std::lock_guard<std::mutex> new_rss_admin(new_rss->admin);
      for (auto && inst_ptr_dep_set: new_rss->mathstatus.math_functions->all_dependencies_of_function) {

	std::shared_ptr<math_function_execution> execfunc = new_rss->mathstatus.function_status.at(inst_ptr_dep_set.first).execfunc;

	if (execfunc) {
	  std::set<std::weak_ptr<recording_set_state>,std::owner_less<std::weak_ptr<recording_set_state>>>::iterator junk_it;
	  bool did_emplace=false;
	    
	  std::lock_guard<std::mutex> ef_admin(execfunc->admin);
	  std::tie(junk_it,did_emplace) = execfunc->referencing_rss.emplace(new_rss);

	  if (did_emplace) {
	    // it's possible that this math function has already completed since we checked above. So add it to need_to_check_if_ready
	    need_to_check_if_ready.emplace(inst_ptr_dep_set.first);
	    
	  }
	}
      }
    }



    // !!!*** Should go through all functions again (maybe after rest of notifies set up?) and check status and use refactored code from math_status::notify_math_function_executed() to detect function completion. ***!!!
    
    // add self-dependencies to the _external_dependencies_on_function of the previous_rss
    //std::vector<std::shared_ptr<instantiated_math_function>> need_to_check_if_ready;

    for (auto && changed_math_function: *changed_math_functions) {
      need_to_check_if_ready.emplace(changed_math_function);
    }
    

    
    if (previous_rss) {  
      std::lock_guard<std::mutex> previous_rss_admin(previous_rss->admin);

      std::shared_ptr<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> new_prevglob_extdep = previous_rss->mathstatus.begin_atomic_external_dependencies_on_function_update();

      std::set<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>> completed_self_deps_to_remove_from_missing_external_function_prerequisites;
      
      for (auto && self_dep : self_dependencies) {
	// Check to make sure previous new_rss even has this exact math function.
	// if so it should be a key in the in the mathstatus.math_functions->all_dependencies_of_function unordered_map
	if (previous_rss->mathstatus.math_functions->all_dependencies_of_function.find(self_dep) != previous_rss->mathstatus.math_functions->all_dependencies_of_function.end()) {
	  // Got it.
	  // Check the status -- may have changed since the above.
	  if (previous_rss->mathstatus.function_status.at(self_dep).complete) {
	    
	    // complete -- perform ready check on this recording
	    need_to_check_if_ready.emplace(self_dep); // should already be there
	    completed_self_deps_to_remove_from_missing_external_function_prerequisites.emplace(std::make_tuple(previous_rss,self_dep));
	  } else {
	    //add to previous new_rss's _external_dependencies
	    new_prevglob_extdep->at(self_dep).push_back(std::make_tuple(new_rss,self_dep));
	  }
	} else {
	  // previous RSS doesn't have this function

	  need_to_check_if_ready.emplace(self_dep); // should already be there
	  completed_self_deps_to_remove_from_missing_external_function_prerequisites.emplace(std::make_tuple(previous_rss,self_dep));
	  
	  
	}
      }
      previous_rss->mathstatus.end_atomic_external_dependencies_on_function_update(new_prevglob_extdep);

      for (auto && prev_rss_completed_self_dep: completed_self_deps_to_remove_from_missing_external_function_prerequisites) {

	std::weak_ptr<recording_set_state> prev_rss_weak = std::get<0>(prev_rss_completed_self_dep);
	std::shared_ptr<recording_set_state> prev_rss = prev_rss_weak.lock();

	if (prev_rss) {
	  std::shared_ptr<instantiated_math_function> self_dep = std::get<1>(prev_rss_completed_self_dep);
	  std::lock_guard<std::mutex> new_rss_admin(new_rss->admin);
	  math_function_status &mf_status = new_rss->mathstatus.function_status.at(self_dep);
	  
	  // erase it from missing_external_function_prerequisites since it
	  // isn't really a prerequisite if it doesn't exist
	  mf_status.missing_external_function_prerequisites.erase(std::make_tuple(prev_rss,self_dep));
	}
      }
      
      
    } 
    

    for (auto && readycheck : need_to_check_if_ready) {
      std::unique_lock<std::mutex> new_rss_admin(new_rss->admin);
      math_function_status &readycheck_status = new_rss->mathstatus.function_status.at(readycheck);

      new_rss->mathstatus.check_dep_fcn_ready(recdb,new_rss,readycheck,&readycheck_status,*ready_to_execute,*may_need_completion_notification,new_rss_admin);
      
    }
    
    // Go through unchanged_incomplete_channels and unchanged_incomplete_mdonly_channels and get notifies when these become complete

    // Create a shadow pointer of new_rss if new_rss is actually a globalrev
    
    std::shared_ptr<globalrevision> new_globalrev = std::dynamic_pointer_cast<globalrevision>(new_rss);
    std::shared_ptr<globalrevision> prev_globalrev = std::dynamic_pointer_cast<globalrevision>(previous_rss);

    if (new_globalrev) {
      for (auto && chanstate: unchanged_incomplete_channels) { // chanstate is a channel_state &
	channel_state &previous_state = previous_rss->recstatus.channel_map->at(chanstate->config->channelpath);
	std::shared_ptr<_unchanged_channel_notify> unchangednotify=std::make_shared<_unchanged_channel_notify>(recdb,prev_globalrev,new_globalrev,previous_state,*chanstate,false);
	unchangednotify->apply_to_rss(previous_rss); 
	/*this code replaced by apply_to_rss()
	std::unique_lock<std::mutex> previous_rss_admin(previous_rss->admin);

	// queue up notification
	std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_set = previous_state.begin_atomic_notify_about_this_channel_ready_update();
	notify_set->emplace(unchangednotify);
	previous_state.end_atomic_notify_about_this_channel_ready_update(notify_set);
	
	unchangednotify->check_all_criteria(new_globalrev);
	*/
      }

      for (auto && chanstate: unchanged_incomplete_mdonly_channels) { // chanstate is a channel_state &
	channel_state &previous_state = previous_rss->recstatus.channel_map->at(chanstate->config->channelpath);
	std::shared_ptr<_unchanged_channel_notify> unchangednotify=std::make_shared<_unchanged_channel_notify>(recdb,prev_globalrev,new_globalrev,previous_state,*chanstate,true);
	unchangednotify->apply_to_rss(previous_rss); 

	/* this code replaced by apply_to_rss()
	std::unique_lock<std::mutex> previous_rss_admin(previous_rss->admin);
	
	// queue up notification
	std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_set = previous_state.begin_atomic_notify_about_this_channel_metadataonly_update();
	notify_set->emplace(unchangednotify);
	previous_state.end_atomic_notify_about_this_channel_metadataonly_update(notify_set);

	unchangednotify->check_all_criteria(new_globalrev);
	*/
	
      }

    }
    
  }
  
  
  std::shared_ptr<globalrevision> active_transaction::end_transaction()
  // Warning: we may be called by the active_transaction destructor, so calling e.g. virtual methods on the active transaction
  // should be avoided.
  // Caller must ensure that all updating processes related to the transaction are complete. Therefore we don't have to worry about locking the current_transaction
  // ***!!!! Much of this needs to be refactored because it is mostly math-based and applicable to on-demand recording groups.
  // (Maybe not; this just puts things into the channel_map so far, But we should factor out code
  // that would define a channel_map and status hash tables needed for an on_demand group)
  {

    std::shared_ptr<recdatabase> recdb_strong=recdb.lock();

    if (!recdb_strong) {
      previous_globalrev=nullptr;
      return nullptr;
    }
    assert(recdb_strong->current_transaction);

    if (transaction_ended) {
      throw snde_error("A single transaction cannot be ended twice");
    }


    std::shared_ptr<globalrevision> globalrev;
    std::map<std::string,std::shared_ptr<channelconfig>> all_channels_by_name;

    // set of channels definitely changed, according to whether we've dispatched them in our graph search
    // for possibly dependent channels 
    std::unordered_set<std::shared_ptr<channelconfig>> changed_channels_need_dispatch;
    std::unordered_set<std::shared_ptr<channelconfig>> changed_channels_dispatched;

    // archive of definitively changed channels
    //std::unordered_set<std::shared_ptr<channelconfig>> definitively_changed_channels;

    // set of channels not yet known to be changed
    std::unordered_set<std::shared_ptr<channelconfig>> unknownchanged_channels;

    
    // set of math functions not known to be changed or unchanged
    std::unordered_set<std::shared_ptr<instantiated_math_function>> unknownchanged_math_functions; 
    
    // set of math functions known to be (definitely) changed
    std::unordered_set<std::shared_ptr<instantiated_math_function>> changed_math_functions; 



    
    {
      std::lock_guard<std::mutex> recdb_lock(recdb_strong->admin);
      // assemble the channel_map
      std::map<std::string,channel_state> initial_channel_map;
      for (auto && channel_name_chan_ptr : recdb_strong->_channels) {
	if (!channel_name_chan_ptr.second->deleted) {
	  initial_channel_map.emplace(std::piecewise_construct,
				      std::forward_as_tuple(channel_name_chan_ptr.first),
				      std::forward_as_tuple(channel_name_chan_ptr.second,channel_name_chan_ptr.second->config(),nullptr,false)); // tentatively mark channel_state as not updated
	}
      }
      
      // build a class globalrevision from recdb->current_transaction using this new channel_map
      globalrev = std::make_shared<globalrevision>(recdb_strong->current_transaction->globalrev,recdb_strong->current_transaction,recdb_strong,recdb_strong->_instantiated_functions,initial_channel_map,previous_globalrev,recdb_strong->current_transaction->rss_unique_index);
      //globalrev->recstatus.channel_map->reserve(recdb_strong->_channels.size());

      globalrev->mutable_recordings_need_holder=std::make_shared<globalrev_mutable_lock>(recdb_strong,globalrev);

    }
  
    // initially all recordings in this globalrev are "defined"
    auto channel_map_ptr = globalrev->recstatus.channel_map;
    for (auto && channame_chanstate: *channel_map_ptr) {
      globalrev->recstatus.defined_recordings.emplace(std::piecewise_construct,
						      std::forward_as_tuple(channame_chanstate.second.config),
						      std::forward_as_tuple(&channame_chanstate.second));      
    }
    
    
      
    
    // Build a temporary map of all channels 
    for (auto && channelname_channelstate: *channel_map_ptr) {
      std::shared_ptr<channelconfig> config = channelname_channelstate.second.config;
      all_channels_by_name.emplace(std::piecewise_construct,
				   std::forward_as_tuple(config->channelpath),
				   std::forward_as_tuple(config));
      
      unknownchanged_channels.emplace(config);
	
    }
    
  
    // Build a temporary map of all functions
    for (auto && instfcnptr_depset: globalrev->mathstatus.math_functions->all_dependencies_of_function) {
      unknownchanged_math_functions.emplace(instfcnptr_depset.first);
    }


    // make sure hash tables won't rehash and screw up iterators or similar
    changed_channels_dispatched.reserve(unknownchanged_channels.size());
    changed_channels_need_dispatch.reserve(unknownchanged_channels.size());

    // mark all new channels/recordings in changed_channels_need_dispatched and remove them from unknownchanged_channels
    for (auto && new_rec_chanpath_ptr: recdb_strong->current_transaction->new_recordings) {

      const std::string &chanpath=new_rec_chanpath_ptr.first;
      const std::shared_ptr<recording_base> &rec = new_rec_chanpath_ptr.second;
      
      std::shared_ptr<channelconfig> config = all_channels_by_name.at(chanpath);
      
      changed_channels_need_dispatch.emplace(config);
      //definitively_changed_channels.emplace(config);
      unknownchanged_channels.erase(config);

      //// For new recordings, set the storage manager if not already set
      //if (!rec->storage_manager) {
      //rec->storage_manager = select_storage_manager_for_recording(recdb_strong,chanpath,globalrev);
      //}
      
      
    }


    // Check for math messages -- erase from unknownchanged_channels see 3372/3373
    // also add to new globalrev->math_status
    for (auto mathfcn : recdb_strong->current_transaction->math_messages) {
      changed_math_functions.emplace(mathfcn.first);
      unknownchanged_math_functions.erase(mathfcn.first);
    }
    globalrev->mathstatus.math_messages = recdb_strong->current_transaction->math_messages;




    for (auto && updated_chan: recdb_strong->current_transaction->updated_channels) {
      std::shared_ptr<channelconfig> config = updated_chan->config();

      std::unordered_set<std::shared_ptr<channelconfig>>::iterator mcc_it = unknownchanged_channels.find(config);

      if (mcc_it != unknownchanged_channels.end()) {
	changed_channels_need_dispatch.emplace(config);
	//definitively_changed_channels.emplace(config);
	unknownchanged_channels.erase(mcc_it);
      }
      if (config && config->math) {
	// updated math channel (presumably with modified function)
	changed_math_functions.emplace(config->math_fcn);
	unknownchanged_math_functions.erase(config->math_fcn);
      }
    }


    std::unordered_set<std::shared_ptr<channelconfig>> explicitly_updated_channels;

    // ensure that recordings that were updated in the transaction, as well as newly-defined channels
    // have a recording define. Keep track of these in explicitly_updated_channels
    // because they won't need dispatch to fill them in.

    // First, if we have an instantiated new recording, place this in the channel_map
    for (auto && new_rec_chanpath_ptr: recdb_strong->current_transaction->new_recordings) {
      std::shared_ptr<channelconfig> config = all_channels_by_name.at(new_rec_chanpath_ptr.first);

      if (config->math) {
	// Not allowed to manually update a math channel
	throw snde_error("Manual instantiation of math channel %s",config->channelpath.c_str());
      }
      auto cm_it = globalrev->recstatus.channel_map->find(new_rec_chanpath_ptr.first);
      assert(cm_it != globalrev->recstatus.channel_map->end());
      //cm_it->second._rec = new_rec_chanpath_ptr.second;
      cm_it->second.end_atomic_rec_update(new_rec_chanpath_ptr.second);

      // assign the .revision field (now implicit)
      //cm_it->second.revision = std::make_shared<uint64_t>(new_rec_chanpath_ptr.second->info->revision);
      cm_it->second.updated=true; 
      
      //
      //						 emplace(std::piecewise_construct,
      //						    std::forward_as_tuple(new_rec_chanpath_ptr.first),
      //						    std::forward_as_tuple(config,new_rec_chanpath_ptr.second,true)).first; // mark updated=true
      
      // mark it as instantiated
      globalrev->recstatus.defined_recordings.erase(config);
      globalrev->recstatus.instantiated_recordings.emplace(std::piecewise_construct,
						std::forward_as_tuple(config),
						std::forward_as_tuple(&cm_it->second));
      explicitly_updated_channels.emplace(config);
    }
    

    // Second, make sure if a channel was created, it has a recording present and gets put in the channel_map
    for (auto && updated_chan: recdb_strong->current_transaction->updated_channels) {

      if (!updated_chan->deleted) {
	std::shared_ptr<channelconfig> config = updated_chan->config();
	//std::shared_ptr<multi_ndarray_recording> new_rec;
	//std::shared_ptr<ndarray_recording_ref> new_rec_ref;
	std::shared_ptr<null_recording> new_rec;
	
	if (config->math) {
	  continue; // math channels get their recordings defined automatically
	}
	
	if (explicitly_updated_channels.find(config) != explicitly_updated_channels.end()) {
	  // already processed above because an explicit new recording was provided. All done here.
	  continue;
	}
      
	auto new_recording_it = recdb_strong->current_transaction->new_recordings.find(config->channelpath);
	
	// new recording should be required but not present; create one
	assert(recdb_strong->current_transaction->new_recording_required.at(config->channelpath) && new_recording_it==recdb_strong->current_transaction->new_recordings.end());
	//new_rec = std::make_shared<recording_base>(recdb_strong,updated_chan,config->owner_id,SNDE_RTN_FLOAT32); // constructor adds itself to current transaction
	//new_rec_ref = create_ndarray_ref(recdb_strong,updated_chan,config->owner_id,SNDE_RTN_FLOAT32);
	new_rec = std::make_shared<null_recording>(recdb_strong,select_storage_manager_for_recording_during_transaction(recdb_strong,config->channelpath.c_str()),recdb_strong->current_transaction,config->channelpath,globalrev,++updated_chan->latest_revision);
	//recordings_needing_finalization.emplace(new_rec); // Since we provided this, we need to make it ready, below
	
	// insert new recording into channel_map
	//auto cm_it = globalrev->recstatus.channel_map->emplace(std::piecewise_construct,
	//							    std::forward_as_tuple(config->channelpath),
	//std::forward_as_tuple(config,new_rec,true)).first; // mark updated=true
	
	auto cm_it = globalrev->recstatus.channel_map->find(config->channelpath);
	assert(cm_it != globalrev->recstatus.channel_map->end());
	//cm_it->second._rec = new_rec;
	cm_it->second.end_atomic_rec_update(new_rec);

	cm_it->second.updated=true; 
	//cm_it->second.revision = std::make_shared<uint64_t>(new_rec->info->revision); (now implicit)
	
	// assign blank waveform content
	new_rec->metadata = std::make_shared<immutable_metadata>();
	new_rec->mark_metadata_done();
	//new_rec->allocate_storage(0,std::vector<snde_index>());
	new_rec->mark_data_ready();
	
	// mark it as completed
	globalrev->recstatus.defined_recordings.erase(config);
	globalrev->recstatus.completed_recordings.emplace(std::piecewise_construct,
							  std::forward_as_tuple(config),
							  std::forward_as_tuple(&cm_it->second));
	
	// mark this as explictly_updated so that it will be removed  from channels_to_process, as it has been already been inserted into the channel_map
	explicitly_updated_channels.emplace(config);
      }
    }
    
    // set of ready channels
    std::unordered_set<channel_state *> ready_channels; // references into the new_rss->recstatus.channel_map

    std::vector<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>> ready_to_execute;
    std::set<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>>,mncn_lessthan> may_need_completion_notification;
    bool all_ready=false;

    std::unordered_set<std::shared_ptr<channelconfig>> unchanged_channels;
    
    build_rss_from_functions_and_channels(recdb_strong,
					  previous_globalrev,
					  globalrev,
					  all_channels_by_name,
					  // set of channels definitely changed, according to whether we've dispatched them in our graph search
					  // for possibly dependent channels 
					  &changed_channels_need_dispatch,
					  &changed_channels_dispatched,
					  // set of channels known to be unchanged (pass as empty)
					  &unchanged_channels,
					  // set of channels not yet known to be changed
					  &unknownchanged_channels,
					  // set of math functions not known to be changed or unchanged
					  &unknownchanged_math_functions,
					  // set of math functions known to be (definitely) changed
					  &changed_math_functions,
					  &explicitly_updated_channels,
					  &ready_channels,
					  &ready_to_execute,
					  &may_need_completion_notification,
					  &all_ready,
					  false); // ondemand_only

    
    // ***!!! Need to got through unchanged_incomplete_math_functions and get notifies when these become complete, if necessary (?)
    // Think it should be unnecessary.
    // ***!!!! This stuff doesn't belong in recstore_display_transforms::update()
    // so from here on we just keep parallel code

    std::vector<std::shared_ptr<channel_notify>> transaction_pending_channel_notifies;  // from the pending_channel_notifies of the transaction

    {
      std::lock_guard<std::mutex> recdb_lock(recdb_strong->admin);

      {

	recdb_strong->_globalrevs.emplace(recdb_strong->current_transaction->globalrev,globalrev);


	std::shared_ptr<globalrevision> previous_latest = recdb_strong->latest_defined_globalrev();
	// atomic update of _latest_defined_globalrev
	std::atomic_store(&recdb_strong->_latest_defined_globalrev,globalrev);

	// if the previously _latest_defined globalrev is ready but still has its mutable_recordings_need_holder,
	// we can clear that now, because anyone who needed it should have gotten it by now
	if (previous_latest && previous_latest->ready) {
	  std::lock_guard<std::mutex> previous_latest_lock(previous_latest->admin);
	  
	  if (previous_latest->mutable_recordings_need_holder) {
	    previous_latest->mutable_recordings_need_holder = nullptr;
	  }
	}
	
	
	std::lock_guard<std::mutex> globalrev_lock(globalrev->admin);
	
	{
	  // mark transaction's resulting_globalrev 
	  std::lock_guard<std::mutex> transaction_admin(recdb_strong->current_transaction->admin);
	  recdb_strong->current_transaction->_resulting_globalrevision = globalrev;

	  // extract pending_channel_notifies
	  transaction_pending_channel_notifies = recdb_strong->current_transaction->pending_channel_notifies;
	  recdb_strong->current_transaction->pending_channel_notifies.clear();
	  
	  // While we hold both recdb and our globalrev, and
	  // the transaction locks, we need to
	  // go through the globalrev rss and move any defined_recordings,
	  // instantiated_recordings, (or data-complete metadataonly_recordings)
	  // into their proper slot, because until shortly above the RSS was not
	  // available for notification.
	  // (after this we need to hold the globalrev_lock until after
	  // the current transaction's resulting_globalrevision
	  // gets assigned (below))
	  globalrev->_update_recstatus__rss_admin_transaction_admin_locked();
	

	  
	}
	
	// this transaction isn't current any more
	recdb_strong->current_transaction = nullptr; 
	assert(!transaction_ended);
	transaction_ended=true;
	transaction_lock_holder.unlock();
      }
    }

    // Perform notifies that unchanged copied recordings from prior revs are now ready
    // (and that globalrev is ready if there is nothing pending!)
    for (auto && readychan : ready_channels) { // readychan is a channel_state &
      readychan->issue_nonmath_notifications(globalrev);
    }

    // queue up everything we marked as ready_to_execute
    for (auto && ready_rss_ready_fcn: ready_to_execute) {
      // Need to queue as a pending_computation
      std::shared_ptr<recording_set_state> ready_rss;
      std::shared_ptr<instantiated_math_function> ready_fcn;

      std::tie(ready_rss,ready_fcn) = ready_rss_ready_fcn;
      recdb_strong->compute_resources->queue_computation(recdb_strong,ready_rss,ready_fcn);
    }
    
    // Run any possibly needed completion notifications
    for (auto && complete_rss_complete_execfunc: may_need_completion_notification) {
      std::shared_ptr<recording_set_state> complete_rss;
      std::shared_ptr<math_function_execution> complete_execfunc;

      std::tie(complete_rss,complete_execfunc) = complete_rss_complete_execfunc;
      execution_complete_notify_single_referencing_rss(recdb_strong,complete_execfunc,complete_execfunc->mdonly,true,complete_rss);
    }
  
    // Check if everything is done; issue notification
    if (all_ready) {
      std::unique_lock<std::mutex> rss_admin(globalrev->admin);
      std::unordered_set<std::shared_ptr<channel_notify>> recordingset_complete_notifiers=std::move(globalrev->recordingset_complete_notifiers);
      globalrev->recordingset_complete_notifiers.clear();
      rss_admin.unlock();

      for (auto && channel_notify_ptr: recordingset_complete_notifiers) {
	channel_notify_ptr->notify_recordingset_complete();
      }
    }

    // go through the pending_channel_notifies from the transaction
    for (auto && pending_channel_notify:    transaction_pending_channel_notifies) {
      pending_channel_notify->apply_to_rss(globalrev);
    }

    //// get notified so as to remove entries from available_compute_resource_database blocked_list
    //// once the previous globalrev is complete.
    //if (previous_globalrev) {
    //  std::shared_ptr<_previous_globalrev_nolongerneeded_notify> prev_nolongerneeded_notify = std::make_shared<_previous_globalrev_nolongerneeded_notify>(recdb,previous_globalrev,globalrev);
    //
    //  prev_nolongerneeded_notify->apply_to_rss(previous_globalrev);
    //}

    // Set up notification when this globalrev is complete
    // So that we can remove obsolete entries from the _globalrevs
    // database and so we can notify anyone monitoring
    // that there is a ready globalrev.
    
    std::shared_ptr<_globalrev_complete_notify> complete_notify=std::make_shared<_globalrev_complete_notify>(recdb,globalrev);
    
    snde_debug(SNDE_DC_NOTIFY,"creating _globalrev_complete_notify 0x%llx for globalrev %llu (0x%llx)",(unsigned long long)complete_notify.get(),(unsigned long long)globalrev->globalrev,(unsigned long long)globalrev.get());
    complete_notify->apply_to_rss(globalrev);
    snde_debug(SNDE_DC_NOTIFY,"_globalrev_complete_notify 0x%llx applies to globalrev %llu (0x%llx)",(unsigned long long)complete_notify.get(),(unsigned long long)globalrev->globalrev,(unsigned long long)globalrev.get());

    // clear out datastructure
    previous_globalrev = nullptr;
    
    
    return globalrev;
  }

  std::shared_ptr<transaction> active_transaction::run_in_background_and_end_transaction(std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<void> params)> fcn,std::shared_ptr<void> params)
  {
    std::shared_ptr<recdatabase> recdb_strong = recdb.lock();
    if (!recdb_strong) {
      return nullptr;
    }

    std::shared_ptr<transaction> retval = recdb_strong->current_transaction;
    
    std::lock_guard<std::mutex> transaction_background_end_lockholder(recdb_strong->transaction_background_end_lock);

    assert(!recdb_strong->transaction_background_end_fcn);  // should not set two background end functions simultaneously
    recdb_strong->transaction_background_end_fcn = fcn;
    recdb_strong->transaction_background_end_params = params;
    recdb_strong->transaction_background_end_acttrans = shared_from_this();

    recdb_strong->transaction_background_end_condition.notify_all();

    return retval;
  }

  
  active_transaction::~active_transaction()
  {
    //recdb->_transaction_raii_holder=nullptr;
    if (!transaction_ended) {
      end_transaction();
    }
  }


  channelconfig::channelconfig(std::string channelpath, std::string owner_name, void *owner_id,bool hidden,std::shared_ptr<recording_storage_manager> storage_manager) : // storage_manager parameter defaults to nullptr
    channelpath(channelpath),
    owner_name(owner_name),
    owner_id(owner_id),
    hidden(hidden),
    math(false),
    storage_manager(storage_manager),
    math_fcn(nullptr),
    mathintermediate(false),
    ondemand(false),
    data_requestonly(false),
    data_mutable(false)
  {

  }

  channel::channel(std::shared_ptr<channelconfig> initial_config) :
    _config(nullptr),
    latest_revision(0),
    deleted(false)
  {
    std::atomic_store(&_config,initial_config);
    
  }
  
  std::shared_ptr<channelconfig> channel::config()
  {
    return std::atomic_load(&_config);
  }

  
  void channel::end_atomic_config_update(std::shared_ptr<channelconfig> new_config)
  {
    // admin must be locked when calling this function. It accepts the modified copy of the atomically guarded data
    std::atomic_store(&_config,new_config);
  }

    
  channel_state::channel_state(std::shared_ptr<channel> chan,std::shared_ptr<channelconfig> config,std::shared_ptr<recording_base> rec_param,bool updated) :
    config(config),
    _channel(chan),
    updated(updated)
  {
    // warning/note: recdb may be locked when this constructor is called. (called within end_transaction to create a prototype that is later copied into the globalrevision structure. 
    //std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_nullptr;
    std::shared_ptr<uint64_t> null_revision_ptr;
    std::atomic_store(&_revision,null_revision_ptr);
    std::atomic_store(&_rec,rec_param);
    std::atomic_store(&_notify_about_this_channel_metadataonly,std::make_shared<std::unordered_set<std::shared_ptr<channel_notify>>>());
    std::atomic_store(&_notify_about_this_channel_ready,std::make_shared<std::unordered_set<std::shared_ptr<channel_notify>>>());
  }
  
  channel_state::channel_state(const channel_state &orig) :
    config(orig.config),
    _channel(orig._channel),
    updated((bool)orig.updated)
    // copy constructor used for initializing channel_map from prototype defined in end_transaction()
  {
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_nullptr;
    std::atomic_store(&_revision,orig.revision());
 
    //assert(!orig._notify_about_this_channel_metadataonly);
    //assert(!orig._notify_about_this_channel_ready);
    
    std::atomic_store(&_rec,orig.rec());
    std::atomic_store(&_notify_about_this_channel_metadataonly,std::make_shared<std::unordered_set<std::shared_ptr<channel_notify>>>(*orig._notify_about_this_channel_metadataonly));
    std::atomic_store(&_notify_about_this_channel_ready,std::make_shared<std::unordered_set<std::shared_ptr<channel_notify>>>(*orig._notify_about_this_channel_ready));
  }

  std::shared_ptr<recording_base> channel_state::rec() const
  {
    return std::atomic_load(&_rec);
  }

  std::shared_ptr<uint64_t> channel_state::revision() const
  {
    return std::atomic_load(&_revision);
  }

  std::shared_ptr<recording_base> channel_state::recording_is_complete(bool mdonly)
  {
    std::shared_ptr<recording_base> retval = rec();
    if (retval) {
      int info_state = retval->info_state;
      if (mdonly) {
	if (info_state & SNDE_RECF_ALLMETADATAREADY) {
	  return retval;
	} else {
	  return nullptr;
	}
      } else {
	if (info_state==SNDE_RECS_FULLYREADY) {
	  return retval;
	} else {
	  return nullptr;
	}	
      }
    }
    return nullptr;
  }

  void channel_state::issue_nonmath_notifications(std::shared_ptr<recording_set_state> rss) // rss is used to lock this channel_state object
  // Must be called without anything locked. Issue notifications requested in _notify* and remove those notification requests,
  // based on the channel_state's current status
  // Note that this might be called more than once for a given situation,
  // but because the notifications it issues are one-shots, it will only
  // pass on the notifications the first time.

  // NOTE: There may be many excess calls to issue_nonmath_notifications,
  // perhaps should troubleshoot this on a performance basis
  {

    // !!!*** This code is largely redundant with recording::mark_data_ready, and the excess should probably be consolidated  ***!!!
    // Note that compared to recording::mark_data_ready this is also used in circumstances
    // where an already completed recording is being assigned to a new
    // recording_set_state

    
    // Issue metadataonly notifications
    if (recording_is_complete(true)) {
      // at least complete through mdonly
      std::unique_lock<std::mutex> rss_admin(rss->admin);
      std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> local_notify_about_this_channel_metadataonly = begin_atomic_notify_about_this_channel_metadataonly_update();
      if (local_notify_about_this_channel_metadataonly) {
	end_atomic_notify_about_this_channel_metadataonly_update(nullptr); // clear out notification list pointer
	rss_admin.unlock();
	
	// perform notifications
	for (auto && channel_notify_ptr: *local_notify_about_this_channel_metadataonly) {
	  channel_notify_ptr->notify_metadataonly(config->channelpath);
	}
      }
    }
    
    // Issue ready notifications
    if (recording_is_complete(false)) {
      
      std::unique_lock<std::mutex> rss_admin(rss->admin);
      std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> local_notify_about_this_channel_ready = begin_atomic_notify_about_this_channel_ready_update();
      if (local_notify_about_this_channel_ready) {
	end_atomic_notify_about_this_channel_ready_update(nullptr); // clear out notification list pointer
	rss_admin.unlock();
	
	// perform notifications
	for (auto && channel_notify_ptr: *local_notify_about_this_channel_ready) {
	  channel_notify_ptr->notify_ready(config->channelpath);
	}
      }
      
    }

    bool all_ready=false;
    {
      std::lock_guard<std::mutex> rss_admin(rss->admin);
      all_ready = !rss->recstatus.defined_recordings.size() && !rss->recstatus.instantiated_recordings.size();      
    }
    snde_debug(SNDE_DC_NOTIFY,"issue_nonmath_notifications: rss=0x%llx; all_ready=%d",(unsigned long long)rss.get(),(int)all_ready);
    
    if (all_ready) {
      std::unordered_set<std::shared_ptr<channel_notify>> rss_complete_notifiers_copy;
      {
	std::lock_guard<std::mutex> rss_admin(rss->admin);
	rss_complete_notifiers_copy = rss->recordingset_complete_notifiers;
      }
      for (auto && notifier: rss_complete_notifiers_copy) {
	notifier->check_recordingset_complete(rss);
      }
    } else {

      /*
      // debugging
      std::shared_ptr<globalrevision> globalrev=std::dynamic_pointer_cast<globalrevision>(rss);
      if (globalrev) {
	std::lock_guard<std::mutex> rss_admin(rss->admin);
	snde_warning("issue_nonmath_notifications: globalrev %llu is not all_ready; defrecsize=%u, irecsize=%u",(unsigned long long)globalrev->globalrev,(unsigned)rss->recstatus.defined_recordings.size(),(unsigned)rss->recstatus.instantiated_recordings.size());

	if (rss->recstatus.defined_recordings.size()==1) {
	  snde_warning("defined recording is %s",rss->recstatus.defined_recordings.begin()->first->channelpath.c_str());
	}
	
      }
      */
	
    }
  }



  static void issue_math_notifications_check_dependent_channel(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> dep_fcn,std::shared_ptr<channelconfig> config,std::shared_ptr<recording_base> new_rec, bool got_mdonly, bool got_fullyready,std::vector<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>> &ready_to_execute,std::set<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>>,mncn_lessthan> &may_need_completion_notification)
  {
    // dep_fcn is a shared_ptr to an instantiated_math_function
    std::unique_lock<std::mutex> rss_admin(rss->admin);
    
    std::set<std::shared_ptr<channelconfig>>::iterator missing_prereq_it;
    math_function_status &dep_fcn_status = rss->mathstatus.function_status.at(dep_fcn);

    //std::shared_ptr<math_function_execution> execfunc = dep_fcn_status.execfunc;
    
    snde_debug(SNDE_DC_RECMATH,"issue_math_notifications_check_dependent_channel %s -> %s",config->channelpath.c_str(),dep_fcn->definition->definition_command.c_str());

    
    // If the dependent function is mdonly then we only have to be mdonly. Otherwise we have to be fullyready
    if (got_fullyready || (got_mdonly && dep_fcn_status.mdonly)) {

      //const recording_status &prior_rss_status = prerequisite_state->recstatus;
      
      bool channel_modified = rss->recstatus.channel_map->at(config->channelpath).updated;;
      //auto prior_cm_it = prerequisite_rss_channel_map->find(config->channelpath);
      //if (prior_cm_it == prerequisite_rss_channel_map->end()) {
      //channel_modified=true;
      //} else {
      //std::shared_ptr<recording_base> prior_recording=prior_cm_it->second.rec();
      //if (new_rec && new_rec != prior_recording) {
      //channel_modified=true;
      //	}
      //}
      
      // Remove us as a missing prerequisite
      missing_prereq_it = dep_fcn_status.missing_prerequisites.find(config);
      if (missing_prereq_it != dep_fcn_status.missing_prerequisites.end()) {
	dep_fcn_status.missing_prerequisites.erase(missing_prereq_it);
	if (channel_modified) {
	  // Don't count self-dependencies here (at least not unless the definition has changed) ... but shouldn't be possible to get this code path on a self-dependency
	  dep_fcn_status.num_modified_prerequisites++;
	}
      }


	
      rss->mathstatus.check_dep_fcn_ready(recdb,rss,dep_fcn,&dep_fcn_status,ready_to_execute,may_need_completion_notification,rss_admin);
      

    }
    
  }
  
  void channel_state::issue_math_notifications(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_set_state> rss) // rss is used to lock this channel_state object
  // Must be called without anything locked. Issue notifications requested in _notify* and remove those notification requests,
  // based on the channel_state's current status
  {

    std::vector<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>> ready_to_execute;
    std::set<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>>,mncn_lessthan> may_need_completion_notification;
    std::shared_ptr<recording_base> new_rec = rec();
    // Issue metadataonly notifications    
    bool got_mdonly = (bool)recording_is_complete(true);
    bool got_fullyready = (bool)recording_is_complete(false);

    snde_debug(SNDE_DC_RECMATH,"issue_math_notifications %s; new_rec=0x%llx, got_mdonly=%d, got_fullyready=%d",config->channelpath.c_str(), (unsigned long long)new_rec.get(),(int)got_mdonly,(int)got_fullyready);

    if (got_mdonly || got_fullyready) {
      // at least complete through mdonly
      
      //snde_warning("issue_math_notifications: adoc.size()=%u channelpath=%s",(unsigned)rss->mathstatus.math_functions->all_dependencies_of_channel.size(),config->channelpath.c_str());
      
      auto dep_it = rss->mathstatus.math_functions->all_dependencies_of_channel.find(config->channelpath);
      if (dep_it != rss->mathstatus.math_functions->all_dependencies_of_channel.end()) {
	for (auto && dep_fcn: dep_it->second) {
    	  // dep_fcn is a shared_ptr to an instantiated_math_function
	  issue_math_notifications_check_dependent_channel(recdb,rss,dep_fcn,config,new_rec,got_mdonly,got_fullyready,ready_to_execute,may_need_completion_notification);
	}
      }

      // consider extra dependencies arising from dynamic dependencies
      std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::shared_ptr<instantiated_math_function>>>> ext_internal_dep = rss->mathstatus.extra_internal_dependencies_on_channel();

      std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::shared_ptr<instantiated_math_function>>>::iterator ext_internal_dep_it = ext_internal_dep->find(config);
      
      if (ext_internal_dep_it != ext_internal_dep->end()) {
	// found extra internal dependencies
	for (auto && rss_extintdepfcn: ext_internal_dep_it->second ) {
	  issue_math_notifications_check_dependent_channel(recdb,rss,rss_extintdepfcn,config,new_rec,got_mdonly,got_fullyready,ready_to_execute,may_need_completion_notification);
	}
      }
      
      std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> ext_dep_on_channel = rss->mathstatus.external_dependencies_on_channel();
      auto extdep_it = ext_dep_on_channel->find(config);

      if (extdep_it != ext_dep_on_channel->end()) {
	for (auto && rss_extdepfcn: extdep_it->second ) {
	  std::weak_ptr<recording_set_state> &dependent_rss_weak = std::get<0>(rss_extdepfcn);
	  std::shared_ptr<recording_set_state> dependent_rss = dependent_rss_weak.lock();
	  std::shared_ptr<instantiated_math_function> &dependent_func = std::get<1>(rss_extdepfcn);

	  if (dependent_rss) {
	    std::unique_lock<std::mutex> dependent_rss_admin(dependent_rss->admin);
	    math_function_status &function_status = dependent_rss->mathstatus.function_status.at(dependent_func);
	    std::set<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<channelconfig>>>::iterator
	      dependent_prereq_it = function_status.missing_external_channel_prerequisites.find(std::make_tuple(rss,config));
	    
	    if (dependent_prereq_it != function_status.missing_external_channel_prerequisites.end()) {
	      function_status.missing_external_channel_prerequisites.erase(dependent_prereq_it);
	      
	    }
	    dependent_rss->mathstatus.check_dep_fcn_ready(recdb,dependent_rss,dependent_func,&function_status,ready_to_execute,may_need_completion_notification,dependent_rss_admin);
	  }
	}
      }
    }


    for (auto && ready_rss_ready_fcn: ready_to_execute) {
      // Need to queue as a pending_computation
      std::shared_ptr<recording_set_state> ready_rss;
      std::shared_ptr<instantiated_math_function> ready_fcn;

      std::tie(ready_rss,ready_fcn) = ready_rss_ready_fcn;
      recdb->compute_resources->queue_computation(recdb,ready_rss,ready_fcn);
    }

    // Run any possibly needed completion notifications
    for (auto && complete_rss_complete_execfunc: may_need_completion_notification) {
      std::shared_ptr<recording_set_state> complete_rss;
      std::shared_ptr<math_function_execution> complete_execfunc;

      std::tie(complete_rss,complete_execfunc) = complete_rss_complete_execfunc;
      execution_complete_notify_single_referencing_rss(recdb,complete_execfunc,complete_execfunc->mdonly,true,complete_rss);
    }
    
  }

  void channel_state::end_atomic_rec_update(std::shared_ptr<recording_base> new_recording)
  {
    std::atomic_store(&_rec,new_recording);

    if (new_recording) {
      end_atomic_revision_update(new_recording->info->revision);
    }
  }

  void channel_state::end_atomic_revision_update(uint64_t new_rev)
  {
    
    std::atomic_store(&_revision,std::make_shared<uint64_t>(new_rev));
    
  }



  std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> channel_state::begin_atomic_notify_about_this_channel_metadataonly_update()
  {
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> orig = notify_about_this_channel_metadataonly();

    if (orig) {
      return std::make_shared<std::unordered_set<std::shared_ptr<channel_notify>>>(*orig);
    } else {
      return nullptr;
    }
  }
  
  void channel_state::end_atomic_notify_about_this_channel_metadataonly_update(std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> newval)
  {
    std::atomic_store(&_notify_about_this_channel_metadataonly,newval);
  }
  
  std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> channel_state::notify_about_this_channel_metadataonly()
  {
    return std::atomic_load(&_notify_about_this_channel_metadataonly);
  }

    
  std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> channel_state::begin_atomic_notify_about_this_channel_ready_update()
  {
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> orig = notify_about_this_channel_ready();
    if (orig) {
      return std::make_shared<std::unordered_set<std::shared_ptr<channel_notify>>>(*orig);
    } else {
      return nullptr;
    }
  }
  
  void channel_state::end_atomic_notify_about_this_channel_ready_update(std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> newval)
  {
    std::atomic_store(&_notify_about_this_channel_ready,newval);
  }
  
  std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> channel_state::notify_about_this_channel_ready()
  {
    return std::atomic_load(&_notify_about_this_channel_ready);
  }


  recording_status::recording_status(const std::map<std::string,channel_state> & channel_map_param) :
    channel_map(std::make_shared<std::map<std::string,channel_state>>(channel_map_param))
  {

  }

  bool recording_set_state::_urratal_check_mdonly(std::string channelpath)
  // internal use only for recording_set_state::_update_recstatus__rss_admin_transaction_admin_locked()
  {
    
    // check for MDonly
    bool mdonly=false;
    std::map<std::string,std::shared_ptr<instantiated_math_function>>::iterator math_function_it;
    math_function_it = mathstatus.math_functions->defined_math_functions.find(channelpath);
    if (math_function_it != mathstatus.math_functions->defined_math_functions.end()) {
      // channel is a math channel (only math channels can be mdonly)
      math_function_status &funcstat = mathstatus.function_status.at(math_function_it->second);
      
      if (funcstat.mdonly) {
	// yes, an mdonly channel... move it from instantiated recordings to metadataonly_recordings
	mdonly=true;
      }
    }

    return mdonly;
  }
  
  void recording_set_state::_update_recstatus__rss_admin_transaction_admin_locked()
  // This is called during end_transaction() with the recdb admin lock, the rss admin lock, and the transaction admin lock held, to identify anything misplaced in the above unordered_maps. After the call and marking of the transaction's _resulting_globalrevision, placement responsibility shifts to mark_metadata_done() and mark_data_ready() methods of the recording. 
  {

    std::unordered_map<std::shared_ptr<channelconfig>,channel_state *>::iterator mdonlyrec_it,mdonlyrec_next;
    
    for (mdonlyrec_it=recstatus.metadataonly_recordings.begin();mdonlyrec_it != recstatus.metadataonly_recordings.end();mdonlyrec_it = mdonlyrec_next) {
      mdonlyrec_next=mdonlyrec_it;
      ++mdonlyrec_next;
      
      std::shared_ptr<channelconfig> config = mdonlyrec_it->first;
      channel_state &state = *mdonlyrec_it->second;
      
      
      std::shared_ptr<recording_base> rec = state.rec();
      assert(rec);
      
      assert(rec->info_state != SNDE_RECS_INITIALIZING); // state cannot progress backwards
      
      if (rec->info_state==SNDE_RECS_FULLYREADY) {
	// recording is ready
	// go into completed recordings
	recstatus.metadataonly_recordings.erase(mdonlyrec_it);
	recstatus.completed_recordings.emplace(config,&state);
      }
      
      
    }

    
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state *>::iterator instrec_it,instrec_next;
    for (instrec_it=recstatus.instantiated_recordings.begin();instrec_it != recstatus.instantiated_recordings.end();instrec_it = instrec_next) {
      instrec_next=instrec_it;
      ++instrec_next;

      std::shared_ptr<channelconfig> config = instrec_it->first;
      channel_state &state = *instrec_it->second;
      
      
      std::shared_ptr<recording_base> rec = state.rec();
      assert(rec);


      if (rec->info_state == SNDE_RECS_FULLYREADY) {
	// recording is ready
	// go into completed recordings
	recstatus.instantiated_recordings.erase(instrec_it);
	recstatus.completed_recordings.emplace(config,&state);
	
      } else if (rec->info_state & SNDE_RECF_ALLMETADATAREADY) {
	  // check for MDonly
	
	bool mdonly = _urratal_check_mdonly(config->channelpath);
	
	if (mdonly) {
	  // if we have metadata and we are mdonly we go into metadataonly_recordings
	  recstatus.instantiated_recordings.erase(instrec_it);
	  recstatus.metadataonly_recordings.emplace(config,&state);
	} 
	
      } 
      
      
    }

    std::unordered_map<std::shared_ptr<channelconfig>,channel_state *>::iterator defrec_it,defrec_next;
    for (defrec_it=recstatus.defined_recordings.begin();defrec_it != recstatus.defined_recordings.end();defrec_it = defrec_next) {
      defrec_next=defrec_it;
      ++defrec_next;

      std::shared_ptr<channelconfig> config = defrec_it->first;
      channel_state &state = *defrec_it->second;

      std::shared_ptr<recording_base> rec = state.rec();
      if (rec) {
	if (rec->info_state == SNDE_RECS_FULLYREADY) {
	  // recording is ready
	  // go into completed recordings
	  recstatus.defined_recordings.erase(defrec_it);
	  recstatus.completed_recordings.emplace(config,&state);

	} else if (rec->info_state & SNDE_RECF_ALLMETADATAREADY) {
	  // check for MDonly
	  bool mdonly = _urratal_check_mdonly(config->channelpath);
	  
	  if (mdonly) {
	    // if we have metadata and we are mdonly we go into metadataonly_recordings
	    recstatus.defined_recordings.erase(defrec_it);
	    recstatus.metadataonly_recordings.emplace(config,&state);
	  } else {
	    // go into instantiated_recordings
	    recstatus.defined_recordings.erase(defrec_it);
	    recstatus.instantiated_recordings.emplace(config,&state);
	  }
	  
	} else {
	  // go into instantiated_recordings
	  recstatus.defined_recordings.erase(defrec_it);
	  recstatus.instantiated_recordings.emplace(config,&state);
	  
	}
	
      }
    }
  }


  std::string recording_set_state::get_math_status()
  {
    std::lock_guard<std::mutex> rss_admin(admin);
    std::string retval="[\n";

    retval += "pending:\n";
    for (auto && inst: mathstatus.pending_functions) {
      retval += "  \"" + inst->definition->definition_command + "\"\n";
    }
    
    retval += "mdonly_pending:\n";
    for (auto && inst: mathstatus.mdonly_pending_functions) {
      retval += "  \"" + inst->definition->definition_command + "\"\n";
    }

    retval += "completed:\n";
    for (auto && inst: mathstatus.completed_functions) {
      retval += "  \"" + inst->definition->definition_command + "\"\n";
    }
    retval += "completed_mdonly:\n";
    for (auto && inst: mathstatus.completed_mdonly_functions) {
      retval += "  \"" + inst->definition->definition_command + "\"\n";
    }
    retval += "]\n";

    return retval;
  }

  std::string recording_set_state::get_math_function_status(std::string definition_command)
  {
    std::lock_guard<std::mutex> rss_admin(admin);
    
    std::string retval;
    bool gotmatch = false;
    
    for (auto && inst_funcstat: mathstatus.function_status) {
      if (inst_funcstat.first->definition->definition_command==definition_command) {
	gotmatch = true;
	retval += definition_command + " [\n";
	const math_function_status &funcstat = inst_funcstat.second;
	retval += "missing_prerequisites:\n";
	for (auto channelconf_ptr: funcstat.missing_prerequisites) {
	  retval+="  \""+channelconf_ptr->channelpath + "\"\n";
	}
	retval += ssprintf("num_modified_prerequisites: %d\n",funcstat.num_modified_prerequisites);
	retval += ssprintf("execfunc=0x%llx\n",(unsigned long long)funcstat.execfunc.get());
	retval += "missing_external_channel_prerequisites:\n";
	for (auto rss_channelconf: funcstat.missing_external_channel_prerequisites) {
	  retval+= ssprintf("  rss: 0x%llx; \"%s\"\n",(unsigned long long)std::get<0>(rss_channelconf).get(),std::get<1>(rss_channelconf)->channelpath.c_str());
	}
	retval += "missing_external_function_prerequisites:\n";
	for (auto rss_inst: funcstat.missing_external_function_prerequisites) {
	  retval+= ssprintf("  rss: 0x%llx; \"%s\"\n",(unsigned long long)std::get<0>(rss_inst).get(),std::get<1>(rss_inst)->definition->definition_command.c_str());
	}
	retval += "mdonly: " + std::to_string(funcstat.mdonly) + "\n";
	retval += "self_dependent: " + std::to_string(funcstat.self_dependent) + "\n";
	retval += "execution_demanded: " + std::to_string(funcstat.execution_demanded) + "\n";
	retval += "ready_to_execute: " + std::to_string(funcstat.ready_to_execute) + "\n";
	retval += "complete: " + std::to_string(funcstat.complete) + "\n";
	
	retval += "]\n";
      }
    }
    if (!gotmatch) {
      retval += "(function not found)\n";
    }
    return retval;
  }
  
  uint64_t rss_get_unique()
  {
    // return a process-wide unique (incrementing) identifier. Process wide
    // so that even if you run multiple recording databases you still won't
    // get a collision. Use the return from this as the unique_index parameter
    // to the recording_set_state constructor

    // remember that c++11 and higher guarantee atomic static initalization

    // By initializing  with a random number (up to 65535: 4 hex digits)
    // we introduce plenty of entropy to prevent name collisions between shared
    // memory objects even if somehow the process ID # is the same
    // (as the shared memory object naming includes both this and the process ID
    // and the revision)

    // This is also used to seed the "starting revision" used by recstore_display_transforms
    static unsigned seed=time(nullptr);
#ifndef _MSC_VER
    static std::atomic<uint64_t> *index=new std::atomic<uint64_t>((uint64_t)(rand_r(&seed)*65535.0/RAND_MAX));
#else
    srand(seed);
    static std::atomic<uint64_t> *index = new std::atomic<uint64_t>((uint64_t)rand() * 65535.0 / RAND_MAX);
#endif

    return (*index)++;
    
  }
  
  recording_set_state::recording_set_state(std::shared_ptr<recdatabase> recdb,const instantiated_math_database &math_functions,const std::map<std::string,channel_state> & channel_map_param,std::shared_ptr<recording_set_state> prereq_state,uint64_t originating_globalrev_index,uint64_t unique_index) :
    originating_globalrev_index(0), 
    unique_index(unique_index), // must be assigned after construction
    recdb_weak(recdb),
    ready(false),
    recstatus(channel_map_param),
    mathstatus(std::make_shared<instantiated_math_database>(math_functions),channel_map_param),
    _prerequisite_state(nullptr),
    lockmgr(recdb->lockmgr)
  {
    // Note: Does not rebuild the mathstatus's dependency map
    std::atomic_store(&_prerequisite_state,prereq_state);
  }


  bool recording_set_state::check_complete()
  {
    std::lock_guard<std::mutex> rss_admin(admin);
      
    // check if all recordings are ready and all math functions are complete
    bool all_ready = !recstatus.defined_recordings.size() && !recstatus.instantiated_recordings.size() && !mathstatus.pending_functions.size() && !mathstatus.mdonly_pending_functions.size();
    return all_ready;
  }
  void recording_set_state::wait_complete()
  {
    std::shared_ptr<promise_channel_notify> promise_notify=std::make_shared<promise_channel_notify>(std::vector<std::string>(),std::vector<std::string>(),true);
    
    promise_notify->apply_to_rss(shared_from_this());

    //std::future<bool> criteria_satisfied = promise_notify->promise.get_future();
    //criteria_satisfied.wait();

    promise_notify->wait_interruptable();
    
  }
  
  std::string recording_set_state::print_math_status(bool verbose/*=false*/)
  {
    return mathstatus.print_math_status(shared_from_this(),verbose);
  }

  
  std::string recording_set_state::print_recording_status(bool verbose/*=false*/)
  {
    std::string print_buf;
    std::lock_guard<std::mutex> rss_admin(admin);
    
    print_buf += ssprintf("\n");
    print_buf += ssprintf("recording status: \n");
    print_buf += ssprintf("=================\n");
    std::shared_ptr<std::map<std::string,channel_state>> chanmap = recstatus.channel_map;
    for (auto && chan_state: *chanmap) {
      std::string chan = chan_state.first;
      channel_state &state = chan_state.second;
      if (verbose || !state.recording_is_complete(false)) {
	print_buf += ssprintf("channel %s\n",chan.c_str());
	print_buf += ssprintf("-------------------------\n");
	print_buf += ssprintf("updated = %d; revision = %llu; recording present = %d\n",(int)state.updated,(unsigned long long)*state.revision(),(int)((bool)state.rec()));
	if (state.rec()) {
	  bool got_stateflag = false;
	  int recstate = state.rec()->info_state;
	  print_buf += ssprintf("state = ");
	  if (recstate & SNDE_RECF_STATICMETADATAREADY) {
	    print_buf += ssprintf("STATICMETADATAREADY ");
	    got_stateflag = true;
	  }
	  if (recstate & SNDE_RECF_DYNAMICMETADATAREADY) {
	    print_buf += ssprintf("DYNAMICMETADATAREADY ");
	    got_stateflag = true;
	  }
	  if (recstate & SNDE_RECF_ALLMETADATAREADY) {
	    print_buf += ssprintf("ALLMETADATAREADY ");
	    got_stateflag = true;
	  }
	  if (recstate & SNDE_RECF_OBSOLETE) {
	    print_buf += ssprintf("OBSOLETE");
	    got_stateflag = true;
	  }
	  if (!got_stateflag) {
	    print_buf += ssprintf("INITIALIZING");
	  }
	  print_buf += ssprintf("\n");
	  
	  
	}
	print_buf += ssprintf("\n");
	 
      }
    }
    print_buf += ssprintf("%u total recordings; %u accounted for\n",(unsigned)chanmap->size(),(unsigned)(recstatus.defined_recordings.size()+recstatus.instantiated_recordings.size()+recstatus.metadataonly_recordings.size()+recstatus.completed_recordings.size()));
    if (recstatus.defined_recordings.size() > 0) {
      print_buf += ssprintf("defined recordings\n");
      print_buf += ssprintf("------------------\n");
      for (auto && chan_chanstate: recstatus.defined_recordings) {
	print_buf += ssprintf("  %s\n",chan_chanstate.first->channelpath.c_str());
      }
    }
    if (recstatus.instantiated_recordings.size() > 0) {
      print_buf += ssprintf("instantiated recordings\n");
      print_buf += ssprintf("-----------------------\n");
      for (auto && chan_chanstate: recstatus.instantiated_recordings) {
	print_buf += ssprintf("  %s\n",chan_chanstate.first->channelpath.c_str());
      }
    }
    if (recstatus.metadataonly_recordings.size() > 0) {
      if (verbose) {
	print_buf += ssprintf("metadataonly recordings\n");
	print_buf += ssprintf("------------------------\n");
	for (auto && chan_chanstate: recstatus.metadataonly_recordings) {
	  print_buf += ssprintf("  %s\n",chan_chanstate.first->channelpath.c_str());
	}
      }
      else {
        print_buf += ssprintf("%u metadataonly recordings\n",(unsigned)recstatus.metadataonly_recordings.size());
      }
    }
    
    if (recstatus.completed_recordings.size() > 0) {
      if (verbose) {
	print_buf += ssprintf("completed recordings\n");
	print_buf += ssprintf("------------------------\n");
	for (auto && chan_chanstate: recstatus.completed_recordings) {
	  print_buf += ssprintf("  %s\n",chan_chanstate.first->channelpath.c_str());
	}
      }
      else {
        print_buf += ssprintf("%u completed recordings\n",(unsigned)recstatus.completed_recordings.size());
      }
    }
    return print_buf;
  }
  
  std::shared_ptr<recording_base> recording_set_state::get_recording(const std::string &fullpath)
  { // safe to call with or without rss locked
    std::map<std::string,channel_state>::iterator cm_it;
      
    cm_it = recstatus.channel_map->find(fullpath);
    if (cm_it == recstatus.channel_map->end()) {
      throw snde_error("get_recording(): channel %s not found.",fullpath.c_str());
    }
    return cm_it->second.rec();
  }


  std::shared_ptr<recording_base> recording_set_state::check_for_recording(const std::string &fullpath)
  {
    std::map<std::string,channel_state>::iterator cm_it;
      
    cm_it = recstatus.channel_map->find(fullpath);
    if (cm_it == recstatus.channel_map->end()) {
      return nullptr;
    }
    return cm_it->second.rec();
  }

  
  std::shared_ptr<ndarray_recording_ref> recording_set_state::get_ndarray_ref(const std::string &fullpath,size_t array_index)
  {
    std::shared_ptr<recording_base> rec = get_recording(fullpath);
    if (!rec) {
      throw snde_error("Recording channel %s is so-far undefined in this globalrevision/rss",fullpath.c_str());
    }
    
    std::shared_ptr<multi_ndarray_recording> rec_ndarray = std::dynamic_pointer_cast<multi_ndarray_recording>(rec);

    if (!rec_ndarray) {
      throw snde_error("Recording channel %s is not a multi_ndarray in this globalrevision/rss",fullpath.c_str());
    }

    return rec_ndarray->reference_ndarray(array_index);
   
  }

  std::shared_ptr<ndarray_recording_ref> recording_set_state::get_ndarray_ref(const std::string &fullpath,std::string array_name)
  {
    std::shared_ptr<recording_base> rec = get_recording(fullpath);
    if (!rec) {
      throw snde_error("Recording channel %s does not exist",fullpath.c_str());
    }

    std::shared_ptr<multi_ndarray_recording> rec_ndarray = std::dynamic_pointer_cast<multi_ndarray_recording>(rec);

    if (!rec_ndarray) {
      throw snde_error("Recording channel %s is not a multi_ndarray",fullpath.c_str());
    }

    auto name_mapping_it = rec_ndarray->name_mapping.find(array_name);
    if (name_mapping_it == rec_ndarray->name_mapping.end()) {
      throw snde_error("Recording multi_ndarray %s does not have an array %s",fullpath.c_str(),array_name.c_str());
    }
    size_t array_index = name_mapping_it->second;
    
    return rec_ndarray->reference_ndarray(array_index);
  }


  std::shared_ptr<ndarray_recording_ref> recording_set_state::check_for_recording_ref(const std::string &fullpath,size_t array_index)
  {
    std::shared_ptr<recording_base> rec = get_recording(fullpath);
    if (!rec) {
      return nullptr;
    }
    std::shared_ptr<multi_ndarray_recording> rec_ndarray = std::dynamic_pointer_cast<multi_ndarray_recording>(rec);

    if (!rec_ndarray) {
      return nullptr;
    }

    return rec_ndarray->reference_ndarray(array_index);
   
  }

  std::shared_ptr<ndarray_recording_ref> recording_set_state::check_for_recording_ref(const std::string &fullpath,std::string array_name)
  {
    std::shared_ptr<recording_base> rec = get_recording(fullpath);
    if (!rec) {
      return nullptr;
    }
    std::shared_ptr<multi_ndarray_recording> rec_ndarray = std::dynamic_pointer_cast<multi_ndarray_recording>(rec);
    
    if (!rec_ndarray) {
      return nullptr;
    }

    size_t array_index = rec_ndarray->name_mapping.at(array_name);
    
    return rec_ndarray->reference_ndarray(array_index);
  }


  std::shared_ptr<std::vector<std::string>> recording_set_state::list_recordings()
  {
    std::shared_ptr<std::vector<std::string>> retval=std::make_shared<std::vector<std::string>>();

    for (auto && channel_map_it: *recstatus.channel_map) {
      retval->push_back(channel_map_it.first);
    }

    return retval;
  }

#ifdef SIZEOF_LONG_IS_8 // this is a SWIG workaround -- see spatialnde2.i
  std::shared_ptr<std::vector<std::pair<std::string,unsigned long>>> recording_set_state::list_recording_revisions()
#else
  std::shared_ptr<std::vector<std::pair<std::string,unsigned long long>>> recording_set_state::list_recording_revisions()
#endif    
  {
    std::shared_ptr<std::vector<std::pair<std::string,uint64_t>>> retval = std::make_shared<std::vector<std::pair<std::string,uint64_t>>>();
    
    for (auto && channel_map_it: *recstatus.channel_map) {
      uint64_t revision=0;
      if (channel_map_it.second.revision()) {
	revision = *channel_map_it.second.revision();
      }
      retval->push_back(std::make_pair(channel_map_it.first,revision));
    }
    return retval;
  }

  std::shared_ptr<std::vector<std::pair<std::string,std::string>>> recording_set_state::list_recording_refs()
    {
      std::shared_ptr<std::vector<std::pair<std::string,std::string>>> retval = std::make_shared<std::vector<std::pair<std::string,std::string>>>();

      for (auto && channel_map_it: *recstatus.channel_map) {
	const std::string &name = channel_map_it.first;
	const channel_state & state = channel_map_it.second;
	std::shared_ptr<recording_base> rec = state.rec();

	if (rec) {
	  std::shared_ptr<multi_ndarray_recording> ndrec=std::dynamic_pointer_cast<multi_ndarray_recording>(rec);
	  if (ndrec) {
	    std::lock_guard<std::mutex> recadmin(ndrec->admin);
	    size_t num_refs = ndrec->layouts.size();
	    if (!num_refs) {
	      continue;
	    }

	    if (num_refs==1 && ndrec->name_reverse_mapping.size()==0) {
	      if (ndrec->name_mapping.size() < 1) {
		snde_warning("ndarray %s does not provide a name mapping",channel_map_it.first.c_str());
		continue;
	      }
	      retval->push_back(std::make_pair(channel_map_it.first,ndrec->name_mapping.begin()->first));
	    } else {
	      size_t refnum;
	      if (ndrec->name_mapping.size() < 1) {
		snde_warning("ndarray %s does not provide a name mapping",channel_map_it.first.c_str());
		continue;
	      }

	      for (refnum=0;refnum < num_refs;refnum++) {
		retval->push_back(std::make_pair(channel_map_it.first,ndrec->name_reverse_mapping.at(refnum)));
		
	      }
	    }
	    
	  }

	}
      }
      
      return retval;
    }


  
  std::shared_ptr<recording_set_state> recording_set_state::prerequisite_state()
  {
    return std::atomic_load(&_prerequisite_state);
  }

  // sets the prerequisite state to nullptr
  void recording_set_state::atomic_prerequisite_state_clear()
  {
    std::shared_ptr<recording_set_state> null_prerequisite;    
    std::atomic_store(&_prerequisite_state,null_prerequisite);

  }

  long recording_set_state::get_reference_count()
  {
    std::weak_ptr<recording_set_state> weak_this = shared_from_this();

    return weak_this.use_count();
  }

  size_t recording_set_state::num_complete_notifiers() // size of recordingset_complete_notifiers; useful for debugging memory leaks
  {
    std::lock_guard<std::mutex> rss_admin(admin);
    return recordingset_complete_notifiers.size();
  }

  globalrev_mutable_lock::globalrev_mutable_lock(std::weak_ptr<recdatabase> recdb,std::weak_ptr<globalrevision> globalrev) :
    recdb(recdb),
    globalrev(globalrev)
  {

  }

  globalrev_mutable_lock::~globalrev_mutable_lock()
  {
    // This destructor being called indicates that
    // the mutable data from this globalrev can now
    // be allowed to change. We place this
    // globalrev in the recdb's globalrev_mutablenotneeded_pending
    // list and trigger the globalrev_mutablenotneeded_condition variable
    std::shared_ptr<recdatabase> recdb_strong=recdb.lock();
    std::shared_ptr<globalrevision> globalrev_strong=globalrev.lock();
    if (recdb_strong && globalrev_strong) {
      globalrev_strong->mutable_recordings_still_needed=false;
      
      std::lock_guard<std::mutex> globalrev_mutablenotneeded_lockholder(recdb_strong->globalrev_mutablenotneeded_lock);

      //snde_warning("Pushing globalrev %llu onto globalrev_mutablenotneeded_pending",(unsigned long long)globalrev_strong->globalrev);
      
      recdb_strong->globalrev_mutablenotneeded_pending.push_back(globalrev_strong);
      recdb_strong->globalrev_mutablenotneeded_condition.notify_all();
      
      
    }
  }
  
  globalrevision::globalrevision(uint64_t globalrev, std::shared_ptr<transaction> defining_transact, std::shared_ptr<recdatabase> recdb,const instantiated_math_database &math_functions,const std::map<std::string,channel_state> & channel_map_param,std::shared_ptr<recording_set_state> prereq_state,uint64_t unique_index) :
    recording_set_state(recdb,math_functions,channel_map_param,prereq_state,globalrev,unique_index),
    defining_transact(defining_transact),
    globalrev(globalrev),
    mutable_recordings_still_needed(true)
  {
    assert(globalrev<=1 || prereq_state); // prereq_state must be defined unless we are starting out at the very beginning
  }


  recdatabase::recdatabase(std::shared_ptr<lockmanager> lockmgr/*=nullptr*/):
    alignment_requirements(std::make_shared<allocator_alignment>()),
    compute_resources(std::make_shared<available_compute_resource_database>()),
    default_storage_manager(nullptr),
    lockmgr(lockmgr),
    started(false),
    transaction_background_end_mustexit(false),
    monitoring_notify_globalrev(0),
    globalrev_mutablenotneeded_mustexit(false)

    // !!!*** Please note alternate constructor below
  {
    std::shared_ptr<globalrevision> null_globalrev;
    std::atomic_store(&_latest_defined_globalrev,null_globalrev);
    std::atomic_store(&_latest_ready_globalrev,null_globalrev);

    std::atomic_store(&_available_math_functions,std::make_shared<math_function_registry_map>());
    
    if (!this->lockmgr) {
      this->lockmgr = std::make_shared<lockmanager>();
    }

    _instantiated_functions._rebuild_dependency_map();

    // instantiate mutablenotneeded thread
    globalrev_mutablenotneeded_thread = std::thread([this]() { globalrev_mutablenotneeded_code(); });
    //globalrev_mutablenotneeded_thread.detach(); // we won't be join()ing this thread

    transaction_background_end_thread = std::thread([this]() { transaction_background_end_code(); });
    
  }

  

  recdatabase::~recdatabase()
  {
    // Trigger globalrev_mutablenotneeded thread to die, then join() it. 
    {
      std::lock_guard<std::mutex> globalrev_mutablenotneeded_lockholder(globalrev_mutablenotneeded_lock);
      //fprintf(stderr,"recdatabase() setting exit flag\n");

      globalrev_mutablenotneeded_mustexit=true;
      globalrev_mutablenotneeded_condition.notify_all();
    }
    
    globalrev_mutablenotneeded_thread.join();
    

    // Trigger transaction_background_end_thread to die, then join() it. 
    {
      std::lock_guard<std::mutex> transaction_background_end_lockholder(transaction_background_end_lock);

      transaction_background_end_mustexit=true;
      transaction_background_end_condition.notify_all();
    }

    transaction_background_end_thread.join();


  }

  void recdatabase::add_alignment_requirement(size_t nbytes)
  {
    alignment_requirements->add_requirement(nbytes);
  }

  void recdatabase::startup()
  {
    {
      if (started) {
	throw snde_error("recdatabase::startup(): recdb already running!");
      }
      started = true; 
    }

    // insert an empty globalrev so there always is one
    snde::active_transaction transact(shared_from_this());
    transact.end_transaction();
    
    
    
    if (!compute_resources->cpu) {
      throw snde_error("CPU not setup (use setup_cpu())");
    }
    if (!default_storage_manager) {

      throw snde_error("Default storage manager not setup (use setup_storage_manager())");
    }
    compute_resources->start();
  }
  
  std::shared_ptr<active_transaction> recdatabase::start_transaction()
  {
    if (!started) {
      throw snde_error("recdatabase::start_transaction(): Recording database has not been started (use start() method)");
    }
    
    return std::make_shared<active_transaction>(shared_from_this());
  }
  
  std::shared_ptr<globalrevision> recdatabase::end_transaction(std::shared_ptr<active_transaction> act_trans)
  {
    return act_trans->end_transaction();
  }

  std::shared_ptr<transaction> recdatabase::run_in_background_and_end_transaction(std::shared_ptr<active_transaction> act_trans,std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<void> params)> fcn,std::shared_ptr<void> params)
  {
    return act_trans->run_in_background_and_end_transaction(fcn,params);
  }

  void recdatabase::add_math_function(std::shared_ptr<instantiated_math_function> new_function,bool hidden)
  {
    add_math_function_storage_manager(new_function,hidden,nullptr);
  }

  std::shared_ptr<instantiated_math_function> recdatabase::lookup_math_function(std::string full_path)
  {
    std::lock_guard<std::mutex> recdb_admin(admin);

    auto def_fcn_it =  _instantiated_functions.defined_math_functions.find(full_path);
    if (def_fcn_it == _instantiated_functions.defined_math_functions.end()) {
      throw snde_error("Channel %s is not from an instantiated math function",full_path.c_str());
    }
    
    return def_fcn_it->second; 
  }

  void recdatabase::delete_math_function(std::shared_ptr<instantiated_math_function> fcn)
  {
    // must be called within a transaction!
    std::vector<std::tuple<std::string,std::shared_ptr<channel>>> paths_and_channels;

    {
      std::lock_guard<std::mutex> recdb_admin(admin);
    
      for (auto && result_channel_path: fcn->result_channel_paths) {
	if (result_channel_path) {
	  std::string full_path = recdb_path_join(fcn->channel_path_context,*result_channel_path);
	  auto chan_it = _channels.find(full_path);
	  if (chan_it != _channels.end()) {
	    std::lock_guard<std::mutex> channel_lock(chan_it->second->admin);
	    if (chan_it->second->_config->math_fcn == fcn) {
	      chan_it->second->end_atomic_config_update(nullptr);
	      chan_it->second->latest_revision++;
	      chan_it->second->deleted = true; 
	      std::shared_ptr<channel> channel_ptr = chan_it->second;
	      _channels.erase(chan_it);
	      _deleted_channels.emplace(std::make_pair(full_path,channel_ptr));
	      
	      paths_and_channels.emplace_back(std::make_tuple(full_path,chan_it->second));
	    }
	  }
	}
      }
    }
      
    // add to the current transaction
    {
      std::lock_guard<std::mutex> curtrans_lock(current_transaction->admin);
      
      for (auto && path_channelptr: paths_and_channels) {
	std::string full_path;
	std::shared_ptr<channel> channelptr;
	std::tie(full_path,channelptr) = path_channelptr;
	
	current_transaction->updated_channels.emplace(channelptr);
      }
    }
    
    
    // remove from _instantiated_functions
    {
      std::lock_guard<std::mutex> recdb_admin(admin);
      for (auto && path_channelptr: paths_and_channels) {
	std::string full_path;
	std::shared_ptr<channel> channelptr;
	std::tie(full_path,channelptr) = path_channelptr;
	
	_instantiated_functions.defined_math_functions.erase(full_path);
      }

      _instantiated_functions._rebuild_dependency_map();
    }

    
  }

  void recdatabase::send_math_message(std::shared_ptr<instantiated_math_function> func, std::string name, std::shared_ptr<math_instance_parameter> msg)
  {

    // Check and throw if not in transaction 
    if (!current_transaction) {
      throw snde_error("send_math_message: must be called in a transaction");
    }
    
    // add to the current transaction
    {
      std::lock_guard<std::mutex> curtrans_lock(current_transaction->admin);
      auto msgs = current_transaction->math_messages.find(func);
      if (msgs != current_transaction->math_messages.end()) {

	auto check = msgs->second.find(name);
	if (check != msgs->second.end()) {
	  throw snde_error("send_math_message:  can only send specified message '%s' once in transaction", name);
	}
	else {
	  msgs->second.emplace(std::make_pair(name, msg));
	}	
      }
      else {
	std::unordered_map<std::string, std::shared_ptr<math_instance_parameter>> new_msgs;
	new_msgs.emplace(std::make_pair(name, msg));
	current_transaction->math_messages.emplace(std::make_pair(func, new_msgs));
      }     	
    }

  }

  void recdatabase::add_math_function_storage_manager(std::shared_ptr<instantiated_math_function> new_function,bool hidden,std::shared_ptr<recording_storage_manager> storage_manager) 
  {

    std::vector<std::tuple<std::string,std::shared_ptr<channel>>> paths_and_channels;

    // Create objects prior to adding them to transaction because transaction lock is later in the locking order
    
    for (auto && result_channel_path: new_function->result_channel_paths) {
      if (result_channel_path) {
	// if a recording is given for this result_channel
	std::string full_path = recdb_path_join(new_function->channel_path_context,*result_channel_path);
	std::shared_ptr<channelconfig> channel_config=std::make_shared<channelconfig>(full_path,"math",(void *)this,hidden,storage_manager); // recdb pointer is owner of math recordings
	channel_config->math_fcn = new_function;
	channel_config->math=true;
	std::shared_ptr<channel> channelptr = reserve_channel(channel_config);
	
	if (!channelptr) {
	  throw snde_error("Channel %s is already defined",full_path.c_str());
	}
	
	paths_and_channels.push_back(std::make_tuple(full_path,channelptr));
      }
    }

    // add to the current transaction
    {
      std::lock_guard<std::mutex> curtrans_lock(current_transaction->admin);
      
      for (auto && path_channelptr: paths_and_channels) {
	std::string full_path;
	std::shared_ptr<channel> channelptr;
	std::tie(full_path,channelptr) = path_channelptr;
	
	current_transaction->updated_channels.emplace(channelptr);
	current_transaction->new_recording_required.emplace(full_path,false);
      }
    }

    // add to _instantiated_functions
    {
      std::lock_guard<std::mutex> recdb_admin(admin);
      for (auto && path_channelptr: paths_and_channels) {
	std::string full_path;
	std::shared_ptr<channel> channelptr;
	std::tie(full_path,channelptr) = path_channelptr;
	
	_instantiated_functions.defined_math_functions.emplace(full_path,new_function);
      }

      _instantiated_functions._rebuild_dependency_map();
    }

  }

  void recdatabase::register_new_rec(std::shared_ptr<recording_base> new_rec)
  {
    std::lock_guard<std::mutex> recdb_admin(admin);
    std::lock_guard<std::mutex> curtrans_lock(current_transaction->admin);


    // in case one new recording has already been added in this transaction for this
    // channel we erase the old one so the new one gets emplaced. 
    auto old_iter = current_transaction->new_recordings.find(new_rec->info->name);
    if (old_iter != current_transaction->new_recordings.end()) {
      current_transaction->new_recordings.erase(old_iter);
    }
    
    current_transaction->new_recordings.emplace(new_rec->info->name,new_rec);
    
  }


  void recdatabase::register_new_math_rec(void *owner_id,std::shared_ptr<recording_set_state> calc_rss,std::shared_ptr<recording_base> new_rec)
  // registers newly created math recording in the given rss (and extracts mutable flag for the given channel). Also checks owner_id
  {
    channel_state & rss_chan = calc_rss->recstatus.channel_map->at(new_rec->info->name);
    assert(rss_chan.config->owner_id == owner_id);
    assert(rss_chan.config->math);
    new_rec->info->immutable = !rss_chan.config->data_mutable;
    
    std::atomic_store(&rss_chan._rec,new_rec);
  }

  std::shared_ptr<globalrevision> recdatabase::latest_defined_globalrev() // safe to call with or without recdb admin lock held
  {
    return std::atomic_load(&_latest_defined_globalrev);
  }

  std::shared_ptr<globalrevision> recdatabase::latest_globalrev() // safe to call with or without recdb admin lock held
  {
    return std::atomic_load(&_latest_ready_globalrev);
  }

  std::shared_ptr<globalrevision> recdatabase::get_globalrev(uint64_t revnum) 
  {
    std::lock_guard<std::mutex> recdb_admin(admin);

    auto grev_it = _globalrevs.find(revnum);

    if (grev_it == _globalrevs.end()) {
      return nullptr;
    }
    return grev_it->second;
  }

  
  std::shared_ptr<channel> recdatabase::reserve_channel(std::shared_ptr<channelconfig> new_config)
  {
    // Note that this is called with transaction lock held, but that is OK because transaction lock precedes recdb admin lock
    std::shared_ptr<channel> new_chan;
    {
      std::lock_guard<std::mutex> recdb_lock(admin);
      
      std::map<std::string,std::shared_ptr<channel>>::iterator chan_it;
      
      chan_it = _channels.find(new_config->channelpath);
      if (chan_it != _channels.end()) {
	// channel in use
	return nullptr;
      }
      
      chan_it = _deleted_channels.find(new_config->channelpath);
      if (chan_it != _deleted_channels.end()) {
	// repurpose existing deleted channel
	new_chan = chan_it->second;
	_deleted_channels.erase(chan_it);
	{
	  // OK to lock channel because channel locks are after the recdb lock we already hold in the locking order
	  std::lock_guard<std::mutex> channel_lock(new_chan->admin);
	  assert(!new_chan->config()); // should be nullptr
	  new_chan->deleted = false;
	  new_chan->end_atomic_config_update(new_config);
	}
	
      } else {
	new_chan = std::make_shared<channel>(new_config);
      }
      // update _channels map with new channel
      _channels.emplace(new_config->channelpath,new_chan);
    }
    {
      // add new_chan to current transaction
      std::lock_guard<std::mutex> curtrans_lock(current_transaction->admin);

      // verify recording not already updated in current transaction
      if (current_transaction->new_recordings.find(new_config->channelpath) != current_transaction->new_recordings.end()) {
	throw snde_error("Replacing owner of channel %s in transaction where recording already updated",new_config->channelpath);
      }
      
      current_transaction->updated_channels.emplace(new_chan);
      current_transaction->new_recording_required.emplace(new_config->channelpath,true);
    }

    
    
    return new_chan;
  }

  void recdatabase::release_channel(std::string path,void *owner_id) // must be called within a transaction
  {
    std::shared_ptr<channel> chan;

    {
      std::lock_guard<std::mutex> recdb_lock(admin);
      
      std::map<std::string,std::shared_ptr<channel>>::iterator chan_it;
      chan_it = _channels.find(path);
      
      if (chan_it == _channels.end()) {
	// channel nonexistent
	return;
      }
      
      chan = chan_it->second;
      if (chan->config()->owner_id != owner_id) {
	throw snde_error("recdatabase::release_channel: owner_id mismatch: Owned by 0x%p (owner_name=%s) vs. released by 0x%p",chan->config()->owner_id,chan->config()->owner_name,owner_id);
      }
      
      
      _channels.erase(chan_it);
      _deleted_channels.emplace(path,chan);
      
      std::lock_guard<std::mutex> chan_lock(chan->admin);
      chan->latest_revision++;
      chan->deleted = true;
      chan->end_atomic_config_update(nullptr);
    }

    // add to the current transaction
    {
      std::lock_guard<std::mutex> curtrans_lock(current_transaction->admin);

      current_transaction->updated_channels.emplace(chan);

    }
  }
  
  // Define a new channel; throws an error if the channel is already in use
  std::shared_ptr<channel> recdatabase::define_channel(std::string channelpath, std::string owner_name, void *owner_id, bool hidden/*=false*/, std::shared_ptr<recording_storage_manager> storage_manager/* =nullptr */)
  {
    std::shared_ptr<channelconfig> new_config=std::make_shared<channelconfig>(channelpath,owner_name,owner_id,hidden,storage_manager);
    
    std::shared_ptr<channel> new_channel=reserve_channel(new_config);

    if (!new_channel) {
      throw snde_error("Channel %s is not available",channelpath.c_str());
    }
    return new_channel;
  }

  
  //  void recdatabase::wait_recordings(std::share_ptr<recording_set_state> rss, const std::vector<std::shared_ptr<recording_base>> &metadataonly,const std::vector<std::shared_ptr<recording_base>> &ready)
  //// NOTE: python wrapper needs to drop thread context during wait and poll to check for connection drop
  // {
  //#error not implemented
  //}

  
  void recdatabase::wait_recording_names(std::shared_ptr<recording_set_state> rss,const std::vector<std::string> &metadataonly, const std::vector<std::string> fullyready)
  // NOTE: python wrapper needs to drop thread context during wait and poll to check for connection drop
  {
    // should queue up a std::promise for central dispatch
    // then we wait here polling the corresponding future with a timeout so we can accommodate dropped
    // connections.

    std::shared_ptr<promise_channel_notify> promise_notify=std::make_shared<promise_channel_notify>(metadataonly,fullyready,false);

    promise_notify->apply_to_rss(rss);

    //std::future<bool> criteria_satisfied = promise_notify->promise.get_future();
    //criteria_satisfied.wait();
    promise_notify->wait_interruptable();
  }

  std::shared_ptr<monitor_globalrevs> recdatabase::start_monitoring_globalrevs(std::shared_ptr<globalrevision> first/* = nullptr*/,bool inhibit_mutable/* = false */)
  // first is the first globalrev to return (nullptr means latest_defined_globalrev()).
  // inhibit_mutable means that this will prevent writing to mutable waveforms
  // of new globalrevs until you have handled them. To use this
  // functionality call the wait_next_inhibit_mutable() method instead of
  // plain wait_next(), which returns a tuple. The 2nd element of the
  // tuple represents a shared lock that prevents writing to mutable waveforms.
  // Be warned that for the first revision or two the 2nd element may be
  // nullptr, indicating that no such lock was able to be acquired. 
    
  {

    std::lock_guard<std::mutex> recdb_admin(admin);

    if (!first) {
      first = latest_defined_globalrev();
    }

    std::shared_ptr<monitor_globalrevs> monitor=std::make_shared<monitor_globalrevs>(first,inhibit_mutable);

    // Any revisions sequentially starting at 'first' that are
    // already ready must get placed into pending
    
    // It's OK to mess with monitor here without locking it because we haven't published it yet
    std::map<uint64_t,std::shared_ptr<globalrevision>>::iterator rev_it;
    uint64_t revcnt; 
    for (revcnt = first->globalrev,rev_it = _globalrevs.find(first->globalrev);rev_it != _globalrevs.end();rev_it = _globalrevs.find(++revcnt)) {

      if (revcnt <= monitoring_notify_globalrev) {
	std::shared_ptr<globalrev_mutable_lock> null_mutable_lock;
	monitor->pending.emplace(std::make_pair(rev_it->first,std::make_tuple(rev_it->second,null_mutable_lock)));
      } else {
	break; 
      }
    }

    // Once in the monitoring queue, any time a subsequent globalrev becomes ready
    // this monitoring object will be notified. 
    monitoring.emplace(monitor);

    return monitor; 
  }

  void recdatabase::register_ready_globalrev_quicknotifies_called_recdb_locked(std::shared_ptr<std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<globalrevision>)>> quicknotify)
  {
    std::lock_guard<std::mutex> recdb_admin(admin);

    ready_globalrev_quicknotifies_called_recdb_locked.emplace(quicknotify);
  }
  
  void recdatabase::unregister_ready_globalrev_quicknotifies_called_recdb_locked(std::shared_ptr<std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<globalrevision>)>> quicknotify)
  {
    std::lock_guard<std::mutex> recdb_admin(admin);
    
    ready_globalrev_quicknotifies_called_recdb_locked.erase(quicknotify);
    
  }


  void recdatabase::globalrev_mutablenotneeded_code()
  {
    std::unique_lock<std::mutex> globalrev_mutablenotneeded_lockholder(globalrev_mutablenotneeded_lock);
    bool exit_flag=false;

    set_thread_name(nullptr,"snde2 recdb gmnnc");

    //fprintf(stderr,"gmnnc() starting\n");

    while (!exit_flag) {
      //fprintf(stderr,"gmnnc() waiting\n");
      globalrev_mutablenotneeded_condition.wait(globalrev_mutablenotneeded_lockholder);
      std::list<std::shared_ptr<globalrevision>>::iterator pending_it;
      //fprintf(stderr,"gmnnc() wakeup\n");

      while ((pending_it = globalrev_mutablenotneeded_pending.begin()) != globalrev_mutablenotneeded_pending.end()) {
	std::shared_ptr<globalrevision> globalrev_mutablenotneeded = *pending_it;
	globalrev_mutablenotneeded_pending.erase(pending_it);
	
	globalrev_mutablenotneeded_lockholder.unlock();
	//snde_warning("gmnnc() processing mutablenotneeded pending %llu",(unsigned long long)globalrev_mutablenotneeded->globalrev);
	
	// got a globalrev where we no longer need to protect its version of mutable waveforms from update
	{
	  std::unique_lock<std::mutex> computeresources_admin(*compute_resources->admin);

	  //fprintf(stderr,"gmnnc() cr_admin locked\n");

	  // go through compute_resources blocked_list, removing the blocked computations corresponding to this globalrev
	  // and queueing them up. 

	  std::multimap<uint64_t,std::shared_ptr<pending_computation>>::iterator blocked_it; 
	  while ((blocked_it = compute_resources->blocked_list.begin()) != compute_resources->blocked_list.end() && blocked_it->first < (globalrev_mutablenotneeded->globalrev+2)*SNDE_CR_PRIORITY_REDUCTION_LIMIT) {  // rationale for the +2: Once a particular globalrev comes through here, we unblock the NEXT globalrev (that's the first +1) but we want to enable everything (all priorities) from that globalrev; hence strictly less than another +1, i.e. +2.
	    std::shared_ptr<pending_computation> blocked_computation = blocked_it->second;

	    //fprintf(stderr,"gmnnc() got element of blocked list\n");

	    compute_resources->blocked_list.erase(blocked_it);
	    computeresources_admin.unlock();
	    compute_resources->_queue_computation_internal(shared_from_this(),blocked_computation); // note: blocked computation pointer is passed by reference and is nullified by this call, making the various weak references to it able to die once it comes off of the todo_list.  
	    computeresources_admin.lock();
	  }
	

	}
	
	globalrev_mutablenotneeded_lockholder.lock();
      }
      if (globalrev_mutablenotneeded_mustexit) {
	exit_flag=true; 
	//fprintf(stderr,"gmnnc() exit flag set\n");
      } else {
	//fprintf(stderr,"gmnnc() exit flag clear\n");
      }
    }
    //fprintf(stderr,"gmnnc() exit\n");

  }


  void recdatabase::transaction_background_end_code()
  {

    set_thread_name(nullptr,"snde2 recdb tbec");

    std::unique_lock<std::mutex> transaction_background_end_lockholder(transaction_background_end_lock);
    
    //fprintf(stderr,"tbec() starting\n");

    while (true) {
      //fprintf(stderr,"tbec() waiting\n");
      transaction_background_end_condition.wait(transaction_background_end_lockholder);
      //fprintf(stderr,"tbec() wakeup\n");

      if (transaction_background_end_mustexit) {
	return;
      }


      if (transaction_background_end_fcn) {
	transaction_background_end_lockholder.unlock();

	std::shared_ptr<recdatabase> recdb_strong = transaction_background_end_acttrans->recdb.lock();
	if (!recdb_strong) {
	  return;
	}

	transaction_background_end_fcn(recdb_strong,transaction_background_end_params);
	
	transaction_background_end_lockholder.lock();
	
	// empty the std::function
	transaction_background_end_fcn = std::function<void(std::shared_ptr<recdatabase> recdb, std::shared_ptr<void> params)>();
	transaction_background_end_params = nullptr;
	std::shared_ptr<active_transaction> transaction_background_end_acttrans_copy = transaction_background_end_acttrans;
	transaction_background_end_acttrans = nullptr;
	transaction_background_end_lockholder.unlock();

	transaction_background_end_acttrans_copy->end_transaction();
	
	transaction_background_end_lockholder.lock();
      }
    }
    //fprintf(stderr,"gmnnc() exit\n");

  }

  
  std::shared_ptr<math_function_registry_map> recdatabase::available_math_functions()
  {
    return std::atomic_load(&_available_math_functions);
  }

  std::shared_ptr<math_function> recdatabase::lookup_available_math_function(std::string name)
  {
    std::shared_ptr<math_function_registry_map> builtin_functions=math_function_registry();    
    std::shared_ptr<math_function_registry_map> addon_functions=available_math_functions();

    math_function_registry_map::iterator builtin_function = builtin_functions->find(name);
    math_function_registry_map::iterator addon_function = addon_functions->find(name);

    if (builtin_function != builtin_functions->end() && addon_function != addon_functions->end()) {
      snde_warning("recdb addon math function %s overrides c++ builtin function of same name",name.c_str());
      return addon_function->second;
    }

    if (builtin_function == builtin_functions->end() && addon_function == addon_functions->end()) {
      throw snde_error("Math function %s not found",name.c_str());
      
    }
    if (addon_function != addon_functions->end()) {
      return addon_function->second;
    }
    if (builtin_function != builtin_functions->end()) {
      return builtin_function->second;
    }
    assert(0); // should not be possible -- all cases handled above
    return nullptr;
  }

  std::shared_ptr<std::vector<std::string>> recdatabase::list_available_math_functions()
  {
    std::set<std::string> sorter; 
    
    std::shared_ptr<std::vector<std::string>> retval = std::make_shared<std::vector<std::string>>();

    
    std::shared_ptr<math_function_registry_map> builtin_functions=math_function_registry();
    
    std::shared_ptr<math_function_registry_map> addon_functions=available_math_functions();

    for (auto && funcname_function: *builtin_functions) {
      sorter.emplace(funcname_function.first);
    }

    for (auto && funcname_function: *builtin_functions) {
      std::set<std::string>::iterator old_function = sorter.find(funcname_function.first);
      if (old_function != sorter.end()) {
	snde_warning("recdb addon math function %s overrides c++ builtin function of same name",funcname_function.first.c_str());
      }
      sorter.emplace(funcname_function.first);
    }

    for (auto && funcname: sorter) {
    
      retval->push_back(funcname);
    }
    return retval;
  }

  std::shared_ptr<math_function_registry_map> recdatabase::_begin_atomic_available_math_functions_update() // should be called with admin lock held
  {
    std::shared_ptr<math_function_registry_map> new_math_functions = std::make_shared<math_function_registry_map>(*available_math_functions());
    return new_math_functions;
  }
  
  void recdatabase::_end_atomic_available_math_functions_update(std::shared_ptr<math_function_registry_map> new_math_functions) // should be called with admin lock held
  {
    std::atomic_store(&_available_math_functions,new_math_functions);
  }
  
  

  size_t recording_default_info_structsize(size_t param,size_t min)
  // Param is the requested info_structsize parameter, or 0 to indicate use the default
  // min is the minimum for this class
  // returns actual size to allocate
  {
    if (!param) {
      param=min;
    }

    if (param < min) {
      throw snde_error("info_structsize during recording creation (%u) is less than minimum (%u) (?)",(unsigned)param,(unsigned)min);
    }
    
    return param;
  }

  std::shared_ptr<ndarray_recording_ref> create_ndarray_ref(std::shared_ptr<recdatabase> recdb,std::shared_ptr<channel> chan,void *owner_id,unsigned typenum)
  {
    std::shared_ptr<multi_ndarray_recording> rec;
    std::shared_ptr<ndarray_recording_ref> ref;

    // ***!!! Should look up maker method in a runtime-addable database ***!!!

    rec=create_recording<multi_ndarray_recording>(recdb,chan,owner_id,1);
    rec->define_array(0,typenum);
    
    ref=rec->reference_ndarray(0);
    return ref;
  }

  std::shared_ptr<ndarray_recording_ref> create_named_ndarray_ref(std::shared_ptr<recdatabase> recdb,std::shared_ptr<channel> chan,void *owner_id,std::string arrayname,unsigned typenum)
  {
    std::shared_ptr<multi_ndarray_recording> rec;
    std::shared_ptr<ndarray_recording_ref> ref;

    // ***!!! Should look up maker method in a runtime-addable database ***!!!

    rec=create_recording<multi_ndarray_recording>(recdb,chan,owner_id,1);
    rec->define_array(0,typenum,arrayname);
    
    ref=rec->reference_ndarray(0);
    return ref;
  }

  std::shared_ptr<ndarray_recording_ref> create_anonymous_ndarray_ref(std::shared_ptr<recdatabase> recdb,std::string purpose,unsigned typenum)
  {
    std::shared_ptr<multi_ndarray_recording> rec;
    std::shared_ptr<ndarray_recording_ref> ref;
    
    // ***!!! Should look up maker method in a runtime-addable database ***!!!
    
    rec=create_anonymous_recording<multi_ndarray_recording>(recdb,purpose,1);
    rec->define_array(0,typenum);
    
    ref=rec->reference_ndarray(0);
    return ref;

  }


  std::shared_ptr<ndarray_recording_ref> create_anonymous_named_ndarray_ref(std::shared_ptr<recdatabase> recdb,std::string purpose,std::string arrayname,unsigned typenum)
  {
    std::shared_ptr<multi_ndarray_recording> rec;
    std::shared_ptr<ndarray_recording_ref> ref;
    
    // ***!!! Should look up maker method in a runtime-addable database ***!!!
    
    rec=create_anonymous_recording<multi_ndarray_recording>(recdb,purpose,1);
    rec->define_array(0,typenum,arrayname);
    
    ref=rec->reference_ndarray(0);
    return ref;

  }

  
  std::shared_ptr<ndarray_recording_ref> create_ndarray_ref_math(std::string chanpath,std::shared_ptr<recording_set_state> calc_rss,unsigned typenum)
  {
    std::shared_ptr<multi_ndarray_recording> rec;
    std::shared_ptr<ndarray_recording_ref> ref;
    
    std::shared_ptr<recdatabase> recdb = calc_rss->recdb_weak.lock();
    if (!recdb) return nullptr;

    
    rec=create_recording_math<multi_ndarray_recording>(chanpath,calc_rss,1); 
    rec->define_array(0,typenum);
    ref=rec->reference_ndarray(0);

    return ref;
  }


  std::shared_ptr<ndarray_recording_ref> create_named_ndarray_ref_math(std::string chanpath,std::shared_ptr<recording_set_state> calc_rss,std::string arrayname,unsigned typenum)
  {
    std::shared_ptr<multi_ndarray_recording> rec;
    std::shared_ptr<ndarray_recording_ref> ref;
    
    std::shared_ptr<recdatabase> recdb = calc_rss->recdb_weak.lock();
    if (!recdb) return nullptr;

    
    rec=create_recording_math<multi_ndarray_recording>(chanpath,calc_rss,1); 
    rec->define_array(0,typenum,arrayname);
    ref=rec->reference_ndarray(0);

    return ref;
  }


};
