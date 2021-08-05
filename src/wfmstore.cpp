#include "wfmstore.hpp"

namespace snde {


// wtn_typemap is indexed by typeid(type)
  const std::unordered_map<std::type_index,unsigned> wtn_typemap({ // look up typenum based on C++ typeid(type)
      {typeid(snde_float32), SNDE_WTN_FLOAT32},
      {typeid(snde_float64), SNDE_WTN_FLOAT64},
      // half-precision not generally
      // available
#ifdef SNDE_HAVE_FLOAT16
      {typeid(snde_float16), SNDE_WTN_FLOAT16},
#endif // SNDE_HAVE_FLOAT16
      {typeid(uint64_t),SNDE_WTN_UINT64},
      {typeid(int64_t),SNDE_WTN_INT64},
      {typeid(uint32_t),SNDE_WTN_UINT32},
      {typeid(int32_t),SNDE_WTN_INT32},
      {typeid(uint16_t),SNDE_WTN_UINT16},
      {typeid(int16_t),SNDE_WTN_INT16},
      {typeid(uint8_t),SNDE_WTN_UINT8},
      {typeid(int8_t),SNDE_WTN_INT8},
      {typeid(snde_rgba),SNDE_WTN_RGBA32},
      {typeid(std::complex<snde_float32>),SNDE_WTN_COMPLEXFLOAT32},
      {typeid(std::complex<snde_float64>),SNDE_WTN_COMPLEXFLOAT64},
      {typeid(snde_rgbd),SNDE_WTN_RGBD64}
      
  });
  
  // wtn_typesizemap is indexed by SNDE_WTN_xxx
  const std::unordered_map<unsigned,size_t> wtn_typesizemap({ // Look up element size bysed on typenum
      {SNDE_WTN_FLOAT32,sizeof(snde_float32)},
      {SNDE_WTN_FLOAT64,sizeof(snde_float64)},
      // half-precision not generally
      // available
#ifdef SNDE_HAVE_FLOAT16
      {SNDE_WTN_FLOAT16,sizeof(snde_float16)},
#else // SNDE_HAVE_FLOAT16
      {SNDE_WTN_FLOAT16,2},
#endif
      {SNDE_WTN_UINT64,sizeof(uint64_t)},
      {SNDE_WTN_INT64,sizeof(int64_t)},
      {SNDE_WTN_UINT32,sizeof(uint32_t)},
      {SNDE_WTN_INT32,sizeof(int32_t)},
      {SNDE_WTN_UINT16,sizeof(uint16_t)},
      {SNDE_WTN_INT16,sizeof(int16_t)},
      {SNDE_WTN_UINT8,sizeof(uint8_t)},
      {SNDE_WTN_INT8,sizeof(int8_t)},
      {SNDE_WTN_RGBA32,sizeof(snde_rgba)},
      {SNDE_WTN_COMPLEXFLOAT32,sizeof(std::complex<snde_float32>)},
      {SNDE_WTN_COMPLEXFLOAT64,sizeof(std::complex<snde_float64>)},
#ifdef SNDE_HAVE_FLOAT16
      {SNDE_WTN_COMPLEXFLOAT16,sizeof(snde_float16)},
#else // SNDE_HAVE_FLOAT16
      {SNDE_WTN_COMPLEXFLOAT16,4},
#endif
      {SNDE_WTN_RGBD64,sizeof(snde_rgbd)},
    });
  

  const std::unordered_map<unsigned,std::string> wtn_ocltypemap({ // Look up opencl type string based on typenum
      {SNDE_WTN_FLOAT32,"float"},
      {SNDE_WTN_FLOAT64,"double"},
      // half-precision not generally
      // available
      {SNDE_WTN_FLOAT16,"half"},
      {SNDE_WTN_UINT64,"unsigned long"},
      {SNDE_WTN_INT64,"long"},
      {SNDE_WTN_UINT32,"unsigned int"},
      {SNDE_WTN_INT32,"int"},
      {SNDE_WTN_UINT16,"unsigned short"},
      {SNDE_WTN_INT16,"short"},
      {SNDE_WTN_UINT8,"unsigned char"},
      {SNDE_WTN_INT8,"char"},
      {SNDE_WTN_RGBA32,"snde_rgba"},
      {SNDE_WTN_COMPLEXFLOAT32,"struct { float real; float imag; }"},
      {SNDE_WTN_COMPLEXFLOAT64,"struct { double real; double imag; }"},
      {SNDE_WTN_COMPLEXFLOAT16,"struct { half real; half imag; }"},
      {SNDE_WTN_RGBD64,"snde_rgbd"},
      
    });
  // see https://stackoverflow.com/questions/38644146/choose-template-based-on-run-time-string-in-c

  
  waveform_base::waveform_base(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<channel> chan,void *owner_id,size_t info_structsize) :
    // This constructor is to be called by everything except the math engine
    //  * Should be called by the owner of the given channel, as verified by owner_id
    //  * Should be called within a transaction (i.e. the wfmdb transaction_lock is held by the existance of the transaction)
    //  * Because within a transaction, wfmdb->current_transaction is valid
    // This constructor automatically adds the new waveform to the current transaction
    info{nullptr},
    info_state(SNDE_WFMS_INITIALIZING),
    metadata(nullptr),
    storage_manager(wfmdb->default_storage_manager),
    storage(nullptr),
    _wfmdb(wfmdb),
    _originating_globalrev_index(wfmdb->current_transaction->globalrev),
    _originating_wss()
  {
    uint64_t new_revision = ++chan->latest_revision; // atomic variable so it is safe to pre-increment
    info = (snde_waveform_base *)calloc(1,info_structsize);
    
    assert(info_structsize >= sizeof(snde_waveform_base));
    
    snde_waveform_base info_prototype{
      .name=strdup(chan->config()->channelpath.c_str()),
      .revision=new_revision,
      .state=info_state,
      .metadata=nullptr,
      .metadata_valid=false,
      .deletable=false,
      .immutable=true,
    };
    *info = info_prototype;
    
    {
      std::lock_guard<std::mutex> curtrans_lock(wfmdb->current_transaction->admin);
      wfmdb->current_transaction->new_waveforms.emplace(chan->config()->channelpath,shared_from_this());
    }
  }
  
  waveform_base::waveform_base(std::shared_ptr<wfmdatabase> wfmdb,std::string chanpath,void *owner_id,std::shared_ptr<waveform_set_state> calc_wss,size_t info_structsize) :
    // This constructor is reserved for the math engine
    // Creates waveform structure and adds to the pre-existing globalrev.
    info{nullptr},
    info_state(SNDE_WFMS_INITIALIZING),
    metadata(nullptr),
    storage_manager(wfmdb->default_storage_manager),
    storage(nullptr),
    _wfmdb(wfmdb),
    _originating_globalrev_index(wfmdb->current_transaction->globalrev),
    _originating_wss(calc_wss)
  {

    uint64_t new_revision=0;

    if (std::dynamic_pointer_cast<globalrevision>(calc_wss)) {
      // if calc_wss is really a globalrevision (i.e. not an ondemand calculation)
      // then we need to define a new revision of this waveform
            
      channel_state &state = calc_wss->wfmstatus.channel_map.at(chanpath);
      assert(!state.config->ondemand);

      // if the new_revision_optional flag is set, then we have to define the new revision now;
      if (state.config->math_fcn->fcn->new_revision_optional) {
	new_revision = ++state._channel->latest_revision; // latest_revision is atomic; correct ordering guaranteed by the implicit self-dependency that comes with new_revision_optional flag
      } else{
	// new_revision_optional is clear: grab revision from channel_state
	new_revision = state.revision;
      }
      
    }
    info = (snde_waveform_base *)calloc(1,info_structsize);
    assert(info_structsize >= sizeof(snde_waveform_base));

    snde_waveform_base info_prototype{
      .name=strdup(chanpath.c_str()),
      .revision=new_revision,
      .state=info_state,
      .metadata=nullptr,
      .metadata_valid=false,
      .deletable=false,
      .immutable=true, // overridden below from data_mutable flag of the channelconfig 
    };
    *info = info_prototype;

    
    channel_state & wss_chan = calc_wss->wfmstatus.channel_map.at(chanpath);
    assert(wss_chan.config->owner_id == owner_id);
    assert(wss_chan.config->math);
    info->immutable = wss_chan.config->data_mutable;
    
    std::atomic_store(&wss_chan._wfm,shared_from_this());
  }

  waveform_base::~waveform_base()
  {
    free(info->name);
    info->name=nullptr;
    free(info);
    info=nullptr;
  }


  std::shared_ptr<waveform_set_state> waveform_base::_get_originating_wss_wfmdb_and_wfm_admin_prelocked()
  // version of get_originating_wss() to use if you have the wfmdb and wfm admin locks already locked.
  {
    std::shared_ptr<waveform_set_state> originating_wss_strong;
    std::shared_ptr<wfmdatabase> wfmdb_strong(_wfmdb);
    bool originating_wss_is_expired=false; // will assign to true if we get an invalid pointer and it turns out to be expired rather than null

    {
      // get originating_wss from _originating_wss weak ptr in class and
      // if unavailable determine it is expired (see https://stackoverflow.com/questions/26913743/can-an-expired-weak-ptr-be-distinguished-from-an-uninitialized-one)
      originating_wss_strong = _originating_wss.lock();
      if (!originating_wss_strong) {
	// check if merely unassigned vs. expired by testing with owner_before on a nullptr
	std::weak_ptr<waveform_set_state> null_weak_ptr;
	if (null_weak_ptr.owner_before(_originating_wss) || _originating_wss.owner_before(null_weak_ptr)) {
	  // this is distinct from the nullptr
	  originating_wss_is_expired=true; 
	}
      }
    }

    // OK; Now we have a strong ptr, which may be null, and
    // if so originating_wss_is_expired is true iff it was
    // once valid
    
    if (!originating_wss_strong && originating_wss_is_expired) {
      throw snde_error("Attempting to get expired originating waveform set state (channel %s revision %llu", info->name,(unsigned long long)info->revision);
      
    }

    if (!originating_wss_strong) {
      // in this case originating_wss was never assigned. We need to extract it
      originating_wss_strong = wfmdb_strong->_globalrevs.at(_originating_globalrev_index);
    }

    return originating_wss_strong;
  }
  
  std::shared_ptr<waveform_set_state> waveform_base::get_originating_wss()
  // Get the originating waveform set state (often a globalrev)
  // You should only call this if you are sure that originating wss must still exist
  // (otherwise may generate a snde_error), such as before the creator has declared
  // the waveform "ready". This will lock the waveform database and wfm admin locks,
  // so any locks currently held must precede both in the locking order
  {

    std::shared_ptr<waveform_set_state> originating_wss_strong = _originating_wss.lock();
    if (originating_wss_strong) return originating_wss_strong;
    
    std::shared_ptr<wfmdatabase> wfmdb_strong = _wfmdb.lock();
    if (!wfmdb_strong) return nullptr; // shouldn't be possible in general
    std::lock_guard<std::mutex> wfmdbadmin(wfmdb_strong->admin);
    std::lock_guard<std::mutex> wfmadmin(admin);
    return _get_originating_wss_wfmdb_and_wfm_admin_prelocked();
  }

  
  ndarray_waveform::ndarray_waveform(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<channel> chan,void *owner_id,unsigned typenum,size_t info_structsize) :
    // This constructor is to be called by everything except the math engine
    //  * Should be called by the owner of the given channel, as verified by owner_id
    //  * Should be called within a transaction (i.e. the wfmdb transaction_lock is held by the existance of the transaction)
    //  * Because within a transaction, wfmdb->current_transaction is valid
    // This constructor automatically adds the new waveform to the current transaction
    waveform_base(wfmdb,chan,owner_id,info_structsize),
    layout(arraylayout(std::vector<snde_index>()))
  {

    assert(info_structsize >= sizeof(snde_ndarray_waveform));

    ndinfo()->dims_valid=false;
    ndinfo()->data_valid=false;
    ndinfo()->ndim=0;
    ndinfo()->base_index=0;
    ndinfo()->dimlen=nullptr;
    ndinfo()->strides=nullptr;
    ndinfo()->owns_dimlen_strides=false;
    ndinfo()->typenum=typenum;
    ndinfo()->elementsize=0;
    ndinfo()->basearray = nullptr;
    ndinfo()->basearray_holder = nullptr;
    
  }

ndarray_waveform::ndarray_waveform(std::shared_ptr<wfmdatabase> wfmdb,std::string chanpath,void *owner_id,std::shared_ptr<waveform_set_state> calc_wss,unsigned typenum,size_t info_structsize) :
  // This constructor is reserved for the math engine
  // Creates waveform structure and adds to the pre-existing globalrev. 
  waveform_base(wfmdb,chanpath,owner_id,calc_wss,info_structsize),
  layout(arraylayout(std::vector<snde_index>()))
  //mutable_lock(nullptr),  // for simply mutable math waveforms will need to init with std::make_shared<rwlock>();
  {
    assert(info_structsize >= sizeof(snde_ndarray_waveform));

    ndinfo()->dims_valid=false;
    ndinfo()->data_valid=false;
    ndinfo()->ndim=0;
    ndinfo()->base_index=0;
    ndinfo()->dimlen=nullptr;
    ndinfo()->strides=nullptr;
    ndinfo()->owns_dimlen_strides=false;
    ndinfo()->typenum=typenum;
    ndinfo()->elementsize=0;
    ndinfo()->basearray = nullptr;
    ndinfo()->basearray_holder = nullptr;

  }

  
  ndarray_waveform::~ndarray_waveform()
  {
    // c pointers get freed automatically because they point into the c++ structs. 
  }


  /*static */ std::shared_ptr<ndarray_waveform> ndarray_waveform::create_typed_waveform(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<channel> chan,void *owner_id,unsigned typenum)
  {
    switch (typenum) {
    case SNDE_WTN_FLOAT32:
      return std::make_shared<ndtyped_waveform<snde_float32>>(wfmdb,chan,owner_id);

    case SNDE_WTN_FLOAT64: 
      return std::make_shared<ndtyped_waveform<snde_float64>>(wfmdb,chan,owner_id);

#ifdef SNDE_HAVE_FLOAT16
    case SNDE_WTN_FLOAT16: 
      return std::make_shared<ndtyped_waveform<snde_float16>>(wfmdb,chan,owner_id);
#endif

    case SNDE_WTN_UINT64:
      return std::make_shared<ndtyped_waveform<uint64_t>>(wfmdb,chan,owner_id);
      
    case SNDE_WTN_INT64:
      return std::make_shared<ndtyped_waveform<int64_t>>(wfmdb,chan,owner_id);

    case SNDE_WTN_UINT32:
      return std::make_shared<ndtyped_waveform<uint32_t>>(wfmdb,chan,owner_id);
      
    case SNDE_WTN_INT32:
      return std::make_shared<ndtyped_waveform<int32_t>>(wfmdb,chan,owner_id);

    case SNDE_WTN_UINT16:
      return std::make_shared<ndtyped_waveform<uint16_t>>(wfmdb,chan,owner_id);
      
    case SNDE_WTN_INT16:
      return std::make_shared<ndtyped_waveform<int16_t>>(wfmdb,chan,owner_id);
      
    case SNDE_WTN_UINT8:
      return std::make_shared<ndtyped_waveform<uint8_t>>(wfmdb,chan,owner_id);
      
    case SNDE_WTN_INT8:
      return std::make_shared<ndtyped_waveform<int8_t>>(wfmdb,chan,owner_id);

    case SNDE_WTN_RGBA32:
      return std::make_shared<ndtyped_waveform<snde_rgba>>(wfmdb,chan,owner_id);

    case SNDE_WTN_COMPLEXFLOAT32:
      return std::make_shared<ndtyped_waveform<std::complex<snde_float32>>>(wfmdb,chan,owner_id);

    case SNDE_WTN_COMPLEXFLOAT64:
      return std::make_shared<ndtyped_waveform<std::complex<snde_float64>>>(wfmdb,chan,owner_id);

#ifdef SNDE_HAVE_FLOAT16
    case SNDE_WTN_COMPLEXFLOAT16:
      return std::make_shared<ndtyped_waveform<std::complex<snde_float16>>>(wfmdb,chan,owner_id);
#endif
      
    case SNDE_WTN_RGBD64:
      return std::make_shared<ndtyped_waveform<snde_rgbd>>(wfmdb,chan,owner_id);
      
    default:
      throw snde_error("ndarray_waveform::create_typed_waveform(): Unknown type number %u",typenum);
    }
  }
  /* static */ std::shared_ptr<ndarray_waveform> ndarray_waveform::create_typed_waveform(std::shared_ptr<wfmdatabase> wfmdb,std::string chanpath,void *owner_id,std::shared_ptr<waveform_set_state> calc_wss,unsigned typenum)
  {
    switch (typenum) {
    case SNDE_WTN_FLOAT32:
      return std::make_shared<ndtyped_waveform<snde_float32>>(wfmdb,chanpath,owner_id,calc_wss);

    case SNDE_WTN_FLOAT64: 
      return std::make_shared<ndtyped_waveform<snde_float64>>(wfmdb,chanpath,owner_id,calc_wss);

#ifdef SNDE_HAVE_FLOAT16
    case SNDE_WTN_FLOAT16: 
      return std::make_shared<ndtyped_waveform<snde_float16>>(wfmdb,chanpath,owner_id,calc_wss);
#endif

    case SNDE_WTN_UINT64:
      return std::make_shared<ndtyped_waveform<uint64_t>>(wfmdb,chanpath,owner_id,calc_wss);
      
    case SNDE_WTN_INT64:
      return std::make_shared<ndtyped_waveform<int64_t>>(wfmdb,chanpath,owner_id,calc_wss);

    case SNDE_WTN_UINT32:
      return std::make_shared<ndtyped_waveform<uint32_t>>(wfmdb,chanpath,owner_id,calc_wss);
      
    case SNDE_WTN_INT32:
      return std::make_shared<ndtyped_waveform<int32_t>>(wfmdb,chanpath,owner_id,calc_wss);

    case SNDE_WTN_UINT16:
      return std::make_shared<ndtyped_waveform<uint16_t>>(wfmdb,chanpath,owner_id,calc_wss);
      
    case SNDE_WTN_INT16:
      return std::make_shared<ndtyped_waveform<int16_t>>(wfmdb,chanpath,owner_id,calc_wss);
      
    case SNDE_WTN_UINT8:
      return std::make_shared<ndtyped_waveform<uint8_t>>(wfmdb,chanpath,owner_id,calc_wss);
      
    case SNDE_WTN_INT8:
      return std::make_shared<ndtyped_waveform<int8_t>>(wfmdb,chanpath,owner_id,calc_wss);

    case SNDE_WTN_RGBA32:
      return std::make_shared<ndtyped_waveform<snde_rgba>>(wfmdb,chanpath,owner_id,calc_wss);

    case SNDE_WTN_COMPLEXFLOAT32:
      return std::make_shared<ndtyped_waveform<std::complex<snde_float32>>>(wfmdb,chanpath,owner_id,calc_wss);

    case SNDE_WTN_COMPLEXFLOAT64:
      return std::make_shared<ndtyped_waveform<std::complex<snde_float64>>>(wfmdb,chanpath,owner_id,calc_wss);

#ifdef SNDE_HAVE_FLOAT16
    case SNDE_WTN_COMPLEXFLOAT16:
      return std::make_shared<ndtyped_waveform<std::complex<snde_float16>>>(wfmdb,chanpath,owner_id,calc_wss);
#endif
      
    case SNDE_WTN_RGBD64:
      return std::make_shared<ndtyped_waveform<snde_rgbd>>(wfmdb,chanpath,owner_id,calc_wss);
      
    default:
      throw snde_error("ndarray_waveform::create_typed_waveform() (math): Unknown type number %u",typenum);
    }
    
  }

  
  void ndarray_waveform::allocate_storage(std::vector<snde_index> dimlen, bool fortran_order) // fortran_order defaults to false
  // must assign info.elementsize and info.typenum before calling allocate_storage()
  // fortran_order only affects physical layout, not logical layout (interpretation of indices)
  {
    snde_index nelem=1;
    snde_index base_index;
    snde_index stride=1;
    size_t dimnum;

    for (dimnum=0;dimnum < dimlen.size();dimnum++) {
      nelem *= dimlen.at(dimnum);
    }
    
    std::tie(storage,base_index) = storage_manager->allocate_waveform(info->name,info->revision,ndinfo()->elementsize,ndinfo()->typenum,nelem);
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
	stride *= dimlen.at(dimnum);
      }
      
    }
    
    layout=arraylayout(dimlen,strides,base_index);
    
    ndinfo()->base_index=layout.base_index;
    ndinfo()->ndim=layout.dimlen.size();
    ndinfo()->dimlen=layout.dimlen.data();
    ndinfo()->strides=layout.strides.data();
  }

  void ndarray_waveform::reference_immutable_waveform(std::shared_ptr<ndarray_waveform> wfm,std::vector<snde_index> dimlen,std::vector<snde_index> strides,snde_index base_index)
  {
    snde_index first_index;
    snde_index last_index;
    size_t dimnum;

    if (!wfm->storage->finalized) {
      throw snde_error("Waveform %s trying to reference non-final data from waveform %s",info->name,wfm->info->name);      
      
    }
    
    ndinfo()->typenum = wfm->ndinfo()->typenum;

    if (ndinfo()->elementsize != 0 && ndinfo()->elementsize != wfm->ndinfo()->elementsize) {
      throw snde_error("Element size mismatch in waveform %s trying to reference data from waveform %s",info->name,wfm->info->name);
    }
    ndinfo()->elementsize = wfm->ndinfo()->elementsize;
    
    
    first_index=base_index;
    for (dimnum=0;dimnum < dimlen.size();dimnum++) {
      if (strides.at(dimnum) < 0) { // this is somewhat academic because strides is currently snde_index, which is currently unsigned so can't possibly be negative. This test is here in case we make snde_index signed some time in the future... 
	first_index += strides.at(dimnum)*(dimlen.at(dimnum))-1;
      }
    }
    if (first_index < 0) {
      throw snde_error("Referencing negative indices in waveform %s trying to reference data from waveform %s",info->name,wfm->info->name);
    }

    last_index=base_index;
    for (dimnum=0;dimnum < dimlen.size();dimnum++) {
      if (strides.at(dimnum) > 0) { 
	last_index += strides.at(dimnum)*(dimlen.at(dimnum))-1;
      }
    }
    if (last_index >= wfm->storage->nelem) {
      throw snde_error("Referencing out-of-bounds indices in waveform %s trying to reference data from waveform %s",info->name,wfm->info->name);
    }

    
    storage = wfm->storage;
    layout=arraylayout(dimlen,strides,base_index);
    
    ndinfo()->base_index=layout.base_index;
    ndinfo()->ndim=layout.dimlen.size();
    ndinfo()->dimlen=layout.dimlen.data();
    ndinfo()->strides=layout.strides.data();
    
  }


 
  double ndarray_waveform::element_double(const std::vector<snde_index> &idx)
  {
    throw snde_error("Cannot access elements of untyped waveform. Create waveform with .create_typed_waveform() instead.");
  }
  
  void ndarray_waveform::assign_double(const std::vector<snde_index> &idx,double val)
  {
    throw snde_error("Cannot access elements of untyped waveform. Create waveform with .create_typed_waveform() instead.");
  }
  int64_t ndarray_waveform::element_int(const std::vector<snde_index> &idx)
  {
    throw snde_error("Cannot access elements of untyped waveform. Create waveform with .create_typed_waveform() instead.");
  }
  void ndarray_waveform::assign_int(const std::vector<snde_index> &idx,int64_t val)
  {
    throw snde_error("Cannot access elements of untyped waveform. Create waveform with .create_typed_waveform() instead.");
  }
  
  uint64_t ndarray_waveform::element_unsigned(const std::vector<snde_index> &idx)
  {
    throw snde_error("Cannot access elements of untyped waveform. Create waveform with .create_typed_waveform() instead.");
  }
  
  void ndarray_waveform::assign_unsigned(const std::vector<snde_index> &idx,uint64_t val)
  {
    throw snde_error("Cannot access elements of untyped waveform. Create waveform with .create_typed_waveform() instead.");
  }
  
#if 0
  // ***!!! This code is probably obsolete ***!!!
  rwlock_token_set waveform::lock_storage_for_write()
  {
    bool immutable;
    {
      std::lock_guard<std::mutex> adminlock(admin);
      immutable=info->immutable;
    }
    if (immutable) {
      // storage locking not required for waveform data for waveforms that (will) be immutable
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

  rwlock_token_set waveform::lock_storage_for_read()
  {
    bool immutable;
    {
      std::lock_guard<std::mutex>(admin);
      immutable=info->immutable;
    }
    if (immutable) {
      // storage locking not required for waveform data for waveforms that (will) be immutable
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

  /* std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> */ void waveform_base::_mark_metadata_done_internal(/*std::shared_ptr<waveform_set_state> wss,const std::string &channame*/)
  // internal use only. Should be called with the waveform admin lock held
  {

      
    assert(info->state == info_state && info->state==SNDE_WFMS_INITIALIZING);

    info->state = SNDE_WFMS_METADATAREADY;
    info_state = SNDE_WFMS_METADATAREADY;

    // this code replaced by issue_nonmath_notifications, below
    //channel_state &chanstate = wss->wfmstatus.channel_map.at(channame);
    //notify_about_this_channel_metadataonly = chanstate.notify_about_this_channel_metadataonly();
    //chanstate.end_atomic_notify_about_this_channel_metadataonly_update(nullptr); // all notifications are now our responsibility
    // return notify_about_this_channel_metadataonly;
    
  }
  
  void waveform_base::mark_metadata_done()
  {
    // This should be called, not holding locks, (except perhaps dg_python context) after info->metadata is finalized

    std::shared_ptr<wfmdatabase> wfmdb = _wfmdb.lock();
    if (!wfmdb) return;
    
    
    std::string channame;
    //std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_metadataonly;
    {
      //std::lock_guard<std::mutex> wssadminlock(wss->admin);  (not actually necessary for anything we are doing
      std::lock_guard<std::mutex> adminlock(admin);
      assert(info->metadata);
      
      if (info_state == SNDE_WFMS_METADATAREADY || info_state==SNDE_WFMS_READY || info_state == SNDE_WFMS_OBSOLETE) {
	return; // already ready (or beyond)
      }

      channame = info->name;
      /*notify_about_this_channel_metadataonly = */_mark_metadata_done_internal(/*wss,channame*/);
    }

    //// perform notifications


    if (!wfmdb->latest_globalrev() || wfmdb->latest_globalrev()->globalrev < _originating_globalrev_index) {
      // this transaction is still in progress; notifications will be handled by end_transaction
      return;
    }

    // with transaction complete, should be able to get an originating wss
    std::shared_ptr<waveform_set_state> wss = get_originating_wss();
    
    //for (auto && notify_ptr: *notify_about_this_channel_metadataonly) {
    //  notify_ptr->notify_metadataonly(channame);
    //}

    // Above replaced by chanstate.issue_nonmath_notifications

    channel_state &chanstate = wss->wfmstatus.channel_map.at(channame);
    chanstate.issue_math_notifications(wfmdb,wss);
    chanstate.issue_nonmath_notifications(wss);

    
  }
  
  void waveform_base::mark_as_ready()  
  {
    std::string channame;
    //std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_ready;
    //std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_metadataonly;

    // This should be called, not holding locks, (except perhaps dg_python context) after info->metadata is finalized
    
    std::shared_ptr<wfmdatabase> wfmdb = _wfmdb.lock();
    if (!wfmdb) return;
    

    {
      std::lock_guard<std::mutex> wfm_admin(admin);

      assert(info->metadata);
      
      if (info_state==SNDE_WFMS_READY || info_state == SNDE_WFMS_OBSOLETE) {
	return; // already ready (or beyond)
      }
      channame = info->name;
      
      if (info_state==SNDE_WFMS_INITIALIZING) {
	// need to perform metadata notifies too
	/*notify_about_this_channel_metadataonly =*/ /* _mark_metadata_done_internal(wss,channame); */
	
      }

      info->state = SNDE_WFMS_READY;
      info_state = SNDE_WFMS_READY;

      // These next few lines replaced by chanstate.issue_nonmath_notifications, below
      //channel_state &chanstate = wss->wfmstatus.channel_map.at(channame);
      //notify_about_this_channel_ready = chanstate.notify_about_this_channel_ready();
      //chanstate.end_atomic_notify_about_this_channel_ready_update(nullptr); // all notifications are now our responsibility
      
    }

    // perform notifications (replaced by issue_nonmath_notifications())
    //for (auto && notify_ptr: *notify_about_this_channel_metadataonly) {
    //  notify_ptr->notify_metadataonly(channame);
    //}
    //for (auto && notify_ptr: *notify_about_this_channel_ready) {
    //  notify_ptr->notify_ready(channame);
    //}

    
    if (!wfmdb->latest_globalrev() || wfmdb->latest_globalrev()->globalrev < _originating_globalrev_index) {
      // this transaction is still in progress; notifications will be handled by end_transaction()
      return;
    }

    // with transaction complete, should be able to get an originating wss
    std::shared_ptr<waveform_set_state> wss = get_originating_wss();

    channel_state &chanstate = wss->wfmstatus.channel_map.at(channame);
    chanstate.issue_math_notifications(wfmdb,wss);
    chanstate.issue_nonmath_notifications(wss);
  }

  active_transaction::active_transaction(std::shared_ptr<wfmdatabase> wfmdb) :
    wfmdb(wfmdb),
    transaction_ended(false)
  {
    std::unique_lock<std::mutex> tr_lock_acquire(wfmdb->transaction_lock);;

    tr_lock_acquire.swap(transaction_lock_holder); // transfer lock into holder
    //wfmdb->_transaction_raii_holder=shared_from_this();

    assert(!wfmdb->current_transaction);

    wfmdb->current_transaction=std::make_shared<transaction>();

    
    uint64_t previous_globalrev_index = 0;
    {
      std::lock_guard<std::mutex> wfmdb_lock(wfmdb->admin);
      std::map<uint64_t,std::shared_ptr<globalrevision>>::iterator last_globalrev_ptr = wfmdb->_globalrevs.end();
      --last_globalrev_ptr; // change from final+1 to final entry

      if (last_globalrev_ptr != wfmdb->_globalrevs.end()) {
	// if there is a last globalrev (otherwise we are starting the first!)
	previous_globalrev_index = last_globalrev_ptr->first;
	previous_globalrev = last_globalrev_ptr->second;

      } else {
	previous_globalrev = nullptr; 
      }
    }      
    wfmdb->current_transaction->globalrev = previous_globalrev_index+1;
    

    
  }

  static void _identify_changed_channels(std::map<std::string,std::shared_ptr<channelconfig>> &all_channels_by_name,instantiated_math_database &mathdb, std::unordered_set<std::shared_ptr<channelconfig>> &maybechanged_channels,std::unordered_set<std::shared_ptr<channelconfig>> &changed_channels_dispatched,std::unordered_set<std::shared_ptr<channelconfig>> &changed_channels_need_dispatch,std::unordered_set<std::shared_ptr<channelconfig>>::iterator channel_to_dispatch_it ) // channel_to_dispatch should be in changed_channels_need_dispatch
  // mathdb must be immutable; (i.e. it must already be copied into the global revision)
  {
    // std::unordered_set iterators remain valid so long as the iterated element itself is not erased, and the set does not need rehashing.
    // Therefore the changed_channels_dispatched and changed_channels_need_dispatch set needs to have reserve() called on it first with the maximum possible growth.
    
    // We do add to changed_channels_need dispatch
    //std::unordered_set<std::shared_ptr<channelconfig>>::iterator changed_chan;

    std::shared_ptr<channelconfig> channel_to_dispatch = *channel_to_dispatch_it;
    changed_channels_need_dispatch.erase(channel_to_dispatch_it);
    changed_channels_dispatched.emplace(channel_to_dispatch);

    if (channel_to_dispatch->math) {
      // look up what is dependent on this channel
      std::unordered_set<std::shared_ptr<instantiated_math_function>> &dependent_math_functions=mathdb.all_dependencies_of_channel.at(channel_to_dispatch->channelpath);

      for (auto && instantiated_math_ptr: dependent_math_functions) {
	if (instantiated_math_ptr->ondemand || instantiated_math_ptr->disabled) {
	  // ignore disabled and ondemand dependent channels (for now)
	  continue; 
	}
	for (auto && result_chanpath_name_ptr: instantiated_math_ptr->result_channel_paths) {
	  if (result_chanpath_name_ptr) {
	    // Found a dependent channel name
	    // Could probably speed this up by copying result_channel_paths into a form whre it points directly at channelconfs. 
	    std::shared_ptr<channelconfig> channelconf = all_channels_by_name.at(*result_chanpath_name_ptr);
	    
	    std::unordered_set<std::shared_ptr<channelconfig>>::iterator maybechanged_it = maybechanged_channels.find(channelconf);
	    // Is the dependenent channel in maybechanged_channels?... if so it is definitely changed, but not yet dispatched
	    if (maybechanged_it != maybechanged_channels.end()) {
	      // put it in the changed dispatch set
	      changed_channels_need_dispatch.emplace(*maybechanged_it);
	      // remove it from the maybe changed set
	      maybechanged_channels.erase(maybechanged_it);
	    } 
	  }
	}
      }
    }

        
  }

  static void mark_prospective_calculations(std::shared_ptr<waveform_set_state> state,std::unordered_set<std::shared_ptr<channelconfig>> definitively_changed_channels)
  // marks execution_demanded flag in the math_function_status for all math functions directly
  // dependent on the definitively_changed_channels.
  // (should this really be in wfmmath.cpp?)
  {
    // Presumes any state admin lock already held or not needed because not yet published

    // Go through the all_dependencies_of_channel of the instantiated_math_database, accumulating
    // all of the instantiated_math_functions into a set
    // Then go through the set, look up the math_function_status from the math_status and set the execution_demanded flag

    //std::unordered_set<std::shared_ptr<instantiated_math_function>> dependent_functions;
    
    for (auto && changed_channelconfig_ptr : definitively_changed_channels) {
      
      std::unordered_set<std::shared_ptr<instantiated_math_function>> & dependent_functions_of_this_channel = state->mathstatus.math_functions->all_dependencies_of_channel.at(changed_channelconfig_ptr->channelpath);

      for (auto && affected_math_function: dependent_functions_of_this_channel) {
	// merge into set 
        //dependent_functions.emplace(affected_math_function);

	// instead of merging into a set and doing it later we just go ahead and set the execution_demanded flag directly
	if (!affected_math_function->disabled || !affected_math_function->ondemand) {
	  state->mathstatus.function_status.at(affected_math_function).execution_demanded = true; 
	}
      }
      

    }

    // go over all of the dependent functions and mark their execution_demanded flag
    //for (auto && affected_math_function: dependent_functions) {
    //  state->mathstatus.function_status.at(affected_math_function).execution_demanded = true; 
    //}
  }

  static bool check_all_dependencies_mdonly(
					    std::shared_ptr<waveform_set_state> state,
					    std::unordered_set<std::shared_ptr<instantiated_math_function>> &unchecked_functions,
					    std::unordered_set<std::shared_ptr<instantiated_math_function>> &passed_mdonly_functions,
					    std::unordered_set<std::shared_ptr<instantiated_math_function>> &failed_mdonly_functions,
					    std::unordered_set<std::shared_ptr<instantiated_math_function>> &checked_regular_functions,
					    std::unordered_set<std::shared_ptr<instantiated_math_function>>::iterator &function_to_check_it)
  // returns true if function_to_check_it refers to an mdonly function with all dependencies also being mdonly
  // assumes state is admin-locked or still under construction and not needing locks. 
  {
    std::shared_ptr<instantiated_math_function> function_to_check = *function_to_check_it;
    unchecked_functions.erase(function_to_check_it);  // remove function_to_check from unchecked_functions
    
    if (!function_to_check->mdonly) {
      checked_regular_functions.emplace(function_to_check);
      return false;
    }

    bool all_mdonly = true;
    std::unordered_set<std::shared_ptr<instantiated_math_function>> &dependent_functions = state->mathstatus.math_functions->all_dependencies_of_function.at(function_to_check);

    for (auto && dependent_function : dependent_functions) {
      if (passed_mdonly_functions.find(dependent_function) != passed_mdonly_functions.end()) {
	// found previously passed function. all good
	continue; 
      }
      if (failed_mdonly_functions.find(dependent_function) != failed_mdonly_functions.end()) {
	// Found previously failed function. Failed.
	all_mdonly=false;
	break;
      }
      if (checked_regular_functions.find(dependent_function) != checked_regular_functions.end()) {
	// This dependent function is not mdonly so we are failed.
	all_mdonly=false;
	break;
      }

      std::unordered_set<std::shared_ptr<instantiated_math_function>>::iterator unchecked_it = unchecked_functions.find(dependent_function);
      if (unchecked_it != unchecked_functions.end()) {
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

  
  
  std::shared_ptr<globalrevision> active_transaction::end_transaction()
  // Warning: we may be called by the active_transaction destructor, so calling e.g. virtual methods on the active transaction
  // should be avoided.
  // Caller must ensure that all updating processes related to the transaction are complete. Therefore we don't have to worry about locking the current_transaction
  // ***!!!! Much of this needs to be refactored because it is mostly math-based and applicable to on-demand waveform groups.
  // (Maybe not; this just puts things into the channel_map so far, But we should factor out code
  // that would define a channel_map and status hash tables needed for an on_demand group)
  {
    std::set<std::shared_ptr<waveform_base>> waveforms_needing_finalization; // automatic waveforms created here that need to be marked as ready

    std::shared_ptr<wfmdatabase> wfmdb_strong=wfmdb.lock();

    if (!wfmdb_strong) {
      previous_globalrev=nullptr;
      return nullptr;
    }
    assert(wfmdb_strong->current_transaction);


    std::shared_ptr<globalrevision> globalrev;
    std::map<std::string,std::shared_ptr<channelconfig>> all_channels_by_name;

    // set of channels definitely changed, according to whether we've dispatched them in our graph search
    // for possibly dependent channels 
    std::unordered_set<std::shared_ptr<channelconfig>> changed_channels_need_dispatch;
    std::unordered_set<std::shared_ptr<channelconfig>> changed_channels_dispatched;

    // archive of definitively changed channels
    std::unordered_set<std::shared_ptr<channelconfig>> definitively_changed_channels;

    // set of channels not yet known to be changed
    std::unordered_set<std::shared_ptr<channelconfig>> maybechanged_channels;


    // set of ready channels
    std::unordered_set<channel_state *> ready_channels; // references into the globalrev->wfmstatus.channel_map
    // set of unchanged incomplete channels
    std::unordered_set<channel_state *> unchanged_incomplete_channels; // references into the globalrev->wfmstatus.channel_map
    std::unordered_set<channel_state *> unchanged_incomplete_mdonly_channels; // references into the globalrev->wfmstatus.channel_map
    
    {
      std::lock_guard<std::mutex> wfmdb_lock(wfmdb_strong->admin);
      // assemble the channel_map
      std::map<std::string,channel_state> initial_channel_map;
      for (auto && channel_name_chan_ptr : wfmdb_strong->_channels) {
	initial_channel_map.emplace(std::piecewise_construct,
				    std::forward_as_tuple(channel_name_chan_ptr.first),
				    std::forward_as_tuple(channel_name_chan_ptr.second,nullptr,false)); // tentatively mark channel_state as not updated
	
      }
      
      // build a class globalrevision from wfmdb->current_transaction
      globalrev = std::make_shared<globalrevision>(wfmdb_strong->current_transaction->globalrev,wfmdb_strong->_math_functions,initial_channel_map,previous_globalrev);
      //globalrev->wfmstatus.channel_map.reserve(wfmdb_strong->_channels.size());
      
      // Build a temporary map of the channels we still need to dispatch
      for (auto && channelname_channelpointer: wfmdb_strong->_channels) {
	std::shared_ptr<channelconfig> config = channelname_channelpointer.second->config();
	all_channels_by_name.emplace(std::piecewise_construct,
	std::forward_as_tuple(config->channelpath),
	std::forward_as_tuple(config));

	maybechanged_channels.emplace(config);

      }
      
    }

    // make sure hash tables won't rehash and screw up iterators or similar
    changed_channels_dispatched.reserve(maybechanged_channels.size());
    changed_channels_need_dispatch.reserve(maybechanged_channels.size());

    // mark all new channels/waveforms as changed_channels_need_dispatched
    for (auto && new_wfm_chanpath_ptr: wfmdb_strong->current_transaction->new_waveforms) {

      const std::string &chanpath=new_wfm_chanpath_ptr.first;
      
      std::shared_ptr<channelconfig> config = all_channels_by_name.at(chanpath);
      
      changed_channels_need_dispatch.emplace(config);
      definitively_changed_channels.emplace(config);
      maybechanged_channels.erase(config);

    }
    for (auto && updated_chan: wfmdb_strong->current_transaction->updated_channels) {
      std::shared_ptr<channelconfig> config = updated_chan->config();

      std::unordered_set<std::shared_ptr<channelconfig>>::iterator mcc_it = maybechanged_channels.find(config);

      if (mcc_it != maybechanged_channels.end()) {
	changed_channels_need_dispatch.emplace(config);
	definitively_changed_channels.emplace(config);
	maybechanged_channels.erase(mcc_it);
      }
    }

    // Now empty out changed_channels_need_dispatch, adding into it any dependencies of currently referenced changed channels
    // This progressively identifies all (possibly) changed channels
    std::unordered_set<std::shared_ptr<channelconfig>>::iterator channel_to_dispatch_it = changed_channels_need_dispatch.begin();
    
    while (channel_to_dispatch_it != changed_channels_need_dispatch.end()) {
      _identify_changed_channels(all_channels_by_name,*globalrev->mathstatus.math_functions, maybechanged_channels,changed_channels_dispatched,changed_channels_need_dispatch,channel_to_dispatch_it );

      channel_to_dispatch_it = changed_channels_need_dispatch.begin();
    }
    
    // now changed_channels_need_dispatch is empty, changed_channels_dispatched represents all changed or possibly-changed (i.e. math may decide presence of a new revision) channels,
    // and maybechanged_channels represents channels which are known to be definitively unchanged (not dependent directly or indirectly on anything that may have changed)
    std::unordered_set<std::shared_ptr<channelconfig>> &possibly_changed_channels_to_process=changed_channels_dispatched;
    std::unordered_set<std::shared_ptr<channelconfig>> &unchanged_channels=maybechanged_channels;


    std::unordered_set<std::shared_ptr<instantiated_math_function>> unchanged_complete_math_functions;
    std::unordered_set<std::shared_ptr<instantiated_math_function>> unchanged_complete_math_functions_mdonly;

    std::unordered_set<std::shared_ptr<instantiated_math_function>> unchanged_incomplete_math_functions;


    // Figure out which functions are actually mdonly...
    // Traverse graph from all mdonly_functions, checking that their dependencies are all mdonly. If not
    // they need to be moved from mdonly_pending_functions or completed_mdonly_functions into pending_functions
    std::unordered_set<std::shared_ptr<instantiated_math_function>> unchecked_functions;
    std::unordered_set<std::shared_ptr<instantiated_math_function>> passed_mdonly_functions;
    std::unordered_set<std::shared_ptr<instantiated_math_function>> failed_mdonly_functions;
    std::unordered_set<std::shared_ptr<instantiated_math_function>> checked_regular_functions;

    std::unordered_set<std::shared_ptr<instantiated_math_function>>::iterator unchecked_function = unchecked_functions.begin();
    while (unchecked_function != unchecked_functions.end()) {
      check_all_dependencies_mdonly(globalrev,unchecked_functions,passed_mdonly_functions,failed_mdonly_functions,checked_regular_functions,unchecked_function);

      unchecked_function = unchecked_functions.begin();
    }

    // any failed_mdonly_functions need to have their mdonly bool cleared in their math_function_status
    // and need to be moved from mdonly_pending_functions into pending_functions
    for (auto && failed_mdonly_function: failed_mdonly_functions) {
      globalrev->mathstatus.function_status.at(failed_mdonly_function).mdonly=false;
      globalrev->mathstatus.mdonly_pending_functions.erase(globalrev->mathstatus.mdonly_pending_functions.find(failed_mdonly_function));
      globalrev->mathstatus.pending_functions.emplace(failed_mdonly_function);
    }


    // Now reference previous revision of all unchanged channels, inserting into the new globalrev's channel_map
    // and also marking any corresponding function_status as complete
    for (auto && unchanged_channel: unchanged_channels) {
      std::shared_ptr<waveform_base> channel_wfm;
      std::shared_ptr<waveform_base> channel_wfm_is_complete;

      std::shared_ptr<instantiated_math_function> unchanged_channel_math_function;
      bool is_mdonly = false;
      
      if (unchanged_channel->math) {
	unchanged_channel_math_function = globalrev->mathstatus.math_functions->defined_math_functions.at(unchanged_channel->channelpath);
	is_mdonly = globalrev->mathstatus.function_status.at(unchanged_channel_math_function).mdonly;
	
      }
      
      assert(previous_globalrev); // Should always be present, because if we are on the first revision, each channel should have had a new waveform and therefore is changed!
      {
	std::lock_guard<std::mutex> previous_globalrev_admin(previous_globalrev->admin);
	
	std::map<std::string,channel_state>::iterator previous_globalrev_chanmap_entry = previous_globalrev->wfmstatus.channel_map.find(unchanged_channel->channelpath);
	if (previous_globalrev_chanmap_entry==previous_globalrev->wfmstatus.channel_map.end()) {
	  throw snde_error("Existing channel %s has no prior waveform",unchanged_channel->channelpath.c_str());
	}
	channel_wfm_is_complete = previous_globalrev_chanmap_entry->second.waveform_is_complete(is_mdonly);
	if (channel_wfm_is_complete) {
	  channel_wfm = channel_wfm_is_complete;
	} else {
	  channel_wfm = previous_globalrev_chanmap_entry->second.wfm();
	}
      }
      std::map<std::string,channel_state>::iterator channel_map_it;
      bool added_successfully;
      //std::tie(channel_map_it,added_successfully) =
      //globalrev->wfmstatus.channel_map.emplace(std::piecewise_construct,
      //std::forward_as_tuple(unchanged_channel->channelpath),
      //						 std::forward_as_tuple(unchanged_channel,channel_wfm,false)); // mark channel_state as not updated
      channel_map_it = globalrev->wfmstatus.channel_map.find(unchanged_channel->channelpath);
      channel_map_it->second._wfm = channel_wfm;
      
      assert(channel_map_it != globalrev->wfmstatus.channel_map.end());

      // check if added channel is complete
      if (channel_wfm_is_complete) {
      
	// if it is math, queue it to mark the function_status as complete, because if we are looking at it here, all
	// direct or indirect prerequisites must also be unchanged
	if (unchanged_channel->math) {
	  std::shared_ptr<instantiated_math_function> channel_math_function = globalrev->mathstatus.math_functions->defined_math_functions.at(unchanged_channel->channelpath);
	  if (is_mdonly) {
	    unchanged_complete_math_functions_mdonly.emplace(channel_math_function);
	  } else {
	    unchanged_complete_math_functions.emplace(channel_math_function);
	  }
	}
	// Queue notification that this channel is complete, except we are currently holding the admin lock, so we can't
	// do it now. Instead queue up a reference into the channel_map
	ready_channels.emplace(&channel_map_it->second);

	// place in globalrev->wfmstatus.completed_waveforms
	globalrev->wfmstatus.completed_waveforms.emplace(std::piecewise_construct,
							 std::forward_as_tuple((std::shared_ptr<channelconfig>)unchanged_channel),
							 std::forward_as_tuple((channel_state *)&channel_map_it->second));
      } else {
	// in this case channel_wfm may or may not exist and is NOT complete
	if (unchanged_channel->math) {
	  
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
    
    
    for (auto && unchanged_complete_math_function: unchanged_complete_math_functions) {
      // For all fully ready math functions with no inputs changed, mark them as complete
      // and put them into the appropriate completed set in math_status.

      math_function_status &ucmf_status = globalrev->mathstatus.function_status.at(unchanged_complete_math_function);
      ucmf_status.mdonly_executed=true; 
      ucmf_status.complete = true;
      // remove from the appropriate set

      auto mpf_it = globalrev->mathstatus.mdonly_pending_functions.find(unchanged_complete_math_function);
      if (mpf_it != globalrev->mathstatus.mdonly_pending_functions.end()) {

	globalrev->mathstatus.mdonly_pending_functions.erase(mpf_it);

      } else {
	auto pf_it = globalrev->mathstatus.pending_functions.find(unchanged_complete_math_function);
	assert(pf_it != globalrev->mathstatus.pending_functions.end());

	
	globalrev->mathstatus.pending_functions.erase(pf_it);
	
      }
      // add to the complete fully ready set. 
      globalrev->mathstatus.completed_functions.emplace(unchanged_complete_math_function);
      
      
    }
  
    for (auto && unchanged_complete_math_function: unchanged_complete_math_functions_mdonly) {
      // For all fully ready mdonly math functions with no inputs changed, mark them as complete
      // and put them into the appropriate completed set in math_status.
      
      globalrev->mathstatus.function_status.at(unchanged_complete_math_function).complete = true;
      assert(unchanged_complete_math_function->mdonly); // shouldn't be in this list if we aren't mdonly.
      
      auto mpf_it = globalrev->mathstatus.mdonly_pending_functions.find(unchanged_complete_math_function);
      assert(mpf_it != globalrev->mathstatus.mdonly_pending_functions.end());
	
      globalrev->mathstatus.mdonly_pending_functions.erase(mpf_it);

      globalrev->mathstatus.completed_mdonly_functions.emplace(unchanged_complete_math_function);
	
      
    }
  
    
    // First, if we have an instantiated new waveform, place this in the channel_map
    for (auto && new_wfm_chanpath_ptr: wfmdb_strong->current_transaction->new_waveforms) {
      std::shared_ptr<channelconfig> config = all_channels_by_name.at(new_wfm_chanpath_ptr.first);

      if (config->math) {
	// Not allowed to manually update a math channel
	throw snde_error("Manual instantiation of math channel %s",config->channelpath.c_str());
      }
      auto cm_it = globalrev->wfmstatus.channel_map.find(new_wfm_chanpath_ptr.first);
      assert(cm_it != globalrev->wfmstatus.channel_map.end());
      cm_it->second._wfm = new_wfm_chanpath_ptr.second;

      // assign the .revision field
      cm_it->second.revision = new_wfm_chanpath_ptr.second->info->revision;
      cm_it->second.updated=true; 
      
      //
      //						 emplace(std::piecewise_construct,
      //						    std::forward_as_tuple(new_wfm_chanpath_ptr.first),
      //						    std::forward_as_tuple(config,new_wfm_chanpath_ptr.second,true)).first; // mark updated=true
      
      // mark it as instantiated
      globalrev->wfmstatus.instantiated_waveforms.emplace(std::piecewise_construct,
						std::forward_as_tuple(config),
						std::forward_as_tuple(&cm_it->second));
      
      possibly_changed_channels_to_process.erase(config); // no further processing needed here
    }
    
    // Second, make sure if a channel was created, it has a waveform present and gets put in the channel_map
    for (auto && updated_chan: wfmdb_strong->current_transaction->updated_channels) {
      std::shared_ptr<channelconfig> config = updated_chan->config();
      std::shared_ptr<waveform_base> new_wfm;

      if (config->math) {
	continue; // math channels get their waveforms defined automatically
      }
      
      if (possibly_changed_channels_to_process.find(config)==possibly_changed_channels_to_process.end()) {
	// already processed above because an explicit new waveform was provided. All done here.
	continue;
      }
      
      auto new_waveform_it = wfmdb_strong->current_transaction->new_waveforms.find(config->channelpath);
      
      // new waveform should be required but not present; create one
      assert(wfmdb_strong->current_transaction->new_waveform_required.at(config->channelpath) && new_waveform_it==wfmdb_strong->current_transaction->new_waveforms.end());
      new_wfm = std::make_shared<waveform_base>(wfmdb_strong,updated_chan,config->owner_id,SNDE_WTN_FLOAT32); // constructor adds itself to current transaction
      waveforms_needing_finalization.emplace(new_wfm); // Since we provided this, we need to make it ready, below
      
      // insert new waveform into channel_map
      //auto cm_it = globalrev->wfmstatus.channel_map.emplace(std::piecewise_construct,
      //							    std::forward_as_tuple(config->channelpath),
      //std::forward_as_tuple(config,new_wfm,true)).first; // mark updated=true
      
      auto cm_it = globalrev->wfmstatus.channel_map.find(config->channelpath);
      assert(cm_it != globalrev->wfmstatus.channel_map.end());
      cm_it->second._wfm = new_wfm;
      cm_it->second.updated=true; 
      cm_it->second.revision = new_wfm->info->revision;
      
      // mark it as instantiated
      globalrev->wfmstatus.instantiated_waveforms.emplace(std::piecewise_construct,
						std::forward_as_tuple(config),
						std::forward_as_tuple(&cm_it->second));
      
      // remove this from channels_to_process, as it has been already been inserted into the channel_map
      possibly_changed_channels_to_process.erase(config);
    }

    for (auto && possibly_changed_channel: possibly_changed_channels_to_process) {
      // Go through the set of possibly_changed channels which were not processed above
      // (These must be math dependencies))
      // Since the math hasn't started, we don't have defined waveforms yet
      // so the waveform is just nullptr
      
      //auto cm_it = globalrev->wfmstatus.channel_map.emplace(std::piecewise_construct,
      //std::forward_as_tuple(possibly_changed_channel->channelpath),
      //							    std::forward_as_tuple(possibly_changed_channel,nullptr,false)).first; // mark updated as false (so far, at least)
      auto cm_it = globalrev->wfmstatus.channel_map.find(possibly_changed_channel->channelpath);
      assert(cm_it != globalrev->wfmstatus.channel_map.end());
      
      
      // These waveforms are defined but not instantiated
      globalrev->wfmstatus.defined_waveforms.emplace(std::piecewise_construct,
      						     std::forward_as_tuple(possibly_changed_channel),
      						     std::forward_as_tuple(&cm_it->second));      
      
    }
  

    // definitively_changed_channels drive the need to execute math functions,
    // including insertion of implicit and explicit self-dependencies into the
    // calculation graph. 
    mark_prospective_calculations(globalrev,definitively_changed_channels);

    // How do we splice implicit and explicit self-dependencies into the calculation graph???
    // Simple: We need to add to the _external_dependencies_on_[channel|function] of the previous_globalrev and
    // add to the missing_external_[channel|function]_prerequisites of this globalrev.

    // Define variable to store all self-dependencies and fill it up
    std::vector<std::shared_ptr<instantiated_math_function>> self_dependencies;
    for (auto && mathfunction_alldeps: globalrev->mathstatus.math_functions->all_dependencies_of_function) {
      // Enumerate all self-dependencies here
      std::shared_ptr<instantiated_math_function> mathfunction = mathfunction_alldeps.first;

      if (!globalrev->mathstatus.function_status.at(mathfunction).complete) { // Complete marked above, if none of the direct or indirect inputs has changed. If complete, nothing else matters. 
	if (mathfunction->fcn->self_dependent || mathfunction->fcn->mandatory_mutable || mathfunction->fcn->pure_optionally_mutable || mathfunction->fcn->new_revision_optional) {
	  // implicit or explicit self-dependency
	  self_dependencies.push_back(mathfunction);
	}
      }
      
    }

    
    //  Need to copy repetitive_notifies into place.
    {
      std::lock_guard<std::mutex> wfmdb_lock(wfmdb_strong->admin);

      for (auto && repetitive_notify: wfmdb_strong->repetitive_notifies) {
	// copy notify into this globalrev
	std::shared_ptr<channel_notify> chan_notify = repetitive_notify->create_notify_instance();

	// insert this notify into globalrev data structures.
	/// NOTE: ***!!! This could probably be simplified by leveraging apply_to_wss() method
	// but that would lose flexibility to ignore missing channels
	{
	  std::lock_guard<std::mutex> criteria_lock(chan_notify->criteria.admin);
	  for (auto && mdonly_channame : chan_notify->criteria.metadataonly_channels) {
	    auto channel_map_it = globalrev->wfmstatus.channel_map.find(mdonly_channame);
	    if (channel_map_it == globalrev->wfmstatus.channel_map.end()) {
	      std::string other_channels="";
	      for (auto && mdonly_channame2 : chan_notify->criteria.metadataonly_channels) {
		other_channels += mdonly_channame2+";";
	      }
	      snde_warning("MDOnly notification requested on non-existent channel %s; other MDOnly channels=%s. Ignoring.",mdonly_channame,other_channels);
	      continue;
	    }

	    channel_state &chanstate=channel_map_it->second;

	    // Add notification unless criterion already met
	    std::shared_ptr<waveform_base> wfm_is_complete = chanstate.waveform_is_complete(true);

	    if (!wfm_is_complete) {
	      // Criterion not met; add notification
	      chanstate._notify_about_this_channel_metadataonly->emplace(chan_notify);
	    }
	    
	  }


	  for (auto && fullyready_channame : chan_notify->criteria.fullyready_channels) {
	    auto channel_map_it = globalrev->wfmstatus.channel_map.find(fullyready_channame);
	    if (channel_map_it == globalrev->wfmstatus.channel_map.end()) {
	      std::string other_channels="";
	      for (auto && fullyready_channame2 : chan_notify->criteria.fullyready_channels) {
		other_channels += fullyready_channame2+";";
	      }
	      snde_warning("FullyReady notification requested on non-existent channel %s; other FullyReady channels=%s. Ignoring.",fullyready_channame,other_channels);
	      continue;
	    }
	    
	    channel_state &chanstate=channel_map_it->second;

	    // Add notification unless criterion already met
	    std::shared_ptr<waveform_base> wfm = chanstate.wfm();
	    int wfm_state = wfm->info_state;
	    
	    if (wfm && wfm_state==SNDE_WFMS_READY) {
	      // Criterion met. Nothing to do 
	    } else {
	      // Criterion not met; add notification
	      chanstate._notify_about_this_channel_ready->emplace(chan_notify);
	    }
	    

	    
	  }

	  
	}

	
      }
      
    }
    
    // globalrev->wfmstatus should have exactly one entry entry in the _waveforms maps
    // per channel_map entry
    bool all_ready;
    {
      // std::lock_guard<std::mutex> globalrev_admin(globalrev->admin);  // globalrev not yet published so holding lock is unnecessary
      assert(globalrev->wfmstatus.channel_map.size() == (globalrev->wfmstatus.defined_waveforms.size() + globalrev->wfmstatus.instantiated_waveforms.size() + globalrev->wfmstatus.metadataonly_waveforms.size() + globalrev->wfmstatus.completed_waveforms.size()));

      all_ready = !globalrev->wfmstatus.defined_waveforms.size() && !globalrev->wfmstatus.instantiated_waveforms.size();
    }



    
    // Iterate through all non-complete math functions, filling out missing_prerequisites 
    for (auto && mathfunction_alldeps: globalrev->mathstatus.math_functions->all_dependencies_of_function) {
      std::shared_ptr<instantiated_math_function> mathfunction = mathfunction_alldeps.first;

      if (mathfunction->disabled || mathfunction->ondemand) { // don't worry about disabled or on-demand functions
	continue;
      }
      
      // If mathfunction is unchanged from prior rev, then we don't worry about prerequisites as we will
      // just be copying its value from the prior rev
      if (unchanged_incomplete_math_functions.find(mathfunction) != unchanged_incomplete_math_functions.end()) {
	continue;
      }
      
      if (!globalrev->mathstatus.function_status.at(mathfunction).complete) {
	// This math function is not complete

	bool mathfunction_is_mdonly = false;
	if (mathfunction->mdonly) { //// to genuinely be mdonly our instantiated_math_function must be marked as such AND we should be in the mdonly_pending_functions set of the math_status (now there is a specific flag)
	  // mathfunction_is_mdonly = (globalrev->mathstatus.mdonly_pending_functions.find(mathfunction) != globalrev->mathstatus.mdonly_pending_functions.end());
	  mathfunction_is_mdonly = globalrev->mathstatus.function_status.at(mathfunction).mdonly;
	}
	// iterate over all parameters that are dependent on other channels
	for (auto && parameter: mathfunction->parameters) {

	  // get the set of prerequisite channels
	  std::set<std::string> prereq_channels = parameter->get_prerequisites(/*globalrev,*/mathfunction->channel_path_context);
	  for (auto && prereq_channel: prereq_channels) {

	    // for each prerequisite channel, look at it's state. 
	    channel_state &prereq_chanstate = globalrev->wfmstatus.channel_map.at(prereq_channel);

	    bool prereq_complete = false; 
	    
	    std::shared_ptr<waveform_base> prereq_wfm = prereq_chanstate.wfm();
	    int prereq_wfm_state = prereq_wfm->info_state;
	    
	    if (prereq_chanstate.config->math) {
	      std::shared_ptr<instantiated_math_function> math_prereq = globalrev->mathstatus.math_functions->defined_math_functions.at(prereq_channel);

	      if (mathfunction_is_mdonly && math_prereq->mdonly && prereq_wfm && (prereq_wfm_state == SNDE_WFMS_METADATAREADY || prereq_wfm_state==SNDE_WFMS_READY)) {
		// If we are mdonly, and the prereq is mdonly and the waveform exists and is metadataready or fully ready,
		// then this prerequisite is complete and nothing is needed.
		
		prereq_complete = true; 
	      }
	    }
	    if (prereq_wfm && prereq_wfm_state == SNDE_WFMS_READY) {
	      // If the waveform exists and is fully ready,
	      // then this prerequisite is complete and nothing is needed.
	      prereq_complete = true; 
	    }
	    if (!prereq_complete) {
	      // Prerequisite is not complete; Need to mark this in the missing_prerequisites of the math_function_status
	      globalrev->mathstatus.function_status.at(mathfunction).missing_prerequisites.emplace(prereq_chanstate.config);
	    }
	    
	    
	  }
	}
      }
    }
    
    // add self-dependencies to the missing_external_prerequisites of this globalrev
    for (auto && self_dep : self_dependencies) {
      globalrev->mathstatus.function_status.at(self_dep).missing_external_function_prerequisites.emplace(std::make_tuple(previous_globalrev,self_dep));
    }

    
    
    // add self-dependencies to the _external_dependencies_on_function of the previous_globalrev
    // Note from hereon we have published our new globalrev so we have to be a bit more careful about
    // locking it because we might get notification callbacks or similar if one of those external waveforms becomes ready
    std::vector<std::shared_ptr<instantiated_math_function>> need_to_check_if_ready;
    {
      std::lock_guard<std::mutex> previous_globalrev_admin(previous_globalrev->admin);

      std::shared_ptr<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>>>> new_prevglob_extdep = previous_globalrev->mathstatus.begin_atomic_external_dependencies_on_function_update();

      for (auto && self_dep : self_dependencies) {
	// Check to make sure previous globalrev even has this exact math function.
	// if so it should be a key in the in the mathstatus.math_functions->all_dependencies_of_function unordered_map
	if (previous_globalrev->mathstatus.math_functions->all_dependencies_of_function.find(self_dep) != previous_globalrev->mathstatus.math_functions->all_dependencies_of_function.end()) {
	  // Got it.
	  // Check the status -- may have changed since the above.
	  if (previous_globalrev->mathstatus.function_status.at(self_dep).complete) {
	    
	    // complete -- perform ready check on this waveform
	    need_to_check_if_ready.push_back(self_dep);
	  } else {
	    //add to previous globalrev's _external_dependencies
	    new_prevglob_extdep->at(self_dep).push_back(std::make_tuple(globalrev,self_dep));
	  }
	}
      }
      previous_globalrev->mathstatus.end_atomic_external_dependencies_on_function_update(new_prevglob_extdep);
      
      
    }

    std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>> ready_to_execute;

    for (auto && readycheck : need_to_check_if_ready) {
      std::lock_guard<std::mutex> globalrev_admin(globalrev->admin);
      math_function_status &readycheck_status = globalrev->mathstatus.function_status.at(readycheck);

      globalrev->mathstatus.check_dep_fcn_ready(globalrev,readycheck,&readycheck_status,ready_to_execute);
    }
    
    // Go through unchanged_incomplete_channels and unchanged_incomplete_mdonly_channels and get notifies when these become complete
    {
      for (auto && chanstate: unchanged_incomplete_channels) { // chanstate is a channel_state &
	channel_state &previous_state = previous_globalrev->wfmstatus.channel_map.at(chanstate->config->channelpath);
	std::shared_ptr<_unchanged_channel_notify> unchangednotify=std::make_shared<_unchanged_channel_notify>(wfmdb_strong,globalrev,previous_state,*chanstate,false);
	unchangednotify->apply_to_wss(previous_globalrev); 
	/*this code replaced by apply_to_wss()
	std::unique_lock<std::mutex> previous_globalrev_admin(previous_globalrev->admin);

	// queue up notification
	std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_set = previous_state.begin_atomic_notify_about_this_channel_ready_update();
	notify_set->emplace(unchangednotify);
	previous_state.end_atomic_notify_about_this_channel_ready_update(notify_set);
	
	unchangednotify->check_all_criteria(globalrev);
	*/
      }

      for (auto && chanstate: unchanged_incomplete_mdonly_channels) { // chanstate is a channel_state &
	channel_state &previous_state = previous_globalrev->wfmstatus.channel_map.at(chanstate->config->channelpath);
	std::shared_ptr<_unchanged_channel_notify> unchangednotify=std::make_shared<_unchanged_channel_notify>(wfmdb,globalrev,previous_state,*chanstate,true);
	unchangednotify->apply_to_wss(previous_globalrev); 

	/* this code replaced by apply_to_wss()
	std::unique_lock<std::mutex> previous_globalrev_admin(previous_globalrev->admin);
	
	// queue up notification
	std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_set = previous_state.begin_atomic_notify_about_this_channel_metadataonly_update();
	notify_set->emplace(unchangednotify);
	previous_state.end_atomic_notify_about_this_channel_metadataonly_update(notify_set);

	unchangednotify->check_all_criteria(globalrev);
	*/
	
      }

    }
    // ***!!! Need to got through unchanged_incomplete_math_functions and get notifies when these become complete, if necessary (?)
    // Think it should be unnecessary. 
    {
      std::lock_guard<std::mutex> wfmdb_lock(wfmdb_strong->admin);
      wfmdb_strong->_globalrevs.emplace(wfmdb_strong->current_transaction->globalrev,globalrev);

      // atomic update of _latest_globalrev
      std::atomic_store(&wfmdb_strong->_latest_globalrev,globalrev);
      
      wfmdb_strong->current_transaction = nullptr; 
      assert(!transaction_ended);
      transaction_ended=true;
      transaction_lock_holder.unlock();
    }

    // Perform notifies that unchanged copied waveforms from prior revs are now ready
    // (and that globalrev is ready if there is nothing pending!)
    for (auto && readychan : ready_channels) { // readychan is a channel_state &
      readychan->issue_nonmath_notifications(globalrev);
    }

    // queue up everything we marked as ready_to_execute
    for (auto && ready_wss_ready_fcn: ready_to_execute) {
      // Need to queue as a pending_computation
      std::shared_ptr<waveform_set_state> ready_wss;
      std::shared_ptr<instantiated_math_function> ready_fcn;

      std::tie(ready_wss,ready_fcn) = ready_wss_ready_fcn;
      wfmdb_strong->compute_resources.queue_computation(ready_wss,ready_fcn);
    }


    // Check if everything is done; issue notification
    if (all_ready) {
      std::unique_lock<std::mutex> wss_admin(globalrev->admin);
      std::unordered_set<std::shared_ptr<channel_notify>> waveformset_complete_notifiers=std::move(globalrev->waveformset_complete_notifiers);
      globalrev->waveformset_complete_notifiers.clear();
      wss_admin.unlock();

      for (auto && channel_notify_ptr: waveformset_complete_notifiers) {
	channel_notify_ptr->notify_waveformset_complete();
      }
    }

    // get notified so as to remove entries from available_compute_resource_database blocked_list
    // once the previous globalrev is complete.
    std::shared_ptr<_previous_globalrev_done_notify> prev_done_notify = std::make_shared<_previous_globalrev_done_notify>(wfmdb,previous_globalrev,globalrev);
    {
      std::lock_guard<std::mutex> prev_admin(previous_globalrev->admin);
      previous_globalrev->waveformset_complete_notifiers.emplace(prev_done_notify);
    }

    return globalrev;
  }
  
  active_transaction::~active_transaction()
  {
    //wfmdb->_transaction_raii_holder=nullptr;
    if (!transaction_ended) {
      end_transaction();
    }
  }


  channelconfig::channelconfig(std::string channelpath, std::string owner_name, void *owner_id,bool hidden,std::shared_ptr<waveform_storage_manager> storage_manager) : // storage_manager parameter defaults to nullptr
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

  channel_notification_criteria::channel_notification_criteria() :
    waveformset_complete(false)
  {
    
  }

  // copy assignment operator -- copies but ignores non-copyable mutex
  channel_notification_criteria & channel_notification_criteria::operator=(const channel_notification_criteria &orig)
  {
    std::lock_guard<std::mutex> orig_admin(orig.admin);

    waveformset_complete = orig.waveformset_complete;
    metadataonly_channels = orig.metadataonly_channels;
    fullyready_channels = orig.fullyready_channels;

    return *this;
  }
  
  // copy constructor -- copies but ignores non-copyable mutex
  channel_notification_criteria::channel_notification_criteria(const channel_notification_criteria &orig)
  {
    std::lock_guard<std::mutex> orig_admin(orig.admin);

    waveformset_complete = orig.waveformset_complete;
    metadataonly_channels = orig.metadataonly_channels;
    fullyready_channels = orig.fullyready_channels;
    
  }

  void channel_notification_criteria::add_waveformset_complete() 
  {
    // only allowed during creation so we don't worry about locking
    waveformset_complete=true;
    
  }

  void channel_notification_criteria::add_fullyready_channel(std::string channelname) 
  {
    // only allowed during creation so we don't worry about locking
    fullyready_channels.emplace(channelname);    
  }

  void channel_notification_criteria::add_metadataonly_channel(std::string channelname) 
  {
    // only allowed during creation so we don't worry about locking
    metadataonly_channels.emplace(channelname);    
  }

  channel_notify::channel_notify() :
    criteria()
  {

  }
  
  channel_notify::channel_notify(const channel_notification_criteria &criteria_to_copy) :
    criteria(criteria_to_copy)
  {
    
  }

  void channel_notify::notify_metadataonly(const std::string &channelpath) // notify this notifier that the given channel has satisified metadataonly (not usually modified by subclass)
  {

    bool generate_notify=false;
    {
      std::lock_guard<std::mutex> criteria_admin(criteria.admin);
      
      std::unordered_set<std::string>::iterator mdo_channels_it = criteria.metadataonly_channels.find(channelpath);
      
      assert(mdo_channels_it != criteria.metadataonly_channels.end()); // should only be able to do a notify on something in the set!
      criteria.metadataonly_channels.erase(mdo_channels_it);

      if (!criteria.metadataonly_channels.size() && !criteria.fullyready_channels.size() && !criteria.waveformset_complete) {
	// all criteria removed; ready for notification
	generate_notify=true;
      }
    }
    if (generate_notify) {
      perform_notify();
    }
  }
  
  void channel_notify::notify_ready(const std::string &channelpath) // notify this notifier that the given channel has satisified ready (not usually modified by subclass)
  {
    bool generate_notify=false;
    {
      std::lock_guard<std::mutex> criteria_admin(criteria.admin);
      
      std::unordered_set<std::string>::iterator fullyready_channels_it = criteria.fullyready_channels.find(channelpath);
      
      assert(fullyready_channels_it != criteria.fullyready_channels.end()); // should only be able to do a notify on something in the set!
      criteria.fullyready_channels.erase(fullyready_channels_it);

      if (!criteria.metadataonly_channels.size() && !criteria.fullyready_channels.size() && !criteria.waveformset_complete) {
	// all criteria removed; ready for notification
	generate_notify=true;
      }
    }
    if (generate_notify) {
      perform_notify();
    }
  }
  
  void channel_notify::notify_waveformset_complete() // notify this notifier that all waveforms in this set are complete.
  {
    bool generate_notify=false;
    {
      std::lock_guard<std::mutex> criteria_admin(criteria.admin);

      assert(criteria.waveformset_complete);
      criteria.waveformset_complete=false; // criterion is now satisfied, so we no longer need to wait for it

      
      if (!criteria.metadataonly_channels.size() && !criteria.fullyready_channels.size() && !criteria.waveformset_complete) {
	// all criteria removed; ready for notification
	generate_notify=true;
      }
    }
    if (generate_notify) {
      perform_notify();
    }

  }

  void channel_notify::check_waveformset_complete(std::shared_ptr<waveform_set_state> wss)
  {
    bool generate_notify=false;

    {
      std::lock_guard<std::mutex> wss_admin(wss->admin);
      std::lock_guard<std::mutex> criteria_admin(criteria.admin);
      
      // check if all waveforms are ready;
      bool all_ready = !wss->wfmstatus.defined_waveforms.size() && !wss->wfmstatus.instantiated_waveforms.size();      
      
      if (criteria.waveformset_complete && all_ready) {
	criteria.waveformset_complete = false;
	wss->waveformset_complete_notifiers.erase(shared_from_this());
      }
      
      if (!criteria.metadataonly_channels.size() && !criteria.fullyready_channels.size() && !criteria.waveformset_complete) {
	// all criteria removed; ready for notification
	generate_notify=true;
      }
    }
    if (generate_notify) {
      perform_notify();
    }
    
    
  }
  bool channel_notify::_check_all_criteria_locked(std::shared_ptr<waveform_set_state> wss,bool notifies_already_applied_to_wss)
  // Internal only: Should be called with wss admin lock and criteria admin locks locked. Returns true if an immediate notification is due
  {
    bool generate_notify=false;
    std::vector<std::string> mdonly_satisfied;
    std::vector<std::string> fullyready_satisfied;
    
    for (auto && md_channelname: criteria.metadataonly_channels) {
      channel_state & chanstate = wss->wfmstatus.channel_map.at(md_channelname);
      
      if (chanstate.waveform_is_complete(true)) {
	if (notifies_already_applied_to_wss) {
	  std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_metadataonly = chanstate.begin_atomic_notify_about_this_channel_metadataonly_update();	  
	  notify_about_this_channel_metadataonly->erase(shared_from_this());
	  chanstate.end_atomic_notify_about_this_channel_metadataonly_update(notify_about_this_channel_metadataonly);
	}
	mdonly_satisfied.push_back(md_channelname);
      }
    }
    
    for (auto && fr_channelname: criteria.fullyready_channels) {
      channel_state & chanstate = wss->wfmstatus.channel_map.at(fr_channelname);
      
      if (chanstate.waveform_is_complete(false)) {
	if (notifies_already_applied_to_wss) {
	  std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_ready = chanstate.begin_atomic_notify_about_this_channel_ready_update();
	  
	  notify_about_this_channel_ready->erase(shared_from_this());
	  chanstate.end_atomic_notify_about_this_channel_ready_update(notify_about_this_channel_ready);
	}
	fullyready_satisfied.push_back(fr_channelname);
	
      }
    }
    
    // check if all waveforms are ready;
    bool all_ready = !wss->wfmstatus.defined_waveforms.size() && !wss->wfmstatus.instantiated_waveforms.size();
    
    // update criteria according to satisfied conditions
    for (auto && md_channelname: mdonly_satisfied) {
      criteria.metadataonly_channels.erase(md_channelname);
    }
    
    for (auto && fr_channelname: fullyready_satisfied) {
      criteria.fullyready_channels.erase(fr_channelname);
    }
    
    if (criteria.waveformset_complete && all_ready) {
      criteria.waveformset_complete = false; 
      if (notifies_already_applied_to_wss) {
	wss->waveformset_complete_notifiers.erase(shared_from_this());
      }
    }
    
    if (!criteria.metadataonly_channels.size() && !criteria.fullyready_channels.size() && !criteria.waveformset_complete) {
      // all criteria removed; ready for notification
      generate_notify=true;
    }

    return generate_notify;
  }
  
    
  

  void channel_notify::check_all_criteria(std::shared_ptr<waveform_set_state> wss)
  {
    bool generate_notify=false;
	

    {
      std::lock_guard<std::mutex> wss_admin(wss->admin);
      std::lock_guard<std::mutex> criteria_admin(criteria.admin);

      generate_notify=_check_all_criteria_locked(wss,true);
      
      
    }
    
    if (generate_notify) {
      perform_notify();
    }

  }
  
  std::shared_ptr<channel_notify> channel_notify::notify_copier()
  {
    throw snde_error("Copier must be provided for repetitive channel notifications");
    return nullptr;
  }
  
  void channel_notify::apply_to_wss(std::shared_ptr<waveform_set_state> wss) // apply this notification process to a particular waveform_set_state. WARNING: May trigger the notification immediately
  {
    bool generate_notify;
    {
      std::lock_guard<std::mutex> wss_admin(wss->admin);
      std::lock_guard<std::mutex> criteria_admin(criteria.admin);

      generate_notify=_check_all_criteria_locked(wss,false);
      
      // Add criteria to this waveform set state


      for (auto && md_channelname: criteria.metadataonly_channels) {
	channel_state & chanstate = wss->wfmstatus.channel_map.at(md_channelname);
      
	std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_metadataonly = chanstate.begin_atomic_notify_about_this_channel_metadataonly_update();	  
	notify_about_this_channel_metadataonly->emplace(shared_from_this());
	chanstate.end_atomic_notify_about_this_channel_metadataonly_update(notify_about_this_channel_metadataonly);		
      }


      
      for (auto && fr_channelname: criteria.fullyready_channels) {
	channel_state & chanstate = wss->wfmstatus.channel_map.at(fr_channelname);
      
	std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_ready = chanstate.begin_atomic_notify_about_this_channel_ready_update();
	  
	notify_about_this_channel_ready->emplace(shared_from_this());
	chanstate.end_atomic_notify_about_this_channel_ready_update(notify_about_this_channel_ready);
      }
    }

      
    if (criteria.waveformset_complete) {
      wss->waveformset_complete_notifiers.emplace(shared_from_this());
    }
    
    
    if (generate_notify) {
      perform_notify();
    }
  }

  std::shared_ptr<channel_notify> repetitive_channel_notify::create_notify_instance()
  // this default implementation uses the channel_notify's notify_copier() to create the instance
  {
    return notify->notify_copier();
  }

  promise_channel_notify::promise_channel_notify(const std::vector<std::string> &mdonly_channels,const std::vector<std::string> &ready_channels)
  {
    for (auto && mdonly_channel: mdonly_channels) {
      criteria.add_metadataonly_channel(mdonly_channel);
    }
    for (auto && ready_channel: ready_channels) {
      criteria.add_fullyready_channel(ready_channel);
    }
  }

  void promise_channel_notify::perform_notify()
  {
    promise.set_value();
  }
  
  _unchanged_channel_notify::_unchanged_channel_notify(std::weak_ptr<wfmdatabase> wfmdb,std::shared_ptr<globalrevision> subsequent_globalrev,channel_state &current_channelstate,channel_state & sg_channelstate,bool mdonly) :
    wfmdb(wfmdb),
    subsequent_globalrev(subsequent_globalrev),
    current_channelstate(current_channelstate),
    sg_channelstate(sg_channelstate)
  {
    if (mdonly) {
      criteria.add_metadataonly_channel(sg_channelstate.config->channelpath);
    } else {
      criteria.add_fullyready_channel(sg_channelstate.config->channelpath);
    }
  }
  
  void _unchanged_channel_notify::perform_notify()
  {
    {
      std::lock_guard<std::mutex> subsequent_globalrev_admin(subsequent_globalrev->admin);
    
      // Pass completed waveform from this channel_state to subsequent_globalrev's channelstate
      sg_channelstate.end_atomic_wfm_update(current_channelstate.wfm());
    }  
    sg_channelstate.issue_nonmath_notifications(subsequent_globalrev);

    std::shared_ptr<wfmdatabase> wfmdb_strong=wfmdb.lock();
    if (wfmdb_strong) {
      sg_channelstate.issue_math_notifications(wfmdb_strong,subsequent_globalrev);
    }
  }
    
  channel_state::channel_state(std::shared_ptr<channel> chan,std::shared_ptr<waveform_base> wfm,bool updated) :
    _channel(chan),
    config(_channel->config()),
    _wfm(nullptr),
    updated(updated),
    revision(0)
  {
    // warning/note: wfmdb may be locked when this constructor is called. (called within end_transaction to create a prototype that is later copied into the globalrevision structure. 
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_nullptr;
    std::atomic_store(&_wfm,wfm);
    std::atomic_store(&_notify_about_this_channel_metadataonly,notify_nullptr);
    std::atomic_store(&_notify_about_this_channel_ready,notify_nullptr);
  }

  channel_state::channel_state(const channel_state &orig) :
    config(orig.config),
    _wfm(nullptr),
    updated((bool)orig.updated)
  {
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_nullptr;

    assert(!orig._notify_about_this_channel_metadataonly);
    assert(!orig._notify_about_this_channel_ready);
    
    std::atomic_store(&_wfm,orig.wfm());
    std::atomic_store(&_notify_about_this_channel_metadataonly,notify_nullptr);
    std::atomic_store(&_notify_about_this_channel_ready,notify_nullptr);
  }

  std::shared_ptr<waveform_base> channel_state::wfm() const
  {
    return std::atomic_load(&_wfm);
  }

  std::shared_ptr<waveform_base> channel_state::waveform_is_complete(bool mdonly)
  {
    std::shared_ptr<waveform_base> retval = wfm();
    if (retval) {
      int info_state = retval->info_state;
      if (mdonly) {
	if (info_state == SNDE_WFMS_METADATAREADY || info_state==SNDE_WFMS_READY) {
	  return retval;
	} else {
	  return nullptr;
	}
      } else {
	if (info_state==SNDE_WFMS_READY) {
	  return retval;
	} else {
	  return nullptr;
	}	
      }
    }
    return nullptr;
  }

  void channel_state::issue_nonmath_notifications(std::shared_ptr<waveform_set_state> wss) // wss is used to lock this channel_state object
  // Must be called without anything locked. Issue notifications requested in _notify* and remove those notification requests,
  // based on the channel_state's current status
  {

    // !!!*** This code is largely redundant with waveform::mark_as_ready, and the excess should probably be removed ***!!!

    
    // Issue metadataonly notifications
    if (waveform_is_complete(true)) {
      // at least complete through mdonly
      std::unique_lock<std::mutex> wss_admin(wss->admin);
      std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> local_notify_about_this_channel_metadataonly = begin_atomic_notify_about_this_channel_metadataonly_update();
      if (local_notify_about_this_channel_metadataonly) {
	end_atomic_notify_about_this_channel_metadataonly_update(nullptr); // clear out notification list pointer
	wss_admin.unlock();
	
	// perform notifications
	for (auto && channel_notify_ptr: *local_notify_about_this_channel_metadataonly) {
	  channel_notify_ptr->notify_metadataonly(config->channelpath);
	}
      }
    }
    
    // Issue ready notifications
    if (waveform_is_complete(false)) {
      
      std::unique_lock<std::mutex> wss_admin(wss->admin);
      std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> local_notify_about_this_channel_ready = begin_atomic_notify_about_this_channel_ready_update();
      if (local_notify_about_this_channel_ready) {
	end_atomic_notify_about_this_channel_ready_update(nullptr); // clear out notification list pointer
	wss_admin.unlock();
	
	// perform notifications
	for (auto && channel_notify_ptr: *local_notify_about_this_channel_ready) {
	  channel_notify_ptr->notify_ready(config->channelpath);
	}
      }
      
    }

    bool all_ready=false;
    {
      std::lock_guard<std::mutex> wss_admin(wss->admin);
      all_ready = !wss->wfmstatus.defined_waveforms.size() && !wss->wfmstatus.instantiated_waveforms.size();      
    }
    
    if (all_ready) {
      std::unordered_set<std::shared_ptr<channel_notify>> wss_complete_notifiers_copy;
      {
	std::lock_guard<std::mutex> wss_admin(wss->admin);
	wss_complete_notifiers_copy = wss->waveformset_complete_notifiers;
      }
      for (auto && notifier: wss_complete_notifiers_copy) {
	notifier->check_waveformset_complete(wss);
      }
    }
  }

  
  void channel_state::issue_math_notifications(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<waveform_set_state> wss) // wss is used to lock this channel_state object
  // Must be called without anything locked. Issue notifications requested in _notify* and remove those notification requests,
  // based on the channel_state's current status
  {

    std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>> ready_to_execute;
    
    // Issue metadataonly notifications
    bool got_mdonly = (bool)waveform_is_complete(true);
    bool got_fullyready = (bool)waveform_is_complete(false);
    if (got_mdonly || got_fullyready) {
      // at least complete through mdonly
      
      for (auto && dep_fcn: wss->mathstatus.math_functions->all_dependencies_of_channel.at(config->channelpath)) {
	// dep_fcn is a shared_ptr to an instantiated_math_function
        std::lock_guard<std::mutex> wss_admin(wss->admin);
	
	std::set<std::shared_ptr<channelconfig>>::iterator missing_prereq_it;
	math_function_status &dep_fcn_status = wss->mathstatus.function_status.at(dep_fcn);
	if (!dep_fcn_status.execution_demanded) {
	  continue;  // ignore unless we are about execution
	}

	// If the dependent function is mdonly then we only have to be mdonly. Otherwise we have to be fullyready
	if (got_fullyready || (got_mdonly && dep_fcn_status.mdonly)) {

	  // Remove us as a missing prerequisite
	  missing_prereq_it = dep_fcn_status.missing_prerequisites.find(config);
	  if (missing_prereq_it != dep_fcn_status.missing_prerequisites.end()) {
	    dep_fcn_status.missing_prerequisites.erase(missing_prereq_it);	  
	  }
	  
	  wss->mathstatus.check_dep_fcn_ready(wss,dep_fcn,&dep_fcn_status,ready_to_execute);
	}
      }
      for (auto && wss_extdepfcn: wss->mathstatus.external_dependencies_on_channel()->at(config)) {
	std::shared_ptr<waveform_set_state> &dependent_wss = std::get<0>(wss_extdepfcn);
	std::shared_ptr<instantiated_math_function> &dependent_func = std::get<1>(wss_extdepfcn);

	std::lock_guard<std::mutex> dependent_wss_admin(dependent_wss->admin);
	math_function_status &function_status = dependent_wss->mathstatus.function_status.at(dependent_func);
	std::set<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<channelconfig>>>::iterator
	  dependent_prereq_it = function_status.missing_external_channel_prerequisites.find(std::make_tuple(wss,config));

	if (dependent_prereq_it != function_status.missing_external_channel_prerequisites.end()) {
	  function_status.missing_external_channel_prerequisites.erase(dependent_prereq_it);
	  
	}
	dependent_wss->mathstatus.check_dep_fcn_ready(dependent_wss,dependent_func,&function_status,ready_to_execute);
	
      }
      
    }


    for (auto && ready_wss_ready_fcn: ready_to_execute) {
      // Need to queue as a pending_computation
      std::shared_ptr<waveform_set_state> ready_wss;
      std::shared_ptr<instantiated_math_function> ready_fcn;

      std::tie(ready_wss,ready_fcn) = ready_wss_ready_fcn;
      wfmdb->compute_resources.queue_computation(ready_wss,ready_fcn);
    }
  }

  void channel_state::end_atomic_wfm_update(std::shared_ptr<waveform_base> new_waveform)
  {
    std::atomic_store(&_wfm,new_waveform);

  }



  std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> channel_state::begin_atomic_notify_about_this_channel_metadataonly_update()
  {
    return std::make_shared<std::unordered_set<std::shared_ptr<channel_notify>>>(*notify_about_this_channel_metadataonly());
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
    return std::make_shared<std::unordered_set<std::shared_ptr<channel_notify>>>(*notify_about_this_channel_ready());
  }
  
  void channel_state::end_atomic_notify_about_this_channel_ready_update(std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> newval)
  {
    std::atomic_store(&_notify_about_this_channel_ready,newval);
  }
  
  std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> channel_state::notify_about_this_channel_ready()
  {
    return std::atomic_load(&_notify_about_this_channel_ready);
  }


  waveform_status::waveform_status(const std::map<std::string,channel_state> & channel_map_param) :
    channel_map(channel_map_param)
  {

  }

  waveform_set_state::waveform_set_state(const instantiated_math_database &math_functions,const std::map<std::string,channel_state> & channel_map_param,std::shared_ptr<waveform_set_state> prereq_state) :
    ready(false),
    wfmstatus(channel_map_param),
    mathstatus(std::make_shared<instantiated_math_database>(math_functions),channel_map_param),
    _prerequisite_state(nullptr)
  {
    std::atomic_store(&_prerequisite_state,prereq_state);
  }

  std::shared_ptr<waveform_set_state> waveform_set_state::prerequisite_state()
  {
    return std::atomic_load(&_prerequisite_state);
  }

  // sets the prerequisite state to nullptr
  void waveform_set_state::atomic_prerequisite_state_clear()
  {
    std::shared_ptr<waveform_set_state> null_prerequisite;    
    std::atomic_store(&_prerequisite_state,null_prerequisite);

  }

  
  globalrevision::globalrevision(uint64_t globalrev, const instantiated_math_database &math_functions,const std::map<std::string,channel_state> & channel_map_param,std::shared_ptr<waveform_set_state> prereq_state) :
    waveform_set_state(math_functions,channel_map_param,prereq_state),
    globalrev(globalrev)
  {
    
  }


  wfmdatabase::wfmdatabase()
  {
    std::shared_ptr<globalrevision> null_globalrev;
    std::atomic_store(&_latest_globalrev,null_globalrev);
  }
  
  
  std::shared_ptr<active_transaction> wfmdatabase::start_transaction()
  {
    return std::make_shared<active_transaction>(shared_from_this());
  }
  
  std::shared_ptr<globalrevision> wfmdatabase::end_transaction(std::shared_ptr<active_transaction> act_trans)
  {
    return act_trans->end_transaction();
  }

  std::shared_ptr<globalrevision> wfmdatabase::latest_globalrev()
  {
    return std::atomic_load(&_latest_globalrev);
  }

  std::shared_ptr<channel> wfmdatabase::reserve_channel(std::shared_ptr<channelconfig> new_config)
  {
    // Note that this is called with transaction lock held, but that is OK because transaction lock precedes wfmdb admin lock
    std::shared_ptr<channel> new_chan;
    {
      std::lock_guard<std::mutex> wfmdb_lock(admin);
      
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
	  // OK to lock channel because channel locks are after the wfmdb lock we already hold in the locking order
	  std::lock_guard<std::mutex> channel_lock(new_chan->admin);
	  assert(!new_chan->config()); // should be nullptr
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

      // verify waveform not already updated in current transaction
      if (current_transaction->new_waveforms.find(new_config->channelpath) != current_transaction->new_waveforms.end()) {
	throw snde_error("Replacing owner of channel %s in transaction where waveform already updated",new_config->channelpath);
      }
      
      current_transaction->updated_channels.emplace(new_chan);
      current_transaction->new_waveform_required.emplace(new_config->channelpath,true);
    }

    
    
    return new_chan;
  }

  //  void wfmdatabase::wait_waveforms(std::share_ptr<waveform_set_state> wss, const std::vector<std::shared_ptr<waveform_base>> &metadataonly,const std::vector<std::shared_ptr<waveform_base>> &ready)
  //// NOTE: python wrapper needs to drop thread context during wait and poll to check for connection drop
  // {
  //#error not implemented
  //}
  
  void wfmdatabase::wait_waveform_names(std::shared_ptr<waveform_set_state> wss,const std::vector<std::string> &metadataonly, const std::vector<std::string> fullyready)
  // NOTE: python wrapper needs to drop thread context during wait and poll to check for connection drop
  {
    // should queue up a std::promise for central dispatch
    // then we wait here polling the corresponding future with a timeout so we can accommodate dropped
    // connections.

    std::shared_ptr<promise_channel_notify> promise_notify=std::make_shared<promise_channel_notify>(metadataonly,fullyready);

    promise_notify->apply_to_wss(wss);

    std::future<void> criteria_satisfied = promise_notify->promise.get_future();
    criteria_satisfied.wait();
  }

  _previous_globalrev_done_notify::_previous_globalrev_done_notify(std::weak_ptr<wfmdatabase> wfmdb,std::shared_ptr<globalrevision> previous_globalrev,std::shared_ptr<globalrevision> current_globalrev) :
    wfmdb(wfmdb),
    previous_globalrev(previous_globalrev),
    current_globalrev(current_globalrev)
  {
    criteria.add_waveformset_complete();
  }
    
  void _previous_globalrev_done_notify::perform_notify()
  {
    {
      std::shared_ptr<wfmdatabase> wfmdb_strong=wfmdb.lock();
      if (!wfmdb_strong) return; 
      std::unique_lock<std::mutex> computeresources_admin(*wfmdb_strong->compute_resources.admin);

      // go through compute_resources blocked_list, removing the blocked computations
      // and queueing them up. 
      std::multimap<uint64_t,std::shared_ptr<pending_computation>>::iterator blocked_it; 
      while ((blocked_it = wfmdb_strong->compute_resources.blocked_list.begin()) != wfmdb_strong->compute_resources.blocked_list.end() && blocked_it->first <= previous_globalrev->globalrev) {
	std::shared_ptr<pending_computation> blocked_computation = blocked_it->second;
	wfmdb_strong->compute_resources.blocked_list.erase(blocked_it);
	computeresources_admin.unlock();
	wfmdb_strong->compute_resources._queue_computation_internal(blocked_computation);
	computeresources_admin.lock();
      }
    }

    // clear previous_globalrev's prerequisite state now that previous_globalrev is entirely ready
    previous_globalrev->atomic_prerequisite_state_clear();
  }
  
  
};
