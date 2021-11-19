%shared_ptr(snde::recording_base);
snde_rawaccessible(snde::recording_base);
%shared_ptr(snde::multi_ndarray_recording);
snde_rawaccessible(snde::multi_ndarray_recording);
%shared_ptr(snde::ndarray_recording_ref);
snde_rawaccessible(snde::ndarray_recording_ref);
%shared_ptr(snde::channelconfig);
snde_rawaccessible(snde::channelconfig);
%shared_ptr(snde::channel);
snde_rawaccessible(snde::channel);
%shared_ptr(snde::recording_set_state);
snde_rawaccessible(snde::recording_set_state);
%shared_ptr(snde::globalrev_mutable_lock);
snde_rawaccessible(snde::globalrev_mutable_lock);
%shared_ptr(snde::globalrevision);
snde_rawaccessible(snde::globalrevision);
%shared_ptr(snde::recdatabase);
snde_rawaccessible(snde::recdatabase);
%shared_ptr(snde::instantiated_math_function);
snde_rawaccessible(snde::instantiated_math_function);

%{
#include "recstore.hpp"

%}


namespace snde {

  // forward references
  class recdatabase;
  class channel;
  class multi_ndarray_recording;
  class ndarray_recording_ref;
  class globalrevision;
  class channel_state;
  class transaction;
  class recording_set_state;
  class arraylayout;
  class math_status;
  class instantiated_math_database;
  class instantiated_math_function;
  class math_definition;

  class channel_notify; // from notify.hpp
  class repetitive_channel_notify; // from notify.hpp
  class promise_channel_notify; 
  class _globalrev_complete_notify;
  class monitor_globalrevs;
  
  extern const std::unordered_map<unsigned,std::string> rtn_typenamemap;
  extern const std::unordered_map<unsigned,size_t> rtn_typesizemap; // Look up element size bysed on typenum
  extern const std::unordered_map<unsigned,std::string> rtn_ocltypemap; // Look up opencl type string based on typenum

  %typemap(in) void *owner_id {
    $1 = (void *)$input; // stores address of the PyObject
  }
  %typecheck(SWIG_TYPECHECK_POINTER) (void *) {
    $1 = 1; // always satisifed
  }

  //
  //%typecheck(SWIG_TYPECHECK_POINTER) (std::shared_ptr<lockmanager>) {
  //  $1 = SWIG_CheckState(SWIG_ConvertPtr($input, 0, SWIGTYPE_p_std__shared_ptrT_snde__lockmanager_t, 0));
  //}
  
  %typemap(out) void * {
    $result = PyLong_FromVoidPtr($1);
  }

  class recording_base /* : public std::enable_shared_from_this<recording_base> */  {
    // may be subclassed by creator
    // mutable in certain circumstances following the conventions of snde_recording

    // lock required to safely read/write mutable portions unless you are the owner and
    // you are single threaded and the information you are writing is for a subsequent state (info->state/info_state);
    // last lock in the locking order except for Python GIL
  public:
    //std::mutex admin; 
    struct snde_recording_base *info; // owned by this class and allocated with malloc; often actually a sublcass such as snde_multi_ndarray_recording
    %immutable;
    /*std::atomic_int*/ int info_state; // atomic mirror of info->state
    %mutable;
    std::shared_ptr<immutable_metadata> metadata; // pointer may not be changed once info_state reaches METADATADONE. The pointer in info is the .get() value of this pointer. 

    std::shared_ptr<recording_storage_manager> storage_manager; // pointer initialized to a default by recording constructor, then used by the allocate_storage() method. Any assignment must be prior to that. may not be used afterward; see recording_storage in recstore_storage.hpp for details on pointed structure.

    // These next three items relate to the __originating__ globalrevision or recording set state
    // wss, but depending on the state _originating_wss may not have been assigned yet and may
    // need to extract from recdb_weak and _originating_globalrev_index.
    // DON'T ACCESS THESE DIRECTLY! Use the .get_originating_wss() and ._get_originating_wss_recdb_and_rec_admin_prelocked() methods.
    std::weak_ptr<recdatabase> recdb_weak;  // Right now I think this is here solely so that we can get access to the available_compute_resources_database to queue more calculations after a recording is marked as ready. 
    std::weak_ptr<transaction> defining_transact; // This pointer should be valid for a recording defined as part of a transaction; nullptr for an ondemand math recording, for example. Weak ptr should be convertible to strong as long as the originating_wss is still current.
    
    std::weak_ptr<recording_set_state> _originating_wss; // locked by admin mutex; if expired than originating_wss has been freed. if nullptr then this was defined as part of a transaction that was may still be going on when the recording was defined. Use get_originating_wss() which handles locking and getting the originating_wss from the defining_transact

    // Need typed template interface !!! ***
    
    // Some kind of notification that recording is done to support
    // e.g. finishing a synchronous process such as a render.

    // This constructor is to be called by everything except the math engine
    //  * Should be called by the owner of the given channel, as verified by owner_id
    //  * Should be called within a transaction (i.e. the recdb transaction_lock is held by the existance of the transaction)
    //  * Because within a transaction, recdb->current_transaction is valid
    // must call recdb->register_new_rec() on the constructed recording
    recording_base(std::shared_ptr<recdatabase> recdb,std::shared_ptr<channel> chan,void *owner_id,size_t info_structsize=sizeof(struct snde_recording_base));

    // This constructor is reserved for the math engine
    // must call recdb->register_new_math_rec() on the constructed recording
    // Note: defining_transact will be nullptr if this calculation
    // is not from a globalrevision originating from a transaction
    recording_base(std::shared_ptr<recdatabase> recdb,std::string chanpath,std::shared_ptr<recording_set_state> calc_wss,size_t info_structsize=sizeof(struct snde_recording_base));

    // rule of 3
    recording_base & operator=(const recording_base &) = delete; 
    recording_base(const recording_base &orig) = delete;
    virtual ~recording_base(); // virtual destructor so we can be subclassed

    std::shared_ptr<multi_ndarray_recording> cast_to_multi_ndarray();

    virtual std::shared_ptr<recording_set_state> _get_originating_wss_rec_admin_prelocked(); // version of get_originating_wss() to use if you have the recording database and recording's admin locks already locked.
    std::shared_ptr<recording_set_state> _get_originating_wss_recdb_admin_prelocked(); // version of get_originating_wss() to use if you have the recording database admin lock already locked.


    virtual std::shared_ptr<recording_set_state> get_originating_wss(); // Get the originating recording set state (often a globalrev). You should only call this if you are sure that originating wss must still exist (otherwise may generate a snde_error), such as before the creator has declared the recording "ready". This will lock the recording database and rec admin locks, so any locks currently held must precede both in the locking order
    virtual bool _transactionrec_transaction_still_in_progress_admin_prelocked(); // with the recording admin locked,  return if this is a transaction recording where the transaction is still in progress and therefore we can't get the recording_set_state

    // Mutable recording only ***!!! Not properly implemented yet ***!!!
    /*
    virtual rwlock_token_set lock_storage_for_write();
    virtual rwlock_token_set lock_storage_for_read();
    */

    
    
    virtual void _mark_metadata_done_internal(/*std::shared_ptr<recording_set_state> wss,const std::string &channame*/);
    virtual void mark_metadata_done();  // call WITHOUT admin lock (or other locks?) held. 
    virtual void mark_as_ready();  // call WITHOUT admin lock (or other locks?) held. 
  };

  class multi_ndarray_recording : public recording_base {
  public:
    std::vector<arraylayout> layouts; // changes to layouts must be propagated to info.arrays[idx].ndim, info.arrays[idx]base_index, info.arrays[idx].dimlen, and info.arrays[idx].strides NOTE THAT THIS MUST BE PREALLOCATED TO THE NEEDED SIZE BEFORE ANY ndarray_recording_ref()'s ARE CREATED!

    std::unordered_map<std::string,size_t> name_mapping; // mapping from array name to array index. Names should not begin with a digit.
    // if name_mapping is non-empty then name_reverse_mapping
    // must be maintained to be identical but reverse
    std::unordered_map<size_t,std::string> name_reverse_mapping;
    
    //std::shared_ptr<rwlock> mutable_lock; // for simply mutable recordings; otherwise nullptr

    std::vector<std::shared_ptr<recording_storage>> storage; // pointers immutable once initialized  by allocate_storage() or reference_immutable_recording().  immutable afterward; see recording_storage in recstore_storage.hpp for details on pointed structure.


    // This constructor is to be called by everything except the math engine to create a multi_ndarray_recording with multiple ndarrays
    //  * Should be called by the owner of the given channel, as verified by owner_id
    //  * Should be called within a transaction (i.e. the recdb transaction_lock is held by the existance of the transaction)
    //  * Because within a transaction, recdb->current_transaction is valid
    //  * Need to call .define_array() on each array up to num_ndarrays. 
    // WARNING: Don't call directly as this constructor doesn't add to the transaction (need to call recdb->register_new_recording()) and many math functions won't work if they rely on
    // ndtyped_recordings!!!
    // OBSOLETE: 
    //    * If the type is known at compile time, better to call ndtyped_recording<T>::create_recording(...)
    //    * If the type is known only at run time, call ::create_typed_recording() method
    multi_ndarray_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<channel> chan,void *owner_id,size_t num_ndarrays,size_t info_structsize=sizeof(struct snde_multi_ndarray_recording));

    
    // This constructor is reserved for the math engine to create multi_ndarrays with a single ndarray
    // Creates recording structure . 
    // WARNING: Don't call directly as this constructor doesn't add  to the pre-existing globalrev (need to call recdb->register_new_math_rec()) and many math functions won't work if they rely on
    // ndtyped_recordings!!!
    //    * If the type is known at compile time, better to call ndtyped_recording<T>::create_recording(...)
    //    * If the type is known only at run time, call ::create_typed_recording() method
    //     * Need to call .define_array() on each array up to num_ndarrays. 

    multi_ndarray_recording(std::shared_ptr<recdatabase> recdb,std::string chanpath,std::shared_ptr<recording_set_state> calc_wss,size_t num_ndarrays,size_t info_structsize=sizeof(struct snde_multi_ndarray_recording));

    // rule of 3
    multi_ndarray_recording & operator=(const multi_ndarray_recording &) = delete; 
    multi_ndarray_recording(const multi_ndarray_recording &orig) = delete;
    virtual ~multi_ndarray_recording();

    inline snde_multi_ndarray_recording *mndinfo() {return (snde_multi_ndarray_recording *)info;}
    inline snde_ndarray_info *ndinfo(size_t index) {return &((snde_multi_ndarray_recording *)info)->arrays[index];}

    void define_array(size_t index,unsigned typenum);   // should be called exactly once for each index < mndinfo()->num_arrays

    
    // static factory methods for creating recordings with single runtime-determined types
    // for regular (non-math) use. Automatically registers the new recording
    static std::shared_ptr<ndarray_recording_ref> create_typed_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<channel> chan,void *owner_id,unsigned typenum);
    static std::shared_ptr<ndarray_recording_ref> create_typed_recording_math(std::shared_ptr<recdatabase> recdb,std::string chanpath,void *owner_id,std::shared_ptr<recording_set_state> calc_wss,unsigned typenum); // math use only
    
    std::shared_ptr<ndarray_recording_ref> reference_ndarray(size_t index=0);


    // must assign info.elementsize and info.typenum before calling allocate_storage()
    // fortran_order only affects physical layout, not logical layout (interpretation of indices)
    virtual void allocate_storage(size_t array_index,std::vector<snde_index> dimlen, bool fortran_order=false);

    // alternative to allocating storage: Referencing an existing recording
    virtual void reference_immutable_recording(size_t array_index,std::shared_ptr<ndarray_recording_ref> rec,std::vector<snde_index> dimlen,std::vector<snde_index> strides);

    
    inline void *void_shifted_arrayptr(size_t array_index);
    
    inline void *element_dataptr(size_t array_index,const std::vector<snde_index> &idx);  // returns a pointer to an element, which is of size ndinfo()->elementsize
    inline size_t element_offset(size_t array_index,const std::vector<snde_index> &idx);

    double element_double(size_t array_index,const std::vector<snde_index> &idx); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!

    void assign_double(size_t array_index,const std::vector<snde_index> &idx,double val); // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published.


    int64_t element_int(size_t array_index,const std::vector<snde_index> &idx); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    void assign_int(size_t array_index,const std::vector<snde_index> &idx,int64_t val); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 

    uint64_t element_unsigned(size_t array_index,const std::vector<snde_index> &idx); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    void assign_unsigned(size_t array_index,const std::vector<snde_index> &idx,uint64_t val); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    
  };

  class ndarray_recording_ref {
    // reference to a single ndarray within an multi_ndarray_recording
    // once the multi_ndarray_recording is published and sufficiently complete, its fields are immutable, so these are too
  public:
    std::shared_ptr<multi_ndarray_recording> rec; // the referenced recording
    size_t rec_index; // index of referenced ndarray within recording.
    unsigned typenum;
    %immutable;
       /* std::atomic_int*/ int &info_state; // reference to rec->info_state
    %mutable;
    arraylayout &layout; // reference  to rec->layouts.at(rec_index)
    std::shared_ptr<recording_storage> &storage;

    ndarray_recording_ref(std::shared_ptr<multi_ndarray_recording> rec,size_t rec_index,unsigned typenum);

    // rule of 3 
    ndarray_recording_ref & operator=(const ndarray_recording_ref &) = delete;
    ndarray_recording_ref(const ndarray_recording_ref &orig) = delete; // could easily be implemented if we wanted
    virtual ~ndarray_recording_ref();

    virtual void allocate_storage(std::vector<snde_index> dimlen, bool fortran_order=false);

    
    inline snde_multi_ndarray_recording *mndinfo() {return (snde_multi_ndarray_recording *)rec->info;}
    inline snde_ndarray_info *ndinfo() {return &((snde_multi_ndarray_recording *)rec->info)->arrays[rec_index];}


    inline void *void_shifted_arrayptr();
    
    inline void *element_dataptr(const std::vector<snde_index> &idx)  // returns a pointer to an element, which is of size ndinfo()->elementsize
    {
      snde_ndarray_info *array_ndinfo = ndinfo();
      char *base_charptr = (char *) (*array_ndinfo->basearray);
      
      char *cur_charptr = base_charptr + array_ndinfo->elementsize*array_ndinfo->base_index;
      for (size_t dimnum=0;dimnum < array_ndinfo->ndim;dimnum++) {
	snde_index thisidx = idx.at(dimnum);
	assert(thisidx < array_ndinfo->dimlen[dimnum]);
	cur_charptr += array_ndinfo->strides[dimnum]*array_ndinfo->elementsize*thisidx;
      }
      
      return (void *)cur_charptr;
    }

    inline size_t element_offset(const std::vector<snde_index> &idx)
    {      
      snde_ndarray_info *array_ndinfo = ndinfo();
      size_t cur_offset = array_ndinfo->base_index;
      
      for (size_t dimnum=0;dimnum < array_ndinfo->ndim;dimnum++) {
	snde_index thisidx = idx.at(dimnum);
	assert(thisidx < array_ndinfo->dimlen[dimnum]);
	cur_offset += array_ndinfo->strides[dimnum]*thisidx;
      }
      
      return cur_offset;
      
    }
    inline size_t element_offset(snde_index idx,bool fortran_order);
    inline size_t element_offset(snde_index idx);

    virtual double element_double(const std::vector<snde_index> &idx); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    virtual double element_double(snde_index idx, bool fortran_order); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    virtual double element_double(snde_index idx); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    virtual void assign_double(const std::vector<snde_index> &idx,double val); // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    virtual void assign_double(snde_index idx,double val,bool fortran_order); // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    virtual void assign_double(snde_index idx,double val); // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 

    virtual int64_t element_int(const std::vector<snde_index> &idx); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    virtual int64_t element_int(snde_index idx,bool fortran_order); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    virtual int64_t element_int(snde_index idx); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    virtual void assign_int(const std::vector<snde_index> &idx,int64_t val); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    virtual void assign_int(snde_index idx,int64_t val,bool fortran_order); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    virtual void assign_int(snde_index idx,int64_t val); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 

    virtual uint64_t element_unsigned(const std::vector<snde_index> &idx); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    virtual uint64_t element_unsigned(snde_index idx,bool fortran_order); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    virtual uint64_t element_unsigned(snde_index idx); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    virtual void assign_unsigned(const std::vector<snde_index> &idx,uint64_t val); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    virtual void assign_unsigned(snde_index idx,uint64_t val,bool fortran_order); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    virtual void assign_unsigned(snde_index idx,uint64_t val); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 



  };


  %extend ndarray_recording_ref {
    PyObject *data()
    {
      PyArray_Descr *ArrayDescr = snde::rtn_numpytypemap.at(self->ndinfo()->typenum);

      // make npy_intp dims and strides from layout.dimlen and layout.strides
      std::vector<npy_intp> dims;
      std::vector<npy_intp> strides;
      std::copy(self->layout.dimlen.begin(),self->layout.dimlen.end(),std::back_inserter(dims));

      for (auto && stride: self->layout.strides) {
	strides.push_back(stride*self->ndinfo()->elementsize); // our strides are in numbers of elements vs numpy does it in bytes;
      }
      int flags = 0;
      if (self->info_state != SNDE_RECS_READY && self->info_state != SNDE_RECS_OBSOLETE) {
	flags = NPY_ARRAY_WRITEABLE; // only writeable if it's not marked as ready yet.
      }

      // Need to grab the GIL before Python calls because
      // swig wrapped us with something that dropped it (!)
      PyGILState_STATE gstate = PyGILState_Ensure();
      Py_IncRef((PyObject *)ArrayDescr); // because PyArray_NewFromDescr steals a reference to its descr parameter
      PyArrayObject *obj = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,ArrayDescr,self->layout.dimlen.size(),dims.data(),strides.data(),self->void_shifted_arrayptr(),0,nullptr);

      // memory_holder_obj contains a shared_ptr to "this", i.e. the ndarray_recording.  We will store this in the "base" property of obj so that as long as obj lives, so will the ndarray_recording, and hence its memory.
      // (This code is similar to the code returned by _wrap_recording_base_cast_to_ndarray()
      std::shared_ptr<snde::ndarray_recording_ref> rawresult = self->shared_from_this();
      assert(rawresult);
      std::shared_ptr<snde::ndarray_recording_ref> *smartresult = new std::shared_ptr<snde::ndarray_recording_ref>(rawresult);
      PyObject *memory_holder_obj = SWIG_NewPointerObj(SWIG_as_voidptr(smartresult), SWIGTYPE_p_std__shared_ptrT_snde__ndarray_recording_ref_t, SWIG_POINTER_OWN/*|SWIG_POINTER_NOSHADOW*/);
      PyArray_SetBaseObject(obj,memory_holder_obj); // steals reference to memory_holder_obj
      PyGILState_Release(gstate);
      return (PyObject *)obj;
    }
  }
  

  class transaction {
  public:
    // mutable until end of transaction when it is destroyed and converted to a globalrev structure
    //std::mutex admin; // last in the locking order except before python GIL. Must hold this lock when reading or writing structures within. Does not cover the channels/recordings themselves.

    uint64_t globalrev; // globalrev index for this transaction. Immutable once published
    std::unordered_set<std::shared_ptr<channel>> updated_channels;
    //Keep track of whether a new recording is required for the channel (e.g. if it has a new owner) (use false for math recordings)
    std::map<std::string,bool> new_recording_required; // index is channel name for updated channels
    std::unordered_map<std::string,std::shared_ptr<recording_base>> new_recordings;

    // end of transaction propagates this structure into an update of recdatabase._channels
    // and a new globalrevision

    // when the transaction is complete, resulting_globalrevision() is assigned
    // so that if you have a pointer to the transaction you can get the globalrevision (which is also a recording_set_state)
    std::weak_ptr<globalrevision> _resulting_globalrevision; // locked by transaction admin mutex; use resulting_globalrevision() accessor. 
    std::pair<std::shared_ptr<globalrevision>,bool> resulting_globalrevision(); // returned bool true means null pointer indicates expired pointer, rather than in-progress transaction

    
  };


  class active_transaction /* : public std::enable_shared_from_this<active_transaction> */ {
    // RAII interface to transaction
    // Don't use this directly from dataguzzler-python, because there
    // you need to drop the thread context before starting the
    // transaction and then reacquire after you have the
    // transaction lock.
  public:
    //std::unique_lock<std::mutex> transaction_lock_holder;
    std::weak_ptr<recdatabase> recdb;
    std::shared_ptr<globalrevision> previous_globalrev;
    bool transaction_ended;
    
    active_transaction(std::shared_ptr<recdatabase> recdb);


    // rule of 3
    active_transaction& operator=(const active_transaction &) = delete; 
    active_transaction(const active_transaction &orig) = delete;
    ~active_transaction(); // destructor releases transaction_lock from holder

    std::shared_ptr<globalrevision> end_transaction();
    
  };

  // need global revision class which will need to have a snapshot of channel configs, including math channels
  // since new math channel revisions are no longer defined until calculated, we will rely on the snapshot to
  // figure out which new channels need to be written. 
  
  class channelconfig {
    // The channelconfig is immutable once published; However it may be copied, privately updated by its owner, (if subclassed, you must use the correct subclasses
    // copy constructor!) and republished.  It may be
    // freed once no references to it exist any more.
    // can be subclassed to accommodate e.g. geometry scene graph entries, geometry parameterizations, etc. 
  public:
    
    std::string channelpath; // Path of this channel in recording database
    std::string owner_name; // Name of owner, such as a dataguzzler_python module
    void *owner_id; // pointer private to the owner (such as dataguzzler_python PyObject of the owner's representation of this channel) that
    // the owner can use to verify its continued ownership.

    bool hidden; // explicitly hidden channel
    bool math; // math channel


    std::shared_ptr<recording_storage_manager> storage_manager; // storage manager for newly defined recordings... Note that while the pointer is immutable, the pointed storage_manager is NOT immutable. 
    
    // The following only apply to math channels
    std::shared_ptr<instantiated_math_function> math_fcn; // If math is set, then this channel is one of the outputs of the given math_fcn  math_fcn is also immutable once published
    bool mathintermediate; // intermediate result of math calculation -- makes it implicitly hidden
    bool ondemand; // if the output is not actually stored in the database but just delivered on-demand
    bool data_requestonly; // if the output is to be stored in the database with the metadata always calculated but the underlying data only triggered to be computed if requested or needed by another recording.
    bool data_mutable; // if the output is mutable

    channelconfig(std::string channelpath, std::string owner_name, void *owner_id,bool hidden,std::shared_ptr<recording_storage_manager> storage_manager=nullptr);
    // rule of 3
    //channelconfig& operator=(const channelconfig &) = default; 
    channelconfig(const channelconfig &orig) = default;
    virtual ~channelconfig() = default; // virtual destructor required so we can be subclassed

  };
  
  class channel {
  public:
    // Channel objects are persistent and tracked with shared pointers.
    // They are never destroyed once created and the channelpath in
    // _config must remain fixed over the lifetime (although _config
    // can be freely replaced with a new one with the same channelpath)

    // Pointers can be safely kept around. Members are atomic so can
    // be safely read without locks; reading multiple members or writing
    // _config or multiple members is synchronized by the
    // admin mutex

    // Should the channel have some means to give notification of updates?
    // What thread/context should that be in and how does it relate to the end of the transaction or
    // the completion of math?
    
    std::shared_ptr<channelconfig> _config; // atomic shared pointer to immutable data structure; nullptr for a deleted channel
    %immutable;
    /*std::atomic<*/uint64_t/*>*/ latest_revision; // 0 means "invalid"; generally 0 (or previous latest) during channel creation/undeletion; incremented to 1 (with an empty recording if necessary) after transaction end 
    /*std::atomic<*/bool/*>*/ deleted; // is the channel currently defined?
    %mutable;
       
    //std::mutex admin; // last in the locking order except before python GIL. Used to ensure there can only be one _config update at a time. 


    channel(std::shared_ptr<channelconfig> initial_config);

    std::shared_ptr<channelconfig> config(); // Use this method to safely get the current channelconfig pointer

    template<typename T>
    std::shared_ptr<T> begin_atomic_config_update()
    // channel admin lock must be locked when calling this function. It is a template because channelconfig can be subclassed. Call it as begin_atomic_config_update<channelconfig>() if you need to subclass It returns a new modifiable copy of the atomically guarded data
    {
      std::shared_ptr<channelconfig> new_config=std::make_shared<T>(*std::dynamic_pointer_cast<T>(_config));
      return new_config;
    }
    
    void end_atomic_config_update(std::shared_ptr<channelconfig> new_config); // admin must be locked when calling this function. It accepts the modified copy of the atomically guarded data
    
    
  };


    
  class channel_state {
  public:
    // for atomic updates to notify_ ... atomic shared pointers, you must lock the recording_set_state's admin lock
    std::shared_ptr<channelconfig> config; // immutable
    std::shared_ptr<channel> _channel; // immutable pointer, but pointed data is not immutable, (but you shouldn't generally need to access this)
    std::shared_ptr<recording_base> _rec; // atomic shared ptr to recording structure created to store the ouput; may be nullptr if not (yet) created. Always nullptr for ondemand recordings... recording contents may be mutable but have their own admin lock

    %immutable;
    /*std::atomic<*/bool/*>*/ updated; // this field is only valid once rec() returns a valid pointer and once rec()->state is READY or METADATAREADY. It is true if this particular recording has a new revision particular to the enclosing recording_set_state
    %mutable;
    std::shared_ptr<uint64_t> revision; // This is assigned when the channel_state is created from _rec->info->revision for manually created recordings. (For ondemand math recordings this is not meaningful?) For math recordings with the math_function's new_revision_optional (config->math_fcn->fcn->new_revision_optional) flag clear, this is defined during end_transaction() before the channel_state is published. If the new_revision_optional flag is set, this left nullptr; once the math function determines whether a new recording will be instantiated the revision will be assigned when the recording is define, with ordering ensured by the implicit self-dependency implied by the new_revision_optional flag (recmath_compute_resource.cpp)
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> _notify_about_this_channel_metadataonly; // atomic shared ptr to immutable set of channel_notifies that need to be updated or perhaps triggered when this channel becomes metadataonly; set to nullptr at end of channel becoming metadataonly. 
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> _notify_about_this_channel_ready; // atomic shared ptr to immutable set of channel_notifies that need to be updated or perhaps triggered when this channel becomes ready; set to nullptr at end of channel becoming ready. 

    channel_state(std::shared_ptr<channel> chan,std::shared_ptr<channelconfig> config,std::shared_ptr<recording_base> rec,bool updated);

    channel_state(const channel_state &orig); // copy constructor used for initializing channel_map from prototype defined in end_transaction()

    std::shared_ptr<recording_base> rec() const;
    std::shared_ptr<recording_base> recording_is_complete(bool mdonly); // uses only atomic members so safe to call in all circumstances. Set to mdonly if you only care that the metadata is complete. Normally call recording_is_complete(false). Returns recording pointer if recording is complete to the requested condition, otherwise nullptr. 
    void issue_nonmath_notifications(std::shared_ptr<recording_set_state> wss); // Must be called without anything locked. Issue notifications requested in _notify* and remove those notification requests
    void issue_math_notifications(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_set_state> wss,bool channel_modified); // Must be called without anything locked. Check for any math updates from the new status of this recording
    
    void end_atomic_rec_update(std::shared_ptr<recording_base> new_recording);


    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> begin_atomic_notify_about_this_channel_metadataonly_update();
    void end_atomic_notify_about_this_channel_metadataonly_update(std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> newval);
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_metadataonly();

    
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> begin_atomic_notify_about_this_channel_ready_update();
    void end_atomic_notify_about_this_channel_ready_update(std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> newval);
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_ready();
  };


  class recording_status {
  public:
    std::map<std::string,channel_state> channel_map; // key is full channel path... The map itself (not the embedded states) is immutable once the recording_set_state is published
    
    /// all of these are indexed by their their full path. Every entry in channel_map should be in exactly one of these. Locked by wss admin mutex per above
    // The index is the shared_ptr in globalrev_channel.config
    // primary use for these is determining when our globalrev/wss is
    // complete: Once call recordings are in metadataonly or completed,
    // then it should be complete
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state *> defined_recordings;
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state *> instantiated_recordings;
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state *> metadataonly_recordings; // only move recordings to here if they are mdonly recordings
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state *> completed_recordings;

    recording_status(const std::map<std::string,channel_state> & channel_map_param);
  };


  class recording_set_state /* : public std::enable_shared_from_this<recording_set_state>*/ {
  public:
    //std::mutex admin; // locks changes to recstatus including channel_map contents (map itself is immutable once published), mathstatus,  and the _recordings reference maps/sets and notifiers. Precedes recstatus.channel_map.rec.admin and Python GIL in locking order
    uint64_t originating_globalrev_index; // this wss may not __be__ a globalrev but it (almost certainly?) is built on one. 
    std::weak_ptr<recdatabase> recdb_weak;
    %immutable;
    /*std::atomic<*/bool/*>*/ ready; // indicates that all recordings except ondemand and data_requestonly recordings are READY (mutable recordings may be OBSOLETE)
    %mutable;
       
    recording_status recstatus;
    math_status mathstatus; // note math_status.math_functions is immutable
    std::shared_ptr<recording_set_state> _prerequisite_state; // C++11 atomic shared pointer. recording_set_state to be used for self-dependencies and any missing dependencies not present in this state. This is an atomic shared pointer (read with .prerequisite_state()) that is set to nullptr once a new globalrevision is ready, so as to allow prior recording revisions to be freed.
    std::unordered_set<std::shared_ptr<channel_notify>> recordingset_complete_notifiers; // Notifiers waiting on this recording set state being complete. Criteria will be removed as they are satisifed and entries will be removed as the notifications are performed.

    std::shared_ptr<lockmanager> lockmgr; // pointer is immutable after initialization

    
    recording_set_state(std::shared_ptr<recdatabase> recdb,const instantiated_math_database &math_functions,const std::map<std::string,channel_state> & channel_map_param,std::shared_ptr<recording_set_state> prereq_state); // constructor
    // Rule of 3
    recording_set_state& operator=(const recording_set_state &) = delete; 
    recording_set_state(const recording_set_state &orig) = delete;
    virtual ~recording_set_state()=default;

    void wait_complete(); // wait for all the math in this recording_set_state or globalrev to reach nominal completion (metadataonly or ready, as configured)
    std::shared_ptr<recording_base> get_recording(const std::string &fullpath);
    std::shared_ptr<ndarray_recording_ref> get_recording_ref(const std::string &fullpath,size_t array_index=0);
    std::shared_ptr<ndarray_recording_ref> get_recording_ref(const std::string &fullpath,std::string array_name);

    // admin lock must be locked when calling this function. Returns
    std::shared_ptr<recording_set_state> prerequisite_state();
    void atomic_prerequisite_state_clear(); // sets the prerequisite state to nullptr
    
    long get_reference_count(); // get the shared_ptr reference count; useful for debugging memory leaks

    size_t num_complete_notifiers(); // size of recordingset_complete_notifiers; useful for debugging memory leaks
  };

  class globalrev_mutable_lock {
  public:
    // See comment above mutable_recordings_still_needed field of globalrevision, below for explanation of what this is for and how it works
    globalrev_mutable_lock(std::weak_ptr<recdatabase> recdb,std::weak_ptr<globalrevision> globalrev);
      
    globalrev_mutable_lock & operator=(const globalrev_mutable_lock &) = delete; 
    globalrev_mutable_lock(const globalrev_mutable_lock &orig) = delete;
    ~globalrev_mutable_lock();


    std::weak_ptr<recdatabase> recdb;
    std::weak_ptr<globalrevision> globalrev; 
  };

  class globalrevision: public recording_set_state { // should probably be derived from a class math_calculation_set or similar, so code can be reused for ondemand recordings
    // channel_map is mutable until the ready flag is set. Once the ready flag is set only mutable and data_requestonly recordings may be modified.
  public:
    // These commented members are really part of the recording_set_state we are derived from
    //std::mutex admin; // locks changes to recstatus, mathstatus, recstatus.channel_map and the _recordings reference maps/sets. Precedes recstatus.channel_map.rec.admin and Python GIL in locking order
    //std::atomic<bool> ready; // indicates that all recordings except ondemand and data_requestonly recordings are READY (mutable recordings may be OBSOLETE)
    //recording_status recstatus;
    //math_status mathstatus; // note math_status.math_functions is immutable

    
    uint64_t globalrev;
    std::shared_ptr<transaction> defining_transact; // This keeps the transaction data structure (pointed to by weak pointers in the recordings created in the transaction) in memory at least as long as the globalrevision is current. 

    std::shared_ptr<globalrev_mutable_lock> mutable_recordings_need_holder;
    //std::atomic<bool> mutable_recordings_still_needed; 

    
    globalrevision(uint64_t globalrev, std::shared_ptr<transaction> defining_transact, std::shared_ptr<recdatabase> recdb,const instantiated_math_database &math_functions,const std::map<std::string,channel_state> & channel_map_param,std::shared_ptr<recording_set_state> prereq_state);   
  };
  

  class recdatabase /* : public std::enable_shared_from_this<recdatabase> */ {
  public:
    //std::mutex admin; // Locks access to _channels and _deleted_channels and _math_functions, _globalrevs and repetitive_notifies. In locking order, precedes channel admin locks, available_compute_resource_database, globalrevision admin locks, recording admin locks, and Python GIL. 
    std::map<std::string,std::shared_ptr<channel>> _channels; // Generally don't use the channel map directly. Grab the latestglobalrev and use the channel map from that. 
    std::map<std::string,std::shared_ptr<channel>> _deleted_channels; // Channels are put here after they are deleted. They can be moved back into the main list if re-created. 
    instantiated_math_database _math_functions; 
    
    std::map<uint64_t,std::shared_ptr<globalrevision>> _globalrevs; // Index is global revision counter. The first element in this is the latest globalrev with all mandatory immutable channels ready. The last element in this is the most recently defined globalrev.
    std::shared_ptr<globalrevision> _latest_globalrev; // atomic shared pointer -- access with latest_globalrev() method;
    std::vector<std::shared_ptr<repetitive_channel_notify>> repetitive_notifies; 

    std::shared_ptr<available_compute_resource_database> compute_resources; // has its own admin lock.
    

    std::shared_ptr<recording_storage_manager> default_storage_manager; // pointer is immutable once created; contents not necessarily immutable; see recstore_storage.hpp

    std::shared_ptr<lockmanager> lockmgr; // pointer immutable after initialization; contents have their own admin lock, which is used strictly internally by them

    //std::mutex transaction_lock; // ***!!! Before any dataguzzler-python module locks, etc.
    std::shared_ptr<transaction> current_transaction; // only valid while transaction_lock is held.

    std::set<std::weak_ptr<monitor_globalrevs>,std::owner_less<std::weak_ptr<monitor_globalrevs>>> monitoring;
    uint64_t monitoring_notify_globalrev; // latest globalrev for which monitoring has already been notified


    std::list<std::shared_ptr<globalrevision>> globalrev_mutablenotneeded_pending;
    bool globalrev_mutablenotneeded_mustexit;

    //recdatabase(std::shared_ptr<recording_storage_manager> default_storage_manager=nullptr,std::shared_ptr<lockmanager> lockmgr=nullptr);
    // default argument split into three separate entries here
    // to work around swig bug with default parameter
    recdatabase(std::shared_ptr<allocator_alignment> alignment_requirements,std::shared_ptr<lockmanager> lockmgr);
    recdatabase(std::shared_ptr<allocator_alignment> alignment_requirements);
    recdatabase(std::shared_ptr<recording_storage_manager> default_storage_manager,std::shared_ptr<lockmanager> lockmgr);
    recdatabase(std::shared_ptr<recording_storage_manager> default_storage_manager);

    
    recdatabase & operator=(const recdatabase &) = delete; 
    recdatabase(const recdatabase &orig) = delete;
    ~recdatabase();
        
    // avoid using start_transaction() and end_transaction() from C++; instantiate the RAII wrapper class active_transaction instead
    // (start_transaction() and end_transaction() are intended for C and perhaps Python)

    // a transaction update can be multi-threaded but you shouldn't call end_transaction()  (or the end_transaction method on the
    // active_transaction or delete the active_transaction) until all other threads are finished with transaction actions
    std::shared_ptr<active_transaction> start_transaction();
    std::shared_ptr<globalrevision> end_transaction(std::shared_ptr<active_transaction> act_trans);
    // add_math_function() must be called within a transaction
    void add_math_function(std::shared_ptr<instantiated_math_function> new_function,bool hidden); // Use separate functions with/without storage manager because swig screws up the overload
    void add_math_function_storage_manager(std::shared_ptr<instantiated_math_function> new_function,bool hidden,std::shared_ptr<recording_storage_manager> storage_manager);

    void register_new_rec(std::shared_ptr<recording_base> new_rec);
    void register_new_math_rec(void *owner_id,std::shared_ptr<recording_set_state> calc_wss,std::shared_ptr<recording_base> new_rec); // registers newly created math recording in the given wss (and extracts mutable flag for the given channel into the recording structure)). 

    std::shared_ptr<globalrevision> latest_globalrev();

    // Allocate channel with a specific name; returns nullptr if the name is inuse
    std::shared_ptr<channel> reserve_channel(std::shared_ptr<channelconfig> new_config);

    //std::shared_ptr<channel> lookup_channel_live(std::string channelpath);


    // NOTE: python wrappers for wait_recordings and wait_recording_names need to drop dgpython thread context during wait and poll to check for connection drop
    //void wait_recordings(std::vector<std::shared_ptr<recording>> &);
    void wait_recording_names(std::shared_ptr<recording_set_state> wss,const std::vector<std::string> &metadataonly, const std::vector<std::string> fullyready);

    std::shared_ptr<monitor_globalrevs> start_monitoring_globalrevs();

  };

  
  

  
};
