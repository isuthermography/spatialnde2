// override __setitem__ from std::map definition within Swig's python/std_map.i
// because otherwise this calls the copy assignment operator on snde::channel_state, which
// doesn't exist
//%extend std::map<std::string,snde::channel_state> {
//  
//  std::map<std::string,snde::channel_state>(const std::map< std::string,snde::channel_state >  &) = delete; // explicitly delete copy constructor
//   
//    void __setitem__(const key_type& key, const mapped_type& x) throw (std::out_of_range) {
//      throw snde::snde_error("Attempt to modify immutable channel map");
//    } // This line should generate a warning because it intentionally overrides the definition from std/std_map.i 
//
//};

%shared_ptr(snde::recording_base);
snde_rawaccessible(snde::recording_base);
%shared_ptr(snde::recording_group);
snde_rawaccessible(snde::recording_group);
%shared_ptr(snde::null_recording);
snde_rawaccessible(snde::null_recording);
%shared_ptr(snde::multi_ndarray_recording);
snde_rawaccessible(snde::multi_ndarray_recording);
%shared_ptr(snde::ndarray_recording_ref);
snde_rawaccessible(snde::ndarray_recording_ref);
%shared_ptr(snde::fusion_ndarray_recording);
snde_rawaccessible(snde::fusion_ndarray_recording);
%shared_ptr(snde::channelconfig);
snde_rawaccessible(snde::channelconfig);
%shared_ptr(snde::channel);
snde_rawaccessible(snde::channel);
%shared_ptr(snde::reserved_channel);
snde_rawaccessible(snde::reserved_channel);
%shared_ptr(snde::recording_set_state);
snde_rawaccessible(snde::recording_set_state);
%shared_ptr(snde::rss_reference);
snde_rawaccessible(snde::rss_reference);
%shared_ptr(snde::globalrev_mutable_lock);
snde_rawaccessible(snde::globalrev_mutable_lock);
%shared_ptr(snde::globalrevision);
snde_rawaccessible(snde::globalrevision);
%shared_ptr(snde::recdatabase);
snde_rawaccessible(snde::recdatabase);
%shared_ptr(snde::instantiated_math_function);
snde_rawaccessible(snde::instantiated_math_function);
%shared_ptr(snde::active_transaction);
snde_rawaccessible(snde::active_transaction);
%shared_ptr(snde::transaction);
snde_rawaccessible(snde::transaction);
%shared_ptr(snde::transaction_math);
snde_rawaccessible(snde::transaction_math);

//%shared_ptr(std::map<std::string,snde::channel_state>);
//snde_rawaccessible(std::map<std::string,snde::channel_state>);


// These really belong in recmath.i but they were moved here so that they are fully defined
// earlier in the order
%shared_ptr(snde::math_instance_parameter);
snde_rawaccessible(snde::math_instance_parameter);
%shared_ptr(snde::list_math_instance_parameter);
snde_rawaccessible(snde::list_math_instance_parameter);
%shared_ptr(snde::dict_math_instance_parameter);
snde_rawaccessible(snde::dict_math_instance_parameter);
%shared_ptr(snde::string_math_instance_parameter);
snde_rawaccessible(snde::string_math_instance_parameter);
%shared_ptr(snde::int_math_instance_parameter);                  
snde_rawaccessible(snde::int_math_instance_parameter);
%shared_ptr(snde::double_math_instance_parameter);
snde_rawaccessible(snde::double_math_instance_parameter);
%shared_ptr(snde::pending_math_definition_result_channel);
snde_rawaccessible(snde::pending_math_definition_result_channel);
%shared_ptr(snde::python_math_definition);
snde_rawaccessible(snde::python_math_definition);
%shared_ptr(snde::pending_math_definition);
snde_rawaccessible(snde::pending_math_definition);
%shared_ptr(snde::pending_math_intermediate_channels);
snde_rawaccessible(snde::pending_math_intermediate_channels);

%{
#include "snde/recstore.hpp"

%}


namespace snde {

  // forward references
  class recdatabase;
  class channel;
  class reserved_channel;
  class multi_ndarray_recording;
  class ndarray_recording_ref;
  class globalrevision;
  class channel_state;
  class transaction;
  class recording_set_state;
  class rss_reference;
  class arraylayout;
  class math_status;
  class instantiated_math_database;
  class instantiated_math_function;
  class math_definition;
  class image_reference;
  class recording_class_info;
  class math_instance_parameter;

  class channel_notify; // from notify.hpp
  class repetitive_channel_notify; // from notify.hpp
  class promise_channel_notify; 
  class _globalrev_complete_notify;
  class monitor_globalrevs;
  class transaction_manager; // recstore_transaction_manager.hpp
  
  extern const std::unordered_map<unsigned,std::string> rtn_typenamemap;
  extern const std::unordered_map<unsigned,size_t> rtn_typesizemap; // Look up element size bysed on typenum
  extern const std::unordered_map<unsigned,std::string> rtn_ocltypemap; // Look up opencl type string based on typenum

  %typemap(in) void *owner_id {

    if (PyLong_Check($input)) {
      // use value of long
      $1 = PyLong_AsVoidPtr($input);
    } else {
      $1 = (void *)$input; // stores address of the PyObject
    }
  }
  %typecheck(SWIG_TYPECHECK_POINTER) (void *) {
    $1 = 1; // always satisifed
  }

  //
  //%typecheck(SWIG_TYPECHECK_POINTER) (std::shared_ptr<lockmanager>) {
  //  $1 = SWIG_CheckState(SWIG_ConvertPtr($input, 0, SWIGTYPE_p_std__shared_ptrT_snde__lockmanager_t, 0));
  //}

  // output typemap for rec_classes
  %typemap(out) std::vector<recording_class_info> (size_t cnt){
    $result = PyList_New($1.size());
    for (cnt=0;cnt < $1.size();cnt++) {
      PyList_SetItem($result,(Py_ssize_t)cnt,PyUnicode_FromString($1.at(cnt).c_str()));
    }    
  }

  // downcasting typemap for recording_base
  %typemap(out) std::shared_ptr<recording_base> (int derivation_level){

    // try classes from deepest to shallowest until we find something SWIG-wrapped
    for (derivation_level = ((int)$1->rec_classes.size())-1;derivation_level >= 0; derivation_level--) {
      const snde::recording_class_info &classinfo = $1->rec_classes.at(derivation_level);
      const std::string swig_typename = std::string("std::shared_ptr <")+classinfo.classname+std::string("> *");
      swig_type_info *const rettype = SWIG_TypeQuery(swig_typename.c_str());
      if (rettype) {
	void *smartresult = classinfo.ptr_to_new_shared($1);
	$result = SWIG_NewPointerObj(smartresult, rettype, SWIG_POINTER_OWN);
	break;
      } else {
	snde::snde_warning("recording_base output typemap: typequery for %s failed.",swig_typename.c_str());
      }
    }    
  }

  std::shared_ptr<recording_storage_manager> select_storage_manager_for_recording_during_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<reserved_channel> proposed_chan,std::shared_ptr<channelconfig> proposed_config);

  std::shared_ptr<recording_storage_manager> select_storage_manager_for_recording(std::shared_ptr<recdatabase> recdb,std::string chanpath,std::shared_ptr<recording_set_state> rss);

  struct recording_params {
    std::shared_ptr<recdatabase> recdb;
    std::shared_ptr<recording_storage_manager> storage_manager;
    std::shared_ptr<rss_reference> prerequisite_state;
    std::string chanpath;
    std::shared_ptr<rss_reference> originating_state;
    uint64_t new_revision; // set to SNDE_REVISION_INVALID unless you have a definitive new revision to supply. 
  };

  class recording_base /* : public std::enable_shared_from_this<recording_base> */  {
    // may be subclassed by creator
    // mutable in certain circumstances following the conventions of snde_recording

    // lock required to safely read/write mutable portions unless you are the owner and
    // you are single threaded and the information you are writing is for a subsequent state (info->state/info_state);
    // last lock in the locking order except for Python GIL
  public:
    //std::mutex admin; 
    struct snde_recording_base *info; // owned by this class and allocated with malloc; often actually a sublcass such as snde_multi_ndarray_recording
    //std::atomic<uint64_t> info_revision; //atomic mirror of info->revision
    %immutable;
    /*std::atomic_int*/ int info_state; // atomic mirror of info->state
    %mutable;

    std::vector<recording_class_info> rec_classes; // ordered inheritance: First entry is recording_base, then subclasses in order. Must be filled out by constructors then immutable after that.

    std::shared_ptr<constructible_metadata> pending_metadata; 
    std::shared_ptr<immutable_metadata> metadata; // pointer may not be changed once info_state reaches METADATADONE. The pointer in info is the .get() value of this pointer. 

    bool needs_dynamic_metadata;
    std::shared_ptr<constructible_metadata> pending_dynamic_metadata; 
    std::shared_ptr<reserved_channel> chan; //  immutable reference to the channel provided when the recording was created. Needed for assigning storage manager
    std::shared_ptr<channelconfig> chanconfig; // current proposed channel configuration as of when this recording was created (immutable)
    std::shared_ptr<recording_storage_manager> storage_manager; // pointer initialized to a default by recording constructor, then used by the allocate_storage() method. Any assignment must be prior to that. may not be used afterward; see recording_storage in recstore_storage.hpp for details on pointed structure.

    // These next three items relate to the __originating__ globalrevision or recording set state
    // rss, but depending on the state _originating_rss may not have been assigned yet and may
    // need to extract from recdb_weak and _originating_globalrev_index.
    // DON'T ACCESS THESE DIRECTLY! Use the .get_originating_rss() and ._get_originating_rss_recdb_and_rec_admin_prelocked() methods.
    std::weak_ptr<recdatabase> recdb_weak;  // Right now I think this is here solely so that we can get access to the available_compute_resources_database to queue more calculations after a recording is marked as ready. 
    //std::weak_ptr<transaction> defining_transact; // This pointer should be valid for a recording defined as part of a transaction; nullptr for an ondemand math recording, for example. Weak ptr should be convertible to strong as long as the originating_rss is still current.
    
    uint64_t originating_state_unique_id; // must be assigned by creator (i.e. create_recording<>() or create_recording_math<>()) immediately after creation. Immutable from then on.

    //std::weak_ptr<recording_set_state> _originating_rss; // locked by admin mutex; if expired than originating_rss has been freed. if nullptr then this was defined as part of a transaction that was may still be going on when the recording was defined. Use get_originating_rss() which handles locking and getting the originating_rss from the defining_transact

    //note that python access to prerequisite_state and originating_state is not strictly thread safe
    std::shared_ptr<rss_reference> prerequisite_state; // This pointer is cleared when the recording is complete and that allows the prerequisite state to go out of existence.

    std::shared_ptr<rss_reference> originating_state; // This pointer is cleared when the recording is complete and that allows the originating state to go out of existence.
    
    recording_base(struct recording_params params,size_t info_structsize=0);

    // rule of 3
    recording_base & operator=(const recording_base &) = delete; 
    recording_base(const recording_base &orig) = delete;
    virtual ~recording_base(); // virtual destructor so we can be subclassed

    virtual void recording_needs_dynamic_metadata(); // Call this before mark_metadata_done() to list this recording as needing dynamic metadata

    std::shared_ptr<multi_ndarray_recording> cast_to_multi_ndarray();

    //virtual std::shared_ptr<recording_set_state> _get_originating_rss_rec_admin_prelocked(); // version of get_originating_rss() to use if you have (optionally the recording database and) the recording's admin locks already locked.
    //virtual std::shared_ptr<recording_set_state> _get_originating_rss_recdb_admin_prelocked(); // version of get_originating_rss() to use if you have the recording database admin lock already locked.


    //virtual std::shared_ptr<recording_set_state> get_originating_rss(); // Get the originating recording set state (often a globalrev). You should only call this if you are sure that originating rss must still exist (otherwise may generate a snde_error), such as before the creator has declared the recording "ready". This will lock the rec admin locks, so any locks currently held must precede that in the locking order
    //virtual bool _transactionrec_transaction_still_in_progress_admin_prelocked(); // with the recording admin locked,  return if this is a transaction recording where the transaction is still in progress and therefore we can't get the recording_set_state

    // Mutable recording only ***!!! Not properly implemented yet ***!!!
    /*
    virtual rwlock_token_set lock_storage_for_write();
    virtual rwlock_token_set lock_storage_for_read();
    */

    
    
    virtual void mark_metadata_done();  // call WITHOUT admin lock (or other locks?) held. 
    virtual void mark_dynamic_metadata_done();  // call WITHOUT admin lock (or other locks?) held. 
    virtual void mark_data_ready();  // call WITHOUT admin lock (or other locks?) held.
    virtual void mark_data_and_metadata_ready();  // call WITHOUT admin lock (or other locks?) held.

    virtual std::shared_ptr<recording_storage_manager> assign_storage_manager(std::shared_ptr<recording_storage_manager> storman);
    virtual std::shared_ptr<recording_storage_manager> assign_storage_manager();

  };

  class null_recording: public recording_base {
  public:

    null_recording(struct recording_params params,size_t info_structsize=0);
    
    
    // rule of 3
    null_recording & operator=(const null_recording &) = delete; 
    null_recording(const null_recording &orig) = delete;
    virtual ~null_recording()=default;

  };

  class recording_group : public recording_base {
  public:
    // Group elements are not stored here; they are found by searching
    // the channel_map of the recording_set_state or the _channels map
    // of the recdatabase. Because the maps are ordered, you should be
    // able to iterate through the group elements by starting with the
    // group name (including trailing slash) and iterating forward until
    // you get an entry not within the group. 

    //std::shared_ptr<std::string> path_to_primary; // nullptr or the path (generally relative to this group) to the primary content of the group, which should be displayed when the user asks to view the content represented by the group. 


    recording_group(struct recording_params params,size_t info_structsize); //,std::shared_ptr<std::string> path_to_primary);
    
    
    // rule of 3
    recording_group & operator=(const recording_group &) = delete; 
    recording_group(const recording_group &orig) = delete;
    virtual ~recording_group()=default;

    
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


    multi_ndarray_recording(struct recording_params params,size_t info_structsize,size_t num_ndarrays);

    // rule of 3
    multi_ndarray_recording & operator=(const multi_ndarray_recording &) = delete; 
    multi_ndarray_recording(const multi_ndarray_recording &orig) = delete;
    virtual ~multi_ndarray_recording();

    virtual void mark_data_ready();  // call WITHOUT admin lock (or other locks?) held. Passes on ready_notifications to storage

    inline snde_multi_ndarray_recording *mndinfo();
    inline snde_ndarray_info *ndinfo(size_t index);
    inline snde_ndarray_info *ndinfo(std::string name);

    void define_array(size_t index,unsigned typenum);   // should be called exactly once for each index < mndinfo()->num_arrays
    void define_array(size_t index,unsigned typenum,std::string name);   // should be called exactly once for each index < mndinfo()->num_arrays

    std::shared_ptr<std::vector<std::string>> list_arrays();
    
    std::shared_ptr<ndarray_recording_ref> reference_ndarray(size_t index=0);
    std::shared_ptr<ndarray_recording_ref> reference_ndarray(std::string array_name);



    void assign_storage(std::shared_ptr<recording_storage> stor,size_t array_index,const std::vector<snde_index> &dimlen, bool fortran_order=false);
    void assign_storage(std::shared_ptr<recording_storage> stor,std::string array_name,const std::vector<snde_index> &dimlen, bool fortran_order=false);

    // must assign info.elementsize and info.typenum before calling allocate_storage()
    // fortran_order only affects physical layout, not logical layout (interpretation of indices)
    // allocate_storage() does assign_storage_manager() then uses that to allocate_recording(), then performs assign_storage().
    // returns the storage in case you want it. 
    
    virtual std::shared_ptr<recording_storage> allocate_storage(size_t array_index,std::vector<snde_index> dimlen, bool fortran_order=false);
    virtual std::shared_ptr<recording_storage> allocate_storage_in_named_array(size_t array_index,std::string storage_array_name,const std::vector<snde_index> &dimlen, bool fortran_order=false);
    virtual std::shared_ptr<recording_storage> allocate_storage(std::string array_name,const std::vector<snde_index> &dimlen, bool fortran_order=false);

    // alternative to allocating storage: Referencing an existing recording
    virtual void reference_immutable_recording(size_t array_index,std::shared_ptr<ndarray_recording_ref> rec,std::vector<snde_index> dimlen,std::vector<snde_index> strides);

    
    inline void *void_shifted_arrayptr(size_t array_index);
    inline void *void_shifted_arrayptr(std::string array_name);

    inline void *element_dataptr(size_t array_index,const std::vector<snde_index> &idx);  // returns a pointer to an element, which is of size ndinfo()->elementsize
    inline size_t element_offset(size_t array_index,const std::vector<snde_index> &idx);

    double element_double(size_t array_index,const std::vector<snde_index> &idx); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    double element_double(size_t array_index,snde_index idx); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    double element_double(size_t array_index,snde_index idx,bool fortran_order); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!

    void assign_double(size_t array_index,const std::vector<snde_index> &idx,double val); // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published.
    void assign_double(size_t array_index,snde_index idx,double val); // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published.
    void assign_double(size_t array_index,snde_index idx,double val,bool fortran_order); // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published.


    int64_t element_int(size_t array_index,const std::vector<snde_index> &idx); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    int64_t element_int(size_t array_index,snde_index idx); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    int64_t element_int(size_t array_index,snde_index idx,bool fortran_order); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    void assign_int(size_t array_index,const std::vector<snde_index> &idx,int64_t val); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    void assign_int(size_t array_index,snde_index idx,int64_t val); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    void assign_int(size_t array_index,snde_index idx,int64_t val,bool fortran_order); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 

    uint64_t element_unsigned(size_t array_index,const std::vector<snde_index> &idx); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    uint64_t element_unsigned(size_t array_index,snde_index idx); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    uint64_t element_unsigned(size_t array_index,snde_index idx,bool fortran_order); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    
    void assign_unsigned(size_t array_index,const std::vector<snde_index> &idx,uint64_t val); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    void assign_unsigned(size_t array_index,snde_index idx,uint64_t val); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    void assign_unsigned(size_t array_index,snde_index idx,uint64_t val,bool fortran_order); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 

    snde_complexfloat64 element_complexfloat64(size_t array_index,const std::vector<snde_index> &idx); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    snde_complexfloat64 element_complexfloat64(size_t array_index,size_t idx); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    snde_complexfloat64 element_complexfloat64(size_t array_index,size_t idx,bool fortran_order); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    void assign_complexfloat64(size_t array_index,const std::vector<snde_index> &idx,snde_complexfloat64 val); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    void assign_complexfloat64(size_t array_index,size_t idx,snde_complexfloat64 val); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    void assign_complexfloat64(size_t array_index,size_t idx,snde_complexfloat64 val,bool fortran_order); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!

    %pythoncode %{
      @property
      def array(self):
        return array_helper(self)
        
      def __str__(self):
        return f"multi_ndarray_recording for {self.info.name:s}.\nSee .array for embedded arrays and .metadata\n"

      def __repr__(self):
        return self.__str__()

    %}
  };

  
  %pythoncode %{
    class array_helper:
      recording = None
      def __init__(self,recording):
        self.recording = recording
        pass

      def __getitem__(self,name_or_index):
        return self.recording.reference_ndarray(name_or_index)

      def __str__(self):
        return str(self.recording.list_arrays())

      def __repr__(self):
        return self.__str__()

      pass

  %}
  // output typemap for _ndarray_recording_ref
  // that turns it into a numpy PyObject for the .data attribute
  // getter (.shared_from_this()). Note the ::data that limits it
  // to the .data attribute. 
}
%typemap(out) std::shared_ptr<snde::ndarray_recording_ref> const &snde::ndarray_recording_ref::data (std::shared_ptr<snde::ndarray_recording_ref> _self,std::vector<npy_intp> dims,std::vector<npy_intp> strides,PyObject *memory_holder_obj) { // self because this code was derived from a preexisting extend directive 
    _self = (*($1));
    
    auto numpytypemap_it = snde::rtn_numpytypemap.find(_self->ndinfo()->typenum);
    if (numpytypemap_it == snde::rtn_numpytypemap.end()) {
      throw snde::snde_error("No corresponding numpy datatype found for snde type #%u",_self->ndinfo()->typenum);
    }
      
    PyArray_Descr *ArrayDescr = snde::rtn_numpytypemap.at(_self->ndinfo()->typenum);

    // make npy_intp dims and strides from layout.dimlen and layout.strides

    std::copy(_self->layout.dimlen.begin(),_self->layout.dimlen.end(),std::back_inserter(dims));

    for (auto && stride: _self->layout.strides) {
      strides.push_back(stride*_self->ndinfo()->elementsize); // our strides are in numbers of elements vs numpy does it in bytes;
    }
    int flags = 0;
    if (!(_self->info_state & SNDE_RECF_DATAREADY)) {
      flags = NPY_ARRAY_WRITEABLE; // only writeable if it's not marked as ready yet.
    }

    //// Need to grab the GIL before Python calls because
    //// swig wrapped us with something that dropped it (!)
    //PyGILState_STATE gstate = PyGILState_Ensure();
    Py_IncRef((PyObject *)ArrayDescr); // because PyArray_NewFromDescr steals a reference to its descr parameter
    PyArrayObject *obj = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,ArrayDescr,_self->layout.dimlen.size(),dims.data(),strides.data(),_self->void_shifted_arrayptr(),flags,nullptr);

    // memory_holder_obj contains a shared_ptr to "this", i.e. the ndarray_recording.  We will store this in the "base" property of obj so that as long as obj lives, so will the ndarray_recording, and hence its memory.
    // (This code is similar to the code returned by _wrap_recording_base_cast_to_ndarray()
    //std::shared_ptr<snde::ndarray_recording_ref> rawresult = self->shared_from_this();
    //assert(rawresult);
    std::shared_ptr<snde::ndarray_recording_ref> *smartresult = new std::shared_ptr<snde::ndarray_recording_ref>(_self);
    memory_holder_obj = SWIG_NewPointerObj(SWIG_as_voidptr(smartresult), SWIGTYPE_p_std__shared_ptrT_snde__ndarray_recording_ref_t, SWIG_POINTER_OWN/*|SWIG_POINTER_NOSHADOW*/);
    PyArray_SetBaseObject(obj,memory_holder_obj); // steals reference to memory_holder_obj
    //PyGILState_Release(gstate);
    $result= (PyObject *)obj;
  }

  


//enable the ndarray_recording_ref.data attribute using the shared_from_this() getter
%attributestring(snde::ndarray_recording_ref,std::shared_ptr<snde::ndarray_recording_ref>,data,shared_from_this);


namespace snde {
  class ndarray_recording_ref {
    // reference to a single ndarray within an multi_ndarray_recording
    // once the multi_ndarray_recording is published and sufficiently complete, its fields are immutable, so these are too
  public:
    std::shared_ptr<multi_ndarray_recording> rec; // the referenced recording
    size_t rec_index; // index of referenced ndarray within recording.
    unsigned typenum;
    %immutable;
       /* std::atomic_int*/ int info_state; // reference to rec->info_state
    %mutable;
    arraylayout &layout; // reference  to rec->layouts.at(rec_index)
    std::shared_ptr<recording_storage> &storage;

    ndarray_recording_ref(std::shared_ptr<multi_ndarray_recording> rec,size_t rec_index,unsigned typenum);

    // rule of 3 
    ndarray_recording_ref & operator=(const ndarray_recording_ref &) = delete;
    ndarray_recording_ref(const ndarray_recording_ref &orig) = delete; // could easily be implemented if we wanted
    virtual ~ndarray_recording_ref();

    virtual void allocate_storage(std::vector<snde_index> dimlen);
    virtual void allocate_storage(std::vector<snde_index> dimlen, bool fortran_order);
    virtual std::shared_ptr<recording_storage> allocate_storage_in_named_array(std::string storage_array_name,const std::vector<snde_index> &dimlen);
    virtual std::shared_ptr<recording_storage> allocate_storage_in_named_array(std::string storage_array_name,const std::vector<snde_index> &dimlen, bool fortran_order);


    
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



    virtual snde_complexfloat64 element_complexfloat64(const std::vector<snde_index> &idx); //  if array is mutable by others, it should generally be locked for read when calling this function!
    virtual snde_complexfloat64 element_complexfloat64(snde_index idx,bool fortran_order); // if array is mutable by others, it should generally be locked for read when calling this function!
    virtual snde_complexfloat64 element_complexfloat64(snde_index idx); //  if array is mutable by others, it should generally be locked for read when calling this function!
    virtual void assign_complexfloat64(const std::vector<snde_index> &idx,snde_complexfloat64 val); //  May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    virtual void assign_complexfloat64(snde_index idx,snde_complexfloat64 val,bool fortran_order); // May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    virtual void assign_complexfloat64(snde_index idx,snde_complexfloat64 val); // if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 

    %pythoncode %{
      def __str__(self):
        rec_name = self.rec.info.name
        return f"ndarray_recording_ref {rec_name:s}.array[{self.rec_index:d}]\nSee .data and .rec.metadata\n"

      def __repr__(self):
        return self.__str__()

    %}

  };


 
  

  class fusion_ndarray_recording: public multi_ndarray_recording {
  public:
    // fusion_ndarray_recording has two arrays: accumulator, and totals.
    // it is meant to represent the circumstance where the meaningful value is
    // represented as a weighted average: (sum_i value_i*weight_i)/(sum_i weight_i).
    // The "accumulator" array represents sum_i value_i*weight_i
    // and the "totals" array represents sum_i weight_i
    //
    // Then meaningful output can be evaluated by dividing accumulator/totals.

    // typenum parameter applies to the accumulator only (obviously)
    
    fusion_ndarray_recording(struct recording_params params,size_t info_structsize,unsigned typenum);
    
    
  };


class transaction_math {
  public:
  //    std::weak_ptr<active_transaction> trans;
    std::map<std::string,std::pair<std::shared_ptr<reserved_channel>,std::shared_ptr<pending_math_definition_result_channel>>> pending_dict; // Indexed by definition channel name

    
  };
  %extend transaction_math {
    void __setitem__(std::string name, std::shared_ptr<pending_math_definition_result_channel> result_chan)
    {
      auto dict_it = self->pending_dict.find(name);
      if (dict_it != self->pending_dict.end()) {
        self->pending_dict.erase(dict_it);
      }

      std::shared_ptr<snde::transaction> trans_strong = self->trans.lock();
      if (!trans_strong) {
        throw snde::snde_error("transaction_math object has exceeded lifetime of underlying transaction");
      }
      std::shared_ptr<snde::recdatabase> recdb_strong = self->recdb.lock();
      if (!recdb_strong) {
        throw snde::snde_error("transaction_math object has exceeded lifetime of underlying recording database");
      }
      std::shared_ptr<snde::channelconfig> config = std::make_shared<snde::channelconfig>(name,"math",false);
      config->math=true;
      config->math_fcn = result_chan->definition->instantiated;
      config->data_mutable = result_chan->definition->instantiated->is_mutable; // !!!Should probably be result index specific.
      std::shared_ptr<snde::reserved_channel> resvchan = recdb_strong->reserve_math_channel(trans_strong,config);
      
      self->pending_dict.emplace(name,std::make_pair(resvchan,result_chan));
    }

    std::shared_ptr<pending_math_definition_result_channel> __getitem__(std::string name)
    {
      auto dict_it = self->pending_dict.find(name);
      if (dict_it != self->pending_dict.end()) {
        return dict_it->second.second;
      }
      //if we don't find a pending math result,
      // we will search for an existing math channel
      // so we can display its information.
      // This will go into a pending_math_definition_result_channel
      // in "existing_mode"
      std::shared_ptr<snde::recdatabase> recdb_strong = self->recdb.lock();
      if (!recdb_strong) {
        throw snde::snde_error("transaction_math object has exceeded lifetime of underlying recording database");
      }

      {
        std::lock_guard<std::mutex> recdb_admin(recdb_strong->admin);
        auto defined_it = recdb_strong->_instantiated_functions.defined_math_functions.find(name);
        if (defined_it != recdb_strong->_instantiated_functions.defined_math_functions.end()) {
          //found an existing math channel
          std::shared_ptr<snde::instantiated_math_function> instantiated=defined_it->second;
          return std::make_shared<snde::pending_math_definition_result_channel>(instantiated->definition->definition_command,name);
        }
      }
      throw snde::snde_indexerror("no such math channel found");
    }

    void __delitem__(std::string name)
    {
      auto dict_it = self->pending_dict.find(name);
      if (dict_it != self->pending_dict.end()) {
        self->pending_dict.erase(dict_it);
      }
    }
  }
  
  
  class transaction {
  public:
    // mutable until end of transaction when it is destroyed and converted to a globalrev structure
    //std::mutex admin; // last in the locking order except before python GIL. Must hold this lock when reading or writing structures within. Does not cover the channels/recordings themselves.

    
    uint64_t rss_unique_index; // unique_index field that will be assigned to the recording_set_state. Immutable once published
    std::multimap<std::string,std::pair<std::shared_ptr<reserved_channel>,std::shared_ptr<channelconfig>>> updated_channels;
    //Keep track of whether a new recording is required for the channel (e.g. if it has a new owner) (use false for math recordings)
    std::map<std::string,bool> new_recording_required; // index is channel name for updated channels
    std::unordered_map<std::string,std::pair<std::shared_ptr<reserved_channel>,std::shared_ptr<recording_base>>> new_recordings; // Note: for now reserved_channel may be nullptr for a math channel (but that wouldn't be recorded here, would it?)

    // end of transaction propagates this structure into an update of recdatabase._channels
    // and a new globalrevision

    std::vector<std::shared_ptr<channel_notify>> pending_channel_notifies; // channel_notifies that need to be applied to the globalrev once it is created

    // when the transaction is complete, resulting_globalrevision() is assigned
    // so that if you have a pointer to the transaction you can get the globalrevision (which is also a recording_set_state)
   std::shared_ptr<globalrevision> _resulting_globalrevision; // locked by transaction admin mutex; use globalrev() or globalrev_nowait() accessor.

    std::shared_ptr<rss_reference> our_state_reference; // This pointer is created with the transaction. The reference is filled in by _realize_transaction() and MUST BE CLEARED BY THE TRANSACTION_MANAGER AFTER the transaction is realized, and that allows the state to go out of existence.
    
    std::shared_ptr<rss_reference> prerequisite_state; // This pointer is created with the transaction. THE REFERENCE MUST BE FILLED IN BY THE TRANSACTION_MANAGER. The pointer MUST BE CLEARED BY THE TRANSACTION_MANAGER AFTER the transaction is realized and that allows the prerequisite state to go out of existence once other pointers are cleared.
    
    //std::mutex transaction_background_end_lock; // locks the function and pointer immediately below. Last in the locking order
    
    //std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<void> params)> transaction_background_end_fcn;
    std::shared_ptr<void> transaction_background_end_params;
    
    std::shared_ptr<transaction_math> math;


    
    transaction(std::shared_ptr<recdatabase> recdb);
    // rule of 3
    transaction& operator=(const transaction &) = delete; 
    transaction(const transaction &orig) = delete;
    virtual ~transaction();
    
    void register_new_rec(std::shared_ptr<reserved_channel> chan,std::shared_ptr<recording_base> new_rec);
    
    std::shared_ptr<globalrevision> globalrev(); // Wait for the globalrev resulting from this transaction to be complete.
    std::shared_ptr<globalrevision> globalrev_wait(); // Wait for the globalrev resulting from this transaction to be complete. Unlike globalrev() this function will not throw an exception for incomplete recordings; instead it will wait for them to be complete.
    std::shared_ptr<globalrevision> globalrev_available(); // Wait for the globalrev resulting from this transaction to exist, but not necessarily be complete. 
    
    std::shared_ptr<globalrevision> globalrev_nowait(); // Return the globalrevision resulting from this transaction if it exists yet. Returns nullptr otherwise. Even if the globalrev exists, it may not be complete.
    
    //virtual void end_transaction(std::shared_ptr<recdatabase> recdb);
    //std::tuple<std::shared_ptr<globalrevision>,transaction_notifies> _realize_transaction(std::shared_ptr<recdatabase> recdb,uint64_t globalrevision_index);
    
    //void _notify_transaction_globalrev(std::shared_ptr<recdatabase> recdb_strong,std::shared_ptr<globalrevision> globalrev,struct transaction_notifies trans_notify);
    


    bool transaction_globalrev_is_complete();

    // Wait on the return value of get_transaction_globalrev_complete_waiter() by
    // calling its wait_interruptable method.
    // You can call its interrupt() method from another thread to
    // cancel the wait. 
    std::shared_ptr<promise_channel_notify> get_transaction_globalrev_complete_waiter(); 

    
  };

  // Typemaps for run_in_background_and_end_transaction()
  %typemap(in) (std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<void> params)> fcn) {
    Py_INCREF($input); // Matched by the Py_DECREF($input) in the lambda, below)

    PyObject *ribaet_fcn_copy = $input;
    $1 = [ ribaet_fcn_copy ] (std::shared_ptr<snde::recdatabase> recdb,std::shared_ptr<void> params) { // lambda
      std::shared_ptr<PyObject *> ParamsPy = std::static_pointer_cast<PyObject *>(params);

      PyGILState_STATE gstate;
      gstate = PyGILState_Ensure(); // acquire the GIL -- this is run in a gil-free context
      
      //PyObject *ret = PyObject_CallFunction(ribaet_fcn_copy,(char *)"O",*ParamsPy);
      PyObject *ret = PyObject_Call(ribaet_fcn_copy,*ParamsPy,nullptr);
      
      // we don't care about the return value
      if (ret) {
	Py_DECREF(ret);	
      } else {
	// Print the Python exception information
	PyErr_PrintEx(0);
      }
      // This lambda only runs once so this DECREF matches the above INCREF. 
      Py_DECREF(ribaet_fcn_copy);

      // This lambda only runs once so this DECREF matches the INCREF in the params typemap
      Py_DECREF(*ParamsPy);

      PyGILState_Release(gstate); // release the GIL -- this is run in a gil-free context

      /* No Python API allowed beyond this point. */


    };

  }

  // Typemaps for run_in_background_and_end_transaction()
  %typemap(in) (std::shared_ptr<void> params) {
    Py_INCREF($input); // Matched by the Py_DECREF(*ParamsPy) in the
    // std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<void> params)> fcn
    // lambda, above.

    $1 = std::make_shared<PyObject *>($input);

  }


  class active_transaction /* : public std::enable_shared_from_this<active_transaction> */ {
    // RAII interface to transaction
    // Don't use this directly from dataguzzler-python, because there
    // you need to drop the thread context before starting the
    // transaction and then reacquire after you have the
    // transaction lock.
  public:
    std::shared_ptr<recdatabase> recdb;
    //std::shared_ptr<globalrevision> previous_globalrev;
    std::shared_ptr<transaction> trans;
    bool transaction_ended;



    
    // Direct creationg of active_transaction object disabled in Python. Use recdb.start_transaction()
    // instead because that handles dropping dataguzzler-python locks prior to acquiring the
    // transaction lock correctly. 
    //active_transaction(std::shared_ptr<recdatabase> recdb);


    // rule of 3
    active_transaction& operator=(const active_transaction &) = delete; 
    active_transaction(const active_transaction &orig) = delete;
    ~active_transaction(); // destructor releases transaction_lock from holder

    std::shared_ptr<transaction> end_transaction();
    std::shared_ptr<transaction> run_in_background_and_end_transaction(std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<void> params)> fcn, std::shared_ptr<void> params);
    %pythoncode %{
        
      @property
      def math(self):
        
                 
        return self.trans.math



      # context manager protocol
      def __enter__(self):
        return self

      def __exit__(self,exc_type,exc_value,traceback):
        transobj=self.end_transaction()
        # copy bound methods from the transaction object
        # into our active_transaction object so that
        # they are readily available
        setattr(self,"globalrev",transobj.globalrev)
        setattr(self,"globalrev_wait",transobj.globalrev_wait)
        setattr(self,"globalrev_available",transobj.globalrev_available)
        setattr(self,"globalrev_nowait",transobj.globalrev_nowait)
        pass

      def __str__(self):
        if hasattr(self,"globalrev"):
          # after with statement
          return "active_transaction post-end.\nSee .globalrev(), .globalrev_wait(), .globalrev_available(), .globalrev_nowait()\n"
        else:
          return "Active transaction object.\nSee recdb.reserve_channel(), recdb.create_recording(), and .math\n"
        pass

      def __repr__(self):
        return self.__str__()
    %}
  };

  
  
  class channelconfig {
    // The channelconfig is immutable once published; However it may be copied, privately updated by its owner, (if subclassed, you must use the correct subclasses
    // copy constructor!) and republished.  It may be
    // freed once no references to it exist any more.
    // can be subclassed to accommodate e.g. geometry scene graph entries, geometry parameterizations, etc. 
  public:
    
    std::string channelpath; // Path of this channel in recording database
    std::string owner_name; // Name of owner, such as a dataguzzler_python module
    // void *owner_id; // pointer private to the owner (such as dataguzzler_python PyObject of the owner's representation of this channel) that
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

    channelconfig(std::string channelpath, std::string owner_name,bool hidden,std::shared_ptr<recording_storage_manager> storage_manager=nullptr);
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
    
    std::string channelpath; // full channel path (immutable)
    
    // std::shared_ptr<channelconfig> _config; // atomic shared pointer to immutable data structure; nullptr for a deleted channel
    std::shared_ptr<reserved_channel> _realized_owner; // atomic shared pointer
    %immutable;
    /*std::atomic<*/uint64_t/*>*/ latest_revision; // 0 means "invalid"; generally 0 (or previous latest) during channel creation/undeletion; incremented to 1 (with an empty recording if necessary) after transaction end 
    // /*std::atomic<*/bool/*>*/ deleted; // is the channel currently defined?
    %mutable;
       
    //std::mutex admin; // last in the locking order except before python GIL. Used to ensure there can only be one _config update at a time. 


    channel(std::string channelpath,std::shared_ptr<reserved_channel> initial_owner);

    // Any config referenced here can only represent
    // the current status that is realized (from _realize_transaction())
    // not what may be pending in transactions. Therefore, this config()
    // is pretty much useless and generally shouldn't be used.
    // Need to modify transaction structure to track modified channels
    // by the new channelconfig rather than by channel pointer.
    // This way, the modifications get queued with the transaction
    // and synchronized in _realize_transaction(), which is intrisically
    // serial.
    
    std::shared_ptr<channelconfig> realized_config(); // Use this method to safely get the current channelconfig pointer from the most recently realized transaction
    std::shared_ptr<reserved_channel> realized_owner(); // Use this method to safely get the current owning reserved_channel pointer from the most recently realized transaction.

    
    std::shared_ptr<reserved_channel> begin_atomic_realized_owner_update(); // channel admin lock must be locked when calling this function.
  
   
      
   
    
    void end_atomic_realized_owner_update(std::shared_ptr<reserved_channel> new_owner); // admin must be locked when calling this function. It accepts the modified copy of the atomically guarded data

     
  };

  class reserved_channel {
  public:
    // A reserved_channel represents that a channel has been defined
    // for the purpose of storing recordings. The reserved_channel pointer
    // will be passed when defining new recordings for the channel.

     // std::mutex admin; // last in the locking order except before python GIL. Used to ensure there can only be one _config update at a time.
    std::shared_ptr<reserved_channel> math_actual_reschan;
    std::shared_ptr<channelconfig> _proposed_config; // Atomic shared pointer to current proposed configuration (latest changes). Will be nullptr if the channel is deleted.
    std::shared_ptr<channelconfig> _realized_config; // Atomic shared pointer to current realized configuration (status as of the most recently realized transaction). Will be nullptr if the channel is deleted.


    std::shared_ptr<channel> chan; // Immutable
    reserved_channel();
    
    reserved_channel(const reserved_channel& other);
    reserved_channel& operator=(const reserved_channel&) =delete;
    std::shared_ptr<channelconfig> proposed_config(); //Use this method to get the most recently assigned configuration.
    std::shared_ptr<channelconfig> realized_config(); // Use this method to safely get the current channelconfig pointer for the most recently realized transaction. Returns nullptr for a deleted channel.

    // NOTE: the config may only be updated by various calls
    // that reconfigure a channel
    // so don't call these update methods yourself.
    // That means this template complexity may be unnecessary.
    template<typename T>
    std::shared_ptr<T> begin_atomic_proposed_config_update()
    // channel admin lock must be locked when calling this function. It is a template because channelconfig can be subclassed. Call it as begin_atomic_proposed_config_update<channelconfig>() if you need to subclass It returns a new modifiable copy of the atomically guarded data
    // (it returns nullptr if the existing config doesn't match T)
    {
      std::shared_ptr<T> old_config = std::dynamic_pointer_cast<T>(std::atomic_load(&_proposed_config));
      if (!old_config) {
	return nullptr;
      }
      std::shared_ptr<T> new_config=std::make_shared<T>(*old_config);
      return new_config;
    }
    
    void end_atomic_proposed_config_update(std::shared_ptr<channelconfig> new_config); // admin must be locked when calling this function. It accepts the modified copy of the atomically guarded data
    
 
  
    // NOTE: the config may only be updated by realize_transaction()
    // so don't call these update methods yourself.
    // That means this template complexity is unnecessary.
    template<typename T>
    std::shared_ptr<T> begin_atomic_realized_config_update()
    // channel admin lock must be locked when calling this function. It is a template because channelconfig can be subclassed. Call it as begin_atomic_config_update<channelconfig>() if you need to subclass It returns a new modifiable copy of the atomically guarded data
    // (it returns nullptr if the existing config doesn't match T)
    {
      std::shared_ptr<T> old_config = std::dynamic_pointer_cast<T>(std::atomic_load(&_realized_config));
      if (!old_config) {
	return nullptr;
      }
      std::shared_ptr<T> new_config=std::make_shared<T>(*old_config);
      return new_config;
    }
    
    void end_atomic_realized_config_update(std::shared_ptr<channelconfig> new_config); // admin must be locked when calling this function. It accepts the modified copy of the atomically guarded data
    
  };  
}
%template(reserved_channel_vector) std::vector<std::shared_ptr<snde::reserved_channel>>;

namespace snde { 
  class channel_state {
  public:
    // for atomic updates to notify_ ... atomic shared pointers, you must lock the recording_set_state's admin lock
    std::shared_ptr<channelconfig> config; // immutable
    std::shared_ptr<reserved_channel> chan; // immutable
    // std::shared_ptr<channel> _channel; // immutable pointer, but pointed data is not immutable, (but you shouldn't generally need to access this)
    std::shared_ptr<recording_base> _rec; // atomic shared ptr to recording structure created to store the ouput; may be nullptr if not (yet) created. Always nullptr for ondemand recordings... recording contents may be mutable but have their own admin lock

    %immutable;
    /*std::atomic<*/bool/*>*/ updated; // this field is only valid once rec() returns a valid pointer and once rec()->state is READY or METADATAREADY. It is true if this particular recording has a new revision particular to the enclosing recording_set_state
    %mutable;
    std::shared_ptr<uint64_t> _revision; // This is assigned when the channel_state is created from _rec->info->revision for manually created recordings. (For ondemand math recordings this is not meaningful?) For math recordings with the math_function's new_revision_optional (config->math_fcn->fcn->new_revision_optional) flag clear, this is defined during end_transaction() before the channel_state is published. If the new_revision_optional flag is set, this left nullptr; once the math function determines whether a new recording will be instantiated the revision will be assigned when the recording is define, with ordering ensured by the implicit self-dependency implied by the new_revision_optional flag (recmath_compute_resource.cpp)
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> _notify_about_this_channel_metadataonly; // atomic shared ptr to immutable set of channel_notifies that need to be updated or perhaps triggered when this channel becomes metadataonly; set to nullptr at end of channel becoming metadataonly. 
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> _notify_about_this_channel_ready; // atomic shared ptr to immutable set of channel_notifies that need to be updated or perhaps triggered when this channel becomes ready; set to nullptr at end of channel becoming ready. 

    channel_state(std::shared_ptr<reserved_channel> owner,std::shared_ptr<channelconfig> config,std::shared_ptr<recording_base> rec,bool updated);

    channel_state(const channel_state &orig); // copy constructor used for initializing channel_map from prototype defined in realize_transaction()

    // Copy assignment operator deleted
    channel_state& operator=(const channel_state &) = delete;

    // default destructor
    ~channel_state() = default; 


    std::shared_ptr<recording_base> rec() const;
    //std::shared_ptr<uint64_t> revision() const;
    std::shared_ptr<recording_base> recording_is_complete(bool mdonly); // uses only atomic members so safe to call in all circumstances. Set to mdonly if you only care that the metadata is complete. Normally call recording_is_complete(false). Returns recording pointer if recording is complete to the requested condition, otherwise nullptr. 
    void issue_nonmath_notifications(std::shared_ptr<recording_set_state> rss); // Must be called without anything locked. Issue notifications requested in _notify* and remove those notification requests
    void issue_math_notifications(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_set_state> rss); // Must be called without anything locked. Check for any math updates from the new status of this recording
    
    void end_atomic_rec_update(std::shared_ptr<recording_base> new_recording);


    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> begin_atomic_notify_about_this_channel_metadataonly_update();
    void end_atomic_notify_about_this_channel_metadataonly_update(std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> newval);
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_metadataonly();

    
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> begin_atomic_notify_about_this_channel_ready_update();
    void end_atomic_notify_about_this_channel_ready_update(std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> newval);
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_ready();
  };

};
// Make channel_map type accessible from python (marked as shared_ptr, above)
//%template(ChannelMap) std::map<std::string,snde::channel_state>;


namespace snde {
  
  class recording_status {
  public:
    //std::shared_ptr<std::map<std::string,channel_state>> channel_map; // key is full channel path... The map itself (not the embedded states) is immutable once the recording_set_state is published
    
    /// all of these are indexed by their their full path. Every entry in channel_map should be in exactly one of these. Locked by rss admin mutex per above
    // The index is the shared_ptr in globalrev_channel.config
    // primary use for these is determining when our globalrev/rss is
    // complete: Once call recordings are in metadataonly or completed,
    // then it should be complete
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state *> defined_recordings;
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state *> instantiated_recordings;
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state *> metadataonly_recordings; // only move recordings to here if they are mdonly recordings
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state *> completed_recordings;

    recording_status(const std::map<std::string,channel_state> & channel_map_param);
  };

  // return a process-wide unique (incrementing) identifier. Process wide
  // so that even if you run multiple recording databases you still won't
  // get a collision. Use a value from this function as the 
  uint64_t rss_get_unique();

  class recording_set_state /* : public std::enable_shared_from_this<recording_set_state>*/ {
  public:
    //std::mutex admin; // locks changes to recstatus including channel_map contents (map itself is immutable once published), mathstatus,  and the _recordings reference maps/sets and notifiers. Precedes recstatus.channel_map.rec.admin and Python GIL in locking order
    uint64_t originating_globalrev_index; // this rss may not __be__ a globalrev but it (almost certainly?) is built on one. 
    uint64_t unique_index; // This is a number unique within the process for this RSS. Used to disambiguate shared memory names, for example. Immutable post-construction 
    std::weak_ptr<recdatabase> recdb_weak;
    %immutable;
    /*std::atomic<*/bool/*>*/ ready; // indicates that all recordings except ondemand and data_requestonly recordings are READY (mutable recordings may be OBSOLETE)
    %mutable;
       
    recording_status recstatus;
    math_status mathstatus; // note math_status.math_functions is immutable
    std::shared_ptr<rss_reference> _prerequisite_state; // This atomic shared pointer is cleared when the rss is complete and that allows the prerequisite state to go out of existence. Use prerequisite_state() accessor.
    
    std::shared_ptr<rss_reference> our_state_reference; // This pointer is cleared when the rss is complete, and that allows the state to go out of existence.

    std::unordered_set<std::shared_ptr<channel_notify>> recordingset_complete_notifiers; // Notifiers waiting on this recording set state being complete. Criteria will be removed as they are satisifed and entries will be removed as the notifications are performed.

    std::shared_ptr<lockmanager> lockmgr; // pointer is immutable after initialization

    
    recording_set_state(std::shared_ptr<recdatabase> recdb,const instantiated_math_database &math_functions,const std::map<std::string,channel_state> & channel_map_param,std::shared_ptr<rss_reference> prerequisite_state,uint64_t originating_globalrev_index,uint64_t unique_index); // constructor
    // Rule of 3
    recording_set_state& operator=(const recording_set_state &) = delete; 
    recording_set_state(const recording_set_state &orig) = delete;
    virtual ~recording_set_state()=default;

    void wait_complete(); // wait for all the math in this recording_set_state or globalrev to reach nominal completion (metadataonly or ready, as configured)
    std::string print_math_status(bool verbose=false);
    std::string print_recording_status(bool verbose=false);

    std::shared_ptr<recording_base> get_recording(const std::string &fullpath);
    std::shared_ptr<rss_reference> prerequisite_state();
    void prerequisite_state_clear();
    void prerequisite_state_assign(std::shared_ptr<rss_reference> state);
    std::shared_ptr<ndarray_recording_ref> get_ndarray_ref(const std::string &fullpath,size_t array_index=0);
    std::shared_ptr<ndarray_recording_ref> get_ndarray_ref(const std::string &fullpath,std::string array_name);

    std::shared_ptr<ndarray_recording_ref> check_for_recording_ref(const std::string &fullpath,size_t array_index=0);
    std::shared_ptr<ndarray_recording_ref> check_for_recording_ref(const std::string &fullpath,std::string array_name);

    std::shared_ptr<std::vector<std::string>> list_recordings();

#ifdef SIZEOF_LONG_IS_8 // this is a SWIG workaround -- see spatialnde2.i
    std::shared_ptr<std::vector<std::pair<std::string,unsigned long>>> list_recording_revisions();
#else
    std::shared_ptr<std::vector<std::pair<std::string,unsigned long long>>> list_recording_revisions();
#endif    
    std::shared_ptr<std::vector<std::pair<std::string,std::string>>> list_ndarray_refs();

       
    long get_reference_count(); // get the shared_ptr reference count; useful for debugging memory leaks

    size_t num_complete_notifiers(); // size of recordingset_complete_notifiers; useful for debugging memory leaks
    
    std::string get_math_status();
    std::string get_math_function_status(std::string definition_command);

  };
class rss_reference {
    // A single rss_reference to a particular rss or globalrev is
    // created during the transaction that created the rss. It should
    // never be copied and no other reference to the same rss should
    // be created until the rss actually exists. But, shared pointers
    // to it can be passed around. Its internal shared pointer gets
    // initialized when the transaction is realized. It is stored in
    // recordings as prerequisite_state and/or originating_state and
    // the shared_ptr to the rss_reference gets cleared when the
    // recording is complete. As such, it holds previous state in
    // memory as long as needed, but no longer. It is also stored in
    // recording_set_states as prerequisite_state. That pointer is
    // likewise cleared when the rss or globalrev is complete.
  public:
  // std::shared_ptr<recording_set_state> _rss; // atomic shared pointer; use rss() accessor
    rss_reference();
    rss_reference(std::shared_ptr<recording_set_state> rss);
    // ~rss_reference();
    std::shared_ptr<recording_set_state> rss();
    void rss_assign(std::shared_ptr<recording_set_state> rss);
    };
  %extend rss_reference {
    std::string __str__()
    {
      return snde::ssprintf("rss_reference 0x%llx to 0x%llx",(unsigned long long)self,(unsigned long long)self->rss().get());
    }

  }
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
    //std::shared_ptr<transaction> defining_transact; // This keeps the transaction data structure (pointed to by weak pointers in the recordings created in the transaction) in memory at least as long as the globalrevision is current. 

    std::shared_ptr<globalrev_mutable_lock> mutable_recordings_need_holder;
    //std::atomic<bool> mutable_recordings_still_needed; 

    
    globalrevision(uint64_t globalrev, std::shared_ptr<transaction> defining_transact, std::shared_ptr<recdatabase> recdb,const instantiated_math_database &math_functions,const std::map<std::string,channel_state> & channel_map_param,std::shared_ptr<rss_reference> prerequisite_state,uint64_t rss_unique_index);   
    %pythoncode %{
      @property
      def rec(self):
        return rec_helper(self)

      @property
      def ref(self):
        return ref_helper(self)

      def __str__(self):
        return f"globalrevision #{self.globalrev:d}.\nSee .rec for recordings and .ref for ndarray_recording_refs.\n"

      def __repr__(self):
        return self.__str__()
    %}
  };
  
  %pythoncode %{
    class rec_helper:
      globalrev=None
      def __init__(self,globalrev):
        self.globalrev=globalrev
        pass


      def __getitem__(self,name):
        return self.globalrev.get_recording(name)

      def __str__(self):
        return str(self.globalrev.list_recordings())

      def __repr__(self):
        return self.__str__()
      pass

    class ref_helper:
      globalrev=None
      def __init__(self,globalrev):
        self.globalrev=globalrev
        pass


      def __getitem__(self,name):
        if isinstance(name,tuple):
          # actually a tuple of (name,array_index) or (name,array_name)
          return self.globalrev.get_ndarray_ref(*name)
          
        return self.globalrev.get_ndarray_ref(name)

      def __str__(self):
        return str(self.globalrev.list_ndarray_refs())


      def __repr__(self):
        return self.__str__()
      pass


        

  %}
  class recdatabase /* : public std::enable_shared_from_this<recdatabase> */ {
  public:
    //std::mutex admin; // Locks access to _channels and _deleted_channels and _math_functions, _globalrevs and repetitive_notifies. In locking order, precedes channel admin locks, available_compute_resource_database, globalrevision admin locks, recording admin locks, and Python GIL. 
    std::map<std::string,std::shared_ptr<channel>> _channels; // Generally don't use the channel map directly. Grab the latestglobalrev and use the channel map from that. 
    // std::map<std::string,std::shared_ptr<channel>> _deleted_channels; // Channels are put here after they are deleted. They can be moved back into the main list if re-created. 
    instantiated_math_database _instantiated_functions; 
    
    std::map<uint64_t,std::shared_ptr<globalrevision>> _globalrevs; // Index is global revision counter. The first element in this is the latest globalrev with all mandatory immutable channels ready. The last element in this is the most recently defined globalrev.
    //std::shared_ptr<globalrevision> _latest_defined_globalrev; // atomic shared pointer -- access with latest_defined_globalrev() method;
    //std::shared_ptr<globalrevision> _latest_ready_globalrev; // atomic shared pointer -- access with latest_globalrev() method;
    std::vector<std::shared_ptr<repetitive_channel_notify>> repetitive_notifies; 

    std::shared_ptr<allocator_alignment> alignment_requirements; // Pointer is immutable; pointed structure has its own locking
    std::shared_ptr<available_compute_resource_database> compute_resources; // has its own admin lock.
    

    std::shared_ptr<memallocator> lowlevel_alloc; // pointer is immutable once created during startup; contents not necessarily immutable; see memallocator.hpp
    std::shared_ptr<recording_storage_manager> default_storage_manager; // pointer is immutable once created during startup; contents not necessarily immutable; see recstore_storage.hpp

    std::shared_ptr<lockmanager> lockmgr; // pointer immutable after initialization; contents have their own admin lock, which is used strictly internally by them
    /*std::atomic<*/bool/*>*/ started;

    //std::mutex transaction_lock; // ***!!! Before any dataguzzler-python module locks, etc.
    //std::shared_ptr<transaction> current_transaction; // only valid while transaction_lock is held.

    std::shared_ptr<transaction_manager> transmgr; // pointer is immutable once created during startup.
    std::set<std::weak_ptr<monitor_globalrevs>,std::owner_less<std::weak_ptr<monitor_globalrevs>>> monitoring;
    uint64_t monitoring_notify_globalrev; // latest globalrev for which monitoring has already been notified


    std::list<std::shared_ptr<globalrevision>> globalrev_mutablenotneeded_pending;
    bool globalrev_mutablenotneeded_mustexit;

    std::unordered_map<std::string,std::shared_ptr<reserved_channel>> reserved_math_channels; // locked by admin lock
    
    //recdatabase(std::shared_ptr<recording_storage_manager> default_storage_manager=nullptr,std::shared_ptr<lockmanager> lockmgr=nullptr);
    // default argument split into three separate entries here
    // to work around swig bug with default parameter
    
    recdatabase(std::shared_ptr<lockmanager> lockmgr);
    
    recdatabase();

    
    recdatabase & operator=(const recdatabase &) = delete; 
    recdatabase(const recdatabase &orig) = delete;
    ~recdatabase();

    void add_alignment_requirement(size_t nbytes); // note all alignment requirements (e.g. from GPU's) must be added before initializing storage managers

    void startup(); // gets the math engine running, etc. 


    // a transaction update can be multi-threaded but you shouldn't call end_transaction()  (or the end_transaction method on the
    // active_transaction or delete the active_transaction) until all other threads are finished with transaction actions

    // NOTE start_transaction is wrapped manually with an %extend block, below
    //std::shared_ptr<active_transaction> start_transaction();
    std::shared_ptr<transaction> end_transaction(std::shared_ptr<active_transaction> act_trans);

    std::shared_ptr<transaction> run_in_background_and_end_transaction(std::shared_ptr<active_transaction> act_trans,std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<void> params)> fcn, std::shared_ptr<void> params);
    std::shared_ptr<reserved_channel> reserve_math_channel(std::shared_ptr<transaction> trans,std::shared_ptr<channelconfig> new_config);
    std::shared_ptr<reserved_channel> lookup_math_channel_recdb_locked(std::string channelname);
    std::shared_ptr<reserved_channel> lookup_math_channel(std::string channelname);
    
    // add_math_function() must be called within a transaction
    std::vector<std::shared_ptr<reserved_channel>> add_math_function(std::shared_ptr<active_transaction> trans,std::shared_ptr<instantiated_math_function> new_function,bool hidden); // Use separate functions with/without storage manager because swig screws up the overload
    std::shared_ptr<instantiated_math_function> lookup_math_function(std::string fullpath);
    std::vector<std::string> list_math_function_defs();
    void delete_math_function(std::shared_ptr<active_transaction> trans,std::vector<std::string> chans,std::shared_ptr<instantiated_math_function> fcn);
    std::vector<std::shared_ptr<reserved_channel>> add_math_function_storage_manager(std::shared_ptr<active_transaction> trans,std::shared_ptr<instantiated_math_function> new_function,bool hidden,std::shared_ptr<recording_storage_manager> storage_manager);
    void send_math_message(std::shared_ptr<active_transaction> trans,std::shared_ptr<instantiated_math_function> func, std::string name, std::shared_ptr<math_instance_parameter> msg);

    //void register_new_rec(std::shared_ptr<recording_base> new_rec);
    //void register_new_math_rec(void *owner_id,std::shared_ptr<recording_set_state> calc_rss,std::shared_ptr<recording_base> new_rec); // registers newly created math recording in the given rss (and extracts mutable flag for the given channel into the recording structure)).
    
    std::shared_ptr<globalrevision> latest_defined_globalrev(); // safe to call with or without recdb admin lock held

    std::shared_ptr<globalrevision> latest_globalrev(); // safe to call with or without recdb admin lock held. Returns latest globalrev which is ready and for which all prior globalrevs are ready

    std::shared_ptr<globalrevision> get_globalrev(uint64_t revnum);

    // Allocate channel with a specific name; returns nullptr if the name is inuse
    

    std::shared_ptr<reserved_channel> reserve_channel(std::shared_ptr<active_transaction> trans,std::shared_ptr<channelconfig> new_config); // must be called within a transaction
    std::shared_ptr<reserved_channel> reserve_channel(std::shared_ptr<transaction> trans,std::shared_ptr<channelconfig> new_config); // must be called within a transaction
    void release_channel(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan); // must be called within a transaction

    // Define a new channel; throws an error if the channel is already in use
    //std::shared_ptr<channel> define_channel(std::string channelpath, std::string owner_name, void *owner_id, bool hidden=false, std::shared_ptr<recording_storage_manager> storage_manager=nullptr);

    std::shared_ptr<reserved_channel> define_channel(std::shared_ptr<active_transaction> trans,std::string channelpath, std::string owner_name);

    std::shared_ptr<reserved_channel> define_channel(std::shared_ptr<active_transaction> trans,std::string channelpath, std::string owner_name, bool hidden);
    

    std::shared_ptr<reserved_channel> define_channel(std::shared_ptr<active_transaction> trans,std::string channelpath, std::string owner_name, bool hidden, std::shared_ptr<recording_storage_manager> storage_manager);
    

    //std::shared_ptr<channel> lookup_channel_live(std::string channelpath);


    // NOTE: python wrappers for wait_recordings and wait_recording_names need to drop dgpython thread context during wait and poll to check for connection drop
    //void wait_recordings(std::vector<std::shared_ptr<recording>> &);
    void wait_recording_names(std::shared_ptr<recording_set_state> rss,const std::vector<std::string> &metadataonly, const std::vector<std::string> fullyready);

    std::shared_ptr<monitor_globalrevs> start_monitoring_globalrevs();
    std::shared_ptr<monitor_globalrevs> start_monitoring_globalrevs(std::shared_ptr<globalrevision> first);
    std::shared_ptr<monitor_globalrevs> start_monitoring_globalrevs(std::shared_ptr<globalrevision> first ,bool inhibit_mutable);
    void globalrev_mutablenotneeded_code(); 
    //void transaction_background_end_code();


    std::shared_ptr<math_function_registry_map> available_math_functions();

    std::shared_ptr<math_function> lookup_available_math_function(std::string name);
    std::shared_ptr<std::vector<std::string>> list_available_math_functions();

    %pythoncode %{
      @property
      def latest(self):
        return self.latest_globalrev()

      @property      
      def math(self):
        return recdb_math_helper(self)

      def __str__(self):
        latest_ready = self.latest_globalrev()
        latest_defined = self.latest_defined_globalrev()
        return f"spatialnde2 recording database latest ready/defined = {latest_ready.globalrev:d}/{latest_defined.globalrev:d}\nLook at .latest for latest ready. See also .math\n"

      def __repr__(self):
        return self.__str__()

    %}
  };

  %pythoncode %{
    class recdb_math_helper:
      recdb=None
      def __init__(self,recdb):
        self.recdb = recdb
        pass

      def __getitem__(self,name):
        instantiated = self.recdb.lookup_math_function(name)
        return instantiated

      def __setitem__(self,name,pending_definition_channel):
        with self.recdb.start_transaction() as trans:
          trans.math[name] = pending_definition_channel
          pass
        pass

      def __str__(self):
        return str(self.recdb.list_math_function_defs())

      def __repr__(self):
        return self.__str__()

      pass
  %}
  // unfortunately this %nothread (suggested by https://sourceforge.net/p/swig/mailman/swig-user/?viewmonth=200902&style=flat&viewday=4 ) doesn't work.
  // instead it seems to affect things only beyond the %extend (?)
  // and then the %thread afterward is ineffective...
  // so instead we just use SWIG_PYTHON_THREAD_BEGIN_BLOCK...
  
  //%nothread;  // temporarily turn off swig threading support for this definition ***!!! MUST BE PAIRED WITH A %thread; after !!!***
  %extend recdatabase {
    // manual wrapping of start_transaction() so that we drop the dataguzzler_python context
    // (if present) while acquiring the transaction lock
   

    std::shared_ptr<active_transaction> start_transaction(std::shared_ptr<measurement_time> timestamp)
    {
      // ***!!!NOTE: See parallel code below in start_transaction() (with no parameter)
      PyObject *dgpython_context_module=nullptr;
      PyObject *PopThreadContext=nullptr;
      {
	SWIG_PYTHON_THREAD_BEGIN_BLOCK;
      
	
	// check for presence of dataguzzler-python (must have already been imported by something else)
	//dgpython_context_module = PyImport_ImportModule("dataguzzler_python.context");
	PyObject *dgpython_context_module_name = PyUnicode_FromString("dataguzzler_python.context");
	
	dgpython_context_module = PyImport_GetModule(dgpython_context_module_name);
	Py_DECREF(dgpython_context_module_name);
	if (dgpython_context_module) {
	  // get PopThreadContext() and PushThreadContext() functions
	  PopThreadContext = PyObject_GetAttrString(dgpython_context_module,"PopThreadContext");
	  PyObject *PushThreadContext = PyObject_GetAttrString(dgpython_context_module,"PushThreadContext");
	  
	  // Call PushThreadContext(None) to drop the current context
	  PyObject *ret = PyObject_CallFunction(PushThreadContext,(char *)"O",Py_None);

	  if (ret) {
	    Py_DECREF(ret);
	  }
	  else {
	    // Print the Python exception information
	    PyErr_PrintEx(0);  
	  }
	  
	  Py_DECREF(PushThreadContext);
	  
	} else {
	  PyErr_Clear();
	  //snde::snde_warning("start_transaction(): No dataguzzler_python context found");
	}
	
	
	// Drop the GIL and acquire the transaction lock
	SWIG_PYTHON_THREAD_END_BLOCK;
      }
      std::shared_ptr<snde::active_transaction> retval;
      //Py_BEGIN_ALLOW_THREADS;
      {
	//SWIG_PYTHON_THREAD_BEGIN_ALLOW;
	retval = self->start_transaction(timestamp);
	//SWIG_PYTHON_THREAD_END_ALLOW;
      }
      //Py_END_ALLOW_THREADS;
      {
	SWIG_PYTHON_THREAD_BEGIN_BLOCK;
	
	// Pop the thread context to reaquire our context lock
	if (dgpython_context_module) {
	  PyObject *ret = PyObject_CallFunction(PopThreadContext,(char *)"");
	  Py_DECREF(ret);
	  Py_DECREF(PopThreadContext);
	  Py_DECREF(dgpython_context_module);
	}
	
	SWIG_PYTHON_THREAD_END_BLOCK;
      }
      return retval;
    }

     std::shared_ptr<active_transaction> start_transaction()
{
      // ***!!!NOTE: See parallel code below in start_transaction(timestamp) 
      PyObject *dgpython_context_module=nullptr;
      PyObject *PopThreadContext=nullptr;
      {
	SWIG_PYTHON_THREAD_BEGIN_BLOCK;
      
	
	// check for presence of dataguzzler-python (must have already been imported by something else)
	//dgpython_context_module = PyImport_ImportModule("dataguzzler_python.context");
	PyObject *dgpython_context_module_name = PyUnicode_FromString("dataguzzler_python.context");
	
	dgpython_context_module = PyImport_GetModule(dgpython_context_module_name);
	Py_DECREF(dgpython_context_module_name);
	if (dgpython_context_module) {
	  // get PopThreadContext() and PushThreadContext() functions
	  PopThreadContext = PyObject_GetAttrString(dgpython_context_module,"PopThreadContext");
	  PyObject *PushThreadContext = PyObject_GetAttrString(dgpython_context_module,"PushThreadContext");
	  
	  // Call PushThreadContext(None) to drop the current context
	  PyObject *ret = PyObject_CallFunction(PushThreadContext,(char *)"O",Py_None);

	  if (ret) {
	    Py_DECREF(ret);
	  }
	  else {
	    // Print the Python exception information
	    PyErr_PrintEx(0);  
	  }
	  
	  Py_DECREF(PushThreadContext);
	  
	} else {
	  PyErr_Clear();
	  //snde::snde_warning("start_transaction(): No dataguzzler_python context found");
	}
	
	
	// Drop the GIL and acquire the transaction lock
	SWIG_PYTHON_THREAD_END_BLOCK;
      }
      std::shared_ptr<snde::active_transaction> retval;
      //Py_BEGIN_ALLOW_THREADS;
      {
	//SWIG_PYTHON_THREAD_BEGIN_ALLOW;
	retval = self->start_transaction();
	//SWIG_PYTHON_THREAD_END_ALLOW;
      }
      //Py_END_ALLOW_THREADS;
      {
	SWIG_PYTHON_THREAD_BEGIN_BLOCK;
	
	// Pop the thread context to reaquire our context lock
	if (dgpython_context_module) {
	  PyObject *ret = PyObject_CallFunction(PopThreadContext,(char *)"");
	  Py_DECREF(ret);
	  Py_DECREF(PopThreadContext);
	  Py_DECREF(dgpython_context_module);
	}
	
	SWIG_PYTHON_THREAD_END_BLOCK;
      }
      return retval;
    }
    //%thread;  // reenable swig threading support after the %extend above. 
    

  };
  
  template <typename T>
  class ndtyped_recording_ref : public ndarray_recording_ref {
    // internals not swig-wrapped; we use typemaps below instead 
  };


  // implement input typemaps to give ndtyped_recording_ref<>...
  %typemap(input) ndtyped_recording_ref<snde_kdnode> (int res=0,void *argp,std::shared_ptr<ndtyped_recording_ref<snde_kdnode>> templated) {
    res = SWIG_ConvertPtr($input,&argp,$descriptor(ndarray_recording_ref), $disown | 0);
    if (!SWIG_IsOK(res)) {
      SWIG_exception_fail(SWIG_ArgError(res), "in method '" "$symname" "', argument "
			  "$argnum"" is not convertable to a snde::ndarray_recording_ref");
      
    }
    templated=std::dynamic_pointer_cast<ndtyped_recording_ref<snde_kdnode>>(*(std::shared_ptr<ndarray_recording_ref> *)argp);

    if (!templated) {
      SWIG_exception_fail(SWIG_ArgError(res), "in method '" "$symname" "', argument "
			  "$argnum"" is not convertable to a snde::ndtyped_recording_ref<" "snde_kdnode" ">");
      
    }
    $1 = templated; 
  };
  

  size_t recording_default_info_structsize(size_t param,size_t min);

  //template <typename T,typename ... Args>
  //std::shared_ptr<T> create_recording(std::shared_ptr<recdatabase> recdb,std::shared_ptr<reserved_channel> chan,Args && ... args);
  
  template <typename T,typename ... Args>
  std::shared_ptr<T> create_recording_math(std::shared_ptr<recdatabase> recdb,std::string chanpath,std::shared_ptr<recording_set_state> calc_rss,Args && ... args);
  
  // for non math-functions operating in a transaction
  template <typename T>
  std::shared_ptr<ndtyped_recording_ref<T>> create_typed_ndarray_ref(std::shared_ptr<recdatabase> recdb,std::shared_ptr<reserved_channel> chan);

  template <typename S,typename T,typename ... Args>
    std::shared_ptr<ndtyped_recording_ref<T>> create_typed_subclass_ndarray_ref(std::shared_ptr<recdatabase> recdb,std::shared_ptr<reserved_channel> chan,Args && ... args);


  // These next two templates are commented out because they
  // cause SWIG syntax errors for no apparent reason whatsoever (?)
  
  //template <typename T>
  //std::shared_ptr<ndtyped_recording_ref<T>> create_anonymous_typed_ndarray_ref(std::shared_ptr<recdatabase> recdb,std::string purpose,unsigned typenum=rtn_typemap.at(typeid(T))); // purpose is used for naming shared memory objects

  //template <typename S,typename T,typename ... Args>
  //std::shared_ptr<ndtyped_recording_ref<T>> create_anonymous_typed_subclass_ndarray_ref(std::shared_ptr<recdatabase> recdb,std::string purpose,Args && ... args); // purpose is used for naming shared memory objects

  
  // for math_recordings_only (no transaction)
  template <typename T>
  std::shared_ptr<ndtyped_recording_ref<T>> create_typed_ndarray_ref_math(std::string chanpath,std::shared_ptr<recording_set_state> calc_rss);


  template <typename S,typename T,typename ... Args>
  std::shared_ptr<ndtyped_recording_ref<T>> create_typed_subclass_ndarray_ref_math(std::string chanpath,std::shared_ptr<recording_set_state> calc_rss,Args && ... args);

  std::shared_ptr<ndarray_recording_ref> create_ndarray_ref(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,unsigned typenum);

  std::shared_ptr<ndarray_recording_ref> create_named_ndarray_ref(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,std::string arrayname,unsigned typenum);

  template <typename S,typename ... Args> 
  std::shared_ptr<ndarray_recording_ref> create_subclass_ndarray_ref(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,unsigned typenum,Args && ... args);

  std::shared_ptr<ndarray_recording_ref> create_anonymous_ndarray_ref(std::shared_ptr<recdatabase> recdb,std::string purpose,unsigned typenum); // purpose is used for naming shared memory objects

  std::shared_ptr<ndarray_recording_ref> create_anonymous_named_ndarray_ref(std::shared_ptr<recdatabase> recdb,std::string purpose,std::string arrayname,unsigned typenum); // purpose is used for naming shared memory objects

  template <typename S,typename ... Args> 
  std::shared_ptr<ndarray_recording_ref> create_anonymous_subclass_ndarray_ref(std::shared_ptr<recdatabase> recdb,std::string purpose,unsigned typenum, Args && ... args); // purpose is used for naming shared memory objects

  
  std::shared_ptr<ndarray_recording_ref> create_ndarray_ref_math(std::string chanpath,std::shared_ptr<recording_set_state> calc_rss,unsigned typenum); // math use only... ok to specify typenum as SNDE_RTM_UNASSIGNED if you don't know the final type yet. Then use assign_recording_type() method to get a new fully typed reference 
  std::shared_ptr<ndarray_recording_ref> create_named_ndarray_ref_math(std::string chanpath,std::shared_ptr<recording_set_state> calc_rss,std::string arrayname,unsigned typenum); // math use only... ok to specify typenum as SNDE_RTM_UNASSIGNED if you don't know the final type yet. Then use assign_recording_type() method to get a new fully typed reference 

  template <typename S,typename ... Args>
  std::shared_ptr<ndarray_recording_ref> create_subclass_ndarray_ref_math(std::string chanpath,std::shared_ptr<recording_set_state> calc_rss,unsigned typenum,Args && ... args); // math use only... ok to specify typenum as SNDE_RTM_UNASSIGNED if you don't know the final type yet. Then use assign_recording_type() method to get a new fully typed reference 

  // create recording templates
  // Work around SWIG not supporting variadic templates
  // by creating multiple non-variadic templates that
  // swig thinks exist, that are then mapped onto the
  // original by #define

  // first, template for no extra recording arguments
  template <class T>
    std::shared_ptr<T> create_recording_noargs(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan);
  %{
#define create_recording_noargs create_recording
   %}

  // template for one extra recording argument that is a shared_ptr to a std::string
  template <class T>
    std::shared_ptr<T> create_recording_ptr_to_string(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,std::shared_ptr<std::string> path_to_primary);
  %{
#define create_recording_ptr_to_string create_recording
   %}


  // template for one extra recording argument that is a std::string
  template <class T>
    std::shared_ptr<T> create_recording_string(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,std::string stringarg);
  %{
#define create_recording_string create_recording
   %}

  
  // template for one extra recording argument that is a size_t
  template <class T>
    std::shared_ptr<T> create_recording_size_t(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,size_t);
  %{
#define create_recording_size_t create_recording
   %}

  // template for one extra recording argument that is an unsigned
  template <class T>
    std::shared_ptr<T> create_recording_unsigned(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,unsigned);
  %{
#define create_recording_unsigned create_recording
   %}

  
    // template for one extra recording argument that is a const vector of string-orientation pairs
  template <class T> 
    std::shared_ptr<T> create_recording_const_vector_of_string_orientation_pairs(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,/* const */std::vector<std::pair<std::string,snde_orientation3>> /*&*/ pieces);
  %{
#define create_recording_const_vector_of_string_orientation_pairs create_recording
   %}

  // template for three extra string recording arguments
  template <class T> 
    std::shared_ptr<T> create_recording_three_strings(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,std::string param1,std::string param2,std::string param3);
  %{
#define create_recording_three_strings create_recording
   %}


  // template for textured_part info arguments

  // Moved to graphics_recording.i
  //  template <class T>
  //   std::shared_ptr<T> create_recording_textured_part_info(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,std::string part_name, std::shared_ptr<std::string> parameterization_name, std::vector<std::pair<snde_index,std::shared_ptr<image_reference>>> texture_refs);
  // %{
  //#define create_recording_textured_part_info create_recording
  // %}

  
  // template for recording ref for a ndarray subclass with one extra recording argument that is a std::string
  template <class T>
    std::shared_ptr<ndarray_recording_ref> create_subclass_ndarray_ref_string(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,unsigned typenum,std::string stringarg);
  %{
#define create_subclass_ndarray_ref_string create_subclass_ndarray_ref
   %}


  
  %template(create_null_recording) snde::create_recording_noargs<snde::null_recording>;
  //%template(create_recording_group) snde::create_recording_ptr_to_string<snde::recording_group>;
  %template(create_recording_group) snde::create_recording_noargs<snde::recording_group>;
  %template(create_multi_ndarray_recording) snde::create_recording_size_t<snde::multi_ndarray_recording>;
  %template(create_fusion_ndarray_recording) snde::create_recording_unsigned<snde::fusion_ndarray_recording>;

  
};
