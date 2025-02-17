// Mostly immutable recstore concept:

// "Channel" is a permanent, non-renameable
// identifier/structure that keeps track of revisions
// for each named address (recording path).
// Channels can be "Deleted" in which case
// they are omitted from current lists, but
// their revisions are kept so if the name
// is reused the revision counter will
// increment, not reset.
// A channel is "owned" by a module, which
// is generally the only source of new
// recordings on the channel.
// The module requests a particular channel
// name, but the obtained name might
// be different if the name is already in use.

// Channels can have sub-channels, generally with
// their own data. In some cases -- for example
// a renderable geometry collection -- the parent channel
// mediates the storage of sub-channels.
//
// Channel cross-references are usually by name
// but an exception is possible for renderable
// geometry collections. 

// The recording database keeps a collection
// of named channels and the current owners.
// Entries can only be
// added or marked as deleted during a
// transaction.

// "Recording" represents a particular revision
// of a channel. The recording structure itself
// and metadata is always immutable once READY,
// except for the state field and the presence
// of an immutable data copy (mutable only).
// The underlying data can be
// mutable or immutable.
// Recordings have three states: INITIALIZING, READY,
// and OBSOLETE (mutable only).
// INITIALIZING is the status while its creator is
// filling out the data structure
//
// Once the entire recording (including all metadata
// and the underlying data) is complete, the recording
// becomes READY. If the recording is mutable, before
// its is allowed to be mutated, the recording should
// be marked as OBSOLETE

// Mutable recordings can only be modified in one situation:
//  * As an outcome of a math computation
// Thus mutation of a particular channel/recording can be
// prevented by inhibiting its controlling computation

// Live remote access does not distinguish between READY and
// OBSOLETE recordings; thus code that accesses such recordings
// needs to be robust in the presence of corrupted data,
// as it is expected to be quite common.
//
// Local rendering can inhibit the controlling computation
// of mutable math channels. 
// so as to be able to render a coherent picture. Remote
// access can similarly inhibit the controlling computation
// while a coherent copy is made. This defeats the zero copy
// characteristic, however.

// RECStore structure:
// std::map of ref's to global revision structures
// each global revision structure has pointers to either a placeholder
// or recording per non-deleted channel (perhaps a hash table by recording path?).
// The placeholder must be replaced by the recording when or before the
// recording becomes READY
// this recording has a revision index (derived from the channel),
// flags, references an underlying datastore, etc.

// Transactions:
// Unlike traditional databases, simultaneous transactions are NOT permitted.
// That said, most of the work of a transaction is expected to be in the
// math, and the math can proceed in parallel for multiple transactions.

// StartTransaction() acquires the transaction lock, defines a new
// global revision index, and starts recording created recordings. 
// EndTransaction() treats the new recordings as an atomic update to
// the recording database. Essentially we copy recdatabase._channels
// into the channel_map of a new globalrevision. We insert all explicitly-defined
// new recordings from the transaction into the global revision, we carry-forward
// unchanged non-math functions. We build (or update/copy?) a mapping of
// what is dependent on each channel (and also create graph edges from
// the previous revision for sel-dependencies) and walk the dependency graph, defining new
// revisions of all dependent math recordings without
// implicit or explicit self-dependencies (see recmath.hpp) that have
// all inputs ready. (Remember that a self-dependency alone does not
// trigger a new version). Then we can queue computation of any remaining
// math recordings that have all inputs ready.
// We do that by putting all of those computations as pending_computations in the todo list of the available_compute_resource_database and queue them up as applicable
// in the prioritized_computations of the applicable
// available_compute_resource(s).
// As we get notified (how?) of these computations finishing
// we follow the same process, identifying recordings whose inputs
// are all ready, etc. 

// StartTransaction() and EndTransaction() are nestable, and only
// the outer pair are active.
//   *** UPDATE: Due to C++ lack of finally: blocks, use a RAII
// "Transaction" object to hold the transaction lock. instead of
// StartTransaction() and EndTransaction().
// i.e. snde::recstore_scoped_transaction transaction;

#ifndef SNDE_RECSTORE_HPP
#define SNDE_RECSTORE_HPP

// CMake's CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS is set, but some global variables still require 
// explicit import and export flags to compile properly with MSVC and other compilers that
// behave similarly.  Newer versions of GCC shouldn't care about the presence of dllimport
// or dllexport, but it doesn't need it.
#ifdef _WIN32
    #ifdef SPATIALNDE2_SHAREDLIB_EXPORT
        #define SNDE_API __declspec(dllexport)
    #else
        #define SNDE_API __declspec(dllimport)
    #endif
#else
    #define SNDE_API
#endif

#include <unordered_set>
#include <typeindex>
#include <memory>
#include <complex>
#include <future>
#include <type_traits>

#include "snde/recording.h" // definition of snde_recording

#include "snde/arrayposition.hpp"
#include "snde/recstore_storage.hpp"
#include "snde/lockmanager.hpp"
#include "snde/recmath.hpp"

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
  template <typename T> class ndtyped_recording_ref;

  class math_instance_parameter; // from recmath.hpp
  class channel_notify; // from notify.hpp
  class repetitive_channel_notify; // from notify.hpp
  class promise_channel_notify;
  class _globalrev_complete_notify;
  class monitor_globalrevs;
  class recording_storage; // recstore_storage.hpp
  class recording_storage_manager; // recstore_storage.hpp
  class allocator_alignment; // allocator.hpp
  class transaction_manager; // recstore_transaction_manager.hpp
  class measurement_time; // recstore_transaction_manager.hpp
  
  // constant data structures with recording type number information
  SNDE_API extern const std::unordered_map<std::type_index,unsigned> rtn_typemap; // look up typenum based on C++ typeid(type)
  SNDE_API extern const std::unordered_map<unsigned,std::string> rtn_typenamemap;
  SNDE_API extern const std::unordered_map<unsigned,size_t> rtn_typesizemap; // Look up element size bysed on typenum
  SNDE_API extern const std::unordered_map<unsigned,std::string> rtn_ocltypemap; // Look up opencl type string based on typenum
  SNDE_API extern const std::unordered_map<unsigned,std::set<unsigned>> rtn_compatible_types;

  // https://stackoverflow.com/questions/41494844/check-if-object-is-instance-of-class-with-template
  // https://stackoverflow.com/questions/43587405/constexpr-if-alternative
  template <typename> 
  struct rtn_type_is_shared_ptr: public std::false_type { };

  template <typename T> 
  struct rtn_type_is_shared_ptr<std::shared_ptr<T>>: public std::true_type { };
  
  
  template <typename T>
  typename std::enable_if<!rtn_type_is_shared_ptr<T>::value,bool>::type 
  rtn_type_is_shared_recordingbase_ptr()
  {
    return false;
  }

  template <typename T>
  typename std::enable_if<rtn_type_is_shared_ptr<T>::value,bool>::type 
  rtn_type_is_shared_recordingbase_ptr()
  {
    return (bool)std::is_base_of<recording_base,typename T::element_type>::value;
  }


  template <typename T>
  typename std::enable_if<!rtn_type_is_shared_ptr<T>::value,bool>::type 
  rtn_type_is_shared_ndarrayrecordingref_ptr()
  {
    return false;
  }

  template <typename T>
  typename std::enable_if<rtn_type_is_shared_ptr<T>::value,bool>::type 
  rtn_type_is_shared_ndarrayrecordingref_ptr()
  {
    return (bool)std::is_base_of<ndarray_recording_ref,typename T::element_type>::value;
  }

  template <typename T>
  typename std::enable_if<!rtn_type_is_shared_ptr<T>::value, bool>::type
    rtn_type_is_shared_constructiblemetadata_ptr()
  {
    return false;
  }

  template <typename T>
  typename std::enable_if<rtn_type_is_shared_ptr<T>::value, bool>::type
    rtn_type_is_shared_constructiblemetadata_ptr()
  {
    return (bool)std::is_base_of<constructible_metadata, typename T::element_type>::value;
  }

  
  
  template <typename T>
  unsigned rtn_fromtype()
  // works like rtn_typemap but can accommodate instances and subclasses of recording_base. Also nicer error message
  {

    // T may be a snde::recording_base subclass instance
    if (rtn_type_is_shared_recordingbase_ptr<T>()) {
      return SNDE_RTN_RECORDING;
    } else {
      if (rtn_type_is_shared_ndarrayrecordingref_ptr<T>()) {
	return SNDE_RTN_RECORDING_REF;
      } else {
	if (rtn_type_is_shared_constructiblemetadata_ptr<T>()) {
	  return SNDE_RTN_CONSTRUCTIBLEMETADATA;
	}
	else {
	  std::unordered_map<std::type_index, unsigned>::const_iterator wtnt_it = rtn_typemap.find(std::type_index(typeid(T)));
	  if (wtnt_it != rtn_typemap.end()) {
	    return wtnt_it->second;
	  }
	  else {

	    throw snde_error("Type %s is not supported in this context", demangle_type_name(typeid(T).name()).c_str());
	  }
	}
      }
    }
  }



  // Detection of complex types: See https://stackoverflow.com/questions/41438493/how-to-identifying-whether-a-template-argument-is-stdcomplex
  template<typename T>
  struct is_complex: public std::false_type {};
  
  // mark snde_complexfloat32 as complex
  template<>
  struct is_complex<snde_complexfloat32>: public std::true_type {};
  
  // mark snde_complexfloat64 as complex
  template<>
  struct is_complex<snde_complexfloat64>: public std::true_type {};
  
  // mark std::complex as complex
  template<typename T>
  struct is_complex<std::complex<T>>: public std::true_type {};
  
  
  

  class recording_creator_data {
  public:
    recording_creator_data() = default;
    // rule of 3
    recording_creator_data & operator=(const recording_creator_data &) = delete; 
    recording_creator_data(const recording_creator_data &orig) = delete;
    virtual ~recording_creator_data()=default; // virtual destructor so we can be subclassed by different types of creators    
  };
  

  std::shared_ptr<recording_storage_manager> select_storage_manager_for_recording_during_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<reserved_channel> proposed_chan,std::shared_ptr<channelconfig> proposed_config);

  std::shared_ptr<recording_storage_manager> select_storage_manager_for_recording(std::shared_ptr<recdatabase> recdb,std::string chanpath,std::shared_ptr<recording_set_state> rss);

  template <typename T>
  void *ptr_to_new_shared_impl(std::shared_ptr<recording_base> baseptr)
  {
    std::shared_ptr<T> *new_shared_ptr = new std::shared_ptr<T>(std::dynamic_pointer_cast<T>(baseptr));

    return const_cast<void *>(static_cast<const void *>(new_shared_ptr)); // like SWIG_as_voidptr
  }

  class recording_class_info {
  public:
    std::string classname;
    std::type_index index;
    std::function<void *(std::shared_ptr<recording_base> baseptr)> ptr_to_new_shared; // used for SWIG downcasting
    
    recording_class_info(std::string classname,std::type_index index,std::function<void *(std::shared_ptr<recording_base> baseptr)> ptr_to_new_shared) :
      classname(classname),
      index(index),
      ptr_to_new_shared(ptr_to_new_shared)
    {

    }
  };

  struct recording_params {
    std::shared_ptr<recdatabase> recdb;
    std::shared_ptr<recording_storage_manager> storage_manager;
    std::shared_ptr<rss_reference> prerequisite_state;
    std::string chanpath;
    std::shared_ptr<rss_reference> originating_state;
    uint64_t new_revision; // set to SNDE_REVISION_INVALID unless you have a definitive new revision to supply. 
  };
  
  class recording_base: public std::enable_shared_from_this<recording_base>  {
    // may be subclassed by creator
    // mutable in certain circumstances following the conventions of snde_recording

    // lock required to safely read/write mutable portions unless you are the owner and
    // you are single threaded and the information you are writing is for a subsequent state (info->state/info_state);
    // last lock in the locking order except for Python GIL
  public:
    std::mutex admin;
    
    struct snde_recording_base *info; // owned by this class and allocated with malloc; often actually a sublcass such as snde_multi_ndarray_recording
    std::atomic<uint64_t> info_revision; //atomic mirror of info->revision 
    std::atomic_int info_state; // atomic mirror of info->state ***!!! This is referenced by the ndarray_recording_ref->info_state

    // class inheritance info (managed by constructors of this and derived classes)
    std::vector<recording_class_info> rec_classes; // ordered inheritance: First entry is recording_base, then subclasses in order. Must be filled out by constructors then immutable after that.
    
    std::shared_ptr<constructible_metadata> pending_metadata; 
    std::shared_ptr<immutable_metadata> metadata; // pointer may not be changed once info_state ALLMETADATAREADY flag set. The pointer in info is the .get() value of this pointer. This is really the only metadata field that matters, and it is build from the combination of metadata, pending_metadata, and pending_dynamic_metadata in _merge_static_and_dynamic_metadata_admin_locked()

    bool needs_dynamic_metadata; // assign with recording_needs_dynamic_metadata() method
    std::shared_ptr<constructible_metadata> pending_dynamic_metadata; 

    std::shared_ptr<reserved_channel> chan; //  immutable reference to the channel provided when the recording was created. Needed for assigning storage manager
    std::shared_ptr<channelconfig> chanconfig; // current proposed channel configuration as of when this recording was created (immutable)
    std::shared_ptr<recording_storage_manager> storage_manager; // pointer initialized to a default by recording constructor, then used by the allocate_storage() method. Any assignment must be prior to that. may not be used afterward; see recording_storage in recstore_storage.hpp for details on pointed structure.

    // These next three items relate to the __originating__ globalrevision or recording set state
    // rss, but depending on the state _originating_rss may not have been assigned yet and may
    // need to extract from recdb_weak and _originating_globalrev_index.
    // DON'T ACCESS THESE DIRECTLY! Use the .get_originating_rss() and ._get_originating_rss_recdb_and_rec_admin_prelocked() methods.
    std::weak_ptr<recdatabase> recdb_weak;  // Right now I think this is here solely so that we can get access to the available_compute_resources_database to queue more calculations after a recording is marked as ready. Also used by assign_storage_manager(). Immutable once created so safe to read.  
    //std::weak_ptr<transaction> defining_transact; // This pointer should be valid for a recording defined as part of a transaction; nullptr for an ondemand math recording, for example. Weak ptr should be convertible to strong as long as the originating_rss is still current.

    uint64_t originating_state_unique_id; // must be assigned by creator (i.e. create_recording<>() or create_recording_math<>()) immediately after creation. Immutable from then on. 
    
    //std::weak_ptr<recording_set_state> _originating_rss; // locked by admin mutex; if expired than originating_rss has been freed. if nullptr then this was defined as part of a transaction that was may still be going on when the recording was defined. Use get_originating_rss() which handles locking and getting the originating_rss from the defining_transact

    std::shared_ptr<rss_reference> prerequisite_state; // This pointer is cleared when the recording is complete and that allows the prerequisite state to go out of existence.

    std::shared_ptr<rss_reference> originating_state; // This pointer is cleared when the recording is complete and that allows the originating state to go out of existence.
    
    std::shared_ptr<recording_creator_data> creator_data; // responsibility of owner of channel who wrote the recording. The shared pointer itself is locked by the admin mutex. 
    
    // Some kind of notification that recording is done to support
    // e.g. finishing a synchronous process such as a render.

    // This constructor is to be called via create_recording<>() or create_recording_math<>
    // or create_ndarray_ref() or create_ndarray_ref_math(),
    // or create_typed_ndarray_ref<>() or create_typed_ndarray_ref_math<>().
    
    //  * Should be called by the owner of the given channel, as verified by owner_id
    //  * Should be called within a transaction (i.e. the recdb transaction_lock is held by the existance of the transaction) or part of a math calculation
    // The various helpers above must call recdb->register_new_rec() or register_new_math_rec() on the constructed recording
    recording_base(struct recording_params params,size_t info_structsize=0);

    // rule of 3
    recording_base & operator=(const recording_base &) = delete; 
    recording_base(const recording_base &orig) = delete;
    virtual ~recording_base(); // virtual destructor so we can be subclassed

    virtual void recording_needs_dynamic_metadata(); // Call this before mark_metadata_done() to list this recording as needing dynamic metadata

    
    std::shared_ptr<multi_ndarray_recording> cast_to_multi_ndarray();

    // graphics_subcomponents() returns a mapping from unique strings determined by the recording subclass to the channel (which may be relative) storing the subcomponent. This function should only be called once the recording is complete. The returned map should be considered immutable.
    // if the provided rss parameter is not nullptr, then this routine includes in the mapping the parameters to lockmanager::lock_recording_arrays() to acquire any necessary locks for the subcomponent orientations. After acquiring these locks, you can then make a call to graphics_subcomponents_lockedorientations() to obtain a mapping with the actual orientations. 
    virtual const std::shared_ptr<std::map<std::string,std::pair<std::string,std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>>>> graphics_subcomponents_orientation_lockinfo(std::shared_ptr<recording_set_state> rss);

    virtual const std::shared_ptr<std::map<std::string,std::pair<std::string,snde_orientation3>>> graphics_subcomponents_lockedorientations(std::shared_ptr<recording_set_state> rss);

    virtual std::shared_ptr<std::set<std::string>> graphics_componentpart_channels(std::shared_ptr<recording_set_state> rss,std::vector<std::string> processing_tags);
    
    //virtual std::shared_ptr<recording_set_state> _get_originating_rss_rec_admin_prelocked(); // version of get_originating_rss() to use if you have (optionally the recording database and) the recording's admin locks already locked.
    //virtual std::shared_ptr<recording_set_state> _get_originating_rss_recdb_admin_prelocked(); // version of get_originating_rss() to use if you have the recording database admin lock already locked.


    //virtual std::shared_ptr<recording_set_state> get_originating_rss(); // Get the originating recording set state (often a globalrev). You should only call this if you are sure that originating rss must still exist (otherwise may generate a snde_error), such as before the creator has declared the recording "ready". This will lock the rec admin locks, so any locks currently held must precede that in the locking order
    //virtual bool _transactionrec_transaction_still_in_progress_admin_prelocked(); // with the recording admin locked,  return if this is a transaction recording where the transaction is still in progress and therefore we can't get the recording_set_state

    // Mutable recording only ***!!! Not properly implemented yet ***!!!
    /*
    virtual rwlock_token_set lock_storage_for_write();
    virtual rwlock_token_set lock_storage_for_read();
    */

    virtual void _check_move_to_metadataonly__rss_and_recording_locked(std::shared_ptr<recording_set_state> rss,channel_state *chanstate);
    
    virtual void _merge_static_and_dynamic_metadata_admin_locked();

    virtual void _move_to_completed__rss_and_recording_locked(std::shared_ptr<recording_set_state> rss,channel_state *chanstate);


    virtual void mark_metadata_done();  // call WITHOUT admin lock (or other locks?) held. 
    virtual void mark_dynamic_metadata_done();  // call WITHOUT admin lock (or other locks?) held. 
    virtual void mark_data_ready();  // call WITHOUT admin lock (or other locks?) held.
    virtual void mark_data_and_metadata_ready();  // call WITHOUT admin lock (or other locks?) held.
    virtual void _mark_storage_as_finalized_internal();

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
    std::vector<arraylayout> layouts; // changes to layouts must be propagated to info.arrays[idx].ndim, info.arrays[idx]base_index, info.arrays[idx].dimlen, and info.arrays[idx].strides NOTE THAT THIS MUST BE PREALLOCATED TO THE NEEDED SIZE BEFORE ANY ndarray_recording_ref()'s ARE CREATED! ... ALSO THE ELEMENTS OF THIS VECTOR ARE DIRECTLY REFERENCED BY ndarray_recording_ref->layout

    std::unordered_map<std::string,size_t> name_mapping; // mapping from array name to array index. Names should not begin with a digit.
    // if name_mapping is non-empty then name_reverse_mapping
    // must be maintained to be identical but reverse
    std::unordered_map<size_t,std::string> name_reverse_mapping;
    
    //std::shared_ptr<rwlock> mutable_lock; // for simply mutable recordings; otherwise nullptr

    std::vector<std::shared_ptr<recording_storage>> storage; // pointers immutable once initialized  by allocate_storage() or reference_immutable_recording().  immutable afterward; see recording_storage in recstore_storage.hpp for details on pointed structure. ... ALSO THE ELEMENTS OF THIS VECTOR ARE DIRECTLY REFERENCED BY ndarray_recording_ref->storage


    // This constructor used to create a multi_ndarray_recording with multiple ndarrays. But don't use it directly. Use create_recording and friends instead!
    //  Immediately after construction, need to call .define_array() on each array up to num_ndarrays!!! (create...ref() and create...ref_math() do this automatically)
    // WARNING: Don't call directly as this constructor doesn't add to the transaction (need to call recdb->register_new_recording()) and many math functions 
    multi_ndarray_recording(struct recording_params params,size_t info_structsize,size_t num_ndarrays);


    // rule of 3
    multi_ndarray_recording & operator=(const multi_ndarray_recording &) = delete; 
    multi_ndarray_recording(const multi_ndarray_recording &orig) = delete;
    virtual ~multi_ndarray_recording();


    // NOTE: This can only be called on a recording that was constructed with num_ndarrays
    // set to zero, and must be called before any arrays are defined, etc. etc.,
    // i.e. first thing after construction 
    virtual void set_num_ndarrays(size_t num_ndarrays); 

    virtual void mark_data_ready();  // call WITHOUT admin lock (or other locks?) held. Passes on ready_notifications to storage
    virtual void _mark_storage_as_finalized_internal();

    
    inline snde_multi_ndarray_recording *mndinfo() {return (snde_multi_ndarray_recording *)info;}
    inline snde_ndarray_info *ndinfo(size_t index) {return &((snde_multi_ndarray_recording *)info)->arrays[index];}
    inline snde_ndarray_info *ndinfo(std::string name) {return &((snde_multi_ndarray_recording *)info)->arrays[name_mapping.at(name)];}
    
    void define_array(size_t index,unsigned typenum);   // should be called exactly once for each index < mndinfo()->num_arrays
    void define_array(size_t index,unsigned typenum,std::string name);   // should be called exactly once for each index < mndinfo()->num_arrays

    std::shared_ptr<std::vector<std::string>> list_arrays();

    // Throw an snde_error exception if the recording specifies a scale
    // or offset. This should be used whenever interpreting array data
    // in a context that does not include multiplying by the scale and
    // then adding the offset (ande_array-ampl_scale and
    // ande_array-ampl_offset metadata entries, respectively)
    // The identifier parameter should identify the e.g. math function
    // or similar so that the source of the problem is readily
    // identifiable
    void assert_no_scale_or_offset(std::string identifier);

    // return (scale,offset) tuple from metadata
    std::tuple<snde_float64,snde_float64> get_ampl_scale_offset();
    
    std::shared_ptr<ndarray_recording_ref> reference_ndarray(size_t index=0);
    std::shared_ptr<ndarray_recording_ref> reference_ndarray(const std::string &array_name);

    template <typename T>
    std::shared_ptr<ndtyped_recording_ref<T>> reference_typed_ndarray(size_t index=0);

    template <typename T>
    std::shared_ptr<ndtyped_recording_ref<T>> reference_typed_ndarray(const std::string &array_name);

    void assign_storage(std::shared_ptr<recording_storage> stor,size_t array_index,const std::vector<snde_index> &dimlen, bool fortran_order=false);
    void assign_storage(std::shared_ptr<recording_storage> stor,std::string array_name,const std::vector<snde_index> &dimlen, bool fortran_order=false);
    
    void assign_storage_strides(std::shared_ptr<recording_storage> stor,size_t array_index,const std::vector<snde_index> &dimlen, const std::vector<snde_index> &strides);

    void assign_storage_portion(std::shared_ptr<recording_storage> stor,size_t array_index,const std::vector<snde_index> &new_dimlen, bool fortran_order,snde_index new_base_index);
    
    // must assign info.elementsize and info.typenum before calling allocate_storage()
    // fortran_order only affects physical layout, not logical layout (interpretation of indices)
    // allocate_storage() does assign_storage_manager() then uses that to allocate_recording(), then performs assign_storage().
    // returns the storage in case you want it. 
    
    virtual std::shared_ptr<recording_storage> allocate_storage(size_t array_index,const std::vector<snde_index> &dimlen, bool fortran_order=false);
    virtual std::shared_ptr<recording_storage> allocate_storage_in_named_array(size_t array_index,std::string storage_array_name,const std::vector<snde_index> &dimlen, bool fortran_order=false);

    virtual std::shared_ptr<recording_storage> allocate_storage(std::string array_name,const std::vector<snde_index> &dimlen, bool fortran_order=false);


    
    // alternative to allocating storage: Referencing an existing recording
    virtual void reference_immutable_recording(size_t array_index,std::shared_ptr<ndarray_recording_ref> rec,std::vector<snde_index> dimlen,std::vector<snde_index> strides);

    
    inline void *void_shifted_arrayptr(size_t array_index)
    // returns base of allocation (basearray + base_index) as a void pointer
    {
      struct snde_ndarray_info *info=ndinfo(array_index);
      if (info->shiftedarray) {
	return info->shiftedarray;
      }
      return (((char *)(*info->basearray)) + info->base_index*info->elementsize);
      //return * storage[array_index]->addr();
    }
    
    inline void *void_shifted_arrayptr(std::string array_name)
    {
      return void_shifted_arrayptr(name_mapping.at(array_name));
    }

    
    inline void *element_dataptr(size_t array_index,const std::vector<snde_index> &idx)  // returns a pointer to an element, which is of size ndinfo()->elementsize
    {
      snde_ndarray_info *array_ndinfo = ndinfo(array_index);
      char *cur_charptr;

      if (array_ndinfo->shiftedarray) {
	cur_charptr=(char *)array_ndinfo->shiftedarray;
      } else {
	cur_charptr = ((char *)(*array_ndinfo->basearray)) + array_ndinfo->base_index*array_ndinfo->elementsize;
      }
      
      for (size_t dimnum=0;dimnum < array_ndinfo->ndim;dimnum++) {
	snde_index thisidx = idx.at(dimnum);
	assert(thisidx < array_ndinfo->dimlen[dimnum]);
	cur_charptr += array_ndinfo->strides[dimnum]*array_ndinfo->elementsize*thisidx;
      }
      
      return (void *)cur_charptr;
    }

    inline size_t element_offset(size_t array_index,const std::vector<snde_index> &idx)
    // element_offset in terms of elements, not including base_index
    {      
      snde_ndarray_info *array_ndinfo = ndinfo(array_index);
      size_t cur_offset = 0; //array_ndinfo->base_index;
      
      for (size_t dimnum=0;dimnum < array_ndinfo->ndim;dimnum++) {
	snde_index thisidx = idx.at(dimnum);
	assert(thisidx < array_ndinfo->dimlen[dimnum]);
	cur_offset += array_ndinfo->strides[dimnum]*thisidx;
      }
      
      return cur_offset;
      
    }


    
    double element_double(size_t array_index,const std::vector<snde_index> &idx); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    double element_double(size_t array_index,snde_index idx, bool fortran_order=false); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    
    void assign_double(size_t array_index,const std::vector<snde_index> &idx,double val); // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    void assign_double(size_t array_index,snde_index idx,double val,bool fortran_order=false); // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 

    int64_t element_int(size_t array_index,const std::vector<snde_index> &idx); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    int64_t element_int(size_t array_index,snde_index idx,bool fortran_order=false); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    void assign_int(size_t array_index,const std::vector<snde_index> &idx,int64_t val); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    void assign_int(size_t array_index,snde_index idx,int64_t val,bool fortran_order=false); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 

    uint64_t element_unsigned(size_t array_index,const std::vector<snde_index> &idx); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    uint64_t element_unsigned(size_t array_index,snde_index idx,bool fortran_order=false); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    void assign_unsigned(size_t array_index,const std::vector<snde_index> &idx,uint64_t val); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    void assign_unsigned(size_t array_index,snde_index idx,uint64_t val,bool fortran_order=false); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 


    
    snde_complexfloat64 element_complexfloat64(size_t array_index,const std::vector<snde_index> &idx); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    snde_complexfloat64 element_complexfloat64(size_t array_index,snde_index idx,bool fortran_order=false); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    void assign_complexfloat64(size_t array_index,const std::vector<snde_index> &idx,snde_complexfloat64 val); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    void assign_complexfloat64(size_t array_index,snde_index idx,snde_complexfloat64 val,bool fortran_order=false); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!

  };


  
  
  class ndarray_recording_ref: public std::enable_shared_from_this<ndarray_recording_ref> {
    // reference to a single ndarray within an multi_ndarray_recording
    // once the multi_ndarray_recording is published and sufficiently complete, its fields are immutable, so these are too
  public:
    std::shared_ptr<multi_ndarray_recording> rec; // the referenced recording
    size_t rec_index; // index of referenced ndarray within recording.
    unsigned &typenum; // reference into rec->ndarray(rec_index)->typenum
    std::atomic_int &info_state; // reference into rec->info_state
    arraylayout &layout; // reference  into rec->layouts.at(rec_index)
    std::shared_ptr<recording_storage> &storage; // reference into rec->storage.at(rec_index)


    ndarray_recording_ref(std::shared_ptr<multi_ndarray_recording> rec,size_t rec_index,unsigned typenum);

    // rule of 3 
    ndarray_recording_ref & operator=(const ndarray_recording_ref &) = delete;
    ndarray_recording_ref(const ndarray_recording_ref &orig) = delete; // could easily be implemented if we wanted
    virtual ~ndarray_recording_ref();

    virtual void allocate_storage(std::vector<snde_index> dimlen, bool fortran_order=false);
    virtual std::shared_ptr<recording_storage> allocate_storage_in_named_array(std::string storage_array_name,const std::vector<snde_index> &dimlen, bool fortran_order=false);
    template <typename T>
    std::shared_ptr<ndtyped_recording_ref<T>> reference_typed_ndarray();
    
    inline snde_multi_ndarray_recording *mndinfo() {return (snde_multi_ndarray_recording *)rec->info;}
    inline snde_ndarray_info *ndinfo() {return &((snde_multi_ndarray_recording *)rec->info)->arrays[rec_index];}

    
    
    // Throw an snde_error exception if the recording specifies a scale
    // or offset. This should be used whenever interpreting array data
    // in a context that does not include multiplying by the scale and
    // then adding the offset (ande_array-ampl_scale and
    // ande_array-ampl_offset metadata entries, respectively)
    // The identifier parameter should identify the e.g. math function
    // or similar so that the source of the problem is readily
    // identifiable
    inline void assert_no_scale_or_offset(std::string identifier)
    {
      rec->assert_no_scale_or_offset(identifier);
    }
    
    // return (scale,offset) tuple from metadata
    inline std::tuple<snde_float64,snde_float64> get_ampl_scale_offset()
    {
      return rec->get_ampl_scale_offset();
    }
    

    inline void *void_shifted_arrayptr()
    {
      if (ndinfo()->shiftedarray) {
	return ndinfo()->shiftedarray;
      } 
      return (void *)(((char *)(*ndinfo()->basearray)) + ndinfo()->base_index*ndinfo()->elementsize);
    }
    
    inline void *element_dataptr(const std::vector<snde_index> &idx)  // returns a pointer to an element, which is of size ndinfo()->elementsize
    {
      snde_ndarray_info *array_ndinfo = ndinfo();
      char *cur_charptr;
      
      if (array_ndinfo->shiftedarray) {
	cur_charptr=(char *)array_ndinfo->shiftedarray;
      } else {
	cur_charptr = ((char *)(*array_ndinfo->basearray)) + array_ndinfo->base_index*array_ndinfo->elementsize;
      }
      
      for (size_t dimnum=0;dimnum < array_ndinfo->ndim;dimnum++) {
	snde_index thisidx = idx.at(dimnum);
	assert(thisidx < array_ndinfo->dimlen[dimnum]);
	cur_charptr += array_ndinfo->strides[dimnum]*array_ndinfo->elementsize*thisidx;
      }
      
      return (void *)cur_charptr;
    }

    inline size_t element_offset(const std::vector<snde_index> &idx)
    // element_offset in terms of elements, not including base_index
    {      
      snde_ndarray_info *array_ndinfo = ndinfo();
      size_t cur_offset = 0; //array_ndinfo->base_index;
      
      for (size_t dimnum=0;dimnum < array_ndinfo->ndim;dimnum++) {
	snde_index thisidx = idx.at(dimnum);
	assert(thisidx < array_ndinfo->dimlen[dimnum]);
	cur_offset += array_ndinfo->strides[dimnum]*thisidx;
      }
      
      return cur_offset;
      
    }


    inline size_t element_offset(snde_index idx,bool fortran_order=false)
    // element_offset in terms of elements, not including base_index,
    // for unwrapped index idx, interpreted by fortran or C convention
    // (see also ndtyped_recording_ref::element)
    {      
      std::vector<snde_index> vidx;
      vidx.reserve(layout.dimlen.size());

      if (fortran_order) {
	for (size_t dimnum=0;dimnum < layout.dimlen.size();dimnum++) {
	  vidx.push_back(idx % layout.dimlen[dimnum]);
	  idx /= layout.dimlen[dimnum];
	}
      } else {
	for (size_t dimnum=0;dimnum < layout.dimlen.size();dimnum++) {
	  vidx.insert(vidx.begin(),idx % layout.dimlen[layout.dimlen.size()-dimnum-1]);
	  idx /= layout.dimlen[layout.dimlen.size()-dimnum-1];
	}

      }
      return element_offset(vidx);
    }

    
    virtual std::shared_ptr<ndarray_recording_ref> assign_recording_type(unsigned typenum); // returns a new fully-typed reference. 
    
    virtual double element_double(const std::vector<snde_index> &idx); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    virtual double element_double(snde_index idx, bool fortran_order=false); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    virtual void assign_double(const std::vector<snde_index> &idx,double val); // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    virtual void assign_double(snde_index idx,double val,bool fortran_order=false); // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 

    virtual int64_t element_int(const std::vector<snde_index> &idx); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    virtual int64_t element_int(snde_index idx,bool fortran_order=false); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    virtual void assign_int(const std::vector<snde_index> &idx,int64_t val); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    virtual void assign_int(snde_index idx,int64_t val,bool fortran_order=false); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 

    virtual uint64_t element_unsigned(const std::vector<snde_index> &idx); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    virtual uint64_t element_unsigned(snde_index idx,bool fortran_order=false); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    virtual void assign_unsigned(const std::vector<snde_index> &idx,uint64_t val); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    virtual void assign_unsigned(snde_index idx,uint64_t val,bool fortran_order=false); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    

    virtual snde_complexfloat64 element_complexfloat64(const std::vector<snde_index> &idx);
    virtual snde_complexfloat64 element_complexfloat64(snde_index idx,bool fortran_order=false);

    virtual void assign_complexfloat64(const std::vector<snde_index> &idx,snde_complexfloat64 val);
    virtual void assign_complexfloat64(snde_index idx,snde_complexfloat64 val,bool fortran_order=false);

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

  struct transaction_notifies {
    bool all_ready;
    std::unordered_set<channel_state *> ready_channels; // references into the new_rss->recstatus.channel_map
    std::vector<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>> ready_to_execute;
    std::set<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>>,mncn_lessthan> may_need_completion_notification;
    std::vector<std::shared_ptr<channel_notify>> transaction_pending_channel_notifies; // from the pending_channel_notifies of the transaction
    

  };
  
  class transaction_math {
  public:
    std::weak_ptr<recdatabase> recdb;
    std::weak_ptr<transaction> trans;
    std::map<std::string,std::pair<std::shared_ptr<reserved_channel>,std::shared_ptr<pending_math_definition_result_channel>>> pending_dict; // Indexed by definition channel name

    // Note: extended in python swig bindings to act like a dictionary 
  };
  
  class transaction: public std::enable_shared_from_this<transaction> {
  public:
    // mutable until end of transaction when it is destroyed and converted to a globalrev structure
    std::mutex admin; // after the recording_set_state in the locking order and after the class channel's admin lock, but before recordings. Must hold this lock when reading or writing structures within. Does not cover the channels/recordings themselves.

    
    uint64_t rss_unique_index; // unique_index field that will be assigned to the recording_set_state. Immutable once published
    std::multimap<std::string,std::pair<std::shared_ptr<reserved_channel>,std::shared_ptr<channelconfig>>> updated_channels;
    //Keep track of whether a new recording is required for the channel (e.g. if it has a new owner) (use false for math recordings)
    std::map<std::string,bool> new_recording_required; // index is channel name for updated channels
    std::unordered_map<std::string,std::pair<std::shared_ptr<reserved_channel>,std::shared_ptr<recording_base>>> new_recordings; // Note: for now reserved_channel may be nullptr for a math channel (but that wouldn't be recorded here, would it?)
    std::unordered_map<std::shared_ptr<instantiated_math_function>, std::unordered_map<std::string, std::shared_ptr<math_instance_parameter>>> math_messages;

    // end of transaction propagates this structure into an update of recdatabase._channels
    // and a new globalrevision

    std::vector<std::shared_ptr<channel_notify>> pending_channel_notifies; // channel_notifies that need to be applied to the globalrev once it is created

    // when the transaction is complete, resulting_globalrevision() is assigned
    // so that if you have a pointer to the transaction you can get the globalrevision (which is also a recording_set_state)
    std::shared_ptr<globalrevision> _resulting_globalrevision; // locked by transaction admin mutex; use globalrev() or globalrev_nowait() accessor.

    std::shared_ptr<rss_reference> our_state_reference; // This pointer is created with the transaction. The reference is filled in by _realize_transaction() and MUST BE CLEARED BY THE TRANSACTION_MANAGER AFTER the transaction is realized, and that allows the state to go out of existence.
    
    std::shared_ptr<rss_reference> prerequisite_state; // This pointer is created with the transaction. THE REFERENCE MUST BE FILLED IN BY THE TRANSACTION_MANAGER. The pointer MUST BE CLEARED BY THE TRANSACTION_MANAGER AFTER the transaction is realized and that allows the prerequisite state to go out of existence once other pointers are cleared.

    std::mutex transaction_background_end_lock; // locks the function and pointer immediately below. Last in the locking order
    
    std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<void> params)> transaction_background_end_fcn;
    std::shared_ptr<void> transaction_background_end_params;

    std::shared_ptr<transaction_math> math;


    
    transaction(std::shared_ptr<recdatabase> recdb);
    // rule of 3
    transaction& operator=(const transaction &) = delete; 
    transaction(const transaction &orig) = delete;
    virtual ~transaction();
    
    void register_new_rec(std::shared_ptr<reserved_channel> chan,std::shared_ptr<recording_base> new_rec);

    std::shared_ptr<globalrevision> globalrev(); // Wait for the globalrev resulting from this transaction to be complete. Throws an exception if any of the non-math recordings initiated in this transaction are not complete.
    std::shared_ptr<globalrevision> globalrev_wait(); // Wait for the globalrev resulting from this transaction to be complete. Unlike globalrev() this function will not throw an exception for incomplete recordings; instead it will wait for them to be complete.
    
    std::shared_ptr<globalrevision> globalrev_available(); // Wait for the globalrev resulting from this transaction to exist, but not necessarily be complete. 
    
    std::shared_ptr<globalrevision> globalrev_nowait(); // Return the globalrevision resulting from this transaction if it exists yet. Returns nullptr otherwise. Even if the globalrev exists, it may not be complete.

    //virtual void end_transaction(std::shared_ptr<recdatabase> recdb);
    std::tuple<std::shared_ptr<globalrevision>,transaction_notifies> _realize_transaction(std::shared_ptr<recdatabase> recdb,uint64_t globalrevision_index);

    void _notify_transaction_globalrev(std::shared_ptr<recdatabase> recdb_strong,std::shared_ptr<globalrevision> globalrev,struct transaction_notifies trans_notify);


    bool transaction_globalrev_is_complete();


    // Wait on the return value of get_transaction_globalrev_complete_waiter() by
    // calling its wait_interruptable method.
    // You can call its interrupt() method from another thread to
    // cancel the wait. 
    std::shared_ptr<promise_channel_notify> get_transaction_globalrev_complete_waiter(); 

    virtual void update_timestamp(std::shared_ptr<transaction_manager> transmgr,std::shared_ptr<measurement_time> new_timestamp);
  };


  // helper function used by transaction::realize_transaction() and recstore_display_transforms::update()
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
					     bool ondemand_only);

  
  
  
  class active_transaction  : public std::enable_shared_from_this<active_transaction>  {
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

    
    
    
    active_transaction(std::shared_ptr<recdatabase> recdb,std::shared_ptr<measurement_time> timestamp=nullptr);


    // rule of 3
    active_transaction& operator=(const active_transaction &) = delete; 
    active_transaction(const active_transaction &orig) = delete;
    ~active_transaction(); // destructor releases transaction_lock from holder

        
    std::shared_ptr<transaction> end_transaction();
    void update_timestamp(std::shared_ptr<measurement_time> new_timestamp);
    std::shared_ptr<transaction> run_in_background_and_end_transaction(std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<void> params)> fcn,std::shared_ptr<void> params);
    
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
    //void *owner_id; // pointer private to the owner (such as dataguzzler_python PyObject of the owner's representation of this channel) that
    // the owner can use to verify its continued ownership.

    bool hidden; // explicitly hidden channel
    bool math; // math channel


    std::shared_ptr<recording_storage_manager> storage_manager; // storage manager for newly defined recordings or nullptr to implicitly refer to an upper level or the default... Note that while the pointer is immutable, the pointed storage_manager is NOT immutable. 
    
    // The following only apply to math channels
    std::shared_ptr<instantiated_math_function> math_fcn; // If math is set, then this channel is one of the outputs of the given math_fcn  math_fcn is also immutable once published
    bool mathintermediate; // intermediate result of math calculation -- makes it implicitly hidden
    bool ondemand; // if the output is not actually stored in the database but just delivered on-demand
    bool data_requestonly; // if the output is to be stored in the database with the metadata always calculated but the underlying data only triggered to be computed if requested or needed by another recording.
    bool data_mutable; // if the output is mutable

    channelconfig(std::string channelpath, std::string owner_name,bool hidden,std::shared_ptr<recording_storage_manager> storage_manager=nullptr);
    // rule of 3
    channelconfig& operator=(const channelconfig &) = default; 
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
    std::atomic<uint64_t> latest_revision; // 0 means "invalid"; generally 0 (or previous latest) during channel creation/undeletion; incremented to 1 (with an empty recording if necessary) after transaction end 
    // std::atomic<bool> deleted; // is the channel currently defined? (replaced by owner being null)

    std::mutex admin; // last in the locking order except before python GIL. Used to ensure there can only be one _config update at a time. 


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

    std::mutex admin; // last in the locking order except before python GIL. Used to ensure there can only be one _config update at a time.

    // Most channels are owned by their creator and the reserved_channel
    // object is owned by their creator and the creator is responsible
    // for making changes in a reasonable order and not having multiple
    // proposed configurations simultaneously. In contrast, all
    // math channels are owned centrally and therefore multiple
    // simultaneous changes are more plausible. The recording database
    // has a map of reserved_channels (reserved_math_channels)
    // but because these are shared we don't want to propose changes
    // to them because only one proposal can be handled at a time.
    // Therefore, for math channels we clone the reserved_channel
    // in reserve_math_channel() and return a clone with the
    // math_actual_reschan pointing at the original and with
    // the appropriate proposed_config set.
    
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
    
  class channel_state {
  public:
    // for atomic updates to notify_ ... atomic shared pointers, you must lock the recording_set_state's admin lock
    std::shared_ptr<channelconfig> config; // immutable
    std::shared_ptr<reserved_channel> chan; // immutable
    // std::shared_ptr<channel> _channel; // immutable pointer, but pointed data is not immutable, (but you shouldn't generally need to access this)
    std::shared_ptr<recording_base> _rec; // atomic shared ptr to recording structure created to store the ouput; may be nullptr if not (yet) created. Always nullptr for ondemand recordings... recording contents may be mutable but have their own admin lock
    std::atomic<bool> updated; // this field is only valid once the creating transaction is ended and rec() returns a valid pointer and once rec()->state is READY or METADATAREADY. It is true if this particular recording has a new revision particular to the enclosing recording_set_state. For math recordings this is assigned in the create_recording_math<T> template (below). For non-math recordings it is assigned in realize_transaction(). 
    std::shared_ptr<uint64_t> _revision; // Atomic shared pointer... This is assigned when the channel_state is created from _rec->info->revision for manually created recordings. (For ondemand math recordings this is not meaningful?) For math recordings with the math_function's new_revision_optional (config->math_fcn->fcn->new_revision_optional) flag clear, this is defined during realize_transaction() before the channel_state is published. If the new_revision_optional flag is set, this left nullptr; once the math function determines whether a new recording will be instantiated the revision will be assigned when the recording is define, with ordering ensured by the implicit self-dependency implied by the new_revision_optional flag (recmath_compute_resource.cpp)
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> _notify_about_this_channel_metadataonly; // atomic shared ptr to immutable set of channel_notifies that need to be updated or perhaps triggered when this channel becomes metadataonly; set to nullptr at end of channel becoming metadataonly. 
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> _notify_about_this_channel_ready; // atomic shared ptr to immutable set of channel_notifies that need to be updated or perhaps triggered when this channel becomes ready; set to nullptr at end of channel becoming ready. 

    channel_state(std::shared_ptr<reserved_channel> owner,std::shared_ptr<channelconfig> config,std::shared_ptr<recording_base> rec,bool updated);

    channel_state(const channel_state &orig); // copy constructor used for initializing channel_map from prototype defined in realize_transaction()

    // Copy assignment operator deleted
    channel_state& operator=(const channel_state &) = delete;

    // default destructor
    ~channel_state() = default; 

    std::shared_ptr<recording_base> rec() const;
    std::shared_ptr<uint64_t> revision() const;
    std::shared_ptr<recording_base> recording_is_complete(bool mdonly); // uses only atomic members so safe to call in all circumstances. Set to mdonly if you only care that the metadata is complete. Normally call recording_is_complete(false). Returns recording pointer if recording is complete to the requested condition, otherwise nullptr. 
    void issue_nonmath_notifications(std::shared_ptr<recording_set_state> rss); // Must be called without anything locked. Issue notifications requested in _notify* and remove those notification requests
    void issue_math_notifications(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_set_state> rss); // Must be called without anything locked. Check for any math updates from the new status of this recording
    
    void end_atomic_rec_update(std::shared_ptr<recording_base> new_recording); // also updates revision
    void end_atomic_revision_update(uint64_t new_revision);


    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> begin_atomic_notify_about_this_channel_metadataonly_update();
    void end_atomic_notify_about_this_channel_metadataonly_update(std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> newval);
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_metadataonly();

    
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> begin_atomic_notify_about_this_channel_ready_update();
    void end_atomic_notify_about_this_channel_ready_update(std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> newval);
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_ready();
  };


  class recording_status {
  public:
    std::shared_ptr<std::map<std::string,channel_state>> channel_map; // key is full channel path... The map itself (not the embedded states) is immutable once the recording_set_state is published. The map itself may also outlive the enclosing recording_set_state.
    
    /// all of these are indexed by their their full path. Every entry in channel_map should be in exactly one of these. Locked by rss admin mutex per above
    // The index is the shared_ptr in globalrev_channel.config
    // primary use for these is determining when our globalrev/rss is
    // complete: Once call recordings are in metadataonly or completed,
    // then it should be complete
    // NOTE: ***!!! This doesn't work as cleanly as we would like because ideally
    // the channel would move around atomically once instantiated, etc. but that's not
    // really possible because it would require excessive locking (in math functions the
    // output can affect multiple RSS's and the timing of when notifications get added is
    // tricky, so there is catch-up logic in various places). 
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

  class recording_set_state : public std::enable_shared_from_this<recording_set_state> {
  public:
    std::mutex admin; // locks changes to recstatus including channel_map contents (map itself is immutable once published), mathstatus,  and the _recordings reference maps/sets and notifiers. Precedes transaction->admin (i.e. recdb->current_transaction->admin) , recstatus.channel_map.rec.admin and Python GIL in locking order
    uint64_t originating_globalrev_index; // this rss may not __be__ a globalrev but it (almost certainly?) is built on one.
    uint64_t unique_index; // This is a number unique within the process for this RSS. Used to disambiguate shared memory names, for example. Immutable post-construction 
    std::weak_ptr<recdatabase> recdb_weak;
    std::atomic<bool> ready; // indicates that all recordings except ondemand and data_requestonly recordings are READY (mutable recordings may be OBSOLETE)
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
    virtual ~recording_set_state();

    bool check_complete();
    
    void wait_complete(); // wait for all the recordings and math in this recording_set_state or globalrev to reach nominal completion (metadataonly or ready, as configured)
    
    std::string print_math_status(bool verbose=false);
    std::string print_recording_status(bool verbose=false);

    // get_xxxx throws an exception if the recording is not present
    // check_for_xxxx returns nullptr if the recording is not present
    std::shared_ptr<recording_base> get_recording(const std::string &fullpath);
    std::shared_ptr<recording_base> check_for_recording(const std::string &fullpath);
    
    std::shared_ptr<ndarray_recording_ref> get_ndarray_ref(const std::string &fullpath,size_t array_index=0);
    std::shared_ptr<ndarray_recording_ref> get_ndarray_ref(const std::string &fullpath,std::string array_name);

    template <typename T>
    std::shared_ptr<ndarray_recording_ref> get_typed_recording_ref(const std::string &fullpath,size_t array_index=0)
    {
      std::shared_ptr<ndarray_recording_ref> rawref = get_ndarray_ref(fullpath,array_index);
      std::shared_ptr<ndtyped_recording_ref<T>> retval = std::dynamic_pointer_cast<ndtyped_recording_ref<T>>(rawref);
      if (!retval) {
	throw snde_error("Recording channel %s array index %llu has type %s which appears not to match %s",fullpath.c_str(),(unsigned long long)array_index,rtn_typenamemap.at(rawref->typenum).c_str(),rtn_typenamemap.at(rtn_typemap.at(typeid(T))).c_str());
      }
      return retval;
    }
    
    template <typename T>
    std::shared_ptr<ndarray_recording_ref> get_typed_recording_ref(const std::string &fullpath,std::string array_name)
    {
      std::shared_ptr<ndarray_recording_ref> rawref = get_ndarray_ref(fullpath,array_name);
      std::shared_ptr<ndtyped_recording_ref<T>> retval = std::dynamic_pointer_cast<ndtyped_recording_ref<T>>(rawref);
      if (!retval) {
	throw snde_error("Recording channel %s array name %s has type %s which appears not to match %s",fullpath.c_str(),array_name.c_str(),rtn_typenamemap.at(rawref->typenum).c_str(),rtn_typenamemap.at(rtn_typemap.at(typeid(T))).c_str());
      }
      return retval;
      
    }

    
    std::shared_ptr<rss_reference> prerequisite_state();
    void prerequisite_state_clear();
    void prerequisite_state_assign(std::shared_ptr<rss_reference> state);
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



      // internal use only for recording_set_state::_update_recstatus__rss_admin_transaction_admin_locked()
    bool _urratal_check_mdonly(std::string channelpath);
    
    void _update_recstatus__rss_admin_transaction_admin_locked(); // This is called during realize_transaction() with the recdb admin lock, the rss admin lock, and the transaction admin lock held, to identify anything misplaced in the above unordered_maps. After the call and marking of the transaction's _resulting_globalrevision, placement responsibility shifts to mark_metadata_done() and mark_data_ready() methods of the recording. 

    void register_new_math_rec(std::shared_ptr<recording_base> new_rec); // registers newly created math recording in the given rss (and extracts mutable flag for the given channel into the recording structure)).
    
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
    std::shared_ptr<recording_set_state> _rss; // atomic shared pointer; use rss() accessor
    rss_reference();
    rss_reference(std::shared_ptr<recording_set_state> rss);
    ~rss_reference();
    std::shared_ptr<recording_set_state> rss();
    void rss_assign(std::shared_ptr<recording_set_state> rss);
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
    //std::mutex admin; // locks changes to recstatus, mathstatus, recstatus.channel_map, the _recordings reference maps/sets, and the mutable_recordings_need_holder. Precedes recstatus.channel_map.rec.admin and Python GIL in locking order
    //std::atomic<bool> ready; // indicates that all recordings except ondemand and data_requestonly recordings are READY (mutable recordings may be OBSOLETE)
    //recording_status recstatus;
    //math_status mathstatus; // note math_status.math_functions is immutable

    
    uint64_t globalrev; // immutable
    //std::shared_ptr<transaction> defining_transact; // This keeps the transaction data structure (pointed to by weak pointers in the recordings created in the transaction) in memory at least as long as the globalrevision is current.

    // With the globalrevision object there is a single object of class globalrev_mutable_lock. This is created with the globalrevision
    // and pointed to here (mutable_recordings_still_needed) until the globalrev is ready. Once ready, notify.cpp:_globalrev_complete_notify::perform_notify()
    // places this pointer in the pending_map of each monitor_globalrevs, and once the globalrev is both ready and there is a subsequent globalrev, the
    // mutable_recordings_need_holder is emptied. When the last shared_ptr expires, the globalrev_mutable_lock destructor
    // is called, clearing the mutable_recordings_still_needed flag and triggering the globalrev_monitoring_thread to wake up and queue the
    // relevant computations from the
    // blocked_list (recdatabase::globalrev_mutablenotneeded_code())
    // This pointer is also held by the display logic in openscenegraph_compositor to
    // keep mutable recordings in a consistent state for viewing.
    std::shared_ptr<globalrev_mutable_lock> mutable_recordings_need_holder;
    std::atomic<bool> mutable_recordings_still_needed; 

    globalrevision(uint64_t globalrev, std::shared_ptr<transaction> defining_transact, std::shared_ptr<recdatabase> recdb,const instantiated_math_database &math_functions,const std::map<std::string,channel_state> & channel_map_param,std::shared_ptr<rss_reference> prerequisite_state,uint64_t rss_unique_index);   
  };
  

  class recdatabase : public std::enable_shared_from_this<recdatabase> {
  public:
    std::mutex admin; // Locks access to _channels and _deleted_channels and _available_math_functions, _globalrevs, repetitive_notifies,  and monitoring. In locking order, precedes transaction_manager locks, channel admin locks, available_compute_resource_database, globalrevision admin locks, recording admin locks, and Python GIL. 
    std::map<std::string,std::shared_ptr<channel>> _channels; // Generally don't use the channel map directly. Grab the latestglobalrev and use the channel map from that. 
    // std::map<std::string,std::shared_ptr<channel>> _deleted_channels; // Channels are put here after they are deleted. They can be moved back into the main list if re-created. 
    instantiated_math_database _instantiated_functions; 
    
    std::map<uint64_t,std::shared_ptr<globalrevision>> _globalrevs; // Index is global revision counter. Includes at least one globalrev that is fully ready plus any that are still udndergoing computation. The last element in this is the most recently defined globalrev.
    std::shared_ptr<globalrevision> _latest_defined_globalrev; // atomic shared pointer -- access with latest_defined_globalrev() method;
    std::shared_ptr<globalrevision> _latest_ready_globalrev; // atomic shared pointer to latest ready globalrev -- access with latest_globalrev() method;
    std::vector<std::shared_ptr<repetitive_channel_notify>> repetitive_notifies; 

    std::shared_ptr<allocator_alignment> alignment_requirements; // Pointer is immutable; pointed structure has its own locking
    std::shared_ptr<available_compute_resource_database> compute_resources; // has its own admin lock.
    

    std::shared_ptr<memallocator> lowlevel_alloc; // pointer is immutable once created during startup; contents not necessarily immutable; see memallocator.hpp
    std::shared_ptr<recording_storage_manager> default_storage_manager; // pointer is immutable once created during startup; contents not necessarily immutable; see recstore_storage.hpp
    std::shared_ptr<recording_storage_manager> nonlocking_storage_manager; // storage manager that never imposes its own locking requirements. GPU use may still impose locking requirements, though

    std::shared_ptr<lockmanager> lockmgr; // pointer immutable after initialization; contents have their own admin lock, which is used strictly internally by them

    std::atomic<bool> started;

    std::shared_ptr<transaction_manager> transmgr; // pointer is immutable once created during startup. 
    
    std::set<std::weak_ptr<monitor_globalrevs>,std::owner_less<std::weak_ptr<monitor_globalrevs>>> monitoring;
    uint64_t monitoring_notify_globalrev; // latest globalrev for which monitoring has already been notified


    
    std::thread globalrev_mutablenotneeded_thread;
    std::mutex globalrev_mutablenotneeded_lock; // locks the condition vector, list, and bool immediately below.  Last in the locking order
    std::condition_variable globalrev_mutablenotneeded_condition;
    std::list<std::shared_ptr<globalrevision>> globalrev_mutablenotneeded_pending;
    bool globalrev_mutablenotneeded_mustexit;

    std::set<std::shared_ptr<std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<globalrevision>)>>> ready_globalrev_quicknotifies_called_recdb_locked; // locked by admin lock.  
    
    std::shared_ptr<math_function_registry_map> _available_math_functions; // atomic shared pointer... use available_math_functions() accessor. 
    
    std::unordered_map<std::string,std::shared_ptr<reserved_channel>> reserved_math_channels; // locked by admin lock
    
    recdatabase(std::shared_ptr<lockmanager> lockmgr=nullptr);
    recdatabase & operator=(const recdatabase &) = delete; 
    recdatabase(const recdatabase &orig) = delete;
    ~recdatabase();
        
    void add_alignment_requirement(size_t nbytes); // note all alignment requirements (e.g. from GPU's) must be added before initializing storage managers

    void startup(); // gets the math engine running, etc. 


    // a transaction update can be multi-threaded but you shouldn't call end_transaction()  (or the end_transaction method on the
    // active_transaction or delete the active_transaction) until all other threads are finished with transaction actions
    std::shared_ptr<active_transaction> start_transaction(std::shared_ptr<measurement_time> timestamp=nullptr);
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

    void send_math_message(std::shared_ptr<active_transaction> trans,std::shared_ptr<instantiated_math_function> func, std::string name, std::shared_ptr<math_instance_parameter> msg);
    
    std::vector<std::shared_ptr<reserved_channel>> add_math_function_storage_manager(std::shared_ptr<active_transaction> trans,std::shared_ptr<instantiated_math_function> new_function,bool hidden,std::shared_ptr<recording_storage_manager> storage_manager);

    std::shared_ptr<globalrevision> latest_defined_globalrev(); // safe to call with or without recdb admin lock held

    std::shared_ptr<globalrevision> latest_globalrev(); // safe to call with or without recdb admin lock held. Returns latest globalrev which is ready and for which all prior globalrevs are ready

    std::shared_ptr<globalrevision> get_globalrev(uint64_t revnum);
    std::shared_ptr<reserved_channel> _reserve_channel_create_new_recdb_locked(std::shared_ptr<channelconfig> new_config);
    std::shared_ptr<reserved_channel> _reserve_channel_add_to_trans(std::shared_ptr<transaction> trans,std::shared_ptr<channelconfig> new_config,std::shared_ptr<reserved_channel> reservation);
    // Allocate channel with a specific name; returns nullptr if the name is inuse
    std::shared_ptr<reserved_channel> reserve_channel(std::shared_ptr<active_transaction> trans,std::shared_ptr<channelconfig> new_config); // must be called within a transaction
    std::shared_ptr<reserved_channel> reserve_channel(std::shared_ptr<transaction> trans,std::shared_ptr<channelconfig> new_config); // must be called within a transaction
    void release_channel(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan); // must be called within a transaction

    // Define a new channel; throws an error if the channel is already in use.
    // Must be called within a transaction
    std::shared_ptr<reserved_channel> define_channel(std::shared_ptr<active_transaction> trans,std::string channelpath, std::string owner_name, bool hidden=false, std::shared_ptr<recording_storage_manager> storage_manager=nullptr);
    
    //std::shared_ptr<channel> lookup_channel_live(std::string channelpath);


    // NOTE: python wrappers for wait_recordings and wait_recording_names need to drop dgpython thread context during wait and poll to check for connection drop
    //void wait_recordings(std::vector<std::shared_ptr<recording>> &);
    void wait_recording_names(std::shared_ptr<recording_set_state> rss,const std::vector<std::string> &metadataonly, const std::vector<std::string> fullyready);

    std::shared_ptr<monitor_globalrevs> start_monitoring_globalrevs(std::shared_ptr<globalrevision> first = nullptr,bool inhibit_mutable = false);

    // These functions can be used to manage quicknotifies that are called when a new globalrev
    // becomes ready. They are called with the recdb locked (be aware of locking order!!!)
    // and MUST return very rapidly -- shouldn't do anything of substance: Generally
    // just queue some sort of event for some thread or event loop to take care of later
    // IN MOST USAGE TO GET GLOBALREV UPDATES YOU SHOULD USE start_monitoring_globalrevs()!!!
    void register_ready_globalrev_quicknotifies_called_recdb_locked(std::shared_ptr<std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<globalrevision>)>> quicknotify);
    void unregister_ready_globalrev_quicknotifies_called_recdb_locked(std::shared_ptr<std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<globalrevision>)>> quicknotify);

    void globalrev_mutablenotneeded_code(); 
    
    
    
    std::shared_ptr<math_function_registry_map> available_math_functions(); // Note that this includes ONLY the custom-added math functions, not the built ins. Use math_function_registry()  to get the c++ built in ones

    std::shared_ptr<math_function> lookup_available_math_function(std::string name); // look up an a available math function considers both custom-added and c++ builtin math functions
    std::shared_ptr<std::vector<std::string>> list_available_math_functions(); // considers both custom-added and c++ builtin available math functions
    
    std::shared_ptr<math_function_registry_map> _begin_atomic_available_math_functions_update(); // should be called with admin lock held
    void _end_atomic_available_math_functions_update(std::shared_ptr<math_function_registry_map>); // should be called with admin lock held

  };


  
  template <typename T>
  class ndtyped_recording_ref : public ndarray_recording_ref {
  public:
    typedef T dtype;

    ndtyped_recording_ref(std::shared_ptr<multi_ndarray_recording> rec,unsigned typenum,size_t rec_index) :
      ndarray_recording_ref(rec,rec_index,typenum)
    {
      unsigned templated_typenum=rtn_typemap.at(typeid(T));
      if (templated_typenum != typenum) {
	bool found_compatible=false;
	auto rct_it = rtn_compatible_types.find(templated_typenum);
	if (rct_it != rtn_compatible_types.end()) {
	  auto rct_compat_it = rct_it->second.find(typenum);
	  if (rct_compat_it != rct_it->second.end()) {
	    found_compatible = true; 
	  }
	}

	if (!found_compatible) {
	  throw snde_error("Attempting to create reference of typenum %u that is incompatible with data specified as type %u",(unsigned)templated_typenum,typenum);
	}
	
      }

			      
    }
    

    
    inline T *shifted_arrayptr() { return (T*)void_shifted_arrayptr(); }

    inline T& element(const std::vector<snde_index> &idx) { return *(shifted_arrayptr()+element_offset(idx)); }

    inline T& element(snde_index idx,bool fortran_order=false) {
      // reference element by unwrapped index idx, interpreted by
      // fortran or C convention
      // (see also ndarray_recording_ref::element_offset())
      std::vector<snde_index> vidx;
      vidx.reserve(layout.dimlen.size());

      if (fortran_order) {
	for (size_t dimnum=0;dimnum < layout.dimlen.size();dimnum++) {
	  vidx.push_back(idx % layout.dimlen[dimnum]);
	  idx /= layout.dimlen[dimnum];
	}
      } else {
	for (size_t dimnum=0;dimnum < layout.dimlen.size();dimnum++) {
	  vidx.insert(vidx.begin(),idx % layout.dimlen[layout.dimlen.size()-dimnum-1]);
	  idx /= layout.dimlen[layout.dimlen.size()-dimnum-1];
	}

      }
      return element(vidx);
    }
    
    // rule of 3
    ndtyped_recording_ref & operator=(const ndtyped_recording_ref &) = delete; 
    ndtyped_recording_ref(const ndtyped_recording_ref &orig) = delete;
    ~ndtyped_recording_ref() { }


    
    // see https://stackoverflow.com/questions/12073689/c11-template-function-specialization-for-integer-types/12073915
    // https://stackoverflow.com/questions/57964743/cannot-be-overloaded-error-while-trying-to-enable-sfinae-with-enable-if
    // but the above method requires that the methods be templates to use SFINAE,
    // but template methods cannot override a parent's virtual methods per
    // https://stackoverflow.com/questions/2778352/template-child-class-overriding-a-parent-classs-virtual-function
    // So we need non-template wrappers that call the SFINAE templates


    // Here are the SFINAE templates
    // element_double for arithmetic types
    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,double>::type _element_double(const std::vector<snde_index> &idx) // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    {
      size_t offset = element_offset(idx);
      return (double)shifted_arrayptr()[offset];
    }
    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,double>::type _element_double(snde_index idx,bool fortran_order) // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    {
      size_t offset = element_offset(idx,fortran_order);
      return (double)shifted_arrayptr()[offset];
    }

    // element_double for nonarithmetic types
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value,double>::type _element_double(const std::vector<snde_index> &idx) // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    {
      throw snde_error("Cannot extract floating point value from non-arithmetic type");
    }
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value,double>::type _element_double(snde_index idx,bool fortran_order) // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    {
      throw snde_error("Cannot extract floating point value from non-arithmetic type");
    }
    
    // assign_double for arithmetic types
    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,void>::type _assign_double(const std::vector<snde_index> &idx,double val) // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      size_t offset = element_offset(idx);
      shifted_arrayptr()[offset]=(T)val;      
    }
    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,void>::type _assign_double(snde_index idx,double val,bool fortran_order) // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      size_t offset = element_offset(idx,fortran_order);
      shifted_arrayptr()[offset]=(T)val;      
    }

    // assign_double for nonarithmetic types
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value,void>::type _assign_double(const std::vector<snde_index> &idx,double val) // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      throw snde_error("Cannot assign floating point value from non-arithmetic type");
    }
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value,void>::type _assign_double(snde_index idx,double val,bool fortran_order) // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      throw snde_error("Cannot assign floating point value from non-arithmetic type");
    }

    
    // element_int for arithmetic types
    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,int64_t>::type _element_int(const std::vector<snde_index> &idx) // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    {
      size_t offset = element_offset(idx);
      return (int64_t)shifted_arrayptr()[offset];
    }
    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,int64_t>::type _element_int(snde_index idx,bool fortran_order) // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    {
      size_t offset = element_offset(idx,fortran_order);
      return (int64_t)shifted_arrayptr()[offset];
    }

    // element_int for non-arithmetic types
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value,int64_t>::type _element_int(const std::vector<snde_index> &idx) // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    {
      throw snde_error("Cannot extract integer value from non-arithmetic type");
    }
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value,int64_t>::type _element_int(snde_index idx,bool fortran_order) // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    {
      throw snde_error("Cannot extract integer value from non-arithmetic type");
    }
    
    // assign_int for arithmetic types
    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,void>::type  _assign_int(const std::vector<snde_index> &idx,int64_t val) // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      size_t offset = element_offset(idx);
      shifted_arrayptr()[offset]=(T)val;
    }
    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,void>::type  _assign_int(snde_index idx,int64_t val,bool fortran_order) // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      size_t offset = element_offset(idx,fortran_order);
      shifted_arrayptr()[offset]=(T)val;
    }

    // assign_int for non-arithmetic types
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value,void>::type  _assign_int(const std::vector<snde_index> &idx,int64_t val) // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      throw snde_error("Cannot assign integer value to non-arithmetic type");
    }
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value,void>::type  _assign_int(snde_index idx,int64_t val,bool fortran_order) // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      throw snde_error("Cannot assign integer value to non-arithmetic type");
    }
    
    // element_unsigned for arithmetic types
    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,uint64_t>::type _element_unsigned(const std::vector<snde_index> &idx) // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    {
      size_t offset = element_offset(idx);
      return (uint64_t)shifted_arrayptr()[offset];
    }
    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,uint64_t>::type _element_unsigned(snde_index idx,bool fortran_order) // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    {
      size_t offset = element_offset(idx,fortran_order);
      return (uint64_t)shifted_arrayptr()[offset];
    }
    
    // element_unsigned for non-arithmetic types
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value,uint64_t>::type _element_unsigned(const std::vector<snde_index> &idx) // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    {
      throw snde_error("Cannot extract integer value from non-arithmetic type");
    }
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value,uint64_t>::type _element_unsigned(snde_index idx,bool fortran_order) // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    {
      throw snde_error("Cannot extract integer value from non-arithmetic type");
    }

    
    // assign_unsigned for arithmetic types
    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,void>::type _assign_unsigned(const std::vector<snde_index> &idx,uint64_t val) // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      size_t offset = element_offset(idx);
      shifted_arrayptr()[offset]=(T)val;
    }


    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,void>::type _assign_unsigned(snde_index idx,uint64_t val,bool fortran_order) // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      size_t offset = element_offset(idx,fortran_order);
      shifted_arrayptr()[offset]=(T)val;
    }

    

    // assign_unsigned for non-arithmetic types
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value,void>::type _assign_unsigned(const std::vector<snde_index> &idx,uint64_t val) // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      throw snde_error("Cannot assign integer value to non-arithmetic type");
    }

    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value,void>::type _assign_unsigned(snde_index idx,uint64_t val,bool fortran_order) // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      throw snde_error("Cannot assign integer value to non-arithmetic type");
    }




    // element_complexfloat64 for complex types
    template <typename U = T>
    typename std::enable_if<is_complex<U>::value,snde_complexfloat64>::type _element_complexfloat64(const std::vector<snde_index> &idx) //  if array is mutable by others, it should generally be locked for read when calling this function!
    {
      size_t offset = element_offset(idx);
      snde_complexfloat64 retval;
      retval.real = shifted_arrayptr()[offset].real;
      retval.imag = shifted_arrayptr()[offset].imag;

      return retval;
    }
    template <typename U = T>
    typename std::enable_if<is_complex<U>::value,snde_complexfloat64>::type _element_complexfloat64(snde_index idx,bool fortran_order) //  if array is mutable by others, it should generally be locked for read when calling this function!
    {
      size_t offset = element_offset(idx,fortran_order);
      snde_complexfloat64 retval;
      retval.real = shifted_arrayptr()[offset].real;
      retval.imag = shifted_arrayptr()[offset].imag;

      return retval;
    }

    // element_complexfloat64 for arithmetic types: returns value with real part, imaginary part of 0
    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,snde_complexfloat64>::type _element_complexfloat64(const std::vector<snde_index> &idx) // if array is mutable by others, it should generally be locked for read when calling this function!
    {
      size_t offset = element_offset(idx);
      snde_complexfloat64 retval;
      retval.real = shifted_arrayptr()[offset];
      retval.imag = 0.0;

      return retval; 
    }
    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,snde_complexfloat64>::type _element_complexfloat64(snde_index idx,bool fortran_order) // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    {
      size_t offset = element_offset(idx,fortran_order);
      
      snde_complexfloat64 retval;
      retval.real = shifted_arrayptr()[offset];
      retval.imag = 0.0;

      return retval; 
    }
    
    // element_complexfloat64 for non-arithmetic types
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value && !is_complex<U>::value,snde_complexfloat64>::type _element_complexfloat64(const std::vector<snde_index> &idx) // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    {
      throw snde_error("Cannot extract complex value from non-complex and non-arithmetic type");
    }
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value && !is_complex<U>::value,snde_complexfloat64>::type _element_complexfloat64(snde_index idx,bool fortran_order) // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    {
      throw snde_error("Cannot extract complex value from non-complex and non-arithmetic type");
    }



    // assign_complexfloat64 for complex types
    template <typename U = T>
    typename std::enable_if<is_complex<U>::value,void>::type _assign_complexfloat64(const std::vector<snde_index> &idx,snde_complexfloat64 val) // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      size_t offset = element_offset(idx);

      T to_assign;
      to_assign.real = val.real;
      to_assign.imag = val.imag;
      
      shifted_arrayptr()[offset]=to_assign;
    }


    template <typename U = T>
    typename std::enable_if<is_complex<U>::value,void>::type _assign_complexfloat64(snde_index idx,snde_complexfloat64 val,bool fortran_order) // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      size_t offset = element_offset(idx,fortran_order);
      
      T to_assign;
      to_assign.real = val.real;
      to_assign.imag = val.imag;
      
      shifted_arrayptr()[offset]=to_assign;
    }



    // assign_complexfloat64 for non-complex types
    template <typename U = T>
    typename std::enable_if<!is_complex<U>::value,void>::type _assign_complexfloat64(const std::vector<snde_index> &idx,snde_complexfloat64 val) // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      throw snde_error("Cannot assign complex value to non-complex type");
    }

    template <typename U = T>
    typename std::enable_if<!is_complex<U>::value,void>::type _assign_complexfloat64(snde_index idx,snde_complexfloat64 val,bool fortran_order) // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      throw snde_error("Cannot assign complex value to non-complex type");
    }

    

    // ... and here are the non-template wrappers
    double element_double(const std::vector<snde_index> &idx)
    {
      return _element_double(idx);
    }

    double element_double(snde_index idx,bool fortran_order=false)
    {
      return _element_double(idx,fortran_order);
    }

    void assign_double(const std::vector<snde_index> &idx,double val) // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      _assign_double(idx,val);
    }

    void assign_double(snde_index idx,double val,bool fortran_order=false) // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      _assign_double(idx,val,fortran_order);
    }

    int64_t element_int(const std::vector<snde_index> &idx)
    {
      return _element_int(idx);
    }

    int64_t element_int(snde_index idx,bool fortran_order=false)
    {
      return _element_int(idx,fortran_order);
    }

    void assign_int(const std::vector<snde_index> &idx,int64_t val) // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      _assign_int(idx,val);
    }

    void assign_int(snde_index idx,int64_t val,bool fortran_order=false) // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      _assign_int(idx,val,fortran_order);
    }

    uint64_t element_unsigned(const std::vector<snde_index> &idx)
    {
      return _element_unsigned(idx);
    }

    uint64_t element_unsigned(snde_index idx,bool fortran_order=false)
    {
      return _element_unsigned(idx,fortran_order);
    }

    void assign_unsigned(const std::vector<snde_index> &idx,uint64_t val) // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      _assign_unsigned(idx,val);
    }

    void assign_unsigned(snde_index idx,uint64_t val,bool fortran_order=false) // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      _assign_unsigned(idx,val,fortran_order);
    }

    
    snde_complexfloat64 element_complexfloat64(const std::vector<snde_index> &idx)
    {
      return _element_complexfloat64(idx);
    }
    
    snde_complexfloat64 element_complexfloat64(snde_index idx,bool fortran_order=false)
    {
      return _element_complexfloat64(idx,fortran_order);
      
    }

    void assign_complexfloat64(const std::vector<snde_index> &idx,snde_complexfloat64 val)
    {
      _assign_complexfloat64(idx,val);
    }
    
    void assign_complexfloat64(snde_index idx,snde_complexfloat64 val,bool fortran_order=false)
    {
      _assign_complexfloat64(idx,val,fortran_order);      
    }


  };



  size_t recording_default_info_structsize(size_t param,size_t min);
  
  template <typename T,typename ... Args>
  std::shared_ptr<T> create_recording(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,Args && ... args)
  {
    //if (!recdb->current_transaction) {
    //  throw snde_error("create_recording() outside of a transaction!");
    //}



    std::shared_ptr<channelconfig> chanconfig = chan->proposed_config();
    std::shared_ptr<recording_storage_manager> storage_manager = select_storage_manager_for_recording_during_transaction(trans->recdb,chan,chanconfig);
    
    //***!!! uint64_t new_revision = ++chan->latest_revision; // atomic variable so it is safe to pre-increment

    
    // It is safe to use recdb->current_transaction here even with out locking because
    // the caller is responsible for making sure we are in a transaction
    // i.e. appropriate synchronization must happen before and after. 
    
    std::shared_ptr<T> new_rec = std::make_shared<T>(recording_params{trans->recdb,storage_manager,trans->trans->prerequisite_state,chanconfig->channelpath,trans->trans->our_state_reference,SNDE_REVISION_INVALID},0,args...);
    new_rec->chan=chan;
    new_rec->chanconfig=chanconfig;
    new_rec->originating_state_unique_id = trans->trans->rss_unique_index;
    
    trans->trans->register_new_rec(chan,new_rec);
    return new_rec;
  }


  // create_anonymous_recording() creates a recording that is not attached to a channel and cannot be used for automated math calculations
  // Note that because it is not connected to a channel, it will always use the default storage manager. That said, it is OK
  // to override the storage manager by reassigning it. 
  template <typename T,typename ... Args>
  std::shared_ptr<T> create_anonymous_recording(std::shared_ptr<recdatabase> recdb,std::string purpose,Args && ... args) // purpose is used for naming shared memory objects
  {
    std::shared_ptr<recording_storage_manager> storage_manager = recdb->default_storage_manager;

    
    
    std::shared_ptr<T> new_rec = std::make_shared<T>(recording_params{recdb,storage_manager,nullptr,std::string("//anonymous/")+purpose,nullptr,SNDE_REVISION_INVALID},0,args...);
    new_rec->originating_state_unique_id = rss_get_unique();
    
    return new_rec;
  }
  
  
  template <typename T,typename ... Args>
  std::shared_ptr<T> create_recording_math(std::string chanpath,std::shared_ptr<recording_set_state> calc_rss,Args && ... args) // math use only... 
  {
    std::shared_ptr<recdatabase> recdb = calc_rss->recdb_weak.lock();
    if (!recdb) return nullptr;
    
    std::shared_ptr<recording_storage_manager> storage_manager = select_storage_manager_for_recording(recdb,chanpath,calc_rss);
    //std::shared_ptr<transaction> defining_transact = (std::dynamic_pointer_cast<globalrevision>(calc_rss)) ? (std::dynamic_pointer_cast<globalrevision>(calc_rss)->defining_transact):nullptr;
    uint64_t new_revision=0;

    channel_state &state = calc_rss->recstatus.channel_map->at(chanpath);
    state.updated = true; // mark as updated but ONLY in the RSS of the calculation itself (not in subsequent referencing rss's)

    
    if (std::dynamic_pointer_cast<globalrevision>(calc_rss)) {
      // if calc_rss is really a globalrevision (i.e. not an ondemand calculation)
      // then we need to define a new revision of this recording
            
      assert(!state.config->ondemand);

      // if the new_revision_optional flag is set, then we have to define the new revision now;
      if (/*state.config->math_fcn->fcn->new_revision_optional && */ !state.revision()) {
	// We no longer require the new_revision_optional flag here,
	// but we shouldn't be creating new math recordings until
	// the transaction has ended, at which point
	// build_rss_from_functions_and_channels() should have
	// instantiated any mandatory updated recordings.
	// If we got to this code path, this is not a mandatory update,
	// which means that our channel, or something we are
	// dependent on, is new_revision_optional.

	// !!!***There might be the possibility of revision index
	// ordering bugs if we
	// are dependent on a new_revision_optional math channel
	// as well as other channels and get a mix of mandatory
	// and optional updates. 
	new_revision = ++state.chan->chan->latest_revision; // latest_revision is atomic; correct ordering guaranteed by the implicit self-dependency that comes with new_revision_optional flag -- this should even work transitively for recordings dependent on new-revision-optional recordings
	state.end_atomic_revision_update(new_revision);
      } else{ 
	// new_revision_optional is clear: grab revision from channel_state
	new_revision = *state.revision();
      }
      
    }

    std::shared_ptr<T> new_rec = std::make_shared<T>(recording_params{recdb,storage_manager,calc_rss->prerequisite_state(),chanpath,calc_rss->our_state_reference,new_revision},0,args...);
    
    new_rec->originating_state_unique_id = calc_rss->unique_index;
    new_rec->chan=state.chan;
    new_rec->chanconfig=state.config;
    
    calc_rss->register_new_math_rec(new_rec);
    return new_rec;
    
  }


  
  // for non math-functions operating in a transaction
  template <typename T>
  std::shared_ptr<ndtyped_recording_ref<T>> create_typed_ndarray_ref(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,unsigned typenum=rtn_typemap.at(typeid(T)))
  {
    //std::shared_ptr<multi_ndarray_recording> new_rec = std::make_shared<multi_ndarray_recording>(recdb,chan,owner_id,1);
    
    std::shared_ptr<multi_ndarray_recording> new_rec = create_recording<multi_ndarray_recording>(trans,chan,1);
    new_rec->define_array(0,typenum);
    
    return std::make_shared<ndtyped_recording_ref<T>>(new_rec,typenum,0);
  }

    template <typename T>
    std::shared_ptr<ndtyped_recording_ref<T>> create_typed_named_ndarray_ref(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,std::string arrayname,unsigned typenum=rtn_typemap.at(typeid(T)))
  {
    //std::shared_ptr<multi_ndarray_recording> new_rec = std::make_shared<multi_ndarray_recording>(recdb,chan,owner_id,1);
    
    std::shared_ptr<multi_ndarray_recording> new_rec = create_recording<multi_ndarray_recording>(trans,chan,1);
    new_rec->define_array(0,typenum,arrayname);
    
    return std::make_shared<ndtyped_recording_ref<T>>(new_rec,typenum,0);
  }

  
  template <typename S,typename T,typename ... Args>
  std::shared_ptr<ndtyped_recording_ref<T>> create_typed_subclass_ndarray_ref(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,Args && ... args)
  {

    unsigned typenum=rtn_typemap.at(typeid(T));
    //std::shared_ptr<multi_ndarray_recording> new_rec = std::make_shared<multi_ndarray_recording>(recdb,chan,owner_id,1);
    
    std::shared_ptr<multi_ndarray_recording> new_rec = create_recording<S>(trans,chan,args...);
    new_rec->define_array(0,typenum);
    
    return std::make_shared<ndtyped_recording_ref<T>>(new_rec,typenum,0);
  }



  template <typename T>
  std::shared_ptr<ndtyped_recording_ref<T>> create_anonymous_typed_ndarray_ref(std::shared_ptr<recdatabase> recdb,std::string purpose,unsigned typenum=rtn_typemap.at(typeid(T))) // purpose is used for naming the shared memory objects used for the underlying storage
  {
    //std::shared_ptr<multi_ndarray_recording> new_rec = std::make_shared<multi_ndarray_recording>(recdb,chan,owner_id,1);
    
    std::shared_ptr<multi_ndarray_recording> new_rec = create_anonymous_recording<multi_ndarray_recording>(recdb,purpose,1);
    new_rec->define_array(0,typenum);
    
    return std::make_shared<ndtyped_recording_ref<T>>(new_rec,typenum,0);
  }

  template <typename T>
  std::shared_ptr<ndtyped_recording_ref<T>> create_anonymous_typed_named_ndarray_ref(std::shared_ptr<recdatabase> recdb,std::string purpose,std::string arrayname, unsigned typenum=rtn_typemap.at(typeid(T))) // purpose is used for naming the shared memory objects used for the underlying storage
  {
    //std::shared_ptr<multi_ndarray_recording> new_rec = std::make_shared<multi_ndarray_recording>(recdb,chan,owner_id,1);
    
    std::shared_ptr<multi_ndarray_recording> new_rec = create_anonymous_recording<multi_ndarray_recording>(recdb,purpose,1);
    new_rec->define_array(0,typenum,arrayname);
    
    return std::make_shared<ndtyped_recording_ref<T>>(new_rec,typenum,0);
  }

  
  template <typename S,typename T,typename ... Args>
  std::shared_ptr<ndtyped_recording_ref<T>> create_anonymous_typed_subclass_ndarray_ref(std::shared_ptr<recdatabase> recdb,std::string purpose,Args && ... args) // purpose is used for naming shared memory objects
  {
    unsigned typenum=rtn_typemap.at(typeid(T));
    //std::shared_ptr<multi_ndarray_recording> new_rec = std::make_shared<multi_ndarray_recording>(recdb,chan,owner_id,1);
    
    std::shared_ptr<multi_ndarray_recording> new_rec = create_anonymous_recording<S>(recdb,purpose,args...);
    new_rec->define_array(0,typenum);
    
    return std::make_shared<ndtyped_recording_ref<T>>(new_rec,typenum,0);
  }

  
  
  // for math_recordings_only (no transaction)
  template <typename T>
  std::shared_ptr<ndtyped_recording_ref<T>> create_typed_ndarray_ref_math(std::string chanpath,std::shared_ptr<recording_set_state> calc_rss,unsigned typenum=rtn_typemap.at(typeid(T)))
  {
    std::shared_ptr<recdatabase> recdb = calc_rss->recdb_weak.lock();
    if (!recdb) return nullptr;

    //std::shared_ptr<multi_ndarray_recording> new_rec = std::make_shared<multi_ndarray_recording>(recdb,chanpath,calc_rss,1); // owner id for math recordings is recdb raw pointer recdb.get()
    std::shared_ptr<multi_ndarray_recording> new_rec = create_recording_math<multi_ndarray_recording>(chanpath,calc_rss,1);
    new_rec->define_array(0,typenum);
    return std::make_shared<ndtyped_recording_ref<T>>(new_rec,typenum,0);
  }

  // for math_recordings_only (no transaction)
  template <typename T>
  std::shared_ptr<ndtyped_recording_ref<T>> create_typed_named_ndarray_ref_math(std::string chanpath,std::shared_ptr<recording_set_state> calc_rss,std::string arrayname,unsigned typenum=rtn_typemap.at(typeid(T)))
  {
    std::shared_ptr<recdatabase> recdb = calc_rss->recdb_weak.lock();
    if (!recdb) return nullptr;

    //std::shared_ptr<multi_ndarray_recording> new_rec = std::make_shared<multi_ndarray_recording>(recdb,chanpath,calc_rss,1); // owner id for math recordings is recdb raw pointer recdb.get()
    std::shared_ptr<multi_ndarray_recording> new_rec = create_recording_math<multi_ndarray_recording>(chanpath,calc_rss,1);
    new_rec->define_array(0,typenum,arrayname);
    return std::make_shared<ndtyped_recording_ref<T>>(new_rec,typenum,0);
  }

  
  template <typename S,typename T,typename ... Args>
  std::shared_ptr<ndtyped_recording_ref<T>> create_typed_subclass_ndarray_ref_math(std::string chanpath,std::shared_ptr<recording_set_state> calc_rss,Args && ... args)
  {
    unsigned typenum=rtn_typemap.at(typeid(T));
    std::shared_ptr<recdatabase> recdb = calc_rss->recdb_weak.lock();
    if (!recdb) return nullptr;

    //std::shared_ptr<multi_ndarray_recording> new_rec = std::make_shared<multi_ndarray_recording>(recdb,chanpath,calc_rss,1); // owner id for math recordings is recdb raw pointer recdb.get()
    std::shared_ptr<multi_ndarray_recording> new_rec = create_recording_math<S>(chanpath,calc_rss,args...);
    new_rec->define_array(0,typenum);
    return std::make_shared<ndtyped_recording_ref<T>>(new_rec,typenum,0);
  }

  
  // static factory methods for creating recordings with single runtime-determined types
  // for regular (non-math) use. Automatically registers the new recording
  // ok to specify typenum as SNDE_RTM_UNASSIGNED if you don't know the final type yet. Then use assign_recording_type() method to get a new fully typed reference 
  std::shared_ptr<ndarray_recording_ref> create_ndarray_ref(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,unsigned typenum);

  std::shared_ptr<ndarray_recording_ref> create_named_ndarray_ref(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,std::string arrayname,unsigned typenum);

  template <typename S,typename ... Args> 
  std::shared_ptr<ndarray_recording_ref> create_subclass_ndarray_ref(std::shared_ptr<active_transaction> trans,std::shared_ptr<reserved_channel> chan,unsigned typenum,Args && ... args)
  {
    std::shared_ptr<multi_ndarray_recording> rec;
    std::shared_ptr<ndarray_recording_ref> ref;

    // ***!!! Should look up maker method in a runtime-addable database ***!!!

    rec=create_recording<S>(trans,chan,args...);
    rec->define_array(0,typenum);
    
    ref=rec->reference_ndarray(0);
    return ref;
  }
  
  std::shared_ptr<ndarray_recording_ref> create_anonymous_ndarray_ref(std::shared_ptr<recdatabase> recdb,std::string purpose,unsigned typenum); // purpose is used for naming shared memory objects

  std::shared_ptr<ndarray_recording_ref> create_anonymous_named_ndarray_ref(std::shared_ptr<recdatabase> recdb,std::string purpose,std::string arrayname,unsigned typenum); // purpose is used for naming shared memory objects

  template <typename S,typename ... Args> 
  std::shared_ptr<ndarray_recording_ref> create_anonymous_subclass_ndarray_ref(std::shared_ptr<recdatabase> recdb,std::string purpose,unsigned typenum, Args && ... args) // purpose is used for naming shared memory objects
  {
    std::shared_ptr<multi_ndarray_recording> rec;
    std::shared_ptr<ndarray_recording_ref> ref;
    
    // ***!!! Should look up maker method in a runtime-addable database ***!!!
    
    rec=create_anonymous_recording<S>(recdb,purpose,args...);
    rec->define_array(0,typenum);
    
    ref=rec->reference_ndarray(0);
    return ref;

  }

  std::shared_ptr<ndarray_recording_ref> create_ndarray_ref_math(std::string chanpath,std::shared_ptr<recording_set_state> calc_rss,unsigned typenum); // math use only... ok to specify typenum as SNDE_RTM_UNASSIGNED if you don't know the final type yet. Then use assign_recording_type() method to get a new fully typed reference 

  std::shared_ptr<ndarray_recording_ref> create_named_ndarray_ref_math(std::string chanpath,std::shared_ptr<recording_set_state> calc_rss,std::string arrayname,unsigned typenum); // math use only... ok to specify typenum as SNDE_RTM_UNASSIGNED if you don't know the final type yet. Then use assign_recording_type() method to get a new fully typed reference 

  
  template <typename S,typename ... Args>
  std::shared_ptr<ndarray_recording_ref> create_subclass_ndarray_ref_math(std::string chanpath,std::shared_ptr<recording_set_state> calc_rss,unsigned typenum,Args && ... args) // math use only... ok to specify typenum as SNDE_RTM_UNASSIGNED if you don't know the final type yet. Then use assign_recording_type() method to get a new fully typed reference 
  {
    std::shared_ptr<multi_ndarray_recording> rec;
    std::shared_ptr<ndarray_recording_ref> ref;
    
    std::shared_ptr<recdatabase> recdb = calc_rss->recdb_weak.lock();
    if (!recdb) return nullptr;

    
    rec=create_recording_math<S>(chanpath,calc_rss,args...); 
    rec->define_array(0,typenum);
    ref=rec->reference_ndarray(0);

    return ref;
  }



  template <typename T>
  std::shared_ptr<ndtyped_recording_ref<T>> multi_ndarray_recording::reference_typed_ndarray(size_t index/*=0*/)
  {
    std::shared_ptr<ndarray_recording_ref> untyped_ref = reference_ndarray(index);
    std::shared_ptr<ndtyped_recording_ref<T>> typed_ref = std::dynamic_pointer_cast<ndtyped_recording_ref<T>>(untyped_ref);
    
    if (!typed_ref) {
      throw snde_error("reference_typed_ndarray(): Cannot cast ndarray with type %s to type %s",rtn_typenamemap.at(untyped_ref->typenum).c_str(),rtn_typenamemap.at(rtn_typemap.at(typeid(T))).c_str());
    }
    return typed_ref; 
  }
    
  template <typename T>
  std::shared_ptr<ndtyped_recording_ref<T>> multi_ndarray_recording::reference_typed_ndarray(const std::string &array_name)
  {
    std::shared_ptr<ndarray_recording_ref> untyped_ref = reference_ndarray(array_name);
    std::shared_ptr<ndtyped_recording_ref<T>> typed_ref = std::dynamic_pointer_cast<ndtyped_recording_ref<T>>(untyped_ref);
    
    if (!typed_ref) {
      throw snde_error("reference_typed_ndarray(): Cannot cast ndarray with type %s to type %s",rtn_typenamemap.at(untyped_ref->typenum).c_str(),rtn_typenamemap.at(rtn_typemap.at(typeid(T))).c_str());
    }
    return typed_ref; 
    
  }
  
  template <typename T>
  std::shared_ptr<ndtyped_recording_ref<T>> ndarray_recording_ref::reference_typed_ndarray()
  {
   
    std::shared_ptr<ndtyped_recording_ref<T>> typed_ref = std::dynamic_pointer_cast<ndtyped_recording_ref<T>>(shared_from_this());
    
    if (!typed_ref) {
      throw snde_error("reference_typed_ndarray(): Cannot cast ndarray with type %s to type %s",rtn_typenamemap.at(typenum).c_str(),rtn_typenamemap.at(rtn_typemap.at(typeid(T))).c_str());
    }
    return typed_ref; 
  }
  
};

#endif // SNDE_RECSTORE_HPP
