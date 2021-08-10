// Mostly immutable wfmstore concept:

// "Channel" is a permanent, non-renameable
// identifier/structure that keeps track of revisions
// for each named address (waveform path).
// Channels can be "Deleted" in which case
// they are omitted from current lists, but
// their revisions are kept so if the name
// is reused the revision counter will
// increment, not reset.
// A channel is "owned" by a module, which
// is generally the only source of new
// waveforms on the channel.
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

// The waveform database keeps a collection
// of named channels and the current owners.
// Entries can only be
// added or marked as deleted during a
// transaction.

// "Waveform" represents a particular revision
// of a channel. The waveform structure itself
// and metadata is always immutable once READY,
// except for the state field and the presence
// of an immutable data copy (mutable only).
// The underlying data can be
// mutable or immutable.
// Waveforms have three states: INITIALIZING, READY,
// and OBSOLETE (mutable only).
// INITIALIZING is the status while its creator is
// filling out the data structure
//
// Once the entire waveform (including all metadata
// and the underlying data) is complete, the waveform
// becomes READY. If the waveform is mutable, before
// its is allowed to be mutated, the waveform should
// be marked as OBSOLETE

// Mutable waveforms can only be modified in one situation:
//  * As an outcome of a math computation
// Thus mutation of a particular channel/waveform can be
// prevented by inhibiting its controlling computation

// Live remote access does not distinguish between READY and
// OBSOLETE waveforms; thus code that accesses such waveforms
// needs to be robust in the presence of corrupted data,
// as it is expected to be quite common.
//
// Local rendering can inhibit the controlling computation
// of mutable math channels. 
// so as to be able to render a coherent picture. Remote
// access can similarly inhibit the controlling computation
// while a coherent copy is made. This defeats the zero copy
// characteristic, however.

// WFMStore structure:
// std::map of ref's to global revision structures
// each global revision structure has pointers to either a placeholder
// or waveform per non-deleted channel (perhaps a hash table by waveform path?).
// The placeholder must be replaced by the waveform when or before the
// waveform becomes READY
// this waveform has a revision index (derived from the channel),
// flags, references an underlying datastore, etc.

// Transactions:
// Unlike traditional databases, simultaneous transactions are NOT permitted.
// That said, most of the work of a transaction is expected to be in the
// math, and the math can proceed in parallel for multiple transactions.

// StartTransaction() acquires the transaction lock, defines a new
// global revision index, and starts recording created waveforms. 
// EndTransaction() treats the new waveforms as an atomic update to
// the waveform database. Essentially we copy wfmdatabase._channels
// into the channel_map of a new globalrevision. We insert all explicitly-defined
// new waveforms from the transaction into the global revision, we carry-forward
// unchanged non-math functions. We build (or update/copy?) a mapping of
// what is dependent on each channel (and also create graph edges from
// the previous revision for sel-dependencies) and walk the dependency graph, defining new
// revisions of all dependent math waveforms without
// implicit or explicit self-dependencies (see wfmmath.hpp) that have
// all inputs ready. (Remember that a self-dependency alone does not
// trigger a new version). Then we can queue computation of any remaining
// math waveforms that have all inputs ready.
// We do that by putting all of those computations as pending_computations in the todo list of the available_compute_resource_database and queue them up as applicable
// in the prioritized_computations of the applicable
// available_compute_resource(s).
// As we get notified (how?) of these computations finishing
// we follow the same process, identifying waveforms whose inputs
// are all ready, etc. 

// StartTransaction() and EndTransaction() are nestable, and only
// the outer pair are active.
//   *** UPDATE: Due to C++ lack of finally: blocks, use a RAII
// "Transaction" object to hold the transaction lock. instead of
// StartTransaction() and EndTransaction().
// i.e. snde::wfmstore_scoped_transaction transaction;

#ifndef SNDE_WFMSTORE_HPP
#define SNDE_WFMSTORE_HPP

#include <unordered_set>
#include <typeindex>
#include <memory>
#include <complex>
#include <future>
#include <type_traits>

#include "waveform.h" // definition of snde_waveform

#include "arrayposition.hpp"
#include "wfmstore_storage.hpp"
#include "lockmanager.hpp"
#include "wfmmath.hpp"

namespace snde {

  // forward references
  class wfmdatabase;
  class channel;
  class globalrevision;
  class channel_state;

  // constant data structures with waveform type number information
  extern const std::unordered_map<std::type_index,unsigned> wtn_typemap; // look up typenum based on C++ typeid(type)
  extern const std::unordered_map<unsigned,std::string> wtn_typenamemap;
  extern const std::unordered_map<unsigned,size_t> wtn_typesizemap; // Look up element size bysed on typenum
  extern const std::unordered_map<unsigned,std::string> wtn_ocltypemap; // Look up opencl type string based on typenum

  // https://stackoverflow.com/questions/41494844/check-if-object-is-instance-of-class-with-template
  // https://stackoverflow.com/questions/43587405/constexpr-if-alternative
  template <typename> 
  struct wtn_type_is_shared_ptr: public std::false_type { };

  template <typename T> 
  struct wtn_type_is_shared_ptr<std::shared_ptr<T>>: public std::true_type { };
  
  
  template <typename T>
  typename std::enable_if<!wtn_type_is_shared_ptr<T>::value,bool>::type 
  wtn_type_is_shared_waveformbase_ptr()
  {
    return false;
  }

  template <typename T>
  typename std::enable_if<wtn_type_is_shared_ptr<T>::value,bool>::type 
  wtn_type_is_shared_waveformbase_ptr()
  {
    return (bool)std::is_base_of<waveform_base,typename T::element_type>::value;
  }
  
  
  template <typename T>
  unsigned wtn_fromtype()
  // works like wtn_typemap but can accommodate instances and subclasses of waveform_base. Also nicer error message
  {
    std::unordered_map<std::type_index,unsigned>::const_iterator wtnt_it = wtn_typemap.find(std::type_index(typeid(T)));

    if (wtnt_it != wtn_typemap.end()) {
      return wtnt_it->second;
    } else {
      // T may be a snde::waveform_base subclass instance
      if (wtn_type_is_shared_waveformbase_ptr<T>()) {
	return SNDE_WTN_WAVEFORM;
      } else {
	throw snde_error("Type %s is not supported in this context",typeid(T).name());
      }
    }
  }

  
  class waveform_base  {
    // may be subclassed by creator
    // mutable in certain circumstances following the conventions of snde_waveform

    // lock required to safely read/write mutable portions unless you are the owner and
    // you are single threaded and the information you are writing is for a subsequent state (info->state/info_state);
    // last lock in the locking order except for Python GIL
  public:
    std::mutex admin; 
    struct snde_waveform_base *info; // owned by this class and allocated with malloc; often actually a sublcass such as snde_ndarray_waveform
    std::atomic_int info_state; // atomic mirror of info->state
    std::shared_ptr<immutable_metadata> metadata; // pointer may not be changed once info_state reaches METADATADONE. The pointer in info is the .get() value of this pointer. 

    std::shared_ptr<waveform_storage_manager> storage_manager; // pointer initialized to a default by waveform constructor, then used by the allocate_storage() method. Any assignment must be prior to that. may not be used afterward; see waveform_storage in wfmstore_storage.hpp for details on pointed structure.
    std::shared_ptr<waveform_storage> storage; // pointer immutable once initialized  by allocate_storage() or reference_immutable_waveform().  immutable afterward; see waveform_storage in wfmstore_storage.hpp for details on pointed structure.

    // These next three items relate to the __originating__ globalrevision or waveform set state
    // wss, but depending on the state _originating_wss may not have been assigned yet and may
    // need to extract from _wfmdb and _originating_globalrev_index.
    // DON'T ACCESS THESE DIRECTLY! Use the .get_originating_wss() and ._get_originating_wss_wfmdb_and_wfm_admin_prelocked() methods.
    std::weak_ptr<wfmdatabase> _wfmdb;
    uint64_t _originating_globalrev_index;
    std::weak_ptr<waveform_set_state> _originating_wss;

    // Need typed template interface !!! ***
    
    // Some kind of notification that waveform is done to support
    // e.g. finishing a synchronous process such as a render.

    // This constructor is to be called by everything except the math engine
    //  * Should be called by the owner of the given channel, as verified by owner_id
    //  * Should be called within a transaction (i.e. the wfmdb transaction_lock is held by the existance of the transaction)
    //  * Because within a transaction, wfmdb->current_transaction is valid
    // must call wfmdb->register_new_wfm() on the constructed waveform
    waveform_base(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<channel> chan,void *owner_id,size_t info_structsize=sizeof(struct snde_waveform_base));

    // This constructor is reserved for the math engine
    // must call wfmdb->register_new_math_wfm() on the constructed waveform
    waveform_base(std::shared_ptr<wfmdatabase> wfmdb,std::string chanpath,std::shared_ptr<waveform_set_state> calc_wss,size_t info_structsize=sizeof(struct snde_waveform_base));

    // rule of 3
    waveform_base & operator=(const waveform_base &) = delete; 
    waveform_base(const waveform_base &orig) = delete;
    virtual ~waveform_base(); // virtual destructor so we can be subclassed

    virtual std::shared_ptr<waveform_set_state> _get_originating_wss_wfmdb_and_wfm_admin_prelocked(); // version of get_originating_wss() to use if you have the waveform database and waveform's admin locks already locked.
    std::shared_ptr<waveform_set_state> _get_originating_wss_wfmdb_admin_prelocked(); // version of get_originating_wss() to use if you have the waveform database admin lock already locked.

    virtual std::shared_ptr<waveform_set_state> get_originating_wss(); // Get the originating waveform set state (often a globalrev). You should only call this if you are sure that originating wss must still exist (otherwise may generate a snde_error), such as before the creator has declared the waveform "ready". This will lock the waveform database and wfm admin locks, so any locks currently held must precede both in the locking order
    

    // Mutable waveform only ***!!! Not properly implemented yet ***!!!
    /*
    virtual rwlock_token_set lock_storage_for_write();
    virtual rwlock_token_set lock_storage_for_read();
    */
    
    virtual void _mark_metadata_done_internal(/*std::shared_ptr<waveform_set_state> wss,const std::string &channame*/);
    virtual void mark_metadata_done();  // call WITHOUT admin lock (or other locks?) held. 
    virtual void mark_as_ready();  // call WITHOUT admin lock (or other locks?) held. 
  };


  class ndarray_waveform : public waveform_base {
  public:
    arraylayout layout; // changes to layout must be propagated to info.ndim, info.base_index, info.dimlen, and info.strides
    //std::shared_ptr<rwlock> mutable_lock; // for simply mutable waveforms; otherwise nullptr


    // This constructor is to be called by everything except the math engine
    //  * Should be called by the owner of the given channel, as verified by owner_id
    //  * Should be called within a transaction (i.e. the wfmdb transaction_lock is held by the existance of the transaction)
    //  * Because within a transaction, wfmdb->current_transaction is valid
    // WARNING: Don't call directly as this constructor doesn't add to the transaction (need to call wfmdb->register_new_waveform())
    //    * If the type is known at compile time, better to call ndtyped_waveform<T>::create_waveform(...)
    //    * If the type is known only at run time, call ::create_typed_waveform() method
    ndarray_waveform(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<channel> chan,void *owner_id,unsigned typenum,size_t info_structsize=sizeof(struct snde_ndarray_waveform));
    
    // This constructor is reserved for the math engine
    // Creates waveform structure . 
    // WARNING: Don't call directly as this constructor doesn't and  to the pre-existing globalrev (need to call wfmdb->register_new_math_wfm())
    //    * If the type is known at compile time, better to call ndtyped_waveform<T>::create_waveform(...)
    //    * If the type is known only at run time, call ::create_typed_waveform() method
    ndarray_waveform(std::shared_ptr<wfmdatabase> wfmdb,std::string chanpath,std::shared_ptr<waveform_set_state> calc_wss,unsigned typenum,size_t info_structsize=sizeof(struct snde_ndarray_waveform));

    // rule of 3
    ndarray_waveform & operator=(const ndarray_waveform &) = delete; 
    ndarray_waveform(const ndarray_waveform &orig) = delete;
    virtual ~ndarray_waveform();

    inline snde_ndarray_waveform *ndinfo() {return (snde_ndarray_waveform *)info;}

    // static factory methods for creating waveforms with runtime-determined types
    // for regular (non-math) use. Automatically registers the new waveform
    static std::shared_ptr<ndarray_waveform> create_typed_waveform(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<channel> chan,void *owner_id,unsigned typenum);
    static std::shared_ptr<ndarray_waveform> create_typed_waveform(std::shared_ptr<wfmdatabase> wfmdb,std::string chanpath,void *owner_id,std::shared_ptr<waveform_set_state> calc_wss,unsigned typenum);
    
    
    // must assign info.elementsize and info.typenum before calling allocate_storage()
    // fortran_order only affects physical layout, not logical layout (interpretation of indices)
    virtual void allocate_storage(std::vector<snde_index> dimlen, bool fortran_order=false);

    // alternative to allocating storage: Referencing an existing waveform
    virtual void reference_immutable_waveform(std::shared_ptr<ndarray_waveform> wfm,std::vector<snde_index> dimlen,std::vector<snde_index> strides,snde_index base_index);

    
    inline void *void_dataptr()
    {
      return *ndinfo()->basearray;
    }
    
    inline void *element_dataptr(const std::vector<snde_index> &idx)  // returns a pointer to an element, which is of size ndinfo()->elementsize
    {
      char *base_charptr = (char *) (*ndinfo()->basearray);
      
      char *cur_charptr = base_charptr + ndinfo()->elementsize*ndinfo()->base_index;
      for (size_t dimnum=0;dimnum < ndinfo()->ndim;dimnum++) {
	snde_index thisidx = idx.at(dimnum);
	assert(thisidx < ndinfo()->dimlen[dimnum]);
	cur_charptr += ndinfo()->strides[dimnum]*ndinfo()->elementsize*thisidx;
      }
      
      return (void *)cur_charptr;
    }

    inline size_t element_offset(const std::vector<snde_index> &idx)
    {      
      size_t cur_offset = ndinfo()->base_index;
      
      for (size_t dimnum=0;dimnum < ndinfo()->ndim;dimnum++) {
	snde_index thisidx = idx.at(dimnum);
	assert(thisidx < ndinfo()->dimlen[dimnum]);
	cur_offset += ndinfo()->strides[dimnum]*thisidx;
      }
      
      return cur_offset;
      
    }
    virtual double element_double(const std::vector<snde_index> &idx); // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    virtual void assign_double(const std::vector<snde_index> &idx,double val); // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 

    virtual int64_t element_int(const std::vector<snde_index> &idx); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    virtual void assign_int(const std::vector<snde_index> &idx,int64_t val); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 

    virtual uint64_t element_unsigned(const std::vector<snde_index> &idx); // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    virtual void assign_unsigned(const std::vector<snde_index> &idx,uint64_t val); // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    
  };


  class transaction {
  public:
    // mutable until end of transaction when it is destroyed and converted to a globalrev structure
    std::mutex admin; // last in the locking order except before python GIL. Must hold this lock when reading or writing structures within. Does not cover the channels/wavaveforms themselves.

    uint64_t globalrev; // globalrev for this transaction. Immutable once published
    std::unordered_set<std::shared_ptr<channel>> updated_channels;
    //Keep track of whether a new waveform is required for the channel (e.g. if it has a new owner) (use false for math waveforms)
    std::map<std::string,bool> new_waveform_required; // index is channel name for updated channels
    std::unordered_map<std::string,std::shared_ptr<waveform_base>> new_waveforms;

    // end of transaction propagates this structure into an update of wfmdatabase._channels
    // and a new globalrevision 


    
  };


  class active_transaction /* : public std::enable_shared_from_this<active_transaction> */ {
    // RAII interface to transaction
    // Don't use this directly from dataguzzler-python, because there
    // you need to drop the thread context before starting the
    // transaction and then reacquire after you have the
    // transaction lock.
  public:
    std::unique_lock<std::mutex> transaction_lock_holder;
    std::weak_ptr<wfmdatabase> wfmdb;
    std::shared_ptr<globalrevision> previous_globalrev;
    bool transaction_ended;
    
    active_transaction(std::shared_ptr<wfmdatabase> wfmdb);


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
    
    std::string channelpath; // Path of this channel in waveform database
    std::string owner_name; // Name of owner, such as a dataguzzler_python module
    void *owner_id; // pointer private to the owner (such as dataguzzler_python PyObject of the owner's representation of this channel) that
    // the owner can use to verify its continued ownership.

    bool hidden; // explicitly hidden channel
    bool math; // math channel


    std::shared_ptr<waveform_storage_manager> storage_manager; // storage manager for newly defined waveforms... Note that while the pointer is immutable, the pointed storage_manager is NOT immutable. 
    
    // The following only apply to math channels
    std::shared_ptr<instantiated_math_function> math_fcn; // If math is set, then this channel is one of the outputs of the given math_fcn  math_fcn is also immutable once published
    bool mathintermediate; // intermediate result of math calculation -- makes it implicitly hidden
    bool ondemand; // if the output is not actually stored in the database but just delivered on-demand
    bool data_requestonly; // if the output is to be stored in the database with the metadata always calculated but the underlying data only triggered to be computed if requested or needed by another waveform.
    bool data_mutable; // if the output is mutable

    channelconfig(std::string channelpath, std::string owner_name, void *owner_id,bool hidden,std::shared_ptr<waveform_storage_manager> storage_manager=nullptr);
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
    
    std::shared_ptr<channelconfig> _config; // atomic shared pointer to immutable data structure; nullptr for a deleted channel
    std::atomic<uint64_t> latest_revision; // 0 means "invalid"; generally 0 (or previous latest) during channel creation/undeletion; incremented to 1 (with an empty waveform if necessary) after transaction end 
    std::atomic<bool> deleted; // is the channel currently defined?

    std::mutex admin; // last in the locking order except before python GIL. Used to ensure there can only be one _config update at a time. 


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

  class channel_notification_criteria {
    // NOTE: This class should be considered no longer mutable by its creator once published.
    // The externally exposed mutation methods below are intended solely for the creation process
    
    // When in place within a channel_notify for a particular waveform_set_state the notification logic
    // may access/modify it so long as the waveform_set_state admin lock is held (removing criteria that are already satisifed)
    // Internal members should generally be treated as private from an external API perspective
  public:
    mutable std::mutex admin; // must be locked to read/modify waveformset_complete, metadataonly_channels and/or fullready_channels; last lock except for python GIL
    // (may also be interpreted by channel_notify subclasses as protecting subclass data)
    // Mutable so we can lock it even with a const reference
    bool waveformset_complete; // true if this notification is to occur only once the entire waveform_set/globalrev is marked complete

    // These next two members: entries keep getting removed from the sets as the criteria are satisfied
    std::unordered_set<std::string> metadataonly_channels; // specified channels must reach metadata only status (note: fullyready also satisifies criterion)
    std::unordered_set<std::string> fullyready_channels; // specified channels must reach fully ready status

    channel_notification_criteria();
    channel_notification_criteria & operator=(const channel_notification_criteria &); 
    channel_notification_criteria(const channel_notification_criteria &orig);
    ~channel_notification_criteria() = default;
    void add_waveformset_complete();
    void add_fullyready_channel(std::string);
    void add_metadataonly_channel(std::string);
  };
  
  class channel_notify : public std::enable_shared_from_this<channel_notify> {
  public:
    // base class
    // derive from this class if you want to get notified
    // when a channel or waveform, or channel or waveform set,
    // becomes ready.

    // Note that all channels must be in the same waveform_set_state/globalrevision
    // Notification occurs once all criteria are satisfied.

    // !!!*** channel_notify should probably be simplified such that adding criteria
    // not only goes into criteria but also adds into waveform_state_set removing error
    // prone extra code to manually add it in and when channel_notify gets copied in
    // during end_transaction()
    channel_notification_criteria criteria; 

    channel_notify();  // initialize with empty criteria; may add with criteria methods .criteria.add_waveformset_complete(), .criteria.add_fullyready_channel(), .criteria.add_mdonly_channel(); NOTE: After instantiating and setting criteria must call apply_to_wss() to apply it to a waveform_set_state or globalrev
    channel_notify(const channel_notification_criteria &criteria_to_copy);
    
    // rule of 3
    channel_notify & operator=(const channel_notify &) = delete; 
    channel_notify(const channel_notify &orig) = delete;
    virtual ~channel_notify()=default;
    
    virtual void perform_notify()=0; // will be called once ALL criteria are satisfied. May be called in any thread or context; must return quickly. Shouldn't do more than acquire a non-heavily-contended lock and perform a simple operation. NOTE: WILL NEED TO SPECIFY WHAT EXISTING LOCKS IF ANY MIGHT BE HELD WHEN THIS IS CALLED

    // These next three methods are called when one of the criteria has been satisifed
    virtual void notify_metadataonly(const std::string &channelpath); // notify this notifier that the given channel has satisified metadataonly (not usually modified by subclass)
    virtual void notify_ready(const std::string &channelpath); // notify this notifier that the given channel has satisified ready (not usually modified by subclass)
    virtual void notify_waveformset_complete(); // notify this notifier that all waveforms in this set are complete

    // check to see if any waveformset criterion is satisfied and notify if everything is satisfied
    virtual void check_waveformset_complete(std::shared_ptr<waveform_set_state> wss);


    // Internal only: Should be called with wss admin lock and criteria admin locks locked. Returns true if an immediate notification is due
    bool _check_all_criteria_locked(std::shared_ptr<waveform_set_state> wss,bool notifies_already_applied_to_wss);

    // check all criteria and notify if everything is satisfied. 
    virtual void check_all_criteria(std::shared_ptr<waveform_set_state> wss);



    virtual std::shared_ptr<channel_notify> notify_copier(); // default implementation throws a snde_error. Derived classes should use channel_notify(criteria) superclass constructor


    virtual void apply_to_wss(std::shared_ptr<waveform_set_state> wss); // apply this notification process to a particular waveform_set_state. WARNING: May trigger the notification immediately
  };

  class repetitive_channel_notify {
  public:
    // base class
    // either derive from this class or use our default implementation
    // with a derived channel_notify and an explicit notify_copier()
    std::shared_ptr<channel_notify> notify;

    // rule of 3
    repetitive_channel_notify & operator=(const repetitive_channel_notify &) = delete; 
    repetitive_channel_notify(const repetitive_channel_notify &orig) = delete;
    virtual ~repetitive_channel_notify()=default;

    virtual std::shared_ptr<channel_notify> create_notify_instance(); // default implementation uses the channel_notify's notify_copier() to create the instance
  };


  class promise_channel_notify: public channel_notify {
    // has a .promise member
    // with a .get_future() that you can
    // wait on  (be sure to drop all locks before waiting)
  public:
    std::promise<void> promise;

    // After construction, need to call .apply_to_wss() method!!!
    promise_channel_notify(const std::vector<std::string> &mdonly_channels,const std::vector<std::string> &ready_channels,bool waveformset_complete);
    // rule of 3
    promise_channel_notify & operator=(const promise_channel_notify &) = delete; 
    promise_channel_notify(const promise_channel_notify &orig) = delete;
    virtual ~promise_channel_notify()=default;

    
    void perform_notify();

  };
  
  
  class _unchanged_channel_notify: public channel_notify {
  public:
    // used internally to get notifications for subsequent globalrev that needs to have a reference to the version (that is not ready yet) in this waveform.
    std::weak_ptr<wfmdatabase> wfmdb;
    std::shared_ptr<globalrevision> subsequent_globalrev;
    channel_state &current_channelstate; 
    channel_state &sg_channelstate; 


    _unchanged_channel_notify(std::weak_ptr<wfmdatabase> wfmdb,std::shared_ptr<globalrevision> subsequent_globalrev,channel_state & current_channelstate,channel_state & sg_channelstate,bool mdonly); // After construction, need to call .apply_to_wss() method!!!

    virtual ~_unchanged_channel_notify()=default;
    
    virtual void perform_notify();
  };
  
  class channel_state {
  public:
    // for atomic updates to notify_ ... atomic shared pointers, you must lock the waveform_set_state's admin lock
    std::shared_ptr<channelconfig> config; // immutable
    std::shared_ptr<channel> _channel; // immutable pointer, but pointed data is not immutable, (but you shouldn't generally need to access this)
    std::shared_ptr<waveform_base> _wfm; // atomic shared ptr to waveform structure created to store the ouput; may be nullptr if not (yet) created. Always nullptr for ondemand waveforms... waveform contents may be mutable but have their own admin lock
    std::atomic<bool> updated; // this field is only valid once wfm() returns a valid pointer and once wfm()->state is READY or METADATAREADY. It is true if this particular waveform has a new revision particular to the enclosing waveform_set_state
    uint64_t revision; // This is assigned when the channel_state is created from _wfm->info->revision for manually created waveforms. For ondemand math waveforms this is not meaningful. For math waveforms with the math_function's new_revision_optional (config->math_fcn->fcn->new_revision_optional) flag clear, this is defined during end_transaction() ***!!!. If the new_revision_optional flag is set, this is defined when the math function instantiates its waveform in the waveform_base constructor
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> _notify_about_this_channel_metadataonly; // atomic shared ptr to immutable set of channel_notifies that need to be updated or perhaps triggered when this channel becomes metadataonly; set to nullptr at end of channel becoming metadataonly. 
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> _notify_about_this_channel_ready; // atomic shared ptr to immutable set of channel_notifies that need to be updated or perhaps triggered when this channel becomes ready; set to nullptr at end of channel becoming ready. 

    channel_state(std::shared_ptr<channel> chan,std::shared_ptr<waveform_base> wfm,bool updated);

    channel_state(const channel_state &orig); // copy constructor used for initializing channel_map from prototype defined in end_transaction()

    std::shared_ptr<waveform_base> wfm() const;
    std::shared_ptr<waveform_base> waveform_is_complete(bool mdonly); // uses only atomic members so safe to call in all circumstances. Set to mdonly if you only care that the metadata is complete. Normally call waveform_is_complete(false). Returns waveform pointer if waveform is complete to the requested condition, otherwise nullptr. 
    void issue_nonmath_notifications(std::shared_ptr<waveform_set_state> wss); // Must be called without anything locked. Issue notifications requested in _notify* and remove those notification requests
    void issue_math_notifications(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<waveform_set_state> wss); // Must be called without anything locked. Check for any math updates from the new status of this waveform
    
    void end_atomic_wfm_update(std::shared_ptr<waveform_base> new_waveform);


    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> begin_atomic_notify_about_this_channel_metadataonly_update();
    void end_atomic_notify_about_this_channel_metadataonly_update(std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> newval);
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_metadataonly();

    
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> begin_atomic_notify_about_this_channel_ready_update();
    void end_atomic_notify_about_this_channel_ready_update(std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> newval);
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_ready();
  };


  class waveform_status {
  public:
    std::map<std::string,channel_state> channel_map; // key is full channel path... The map itself (not the embedded states) is immutable once the waveform_set_state is published
    
    /// all of these are indexed by their their full path. Every entry in channel_map should be in exactly one of these. Locked by wss admin mutex per above
    // The index is the shared_ptr in globalrev_channel.config
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state *> defined_waveforms;
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state *> instantiated_waveforms;
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state *> metadataonly_waveforms;
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state *> completed_waveforms;

    waveform_status(const std::map<std::string,channel_state> & channel_map_param);
  };


  class waveform_set_state : public std::enable_shared_from_this<waveform_set_state> {
  public:
    std::mutex admin; // locks changes to wfmstatus including channel_map contents (map itself is immutable once published), mathstatus,  and the _waveforms reference maps/sets and notifiers. Precedes wfmstatus.channel_map.wfm.admin and Python GIL in locking order
    std::atomic<bool> ready; // indicates that all waveforms except ondemand and data_requestonly waveforms are READY (mutable waveforms may be OBSOLETE)
    waveform_status wfmstatus;
    math_status mathstatus; // note math_status.math_functions is immutable
    std::shared_ptr<waveform_set_state> _prerequisite_state; // C++11 atomic shared pointer. waveform_set_state to be used for self-dependencies and any missing dependencies not present in this state. This is an atomic shared pointer (read with .prerequisite_state()) that is set to nullptr once a new globalrevision is ready, so as to allow prior waveform revisions to be freed.
    std::unordered_set<std::shared_ptr<channel_notify>> waveformset_complete_notifiers; // Notifiers waiting on this waveform set state being complete. Criteria will be removed as they are satisifed and entries will be removed as the notifications are performed.

    
    waveform_set_state(const instantiated_math_database &math_functions,const std::map<std::string,channel_state> & channel_map_param,std::shared_ptr<waveform_set_state> prereq_state); // constructor
    // Rule of 3
    waveform_set_state& operator=(const waveform_set_state &) = delete; 
    waveform_set_state(const waveform_set_state &orig) = delete;
    virtual ~waveform_set_state()=default;

    void wait_complete(); // wait for all the math in this waveform_set_state or globalrev to reach nominal completion (metadataonly or ready, as configured)


    // admin lock must be locked when calling this function. Returns
    std::shared_ptr<waveform_set_state> prerequisite_state();
    void atomic_prerequisite_state_clear(); // sets the prerequisite state to nullptr
    
  };
  
  class globalrevision: public waveform_set_state { // should probably be derived from a class math_calculation_set or similar, so code can be reused for ondemand waveforms
    // channel_map is mutable until the ready flag is set. Once the ready flag is set only mutable and data_requestonly waveforms may be modified.
  public:
    // These commented members are really part of the waveform_set_state we are derived from
    //std::mutex admin; // locks changes to wfmstatus, mathstatus, wfmstatus.channel_map and the _waveforms reference maps/sets. Precedes wfmstatus.channel_map.wfm.admin and Python GIL in locking order
    //std::atomic<bool> ready; // indicates that all waveforms except ondemand and data_requestonly waveforms are READY (mutable waveforms may be OBSOLETE)
    //waveform_status wfmstatus;
    //math_status mathstatus; // note math_status.math_functions is immutable

    
    uint64_t globalrev;

    globalrevision(uint64_t globalrev, const instantiated_math_database &math_functions,const std::map<std::string,channel_state> & channel_map_param,std::shared_ptr<waveform_set_state> prereq_state);   
  };
  

  class wfmdatabase : public std::enable_shared_from_this<wfmdatabase> {
  public:
    std::mutex admin; // Locks access to _channels and _deleted_channels and _math_functions, _globalrevs and repetitive_notifies. In locking order, precedes channel admin locks, available_compute_resource_database, globalrevision admin locks, waveform admin locks, and Python GIL. 
    std::map<std::string,std::shared_ptr<channel>> _channels; // Generally don't use the channel map directly. Grab the latestglobalrev and use the channel map from that. 
    std::map<std::string,std::shared_ptr<channel>> _deleted_channels; // Channels are put here after they are deleted. They can be moved back into the main list if re-created. 
    instantiated_math_database _math_functions; 
    
    std::map<uint64_t,std::shared_ptr<globalrevision>> _globalrevs; // Index is global revision counter. The first element in this is the latest globalrev with all mandatory immutable channels ready. The last element in this is the most recently defined globalrev.
    std::shared_ptr<globalrevision> _latest_globalrev; // atomic shared pointer -- access with latest_globalrev() method;
    std::vector<std::shared_ptr<repetitive_channel_notify>> repetitive_notifies; 

    available_compute_resource_database compute_resources; // has its own admin lock.
    

    std::shared_ptr<waveform_storage_manager> default_storage_manager; // pointer is immutable once created; contents not necessarily immutable; see wfmstore_storage.hpp


    std::mutex transaction_lock; // ***!!! Before any dataguzzler-python module locks, etc.
    std::shared_ptr<transaction> current_transaction; // only valid while transaction_lock is held. 

    wfmdatabase();
        
    // avoid using start_transaction() and end_transaction() from C++; instantiate the RAII wrapper class active_transaction instead
    // (start_transaction() and end_transaction() are intended for C and perhaps Python)

    // a transaction update can be multi-threaded but you shouldn't call end_transaction()  (or the end_transaction method on the
    // active_transaction or delete the active_transaction) until all other threads are finished with transaction actions
    std::shared_ptr<active_transaction> start_transaction();
    std::shared_ptr<globalrevision> end_transaction(std::shared_ptr<active_transaction> act_trans);
    // add_math_function() must be called within a transaction
    void add_math_function(std::shared_ptr<instantiated_math_function> new_function,bool hidden,std::shared_ptr<waveform_storage_manager> storage_manager=nullptr);

    void register_new_wfm(std::shared_ptr<waveform_base> new_wfm);
    void register_new_math_wfm(void *owner_id,std::shared_ptr<waveform_set_state> calc_wss,std::shared_ptr<waveform_base> new_wfm); // registers newly created math waveform in the given wss (and extracts mutable flag for the given channel into the waveform structure)). 

    std::shared_ptr<globalrevision> latest_globalrev();

    // Allocate channel with a specific name; returns nullptr if the name is inuse
    std::shared_ptr<channel> reserve_channel(std::shared_ptr<channelconfig> new_config);

    //std::shared_ptr<channel> lookup_channel_live(std::string channelpath);


    // NOTE: python wrappers for wait_waveforms and wait_waveform_names need to drop dgpython thread context during wait and poll to check for connection drop
    //void wait_waveforms(std::vector<std::shared_ptr<waveform>> &);
    void wait_waveform_names(std::shared_ptr<waveform_set_state> wss,const std::vector<std::string> &metadataonly, const std::vector<std::string> fullyready);
    
  };


  class _previous_globalrev_done_notify: public channel_notify {
  public:
    // used internally to get notification that a previous globalrev is complete so as to remove entries from available_compute_resource_database blocked_list
    std::weak_ptr<wfmdatabase> wfmdb;
    std::shared_ptr<globalrevision> previous_globalrev;
    std::shared_ptr<globalrevision> current_globalrev;
    

    _previous_globalrev_done_notify(std::weak_ptr<wfmdatabase> wfmdb,std::shared_ptr<globalrevision> previous_globalrev,std::shared_ptr<globalrevision> current_globalrev);

    virtual ~_previous_globalrev_done_notify()=default;
    virtual void perform_notify();
  };
  
  template <typename T>
  class ndtyped_waveform : public ndarray_waveform {
  public:
    typedef T dtype;

    // This constructor is to be called by everything except the math engine
    //  * Should be called by the owner of the given channel, as verified by owner_id
    //  * Should be called within a transaction (i.e. the wfmdb transaction_lock is held by the existance of the transaction)
    //  * Because within a transaction, wfmdb->current_transaction is valid
    // WARNING: Don't call directly as this constructor doesn't add to the transaction (need to call wfmdb->register_new_waveform(). Use ndtyped_waveform<T>::create_waveform() instead
    ndtyped_waveform(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<channel> chan,void *owner_id) :
      ndarray_waveform(wfmdb,chan,owner_id,wtn_typemap.at(typeid(T)))
    {
      
    }
    
    // This constructor is reserved for the math engine
    // Creates waveform structure
    // WARNING: Don't call directly as this constructor doesn't add to the transaction (need to call wfmdb->register_new_math_waveform(). Use ndtyped_waveform<T>::create_waveform() instead
    ndtyped_waveform(std::shared_ptr<wfmdatabase> wfmdb,std::string chanpath,std::shared_ptr<waveform_set_state> calc_wss) :
      ndarray_waveform(wfmdb,chanpath,calc_wss,wtn_typemap.at(typeid(T)))
    {
      
    }

    static std::shared_ptr<ndtyped_waveform> create_waveform(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<channel> chan,void *owner_id)
    {
      std::shared_ptr<ndtyped_waveform> new_wfm = std::make_shared<ndtyped_waveform>(wfmdb,chan,owner_id);
      wfmdb->register_new_wfm(new_wfm);
      return new_wfm;
    }

    
    static std::shared_ptr<ndtyped_waveform> create_waveform(std::shared_ptr<wfmdatabase> wfmdb,std::string chanpath,void *owner_id,std::shared_ptr<waveform_set_state> calc_wss)
    {
      std::shared_ptr<ndtyped_waveform> new_wfm = std::make_shared<ndtyped_waveform>(wfmdb,chanpath,owner_id,calc_wss);
      wfmdb->register_new_math_wfm(calc_wss,new_wfm);
      return new_wfm;
    }
    
    T *dataptr() { return (T*)void_dataptr(); }

    T& element(std::vector<snde_index> idx);
    
    // rule of 3
    ndtyped_waveform & operator=(const ndtyped_waveform &) = delete; 
    ndtyped_waveform(const ndtyped_waveform &orig) = delete;
    ~ndtyped_waveform() { }


    
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
      return (double)dataptr()[offset];
    }

    // element_double for nonarithmetic types
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value,double>::type _element_double(const std::vector<snde_index> &idx) // WARNING: if array is mutable by others, it should generally be locked for read when calling this function!
    {
      throw snde_error("Cannot extract floating point value from non-arithmetic type");
    }
    
    // assign_double for arithmetic types
    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,void>::type _assign_double(const std::vector<snde_index> &idx,double val) // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      size_t offset = element_offset(idx);
      dataptr()[offset]=(T)val;      
    }

    // assign_double for nonarithmetic types
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value,void>::type _assign_double(const std::vector<snde_index> &idx,double val) // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      throw snde_error("Cannot assign floating point value from non-arithmetic type");
    }

    
    // element_int for arithmetic types
    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,int64_t>::type _element_int(const std::vector<snde_index> &idx) // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    {
      size_t offset = element_offset(idx);
      return (int64_t)dataptr()[offset];
    }

    // element_int for non-arithmetic types
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value,int64_t>::type _element_int(const std::vector<snde_index> &idx) // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    {
      throw snde_error("Cannot extract integer value from non-arithmetic type");
    }
    
    // assign_int for arithmetic types
    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,void>::type  _assign_int(const std::vector<snde_index> &idx,int64_t val) // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      size_t offset = element_offset(idx);
      dataptr()[offset]=(T)val;
    }

    // assign_int for non-arithmetic types
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value,void>::type  _assign_int(const std::vector<snde_index> &idx,int64_t val) // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      throw snde_error("Cannot assign integer value to non-arithmetic type");
    }
    
    // element_unsigned for arithmetic types
    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,uint64_t>::type _element_unsigned(const std::vector<snde_index> &idx) // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    {
      size_t offset = element_offset(idx);
      return (uint64_t)dataptr()[offset];
    }
    
    // element_unsigned for non-arithmetic types
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value,uint64_t>::type _element_unsigned(const std::vector<snde_index> &idx) // WARNING: May overflow; if array is mutable by others, it should generally be locked for read when calling this function!
    {
      throw snde_error("Cannot extract integer value from non-arithmetic type");
    }

    
    // assign_unsigned for arithmetic types
    template <typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value,void>::type _assign_unsigned(const std::vector<snde_index> &idx,uint64_t val) // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      size_t offset = element_offset(idx);
      dataptr()[offset]=(T)val;
    }


    // assign_unsigned for non-arithmetic types
    template <typename U = T>
    typename std::enable_if<!std::is_arithmetic<U>::value,void>::type _assign_unsigned(const std::vector<snde_index> &idx,uint64_t val) // WARNING: May overflow; if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      throw snde_error("Cannot assign integer value to non-arithmetic type");
    }


    // ... and here are the non-template wrappers
    double element_double(const std::vector<snde_index> &idx)
    {
      return _element_double(idx);
    }
    void assign_double(const std::vector<snde_index> &idx,double val) // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      _assign_double(idx,val);
    }
    
    int64_t element_int(const std::vector<snde_index> &idx)
    {
      return _element_int(idx);
    }
    void assign_int(const std::vector<snde_index> &idx,int64_t val) // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      _assign_int(idx,val);
    }

    uint64_t element_unsigned(const std::vector<snde_index> &idx)
    {
      return _element_unsigned(idx);
    }
    void assign_unsigned(const std::vector<snde_index> &idx,uint64_t val) // WARNING: if array is mutable by others, it should generally be locked for write when calling this function! Shouldn't be performed on an immutable array once the array is published. 
    {
      _assign_unsigned(idx,val);
    }

  };
  

  
};

#endif // SNDE_WFMSTORE_HPP
