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

#ifndef __cplusplus
typedef void immutable_metadata; // must treat metadata pointer as opaque from C
#endif

struct snde_waveform {
  // This structure and pointed data are fully mutable during the INITIALIZING state
  // In METADATAREADY state metadata_valid should be set and metadata storage becomes immutable
  // in READY state the rest of the structure and data pointed to is immutable except in the case of a mutable waveform for the state variable, which could could change to OBSOLETE and the data pointed to, which could change as well once the state becomes OBSOLETE
  // Note that in a threaded environment you can't safely read the state without an assocated lock, or you can read a mirrored atomic state variable, such as in class waveform. 
  
  char *name; // separate malloc() 
  uint64_t revision;
  int state; // see SNDE_WFMS... defines below

  immutable_metadata *metadata; 
  
  // what has been filled out in this struct so far
  snde_bool metadata_valid;
  snde_bool dims_valid;
  snde_bool data_valid;
  snde_bool deletable;  // whether it is OK to call snde_waveform_delete() on this structure

  // This info must be kept sync'd with class waveform.layout
  snde_index ndim;
  snde_index base_index; // index in elements beyond (*basearray)
  snde_index *dimlen; // pointer often from waveform.layout.dimlen.get()
  snde_index *strides; // pointer often from waveform.layout.strides.get()

  snde_bool owns_dimlen_strides; // if set, dimlen and strides should be free()'d with this data structure.
  snde_bool immutable; // doesn't mean necessarily immutable __now__, just immutable once ready

  unsigned typenum; /// like mutablewfmstore.hpp typenum?
  size_t elementsize; 

  // *** Need Metadata storage ***
  // Use C or C++ library interface???

  
  // physical data storage
  void **basearray; // pointer into low-level store
};
#define SNDE_WFMS_INITIALIZING 0
#define SNDE_WFMS_METADATAREADY 1
#define SNDE_WFMS_READY 2
#define SNDE_WFMS_OBSOLETE 3

namespace snde {

  class waveform: public std::enable_shared_from_this<waveform> {
    // may be subclassed by creator
    // mutable in certain circumstances following the conventions of snde_waveform

    // lock required to safely read/write mutable portions unless you are the owner and
    // you are single threaded and the information you are writing is for a subsequent state (info->state/info_state);
    // last lock in the locking order except for Python GIL
    std::mutex admin; 
    struct snde_waveform info;
    atomic_int info_state; // atomic mirror of info->state
    std::shared_ptr<immutable_metadata> metadata; // pointer may not be changed once info_state reaches METADATADONE. The pointer in info is the .get() value of this pointer. 
    arraylayout layout; // changes to layout must be propagated to info.ndim, info.base_index, info.dimlen, and info.strides
    std::shared_ptr<rwlock> mutable_lock; // for simply mutable waveforms; otherwise nullptr

    std::shared_ptr<waveform_storage_manager> storage_manager; // pointer initialized to a default by waveform constructor, then used by the allocate_storage() method. Any assignment must be prior to that. may not be used afterward; see waveform_storage in wfmstore_storage.hpp for details on pointed structure.
    std::shared_ptr<waveform_storage> storage; // pointer immutable once initialized  by allocate_storage() or reference_immutable_waveform().  immutable afterward; see waveform_storage in wfmstore_storage.hpp for details on pointed structure.

    // Need typed template interface !!! ***
    
    // Some kind of notification that waveform is done to support
    // e.g. finishing a synchronous process such as a render.

    // This constructor is to be called by everything except the math engine
    //  * Should be called by the owner of the given channel, as verified by owner_id
    //  * Should be called within a transaction (i.e. the wfmdb transaction_lock is held by the existance of the transaction)
    //  * Because within a transaction, wfmdb->current_transaction is valid
    // This constructor automatically adds the new waveform to the current transaction
    // ***!!! Should we require a transaction pointer ***???!!!
    waveform(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<channel> chan,void *owner_id);

    // This constructor is reserved for the math engine
    // Creates waveform structure and adds to the pre-existing globalrev. 
    waveform(std::shared_ptr<wfmdatabase> wfmdb,std::string chanpath,void *owner_id,std::shared_ptr<globalrevision> globalrev);

    // rule of 3
    waveform & operator=(const waveform &) = delete; 
    waveform(const waveform &orig) = delete;
    virtual ~waveform(); // virtual destructor so we can be subclassed

    // must assign info.elementsize and info.typenum before calling allocate_storage()
  // fortran_order only affects physical layout, not logical layout (interpretation of indices)
    virtual void allocate_storage(std::vector<snde_index> dimlen, fortran_order=false);

    // alternative to allocating storage: Referencing an existing waveform
    virtual void reference_immutable_waveform(std::shared_ptr<waveform> wfm,std::vector<snde_index> dimlen,std::vector<snde_index> strides,snde_index base_index);

    virtual rwlock_token_set lock_storage_for_write();
    virtual rwlock_token_set lock_storage_for_read();

    virtual void mark_metadata_done();  // call WITHOUT admin lock (or other locks?) held
    virtual void mark_as_ready();  // call WITHOUT admin lock (or other locks?) held
  };

  
  class transaction {
    // mutable until end of transaction when it is destroyed and converted to a globalrev structure
    std::mutex admin; // last in the locking order except before python GIL. Must hold this lock when reading or writing structures within. Does not cover the channels/wavaveforms themselves.

    uint64_t globalrev; // globalrev for this transaction
    std::unordered_set<std::shared_ptr<channel>> updated_channels;
    //Keep track of whether a new waveform is required for the channel (e.g. if it has a new owner)
    std::map<std::string,bool> new_waveform_required; // index is channel name for updated channels
    std::unordered_map<std::string,std::shared_ptr<waveform>> new_waveforms;

    // end of transaction propagates this structure into an update of wfmdatabase._channels
    // and a new globalrevision 


    
  };


  class active_transaction /* : public std::enable_shared_from_this<active_transaction> */ {
    // RAII interface to transaction
    // Don't use this directly from dataguzzler-python, because there
    // you need to drop the thread context before starting the
    // transaction and then reacquire after you have the
    // transaction lock.
    std::unique_lock<std::mutex> transaction_lock_holder;
    std::weak_ptr<wfmdatabase> wfmdb;
    std::shared_ptr<globalrevision> previous_globalrev;
    bool transaction_ended;
    
    active_transaction(std::shared_ptr<wfmdatabase> wfmdb);


    // rule of 3
    active_transaction& operator=(const active_transaction &) = delete; 
    active_transaction(const active_transaction &orig) = delete;
    ~active_transaction(); // destructor releases transaction_lock from holder

    void end_transaction();
    
  };

  // need global revision class which will need to have a snapshot of channel configs, including math channels
  // since new math channel revisions are no longer defined until calculated, we will rely on the snapshot to
  // figure out which new channels need to be written. 
  
  class channelconfig {
    // The channelconfig is immutable once published; However it may be copied, privately updated by its owner, (if subclassed, you must use the correct subclasses
    // copy constructor!) and republished.  It may be
    // freed once no references to it exist any more.
    // can be subclassed to accommodate e.g. geometry scene graph entries, geometry parameterizations, etc. 

    channelconfig(std::string channelpath, std::string owner_name, void *owner_id,bool hidden,std::shared_ptr<waveform_storage_manager> storage_manager=nullptr);
    // rule of 3
    channelconfig& operator=(const channelconfig &) = default; 
    channelconfig(const channelconfig &orig) = default;
    virtual ~channelconfig() = default; // virtual destructor required so we can be subclassed
    
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
    
  };
  
  class channel {
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
    std::mutex admin; // must be locked to read/modify waveformset_complete, metadataonly_channels and/or fullready_channels; last lock except for python GIL
    // (may also be interpreted by channel_notify subclasses as protecting subclass data)
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
  
  class channel_notify {
    // base class
    // derive from this class if you want to get notified
    // when a channel or waveform, or channel or waveform set,
    // becomes ready.

    // Note that all channels must be in the same waveform_set_state/globalrevision
    // Notification occurs once all criteria are satisfied. 
    channel_notification_criteria criteria; 

    channel_notify();  // initialize with empty criteria; may add with criteria methods .criteria.add_waveformset_complete(), .criteria.add_fullyready_channel(), .criteria.add_mdonly_channel();
    channel_notify(const channel_notifiation_criteria &criteria_to_copy);
    
    // rule of 3
    channel_notify & operator=(const channel_notify &) = delete; 
    channel_notify(const channel_notify &orig) = delete;
    virtual ~channel_notify=default;
    
    virtual void perform_notify()=0; // will be called once ALL criteria are satisfied. May be called in any thread or context; must return quickly. Shouldn't do more than acquire a non-heavily-contended lock and perform a simple operation. NOTE: WILL NEED TO SPECIFY WHAT EXISTING LOCKS IF ANY MIGHT BE HELD WHEN THIS IS CALLED

    // These next three methods are called when one of the criteria has been satisifed
    virtual void notify_metadataonly(const std::string &channelpath); // notify this notifier that the given channel has satisified metadataonly (not usually modified by subclass)
    virtual void notify_ready(const std::string &channelpath); // notify this notifier that the given channel has satisified ready (not usually modified by subclass)
    virtual void notify_waveformset_complete(); // notify this notifier that all waveforms in this set are complete.



    virtual std::shared_ptr<channel_notify> notify_copier(); // default implementation throws a snde_error. Derived classes should use channel_notify(criteria) superclass constructor
    
  };

  class repetitive_channel_notify {
    // base class
    // either derive from this class or use our default implementation
    // with a derived channel_notify and an explicit notify_copier()
    std::shared_ptr<channel_notify> notify;

    // rule of 3
    repetitive_channel_notify & operator=(const repetitive_channel_notify &) = delete; 
    repetitive_channel_notify(const repetitive_channel_notify &orig) = delete;
    virtual ~repetitive_channel_notify=default;

    virtual std::shared_ptr<channel_notify> create_notify_instance(); // default implementation uses the channel_notify's notify_copier() to create the instance
  };


  class _unchanged_channel_notify: public channel_notify {
    // used internally to get notifications for subsequent globalrev that needs to have a reference to the version (that is not ready yet) in this waveform.
    std::weak_ptr<wfmdatabase> wfmdb
    std::shared_ptr<globalrev> subsequent_globalrev;
    channel_state &current_channelstate; 
    channel_state &sg_channelstate; 


    _unchanged_channel_notify(std::weak_ptr<wfmdatabase> wfmdb,std::shared_ptr<globalrev> subsequent_globalrev,channel_state & current_channelstate,channel_state & sg_channelstate,bool mdonly);
    
    virtual void perform_notify();
  };
  
  class channel_state {
    // for atomic updates to notify_ ... atomic shared pointers, you must lock the waveform_set_state's admin lock
    std::shared_ptr<channelconfig> config; // immutable
    std::shared_ptr<waveform> _wfm; // atomic shared ptr to waveform structure created to store the ouput; may be nullptr if not (yet) created. Always nullptr for ondemand waveforms... waveform contents may be mutable but have their own admin lock
    std::atomic<bool> updated; // this field is only valid once wfm() returns a valid pointer and once wfm()->state is READY or METADATAREADY. It is true if this particular waveform has a new revision particular to the enclosing waveform_set_state
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> _notify_about_this_channel_metadataonly; // atomic shared ptr to set of channel_notifies that need to be updated or perhaps triggered when this channel becomes metadataonly; set to nullptr at end of channel becoming metadataonly. 
    std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> _notify_about_this_channel_ready; // atomic shared ptr to set of channel_notifies that need to be updated or perhaps triggered when this channel becomes ready; set to nullptr at end of channel becoming ready. 

    channel_state(std::shared_ptr<channelconfig> config,std::shared_ptr<waveform> wfm,bool updated);
    std::shared_ptr<waveform> wfm();
    std::shared_ptr<waveform> waveform_is_complete(bool mdonly); // uses only atomic members so safe to call in all circumstances. Set to mdonly if you only care that the metadata is complete. Normally call waveform_is_complete(false). Returns waveform pointer if waveform is complete to the requested condition, otherwise nullptr. 
    void issue_nonmath_notifications(std::shared_ptr<waveform_set_state> wss); // Must be called without anything locked. Issue notifications requested in _notify* and remove those notification requests
    void issue_math_notifications(std::shared_ptr<waveform_set_state> wss); // Must be called without anything locked. Check for any math updates from the new status of this waveform
    
    void end_atomic_wfm_update(std::shared_ptr<waveform> new_waveform);
  };


  class waveform_status {
    std::map<std::string,channel_state> channel_map; // key is full channel path... The map itself (not the embedded states) are immutable once the waveform_set_state is published
    
    /// all of these are indexed by their their full path. Every entry in channel_map should be in exactly one of these. Locked by admin mutex per above
    // The index is the shared_ptr in globalrev_channel.config
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state&> defined_waveforms;
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state&> instantiated_waveforms;
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state&> metadataonly_waveforms;
    std::unordered_map<std::shared_ptr<channelconfig>,channel_state&> completed_waveforms;

  };


  class waveform_set_state {
  public:
    std::mutex admin; // locks changes to channel_map contents (map itself is immutable once published) and the _waveforms reference maps/sets and notifiers. Precedes wfmstatus.channel_map.wfm.admin and Python GIL in locking order
    std::atomic<bool> ready; // indicates that all waveforms except ondemand and data_requestonly waveforms are READY (mutable waveforms may be OBSOLETE)
    waveform_status wfmstatus;
    math_status mathstatus; // note math_status.math_functions is immutable
    std::shared_ptr<waveform_set_state> _prerequisite_state; // C++11 atomic shared pointer. waveform_set_state to be used for self-dependencies and any missing dependencies not present in this state. This is an atomic shared pointer (read with .prerequisite_state()) that is set to nullptr once a new globalrevision is ready, so as to allow prior waveform revisions to be freed.
    std::unordered_set<std::shared_ptr<channel_notify>> waveformset_complete_notifiers; // Notifiers waiting on this waveform set state being complete. Criteria will be removed as they are satisifed and entries will be removed as the notifications are performed.

    
    waveform_set_state(); // constructor

    // Rule of 3
    waveform_set_state& operator=(const waveform_set_state &) = delete; 
    waveform_set_state(const waveform_set_state &orig) = delete;
    virtual ~waveform_set_state()=default;

    // admin lock must be locked when calling this function. Returns
    std::shared_ptr<waveform_set_state> prerequisite_state();
    void atomic_prerequisite_state_clear(); // sets the prerequisite state to nullptr
    
  }
  
  class globalrevision: public waveform_set_state { // should probably be derived from a class math_calculation_set or similar, so code can be reused for ondemand waveforms
    // channel_map is mutable until the ready flag is set. Once the ready flag is set only mutable and data_requestonly waveforms may be modified.
  public:
    // These commented members are really part of the waveform_set_state we are derived from
    //std::mutex admin; // locks changes to wfmstatus, mathstatus, wfmstatus.channel_map and the _waveforms reference maps/sets. Precedes wfmstatus.channel_map.wfm.admin and Python GIL in locking order
    //std::atomic<bool> ready; // indicates that all waveforms except ondemand and data_requestonly waveforms are READY (mutable waveforms may be OBSOLETE)
    //waveform_status wfmstatus;
    //math_status mathstatus; // note math_status.math_functions is immutable

    
    uint64_t globalrev;

    
    globalrevision(uint64_t globalrev);
  };
  

  class wfmdatabase {
    std::mutex admin; // Locks access to _channels and _deleted_channels and _math_functions, _globalrevs and repetitive_notifies. In locking order, precedes channel admin locks, available_compute_resource_database, globalrevision admin locks, waveform admin locks, and Python GIL. 
    std::map<std::string,std::shared_ptr<channel>> _channels; // Generally don't use the channel map directly. Grab the latestglobalrev and use the channel map from that. 
    std::map<std::string,std::shared_ptr<channel>> _deleted_channels; // Channels are put here after they are deleted. They can be moved back into the main list if re-created. 
    instantiated_math_database _math_functions; 
    
    std::map<uint64_t,std::shared_ptr<globalrevision>> _globalrevs; // Index is global revision counter. The first element in this is the latest globalrev with all mandatory immutable channels ready. The last element in this is the most recently defined globalrev. 
    std::vector<std::shared_ptr<repetitive_channel_notify>> repetitive_notifies; 

    available_compute_resource_database compute_resources; // has its own admin lock.
    

    std::shared_ptr<waveform_storage_manager> default_storage_manager; // pointer is immutable once created; contents not necessarily immutable; see wfmstore_storage.hpp


    std::mutex transaction_lock; // ***!!! Before any dataguzzler-python module locks, etc.
    std::shared_ptr<transaction> current_transaction; // only valid while transaction_lock is held. 

      ,    
    // avoid using start_transaction() and end_transaction() from C++; instantiate the RAII wrapper class active_transaction instead
    // (start_transaction() and end_transaction() are intended for C and perhaps Python)

    // a transaction update can be multi-threaded but you shouldn't call end_transaction()  (or the end_transaction method on the
    // active_transaction or delete the active_transaction) until all other threads are finished with transaction actions
    std::shared_ptr<active_transaction> start_transaction();
    void end_transaction(std::shared_ptr<active_transaction> act_trans);

    // Allocate channel with a specific name; returns nullptr if the name is inuse
    std::shared_ptr<channel> reserve_channel(std::string channelpath,std::string owner_name,void *owner_id,bool hidden,std::shared_ptr<waveform_storage_manager> storage_manager=nullptr);

    //std::shared_ptr<channel> lookup_channel_live(std::string channelpath);


    // NOTE: python wrappers for wait_waveforms and wait_waveform_names need to drop dgpython thread context during wait and poll to check for connection drop
    void wait_waveforms(std::vector<std::shared_ptr<waveform>> &);
    void wait_waveform_names(std::shared_ptr<globalrevision> globalrev,std::vector<std::string> &);
    
  };

  // Need function to walk all channels and find all math channels?
  
};
