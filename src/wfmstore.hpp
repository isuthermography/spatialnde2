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

struct snde_waveform {
  // This structure and pointed data are fully mutable during the INITIALIZING state
  // In METADATAREADY state metadata_valid should be set and metadata storage becomes immutable
  // in READY state the rest of the structure and data pointed to is immutable except in the case of a mutable waveform for the state variable, which could could change to OBSOLETE and the data pointed to, which could change as well once the state becomes OBSOLETE
  // Note that in a threaded environment you can't safely read the state without an assocated lock, or you can read a mirrored atomic state variable, such as in class waveform. 
  
  char *name; // separate malloc() 
  uint64_t revision;
  int state; // see SNDE_WFMS... defines below

  // what has been filled out in this struct so far
  snde_bool metadata_valid;
  snde_bool dims_valid;
  snde_bool data_valid; 

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
    std::mutex admin; // lock required to safely write mutable portions; last lock in the locking order except for Python GIL
    struct snde_waveform info;
    atomic_int info_state; // atomic mirror of info->state
    arraylayout layout; // changes to layout must be propagated to info.ndim, info.base_index, info.dimlen, and info.strides

    std::shared_ptr<waveform_storage> storage; // pointer is immutable once created; see waveform_storage in wfmstore_storage.hpp for details on pointed strcture.

    // Need typed template interface !!! ***  
    
    // Some kind of notification that waveform is done to support
    // e.g. finishing a synchronous process such as a render.

    // This constructor is to be called by everything except the math engine
    //  * Should be called by the owner of the given channel, as verified by owner_id
    //  * Should be called within a transaction (i.e. the wfmdb transaction_lock is held by the existance of the transaction)
    //  * Because within a transaction, wfmdb->current_transaction is valid
    // This constructor automatically adds the new waveform to the current transaction
    waveform(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<channel> chan,void *owner_id);

    // This constructor is reserved for the math engine
    // Creates waveform structure and adds to the pre-existing globalrev. 
    waveform(std::shared_ptr<wfmdatabase> wfmdb,std::string chanpath,void *owner_id,std::shared_ptr<globalrevision> globalrev);

    // rule of 3
    waveform & operator=(const waveform &) = delete; 
    waveform(const waveform &orig) = delete;
    virtual ~waveform(); // virtual destructor so we can be subclassed

    
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
    std::shared_ptr<wfmdatabase> wfmdb;
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

  class globalrev_channel {
    // immutable once published until destroyed, but
    // the pointed waveform is mutable under some circumstances
    // with its own admin lock
    std::shared_ptr<channelconfig> config; // immutable
    std::shared_ptr<waveform> wfm; // waveform structure created to store the ouput; may be nullptr if not (yet) created. Always nullptr for ondemand waveforms

    globalrev_channel(std::shared_ptr<channelconfig> config,std::shared_ptr<waveform> wfm);
  };
  
  class globalrevision {
    // channel_map is mutable until the ready flag is set. Once the ready flag is set only mutable and data_requestonly waveforms may be modified. 
    std::mutex admin; // locks changes to channel_map. Precedes channel_map.wfm.admin and Python GIL in locking order
    std::atomic<bool> ready; // indicates that all waveforms except ondemand and data_requestonly waveforms are READY (mutable waveforms may be OBSOLETE)
    std::map<std::string,globalrev_channel> channel_map; // key is full channel path
    instantiated_math_database math_functions; // immutable once copied in on construction of the globalrevision
    uint64_t globalrev;

    // !!!*** Need math stuff here !!!***
    //std::map<std::string,waveform> math
    
    globalrevision(uint64_t globalrev);
  };
  

  class wfmdatabase {
    std::mutex admin; // Locks access to _channels and _deleted_channels and _math_functions and _globalrevs. In locking order, precedes channel admin locks, available_compute_resource_database, globalrevision admin locks, waveform admin locks, and Python GIL. 
    std::map<std::string,std::shared_ptr<channel>> _channels; // Generally don't use the channel map directly. Grab the latestglobalrev and use the channel map from that. 
    std::map<std::string,std::shared_ptr<channel>> _deleted_channels; // Channels are put here after they are deleted. They can be moved back into the main list if re-created. 
    instantiated_math_database _math_functions; 
    
    std::map<uint64_t,std::shared_ptr<globalrevision>> _globalrevs; // Index is global revision counter. The first element in this is the latest globalrev with all mandatory immutable channels ready. The last element in this is the most recently defined globalrev. 

    available_compute_resource_database compute_resources; // has its own admin lock.


    std::shared_ptr<waveform_storage_manager> default_storage_manager; // pointer is immutable once created; contents not necessarily immutable; see wfmstore_storage.hpp


    std::mutex transaction_lock; // ***!!! Before any dataguzzler-python module locks, etc.
    std::shared_ptr<transaction> current_transaction; // only valid while transaction_lock is held. 


    
    // avoid using start_transaction() and end_transaction() from C++; instantiate the RAII wrapper class active_transaction instead
    // (start_transaction() and end_transaction() are intended for C and perhaps Python)

    // a transaction update can be multi-threaded but you shouldn't call end_transaction()  (or the end_transaction method on the
    // active_transaction or delete the active_transaction) until all other threads are finished with transaction actions
    std::shared_ptr<active_transaction> start_transaction();
    void end_transaction(std::shared_ptr<active_transaction> act_trans);

    // Allocate channel with a specific name; returns nullptr if the name is inuse
    std::shared_ptr<channel> reserve_channel(std::string channelpath,std::string owner_name,void *owner_id,bool hidden,std::shared_ptr<waveform_storage_manager> storage_manager=nullptr);

    //std::shared_ptr<channel> lookup_channel_live(std::string channelpath);
    
  };

  // Need function to walk all channels and find all math channels?
  
};
