#include "wfmstore.hpp"

namespace snde {
  waveform::waveform(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<channel> chan,void *owner_id) :
    // This constructor is to be called by everything except the math engine
    //  * Should be called by the owner of the given channel, as verified by owner_id
    //  * Should be called within a transaction (i.e. the wfmdb transaction_lock is held by the existance of the transaction)
    //  * Because within a transaction, wfmdb->current_transaction is valid
    // This constructor automatically adds the new waveform to the current transaction
    info{
      .name=strdup(chan->config()->channelpath.c_str());
      .revision=0,
      .state=SNDE_WFMS_INITIALIZING,
      .metadata_valid=false,
      .dims_valid=false,
      .data_valid=false,
      .ndim=0,
      .base_index=0,
      .dimlen=nullptr,
      .strides=nullptr,
      .owns_dimlen_strides=nullptr,
      .immutable=true,
      .typenum=,
      .elementsize=0,
      .basearray = nullptr;
    },
    info_state(info.state),
    layout(),
    mutable_lock(nullptr),
    storage_manager(wfmdb->default_storage_manager),
    storage(nullptr)
  {
    uint64_t new_revision = ++chan->latest_revision; // atomic variable so it is safe to increment
    info.revision=new_revision;
    {
      std::lock_guard<std::mutex> curtrans_lock(wfmdb->current_transaction->admin);
      wfmdb->current_transaction->new_waveforms.emplace(chan->config()->channelpath,shared_from_this());
    }
  }
  
  waveform::waveform(std::shared_ptr<wfmdatabase> wfmdb,std::string chanpath,void *owner_id,std::shared_ptr<globalrevision> globalrev) :
    // This constructor is reserved for the math engine
    // Creates waveform structure and adds to the pre-existing globalrev. 
    info{
      .name=strdup(chanpath.c_str());
      .revision=0,
      .state=SNDE_WFMS_INITIALIZING,
      .metadata_valid=false,
      .dims_valid=false,
      .data_valid=false,
      .ndim=0,
      .base_index=0,
      .dimlen=nullptr,
      .strides=nullptr,
      .owns_dimlen_strides=nullptr,
      .immutable=true,
      .typenum=,
      .elementsize=0,
      .basearray = nullptr;
    },
    info_state(info.state),
    layout(),
    mutable_lock(nullptr),  // for simply mutable math waveforms will need to init with std::make_shared<rwlock>();
    storage_manager(wfmdb->default_storage_manager),
    storage(nullptr)
  {
    std::lock_guard<std::mutex> globalrev_admin(globalrev->admin);
    channel_state & globalrev_chan = globalrev->wfmstatus.channel_map.at(chanpath);
    assert(globalrev_chan.config->owner_id == owner_id);
    assert(globalrev_chan.config->math);
    info.immutable = globalrev_chan.config->data_mutable;
    
    globalrev_chan.wfm = shared_from_this();
  }

  virtual waveform::~waveform()
  {
    free(info->name);
    info->name=nullptr;
  }


  virtual void waveform::allocate_storage(std::vector<snde_index> dimlen, fortran_order=false)
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
    
    std::tie(storage,base_index) = storage_manager->allocate_waveform(info.name,info.revision,info.elementsize,info.typenum,nelem);
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
	strides.push_front(stride);
	stride *= dimlen.at(dimnum);
      }
      
    }
    
    layout=arraylayout(dimlen,strides,base_index);
    
    info.base_index=layout.base_index;
    info.ndim=layout.dimlen.size();
    info.dimlen=layout.dimlen.data;
    info.strides=layout.strides.data;
  }

  virtual void waveform::reference_immutable_waveform(std::shared_ptr<waveform> wfm,std::vector<snde_index> dimlen,std::vector<snde_index> strides,snde_index base_index)
  {
    snde_index first_index;
    snde_index last_index;
    size_t dimnum;

    if (!wfm->storage->finalized) {
      raise snde_error("Waveform %s trying to reference non-final data from waveform %s",info.name,wfm->info.name);      
      
    }
    
    info.typenum = wfm->info.typenum;

    if (info.elemsize != 0 && info.elemsize != wfm->info.elemsize) {
      raise snde_error("Element size mismatch in waveform %s trying to reference data from waveform %s",info.name,wfm->info.name);
    }
    info.elemsize = wfm->info.elemsize;

    
    first_index=base_index;
    for (dimnum=0;dimnum < dimlen;dimnum++) {
      if (strides.at(dimnum) < 0) { // this is somewhat academic because strides is currently snde_index, which is currently unsigned so can't possibly be negative. This test is here in case we make snde_index signed some time in the future... 
	first_index += strides.at(dimnum)*(dimlen.at(dimnum))-1;
      }
    }
    if (first_index < 0) {
      raise snde_error("Referencing negative indices in waveform %s trying to reference data from waveform %s",info.name,wfm->info.name);
    }

    last_index=base_index;
    for (dimnum=0;dimnum < dimlen;dimnum++) {
      if (strides.at(dimnum) > 0) { 
	last_index += strides.at(dimnum)*(dimlen.at(dimnum))-1;
      }
    }
    if (last_index >= wfm->storage->nelem) {
      raise snde_error("Referencing out-of-bounds indices in waveform %s trying to reference data from waveform %s",info.name,wfm->info.name);
    }

    
    storage = wfm->storage;
    layout=arraylayout(dimlen,strides,base_index);
    
    info.base_index=layout.base_index;
    info.ndim=layout.dimlen.size();
    info.dimlen=layout.dimlen.data;
    info.strides=layout.strides.data;
    
  }

  virtual rwlock_token_set waveform::lock_storage_for_write()
  {
    bool immutable;
    {
      std::lock_guard<std::mutex>(admin);
      immutable=info.immutable;
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

  virtual rwlock_token_set waveform::lock_storage_for_read()
  {
    bool immutable;
    {
      std::lock_guard<std::mutex>(admin);
      immutable=info.immutable;
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

  virtual void waveform::mark_as_ready()
  {
    #error not implemented
  }

  active_transaction::active_transaction(std::shared_ptr<wfmdatabase> wfmdb) :
    wfmdb(wfmdb)
    transaction_ended(false);
  {
    std::unique_lock<std::mutex> tr_lock_acquire(wfmdb->transaction_lock);;

    tr_lock_acquire.swap(transaction_lock_holder); // transfer lock into holder
    //wfmdb->_transaction_raii_holder=shared_from_this();

    assert(!wfmdb->current_transaction);

    wfmdb->current_transaction=std::make_shared<transaction>();

    {
      std::lock_guard<mutex> wfmdb_lock(wfmdb->admin);
      std::map<uint64_t,std::shared_ptr<globalrevision>>::iterator last_globalrev_ptr = wfmdb->_globalrevs->end();
      --last_globalrev_ptr; // change from final+1 to final entry

      uint64_t previous_globalrev_index = 0;
      if (last_globalrev_ptr != wfmdb->_globalrevs->end()) {
	// if there is a last globalrev (otherwise we are starting the first!)
	previous_globalrev_index = last_globalrev_ptr->first;
	previous_globalrev = last_globalrev_ptr->second;

      } else {
	previous_globalrev = nullptr; 
      }
    }      
    wfmdb->current_transaction->globalrev = previous_globalrev_index+1;
    

    
  }

  static void _identify_changed_channels(std::map<std::string,std::shared_ptr<channelconfig>> &all_channels_by_name,instantiated_math_database &mathdb, std::unordered_set<std::shared_ptr<channelconfig>> &maybechanged_channels,std::unordered_set<std::shared_ptr<channelconfig>> &changed_channels_dispatched,std::unordered_set<std::shared_ptr<channelconfig>> &changed_channels_need_dispatch,std::shared_ptr<channelconfig> channel_to_dispatch ) // channel_to_dispatch should be in changed_channels_need_dispatch
  // mathdb must be immutable; (i.e. it must already be copied into the global revision)
  {
    // std::unordered_set iterators remain valid so long as the iterated element itself is not erased, and the set does not need rehashing.
    // Therefore the changed_channels_dispatched and changed_channels_need_dispatch set needs to have reserve() called on it first with the maximum possible growth.
    
    // We do add to changed_channels_need dispatch
    //std::unordered_set<std::shared_ptr<channelconfig>>::iterator changed_chan;

    changed_channels_need_dispatch.erase(channel_to_dispatch);
    changed_channels_dispatch.emplace(channel_to_dispatch);

    if (channel_to_dispatch->math) {
      // look up what is dependent on this channel
      std::set<std::shared_ptr<instantiated_math_function>> &dependent_math_functions=mathdb.all_dependencies_of_channel.at(channel_to_dispatch->channelpath);

      for (auto && instantiated_math_ptr: dependent_math_functions) {
	if (instantiated_math_ptr->ondemand || instantiated_math_ptr->disabled) {
	  // ignore disabled and ondemand channels (for now)
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
	      changed_channels_to_dispatch.emplace(*maybechanged_it);
	      // remove it from the maybe changed set
	      maybechanged_channels.erase(maybechanged_it);
	    } 
	  }
	}
      }
    }

        
  }

  
  void active_transaction::end_transaction()
  // Warning: we may be called by the active_transaction destructor, so calling e.g. virtual methods on the active transaction
  // should be avoided.
  // Caller must ensure that all updating processes related to the transaction are complete. Therefore we don't have to worry about locking the current_transaction
  // ***!!!! Much of this needs to be refactored because it is mostly math-based and applicable to on-demand waveform groups.
  // (Maybe not; this just puts things into the channel_map so far, But we should factor out code
  // that would define a channel_map and status hash tables needed for an on_demand group)
  {
    std::set<std::shared_ptr<waveform>> waveforms_needing_finalization; // automatic waveforms created here that need to be marked as ready

    assert(wfmdb->current_transaction);


    std::shared_ptr<globalrevision> globalrev;
    std::map<std::string,std::shared_ptr<channelconfig>> all_channels_by_name;

    // set of channels definitely changed
    std::unordered_set<std::shared_ptr<channelconfig>> changed_channels_need_dispatch;
    std::unordered_set<std::shared_ptr<channelconfig>> changed_channels_dispatched;

    // set of channels not yet known to be changed
    std::unordered_set<std::shared_ptr<channelconfig>> maybechanged_channels;


    // set of ready channels
    std::unordered_set<channel_state &> ready_channels; // references into the globalrev->wfmstatus.channel_map
    
    {
      std::lock_guard<mutex> wfmdb_lock(wfmdb->admin);

      // build a class globalrevision from wfmdb->current_transaction
      globalrev = std::make_shared<globalrevision>(wfmdb->current_transaction->globalrev,wfmdb->_math_functions,previous_globalrev);
      globalrev->wfmstatus.channel_map.reserve(wfmdb->_channels.size());
      
      // Build a temporary map of the channels we still need to dispatch
      for (auto && channel_pointer: wfmdb->_channels) {
	std::shared_ptr<channelconfig> config = channel_pointer->config();
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
    for (auto && new_wfm_chanpath_ptr: wfmdb->current_transaction->new_waveforms) {

      std::string &chanpath=new_wfm_chanpath_ptr.first;
      
      std::shared_ptr<channelconfig> config = all_channels_by_name.at(chanpath);
      
      changed_channels_need_dispatch.emplace(config);
      maybechanged_channels.erase(config);

    }
    for (auto && updated_chan: wfmdb->current_transaction->updated_channels) {
      std::shared_ptr<channelconfig> config = updated_chan->config();

      std::unordered_set<std::shared_ptr<channelconfig>>::iterator mcc_it = maybechanged_channels.find(config);

      if (mcc_it != maybechanged_channels.end()) {
	changed_channels_need_dispatch.emplace(config);
	maybechanged_channels.erase(mcc_it);
      }
    }

    // Now empty out changed_channels_need_dispatch, adding into it any dependencies of currently referenced changed channels
    // This progressively identifies all (possibly) changed channels
    while ((std::unordered_set<std::shared_ptr<channelconfig>>::iterator channel_to_dispatch = changed_channels_need_dispatch.begin()) != changed_channels_need_dispatch.end()) {
      _identify_changed_channels(all_channels_by_name,*globalrev->status.math_functions, maybechanged_channels,changed_channels_dispatched,changed_channels_need_dispatch,channel_to_dispatch );
    }
    
    // now changed_channels_need_dispatch is empty, changed_channels_dispatched represents all changed channels,
    // and maybechanged_channels represents unchanged channels
    std::unordered_set<std::shared_ptr<channelconfig>> &changed_channels_to_process=changed_channels_dispatched;
    std::unordered_set<std::shared_ptr<channelconfig>> &unchanged_channels=maybechanged_channels;


    // Now reference previous revision of all unchanged channels, inserting into the new globalrev's channel_map
    for (auto && unchanged_channel: unchanged_channels) {
      assert(previous_globalrev); // Should always be present, because if we are on the first revision, each channel should have had a new waveform and therefore is changed!
      
      std::map<std::string,channel_state>::iterator previous_globalrev_chanmap_entry = previous_globalrev->wfmstatus.channel_map.find(unchanged_channel->channelpath);
      if (previous_globalrev_chanmap_entry==previous_globalrev->wfmstatus.channel_map.end()) {
	raise snde_error("Existing channel %s has no prior waveform",unchanged_channel->channelpath.c_str());
      }
      std::shared_ptr<waveform> channel_wfm = previous_globalrev_chanmap_entry->second.wfm;

      std::map<std::string,channel_state>::iterator channel_map_it;
      bool added_successfully;
      std::tie(channel_map_it,added_successfully) =
	globalrev->wfmstatus.channel_map.emplace(std::piecewise_construct,
						 std::forward_as_tuple(unchanged_channel->channelpath),
						 std::forward_as_tuple(unchanged_channel,channel_wfm));

      assert(added_successfully);
      // Queue notification that this channel is complete, except we are currently holding the admin lock, so we can't
      // do it now. Instead queue up a reference into the channel_map
      ready_channels.emplace(*channel_map_it);

      // place in globalrev->wfmstatus.completed_waveforms
      globalrev->wfmstatus.completed_waveforms.emplace(*channel_map_it);
    }

  
    
    // First, if we have an instantiated new waveform, place this in the channel_map
    for (auto && new_wfm_chanpath_ptr: wfmdb->current_transaction->new_waveforms) {
      std::shared_ptr<channelconfig> config = all_channels_by_name.at(new_wfm_chanpath_ptr.first);
      
      auto cm_it = globalrev->wfmstatus.channel_map.emplace(std::piecewise_construct,
						  std::forward_as_tuple(new_wfm_chanpath_ptr.first),
						  std::forward_as_tuple(config,new_wfm_chanpath_ptr.second)).first;
      // mark it as instantiated
      globalrev->wfmstatus.instantiated_waveforms.emplace(std::piecewise_construct,
						std::forward_as_tuple(config),
						std::forward_as_tuple(*cm_it));
      
      changed_channels_to_process.erase(config); // no further processing needed here
    }
    
    // Second, make sure if a channel was created, it has a waveform present and gets put in the channel_map
    for (auto && updated_chan: wfmdb->current_transaction->updated_channels) {
      std::shared_ptr<channelconfig> config = updated_chan->config();
      std::shared_ptr<waveform> new_wfm;

      if (changed_channels_to_process.find(config)==changed_channels_to_process.end()) {
	// already processed above because an explicit new waveform was provided. All done here.
	continue;
      }
      
      auto new_waveform_it = wfmdb->current_transaction->new_waveforms.find(config->channelpath);
      
      // new waveform should be required but not present; create one
      assert(wfmdb->current_transaction->new_waveform_required.at(config->channelpath) && new_waveform_it==wfmdb->current_transaction->new_waveforms.end());
      new_wfm = std::make_shared<waveform>(wfmdb,updated_chan,updated_chan->owner_id); // constructor adds itself to current transaction
      waveforms_needing_finalization.emplace(new_wfm); // Since we provided this, we need to make it ready, below
      
      // insert new waveform into channel_map
      auto cm_it = globalrev->wfmstatus.channel_map.emplace(std::piecewise_construct,
						  std::forward_as_tuple(config->channelpath),
						  std::forward_as_tuple(config,new_wfm));
      
      // mark it as instantiated
      globalrev->wfmstatus.instantiated_waveforms.emplace(std::piecewise_construct,
						std::forward_as_tuple(config),
						std::forward_as_tuple(*cm_it));
      
      // remove this from channels_to_process, as it has been already been inserted into the channel_map
      changed_channels_to_process.erase(config);
    }

    for (auto && changed_channel: changed_channels_to_process) {
      // Go through the set of changed channels which were not processed above
      // (These must be math dependencies))
      // Since the math hasn't started, we don't have defined waveforms yet
      // so the waveform is just nullptr
      auto cm_it = globalrev->wfmstatus.channel_map.emplace(std::piecewise_construct,
						  std::forward_as_tuple(changed_channel->channelpath),
						  std::forward_as_tuple(changed_channel,nullptr));
      
      // These waveforms are defined but not instantiated
      globalrev->defined_waveforms.wfmstatus.emplace(std::piecewise_construct,
						     std::forward_as_tuple(config),
						     std::forward_as_tuple(*cm_it));      
      
    }
  
    
    // !!!*** How do we splice implicit and explicit self-dependencies into the calculation graph???
    // Simple: We need to add to the _external_dependencies of the prior globalrev and
    // add to the missing_external_prerequisites of this globalrev. 

    // ***!!! Need to copy repetitive_notifies into place. 


    // globalrev->wfmstatus should have exactly one entry entry in the _waveforms maps
    // per channel_map etnry
    assert(globalrev->wfmstatus.channel_map.size() == (globalrev->wfmstatus.defined_waveforms.size() + globalrev->wfmstatus.instantiated_waveforms.size() + globalrev->wfmstatus.metadataonly_waveforms.size() + globalrev->wfmstatus.completed_waveforms.size()));

    bool all_ready = !globalrev->wfmstatus.defined_waveforms.size() && !globalrev->wfmstatus.instantiated_waveforms.size();
    
    {
      std::lock_guard<mutex> wfmdb_lock(wfmdb->admin);
      wfmdb->_globalrevs.emplace(wfmdb->current_transaction->globalrev,globalrev);
    
      wfmdb->current_transaction = nullptr; 
      assert(!transaction_ended);
      transaction_ended=true;
      transaction_lock_holder.unlock();
    }

    // Perform notifies that unchanged copied waveforms from prior revs are now ready
    // (and that globalrev is ready if there is nothing pending!)
    for (auto && readychan : ready_channels) { // readychan is a channel_state &
      // Issue metadataonly notifications
      {
	std::unique_lock<std::mutex> wss_admin(globalrev->admin);
	std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_metadataonly = readychan.begin_atomic_notify_about_this_channel_metadataonly_update();
	if (notify_about_this_channel_metadataonly) {
	  readychan.end_atomic_notify_about_this_channel_metadataonly_update(nullptr); // clear out notification list pointer
	  wss_admin.unlock();

	  // perform notifications
	  for (auto && channel_notify_ptr: notify_about_this_channel_metadataonly) {
	    channel_notify_ptr->notify_metadataonly(readychan.config->channelpath);
	  }
	}
      }

      // Issue ready notifications
      {

	std::unique_lock<std::mutex> wss_admin(globalrev->admin);
	std::shared_ptr<std::unordered_set<std::shared_ptr<channel_notify>>> notify_about_this_channel_ready = readychan.begin_atomic_notify_about_this_channel_ready_update();
	if (notify_about_this_channel_ready) {
	  readychan.end_atomic_notify_about_this_channel_ready_update(nullptr); // clear out notification list pointer
	  wss_admin.unlock();
	  
	  // perform notifications
	  for (auto && channel_notify_ptr: notify_about_this_channel_ready) {
	    channel_notify_ptr->notify_ready(readychan.config->channelpath);
	  }
	}

      }
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
    
  }
  
  active_transaction::~active_transaction()
  {
    //wfmdb->_transaction_raii_holder=nullptr;
    if (!transaction_ended) {
      end_transaction();
    }
  }


  channelconfig::channelconfig(std::string channelpath, std::string owner_name, void *owner_id,bool hidden,std::shared_ptr<waveform_storage_manager> storage_manager=nullptr) :
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
  
  std::shared_ptr<channelconfig> channel::config();
  {
    return std::atomic_load(&_config);
  }

  
  void channel::end_atomic_config_update(std::shared_ptr<channelconfig> new_config)
  {
    // admin must be locked when calling this function. It accepts the modified copy of the atomically guarded data
    std::atomic_store(&_config,new_config);
  }

  channel_notification_criteria::channel_notification_criteria() :
    waveformset_complete(false);
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

  channel_notification_criteria::add_waveformset_complete() 
  {
    // only allowed during creation so we don't worry about locking
    waveformset_complete=true;
    
  }

  channel_notification_criteria::add_fullyready_channel(std::string channelname) 
  {
    // only allowed during creation so we don't worry about locking
    fullyready_channels.emplace(channelname);    
  }

  channel_notification_criteria::add_metadataonly_channel(std::string channelname) 
  {
    // only allowed during creation so we don't worry about locking
    metadataonly_channels.emplace(channelname);    
  }

  channel_notify::channel_notify() :
    criteria()
  {

  }
  
  channel_notify::channel_notify(const channel_notifiation_criteria &criteria_to_copy) :
    criteria(criteria_to_copy)
  {
    
  }

  virtual void channel_notify::notify_metadataonly(const std::string &channelpath) // notify this notifier that the given channel has satisified metadataonly (not usually modified by subclass)
  {

    bool generate_notify=false;
    {
      std::lock_guard<std::mutex> criteria_admin(criteria.admin);
      
      std::unordered_set<std::string> mdo_channels_it = criteria.metadataonly_channels.find(channelpath);
      
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
  
  virtual void channel_notify::notify_ready(const std::string &channelpath) // notify this notifier that the given channel has satisified ready (not usually modified by subclass)
  {
    bool generate_notify=false;
    {
      std::lock_guard<std::mutex> criteria_admin(criteria.admin);
      
      std::unordered_set<std::string> fullyready_channels_it = criteria.fullyready_channels.find(channelpath);
      
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
  
  virtual void channel_notify::notify_waveformset_complete() // notify this notifier that all waveforms in this set are complete.
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
  
  std::shared_ptr<channel_notify> channel_notify::notify_copier()
  {
    raise snde_error("Copier must be provided for repetitive channel notifications");
  }

  virtual std::shared_ptr<channel_notify> repetitive_channel_notify::create_notify_instance()
  // this default implementation uses the channel_notify's notify_copier() to create the instance
  {
    return notify->notify_copier();
  }
  
  channel_state::channel_state(std::shared_ptr<channelconfig> config,std::shared_ptr<waveform> wfm) :
    config(config),
    _wfm(nullptr),
    _notifies(nullptr)
  {
    std::atomic_store(&_wfm,wfm);
    std::atomic_store(&_notifies,nullptr);
  }
  std::shared_ptr<waveform> channel_state::wfm()
  {
    return std::atomic_load(&_wfm);
  }

  void channel_state::end_atomic_wfm_update(std::shared_ptr<waveform> new_waveform)
  {
    std::atomic_store(&_wfm,new_waveform);

  }
  
  waveform_set_state::waveform_set_state(const instantiated_math_database &math_functions,std::shared_ptr<waveform_set_state> prereq_state) :
    ready(false),
    mathstatus(std::make_shared<instantiated_math_database>(math_functions)),
    _prerequisite_state(nullptr),
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
    std::atomic_store(&_prerequisite_state,nullptr);

  }

  
  globalrevision::globalrevision(uint64_t globalrev, const instantiated_math_database &math_functions,std::shared_ptr<waveform_set_state> prereq_state) :
    waveform_set_state(math_functions,prereq_state),
    globalrev(globalrev)
  {
    
  }

  
  
  std::shared_ptr<active_transaction> wfmdatabase::start_transaction()
  {
    return std::make_shared<active_transaction>();
  }
  
  void wfmdatabase::end_transaction(std::shared_ptr<active_transaction> act_trans)
  {
    act_trans->end_transaction();
  }

  std::shared_ptr<channel> wfmdatabase::reserve_channel(std::shared_ptr<channelconfig> new_config)
  {
    // Note that this is called with transaction lock held, but that is OK because transaction lock precedes wfmdb admin lock
    {
      std::lock_guard<std::mutex> wfmdb_lock(admin);
      
      std::map<std::string,std::shared_ptr<channel>>::iterator chan_it;
      std::shared_ptr<channel> new_chan;
      
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
      _channels->emplace(new_config->channelpath,new_chan);
    }
    {
      // add new_chan to current transaction
      std::lock_guard<std::mutex> curtrans_lock(current_transaction->admin);

      // verify waveform not already updated in current transaction
      if (current_transaction->new_waveforms.find(new_config->channelpath) != current_transaction->new_waveforms.end()) {
	raise snde_error("Replacing owner of channel %s in transaction where waveform already updated",new_config->channelpath);
      }
      
      current_transaction->updated_channels.emplace(new_chan);
      current_transaction->new_waveform_required.emplace(new_config->channelpath,true);
    }

    
    
    return new_chan;
  }

  void wfmdatabase::wait_waveforms(std::vector<std::shared_ptr<waveform>> &)
  // NOTE: python wrapper needs to drop thread context during wait and poll to check for connection drop
  {
#error not implemented
  }
  
  void wfmdatabase::wait_waveform_names(std::shared_ptr<globalrevision> globalrev,std::vector<std::string> &)
  // NOTE: python wrapper needs to drop thread context during wait and poll to check for connection drop
  {
    // should queue up a std::promise for central dispatch
    // then we wait here polling the corresponding future with a timeout so we can accommodate dropped
    // connections. 
#error not implemented
  }

}
