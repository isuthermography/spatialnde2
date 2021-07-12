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
    storage(nullptr)
  {
    std::lock_guard<std::mutex> globalrev_admin(globalrev->admin);
    globalrev_channel & globalrev_chan = globalrev->channel_map.at(chanpath);
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

  void active_transaction::end_transaction()
  // Warning: we may be called by the active_transaction destructor, so calling e.g. virtual methods on the active transaction
  // should be avoided.
  // Caller must ensure that all updating processes related to the transaction are complete. Therefore we don't have to worry about locking the current_transaction
  {
    assert(wfmdb->current_transaction);

    
    {
      std::lock_guard<mutex> wfmdb_lock(wfmdb->admin);

      // build a class globalrevision from wfmdb->current_transaction
      std::shared_ptr<globalrevision> globalrev = std::make_shared<globalrevision>(wfmdb->current_transaction->globalrev,wfmdb->_math_functions);

      // make sure all required waveforms are present
      for (auto && updated_chan: wfmdb->current_transaction->updated_channels) {
	if (wfmdb->current_transaction->new_waveform_required.at(updated_chan->config()->channelpath) && wfmdb->current_transaction->new_waveforms.find(updated_chan->config()->channelpath)==wfmdb->current_transaction->new_waveforms.end()) {
	  // new waveform required but not present; create one
	  std::shared_ptr<waveform> new_wfm = std::make_shared<waveform>(wfmdb,updated_chan,updated_chan->owner_id); // constructor adds itself to current transaction
	}

	// Iterate through all current channels
	for (auto && channel: wfmdb->_channels) {
	  std::shared_ptr<channelconfig> config = channel->config();
	  std::string &channelpath = config->channelpath;

	  std::shared_ptr<waveform> channel_wfm=nullptr;
	  
	  std::unordered_map<std::string,std::shared_ptr<waveform>>::iterator new_wfm_it = wfmdb->current_transaction->new_waveforms.find(channelpath);
	  if (new_wfm_it != wfmdb->current_transaction_new_waveforms.end()) {
	    // got new waveform in this channel
	    channel_wfm = new_wfm_it->second; 
	  } else {
	    if (!config->math) {
	      // non-math waveforms: If it exists it is part of the channel map, so pull waveform from prior revision
	      assert(previous_globalrev); // Should always be present, because if we are on the first revision, each channel should have had a new waveform!
	      
	      std::map<std::string,globalrev_channel>::iterator previous_globalrev_chanmap_entry = previous_globalrev->channel_map.find(channelpath);
	      if (previous_globalrev_chanmap_entry==previous_globalrev->channel_map.end()) {
		raise snde_error("New channel %s with no waveform",channelpath.c_str());
	      }
	      channel_wfm=previous_globalrev_chanmap_entry->second.wfm;
	    } else {
	      // this is a math channel
	      std::shared_ptr<instantiated_math_function> instmath = globalrev->math_functions->defined_math_functions.at(channelpath);

	      instmath->initialize_globalrev(globalrev);
	    }
	  }
	  
	  // add this channel to globalrev channel map
	  globalrev->channel_map.emplace(std::piecewise_construct,
					 std::forward_as_tuple(channelpath),
					 std::forward_as_tuple(config,channel_wfm));

	  
	}
	
      }

      
      
      
      wfmdb->_globalrevs.emplace(wfmdb->current_transaction->globalrev,globalrev);
    }
    wfmdb->current_transaction = nullptr; 
    assert(!transaction_ended);
    transaction_ended=true;
    transaction_lock_holder.unlock();
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
    _config(initial_config),
    latest_revision(0),
    deleted(false)
  {
    
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
  
  globalrev_channel::globalrev_channel(std::shared_ptr<channelconfig> config,std::shared_ptr<waveform> wfm) :
    config(config),
    wfm(wfm)
  {
    
  }
  
  
  globalrevision::globalrevision(uint64_t globalrev, const instantiated_math_database &math_functions) :
    ready(false),
    globalrev(globalrev),
    math_functions(math_functions)
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



}
