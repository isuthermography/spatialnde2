%shared_ptr(snde::channel_notify);
%shared_ptr(snde::repetitive_channel_notify);
%shared_ptr(snde::promise_channel_notify);
%shared_ptr(snde::monitor_globalrevs);

%{
  #include "notify.hpp"

%}
namespace snde {
  // forward references
  class recording_state_set; //recstore.hpp
  class recdatabase; //recstore.hpp
  class globalrevision; //recstore.hpp
  
  class channel_notification_criteria {
    // NOTE: This class should be considered no longer mutable by its creator once published.
    // The externally exposed mutation methods below are intended solely for the creation process
    
    // When in place within a channel_notify for a particular recording_set_state the notification logic
    // may access/modify it so long as the recording_set_state admin lock is held (removing criteria that are already satisifed)
    // Internal members should generally be treated as private from an external API perspective
  public:
    //mutable std::mutex admin; // must be locked to read/modify recordingset_complete, metadataonly_channels and/or fullready_channels; last lock except for python GIL
    // (may also be interpreted by channel_notify subclasses as protecting subclass data)
    // Mutable so we can lock it even with a const reference
    bool recordingset_complete; // true if this notification is to occur only once the entire recording_set/globalrev is marked complete

    // These next two members: entries keep getting removed from the sets as the criteria are satisfied
    //std::unordered_set<std::string> metadataonly_channels; // specified channels must reach metadata only status (note: fullyready also satisifies criterion)
    //std::unordered_set<std::string> fullyready_channels; // specified channels must reach fully ready status

    channel_notification_criteria();
    //channel_notification_criteria & operator=(const channel_notification_criteria &); 
    channel_notification_criteria(const channel_notification_criteria &orig);
    ~channel_notification_criteria() = default;
    void add_recordingset_complete();
    
    void add_completion_channel(std::shared_ptr<recording_set_state> wss,std::string); // satisfied once the specified channel reaches the current (when this criteria is defined) definition of completion for that channel (mdonly vs fullyready)
    
    void add_fullyready_channel(std::string); // satisified once the specified channel becomes fullyready (inapplicable to mdonly channels -- may never trigger unless fullyready is requested)
    void add_metadataonly_channel(std::string); // satisfied once the specified channel achieves mdonly (if applied to an mdonly channel you may not get notified until the channel is fullyready)
  };
  
  class channel_notify /* : public std::enable_shared_from_this<channel_notify> */ {
  public:
    // base class
    // derive from this class if you want to get notified
    // when a channel or recording, or channel or recording set,
    // becomes ready.

    // Note that all channels must be in the same recording_set_state/globalrevision
    // Notification occurs once all criteria are satisfied.

    // !!!*** channel_notify should probably be simplified such that adding criteria
    // not only goes into criteria but also adds into recording_state_set removing error
    // prone extra code to manually add it in and when channel_notify gets copied in
    // during end_transaction()
    channel_notification_criteria criteria; 

    channel_notify();  // initialize with empty criteria; may add with criteria methods .criteria.add_recordingset_complete(), .criteria.add_fullyready_channel(), .criteria.add_mdonly_channel(); NOTE: After instantiating and setting criteria must call apply_to_wss() to apply it to a recording_set_state or globalrev
    channel_notify(const channel_notification_criteria &criteria_to_copy);
    
    // rule of 3
    channel_notify & operator=(const channel_notify &) = delete; 
    channel_notify(const channel_notify &orig) = delete;
    virtual ~channel_notify()=default;
    
    virtual void perform_notify()=0; // will be called once ALL criteria are satisfied. May be called in any thread or context; must return quickly. Shouldn't do more than acquire a non-heavily-contended lock and perform a simple operation. NOTE: WILL NEED TO SPECIFY WHAT EXISTING LOCKS IF ANY MIGHT BE HELD WHEN THIS IS CALLED

    // These next three methods are called when one of the criteria has been satisifed
    virtual void notify_metadataonly(const std::string &channelpath); // notify this notifier that the given mdonly channel has satisified metadataonly (not usually modified by subclass)
    virtual void notify_ready(const std::string &channelpath); // notify this notifier that the given channel has satisified ready (not usually modified by subclass)
    virtual void notify_recordingset_complete(); // notify this notifier that all recordings in this set are complete

    // check to see if any recordingset criterion is satisfied and notify if everything is satisfied
    virtual void check_recordingset_complete(std::shared_ptr<recording_set_state> wss);


    // Internal only: Should be called with wss admin lock and criteria admin locks locked. Returns true if an immediate notification is due
    bool _check_all_criteria_locked(std::shared_ptr<recording_set_state> wss,bool notifies_already_applied_to_wss);

    // check all criteria and notify if everything is satisfied. 
    virtual void check_all_criteria(std::shared_ptr<recording_set_state> wss);



    virtual std::shared_ptr<channel_notify> notify_copier(); // default implementation throws a snde_error. Derived classes should use channel_notify(criteria) superclass constructor


    virtual void apply_to_wss(std::shared_ptr<recording_set_state> wss); // apply this notification process to a particular recording_set_state. WARNING: May trigger the notification immediately
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
    //std::promise<void> promise;

    // After construction, need to call .apply_to_wss() method!!!
    promise_channel_notify(const std::vector<std::string> &mdonly_channels,const std::vector<std::string> &ready_channels,bool recordingset_complete);
    // rule of 3
    promise_channel_notify & operator=(const promise_channel_notify &) = delete; 
    promise_channel_notify(const promise_channel_notify &orig) = delete;
    virtual ~promise_channel_notify()=default;

    
    void perform_notify();

  };
  

  
  class monitor_globalrevs /* : public std::enable_shared_from_this<monitor_globalrevs> */{
    // created by recdatabase::start_monitoring_globalrevs()
  public:
    uint64_t next_globalrev_index;
    bool active;
    //std::mutex admin; // after the recdb admin lock in locking order; before the Python GIL
    //std::condition_variable ready_globalrev;

    //std::map<uint64_t,std::shared_ptr<globalrevision>> pending; // pending global revisions that are ready but have not been waited for
    
    monitor_globalrevs(std::shared_ptr<globalrevision> first);

    // rule of 3
    monitor_globalrevs & operator=(const monitor_globalrevs &) = delete; 
    monitor_globalrevs(const monitor_globalrevs &orig) = delete;
    virtual ~monitor_globalrevs()=default;

    std::shared_ptr<globalrevision> wait_next(std::shared_ptr<recdatabase> recdb);
    void close(std::shared_ptr<recdatabase> recdb); // permanently disable the monitoring
    
  };
  
};

