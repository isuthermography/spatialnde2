#ifndef SNDE_NOTIFY_HPP
#define SNDE_NOTIFY_HPP

#include <unordered_set>
#include <memory>
#include <future>

/* 
NOTIFICATION PRINCIPLES

Rigorous notification is a core challenge here because of the parallelism and the difficulty in arranging 
atomic operations with the necessary combinations of ordered locks. 

So in general, the notification system should have the following characteristics:
  * The characteristic being notified for is testable
  * The testing can be done in any thread and context, so long as locks are not previously being held
  * The characteristic being notified must be satisfied prior to or during (atomically) 
    the initiation of the notification
  * Redundant notifications should be ignored
  * Since an event may occur after the characteristic is tested but before the request for notification 
    is fully posted, when posting a notification, should always explicitly test right after posting. 
  * A thread that is planning to perform a notification and that has verified that the notification is
    appropriate can remove the notification request, taking responsibility itself for performing the
    notification

So the general process (Method A):
  * (optional) Requestor checks criterion
  * Requestor posts notification request
  * Requestor checks criteron; delists notification if satisfied (and still posted)
  * Event occurs in some thread "A" triggering criterion check, which is satsified
  * Thread A delists the notification, then performs the notification
Note that the above process may trigger an excess notification; therefore the recipient
must be robust against duplicated notifications. 

The only exceptions to the above are scenarios where the criterion check and the listing/delisting 
can be performed atomically, in which case (Method B): 
  * Requestor atomically checks criterion and posts notification request if not satisfied 
  * Event occurs in some thread "A" triggering atomic criterion check (which is satsified) and delist
    of the notification. 
  * Thread A then performs the notification


Recording database notification scenarios: 

1. channel_notify and derivatives. This uses Method B with notifications in channel_state::_notify_about_this_channel_metadataonly
   and channel_state::_notify_about_this_channel_ready. Criterion checking performed by channel_notify::_check_all_criteria_locked()
   based on the internal completion flags in recordings and channel_state::issue_nonmath_notifications()
2. Recording math notifications within a particular recording_set_state: Method A. These are used to trigger dependent functions and rely on
   math_status::pending_functions, mdonly_pending_functions, completed_functions, and completed_mdonly_functions sets. 
  * These are triggered by channel_state::issue_math_notifications()
3. Recording math and channel notifications from a different recording_set_state: Method A:
  * math_function_status::missing_external_channel_prerequisites and math_status::_external_dependencies_on_channel: channel_state::issue_math_notifications()
  * math_function_status::missing_external_function_preqrequisites and math_status::_external_dependencies_on_function (this is currently solely self-dependencies): math_status::notify_math_function_executed()
  



*/


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
    mutable std::mutex admin; // must be locked to read/modify recordingset_complete, metadataonly_channels and/or fullready_channels; last lock except for python GIL
    // (may also be interpreted by channel_notify subclasses as protecting subclass data)
    // Mutable so we can lock it even with a const reference
    bool recordingset_complete; // true if this notification is to occur only once the entire recording_set/globalrev is marked complete

    // These next two members: entries keep getting removed from the sets as the criteria are satisfied
    std::unordered_set<std::string> metadataonly_channels; // specified channels must reach metadata only status (note: fullyready also satisifies criterion)
    std::unordered_set<std::string> fullyready_channels; // specified channels must reach fully ready status

    channel_notification_criteria();
    channel_notification_criteria & operator=(const channel_notification_criteria &); 
    channel_notification_criteria(const channel_notification_criteria &orig);
    ~channel_notification_criteria() = default;
    void add_recordingset_complete();
    
    void add_completion_channel(std::shared_ptr<recording_set_state> wss,std::string); // satisfied once the specified channel reaches the current (when this criteria is defined) definition of completion for that channel (mdonly vs fullyready)
    
    void add_fullyready_channel(std::string); // satisified once the specified channel becomes fullyready (inapplicable to mdonly channels -- may never trigger unless fullyready is requested)
    void add_metadataonly_channel(std::string); // satisfied once the specified channel achieves mdonly (if applied to an mdonly channel you may not get notified until the channel is fullyready)
  };
  
  class channel_notify : public std::enable_shared_from_this<channel_notify> {
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
    std::promise<void> promise;

    // After construction, need to call .apply_to_wss() method!!!
    promise_channel_notify(const std::vector<std::string> &mdonly_channels,const std::vector<std::string> &ready_channels,bool recordingset_complete);
    // rule of 3
    promise_channel_notify & operator=(const promise_channel_notify &) = delete; 
    promise_channel_notify(const promise_channel_notify &orig) = delete;
    virtual ~promise_channel_notify()=default;

    
    void perform_notify();

  };
  
  class _unchanged_channel_notify: public channel_notify {
  public:
    // used internally to get notifications for subsequent globalrev that needs to have a reference to the version (that is not ready yet) in this recording.
    std::weak_ptr<recdatabase> recdb;
    std::shared_ptr<globalrevision> subsequent_globalrev;
    channel_state &current_channelstate; 
    channel_state &sg_channelstate; 
    bool mdonly;

    _unchanged_channel_notify(std::weak_ptr<recdatabase> recdb,std::shared_ptr<globalrevision> subsequent_globalrev,channel_state & current_channelstate,channel_state & sg_channelstate,bool mdonly); // After construction, need to call .apply_to_wss() method!!!

    virtual ~_unchanged_channel_notify()=default;
    
    virtual void perform_notify();
  };
  

  class _previous_globalrev_done_notify: public channel_notify {
  public:
    // used internally to get notification that a previous globalrev is complete so as to remove entries from available_compute_resource_database blocked_list
    std::weak_ptr<recdatabase> recdb;
    std::shared_ptr<globalrevision> previous_globalrev;
    std::shared_ptr<globalrevision> current_globalrev;
    

    _previous_globalrev_done_notify(std::weak_ptr<recdatabase> recdb,std::shared_ptr<globalrevision> previous_globalrev,std::shared_ptr<globalrevision> current_globalrev);

    virtual ~_previous_globalrev_done_notify()=default;
    virtual void perform_notify();
  };
};

#endif // SNDE_NOTIFY_HPP
