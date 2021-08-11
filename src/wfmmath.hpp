#ifndef SNDE_WFMMATH_HPP
#define SNDE_WFMMATH_HPP

#include <functional>

#include "wfmmath_compute_resource.hpp"
#include "wfmmath_parameter.hpp"
#include "wfmdb_paths.hpp"

namespace snde {
  // NEED TO CONSIDER THREAD SAFETY ACCESS OF ALL THE STRUCTURES IN HERE !!! (DONE) ***
  // Probably need a lock that is acquired AFTER relevant wfmstore lock(s)
  // because we will have to be going through the wfmstore to put computations
  // on hold in order to safely render mutable data. While going through it
  // we will need pull things off the todo_list and associated prioritized_computations multimaps
  //

  // PROBLEM: How to print back the defined math function:
  // SOLUTION: Need a data structure within the waveform database that "remembers" how it was
  // originally defined and verifies that that original definition is still
  // valid

  
  // implicit self-dependency: Present for any mutable math functions
  // or any math functions with new_revision_optional flag set


  // IDEA: Allow math waveforms to be dependent on external registers (akin to hw registers) that
  // have a typed interface and can notify when they are updated. They can allow implicit transactions if so configured
  // and have calcsync behavior so too much doesn't get queued up. -- maybe some sort of explicit grouping? 
  
  // defines for the type entry of the param_names_types list in a math_function... so far identical to DGM_MDT_... in dg_metadata.h
#define SNDE_MFPT_INT 0
#define SNDE_MFPT_STR 1
#define SNDE_MFPT_DBL 2
  // 3 is for an ancillary string
#define SNDE_MFPT_WAVEFORM 4

  // forward declarations
  class channelconfig; // defined in wfmstore.hpp
  class waveform_status; // defined in wfmstore.hpp
  class channel_state; // defined in wfmstore.hpp

  class math_status;
  class math_definition;
  
  class math_function : public std::enable_shared_from_this<math_function> { // a math function that is defined accessable so it can be instantiated
    // Immutable once published; that said it may be replaced in the database due to a reloading operation. 
  public:

    math_function(size_t num_results,const std::list<std::tuple<std::string,unsigned>> &param_names_types,std::function<std::shared_ptr<executing_math_function>(std::shared_ptr<waveform_set_state> wss,std::shared_ptr<instantiated_math_function> instantiated,bool is_mutable,bool mdonly)> initiate_execution);

    // Rule of 3
    math_function(const math_function &) = delete;
    math_function& operator=(const math_function &) = delete; 
    virtual ~math_function()=default;  // virtual destructor required so we can be subclassed

    
    // Should we put the name (of the function, not the channel) here???
    size_t num_results;
    std::list<std::tuple<std::string,unsigned>> param_names_types; // list of (name,type) tuples

    std::vector<bool> result_mutability; // for each result, is it mutable (if we are in mutable mode)
    
    bool new_revision_optional; // set if the function sometimes chooses not to create a new revision. Causes an implicit self-dependency, because we have to wait for the prior revision to finish to find out if that version was actually different. 
    bool pure_optionally_mutable; // set if the function is "pure" and can optionally operate on its previous output, only rewriting the modified area according to bounding_hyperboxes. If optionally_mutable is taken advantage of, there is an implicit self-dependency on the prior-revision
    bool mandatory_mutable; // set if the function by design mutates its previous output. Creates an implicit self-dependency.
    bool self_dependent; // set if the function by design is dependent on the prior revision of its previous output
    bool mdonly_allowed; // set if it is OK to instantiate this function in metadataonly form
    
    std::function<std::shared_ptr<executing_math_function>(std::shared_ptr<waveform_set_state> wss,std::shared_ptr<instantiated_math_function> instantiated,bool is_mutable,bool mdonly)> initiate_execution;
    
    // WARNING: If there is no implict or explicit self-dependency multiple computations for the same math function
    // but different versions can happen in parallel. 

    // note: instantiated_math_function returned by instantiate() needs to be explicitly added to waveform database/channels created/etc. !!!***
    virtual std::shared_ptr<instantiated_math_function> instantiate(const std::vector<std::shared_ptr<math_parameter>> & parameters,
								    const std::vector<std::shared_ptr<std::string>> & result_channel_paths,
								    std::string channel_path_context,
								    bool is_mutable,
								    bool ondemand,
								    bool mdonly,
								    std::shared_ptr<math_definition> definition,
								    std::string extra_params)=0;
								    
    // get_compute_options() returns a list of compute_resource_options, each of which has a compute_code pointer
    // NOTE: Somehow get_compute_options() or similar needs to consider the types of the parameter arrays and select
    // or configure code appropriately.
    //virtual std::shared_ptr<executing_math_function> initiate_execution(std::shared_ptr<waveform_set_state> wss,std::shared_ptr<instantiated_math_function> instantiated)=0; // usually returns a sub-class
  };

  // ***!!! compute_code is obsolete!!!***
  class compute_code {
    // This class represents the information provided by the math_function with the compute_resource_option; basically an offer to compute
    // immutable once published
    
    // BUG: The nature of code to provide may depend upon the resource. i.e. code to run in a C/C++ thread different from code to run
    // in a subprocess, possibly different from code that can run remotely over MPI, etc.
    // In particular anything to be run in a subprocess or over MPI we really need to know how to __find__ the code from the foreign context
    // once the input data has been marshalled.

    // BUG: How/where should locking be done for mutable waveforms?
    // BUG: How does allocation process work? For mutable waveforms? For immutable waveforms?
    
  public:
    compute_code() = default;
    // Rule of 3
    compute_code(const compute_code &) = delete;  // CC and CAO are deleted because we don't anticipate needing them. 
    compute_code& operator=(const compute_code &) = delete; 
    virtual ~compute_code()=default;  // virtual destructor required so we can be subclassed


    // !!!*** Should all of these methods be replaced by one overarching method
    // that then is implemented in subclasses for
    // various common patterns???
    virtual void determine_size(std::shared_ptr<waveform_set_state> wss, std::shared_ptr<executing_math_function> fcn, std::shared_ptr<compute_resource_option> option)=0;
    virtual void do_metadata_only(std::shared_ptr<waveform_set_state> wss, std::shared_ptr<executing_math_function> fcn, std::shared_ptr<compute_resource_option> option)=0;
    virtual void do_compute_from_metadata(std::shared_ptr<waveform_set_state> wss, std::shared_ptr<executing_math_function> fcn, std::shared_ptr<compute_resource_option> option)=0;
    virtual void do_compute(std::shared_ptr<waveform_set_state> wss, std::shared_ptr<executing_math_function> fcn, std::shared_ptr<compute_resource_option> option)=0;
  };


  
  

  class math_function_database {
    // represents the full set of available functions
    // _functions can be updated atomically using the admin lock.
  public:
    std::mutex admin; // last lock in order except for Python GIL
    std::shared_ptr<std::map<std::string,std::shared_ptr<math_function>>> _functions; // Actually C++11 atomic shared pointer to immutable map
    
  };

  class math_definition {
    // Represents the way an instantiated_math_function was defined
    // immutable once published. Needed for saving settings. 
  public:
    std::string definition_command;

    math_definition(std::string definition_command);
  };
  
  class instantiated_math_function: public std::enable_shared_from_this<instantiated_math_function>  {
    // This structure represents a defined math function. It is immutable
    // once published. 
    // but you may copy it for the purpose of changing it (using clone()) and replace it in the master database.
    // The clone() function should clear the .definition
    // member in the copy and point original_function at the original
    // (if not already defined) with the valid .definition member
  public:
    std::list<std::shared_ptr<math_parameter>> parameters; 
    //std::list<std::shared_ptr<channel>> results; // Note that null entries are legitimate if results are being ignored.
    std::vector<std::shared_ptr<std::string>> result_channel_paths; // Note that null entries are legitimate if results are being ignored.
    
    std::string channel_path_context; // context for parameters and result_channel_paths, if any are relative. 
    bool disabled; // if this math function is temporarily disabled
    bool is_mutable; // should be set if the function is mutable for any reason
    bool self_dependent; // this function is self-dependent: fcn->new_revision_optional || is_mutable || fcn->self_dependent;
    bool ondemand;
    bool mdonly; // Note: This determines whether the instantiation is mdonly. For the execution to be mdonly, the mdonly flag in the math_function_status must be true as well. 
    std::shared_ptr<math_function> fcn;
    std::shared_ptr<math_definition> definition;
    
    std::shared_ptr<instantiated_math_function> original_function; // null originally 
    // Should point to allocation interface here? ... No. Allocation interface comes from the channelconfig's storage_manager

    instantiated_math_function(const std::vector<std::shared_ptr<math_parameter>> & parameters,
			       const std::vector<std::shared_ptr<std::string>> & result_channel_paths,
			       std::string channel_path_context,
			       bool is_mutable,
			       bool ondemand,
			       bool mdonly,
			       std::shared_ptr<math_function> fcn,
			       std::shared_ptr<math_definition> definition);

    // Rule of 3
    instantiated_math_function(const instantiated_math_function &) = default;  // CC is present so subclass copy constructor can initialize us more easily
    instantiated_math_function& operator=(const instantiated_math_function &) = delete;
    virtual ~instantiated_math_function()=default;  // virtual destructor required so we can be subclassed


    //virtual bool check_dependencies(waveform_status &waveformstatus, math_status &mathstatus)=0; 
    // virtual clone method -- must be implemented in all subclasses. If .definition is non nullptr and definition_change is set, it clears the copy and points original_function at the old .definition
    virtual std::shared_ptr<instantiated_math_function> clone(bool definition_change=true); // only clone with definition_change=false for enable/disable of the function
    
  };

  class instantiated_math_database {
    // Used to represent currently defined functions. Both in main waveform database and then copied into each global revision.
    // In main waveform database, locked by waveform database admin lock;
    // Immutable once copied into a global revision
  public:
    std::map<std::string,std::shared_ptr<instantiated_math_function>> defined_math_functions; // key is channel path and channel_path_context; note that several keys will point to the same instantiated_math_function. Any changes to any functions require calling rebuild_dependency_map (below)
    
    // Hash table here so that we can look up the math channels that are dependent on an input which may have changed.
    std::unordered_map<std::string,std::unordered_set<std::shared_ptr<instantiated_math_function>>> all_dependencies_of_channel;

    // Hash table here so that we can look up the math functions that are dependent on a given function (within this global revision; does not include ondemand dependencies or implict or explicit self-dependencies 
    std::unordered_map<std::shared_ptr<instantiated_math_function>,std::unordered_set<std::shared_ptr<instantiated_math_function>>> all_dependencies_of_function;

    void rebuild_dependency_map(); // rebuild all_dependencies_of_channel and all_dependencies_of_function hash tables. Must be called any time any of the defined_math_functions changes. May only be called for the instantiated_math_database within the main waveform database, and the main waveform database admin lock must be locked when this is called. 

  };


  class math_function_status {
    // locked with math_status by the parent waveform_set_state's admin lock
    
    // A math function must be executed in a waveform_set_state/globalrev if it is not ondemand and:
    //   * It has changed, or
    //   * There is no prior revision, or
    //   * at least one of its (non-self) prerequisites has changed (only known for certain once at least one prereq has completed with an updated output (channel_state.updated))
    // The execution may independently define each of its output(s) as changed or not changed.
    // Therefore:
    //   * We need to track prerequisite status changes even if we don't know whether the function will need to be executed at all
    //   * When a preqreq status change comes in, if the prereq (except for a self-dep) has changed we need to OR that in to execution_demanded
    //   * If we have a self-dependency, the self-dep should be added to missing_external_dependencies AND
    //     added to the _external_dependencies of the prior globalrev. This is normally done in end_transaction()

  public:
    // Should this next map be replaced by a map of general purpose notifies that trigger when all missing prerequisites are satisified? (probably not but we still need the notification functionality)
    std::set<std::shared_ptr<channelconfig>> missing_prerequisites; // all missing (non-ready) local (in this waveform_set_state/globalrev) prerequisites of this function. Remove entries from the set as they become ready. When the set is empty, the math function represented by the key is dispatchable and should be marked as ready_to_execute
    
    std::set<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<channelconfig>>> missing_external_channel_prerequisites; // all missing (non-ready) external prerequisites of this function. Remove entries from the set as they become ready. 
    std::set<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>> missing_external_function_prerequisites; // all missing (non-ready) external prerequisites of this function. Remove entries from the set as they become ready. 

    bool mdonly; // if this execution is actually mdonly
    bool mdonly_executed; // if execution has completed at least through mdonly;
    bool is_mutable; // if this execution does in-place mutation of one or more of its parameters.
    bool execution_in_progress; // set while holding the wss's admin lock right before the executing_math_function is generated. Cleared when the executing_math_function is released. 
    
    bool execution_demanded; // even once all prereqs are satisfied, do we need to actually execute? This is only set if at least one of our non-self dependencies has changed and we are not disabled (or ondemand in a regular globalrev)
    bool ready_to_execute;
    bool metadataonly_complete; // if we are only executing to metadataonly, this is the complete flag
    bool complete; // set to true once fully executed; Note that this can shift from true back to false for a formerly metadataonly math function where the full data has been requested

    math_function_status(bool mdonly,bool is_mutable);
  };

  class math_status {
    // status of execution of math functions
    // locked by the parent waveform_set_state's admin lock. Be warned that
    // you cannot hold the admin locks of two waveform_set_states simultaneously. 
  public:
    std::shared_ptr<instantiated_math_database> math_functions; // immutable once copied in on construction
    std::unordered_map<std::shared_ptr<instantiated_math_function>,math_function_status> function_status; // lookup dependency and status info on this instantiated_math_function in our waveform_set_state/globalrev. You must hold the waveform_set_state's admin lock. 


    // NOTE: an entry in either _external_dependencies_on_channel or _external_dependencies_on_function is sufficient to get you the needed callback. 
    std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>>>> _external_dependencies_on_channel; // Lookup external math functions that are dependent on this channel -- usually subsequent revisions of the same function. May result from implicit or explicit self-dependencies. This map is immutable and pointed to by a C++11 atomic shared pointer it is safe to look it up with the external_dependencies() method without holding your waveform_set_state's admin lock. 

    std::shared_ptr<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>>>> _external_dependencies_on_function; // Lookup external math functions that are dependent on this math function -- usually subsequent revisions of the same function. May result from implicit or explicit self-dependencies. This map is immutable and pointed to by a C++11 atomic shared pointer it is safe to look it up with the external_dependencies() method without holding your waveform_set_state's admin lock. 


    // for the rest of these, you must own the waveform_set_state's admin lock
    std::unordered_set<std::shared_ptr<instantiated_math_function>> pending_functions; // pending functions where goal is full result
    std::unordered_set<std::shared_ptr<instantiated_math_function>> mdonly_pending_functions; // pending functions where goal is metadata only
    std::unordered_set<std::shared_ptr<instantiated_math_function>> completed_functions;  // completed functions with full result
    std::unordered_set<std::shared_ptr<instantiated_math_function>> completed_mdonly_functions; // pending functions where goal is metadata only and metadata is done (note that it is possible for fully ready functions to be in this list in some circumstances, for example if the full result was requested in another globalrev that references the same waveform structure. 
    
    
    math_status(std::shared_ptr<instantiated_math_database> math_functions,const std::map<std::string,channel_state> & channel_map);

    std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>>>> begin_atomic_external_dependencies_on_channel_update(); // must be called with waveform_set_state's admin lock held
    void end_atomic_external_dependencies_on_channel_update(std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>>>> newextdep);
    std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>>>> external_dependencies_on_channel();
    
    std::shared_ptr<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>>>> begin_atomic_external_dependencies_on_function_update(); // must be called with waveform_set_state's admin lock held
    void end_atomic_external_dependencies_on_function_update(std::shared_ptr<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>>>> newextdep);
    std::shared_ptr<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>>>> external_dependencies_on_function(); 


    void notify_math_function_executed(std::shared_ptr<wfmdatabase> wfmdb,std::shared_ptr<waveform_set_state> wss,std::shared_ptr<instantiated_math_function> fcn,bool mdonly);
    
    // check_dep_fcn_ready() assumes dep_wss admin lock is already held
    void check_dep_fcn_ready(std::shared_ptr<waveform_set_state> dep_wss,
			     std::shared_ptr<instantiated_math_function> dep_fcn,
			     math_function_status *mathstatus_ptr,
			     std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>> &ready_to_execute_appendvec);

  };

  
  class executing_math_function {
    // generated once we are ready to execute a math function
    // (all prerequisites are done)
    // usually subclassed
  public:
    std::shared_ptr<waveform_set_state> wss; // waveform set state in which we are executing
    std::shared_ptr<instantiated_math_function> inst;     // This attribute is immutable once published

    bool is_mutable;
    bool mdonly; 
    // should also have parameter values, references, etc. here
    
    // parameter values should include bounding_hyperboxes of the domains that actually matter,
    // at least if the function is optionally_mutable


    // self_dependent_waveforms is auto-created by the constructor
    std::list<std::shared_ptr<waveform_base>> self_dependent_waveforms; // only valid (size() > 0) with implicit/explict self dependency. entries will be nullptr first time through anyway. Entries may also be nullptr if the function output is being ignored rather than stored in the waveform database. 

    // compute_resource is assigned post-creation
    std::shared_ptr<assigned_compute_resource> compute_resource; // locked by acrd's admin lock
    // These next two elements are locked by the parent available_compute_resources_database admin lock
    // THEY HAE BEEN REPLACED BY THE ASSIGNED COMPUTE RESOURCE
    //std::vector<size_t> cpu_cores;  // vector of indices into available_compute_resource_cpu::functions_using_cores representing assigned CPU cores; from assigned_compute_resource_option_cpu and/or assigned_compute_resource_option_opencl
    //std::vector<size_t> opencl_jobs;  // vector of indices into available_compute_resource_cpu::functions_using_devices representing assigned GPU jobs; from assigned_compute_resource_option_opencl


    executing_math_function(std::shared_ptr<waveform_set_state> wss,std::shared_ptr<instantiated_math_function> fcn,bool is_mutable, bool mdonly);

    // Rule of 3
    executing_math_function(const executing_math_function &) = delete;
    executing_math_function& operator=(const executing_math_function &) = delete; 
    virtual ~executing_math_function()=default;  // virtual destructor required so we can be subclasse

    virtual bool perform_decide_new_revision()=0; // perform_decide_new_revision asks the executing math function to determine whether a new revision is to be created ***!!! SHOULD THIS BE SEPARATE FOR EACH OUTPUT CHANNEL RATHER THAN A SINGLE BOOL!!!???
    virtual std::list<std::shared_ptr<compute_resource_option>> perform_compute_options()=0; // perform_compute_options asks the executing math function to perform its compute_options step (which should not be compute intensive)

    virtual void perform_define_wfms()=0;
    virtual void perform_metadata()=0;
    virtual void perform_lock_alloc()=0;
    virtual void perform_exec()=0;

  };
}

#endif // SNDE_WFMMATH_HPP
