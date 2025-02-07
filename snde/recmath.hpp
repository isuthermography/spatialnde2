#ifndef SNDE_RECMATH_HPP
#define SNDE_RECMATH_HPP

#include <functional>

#include "snde/recmath_compute_resource.hpp"
#include "snde/recmath_parameter.hpp"
#include "snde/recdb_paths.hpp"

namespace snde {
  // NEED TO CONSIDER THREAD SAFETY ACCESS OF ALL THE STRUCTURES IN HERE !!! (DONE) ***
  // Probably need a lock that is acquired AFTER relevant recstore lock(s)
  // because we will have to be going through the recstore to put computations
  // on hold in order to safely render mutable data. While going through it
  // we will need pull things off the todo_list and associated prioritized_computations multimaps
  //

  // PROBLEM: How to print back the defined math function:
  // SOLUTION: Need a data structure within the recording database that "remembers" how it was
  // originally defined and verifies that that original definition is still
  // valid

  
  // implicit self-dependency: Present for any mutable math functions
  // or any math functions with new_revision_optional flag set
  // or any dynamic dependencies
  // or anything dependent directly or indirectly on something with
  // uncertain execution

  // IDEA: Allow math recordings to be dependent on external registers (akin to hw registers) that
  // have a typed interface and can notify when they are updated. They can allow implicit transactions if so configured
  // and have calcsync behavior so too much doesn't get queued up. -- maybe some sort of explicit grouping? 
  
  // defines for the type entry of the param_names_types list in a math_function... so far identical to DGM_MDT_... in dg_metadata.h
#define SNDE_MFPT_INT 0
#define SNDE_MFPT_STR 1
#define SNDE_MFPT_DBL 2
  // 3 is for an ancillary string
#define SNDE_MFPT_BOOL 4
#define SNDE_MFPT_RECORDING 5
#define SNDE_MFPT_VECTOR 6
#define SNDE_MFPT_ORIENTATION 7
#define SNDE_MFPT_INDEXVEC 8 // vector of indices
#define SNDE_MFPT_MAP 9 // map
#define SNDE_MFPT_OFFSETCALIB 10 // used by offset_calibration.cpp internal messages
#define SNDE_MFPT_METADATA 11 // constructible_metadata

  // forward declarations
  class channelconfig; // defined in recstore.hpp
  class recording_status; // defined in recstore.hpp
  class channel_state; // defined in recstore.hpp
  class lockmanager; // defined in lockmanager.hpp

  class math_status;
  class math_function_status;
  class math_definition;
  class pending_math_definition;
  class math_parameter;
  class instantiated_math_database;
  class instantiated_math_function;
  class math_definition;

  struct mncn_lessthan { // less than operator for may_need_completion_notification so it can be used in a std::set
    // std::set<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>>,mncn_lessthan> 
    bool operator()( const std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>> & lhs,
                 const std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>> & rhs ) const noexcept;
  };

  
  class math_instance_parameter {
  public:
    unsigned paramtype; // SNDE_MFPT_XXX from above

    math_instance_parameter(unsigned paramtype);

    // Rule of 3
    math_instance_parameter(const math_instance_parameter &) = delete;
    math_instance_parameter& operator=(const math_instance_parameter &) = delete;
    virtual ~math_instance_parameter()=default;  // virtual destructor required so we can be subclassed

    virtual bool operator==(const math_instance_parameter &ref)=0; // used for comparing extra parameters to instantiated_math_functions
    virtual bool operator!=(const math_instance_parameter &ref)=0;
  };
  
  class list_math_instance_parameter : public math_instance_parameter {
  public:
    std::vector<std::shared_ptr<math_instance_parameter>> list;

    list_math_instance_parameter(std::vector<std::shared_ptr<math_instance_parameter>> list);
    
    virtual bool operator==(const math_instance_parameter &ref); // used for comparing extra parameters to instantiated_math_functions
    virtual bool operator!=(const math_instance_parameter &ref);
  };
    
  class dict_math_instance_parameter : public math_instance_parameter {
  public:
    std::unordered_map<std::string, std::shared_ptr<math_instance_parameter>> dict;

    dict_math_instance_parameter(std::unordered_map<std::string, std::shared_ptr<math_instance_parameter>> dict);
    
    virtual bool operator==(const math_instance_parameter &ref); // used for comparing extra parameters to instantiated_math_functions
    virtual bool operator!=(const math_instance_parameter &ref);
  };
  
  class string_math_instance_parameter : public math_instance_parameter {
  public:
    std::string value;

    string_math_instance_parameter(std::string value);

    virtual bool operator==(const math_instance_parameter &ref); // used for comparing extra parameters to instantiated_math_functions
    virtual bool operator!=(const math_instance_parameter &ref);
  };
  
  class int_math_instance_parameter : public math_instance_parameter {
  public:
    int64_t value;

    int_math_instance_parameter(int64_t value);

    virtual bool operator==(const math_instance_parameter &ref); // used for comparing extra parameters to instantiated_math_functions
    virtual bool operator!=(const math_instance_parameter &ref);
  };
  
  class double_math_instance_parameter : public math_instance_parameter {
  public:
    double value;

    double_math_instance_parameter(double value);
    
    virtual bool operator==(const math_instance_parameter &ref); // used for comparing extra parameters to instantiated_math_functions
    virtual bool operator!=(const math_instance_parameter &ref);
  };

  
  class math_function : public std::enable_shared_from_this<math_function> { // a math function that is defined accessable so it can be instantiated
    // Immutable once published; that said it may be replaced in the database due to a reloading operation. 
  public:

    math_function(std::string function_name,size_t num_results,const std::vector<std::pair<std::string,unsigned>> &param_names_types,std::function<std::shared_ptr<executing_math_function>(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> instantiated)> initiate_execution);

    // Rule of 3
    math_function(const math_function &) = delete;
    math_function& operator=(const math_function &) = delete; 
    virtual ~math_function()=default;  // virtual destructor required so we can be subclassed

    std::string function_name;
    size_t num_results;
    std::vector<std::pair<std::string,unsigned>> param_names_types; // list of (name,type) tuples

    
    bool new_revision_optional; // set if the function sometimes chooses not to create a new revision. Causes an implicit self-dependency, because we have to wait for the prior revision to finish to find out if that version was actually different. Note that new_revision_optional implies that execution is optional but execution of a new_revision_optional math function does not guarantee it will actually create new revisions but may still reference prior revs. Execution of a non-new_revision_optional math function is guaranteed to define new recordings in each result channel. 
    bool pure_optionally_mutable; // set if the function is "pure" and can optionally operate on its previous output, only rewriting the modified area according to bounding_hyperboxes. If optionally_mutable is taken advantage of, there is an implicit self-dependency on the prior-revision
    bool mandatory_mutable; // set if the function by design mutates its previous output. Creates an implicit self-dependency.
    bool self_dependent; // set if the function by design is dependent on the prior revision of its previous output
    bool mdonly_allowed; // set if it is OK to instantiate this function in metadataonly form. Note mdonly is incompatible with new_reivision_optional

    bool dynamic_dependency;  // dynamic_dependency is where at least one recording the function is dependent on may by its content induce additional dependenc(ies). Thus changes to the function parameters are a sufficient condition to trigger execution (or at least a new_revision_optional assessment), but they are not a necessary condition. As such the function needs to be asked whether execution might be necessary even in situations where it might appear not. The function will need access to its parameters to make that determination. ***!!! will need to track versions over dep. tree to decide whether to execute. Probably implicit self-dependency
    
    std::function<std::shared_ptr<executing_math_function>(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> instantiated)> initiate_execution;

    std::shared_ptr<std::function<bool(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> instantiated, math_function_status *mathstatus_ptr)>> find_additional_deps; // the find_additional_deps method is nullptr unless the function is a dynamic_dependency. For dynamic_dependencies this function walks the current list of dependencies (for instantiated parameter) with rss set state and adds additional dependencies that it has discovered to the mathstatus_ptr missing_prerequisites, missing_external_channel_prerequisites, and/or missing_external_function_prerequisites. It returns true if any were added (or if a repeated call is otherwise necessary); false otherwise. Any additions to missing_external_channel_prerequisites needs a corresponding addition in the prerequisite state to math_status::external_dependencies_on_channel. Any addition to missing_external_function_prerequisites requires an addition in the prerequisite state to math_status::external_dependencies_on_function. ANY ADDITION TO missing_prerequisites REQUIRES AN ADDITION TO math_status::extra_internal_dependencies_on_channel!!! (and the prerequisite must actually be missing; if present should simply increment num_modified_prerequisites if the prerequisite has been modified. 
    
    // WARNING: If there is no implict or explicit self-dependency multiple computations for the same math function
    // but different versions can happen in parallel. 

    // note: instantiated_math_function returned by instantiate() needs to be explicitly added to recording database/channels created/etc. !!!***
    virtual std::shared_ptr<instantiated_math_function> instantiate(const std::vector<std::shared_ptr<math_parameter>> & parameters,
								    const std::vector<std::shared_ptr<std::string>> & result_channel_paths,
								    std::string channel_path_context,
								    bool is_mutable,
								    bool ondemand,
								    bool mdonly,
								    std::shared_ptr<math_definition> definition,
								    std::set<std::string> execution_tags,
								    std::shared_ptr<math_instance_parameter> extra_params)=0;
								    
    // get_compute_options() returns a list of compute_resource_options, each of which has a compute_code pointer
    // NOTE: Somehow get_compute_options() or similar needs to consider the types of the parameter arrays and select
    // or configure code appropriately.
    //virtual std::shared_ptr<executing_math_function> initiate_execution(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> instantiated)=0; // usually returns a sub-class
  };

  // ***!!! compute_code is obsolete!!!***
  class compute_code {
    // This class represents the information provided by the math_function with the compute_resource_option; basically an offer to compute
    // immutable once published
    
    // BUG: The nature of code to provide may depend upon the resource. i.e. code to run in a C/C++ thread different from code to run
    // in a subprocess, possibly different from code that can run remotely over MPI, etc.
    // In particular anything to be run in a subprocess or over MPI we really need to know how to __find__ the code from the foreign context
    // once the input data has been marshalled.

    // BUG: How/where should locking be done for mutable recordings?
    // BUG: How does allocation process work? For mutable recordings? For immutable recordings?
    
  public:
    compute_code() = default;
    // Rule of 3
    compute_code(const compute_code &) = delete;  // CC and CAO are deleted because we don't anticipate needing them. 
    compute_code& operator=(const compute_code &) = delete; 
    virtual ~compute_code()=default;  // virtual destructor required so we can be subclassed


    // !!!*** Should all of these methods be replaced by one overarching method
    // that then is implemented in subclasses for
    // various common patterns???
    virtual void determine_size(std::shared_ptr<recording_set_state> rss, std::shared_ptr<executing_math_function> fcn, std::shared_ptr<compute_resource_option> option)=0;
    virtual void do_metadata_only(std::shared_ptr<recording_set_state> rss, std::shared_ptr<executing_math_function> fcn, std::shared_ptr<compute_resource_option> option)=0;
    virtual void do_compute_from_metadata(std::shared_ptr<recording_set_state> rss, std::shared_ptr<executing_math_function> fcn, std::shared_ptr<compute_resource_option> option)=0;
    virtual void do_compute(std::shared_ptr<recording_set_state> rss, std::shared_ptr<executing_math_function> fcn, std::shared_ptr<compute_resource_option> option)=0;
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

    virtual ~math_definition()=default;
    virtual std::shared_ptr<math_definition> rebuild(std::shared_ptr<instantiated_math_function> fcn);
    virtual bool operator==(const math_definition &ref); // used in comparisons of instantiated_math_functions
    virtual bool operator!=(const math_definition &ref);
  };
  
  class instantiated_math_function: public std::enable_shared_from_this<instantiated_math_function>  {
    // This structure represents a defined math function. It is immutable
    // once published. 
    // but you may copy it for the purpose of changing it (using clone()) and replace it in the master database.
    // The clone() function should clear the .definition
    // member in the copy and point original_function at the original
    // (if not already defined) with the valid .definition member
  public:
    std::vector<std::shared_ptr<math_parameter>> parameters; // ***!!! MUST BE COMBINED WITH channel_path_context ***!!!
    //std::list<std::shared_ptr<channel>> results; // Note that null entries are legitimate if results are being ignored.
    std::vector<std::shared_ptr<std::string>> result_channel_paths; // Note that null entries are legitimate if results are being ignored. ***!!! MUST BE COMBINED WITH channel_path_context ***!!!
    
    std::vector<bool> result_mutability; // for each result, is it mutable (if we are in mutable mode)
    size_t num_results; // number of results generated by this math function
    
    std::string channel_path_context; // context for parameters and result_channel_paths, if any are relative. 
    bool disabled; // if this math function is temporarily disabled
    bool is_mutable; // should be set if the function is mutable for any reason
    bool self_dependent; // this function is always self-dependent: fcn->new_revision_optional || is_mutable || fcn->self_dependent || fcn->dynamic_dependeny;
    bool ondemand;
    bool mdonly; // Note: This determines whether the instantiation is mdonly. For the execution to be mdonly, the mdonly flag in the math_function_status must be true as well. 
    std::shared_ptr<math_function> fcn;
    std::shared_ptr<math_definition> definition;
    std::set<std::string> execution_tags;
    std::shared_ptr<math_instance_parameter> extra_params;
    
    std::shared_ptr<instantiated_math_function> original_function; // null originally
    // WARNING: If you add more member variables, you also need to check
    // equality of them in operator==() !!!***
    
    // Should point to allocation interface here? ... No. Allocation interface comes from the channelconfig's storage_manager

    instantiated_math_function(const std::vector<std::shared_ptr<math_parameter>> & parameters,
			       const std::vector<std::shared_ptr<std::string>> & result_channel_paths,
			       std::string channel_path_context,
			       bool is_mutable,
			       bool ondemand,
			       bool mdonly,
			       std::shared_ptr<math_function> fcn,
			       std::shared_ptr<math_definition> definition,
			       std::set<std::string> execution_tags,
			       std::shared_ptr<math_instance_parameter> extra_params);

    // Rule of 3
    instantiated_math_function(const instantiated_math_function &) = default;  // CC is present so subclass copy constructor can initialize us more easily
    instantiated_math_function& operator=(const instantiated_math_function &) = delete;
    virtual ~instantiated_math_function()=default;  // virtual destructor required so we can be subclassed
    virtual bool operator==(const instantiated_math_function &ref); // May want to be defined by subclasses... used to compare newly created display functions with prior display functions to see if they need updated. Also to uniquify newly created display functions
    virtual bool operator!=(const instantiated_math_function &ref); // May want to be defined by subclasses... used to compare newly created display functions with prior display functions to see if they need updated.

    //virtual bool check_dependencies(recording_status &recordingstatus, math_status &mathstatus)=0; 
    // virtual clone method -- must be implemented in all subclasses. If .definition is non nullptr and definition_change is set, it clears the copy and points original_function at the old .definition
    virtual std::shared_ptr<instantiated_math_function> clone(bool definition_change=true); // only clone with definition_change=false for enable/disable of the function
    
  };

  class instantiated_math_database {
    // Used to represent currently defined functions. Both in main recording database and then copied into each global revision.
    // In main recording database, locked by recording database admin lock;
    // Immutable once copied into a global revision
  public:
    std::map<std::string,std::shared_ptr<instantiated_math_function>> defined_math_functions; // key is channel path and channel_path_context; note that several keys will point to the same instantiated_math_function. Any changes to any functions require calling rebuild_dependency_map (below)
    
    // Hash table here so that we can look up the math channels that are dependent on an input which may have changed.
    std::unordered_map<std::string,std::unordered_set<std::shared_ptr<instantiated_math_function>>> all_dependencies_of_channel;

    // Hash table here so that we can look up the math functions that are dependent on a given function (within this global revision; does not include ondemand dependencies or implict or explicit self-dependencies 
    std::unordered_map<std::shared_ptr<instantiated_math_function>,std::unordered_set<std::shared_ptr<instantiated_math_function>>> all_dependencies_of_function;
    std::unordered_set<std::shared_ptr<instantiated_math_function>> mdonly_functions; // all functions with mdonly set... updated by rebuild_dependency_map()

    void _rdm_remove_from_definition(std::unordered_map<std::shared_ptr<instantiated_math_function>,std::shared_ptr<instantiated_math_function>> *updated_fcns,size_t result_index,std::shared_ptr<instantiated_math_function> orig_fcn_ptr);
    
    void _rebuild_dependency_map(std::shared_ptr<recdatabase> recdb,bool allow_non_reserved_math_channels); // rebuild all_dependencies_of_channel and all_dependencies_of_function hash tables. Must be called any time any of the defined_math_functions changes. May only be called for the instantiated_math_database within the main recording database, and the main recording database admin lock must be locked when this is called. 

  };


  class math_function_status {
    // locked with math_status by the parent recording_set_state's admin lock
    
    // A math function must be executed in a recording_set_state/globalrev if it is not ondemand and:
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
    std::set<std::shared_ptr<channelconfig>> missing_prerequisites; // all missing (non-ready -- or !!!*** non-mdonly as appropriate) local (in this recording_set_state/globalrev) prerequisites of this function. Remove entries from the set as they become ready. When the set is empty, the math function represented by the key is dispatchable and should be marked as ready_to_execute
    size_t num_modified_prerequisites; // count of the number of prerequisites no longer (or never) listed in missing_prerequisites that have indeed been modified. Used solely to determine whether execution is actually needed. Anybody modifying missing_prerequisites is responsible for updating this. Pre-initialize to 1 to always force execution.  Should not include self-dependencies
    
    // !!!*** Do we need to list out potentially-missing prerequisites based on channels determined by math functions that may or may not execute !!!***???
    // !!!*** A math function that decides not to execute needs to define its outputs as ready, referencing the previous state !!!***
    // !!!*** the execution of functions dependent on a function that decided not to execute may need to be cancelled.  
    
    std::shared_ptr<math_function_execution> execfunc; // tracking of this particular execution. Later globalrevs that will use the result unchanged may also point to this. (Each pointing globalrev should be registered in the execfunc's referencing_rss set so that it can get callbacks on completion). Note that this may be nullptr in various scenarios:  These might be new_revision_optional with changed inputs, dynamic dependencies with unchanged inputs, or functions dependent on those sorts of functions   (!!!*** Note than an unassigned execfunc needs to create an implicit self-dependency)
    
    std::set<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<channelconfig>>> missing_external_channel_prerequisites; // all missing (non-ready...or !!!*** nonmdonly (as appropriate)) external prerequisites of this function. Remove entries from the set as they become ready. Will be used e.g. for dependencies of on-demand recordings calculated in their own rss context
    std::set<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>> missing_external_function_prerequisites; // all missing (non-ready... or !!!*** non-mdonly (as appropriate)) external prerequisites of this function. Remove entries from the set as they become ready. Currently used solely for self-dependencies (which can't be mdonly)... Paired with external_dependencies_on_function

    std::set<std::tuple<std::shared_ptr<channelconfig>,std::shared_ptr<instantiated_math_function>>> extra_internal_dependencies_on_function; // Use this field to indicate that there are additional math functions within this rss that need to be notified when this function is complete. The first element of the tuple identifies the output channel of this math function that the referenced math function is dependent on. Please note that it may be a dynamic (implicit) dependency. 
    
    bool mdonly; // if this execution is actually mdonly. Feeds into execfunc->mdonly

    bool self_dependent; // if there is an implicit or explicit self-dependency
    
    //bool mdonly_executed; // if execution has completed at least through mdonly; Now replaced with execfunc->mdonly_executed
    //bool is_mutable; // if this execution does in-place mutation of one or more of its parameters. Now replaced with execfunc->is_mutable
    //bool execution_in_progress; // set while holding the rss's admin lock right before the executing_math_function is generated. Cleared when the executing_math_function is released. 
    
    bool execution_demanded; // even once all prereqs are satisfied, do we need to actually execute? This is only set if at least one of our non-self dependencies has changed and we are not disabled (or ondemand in a regular globalrev) and !!!*** the execfunc was generated for this revision
    bool ready_to_execute;
    // bool metadataonly_complete; // if we are only executing to metadataonly, this is the complete flag.... move to execfunc->
    bool complete; // set to true once fully executed OR once we decide execution is not demanded; Note that this can shift from true back to false for a formerly metadataonly math function where the full data has been requested

    math_function_status(bool self_dependent);
  };

  class math_status {
    // status of execution of math functions
    // locked by the parent recording_set_state's admin lock. Be warned that
    // you cannot hold the admin locks of two recording_set_states simultaneously. 
  public:
    std::shared_ptr<instantiated_math_database> math_functions; // immutable once copied in on construction
    std::unordered_map<std::shared_ptr<instantiated_math_function>,math_function_status> function_status; // lookup dependency and status info on this instantiated_math_function in our recording_set_state/globalrev. You must hold the recording_set_state's admin lock to mess with the function_status, but the map itself is only modified during construction so it is safe to do lookups on this map and keep pointers to the math_function_status without holding any locks. 

    // NOTE: an entry in either _external_dependencies_on_channel or _external_dependencies_on_function is sufficient to get you the needed callback. 
    std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> _external_dependencies_on_channel; // Lookup external math functions that are dependent on this channel -- usually subsequent revisions of the same function. May result from implicit or explicit self-dependencies. This map is immutable and pointed to by a C++11 atomic shared pointer it is safe to look it up with the external_dependencies() method without holding your recording_set_state's admin lock. 

    std::shared_ptr<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> _external_dependencies_on_function; // Lookup external math functions that are dependent on this math function -- usually subsequent revisions of the same function. May result from implicit or explicit self-dependencies. This map is immutable and pointed to by a C++11 atomic shared pointer it is safe to look it up with the external_dependencies() method without holding your recording_set_state's admin lock. 

    std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::shared_ptr<instantiated_math_function>>>> _extra_internal_dependencies_on_channel; // atomic shared pointer; use accessor and update functions. used for dynamically added dependencies of each function. 

    // for the rest of these, you must own the recording_set_state's admin lock
    std::unordered_set<std::shared_ptr<instantiated_math_function>> pending_functions; // pending functions where goal is full result
    std::unordered_set<std::shared_ptr<instantiated_math_function>> mdonly_pending_functions; // pending functions where goal is metadata only
    std::unordered_set<std::shared_ptr<instantiated_math_function>> completed_functions;  // completed functions with full result
    std::unordered_set<std::shared_ptr<instantiated_math_function>> completed_mdonly_functions; // pending functions where goal is metadata only and metadata is done (note that it is possible for fully ready functions to be in this list in some circumstances, for example if the full result was requested in another globalrev that references the same recording structure. 
    
    std::unordered_map<std::shared_ptr<instantiated_math_function>, std::unordered_map<std::string, std::shared_ptr<math_instance_parameter>>> math_messages; // Messages to dispatch to math functions in this rss only

    
    
    math_status(std::shared_ptr<instantiated_math_database> math_functions,const std::map<std::string,channel_state> & channel_map);

    std::string print_math_status(std::shared_ptr<recording_set_state> rss, bool verbose=false);
    
    std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> begin_atomic_external_dependencies_on_channel_update(); // must be called with recording_set_state's admin lock held
    void end_atomic_external_dependencies_on_channel_update(std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> newextdep);
    std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> external_dependencies_on_channel();
    
    std::shared_ptr<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> begin_atomic_external_dependencies_on_function_update(); // must be called with recording_set_state's admin lock held
    void end_atomic_external_dependencies_on_function_update(std::shared_ptr<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> newextdep);
    std::shared_ptr<std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::weak_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>>>> external_dependencies_on_function(); 

    std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::shared_ptr<instantiated_math_function>>>> begin_atomic_extra_internal_dependencies_on_channel_update(); // must be called with recording_set_state's admin lock held
    void end_atomic_extra_internal_dependencies_on_channel_update(std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::shared_ptr<instantiated_math_function>>>> newextdep);
    std::shared_ptr<std::unordered_map<std::shared_ptr<channelconfig>,std::vector<std::shared_ptr<instantiated_math_function>>>> extra_internal_dependencies_on_channel();

    void notify_math_function_executed(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> fcn,bool mdonly,bool possibly_redundant);
    
    // check_dep_fcn_ready() assumes dep_rss admin lock is already held
    void check_dep_fcn_ready(std::shared_ptr<recdatabase> recdb,
			     std::shared_ptr<recording_set_state> dep_rss,
			     std::shared_ptr<instantiated_math_function> dep_fcn,
			     math_function_status *mathstatus_ptr,
			     std::vector<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<instantiated_math_function>>> &ready_to_execute_appendvec,
			     std::set<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>>,mncn_lessthan> &may_need_completion_notification,
			     std::unique_lock<std::mutex> &dep_rss_admin_holder);

  };

  class math_function_execution {
    // tracks the status of a math function and its execution across multiple rss/globalrevs
    // since this structure is persistent, subclasses must be careful not to maintain shared_ptrs that will
    // keep old data in memory
    
    // lock-free behavior:
    //    * rss should be valid from creation as long as mdonly_executed is not set.
    //    * inst is immutable
    //    * bools are atomic so they can be read safely (but if you read several you may not get a consistent state without locking

    // The 'execution ticket' is the non-false value of .executing and is acquired by calling try_execution_ticket() and having that
    // return true. Once you have the execution ticket you can create a pending computation, assign the compute resource, etc.

  public:
    std::mutex admin; // guards referencing_rss and groups operations on the bools. Also acquire this before clearing rss. Last in the locking order except python GIL

    std::shared_ptr<recording_set_state> rss; // recording set state in which we are executing. Set to nullptr by owner of execution ticket after metadata phase to avoid a reference loop that will keep old recordings in memory.
    std::shared_ptr<std::map<std::string,channel_state>> rss_channel_map; // channel_map from rss; outlives rss pointer
    std::shared_ptr<instantiated_math_function> inst;     // This attribute is immutable once published

    std::set<std::weak_ptr<recording_set_state>,std::owner_less<std::weak_ptr<recording_set_state>>> referencing_rss; // all recording set states that reference this executing_math_function

    std::atomic<bool> executing; // the execution ticket (see try_execution_ticket() method)
    std::atomic<bool> is_mutable; // if this execution does in-place mutation of one or more of its parameters.
    std::atomic<bool> mdonly; // if this execution is actually mdonly
    std::atomic<bool> instantiated; // if recordings are defined
    std::atomic<bool> metadata_executed; // if execution has completed at least through metadata
    std::atomic<bool> metadataonly_complete; // if we are only executing to metadataonly, this is the complete flag.
    std::atomic<bool> fully_complete; // function has fully executed to ready state
    std::shared_ptr<executing_math_function> execution_tracker; // only valid once prerequisites are complete because we can't instantiate it until we have resolved the types of the parameters to identify which code to execute. 

    math_function_execution(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst,bool mdonly,bool is_mutable);  // automatically adds rss to referencing_rss set
    
    inline bool try_execution_ticket()
    // if this returns true, you have the execution ticket. Don't forget to assign executing=false when you are done. 
    {
      if (fully_complete) {
	return false;
      }
      bool expected=false;
      return executing.compare_exchange_strong(expected,true);
    }
    
  };

 
  class executing_math_function {
    // generated to track the execution of a math function
    // each executing_math_function has a single unique math_function_execution. It
    // is not allowed for multiple math_function_executions to point at the same executing_math_function
    
    // executing_math_function is created once all prerequisites are done
    // and usually subclassed

    // lock-free behavior:
    //    * rss should be valid from creation as long as mdonly_executed is not set. Safe to read by holder of parent's execution ticket, including the corresponding execution methods
    //    * inst is immutable
    //    * self_dependent_recordings: safe for owner of parent math_function_execution's 'execution ticket' to read lock-free
    //    * compute_resource: shared_ptr may be assigned/read by the owner of the parent math_function_execution's 'execution ticket'. The pointed data structure is locked by
    //      the assigned compute resource database's admin lock
    //
    
  public:
    std::shared_ptr<recording_set_state> rss; // recording set state in which we are executing. Set to nullptr by owner of parent's execution ticket after metadata phase to avoid a reference loop that will keep old recordings in memory. 
    std::shared_ptr<instantiated_math_function> inst;     // This attribute is immutable once published
    std::shared_ptr<recdatabase> recdb; // to avoid reference loops, this is only to be read by the function code itself, managed by the executing thread (recomath_compute_resource pool_code), and set to nullptr when execution is not actually occuring. 
    std::shared_ptr<lockmanager> lockmgr;

    std::unordered_map<std::string, std::shared_ptr<math_instance_parameter>> msgs;
    

    // should also have parameter values, references, etc. here
    
    // parameter values should include bounding_hyperboxes of the domains that actually matter,
    // at least if the function is optionally_mutable


    // self_dependent_recordings is auto-created by the constructor
    std::vector<std::shared_ptr<recording_base>> self_dependent_recordings; // only valid (size() > 0) with implicit/explict self dependency. entries will be nullptr first time through anyway. Entries may also be nullptr if the function output is being ignored rather than stored in the recording database. ***!!! Must be cleared to nullptr after execution to avoid keeping old recordings alive ***!!!  -- read access OK by the holder of the execution ticket; Otherwise accesses should be protected by the math_function_execution's admin lock


    // compute_resource is assigned post-creation and will be compatible with the selected_compute_option
    // These should be finalized and safe to read from any thread once we are in the define_recs function.
    // Until that time they should be treated as locked with the assigned_compute_resource_database admin lock. 
    std::shared_ptr<assigned_compute_resource> compute_resource; 
    std::shared_ptr<compute_resource_option> selected_compute_option; // This is one of the entries returned by the compute_options step. 
    
    // These next two elements are locked by the parent available_compute_resources_database admin lock
    // THEY HAE BEEN REPLACED BY THE ASSIGNED COMPUTE RESOURCE
    //std::vector<size_t> cpu_cores;  // vector of indices into available_compute_resource_cpu::functions_using_cores representing assigned CPU cores; from assigned_compute_resource_option_cpu and/or assigned_compute_resource_option_opencl
    //std::vector<size_t> opencl_jobs;  // vector of indices into available_compute_resource_cpu::functions_using_devices representing assigned GPU jobs; from assigned_compute_resource_option_opencl


    executing_math_function(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> fcn,std::shared_ptr<lockmanager> lockmgr);

    // Rule of 3
    executing_math_function(const executing_math_function &) = delete;
    executing_math_function& operator=(const executing_math_function &) = delete; 
    virtual ~executing_math_function()=default;  // virtual destructor required so we can be subclasse

    
    virtual bool perform_decide_execution()=0; // perform_decide_execution asks the new_revision_optional executing math function to determine whether it needs to execute, potentially creating new revisions of its output
    virtual std::vector<std::shared_ptr<compute_resource_option>> perform_compute_options()=0; // perform_compute_options asks the executing math function to perform its compute_options step (which should not be compute intensive)
    virtual std::string get_result_channel_path(size_t result_index);
    virtual size_t get_num_result_channels();
    
    virtual void perform_define_recs()=0;
    virtual void perform_metadata()=0;
    virtual void perform_lock_alloc()=0;
    virtual void perform_exec()=0;

  };

  

class pending_math_definition_result_channel {
public:
  //A pending_math_definition_result_channel is returned
  //when you try to iterate over a pending_math_definition.
  //This provides a way to separate out the different
  //result_channels.

  //A mutable _result_channel is returned by the
  //pending_math_definition __iter__() method as an iterator.
  //This iterator copies itself, returning immutable copies.

  //This class has a dual use of being able to describe
  //existing math channels with a python __str__/__repr__
  //method. In this case, the class operates in "existing" mode
  //where it just stores the math definition as a string
  //alongside the channel name.
  

  std::shared_ptr<pending_math_definition> definition;
  size_t result_channel_index;

  bool existing_mode;
  std::string existing_mode_math_definition;
  std::string existing_mode_channel_name;
  
  pending_math_definition_result_channel(std::shared_ptr<pending_math_definition> definition,size_t result_channel_index);
  

  pending_math_definition_result_channel(std::string existing_mode_math_definition,std::string existing_mode_channel_name);
};

  class python_math_definition: public math_definition {
  public:
    std::string function_name; //name of function (will need snde. prefix)
    std::vector<std::string> args; //list of python interpretable string arguments

    python_math_definition(std::string function_name,std::vector<std::string> args);
  

    virtual std::shared_ptr<math_definition> rebuild(std::shared_ptr<instantiated_math_function> fcn);
  
 
    virtual void evaluate(std::shared_ptr<instantiated_math_function> instantiated,std::vector<std::shared_ptr<std::string>> result_channel_paths);

  };

  class pending_math_intermediate_channels {
  public:
    // this class exists because the template below is too complicated for swig to handle. therefore this separate class can be managed with a shared pointer.
    std::vector<std::pair<std::string,std::shared_ptr<pending_math_definition_result_channel>>> intermediate_channels; //list of (channel_name,pending_math_definition or pending_math_definition_result_channel) for any required intermediate channels. Note that this can create a reference loop so it must be cleared when the pending part is no longer necessary.
    void append(std::string channel_name, std::shared_ptr<pending_math_definition_result_channel> result_chan);
  };
  
class pending_math_definition: public python_math_definition {
public:
  //A pending_math_definition is returned when you use
  //the python shorthand for instantiating a math function. This
  //gets stored by the trans.math[] setitem method into the
  //transaction and then the definition is finalized during
  //_realize_transaction()
 
  std::shared_ptr<instantiated_math_function> instantiated; //the (possibly incomplete) instantiated_math_function. Note that this can create a reference loop so it must be cleared when the pending part is no longer necessary.
  std::vector<std::pair<std::string,std::shared_ptr<pending_math_definition_result_channel>>> intermediate_channels; //list of (channel_name,pending_math_definition or pending_math_definition_result_channel) for any required intermediate channels. Note that this can create a reference loop so it must be cleared when the pending part is no longer necessary.

  size_t num_results;

  pending_math_definition(std::string function_name,std::vector<std::string> args,std::shared_ptr<pending_math_intermediate_channels> intermediate_channels,size_t num_results);
 

  // Rule of 3
  pending_math_definition(const pending_math_definition &) = delete;
  pending_math_definition& operator=(const pending_math_definition &) = delete; 
  virtual ~pending_math_definition()=default;  // virtual destructor required so we can be subclassed
  
  virtual void evaluate(std::shared_ptr<instantiated_math_function> instantiated,std::vector<std::shared_ptr<std::string>> result_channel_paths);
  
};


  class math_parameter_mismatch: public snde_error {
  public:
    template<typename ... Args>
    math_parameter_mismatch(std::string fmt, Args && ... args) :
      snde_error(fmt,std::forward<Args>(args) ...)
    {

    }
    virtual ~math_parameter_mismatch() = default;
  };

  class silent_math_parameter_mismatch: public math_parameter_mismatch {
  public:
    // mismatch class that doesn't print the error, to accommodate
    // e.g. null recording parameters without making a big todo. 
    template<typename ... Args>
    silent_math_parameter_mismatch(std::string fmt, Args && ... args) :
      math_parameter_mismatch(fmt,std::forward<Args>(args) ...)
    {

    }
    virtual ~silent_math_parameter_mismatch() = default;
  };

  void annotate_incomplete_deps(std::shared_ptr<recording_set_state> rss, std::shared_ptr<instantiated_math_function> inst, math_function_status *mathstatus_ptr, const std::vector<std::string> &processing_tags, std::shared_ptr<std::set<std::string>> deps);


  // Dynamic dependencies have a find_additional_deps() method
  // that is needed to identify implicit dependencies.
  // This function is designed to fill that role for graphics
  // dependencies. Wrap this with a lambda that knows your
  // processing tags and assign it when the math_function is
  // created.
  //
  // Note that this only considers recordings that are not ready
  // as if the recording is ready any dependence is already
  // satisfied. 
  bool traverse_scenegraph_find_graphics_deps(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst,math_function_status *mathstatus_ptr,const std::vector<std::string> &processing_tags,const std::set<std::string> &filter_channels);


  // traverse_scenegraph_find_graphics_deps() searches all explicit
  // dependencies (parameters of the math function) for referenced graphics.
  // traverse_scenegraph_find_graphics_deps_under_channel() searches
  // only within a specified channel
  
  bool traverse_scenegraph_find_graphics_deps_under_channel(std::shared_ptr<recording_set_state> rss,std::shared_ptr<instantiated_math_function> inst,math_function_status *mathstatus_ptr,const std::vector<std::string> &processing_tags,const std::set<std::string> &filter_channels,std::string channel_path);
  
  //class registered_math_function {
  //public:
  //  std::string name; // Suggest python-style package.module.function, etc.
  //  std::function<std::shared_ptr<math_function>(/*std::shared_ptr<recdatabase> recdb*/)> builder_function;
  //  
  //  registered_math_function(std::string name,std::function<std::shared_ptr<math_function>(/*std::shared_ptr<recdatabase> recdb*/)> builder_function);
  //  virtual ~registered_math_function()=default; // may be subclassed
  //  
  //};
  
  typedef std::unordered_map<std::string,std::shared_ptr<math_function>> math_function_registry_map;
  
  std::shared_ptr<math_function_registry_map> math_function_registry();
 
  // registered name is usually a python-style package/module path
  // register_math_function() returns a value so it can be used an an initializer
  int register_math_function(std::shared_ptr<math_function> fcn);
  

  // Idea: we could make a function to run math manually.
  // But we would first have to create the ability to define
  // untethered recordings outside of a transaction...
  // e.g.
  // std::shared_ptr<recording_set_state> execute_math(std::vector<std::shared_ptr<instantiated_math_function>> math_functions,std::vector<std::shared_ptr<recording_base>>);
  
};



#endif // SNDE_RECMATH_HPP
