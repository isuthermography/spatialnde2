#include "wfmmath_compute_resource.hpp"
#include "wfmmath_parameter.hpp"

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
  
  class math_function { // a math function that is defined accessable so it can be instantiated
    // Immutable once published; that said it may be replaced in the database due to a reloading operation. 


    math_function(size_t num_results,const std::list<std::tuple<std::string,unsigned>> &param_names_types);

    // Rule of 3
    math_function(const math_function &) = default;
    math_function& operator=(const math_function &) = default; 
    virtual ~math_function()=default;  // virtual destructor required so we can be subclassed

    
    // Should we put the name (of the function, not the channel) here???
    size_t num_results;
    std::list<std::tuple<std::string,unsigned>> param_names_types; // list of (name,type) tuples

    bool new_revision_optional; // set if the function sometimes chooses not to create a new revision. Causes an implicit self-dependency, because we have to wait for the prior revision to finish to find out if that version was actually different. 
    bool pure_optionally_mutable; // set if the function is "pure" and can optionally operate on its previous output, only rewriting the modified area according to bounding_hyperboxes. If optionally_mutable is taken advantage of, there is an implicit self-dependency on the prior-revision
    bool mandatory_mutable; // set if the function by design mutates its previous output. Creates an implicit self-dependency.
    bool self_dependent; // set if the function by design is dependent on the prior revision of its previous output
    
    // WARNING: If there is no implict or explicit self-dependency multiple computations for the same math function
    // but different versions can happen in parallel. 
    
    // get_compute_options() returns a list of compute_resource_options, each of which has a compute_code pointer
    // NOTE: Somehow get_compute_options() or similar needs to consider the types of the parameter arrays and select
    // or configure code appropriately. 
    virtual std::list<std::shared_ptr<compute_resource_option>> get_compute_options(std::shared_ptr<executing_math_function> fcn)=0;
  };

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

    
    virtual determine_size(std::shared_ptr<assigned_compute_resource_option> compute_resource,executing_math_function fcn)=0;
    virtual do_metadata_only(std::shared_ptr<assigned_compute_resource_option> compute_resource,executing_math_function fcn)=0;
    virtual do_compute(std::shared_ptr<assigned_compute_resource_option> compute_resource,executing_math_function fcn)=0;
  };
  

  class math_function_database {
    // represents the full set of available functions
    // _functions can be updated atomically using the admin lock. 
    std::mutex admin; // last lock in order except for Python GIL
    std::shared_ptr<std::map<std::string,math_function>> _functions; // Actually C++11 atomic shared pointer to immutable map
    
  };

  class math_definition {
    // Represents the way an instantiated_math_function was defined
    // immutable once published. Needed for saving settings. 
  public:
    std::string definition_command;
  }
  
  class instantiated_math_function: public std::enable_shared_from_this<instantiated_math_function>  {
    // This structure represents a defined math function. It is immutable
    // once published. 
    // but you may copy it for the purpose of changing it (using clone()) and replace it in the master database.
    // The clone() function should clear the .definition
    // member in the copy and point original_function at the original
    // (if not already defined) with the valid .definition member

    std::list<math_parameter> parameters; 
    //std::list<std::shared_ptr<channel>> results; // Note that null entries are legitimate if results are being ignored.
    std::string channel_path_context; // context for parameters and result_channel_paths, if any are relative. 
    std::list<std::shared_ptr<std::string>> result_channel_paths; // Note that null entries are legitimate if results are being ignored.
    bool disabled; // if this math function is temporarily disabled
    bool ondemand;
    bool mdonly; 
    std::shared_ptr<math_function> fcn;
    std::shared_ptr<math_definition> definition;
    
    std::shared_ptr<instantiated_math_function> original_function; // null originally 
    // Should point to allocation interface here? ... No. Allocation interface comes from the channelconfig's storage_manager

    instantiated_math_function();
    // Rule of 3
    instantiated_math_function(const instantiated_math_function &) = default;  // CC and CAO are deleted because we don't anticipate needing them. 
    instantiated_math_function& operator=(const instantiated_math_function &) = default;
    virtual ~instantiated_math_function()=default;  // virtual destructor required so we can be subclassed


    virtual bool check_dependencies(waveform_status &waveformstatus, math_status &mathstatu); 
    // virtual clone method -- must be implemented in all subclasses. If .definition is non nullptr, it clears the copy and points original_function at the old .definition
    virtual std::shared_ptr<instantiated_math_function> clone(instantiated_math_function &orig);
    
  };

  class instantiated_math_database {
    // Used to represent currently defined functions. Both in main waveform database and then copied into each global revision.
    // In main waveform database, locked by waveform database admin lock;
    // Immutable once copied into a global revision
    
    std::map<std::string,std::shared_ptr<instantiated_math_function>> defined_math_functions; // key is channel path and channel_path_context; note that several keys will point to the same instantiated_math_function. Any changes to any functions require calling rebuild_dependency_map (below)
    
    // Hash table here so that we can look up the math channels that are dependent on an input which may have changed.
    std::unordered_map<std::string,std::set<std::shared_ptr<instantiated_math_function>>> all_dependencies_of_channel;

    // Hash table here so that we can look up the math functions that are dependent on a given function (within this global revision; does not include ondemand dependencies or implict or explicit self-dependencies 
    std::unordered_map<std::shared_ptr<instantiated_math_function>,std::set<std::shared_ptr<instantiated_math_function>>> all_dependencies_of_function;


    void rebuild_dependency_map(); // rebuild all_dependencies_of_channel and all_dependencies_of_function hash tables. Must be called any time any of the defined_math_functions changes. May only be called for the instantiated_math_database within the main waveform database, and the main waveform database admin lock must be locked when this is called. 

  };


  class math_function_status {
    // Should this next map be replaced by a map of general purpose notifies that trigger when all missing prerequisites are satisified? (probably not but we still need the notification functionality)
    std::set<std::shared_ptr<instantiated_math_function>> missing_prerequisites; // all missing (non-ready) local (in this waveform_set_state/globalrev) prerequisites of this function. Remove entries from the set as they become ready. When the set is empty, the math function represented by the key is dispatchable.
    
    std::set<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>> missing_external_prerequisites; // all missing (non-ready) external prerequisites of this function. Remove entries from the set as they become ready

  };

  class math_status {
    // status of execution of math functions
    // locked by the parent waveform_set_state's admin lock. Be warned that
    // you cannot hold the admin locks of two waveform_set_states simultaneously. 
  public:
    std::shared_ptr<instantiated_math_database> math_functions; // immutable once copied in on construction
    std::unordered_map<std::shared_ptr<instantiated_math_function>,math_function_status> function_status; // lookup dependency and status info on this instantiated_math_function in our waveform_set_state/globalrev. You must hold the waveform_set_state's admin lock. 
    
    std::unordered_map<std::shared_ptr<instantiated_math_function>,std::vector<std::tuple<std::shared_ptr<waveform_set_state>,std::shared_ptr<instantiated_math_function>>>> _external_dependencies; // Lookup external math functions that are dependent on this math function -- usually subsequent revisions of the same function. May result from implicit or explicit self-dependencies. This map is immutable and pointed to by a C++11 atomic shared pointer it is safe to look it up with the external_dependencies() method without holding your waveform_set_state's admin lock. 


    // for the rest of these, you must own the waveform_set_state's admin lock
    std::unordered_set<std::shared_ptr<instantiated_math_function>> pending_functions; // pending functions where goal is full result
    std::unordered_set<std::shared_ptr<instantiated_math_function>> mdonly_pending_functions; // pending functions where goal is metadata only
    std::unordered_set<std::shared_ptr<instantiated_math_function>> completed_functions;  // completed functions with full result
    std::unordered_set<std::shared_ptr<instantiated_math_function>> completed_mdonly_functions; // pending functions where goal is metadata only and metadata is done
    
    
    math_status(std::shared_ptr<instantiated_math_database> math_functions);
  };

  
  class executing_math_function {
    std::shared_ptr<instantiated_math_function> fcn;     // This attribute is immutable once published
    
    // should also have parameter values, references, etc. here
    
    // parameter values should include bounding_hyperboxes of the domains that actually matter,
    // at least if the function is optionally_mutable

    // These next two elements are locked by the parent available_compute_resources_database admin lock
    std::vector<size_t> cpu_cores;  // vector of indices into available_compute_resource_cpu::functions_using_cores representing assigned CPU cores; from assigned_compute_resource_option_cpu and/or assigned_compute_resource_option_opencl
    std::vector<size_t> opencl_jobs;  // vector of indices into available_compute_resource_cpu::functions_using_devices representing assigned GPU jobs; from assigned_compute_resource_option_opencl
  };
}
