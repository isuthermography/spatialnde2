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
    
    instantiated_math_function() = default;
    // Rule of 3
    instantiated_math_function(const instantiated_math_function &) = default;  // CC and CAO are deleted because we don't anticipate needing them. 
    instantiated_math_function& operator=(const instantiated_math_function &) = default;
    virtual ~instantiated_math_function()=default;  // virtual destructor required so we can be subclassed

    // virtual clone method -- must be implemented in all subclasses. If .definition is non nullptr, it clears the copy and points original_function at the old .definition
    virtual std::shared_ptr<instantiated_math_function> clone(instantiated_math_function &orig);
    
    std::list<math_parameter> parameters; 
    //std::list<std::shared_ptr<channel>> results; // Note that null entries are legitimate if results are being ignored.
    std::string channel_path_context; // context for parameters and result_channel_paths, if any are relative. 
    std::list<std::shared_ptr<std::string>> result_channel_paths; // Note that null entries are legitimate if results are being ignored.
    bool disabled; // if this math function is temporarily disabled
    std::shared_ptr<math_function> fcn;
    std::shared_ptr<math_definition> definition;
    
    std::shared_ptr<instantiated_math_function> original_function; // null originally 
    // Should point to allocation interface here? ... No. Allocation interface comes from the channelconfig's storage_manager
  };

  class instantiated_math_database {
    // Used to represent currently defined functions. Both in main waveform database and then copied into each global revision.
    // In main waveform database, locked by waveform database admin lock;
    // Immutable once copied into a global revision
    
    std::map<std::string,std::shared_ptr<instantiated_math_function>> defined_math_functions; // key is channel path and channel_path_context; note that several keys will point to the same instantiated_math_function
    
    // !!!*** Need hash table here so that we can look up the math functions that are dependent on an input which may have changed.
    // !!!*** Need a structure (not here, but in with the global revision) for tracking progress -- but also need to accommodate
    // ondemand waveforms that might be not calculated or calculated multiple times within a globalrev. 

    // Initialize the given globalrev during end_transaction().
    // WARNING: Called with the wfmdb mutex already locked (!)
    // Goes though the defined_math_functions and sorts out which functions have no input changes,
    // and therefore their outputs can be propagated directly into the new globalrev,
    // which functions have input changes and all prerequisites are ready,
    // and which functions may or may not have input changes because not
    // all prerequisites are ready. It will need to store things
    // in with the globalrevision structures, which is OK, and interface with
    // the callbacks we get when calculations are complete. 
    void initialize_globalrev(std::shared_ptr<globalrevision> globalrev);
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
