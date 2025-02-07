%shared_ptr(snde::compute_resource_option)
snde_rawaccessible(snde::compute_resource_option);
%shared_ptr(snde::compute_resource_option_cpu)
snde_rawaccessible(snde::compute_resource_option_cpu);
%shared_ptr(snde::available_compute_resource_database)
snde_rawaccessible(snde::available_compute_resource_database);
  %shared_ptr(snde::pending_computation)
snde_rawaccessible(snde::pending_computation);
  %shared_ptr(snde::available_compute_resource)
snde_rawaccessible(snde::available_compute_resource);
  %shared_ptr(snde::available_compute_resource_cpu)
snde_rawaccessible(snde::available_compute_resource_cpu);
  %shared_ptr(snde::assigned_compute_resource)
snde_rawaccessible(snde::assigned_compute_resource);
  %shared_ptr(snde::assigned_compute_resource_cpu)
snde_rawaccessible(snde::assigned_compute_resource_cpu);


%template(snde_available_compute_resource_list) std::list<std::shared_ptr<snde::available_compute_resource>>;


%{
  #include "recmath_compute_resource.hpp"  
%}


namespace snde {

  
  
#define SNDE_CR_PRIORITY_REDUCTION_LIMIT 8 // number of priority reduction levels within a globalrev (priority reductions 0..(SNDE_CR_PRIORITY_REDUCTION_LIMIT-1)
  

#define SNDE_CR_CPU 0
#define SNDE_CR_OPENCL 1
#define SNDE_CR_CUDA 2

  // forward references 
  class compute_code; // defined in recstore.hpp
  class recording_set_state; // defined in recstore.hpp
  class recdatabase; // defined in recstore.hpp
  class recording_base; // defined in recstore.hpp
  class channel_state; // defined in recstore.hpp
  class instantiated_math_function; // defined in recmath.hpp
  class executing_math_function; // defined in recmath.hpp
  class math_function_execution; // defined in recmath.hpp
  
  class available_compute_resource;
  class assigned_compute_resource_cpu;
  class pending_computation;
  
  class compute_resource_option {
    // A list of shared_ptrs to these are returned from the executing math_function's perform_compute_options() method
    // they are immutable once published
  public:
    std::set<std::string> execution_tags;
    
    compute_resource_option(unsigned type,std::set<std::string> execution_tags, size_t metadata_bytes,size_t data_bytes); // metadata_bytes and data_bytes are transfer requirement estimate -- intended to decide about transfer to a cluster node or similar

    // Rule of 3
    compute_resource_option(const compute_resource_option &) = delete;  // CC and CAO are deleted because we don't anticipate needing them. 
    compute_resource_option& operator=(const compute_resource_option &) = delete; 
    virtual ~compute_resource_option()=default;  // virtual destructor required so we can be subclassed

    virtual bool compatible_with(std::shared_ptr<available_compute_resource> available)=0;
    
    unsigned type; // SNDE_CR_...

    // transfer requirement estimate -- intended to decide about transfer to a cluster node or similar
    size_t metadata_bytes;
    size_t data_bytes;
  };

  class compute_resource_option_cpu: public compute_resource_option {
  public:
    compute_resource_option_cpu(std::set<std::string> execution_tags,
				size_t metadata_bytes,
				size_t data_bytes,
				snde_float64 flops,
				size_t max_effective_cpu_cores,
				size_t useful_cpu_cores);
    snde_float64 flops;  // not currently used
    size_t max_effective_cpu_cores;  // not currently used
    size_t useful_cpu_cores; // recommended number of cpu cores to use
  };
  
  
  class available_compute_resource_database {
    // Represents the full set of compute resources available
    // for recording math calculations
  public:
    //std::shared_ptr<std::mutex> admin; // locks compute_resources and todo_list, including all pending_computations but not their embedded executing_math_function; After the recdatabase; in the locking order precedes just Python GIL
    std::multimap<int,std::shared_ptr<available_compute_resource>> compute_resources; // map key is priority

    // everything in todo_list is queued in with one or more of the compute_resources
    std::unordered_set<std::shared_ptr<pending_computation>> todo_list;  // Must be very careful with enclosing scope of references to pending_computation elements. They must be removed from todo_list and drop off the available_compute_resource prioritized_computations multimaps by expiration of their weak_ptrs atomically.  But in any case the enclosing scope for the extracted shared_ptr must terminate before the lock is released, so you must always release the shared pointer prior to the acrdb admin lock. 

    std::multimap<uint64_t,std::shared_ptr<pending_computation>> blocked_list; // indexed by global revision; map of pending computations that have been blocked from todo_list because we are waiting for all mutable calcs in prior revision to complete.

    //std::thread dispatcher_thread; // started by start()
	
    //std::condition_variable computations_added_or_completed; // associated mutex is the admin lock of the available_compute_resource_database



    available_compute_resource_database();

    void add_resource(std::shared_ptr<available_compute_resource> new_resource);

    void queue_computation(std::shared_ptr<recdatabase> recdb,std::shared_ptr<recording_set_state> ready_rss,std::shared_ptr<instantiated_math_function> ready_fcn);
    //void _queue_computation_internal(std::shared_ptr<pending_computation> &computation); // NOTE: Sets computation to nullptr once queued
    void start(); // start all of the compute_resources
    void dispatch_code();

    void notify_acrd_of_changes_to_prioritized_computations(); // should be called WITH ACRD's admin lock held

  };

  class pending_computation {
    // Once the result recordings are ready we need to copy this and get it off the todo_list,
    // so as to invalidate other entries on the prioritized_computations
    // of other available_compute_resources
    // locked by the available_compute_resource_database's admin lock but the pointed to executing_math_function is locked separately
  public:
    std::shared_ptr<math_function_execution> function_to_execute; 
    std::shared_ptr<recording_set_state> recstate;
    uint64_t globalrev;
    uint64_t priority_reduction; // 0..(SNDE_CR_PRIORITY_REDUCTION_LIMIT-1)

    pending_computation(std::shared_ptr<math_function_execution> function_to_execute,std::shared_ptr<recording_set_state> recstate,uint64_t globalrev,uint64_t priority_reduction);
    
  };
  
  class available_compute_resource /*: public std::enable_shared_from_this<available_compute_resource>*/ {
    // Locked by the available_compute_resource_database's admin lock
  public:

    std::set<std::string> tags; // a list of execution tags that will preferentially be handled by this compute resource
    
    available_compute_resource(std::shared_ptr<recdatabase> recdb,unsigned type,std::set<std::string> tags);
    // Rule of 3
    available_compute_resource(const available_compute_resource &) = delete;
    available_compute_resource& operator=(const available_compute_resource &) = delete; 
    virtual ~available_compute_resource()=default;  // virtual destructor required so we can be subclassed

    virtual void start()=0; // set the compute resource going
    
    unsigned type; // SNDE_CR_...

    //std::shared_ptr<std::mutex> acrd_admin; // so we can access the ACRD's admin lock.
    std::weak_ptr<recdatabase> recdb;
    
    std::multimap<uint64_t,std::tuple<std::weak_ptr<pending_computation>,std::shared_ptr<compute_resource_option>>> prioritized_computations;  // indexed by (global revision)*8, with priority reductions 0..(SNDE_CR_PRIORITY_REDUCTION_LIMIT-1) added. So  lowest number means highest priority. The values are weak_ptrs, so the same pending_computation can be assigned to multiple compute_resources. When the pending_computation is dispatched the strong shared_ptr to it must be cleared atomically with the dispatch so it appears null in any other maps. As we go to compute, we will just keep popping off the first element

    //virtual void notify_acr_of_changes_to_prioritized_computations()=0; // should be called WITH ACRD's admin lock held
    //virtual std::vector<std::shared_ptr<executing_math_function>> currently_executing_functions()=0; // Subclass extract functions that are actually executing right now. 
  };

  // Defines for the dispatch priority of a compute resource.
  // Smaller or more negative numbers are higher priority.
#define SNDE_ACRP_GPU_SPECIALIZEDAPI 0 // specialized API such as HIP/CUDA. If we have an implementation in such an API and a compute resource that can handle it, then we almost certainly want to use it. 
#define SNDE_ACRP_GPU_GENERALAPI 1 // Portable API such as OpenCL/SYCL
#define SNDE_ACRP_CPU 2 // CPU priority is always lower (higher number) so that GPU implementations take priority if there is a suitable GPU available and not contended for 
#define SNDE_ACRP_CPU_AS_GPU 5 // Very lowest priority is CPU fallback for GPU code 

  class available_compute_resource_cpu: public available_compute_resource {
  public:

    // NOTE: We should probably implement core affinity here and limit execution to certain sets of cores. 
    size_t total_cpu_cores_available;
    std::vector<std::shared_ptr<math_function_execution>> functions_using_cores; // contains core assignments, total length should match total_cpu_cores_available.

    //std::condition_variable computations_added_or_completed; // associated mutex is the admin lock of the available_compute_resource_database
    //std::thread dispatch_thread;

    // each pool thread should either be in available_threads or assigned_threads at any given time
    //std::vector<std::thread> available_threads; // mapped according to functions_using_cores
    //std::vector<std::shared_ptr<std::condition_variable>> thread_triggers; // mapped according to functions_using_cores  (use shared_ptrs because condition_variable is not MoveConstructable)
    std::vector<std::tuple<std::shared_ptr<recording_set_state>,std::shared_ptr<math_function_execution>,std::shared_ptr<assigned_compute_resource_cpu>>> thread_actions; // mapped according to first thread assigned to a particular math function
    //std::thread dispatcher_thread; // started by constructor
    
    available_compute_resource_cpu(std::shared_ptr<recdatabase> recdb,std::set<std::string> tags,size_t total_cpu_cores_available);


    virtual void start(); // set the compute resource going
    //virtual bool dispatch_code(std::unique_lock<std::mutex> &acrd_admin_lock);
    void pool_code(size_t threadidx);
  };


  // The assigned_compute_resource structures
  // are passed to the recmath class compute_code virtual methods to
  // tell them the resources they are allowed to use. The subclass
  // provided will always match the subclass of the compute_resource_option
  // corresponding to the compute_code structure being called. 
  class assigned_compute_resource {
  public:
    assigned_compute_resource(unsigned type,std::shared_ptr<available_compute_resource> resource);
    // Rule of 3
    assigned_compute_resource(const assigned_compute_resource &) = delete;
    assigned_compute_resource& operator=(const assigned_compute_resource &) = delete; 
    virtual ~assigned_compute_resource()=default;  // virtual destructor required so we can be subclassed
    virtual void release_assigned_resources(std::unique_lock<std::mutex> &acrd_admin_holder)=0; // resources referenced below no longer meaningful once this is called. Must be called with acrd admin lock locked

    unsigned type; // SNDE_CR_...
    std::shared_ptr<available_compute_resource> resource; 
  };
    
  class assigned_compute_resource_cpu : public assigned_compute_resource {
  public:
    assigned_compute_resource_cpu(std::shared_ptr<available_compute_resource> resource,const std::vector<size_t> &assigned_cpu_core_indices);
    
    std::vector<size_t> assigned_cpu_core_indices;
    //size_t number_of_cpu_cores;
    
    virtual void release_assigned_resources(std::unique_lock<std::mutex> &acrd_admin_holder); // resources referenced below no longer meaningful once this is called. Must be called with acrd admin lock locked
    
  };




  
};

 
