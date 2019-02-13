/* transactional revision manager */
/* ***!!!!  SHOULD REDESIGN WITH PROPER DATABASE INDEXED BY 
   WHAT THE FUNCTION IS DEPENDENT ON

   ***!!!! Need some way to declare dependence on waveform metadata ***!!!


   ***!!! Need some way to auto-lock output at the end of transaction ***!!!   

   ***!!! SHOULD IMPLEMENT SPECIFYING ON-DEMAND ARRAYS so that
   functions which only generate stuff in on-demand arrays get 
   postponed until output is needed 

   *** Should implement lock "keys" whereby the revision manager acquires the keys, 
   and then only lockers with access to the keys are are allowed to lock the arrays. 
   Others have to wait...   ... acts as a prior lock in the locking order... 
*/

#include <unordered_set>
#include <set>
#include <atomic>
#include "gsl-lite.hpp"
#include "arraymanager.hpp"

#include "mutablewfmstore.hpp" // for wfmdirty_notification_receiver and class mutableinfostore

#ifndef SNDE_REVISION_MANAGER_HPP
#define SNDE_REVISION_MANAGER_HPP


namespace snde {
  /* The transactional revision manager organizes changes to a set of 
     arrays into discrete versions of the entire set. 

     Changes nominally happen instantaneously, although some looser access is permitted.

     Specifically, it is possible to hold read locks to an old version even after 
     a new version has been defined, so long as the changes have not (yet) affected
     what those read locks are protecting.


     All changes to the associated arrays SHOULD be managed through this manager.

     Dependencies can be registered with the TRM so that outputs can be automatically
     recalculated from inputs. 


     TRM can manage arrays that cross the boundary between arraymanagers, but
     you still need to follow the locking order, according to array creation, 
     and with a single lockmanager. 

     ***!!! Should be modified to support functions that are only executed on-demand...
i.e. a transaction that does nothing but mark demands on some outputs.... Those outputs
are otherwise never generated, even if their input changes ***!!!
  */

  // forward declaration
  class trm;

  
  class trm_arrayregion {
  public:
    std::shared_ptr<arraymanager> manager;
    void **array;
    snde_index start;
    snde_index len;

    trm_arrayregion()
    {
      array=NULL;
      start=SNDE_INDEX_INVALID;
      len=0;
    }
    
    trm_arrayregion(std::shared_ptr<arraymanager> manager,
		    void **array,
		    snde_index start,
		    snde_index len) : manager(manager),
				      array(array),
				      start(start),
				      len(len)
    {
      
    }
						   

    
    bool operator==(const trm_arrayregion &rhs) const
    {
      return (array==rhs.array) && (manager==rhs.manager) && (start==rhs.start) && (len==rhs.len);      
    }

    bool overlaps(const trm_arrayregion &other)
    {
      if (array != other.array) return false;

      assert(manager==other.manager); // Same array should be managed by a consistent manager

      // if other ends before we start, no overlap
      if ((other.start+other.len) <= start) return false;

      // if we end before other starts, no overlap
      if ((start+len) < other.start) return false;

      // otherwise overlap
      return true;
    }
  };


  // singleton class is a marker for extract_regions, to indicate that only a single element is expected, and to extract a reference rather than a gsl::span...
    template <typename T>
    class singleton  { // not sure if the content here is even necessary... but it can't hurt.
      typedef T wrapped;
      T content;
    public:
      singleton(T content) {
	this->content=content;
      }
    };

  // define is_singleton<type> to check whether type is from our singleton class
  template<class T> struct is_singleton_helper: std::false_type {};
  template<class T> struct is_singleton_helper<singleton<T>>: std::true_type {};
  template<class T> struct is_singleton: is_singleton_helper<typename std::remove_cv<T>::type> {};


    // indexrange class is a marker for extract_regions, to indicate that only a range is expected, and to extract that range rather than a gsl::span...
    struct indexrange  { };
      
    // rawregion class is a marker for extract_regions, to indicate that a trm_arrayregion is expected, and to extract that trm_arrayregion rather than a gsl::span...
    struct rawregion  { };

      
    // extract_region for an array (gsl::span)
    // Wrap in an extra struct because C++ doesn't support partial specialization of bare functions
    template <typename T>
    struct extract_region_impl_wrap {
      static gsl::span<T> extract_region_impl(trm_arrayregion &region)
      {
	T *pointer = *((T **)region.array);
	
	return gsl::span<T>(pointer+region.start,region.len);
	
      }
    };
    // extract_region specialization for a marked singleton (simple reference) 
    template <typename T>
    struct extract_region_impl_wrap<singleton<T>> {
      static T& extract_region_impl(trm_arrayregion &region)
      {
	T *pointer = *((T **)region.array);
	assert(region.len==1); // if this trips, then a marked singleton corresponds to an arrayregion with size != 1
	return *(pointer+region.start); 
	
      }
    };

    // extract_region specialization for indexrange (snde_indexrange)
    template <>
    struct extract_region_impl_wrap<indexrange> {
      static snde_indexrange extract_region_impl(trm_arrayregion &region)
      // Note this returns a value, not a reference! 
      {
	snde_indexrange range;
	range.start = region.start;
	range.len=region.len;
	return range; 
	
      }
    };

      // extract_region specialization for rawregion (trm_arrayregion)
    template <>
    struct extract_region_impl_wrap<rawregion> {
      static trm_arrayregion & extract_region_impl(trm_arrayregion &region)
      // note that this returns a reference to the region!
      {
	return region; 
      }
    };

  
  // primary template
  template <typename... T>
  struct extract_regions_impl_wrap;

  // void specialization
  template <>
  struct extract_regions_impl_wrap<> {
    static std::tuple<> extract_regions_impl(std::vector<trm_arrayregion> &regions, size_t firstregion)
    {
      assert(firstregion==regions.size()); // If this assertion fails, it means the parent extract_regions() was given not enough types for
      // the size of the vector 
      return std::make_tuple();
    }
  };

  // recursive specialization
  //!!!*** this requires c++14 to deduce the return type... could probably write a
  // trailing return type somehow (!!!???)
  template <typename T,typename... types>
  struct extract_regions_impl_wrap<T,types...> {
    static auto extract_regions_impl(std::vector<trm_arrayregion> &regions, size_t firstregion)
    {
      assert(firstregion < regions.size()); // If this assertion fails, it means the parent extract_regions() was given too many types for
	// the size of the vector 
	trm_arrayregion &region = regions[firstregion];
	
	auto this_element = extract_region_impl_wrap<T>::extract_region_impl(regions[firstregion]);
	
	return std::tuple_cat(std::make_tuple(this_element),extract_regions_impl_wrap<types...>::extract_regions_impl(regions,firstregion+1));
      }
    };
  
    template <typename... types>
    auto extract_regions(std::vector<trm_arrayregion> regions) {
      assert(regions.size()==sizeof...(types));
      return extract_regions_impl_wrap<types...>::extract_regions_impl(regions,0); // std::tuple<types...> std::make_index_sequence(sizeof...(types)));
    }

  /* Ok. Here's how you use extract_regions()... Let's suppose
     you are expecting 3 parameters: a single meshedpart, an  
     array of snde_coord3's, and an array of snde_indexes.
    
     // Declare variables
     snde_meshedpart meshedpart; // singleton
     // Note: We would really rather meshedpart
     // be a reference but there is no good way to do that until C++17
     // See: https://stackoverflow.com/questions/39103792/initializing-multiple-references-with-stdtie
     // .. as is, a singleton such as meshedpart will be read-only, unless
     // manually extracted with std::get<>.
     
     gsl_span<snde_coord3> coords;
     snde_indexrange indexes; 
     trm_arrayregion region;

     std::tie(meshedpart,coords,indexes,region) = extract_regions<singleton<meshedpart>,snde_coord3,indexrange,rawregion>(inputs);

     // Note that this does nothing in terms of locking, which generally must be done separately (and 
     // before calling extract_regions<>() -- at least if you are extracting anything but rawregions)
  */


  
  
  class trm_dependency : public std::enable_shared_from_this<trm_dependency> { /* dependency of one memory region on another */
  public:
    std::weak_ptr<trm> revman;
    std::function<void(snde_index newversion,std::shared_ptr<trm_dependency> dep,std::vector<rangetracker<markedregion>> &inputchangedregions)> function; /* returns updated output region array */
    /* if function is NULL, that means that this is an input, i.e. one of the input arrays that is locked for 
       write and that we will be responding to changes from */
    
    std::function<std::vector<trm_arrayregion>(std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs,std::vector<trm_arrayregion> inputs)> regionupdater; /* to be called when an input changes... returns updated input region array. Should try to return quickly. Arrays can be locked following the locking order, and only for read. */

    std::function<std::vector<trm_arrayregion>(std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs,std::vector<trm_arrayregion> inputs,std::vector<std::shared_ptr<mutableinfostore>> metadata_outputs,std::vector<trm_arrayregion> outputs)> update_output_regions;

    std::function<void(std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs,std::vector<trm_arrayregion> inputs,std::vector<trm_arrayregion> outputs)> cleanup;
    
    std::vector<rangetracker<markedregion>> inputchangedregion; // rangetracker for changed zones, for each input 
    std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs;
    std::vector<trm_arrayregion> inputs;
    std::vector<std::shared_ptr<mutableinfostore>> metadata_outputs;
    std::vector<trm_arrayregion> outputs;

    std::vector<std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>>> input_dependencies; /* vector of input dependencies,  ordered per metadatainput then per input... These are sets because of possible overlap of regions or one output being used by multiple inputs */
    std::vector<std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>>> output_dependencies; /* vector of output dependencies, per metadataoutput then per output  */

    std::weak_ptr<trm_dependency> weak_this; // used in destructor
    bool force_full_rebuild; // used to trigger full rebuild of all outputs for newly created dependency

    /* pending_input_dependencies is only valid during a transaction, and lists
       input dependencies that will be modified by other dependencies */
    //std::vector<std::weak_ptr<trm_dependency>> pending_input_dependencies;

    

    trm_dependency(std::shared_ptr<trm> revman,
		   std::function<void(snde_index newversion,std::shared_ptr<trm_dependency> dep,std::vector<rangetracker<markedregion>> &inputchangedregions)> function,
		   std::function<std::vector<trm_arrayregion>(std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs,std::vector<trm_arrayregion> inputs)> regionupdater,
		   std::function<std::vector<trm_arrayregion>(std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs,std::vector<trm_arrayregion> inputs,std::vector<std::shared_ptr<mutableinfostore>> metadata_outputs,std::vector<trm_arrayregion> outputs)> update_output_regions,
		   std::function<void(std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs,std::vector<trm_arrayregion> inputs,std::vector<trm_arrayregion> outputs)> cleanup,
		   std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs,
		   std::vector<trm_arrayregion> inputs,
		   std::vector<std::shared_ptr<mutableinfostore>> metadata_outputs,
		   std::vector<trm_arrayregion> outputs) :
      revman(revman),
      function(function),
      regionupdater(regionupdater),
      update_output_regions(update_output_regions),
      cleanup(cleanup),
      metadata_inputs(metadata_inputs),
      inputs(inputs),
      metadata_outputs(metadata_outputs),
      outputs(outputs),
      force_full_rebuild(true)
    {
      // weak_this=shared_from_this();
    }
		   
    trm_dependency(const trm_dependency &)=delete; // copy constructor disabled
    trm_dependency& operator=(const trm_dependency &)=delete; // copy assignment disabled
    
    ~trm_dependency(); // destructor in .cpp file to avoid circular class dependency
  };


  /* #defines for trm::state */ 
#define TRMS_IDLE 0
#define TRMS_TRANSACTION 1 /* Between BeginTransaction() and EndTransaction() */
#define TRMS_REGIONUPDATE 2 /* performing region updates inside EndTransaction() */
#define TRMS_DEPENDENCY 3 /* performing dependency updates inside EndTransaction() */

  
  class trm : public std::enable_shared_from_this<trm> { /* transactional revision manager */
    /* General rule: You should not write to any of the 
       managed arrays without doing so as part of a transaction. 
       
       So locking of the managed arrays for write should be done only
       through the transaction process, or when executing a registered
       dependency update. 

       Locking the managed arrays for read should generally be done 
       through trm::lock_arrays_read() (NOT YET IMPLEMENTED), which 
       will always get you 
       a consistent set and which will also minimize the risk of 
       starving the write processes of access. */

  public:
    std::shared_ptr<rwlock> transaction_update_lock; /* Allows only one transaction at a time. Locked BEFORE any read or write locks acquired by a process 
							that will write. Write lock automatically acquired and placed in transaction_update_writelock_holder 
						        during start_transaction().... Not acquired as part of a locking process */
    /* NOTE: We rely on the fact that rwlock's can be unlocked by a different thread when End_Transaction() delegates release of this lock to a thread 
       that waits for all transaction functions to complete */
    
    
    std::unique_lock<rwlock_lockable> transaction_update_writelock_holder; // access to this holder is managed by dependency_table_lock

    std::atomic<size_t> state; /* TRMS_IDLE, TRMS_TRANSACTION, TRMS_REGIONUPDATE, or TRMS_DEPENDENCY */ 

    std::atomic<snde_index> currevision;
    
    std::recursive_mutex dependency_table_lock; /* ordered after transaction_update_lock but before locking of arrays; locks dependencies, contents of dependencies,  and dependency execution tracking variables */
    /* dependency_tabel_lock is a recursive mutex so it can be safely re-locked when a trm_dependency's 
       destructor is called, auto removing the trm_dependency from the various sets */
    std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>> dependencies; /* list of all dependencies */
    
    /* dependency graph edges map inputs to outputs */
    /* we can execute the graph by:
      0. Clear all execution flags; 
      1.  Starting at any unexecuted node. 
      2.  Look for the node's first unexecuted input dependency, 
         2a. If there is such an unexecuted input dependency switch to that node and return to step 2. 
      3. Execute this node; set its execution flag (this step may be parallelized if there are multiple cases)
      4. Executing this node may have changed its output regions. Look through all the output 
         dependencies corresponding to all the output regions that have changed and call their
         regionupdater functions. 
      5. Move to  the first output (if present) and go to step 2. 
      6. If no output go to step 1. 

      NOTE: The dependency graph MAY be changed during the execution, but obviously 
      inputs or outputs of elements that have been executed MUST NOT be changed. 


      Parallel model:
      0. Clear all execution flags
      1. Split into team. Each member: 
         a.  Identify an unexecuted node with no unexecuted input dependencies and atomically mark it as executing
         b.  Acquire the write locks for each of its output arrays, following the correct locking order. Execute the node. Release the output arrays. 
         c.  Executing this node may have changed its output regions. Look through all the 
             output dependencies corresponding to all the output regions that have changed and call their
	     regionupdater functions. 
         d.  Mark the node as complete
         e.  Return to step a. 

    */
    
    /* To execute process: 
        1. lock "transaction update" lock for write
        1a. Increment version 
	2. Run parallel model above
        4a. Release "process update" lock, allowing 
            queued readers to read. Once they are 
            done it will again be possible to lock it
            for write, allowing yet another new version
     */
    /*
      To execute dependency node: 
        1. Look up inputs (vector of arrayregions) 
        2. For each input, figure out from modified_db which 
           subregions have been modified as part of
	   this transaction update
        3. If any have been modified, call the dependency function,
           extract what output regions it actually modified, 
           and store those in the modified db. 
        
     */
    
    /* dependency execution tracking variables (locked by dependency_table_lock) */
    /* During a transactional update, each trm_dependency pointer
       should be in exactly one of these unordered_sets. When 
       The transactional update ends, they should all be moved 
       into unsorted
    */

    /* *** IDEA: Should add category that allows lazy evaluation. Then if we ask to read it, it will 
       trigger the calculation... How does this interact with locking order?  Any ask to read after new
       version is defined must wait for new version. */
    
    std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>> unsorted; /* not yet idenfified into one of the other categories */
    std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>> no_need_to_execute; /* no (initial) need to execute, but may still be dependent on something */
    
    std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>> unexecuted_with_deps; // Once deps are complete, these move into unexecuted_needs_regionupdater
    std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>> unexecuted_needs_regionupdater; // dropped into unexecuted_regionupdated when in TRMS_REGIONUPDATE phase, directly processed either into unexecuted_no_deps (if inputs are fully evaluated), or back into unexecuted_with_deps (if incomplete inputs still present) in TRMS_DEPENDENCY phase
    std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>> unexecuted_regionupdated;  // these deps will be dispatched either into unexecuted_no_deps (if inputs are fully evaluated), or back into unexecuted_with_deps (if incomplete inputs still present) by _figure_out_unexecuted_deps() 
    std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>> unexecuted_no_deps; // these are ready to execute
    std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>> executing_regionupdater;
    std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>> executing;
    std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>> done;

    /* locked by dependency_table_lock; modified_db is a database of which array
       regions have been modified during (or before?) this transaction. Should be
       cleared at end of transaction */
    std::unordered_map<void **,std::pair<std::shared_ptr<arraymanager>,rangetracker<markedregion>>> modified_db;

    /* locked by dependency_table_lock; modified_metadata_db is a database of which mutableinfostore's 
       metadata have been modified during (or before?) this transaction. Should be
       cleared at end of transaction */
    std::set<std::weak_ptr<mutableinfostore>,std::owner_less<std::weak_ptr<mutableinfostore>>> modified_metadata_db;
    
    
    // note: these condition variables are condition_variable_any instead of
    // condition_variable because that is what is required for compatibility with
    // our recursive mutex.
    // ... it's recursive so the destructor can re-lock if necessary. Other
    // recursion will generally NOT be OK because wait() only unlocks it once...
    std::condition_variable_any job_to_do; /* associated with dependency_table_lock  mutex */
    std::condition_variable_any regionupdates_done; /* associated with dependency_table_lock mutex... indicator set by calculation thread when all region updates are complete in state TRMS_REGIONUPDATE */
    std::condition_variable_any jobs_done; /* associated with dependency_table_lock mutex... indicator used by transaction_wait_thread to see when it is time to wrap-up the transaction */
    std::condition_variable_any all_done; /* associated with dependency_table_lock mutex... indicator that the transaction is fully wrapped up and next one can start */
    std::vector<std::thread> threadpool;

    std::thread transaction_wait_thread; /* This thread waits for all computation functions to complete, then releases transaction_update_lock */

    bool threadcleanup; /* guarded by dependency_table_lock */


    /* the change_detection_pseudo_cache is registered as a cachemanager with the various arraymanagers for the input
       dependencies, so we get notified when things are modified, and can update our modified_db, 
       ... we are registered under "our_unique_name" which is "trm_" followed by our pointer address */ 
    std::shared_ptr<cachemanager> change_detection_pseudo_cache;
    std::string our_unique_name;

    // Nested class
    class trm_change_detection_pseudo_cache: public cachemanager {
    public:
      std::weak_ptr<trm> revman;
      
      trm_change_detection_pseudo_cache(std::shared_ptr<trm> revman) :
	revman(revman)
      {
	
      }
      virtual void mark_as_dirty(std::shared_ptr<arraymanager> manager,void **arrayptr,snde_index pos,snde_index numelem)
      {
	// Warning: Various arrays may be locked when this is called!
	std::shared_ptr<trm> revman_strong(revman);
	std::lock_guard<std::recursive_mutex> dep_tbl(revman_strong->dependency_table_lock);
	
	revman_strong->_mark_region_as_modified(trm_arrayregion(manager,arrayptr,pos,numelem));
      }
      
      virtual ~trm_change_detection_pseudo_cache() {};
      
    };

    // another nested class:
    class trm_wfmdirty_notification: public wfmdirty_notification_receiver {
    public:
      trm *revman;
      
      trm_wfmdirty_notification(trm *revman) :
	wfmdirty_notification_receiver(),
	revman(revman)
      {
	
      }

      virtual void mark_as_dirty(std::shared_ptr<mutableinfostore> infostore)
      {
	// Warning: Various arrays may be locked when this is called!
	//std::shared_ptr<trm> revman_strong(revman);
	std::lock_guard<std::recursive_mutex> dep_tbl(revman->dependency_table_lock);
	
	revman->_mark_infostore_as_modified(infostore);

      }
      
      virtual ~trm_wfmdirty_notification() {};
    };

    std::shared_ptr<trm_wfmdirty_notification> wfmdb_notifier;
    
    
    trm(const trm &)=delete; /* copy constructor disabled */
    trm& operator=(const trm &)=delete; /* copy assignment disabled */

    
    trm(std::shared_ptr<mutablewfmdb> wfmdb, int num_threads=-1)
    {
      currevision=1;
      threadcleanup=false;
      state=TRMS_IDLE;
      transaction_update_lock=std::make_shared<rwlock>();

      wfmdb_notifier = std::make_shared<trm_wfmdirty_notification>(this);
      wfmdb->add_dirty_notification_receiver(wfmdb_notifier);
      our_unique_name="trm_" + std::to_string((unsigned long long)this);

      // Assigment of change_detection_pseudo_cache moved to first
      // use because we are not allowed to use shared_from_this() in
      // a constructor...
      //change_detection_pseudo_cache = std::make_shared<trm_change_detection_pseudo_cache>(shared_from_this());

      if (num_threads==-1) {
	num_threads=std::thread::hardware_concurrency();
      }

      for (size_t cnt=0;cnt < num_threads;cnt++) {
	threadpool.push_back(std::thread( [ this ]() {
	      std::unique_lock<std::recursive_mutex> dep_tbl(dependency_table_lock);
	      for (;;) {
		job_to_do.wait(dep_tbl,[ this ]() { return threadcleanup || (unexecuted_needs_regionupdater.size() > 0 && state==TRMS_REGIONUPDATE) || ((unexecuted_needs_regionupdater.size() > 0 || unexecuted_no_deps.size() > 0) && state==TRMS_DEPENDENCY); } );

		if (threadcleanup) {
		  return; 
		}

		auto updaterjob=unexecuted_needs_regionupdater.begin();

		if (updaterjob != unexecuted_needs_regionupdater.end()) {
		  std::shared_ptr<trm_dependency> job_ptr = updaterjob->lock();
		  if (job_ptr) {
		    /* Call the region updater code for a dependency. */

		    // remove from unexecuted_needs_regionupdater
		    unexecuted_needs_regionupdater.erase(job_ptr);
		    if (job_ptr->regionupdater) {
		      //std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>>::iterator ex_ru_iter;
		      //bool added_to_ex_ru=false;
		      
		      //std::tie(ex_ru_iter,added_to_ex_ru)=
		      executing_regionupdater.emplace(job_ptr);
		      
		      //assert(added_to_ex_ru); // dependency is allowed to be in exactly one set at a time
		      
		      std::vector<trm_arrayregion> newinputs;
		      std::vector<trm_arrayregion> newoutputs;
		      bool inputs_changed=false;
		      
		      dep_tbl.unlock();		
		      newinputs = job_ptr->regionupdater(job_ptr->metadata_inputs,job_ptr->inputs);
		      dep_tbl.lock();
		      
		      if (!(newinputs == job_ptr->inputs)) { /* NOTE: Do not change to != because operator== is properly overloaded but operator!= is not (!) */
			inputs_changed=true;
			_ensure_input_cachemanagers_registered(newinputs);
			job_ptr->inputs=newinputs;
		      }
		      
		      dep_tbl.unlock();		
		      newoutputs = job_ptr->update_output_regions(job_ptr->metadata_inputs,job_ptr->inputs,job_ptr->metadata_outputs,job_ptr->outputs);
		      dep_tbl.lock();		
		      
		      if (inputs_changed || !(newoutputs == job_ptr->outputs)) {
			/* NOTE: Do not change to != because operator== is properly overloaded but operator!= is not (!) */			
						
			_rebuild_depgraph_node_edges(job_ptr); 
		      } 
		      executing_regionupdater.erase(job_ptr); //ex_ru_iter);
		    }

		    unexecuted_regionupdated.emplace(job_ptr);
		    if (state != TRMS_REGIONUPDATE) {
		      /* dispatch either into unexecuted_no_deps (if inputs are fully evaluated) or 
			 into unexecuted_with_deps (if incomplete inputs still present) 
			 (the dispatch is batched after TRMS_REGIONUPDATE state, so if we are in that 
			 state no need to do the dispatching) 
		      */
		      _figure_out_unexecuted_deps();
		    }
		  } else {
		    // job_ptr == null -- this dependency doesn't exist any more... just throw it away!
		    unexecuted_needs_regionupdater.erase(updaterjob);
		  }
		  
		} else {
		  auto job = unexecuted_no_deps.begin(); /* iterator pointing to a dependency pointer */
		
		  
		  if (job != unexecuted_no_deps.end()) {
		    size_t changedcnt=0;
		    
		    std::shared_ptr<trm_dependency> job_ptr = job->lock();
		    
		    //std::vector<std::vector<trm_arrayregion>> inputchangedregions;
		    if (job_ptr) {
		      // Fill inputchangedregions with empty trackers if it is too small 
		      for (size_t inpcnt=job_ptr->inputchangedregion.size();inpcnt < job_ptr->inputs.size();inpcnt++) {
			job_ptr->inputchangedregion.emplace_back(); 
		      }

		      
		      if (job_ptr->force_full_rebuild) {
			// force full rebuild: Mark everything as changed
			changedcnt=1;
			size_t inputcnt=0;
			for (auto & input: job_ptr->inputs) {
			  job_ptr->inputchangedregion[inputcnt].mark_region(0,SNDE_INDEX_INVALID);
			  changedcnt++;
			  inputcnt++;
			}
		      } else {
			size_t inputcnt=0;
			for (auto & input: job_ptr->inputs) {
			  //std::vector<trm_arrayregion> inputchangedregion=_modified_regions(input);
			  _merge_modified_regions(job_ptr->inputchangedregion[inputcnt],input);
			  //inputchangedregions.push_back(inputchangedregion);
			  
			  job_ptr->inputchangedregion[inputcnt].merge_adjacent_regions();
			  
			  changedcnt += job_ptr->inputchangedregion[inputcnt].size(); /* count of changed regions in this input */
			  
			  inputcnt++;
			}
		      }
		      /* Here is where we call the dependency function ... but we need to be able to figure out the parameters. 
			 Also if the changes did not line up with the 
			 dependency inputs it should be moved to no_need_to_execute
			 instead */
		      
		      
		      std::vector<rangetracker<markedregion>> outputchangedregions;
		      
		      if (changedcnt > 0) {

			// only execute if something is changed (or force_full_rebuild, above)
			
			unexecuted_no_deps.erase(job);
			executing.insert(job_ptr);
			std::vector<trm_arrayregion> newoutputs;
			bool outputs_changed=false;

			// update output regions first 
			dep_tbl.unlock();
			newoutputs=job_ptr->update_output_regions(job_ptr->metadata_inputs,job_ptr->inputs,job_ptr->metadata_outputs,job_ptr->outputs);
			dep_tbl.lock();
			
			if (!(newoutputs == job_ptr->outputs)) { /* NOTE: Do not change to != because operator== is properly overloaded but operator!= is not (!) */
			  outputs_changed=true;
			  job_ptr->outputs=newoutputs;

			}

			// now execute!
			dep_tbl.unlock();
			job_ptr->function(this->currevision,job_ptr,job_ptr->inputchangedregion);
			dep_tbl.lock();
			
			if (outputs_changed) {			  
			  _rebuild_depgraph_node_edges(job_ptr); 
			} 
			
			job_ptr->force_full_rebuild=false; // rebuild is done!
			executing.erase(job_ptr);
			done.insert(job_ptr);
			
		      } else {
			
			unexecuted_no_deps.erase(job);
			
			no_need_to_execute.insert(job_ptr);
		      }
		      
		      
		      size_t outcnt=0;
		      //for (auto & ocr_entry: outputchangedregions) {
		      //  _mark_regions_as_modified(job_ptr->outputs[outcnt].manager,job_ptr->outputs[outcnt].array,ocr_entry);
		      //
		      //
		      //  if (ocr_entry.size() > 0) {
		      //    for (auto & outdep: job_ptr->output_dependencies[outcnt]) {
		      //	/* !!!*** Is there any way to check whether we have really messed with an input of this output dependency? */
		      // _call_regionupdater(outdep);
		      //
		      //    }
		      //  }
		      
		      /* ***!!! We should have a better way to map the dirty() call from the 
			 function code back to this dependency so we don't just have to blindly
			 iterate over all of them ... */
		      for (outcnt=0; outcnt < job_ptr->output_dependencies.size();outcnt++) {
			
			for (auto & outdep: job_ptr->output_dependencies[outcnt]) {
			  
			  
			  if (unexecuted_with_deps.count(outdep)) {
			    /* this still needs to be executed */
			    /* are all of its input dependencies complete? */
			    bool deps_complete=true;
			    std::shared_ptr<trm_dependency> outdep_strong=outdep.lock();
			    if (outdep_strong) {
			      for (size_t indepcnt=0;indepcnt < outdep_strong->input_dependencies.size();indepcnt++) {
				for (auto & indep: outdep_strong->input_dependencies[indepcnt]) {
				  std::shared_ptr<trm_dependency> indep_strong=indep.lock();
				  if (indep_strong) {
				    if (executing.count(indep_strong) || unexecuted_with_deps.count(indep) || unexecuted_no_deps.count(indep)) {
				      deps_complete=false;
				    }
				  }
				}
			      }
			    }
			    if (deps_complete) {
			      /* This dep has all input dependencies satisfied... move it into unexecuted_no_deps */
			      //outdep_ptr=*outdep;
			      unexecuted_with_deps.erase(outdep);
			      unexecuted_no_deps.insert(outdep);
			    }
			  }
			}
		      }
		    
		    } else {
		      // job_ptr == null -- this dependency doesn't exist any more... just throw it away!
		      unexecuted_no_deps.erase(job);
		    }
		  }
		}
		/*  signal job_to_do condition variable
		    according to the (number of entries in
		    unexecuted_no_deps + number of entries in unexecuted_needs_regionupdater)-1.... because if there's only
		    one left, we can handle it ourselves when we loop back */
		
		size_t njobs=unexecuted_needs_regionupdater.size();

		if (njobs==0 && state==TRMS_REGIONUPDATE) {
		  regionupdates_done.notify_all();
		}
		
		if (state==TRMS_DEPENDENCY) {
		  njobs+=unexecuted_no_deps.size();
		}
		if (njobs==0 && state==TRMS_DEPENDENCY) {
		  jobs_done.notify_all();
		}

		while (njobs > 1) {
		  job_to_do.notify_one();
		  njobs--;
		}

		
		
	      
	      }
	      
	    }));
      }

      transaction_wait_thread=std::thread([ this ]() {
					    std::unique_lock<std::recursive_mutex> dep_tbl(dependency_table_lock);
					    for (;;) {
					      
					      jobs_done.wait(dep_tbl,[ this ]() { return (unexecuted_no_deps.size()==0 && unexecuted_with_deps.size()==0 && executing.size()==0 && state==TRMS_DEPENDENCY) || threadcleanup;});

					      if (threadcleanup) {
						return; 
					      }


					      assert(state==TRMS_DEPENDENCY);
					      // all jobs are done. Now the transaction_update_writelock_holder can be released
					      std::unique_lock<rwlock_lockable> holder;

					      // Clear out modified_db
					      // **** NOTE: If stuff is modified externally between End_Transaction() and
					      // the end of computation we may miss it because of these clear() calls
					      modified_db.clear();
					      modified_metadata_db.clear();

					      // move everything from done into unsorted
					      for (auto done_iter = done.begin();done_iter != done.end();done_iter=done.begin()) {
						unsorted.insert(*done_iter);
						done.erase(done_iter);
					      }

					      // move everything from no_need_to_execute into unsorted
					      for (auto nnte_iter = no_need_to_execute.begin();nnte_iter != no_need_to_execute.end();nnte_iter=no_need_to_execute.begin()) {
						unsorted.insert(*nnte_iter);
						no_need_to_execute.erase(nnte_iter);
					      }

					      state=TRMS_IDLE; 
					      
					      holder.swap(transaction_update_writelock_holder);
					      assert(holder.owns_lock()); // Should always be true because our thread has the exclusive right to release the lock and is only notified when it is appropriate to do this.
					      // holder dropping out of scope releases the lock
					      all_done.notify_all();
					      
					    }
					  });

    }

    ~trm()
    {
      /* clean up threads */
      {
	std::lock_guard<std::recursive_mutex> dep_tbl(dependency_table_lock);
	threadcleanup=true;
	job_to_do.notify_all();
	jobs_done.notify_all();
      }
      for (size_t cnt=0;cnt < threadpool.size();cnt++) {
	threadpool[cnt].join();	
      }
      transaction_wait_thread.join();

      wfmdb_notifier=nullptr; // trigger deletion of wfmdb_notifier before we disappear ourselves, because it has a pointer to us. 

    }


    bool _region_in_modified_db(const trm_arrayregion &region)
    {
      /* dependency_table_lock should be locked when calling this method */
      auto dbregion = modified_db.find(region.array);
      if (dbregion == modified_db.end()) {
	return false;
      }
      std::pair<std::shared_ptr<arraymanager>,rangetracker<markedregion>> &manager_tracker = dbregion->second;
      std::shared_ptr<arraymanager> &manager=manager_tracker.first;
      rangetracker<markedregion> &tracker=manager_tracker.second;

      rangetracker<markedregion> subtracker=tracker.iterate_over_marked_portions(region.start,region.len);

      return !(subtracker.begin()==subtracker.end());
      
    }
    
    void _merge_modified_regions(rangetracker<markedregion> &inputchangedregion,const trm_arrayregion &input)
    /* evaluate modified regions of input, per the modified_db */
    {
      /* dependency_table_lock should be locked when calling this method */
      if (modified_db.count(input.array)) {
	std::pair<std::shared_ptr<arraymanager>,rangetracker<markedregion>> &manager_tracker = modified_db.at(input.array);
	std::shared_ptr<arraymanager> &manager=manager_tracker.first;
	rangetracker<markedregion> &tracker=manager_tracker.second;

	std::vector<trm_arrayregion> retval;
      
	rangetracker<markedregion> subtracker=tracker.iterate_over_marked_portions(input.start,input.len);

	for (auto & subregion: subtracker) {
	  inputchangedregion.mark_region(subregion.second->regionstart,subregion.second->regionend-subregion.second->regionstart);
	  //trm_arrayregion newregion(manager,input.array,subregion.second->regionstart,subregion.second->regionend-subregion.second->regionstart);
	
	  //retval.push_back(newregion);
	}
      }
      //return retval;
    }
      
    void _remove_depgraph_node_edges(std::weak_ptr<trm_dependency> dependency,std::vector<std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>>> &input_dependencies,std::vector<std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>>> &output_dependencies)
    /* Clear out the graph node edges that impinge on dependency, 
       based on its professed inputs and outputs */
    /* dependency_table_lock must be held by caller */
    {
      for (auto & old_input: input_dependencies) {
	for (auto & old_input_dep: old_input ) { // step through the set for this input
	  std::shared_ptr<trm_dependency> old_input_dep_strong=old_input_dep.lock();
	  if (old_input_dep_strong) {
	    for (size_t outcnt=0;outcnt < old_input_dep_strong->output_dependencies.size();outcnt++) {
	      if (old_input_dep_strong->output_dependencies[outcnt].count(dependency)) {
		old_input_dep_strong->output_dependencies[outcnt].erase(dependency);
	      }
	    }
	    
	  }
	}
      }

      for (auto & old_output: output_dependencies) {
	for (auto & old_output_dep: old_output) { // step through the set for this output
	  std::shared_ptr<trm_dependency> old_output_dep_strong=old_output_dep.lock();
	  if (old_output_dep_strong) {
	    for (size_t inpcnt=0;inpcnt < old_output_dep_strong->input_dependencies.size();inpcnt++) {
	      if (old_output_dep_strong->input_dependencies[inpcnt].count(dependency)) {
		old_output_dep_strong->input_dependencies[inpcnt].erase(dependency);
	      }
	    }
	  }
	
	}
      }


    } 
    void _rebuild_depgraph_node_edges(std::shared_ptr<trm_dependency> dependency)
    /* Clear out and rebuild the dependency graph node edges that impinge on dependency, 
       based on its professed inputs and outputs */
    /* dependency_table_lock must be held by caller */
    {
      _remove_depgraph_node_edges(dependency,dependency->input_dependencies,dependency->output_dependencies);

      /* make sure dependency has a big enough input_dependencies array */
      while (dependency->input_dependencies.size() < dependency->metadata_inputs.size() + dependency->inputs.size()) {
	dependency->input_dependencies.emplace_back();
      }
      
      /* Iterate over all existing dependencies we could have a relationship to */
      for (auto & existing_dep: dependencies) {

	std::shared_ptr<trm_dependency> existing_dep_strong=existing_dep.lock();
	/* make sure existing dep has a big enough output_dependencies array */
	
	if (existing_dep_strong) {
	  while (existing_dep_strong->output_dependencies.size() < existing_dep_strong->metadata_outputs.size() + existing_dep_strong->outputs.size()) {
	    existing_dep_strong->output_dependencies.emplace_back();
	  }
	
	  
	  
	  /* For each of our input metadata dependencies, does the existing dependency have an output
	     dependency? */
	  size_t inpcnt=0;
	  for (auto & input: dependency->metadata_inputs) { // input is a weak_ptr to an infostore
	    
	    
	    auto & this_input_depset = dependency->input_dependencies[inpcnt];
	    for (size_t outcnt=0;outcnt < existing_dep_strong->metadata_outputs.size();outcnt++) {
	      // existing_dep_strong->metadata_outputs.at(outcnt) is a weak pointer to an infostore...
	      // need to compare with input

	      // use owner_before() attributes for comparison so comparison is legitimate even if
	      // weak_pointers have been released.
	      const std::weak_ptr<mutableinfostore> & output = existing_dep_strong->metadata_outputs.at(outcnt);

	      // basically we are looking for input==output
	      // expressed as !(input < output) && !(output < input)
	      // where input < output expressed as input.owner_before(output)
	      if (!input.owner_before(output) && !output.owner_before(input)) {
	      
		this_input_depset.emplace(existing_dep_strong);
		existing_dep_strong->output_dependencies[outcnt].emplace(dependency);
	      } 
	    }
	    inpcnt++;
	  }

	  /* For each of our input array dependencies, does the existing dependency have an output
	     dependency? */
	  inpcnt=0;
	  for (auto & input: dependency->inputs) { // input is a trm_arrayregion
	    
	    
	    auto & this_input_depset = dependency->input_dependencies[dependency->metadata_inputs.size() + inpcnt];
	    for (size_t outcnt=0;outcnt < existing_dep_strong->outputs.size();outcnt++) {
	      
	    
	      if (input.overlaps(existing_dep_strong->outputs[outcnt])) {
		this_input_depset.emplace(existing_dep_strong);
		existing_dep_strong->output_dependencies[existing_dep_strong->metadata_outputs.size() +outcnt].emplace(dependency);
	      } 
	    }
	    inpcnt++;
	  }
	
	  /* For each of our output metadata dependencies, does the existing dependency have an input
	     dependency? */
	  size_t outcnt=0;
	  for (auto & output: dependency->metadata_outputs) { // output is a weak_ptr to an infostore	    

	    auto & this_output_depset = dependency->output_dependencies[outcnt];
	    for (inpcnt=0;inpcnt < existing_dep_strong->metadata_inputs.size();inpcnt++) {
	      // existing_dep_strong->metadata_inputs.at(inpcnt) is a weak pointer to an infostore...
	      // need to compare with output

	      // use owner_before() attributes for comparison so comparison is legitimate even if
	      // weak_pointers have been released.
	      const std::weak_ptr<mutableinfostore> & input = existing_dep_strong->metadata_inputs.at(inpcnt);
	      
	      // basically we are looking for input==output
	      // expressed as !(input < output) && !(output < input)
	      // where input < output expressed as input.owner_before(output)
	      if (!input.owner_before(output) && !output.owner_before(input)) {
	      
		this_output_depset.emplace(existing_dep_strong);
		existing_dep_strong->input_dependencies[inpcnt].emplace(dependency);
	      } 

	      
	    }
	    outcnt++;
	  }

	  
	  /* For each of our output array dependencies, does the existing dependency have an input
	     dependency? */
	  outcnt=0;
	  for (auto & output: dependency->outputs) {
	    auto & this_output_depset = dependency->output_dependencies[dependency->metadata_outputs.size() + outcnt];
	    for (inpcnt=0;inpcnt < existing_dep_strong->inputs.size();inpcnt++) {
	      if (existing_dep_strong->inputs[inpcnt].overlaps(output)) {
		this_output_depset.emplace(existing_dep_strong);
		existing_dep_strong->input_dependencies[existing_dep_strong->metadata_inputs.size() + inpcnt].emplace(dependency);
	      }
	    }
	    outcnt++;
	  }
	}
      }
      
    }
    
    std::shared_ptr<trm_dependency>  add_dependency(std::function<void(snde_index newversion,std::shared_ptr<trm_dependency> dep,std::vector<rangetracker<markedregion>> &inputchangedregions)> function,
						    std::function<std::vector<trm_arrayregion>(std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs,std::vector<trm_arrayregion> inputs)> regionupdater,
						    std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs,
						    std::vector<trm_arrayregion> inputs, // inputs array does not need to be complete; will be passed immediately to regionupdater() -- so this need only be a valid seed. 
						    //std::vector<trm_arrayregion> outputs)
						    std::vector<std::shared_ptr<mutableinfostore>> metadata_outputs,
						    std::function<std::vector<trm_arrayregion>(std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs, std::vector<trm_arrayregion> inputs,std::vector<std::shared_ptr<mutableinfostore>> metadata_outputs,std::vector<trm_arrayregion> outputs)> update_output_regions,
						    std::function<void(std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs,std::vector<trm_arrayregion> inputs,std::vector<trm_arrayregion> outputs)> cleanup) // cleanup() should not generally do any locking but just free regions. 
    {
      /* Add a dependency outside StartTransaction()...EndTransaction()... First execution opportunity will be at next call to EndTransaction() */
      /* acquire necessary read lock to allow modifying dependency tree */
      std::lock_guard<rwlock_lockable> ourlock(transaction_update_lock->reader);
      return add_dependency_during_update(function,
					  regionupdater,
					  metadata_inputs,
					  inputs,
					  metadata_outputs,
					  update_output_regions,
					  cleanup);
    }

    void _categorize_dependency(std::shared_ptr<trm_dependency> dependency)
    {
      /* During EndTransaction() or equivalent we have to move each dependency 
	 where it belongs. This looks at the inputs and outputs of the given dependency, 
	 which should be in unsorted, and moves into unexecuted_needs_regionupdater  
         (so that its regionupdater will be called) if an immediate need
	 to execute is identified, or into no_need_to_execute otherwise.

	 The dependency_table_lock should be locked in order to call this 
	 method.
      */

      bool modified_input_dep=false;
      //assert(unsorted.count(dependency)==1);
      //assert(dependency->pending_input_dependencies.empty());
      
      //unsorted.erase(dependency);

      if (dependency->force_full_rebuild) {
	modified_input_dep=true;
      }
      
      if (!modified_input_dep) {
	for (auto & metadatainput: dependency->metadata_inputs) {
	  if (modified_metadata_db.find(metadatainput) != modified_metadata_db.end()) {
	    // marked as modified
	    modified_input_dep=true;
	    break;
	  }
	}
      }

      if (!modified_input_dep) {
	for (auto & input: dependency->inputs) {
	  if (_region_in_modified_db(input)) {
	    modified_input_dep=true;
	    break;
	    //dependency.pending_input_dependencies.push_back();
	  }
	}
      }
      if (modified_input_dep) {
	/* temporarily mark with no_deps... will have to walk 
	   dependency tree and shift to unexecuted_with_deps
	   (if appropriate) later */
	
	/* Call regionupdater function and update dependency graph if necessary */ 
	//_call_regionupdater(dependency); 
	
	unexecuted_needs_regionupdater.insert(dependency);
      } else {
	no_need_to_execute.insert(dependency);
      }
      
    }

    void _figure_out_unexecuted_deps()
    {
      /* Figure out whether the dependencies listed in unexecuted_regionupdated (and/or given dep) should 
       go into unexecuted_with_deps or unexecuted_no_deps 
       
       The dependency_table_lock should be locked in order to call this 
       method.
      */
      
      /* Iterate recursively over the output dependencies of dep, unexecuted_regionupdated, unexecuted_with_deps, unexecuted_needs_regionupdater, unexecuted_no_deps, executing_regionupdater, and executing, move them into 
	 unexecuted_with_deps. ... be careful about iterator validity
	 
	 Anything that remains in unexecuted can be shifted into unexecuted_no_deps */

      // accumulate all unexecuted dependencies into a giant vector
      std::vector<std::weak_ptr<trm_dependency>> unexecuted(unexecuted_regionupdated.begin(),unexecuted_regionupdated.end());
      unexecuted.insert(unexecuted.end(),unexecuted_with_deps.begin(),unexecuted_with_deps.end());
      unexecuted.insert(unexecuted.end(),unexecuted_needs_regionupdater.begin(),unexecuted_needs_regionupdater.end());
      unexecuted.insert(unexecuted.end(),unexecuted_no_deps.begin(),unexecuted_no_deps.end());
      unexecuted.insert(unexecuted.end(),executing_regionupdater.begin(),executing_regionupdater.end());
      unexecuted.insert(unexecuted.end(),executing.begin(),executing.end());
      
      
      for (auto & dependency: unexecuted) {

	std::shared_ptr<trm_dependency> dep_strong(dependency);
	if (dep_strong) {
	  _output_deps_into_unexecwithdeps(dep_strong,false);
	}
      }
      
      /* shift any that remain in unexecuted_regionupdated into unexecuted_no_deps */
      std::vector<std::weak_ptr<trm_dependency>> unexecuted_regionupdated_copy(unexecuted_regionupdated.begin(),unexecuted_regionupdated.end());
      for (auto & dependency: unexecuted_regionupdated_copy) {
	unexecuted_regionupdated.erase(dependency);
	unexecuted_no_deps.insert(dependency);
      }
      
    }
    

    void _erase_dep_from_tree(std::weak_ptr<trm_dependency> dependency,std::vector<std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>>> &input_dependencies,std::vector<std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>>> &output_dependencies)
    // called by trm_dependency's destructor
    {
      // must hold dependency table lock
      std::lock_guard<std::recursive_mutex> dep_tbl(dependency_table_lock);
      // not permitted to do anything that requires a mutex here,
      // as we may be called during a destructor and we already
      // own the dep table lock
      
      _remove_depgraph_node_edges(dependency,input_dependencies,output_dependencies);


      /* remove from full list of all dependencies */
      dependencies.erase(dependency);

      /* remove from queue to execute */
      if (unsorted.find(dependency) != unsorted.end()) {
	unsorted.erase(dependency);
      }
      if (no_need_to_execute.find(dependency) != no_need_to_execute.end()) {
	no_need_to_execute.erase(dependency);
      }
      if (unexecuted_with_deps.find(dependency) != unexecuted_with_deps.end()) {
	unexecuted_with_deps.erase(dependency);
      }
      if (unexecuted_needs_regionupdater.find(dependency) != unexecuted_needs_regionupdater.end()) {
	unexecuted_needs_regionupdater.erase(dependency);
      }
      if (unexecuted_regionupdated.find(dependency) != unexecuted_regionupdated.end()) {
	unexecuted_regionupdated.erase(dependency);
      }
      if (unexecuted_no_deps.find(dependency) != unexecuted_no_deps.end()) {
	unexecuted_no_deps.erase(dependency);
      }
      // no need to look at the executing set, because those are shared_ptrs,
      // so we wouldn't be called if in there!
      if (done.find(dependency) != done.end()) {
	done.erase(dependency);
      }

    }


    // remove_dependency() no longer necessary... just allow all shared_ptr references
    // to the trm_dependency to expire!
    
    //void remove_dependency(std::weak_ptr<trm_dependency> dependency)
    //{
    //  ///* Can only remove dependency while an update is not in progress */
    //  //std::lock_guard<rwlock_lockable> ourlock(transaction_update_lock->reader);
    //
    //  /* must hold dependency_table_lock */ 
    //  std::lock_guard<std::recursive_mutex> dep_tbl(dependency_table_lock);
    //  // (content moved into _erase_dep_from_tree)
    //  _remove_dependency(dependency);
    //}
    
    /* add_dependency_during_update may only be called during a transaction */
    /* MUST HOLD WRITE LOCK for all output_arrays specified... may reallocate these arrays! */
    std::shared_ptr<trm_dependency> add_dependency_during_update(std::function<void(snde_index newversion,std::shared_ptr<trm_dependency> dep,std::vector<rangetracker<markedregion>> &inputchangedregions)> function,
								 std::function<std::vector<trm_arrayregion>(std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs,std::vector<trm_arrayregion> inputs)> regionupdater,
								 std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs,
								 std::vector<trm_arrayregion> inputs, // inputs array does not need to be complete; will be passed immediately to regionupdater() -- so this need only be a valid seed.
								 std::vector<std::shared_ptr<mutableinfostore>> metadata_outputs,
								 //std::vector<trm_arrayregion> outputs)
								 std::function<std::vector<trm_arrayregion>(std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs,std::vector<trm_arrayregion> inputs,std::vector<std::shared_ptr<mutableinfostore>> metadata_outputs,std::vector<trm_arrayregion> outputs)> update_output_regions,
								 std::function<void(std::vector<std::shared_ptr<mutableinfostore>> metadata_inputs, std::vector<trm_arrayregion> inputs,std::vector<trm_arrayregion> outputs)> cleanup) // cleanup() should not generally do any locking but just free regions. 
    /* May only be called while holding transaction_update_lock, either as a reader(maybe?) or as a writer */
    {

      /* Construct inputs */
      inputs=regionupdater(metadata_inputs,inputs);
      _ensure_input_cachemanagers_registered(inputs);
      
      /* construct empty output regions */
      std::vector<trm_arrayregion> outputs; // start with blank output array
      //outputs=update_output_regions(inputs,outputs);
      
      std::shared_ptr<trm_dependency> dependency=std::make_shared<trm_dependency>(shared_from_this(),function,regionupdater,update_output_regions,cleanup,metadata_inputs,inputs,metadata_outputs,outputs);
      dependency->weak_this = dependency; // couldn't be set in constructor because you can't call shared_form_this() in constructor, but it is needed in the destructor and can't be created there either(!)

      
      std::lock_guard<std::recursive_mutex> dep_tbl(dependency_table_lock);
      
      /*  Check input and output dependencies; 
	  if we are inside a transactional update and there are 
	  no unexecuted dependencies, we should drop into unexecuted_no_deps, unexecuted_with_deps, no_need_to_execute, etc. instead of unsorted */

      //_call_regionupdater(dependency); // Make sure we have full list of dependencies.
      // 
      //// Fill inputchangedregions with full trackers according to number of inputs (now replaced with force_full_rebuild auto-initialized to true)
      //for (size_t inpcnt=0;inpcnt < dependency->inputs.size();inpcnt++) {
      //dependency->inputchangedregion.emplace_back();
      //dependency->inputchangedregion[inpcnt].mark_region(0,SNDE_INDEX_INVALID); // Mark entire block
      //}
      
      
      dependencies.emplace(dependency);

      _rebuild_depgraph_node_edges(dependency);
      

      if (state==TRMS_DEPENDENCY) {
	unexecuted_needs_regionupdater.insert(dependency);
	job_to_do.notify_one();
      } else {
	unsorted.insert(dependency);	
      }
      return dependency;
    }

    void _output_deps_into_unexecwithdeps(std::shared_ptr<trm_dependency> dependency,bool is_an_output_dependency)
    /* Iterate recursively over dependency and its output dependencies of <dependency> moving them 
       from the unexecuted or no_need_to_execute list (if present) into the unexecuted_with_deps set if they have dependencies */
    {
      // If this is already an output dependency we are processing, then
      // we have to move it into unexecuted_with_deps.
      if (is_an_output_dependency) {
	if (unexecuted_regionupdated.count(dependency)) {
	  unexecuted_regionupdated.erase(dependency);
	  unexecuted_with_deps.insert(dependency);
	  
	} else if (no_need_to_execute.count(dependency)) {
	  no_need_to_execute.erase(dependency);
	  unexecuted_with_deps.insert(dependency);	
	} else if (unexecuted_no_deps.count(dependency)) {
	  unexecuted_no_deps.erase(dependency);
	  unexecuted_with_deps.insert(dependency);
	}
      }
      /* recursive loop */
      for (auto & out : dependency->output_dependencies) {
	for (auto & outdep : out) {
	  std::shared_ptr<trm_dependency> outdep_strong=outdep.lock();
	  if (outdep_strong) {
	    _output_deps_into_unexecwithdeps(outdep_strong,true);
	  }
	}
      }
    }


    void _Start_Transaction(std::unique_lock<rwlock_lockable> &ourlock)
    {
      /* assumes dependency_table_lock is held already */
      assert(!transaction_update_writelock_holder.owns_lock());
      
      assert(no_need_to_execute.empty());
      assert(unexecuted_with_deps.empty());
      assert(unexecuted_needs_regionupdater.empty());
      assert(unexecuted_regionupdated.empty());
      assert(unexecuted_no_deps.empty());
      assert(executing_regionupdater.empty());
      assert(executing.empty());
      assert(done.empty());
      
      assert(modified_db.empty());
      assert(modified_metadata_db.empty());
      
      
      state=TRMS_TRANSACTION;
      
      currevision++;
      //fprintf(stderr,"_Start_Transaction(%u)\n",(unsigned)currevision);
      
      // Move transaction lock to holder 
      ourlock.swap(transaction_update_writelock_holder);
      
      
      for (auto & dependency : dependencies) {
	// Clear out inputchangedregions
	std::shared_ptr<trm_dependency> dep_strong=dependency.lock();
	if (dep_strong) {
	  dep_strong->inputchangedregion.empty();
	
	  //for (auto & icr : dependency->inputchangedregion) {
	  //  icr.clear_all();
	  //}
	  
	// Fill inputchangedregions with empty trackers according to number of inputs
	  for (size_t inpcnt=0;inpcnt < dep_strong->inputs.size();inpcnt++) {
	    dep_strong->inputchangedregion.emplace_back(); 
	  }
	}
      }
      
    }

  

    void Start_Transaction()
    {
      {
	std::unique_lock<std::recursive_mutex> dep_tbl(dependency_table_lock);
	_Wait_Computation(currevision,dep_tbl);
      }
      
      std::unique_lock<rwlock_lockable> ourlock(transaction_update_lock->writer);
      

      {
	std::lock_guard<std::recursive_mutex> dep_tbl(dependency_table_lock);

	_Start_Transaction(ourlock);

      }
    }
    
    void _mark_infostore_as_modified(std::shared_ptr<mutableinfostore> infostore)
    {
      /* dependency_table_lock must be locked when this function is called */
      auto dbregion = modified_metadata_db.find(infostore);

      if (dbregion==modified_metadata_db.end()) {
	/* this infostore not currently marked as modified */
	modified_metadata_db.emplace(infostore);
      }
      
    }

    void _mark_region_as_modified(const trm_arrayregion &modified)
    {
      /* dependency_table_lock must be locked when this function is called */
      auto dbregion = modified_db.find(modified.array);

      if (dbregion==modified_db.end()) {
	/* No existing entry for this array */
	//dbregion=modified_db.emplace(0,std::make_pair<std::shared_ptr<arraymanager>,rangetracker<arrayregion>>(modified.manager,rangetracker<markedregion>())).first;
	dbregion=modified_db.emplace((void **)NULL,std::make_pair(modified.manager,rangetracker<markedregion>())).first;
      }
      dbregion->second.second.mark_region_noargs(modified.start,modified.len);
      
    }

    void _mark_region_as_modified(std::shared_ptr<arraymanager> manager,void **array,const markedregion &modified)
    {
      /* dependency_table_lock must be locked when this function is called */
      auto dbregion = modified_db.find(array);

      if (dbregion==modified_db.end()) {
	/* No existing entry for this array */
	//dbregion=modified_db.emplace(0,std::make_pair<std::shared_ptr<arraymanager>,rangetracker<arrayregion>>(modified.manager,rangetracker<markedregion>())).first;
	dbregion=modified_db.emplace((void **)NULL,std::make_pair(manager,rangetracker<markedregion>())).first;
      }
      dbregion->second.second.mark_region_noargs(modified.regionstart,modified.regionend-modified.regionstart);
      
    }

    /* ***!!!!!*** SHOULD REMOVE CALLS TO THIS AND REPLACE WITH arraymanager's MARK_AS_DIRTY */
    //void Transaction_Mark_Modified(const trm_arrayregion &modified)
    //{
    //  std::lock_guard<std::recursive_mutex> dep_tbl(dependency_table_lock);
    //
    //  _mark_region_as_modified(modified);
    //  
    //}


    void _mark_regions_as_modified(std::shared_ptr<arraymanager> manager,void **array,rangetracker<markedregion> & modified)
    {
      /* dependency_table_lock must be locked when this function is called */
      for (auto & region : modified) {
	_mark_region_as_modified(manager,array,*region.second);
      }
    }

    void _mark_regions_as_modified(std::vector<trm_arrayregion> & modified)
    {
      /* dependency_table_lock must be locked when this function is called */
      for (auto & region : modified) {
	_mark_region_as_modified(region);
      }
    }

    
    /* ***!!!!!*** SHOULD REMOVE CALLS TO THIS AND REPLACE WITH arraymanager's MARK_AS_DIRTY 
     through arraymanager */
    //void Transaction_Mark_Modified(std::vector<trm_arrayregion> &modified)
    //{
    //  std::lock_guard<std::recursive_mutex> dep_tbl(dependency_table_lock);
    //  
    //  _mark_regions_as_modified(modified);
    //  
    //}

    /* ***!!!!!*** SHOULD REMOVE CALLS TO THIS AND REPLACE WITH arraymanager's MARK_AS_DIRTY 
     through arraymanager */
    //void Transaction_Mark_Modified(std::shared_ptr<std::vector<trm_arrayregion>> modified)
    //{
    //  std::lock_guard<std::recursive_mutex> dep_tbl(dependency_table_lock);
    //  
    //  _mark_regions_as_modified(*modified);
    //  
    //}

    /*void _Modified_Dependencies()
    {
      
    }*/

    void _ensure_input_cachemanagers_registered(std::vector<trm_arrayregion> inputs)
    /* Ensure that our trm is registered as a cachemanager for each of the arraymanagers corresponding to the given inputs, 
       so that we are kept abreast of changes to those inputs and can update our modified_db */ 
    {

      if (!change_detection_pseudo_cache) {
	// change_detection_pseudo_cache is not created in our constructor
	// because we are not allowed to use shared_from_this() in that context. 
	change_detection_pseudo_cache = std::make_shared<trm_change_detection_pseudo_cache>(shared_from_this());
      }
      
      for (auto & input: inputs) {
	if (!input.manager->has_cache(our_unique_name)) {
	  input.manager->set_undefined_cache(our_unique_name,change_detection_pseudo_cache);
	}
      }
    }



    snde_index End_Transaction()
    /* Can call Wait_Computation on returned revision # to wait for 
       computation to be complete */
    // !!!*** NOTE: If stuff is modified externally between End_Transaction() and
    // the end of computation, we may miss it because we clear the modified_db
    // at the end of computation... should we incrementally remove stuff from
    // the modified db during the categorization process? ... and the NOT do
    // the clear() in transaction_wait_thread?
    {
    
      snde_index retval=currevision;

      //fprintf(stderr,"_End_Transaction(%u)\n",(unsigned)currevision);

      std::unique_lock<std::recursive_mutex> dep_tbl(dependency_table_lock);
	
      /* Now need to go through our dependencies and see which have been modified */
      for (auto dependency=unsorted.begin();dependency != unsorted.end();dependency=unsorted.begin()) {
	std::shared_ptr<trm_dependency> dep_strong=dependency->lock();
	unsorted.erase(dependency);
	
	if (dep_strong) {
	  _categorize_dependency(dep_strong);
	}
	
      }
      
      state=TRMS_REGIONUPDATE;
      /* Run region update process and wait for it to finish */

      do {
	size_t nupdates = unexecuted_needs_regionupdater.size();
	while (nupdates > 0) {
	  job_to_do.notify_one();
	  nupdates--;
	}

	regionupdates_done.wait(dep_tbl, [ this ]() { return unexecuted_needs_regionupdater.size()==0; });
      } while (unexecuted_needs_regionupdater.size() > 0);
      
      
      
      _figure_out_unexecuted_deps();

      state=TRMS_DEPENDENCY;
      /* Initiate execution process */
      
      size_t njobs=unexecuted_no_deps.size();
      if (!njobs) {
	jobs_done.notify_all(); /* if no jobs, notify anybody who is waiting so that we get our cleanup */
      }
      while (njobs > 0) {
	job_to_do.notify_one();
	njobs--;
      }
      
      //jobs_done.wait( dep_tbl, [ this ]() { return unexecuted_no_deps.size()==0 && unexecuted_with_deps.size()==0 && executing.size()==0;});

      // Computations may not be done.... need to Wait_Computation to be assured
      // of completions

      //Transaction_Mark_Modified(modified);

      return retval;

    }
    

    void _Wait_Computation(snde_index revnum,std::unique_lock<std::recursive_mutex> &dep_tbl)
    /* Wait for computation of revnum to complete */
    /* assumes dependency_table_lock is held by given unique_lock */
    {
      
      //fprintf(stderr,"_Wait_Computation(%u)\n",revnum);
      assert(revnum <= currevision);
      if (revnum==currevision) {
	all_done.wait( dep_tbl, [ this,revnum ]() { return revnum < currevision || !transaction_update_writelock_holder.owns_lock();});
      }
      //fprintf(stderr,"_Wait_Computation(%u) complete.\n",revnum);
    }

    void Wait_Computation(snde_index revnum)
    /* Wait for computation of revnum to complete */
    {
      std::unique_lock<std::recursive_mutex> dep_tbl(dependency_table_lock);

      _Wait_Computation(revnum,dep_tbl);
    }

    /* ***!!!!! Should implement end_transaction() + wait_computation() that 
       acquires and returns locks for all dependency inputs/regions to ensure consistency */
  };
  


};
#endif // SNDE_REVISION_MANAGER_HPP
