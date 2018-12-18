/* transactional revision manager */
/* ***!!!!  SHOULD REDESIGN WITH PROPER DATABASE INDEXED BY 
   WHAT THE FUNCTION IS DEPENDENT ON
   ***!!! SHOULD IMPLEMENT SPECIFYING ON-DEMAND ARRAYS so that
   functions which only generate stuff in on-demand arrays get 
   postponed until output is needed */

#include <unordered_set>
#include <atomic>
#include "gsl-lite.hpp"
#include "arraymanager.hpp"

#ifndef SNDE_REVISION_MANAGER_HPP
#define SNDE_REVISION_MANAGER_HPP


namespace snde {
  /* The transactional revision manager organizes changes to a set of 
     arrays into discrete versions of the entire set. 

     Changes nominally happen instantaneously, although some looser access is permitted.

     Specifically, it is possible to hold read locks to an old version even after 
     a new version has been defined, so long as the changes have not (yet) affected
     what those read locks are protecting.


     All changes to the associated arrays must be managed through this manager, and 
     obtaining write locks to them in other ways (beyond what is facilitated/permitted 
     by the TRM) is not allowed. 

     Dependencies can be registered with the TRM so that outputs can be automatically
     recalculated from inputs. 


     TRM can manage arrays that cross the boundary between arraymanagers, but
     you still need to follow the locking order.. e.g. stuff managed by 
     one arraymanager should be locked prior to stuff managed by 
     the other arraymanager or some other consistent rule. 


     ***!!! Should be modified to support functions that are only executed on-demand...
i.e. a transaction that does nothing but mark demands on some outputs.... Those outputs
are otherwise never generated, even if their input changes ***!!!
  */
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

     // Note that this does nothing in terms of locking, which probably must be done separately
  */


  
  
  class trm_dependency { /* dependency of one memory region on another */
  public:
    std::function<std::vector<rangetracker<markedregion>>(snde_index newversion,std::shared_ptr<trm_dependency> dep,std::vector<rangetracker<markedregion>> &inputchangedregions)> function; /* returns updated output region array */
    /* if function is NULL, that means that this is an input, i.e. one of the input arrays that is locked for 
       write and that we will be responding to changes from */
    
    std::function<std::vector<trm_arrayregion>(std::vector<trm_arrayregion> inputs)> regionupdater; /* to be called when an input changes... returns updated input region array ... Must not be able to make calls that will try to mess with the dependency graph, as that will already be locked when this call is made */

    std::function<std::vector<trm_arrayregion>(std::vector<trm_arrayregion> inputs,std::vector<trm_arrayregion> outputs)> update_output_regions;
    
    std::vector<rangetracker<markedregion>> inputchangedregion; // rangetracker for changed zones, for each input 
    std::vector<trm_arrayregion> inputs;
    std::vector<trm_arrayregion> outputs;

    std::vector<std::unordered_set<std::shared_ptr<trm_dependency>>> input_dependencies; /* vector of input dependencies,  per input DOES IT REALLY NEED TO BE A SET? DON'T THINK SO!!! */
    std::vector<std::unordered_set<std::shared_ptr<trm_dependency>>> output_dependencies; /* vector of output dependencies, per output */


    /* pending_input_dependencies is only valid during a transaction, and lists
       input dependencies that will be modified by other dependencies */
    //std::vector<std::weak_ptr<trm_dependency>> pending_input_dependencies;

    

    trm_dependency(std::function<std::vector<rangetracker<markedregion>>(snde_index newversion,std::shared_ptr<trm_dependency> dep,std::vector<rangetracker<markedregion>> &inputchangedregions)> function,
		   std::function<std::vector<trm_arrayregion>(std::vector<trm_arrayregion> inputs)> regionupdater,
		   std::function<std::vector<trm_arrayregion>(std::vector<trm_arrayregion> inputs,std::vector<trm_arrayregion> outputs)> update_output_regions,
		   std::vector<trm_arrayregion> inputs,
		   std::vector<trm_arrayregion> outputs) :
      function(function),
      regionupdater(regionupdater),
      update_output_regions(update_output_regions),
      inputs(inputs),
      outputs(outputs)
    {
      
    }
		   

  };


  /* #defines for trm::state */ 
#define TRMS_IDLE 0
#define TRMS_TRANSACTION 1 /* Between BeginTransaction() and EndTransaction() */
#define TRMS_DEPENDENCY 2 /* performing dependency updates inside EndTransaction() */

  
  class trm { /* transactional revision manager */
    /* General rule: You are not allowed to write to any of the 
       managed arrays without doing so as part of a transaction. 
       
       So locking of the managed arrays for write may be done only
       through the transaction process, or when executing a registered
       dependency update. 

       Locking the managed arrays for read should generally be done 
       through trm::lock_arrays_read(), which will always get you 
       a consistent set and which will also minimize the risk of 
       starving the write processes of access. */

  public:
    std::shared_ptr<rwlock> transaction_update_lock; /* Allows only one transaction at a time. Locked BEFORE any read or write locks acquired by a process 
							that will write. Write lock automatically acquired and placed in transaction_update_writelock_holder 
						        during start_transaction().... Not acquired as part of a locking process */
    /* NOTE: We rely on the fact that rwlock's can be unlocked by a different thread when End_Transaction() delegates release of this lock to a thread 
       that waits for all transaction functions to complete */
    
    
    std::unique_lock<rwlock_lockable> transaction_update_writelock_holder; // access to this holder is managed by dependency_table_lock

    std::atomic<size_t> state; /* TRMS_IDLE, TRMS_TRANSACTION, or TRMS_DEPENDENCY */ 

    std::atomic<snde_index> currevision;
    
    std::mutex dependency_table_lock; /* Must be final lock in order (after transaction_update_lock and after locking of arrays; locks dependencies, contents of dependencies,  and dependency execution tracking variables */
    std::unordered_set<std::shared_ptr<trm_dependency>> dependencies; /* list of all dependencies */
    
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
    
    std::unordered_set<std::shared_ptr<trm_dependency>> unsorted; /* not yet idenfified into one of the other categories */
    std::unordered_set<std::shared_ptr<trm_dependency>> unexecuted; /* will need execution but haven't figured out if we have to wait on deps or not */
    std::unordered_set<std::shared_ptr<trm_dependency>> no_need_to_execute; /* no (initial) need to execute, but may still be dependent on something */
    
    std::unordered_set<std::shared_ptr<trm_dependency>> unexecuted_with_deps;
    std::unordered_set<std::shared_ptr<trm_dependency>> unexecuted_no_deps;
    std::unordered_set<std::shared_ptr<trm_dependency>> executing;
    std::unordered_set<std::shared_ptr<trm_dependency>> done;

    /* locked by dependency_table_lock; modified_db is a database of which array
       regions have been modified during this transaction. Should be
       cleared at end of transaction */
    std::unordered_map<void **,std::pair<std::shared_ptr<arraymanager>,rangetracker<markedregion>>> modified_db;


    std::condition_variable job_to_do; /* associated with dependency_table_lock  mutex */
    std::condition_variable jobs_done; /* associated with dependency_table_lock mutex... indicator used by transaction_wait_thread to see when it is time to wrap-up the transaction */
    std::condition_variable all_done; /* associated with dependency_table_lock mutex... indicator that the transaction is fully wrapped up and next one can start */
    std::vector<std::thread> threadpool;

    std::thread transaction_wait_thread; /* This thread waits for all computation functions to complete, then releases transaction_update_lock */

    bool threadcleanup; /* guarded by dependency_table_lock */

    trm(const trm &)=delete; /* copy constructor disabled */
    trm& operator=(const trm &)=delete; /* copy assignment disabled */

    
    trm(int num_threads=-1)
    {
      currevision=1;
      threadcleanup=false;
      state=TRMS_IDLE;
      transaction_update_lock=std::make_shared<rwlock>();

      if (num_threads==-1) {
	num_threads=std::thread::hardware_concurrency();
      }

      for (size_t cnt=0;cnt < num_threads;cnt++) {
	threadpool.push_back(std::thread( [ this ]() {
	      std::unique_lock<std::mutex> dep_tbl(dependency_table_lock);
	      for (;;) {
		job_to_do.wait(dep_tbl,[ this ]() { return threadcleanup || (unexecuted_no_deps.size() > 0 && state==TRMS_DEPENDENCY); } );

		if (threadcleanup) {
		  return; 
		}
		
		auto job = unexecuted_no_deps.begin(); /* iterator pointing to a dependency pointer */
		

		if (job != unexecuted_no_deps.end()) {
		  size_t changedcnt=0;

		  std::shared_ptr<trm_dependency> job_ptr = *job;
		  
		  //std::vector<std::vector<trm_arrayregion>> inputchangedregions;
		  size_t inputcnt=0;
		  for (auto & input: job_ptr->inputs) {
		    //std::vector<trm_arrayregion> inputchangedregion=_modified_regions(input);
		    _merge_modified_regions(job_ptr->inputchangedregion[inputcnt],input);
		    //inputchangedregions.push_back(inputchangedregion);

		    job_ptr->inputchangedregion[inputcnt].merge_adjacent_regions();
		    
		    changedcnt += job_ptr->inputchangedregion[inputcnt].size(); /* count of changed regions in this input */

		    inputcnt++;
		  }
		  /* Here is where we call the dependency function ... but we need to be able to figure out the parameters. 
		     Also if the changes did not line up with the 
		     dependency inputs it should be moved to no_need_to_execute
		     instead */

		  
		  std::vector<rangetracker<markedregion>> outputchangedregions;

		  if (changedcnt > 0) {

		    unexecuted_no_deps.erase(job);
		    executing.insert(job_ptr);		    
		    dep_tbl.unlock();
		    job_ptr->outputs=job_ptr->update_output_regions(job_ptr->inputs,job_ptr->outputs);
		    outputchangedregions=job_ptr->function(this->currevision,job_ptr,job_ptr->inputchangedregion);
		    dep_tbl.lock();
		    executing.erase(job_ptr);
		    done.insert(job_ptr);
		    
		  } else {

		    unexecuted_no_deps.erase(job);

		    no_need_to_execute.insert(job_ptr);
		  }


		  size_t outcnt=0;
		  for (auto & ocr_entry: outputchangedregions) {
		    _mark_regions_as_modified(job_ptr->outputs[outcnt].manager,job_ptr->outputs[outcnt].array,ocr_entry);
		    

		    if (ocr_entry.size() > 0) {
		      for (auto & outdep: job_ptr->output_dependencies[outcnt]) {
			/* !!!*** Is there any way to check whether we have really messed with an input of this output dependency? */
			_call_regionupdater(outdep);
			
		      }
		    }

		    
		    for (auto & outdep: job_ptr->output_dependencies[outcnt]) {
		      if (unexecuted_with_deps.count(outdep)) {
			/* this still needs to be executed */
			/* are all of its input dependencies complete? */
			bool deps_complete=true;
			
			for (size_t indepcnt=0;indepcnt < outdep->input_dependencies.size();indepcnt++) {
			  for (auto & indep: outdep->input_dependencies[indepcnt]) {
			    if (executing.count(indep) || unexecuted_with_deps.count(indep) || unexecuted_no_deps.count(indep)) {
			      deps_complete=false;
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
		    outcnt++;
		  }


		  /*  signal job_to_do condition variable
		     according to the (number of entries in
		     unexecuted_no_deps)-1.... because if there's only
		     one left, we can handle it ourselves when we loop back */

		  size_t njobs=unexecuted_no_deps.size();
		  while (njobs > 1) {
		    job_to_do.notify_one();
		    njobs--;
		  }

		  if (njobs==0) {
		    jobs_done.notify_all();
		  }
		}
		
	      }
	      
	    }));
      }

      transaction_wait_thread=std::thread([ this ]() {
					    std::unique_lock<std::mutex> dep_tbl(dependency_table_lock);
					    for (;;) {
					      jobs_done.wait(dep_tbl,[ this ]() { return (unexecuted_no_deps.size()==0 && unexecuted_with_deps.size()==0 && executing.size()==0 && state==TRMS_DEPENDENCY) || threadcleanup;});

					      if (threadcleanup) {
						return; 
					      }


					      assert(state==TRMS_DEPENDENCY);
					      // all jobs are done. Now the transaction_update_writelock_holder can be released
					      std::unique_lock<rwlock_lockable> holder;

					      // Clear out modified_db
					      modified_db.clear();
					      
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
	std::lock_guard<std::mutex> dep_tbl(dependency_table_lock);
	threadcleanup=true;
	job_to_do.notify_all();
	jobs_done.notify_all();
      }
      for (size_t cnt=0;cnt < threadpool.size();cnt++) {
	threadpool[cnt].join();	
      }
      transaction_wait_thread.join();
      
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
      
    void _remove_depgraph_node_edges(std::shared_ptr<trm_dependency> dependency)
    /* Clear out the graph node edges that impinge on dependency, 
       based on its professed inputs and outputs */
    /* dependency_table_lock must be held by caller */
    {
      for (auto & old_input: dependency->input_dependencies) {
	for (auto & old_input_dep: old_input) { // step through the set for this input
	  for (size_t outcnt=0;outcnt < old_input_dep->output_dependencies.size();outcnt++) {
	    if (old_input_dep->output_dependencies[outcnt].count(dependency)) {
	      old_input_dep->output_dependencies[outcnt].erase(dependency);
	    }
	  }
	}
      }

      for (auto & old_output: dependency->output_dependencies) {
	for (auto & old_output_dep: old_output) { // step through the set for this output
	  for (size_t inpcnt=0;inpcnt < old_output_dep->input_dependencies.size();inpcnt++) {
	    if (old_output_dep->input_dependencies[inpcnt].count(dependency)) {
	      old_output_dep->input_dependencies[inpcnt].erase(dependency);
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
      _remove_depgraph_node_edges(dependency);

      /* make sure dependency has a big enough input_dependencies array */
      while (dependency->input_dependencies.size() < dependency->inputs.size()) {
	dependency->input_dependencies.emplace_back();
      }
      
      /* Iterate over all existing dependencies we could have a relationship to */
      for (auto & existing_dep: dependencies) {

	/* make sure existing dep has a big enough output_dependencies array */
	while (existing_dep->output_dependencies.size() < existing_dep->outputs.size()) {
	  existing_dep->output_dependencies.emplace_back();
	}
	
	
	
	/* For each of our input dependencies, does the existing dependency have an output
	   dependency? */
	size_t inpcnt=0;
	for (auto & input: dependency->inputs) { // input is a trm_arrayregion
	  
	  
	  auto & this_input_depset = dependency->input_dependencies[inpcnt];
	  for (size_t outcnt=0;outcnt < existing_dep->outputs.size();outcnt++) {

	    
	    if (input.overlaps(existing_dep->outputs[outcnt])) {
	      this_input_depset.emplace(existing_dep);
	      existing_dep->output_dependencies[outcnt].emplace(dependency);
	    } 
	  }
	  inpcnt++;
	}

	/* For each of our output dependencies, does the existing dependency have an input
	   dependency? */
	size_t outcnt=0;
	for (auto & output: dependency->outputs) {
	  auto & this_output_depset = dependency->output_dependencies[outcnt];
	  for (inpcnt=0;inpcnt < existing_dep->inputs.size();inpcnt++) {
	    if (existing_dep->inputs[inpcnt].overlaps(output)) {
	      this_output_depset.emplace(existing_dep);
	      existing_dep->input_dependencies[inpcnt].emplace(dependency);
	    }
	  }
	  outcnt++;
	}
      }
      
    }
    
    std::shared_ptr<trm_dependency>  add_dependency(std::function<std::vector<rangetracker<markedregion>>(snde_index newversion,std::shared_ptr<trm_dependency> dep,std::vector<rangetracker<markedregion>> &inputchangedregions)> function,
								 std::function<std::vector<trm_arrayregion>(std::vector<trm_arrayregion> inputs)> regionupdater,
								 std::vector<trm_arrayregion> inputs, // inputs array does not need to be complete; will be passed immediately to regionupdater() -- so this need only be a valid seed. 
								 //std::vector<trm_arrayregion> outputs)
								 std::function<std::vector<trm_arrayregion>(std::vector<trm_arrayregion> inputs,std::vector<trm_arrayregion> outputs)> update_output_regions, //,rwlock_token_set all_locks),
								 std::shared_ptr<arraymanager> manager)
    {
      /* Add a dependency outside StartTransaction()...EndTransaction()... First execution opportunity will be at next call to EndTransaction() */
      /* acquire necessary read lock to allow modifying dependency tree */
      std::lock_guard<rwlock_lockable> ourlock(transaction_update_lock->reader);
      return add_dependency_during_update(function,
					  regionupdater,
					  inputs,
					  update_output_regions,
					  manager);
    }

    void _categorize_dependency(std::shared_ptr<trm_dependency> dependency)
    {
      /* During EndTransaction() or equivalent we have to move each dependency 
	 where it belongs. This looks at the inputs and outputs of the given dependency, 
	 which should be in unsorted, and moves into unexecuted (and calls its
	 regionupdater) if an immediate need
	 to execute is identified, or into no_need_to_execute otherwise.

	 The dependency_table_lock should be locked in order to call this 
	 method.
      */

      bool modified_input_dep=false;
      assert(unsorted.count(dependency)==1);
      //assert(dependency->pending_input_dependencies.empty());
      
      unsorted.erase(dependency);
      
      for (auto & input: dependency->inputs) {
	if (_region_in_modified_db(input)) {
	  modified_input_dep=true;
	  //dependency.pending_input_dependencies.push_back();
	}
      }
      if (modified_input_dep) {
	/* temporarily mark with no_deps... will have to walk 
	   dependency tree and shift to unexecuted_with_deps
	   (if appropriate) later */
	
	/* Call regionupdater function and update dependency graph if necessary */ 
	_call_regionupdater(dependency); 
	
	unexecuted.insert(dependency);
      } else {
	no_need_to_execute.insert(dependency);
      }
      
    }

    void _figure_out_unexecuted_deps()
    {
      /* Figure out whether the dependencies listed in unexecuted should 
       go into unexecuted_with_deps or unexecuted_no_deps 
       
       The dependency_table_lock should be locked in order to call this 
       method.
      */
      
      /* Iterate recursively over unexecuted's output dependencies, move them into 
	 unexecuted_with_deps. ... be careful about iterator validity
	 
	 Anything that remains in unexecuted can be shifted into unexecuted_no_deps */
      
      std::vector<std::shared_ptr<trm_dependency>> unexecuted_copy(unexecuted.begin(),unexecuted.end());

      
      
      for (auto & dependency: unexecuted_copy) {
	_output_deps_into_unexecwithdeps(dependency,false);
      }
      
      /* shift any that remain in unexecuted into unexecuted_no_deps */
      std::vector<std::shared_ptr<trm_dependency>> unexecuted_copy2(unexecuted.begin(),unexecuted.end());
      for (auto & dependency: unexecuted_copy2) {
	unexecuted.erase(dependency);
	unexecuted_no_deps.insert(dependency);
      }
      
    }
    
    

    void remove_dependency(std::shared_ptr<trm_dependency> dependency)
    {
      /* Can only remove dependency while an update is not in progress */
      std::lock_guard<rwlock_lockable> ourlock(transaction_update_lock->reader);

      /* must hold dependency_table_lock */ 
      std::lock_guard<std::mutex> dep_tbl(dependency_table_lock);
      
      _remove_depgraph_node_edges(dependency);


      /* remove from full list of all dependencies */
      dependencies.erase(dependency);

      /* remove from queue to execute */
      unsorted.erase(dependency); 
    }
    
    /* add_dependency_during_update may only be called during a transaction */
    /* MUST HOLD WRITE LOCK for all output_arrays specified... may reallocate these arrays! */
    std::shared_ptr<trm_dependency> add_dependency_during_update(std::function<std::vector<rangetracker<markedregion>>(snde_index newversion,std::shared_ptr<trm_dependency> dep,std::vector<rangetracker<markedregion>> &inputchangedregions)> function,
								 std::function<std::vector<trm_arrayregion>(std::vector<trm_arrayregion> inputs)> regionupdater,
								 std::vector<trm_arrayregion> inputs, // inputs array does not need to be complete; will be passed immediately to regionupdater() -- so this need only be a valid seed. 
								 //std::vector<trm_arrayregion> outputs)
								 std::function<std::vector<trm_arrayregion>(std::vector<trm_arrayregion> inputs,std::vector<trm_arrayregion> outputs)> update_output_regions, //,rwlock_token_set all_locks),
								 std::shared_ptr<arraymanager> manager)
    /* May only be called while holding transaction_update_lock, either as a reader(maybe?) or as a writer */
    {

      /* Construct inputs */
      inputs=regionupdater(inputs);
      /* construct empty output regions */
      std::vector<trm_arrayregion> outputs; // start with blank output array
      outputs=update_output_regions(inputs,outputs);
      
      std::shared_ptr<trm_dependency> dependency=std::make_shared<trm_dependency>(function,regionupdater,update_output_regions,inputs,outputs);

      std::lock_guard<std::mutex> dep_tbl(dependency_table_lock);
      
      /*  Check input and output dependencies; 
	  if we are inside a transactional update and there are 
	  no unexecuted dependencies, we should drop into unexecuted_no_deps, unexecuted_with_deps, no_need_to_execute, etc. instead of unsorted */

      _call_regionupdater(dependency); // Make sure we have full list of dependencies.
      
      // Fill inputchangedregions with full trackers according to number of inputs
      for (size_t inpcnt=0;inpcnt < dependency->inputs.size();inpcnt++) {
	dependency->inputchangedregion.emplace_back();
	dependency->inputchangedregion[inpcnt].mark_region(0,SNDE_INDEX_INVALID); // Mark entire block
      }
      
      
      dependencies.emplace(dependency);

      _rebuild_depgraph_node_edges(dependency);
      


      //unsorted.insert(dependency);
      unexecuted.insert(dependency);
      if (state==TRMS_DEPENDENCY) {
	//_categorize_dependency(dependency); (no longer need to categorize as from hereon it is already in unexecuted (MAY NEED TO CHANGE FOR ON-DEMAND DEPENDENCIES!!!) 
	
	//_figure_out_unexecuted_deps();
	/* See if this is dependent on anything with pending execution */
	bool have_input_dependency=false;
	for (size_t inpcnt=0;inpcnt < dependency->input_dependencies.size();inpcnt++) {
	  for (auto & input_dependency: dependency->input_dependencies[inpcnt]) {
	    assert(!executing.count(input_dependency)); /* if this triggers, it would be a potential race condition */
	    if (unexecuted_with_deps.count(input_dependency) || unexecuted_no_deps.count(input_dependency)) {
	      have_input_dependency=true;
	    }
	  }
	}
	//unsorted.erase(dependency);
	unexecuted.erase(dependency);
	if (have_input_dependency) {
	  unexecuted_with_deps.insert(dependency);
	} else {
	  unexecuted_no_deps.insert(dependency);
	}
	_output_deps_into_unexecwithdeps(dependency,false);
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
	if (unexecuted.count(dependency)) {
	  unexecuted.erase(dependency);
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
	  _output_deps_into_unexecwithdeps(outdep,true);
	}
      }
    }

    void Start_Transaction()
    {
      std::unique_lock<rwlock_lockable> ourlock(transaction_update_lock->writer);
      

      {
	std::lock_guard<std::mutex> dep_tbl(dependency_table_lock);
	
	assert(!transaction_update_writelock_holder.owns_lock());
	
	assert(unexecuted.empty());
	assert(no_need_to_execute.empty());
	assert(unexecuted_with_deps.empty());
	assert(unexecuted_no_deps.empty());
	assert(executing.empty());
	assert(done.empty());
	
	assert(modified_db.empty());

	
	state=TRMS_TRANSACTION;
	
	currevision++; 
	
	// Move transaction lock to holder 
	ourlock.swap(transaction_update_writelock_holder);


	for (auto & dependency : dependencies) {
	  // Clear out inputchangedregions
	  dependency->inputchangedregion.empty();
	  
	  //for (auto & icr : dependency->inputchangedregion) {
	  //  icr.clear_all();
	  //}

	  // Fill inputchangedregions with empty trackers according to number of inputs
	  for (size_t inpcnt=0;inpcnt < dependency->inputs.size();inpcnt++) {
	    dependency->inputchangedregion.emplace_back(); 
	  }
	  
	}

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

    void Transaction_Mark_Modified(const trm_arrayregion &modified)
    {
      std::lock_guard<std::mutex> dep_tbl(dependency_table_lock);

      _mark_region_as_modified(modified);
      
    }


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

    
    void Transaction_Mark_Modified(std::vector<trm_arrayregion> &modified)
    {
      std::lock_guard<std::mutex> dep_tbl(dependency_table_lock);
      
      _mark_regions_as_modified(modified);
      
    }

    void Transaction_Mark_Modified(std::shared_ptr<std::vector<trm_arrayregion>> modified)
    {
      std::lock_guard<std::mutex> dep_tbl(dependency_table_lock);
      
      _mark_regions_as_modified(*modified);
      
    }

    /*void _Modified_Dependencies()
    {
      
    }*/


    void _call_regionupdater(std::shared_ptr<trm_dependency> dependency)
    /* Call the region updater code for a dependency. dependency_table_lock must be locked */
      
    {
      if (dependency->regionupdater) {
	std::vector<trm_arrayregion> newinputs;

	newinputs = dependency->regionupdater(dependency->inputs);

	if (!(newinputs == dependency->inputs)) {
	  /* NOTE: Do not change to != because operator== is properly overloaded but operator!= is not (!) */
	  dependency->inputs=newinputs;
	  _rebuild_depgraph_node_edges(dependency); 
	} 
      }
    }
    
    snde_index End_Transaction(std::shared_ptr<std::vector<trm_arrayregion>> modified)
    /* Can call Wait_Computation on returned revision # to wait for 
       computation to be complete */
    {
      std::unique_lock<rwlock_lockable> ourlock;
      snde_index retval=currevision;
      

      Transaction_Mark_Modified(modified);


      
      /* Now need to go through our dependencies and see which have been modified */
      {
	std::lock_guard<std::mutex> dep_tbl(dependency_table_lock);
	
	for (auto & dependency : dependencies) {

	  if (!unexecuted.count(dependency)) {
	    // if execution isn't already pending... 
	    _categorize_dependency(dependency);
	  }
	}

	_figure_out_unexecuted_deps();

      }

      state=TRMS_DEPENDENCY;
      /* Need to run execution process */
      
      {
	std::unique_lock<std::mutex> dep_tbl(dependency_table_lock);

	size_t njobs=unexecuted_no_deps.size();
	if (!njobs) {
	  jobs_done.notify_all(); /* if no jobs, notify anybody who is waiting so that we get our cleanup */
	}
	while (njobs > 0) {
	  job_to_do.notify_one();
	  njobs--;
	}

	//jobs_done.wait( dep_tbl, [ this ]() { return unexecuted_no_deps.size()==0 && unexecuted_with_deps.size()==0 && executing.size()==0;});
	
      }

      // Computations may not be done.... need to Wait_Computation to be assured
      // of completions
      
      return retval;
    }

    void Wait_Computation(snde_index revnum)
    /* Wait for computation of revnum to complete */
    {
      std::unique_lock<std::mutex> dep_tbl(dependency_table_lock);
      assert(revnum <= currevision);
      if (revnum==currevision) {
	all_done.wait( dep_tbl, [ this,revnum ]() { return revnum < currevision || !transaction_update_writelock_holder.owns_lock();});
      }
    }

  };
  


};
#endif // SNDE_REVISION_MANAGER_HPP
