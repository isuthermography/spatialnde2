/* transactional revision manager */
/* ***!!!!  SHOULD REDESIGN WITH PROPER DATABASE INDEXED BY 
   WHAT THE FUNCTION IS DEPENDENT ON

   ***!!!! Need some way to declare dependence on recording metadata ***!!!


   ***!!! Need some way to auto-lock output at the end of transaction ***!!!   

   ***!!! SHOULD IMPLEMENT PLANNED TRANSACTIONS -- where only specified arrays 
can be directly modified. This way, multiple transactions can run in parallel 
provided only the last one will be locked. 


   ***!!! SHOULD IMPLEMENT SPECIFYING ON-DEMAND ARRAYS so that
   functions which only generate stuff in on-demand arrays get 
   postponed until output is needed 



   *** Should (probably not) implement lock "keys" whereby the revision manager acquires the keys, 
   and then only lockers with access to the keys are are allowed to lock the arrays. 
   Others have to wait...   ... acts as a prior lock in the locking order... 



New design concept requirements:
   * Dependency graph for mutable zones
   * Immutable zones with revisions can also exist in dependency graph
   * Immutable recordings also 
   * Recording headers/metadata can also exist in dependency graph
   CANCELLED: 
      * Planned transactions: Pre-specify what will be changed, what will be 
        accessed, what (if-any) output is locked at the end of the transaction
      * Output can (conceptually) be locked because a dependency implicitly owns its 
        outputs so nothing else should be able to modify the outputs unless/until 
        another transaction is started. So these requested outputs become 
        part of the planned transaction graph. 

        A planned transaction can proceed in parallel with other transactions if it
        is orthogonal to the other transactions. 
    * Track rectangular bounding box for changes through from initiated transaction to completion. 
    * Need library to perform intersections on bounding boxes. 
    * Need to tag recordings with their changed box since prior revision (metadata?)
    * Some math functions are marked to be automatically re-run in the viewer
      (live only) and probably don't have any data saved -- like ProcExpr/ProcRGBA -- Can we define helper math functions that assist with viewing the data? -- i.e. math function for background subtraction that evaluates the background average; then a viewing assist? -- Can geometry math such as normals and boxes fit into this framework? 
      * Tag ProcExpr/ProcRGBA/rendering math functions as "local only"
      * Any particular allocation space can only have "local only" or 
        non-"local only" mutable data but not both
	* Local only functions will usually generate mutable output in-place
        * Need a way to lock them once mutable output generated, to enable render before they are overwritten by next rev. 
        * Changes to local only functions do define a new global revision, which
          means that local and remote global revisions will get out of sync
          and a mapping will be required. 
      * Sub-recordings have their own data and revisions, etc. independent 
        of their parent. Change of parent does not imply change of 
        sub-recording. 
    * Execution graph: 
      * Mutable recordings and immutable recordings with self-dependency or ability to do partial recalc (possibly pure immutable recordings that can be bypassed with no input changes as well) all have a dependency on their prior revision
      * Exception for mutable recordings that are only mutable to decrease copying when only a sub-rectangle is changed. 
      * How to handle graphics arrays that aren't actually recordings? 
      * Mutable recordings have implicit dependency on all dependencies of their prior revision (this prevents mutable recording calculation of new revision from occuring before previous revision)
      * Keep graph of everything pending
        * With new revision, build new graph and then splice/graft it on to 
          current graph, or just add entries piecemeal (probably former; we 
          need to start on the left side, which may not otherwise be clear
	* But externally provided recordings may pop into being piecemeal 
          as they become ("Ready").  
    * Everything immediately executable goes onto work queues
      * Different work queues for different types of jobs (what if a particular
        job can be performed by multiple dispatch options?)
      * Dedicated work queue for particular function guarantees it will always 
        run in the same process context and therefore be able to cache 
        intermediate results that may need to be reused. 
    * Work queues dispatch jobs to destinations, with numbers of threads 
      bounded by resource managers. Multiple dispatchers may use the same
      resources (CPU cores from a (possibly limited affinity pool))
      * Tell each job how many cores/GPUs it should use (based on a declared parallel capability). 
    * Job prioritization according to (a) global revision number (older 
      revisions higher priority), (b) number of (immediate? downstream?)
      dependencies within its global revision revision. 
    * Some recordings are on-demand -- only generated if desired
    * Any recording with a mandatory self-dependency (including all 
      mutable recordings) cannot be on-demand. 
    * READY character of a global revision is dependent on recordings of interest: Just mandatory recordings or additional optional recordings? so when asking for READY you need to be specific. Some recordings (local only) not available remotely 
  * Math parser to implement expression logic and implicit intermediates? 
    (pygram-generated?)

  * Explicit mapping process between function returns and channels. 

  * Channels and pending calculation graph reference math functions 
    (any other references are weak) so that once a math function is no
    longer referenced, it goes away. 
  * Explicitly creating a math calculation defines the math channels, which
    then reference the math calculation. The math calculation maintains
    a reference to the channels it outputs on. The reference loop is 
    irrelevant because channels can not be deleted anyway
  * If the channel does not reference the math calculation, then the 
    math calculation is not the channel owner and should redirect its 
    output to an implicit hidden channel
  * Any dead or empty output channel goes to an implicit hidden channel
  * Implicit hidden channels store values of intermediates and ignored
    math function outputs. 
  * Implicit hidden channels are named uniquely according to their source
    and only weakly reference the source so the source is deleted when 
    there are no remaining explicit references
  * Like all channels, implicit hidden channels are permanent. 

  * Distinction between currently defined math function and an executing
    math function for a particular global revision, which may be different. 
  * Math functions should (be able to) keep track of execution CPU/GPU 
    resource use including OpenMP thread CPU usage, GPU usage, subprocess 
    CPU usage, and execution wall time. Data should be stored/profiled 
    according to the execution unit involved. 




 
Rethink identification of inputs and outputs e.g. SDTA_IDENTIFYINPUTS. What are the actual use cases and what do we really need? 
* Need to be able to propagate proposed input changes to corresponding output changes;
  ... But exact location of intermediates doesn't really matter in most cases ... Can we define a proxy for such situations?
  ... May not be able to calculate exact size until execution anyway
* Two classes of inputs: Inputs that may be messed with externally and inputs which are the outputs of revision_manager 
  functions. The latter can be represented by proxies rather than worrying about addresses and locations. 
* But we do, when all is said and done, have to be able to provide output addresses. 

* When an output moves location, do we have to push a notification immediately that the prior address is no longer valid? 
  ... Yes the option is valuable. 

How does TRM affect locking orders? Presence within a transaction? planned transaction? 
  Locking order:  transaction_update_lock -> individual arrays
  dependency_table_lock is after individual arrays in the locking order and should not 
  be held while callbacks are being called. 

Versioned immutable recordings: 
  Defined state vs. Ready state. 

Real time display: 
  * Snapshot current defined state; wait for calculations to complete and corresponding versioned immutables to become ready
    * Can only happen between transactions as transaction should appear atomic
    * Delays new planned transactions
    * Waits for current planned transactions to complete. 
    * Prevents changes/new transactions until rendering complete 
  * Problem: How to get simultaneity/ordering of updates that come in during such a delay???***!!!
  * Need to be able to register to capture each change!

Thought: Maybe if mutable fields are intermediates only: Final outputs generally immutable and raw inputs generally 
immutable, then the global semantics are immutable and a lot of these problems go away. 

*/

#include <unordered_set>
#include <set>
#include <typeinfo>
#include <typeindex>
#include <atomic>
#include "snde/gsl-lite.hpp"
#include "snde/arraymanager.hpp"


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

***!!! Should be modified to identify implicit struct dependencies when evaluating the 
dependency graph

***!!! Should rework to accommodate arrays that are changed outside a transaction

  */

  // forward declaration
  class trm;


class trm_struct_depend_keyimpl_base {
public:
  trm_struct_depend_keyimpl_base(const trm_struct_depend_keyimpl_base &)=delete; // no copy constructor
  trm_struct_depend_keyimpl_base & operator=(const trm_struct_depend_keyimpl_base &)=delete; // no copy assignment

  trm_struct_depend_keyimpl_base()
  {

  }
  
  // objects should be immutable
  virtual bool less_than(const trm_struct_depend_keyimpl_base &other) const
  {
    // return whether this object is less than other
    
    // in your implementation of less than, you may assume that
    // both parameters can be dynamically casted to your type.
    // (otherwise your less_than function wouldn't be called)
    return false;
  }
  
  virtual ~trm_struct_depend_keyimpl_base() {}
};

class trm_struct_depend_key
// index key class for identifying dependency on external structure...
// includes pointer to a trm_struct_depend_keyimpl_base from which
// the actual implementation is derived
{
public:
  std::shared_ptr<trm_struct_depend_keyimpl_base> keyimpl;

  
  trm_struct_depend_key(std::shared_ptr<trm_struct_depend_keyimpl_base> keyimpl) :
    keyimpl(keyimpl)
  {

  }
  
  // need operator < so this can be used as a set key.
  // use set rather than unordered set so that we can
  // use weak pointers as part of the key structure. 
  friend bool operator<(const trm_struct_depend_key &l,const trm_struct_depend_key &r)
  {
    // so we really don't care about the ordering so long as it
    // is consistent and unchanging based on the
    // fundamental fixed parameters
    // of the dependency -- e.g. type, channel name, etc.
    // stored in specializations of this
    // class. 

    // So we use the typeid operator to check the underlying
    // types (since we are polymorphic). If they are the
    // same we call the virtual less_than method. 
    // If they are different we use the ordering of
    // the types instead.

    const std::type_index tl = typeid(*l.keyimpl);
    const std::type_index tr = typeid(*r.keyimpl);

    if (tl==tr) {
      return l.keyimpl->less_than(*r.keyimpl);
    } else {
      return tl < tr; 
    }
  }
  
  
};

class trm_struct_depend_notifier
{
  // base class for notification component that tells us
  // when external structure has changed.

  // notifier has the potential to store the value(s) of interest and only
  // propagate the notification if the value has changed


  // to notify: Call  recipient->mark_struct_depend_as_modified(key);
public:
  
  std::weak_ptr<trm> recipient;
  trm_struct_depend_key key;
  trm_struct_depend_notifier(std::weak_ptr<trm> recipient,
			     trm_struct_depend_key key) :
    recipient(recipient),
    key(key)
  {

  }

  virtual void trm_notify(); // implementation in revision_manager.cpp
  
  virtual ~trm_struct_depend_notifier() {}
};


//  #defines for trm_dependency function actions
#define STDA_IDENTIFYINPUTS (1<<0)
#define STDA_IDENTIFYOUTPUTS (1<<1)
#define STDA_EXECUTE (1<<2)

  // When the dependency function is called, it is
  // with STDA_IDENTIFYINPUTS or
  // STDA_IDENTIFYINPUTS|STDA_IDENTIFYOUTPUTS
  // or STDA_IDENTIFYINPUTS|STDA_IDENTIFYOUTPUTS|STDA_EXECUTE
  // or STDA_CLEANUPOUTPUTS
  
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


static void trm_lock_arrayregions(std::shared_ptr<lockingprocess> lockprocess,const std::vector<trm_arrayregion> &to_lock,bool write=false)
{
  // lock the given array regions, following the locking order

  // Install the regions in an ordered map, ordered by locking position
  std::map<lockingposition,trm_arrayregion> ordered_regions;

  for (auto & region: to_lock) {
    lockindex_t arrayidx = region.manager->locker->_arrayidx()->at(region.array);
    ordered_regions.emplace(lockingposition(arrayidx,region.start,write),region);
  }

  // now preform locking
  for (auto & lockingposition_region: ordered_regions) {
    if (write) {
      lockprocess->get_locks_write_array_region(lockingposition_region.second.array,lockingposition_region.second.start,lockingposition_region.second.len);
    } else {
      lockprocess->get_locks_read_array_region(lockingposition_region.second.array,lockingposition_region.second.start,lockingposition_region.second.len);

    }
  }
}

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


  typedef std::pair<trm_struct_depend_key,std::shared_ptr<trm_struct_depend_notifier>> trm_struct_depend;
  
  class trm_dependency : public std::enable_shared_from_this<trm_dependency> { /* dependency of one memory region on another */
  public:
    std::weak_ptr<trm> revman;


    std::function<void(snde_index newversion,std::shared_ptr<trm_dependency> dep,const std::set<trm_struct_depend_key> &inputchangedstructs,const std::vector<rangetracker<markedregion>> &inputchangedregions,unsigned actions)> function;
    

    std::function<void(trm_dependency *dep)> cleanup;

    std::set<trm_struct_depend_key> inputchangedstructs;
    std::vector<rangetracker<markedregion>> inputchangedregions; // rangetracker for changed zones, for each input 
    std::vector<trm_struct_depend> struct_inputs;
    std::vector<trm_arrayregion> inputs;
    std::vector<trm_struct_depend> struct_outputs;
    std::vector<trm_arrayregion> outputs;
    trm_struct_depend implicit_trm_trmdependency_output;
    
    std::vector<std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>>> input_dependencies; /* vector of input dependencies,  ordered per metadatainput then per input... These are sets because of possible overlap of regions or one output being used by multiple inputs */
    std::vector<std::set<std::weak_ptr<trm_dependency>,std::owner_less<std::weak_ptr<trm_dependency>>>> output_dependencies; /* vector of output dependencies, per metadataoutput then per output  */

    std::weak_ptr<trm_dependency> weak_this; // used in destructor
    bool force_full_rebuild; // used to trigger full rebuild of all outputs for newly created dependency

    /* pending_input_dependencies is only valid during a transaction, and lists
       input dependencies that will be modified by other dependencies */
    //std::vector<std::weak_ptr<trm_dependency>> pending_input_dependencies;

    

    trm_dependency(std::shared_ptr<trm> revman,
		   std::vector<trm_struct_depend> struct_inputs,
		   std::vector<trm_arrayregion> inputs,
		   std::vector<trm_struct_depend> struct_outputs,
		   std::vector<trm_arrayregion> outputs,
		   std::function<void(snde_index newversion,std::shared_ptr<trm_dependency> dep,const std::set<trm_struct_depend_key> &inputchangedstructs,const std::vector<rangetracker<markedregion>> &inputchangedregions,unsigned actions)> function,
		   std::function<void(trm_dependency *dep)> cleanup) :
      revman(revman),
      struct_inputs(struct_inputs),
      inputs(inputs),
      struct_outputs(struct_outputs),
      outputs(outputs),
      function(function),
      cleanup(cleanup),
      force_full_rebuild(true),
      implicit_trm_trmdependency_output(std::make_pair<trm_struct_depend_key,std::shared_ptr<snde::trm_struct_depend_notifier>>(trm_struct_depend_key(nullptr),std::shared_ptr<snde::trm_struct_depend_notifier>(nullptr)))
    {
      // weak_this=shared_from_this();

    }
		   
    trm_dependency(const trm_dependency &)=delete; // copy constructor disabled
    trm_dependency& operator=(const trm_dependency &)=delete; // copy assignment disabled
    
    ~trm_dependency(); // destructor in .cpp file to avoid circular class dependency


    bool update_struct_inputs(const std::vector<trm_struct_depend> &new_struct_inputs);

    bool update_struct_outputs(const std::vector<trm_struct_depend> &new_struct_outputs);

    
    bool update_inputs(const std::vector<trm_arrayregion> &new_inputs);
    bool update_outputs(const std::vector<trm_arrayregion> &new_outputs);

    std::tuple<lockholder_index,rwlock_token_set,std::string> realloc_output_if_needed(std::shared_ptr<lockingprocess> process,std::shared_ptr<arraymanager> manager,size_t outnum,void **output_array,snde_index numelements,std::string name)
    /* Reallocate output <outnum> of this dependency from the given array, with size of numelements, 
       unless the output is already set to a suitable allocation. Lock the output either way. 
       Return the lock in the lock holder under the specified name. 

       If the address in the lockholder matches the address already in the output array, 
       then the output array has not changed. If it is different, then the old array needs to 
       be freed and the output completely regenerated. 

    */
    {
      std::vector<std::tuple<lockholder_index,rwlock_token_set,std::string>> arrayregions;
     
      if (outputs.size() <= outnum || outputs.at(outnum).array != output_array || outputs[outnum].len < numelements) {
	arrayregions = process->alloc_array_region(manager,output_array,numelements,name);

	return arrayregions.at(0); // assumes that parallel allocations from other arrays don't matter or are handled separately
      }
    
      lockholder_index idx;
      rwlock_token_set tokens;
      std::tie(idx,tokens)=process->get_locks_write_array_region(output_array,outputs[outnum].start,outputs[outnum].len);
      if (outputs[outnum].len > numelements)  {
	manager->realloc_down(output_array,outputs[outnum].start,outputs[outnum].len, numelements);
      }

      return std::make_tuple(idx,tokens,name);
    }

    void add_output_to_array(std::vector<trm_arrayregion> &new_outputs,std::shared_ptr<arraymanager> manager,std::shared_ptr<lockholder> holder,size_t outnum,void **output_array,std::string output_name)
    // be sure to call update_outputs() after new_outputs is filled up!
    {
      assert(new_outputs.size()==outnum); // should be adding on to the end
      
      
      snde_index idx = holder->get_alloc(output_array,output_name);
      snde_index len = holder->get_alloc_len(output_array,output_name);

      
      if (outnum >= outputs.size()) {
	// not already in outputs array
	new_outputs.emplace_back(manager,output_array,idx,len);	
      } else {
	// already in outputs array... does it match?
	if (outputs.at(outnum).array != output_array || outputs.at(outnum).start != idx) {
	  // mismatch... must free old array
	  manager->free(outputs.at(outnum).array,outputs.at(outnum).start);
	}
	
	new_outputs.emplace_back(manager,output_array,idx,len);	
	
      }
      
    }

    void free_output(std::vector<trm_arrayregion> &new_outputs,size_t outnum)
    // for use during SDTA_CLEANUP for an output allocated by add_output_to_array()
    // be sure to call update_outputs() after new_outputs is filled up!
    {
      assert(new_outputs.size()==outnum); // should be adding on to the end
      if (outnum >= outputs.size()) { // not currently in outputs; nothing to do
	new_outputs.emplace_back(nullptr,nullptr,SNDE_INDEX_INVALID,0);
	
      } else {
	if (outputs.at(outnum).array && outputs.at(outnum).start != SNDE_INDEX_INVALID) {
	  // mismatch... must free old array
	  outputs.at(outnum).manager->free(outputs.at(outnum).array,outputs.at(outnum).start);
	}
	new_outputs.emplace_back(nullptr,nullptr,SNDE_INDEX_INVALID,0);

      }
    }
    
  };



// trm struct depend on another specific  trm_dependency:
class trm_trmdependency_key: public trm_struct_depend_keyimpl_base {
public:
  std::weak_ptr<trm_dependency> dependent_on;

  trm_trmdependency_key(const trm_trmdependency_key &)=delete; // no copy constructor
  trm_trmdependency_key & operator=(const trm_trmdependency_key &)=delete; // no copy assignment

  trm_trmdependency_key(std::shared_ptr<trm_dependency> dependent_on) :
    dependent_on(dependent_on)
  {
    
  }
  virtual bool less_than(const trm_struct_depend_keyimpl_base &other) const
  {
    // called to identify mapping location of the trm_struct_depend.
    // both l&r should be our class
    const trm_trmdependency_key *op = dynamic_cast<const trm_trmdependency_key *>(&other);

    assert(op);
    
    return dependent_on.owner_before(op->dependent_on);
    
  }

};


class trm_trmdependency_notifier: public trm_struct_depend_notifier {
  // inherited members:
  //   from trm_struct_depend_notifier: 
  //     std::weak_ptr<trm> recipient;
  //     trm_struct_depend_key key;
  //
  //  key has a member keyimpl that can be dynamically pointer casted to trm_trmdependency_key 
public:
  
  trm_trmdependency_notifier(const trm_trmdependency_notifier &)=delete; // no copy constructor
  trm_trmdependency_notifier & operator=(const trm_trmdependency_notifier &)=delete; // no copy assignment
  
  trm_trmdependency_notifier(std::shared_ptr<trm> recipient,std::shared_ptr<trm_dependency> dependent_on) :
    trm_struct_depend_notifier(recipient,trm_struct_depend_key(std::make_shared<trm_trmdependency_key>(dependent_on)))
    
  {

  }

  std::shared_ptr<trm_dependency> get_dependent_on()
  {
    std::shared_ptr<trm_trmdependency_key> keyimpl = std::dynamic_pointer_cast<trm_trmdependency_key>(key.keyimpl);
    std::shared_ptr<trm_dependency> dependent_on=keyimpl->dependent_on.lock();
    return dependent_on;
  }

  // note inherited method:
  //trm_notify();
  
  virtual ~trm_trmdependency_notifier() {}
};

static std::shared_ptr<trm_dependency> get_trmdependency(const trm_struct_depend &depend)
{
  // may return nullptr if dependency doesn't exist anymore
  std::shared_ptr<trm_trmdependency_notifier> notifier = std::dynamic_pointer_cast<trm_trmdependency_notifier>(depend.second);
  assert(notifier);
  
  return notifier->get_dependent_on();

}

static trm_struct_depend trm_trmdependency(std::shared_ptr<trm> revman, std::shared_ptr<trm_dependency> dependent_on)
{
  std::shared_ptr<trm_trmdependency_notifier> notifier = std::make_shared<trm_trmdependency_notifier>(revman,dependent_on);
  
  return std::make_pair(notifier->key,notifier);

}


  
  
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
    
    std::recursive_mutex dependency_table_lock; /* ordered after transaction_update_lock and after locking of arrays; locks dependencies, the table below of dependency references, and modified_db, modified_struct db*/
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
       version is defined must wait for new version.
       
       ... alternative concept: Output can be 'disabled' if nobody cares about it. 

    */
    
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

    /* locked by dependency_table_lock; modified_struct_db is a database of which structures we are 
       dependent on that  have been modified during (or before?) this transaction. Should be
       cleared at end of transaction */

    /* ***!!!!!! Needs to be redone: instead keep a table of dependencies which care about each
       region, or each structure and directly update the changed state of the corresponding dependencies!!!*** */
    
    //std::set<std::weak_ptr<mutableinfostore>,std::owner_less<std::weak_ptr<mutableinfostore>>> modified_metadata_db;
    //std::set<std::string> modified_metadata_db;
    std::set<trm_struct_depend_key> modified_struct_db;
    
    
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

    
    
    trm(const trm &)=delete; /* copy constructor disabled */
    trm& operator=(const trm &)=delete; /* copy assignment disabled */

    
    trm(int num_threads=-1)
    {
      currevision=1;
      threadcleanup=false;
      state=TRMS_IDLE;
      transaction_update_lock=std::make_shared<rwlock>();

      //recdb_notifier = std::make_shared<trm_recdirty_notification>(this);
      //recdb->add_dirty_notification_receiver(recdb_notifier);
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
		    executing_regionupdater.emplace(job_ptr);
		    
		    // note parallel code in add_dependency_during_update 
		    
		    std::vector<trm_arrayregion> oldinputs=job_ptr->inputs;
		    std::vector<trm_arrayregion> oldoutputs=job_ptr->outputs;
		    bool inputs_changed=false;
		    
		    
		    dep_tbl.unlock();		
		    job_ptr->function(0,job_ptr,std::set<trm_struct_depend_key>(),std::vector<rangetracker<markedregion>>(),STDA_IDENTIFYINPUTS|STDA_IDENTIFYOUTPUTS); 
		    dep_tbl.lock();
		    
		    if (!(oldinputs == job_ptr->inputs)) { /* NOTE: Do not change to != because operator== is properly overloaded but operator!= is not (!) */
		      inputs_changed=true;
		      _ensure_input_cachemanagers_registered(job_ptr->inputs);
		    }
		    
		    
		    if (inputs_changed || !(oldoutputs == job_ptr->outputs)) {
		      /* NOTE: Do not change to != because operator== is properly overloaded but operator!= is not (!) */			
		      
		      _rebuild_depgraph_node_edges(job_ptr); 
		    } 
		    executing_regionupdater.erase(job_ptr); //ex_ru_iter);
		    

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
		      for (size_t inpcnt=job_ptr->inputchangedregions.size();inpcnt < job_ptr->inputs.size();inpcnt++) {
			job_ptr->inputchangedregions.emplace_back(); 
		      }

		      
		      if (job_ptr->force_full_rebuild) {
			// force full rebuild: Mark everything as changed
			changedcnt=1;
			size_t inputcnt=0;
			for (auto & input: job_ptr->inputs) {
			  job_ptr->inputchangedregions[inputcnt].mark_region(0,SNDE_INDEX_INVALID);
			  changedcnt++;
			  inputcnt++;
			}
		      } else {
			size_t inputcnt=0;
			for (auto & input: job_ptr->inputs) {
			  //std::vector<trm_arrayregion> inputchangedregions=_modified_regions(input);
			  _merge_modified_regions(job_ptr->inputchangedregions[inputcnt],input);
			  //inputchangedregions.push_back(inputchangedregion);
			  
			  job_ptr->inputchangedregions[inputcnt].merge_adjacent_regions();
			  
			  changedcnt += job_ptr->inputchangedregions[inputcnt].size(); /* count of changed regions in this input */
			  
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
			std::vector<trm_arrayregion> oldinputs=job_ptr->inputs;
			std::vector<trm_arrayregion> oldoutputs=job_ptr->outputs;
			bool inputs_changed=false;
			bool outputs_changed=false;

			// clear out inputchangedregions and inputchangedstructs BEFORE calling function
			// so that we can accumulate any changes that might occur in the background
			// during our call, and therefore reexecute appropriately
			std::set<trm_struct_depend_key> inputchangedstructs(std::move(job_ptr->inputchangedstructs));
			std::vector<rangetracker<markedregion>> inputchangedregions(std::move(job_ptr->inputchangedregions));
			
			job_ptr->inputchangedregions.clear();
			job_ptr->inputchangedstructs.clear();
			
			for (size_t inpcnt=0;inpcnt < job_ptr->inputs.size();inpcnt++) {
			  job_ptr->inputchangedregions.emplace_back(); 
			}


			
			// call function with all flags set
			dep_tbl.unlock();		
			job_ptr->function(0,job_ptr,inputchangedstructs,inputchangedregions,STDA_IDENTIFYINPUTS|STDA_IDENTIFYOUTPUTS|STDA_EXECUTE);
			
			// notify that we have dirtied our implicit output
			
			job_ptr->implicit_trm_trmdependency_output.second->trm_notify();
			dep_tbl.lock();
			
			if (!(oldinputs == job_ptr->inputs)) { /* NOTE: Do not change to != because operator== is properly overloaded but operator!= is not (!) */
			  inputs_changed=true;
			  _ensure_input_cachemanagers_registered(job_ptr->inputs);
			}
			
			if (!(oldoutputs == job_ptr->outputs)) { /* NOTE: Do not change to != because operator== is properly overloaded but operator!= is not (!) */
			  outputs_changed=true;
			  
			}

			
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
					      
					      jobs_done.wait(dep_tbl,[ this ]() { return (unexecuted_no_deps.size()==0 && unexecuted_with_deps.size()==0 && unexecuted_needs_regionupdater.size()==0 && unexecuted_regionupdated.size()==0 && executing_regionupdater.size()==0 &&  executing.size()==0 && state==TRMS_DEPENDENCY) || threadcleanup;});

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
					      modified_struct_db.clear();

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

      //recdb_notifier=nullptr; // trigger deletion of recdb_notifier before we disappear ourselves, because it has a pointer to us. 

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
      while (dependency->input_dependencies.size() < dependency->struct_inputs.size() + dependency->inputs.size()) {
	dependency->input_dependencies.emplace_back();
      }

      /* make sure dependency has a big enough output_dependencies array */
      while (dependency->output_dependencies.size() < dependency->struct_outputs.size() + dependency->outputs.size() +1 ) { // last entry is the implicit struct output dependency that is a trm_trmdependency(dependent_on=ourselves)
	dependency->output_dependencies.emplace_back();
      }

      /* Iterate over all existing dependencies we could have a relationship to */
      for (auto & existing_dep: dependencies) {

	std::shared_ptr<trm_dependency> existing_dep_strong=existing_dep.lock();
	
	if (existing_dep_strong) {
	  
	  /* make sure existing dep has a big enough input_dependencies array */
	  while (existing_dep_strong->input_dependencies.size() < existing_dep_strong->struct_inputs.size() + existing_dep_strong->inputs.size()) {
	    existing_dep_strong->input_dependencies.emplace_back();
	  }

	  /* make sure existing dep has a big enough output_dependencies array */
	  while (existing_dep_strong->output_dependencies.size() < existing_dep_strong->struct_outputs.size() + existing_dep_strong->outputs.size() + 1) {
	    // last entry is the implicit struct output dependency that is a trm_trmdependency(dependent_on=ourselves)
	    existing_dep_strong->output_dependencies.emplace_back();
	  }
	
	  
	  
	  /* For each of our input structure dependencies, does the existing dependency have an output
	     dependency? */
	  size_t inpcnt=0;
	  for (auto & inputstruct: dependency->struct_inputs) { 

	    trm_struct_depend_key &inputkey=inputstruct.first;
	    
	    auto & this_input_depset = dependency->input_dependencies.at(inpcnt);
	    for (size_t outcnt=0;outcnt < existing_dep_strong->struct_outputs.size();outcnt++) {
	      // existing_dep_strong->struct_outputs.at(outcnt) is another trm_struct_depend_key...
	      // need to compare with input
		
	      // use (indirectly) owner_before() attributes for comparison so comparison is legitimate even if
	      // weak_pointers have been released.
	      const trm_struct_depend_key &outputkey = existing_dep_strong->struct_outputs.at(outcnt).first;
		
	      // basically we are looking for input==output
	      // expressed as !(input < output) && !(output < input)
	      // where input < output expressed as input.owner_before(output)
	      if (!(inputkey < outputkey)  && !(outputkey < inputkey)) {
		
		this_input_depset.emplace(existing_dep_strong);
		existing_dep_strong->output_dependencies.at(outcnt).emplace(dependency);
	      } 
	      
	    }
	    
	    const trm_struct_depend_key &outputkey = existing_dep_strong->implicit_trm_trmdependency_output.first;
		
	    // basically we are looking for input==output
	    // expressed as !(input < output) && !(output < input)
	    // where input < output expressed as input.owner_before(output)
	    if (!(inputkey < outputkey)  && !(outputkey < inputkey)) {
	      
	      this_input_depset.emplace(existing_dep_strong);
	      existing_dep_strong->output_dependencies.at(existing_dep_strong->struct_outputs.size() + existing_dep_strong->outputs.size()).emplace(dependency);
	    } 
	    
	    inpcnt++;
	  }

	  /* For each of our input array dependencies, does the existing dependency have an output
	     dependency? */
	  inpcnt=0;
	  for (auto & input: dependency->inputs) { // input is a trm_arrayregion
	    
	    
	    auto & this_input_depset = dependency->input_dependencies.at(dependency->struct_inputs.size() + inpcnt);
	    for (size_t outcnt=0;outcnt < existing_dep_strong->outputs.size();outcnt++) {
	      
	    
	      if (input.overlaps(existing_dep_strong->outputs[outcnt])) {
		this_input_depset.emplace(existing_dep_strong);
		existing_dep_strong->output_dependencies.at(existing_dep_strong->struct_outputs.size() +outcnt).emplace(dependency);
	      } 
	    }
	    inpcnt++;
	  }
	
	  /* For each of our output structure dependencies, does the existing dependency have an input
	     dependency? */
	  size_t outcnt=0;
	  for (auto & outputstruct: dependency->struct_outputs) { 
	    trm_struct_depend_key &outputkey=outputstruct.first;
	    
	    auto & this_output_depset = dependency->output_dependencies.at(outcnt);
	    for (inpcnt=0;inpcnt < existing_dep_strong->struct_inputs.size();inpcnt++) {
	      // existing_dep_strong->struct_inputs.at(inpcnt) is another trm_struct_depend_key
	      // need to compare with output
	      
	      // use (indirectly) owner_before() attributes for comparison so comparison is legitimate even if
	      // weak_pointers have been released.
	      const trm_struct_depend_key & inputkey = existing_dep_strong->struct_inputs.at(inpcnt).first;
		
	      // basically we are looking for input==output
	      // expressed as !(input < output) && !(output < input)
	      // where input < output expressed as input.owner_before(output)
	      if (!(inputkey < outputkey) && !(outputkey < inputkey)) {
		
		this_output_depset.emplace(existing_dep_strong);
		existing_dep_strong->input_dependencies.at(inpcnt).emplace(dependency);
	      } 
	      
		
	    }
	    
	    outcnt++;
	  }

	  trm_struct_depend_key &outputkey=dependency->implicit_trm_trmdependency_output.first;
	  
	  auto & this_output_depset = dependency->output_dependencies.at(dependency->struct_outputs.size() + dependency->outputs.size());
	  for (inpcnt=0;inpcnt < existing_dep_strong->struct_inputs.size();inpcnt++) {
	    // existing_dep_strong->struct_inputs.at(inpcnt) is another trm_struct_depend_key
	    // need to compare with output
	    
	    // use (indirectly) owner_before() attributes for comparison so comparison is legitimate even if
	    // weak_pointers have been released.
	    const trm_struct_depend_key & inputkey = existing_dep_strong->struct_inputs.at(inpcnt).first;
	    
	    // basically we are looking for input==output
	    // expressed as !(input < output) && !(output < input)
	    // where input < output expressed as input.owner_before(output)
	    if (!(inputkey < outputkey) && !(outputkey < inputkey)) {
	      
	      this_output_depset.emplace(existing_dep_strong);
	      existing_dep_strong->input_dependencies.at(inpcnt).emplace(dependency);
	    } 
	    
	    
	  }
	  
	  
	  
	  /* For each of our output array dependencies, does the existing dependency have an input
	     dependency? */
	  outcnt=0;
	  for (auto & output: dependency->outputs) {
	    auto & this_output_depset = dependency->output_dependencies.at(dependency->struct_outputs.size() + outcnt);
	    for (inpcnt=0;inpcnt < existing_dep_strong->inputs.size();inpcnt++) {
	      if (existing_dep_strong->inputs[inpcnt].overlaps(output)) {
		this_output_depset.emplace(existing_dep_strong);
		existing_dep_strong->input_dependencies.at(existing_dep_strong->struct_inputs.size() + inpcnt).emplace(dependency);
	      }
	    }
	    outcnt++;
	  }
	}
      }
      
    }
    
    std::shared_ptr<trm_dependency>  add_dependency(
						    std::vector<trm_struct_depend> struct_inputs,
						    std::vector<trm_arrayregion> inputs, // inputs array does not need to be complete; will be passed immediately to regionupdater() -- so this need only be a valid seed. 
						    //std::vector<trm_arrayregion> outputs)
						    std::vector<trm_struct_depend> struct_outputs,
						    
						    std::function<void(snde_index newversion,std::shared_ptr<trm_dependency> dep,const std::set<trm_struct_depend_key> &inputchangedstructs,const std::vector<rangetracker<markedregion>> &inputchangedregions,unsigned actions)> function,
						    std::function<void(trm_dependency *dep)> cleanup) // cleanup() should not generally do any locking but just free regions. 
    {
      /* Add a dependency outside StartTransaction()...EndTransaction()... First execution opportunity will be at next call to EndTransaction() */
      /* acquire necessary read lock to allow modifying dependency tree */
      std::lock_guard<rwlock_lockable> ourlock(transaction_update_lock->reader);
      return add_dependency_during_update(struct_inputs,
					  inputs,
					  struct_outputs,
					  function,
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
	for (auto & metadatainput: dependency->struct_inputs) {
	  if (modified_struct_db.find(metadatainput.first) != modified_struct_db.end()) {
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

	std::shared_ptr<trm_dependency> dep_strong=dependency.lock();
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
    /* PROBABLY OBSOLETE: MUST HOLD WRITE LOCK for all output_arrays specified... may reallocate these arrays! */
    std::shared_ptr<trm_dependency> add_dependency_during_update(std::vector<trm_struct_depend> struct_inputs,
								 std::vector<trm_arrayregion> inputs, // inputs array does not need to be complete; will be passed immediately to regionupdater() -- so this need only be a valid seed.
								 std::vector<trm_struct_depend> struct_outputs,
								 //std::vector<trm_arrayregion> outputs)
								 std::function<void(snde_index newversion,std::shared_ptr<trm_dependency> dep,const std::set<trm_struct_depend_key> &inputchangedstructs,const std::vector<rangetracker<markedregion>> &inputchangedregions,unsigned actions)> function,
								 std::function<void(trm_dependency *dep)> cleanup) // cleanup() should not generally do any locking but just free regions. 
    /* May only be called while holding transaction_update_lock, either as a reader(maybe?) or as a writer */
    {
      
      std::vector<trm_arrayregion> outputs; // start with blank output array
      
      std::shared_ptr<trm_dependency> dependency=std::make_shared<trm_dependency>(shared_from_this(),struct_inputs,inputs,struct_outputs,outputs,function,cleanup);
      
      /* Update inputs/outputs */
      _ensure_input_cachemanagers_registered(inputs);
      
      
      dependency->weak_this = dependency; // couldn't be set in constructor because you can't call shared_form_this() in constructor, but it is needed in the destructor and can't be created there either(!)
      
      std::unique_lock<std::recursive_mutex> dep_tbl(dependency_table_lock);
      
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
      
      dependency->implicit_trm_trmdependency_output = trm_trmdependency(shared_from_this(),dependency);
      
      dependencies.emplace(dependency);
      
      _rebuild_depgraph_node_edges(dependency);
      
      // Now run the regionupdater and update_output_regions methods..... Have to release the locks, run them
      // like they would be in the sub-threads. (note parallel code in subthread)
      executing_regionupdater.emplace(dependency);
      
      		      
      std::vector<trm_arrayregion> oldinputs=dependency->inputs;
      std::vector<trm_arrayregion> oldoutputs=dependency->outputs;
      bool inputs_changed=false;
      
      dep_tbl.unlock();		
      dependency->function(0,dependency,std::set<trm_struct_depend_key>(),std::vector<rangetracker<markedregion>>(),STDA_IDENTIFYINPUTS|STDA_IDENTIFYOUTPUTS); 
      dep_tbl.lock();
      
      if (!(oldinputs == dependency->inputs)) { /* NOTE: Do not change to != because operator== is properly overloaded but operator!= is not (!) */
	inputs_changed=true;
	_ensure_input_cachemanagers_registered(dependency->inputs);
      }
      
      if (inputs_changed || !(oldoutputs == dependency->outputs)) {
	/* NOTE: Do not change to != because operator== is properly overloaded but operator!= is not (!) */			
	
	_rebuild_depgraph_node_edges(dependency); 
      } 
      executing_regionupdater.erase(dependency); //ex_ru_iter);
      
      for (size_t inpcnt=0;inpcnt < dependency->inputs.size();inpcnt++) {
	dependency->inputchangedregions.emplace_back(); 
      }
      
      
      if (state==TRMS_DEPENDENCY || state==TRMS_REGIONUPDATE) {
	unexecuted_regionupdated.insert(dependency);
	if (state==TRMS_DEPENDENCY) {
	  _figure_out_unexecuted_deps();
	}
      } else {
	unsorted.insert(dependency);	
      }
      
      
      

      
      return dependency;
    }

    void _output_deps_into_unexecwithdeps(std::shared_ptr<trm_dependency> dependency,bool is_an_output_dependency)
    /* Iterate recursively over dependency and its output dependencies of <dependency> moving them 
       from the unexecuted or no_need_to_execute list (if present) into the unexecuted_with_deps set if they have dependencies... The dependency_table lock should be hld while calling this... */
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
      //assert(modified_struct_db.empty());
      
      
      state=TRMS_TRANSACTION;
      
      currevision++;
      //fprintf(stderr,"_Start_Transaction(%u)\n",(unsigned)currevision);
      
      // Move transaction lock to holder 
      ourlock.swap(transaction_update_writelock_holder);
      
      
      //      for (auto & dependency : dependencies) {
      //// Clear out inputchangedregions (NOW DONE IMMEDIATELY AFTER EXECUTION)
      //std::shared_ptr<trm_dependency> dep_strong=dependency.lock();
      //if (dep_strong) {
	  //dep_strong->inputchangedregions.clear();
	
	  //for (auto & icr : dependency->inputchangedregion) {
	  //  icr.clear_all();
	  //}
	  
	// Fill inputchangedregions with empty trackers according to number of inputs
      //}
      /// }
      
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

    void _mark_struct_depend_as_modified(const trm_struct_depend_key &struct_key)
    {
      /* dependency_table_lock must be locked when this function is called */
      auto dbregion = modified_struct_db.find(struct_key);

      if (dbregion==modified_struct_db.end()) {
	/* this infostore not currently marked as modified */
	//assert(state==TRMS_TRANSACTION || state==TRMS_DEPENDENCY);
	modified_struct_db.emplace(struct_key);
      }
      
    }

    void mark_struct_depend_as_modified(const trm_struct_depend_key &struct_key)
    {
      std::lock_guard<std::recursive_mutex> dep_tbl(dependency_table_lock);
      _mark_struct_depend_as_modified(struct_key);
    }
    
    void _mark_region_as_modified(const trm_arrayregion &modified)
    {
      /* dependency_table_lock must be locked when this function is called */

      fprintf(stderr,"_mark_region_as_modified(array=0x%llx; start=%d; len=%d\n",(unsigned long long)((void *)modified.array),(int)modified.start,(int)modified.len);
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
    //

    // ***!!!!! We need to fix so we never clear the modified_db,
    // just move stuff out of it. That way we will never miss a modification ***!!!
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
