%pythonbegin %{
import types as pytypes
%}

%shared_ptr(snde::lockmanager);
%shared_ptr(snde::rwlock_token_content);
%shared_ptr(snde::rwlock_token_set_content);
%shared_ptr(std::vector<snde::rangetracker<snde::markedregion>>);



%template(LockingPositionMap) std::multimap<snde::lockingposition,snde::CountedPyObject>;

%extend std::multimap<snde::lockingposition,snde::CountedPyObject> {
  void emplace_pair(std::pair<snde::lockingposition,snde::CountedPyObject> p)
  {
    self->emplace(p);
  }
}



%template(Region) snde::rangetracker<snde::markedregion>;

%template(VectorOfRegions) std::vector<snde::rangetracker<snde::markedregion>>;
%template(lockingposition_generator) std::pair<snde::lockingposition,snde::CountedPyObject>; 

//%template(voidpp_posn_map) std::unordered_map<void **,size_t>;
%template(voidpp_posn_map) std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>,std::allocator< std::pair< void **const,size_t > > >;

%extend std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>,std::allocator< std::pair< void **const,size_t > > > {
  bool has_key(ArrayPtr key) {
    if (self->find(key)==self->end()) return false;
    return true;
  }
};



size_t voidpp_posn_map_iterator_posn(std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>,std::allocator< std::pair< void **const,size_t > > >::iterator);

%{
size_t voidpp_posn_map_iterator_posn(std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>,std::allocator< std::pair< void **const,size_t > > >::iterator self) {
    return self->second;

}
%}

snde::ArrayPtr voidpp_posn_map_iterator_ptr(std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>,std::allocator< std::pair< void **const,size_t > > >::iterator);
%{
snde::ArrayPtr voidpp_posn_map_iterator_ptr(std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>,std::allocator< std::pair< void **const,size_t > > >::iterator self)
{
  return self->first;
}
%}


/*
// template iterator workaround per http://www.swig.org/Doc1.3/SWIGPlus.html#SWIGPlus_nested_classes
%{
  
  typedef std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>,std::allocator< std::pair< void **const,size_t > > >::iterator voidpp_posn_map_iterator;
%}

class voidpp_posn_map_iterator {
  voidpp_posnmap_iterator(voidpp_posn_map_iterator &);

  //~voidpp_posnmap_iterator();
};

voidpp_posn_map_iterator voidpp_posn_map_iterator_fromiterator(std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>,std::allocator< std::pair< void **const,size_t > > >::iterator it);
%{
  voidpp_posn_map_iterator voidpp_posn_map_iterator_fromiterator(std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>,std::allocator< std::pair< void **const,size_t > > >::iterator it)
  {
    return it;
  }
%}

%extend voidpp_posn_map_iterator {
  ArrayPtr get_ptr() {
    return (*self)->first;
  }
  size_t get_posn() {
    return (*self)->second;
  }
}
*/


%shared_ptr(LockingPositionMap);
%shared_ptr(VectorOfRegions);

%{
  namespace swig {
    template <> struct traits<void>
    {
      typedef pointer_category category;
      static const char *type_name()
      {
        return "void";
      }

    };
  }  
%}

%template(arrayvector) std::vector<void **>;

%{
  
#include "lockmanager.hpp"
%}


namespace snde {

  class rwlock;  // forward declaration

  class rwlock_lockable {
    // private to rwlock
  public:
    int _writer;
    rwlock *_rwlock_obj;
    
    rwlock_lockable(rwlock *lock,int writer);

    void lock();  // implemented in lockmanager.cpp
    void unlock(); // implemented in lockmanager.cpp
  };

  struct arrayregion {
    void **array;
    snde_index indexstart;
    snde_index numelems;
  };

  class markedregion  {
  public:
    snde_index regionstart;
    snde_index regionend;

    markedregion(snde_index regionstart,snde_index regionend)
    {
      this->regionstart=regionstart;
      this->regionend=regionend;
    }

    bool attempt_merge(markedregion &later)
    {
      assert(later.regionstart==regionend);
      regionend=later.regionend;
      return true;
    }
    std::shared_ptr<markedregion> breakup(snde_index breakpoint)
    /* breakup method ends this region at breakpoint and returns
       a new region starting at from breakpoint to the prior end */
    {
      std::shared_ptr<markedregion> newregion=std::make_shared<markedregion>(breakpoint,regionend);
      regionend=breakpoint;

      return newregion;
    }
  };

  class rwlock {
  public:
    // loosely motivated by https://stackoverflow.com/questions/11032450/how-are-read-write-locks-implemented-in-pthread
    // * Instantiate snde::rwlock()
    // * To read lock, define a std::unique_lock<snde::rwlock> readlock(rwlock_object)
    // * To write lock, define a std::unique_lock<snde::rwlock_writer> readlock(rwlock_object.writer)
    // Can alternatively use lock_guard if you don't need to be able to unlock within the context

    std::deque<std::condition_variable *> threadqueue;
    int writelockpending;
    int writelockcount;
    int readlockcount;
    //size_t regionstart; // if in subregions only
    //size_t regionend; // if in subregions only

    //std::mutex admin;  (swig-incompatible)
    rwlock_lockable reader;
    rwlock_lockable writer;


    rangetracker<markedregion> _dirtyregions; /* track dirty ranges during a write (all regions that are locked for write) */

    std::deque<std::function<void(snde_index firstelem,snde_index numelems)>> _dirtynotify; /* locked by admin, but functions should be called with admin lock unlocked (i.e. copy the deque before calling). This write lock will be locked. NOTE: numelems of SNDE_INDEX_INVALID means to the end of the array  */

    // For weak readers to support on-GPU caching of data, we would maintain a list of some sort
    // of these weak readers. Before we give write access, we would have to ask each weak reader
    // to relinquish. Then when write access ends by unlock or downgrade, we offer each weak
    // reader the ability to recache.

    // in addition/alternatively the allocator concept could allocate memory directly on board the GPU...
    // but this alone would be problematic for e.g. triangle lists that need to be accessed both by
    // a GPU renderer and GPGPU computation.


    rwlock();
    
    void _wait_for_top_of_queue(std::condition_variable *cond,std::unique_lock<std::mutex> *adminlock);
    
    void lock_reader();

    void clone_reader();
    void unlock_reader();

    void writer_append_region(snde_index firstelem, snde_index numelems);
    void lock_writer(snde_index firstelem,snde_index numelems);

    void lock_writer();

    void downgrade();

    void sidegrade();

    void clone_writer();
    void unlock_writer();

  };


  typedef std::unique_lock<rwlock_lockable> rwlock_token_content;
  typedef std::shared_ptr<rwlock_token_content> rwlock_token;

  typedef std::unordered_map<rwlock_lockable *,rwlock_token> rwlock_token_set_content;
  //typedef std::shared_ptr<rwlock_token_set_content> rwlock_token_set;
  // Persistent token of lock ownership
  //typedef std::shared_ptr<std::unique_lock<rwlock_lockable>> rwlock_token;
  // Set of tokens
  typedef std::shared_ptr<rwlock_token_set_content> rwlock_token_set;
};

%template(rwlock_token) std::shared_ptr<snde::rwlock_token_content>;  
%template(rwlock_token_set) std::shared_ptr<snde::rwlock_token_set_content>;  

/*
%typemap(out) snde::rwlock_token_set {
  std::shared_ptr<snde::rwlock_token_set_content> *smartresult = bool(result) ? new std::shared_ptr<snde::rwlock_token_set_content>(result SWIG_NO_NULL_DELETER_SWIG_POINTER_NEW) :0;
  %set_output(SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(std::shared_ptr< snde::rwlock_token_set_content > *), SWIG_POINTER_NEW|SWIG_POINTER_OWN));

}*/

namespace snde{


  /* rwlock_token_set semantics: 
     The rwlock_token_set contains a reference-counted set of locks, specific 
     to a particular thread. They can be passed around, copied, etc.
     at will within a thread context. 
     
     A lock is unlocked when: 
        (a) unlock_rwlock_token_set() is called on an rwlock_token_set containing 
	    the lock, or 
        (b) All references to all rwlock_token_sets containing the lock
	    are released (by release_rwlock_token_set()) or go out of scope
     Note that it is an error (std::system_error) to call unlock_rwlock_token_set()
     on an rwlock_token_set that contains any locks that already have been
     unlocked (i.e. by a prior call to unlock_rwlock_token_set()) 

     It is possible to pass a token_set to another thread, but 
     only by creating a completely independent cloned copy and 
     completely delegating the cloned copy to the other thread.

     The cloned copy is created with clone_rwlock_token_set(). 
     The unlock()ing the cloned copy is completely separate from
     unlocking the original.
*/

  static inline void release_rwlock_token_set(rwlock_token_set &tokens);

  static inline void unlock_rwlock_token_set(rwlock_token_set tokens);
  



  static inline snde::rwlock_token_set empty_rwlock_token_set(void);

  static inline bool check_rwlock_token_set(rwlock_token_set tokens);
  

  static inline void merge_into_rwlock_token_set(rwlock_token_set accumulator, rwlock_token_set tomerge);

  static inline rwlock_token_set clone_rwlock_token_set(rwlock_token_set orig);
  
  class arraylock {
  public:
    //std::mutex admin; /* locks access to full_array and subregions members.. obsolete.... just use the lockmanager's admin lock */

%immutable; //avoid swig compilation errors
    rwlock full_array;
%mutable;
    //std::vector<rwlock> subregions;

    arraylock() {

    }
  };

  //typedef std::vector<void **> arrayvector;

  class lockmanager {
  public:
    std::vector<void **> _arrays; /* get array pointer from index */
    %immutable; /* avoid swig trouble */
    // NOTE: can work around swig troubles by explicitly specifying hash
    std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>> _arrayidx; /* get array index from pointer */
    std::deque<arraylock> _locks; /* get lock from index */
    //std::unordered_map<rwlock *,size_t> _idx_from_lockptr; /* get array index from lock pointer */
    %mutable;

    lockmanager();
    size_t get_array_idx(void **array);

    void addarray(void **array);
      /* WARNING: ALL addarray() CALLS MUST BE ON INITIALIZATION
	 FROM INITIALIZATION THREAD, BEFORE OTHER METHODS MAY
	 BE CALLED! */

    bool is_region_granular(void); /* Return whether locking is really granular on a region-by-region basis (true) or just on an array-by-array basis (false) */

    void set_array_size(void **Arrayptr,size_t elemsize,snde_index nelem);

    rwlock_token  _get_preexisting_lock_read_array(rwlock_token_set all_locks, size_t arrayidx);


    rwlock_token  _get_lock_read_array(rwlock_token_set all_locks, size_t arrayidx);

    rwlock_token_set get_locks_read_array(rwlock_token_set all_locks, void **array);

    rwlock_token_set get_preexisting_locks_read_array(rwlock_token_set all_locks, void **array);


    rwlock_token_set get_locks_read_arrays(rwlock_token_set all_locks, std::vector<void **> arrays);


    rwlock_token_set get_preexisting_locks_read_array_region(rwlock_token_set all_locks, void **array,snde_index indexstart,snde_index numelems);

    rwlock_token_set get_locks_read_array_region(rwlock_token_set all_locks, void **array,snde_index indexstart,snde_index numelems);

    rwlock_token_set get_locks_read_arrays_region(rwlock_token_set all_locks, std::vector<struct arrayregion> arrays);




    rwlock_token_set get_locks_read_all(rwlock_token_set all_locks);



    rwlock_token  _get_preexisting_lock_write_array_region(rwlock_token_set all_locks, size_t arrayidx,snde_index indexstart,snde_index numelems);

    rwlock_token  _get_lock_write_array_region(rwlock_token_set all_locks, size_t arrayidx,snde_index indexstart,snde_index numelems);

    rwlock_token_set get_locks_write_array(rwlock_token_set all_locks, void **array);
    rwlock_token_set get_preexisting_locks_write_array(rwlock_token_set all_locks, void **array);

    rwlock_token_set get_locks_write_arrays(rwlock_token_set all_locks, std::vector<void **> arrays);

    rwlock_token_set get_preexisting_locks_write_array_region(rwlock_token_set all_locks, void **array,snde_index indexstart,snde_index numelems);

    rwlock_token_set get_locks_write_array_region(rwlock_token_set all_locks, void **array,snde_index indexstart,snde_index numelems);

    rwlock_token_set get_locks_write_arrays_region(rwlock_token_set all_locks, std::vector<struct arrayregion> arrays);



    rwlock_token_set get_locks_write_all(rwlock_token_set all_locks);

    void downgrade_to_read(rwlock_token_set locks);

  };


  class lockingposition {
  public:
    size_t arrayidx; /* index of array we want to lock, or numeric_limits<size_t>.max() */
    snde_index idx_in_array; /* index within array, or SNDE_INDEX_INVALID*/
    bool write; /* are we trying to lock for write? */ 
    lockingposition();
    lockingposition(size_t arrayidx,snde_index idx_in_array,bool write);

    bool operator<(const lockingposition & other) const;
  };


  class lockingprocess {
      /* lockingprocess is a tool for performing multiple locking
         for multiple objects while satisfying the required
         locking order */

      /* (There was going to be an opencl_lockingprocess that was to be derived
         from this class, but it was cancelled) */

    lockingprocess(const lockingprocess &)=delete; /* copy constructor disabled */
    lockingprocess& operator=(const lockingprocess &)=delete; /* copy assignment disabled */

    virtual rwlock_token_set get_locks_write_array(void **array);
    virtual rwlock_token_set get_locks_write_array_region(void **array,snde_index indexstart,snde_index numelems);
    virtual rwlock_token_set get_locks_read_array(void **array);
    virtual rwlock_token_set get_locks_read_array_region(void **array,snde_index indexstart,snde_index numelems);
    virtual void spawn(std::function<void(void)> f);

    virtual ~lockingprocess();

    

  };
};
  
%{
namespace snde {
  class lockingprocess_pycpp: public lockingprocess {
    lockingprocess_pycpp(const lockingprocess_pycpp &)=delete; /* copy constructor disabled */
    lockingprocess_pycpp& operator=(const lockingprocess_pycpp &)=delete; /* copy assignment disabled */

    lockingprocess_pycpp(std::shared_ptr<lockmanager> manager); 

    //  virtual rwlock_token_set get_locks_write_array(void **array);

    //  virtual rwlock_token_set get_locks_write_array_region(void **array,snde_index indexstart,snde_index numelems);


    //  virtual rwlock_token_set get_locks_read_array(void **array);

    //  virtual rwlock_token_set get_locks_read_array_region(void **array,snde_index indexstart,snde_index numelems);


    //  virtual std::tuple<rwlock_token_set,std::shared_ptr<std::vector<rangetracker<markedregion>>>,std::shared_ptr<std::vector<rangetracker<markedregion>>>> finish();
    // ***!!! Need to implement callbacks back to Python side !!!***
    // (use ctypes?) 
    virtual void spawn(std::function<void(void)> f)
    {

    }

    virtual ~lockingprocess_pycpp()
    {

    }	
  };

}
%}

namespace snde {
 // lockingprocess needs to be re-implemented based
 // on Python generators/resumable functions/etc.
 // and/or wrapped for Python thread support
 // this is a base class for lockingprocess_python (implemented on the Python side)
 
  class lockingprocess_pycpp: public lockingprocess {
    lockingprocess_pycpp(const lockingprocess_pycpp &)=delete; /* copy constructor disabled */
    lockingprocess_pycpp& operator=(const lockingprocess_pycpp &)=delete; /* copy assignment disabled */

    lockingprocess_pycpp(std::shared_ptr<lockmanager> manager); 

    //  virtual rwlock_token_set get_locks_write_array(void **array);

    //  virtual rwlock_token_set get_locks_write_array_region(void **array,snde_index indexstart,snde_index numelems);


    //  virtual rwlock_token_set get_locks_read_array(void **array);

    //  virtual rwlock_token_set get_locks_read_array_region(void **array,snde_index indexstart,snde_index numelems);


    //  virtual std::tuple<rwlock_token_set,std::shared_ptr<std::vector<rangetracker<markedregion>>>,std::shared_ptr<std::vector<rangetracker<markedregion>>>> finish();
    // ***!!! Need to implement callbacks back to Python side !!!***
    // (use ctypes?) 
    virtual void spawn(std::function<void(void)> f);

    virtual ~lockingprocess_pycpp();
  };
  
}




%pythoncode %{

# !!!*** lockingprocess here should be an abstract base class
# defined on the c++ side with
# a specialization that calls Python
class lockingprocess_python(lockingprocess_pycpp):
  manager=None # lockmanager
  waiting_generators=None   # LockingPositionMap
  runnable_generators=None  # list of generators
  
  arrayreadregions=None  # VectorOfRegions
  arraywriteregions=None # VectorOfRegions
  lastlockingposition=None # lockingposition

  all_tokens=None # rwlock_token_set

  def __init__(self,**kwargs):
    for key in kwargs:
      if not hasattr(self,key):
        raise ValueError("Bad attribute")
      setattr(self,key,kwargs[key])
      pass
      
  @classmethod
  def execprocess(cls,manager,*lock_generators):
    arrayreadregions=VectorOfRegions(manager._arrays.size())
    arraywriteregions=VectorOfRegions(manager._arrays.size())
    lastlockingposition=lockingposition(0,0,True)

    all_tokens=empty_rwlock_token_set()

    # locking generators take (all_tokens,arrayreadregions,arraywriteregions)
    # as parameters. They yield either more generators
    # or a lockingposition. If they yield a locking position, the next()
    # call on them will cause them to perform the lock and yield None,
    # then they may yield another generator or locking position, etc. 

    waiting_generators = LockingPositionMap()

    proc=cls(manager=manager,
             waiting_generators=waiting_generators,
             #runnable_generators=runnable_generators,
             arrayreadregions=arrayreadregions,
	     arraywriteregions=arraywriteregions,
             lastlockingposition=lastlockingposition,
             all_tokens=all_tokens)
    proc.runnable_generators=[ lock_generator(proc) for lock_generator in lock_generators ]

  
    while len(waiting_generators) > 0 or len(proc.runnable_generators) > 0:
      while len(proc.runnable_generators) > 0:
        thisgen=proc.runnable_generators[0]
        newgen=next(proc.runnable_generators[0])
        proc.runnable_generators.pop(0) # this generator is no longer runnable

        proc.process_generated(thisgen,newgen)
      
        pass
      # ok... no more runnable generators... do we have a waiting generator?
      if len(waiting_generators) > 0:
        # grab the first waiting generator
	# Use C++ style iteration because that way we iterate
	# over pairs, not over keys
	iterator=waiting_generators.begin()
	(lockpos,lockcall_gen)=iterator.value()
	(lockcall,gen)=lockcall_gen.value()
	

        waiting_generators.erase(iterator)
      
        # perform locking operation
        res=lockcall()
	newgen=None
	try:
	  newgen=gen.send(res)
	except StopIteration:
	  pass
        if newgen is not None:
          proc.process_generated(gen,newgen)
	  pass
        pass
      pass
    return (proc.all_tokens,proc.arrayreadregions,proc.arraywriteregions)
    
  def process_generated(self,thisgen,newgen):
    if isinstance(newgen,pytypes.GeneratorType):
      # Got another generator
      self.runnable_generators.append(thisgen)
      self.runnable_generators.append(newgen)
      pass
    else:
      assert(isinstance(newgen,tuple) and isinstance(newgen[0],lockingposition))
      (posn,lockcall) = newgen	
	
      self.waiting_generators.emplace_pair(lockingposition_generator(posn,CountedPyObject((lockcall,thisgen))))
      pass
    pass

  def get_locks_read_array_region(self,ArrayPtr_Swig,indexstart,numelems):
    #ArrayPtr_Swig = ArrayPtr_fromint(arrayptr)
    if not self.manager._arrayidx.has_key(ArrayPtr_Swig):
      raise ValueError("Array not found")
    
    iterator = self.manager._arrayidx.find(ArrayPtr_Swig)
    arrayidx = voidpp_posn_map_iterator_posn(iterator) # fromiterator(iterator).get_posn()
    #if iterator==self.manager._arrayidx.end():  # ***!!!! This diagnosis does not actually work
    #  raise ValueError("Array not found")
      
    if self.manager.is_region_granular():
      posn=lockingposition(arrayidx,indexstart,False)
      pass
    else:
      posn=lockingposition(arrayidx,0,False)
      pass
    
    def lockcall():
      newset = self.manager.get_locks_read_array_region(self.all_tokens,ArrayPtr_Swig,indexstart,numelems)
      self.arrayreadregions[arrayidx].mark_region_noargs(indexstart,numelems)
      return newset
    return (posn,lockcall)
  pass

def pylockprocess(*args,**kwargs):
  return lockingprocess_python.execprocess(*args,**kwargs)
  pass

class pylockholder(object):
  def store(self,lockname,value):
    setattr(self,lockname,value)
    pass
  pass
    
  
%}