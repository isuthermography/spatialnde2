%pythonbegin %{
import types as pytypes
%}

%shared_ptr(LockingPositionMap);
%shared_ptr(VectorOfRegions);

%shared_ptr(snde::lockmanager);
%shared_ptr(snde::lockholder);
%shared_ptr(snde::lockingprocess_thread);
%shared_ptr(std::vector<snde::rangetracker<snde::markedregion>>);
%shared_ptr(voidpp_voidpp_multimap_pyiterator)

  
 //%shared_ptr(std::vector<void **>);
%shared_ptr(std::unordered_map<void **,snde::lockindex_t,std::hash<void **>,std::equal_to<void **>>);
%shared_ptr(std::unordered_map<snde::lockindex_t,void **>);
//%shared_ptr(std::unordered_map<snde::lockindex_t,std::weak_ptr<snde::mutableinfostore>);
%shared_ptr(std::unordered_map<snde::lockindex_t,std::shared_ptr<snde::datalock>>);
  
//%shared_ptr(std::deque<std::shared_ptr<arraylock>>);

%template(LockingPositionMap) std::multimap<snde::lockingposition,snde::CountedPyObject>;

%template(voidpp_voidpp_multimap) std::multimap<void **,void**>;

//%extend std::multimap<void **,void **> {
//  std::multimap<void **,void**>::iterator lower_bound(const void **& key) {
//
//}

%{ // per https://stackoverflow.com/questions/38404806/error-c2039-type-name-is-not-a-member-of-of-swigtraitsbar
  namespace swig {
    template <> struct traits<snde::rwlock_token_set>
    {
      typedef pointer_category category;
      static const char *type_name()
      {
        return "rwlock_token_set";
      }

    };

    template <> struct traits<std::shared_ptr<snde::alloc_voidpp>>
    {
      typedef pointer_category category;
      static const char *type_name()
      {
        return "std::shared_ptr<snde::alloc_voidpp>";
      }

    };
    
  }  
%}


%{
class Stop_Iteration: public std::exception {
};
%}
class Stop_Iteration: public std::exception {
};

%typemap(throws) Stop_Iteration %{
  PyErr_SetNone(PyExc_StopIteration);
  SWIG_fail;
%}

%{

class voidpp_voidpp_multimap_pyiterator {
public:
  std::multimap<void **,void**> *map;
  void **key;
  std::multimap<void **,void**>::iterator it;
  voidpp_voidpp_multimap_pyiterator(std::multimap<void **,void**> *mp,void **ky,std::multimap<void **,void**>::iterator iter) : map(mp),key(ky),it(iter) {}

  
  void **next() /* throw (Stop_Iteration)*/ {
    std::multimap<void **,void**>::iterator ret;
    //fprintf(stderr,"Iterate next()\n");
    if (it==map->end() || it->first != key) {    
      //fprintf(stderr,"StopIter\n");
      throw Stop_Iteration();
    }
    ret=it;
    it++;
    return ret->second;
  }

  static std::shared_ptr<voidpp_voidpp_multimap_pyiterator> iterate_particular_key(std::multimap<void **,void **> *map,void **key) {
    return std::make_shared<voidpp_voidpp_multimap_pyiterator>(map,key,map->lower_bound(key));
  }

  
};

%}

class voidpp_voidpp_multimap_pyiterator {
public:
  //std::multimap<void **,void**>::iterator it;
  voidpp_voidpp_multimap_pyiterator(std::multimap<void **,void**> *mp,void **ky,std::multimap<void **,void**>::iterator iter);
  void **next() throw (Stop_Iteration) ;
  
  static std::shared_ptr<voidpp_voidpp_multimap_pyiterator> iterate_particular_key(std::multimap<void **,void **> *map,void **key);   
};

%extend voidpp_voidpp_multimap_pyiterator {
  voidpp_voidpp_multimap_pyiterator *__iter__()
  {
    return $self;
  }
  void **__next__() throw (Stop_Iteration)
  {
    return $self->next();
  }

}


%extend std::multimap<snde::lockingposition,snde::CountedPyObject> {
  void emplace_pair(std::pair<snde::lockingposition,snde::CountedPyObject> p)
  {
    self->emplace(p);
  }
}



%template(Region) snde::rangetracker<snde::markedregion>;

%template(PtrVectorOfRegions) std::shared_ptr<std::vector<snde::rangetracker<snde::markedregion>>>;
%template(VectorOfRegions) std::vector<snde::rangetracker<snde::markedregion>>;
%template(lockingposition_generator) std::pair<snde::lockingposition,snde::CountedPyObject>; 

//%template(voidpp_posn_map) std::unordered_map<void **,size_t>;
//%template(voidpp_posn_map) std::unordered_map<void **,snde::lockindex_t,std::hash<void **>,std::equal_to<void *>,std::allocator< std::pair< void **const,snde::lockindex_t > > >;

%extend std::unordered_map<void **,snde::lockindex_t,std::hash<void **>,std::equal_to<void **>,std::allocator< std::pair< void **const,snde::lockindex_t > > > {
  bool has_key(ArrayPtr key) {
    if (self->find((void*)key)==self->end()) return false;
    return true;
  }
};



// NOTE: This iterator std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>,std::allocator< std::pair< void **const,size_t > > >::iterator   is currently causing a memory leak message.... seems to be a swig bug...

//snde::lockindex_t voidpp_posn_map_iterator_posn(std::unordered_map<void **,snde::lockindex_t,std::hash<void **>,std::equal_to<void **>,std::allocator< std::pair< void **const,snde::lockindex_t > > >::iterator);

%{
  snde::lockindex_t voidpp_posn_map_iterator_posn(std::unordered_map<void **,snde::lockindex_t,std::hash<void **>,std::equal_to<void **>,std::allocator< std::pair< void **const,snde::lockindex_t > > >::iterator self) {
    return self->second;

}
%}

//snde::ArrayPtr voidpp_posn_map_iterator_ptr(std::unordered_map<void **,snde::lockindex_t,std::hash<void **>,std::equal_to<void **>,std::allocator< std::pair< void **const,snde::lockindex_t > > >::iterator);
%{
  snde::ArrayPtr voidpp_posn_map_iterator_ptr(std::unordered_map<void **,snde::lockindex_t,std::hash<void **>,std::equal_to<void **>,std::allocator< std::pair< void **const,snde::lockindex_t > > >::iterator self)
{
  return (void**)self->first;
}
%}

// Workaround for memory leak: Never expose the iterator to Python
%extend std::unordered_map<void **,snde::lockindex_t,std::hash<void **>,std::equal_to<void **>,std::allocator< std::pair< void **const,snde::lockindex_t > > > {
  
  snde::lockindex_t get_ptr_posn(snde::ArrayPtr ptr){
    std::unordered_map<void **,snde::lockindex_t,std::hash<void **>,std::equal_to<void **>,std::allocator< std::pair< void **const,snde::lockindex_t > > >::iterator it=self->find((void **)ptr);
    assert(it != self->end()); /* should diagnose lack of entry prior to calling with has_key() */
    	      
    return it->second;
  }

}


  

/*  ***** NOTE: next big section is obsolete and commented out
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

namespace snde{
  class lockholder_index; // forward declaration
  class arraymanager; // forward declaration
  class mutablewfmdb; // forward declaration
  class geometry;
#ifdef SNDE_MUTABLE_WFMDB_SUPPORT
  class lockable_infostore_or_component;
#endif // SNDE_MUTABLE_WFMDB_SUPPORT
  //class component; // forward declaration
  //class parameterization; // forward declaration
  class lockingposition;
  class ndarray_recording_ref; // forward declaration, recstore.hpp
  
struct arrayregion {
    void **array;
    snde_index indexstart;
    snde_index numelems;
  };

  class markedregion  {
  public:
    snde_index regionstart;
    snde_index regionend;

    markedregion(snde_index regionstart,snde_index regionend);

    bool attempt_merge(markedregion &later);
    std::shared_ptr<markedregion> sp_breakup(snde_index breakpoint);
  };


    class datalock {
  public: 
      //std::mutex admin; /* locks access to subregions field of arraylock subclass, lock after everything; see also whole_array_write */
      %immutable;
    std::shared_ptr<rwlock> whole; /* This is in the locking order with the arrays and mutableinfostores. In
				      order to modify subregions you must hold this AND all subregions AND admin (above) 
				      for write... Note: Not used for dirty tracking (put dirty stuff in subregions!) */
    %mutable;
	 
    datalock() {
      whole=std::make_shared<rwlock>();
      
    }
  };
    

  
    class arraylock: public datalock {
  public:
    //std::mutex admin; (swig incompatible) /* locks access to subregions, lock after everything; see also whole_array_write */

    //%immutable; //avoid swig compilation errors
    //rwlock full_array;
    //%mutable;
    //std::vector<rwlock> subregions;
    %immutable;
    std::map<markedregion,std::shared_ptr<rwlock>> subregions;
    %mutable;

    arraylock();
  };

  //typedef std::vector<void **> arrayvector;

    // input typemap for list of (arrayref,bool) parameters to lockmanager::lock_recording_refs
    
    %typemap(in) std::vector<std::pair<std::shared_ptr<snde::ndarray_recording_ref>,bool>> (PyObject *Tup,PyObject *Ref,PyObject *Bol,Py_ssize_t elemnum,Py_ssize_t num_elem,int res,void *argp,int boolval) {
      $1 = std::vector<std::pair<std::shared_ptr<snde::ndarray_recording_ref>,bool>>();
      if (!PySequence_Check($input)) {
	SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			    "$argnum"" is not a sequence.");
      }
      num_elem = PySequence_Size($input);
      $1.reserve(num_elem);
      
      for (elemnum=0;elemnum < num_elem;elemnum++) {
	Tup=nullptr;
	Ref=nullptr;
	Bol=nullptr;
	Tup=PySequence_GetItem($input,elemnum);
	if (!Tup) {
	  SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			      "$argnum"" Cannot get sequence item.");
	}
	if (!PySequence_Check(Tup)) {
	  SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			      "$argnum"" sequence item is not a tuple or short sequence.");
	}
	if (PySequence_Size(Tup) != 2) {
	  SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			      "$argnum"" sequence item sequence length is not 2.");
	}
	Ref=PySequence_GetItem(Tup,0);
	if (!Ref) {
	  SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			      "$argnum"" sequence item failed to get first element.");
	}
	res=SWIG_ConvertPtr(Ref,&argp,$descriptor(std::shared_ptr<snde::ndarray_recording_ref> *),0);
	if (!SWIG_IsOK(res)) {
	  SWIG_exception_fail(SWIG_ArgError(res), "in method '" "$symname" "', argument "
			      "$argnum"" sequence item element is not convertable to a snde::ndarray_recording_ref");	  
	}
	Bol=PySequence_GetItem(Tup,1);
	if (!Bol) {
	  SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			      "$argnum"" sequence item failed to get second element.");
	}
	boolval = PyObject_IsTrue(Bol);
	if (boolval < 0) {
	  SWIG_exception_fail(SWIG_TypeError, "in method '" "$symname" "', argument "
			      "$argnum"" sequence item second element not interpretable as boolean.");
	}
	$1.push_back(std::make_pair(std::shared_ptr<snde::ndarray_recording_ref>(*(std::shared_ptr<snde::ndarray_recording_ref> *)argp),boolval));
	
	Py_XDECREF(Bol);
	Py_XDECREF(Ref);
	Py_XDECREF(Tup);
      }
    }
    

  class lockmanager {
  public:
    //std::vector<void **> _arrays; /* get array pointer from index */
    //%immutable; /* avoid swig trouble */
    //// NOTE: can work around swig troubles by explicitly specifying hash
    //std::unordered_map<void **,size_t,std::hash<void **>,std::equal_to<void **>> _arrayidx; /* get array index from pointer */
    //std::deque<arraylock> _locks; /* get lock from index */
    ////std::unordered_map<rwlock *,size_t> _idx_from_lockptr; /* get array index from lock pointer */
    //%mutable;

    lockmanager();

    /* Accessors for atomic shared pointers */
    std::shared_ptr<std::unordered_map<lockindex_t,void **>> _arrays();
    std::shared_ptr<std::unordered_map<void **,lockindex_t,std::hash<void **>,std::equal_to<void **>>> _arrayidx();
    std::shared_ptr<std::unordered_map<lockindex_t,std::shared_ptr<datalock>>> _locks();

    lockindex_t get_array_idx(void **array);

    void addarray(void **array);
      /* WARNING: ALL addarray() CALLS MUST BE ON INITIALIZATION
	 FROM INITIALIZATION THREAD, BEFORE OTHER METHODS MAY
	 BE CALLED! */

    bool is_region_granular(void); /* Return whether locking is really granular on a region-by-region basis (true) or just on an array-by-array basis (false) */

    void set_array_size(void **Arrayptr,size_t elemsize,snde_index nelem);
    rwlock_token newallocation(rwlock_token_set all_locks,void **arrayptr,snde_index pos,snde_index size,snde_index elemsize);
    void freeallocation(void **arrayptr,snde_index pos, snde_index size,snde_index elemsize);
    void realloc_down_allocation(void **arrayptr, snde_index addr,snde_index orignelem, snde_index newnelem);


    // use lock_recording_refs() to lock a bunch of ndarray_recording refs
    // use brace initialization; the 2nd half of the pair is true for write locking:
    //  mylock = lock_recording_refs({ {inputrec1,false},
    //                                 {inputrec2,false},
    //                                 {outputrec1,true} });
    rwlock_token_set lock_recording_refs(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> recrefs,bool gpu_access);


    rwlock_token  _get_preexisting_lock_read_lockobj(rwlock_token_set all_locks,std::shared_ptr<rwlock> rwlockobj);
    std::pair<rwlock_lockable *,rwlock_token>  _get_preexisting_lock_read_array_region(rwlock_token_set all_locks, lockindex_t arrayidx,snde_index pos,snde_index size);
    std::pair<rwlock_lockable *,rwlock_token>  _get_lock_read_array_region(rwlock_token_set all_locks, lockindex_t arrayidx,snde_index pos,snde_index size);
    rwlock_token_set get_locks_read_array(rwlock_token_set all_locks, void **array);
    rwlock_token_set get_preexisting_locks_read_array(rwlock_token_set all_locks, void **array);
    rwlock_token_set get_preexisting_locks_read_array_region(rwlock_token_set all_locks, void **array,snde_index indexstart,snde_index numelems);
    
    rwlock_token_set get_locks_read_array_region(rwlock_token_set all_locks, void **array,snde_index indexstart,snde_index numelems);
    rwlock_token_set get_locks_read_all(rwlock_token_set all_locks);
    
    rwlock_token  _get_preexisting_lock_write_lockobj(rwlock_token_set all_locks,std::shared_ptr<rwlock> rwlockobj);
    
    std::pair<rwlock_lockable *,rwlock_token>  _get_preexisting_lock_write_array_region(rwlock_token_set all_locks, lockindex_t arrayidx,snde_index indexstart,snde_index numelems);

    std::pair<rwlock_lockable *,rwlock_token>  _get_lock_write_array_region(rwlock_token_set all_locks, lockindex_t arrayidx,snde_index pos,snde_index size);
    rwlock_token_set get_locks_write_array(rwlock_token_set all_locks, void **array);
    rwlock_token_set get_preexisting_locks_write_array(rwlock_token_set all_locks, void **array);
    rwlock_token_set get_preexisting_locks_write_array_region(rwlock_token_set all_locks, void **array,snde_index indexstart,snde_index numelems);
    rwlock_token_set get_locks_write_array_region(rwlock_token_set all_locks, void **array,snde_index indexstart,snde_index numelems);
    rwlock_token_set get_locks_write_all(rwlock_token_set all_locks);

    void downgrade_to_read(rwlock_token_set locks);

  };

  class lockingprocess_thread {
  public:
    virtual ~lockingprocess_thread() {};
  };

  class lockingposition {
  public:
      bool initial_position; // if true this is the blank initial position in the locking order
      bool between_infostores_and_arrays; // if true this is the blank position between infostores and arrays

#ifdef SNDE_MUTABLE_WFMDB_SUPPORT
      std::weak_ptr<lockable_infostore_or_component> lic;
#endif // SNDE_MUTABLE_WFMDB_SUPPORT
    lockindex_t array_idx; // -1 if invalid

    snde_index idx_in_array; /* index within array, or SNDE_INDEX_INVALID*/
    bool write; /* are we trying to lock for write? */ 
    lockingposition();
    static lockingposition lockingposition_before_lic();
    static lockingposition lockingposition_before_arrays();


    lockingposition(lockindex_t array_idx,snde_index idx_in_array,bool write);
#ifdef SNDE_MUTABLE_WFMDB_SUPPORT
    lockingposition(std::weak_ptr<lockable_infostore_or_component> lic,bool write);
#endif //SNDE_MUTABLE_WFMDB_SUPPORT
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

    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_write_array(void **array);
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_write_array_region(void **array,snde_index indexstart,snde_index numelems);
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_read_array(void **array);
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_read_array_region(void **array,snde_index indexstart,snde_index numelems);
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_array_region(void **array,bool write,snde_index indexstart,snde_index numelems);
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_array(void **array,bool write);
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_array_mask(void **array,uint64_t maskentry,uint64_t resizemaskentry,uint64_t readmask,uint64_t writemask,uint64_t resizemask,snde_index indexstart,snde_index numelems);

    virtual std::shared_ptr<lockingprocess_thread> spawn(std::function<void(void)> f);

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
    virtual std::shared_ptr<lockingprocess_thread> spawn(std::function<void(void)> f)
    {
      return nullptr;
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
    virtual std::shared_ptr<lockingprocess_thread> spawn(std::function<void(void)> f);

    virtual ~lockingprocess_pycpp();
  };
  
}



#ifdef SNDE_LOCKMANAGER_COROUTINES_THREADED

  /* ***!!! Should create alternate implementation based on boost stackful coroutines ***!!! */
  /* ***!!! Should create alternate implementation based on C++ resumable functions proposal  */


%typemap(out) std::tuple<rwlock_token_set, std::shared_ptr<std::vector<rangetracker<markedregion>>>, std::shared_ptr<std::vector<rangetracker<markedregion>>>> {
    $result = PyTuple_New(3);
    // Substituted code for converting cl_context here came
    // from a typemap substitution "$typemap(out,cl_context)"
    snde::rwlock_token_set result0 = std::get<0>(*&$1);
    snde::rwlock_token_set *smartresult0 = result0 ? new snde::rwlock_token_set(result0) : 0;
    
    PyTuple_SetItem($result,0,SWIG_NewPointerObj(SWIG_as_voidptr(smartresult0),$descriptor(rwlock_token_set *),SWIG_POINTER_OWN));

    std::shared_ptr<std::vector<snde::rangetracker<snde::markedregion>>> result1 = std::get<1>(*&$1);
    std::shared_ptr<std::vector<snde::rangetracker<snde::markedregion>>>  *smartresult1 = result1 ? new std::shared_ptr<std::vector<snde::rangetracker<snde::markedregion>>>(result1) : 0;
    PyTuple_SetItem($result,1,SWIG_NewPointerObj(SWIG_as_voidptr(smartresult1),$descriptor(std::shared_ptr<std::vector<snde::rangetracker<snde::markedregion>>> *),SWIG_POINTER_OWN));

    std::shared_ptr<std::vector<snde::rangetracker<snde::markedregion>>> result2 = std::get<2>(*&$1);
    std::shared_ptr<std::vector<snde::rangetracker<snde::markedregion>>>  *smartresult2 = result2 ? new std::shared_ptr<std::vector<snde::rangetracker<snde::markedregion>>>(result2) : 0;
    PyTuple_SetItem($result,2,SWIG_NewPointerObj(SWIG_as_voidptr(smartresult2),$descriptor(std::shared_ptr<std::vector<snde::rangetracker<snde::markedregion>>> *),SWIG_POINTER_OWN));


  }

%typemap(in) std::function<void(void)> spawn_func (PyObject *FuncObj,PyObject *SelfObj) {
  FuncObj=$input;
  SelfObj=$self;
  Py_INCREF(FuncObj);
  Py_INCREF(SelfObj);
  arg2 = [ FuncObj, SelfObj ]() { PyObject *res=PyObject_CallFunctionObjArgs(FuncObj,SelfObj,NULL);Py_DECREF(FuncObj);Py_DECREF(SelfObj);Py_XDECREF(res); };

}

namespace snde {

  class lockingprocess_threaded: public lockingprocess {
    /* lockingprocess is a tool for performing multiple locking
       for multiple objects while satisfying the required
       locking order */
      //public: 
      //lockingprocess_threaded(std::shared_ptr<lockmanager> lockmanager);
       // This is defined by the details are hidden from python, so
       // they can't be used... Use the lockingprocess_threaded_python instead
       
    };

class lockingprocess_threaded_python: public lockingprocess_threaded {
  public:
    lockingprocess_threaded_python(std::shared_ptr<lockmanager> lockmanager);
    
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_write_array(void **array);

    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_write_array_region(void **array,snde_index indexstart,snde_index numelems);


    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_read_array(void **array);

    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_read_array_region(void **array,snde_index indexstart,snde_index numelems);
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_array_mask(void **array,uint64_t maskentry,uint64_t resizemaskentry,uint64_t readmask,uint64_t writemask,uint64_t resizemask,snde_index indexstart,snde_index numelems);

    //virtual std::vector<std::tuple<lockholder_index,rwlock_token_set,std::string>> alloc_array_region(std::shared_ptr<arraymanager> manager,void **allocatedptr,snde_index nelem,std::string allocid);

    virtual std::shared_ptr<lockingprocess_thread> spawn(std::function<void(void)> spawn_func);
      
    virtual rwlock_token_set finish();
    virtual ~lockingprocess_threaded_python();

};
};

%{
namespace snde {
  class lockingprocess_threaded_python: public lockingprocess_threaded {
    public:
    lockingprocess_threaded_python(std::shared_ptr<lockmanager> lockmanager) : lockingprocess_threaded(lockmanager)
    {
      
    }

    void *pre_callback()
    {
      PyGILState_STATE *State;
      State=(PyGILState_STATE *)calloc(sizeof(*State),1);
      *State=PyGILState_Ensure();
      return State;
    }

    void post_callback(void *state)
    {
      PyGILState_STATE *State=(PyGILState_STATE *)state;
      PyGILState_Release(*State);
      free(State);
    }

    void *prelock()
    {
      PyThreadState *_save;
      _save=PyEval_SaveThread(); 
      
      return (void *)_save;
    }

    void postunlock(void *prelockstate)
    {
      PyThreadState *_save=(PyThreadState *)prelockstate;
      PyEval_RestoreThread(_save); 
    }



  };
};
%}
 


#endif

namespace snde {


  class lockholder_index {
  public:
    void **array;
    bool write;
    snde_index startidx;
    snde_index numelem;

    lockholder_index(void **_array,bool _write, snde_index _startidx,snde_index _numelem);

    // equality operator for std::unordered_map
    bool operator==(const lockholder_index b) const;
  };


%typemap(in) std::pair<snde::lockholder_index,rwlock_token_set tokens>  (PyObject *Obj,PyObject *li_obj,void *li_ptr=NULL,snde::lockholder_index *li,void *ToksPtr=NULL,rwlock_token_set Toks) {
  int newmem=0;
  Obj=$input;
  assert(PyTuple_Check(Obj));
  assert(PyTuple_Size(Obj)==2);
  //Str=PyString_AsString(PyTuple_GetItem(Obj,0));
  li_obj=PyTuple_GetItem(Obj,0);
  int res0 = SWIG_ConvertPtr(li_obj, &li_ptr,$descriptor(lockholder_index*),0);
  if (!SWIG_IsOK(res0)) {
    SWIG_exception_fail(SWIG_ArgError(res0), "Converting lockholderindex parameter to std::tuple<lockholder_index,rwlock_token_set,std::string> ");
  }
  li=reinterpret_cast<snde::lockholder_index *>(li_ptr);


  int res = SWIG_ConvertPtrAndOwn(PyTuple_GetItem(Obj,1),&ToksPtr,$descriptor(rwlock_token_set &),0,&newmem);
  if (!SWIG_IsOK(res)) {
    SWIG_exception_fail(SWIG_ArgError(res), "Converting parameter to std::pair<void **,rwlock_token_set>"); 
  }
  if (ToksPtr) Toks=*(reinterpret_cast< snde::rwlock_token_set * >(ToksPtr));
  if (newmem & SWIG_CAST_NEW_MEMORY) delete reinterpret_cast< snde::rwlock_token_set * >(ToksPtr);

  $1 = std::pair<snde::lockholder_index,snde::rwlock_token_set>(*li,Toks);

}


%typemap(in) std::tuple<snde::lockholder_index,rwlock_token_set tokens,std::string>  (PyObject *Obj,PyObject *li_obj,void *li_ptr=NULL,snde::lockholder_index *li,void *ToksPtr=NULL,rwlock_token_set Toks,std::string allocid) {
  int newmem=0;
  Obj=$input;
  assert(PyTuple_Check(Obj));
  assert(PyTuple_Size(Obj)==3);
  li_obj=PyTuple_GetItem(Obj,0);
  int res0 = SWIG_ConvertPtr(li_obj, &li_ptr,$descriptor(lockholder_index *),0);
  if (!SWIG_IsOK(res0)) {
    SWIG_exception_fail(SWIG_ArgError(res0), "Converting lockholderindex parameter to std::tuple<lockholder_index,rwlock_token_set,std::string> ");
  }
  li=reinterpret_cast<snde::lockholder_index *>(li_ptr);

  int res = SWIG_ConvertPtrAndOwn(PyTuple_GetItem(Obj,1),&ToksPtr,$descriptor(rwlock_token_set &),0,&newmem);
  if (!SWIG_IsOK(res)) {
    SWIG_exception_fail(SWIG_ArgError(res), "Converting parameter to std::tuple<lockholder_index,rwlock_token_set,std::string>"); 
  }
  if (ToksPtr) Toks=*(reinterpret_cast< snde::rwlock_token_set * >(ToksPtr));
  if (newmem & SWIG_CAST_NEW_MEMORY) delete reinterpret_cast< snde::rwlock_token_set * >(ToksPtr);

  if (PyString_Check(PyTuple_GetItem(Obj,2))) {
    allocid=PyString_AsString(PyTuple_GetItem(Obj,2));
  } else if (PyUnicode_Check(PyTuple_GetItem(Obj,2))) {
#if PY_VERSION_HEX < 0x03000000
    allocid=PyString_AsString(PyTuple_GetItem(Obj,2));
#else
    allocid=PyUnicode_AsUTF8(PyTuple_GetItem(Obj,2));
#endif
  }
    
  $1 = std::tuple<snde::lockholder_index,snde::rwlock_token_set,std::string>(*li,Toks,allocid);

}

%typemap(in) std::vector<std::tuple<lockholder_index,rwlock_token_set,std::string>>  (PyObject *Obj,PyObject *li_obj,void *li_ptr=NULL,snde::lockholder_index *li,void *ToksPtr=NULL,rwlock_token_set Toks,std::string allocid,size_t numentries,size_t cnt,std::vector<std::tuple<lockholder_index,rwlock_token_set,std::string>> buf,PyObject *Tup) {
  int newmem=0;
  Obj=$input;
  assert(PyList_Check(Obj));

  numentries=PyList_Size(Obj);
  for (cnt=0;cnt < numentries;cnt++) {
    Tup=PyList_GetItem(Obj,cnt);
    assert(PyTuple_Size(Tup)==3);
    li_obj=PyTuple_GetItem(Tup,0);
    int res0 = SWIG_ConvertPtr(li_obj, &li_ptr,$descriptor(lockholder_index *),0);
    if (!SWIG_IsOK(res0)) {
      SWIG_exception_fail(SWIG_ArgError(res0), "Converting lockholderindex parameter to std::tuple<lockholder_index,rwlock_token_set,std::string> ");
    }
    li=reinterpret_cast<snde::lockholder_index *>(li_ptr);

    int res = SWIG_ConvertPtrAndOwn(PyTuple_GetItem(Tup,1),&ToksPtr,$descriptor(rwlock_token_set &),0,&newmem);
    if (!SWIG_IsOK(res)) {
      SWIG_exception_fail(SWIG_ArgError(res), "Converting parameter to std::tuple<lockholder_index,rwlock_token_set,std::string>"); 
    }
    if (ToksPtr) Toks=*(reinterpret_cast< snde::rwlock_token_set * >(ToksPtr));
    if (newmem & SWIG_CAST_NEW_MEMORY) delete reinterpret_cast< snde::rwlock_token_set * >(ToksPtr);

    if (PyString_Check(PyTuple_GetItem(Tup,2))) {
      allocid=PyString_AsString(PyTuple_GetItem(Tup,2));
    } else if (PyUnicode_Check(PyTuple_GetItem(Tup,2))) {
#if PY_VERSION_HEX < 0x03000000
      allocid=PyString_AsString(PyTuple_GetItem(Tup,2));
#else
      allocid=PyUnicode_AsUTF8(PyTuple_GetItem(Tup,2));
#endif
    }
    buf.push_back(std::make_tuple(*li,Toks,allocid));
  }

  $1 = buf;
}


//%typemap(typecheck,precedence=SWIG_TYPECHECK_POINTER) lockholder_index   {
//  //$1 = PyTuple_Check($input) && PyString_Check(PyTuple_GetItem($input,0));
//  $1 = PyTuple_Check($input) && PyTuple_Size($input)==5;
//
//}

%typemap(typecheck,precedence=SWIG_TYPECHECK_POINTER) std::tuple<lockholder_index,rwlock_token_set tokens,std::string>  {
  //$1 = PyTuple_Check($input) && PyString_Check(PyTuple_GetItem($input,0));
  $1 = PyTuple_Check($input) && PyTuple_Size($input)==3;
}

%typemap(typecheck,precedence=SWIG_TYPECHECK_POINTER) std::pair<lockholder_index,rwlock_token_set tokens>  {
  //$1 = PyTuple_Check($input) && PyString_Check(PyTuple_GetItem($input,0));
  $1 = PyTuple_Check($input) && PyTuple_Size($input)==2;
}

%typemap(typecheck,precedence=SWIG_TYPECHECK_POINTER) std::vector<std::tuple<lockholder_index,rwlock_token_set,std::string>> {
  $1 = PyList_Check($input);
}
%extend lockholder {
  std::string __repr__()
  {
    return self->as_string();
  }
}
  struct lockholder_index_hash {
    size_t operator()(const lockholder_index &x) const;
  };
  
  struct voidpp_string_hash {
    size_t operator()(const std::pair<void **,std::string> &x) const;
  };
  
//%nodefaultctor lockholder;   // Inhibit SWIG's default constructor so we can replace it with our Python __init__ (below)
  class lockholder {
  public:
    std::unordered_map<lockholder_index,rwlock_token_set,lockholder_index_hash> values;
    std::unordered_map<std::pair<void **,std::string>,std::pair<snde_index,snde_index>,voidpp_string_hash> allocvalues;
    
    
    std::string as_string();
    bool has_lock(void **array,bool write,snde_index indexstart,snde_index numelem);
    
    bool has_alloc(void **array,std::string allocid);
    
    void store(void **array,bool write,snde_index indexstart,snde_index numelem,rwlock_token_set locktoken);
    void store(lockholder_index array_write_startidx_numelem_tokens,rwlock_token_set locktoken);
    void store(std::pair<lockholder_index,rwlock_token_set> idx_locktoken);
    void store_alloc(void **array,bool write,snde_index startidx,snde_index numelem,rwlock_token_set tokens,std::string allocid);
    void store_alloc(lockholder_index idx,rwlock_token_set,std::string allocid);
    void store_alloc(std::tuple<lockholder_index,rwlock_token_set,std::string> idx_tokens_allocid);
    void store_alloc(std::vector<std::tuple<lockholder_index,rwlock_token_set,std::string>> vector_idx_tokens_allocid);


    rwlock_token_set get(void **array,bool write,snde_index indexstart,snde_index numelem);
    rwlock_token_set get_alloc_lock(void **array,std::string allocid);
    snde_index get_alloc(void **array,std::string allocid);
  };
  
// Rewrite the SWIG __getattr__  so that .name and .name_addr act like get() and getaddr()
// but by name
//%extend lockholder {
//  %pythoncode %{
//    def __getattr__(self,name):
//      if name.endswith("_addr") and self.geometry.has_field(name[:-5]) and self.has_addr(self.geometry.addr(name[:-5])):
//        return self.get_addr(self.geometry.addr(name[:-5]))
//      if self.geometry.has_field(name) and self.has_lock(self.geometry.addr(name)):
//        return self.get(self.geometry.addr(name))
//	
//      return _swig_getattr(self,lockholder,name)
//   %}
//}

// replace SWIG's __init__ method, so that Python
// lockholder takes a geometry argument so we can look up
// attributes. 
//%extend lockholder {
//  %pythoncode %{
//    def __init__(self,geometry):
//        this = _spatialnde2.new_lockholder()
//        self.geometry=geometry
//        try:
//            self.this.append(this)
//        except __builtin__.Exception:
//            self.this = this        
//    %}
//}

/* *** Must keep sync'd with lockmanager.hpp */

  /* *** Lock masks for obtain_lock() calls on mutableinfostore,
     part/assembly/component, and parameterization *** */
  
typedef uint64_t snde_infostore_lock_mask_t;
#define SNDE_INFOSTORE_INFOSTORES (1ull<<0) // the snde::mutableinfostore and metadata ... used solely with get_locks_infostore_mask(...)
#define SNDE_INFOSTORE_COMPONENTS (1ull<<1) // the snde::components, i.e. parts and assemblies... used solely with get_locks_infostore_mask(...) 
#define SNDE_INFOSTORE_PARAMETERIZATIONS (1ull<<2) // the snde::parameterizations of the components... used solely with get_locks_infostore_mask(...) 

#define SNDE_INFOSTORE_ALL ((1ull<<3)-(1ull<<0))
  
// 
#define SNDE_COMPONENT_GEOM_PARTS (1ull<<8)
#define SNDE_COMPONENT_GEOM_TOPOS (1ull<<9)
#define SNDE_COMPONENT_GEOM_TOPO_INDICES (1ull<<10)
#define SNDE_COMPONENT_GEOM_TRIS (1ull<<11)
#define SNDE_COMPONENT_GEOM_REFPOINTS (1ull<<12)
#define SNDE_COMPONENT_GEOM_MAXRADIUS (1ull<<13)
#define SNDE_COMPONENT_GEOM_NORMALS (1ull<<14)
#define SNDE_COMPONENT_GEOM_INPLANEMAT (1ull<<15)
#define SNDE_COMPONENT_GEOM_EDGES (1ull<<16)
#define SNDE_COMPONENT_GEOM_VERTICES (1ull<<17)
#define SNDE_COMPONENT_GEOM_PRINCIPAL_CURVATURES (1ull<<18)
#define SNDE_COMPONENT_GEOM_CURVATURE_TANGENT_AXES (1ull<<19)
#define SNDE_COMPONENT_GEOM_VERTEX_EDGELIST_INDICES (1ull<<20)
#define SNDE_COMPONENT_GEOM_VERTEX_EDGELIST (1ull<<21)
#define SNDE_COMPONENT_GEOM_BOXES (1ull<<22)
#define SNDE_COMPONENT_GEOM_BOXCOORD (1ull<<23)
#define SNDE_COMPONENT_GEOM_BOXPOLYS (1ull<<24)

#define SNDE_COMPONENT_GEOM_ALL ((1ull<<17)-(1ull<<8))

// Resizing masks -- mark those arrays that resize together
//#define SNDE_COMPONENT_GEOM_COMPONENT_RESIZE (SNDE_COMPONENT_GEOM_COMPONENT)
#define SNDE_COMPONENT_GEOM_PARTS_RESIZE (SNDE_COMPONENT_GEOM_PARTS)
#define SNDE_COMPONENT_GEOM_TOPOS_RESIZE (SNDE_COMPONENT_GEOM_TOPOS)
#define SNDE_COMPONENT_GEOM_TOPO_INDICES_RESIZE (SNDE_COMPONENT_GEOM_TOPO_INDICES)
#define SNDE_COMPONENT_GEOM_TRIS_RESIZE (SNDE_COMPONENT_GEOM_TRIS|SNDE_COMPONENT_GEOM_REFPOINTS|SNDE_COMPONENT_GEOM_MAXRADIUS|SNDE_COMPONENT_GEOM_NORMALS|SNDE_COMPONENT_GEOM_INPLANEMAT)
#define SNDE_COMPONENT_GEOM_EDGES_RESIZE (SNDE_COMPONENT_GEOM_EDGES)
#define SNDE_COMPONENT_GEOM_VERTICES_RESIZE (SNDE_COMPONENT_GEOM_VERTICES|SNDE_COMPONENT_GEOM_PRINCIPAL_CURVATURES|SNDE_COMPONENT_GEOM_CURVATURE_TANGENT_AXES|SNDE_COMPONENT_GEOM_VERTEX_EDGELIST_INDICES)
#define SNDE_COMPONENT_GEOM_VERTEX_EDGELIST_RESIZE (SNDE_COMPONENT_GEOM_VERTEX_EDGELIST)
#define SNDE_COMPONENT_GEOM_BOXES_RESIZE (SNDE_COMPONENT_GEOM_BOXES|SNDE_COMPONENT_GEOM_BOXCOORD)
#define SNDE_COMPONENT_GEOM_BOXPOLYS_RESIZE (SNDE_COMPONENT_GEOM_BOXPOLYS)


#define SNDE_UV_GEOM_UVS (1ull<<32)
#define SNDE_UV_GEOM_UV_PATCHES (1ull<<33)
#define SNDE_UV_GEOM_UV_TOPOS (1ull<<34)
#define SNDE_UV_GEOM_UV_TOPO_INDICES (1ull<<35)
#define SNDE_UV_GEOM_UV_TRIANGLES (1ull<<36)
#define SNDE_UV_GEOM_INPLANE2UVCOORDS (1ull<<37)
#define SNDE_UV_GEOM_UVCOORDS2INPLANE (1ull<<38)
#define SNDE_UV_GEOM_UV_EDGES (1ull<<39)
#define SNDE_UV_GEOM_UV_VERTICES (1ull<<40)
#define SNDE_UV_GEOM_UV_VERTEX_EDGELIST_INDICES (1ull<<41)
#define SNDE_UV_GEOM_UV_VERTEX_EDGELIST (1ull<<42)
#define SNDE_UV_GEOM_UV_BOXES (1ull<<43)
#define SNDE_UV_GEOM_UV_BOXCOORD (1ull<<44)
#define SNDE_UV_GEOM_UV_BOXPOLYS (1ull<<45)
  //#define SNDE_UV_GEOM_UV_IMAGES (1ull<<13)

#define SNDE_UV_GEOM_ALL ((1ull<<46)-(1ull<<32))

// Resizing masks -- mark those arrays that resize together
#define SNDE_UV_GEOM_UVS_RESIZE (SNDE_UV_GEOM_UVS)
#define SNDE_UV_GEOM_UV_PATCHES_RESIZE (SNDE_UV_GEOM_PATCHES)
#define SNDE_UV_GEOM_UV_TOPOS_RESIZE (SNDE_UV_GEOM_UV_TOPOS)
#define SNDE_UV_GEOM_UV_TOPO_INDICES_RESIZE (SNDE_UV_GEOM_UV_TOPO_INDICES)
#define SNDE_UV_GEOM_UV_TRIANGLES_RESIZE (SNDE_UV_GEOM_UV_TRIANGLES|SNDE_UV_GEOM_INPLANE2UVCOORDS|SNDE_UV_GEOM_UVCOORDS2INPLANE)
#define SNDE_UV_GEOM_UV_EDGES_RESIZE (SNDE_UV_GEOM_UV_EDGES)
#define SNDE_UV_GEOM_UV_VERTICES_RESIZE (SNDE_UV_GEOM_UV_VERTICES|SNDE_UV_GEOM_UV_VERTEX_EDGELIST_INDICES)
#define SNDE_UV_GEOM_UV_VERTEX_EDGELIST_RESIZE (SNDE_UV_GEOM_UV_VERTEX_EDGELIST)
#define SNDE_UV_GEOM_UV_BOXES_RESIZE (SNDE_UV_GEOM_UV_BOXES|SNDE_UV_GEOM_UV_BOXCOORD)
#define SNDE_UV_GEOM_UV_BOXPOLYS_RESIZE (SNDE_UV_GEOM_UV_BOXPOLYS)
  
  
  




  
}; // close namespace

//// Also wrap a c++ constructor to compensate for the one we had
//// to remove with %nodefaultctor (in order that we could write out Python
//// constructor manually
//// see: https://stackoverflow.com/questions/33564645/how-to-add-an-alternative-constructor-to-the-target-language-specifically-pytho
//%inline %{
//  snde::lockholder *new_lockholder(void) {
//    snde::lockholder *lh = new snde::lockholder();
//    return lh;
//  }
//%}



%pythoncode %{

# lockingprocess here has an abstract base class
# defined on the c++ side with
# a specialization that calls Python
class lockingprocess_python(lockingprocess_pycpp):
  #manager=None
  lockmanager=None # lockmanager
  waiting_generators=None   # LockingPositionMap
  runnable_generators=None  # list of (generator,generator_parent,sendvalue)s
  
  #arrayreadregions=None  # VectorOfRegions
  #arraywriteregions=None # VectorOfRegions
  lastlockingposition=None # lockingposition

  # The distinction between all_tokens and used_tokens is that
  # we can temporarily lock things during the locking process,
  # but when it is complete, any of those that haven't been
  # returned can be released. So we actually return
  # only the used_tokens
  
  all_tokens=None # rwlock_token_set
  used_tokens=None # rwlock_token_set

  def __init__(self,**kwargs):
    for key in kwargs:
      if not hasattr(self,key):
        raise ValueError("Bad attribute")
      setattr(self,key,kwargs[key])
      pass
      
  @classmethod
  def execprocess(cls,lockmanager,*lock_generators):
    #arrayreadregions=VectorOfRegions(manager.locker._arrays().size())
    #arraywriteregions=VectorOfRegions(manager.locker._arrays().size())
    lastlockingposition=lockingposition(0,0,True)

    all_tokens=empty_rwlock_token_set()
    used_tokens=empty_rwlock_token_set()

    # locking generators take all_tokens  #,arrayreadregions,arraywriteregions)
    # as parameters. They yield either more generators
    # or a lockingposition. If they yield a locking position, the next()
    # call on them will cause them to perform the lock and yield None,
    # then they may yield another generator or locking position, etc. 

    waiting_generators = LockingPositionMap()

    proc=cls(lockmanager=lockmanager,
             waiting_generators=waiting_generators,
             #runnable_generators=runnable_generators,
             #arrayreadregions=arrayreadregions,
	     #arraywriteregions=arraywriteregions,
             lastlockingposition=lastlockingposition,
             all_tokens=all_tokens,
             used_tokens=used_tokens)
    proc.runnable_generators=[ (lock_generator(proc),None,None) for lock_generator in lock_generators ]

  
    while len(waiting_generators) > 0 or len(proc.runnable_generators) > 0:
      while len(proc.runnable_generators) > 0:
        (thisgen,thisgen_parent,sendvalue)=proc.runnable_generators[0]

        # Pull from generator thisgen, delegating back to parent if there is nothing left 
        gen_out=None	
        while gen_out is None:
          try:
            gen_out=thisgen.send(sendvalue)
          except StopIteration:
            pass
          if gen_out is None and thisgen_parent is not None:
            thisgen=thisgen_parent
            thisgen_parent=None
            if isinstance(thisgen,tuple):
              thisgen_parent=thisgen[1]
              thisgen=thisgen[0]
              pass
            pass
          elif gen_out is None:
            break
          pass
        proc.runnable_generators.pop(0) # this generator is no longer runnable
        
        if gen_out is not None: 
          proc.process_generated(thisgen,thisgen_parent,gen_out)
          pass
	  
        pass
      # ok... no more runnable generators... do we have a waiting generator?
      if len(waiting_generators) > 0:
        # grab the first waiting generator
        # Use C++ style iteration because that way we iterate
        # over pairs, not over keys
        iterator=waiting_generators.begin()

        (lockpos,lockcall_gen_genparent)=iterator.value()

        #sys.stderr.write("Got first waiting generator, idx=%d" % (lockpos.idx))
        #if len(waiting_generators) > 1:
        #  second_iterator=waiting_generators.begin();
        #  second_iterator+=1
        #  sys.stderr.write("; 2nd, idx=%d" % (second_iterator.value()[0].idx))
        #  pass
        #sys.stderr.write("\n")
        (lockcall,gen,genparent)=lockcall_gen_genparent.value()


        waiting_generators.erase(iterator)
        del iterator
	
        # diagnose locking order error
        if lockpos < proc.lastlockingposition :
          preexisting_only=True
          #raise ValueError("Locking order violation")
          pass
        else:
          preexisting_only=False	
          pass
        
        # perform locking operation
        res=lockcall(preexisting_only)
        if not preexisting_only: 
          proc.lastlockingposition=lockpos
          pass
	
        # .... This is now runnable... add to runnablegenerators list
        proc.runnable_generators.append((gen,genparent,res))
        pass
      pass
      
    return proc.used_tokens # ,proc.arrayreadregions,proc.arraywriteregions)
    
  def process_generated(self,thisgen,thisgen_parent,gen_out):
    assert(isinstance(gen_out,tuple))
    if gen_out[0]=="spawn":
      assert(isinstance(gen_out[1],pytypes.GeneratorType))
      # Got another generator
      self.runnable_generators.append((thisgen,thisgen_parent,None))
      self.runnable_generators.append((gen_out[1],None,None))
      pass
    elif gen_out[0]=="alloc":
      # Store with parent, so that final return of child is equivalent to return of parent
      if thisgen_parent is not None:
        self.runnable_generators.append((gen_out[1],(thisgen,thisgen_parent),None)) # append tuple of (gen_out[1], parentgen,sendvalue) to the runnable_generators list
        pass
      else:
        self.runnable_generators.append((gen_out[1],thisgen,None)) # append tuple of (gen_out[1], parentgen,sendvalue) to the runnable_generators list
        pass
      pass
    elif gen_out[0]=="allocret":
      self.runnable_generators.append((thisgen_parent,None,gen_out[1]));
    else:
      assert(gen_out[0]=="lock")
      assert(isinstance(gen_out[1],lockingposition))
      (name,posn,lockcall) = gen_out	

      #sys.stderr.write("Adding waiting generator,posn.idx=%d\n" % (posn.idx))
      self.waiting_generators.emplace_pair(lockingposition_generator(posn,CountedPyObject((lockcall,thisgen,thisgen_parent))))
      pass
    pass

  def spawn(self,lock_generator):
    newgen=lock_generator(self)
    #newfunc = lambda proc: (yield None)
    #newgen=newfunc(self)
    return ("spawn",newgen)

  def get_locks_read_array_region(self,
				  fieldaddr,
				  indexstart,numelems):
    try: 
      indexstart=long(indexstart)
      numelems=long(numelems)
      pass
    except NameError:
      # python 3
      indexstart=int(indexstart)
      numelems=int(numelems)
      pass      
    
    if not self.lockmanager._arrayidx().has_key(fieldaddr):
      raise ValueError("Array not found")
    
    #iterator = self.lockmanager._arrayidx.find(fieldaddr)
    #arrayidx = voidpp_posn_map_iterator_posn(iterator) # fromiterator(iterator).get_posn()
    arrayidx = self.lockmanager._arrayidx().get_ptr_posn(fieldaddr)

      
    if self.lockmanager.is_region_granular():
      posn=lockingposition(arrayidx,indexstart,False)
      pass
    else:
      posn=lockingposition(arrayidx,0,False)
      pass
    
    def lockcall(preexisting_only):
      if (preexisting_only):
        newset = self.lockmanager.get_preexisting_locks_read_array_region(self.all_tokens,fieldaddr,indexstart,numelems)
        pass
      else:
        newset = self.lockmanager.get_locks_read_array_region(self.all_tokens,fieldaddr,indexstart,numelems)
        pass      
      merge_into_rwlock_token_set(self.used_tokens,newset);
      
      #self.arrayreadregions[arrayidx].mark_region_noargs(indexstart,numelems)
      return (lockholder_index(fieldaddr,False,indexstart,numelems),newset)
    return ("lock",posn,lockcall)

  def get_locks_write_array_region(self,
				   fieldaddr,
				   indexstart,numelems,_dont_add_locks_to_used=False): #,_nomarkwriteregions=False):

    try: 
      indexstart=long(indexstart)
      numelems=long(numelems)
      pass
    except NameError:
      # python 3
      indexstart=int(indexstart)
      numelems=int(numelems)
      pass      

    
    
    if not self.lockmanager._arrayidx().has_key(fieldaddr):
      raise ValueError("Array not found")
    
    #iterator = self.lockmanager._arrayidx.find(fieldaddr)
    #arrayidx = voidpp_posn_map_iterator_posn(iterator) # fromiterator(iterator).get_posn()
    arrayidx = self.lockmanager._arrayidx().get_ptr_posn(fieldaddr)

      
    if self.lockmanager.is_region_granular():
      posn=lockingposition(arrayidx,indexstart,True)
      pass
    else:
      posn=lockingposition(arrayidx,0,True)
      pass
    
    def lockcall(preexisting_only):
      #addrstr=AddrStr(fieldaddr)
      #sys.stderr.write("get_locks_write(preex=%s,addr=%s,st=%d,num=%d)\n" % (str(preexisting_only),addrstr,indexstart,numelems))

      if preexisting_only: 
        newset = self.lockmanager.get_preexisting_locks_write_array_region(self.all_tokens,fieldaddr,indexstart,numelems)
        pass
      else:
        newset = self.lockmanager.get_locks_write_array_region(self.all_tokens,fieldaddr,indexstart,numelems)
        pass
     

      if not _dont_add_locks_to_used:
        merge_into_rwlock_token_set(self.used_tokens,newset);
        pass
      
      #if not(_nomarkwriteregions):
      #  self.arraywriteregions[arrayidx].mark_region_noargs(indexstart,numelems)
      #  pass
      return (lockholder_index(fieldaddr,True,indexstart,numelems),newset)
    return ("lock",posn,lockcall)

  def get_locks_array_region(self,
			     fieldname,
			     write,
			     indexstart,numelems):
    if write:
      return self.get_locks_write_array_region(fieldname,indexstart,numelems)
    else:
      return self.get_locks_read_array_region(fieldname,indexstart,numelems)
    pass
  
  
  def get_locks_array(self,
		      fieldname,
		      write):
    if write:
      return self.get_locks_write_array_region(fieldname,0,SNDE_INDEX_INVALID)
    else:
      return self.get_locks_read_array_region(fieldname,0,SNDE_INDEX_INVALID)
    pass
	
  def get_locks_array_mask(self,fieldname,maskentry,resizemaskentry,readmask,writemask,resizemask,indexstart,numelems):
    if resizemask & resizemaskentry:
      return self.get_locks_array(fieldname,True)
    elif writemask & maskentry:
      return self.get_locks_array_region(fieldname,True,indexstart,numelems)
    elif readmask & maskentry:
      return self.get_locks_array_region(fieldname,False,indexstart,numelems)
    pass
    

  # !!!*** Should also implement a realloc() function that takes a lambda
  # that can be called while the entire array is locked to determine the new size
  def alloc_array_region(self,arraymanager,fieldaddr,numelems,allocid):
        
    def alloc_func():
      numarrays_locked=0;

      #fieldaddr=geomstruct.field_address(fieldname)

      # must store iterator in a variable, lest it be a temporary that goes out of context
      # causing segmentation faults
      iter = voidpp_voidpp_multimap_pyiterator.iterate_particular_key(arraymanager.arrays_managed_by_allocator(),fieldaddr)
      for managed_array in iter:
        # lock entire array
	# but don't record it in used_tokens or mark the write regions
        
        #managed_array_str=AddrStr(managed_array)
        #field_array_str=AddrStr(fieldaddr)
        #sys.stderr.write("Getting locks of managed_array %s of array %s (idx %d)\n" % (managed_array_str,field_array_str,  self.lockmanager._arrayidx().get_ptr_posn(managed_array)))
        yield self.get_locks_write_array_region(managed_array,0,SNDE_INDEX_INVALID,_dont_add_locks_to_used=True) #,_nomarkwriteregions=True)
        numarrays_locked+=1
        pass
      assert(numarrays_locked > 0); # if this fails, you probably called with a follower array pointer, not an allocator array pointer


      (ret_fields_tokens,ret_addr) = arraymanager.alloc_arraylocked_swigworkaround(self.all_tokens,fieldaddr,numelems)

      # re-iterate over managed arrays, marking our write credentials
      #iter = voidpp_voidpp_multimap_pyiterator.iterate_particular_key(arraymanager.arrays_managed_by_allocator(),fieldaddr)
      #for managed_array in iter:
      #  arrayidx = self.lockmanager._arrayidx().get_ptr_posn(managed_array)
      #  self.arraywriteregions[arrayidx].mark_region_noargs(ret_addr,numelems)        
      #  pass

      # Iterate over returned pointers and tokens
      result=[]
      for (ret_field,token) in ret_fields_tokens:
        # keep the tokens from the allocation..., add them into self.used_tokens
        merge_into_rwlock_token_set(self.used_tokens,token)
        result.append((lockholder_index(ret_field.value(),True,ret_addr,numelems),token,allocid))
        pass
      #sys.stderr.write("allocation result: %s\n" % (str(result)))
      # Other tokens will fall out of scope
      yield ("allocret",result)
      pass
    

    return ("alloc",alloc_func())
  pass

def pylockprocess(*args,**kwargs):
  return lockingprocess_python.execprocess(*args,**kwargs)
  pass

class pylockholder_obsolete(object):
  def store(self,name_value):

    (lockname,value)=name_value
      
    setattr(self,lockname,value)
    pass

  def get(self,name):
    return getattr(self,name)

  def store_addr(self,name_tokens_addr):
    (name,tokens,addr)=name_tokens_addr
    setattr(self,name,tokens)
    setattr(self,name+"_addr",addr)
    pass
    
  def store_name(self,name,*args):
  
    if isinstance(args[0],tuple):
      # pylockprocess mode... get (name, value)
      name_value=args[0]
      (lockname,value)=name_value
      pass
    else:
      # lockingprocess_threaded_python mode... get just value
      value=args[0]
      pass
    setattr(self,name,value)
    pass
  pass
    
  
%}


