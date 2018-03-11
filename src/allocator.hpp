#ifndef SNDE_ALLOCATOR_HPP
#define SNDE_ALLOCATOR_HPP

#include <cstdint>
#include <cstring>
#include <cmath>

#include "memallocator.hpp"

#include "lockmanager.hpp"

namespace snde {
  

  /* 
     This is a toolkit for allocating and freeing pieces of a larger array.
     The larger array is made up of type <AT>
   
     Pass a pointer to a memory allocator, a pointer to an optional locker, 
     a pointer to the array pointer, 
     and the number of elements to initially reserve space for to the 
     constructor. 

     It is presumed that the creator will take care of keeping the 
     memory allocator object in memory until such time as this 
     object is destroyed. 
   

   
  */

  class allocator_alignment {
  public:
    std::vector<unsigned> address_alignment; /* required address alignments, in bytes */

    allocator_alignment()
    {
      address_alignment.push_back(8); /* always require at least 8-byte (64-bit) alignment */      
    }
    void add_requirement(unsigned alignment)
    {
      address_alignment.push_back(alignment);
    }

    unsigned get_alignment()
    {
      // alignment is least common multiple of the various elements of address_alignment
      std::vector<std::vector<unsigned>> factors;
      std::vector<std::vector<std::pair<unsigned,unsigned>>> factors_powers; 

      std::unordered_map<unsigned,unsigned> factors_maxpowers;
      
      // evaluate prime factorization
      for (unsigned reqnum=0;reqnum < address_alignment.size();reqnum++) {

	unsigned alignment = address_alignment[reqnum];

	unsigned divisor=2;
	unsigned power=0;
	unsigned maxpower=0;

	factors.emplace_back();
	factors_powers.emplace_back();
	
	while (alignment >= 2) {
	  /* evaluate possible factors up to sqrt(value), 
	     dividing them out as we find them */
	  if ((alignment % divisor)==0 ) {
	    factors[reqnum].push_back(divisor);
	    power++;
	    alignment = alignment / divisor;
	  } else {
	    if (power > 0) {
	      factors_powers[reqnum].push_back(std::make_pair(divisor,power));
	      maxpower=0;
	      if (factors_maxpowers.find(divisor) != factors_maxpowers.end()) {
		maxpower=factors_maxpowers.at(divisor);		
	      }
	      if (power > maxpower) {
		factors_maxpowers[divisor]=power;
	      }
	    }
	    power=0;
	    divisor++;
	    unsigned limit=(unsigned)(1.0+sqrt(alignment));
	    if (divisor > limit) {
	      divisor = alignment; // remaining value must be last prime factor
	    }
	    
	  }
	  
	}
	if (power > 0) {
	  factors_powers[reqnum].push_back(std::make_pair(divisor,power));
	  
	  maxpower=0;
	  if (factors_maxpowers.find(divisor) != factors_maxpowers.end()) {
	    maxpower=factors_maxpowers.at(divisor);		
	  }
	  if (power > maxpower) {
	    factors_maxpowers[divisor]=power;
	  }
	  
	}
	
	
      }
      /* Ok. Should have sorted prime factorization of all 
	 alignment requirements now */

      /* least common multiple comes from the product 
	 of the highest powers of each prime factor */

      unsigned result=1;
      for (auto & factor_maxpower : factors_maxpowers) {
	for (unsigned power=0;power < factor_maxpower.second;power++) {
	  result=result*factor_maxpower.first;
	}
      }
      return result;
    }
  };

  
  struct arrayinfo {
    void **arrayptr;
    size_t elemsize;
    bool destroyed; /* locked by allocatormutex */
  };
  
  class alloc_voidpp {
    /* Used to work around SWIG troubles with iterating over a pair with a void ** */
    public:
    void **ptr;
    alloc_voidpp(void **_ptr) : ptr(_ptr) {}
    alloc_voidpp() : ptr(NULL) {}
    void **value() {return ptr;}
  };
  
  class allocation {
    /* protected by allocator's allocatormutex */
  public:
    snde_index regionstart;
    snde_index regionend;
    
    allocation(const allocation &)=delete; /* copy constructor disabled */
    allocation& operator=(const allocation &)=delete; /* copy assignment disabled */
    
    allocation(snde_index regionstart,snde_index regionend)
    {
      this->regionstart=regionstart;
      this->regionend=regionend;
    }

    bool attempt_merge(allocation &later)
    {
      /* !!!*** NOTE: Would be more efficient if we allowed allocations to merge and breakup 
	 thereby skipping over groups of allocations during the free space search */

      assert(later.regionstart==regionend);
      regionend=later.regionend;
      return true;

    }
    
    std::shared_ptr<allocation> sp_breakup(snde_index breakpoint)
    /* breakup method ends this region at breakpoint and returns
       a new region starting at from breakpoint to the prior end */
    {
      std::shared_ptr<allocation> newregion=std::make_shared<allocation>(breakpoint,regionend);
      regionend=breakpoint;

      return newregion;
    }
      ~allocation()
    {
    }
    
    
  };
  
  

  
  class allocator /* : public allocatorbase*/ {
    /* !!!*** NOTE: will want allocator to be able to 
       handle parallel-indexed portions of arrays of different
       things, e.g. vertex array and curvature array 
       should be allocated in parallel */

    /* 
       all add_other_array() calls must be performed from a
       single thread on initialization !!! 
       
       otherwise alloc() and free() are thread safe, */
       

    std::mutex allocatormutex; // Always final mutex in locking order; protects the free list 
    
  public: 
    snde_index _totalnchunks;

    std::shared_ptr<memallocator> _memalloc;
    std::shared_ptr<lockmanager> _locker; // could be NULL if there is no locker
    std::deque<std::shared_ptr<std::function<void(snde_index)>>> realloccallbacks; // locked by allocatormutex

    bool destroyed;
    /* 
       Should lock things on allocation...
     
       Will probably need separate lock for main *_arrayptr so we can 
       wait on everything relinquishing that in order to do a realloc. 
     
    */
  
  
    //void **_arrayptr;
    //size_t _elemsize;

    // The arrays member is genuinely public for read access and
    // may be iterated over. Note that it may only be written
    // from the single thread during the initialization phase
    std::deque<struct arrayinfo> arrays;
    
    
    /* Freelist structure ... 
    */
    rangetracker<allocation> allocations;
    
    snde_index _allocchunksize; // size of chunks we allocate, in numbers of elements
  
    allocator(std::shared_ptr<memallocator> memalloc,std::shared_ptr<lockmanager> locker,std::shared_ptr<allocator_alignment> alignment,void **arrayptr,size_t elemsize,snde_index totalnelem)
    {
      // must hold writelock on array

      destroyed=false;
      
      _memalloc=memalloc;
      _locker=locker; // could be NULL if there is no locker

      arrays.push_back(arrayinfo{arrayptr,elemsize,false});
      
      //_allocchunksize = 2*sizeof(snde_index)/_elemsize;
      // Round up when determining chunk size
      _allocchunksize = (2*sizeof(snde_index) + elemsize-1)/elemsize;

      // satisfy alignment requirement on _allocchunksize
      allocator_alignment our_alignment=*alignment;
      our_alignment.add_requirement(_allocchunksize*elemsize);
      _allocchunksize = our_alignment.get_alignment()/elemsize;
      
      // _totalnchunks = totalnelem / _allocchunksize  but round up. 
      _totalnchunks = (totalnelem + _allocchunksize-1)/_allocchunksize;

      if (_totalnchunks < 2) {
	_totalnchunks=2;
      }
      // Perform memory allocation 
      *arrays[0].arrayptr = _memalloc->malloc(_totalnchunks * _allocchunksize * elemsize);

      if (_locker) {
	_locker->set_array_size(arrays[0].arrayptr,arrays[0].elemsize,_totalnchunks*_allocchunksize);
      }

      // Permanently allocate (and waste) first chunk
      // so that an snde_index==0 is otherwise invalid

      _alloc(_allocchunksize);
    
    }

    allocator(const allocator &)=delete; /* copy constructor disabled */
    allocator& operator=(const allocator &)=delete; /* assignment disabled */

    size_t add_other_array(void **arrayptr, size_t elsize)
    /* returns index */
    {
      assert(!destroyed);
      size_t retval=arrays.size();
      arrays.push_back(arrayinfo {arrayptr,elsize,false});

      if (*arrays[0].arrayptr) {
	/* if main array already allocated */
	*arrayptr=_memalloc->calloc(_totalnchunks*_allocchunksize * elsize);
      } else {
        *arrayptr = nullptr;
      }
      return retval;
    }

    size_t num_arrays(void)
    {
      size_t size=0;
      for (auto & ary: arrays) {
	if (!ary.destroyed) size++;
      }
      return size;
    }
    
    void remove_array(void **arrayptr)
    {
      std::unique_lock<std::mutex> lock(allocatormutex);
      for (auto ary=arrays.begin();ary != arrays.end();ary++) {
	if (ary->arrayptr == arrayptr) {
	  if (ary==arrays.begin()) {
	    /* removing our master array invalidates the entire allocator */
	    destroyed=true; 
	  }
	  _memalloc->free(*ary->arrayptr);
	  //arrays.erase(ary);
	  ary->destroyed=true; 
	  return;
	}
      }
    }
    
    void _realloc(snde_index newnchunks) {
      assert(!destroyed);
      // Must hold write lock on entire array
      // must hold allocatormutex
      _totalnchunks = newnchunks;
      //*arrays[0].arrayptr = _memalloc->realloc(*arrays[0].arrayptr,_totalnchunks * _allocchunksize * _elemsize);

      /* resize all arrays  */
      for (size_t cnt=0;cnt < arrays.size();cnt++) {
	if (arrays[cnt].destroyed) continue;
	*arrays[cnt].arrayptr= _memalloc->realloc(*arrays[cnt].arrayptr,_totalnchunks * _allocchunksize * arrays[cnt].elemsize);
      
	if (_locker) {
	  size_t arraycnt;
	  for (arraycnt=0;arraycnt < arrays.size();arraycnt++) {
	    _locker->set_array_size(arrays[arraycnt].arrayptr,arrays[arraycnt].elemsize,_totalnchunks*_allocchunksize);
	  }
	}
	
      }
      
    }

    snde_index total_nelem() {
      std::lock_guard<std::mutex> lock(allocatormutex); // Lock the allocator mutex 
      assert(!destroyed);
      return _totalnchunks*_allocchunksize;
    }
    

    snde_index _alloc(snde_index nelem)
    {
      // Step through gaps in the range, looking for a chunk big enough
      snde_index retpos;
      
      bool reallocflag=false;


      // Number of chunks we need... nelem/_allocchunksize rounding up
      snde_index allocchunks = (nelem+_allocchunksize-1)/_allocchunksize;
      std::unique_lock<std::mutex> lock(allocatormutex);

      assert(!destroyed);

      std::shared_ptr<allocation> alloc=allocations.find_unmarked_region(0, _totalnchunks, allocchunks);

      if (alloc==nullptr) {
	snde_index newnchunks=(snde_index)((_totalnchunks+allocchunks)*1.7); /* Reserve extra space, not just bare minimum */
	
	
	this->_realloc(newnchunks);
	reallocflag=true;
	
	alloc=allocations.find_unmarked_region(0, _totalnchunks, allocchunks);
      }

      retpos=SNDE_INDEX_INVALID;
      if (alloc) {
	retpos=alloc->regionstart*_allocchunksize;
	allocations.mark_region(alloc->regionstart,alloc->regionend-alloc->regionstart);
      }

      // !!!*** Need to implement merge to get O(1) performance
      allocations.merge_adjacent_regions();
      
            
      if (reallocflag) {
	// notify recipients that we reallocated
	std::deque<std::shared_ptr<std::function<void(snde_index)>>> realloccallbacks_copy(realloccallbacks); // copy can be iterated with allocatormutex unlocked
	
	for (std::deque<std::shared_ptr<std::function<void(snde_index)>>>::iterator reallocnotify=realloccallbacks_copy.begin();reallocnotify != realloccallbacks_copy.end();reallocnotify++) {
	  snde_index new_total_nelem=total_nelem();
	  lock.unlock(); // release allocatormutex
	  (**reallocnotify)(new_total_nelem);
	  lock.lock();
	}
	
      }
      return retpos;
    }
    
    std::pair<snde_index,std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>>> alloc_arraylocked(rwlock_token_set all_locks,snde_index nelem)
    {
      // must hold write lock on entire array... returns write lock on new allocation
      // and position... Note that the new allocation is not included in the
      // original lock on the entire array, so you can freely release the
      // original if you don't otherwise need it

      snde_index retpos = _alloc(nelem);
      std::vector<std::pair<std::shared_ptr<alloc_voidpp>,rwlock_token_set>> retlocks;

      // NOTE: our admin lock is NOT locked for this
      
      // notify locker of new allocation
      if (_locker && retpos != SNDE_INDEX_INVALID) {
	size_t arraycnt;
	
	for (arraycnt=0;arraycnt < arrays.size();arraycnt++) {
	  rwlock_token token;
	  rwlock_token_set token_set;

	  token_set=empty_rwlock_token_set();
	  token=_locker->newallocation(all_locks,arrays[arraycnt].arrayptr,retpos,nelem,arrays[arraycnt].elemsize);
	  (*token_set)[token->mutex()]=token;

	  retlocks.push_back(std::make_pair(std::make_shared<alloc_voidpp>(arrays[arraycnt].arrayptr),token_set));
	}
      }
      
      return std::make_pair(retpos,retlocks);
    }

    //std::pair<rwlock_token_set,snde_index> alloc(snde_index nelem) {
    //  // must be in a locking process or otherwise where we can lock the entire array...
    //  // this locks the entire array, allocates the new element, and
    //  // releases the rest of the array
    //
    //}

    void register_realloc_callback(std::shared_ptr<std::function<void(snde_index)>> callback)
    {
      std::lock_guard<std::mutex> lock(allocatormutex); // Lock the allocator mutex 
      
      realloccallbacks.emplace_back(callback);
    }

    void unregister_realloc_callback(std::shared_ptr<std::function<void(snde_index)>> callback)
    {
      std::lock_guard<std::mutex> lock(allocatormutex); // Lock the allocator mutex 

      for (size_t pos=0;pos < realloccallbacks.size();) {
	
	if (realloccallbacks[pos]==callback) {
	  realloccallbacks.erase(realloccallbacks.begin()+pos);
	} else {
	  pos++;
	}
      }
    }

    void _free(snde_index addr,snde_index nelem)
    {
      
      // Number of chunks we need... nelem/_allocchunksize rounding up


    
      snde_index chunkaddr;
      snde_index freechunks = (nelem+_allocchunksize-1)/_allocchunksize;
      std::lock_guard<std::mutex> lock(allocatormutex); // Lock the allocator mutex 

      assert(!destroyed);

      assert(addr != SNDE_INDEX_INVALID);
      assert(nelem != SNDE_INDEX_INVALID);
      assert(addr > 0); /* addr==0 is wasted element allocated by constructor */
      assert(addr % _allocchunksize == 0); /* all addresses returned by alloc() are multiples of _allocchunksize */


      chunkaddr=addr/_allocchunksize;
    

      allocations.clear_region(chunkaddr,freechunks);
	
    }
    
    void free(snde_index addr,snde_index nelem)
    {
      // must hold write lock on entire array... (or is it sufficient to hold the
      // write lock on just this element...? not currently because
      // we will be modifying freelist pieces outside this element... on
      // the other hand we will hold allocatormutex during the free operation
      // and caches don't do much with free pieces... we'd be OK so long as
      // nobody else has a write lock on the entire array and is modifying the
      // cached copy)



      // notify locker of free operation
      if (_locker) {
	size_t arraycnt;
	for (arraycnt=0;arraycnt < arrays.size();arraycnt++) {
	  _locker->freeallocation(arrays[arraycnt].arrayptr,addr,nelem,arrays[arraycnt].elemsize);
	}
      }
      _free(addr,nelem);
    }
    ~allocator() {
      // _memalloc was provided by our creator and is not freed
      // _locker was provided by our creator and is not freed
      std::lock_guard<std::mutex> lock(allocatormutex); // Lock the allocator mutex 

      // free all arrays
      for (size_t cnt=0;cnt < arrays.size();cnt++) {
	if (*arrays[cnt].arrayptr) {
	  if (!arrays[cnt].destroyed) {
	    _memalloc->free(*arrays[cnt].arrayptr);
	    *(arrays[cnt].arrayptr) = NULL;
	  }
	}
      }
      arrays.clear();
      
    }
  };
  
}

#endif /* SNDE_ALLOCATOR_HPP */
