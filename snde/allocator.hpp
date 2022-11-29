#ifndef SNDE_ALLOCATOR_HPP
#define SNDE_ALLOCATOR_HPP

#include <mutex>
#include <cstdint>
#include <cstring>
#include <cmath>

#include "snde/memallocator.hpp"

#include "snde/lockmanager.hpp"

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

  std::map<size_t,size_t> prime_factorization(size_t number);

  size_t multiply_factors(std::map<size_t,size_t> factors);
  
  class allocator_alignment {
  public:
    std::mutex admin; // locks access to members. Last in locking order
    std::vector<size_t> address_alignment; /* required address alignments, in bytes */
    std::shared_ptr<size_t> cached_alignment;

    allocator_alignment() 
    {
      address_alignment.push_back(8); /* always require at least 8-byte (64-bit) alignment */      
    }
    void add_requirement(size_t alignment)
    {
      std::lock_guard<std::mutex> adminlock(admin);
      address_alignment.push_back(alignment);
      cached_alignment = nullptr; 
    }

    size_t get_alignment()
    {
      std::lock_guard<std::mutex> adminlock(admin);

      if (cached_alignment) {
	return *cached_alignment;
      }
      // alignment is least common multiple of the various elements of address_alignment
      //std::vector<std::vector<unsigned>> factors;
      std::vector<std::map<size_t,size_t>> factors_powers; 

      std::unordered_map<size_t,size_t> factors_maxpowers;

      //fprintf(stderr,"get_alignment():\n");
      // evaluate prime factorization
      for (size_t reqnum=0;reqnum < address_alignment.size();reqnum++) {

	size_t alignment = address_alignment[reqnum];


	//fprintf(stderr," alignment requirement: %d\n",alignment);

	//factors.emplace_back();
	factors_powers.emplace_back(prime_factorization(alignment));


	// adjust stored maximum power for each divisor
	std::vector<std::map<size_t,size_t>>::iterator factors_powers_lastentry = factors_powers.end()-1;
	for (auto && factor_power: *factors_powers_lastentry) {
	  // adjust factors_maxpowers entry if power is greater than this
	  size_t maxpower=0;
	  size_t divisor=factor_power.first;
	  size_t power=factor_power.second;
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

      size_t result=1;
      for (auto & factor_maxpower : factors_maxpowers) {
	for (size_t power=0;power < factor_maxpower.second;power++) {
	  result=result*factor_maxpower.first;
	}
      }
      //fprintf(stderr,"alignment result: %d\n",result);
      cached_alignment=std::make_shared<size_t>(result);
      return result;
    }


    static void *alignment_shift(void *ptr,size_t alignment) // alignment should be from get_alignment(). ptr must have at least this much extra space available
    
    {
      if (!alignment) {
	return ptr;
      }
      
      // check for power of 2
      if ( (alignment & (alignment-1))==0) {
	// power of 2 alignments, use std::align
	size_t space = 1+(size_t)alignment;
	std::align(alignment,1,ptr,space);
	return ptr; 
      }

      // Otherwise cast to uintptr_t and do pointer math
      uintptr_t ptrval = (uintptr_t)ptr;
      uintptr_t modulus = (ptrval % alignment);

      if (!modulus) {
	return ptr; // already aligned
      }

      ptrval += (alignment-modulus); 
      assert(!(ptrval % alignment));

      ptr = (void *)ptrval;
      return ptr; 
    }
    
  };

  
  struct arrayinfo {
    void **arrayptr;
    size_t elemsize;
    memallocator_regionid id;
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
    snde_index nelem; // valid only form allocations_unmerged... number of elements requested in original allocation NOTE THAT THIS IS IN ELEMENTS, NOT CHUNKS.... regionstart and regionend are in CHUNKS -- each one is _allocchunksize elements
    
    allocation(const allocation &)=delete; /* copy constructor disabled */
    allocation& operator=(const allocation &)=delete; /* copy assignment disabled */
    
    allocation(snde_index regionstart,snde_index regionend,snde_index nelem)
    {
      this->regionstart=regionstart;
      this->regionend=regionend;
      this->nelem=nelem;
    }

    bool attempt_merge(allocation &later)
    {
      /* !!!*** NOTE: Would be more efficient if we allowed allocations to merge and breakup 
	 thereby skipping over groups of allocations during the free space search */

      assert(later.regionstart==regionend);
      regionend=later.regionend;
      nelem=SNDE_INDEX_INVALID; /* don't know how many elements after merge */
      return true;

    }
    
    std::shared_ptr<allocation> sp_breakup(snde_index breakpoint,snde_index nelem)
    /* breakup method ends this region at breakpoint and returns
       a new region starting at from breakpoint to the prior end */
    {
      std::shared_ptr<allocation> newregion=std::make_shared<allocation>(breakpoint,regionend,SNDE_INDEX_INVALID);

      nelem=SNDE_INDEX_INVALID; /* don't know how many elements after breakup */
      regionend=breakpoint;
      
      return newregion;
    }
      ~allocation()
    {
    }
    
    
  };
  
  

  
  class allocator /* : public allocatorbase*/ {
    /* NOTE: allocator can 
       handle parallel-indexed portions of arrays of different
       things, e.g. vertex array and curvature array 
       should be allocated in parallel. This is done 
    through add_other_array() */

    /* 
       
       most methods, including alloc() and free() are thread safe, */
       

    std::mutex allocatormutex; // Always final mutex in locking order; protects the free list 
    
  public: 
    snde_index _totalnchunks;
    size_t maxaddressbytes;
    size_t _elemsize;

    std::shared_ptr<memallocator> _memalloc;
    std::shared_ptr<lockmanager> _locker; // could be NULL if there is no locker
    std::deque<std::shared_ptr<std::function<void(snde_index)>>> pool_realloc_callbacks; // locked by allocatormutex

    // These next 4 parameters are immutable once assigned
    std::string recording_path;
    uint64_t recrevision; 
    uint64_t originating_rss_unique_id;
    memallocator_regionid id; // ID of primary array

    bool destroyed;
    /* 
       Should lock things on allocation...
     
       Will probably need separate lock for main *_arrayptr so we can 
       wait on everything relinquishing that in order to do a pool realloc. 
     
    */
  
  
    //void **_arrayptr;
    //size_t _elemsize;

    // The arrays member is genuinely public for read access through
    // the arrays() accessor and
    // may be iterated over. Note that it may only be written
    // to when allocatormutex is locked.
    // Elements may never be deleted, so as to maintain numbering.
    // The shared_ptr around _arrays
    // is atomic, so it may be freely read through the accessor
    std::shared_ptr<std::deque<struct arrayinfo>> _arrays; // atomic shared_ptr 

    // alignment requirement
    allocator_alignment our_alignment; // requirements from our element size and GPU alignment requirements
    allocator_alignment sub_alignment; // like our_alignment but in terms of elements, not bytes!
    
    /* Freelist structure ... 
    */
    rangetracker<allocation> allocations;
    rangetracker<allocation> allocations_unmerged; /* Unmerged copy of allocations... we can use this to know the length of each allocation for 
						      parameterless free() */
    
    snde_index _allocchunksize; // size of chunks we allocate, in numbers of elements
  
    allocator(std::shared_ptr<memallocator> memalloc,std::shared_ptr<lockmanager> locker,std::string recording_path,uint64_t recrevision,uint64_t originating_rss_unique_id,memallocator_regionid id,std::shared_ptr<allocator_alignment> alignment,void **arrayptr,size_t elemsize,snde_index totalnelem,const std::set<snde_index>& follower_elemsizes,size_t maxaddressbytes) :
      recording_path(recording_path),
      recrevision(recrevision),
      originating_rss_unique_id(originating_rss_unique_id),
      id(id),
      maxaddressbytes(maxaddressbytes),
      _elemsize(elemsize)
    {
      // must hold writelock on array

      destroyed=false;
      
      _memalloc=memalloc;
      _locker=locker; // could be NULL if there is no locker

      std::shared_ptr<std::deque<struct arrayinfo>> new_arrays=std::make_shared<std::deque<struct arrayinfo>>();
      new_arrays->push_back(arrayinfo{arrayptr,elemsize,id,false});
      std::atomic_store(&_arrays,new_arrays);      
      
      
      //_allocchunksize = 2*sizeof(snde_index)/_elemsize;
      // Round up when determining chunk size
      _allocchunksize = (2*sizeof(snde_index) + elemsize-1)/elemsize;

      if (alignment) {
	// satisfy alignment requirement on _allocchunksize
	//our_alignment = *alignment;
	std::lock_guard<std::mutex> alignadmin(alignment->admin);
	our_alignment.address_alignment = alignment->address_alignment;
	
      }
      our_alignment.add_requirement(((size_t)_allocchunksize)*elemsize);
      
      //for (auto && follower_elemsize: follower_elemsizes) {
      //our_alignment.add_requirement(follower_elemsize);
      //}

      size_t initial_allocchunksize = our_alignment.get_alignment()/elemsize;

      // identify prime factors of preexisting alignment requirements
      // present in elemsize, and create sub_alignment with those
      // factors divided out (if present)
      std::map<size_t,size_t> elemsizeprimefactors = prime_factorization(elemsize);

      
      //sub_alignment = our_alignment; // copy our_alignment
      sub_alignment.address_alignment=our_alignment.address_alignment;
      
      // now divide factors from elemsize out of sub_alignment
      for (snde_index align_idx=0;align_idx < sub_alignment.address_alignment.size();align_idx++) {
	std::map<size_t,size_t> alignprimefactors = prime_factorization(sub_alignment.address_alignment.at(align_idx));

	for (auto && primefactor_power: elemsizeprimefactors) {
	  auto alignfactorref = alignprimefactors.find(primefactor_power.first);
	  if (alignfactorref != alignprimefactors.end()) {
	    // found this factor; reduce it appropriately.
	    if (alignfactorref->second <= primefactor_power.second) {
	      alignfactorref->second=0;
	    } else {
	      alignfactorref->second -= primefactor_power.second;
	    }
	  }
	}

	sub_alignment.address_alignment.at(align_idx) = multiply_factors(alignprimefactors);
	
      }
      
      // now add in follower element size requirements, likewise
      // with address alignment factors divided out
      for (auto && follower_elemsize: follower_elemsizes) {
	std::map<size_t,size_t> follower_elemsize_primefactors = prime_factorization(follower_elemsize);
	//fprintf(stderr,"follower elemsize: %d\n",(int)follower_elemsize);

	for (auto alignreq: alignment->address_alignment) {
	  // for each address alignment requirement
	  
	  // alignreq is a size_t
	  std::map<size_t,size_t> alignreqprimefactors = prime_factorization(alignreq);
	  // for this follower, any common prime factors in the element size
	  // and the alignment requirement reduce the alignment requirement
	  
	  for (auto && primefactor_power: alignreqprimefactors) {
	    auto followerfactor = follower_elemsize_primefactors.find(primefactor_power.first);
	    if (followerfactor != follower_elemsize_primefactors.end()) {
	      if (followerfactor->second <= primefactor_power.second) {
		primefactor_power.second -= followerfactor->second;
	      } else {
		primefactor_power.second=0;
	      }
	    }
	  }
	  sub_alignment.add_requirement(multiply_factors(alignreqprimefactors));

	  
	}
      }
	
    
      
      _allocchunksize = ((size_t)initial_allocchunksize)*sub_alignment.get_alignment();
            
				    
      // _totalnchunks = totalnelem / _allocchunksize  but round up. 
      _totalnchunks = (totalnelem + _allocchunksize-1)/_allocchunksize;
      
      if (_totalnchunks < 2) {
	_totalnchunks=2;
      }
      // Perform memory allocation 
      *(*arrays()).at(0).arrayptr = _memalloc->malloc(recording_path,recrevision,originating_rss_unique_id,id,_totalnchunks * _allocchunksize * elemsize, maxaddressbytes);

      if (_locker) {
	_locker->set_array_size((*arrays()).at(0).arrayptr,(*arrays()).at(0).elemsize,_totalnchunks*_allocchunksize);
      }

      // Permanently allocate (and waste) first chunk
      // so that an snde_index==0 is otherwise invalid

      _alloc(_allocchunksize);
    
    }

    allocator(const allocator &)=delete; /* copy constructor disabled */
    allocator& operator=(const allocator &)=delete; /* assignment disabled */

    // accessor for atomic _arrays member
    std::shared_ptr<std::deque<struct arrayinfo>> arrays()
    {
      return std::atomic_load(&_arrays);
    }

    std::tuple<std::shared_ptr<std::deque<struct arrayinfo>>> _begin_atomic_update()
    // allocatormutex must be locked when calling this function...
    // it returns new copies of the atomically-guarded data
    {

      // Make copies of atomically-guarded data 
      std::shared_ptr<std::deque<struct arrayinfo>> new_arrays=std::make_shared<std::deque<struct arrayinfo>>(*arrays());

      return std::make_tuple(new_arrays);
    }

    void _end_atomic_update(std::shared_ptr<std::deque<struct arrayinfo>> new_arrays)
    // allocatormutex must be locked when calling this function...
    {
      
      // replace old with new

      std::atomic_store(&_arrays,new_arrays);      
    }



    size_t add_other_array(memallocator_regionid other_array_id,void **arrayptr, size_t elsize)
    /* returns index */
    {
      std::lock_guard<std::mutex> lock(allocatormutex);

      std::shared_ptr<std::deque<struct arrayinfo>> new_arrays;
      std::tie(new_arrays) = _begin_atomic_update();
      
      assert(!destroyed);
      size_t retval=new_arrays->size();
      new_arrays->push_back(arrayinfo {arrayptr,elsize,other_array_id,false});

      size_t newmaxbytes = (size_t)((double)maxaddressbytes * (double)elsize / (double)_elemsize);

      if (*(*new_arrays)[0].arrayptr) {
	/* if main array already allocated */
          //snde_warning("Allocating %ull and reserving %ull for base el size %ull and other el size %ull", _totalnchunks * _allocchunksize * elsize, newmaxbytes, elsize, _elemsize);
	*arrayptr=_memalloc->calloc(recording_path,recrevision,originating_rss_unique_id,other_array_id,_totalnchunks*_allocchunksize * elsize,newmaxbytes);
      } else {
        *arrayptr = nullptr;
      }
      _end_atomic_update(new_arrays);
      return retval;
    }

    size_t num_arrays(void)
    {
      std::lock_guard<std::mutex> lock(allocatormutex);
      size_t size=0;
      auto arrays_loc = arrays();
      for (auto & ary: *arrays_loc) {
	if (!ary.destroyed) size++;
      }
      return size;
    }
    
    void remove_array(void **arrayptr)
    {
      std::unique_lock<std::mutex> lock(allocatormutex);
      size_t index=0;
      // we hold allocatormutex, so  _arrays  should not be accessed directly, but won't change
      
      for (std::deque<struct arrayinfo>::iterator ary=arrays()->begin();ary != arrays()->end();ary++,index++) {
	if (ary->arrayptr == arrayptr) {
	  if (ary==arrays()->begin()) {
	    /* removing our master array invalidates the entire allocator */
	    destroyed=true; 
	  }
	  _memalloc->free(recording_path,recrevision,originating_rss_unique_id,ary->id,*ary->arrayptr);
	  //arrays.erase(ary);
	  ary->destroyed=true; 
	  return;
	}
      }
    }

    
    void _pool_realloc(snde_index newnchunks) {
      /* This routine reallocates the entire array memory pool, in case we run out of space in it */
      
      
      assert(!destroyed);
      // Must hold write lock on entire array
      // must hold allocatormutex... therefore arrays() won't change although it's still unsafe to read _arrays directly
      _totalnchunks = newnchunks;
      //*arrays[0].arrayptr = _memalloc->realloc(*arrays.at(0).arrayptr,_totalnchunks * _allocchunksize * _elemsize,0);

      /* resize all arrays  */
      for (size_t cnt=0;cnt < arrays()->size();cnt++) {
	if ((*arrays()).at(cnt).destroyed) continue;
	*arrays()->at(cnt).arrayptr= _memalloc->realloc(recording_path,recrevision,originating_rss_unique_id,arrays()->at(cnt).id,*arrays()->at(cnt).arrayptr,arrays()->at(cnt).elemsize*_totalnchunks*_allocchunksize);
      
	if (_locker) {
	  size_t arraycnt;
	  for (arraycnt=0;arraycnt < arrays()->size();arraycnt++) {
	    _locker->set_array_size(arrays()->at(arraycnt).arrayptr,arrays()->at(arraycnt).elemsize,_totalnchunks*_allocchunksize);
	  }
	}
	
      }
      
    }

    snde_index _total_nelem()
    // return total number of elements in pool ALLOCATORMUTEX MUST BE LOCKED!
    {
      assert(!destroyed);
      return _totalnchunks*_allocchunksize;
    }

    snde_index total_nelem()
    // return total number of elements in pool
    {
      std::lock_guard<std::mutex> lock(allocatormutex); // Lock the allocator mutex
      return _total_nelem();
    }

    snde_index space_needed()
    // return size (in elements) of minimal pool into which currently
    // allocated data could be copied without changing
    // indices
    {
      std::lock_guard<std::mutex> lock(allocatormutex); // Lock the allocator mutex
      snde_index space_end=0;
      
      auto allocation=allocations.end();

      if (allocation==allocations.begin()) {
	// empty -- no space needed
	return 0;
      }
      allocation--;

      return allocation->second->regionend*_allocchunksize;
    }


    snde_index _alloc(snde_index nelem)
    {
      // Step through gaps in the range, looking for a chunk big enough
      snde_index retpos;
      
      bool pool_reallocflag=false;


      // Number of chunks we need... nelem/_allocchunksize rounding up
      snde_index allocchunks = (nelem+_allocchunksize-1)/_allocchunksize;
      std::unique_lock<std::mutex> lock(allocatormutex);

      assert(!destroyed);

      std::shared_ptr<allocation> alloc=allocations.find_unmarked_region(0, _totalnchunks, allocchunks,SNDE_INDEX_INVALID);

      if (alloc==nullptr) {
	snde_index newnchunks=(snde_index)((_totalnchunks+allocchunks)*1.7); /* Reserve extra space, not just bare minimum */
	
	
	this->_pool_realloc(newnchunks);
	pool_reallocflag=true;
	
	alloc=allocations.find_unmarked_region(0, _totalnchunks, allocchunks,SNDE_INDEX_INVALID);
      }

      retpos=SNDE_INDEX_INVALID;
      if (alloc) {
	retpos=alloc->regionstart*_allocchunksize;
	allocations.mark_region(alloc->regionstart,alloc->regionend-alloc->regionstart,nelem);
	allocations_unmerged.mark_region(alloc->regionstart,alloc->regionend-alloc->regionstart,nelem);
      }

      // !!!*** Need to implement merge to get O(1) performance
      allocations.merge_adjacent_regions();
      
            
      if (pool_reallocflag) {
	// notify recipients that we reallocated
	std::deque<std::shared_ptr<std::function<void(snde_index)>>> realloccallbacks_copy(pool_realloc_callbacks); // copy can be iterated with allocatormutex unlocked
	
	for (std::deque<std::shared_ptr<std::function<void(snde_index)>>>::iterator reallocnotify=realloccallbacks_copy.begin();reallocnotify != realloccallbacks_copy.end();reallocnotify++) {
	  snde_index new_total_nelem=_total_nelem();
	  lock.unlock(); // release allocatormutex
	  (**reallocnotify)(new_total_nelem);
	  lock.lock();
	}
	
      }
      //fprintf(stderr,"_alloc return %d (_allocchunksize=%d)\n",retpos,_allocchunksize);
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
	
	auto curarrays=arrays();
	for (arraycnt=0;arraycnt < curarrays->size();arraycnt++) {
	  rwlock_token token;
	  rwlock_token_set token_set;
	  auto &thisarray=curarrays->at(arraycnt);

	  token_set=empty_rwlock_token_set();
	  token=_locker->newallocation(all_locks,thisarray.arrayptr,retpos,nelem,thisarray.elemsize);
	  (*token_set)[token->mutex()]=token;

	  retlocks.push_back(std::make_pair(std::make_shared<alloc_voidpp>(thisarray.arrayptr),token_set));
	}
      }

      //fprintf(stderr,"alloc_arraylocked(%llu)->%llu\n",nelem,retpos);
      return std::make_pair(retpos,retlocks);
    }


    snde_index alloc_nolocking(snde_index nelem)
    {
      // allocation function for when you are not using locking at all
      
      snde_index retpos = _alloc(nelem);
      return retpos;
    }


    
    //std::pair<rwlock_token_set,snde_index> alloc(snde_index nelem) {
    //  // must be in a locking process or otherwise where we can lock the entire array...
    //  // this locks the entire array, allocates the new element, and
    //  // releases the rest of the array
    //
    //}


    /* Realloc callbacks are used to notify when the array's memory pool changes address
       (i.e. because the entire pool has been reallocated because more space is needed)  

       Note that this happens immediately upon the reallocation. Somebody generally has
       a write lock on the entire array during such allocation. The callback is called
       without allocatormutex being held
    */
    
    void register_pool_realloc_callback(std::shared_ptr<std::function<void(snde_index)>> callback)
    {
      std::lock_guard<std::mutex> lock(allocatormutex); // Lock the allocator mutex 
      
      pool_realloc_callbacks.emplace_back(callback);
    }

    void unregister_pool_realloc_callback(std::shared_ptr<std::function<void(snde_index)>> callback)
    {
      std::lock_guard<std::mutex> lock(allocatormutex); // Lock the allocator mutex 

      for (size_t pos=0;pos < pool_realloc_callbacks.size();) {
	
	if (pool_realloc_callbacks[pos]==callback) {
	  pool_realloc_callbacks.erase(pool_realloc_callbacks.begin()+pos);
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
    

      allocations.clear_region(chunkaddr,freechunks,SNDE_INDEX_INVALID);
      allocations_unmerged.clear_region(chunkaddr,freechunks,SNDE_INDEX_INVALID);
	
    }

    snde_index get_length(snde_index addr)
    {
      std::lock_guard<std::mutex> lock(allocatormutex); // Lock the allocator mutex
      
      std::shared_ptr<allocation> alloc = allocations_unmerged.get_region(addr/_allocchunksize);

      snde_index nelem = alloc->nelem; /* a null pointer here would mean that addr is not a validly allocated region */

      return nelem;
    }
    
    void free(snde_index addr)
    {
      // must hold write lock on this allocation (I DON"T THINK THIS IS NECESSARY ANYMORE ... except possibly for
      // the freeallocation() calls, below)

      snde_index nelem=get_length(addr);
      
      // notify locker of free operation
      if (_locker) {
	
	size_t arraycnt;
	auto curarrays=arrays();
	for (arraycnt=0;arraycnt < curarrays->size();arraycnt++) {
	  auto &thisarray=curarrays->at(arraycnt);
	  
	  _locker->freeallocation(thisarray.arrayptr,addr,nelem,thisarray.elemsize);
	}
      }
      _free(addr,nelem);
    }

    void _realloc_down(snde_index addr, snde_index orignelem, snde_index newnelem)
    /* Shrink an allocation to the newly specified size. You must 
       have a write lock on the allocation, but not necessarily the entire array */
    {
      snde_index chunkaddr;

      assert(orignelem != SNDE_INDEX_INVALID);
      assert(newnelem != SNDE_INDEX_INVALID);

      assert(orignelem >= newnelem);
      
      


      snde_index origchunks = (orignelem+_allocchunksize-1)/_allocchunksize;
      snde_index newchunks = (newnelem+_allocchunksize-1)/_allocchunksize;


      assert(addr != SNDE_INDEX_INVALID);
      assert(addr > 0); /* addr==0 is wasted element allocated by constructor */
      assert(addr % _allocchunksize == 0); /* all addresses returned by alloc() are multiples of _allocchunksize */

      
      std::lock_guard<std::mutex> lock(allocatormutex); // Lock the allocator mutex 

      {
	std::shared_ptr<allocation> alloc = allocations_unmerged.get_region(addr/_allocchunksize);
	assert(alloc->nelem == orignelem);
	
	if (newchunks==origchunks) {
	  alloc->nelem = newnelem;
	  return;
	}
      }
      chunkaddr=addr/_allocchunksize;

      
      allocations.clear_region(chunkaddr+newchunks,origchunks-newchunks,SNDE_INDEX_INVALID);
      allocations_unmerged.clear_region(chunkaddr+newchunks,origchunks-newchunks,SNDE_INDEX_INVALID);

      // must hold write lock on this allocation (NOT ANYMORE?)

      // mark reduced nelem in this allocation
      if (newnelem > 0) {
	std::shared_ptr<allocation> alloc = allocations_unmerged.get_region(chunkaddr);
	alloc->nelem = newnelem;
      }
    }

    void realloc_down(snde_index addr,snde_index orignelem,snde_index newnelem)
    {


      // notify locker of free operation
      if (_locker) {
	size_t arraycnt;
	auto curarrays=arrays();
	for (arraycnt=0;arraycnt < curarrays->size();arraycnt++) {
	  auto &thisarray=curarrays->at(arraycnt);
	  
	  _locker->realloc_down_allocation(thisarray.arrayptr,addr,orignelem,newnelem);
	}
      }
      _realloc_down(addr,orignelem,newnelem);
    }
    
    ~allocator() {
      // _memalloc was provided by our creator and is not freed
      // _locker was provided by our creator and is not freed
      std::lock_guard<std::mutex> lock(allocatormutex); // Lock the allocator mutex 

      // free all arrays
      for (size_t cnt=0;cnt < arrays()->size();cnt++) {
	if (*arrays()->at(cnt).arrayptr) {
	  if (!arrays()->at(cnt).destroyed) {
	    _memalloc->free(recording_path,recrevision,originating_rss_unique_id,arrays()->at(cnt).id,*arrays()->at(cnt).arrayptr);
	    *(arrays()->at(cnt).arrayptr) = NULL;
	  }
	}
      }
      /* Should we explicitly go through and empty arrays? I
	 don't think it will accomplish anything... */
      arrays().reset();
      
    }
  };
  
}

#endif /* SNDE_ALLOCATOR_HPP */
