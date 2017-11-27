#ifndef SNDE_ALLOCATOR
#define SNDE_ALLOCATOR

#include <stdint.h>

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

  template <class AT> class allocator {
    /* !!!*** NOTE: will want allocator to be able to 
       handle parallel-indexed portions of arrays of different
       things, e.g. vertex array and curvature array 
       should be allocated in parallel */

    std::mutex allocatormutex; // Always final mutex in locking order; protects the free list 
    
  public: 
    snde_index _firstfree;
    snde_index _totalnchunks;

    memallocator *_memalloc;
    lockmanager *_locker; // could be NULL if there is no locker
    /* 
       Should lock things on allocation...
     
       Will probably need separate lock for main *_arrayptr so we can 
       wait on everything relinquishing that in order to do a realloc. 
     
    */
  
  
    AT **_arrayptr;

    /* Freelist structure ... But read via memcpy to avoid 
       aliasing problems... 
       snde_index freeblocksize  (in chunks)
       snde_index nextfreeblockstart ... nextfreeblockstart of last block should be _totalnchunks
    */
    snde_index _allocchunksize; // size of chunks we allocate, in numbers of <AT> elements
  
    allocator<AT>(memallocator *memalloc,lockmanager *locker,AT **arrayptr,snde_index totalnelem) {
      // must hold writelock on array
    
      _memalloc=memalloc;
      _locker=locker; // could be NULL if there is no locker
      _firstfree=0;
      _arrayptr=arrayptr;

      //_allocchunksize = 2*sizeof(snde_index)/sizeof(AT);
      // Round up when determining chunk size
      _allocchunksize = (2*sizeof(snde_index) + sizeof(AT)-1)/sizeof(AT);

      // _totalnchunks = totalnelem / _allocchunksize  but round up. 
      _totalnchunks = (totalnelem + _allocchunksize-1)/_allocchunksize;

      if (_totalnchunks < 2) {
	_totalnchunks=2;
      }
      // Perform memory allocation 
      *_arrayptr = (AT *)_memalloc->malloc(_totalnchunks * _allocchunksize * sizeof(AT));
      /* Single entry in free list, 
	 set freeblocksize, marking _totalnchunk chunks free */
      memcpy(*_arrayptr,&_totalnchunks,sizeof(snde_index));
      /* set nextfreeblockstart, */
      memcpy(*_arrayptr + sizeof(snde_index),&_totalnchunks,sizeof(snde_index));
    
      _locker->set_array_size(_arrayptr,sizeof(AT),_totalnchunks*_allocchunksize);

      // Permanently allocate (and waste) first chunk
      // so that an snde_index==0 is otherwise invalid

      alloc(_allocchunksize);
    
    }

  
    void _realloc(snde_index newnchunks) {
      // Must hold write lock on entire array
      // must hold allocatormutex
      // *** NOTE: *** Caller responsible for adding to
      // last free block or adding new free block !!!
      _totalnchunks = newnchunks;
      *_arrayptr = (AT *)_memalloc->realloc(*_arrayptr,_totalnchunks * _allocchunksize * sizeof(AT));

      _locker->set_array_size(_arrayptr,sizeof(AT),_totalnchunks*_allocchunksize);
      
    }
    
    
  
    snde_index alloc(snde_index nelem) {
      // must hold write lock on entire array
      // Step through free list, looking for a chunk big enough
      snde_index freepos,freeblocksize,nextfreeblockstart;

      snde_index newfreeblockstart;
      snde_index newfreeblocknchunks;

      char *freeposptr;
      char *nextfreeblockstartptr;

      // Number of chunks we need... nelem/_allocchunksize rounding up
      snde_index allocchunks = (nelem+_allocchunksize-1)/_allocchunksize;

      std::unique_lock<std::mutex> lock(allocatormutex);

      for (freeposptr=(char *)&_firstfree,freepos=_firstfree;
	   freepos < _totalnchunks;
	   freeposptr=nextfreeblockstartptr,freepos=nextfreeblockstart) {

	assert(!(freepos > _totalnchunks)); /* Last entry should line up with _totalnchunks */
	if (freepos == _totalnchunks) {
	  /* Overrun of allocated area... need to realloc */
	
	  size_t freeposptroffset=0;
	  snde_index newnchunks=(snde_index)((_totalnchunks+allocchunks)*1.7); /* Reserve extra space, not just bare minimum */
	
	  if (freeposptr != (char *)&_firstfree) {
	    freeposptroffset=freeposptr-((char *)*_arrayptr);
	  }

	  this->_realloc(newnchunks);
	  
	  if (freeposptr != (char *)&_firstfree) {
	    freeposptr=((char *)*_arrayptr)+freeposptroffset;
	  }

	  /*** NOTE: We could be a bit more sophisticated here and 
	       add this free block onto the previous, if the previous
	       ends the entire array ****/

	  /* Define size of newly available free block */	
	  newfreeblockstart = freepos; 
	  newfreeblocknchunks = newnchunks - freepos;
	
	  /* freeposptr should already be pointing here */
	  memcpy(*_arrayptr + newfreeblockstart*_allocchunksize,&newfreeblocknchunks,sizeof(snde_index));
	  /* next block index */
	  memcpy(((char *)(*_arrayptr + newfreeblockstart*_allocchunksize))+sizeof(snde_index),&newnchunks,sizeof(snde_index));
	
	  /* reallocation complete... This should be the final loop iteration */
	}
      
	/* normal start of loop begins here */
      
	memcpy(&freeblocksize,*_arrayptr + freepos*_allocchunksize,sizeof(snde_index));
	nextfreeblockstartptr=((char *)(*_arrayptr + freepos*_allocchunksize)) + sizeof(snde_index);
	memcpy(&nextfreeblockstart,nextfreeblockstartptr,sizeof(snde_index));

	if (freeblocksize >= allocchunks) {
	  /* found block big enough */
	  if (freeblocksize == allocchunks) {
	    /* exactly right size */
	    /* Just have free list skip over this */
	    /* repoint *freeposptr to nextfreeblockstart */
	    memcpy(freeposptr,&nextfreeblockstart,sizeof(snde_index));
	  } else {
	    /* Add new free block after allocated size */

	    newfreeblockstart = freepos + allocchunks;
	    newfreeblocknchunks = freeblocksize-allocchunks;

	    /* repoint *freeposptr to the new block */
	    memcpy(freeposptr,&newfreeblockstart,sizeof(snde_index));

	    /* Write the new free block */
	    /* block size */
	    memcpy(*_arrayptr + newfreeblockstart*_allocchunksize,&newfreeblocknchunks,sizeof(snde_index));
	    /* next block index */
	    memcpy(((char *)(*_arrayptr + newfreeblockstart*_allocchunksize))+sizeof(snde_index),&nextfreeblockstart,sizeof(snde_index));
	  
	  
	  }


	  return freepos*_allocchunksize;
	}
      
      }

      return SNDE_INDEX_INVALID;
    }

    void free(snde_index addr,snde_index nelem) {
      // Number of chunks we need... nelem/_allocchunksize rounding up
      snde_index chunkaddr;
      snde_index freepos,nextfreeblockstart;
      snde_index newfreeblockstart;
      snde_index newfreeblocknchunks;
      snde_index newfreeblocknextstart;
      snde_index priorblockpos=0;

      snde_index freeblocksize;
      
      char *freeposptr,*nextfreeblockstartptr;
      char *priorblocksizeptr,*blocksizeptr;
    
      snde_index freechunks = (nelem+_allocchunksize-1)/_allocchunksize;
      std::lock_guard<std::mutex> lock(allocatormutex); // Lock the allocator mutex 

      assert(addr > 0); /* addr==0 is wasted element allocated by constructor */
      assert(addr % _allocchunksize == 0); /* all addresses returned by alloc() are multiples of _allocchunksize */

      chunkaddr=addr/_allocchunksize;
    
      /* Step through free list until we step over this allocation */
    
      for (freeposptr=(char *)&_firstfree,freepos=_firstfree,priorblocksizeptr=NULL;
	   freepos < _totalnchunks;
	   freeposptr=nextfreeblockstartptr,freepos=nextfreeblockstart,priorblocksizeptr=blocksizeptr) {
      
	/* get info on this free block */
	blocksizeptr=(char *)(*_arrayptr + freepos*_allocchunksize);
	memcpy(&freeblocksize,blocksizeptr,sizeof(snde_index));
	nextfreeblockstartptr=((char *)(*_arrayptr + freepos*_allocchunksize)) + sizeof(snde_index);
	memcpy(&nextfreeblockstart,nextfreeblockstartptr,sizeof(snde_index));

      
	if (chunkaddr < freepos) {
	  /* chunk was inside previous allocated block */
	  assert(chunkaddr + freechunks <= freepos); /* chunks being freed should not overlap free zone */
	  if (chunkaddr + freechunks == freepos) {
	    /* extend this free region earlier to contain
	       newly-freed region */

	    newfreeblockstart=chunkaddr;
	    newfreeblocknchunks=freechunks + freeblocksize;
	    newfreeblocknextstart=nextfreeblockstart;
	  } else {
	    /* Insert new free region */
	    newfreeblockstart=chunkaddr;
	    newfreeblocknchunks=freechunks;
	    newfreeblocknextstart=freepos;
	    
	  }
	
	  /* Write to previous free block where this one starts */
	  memcpy(freeposptr,&newfreeblockstart,sizeof(snde_index));

	  /* Write freeblocksize */
	  memcpy(*_arrayptr + newfreeblockstart*_allocchunksize,&newfreeblocknchunks,sizeof(snde_index));
	
	  /* Write nextfreeblockstart */
	  memcpy(((char *)(*_arrayptr + newfreeblockstart*_allocchunksize))+sizeof(snde_index),&newfreeblocknextstart,sizeof(snde_index));
	  

	  /* Now go back and try to join with the prior free region, 
	     if possible */
	  if (priorblocksizeptr) {
	    snde_index priorblocksize;
	    memcpy(&priorblocksize,priorblocksizeptr,sizeof(snde_index));

	    if (priorblockpos + priorblocksize == newfreeblockstart) {
	      /* Merge free blocks */
	      snde_index newnewfreeblockstart;
	      snde_index newnewfreeblocknchunks;
	      snde_index newnewfreeblocknextstart;

	      newnewfreeblockstart = priorblockpos;
	      newnewfreeblocknchunks = priorblocksize+newfreeblocknchunks;
	      newnewfreeblocknextstart = newfreeblocknextstart;
	    
	      /* Write freeblocksize */
	      memcpy(*_arrayptr + newnewfreeblockstart*_allocchunksize,&newnewfreeblocknchunks,sizeof(snde_index));
	      /* Write nextfreeblockstart */
	      memcpy(((char *)(*_arrayptr + newnewfreeblockstart*_allocchunksize))+sizeof(snde_index),&newnewfreeblocknextstart,sizeof(snde_index));
	    }
	  
	  }
	
	  return;
	}

	priorblockpos=freepos;
      }
      assert(0); /* If this fails, it means the chunk being freed was not inside the array (!) */
    }
  };
    
}

#endif /* SNDE_ALLOCATOR */
