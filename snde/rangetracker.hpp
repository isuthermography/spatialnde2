#ifndef SNDE_RANGETRACKER_HPP
#define SNDE_RANGETRACKER_HPP


#include <map>
#include <memory>

namespace snde {
  
  // markedregion is not really specific to rangetracker,
  // but is compatible with rangetracker 
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
    std::shared_ptr<markedregion> sp_breakup(snde_index breakpoint)
    /* breakup method ends this region at breakpoint and returns
       a new region starting at from breakpoint to the prior end */
    {
      std::shared_ptr<markedregion> newregion=std::make_shared<markedregion>(breakpoint,regionend);
      regionend=breakpoint;

      return newregion;
    }
    markedregion breakup(snde_index breakpoint)
    /* breakup method ends this region at breakpoint and returns
       a new region starting at from breakpoint to the prior end */
    {
      markedregion newregion(breakpoint,regionend);
      regionend=breakpoint;

      return newregion;
    }

    bool operator<(const markedregion & other) const {
      if (regionstart < other.regionstart) return true;
      return false;

    }
  };


  
  
  template <class T> // class T should have regionstart and regionend elements
  class rangetracker {
  public:
    // std::map is a tree-based ordered lookup map
    // because it is ordered it has the lower_bound method,
    // which returns an iterator to the first element not less
    // than a given key.
    //
    // We use regionstart as the key, but we can look up the first region
    // that starts at or after a given regionstart with the
    // lower_bound() method
    std::map<snde_index,std::shared_ptr<T>> trackedregions; // indexed by regionstart

    /* iteration iterates over trackedregions */
    typedef typename std::map<snde_index,std::shared_ptr<T>>::iterator iterator;
    typedef typename std::map<snde_index,std::shared_ptr<T>>::const_iterator const_iterator;
    /* iterator is a pairL iterator.first = regionstart; iterator.second = shared_ptr to T */
    
    iterator begin() { return trackedregions.begin(); }
    iterator end() { return trackedregions.end(); }

    size_t size() const
    {
      return trackedregions.size();
    }

    template <typename ... Args>
    void mark_all(snde_index nelem, Args && ... args)
    {
      /* Arguments past nelem are passed to region constructor (after regionstart and regionend */
      trackedregions.clear();
      trackedregions[0]=std::make_shared<T>(0,nelem,std::forward<Args>(args) ...); 
    }

    void clear_all()
    {
      trackedregions.clear();
    }

    void erase(iterator e) {
      trackedregions.erase(e);
    }
    
    template <typename ... Args>
    std::pair<iterator,iterator> _breakupregion(iterator breakupregion, snde_index breakpoint,Args && ... args)
      // Breakup the region specified by the iterator and breakpoint...
      // return iterators to both pieces
    {

      // break into two parts
      // breakup method shrinks the existing region into the first of two
      // and returns the second
      std::shared_ptr<T> firstregion = breakupregion->second;
      std::shared_ptr<T> secondregion = breakupregion->second->sp_breakup(breakpoint,std::forward<Args>(args) ...);

      // erase from map
      trackedregions.erase(breakupregion->first);

      /* emplace first part of broken up region */
      trackedregions[firstregion->regionstart]=firstregion;

      /* emplace second part of broken up region */
      trackedregions[secondregion->regionstart]=secondregion;

      // Create and return iterators to each 
      return std::make_pair(trackedregions.lower_bound(firstregion->regionstart),trackedregions.lower_bound(secondregion->regionstart));
    }
    

    std::shared_ptr<T> get_region(snde_index firstelem)
    {
      iterator region=trackedregions.lower_bound(firstelem);
      
      if (region != trackedregions.end()) {
	return region->second;
      }
      
      return nullptr;
    }
    
    template <typename ... Args>
    iterator _get_starting_region(snde_index firstelem,Args && ... args)
    /* identify a preexisting region or split a preexisting region so that the startpoint >= specified firstelem or trackedregions.end()
       
       The returned iterator will never identify a region that starts prior to firstelem. The iterator may be trackedregions.end() which would mean there is no preexisting region that contained firstelem or started after firstelem

       if the returned iterator identifies a region that start after firstelem, that would mean that there is no preexisting region that contains the space between firstelem and the returned iterator
       */
    {
      iterator region,breakupregion,priorregion;

      /* identify first region where startpoint >= specified firstelem */
      region=trackedregions.lower_bound(firstelem);
      
      if ((region != trackedregions.end() && region->first != firstelem) || (region==trackedregions.end())) {

	if (region==trackedregions.begin()) {
	  return trackedregions.end(); // no such region!
	}
	
	if (region != trackedregions.begin()) {
	  // region we want may start partway through an invalidregion 
	  // break up region
	  breakupregion=region;
	  breakupregion--;

	  
	  if (breakupregion->second->regionend > firstelem) {
	    /* starts partway through breakupregion... perform breakup */
	    std::tie(priorregion,region)=_breakupregion(breakupregion,firstelem,std::forward<Args>(args) ...);
	  
	    //region=trackedregions.lower_bound(firstelem);
	    assert(region->first==firstelem);
	    
	    /* attempt to merge first part with anything prior */
 	    
	    /* {
	      if (priorregion != trackedregions.begin()) {
		iterator firstpieceprior=priorregion;
		firstpieceprior--;
		if (firstpieceprior->second->attempt_merge(*priorregion->second)) {
		  assert(firstpieceprior->second->regionend==firstelem); // successful attempt_merge should have merged with successor 
		  trackedregions.erase(priorregion->first);
		}
	      }
	      } */
	    
	  }
	}

      }
      return region;
    }
    
    void merge_adjacent_regions()
    {
      iterator iter=begin();
      iterator next=iter;


      while(1) {
	if (iter==end()) return;
	
	next++;
	
	assert(iter->first==iter->second->regionstart);
	if (iter->second->regionstart==iter->second->regionend) {
	  // empty region ... remove it!
	  trackedregions.erase(iter);
	}
      
	if (next==end()) return;
	assert(next->first==next->second->regionstart);
	
	for (;;) {
	  if (next->first==iter->second->regionend) {
	    if (iter->second->attempt_merge(*next->second)) {
	      trackedregions.erase(next);
	      next=iter;
	      next++;
	      if (next==end()) return;
	    } else {
	      iter++;
	      next++;
	      if (next==end()) return;
	    }
	  } else {
	    iter++;
	    next++;
	    if (next==end()) return;
	    
	  }
	}
      }
    }


    template <typename ... Args>
    std::shared_ptr<T> mark_unmarked_region(snde_index firstelem, snde_index numelems,Args && ... args)
    /* mark specified region, which must be entirely unmarked. Returns pointer to single region object */
    {
      /* identify first region where startpoint >= specified firstelem, make sure it doesn't overlap */
      iterator region=trackedregions.lower_bound(firstelem);
      if (region != trackedregions.end()) {
	assert(region->regionstart > firstelem); /* must not match specified firstelem, because this region must be unmarked */
	assert(region->regionstart >= firstelem+numelems); /* found region must not overlap with region we are to mark */	
      }


      /* we need
	 to add a region */

      
      snde_index regionend;
      
      if (numelems==SNDE_INDEX_INVALID) {
	regionend=SNDE_INDEX_INVALID; /* rest of array */
      } else {
	regionend=firstelem+numelems;
      }
      
      if (region != trackedregions.end() && regionend > region->first) {
	regionend=region->first;
      }
      
      trackedregions[firstelem]=std::make_shared<T>(firstelem,regionend,std::forward<Args>(args) ...);
      
      region=trackedregions.lower_bound(firstelem);

      return *region; 

    }

    template <typename ... Args>
    rangetracker<T> mark_region(snde_index firstelem, snde_index numelems,Args && ... args)
    /* mark specified region, some portion(s) of which may already be marked. returns iterable rangetracker with blocks representing
       the desired region */
    {
      iterator region;
      rangetracker<T> retval;
      bool first_region_new=false;
      
      if (!numelems) {
	return retval;
      }
      
      region=_get_starting_region(firstelem,std::forward<Args>(args) ...);

      /* region should now be a region where startpoint >= specified firstelem
	 or trackedregions.end()
       */
      
      if (region==trackedregions.end() || region->first != firstelem) {
	/* in this case we didn't break up a region, but we need
	   to add a region */

	
	snde_index regionend;

	if (numelems==SNDE_INDEX_INVALID) {
	  regionend=SNDE_INDEX_INVALID; /* rest of array */
	} else {
	  regionend=firstelem+numelems;
	}

	if (region != trackedregions.end() && regionend > region->first) {
	  regionend=region->first;
	}
	
	trackedregions[firstelem]=std::make_shared<T>(firstelem,regionend,std::forward<Args>(args) ...);

	region=trackedregions.lower_bound(firstelem);
	first_region_new = true; 
      }

      /* now region refers to firstelem */

      snde_index coveredthrough=firstelem;

      snde_index regionend;
      
      if (numelems==SNDE_INDEX_INVALID) {
	regionend=SNDE_INDEX_INVALID; /* rest of array */
      } else {
	regionend=firstelem+numelems;
      }


      bool this_region_new = first_region_new;

      while (coveredthrough < regionend) {



	if (region == trackedregions.end() || coveredthrough < region->second->regionstart) {
	  /* We have a gap. Don't use this region but
	     instead emplace a prior region starting where we are
	     covered through */
	  snde_index newregionend=regionend;
	  if (region != trackedregions.end() && newregionend > region->second->regionstart) {
	    regionend=region->second->regionstart;
	  }
	  
	  
	  trackedregions[coveredthrough]=std::make_shared<T>(coveredthrough,newregionend,std::forward<Args>(args) ...);

	  region=trackedregions.lower_bound(coveredthrough);
	  this_region_new = true;
	}

	/* now we've got a region that starts at coveredthrough */
	assert(region->second->regionstart==coveredthrough);
	
	if (region->second->regionend > regionend) {
	  /* this region goes beyond our ROI...  */
	  /* break it up */
	  iterator secondpieceiterator;
	  
	  std::tie(region,secondpieceiterator)=_breakupregion(region,regionend,std::forward<Args>(args) ...);
	  this_region_new = false;
	  
	  assert(region->second->regionend == regionend);

	  /* attempt to merge second part of broken up region 
	     with following */
	  {
	    iterator secondpiecenext=secondpieceiterator;
	    secondpiecenext++;
	    if (secondpiecenext != trackedregions.end() && secondpiecenext->second->regionstart==secondpieceiterator->second->regionend) {
	      if (secondpieceiterator->second->attempt_merge(*secondpiecenext->second)) {
		/* if merge succeeded, remove second piecenext */
		trackedregions.erase(secondpiecenext->first);
	      }
	    }
	    
	  }
	  

	  
	  
	}
	
	/* now we've got a region that starts at coveredthrough and 
	   ends at or before firstelem+numelems */
	assert (region->second->regionend <= regionend);
	
	if (!this_region_new) { 
	  // if we're re-using an old region,
	  // replace it because we may need to pass on
	  // constructor info, etc.
	  
	  region->second=std::make_shared<T>(region->second->regionstart,region->second->regionend,std::forward<Args>(args) ...);
	}

	
	/* add region to retval */
	retval.trackedregions[region->second->regionstart]=region->second;

	/* increment coveredthrough */
	coveredthrough=region->second->regionend;

	region++;  // move to next region 
	this_region_new = false; // next region presumably not new
      }

      return retval;
    }


    rangetracker<T> mark_region_noargs(snde_index firstelem, snde_index numelems)
    {
      return mark_region(firstelem,numelems);
    }

    template <typename ... Args>
    rangetracker<T> iterate_over_marked_portions(snde_index firstelem, snde_index numelems,Args && ... args)
    // formerly get_regions()
    /* returns iterable rangetracker with blocks representing
       all currently marked segments of the desired region.

       Note that this DOES change the rangetracker, adding region boundaries 
       at firstelem and firstelem+numelems if not already present and within marked zones
       
       Currently marked segments that overlap the desired region
       will be split at the region boundary and only the 
       inside component will be returned. 
    */
    {
      iterator region;
      rangetracker retval;

      if (!numelems) {
	return retval;
      }

      region=_get_starting_region(firstelem,std::forward<Args>(args) ...);
      /* region should now be a region where startpoint >= specified firstelem
	 or trackedregions.end()
       */
      

      
      while (region != trackedregions.end() && region->second->regionstart < firstelem+numelems) {

	
	if (region->second->regionend > firstelem+numelems) {
	  /* this region goes beyond our ROI...  */
	  /* break it up */
	  iterator secondpieceiterator;


	  std::tie(region,secondpieceiterator)=_breakupregion(region,firstelem+numelems,std::forward<Args>(args) ...);

	  assert(region->second->regionend == firstelem+numelems);

	  /* attempt to merge second part of broken up region 
	     with following */
	  {
	    iterator secondpiecenext=secondpieceiterator;
	    secondpiecenext++;
	    if (secondpiecenext != trackedregions.end() && secondpiecenext->first == secondpieceiterator->second->regionend) { // if there is a next region and it is contiguous
	      if (secondpieceiterator->second->attempt_merge(*secondpiecenext->second)) {
		/* if merge succeeded, remove second piecenext */
		trackedregions.erase(secondpiecenext->first);
	      }
	    }
	    
	  }
	  

	  
	  
	}
	
	/* now we've got a region that ends at or before firstelem+numelems */
	assert (region->second->regionend <= firstelem+numelems);

	/* add region to retval */
	retval.trackedregions[region->second->regionstart]=region->second;

	/* increment region */
	region++;
      }

      return retval;
      
    }

    template <typename ... Args>
    rangetracker<T> clear_region(snde_index firstelem, snde_index numelems,Args && ... args)
    /* returns iterable rangetracker with blocks representing
       any removed marked segments of the desired region */
    {
      rangetracker<T> marked_regions=iterate_over_marked_portions(firstelem,numelems,std::forward<Args>(args) ...);

      for (auto & region: marked_regions) {
	trackedregions.erase(region.first);
      }

      return marked_regions;
    }

    template <typename ... Args>
    std::shared_ptr<T> find_unmarked_region(snde_index start, snde_index endplusone,snde_index size,Args && ... args)
    /* returns first unmarked region of specified size between start and endplusone */
    {
      snde_index marked_size,unmarked_size;
      snde_index next_marked_size,next_unmarked_size;
      snde_index pos,nextpos;

      iterator region;

      region=trackedregions.lower_bound(start);
      if (region != trackedregions.end() && region->first >= start)  {
	pos=start;
	unmarked_size=region->first-start;
	marked_size=region->second->regionend-region->second->regionstart;
	if (marked_size+unmarked_size+pos > endplusone) {
	  if (endplusone > pos+unmarked_size) {
	    marked_size=endplusone-unmarked_size-pos;
	  } else {
	    marked_size=0;
	    unmarked_size=endplusone-pos;
	  }
	}
      } else if (region==trackedregions.end()) {
	pos=start;
	unmarked_size=endplusone-start;
	marked_size=0;
      } else {
	assert(0); /* should never happen */
      }
      
      for (;pos < endplusone;pos=nextpos,unmarked_size=next_unmarked_size,marked_size=next_marked_size) {
	/* in general we iterate over blocks that consist of an 
	   unmarked region, followed by a marked region */
	if (unmarked_size >= size) {
	  return std::make_shared<T>(pos,pos+size,std::forward<Args>(args) ...); 
	}

	nextpos=pos+unmarked_size+marked_size;

	if (region!=trackedregions.end()) {
	  region++;
	}
	if (region!=trackedregions.end()) {
	  next_unmarked_size=region->first-nextpos;
	  
	  assert(region->first==region->second->regionstart);
	  next_marked_size=region->second->regionend-region->second->regionstart;

	  if (next_marked_size+next_unmarked_size+nextpos > endplusone) {
	    if (endplusone > nextpos+next_unmarked_size) {
	      next_marked_size=endplusone-next_unmarked_size-nextpos;
	    } else {
	      next_marked_size=0;
	      next_unmarked_size=endplusone-nextpos;
	    }
	  }
	  	  
	} else {
	  /* region==trackedregions.end */
	  next_unmarked_size=endplusone-nextpos;
	  next_marked_size=0;
	}
	
      }
      return nullptr;
    }

  };

  template <class T>
  bool region_overlaps(T & cmpregion,snde_index regionstart,snde_index regionend) {
    // to not overlap, one has to end before the other starts
    // (equality OK because end marker is one past physical end)
    
    if (cmpregion.regionend <= regionstart) return false;
    if (regionend <= cmpregion.regionstart) return false;
    return true;
  }
  
  template <class T,typename ... Args>
  rangetracker<T> range_union(rangetracker <T> &a, rangetracker<T> &b,Args && ... args)
  {
    rangetracker<T> output;

    for (auto & a_region: a) {
      snde_index numelems;

      if (a_region.second->regionend == SNDE_INDEX_INVALID) {
	numelems=SNDE_INDEX_INVALID;
      } else {
	numelems=a_region.second->regionend-a_region.second->regionstart;
      }
      output.mark_region(a_region.second->regionstart,numelems,std::forward<Args>(args) ...);
    }

    for (auto & b_region: b) {
      snde_index numelems;

      if (b_region.second->regionend == SNDE_INDEX_INVALID) {
	numelems=SNDE_INDEX_INVALID;
      } else {
	numelems=b_region.second->regionend-b_region.second->regionstart;
      }
      output.mark_region(b_region.second->regionstart,numelems,std::forward<Args>(args) ...);
    }
    

    return output;
  }
}

#endif /* SNDE_RANGETRACKER_HPP */
