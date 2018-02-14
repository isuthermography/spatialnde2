#ifndef SNDE_RANGETRACKER_HPP
#define SNDE_RANGETRACKER_HPP


#include <map>

namespace snde {
  

  
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

    template <typename ... Args>
    std::pair<iterator,iterator> _breakupregion(iterator breakupregion, snde_index breakpoint,Args && ... args)
      // Breakup the region specified by the iterator and breakpoint...
      // return iterators to both pieces
    {

      // break into two parts
      // breakup method shrinks the existing region into the first of two
      // and returns the second
      std::shared_ptr<T> firstregion = breakupregion->second;
      std::shared_ptr<T> secondregion = breakupregion->second->breakup(breakpoint,std::forward<Args>(args) ...);

      // erase from map
      trackedregions.erase(breakupregion->first);

      /* emplace first part of broken up region */
      trackedregions[firstregion->regionstart]=firstregion;

      /* emplace second part of broken up region */
      trackedregions[secondregion->regionstart]=secondregion;

      // Create and return iterators to each 
      return std::make_pair(trackedregions.lower_bound(firstregion->regionstart),trackedregions.lower_bound(secondregion->regionstart));
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
      
      if (region != trackedregions.end() && region->first != firstelem) {
	
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
	    
	    {
	      if (priorregion != trackedregions.begin()) {
		iterator firstpieceprior=priorregion;
		firstpieceprior--;
		if (firstpieceprior->second->attempt_merge(*priorregion->second)) {
		  assert(firstpieceprior->second->regionend==firstelem); /* successful attempt_merge should have merged with successor */
		  trackedregions.erase(priorregion->first);
		}
	      }
	    }
	    
	  }
	}

      }
      return region;
    }
    

    template <typename ... Args>
    rangetracker<T> mark_region(snde_index firstelem, snde_index numelems,Args && ... args)
    /* mark specified region; returns iterable rangetracker with blocks representing
       the desired region */
    {
      iterator region;

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

      }

      /* now region refers to firstelem */

      rangetracker<T> retval;
      snde_index coveredthrough=firstelem;

      snde_index regionend;
      
      if (numelems==SNDE_INDEX_INVALID) {
	regionend=SNDE_INDEX_INVALID; /* rest of array */
      } else {
	regionend=firstelem+numelems;
      }
      
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
	  
	}

	/* now we've got a region that starts at coveredthrough */
	assert(region->second->regionstart==coveredthrough);
	
	if (region->second->regionend > regionend) {
	  /* this region goes beyond our ROI...  */
	  /* break it up */
	  iterator secondpieceiterator;
	  
	  std::tie(region,secondpieceiterator)=_breakupregion(region,regionend,std::forward<Args>(args) ...);

	  assert(region->second->regionend == regionend);

	  /* attempt to merge second part of broken up region 
	     with following */
	  {
	    iterator secondpiecenext=secondpieceiterator;
	    secondpiecenext++;
	    if (secondpiecenext != trackedregions.end()) {
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

	/* add region to retval */
	retval.trackedregions[region->second->regionstart]=region->second;

	/* increment coveredthrough */
	coveredthrough=region->second->regionend;
      }

      return retval;
    }


    rangetracker<T> mark_region_noargs(snde_index firstelem, snde_index numelems)
    {
      return mark_region(firstelem,numelems);
    }

    template <typename ... Args>
    rangetracker<T> get_regions(snde_index firstelem, snde_index numelems,Args && ... args)
    /* returns iterable rangetracker with blocks representing
       all currently marked segments of the desired region.
       
       Currently marked segments that overlap the desired region
       will be split at the region boundary and only the 
       inside component will be returned. 
    */
    {
      iterator region;

      region=_get_starting_region(firstelem,std::forward<Args>(args) ...);
      /* region should now be a region where startpoint >= specified firstelem
	 or trackedregions.end()
       */
      

      rangetracker retval;
      
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
	    if (secondpiecenext != trackedregions.end()) {
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
      rangetracker<T> marked_regions=get_regions(firstelem,numelems,std::forward<Args>(args) ...);

      for (auto & region: marked_regions) {
	trackedregions.erase(region.first);
      }

      return marked_regions;
    }


  };

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
