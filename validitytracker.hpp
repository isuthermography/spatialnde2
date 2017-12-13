#ifndef SNDE_VALIDITYTRACKER_HPP
#define SNDE_VALIDITYTRACKER_HPP


#include <map>

namespace snde {
  

  
  template <class T> // class T should have regionstart and regionend elements
  class validitytracker {
  public:
    std::map<snde_index,std::shared_ptr<T>> trackedregions; // indexed by regionstart

    /* iteration iterates over trackedregions */
    typedef typename std::map<snde_index,std::shared_ptr<T>>::iterator iterator;
    typedef typename std::map<snde_index,std::shared_ptr<T>>::const_iterator const_iterator;
    /* iterator is a pairL iterator.first = regionstart; iterator.second = shared_ptr to T */
    
    iterator begin() { return trackedregions.begin(); }
    iterator end() { return trackedregions.end(); }

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
      std::pair<std::shared_ptr<T>,std::shared_ptr<T>> regionparts = breakupregion->second->breakup(firstelem,std::forward<Args>(args) ...);

      // erase from map
      trackedregions.erase(breakupregion.first);

      /* emplace first part of broken up region */
      trackedregions[regionparts[0]->regionstart]=std::get<0>(regionparts);

      /* emplace second part of broken up region */
      trackedregions[regionparts[1]->regionstart]=regionparts[1];

      // Create and return iterators to each 
      return std::make_pair(trackedregions.lower_bound(regionparts[0]),trackedregions.lower_bound(regionparts[1]));
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
		if (firstpieceprior->second.attempt_merge(priorregion->second)) {
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
    validitytracker<T> mark_region(snde_index firstelem, snde_index numelems,Args && ... args)
    /* mark specified region; returns iterable validitytracker with blocks representing
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
	snde_index regionend=firstelem+numelems;

	if (region != trackedregions.end() && regionend > region->first) {
	  regionend=region->first;
	}
	
	trackedregions[firstelem]=std::make_shared<T>(firstelem,regionend,std::forward<Args>(args) ...);

	region=trackedregions.lower_bound(firstelem);

      }

      /* now region refers to firstelem */

      validitytracker retval;
      snde_index coveredthrough=firstelem;
      
      while (coveredthrough < firstelem+numelems) {

	if (region == trackedregions.end() || coveredthrough < region->second->regionstart) {
	  /* We have a gap. Don't use this region but
	     instead emplace a prior region starting where we are
	     covered through */
	  snde_index regionend=firstelem+numelems;
	  if (region != trackedregions.end() && regionend > region->second->regionstart) {
	    regionend=region->second->regionstart;
	  }
	  
	  
	  trackedregions[coveredthrough]=std::make_shared<T>(coveredthrough,regionend,std::forward<Args>(args) ...);

	  region=trackedregions.lower_bound(coveredthrough);
	  
	}

	/* now we've got a region that starts at coveredthrough */
	assert(region->second->regionstart==coveredthrough);
	
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
	      if (secondpieceiterator->second->attempt_merge(secondpiecenext->second)) {
		/* if merge succeeded, remove second piecenext */
		trackedregions.erase(secondpiecenext->first);
	      }
	    }
	    
	  }
	  

	  
	  
	}
	
	/* now we've got a region that starts at coveredthrough and 
	   ends at or before firstelem+numelems */
	assert (region->second->regionend <= firstelem+numelems);

	/* add region to retval */
	retval[region->second->regionstart]=region->second;

	/* increment coveredthrough */
	coveredthrough=region->second->regionend;
      }

      return retval;
    }


    

    template <typename ... Args>
    validitytracker<T> get_regions(snde_index firstelem, snde_index numelems,Args && ... args)
    /* returns iterable validitytracker with blocks representing
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
      

      validitytracker retval;
      
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
	      if (secondpieceiterator->second->attempt_merge(secondpiecenext->second)) {
		/* if merge succeeded, remove second piecenext */
		trackedregions.erase(secondpiecenext->first);
	      }
	    }
	    
	  }
	  

	  
	  
	}
	
	/* now we've got a region that ends at or before firstelem+numelems */
	assert (region->second->regionend <= firstelem+numelems);

	/* add region to retval */
	retval[region->second->regionstart]=region->second;

	/* increment region */
	region++;
      }

      return retval;
      
    }

    template <typename ... Args>
    validitytracker<T> clear_region(snde_index firstelem, snde_index numelems,Args && ... args)
    /* returns iterable validitytracker with blocks representing
       any removed marked segments of the desired region */
    {
      validitytracker<T> marked_regions=get_regions(firstelem,numelems,std::forward<Args>(args) ...);

      for (auto & region: marked_regions) {
	trackedregions.erase(region.first);
      }

      return marked_regions;
    }


  };

  
}

#endif /* SNDE_VALIDITYTRACKER_HPP */
