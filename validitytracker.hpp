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
    validitytracker<T> mark_region(snde_index firstelem, snde_index numelems,Args && ... args)
    /* returns iterable validitytracker with blocks representing
       the desired region */
    {
      iterator region,breakupregion;
      
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
	    std::pair<std::shared_ptr<T>,std::shared_ptr<T>> regionparts = breakupregion->second->breakup(firstelem,std::forward<Args>(args) ...);
	    trackedregions.erase(breakupregion.first);

	    /* emplace first part of broken up region */
	    trackedregions[regionparts[0]->regionstart]=regionparts[0];

	    /* attempt to merge first part with anything prior */
	    {
	      iterator firstpieceiterator=trackedregions.lower_bound(regionparts[0]->regionstart);
	      if (firstpieceiterator != trackedregions.begin()) {
		iterator firstpieceprior=firstpieceiterator;
		firstpieceprior--;
		if (firstpieceprior->second.attempt_merge(firstpieceiterator->second)) {
		  assert(firstpieceprior->second->regionend==firstelem); /* successful attempt_merge should have merged with successor */
		  trackedregions.erase(firstpieceiterator->first);
		}
	      }
	    }
	    
	    /* emplace second part of broken up region */
	    trackedregions[regionparts[1]->regionstart]=regionparts[1];
	    region=trackedregions.lower_bound(firstelem);
	    assert(region->first==firstelem);
	  }
	}

      }
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

	  std::pair<std::shared_ptr<T>,std::shared_ptr<T>> regionparts = region->second->breakup(firstelem+numelems);
	  
	  trackedregions.erase(region.first);
	  /* emplace first part of broken up region */
	  trackedregions[regionparts[0]->regionstart]=regionparts[0];
	  /* emplace second part of broken up region */
	  trackedregions[regionparts[1]->regionstart]=regionparts[1];

	  /* attempt to merge second part of broken up region 
	     with following */
	  {
	    iterator secondpieceiterator=trackedregions.lower_bound(regionparts[1]->regionstart);
	    iterator secondpiecenext=secondpieceiterator;
	    secondpiecenext++;
	    if (secondpiecenext != trackedregions.end()) {
	      if (secondpieceiterator->second->attempt_merge(secondpiecenext->second)) {
		/* if merge succeeded, remove second piecenext */
		trackedregions.erase(secondpiecenext->first);
	      }
	    }
	    
	  }
	  
	  /* get interator to first part of broken up region */
	  region=trackedregions.lower_bound(regionparts[0]->regionstart);
	  assert(region->second->regionend == firstelem+numelems);

	  
	  
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

    /* ***!!! Need to implement clear_region() !!!*** */

    template <typename ... Args>
    validitytracker<T> get_regions(snde_index firstelem, snde_index numelems,Args && ... args)
    /* returns iterable validitytracker with blocks representing
       all marked segments of the desired region */
    {
      /* !!! *** */
    }

    template <typename ... Args>
    validitytracker<T> clear_region(snde_index firstelem, snde_index numelems,Args && ... args)
    /* returns iterable validitytracker with blocks representing
       any removed marked segments of the desired region */
    {
      /* !!! *** */
    }


  };

  
}

#endif /* SNDE_VALIDITYTRACKER_HPP */
