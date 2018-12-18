//%shared_ptr(snde::memallocator);
//%shared_ptr(snde::cmemallocator);

%{
  
#include "rangetracker.hpp"
%}


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
    
    //iterator begin(); 
    //iterator end(); 

    size_t size() const;

    template <typename ... Args>
      void mark_all(snde_index nelem, Args && ... args);
    
    void clear_all();
    
    template <typename ... Args>
      std::pair<iterator,iterator> _breakupregion(iterator breakupregion, snde_index breakpoint,Args && ... args);
    
    std::shared_ptr<T> get_region(snde_index firstelem);

    
    template <typename ... Args>
      iterator _get_starting_region(snde_index firstelem,Args && ... args);    

    template <typename ... Args>
      rangetracker<T> mark_region(snde_index firstelem, snde_index numelems,Args && ... args);    

    rangetracker<T> mark_region_noargs(snde_index firstelem, snde_index numelems);

    template <typename ... Args>
      rangetracker<T> iterate_over_marked_portions(snde_index firstelem, snde_index numelems,Args && ... args);
    template <typename ... Args>
      rangetracker<T> clear_region(snde_index firstelem, snde_index numelems,Args && ... args);
    template <typename ... Args>
      std::shared_ptr<T> find_unmarked_region(snde_index start, snde_index endplusone,snde_index size,Args && ... args);


  };

  template <class T,typename ... Args>
    rangetracker<T> range_union(rangetracker <T> &a, rangetracker<T> &b,Args && ... args);
}

