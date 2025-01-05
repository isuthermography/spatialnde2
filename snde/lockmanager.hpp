#ifndef SNDE_LOCKMANAGER_HPP
#define SNDE_LOCKMANAGER_HPP

#include <cstdint>
#include <cassert>

#include <condition_variable>
#include <functional>
#include <deque>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <thread> // Remember to add -pthread to the flags

#include "snde/snde_error.hpp"
#include "snde/geometry_types.h"
#include "snde/lock_types.hpp"
#include "snde/rangetracker.hpp"

/* general locking api information:
 * OLD High level locking order for mutablerecstore and transactional revision manager (TRM):  
 1 TRM transaction_update_lock -- not managed by lockmanager

Used to have object_trees lock. Problem is that with objects both part of a tree and in the recdb
this has to include recdb entries. 
Solution: atomic lists that can be traversed, sort the results, lock, then verify and retry if needed

 2 Recordings/Geometric component tree C++ structures (ordered by shared_ptr std::owner_less)
    ... AND PARAMETERIZATIONS AND UV's
    * Parameterizations are named by the metadata entry "uv_parameterization"
    * UV projection data are named by the comma-separated metadata entry uv_parameterization_channel
    * Challenge: If you add or modify a recording/geom. component, must provide locks and references to
      everything that component references in the tree, directly or indirectly, and pass those
      to the recdb when you update it (so the dirtynotifies can be safely processed). 
      **** THESE ARE LOCKED BY A PROCESS WHEREBY WE EXPLORE THE GRAPH USING ATOMIC ACCESSES (_explore_component) 
           (Must be prior to any lockprocess spawn()s as the unlocking process is incompatible with spawn() )
	   Then create an ordered list and lock, then re-explore, and retry locking if the ordered list has changed.
 3 Data arrays
   * Ordered by position in geometry data structure relative to others in that structure
   *** THESE ARE LOCKED Following the Locking order via spawn() if necessary 
 4 TRM dependency_table_lock -- not managed by lockmanager

 

 *** NEW, UPDATED LOCKING ORDER for mostly-immutable recording database: 
 (rationale for transaction lock preceding dataguzzler-python module lock 
 was that dataguzzler-python module might want to trigger other acquisitions
 within the transaction. Also we might want to do automatic stuff that might access other dataguzzler-python
 modules at EndTransaction(). Similar to addquerymetadatum. 
 * Pitfall: From a Python module, to start transaction, drop the module lock, 
   start the transaction, reacquire the module lock. Could get locked out briefly
   at the most critical juncture for real time recording. 
 * Advantage of this approach is that addquerymetadatum could be implemented at the 
   spatialnde2 level in EndTransaction rather than in each acquisition module. 
   * Also would work for math functions. 
 * Is is possible to avoid the risk of deadlock without dropping the lock on transaction start?
     ... MAYBE. Probably need to define a protocol for the graph of subcalls from an
addquerymetadatum that runs in end_transaction or metadatadone.     
  
 1. Entry into a transaction (StartTransaction()/EndTransaction() or equiv)
 1.5 dataguzzler-python module locks; 
 1.8 openscenegraph_renderer/osg_compositor execution_lock. 
 2. Any locks required to traverse the mostly immutable recdb (hopefully few/none) (consisting of 2.4-2.7, below)
   * StartTransaction() defines a new global revision for the calling thread
     to mess with. Other threads will still get the prior revision and the 
     structures should generally be immutable so little/no locking required.
   * EndTransaction() makes the new global revision current
 2.4 The recdatabase admin lock 
 2.5 Any single recdatabase channel admin lock, or the available_compute_resource_database admin lock, or any single globalrevision admin lock
 2.6 The recdb current_transaction's admin lock. 
 2.7 Any recording admin lock. 
 3. Data array locks such as mutable data arrays. This includes transient locking to do allocations
    for new mutable OR IMMUTABLE sub-arrays.
    Ordering is by geometry structure or data array structure address, 
    and within Ordering is by location within the geometry structure
 4. Recording storage (recstore_storage) locks
 5. cache manager locks
 6. Allocator locks.
 7. Internal locks of other subsystems, etc.
 8. Lockmanager internal locks 
 9. Any "last" locks such as the Python GIL

Note that the above means that any code called from Python that is doing
locking needs to DROP THE GIL FIRST, acquire locks and then re-acquire the
GIL. With the current SWIG wrappers this is ensured by the -thread option to SWIG


 * Data array locking can be done at various levels of granularity:  All arrays
(get_locks_..._all()), multiple arrays (get_locks_..._arrays()), individual arrays (get_locks_..._array()),
individual regions of individual arrays (get_locks_..._array_region),
or multiple regions of multiple arrays (get_locks_..._arrays_region).
(region-granularity is currently present in the API but an attempt to lock a region actually locks the entire array)
 * Can obtain either a read lock or a write lock. Multiple readers are
allowed simultaneously, but only one writer is possible at a time.
i.e. use get_locks_read_...() vs. get_locks_write_...()
 * Locks must be acquired in a specified order to avoid deadlock. See
https://stackoverflow.com/questions/1951275/would-you-explain-lock-ordering
The order is from the top of the struct snde_geometrydata  to the bottom. Once you own a lock for a particular array, you may not lock an array farther up.
 * Within an array, locks are ordered from smallest to largest index.
 * Allocating space in an array requires a write lock on the entire array, as it may cause the array to be reallocated if it must be expanded.
 * Ownership of a lock is denoted by an rwlock_token, but
   as a single locking operation may return multiple locks,
   the locking operations return an rwlock_token_set
 * When obtaining additional locks after you already own one,
   you must pass the preexisting locks to the locking function,
   as the "prior" or "priors" argument. The preexisting locks are unaffected
   (but it is OK to relock them -- the locks will nest)
 * the rwlock_token and rwlock_token_set are NOT thread safe.
   If you want to pass locks acquired in one thread to another
   thread you can create an rwlock_token_set from your locking
   operation, and then call lockmanager->clone_rwlock_token_set() to
   create an independent clone of the rwlock_token_set that can
   then be safely used by another thread.

 * Currently locks are only implemented to array granularity.
 * In the current implementation attempting to simulaneously read lock one part of an array
   and write lock another part of the same array may deadlock.
 * No useful API to do region-by-region locks of the arrays exists
   so far. Such an API would allow identifying all of the sub-regions
   of all arrays that correspond to the parts (objects) of interest.
   Problem is, the parts need to be locked in order and this ordered
   locking must proceed in parallel for all objects. This would be
   a mess. Suggested approach: Use "resumable functions" once they
   make it into the C++ standard. There are also various
   workarounds to implement closures/resumable functions,
   but they tend to be notationally messy,
   https://github.com/vmilea/CppAsync or require Boost++ (boost.fiber)

 * If we wanted to implement region-granular locking we could probably use the
   rangetracker class to identify and track locked regions.


 */

namespace snde {
  class lockholder_index; // forward declaration
  
  //typedef  std::unordered_map<void *,lockindex_t,std::hash<void *>,std::equal_to<void *>,std::allocator< std::pair< void *const,lockindex_t > > > voidp_size_map;
  //typedef voidp_size_map::iterator voidp_size_map_iterator;
  
  class arraymanager; // forward declaration
  //class mutableinfostore; // forward declaration
  class mutablerecdb;
  class geometry;
#ifdef SNDE_MUTABLE_RECDB_SUPPORT
  class lockable_infostore_or_component; // forward declaration
#endif //SNDE_MUTABLE_RECDB_SUPPORT
  //class component; // forward declaration
  //class parameterization; // forward declaration
  class lockingposition;
  class ndarray_recording_ref; // forward declaration, recstore.hpp
  class multi_ndarray_recording;


/* *** Must keep sync'd with lockmanager.i */

  /* *** Lock masks for obtain_lock() calls on mutableinfostore,
     part/assembly/component, and parameterization *** */
  
typedef uint64_t snde_infostore_lock_mask_t;
#define SNDE_INFOSTORE_INFOSTORES (1ull<<0) // the snde::mutableinfostore and metadata ... used solely with get_locks_lockable_mask(...)
  //#define SNDE_INFOSTORE_OBJECT_TREES (1ull<<1)
#define SNDE_INFOSTORE_COMPONENTS (1ull<<1) // the snde::components, i.e. parts and assemblies... used solely with get_locks_lockable_mask(...) 
#define SNDE_INFOSTORE_PARAMETERIZATIONS (1ull<<2) // the snde::parameterizations of the components... used solely with get_locks_lockable_mask(...) 

#define SNDE_INFOSTORE_ALL ((1ull<<3)-(1ull<<0))
  
// 
#define SNDE_COMPONENT_GEOM_PARTS (1ull<<8)
#define SNDE_COMPONENT_GEOM_TOPOS (1ull<<9)
#define SNDE_COMPONENT_GEOM_TOPO_INDICES (1ull<<10)
#define SNDE_COMPONENT_GEOM_TRIS (1ull<<11)
#define SNDE_COMPONENT_GEOM_REFPOINTS (1ull<<12)
#define SNDE_COMPONENT_GEOM_MAXRADIUS (1ull<<13)
#define SNDE_COMPONENT_GEOM_VERTNORMALS (1ull<<14)
#define SNDE_COMPONENT_GEOM_TRINORMALS (1ull<<15)
#define SNDE_COMPONENT_GEOM_INPLANEMATS (1ull<<16)
#define SNDE_COMPONENT_GEOM_EDGES (1ull<<17)
#define SNDE_COMPONENT_GEOM_VERTICES (1ull<<18)
#define SNDE_COMPONENT_GEOM_PRINCIPAL_CURVATURES (1ull<<19)
#define SNDE_COMPONENT_GEOM_CURVATURE_TANGENT_AXES (1ull<<20)
#define SNDE_COMPONENT_GEOM_VERTEX_EDGELIST_INDICES (1ull<<21)
#define SNDE_COMPONENT_GEOM_VERTEX_EDGELIST (1ull<<22)
#define SNDE_COMPONENT_GEOM_BOXES (1ull<<23)
#define SNDE_COMPONENT_GEOM_BOXCOORD (1ull<<24)
#define SNDE_COMPONENT_GEOM_BOXPOLYS (1ull<<25)

#define SNDE_COMPONENT_GEOM_ALL ((1ull<<26)-(1ull<<8))

// Resizing masks -- mark those arrays that resize together
//#define SNDE_COMPONENT_GEOM_COMPONENT_RESIZE (SNDE_COMPONENT_GEOM_COMPONENT)
#define SNDE_COMPONENT_GEOM_PARTS_RESIZE (SNDE_COMPONENT_GEOM_PARTS)
#define SNDE_COMPONENT_GEOM_TOPOS_RESIZE (SNDE_COMPONENT_GEOM_TOPOS)
#define SNDE_COMPONENT_GEOM_TOPO_INDICES_RESIZE (SNDE_COMPONENT_GEOM_TOPO_INDICES)
#define SNDE_COMPONENT_GEOM_TRIS_RESIZE (SNDE_COMPONENT_GEOM_TRIS|SNDE_COMPONENT_GEOM_REFPOINTS|SNDE_COMPONENT_GEOM_MAXRADIUS|SNDE_COMPONENT_GEOM_VERTNORMALS|SNDE_COMPONENT_GEOM_TRINORMALS|SNDE_COMPONENT_GEOM_INPLANEMATS)
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
#define SNDE_UV_GEOM_UV_PATCHES_RESIZE (SNDE_UV_GEOM_UV_PATCHES)
#define SNDE_UV_GEOM_UV_TOPOS_RESIZE (SNDE_UV_GEOM_UV_TOPOS)
#define SNDE_UV_GEOM_UV_TOPO_INDICES_RESIZE (SNDE_UV_GEOM_UV_TOPO_INDICES)
#define SNDE_UV_GEOM_UV_TRIANGLES_RESIZE (SNDE_UV_GEOM_UV_TRIANGLES|SNDE_UV_GEOM_INPLANE2UVCOORDS|SNDE_UV_GEOM_UVCOORDS2INPLANE)
#define SNDE_UV_GEOM_UV_EDGES_RESIZE (SNDE_UV_GEOM_UV_EDGES)
#define SNDE_UV_GEOM_UV_VERTICES_RESIZE (SNDE_UV_GEOM_UV_VERTICES|SNDE_UV_GEOM_UV_VERTEX_EDGELIST_INDICES)
#define SNDE_UV_GEOM_UV_VERTEX_EDGELIST_RESIZE (SNDE_UV_GEOM_UV_VERTEX_EDGELIST)
#define SNDE_UV_GEOM_UV_BOXES_RESIZE (SNDE_UV_GEOM_UV_BOXES|SNDE_UV_GEOM_UV_BOXCOORD)
#define SNDE_UV_GEOM_UV_BOXPOLYS_RESIZE (SNDE_UV_GEOM_UV_BOXPOLYS)
  
  


  
  
  struct arrayregion {
    void **array;
    snde_index indexstart;
    snde_index numelems;
    
  };


  //  class dirtyregion: public markedregion  {
  //  public:
  //    cachemanager *cache_with_valid_data; /* Do not dereference this pointer... NULL means the main CPU store is the one with the valid data */
  //    cl_event FlushDoneEvent;
  //    bool FlushDoneEventComplete;
  //    dirtyregion(cachemanager *cache_with_valid_data,snde_index regionstart, snde_index regionend) : markedregion(regionstart,regionend)
  //    {
  //      this->cache_with_valid_data=cache_with_valid_data;
  //    }
  //  };

  class datalock {
  public: 
    std::mutex admin; /* locks access to subregions field of arraylock subclass, lock after everything; see also whole_array_write */
    std::shared_ptr<rwlock> whole; /* This is in the locking order with the arrays and mutableinfostores. In
				      order to modify subregions you must hold this AND all subregions AND admin (above) 
				      for write... Note: Not used for dirty tracking (put dirty stuff in subregions!) */
    
    datalock() {
      whole=std::make_shared<rwlock>();
      
    }

    virtual ~datalock() {}
  };
    
    
    
  class arraylock: public datalock {
  public:

    //rwlock full_array;
    //std::vector<rwlock> subregions;

    std::map<markedregion,std::shared_ptr<rwlock>> subregions; // locked by admin mutex of datalock superclass
    
    arraylock() : datalock() {
      
    }

    virtual ~arraylock() {}
  };

  class lockmanager {
    /* Manage all of the locks/mutexes/etc. for a class
       which contains a bunch of arrays handled by
       snde::allocator */


    /* Need to add capability for long-term persistent weak read locks
       that can be notified that they need to go away (e.g. to support
       reallocation). These
       can be used to keep data loaded into a rendering pipline
       or an opencl context */

    /* This class handles multiple arrays.


       We define a read/write lock per array, or more precisely
       per allocated array.

       All arrays managed by a particular allocator
       are implicitly locked by locking the primary
       array for that allocator

       The locking order is determined by the order of the
       calls to addarray() which should generally match
       the order of the arrays in your structure that holds them).

       Our API provides a means to request locking of
       sub-regions of an array, but for now these
       simply translate to nested requests to keep the
       entire array locked.... That means you can't lock
       some regions for read and some regions for write
       of the same array simultaneously.

       You can lock all arrays together with get_locks_write_all()
       or get_locks_read_all()

       You can lock one particular array with get_locks_read_array()
       or get_locks_write_array(). Please note that you must
       follow the locking order convention when doing this
       (or already hold the locks, if you just want another lock token
       for a particular array or region)

       You can lock a region of one particular array with
       get_locks_read_array_region() or get_locks_write_array_region().
       Please note that you must follow the locking order convention
       when doing this (or already hold the locks, if you just
       want another lock token for a particular array or region)
       The locking order convention for regions is that later regions
       must be locked after  earlier regions.
       (Please note that the current version does not actually implement
       region-level granularity)

       You can lock several arrays with get_locks_read_arrays(),
       get_locks_write_arrays(), get_locks_read_arrays_regions(),
       or get_locks_write_arrays_regions(). Assuming either
       nothing is locked beforehand or everything is locked beforehand,
       locking order is not an issue because these functions
       sort your vector of arrays into the proper order.

       These functions return an rwlock_token_set, which is really
       a std::shared_ptr. When the rwlock_token_set and any copies/
       assignments/etc. go out of scope, the lock is released.
       Think of the rwlock_token_set as a handle to what you have
       locked. You can add references to the handle, which will
       prevent those tokens from being released. That is distinct
       from locking the same locks a second time, which gives
       you a new rwlock_token_set that can be passed around
       separately. Conceptually a rwlock_token_set is a single
       set of access rights. If you want to delegate access
       rights to somewhere else, lock the same resources
       a second time (reference the rights you already have),
       release your own rights as appropriate, and pass the
       new rights (optionally downgraded) on to the destination.

       If you hold some locks and want more, or want multiple locks
       on the same resource, you should pass the rwlock_token_set(s)
       that you have as parameters to the locking function. Each
       locking function has a version that accepts a rwlock_token_set
       and a second version that accepts a vector of rwlock_token_sets.
       The locking function returns a new rwlock_token_set with your
       new locks and/or references to the preexisting locks.

       a write lock on an array implies a write lock on
       all regions of the array. Likewise a read lock on
       an array implies a read lock on all regions of the
       array. This means that locking a sub region is really
       more like allowing you  to do  a partial unlock, unless
       you don't have the whole array locked when you lock the
       sub-region

       Note that lock upgrading is not permitted; if you
       have a read lock on any part of the array, you may
       not get a write lock on that part (or the whole array)

       Lock downgrading is possible, with the downgrade_to_read()
       method. This downgrades an entire rwlock_token_set from
       write access to read access. Note that no other references
       to the underlying locks should exist, or an
       snde_error exception will be thrown.
       a single downgrade_to_read() call downgrades all references
       to the referenced rwlock_token_set.


       concurrency and multithreading
       ------------------------------
       Initialization (calls to addarray()) is currently not
       thread safe; all initialization should occur from a single
       thread before calls from other threads can be made.

       Once initialization is complete, locks can be acquired
       from any thread. Be aware that the rwlock_token_set
       Objects themselves are not thread safe. They should
       either be accessed from a single thread, or
       they can be delegated from one thread to another
       (with appropriate synchronization) at which point
       they can be used from the other thread. IN THE
       CURRENT VERSION, BECAUSE THE std::unique_lock USED
       IN rwlock_token IS NOT THREAD SAFE, YOU MUST
       RELEASE ALL OTHER rwlock_token_sets YOU HAVE
       BEFORE DELEGATING A rwlock_token_set TO ANOTHER
       THREAD! (this constraint may be removed in a
       later version). NOTE THAT TO RELEASE A
       rwlock_token_set it must either go out of scope
       or its .reset() method must be called.

       One example of such delegation is when
       data will be processed by a GPU and an arbitrary
       thread may do a callback.
    */


    /* Note that the existance of other data structures
       can implicitly define and maintains the existance of
       array regions e.g. the vertex region mentioned in a
       part structure indicates that those array elements
       exist. That tells you where to find the regions,
       but you still need to lock them to protect against
       e.g. a reallocation process that might be taking the
       data and moving it around to accommodate more space. */


    /* Thoughts:
       The default lock order is first to lass, because earlier defined
       structures (e.g. parts) will generally be higher level...
       so as you traverse, you can lock the part database, figure out
       which vertices, lock those, and perhaps narrow or release your
       hold on the part database.
    */


    /* NOTE: All lockmanager initialization (defining the arrays)
       for a particular class must be done from a single thread,
       before others may
       do any locking */


    
    /* Synchronization model for __arrays, __arrayidx, 
       and __locks: Atomic shared pointer for 
       the content for reading. To change the content, lock the 
       admin mutex, make a complete copy, then 
       switch the atomic pointer. 

       non-atomic shared pointer copy retrieved by the allocators(), 
       allocation_arrays(), and arrays_managed_by_allocator() methods
    */
    /* These next few elements may ONLY be modified during
       initialization phase (single thread, etc.) */
  public:
    std::mutex admin; // Should ONLY be held when rearranging __arrays/__arrayidx/__locks, or manipulating next_array/next_infostore

    lockindex_t _next_array;
    //lockindex_t _next_infostore;
    
    /* DO NOT ACCESS THESE ARRAYS DIRECTLY... ALWAYS USE THE ACCESSORS FOR READ OR THE
       _begin_atomic_update()/_end_atomic_update() FOR WRITE */
    // Note: locking index and locking order position is defined based
    // on index into the _arrays() vector and _locks() deque. Note that entries MAY NOT
    // BE REMOVED because that would change succeeding indices,
    // but they may in the future support disabling, and entries may be added
    // (using _begin_atomic_update(), etc.)

    // Update 2/11/19 and 3/2/19:
    //            * Change __arrays/__locks from vector/deque into unordered_map to support array removal
    //            * Change arrayidx to lockindex_t (signed int64_t) -- negative values invalid
    //            * Add __infostores map (removed 3/2/19)
    
    std::shared_ptr<std::unordered_map<lockindex_t,void **>> __arrays; /* atomic shared pointer to get array pointer from index */
    //std::shared_ptr<std::unordered_map<lockindex_t,mutableinfostore *>> __infostores; 
    
    //std::unordered_map<void **,size_t> _arrayidx; /* get array index from pointer */
    std::shared_ptr<std::unordered_map<void **,lockindex_t,std::hash<void **>,std::equal_to<void **>>> __arrayidx; /* atomic shared pointer to get array index from pointer  (void **)  */

    std::shared_ptr<std::unordered_map<lockindex_t,std::shared_ptr<datalock>>> __locks; /* atomic shared pointer to get lock from index for array locks */
    // (infostore, component, and parameterization locks have a .lock attribute
    
    //std::unordered_map<rwlock *,size_t> _idx_from_lockptr; /* get array index from lock pointer */

    // basic lockmanager data structures (arrays, arrayidx,locks).
    //			 do not require locking for read because they are 
    //                   immutable (replaced when changed)
    
    // ... But _locks[...].subregions is mutable
    // and is locked with locks[...].mutex  mutex.
    
    lockmanager() {
      std::atomic_store(&__arrays,std::make_shared<std::unordered_map<lockindex_t,void **>>());
      //std::atomic_store(&__infostores,std::make_shared<std::unordered_map<lockindex_t,mutableinfostore *>>());
      std::atomic_store(&__arrayidx,std::make_shared<std::unordered_map<void **,lockindex_t,std::hash<void **>,std::equal_to<void **>>>());;
      std::atomic_store(&__locks,std::make_shared<std::unordered_map<lockindex_t,std::shared_ptr<datalock>>>());

      _next_array=1;
      //_next_infostore=-1;
    }
    
    /* Accessors for atomic shared pointers */
    std::shared_ptr<std::unordered_map<lockindex_t,void **>> _arrays()
    {
      /* get array pointer from index */    
      return std::atomic_load(&__arrays);
    }

    ///* Accessors for atomic shared pointers */
    //std::shared_ptr<std::unordered_map<lockindex_t,mutableinfostore *>> _infostores()
    //{
    //  /* get array pointer from index */    
    //  return std::atomic_load(&__infostores);
    //}

    std::shared_ptr<std::unordered_map<void **,lockindex_t,std::hash<void **>,std::equal_to<void **>>> _arrayidx()
    {
      /* get array index from pointer */    
      return std::atomic_load(&__arrayidx);
    }
    
    std::shared_ptr<std::unordered_map<lockindex_t,std::shared_ptr<datalock>>> _locks()
    {
      /* Get lock from index */
      return std::atomic_load(&__locks);
    }

    std::tuple<std::shared_ptr<std::unordered_map<lockindex_t,void **>>,
	       std::shared_ptr<std::unordered_map<void **,lockindex_t,std::hash<void **>,std::equal_to<void **>>>,
	       std::shared_ptr<std::unordered_map<lockindex_t,std::shared_ptr<datalock>>>> _begin_atomic_update()
    // adminlock must be locked when calling this function...
    // it returns new copies of the atomically-guarded data
    {

      // Make copies of atomically-guarded data 
      std::shared_ptr<std::unordered_map<lockindex_t,void **>> new__arrays=std::make_shared<std::unordered_map<lockindex_t,void **>>(*_arrays());
      //std::shared_ptr<std::unordered_map<lockindex_t,mutableinfostore *>> new__infostores=std::make_shared<std::unordered_map<lockindex_t,mutableinfostore *>>(*_infostores());
      std::shared_ptr<std::unordered_map<void **,lockindex_t,std::hash<void **>,std::equal_to<void **>>> new__arrayidx=std::make_shared<std::unordered_map<void **,lockindex_t,std::hash<void **>,std::equal_to<void **>>>(*_arrayidx());
      std::shared_ptr<std::unordered_map<lockindex_t,std::shared_ptr<datalock>>> new__locks=std::make_shared<std::unordered_map<lockindex_t,std::shared_ptr<datalock>>>(*_locks());      
      
      return std::make_tuple(new__arrays,new__arrayidx,new__locks);
    }
    
    void _end_atomic_update(std::shared_ptr<std::unordered_map<lockindex_t,void **>> new__arrays,
			    std::shared_ptr<std::unordered_map<void **,lockindex_t,std::hash<void **>,std::equal_to<void **>>> new__arrayidx,
			    std::shared_ptr<std::unordered_map<lockindex_t,std::shared_ptr<datalock>>> new__locks)
    // adminlock must be locked when calling this function...
    {
      
      // replace old with new

      std::atomic_store(&__arrays,new__arrays);
      std::atomic_store(&__arrayidx,new__arrayidx);
      std::atomic_store(&__locks,new__locks);
      
    }
    
    lockindex_t get_array_idx(void **array)
    {
      lockindex_t ret;
      auto arrayidx = _arrayidx();
      assert(arrayidx->find(array) != arrayidx->end());
      ret = (*arrayidx)[array];
      assert(ret > 0);
      return ret;
    }

    /*
    lockindex_t get_infostore_idx(std::shared_ptr<mutableinfostore> infostore)
    {
      lockindex_t ret;
      auto arrayidx = _arrayidx();
      assert(arrayidx->find(infostore.get()) != arrayidx->end());
      ret = (*arrayidx)[infostore.get()];
      assert(ret < 0);
      return ret;
    }
    */

    /*
    void addinfostore_rawptr(mutableinfostore *infostore)
    {
      // array is pointer to pointer to array data, because
      // the pointer to pointer remains fixed even as the array itself may be reallocated
      lockindex_t idx;
      std::lock_guard<std::mutex> lock(admin);
      
      // Make copies of atomically-guarded data 
      std::shared_ptr<std::unordered_map<lockindex_t,void **>> new__arrays;
      std::shared_ptr<std::unordered_map<lockindex_t,mutableinfostore *>> new__infostores;
      std::shared_ptr<std::unordered_map<void **,lockindex_t,std::hash<void **>,std::equal_to<void **>>> new__arrayidx;
      std::shared_ptr<std::unordered_map<lockindex_t,std::shared_ptr<datalock>>> new__locks;

      std::tie(new__arrays,new__infostores,new__arrayidx,new__locks) = _begin_atomic_update();
      
      assert(new__arrayidx->find((void **)infostore)==new__arrayidx->end());
      
      idx=_next_infostore;
      _next_infostore--; // infostore indexes grow downwards
      
      new__infostores->emplace(idx,infostore);
      
      (*new__arrayidx)[(void*)infostore]=idx;
      

      
      new__locks->emplace(idx,std::make_shared<datalock>()); // Create datalock (NOT arraylock)  object 

      //_idx_from_lockptr[&_locks.back().full_array]=idx;
      // replace old with new
      _end_atomic_update(new__arrays,new__infostores,new__arrayidx,new__locks);

    }
    */

    /*
    void addinfostore(std::shared_ptr<mutableinfostore> infostore)
    {

      addinfostore_rawptr(infostore.get());
      
    }
    */
    
    void addarray(void **array) {
      // array is pointer to pointer to array data, because
      // the pointer to pointer remains fixed even as the array itself may be reallocated
      lockindex_t idx;
      std::lock_guard<std::mutex> lock(admin);

      // Make copies of atomically-guarded data 
      std::shared_ptr<std::unordered_map<lockindex_t,void **>> new__arrays;
      std::shared_ptr<std::unordered_map<void **,lockindex_t,std::hash<void **>,std::equal_to<void **>>> new__arrayidx;
      std::shared_ptr<std::unordered_map<lockindex_t,std::shared_ptr<datalock>>> new__locks;

      std::tie(new__arrays,new__arrayidx,new__locks) = _begin_atomic_update();
      
      assert(new__arrayidx->find(array)==new__arrayidx->end());
      
      idx=_next_array;
      _next_array++;
      
      new__arrays->emplace(idx,array);
      
      (*new__arrayidx)[(void**)array]=idx;
      

      
      new__locks->emplace(idx,std::make_shared<arraylock>());  /* Create arraylock object */

      //_idx_from_lockptr[&_locks.back().full_array]=idx;
      // replace old with new
      _end_atomic_update(new__arrays,new__arrayidx,new__locks);


    }

    void remarray(void **array)
    {
      std::lock_guard<std::mutex> lock(admin);

      // Make copies of atomically-guarded data 
      std::shared_ptr<std::unordered_map<lockindex_t,void **>> new__arrays;
      std::shared_ptr<std::unordered_map<void **,lockindex_t,std::hash<void **>,std::equal_to<void **>>> new__arrayidx;
      std::shared_ptr<std::unordered_map<lockindex_t,std::shared_ptr<datalock>>> new__locks;

      std::tie(new__arrays,new__arrayidx,new__locks) = _begin_atomic_update();

      auto new__ai_iter = new__arrayidx->find(array);
      
      assert(new__ai_iter != new__arrayidx->end());
      lockindex_t idx = new__ai_iter->second;
      assert(idx > 0);
      
      
      new__arrayidx->erase(new__ai_iter);
      
      

      auto new__a_iter = new__arrays->find(idx);
      assert(new__a_iter != new__arrays->end());
      new__arrays->erase(new__a_iter);


      auto new__l_iter = new__locks->find(idx);
      assert(new__l_iter != new__locks->end());
      new__locks->erase(new__l_iter);

      
      //_idx_from_lockptr[&_locks.back().full_array]=idx;
      // replace old with new
      _end_atomic_update(new__arrays,new__arrayidx,new__locks);



    }

    /*
    void reminfostore_rawptr(mutableinfostore *infostore)
    {
      std::lock_guard<std::mutex> lock(admin);
      
      // Make copies of atomically-guarded data 
      std::shared_ptr<std::unordered_map<lockindex_t,void **>> new__arrays;
      std::shared_ptr<std::unordered_map<lockindex_t,mutableinfostore *>> new__infostores;
      std::shared_ptr<std::unordered_map<void *,lockindex_t,std::hash<void *>,std::equal_to<void *>>> new__arrayidx;
      std::shared_ptr<std::unordered_map<lockindex_t,std::shared_ptr<datalock>>> new__locks;

      std::tie(new__arrays,new__infostores,new__arrayidx,new__locks) = _begin_atomic_update();

      auto new__ai_iter = new__arrayidx->find((void *)infostore);
      
      assert(new__ai_iter != new__arrayidx->end());
      lockindex_t idx = new__ai_iter->second;
      assert(idx < 0);
      
      
      new__arrayidx->erase(new__ai_iter);
      
      

      auto new__is_iter = new__infostores->find(idx);
      assert(new__is_iter != new__infostores->end());
      new__infostores->erase(new__is_iter);


      auto new__l_iter = new__locks->find(idx);
      assert(new__l_iter != new__locks->end());
      new__locks->erase(new__l_iter);

      
      //_idx_from_lockptr[&_locks.back().full_array]=idx;
      // replace old with new
      _end_atomic_update(new__arrays,new__infostores,new__arrayidx,new__locks);



    }
    void reminfostore(std::shared_ptr<mutableinfostore> infostore)
    {

      reminfostore_rawptr(infostore.get());

    }

    */


    
    bool is_region_granular(void) /* Return whether locking is really granular on a region-by-region basis (true) or just on an array-by-array basis (false) */
    {
      return true;
    }

    void set_array_size(void **Arrayptr,size_t elemsize,snde_index nelem) {
      // We don't currently care about the array size
    }

    //void writer_mark_as_dirty(cachemanager *cache,void **arrayptr,snde_index pos,snde_index size)
    //{
    //  size_t arrayidx=_arrayidx.at(array);
    //
    //
    //  std::unique_lock<std::mutex> arrayadminlock(_locks[arrayidx].admin);
    //
    //  std::map<markedregion,rwlock>::iterator iter=manager->locker->_locks[arrayidx].subregions.lower_bound(pos);

    // if (pos < iter.first.regionstart) { /* probably won't happen due to array layout process, but just in case */
    //assert(iter != _locks[arrayidx].subregions.begin());
    //	iter--;
    //  }
    //
    //// iterate over the subregions of this arraylock
    //  for (;iter != manager->locker->_locks[arrayidx].subregions.end() && iter->first.regionstart < writeregion.second->regionend;iter++) {
    //snde_index regionstart=dirtyregion->regionstart;
    //snde_index regionend=dirtyregion->regionend;
    //
    //
    //if (iter->first.regionstart > regionstart) {
    //regionstart=iter->first.regionstart;
    //}
    //if (regionend > iter->first.regionend) {
    //regionend=iter->first.regionend;
    //}
    //
    //iter->second.writer_mark_as_dirty(this,regionstart,regionend-regionstart);
    //
    //}
    
    //}
    
    rwlock_token newallocation(rwlock_token_set all_locks,void **arrayptr,snde_index pos,snde_index size,snde_index elemsize)
    {
      /* callback from allocator */
      /* returns locked token */
      /* Entire array should be write locked in order to call this */
      lockindex_t arrayidx=_arrayidx()->at((void**)arrayptr);

      std::shared_ptr<arraylock> thislock=std::dynamic_pointer_cast<arraylock>(_locks()->at(arrayidx));
      std::unique_lock<std::mutex> lock(thislock->admin); // lock the subregions field

      
      
      
      // Notification from allocator
      markedregion region(pos,pos+size);
      
      assert(thislock->subregions.count(region)==0);

      // add new rwlock to subregion
      thislock->subregions.emplace(std::piecewise_construct,
				   std::forward_as_tuple(pos,pos+size),
				   std::forward_as_tuple());

      thislock->subregions.at(region)=std::make_shared<rwlock>();
      std::shared_ptr<rwlock> rwlockobj=thislock->subregions.at(region);
      
      rwlock_token retval(new std::unique_lock<rwlock_lockable>(rwlockobj->writer));
      (*all_locks)[&rwlockobj->writer]=retval;
      
      return retval;
    }
    
    void realloc_down_allocation(void **arrayptr, snde_index addr,snde_index orignelem, snde_index newnelem)
    {
      /* callback from allocator */
      /* callback from allocator */
      lockindex_t idx=_arrayidx()->at(arrayptr);
      std::shared_ptr<arraylock> thislock=std::dynamic_pointer_cast<arraylock>(_locks()->at(idx));
      std::lock_guard<std::mutex> lock(thislock->admin);

      // notification from allocator that an allocation has shrunk

      // Find pointer for lock for this region
      std::shared_ptr<rwlock> lock_ptr = thislock->subregions.at(markedregion(addr,addr+orignelem));
      
      // Remove this pointer from the subregions map
      thislock->subregions.erase(markedregion(addr,addr+orignelem));

      // Reinsert it with the new size
      thislock->subregions.emplace(std::make_pair(markedregion(addr,addr+newnelem),lock_ptr));
      
    }
    
    void freeallocation(void **arrayptr,snde_index pos, snde_index size,snde_index elemsize)
    {
      /* callback from allocator */
      lockindex_t idx=_arrayidx()->at(arrayptr);
      std::shared_ptr<arraylock> thislock=std::dynamic_pointer_cast<arraylock>(_locks()->at(idx));
      std::lock_guard<std::mutex> lock(thislock->admin);


      // notification from allocator
      thislock->subregions.erase(markedregion(pos,pos+size));
      
    }


    // use lock_recording_refs() to lock a bunch of ndarray_recording refs
    // use brace initialization; the 2nd half of the pair is true for write locking:
    //  mylock = lock_recording_refs({ {inputrec1,false},
    //                                 {inputrec2,false},
    //                                 {outputrec1,true} });
    // GPU access flag causes it to make the locking decision based on the original recording storage
    // and the requires_locking_read_gpu/requires_locking_write_gpu storage flags instead of requires_locking_read and requires_locking_write
    rwlock_token_set lock_recording_refs(std::vector<std::pair<std::shared_ptr<ndarray_recording_ref>,bool>> recrefs,bool gpu_access=false);
    
    // use lock_recording_arrays() to lock a bunch of multi_ndarray_recording elements 
    // use brace initialization; the 2nd half of the pair is true for write locking:
    //  mylock = lock_recording_arrays({ {inputrec1, {"parts", false} },
    //                                   {inputrec2, {"triangles", false} },
    //                                   {outputrec1, {"edges", true} } });
    // GPU access flag causes it to make the locking decision based on the original recording storage
    // and the requires_locking_read_gpu/requires_locking_write_gpu storage flags instead of requires_locking_read and requires_locking_write
    rwlock_token_set lock_recording_arrays(std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<size_t,bool>>> recrefs,bool gpu_access /* = false */);
    
    rwlock_token_set lock_recording_arrays(std::vector<std::pair<std::shared_ptr<multi_ndarray_recording>,std::pair<std::string,bool>>> recrefs,bool gpu_access=false);
    
    
    rwlock_token  _get_preexisting_lock_read_lockobj(rwlock_token_set all_locks,std::shared_ptr<rwlock> rwlockobj)
    /* returns NULL if there is no such preexisting read lock or
       there is no preexisting write lock that is convertable to a read lock */
    {      // must hold write lock on entire array... returns write lock on new allocation
      // and position

      // prior is like a rwlock_token_set **
      
      rwlock_lockable *lockobj=&rwlockobj->reader;
      rwlock_lockable *writelockobj=&rwlockobj->writer;
      rwlock_token writelocktoken;

	
      if ((*all_locks).count(lockobj)) {
	/* If we have this token */
	/* return a reference */
	return (*all_locks)[lockobj];
      }
      if ((*all_locks).count(writelockobj)) {
	/* There is a write lock for this token */
	writelocktoken=(*all_locks)[writelockobj];
      }
      
      /* if we got here, we do not have a token, need to make one */
      if (writelocktoken) {
	/* There is a write token, but not a read token */
	rwlockobj->sidegrade(); /* add read capability to write lock */
        /* this capability is important to avoid deadlocking
           if a single owner locks one subregion for read and
           another subregion for write, then so long as the write
           lock was done first, it will not deadlock. Unfortunately
           this doesn't cover all situations because the locking order
           specifies that earlier blocks should be allocated first,
           and the earlier block may not be the write block. */
	/* now create a new reader token that adopts the read capability
	   we just added */
	rwlock_token retval=std::make_shared<std::unique_lock<rwlock_lockable>>(*lockobj,std::adopt_lock);
	(*all_locks)[lockobj]=retval;
	
	return retval;
      }
      
      return std::shared_ptr<std::unique_lock<rwlock_lockable>>();

    }

    
    std::pair<rwlock_lockable *,rwlock_token>  _get_preexisting_lock_read_array_region(rwlock_token_set all_locks, lockindex_t arrayidx,snde_index pos,snde_index size)
    /* returns NULL if there is no such preexisting read lock or
       there is no preexisting write lock that is convertable to a read lock */
    {
      // prior is like a rwlock_token_set **
      std::shared_ptr<arraylock> thislock=std::dynamic_pointer_cast<arraylock>(_locks()->at(arrayidx));
      std::unique_lock<std::mutex> lock(thislock->admin); // lock the subregions field

      if (!thislock->subregions.size() && pos==0) {
	// this array does not have regions ... just lock whole array
	lock.unlock();
	return std::make_pair(&thislock->whole->reader,_get_preexisting_lock_read_lockobj(all_locks, thislock->whole));
      } else {
	auto rwlockobj_iter = thislock->subregions.find(markedregion(pos,pos+size));
	if (rwlockobj_iter != thislock->subregions.end()) {
	  
	  std::shared_ptr<rwlock> rwlockobj = rwlockobj_iter->second; //thislock->subregions.at(markedregion(pos,pos+size));
	  // now that we hold the shared pointer it can't be deleted behind our back, and
	  // locking it secures the keepalive. 
	  
	  lock.unlock();
	  return std::make_pair(&rwlockobj->reader,_get_preexisting_lock_read_lockobj(all_locks, rwlockobj));
	} else {
	  // subregion missing. This means it was deleted behind our back and therefore no longer needs to be locked
	  return std::make_pair(nullptr,nullptr);
	  
	}
      }
    }
    


    std::pair<rwlock_lockable *,rwlock_token>  _get_lock_read_array_region(rwlock_token_set all_locks, lockindex_t arrayidx,snde_index pos,snde_index size)
    {

      rwlock_token retval;
      rwlock_lockable *lockobj;
      std::tie(lockobj,retval)=_get_preexisting_lock_read_array_region(all_locks,arrayidx,pos,size);
      
      if (lockobj && !retval) {
	/* if we got here, we do not have a token, need to make one */

	retval = rwlock_token(new std::unique_lock<rwlock_lockable>(*lockobj));
	(*all_locks)[lockobj]=retval;
      }
      return std::make_pair(lockobj,retval);
    }

  
    template <class T> 
    rwlock_token_set get_locks_read_lockable(rwlock_token_set all_locks,std::shared_ptr<T> lockable)
    {
      rwlock_lockable *reader = &lockable->lock->reader;
      rwlock_token tok = std::make_shared<std::unique_lock<rwlock_lockable>>(*reader);
      
      rwlock_token_set ret = std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
      ret->emplace(reader,tok);

      merge_into_rwlock_token_set(all_locks,ret);
      
      return ret;      
    }

    rwlock_token_set get_locks_read_array(rwlock_token_set all_locks, void **array)
    {
      rwlock_token_set token_set=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
      
      lockindex_t idx=_arrayidx()->at(array);
      std::shared_ptr<arraylock> alock=std::dynamic_pointer_cast<arraylock>(_locks()->at(idx));


      // First, get wholearray lock
      
      rwlock_token wholearray_token = _get_preexisting_lock_read_lockobj(all_locks,alock->whole);
      if (wholearray_token==nullptr) {
	wholearray_token = rwlock_token(new std::unique_lock<rwlock_lockable>(alock->whole->reader));
	(*all_locks)[&alock->whole->reader]=wholearray_token;
      }
      (*token_set)[&alock->whole->reader]=wholearray_token;
            
      /* now that we have wholearray_lock, nobody else can do allocations, etc. that may add to subregions, 
	 but we can't safely iterate over it without holding the admin lock because entries might expire 
	 (via freallocation()). What we can do is copy it but we have to be careful in case something is
	 freed behind our back. 
      */

      std::map<markedregion,std::shared_ptr<rwlock>> alock_subregions_copy;
      {
	std::lock_guard<std::mutex> alock_admin(alock->admin);
	alock_subregions_copy = alock->subregions; 
      }

      for (auto & markedregion_rwlock : alock_subregions_copy) {
	rwlock_lockable *lockableptr;
	rwlock_token token;
	std::tie(lockableptr,token)=_get_lock_read_array_region(all_locks,idx,markedregion_rwlock.first.regionstart,markedregion_rwlock.first.regionend-markedregion_rwlock.first.regionstart);
	if (lockableptr) {
	  // can be nullptr if the lockable was freed before we could lock it (otherwise kept in place by the keepalive)
	  (*token_set)[lockableptr]=token;
	}
      }
      return token_set;
    }

    template <class T>
    rwlock_token_set get_preexisting_locks_read_lockable(rwlock_token_set all_locks, std::shared_ptr<T> lockable)
    {
      rwlock_token_set token_set=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();

      //assert(_arrayidx.find(array) != _arrayidx.end());
      

      // First, get whole lock      
      rwlock_token whole_token = _get_preexisting_lock_read_lockobj(all_locks,lockable->lock);
      if (whole_token==nullptr) {
	throw snde_error("Must have valid preexisting lockable lock (this may be a locking order violation)");
      }

      (*token_set)[&lockable->lock->reader]=whole_token;
      
      return token_set;
    }


    rwlock_token_set get_preexisting_locks_read_array(rwlock_token_set all_locks, void **array)
    {
      rwlock_token_set token_set=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();

      //assert(_arrayidx.find(array) != _arrayidx.end());
      lockindex_t idx=_arrayidx()->at(array);
      std::shared_ptr<arraylock> alock=std::dynamic_pointer_cast<arraylock>(_locks()->at(idx));

      // First, get wholearray lock      
      rwlock_token wholearray_token = _get_preexisting_lock_read_lockobj(all_locks,alock->whole);
      if (wholearray_token==nullptr) {
	throw snde_error("Must have valid preexisting whole lock (this may be a locking order violation)");
      }

      (*token_set)[&alock->whole->reader]=wholearray_token;

      /* now that we have wholearray_lock, nobody else can do allocations, etc. that may add to subregions, 
	 but we can't safely iterate over it without holding the admin lock because entries might expire 
	 (via freallocation()). What we can do is copy it but we have to be careful in case something is
	 freed behind our back. In this case the preexisting lock is required, so that should be findable
      */

      std::map<markedregion,std::shared_ptr<rwlock>> alock_subregions_copy;
      {
	std::lock_guard<std::mutex> alock_admin(alock->admin);
	alock_subregions_copy = alock->subregions; 
      }
      
      for (auto & markedregion_rwlock : alock_subregions_copy) {	
      
	rwlock_lockable *lockableptr;
	rwlock_token preexisting_lock;
	std::tie(lockableptr,preexisting_lock)=_get_preexisting_lock_read_array_region(all_locks,idx,markedregion_rwlock.first.regionstart,markedregion_rwlock.first.regionend-markedregion_rwlock.first.regionstart);
	
	if (preexisting_lock==nullptr) {
	  throw snde_error("Must have valid preexisting lock (this may be a locking order violation)");
	}
	//(*token_set)[&markedregion_rwlock.second.reader]=preexisting_lock;
	(*token_set)[lockableptr]=preexisting_lock;

      }
      return token_set;
    }


    
    rwlock_token_set get_preexisting_locks_read_array_region(rwlock_token_set all_locks, void **array,snde_index indexstart,snde_index numelems)
    {
      rwlock_lockable *lockobj;
      rwlock_token retval;

      rwlock_token_set token_set=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();

      if (indexstart==0 && numelems==SNDE_INDEX_INVALID) {
	return get_preexisting_locks_read_array(all_locks,array);
      }

      lockindex_t arrayidx=_arrayidx()->at(array);

      std::tie(lockobj,retval) =  _get_preexisting_lock_read_array_region(all_locks, arrayidx,indexstart,numelems);

      if (retval==nullptr) {
	throw snde_error("Must have valid preexisting lock (this may be a locking order violation)");
      }
      (*token_set)[lockobj]=retval;
	
      return token_set;
    }
  
    rwlock_token_set get_locks_read_array_region(rwlock_token_set all_locks, void **array,snde_index indexstart,snde_index numelems)
    {
      rwlock_lockable *lockobj;

      rwlock_token retval;

      if (indexstart==0 && numelems==SNDE_INDEX_INVALID) {
	return get_locks_read_array(all_locks,array);
      }

      
      rwlock_token_set token_set=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
      lockindex_t arrayidx=_arrayidx()->at(array);

      std::tie(lockobj,retval) =  _get_lock_read_array_region(all_locks, arrayidx,indexstart,numelems);

      (*token_set)[lockobj]=retval;
      return token_set;
    }



    rwlock_token_set get_locks_read_all_arrays(rwlock_token_set all_locks)
    {
      rwlock_token_set tokens=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
      std::vector<lockindex_t> idxs;
      idxs.reserve(_arrays()->size());

      auto arrays = _arrays();
      for (auto & lockindex_voidpp: *arrays) {
	idxs.push_back(lockindex_voidpp.first);
      }
      std::sort(idxs.begin(),idxs.end());
      
      for (size_t cnt=0;cnt < idxs.size();cnt++) {
        rwlock_token_set thisset = get_locks_read_array(all_locks,(*_arrays())[idxs[cnt]]);
	merge_into_rwlock_token_set(tokens,thisset);
      }
      return tokens;
    }

    rwlock_token_set get_locks_read_all(rwlock_token_set all_locks)
    {
      rwlock_token_set tokens=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
      std::vector<lockindex_t> idxs;
      idxs.reserve(_arrayidx()->size());

      auto arrayidx = _arrayidx();
      for (auto & voidptr_lockindex: *arrayidx) {
	idxs.push_back(voidptr_lockindex.second);	
      }
      std::sort(idxs.begin(),idxs.end());
      
      for (size_t cnt=0;cnt < idxs.size();cnt++) {
        rwlock_token_set thisset = get_locks_read_array(all_locks,(*_arrays())[idxs[cnt]]);
	merge_into_rwlock_token_set(tokens,thisset);
      }
      return tokens;
    }
    
    rwlock_token  _get_preexisting_lock_write_lockobj(rwlock_token_set all_locks,std::shared_ptr<rwlock> rwlockobj)
    {
      // but we do store the bounds for notification purposes (NOT ANYMORE; dirty marking is now explicit)

      // prior is like a rwlock_token_set **
      rwlock_token token;
      
      rwlock_lockable *lockobj=&rwlockobj->writer;

      if ((*all_locks).count(lockobj)) {	  /* If we have this token */
      	  /* return a reference */
	  //(**prior)[lockobj]

	return (*all_locks)[lockobj];
      }
      
      return std::shared_ptr<std::unique_lock<rwlock_lockable>>(); /* return nullptr if there is no preexisting lock */
    }
    

    std::pair<rwlock_lockable *,rwlock_token>  _get_preexisting_lock_write_array_region(rwlock_token_set all_locks, lockindex_t arrayidx,snde_index indexstart,snde_index numelems)
    {
      // We currently implement region-granular locking
      // but we do store the bounds for notification purposes (NOT ANYMORE; dirty marking is now explicit)

      // prior is like a rwlock_token_set **
      std::shared_ptr<arraylock> thislock=std::dynamic_pointer_cast<arraylock>(_locks()->at(arrayidx));
      std::unique_lock<std::mutex> lock(thislock->admin); // lock the subregions field


      if (!thislock->subregions.size() && indexstart==0) {
	// This array does not have subregions... lock the whole array
	lock.unlock();
	return std::make_pair(&thislock->whole->writer,_get_preexisting_lock_write_lockobj(all_locks,thislock->whole));
	
      }
      auto rwlockobj_iter = thislock->subregions.find(markedregion(indexstart,indexstart+numelems));
      if (rwlockobj_iter != thislock->subregions.end()) {
	std::shared_ptr<rwlock> rwlockobj = rwlockobj_iter->second; // thislock->subregions.at(markedregion(indexstart,indexstart+numelems));
	// now that we hold the shared pointer it can't be deleted behind our back, and
	// locking it secures the keepalive. 
	lock.unlock(); 
	return std::make_pair(&rwlockobj->writer,_get_preexisting_lock_write_lockobj(all_locks,rwlockobj));
      } else {
	// subregion missing. This means it was deleted behind our back and therefore no longer needs to be locked
	return std::make_pair(nullptr,nullptr);
      }
      
    }
    
    std::pair<rwlock_lockable *,rwlock_token>  _get_lock_write_array_region(rwlock_token_set all_locks, lockindex_t arrayidx,snde_index pos,snde_index size)
    {
      // but we do store the bounds for notification purposes (NOT ANYMORE; dirty marking is now explicit)

      rwlock_token retval;
      rwlock_lockable *lockobj;


      std::tie(lockobj,retval)=_get_preexisting_lock_write_array_region(all_locks,arrayidx,pos,size);
      
      if (lockobj && !retval) {
	
	/* if we got here, we do not have a token, need to make one */
	retval = std::make_shared<std::unique_lock<rwlock_lockable>>(*lockobj);
	(*all_locks)[lockobj]=retval;
	

      }
      // Dirty marking now must be done explicitly
      //lockobj->_rwlock_obj->writer_append_region(indexstart,numelems);
      return std::make_pair(lockobj,retval);
      
    }

    template <class T>
    rwlock_token_set get_locks_write_lockable(rwlock_token_set all_locks,std::shared_ptr<T> lockable)
    {
      rwlock_lockable *writer = &lockable->lock->writer;
      rwlock_token tok = std::make_shared<std::unique_lock<rwlock_lockable>>(*writer);
      
      rwlock_token_set ret = std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
      ret->emplace(writer,tok);
      
      merge_into_rwlock_token_set(all_locks,ret);
      
      return ret;      
    }

    rwlock_token_set get_locks_write_array(rwlock_token_set all_locks, void **array) {
      rwlock_token_set token_set=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
      
      lockindex_t idx=_arrayidx()->at(array);
      std::shared_ptr<arraylock> alock=std::dynamic_pointer_cast<arraylock>(_locks()->at(idx));
      
      // First, get wholearray lock
      
      rwlock_token wholearray_token = _get_preexisting_lock_write_lockobj(all_locks,alock->whole);
      if (wholearray_token==nullptr) {
	wholearray_token = rwlock_token(new std::unique_lock<rwlock_lockable>(alock->whole->writer));
	(*all_locks)[&alock->whole->writer]=wholearray_token;
      }
      (*token_set)[&alock->whole->writer]=wholearray_token;
      
      /* now that we have wholearray_lock, nobody else can do allocations, etc. that may add to subregions, 
	 but we can't safely iterate over it without holding the admin lock because entries might expire 
	 (via freallocation()). What we can do is copy it but we have to be careful in case something is
	 freed behind our back. 
      */

      
      std::map<markedregion,std::shared_ptr<rwlock>> alock_subregions_copy;
      {
	std::lock_guard<std::mutex> alock_admin(alock->admin);
	alock_subregions_copy = alock->subregions; 
      }
      
      for (auto & markedregion_rwlock : alock_subregions_copy) {
	
	//(*token_set)[&markedregion_rwlock.second.writer]
	rwlock_lockable *lockableptr;
	rwlock_token token;
	std::tie(lockableptr,token)=_get_lock_write_array_region(all_locks,idx,markedregion_rwlock.first.regionstart,markedregion_rwlock.first.regionend-markedregion_rwlock.first.regionstart);
	if (lockableptr) {
	  // can be nullptr if the lockable was freed before we could lock it (otherwise kept in place by the keepalive)
	  (*token_set)[lockableptr]=token;
	}
      }
      return token_set;
    }

    template <class T>
    rwlock_token_set get_preexisting_locks_write_lockable(rwlock_token_set all_locks, std::shared_ptr<T> lockable)
    {
      rwlock_token_set token_set=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();

      //assert(_arrayidx.find(array) != _arrayidx.end());
      //lockindex_t idx=_arrayidx()->at((void *)infostore.get());
      //std::shared_ptr<datalock> dlock=_locks()->at(idx);

      // First, get whole lock      
      rwlock_token whole_token = _get_preexisting_lock_write_lockobj(all_locks,lockable->lock);
      if (whole_token==nullptr) {
	throw snde_error("Must have valid preexisting locakble lock (this may be a locking order violation)");
      }

      (*token_set)[&lockable->lock->writer]=whole_token;
      
      return token_set;
    }

    rwlock_token_set get_preexisting_locks_write_array(rwlock_token_set all_locks, void **array) {
      rwlock_token_set token_set=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();

      //assert(_arrayidx.find(array) != _arrayidx.end());
      lockindex_t idx=_arrayidx()->at(array);
      std::shared_ptr<arraylock> alock=std::dynamic_pointer_cast<arraylock>(_locks()->at(idx));

      // First, get wholearray lock      
      rwlock_token wholearray_token = _get_preexisting_lock_write_lockobj(all_locks,alock->whole);
      if (wholearray_token==nullptr) {
	throw snde_error("Must have valid preexisting whole lock (this may be a locking order violation)");
      }

      (*token_set)[&alock->whole->writer]=wholearray_token;
      
      /* now that we have wholearray_lock, nobody else can do allocations, etc. that may add to subregions, 
	 but we can't safely iterate over it without holding the admin lock because entries might expire 
	 (via freallocation()). What we can do is copy it but we have to be careful in case something is
	 freed behind our back. In this case the preexisting lock is required, so that should be findable.
      */

      std::map<markedregion,std::shared_ptr<rwlock>> alock_subregions_copy;
      {
	std::lock_guard<std::mutex> alock_admin(alock->admin);
	alock_subregions_copy = alock->subregions; 
      }
      
      for (auto & markedregion_rwlock : alock_subregions_copy) {	
	rwlock_lockable *lockableptr;
	rwlock_token preexisting_lock;
      
	std::tie(lockableptr,preexisting_lock)=_get_preexisting_lock_write_array_region(all_locks,idx,markedregion_rwlock.first.regionstart,markedregion_rwlock.first.regionend-markedregion_rwlock.first.regionstart);
	
	if (preexisting_lock==nullptr) {
	  throw snde_error("Must have valid preexisting lock (this may be a locking order violation)");
	}
	(*token_set)[&markedregion_rwlock.second->writer]=preexisting_lock;
      }
      return token_set;
    }

    rwlock_token_set get_preexisting_locks_write_array_region(rwlock_token_set all_locks, void **array,snde_index indexstart,snde_index numelems)
    {
      rwlock_lockable *lockobj;
      rwlock_token retval;

      if (indexstart==0 && numelems==SNDE_INDEX_INVALID) {
	return get_preexisting_locks_write_array(all_locks,array);
      }

      rwlock_token_set token_set=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
      lockindex_t arrayidx=_arrayidx()->at(array);

      std::tie(lockobj,retval) =  _get_preexisting_lock_write_array_region(all_locks, arrayidx,indexstart,numelems);

      if (retval==nullptr) {
	throw snde_error("Must have valid preexisting lock (this may be a locking order violation)");
      }
      (*token_set)[lockobj]=retval;
	
      return token_set;

    }

    rwlock_token_set get_locks_write_array_region(rwlock_token_set all_locks, void **array,snde_index indexstart,snde_index numelems)
    {
      rwlock_lockable *lockobj;

      rwlock_token retval;

      if (indexstart==0 && numelems==SNDE_INDEX_INVALID) {
	return get_locks_write_array(all_locks,array);
      }
      
      rwlock_token_set token_set=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
      lockindex_t arrayidx=_arrayidx()->at(array);

      std::tie(lockobj,retval) =  _get_lock_write_array_region(all_locks, arrayidx,indexstart,numelems);

      (*token_set)[lockobj]=retval;
      return token_set;


    }



    rwlock_token_set get_locks_write_all_arrays(rwlock_token_set all_locks)
    {
      rwlock_token_set tokens=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
      std::vector<lockindex_t> idxs;
      idxs.reserve(_arrays()->size());

      auto arrays = _arrays();
      for (auto & lockindex_voidpp: *arrays) {
	idxs.push_back(lockindex_voidpp.first);
      }
      std::sort(idxs.begin(),idxs.end());
      
      for (size_t cnt=0;cnt < idxs.size();cnt++) {
        rwlock_token_set thisset = get_locks_write_array(all_locks,(*_arrays())[idxs[cnt]]);
	merge_into_rwlock_token_set(tokens,thisset);
      }
      return tokens;
    }

    rwlock_token_set get_locks_write_all(rwlock_token_set all_locks)
    {
      rwlock_token_set tokens=std::make_shared<std::unordered_map<rwlock_lockable *,rwlock_token>>();
      std::vector<lockindex_t> idxs;
      idxs.reserve(_arrayidx()->size());

      auto arrayidx = _arrayidx();
      for (auto & voidptr_lockindex: *arrayidx) {
	idxs.push_back(voidptr_lockindex.second);	
      }
      std::sort(idxs.begin(),idxs.end());
      
      for (size_t cnt=0;cnt < idxs.size();cnt++) {
        rwlock_token_set thisset = get_locks_write_array(all_locks,(*_arrays())[idxs[cnt]]);
	merge_into_rwlock_token_set(tokens,thisset);
      }
      return tokens;
    }
    

  
    void downgrade_to_read(rwlock_token_set locks) {
      /* locks within the token_set MUST NOT be referenced more than once.... That means you must
	 have released your all_locks rwlock_token_set object*/

      for (std::unordered_map<rwlock_lockable *,rwlock_token>::iterator lockable_token=locks->begin();lockable_token != locks->end();lockable_token++) {
	// lockable_token.first is reference to the lockable
	// lockable_token.second is reference to the token
	if (lockable_token->second.use_count() != 1) {
	  throw snde_error("Locks referenced by more than one token_set may not be downgraded");
	  lockable_token->first->_rwlock_obj->downgrade();
	}
      }

    }

    template <class T>
    rwlock_token_set lock_lockables(rwlock_token_set all_locks,std::vector<std::shared_ptr<T>> lockables,bool write)
    {
      // sort lockables of the same type into proper locking order and uniqueify them
      // by putting them into a set
      //auto arrayidx = _arrayidx();
      rwlock_token_set new_locks=empty_rwlock_token_set();

      std::set<std::shared_ptr<T>,std::owner_less<std::shared_ptr<T>>> lockables_set(lockables.begin(),lockables.end());

      // now that we have the order figured out, perform the locking.
      
      for (auto & lockable: lockables_set) {
	if (write) {
	  rwlock_token_set lockable_locks=get_locks_write_lockable(all_locks,lockable);
	  merge_into_rwlock_token_set(new_locks,lockable_locks);
	} else {
	  rwlock_token_set lockable_locks=get_locks_read_lockable(all_locks,lockable);
	  merge_into_rwlock_token_set(new_locks,lockable_locks);
	  
	}
      }
      return new_locks;
    }

  
    //virtual rwlock_token_set lock_infostores(rwlock_token_set all_locks,std::shared_ptr<mutablerecdb> recdb,std::set<std::string> channels_to_lock,bool write); // moved to mutablerecstore.hpp
    
  };

class lockholder_index {
public:

  // Not permitted to actually use infostore, comp, or param
  // pointers, as they may have expired.
  //mutableinfostore *infostore;
  //geometry *geom;
  //component *comp;
  //parameterization *param;
#ifdef SNDE_MUTABLE_RECDB_SUPPORT
  lockable_infostore_or_component *lic;
#endif // SNDE_MUTABLE_RECDB_SUPPORT
  void **array;
  bool write;
  snde_index startidx;
  snde_index numelem;
  lockholder_index();
  lockholder_index(void **_array,bool _write, snde_index _startidx,snde_index _numelem);
  
  /*
  lockholder_index(mutableinfostore *_infostore,bool _write) :
    infostore(_infostore), geom(nullptr), comp(nullptr), param(nullptr), array(nullptr), write(_write), startidx(0), numelem(SNDE_INDEX_INVALID)
  {

  }
  */
  /*
    lockholder_index(geometry *_geom,bool _write) :
    infostore(nullptr), geom(_geom), comp(nullptr), param(nullptr), array(nullptr), write(_write), startidx(0), numelem(SNDE_INDEX_INVALID)
  {

  }
  */
  /*
  lockholder_index(component *comp,bool _write) :
    infostore(nullptr), geom(nullptr), comp(comp), param(nullptr), array(nullptr), write(_write), startidx(0), numelem(SNDE_INDEX_INVALID)
  {

  }

  lockholder_index(parameterization *param,bool _write) :
    infostore(nullptr), geom(nullptr), comp(nullptr), param(param), array(nullptr), write(_write), startidx(0), numelem(SNDE_INDEX_INVALID)
  {

  }
  */
  
#ifdef SNDE_MUTABLE_RECDB_SUPPORT
  lockholder_index(lockable_infostore_or_component *_lic,bool _write);
#endif // SNDE_MUTABLE_RECDB_SUPPORT

  
  // equality operator for std::unordered_map
  bool operator==(const lockholder_index b) const;
};

/* provide hash implementation for indices used for lockholder */

struct lockholder_index_hash {
  size_t operator()(const lockholder_index &x) const
  {
    return /*std::hash<void *>{}((void *)x.infostore)+std::hash<void *>{}((void *)x.geom)+std::hash<void *>{}((void *)x.comp)+std::hash<void *>{}((void *)x.param)*/
#ifdef SNDE_MUTABLE_RECDB_SUPPORT
  std::hash<void *>{}((void *)x.lic)+
#endif // SNDE_MUTABLE_RECDB_SUPPORT
               std::hash<void *>{}((void *)x.array) + std::hash<bool>{}(x.write) + std::hash<snde_index>{}(x.startidx) + std::hash<snde_index>{}(x.numelem);
  }
};
  
  class lockingprocess_thread {
  public:
    virtual ~lockingprocess_thread() {};
  };


  class lockingprocess {
      /* lockingprocess is a tool for performing multiple locking
         for multiple objects while satisfying the required
         locking order */

      /* (There was going to be an opencl_lockingprocess that was to be derived
         from this class, but it was cancelled) */
  public:
    //lockingprocess();
    //lockingprocess(const lockingprocess &)=delete; /* copy constructor disabled */
    //lockingprocess& operator=(const lockingprocess &)=delete; /* copy assignment disabled */

    //virtual std::pair<lockholder_index,rwlock_token_set> get_locks_write_lockable(std::shared_ptr<mutableinfostore> infostore)=0;
    //virtual std::pair<lockholder_index,rwlock_token_set> get_locks_write_lockable(std::shared_ptr<geometry> geom)=0;
    //virtual std::pair<lockholder_index,rwlock_token_set> get_locks_write_lockable(std::shared_ptr<component> comp)=0;
    //virtual std::pair<lockholder_index,rwlock_token_set> get_locks_write_lockable(std::shared_ptr<parameterization> param)=0;
#ifdef SNDE_MUTABLE_RECDB_SUPPORT
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_write_lockable(std::shared_ptr<lockable_infostore_or_component> lic)=0;
#endif // SNDE_MUTABLE_RECDB_SUPPORT
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_write_array(void **array)=0;
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_write_array_region(void **array,snde_index indexstart,snde_index numelems)=0;
    virtual rwlock_token_set begin_temporary_locking(lockingposition startpos)=0; /* WARNING: Temporary locking only supported prior to all spawns!!! */

#ifdef SNDE_MUTABLE_RECDB_SUPPORT
    virtual rwlock_token_set get_locks_read_lockable_temporary(rwlock_token_set temporary_lock_pool,std::shared_ptr<lockable_infostore_or_component> lic)=0;
    virtual rwlock_token_set get_locks_write_lockable_temporary(rwlock_token_set temporary_lock_pool,std::shared_ptr<lockable_infostore_or_component> lic)=0;
    virtual rwlock_token_set get_locks_lockable_mask_temporary(rwlock_token_set temporary_lock_pool,std::shared_ptr<lockable_infostore_or_component> lic,uint64_t maskentry,uint64_t readmask,uint64_t writemask)=0;
#endif // SNDE_MUTABLE_RECDB_SUPPORT
    virtual void abort_temporary_locking(rwlock_token_set temporary_lock_pool)=0; /* WARNING: Temporary locking only supported prior to all spawns!!! */
    virtual rwlock_token_set finish_temporary_locking(lockingposition endpos,rwlock_token_set locks)=0; /* WARNING: Temporary locking only supported prior to all spawns!!! */

    //virtual std::pair<lockholder_index,rwlock_token_set> get_locks_read_lockable(std::shared_ptr<mutableinfostore> infostore)=0;
    //virtual std::pair<lockholder_index,rwlock_token_set> get_locks_read_lockable(std::shared_ptr<geometry> geom)=0;
    //virtual rwlock_token_set get_locks_read_lockable_temporary(std::shared_ptr<geometry> geom)=0;
    //virtual std::pair<lockholder_index,rwlock_token_set> get_locks_read_lockable(std::shared_ptr<component> comp)=0;
    //virtual std::pair<lockholder_index,rwlock_token_set> get_locks_read_lockable(std::shared_ptr<parameterization> param)=0;
#ifdef SNDE_MUTABLE_RECDB_SUPPORT
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_read_lockable(std::shared_ptr<lockable_infostore_or_component> lic)=0;
#endif // SNDE_MUTABLE_RECDB_SUPPORT
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_read_array(void **array)=0;
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_read_array_region(void **array,snde_index indexstart,snde_index numelems)=0;
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_array_region(void **array,bool write,snde_index indexstart,snde_index numelems)=0;
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_array(void **array,bool write)=0;
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_array_mask(void **array,uint64_t maskentry,uint64_t resizemaskentry,uint64_t readmask,uint64_t writemask,uint64_t resizemask,snde_index indexstart,snde_index numelems)=0;
    //virtual std::pair<lockholder_index,rwlock_token_set> get_locks_lockable_mask(std::shared_ptr<mutableinfostore> infostore,uint64_t maskentry,uint64_t readmask,uint64_t writemask)=0;
    //virtual std::pair<lockholder_index,rwlock_token_set> get_locks_lockable_mask(std::shared_ptr<geometry> geom,uint64_t maskentry,uint64_t readmask,uint64_t writemask)=0;
    //virtual std::pair<lockholder_index,rwlock_token_set> get_locks_lockable_mask(std::shared_ptr<component> comp,uint64_t maskentry,uint64_t readmask,uint64_t writemask)=0;
    //virtual std::pair<lockholder_index,rwlock_token_set> get_locks_lockable_mask(std::shared_ptr<parameterization> param,uint64_t maskentry,uint64_t readmask,uint64_t writemask)=0;
#ifdef SNDE_MUTABLE_RECDB_SUPPORT
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_lockable_mask(std::shared_ptr<lockable_infostore_or_component> lic,uint64_t maskentry,uint64_t readmask,uint64_t writemask)=0;
#endif // SNDE_MUTABLE_RECDB_SUPPORT
    virtual std::vector<std::tuple<lockholder_index,rwlock_token_set,std::string>> alloc_array_region(std::shared_ptr<arraymanager> manager,void **allocatedptr,snde_index nelem,std::string allocid)=0;
    virtual std::shared_ptr<lockingprocess_thread> spawn(std::function<void(void)> f)=0;
    //virtual rwlock_token_set lock_infostores(std::shared_ptr<mutablerecdb> recdb,std::set<std::string> channels_to_lock,bool write)=0; // moved to mutablerecstore.hpp

    virtual ~lockingprocess()=0;


    // NON-VIRTUAL templates:
    template <class T>
    rwlock_token_set lock_lockables(std::vector<std::shared_ptr<T>> lockables,bool write)
    {
      // sort lockables of the same type into proper locking order and uniqueify them
      // by putting them into a set
      //auto arrayidx = _arrayidx();
      rwlock_token_set new_locks=empty_rwlock_token_set();

      std::set<std::shared_ptr<T>,std::owner_less<std::shared_ptr<T>>> lockables_set(lockables.begin(),lockables.end());

      // now that we have the order figured out, perform the locking.
      
      for (auto & lockable: lockables_set) {
	if (write) {
	  lockholder_index index;
	  rwlock_token_set lockable_locks;
	  std::tie(index,lockable_locks)=get_locks_write_lockable(lockable);
	  merge_into_rwlock_token_set(new_locks,lockable_locks);
	} else {
	  lockholder_index index;
	  rwlock_token_set lockable_locks;
	  std::tie(index,lockable_locks)=get_locks_read_lockable(lockable);
	  merge_into_rwlock_token_set(new_locks,lockable_locks);
	  
	}
      }
      return new_locks;
    }


    
  };



  class lockingposition {
  public:
      // !!!*** For the moment this assumes all infostores
      // are in the same

      // only one of infostore, comp, param, and array_idx may be valid.
      // locking order goes: lic, including all infostores or components, ordered by std::owner_less on their weak pointers
      // all arrays (with sub-ordering of array segments
      bool initial_position; // if true this is the blank initial position in the locking order
      bool between_infostores_and_arrays; // if true this is the blank position between infostores and arrays
    //std::weak_ptr<mutableinfostore> infostore;
    //std::weak_ptr<geometry> geom;
    //std::weak_ptr<component> comp;
    //std::weak_ptr<parameterization> param;
#ifdef SNDE_MUTABLE_RECDB_SUPPORT
      std::weak_ptr<lockable_infostore_or_component> lic;
#endif // SNDE_MUTABLE_RECDB_SUPPORT
      lockindex_t array_idx; // -1 if invalid
      
      snde_index idx_in_array; /* index within array, or SNDE_INDEX_INVALID*/
      bool write; /* are we trying to lock for write? */

      
      lockingposition() :
	// blank lockingposition counts as before everything else
	initial_position(true),
	between_infostores_and_arrays(false),
	array_idx(-1),
	idx_in_array(SNDE_INDEX_INVALID),
	write(false)
      {
	
      }

    static lockingposition lockingposition_before_lic() 
	// since lic's are first in the order currently, we just return a blank lockingposition
    {
      return lockingposition();
    }

    static lockingposition lockingposition_before_arrays() 
    {
      lockingposition pos = lockingposition();
      pos.between_infostores_and_arrays=true;
      return pos;
    }

    
    
    lockingposition(lockindex_t array_idx,snde_index idx_in_array,bool write) :
	initial_position(false),
	between_infostores_and_arrays(false),
	array_idx(array_idx),
	idx_in_array(idx_in_array),
	write(write)
      {
	
      }
    
    /* lockingposition(std::weak_ptr<mutableinfostore> infostore,bool write) :
	initial_position(false),
	infostore(infostore),
	array_idx(-1),
	idx_in_array(SNDE_INDEX_INVALID),
	write(write)

      {

	std::weak_ptr<mutableinfostore> invalid_infostore;

	assert(infostore.owner_before(invalid_infostore) || invalid_infostore.owner_before(infostore)); // we should compare not-equal to the invalid_infostore
	
      }
    */
    /*
      lockingposition(std::weak_ptr<geometry> geom,bool write) :
	initial_position(false),
	geom(geom),
	array_idx(-1),
	idx_in_array(SNDE_INDEX_INVALID),
	write(write)

      {

	//std::weak_ptr<geometry> invalid_geom;

	assert(geom.owner_before(invalid_geom) || invalid_geom.owner_before(geom)); // we should compare not-equal to the invalid_geom
	
      }
    */

    /* lockingposition(std::weak_ptr<component> comp,bool write) :
	initial_position(false),
	comp(comp),
	array_idx(-1),
	idx_in_array(SNDE_INDEX_INVALID),
	write(write)

      {
	std::weak_ptr<component> invalid_comp;

	assert(comp.owner_before(invalid_comp) || invalid_comp.owner_before(comp)); // we should compare not-equal to the invalid_comp
	
	
      }
    */
    /*
      lockingposition(std::weak_ptr<parameterization> param,bool write) :
	initial_position(false),
	param(param),
	array_idx(-1),
	idx_in_array(SNDE_INDEX_INVALID),
	write(write)

      {
	std::weak_ptr<parameterization> invalid_param;

	assert(param.owner_before(invalid_param) || invalid_param.owner_before(param)); // we should compare not-equal to the invalid_param
	

      }
    */

#ifdef SNDE_MUTABLE_RECDB_SUPPORT
    lockingposition(std::weak_ptr<lockable_infostore_or_component> lic,bool write) :
	initial_position(false),
	between_infostores_and_arrays(false),
	lic(lic),
	array_idx(-1),
	idx_in_array(SNDE_INDEX_INVALID),
	write(write)

      {

	std::weak_ptr<lockable_infostore_or_component> invalid_lic;

	assert(lic.owner_before(invalid_lic) || invalid_lic.owner_before(lic)); // we should compare not-equal to the invalid_infostore
	
      }
#endif // SNDE_MUTABLE_RECDB_SUPPORT

      bool operator<(const lockingposition & other) const {
	// handle initial position case
	{
	  if (initial_position && !other.initial_position) {
	    return true; 
	  }
	  if (initial_position && other.initial_position) {
	    return false;
	  }
	  if (other.initial_position && !initial_position) {
	    return false; 
	  }
	}

	// from this point on, neither we nor other are in the initial position
	
	/*
	{
	  bool our_infostore_valid=false;
	  bool other_infostore_valid=false;
	  std::weak_ptr<mutableinfostore> invalid_infostore;
	  
	  if (infostore.owner_before(invalid_infostore) || invalid_infostore.owner_before(infostore)) {
	    // compares not equal to invalid_infostore... must be valid!
	    our_infostore_valid=true;
	  }
	  
	  if (other.infostore.owner_before(invalid_infostore) || invalid_infostore.owner_before(other.infostore)) {
	    // compares not equal to invalid_infostore... must be valid!
	    other_infostore_valid=true;
	  }
	  
	  if (our_infostore_valid && !other_infostore_valid) {
	    return true; 
	  }
	  if (!our_infostore_valid && other_infostore_valid) {
	    return false; 
	  }
	  
	  if (our_infostore_valid && other_infostore_valid) {
	    return infostore.owner_before(other.infostore);
	  }
	}
	*/
	// neither infostore is set... fall back to comparing geom
	/*
	  {
	  bool our_geom_valid=false;
	  bool other_geom_valid=false;
	  std::weak_ptr<geometry> invalid_geom;
	  
	  if (geom.owner_before(invalid_geom) || invalid_geom.owner_before(geom)) {
	    // compares not equal to invalid_geom... must be valid!
	    our_geom_valid=true;
	  }
	  
	  if (other.geom.owner_before(invalid_geom) || invalid_geom.owner_before(other.geom)) {
	    // compares not equal to invalid_geom... must be valid!
	    other_geom_valid=true;
	  }
	  
	  if (our_geom_valid && !other_geom_valid) {
	    return true; 
	  }
	  if (!our_geom_valid && other_geom_valid) {
	    return false; 
	  }
	  
	  if (our_geom_valid && other_geom_valid) {
	    return geom.owner_before(other.geom);
	  }
	}
	*/
	/*
	// neither geom is set... fall back to comparing component
	{
	  bool our_comp_valid=false;
	  bool other_comp_valid=false;
	  std::weak_ptr<component> invalid_comp;
	  
	  if (comp.owner_before(invalid_comp) || invalid_comp.owner_before(comp)) {
	    // compares not equal to invalid_comp... must be valid!
	    our_comp_valid=true;
	  }
	  
	  if (other.comp.owner_before(invalid_comp) || invalid_comp.owner_before(other.comp)) {
	    // compares not equal to invalid_comp... must be valid!
	    other_comp_valid=true;
	  }
	  
	  if (our_comp_valid && !other_comp_valid) {
	    return true; 
	  }
	  if (!our_comp_valid && other_comp_valid) {
	    return false; 
	  }
	  
	  if (our_comp_valid && other_comp_valid) {
	    return comp.owner_before(other.comp);
	  }
	}
	*/
	// neither component is set... fall back to comparing parameterization
	/*
	{
	  bool our_param_valid=false;
	  bool other_param_valid=false;
	  std::weak_ptr<parameterization> invalid_param;
	  
	  if (param.owner_before(invalid_param) || invalid_param.owner_before(param)) {
	    // compares not equal to invalid_param... must be valid!
	    our_param_valid=true;
	  }
	  
	  if (other.param.owner_before(invalid_param) || invalid_param.owner_before(other.param)) {
	    // compares not equal to invalid_comp... must be valid!
	    other_param_valid=true;
	  }
	  
	  if (our_param_valid && !other_param_valid) {
	    return true; 
	  }
	  if (!our_param_valid && other_param_valid) {
	    return false; 
	  }
	  
	  if (our_param_valid && other_param_valid) {
	    return param.owner_before(other.param);
	  }
	}
	*/
	// neither parameterization is set... fall back to comparing lockable_infostore_or_component

#ifdef SNDE_MUTABLE_RECDB_SUPPORT
	{
	  bool our_lic_valid=false;
	  bool other_lic_valid=false;
	  std::weak_ptr<lockable_infostore_or_component> invalid_lic;
	  
	  if (lic.owner_before(invalid_lic) || invalid_lic.owner_before(lic)) {
	    // compares not equal to invalid_infostore... must be valid!
	    our_lic_valid=true;
	  }
	  
	  if (other.lic.owner_before(invalid_lic) || invalid_lic.owner_before(other.lic)) {
	    // compares not equal to invalid_infostore... must be valid!
	    other_lic_valid=true;
	  }
	  
	  if (our_lic_valid && !other_lic_valid) {
	    return true; 
	  }
	  if (!our_lic_valid && other_lic_valid) {
	    return false; 
	  }
	  
	  if (our_lic_valid && other_lic_valid) {
	    return lic.owner_before(other.lic);
	  }
	}
#endif // SNDE_MUTABLE_RECDB_SUPPORT
	// neither lockable_infostore_or_component is set... fall back to comparing arrays

	// handle between_infostores_and_arrays case
	{
	  if (between_infostores_and_arrays && !other.between_infostores_and_arrays) {
	    return true; 
	  }
	  if (between_infostores_and_arrays && other.between_infostores_and_arrays) {
	    return false;
	  }
	  if (other.between_infostores_and_arrays && !between_infostores_and_arrays) {
	    return false; 
	  }
	}


	
	
	assert(array_idx >= 0);
	assert(other.array_idx >= 0);
	
	
        if (array_idx < other.array_idx) return true;
        if (array_idx > other.array_idx) return false;

        if (idx_in_array < other.idx_in_array) return true;
        if (idx_in_array > other.idx_in_array) return false;

        if (write && !other.write) return true;
        if (!write && other.write) return false;

        /* if we got here, everything is equal, i.e. not less than */
        return false;
      }
    };





#ifdef SNDE_LOCKMANAGER_COROUTINES_THREADED

  /* ***!!! Should create alternate implementation based on boost stackful coroutines ***!!! */
  /* ***!!! Should create alternate implementation based on C++ resumable functions proposal  */

  class lockingprocess_threaded_thread: public lockingprocess_thread {
  public:
    std::thread *thread; // Does not own thread. thread will be cleaned up when the locking process finish() is called. thread should be valid until then

    lockingprocess_threaded_thread(std::thread *thread) :
      thread(thread)
    {
      
    }

    ~lockingprocess_threaded_thread() {}
  };
  

  class lockingprocess_threaded: public lockingprocess {
    /* lockingprocess is a tool for performing multiple locking
       for multiple objects while satisfying the required
       locking order */
    
      /* (There was going to be an opencl_lockingprocess that was to be derived
         from this class, but it was cancelled) */
 
    lockingprocess_threaded(const lockingprocess_threaded &)=delete; /* copy constructor disabled */
    lockingprocess_threaded& operator=(const lockingprocess_threaded &)=delete; /* copy assignment disabled */
    
  public:
    //std::shared_ptr<arraymanager> _manager;
    std::shared_ptr<lockmanager> _lockmanager;
    std::mutex _mutex;
    //std::condition_variable _cv;
    std::multimap<lockingposition,std::condition_variable *> _waitingthreads; // locked by _mutex
    std::deque<std::condition_variable *> _runnablethreads; // locked by _mutex.... first entry is the running thread, which is listed as NULL (no cv needed because it is running).
    
    std::deque<std::thread *> _threadarray; // locked by _mutex

    /* ****!!!!!! Need to separate all_tokens into all_tokens and used_tokens, as in 
       the Python implementation in lockmanager.i. Also allow re-locking of stuff in all_tokens */
    rwlock_token_set all_tokens; /* these are all the tokens we have acquired along the way */
    rwlock_token_set used_tokens; /* these are all the tokens we are actually returning */
    
    //std::shared_ptr<std::vector<rangetracker<markedregion>>> arrayreadregions; /* indexed by arrayidx... 0th element empty */
    //std::shared_ptr<std::vector<rangetracker<markedregion>>> arraywriteregions; /* indexed by arrayidx... 0th element empty */

    
    lockingposition lastlockingposition; /* for diagnosing locking order violations */

    std::unique_lock<std::mutex> _executor_lock;
    
      /* The way this works is we spawn off parallel threads for
         each locking task. The individual threads can execute up
         until a lock attempt. All threads must have reached the
         lock attempt (or be finished) before any locking can
         occur, and then only the thread seeking the earliest lock
         (earliest in the locking order) may execute. That thread
         can then execute up until its next lock attempt and
         the process repeats.

         The mapping between locking positions and threads
         is stored in a std::multimap _waitingthreads (locked by _mutex).
         Each thread involved always either has an entry in _waitingthreads
         or is counted in _runnablethreads.

         To avoid synchronization hassles, only one thread can
         actually run at a time (locked by _mutex and managed by
         _executor_lock when running user code)

      */



    lockingprocess_threaded(std::shared_ptr<lockmanager> lockmanager);
    lockingprocess_threaded(std::shared_ptr<lockmanager> lockmanager,rwlock_token_set all_locks);
    
    virtual bool _barrier(lockingposition lockpos); //(lockindex_t idx,snde_index pos,bool write);

    virtual void *pre_callback();
    
    virtual void post_callback(void *state);

    virtual void *prelock();
    
    virtual void postunlock(void *prelockstate);

    //virtual std::pair<lockholder_index,rwlock_token_set>  get_locks_write_lockable(std::shared_ptr<mutableinfostore> infostore);
    //virtual std::pair<lockholder_index,rwlock_token_set>  get_locks_write_lockable(std::shared_ptr<geometry> geom);

    //virtual std::pair<lockholder_index,rwlock_token_set>  get_locks_write_lockable(std::shared_ptr<component> comp);

    //virtual std::pair<lockholder_index,rwlock_token_set>  get_locks_write_lockable(std::shared_ptr<parameterization> param);
#ifdef SNDE_MUTABLE_RECDB_SUPPORT
    virtual std::pair<lockholder_index,rwlock_token_set>  get_locks_write_lockable(std::shared_ptr<lockable_infostore_or_component> lic);
#endif // SNDE_MUTABLE_RECDB_SUPPORT

    virtual std::pair<lockholder_index,rwlock_token_set>  get_locks_write_array(void **array);

    virtual std::pair<lockholder_index,rwlock_token_set>  get_locks_write_array_region(void **array,snde_index indexstart,snde_index numelems);

    virtual rwlock_token_set begin_temporary_locking(lockingposition startpos); /* WARNING: Temporary locking only supported prior to all spawns!!! */
    
#ifdef SNDE_MUTABLE_RECDB_SUPPORT
    virtual rwlock_token_set get_locks_read_lockable_temporary(rwlock_token_set temporary_lock_pool,std::shared_ptr<lockable_infostore_or_component> lic);
    virtual rwlock_token_set get_locks_write_lockable_temporary(rwlock_token_set temporary_lock_pool,std::shared_ptr<lockable_infostore_or_component> lic);
    virtual rwlock_token_set get_locks_lockable_mask_temporary(rwlock_token_set temporary_lock_pool,std::shared_ptr<lockable_infostore_or_component> lic,uint64_t maskentry,uint64_t readmask,uint64_t writemask);
#endif // SNDE_MUTABLE_RECDB_SUPPORT
    virtual void abort_temporary_locking(rwlock_token_set temporary_lock_pool); /* WARNING: Temporary locking only supported prior to all spawns!!! */
    virtual rwlock_token_set finish_temporary_locking(lockingposition endpos,rwlock_token_set locks); /* WARNING: Temporary locking only supported prior to all spawns!!! */


    //virtual std::pair<lockholder_index,rwlock_token_set>  get_locks_read_lockable(std::shared_ptr<mutableinfostore> infostore);
    //virtual std::pair<lockholder_index,rwlock_token_set>  get_locks_read_lockable(std::shared_ptr<geometry> geom);
    //virtual rwlock_token_set get_locks_read_lockable_temporary(std::shared_ptr<geometry> geom);
    //virtual std::pair<lockholder_index,rwlock_token_set>  get_locks_read_lockable(std::shared_ptr<component> comp);
    //virtual std::pair<lockholder_index,rwlock_token_set>  get_locks_read_lockable(std::shared_ptr<parameterization> param);
#ifdef SNDE_MUTABLE_RECDB_SUPPORT
    virtual std::pair<lockholder_index,rwlock_token_set>  get_locks_read_lockable(std::shared_ptr<lockable_infostore_or_component> lic);
#endif // SNDE_MUTABLE_RECDB_SUPPORT
    
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_read_array(void **array);

    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_read_array_region(void **array,snde_index indexstart,snde_index numelems);

    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_array_region(void **array,bool write,snde_index indexstart,snde_index numelems);

    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_array(void **array,bool write);
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_array_mask(void **array,uint64_t maskentry,uint64_t resizemaskentry,uint64_t readmask,uint64_t writemask,uint64_t resizemask,snde_index indexstart,snde_index numelems);
    //virtual std::pair<lockholder_index,rwlock_token_set> get_locks_lockable_mask(std::shared_ptr<mutableinfostore> infostore,uint64_t maskentry,uint64_t readmask,uint64_t writemask);
    //virtual std::pair<lockholder_index,rwlock_token_set> get_locks_lockable_mask(std::shared_ptr<geometry> geom,uint64_t maskentry,uint64_t readmask,uint64_t writemask);
    //virtual std::pair<lockholder_index,rwlock_token_set> get_locks_lockable_mask(std::shared_ptr<component> comp,uint64_t maskentry,uint64_t readmask,uint64_t writemask);
    //virtual std::pair<lockholder_index,rwlock_token_set> get_locks_lockable_mask(std::shared_ptr<parameterization> param,uint64_t maskentry,uint64_t readmask,uint64_t writemask);
#ifdef SNDE_MUTABLE_RECDB_SUPPORT
    virtual std::pair<lockholder_index,rwlock_token_set> get_locks_lockable_mask(std::shared_ptr<lockable_infostore_or_component> lic,uint64_t maskentry,uint64_t readmask,uint64_t writemask);
#endif // SNDE_MUTABLE_RECDB_SUPPORT
    virtual std::vector<std::tuple<lockholder_index,rwlock_token_set,std::string>> alloc_array_region(std::shared_ptr<arraymanager> manager,void **allocatedptr,snde_index nelem,std::string allocid);

    virtual std::shared_ptr<lockingprocess_thread> spawn(std::function<void(void)> f);
      
    //virtual rwlock_token_set lock_infostores(std::shared_ptr<mutablerecdb> recdb,std::set<std::string> channels_to_lock,bool write); // moved to mutablerecstore.hpp

    virtual rwlock_token_set finish();
    virtual ~lockingprocess_threaded();

    };


#endif

  }




namespace snde {
  
  struct voidpp_string_hash {
    size_t operator()(const std::pair<void **,std::string> &x) const
    {
      void **param;
      std::string allocid;
      
      std::tie(param,allocid)=x;

      return std::hash<void *>{}((void *)param) + std::hash<std::string>{}(allocid);      
    }
  };


  class lockholder {
  public:
    std::unordered_map<lockholder_index,rwlock_token_set,lockholder_index_hash> values;
    std::unordered_map<std::pair<void **,std::string>,std::pair<snde_index,snde_index>,voidpp_string_hash> allocvalues;
    //                           ^ arrayptr ^ name               ^addr       ^nelem
    std::string as_string() {
      std::string ret="";

      ret+=ssprintf("snde::lockholder at 0x%llx with  %d allocations and %d locks\n",(unsigned long long)this,(int)allocvalues.size(),(int)values.size());
      ret+=ssprintf("---------------------------------------------------------------------------\n");
      for (auto & array_startidx : allocvalues) {
	ret+=ssprintf("allocation for array 0x%llx allocid=\"%s\" startidx=%lu\n",(unsigned long long)array_startidx.first.first,array_startidx.first.second.c_str(),(unsigned long)array_startidx.second.first);
      }
      for (auto & idx_tokens : values) {
	ret+=ssprintf("locks for array 0x%llx write=%s startidx=%lu numelem=%lu\n",(unsigned long long)idx_tokens.first.array,idx_tokens.first.write ? "true": "false",(unsigned long)idx_tokens.first.startidx,(unsigned long)idx_tokens.first.numelem);
      }
      return ret;
    }
    
    bool has_lock(void **array,bool write,snde_index indexstart,snde_index numelem)
    {
      return !(values.find(lockholder_index(array,write,indexstart,numelem))==values.end());
    }
    
    bool has_alloc(void **array,std::string allocid)
    {
      return !(allocvalues.find(std::make_pair(array,allocid))==allocvalues.end());
    }
    
    void store(void **array,bool write,snde_index indexstart,snde_index numelem,rwlock_token_set locktoken)
    {
      values[lockholder_index(array,write,indexstart,numelem)]=locktoken;
    }
    void store(lockholder_index array_write_startidx_numelem_tokens,rwlock_token_set locktoken)
    {
      
      values[array_write_startidx_numelem_tokens]=locktoken;
    }

    void store(std::pair<lockholder_index,rwlock_token_set> idx_locktoken)
    {
      lockholder_index array_write_startidx_numelem_tokens;
      rwlock_token_set locktoken;

      std::tie(array_write_startidx_numelem_tokens,locktoken)=idx_locktoken;
      store(array_write_startidx_numelem_tokens,locktoken);
    }

    //void store_name(std::string nameoverride,std::pair<std::string,rwlock_token_set> namevalue)
    //{
    //  values[nameoverride]=namevalue.second;
    //
    //}

    void store_alloc(void **array,bool write,snde_index startidx,snde_index numelem,rwlock_token_set tokens,std::string allocid)
    {
      
      values[lockholder_index(array,write,startidx,numelem)]=tokens;
      allocvalues[std::make_pair(array,allocid)]=std::make_pair(startidx,numelem);
    }
    void store_alloc(lockholder_index idx,rwlock_token_set tokens,std::string allocid)
    {
      store_alloc(idx.array,
		  idx.write,
		  idx.startidx,
		  idx.numelem,
		  tokens,
		  allocid);
    }
    void store_alloc(std::tuple<lockholder_index,rwlock_token_set,std::string> idx_tokens_allocid)
    {
      lockholder_index idx;
      rwlock_token_set tokens;
      std::string allocid;
      std::tie(idx,tokens,allocid)=idx_tokens_allocid;
      store_alloc(idx,tokens,allocid);
    }
    
    void store_alloc(std::vector<std::tuple<lockholder_index,rwlock_token_set,std::string>> vector_idx_tokens_allocid)
    {
      size_t cnt;
      for (cnt=0; cnt < vector_idx_tokens_allocid.size();cnt++) {
	store_alloc(vector_idx_tokens_allocid[cnt]);
      }
    }
    
    //void store_addr(void **array,std::pair<rwlock_token_set,snde_index> tokens_addr)
    //{
    //  store_addr(array,std::get<0>(tokens_addr),std::get<1>(tokens_addr));
    //}

    rwlock_token_set get(void **array,bool write,snde_index indexstart,snde_index numelem)
    {
      std::unordered_map<lockholder_index,rwlock_token_set,lockholder_index_hash>::iterator value=values.find(lockholder_index(array,write,indexstart,numelem));


      
      if (value==values.end()) {
	throw std::runtime_error("Specified array and region with given writable status not found in lockholder. Was it locked with the same parameters?");
      }
      return value->second;
    }

    template <class T>
    rwlock_token_set get(std::shared_ptr<T> lockable_ptr,bool write)
    {
      std::unordered_map<lockholder_index,rwlock_token_set,lockholder_index_hash>::iterator value=values.find(lockholder_index(lockable_ptr.get(),write));
      

      
      if (value==values.end()) {
	throw std::runtime_error("Specified lockable with given writable status not found in lockholder. Was it locked with the same parameters?");
      }
      return value->second;
    }


    rwlock_token_set get_alloc_lock(void **array,std::string allocid)
    {
      std::unordered_map<std::pair<void **,std::string>,std::pair<snde_index,snde_index>,voidpp_string_hash>::iterator allocvalue=allocvalues.find(std::make_pair(array,allocid));
      if (allocvalue==allocvalues.end()) {
	
	throw std::runtime_error("Specified array allocation and ID not found in lockholder. Are the array pointer and ID correct?");
      }
      return get(array,true,allocvalue->second.first,allocvalue->second.second);
    }

    snde_index get_alloc(void **array,std::string allocid)
    {
      std::unordered_map<std::pair<void **,std::string>,std::pair<snde_index,snde_index>,voidpp_string_hash>::iterator allocvalue=allocvalues.find(std::make_pair(array,allocid));
      if (allocvalue==allocvalues.end()) {
	throw std::runtime_error("Specified array allocation and ID not found in lockholder. Are the array pointer and ID correct?");
      }
      return allocvalue->second.first;
    }

    snde_index get_alloc_len(void **array,std::string allocid)
    {
      std::unordered_map<std::pair<void **,std::string>,std::pair<snde_index,snde_index>,voidpp_string_hash>::iterator allocvalue=allocvalues.find(std::make_pair(array,allocid));
      if (allocvalue==allocvalues.end()) {
	throw std::runtime_error("Specified array allocation and ID not found in lockholder. Are the array pointer and ID correct?");
      }
      return allocvalue->second.second;
    }

    //rwlock_token_set operator[](void ** array)
    //{
    //  return values.at(array);      
    //}

  };
  


  
}


#endif /* SNDE_LOCKMANAGER_HPP */
