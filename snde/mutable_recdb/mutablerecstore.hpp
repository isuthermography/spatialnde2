#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <functional>
#include <typeinfo>
#include <typeindex>
#include <complex>

#include "snde/gsl-lite.hpp"

#include "snde/geometry_types.h"
#include "snde/arraymanager.hpp"
#include "snde/metadata.hpp"
#include "snde/lockmanager.hpp"
#include "snde/geometry.hpp"
#include "snde/recdb_paths.hpp"


#ifndef SNDE_MUTABLERECSTORE_HPP
#define SNDE_MUTABLERECSTORE_HPP

namespace snde { 

#if defined(_WIN32) || defined(_WIN64)
#define mws_strtok strtok_s
#else
#define mws_strtok strtok_r
#endif


class mutableinfostore;
class mutabledatastore;


static inline snde_index total_numelements(std::vector<snde_index> shape) {
  snde_index numelements=1;

  for (size_t cnt=0; cnt < shape.size();cnt++) {
    numelements *= shape[cnt];
  }
  return numelements;
}



class mutableinfostore : public lockable_infostore_or_component  {
  /* NOTE: if you add more mutableinfostore subclasses, may need to add 
     handlers to OSGData::update() in openscenegraph_data.hpp ! */
public:
  // base class
  std::string leafname;
  std::string fullname; // including path, from base of recdb

  std::shared_ptr<rwlock> lock; // managed by lockmanager
  recmetadata metadata;
  std::shared_ptr<arraymanager> manager;



  
  // lock the infostore with manager->locker->lock_infostores()
  // *** MUST FOLLOW LOCKING ORDER DEFINED BY manager->locker. infostores should be locked
  // PRIOR TO locking data arrays.

  mutableinfostore(std::string leafname,std::string fullname,const recmetadata &metadata,std::shared_ptr<arraymanager> manager) :
    lockable_infostore_or_component(SNDE_INFOSTORE_INFOSTORES),
    leafname(leafname),
    fullname(fullname),
    metadata(metadata),
    manager(manager),
    lock(std::make_shared<rwlock>())
  {
    //manager->locker->addinfostore_rawptr(this);
  }

  // Disable copy constructor and copy assignment operators
  mutableinfostore(const mutableinfostore &)=delete; /* copy constructor disabled */
  mutableinfostore& operator=(const mutableinfostore &)=delete; /* assignment disabled */


  virtual std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>
  explore_component_get_instances(std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> &component_set,
				  std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,
				  snde_orientation3 orientation,
				  std::shared_ptr<immutable_metadata> metadata,
				  std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names)> get_uv_imagedata)
  {
    std::shared_ptr<lockable_infostore_or_component> our_ptr=shared_from_this();
    component_set.emplace(our_ptr);
    
    return std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>();

  }

  virtual void obtain_geom_lock(std::shared_ptr<lockingprocess> process, std::shared_ptr<iterablerecrefs> recdb_reclist=nullptr,std::string recdb_context="/",snde_infostore_lock_mask_t readmask=SNDE_COMPONENT_GEOM_ALL,snde_infostore_lock_mask_t writemask=0,snde_infostore_lock_mask_t resizemask=0)
  {
    // mutablerecstores don't have geometry -- well mutablegeomstores do, indirectly, but they
    // would have this called explicitly on the member geometry
  }
  virtual void obtain_uv_lock(std::shared_ptr<lockingprocess> process, std::shared_ptr<iterablerecrefs> recdb_reclist=nullptr,std::string recdb_context="/",snde_infostore_lock_mask_t readmask=SNDE_COMPONENT_GEOM_ALL,snde_infostore_lock_mask_t writemask=0,snde_infostore_lock_mask_t resizemask=0)
  {
    // mutablerecstores don't have geometry -- well mutablegeomstores do, indirectly, but they
    // would have this called explicitly on the member geometry

  }
  
  
  /*  virtual void obtain_lock(std::shared_ptr<lockingprocess> process, std::shared_ptr<mutablerecdb> recdb=null, std::string recdb_context="", snde_infostore_lock_mask_t readmask=SNDE_INFOSTORE_INFOSTORE, snde_infostore_lock_mask_t writemask=0, snde_infostore_lock_mask_t resizemask=0)
  {
    assert(readmask & SNDE_INFOSTORE_INFOSTORE || writemask & SNDE_INFOSTORE_INFOSTORE); // pointless if we aren't doing this

    process->get_locks_lockable_mask(shared_from_this(),SNDE_INFOSTORE_INFOSTORE,readmask,writemask);
    
  }
  */
  
  virtual ~mutableinfostore()
  {
    //manager->locker->reminfostore_rawptr(this);
    
  };
};

class recdirty_notification_receiver { // abstract base class
public:
  std::string recfullname; // name of the recording we will notify about

  recdirty_notification_receiver(const recdirty_notification_receiver &)=delete; // no copy constructor
  recdirty_notification_receiver & operator=(const recdirty_notification_receiver &)=delete; // no copy assignment
  
  recdirty_notification_receiver(std::string recfullname) :
    recfullname(recfullname)
  {
    
  }
  
  virtual void mark_as_dirty(std::shared_ptr<mutableinfostore> infostore)=0;  
  virtual ~recdirty_notification_receiver() {};
};
  

class iterablerecrefs : public std::enable_shared_from_this<iterablerecrefs> {
public:
  // This class represents a consistent set of data structures, both immutable,
  // that represent the current entires in the rec db.

  // it is a tree structure
  
  // Note that while these data structures are (generally) immutable,
  // what they point to is not!
  struct iterator {
    /* iterator that iterates over entire tree.
       (note that entire tree is immutable) */
    
    std::shared_ptr<iterablerecrefs> refs;  // should probably really be a weak_ptr
    
    std::vector<size_t> pos;


    std::string get_full_name();
    
    iterator resolve()
    // resolve a new or incremented iterator to its correct position or depth level
    {
      std::vector<size_t> newpos=pos;
      std::vector<std::shared_ptr<iterablerecrefs>>refsstack;
      refsstack.push_back(refs);
      size_t index=0;

      while (1) {
	std::shared_ptr<iterablerecrefs> thisrefs=refsstack[refsstack.size()-1];
	while (newpos[index] == thisrefs->recs.size()) {
	  // overflow of this postion
	  // go up one level and increment
	  newpos.pop_back();

	  if (index > 0) {
	    index--;
	    newpos[index]++;
	  } else {
	    // this is the end... return end marker (empty pos)
	    return iterator{refs,{}};
	  }
	}
	
	if (std::get<0>(thisrefs->recs[newpos[index]])) {
	  // got a mutableinfostore... this iterator directly points to an element.
	  // return it
	  assert(newpos.size()==index+1);
	  return iterator{refs,newpos};
	} else {
	  thisrefs = std::get<1>(thisrefs->recs[index]);
	  assert(thisrefs); // should always have an iterablerecrefs if we didn't have a mutableinfostore
	  index++;

	  if (newpos.size() < index) {
	    newpos.push_back(0);
	  }
	  
	}
      }
      return iterator{refs,newpos};
    }


    iterator operator++(int) // postfix++
    {
      iterator oldvalue=iterator{refs,pos};
      
      std::vector<size_t> newpos=pos;

      assert(newpos.size() > 0);
      newpos[newpos.size()-1]++;
      
      *this=iterator{refs,newpos}.resolve();
      
      return oldvalue;
      
    };

    iterator& operator+=(const size_t n)
    {
      for (size_t cnt=0;cnt < n; cnt++) {
	std::vector<size_t> newpos=pos;
	assert(newpos.size() > 0);
	newpos[newpos.size()-1]++;
	
	*this=iterator{refs,newpos}.resolve();
      }
    };

    iterator& operator--() // prefix--
    {
      std::vector<size_t> newpos=pos;
      assert(newpos.size() > 0);

      while (newpos[newpos.size()-1] == 0) {
	
      }
      newpos[newpos.size()-1]--;
	     
      *this=iterator{refs,newpos}.resolve();

      return *this;
    };


    std::shared_ptr<mutableinfostore> operator*() const {
      std::shared_ptr<iterablerecrefs> thisrefs=refs;
      size_t index=0;
      
      while (1) {
	if (index==pos.size()-1) {
	  // last index
	  std::shared_ptr<mutableinfostore> retval=std::get<0>(thisrefs->recs[pos[index]]);
	  // got a mutableinfostore... this iterator directly points to an element.
	  // return it
	  assert(retval);
	  return retval;
	} else {
	  
	  thisrefs = std::get<1>(thisrefs->recs[pos[index]]);
	  assert(thisrefs); // should always have an iterablerecrefs if we didn't have a mutableinfostore
	  index++;
	  
	}
      }
      
      
    }
    bool operator==(const iterator &it) const {
      if (refs==it.refs && pos==it.pos) {
	return true;
      }
      return false;
    }
    bool operator!=(const iterator &it) const {
      if (refs==it.refs && pos==it.pos) {
	return false;
      }
      return true;
    }

    
  };
  
  
  std::string leafname;  
  std::string fullname;  
  std::vector<std::tuple<std::shared_ptr<mutableinfostore>,std::shared_ptr<iterablerecrefs>>> recs; // recs by index... either the mutableinfostore or the iterablerecrefs should be non-nullptr
  std::unordered_map<std::string,size_t> indexbyname; // map by leafname of rec index

  iterablerecrefs() :
    leafname(""),
    fullname("")
  {
    
  }

  // Copy constructor that copies sub-tree
  iterablerecrefs(const iterablerecrefs &orig)
  {
    indexbyname=orig.indexbyname;

    for (auto & mutstore_immutrefs: orig.recs) {
      if (std::get<0>(mutstore_immutrefs)) { // mutableinfostore
	recs.push_back(mutstore_immutrefs);
      } else {
	std::shared_ptr<mutableinfostore> nullinfostore;
	recs.push_back(std::make_tuple(nullinfostore,std::make_shared<iterablerecrefs>(*std::get<1>(mutstore_immutrefs))));
      }
    }
  }

  // copy assignment operator that copies sub-tree
  iterablerecrefs & operator=(const iterablerecrefs & orig)
  {
    recs.empty();

    indexbyname=orig.indexbyname;

    for (auto & mutstore_immutrefs: orig.recs) {
      if (std::get<0>(mutstore_immutrefs)) { // mutableinfostore
	recs.push_back(mutstore_immutrefs);
      } else {
	std::shared_ptr<mutableinfostore> nullinfostore;
	recs.push_back(std::make_tuple(nullinfostore,std::make_shared<iterablerecrefs>(*std::get<1>(mutstore_immutrefs))));
      }
    }
    return *this;
  }
  
  std::shared_ptr<mutableinfostore> lookup(std::string Name)
  {
    // Name can be split into pieces by slashes to traverse into the tree

    //char *NameCopy=strdup(Name.c_str());
    //char *SavePtr=nullptr;

    
    //for (char *subname=mws_strtok(NameCopy,":",&SavePtr);subname;subname=mws_strtok(NULL,":",&SavePtr)) {
    //  
    //}
    //free(NameCopy);
    /**** !!!! Should enforce absolute lookups ***!!! (but we can only do that easily on recdb directly...)  */
    if (Name.at(0)=='/') { // skip leading slash
      Name=Name.substr(1);
    }

    
    auto index_it = indexbyname.find(Name);
    if (index_it != indexbyname.end()) {

      std::shared_ptr<mutableinfostore> retval;
      
      retval = std::get<0>(recs.at(index_it->second));
      assert(retval->leafname==Name);
      return retval;
    }

    size_t SlashPos;


    SlashPos=Name.find_first_of("/");
    if (SlashPos != std::string::npos) {
      std::string IterableName=Name.substr(0,SlashPos);
      std::string SubName=Name.substr(SlashPos+1);
      std::shared_ptr<iterablerecrefs> subtree;
      subtree = subtree->lookup_subtree(IterableName);

      return subtree->lookup(SubName);
    }
    
    return std::shared_ptr<mutableinfostore>(); // return empty pointer

  }

  std::shared_ptr<iterablerecrefs> lookup_subtree(std::string Name)
  {
    auto index_it = indexbyname.find(Name);
    if (index_it != indexbyname.end()) {
      std::shared_ptr<iterablerecrefs> retval;
      retval = std::get<1>(recs.at(index_it->second));
      assert(retval->leafname==Name);
      return retval;
    }

    size_t SlashPos = Name.find_first_of("/");
    if (SlashPos != std::string::npos) {
      std::string IterableName=Name.substr(0,SlashPos);
      std::string SubName=Name.substr(SlashPos+1);
      std::shared_ptr<iterablerecrefs> subtree;
      subtree = subtree->lookup_subtree(IterableName);

      return subtree->lookup_subtree(SubName);
    }
    
    return std::shared_ptr<iterablerecrefs>(); // return empty pointer
    
  }

  iterator begin()
  {
    return (iterator{ shared_from_this(), { 0 } }).resolve();
  }

  iterator end()
  {
    return iterator{ shared_from_this(), { } };
  }

  
};

  
class mutablerecdb {
public:

  
  std::mutex admin; // must be locked during changes to _reclist (replacement of C++11 atomic shared_ptr)... BEFORE the mutableinfostore rwlocks and BEFORE the various array data in the locking order. 
  
  std::shared_ptr<iterablerecrefs> _reclist; // an atomic shared pointer to an immutable reference list of mutable recordings
  std::shared_ptr<std::unordered_map<std::string,std::set<std::weak_ptr<recdirty_notification_receiver>,std::owner_less<std::weak_ptr<recdirty_notification_receiver>>>>> __dirtynotifies; // an atomic shared pointer to a map by recording name of interest of a set of notification receivers
  
  // To iterate over reclist,
  // first create a shared_ptr to <iterablerecrefs> by calling reclist(),
  // then iterate over that.

  mutablerecdb()
  {
    std::shared_ptr<iterablerecrefs> new_reclist=std::make_shared<iterablerecrefs>();
    std::shared_ptr<std::unordered_map<std::string,std::set<std::weak_ptr<recdirty_notification_receiver>,std::owner_less<std::weak_ptr<recdirty_notification_receiver>>>>> new__dirtynotifies=std::make_shared<std::unordered_map<std::string,std::set<std::weak_ptr<recdirty_notification_receiver>,std::owner_less<std::weak_ptr<recdirty_notification_receiver>>>>>();
      
    _end_atomic_reclist_update(new_reclist);
    _end_atomic__dirtynotifies_update(new__dirtynotifies);
  }
  
  
  virtual std::shared_ptr<iterablerecrefs> reclist()
  {
    return std::atomic_load(&_reclist);
  }

  std::shared_ptr<std::unordered_map<std::string,std::set<std::weak_ptr<recdirty_notification_receiver>,std::owner_less<std::weak_ptr<recdirty_notification_receiver>>>>>  _dirtynotifies()
  {
    return std::atomic_load(&__dirtynotifies);
  }

  virtual std::tuple<std::shared_ptr<iterablerecrefs>> _begin_atomic_reclist_update()
  // admin must be locked when calling this function...
  // it returns new copies of the atomically-guarded data
  {
    
      // Make copies of atomically-guarded data 
    std::shared_ptr<iterablerecrefs> new_reclist=std::make_shared<iterablerecrefs>(*reclist());
    
    return std::make_tuple(new_reclist);

  }

  virtual std::tuple<std::shared_ptr<std::unordered_map<std::string,std::set<std::weak_ptr<recdirty_notification_receiver>,std::owner_less<std::weak_ptr<recdirty_notification_receiver>>>>>> _begin_atomic__dirtynotifies_update()
  // admin must be locked when calling this function...
  // it returns new copies of the atomically-guarded data
  {
    
      // Make copies of atomically-guarded data 
    std::shared_ptr<std::unordered_map<std::string,std::set<std::weak_ptr<recdirty_notification_receiver>,std::owner_less<std::weak_ptr<recdirty_notification_receiver>>>>> new__dirtynotifies=std::make_shared<std::unordered_map<std::string,std::set<std::weak_ptr<recdirty_notification_receiver>,std::owner_less<std::weak_ptr<recdirty_notification_receiver>>>>>(*_dirtynotifies());

    
    return std::make_tuple(new__dirtynotifies);

  }

  virtual void _end_atomic_reclist_update(std::shared_ptr<iterablerecrefs> new_reclist)
  {
    std::atomic_store(&_reclist,new_reclist);
  }

  virtual void _end_atomic__dirtynotifies_update(std::shared_ptr<std::unordered_map<std::string,std::set<std::weak_ptr<recdirty_notification_receiver>,std::owner_less<std::weak_ptr<recdirty_notification_receiver>>>>> new__dirtynotifies)
  {
    std::atomic_store(&__dirtynotifies,new__dirtynotifies);
  }

  virtual std::shared_ptr<mutableinfostore> lookup(std::string Name)
  {
    std::shared_ptr<iterablerecrefs> refs = reclist();


    return refs->lookup(Name);
  }

  
  virtual void _rebuildnameindex(std::shared_ptr<iterablerecrefs> new_reclist)
  {
    /* ONLY OPERATE ON NEW UNASSIGNED reclist */
    new_reclist->indexbyname.clear();
    size_t pos;
    
    for (pos=0;pos < new_reclist->recs.size();pos++) {
      std::shared_ptr<mutableinfostore> infostore;
      std::shared_ptr<iterablerecrefs> recrefs;
      std::tie(infostore,recrefs)=new_reclist->recs[pos];

      if (infostore) {
	new_reclist->indexbyname.emplace(infostore->leafname,pos);
      } else {
	new_reclist->indexbyname.emplace(recrefs->leafname,pos);
      }
				       
    }
  }

  static std::shared_ptr<iterablerecrefs> _add_subtree_internal(std::shared_ptr<iterablerecrefs> new_reclist,std::string path,std::string subname)
  {
    std::shared_ptr<iterablerecrefs> new_subtree = std::make_shared<iterablerecrefs>();
    new_subtree->leafname = subname;
    new_subtree->fullname = path + subname;

    return new_subtree;
  }
  
  static int _addinfostore_internal(std::shared_ptr<iterablerecrefs> new_reclist,std::shared_ptr<mutableinfostore> infostore, std::string path,std::string name,bool make_subtrees)
  // returns non-zero for failure
  {
    size_t SlashPos=name.find_first_of("/");
    if (SlashPos != std::string::npos) {
      // add to a subtree
      std::string IterableName=name.substr(0,SlashPos);
      
      std::string SubName=name.substr(SlashPos+1);
      std::shared_ptr<iterablerecrefs> subtree;
      if (SlashPos==0) {
	// absolute reference to root
	assert(path=="");  // references to root from subtrees not currently supported
	return _addinfostore_internal(new_reclist,infostore,"/",SubName,make_subtrees);

      } else {
	subtree = subtree->lookup_subtree(IterableName);
	if (!subtree && make_subtrees) {
	  subtree = _add_subtree_internal(new_reclist,path,SubName);
	} else if (!subtree) {
	  return 1;
	}
      }
      return _addinfostore_internal(subtree,infostore,path+IterableName+"/",SubName,make_subtrees);
    }
    size_t num_recs=new_reclist->recs.size();
    
    assert(name==infostore->leafname);
      
    new_reclist->recs.push_back(std::make_tuple(infostore,std::shared_ptr<iterablerecrefs>(nullptr)));
    new_reclist->indexbyname.emplace(infostore->leafname,num_recs);
    return 0;
  }
  
  virtual int addinfostore(std::shared_ptr<mutableinfostore> infostore,bool make_subtrees=false)
  // returns non-zero for failure
  {
    std::lock_guard<std::mutex> adminlock(admin);
    int retval;
    
    std::shared_ptr<iterablerecrefs> new_reclist;
    
    std::tie(new_reclist)=_begin_atomic_reclist_update();
    // new_reclist is our own private copy so it is for the moment mutable
    retval = _addinfostore_internal(new_reclist,infostore,"", infostore->fullname,make_subtrees);
    
    if (!retval) {
      _end_atomic_reclist_update(new_reclist);
    }
    return retval;
  }

  virtual void add_dirty_notification_receiver(std::shared_ptr<recdirty_notification_receiver> rcvr)
  {
    std::lock_guard<std::mutex> adminlock(admin);
    std::shared_ptr<std::unordered_map<std::string,std::set<std::weak_ptr<recdirty_notification_receiver>,std::owner_less<std::weak_ptr<recdirty_notification_receiver>>>>> new__dirtynotifies;
    std::tie(new__dirtynotifies) = _begin_atomic__dirtynotifies_update();
    

    // recfullname member
    
    auto it = new__dirtynotifies->find(rcvr->recfullname);
    if (it == new__dirtynotifies->end()) {
      new__dirtynotifies->emplace(rcvr->recfullname,std::set<std::weak_ptr<recdirty_notification_receiver>,std::owner_less<std::weak_ptr<recdirty_notification_receiver>>>());
      it = new__dirtynotifies->find(rcvr->recfullname);
    } else {
      // already exists... filter out any dead pointers
      std::set<std::weak_ptr<recdirty_notification_receiver>,std::owner_less<std::weak_ptr<recdirty_notification_receiver>>>::iterator rcvr_it,nextrcvr_it;
      for (rcvr_it=it->second.begin(); rcvr_it != it->second.end(); rcvr_it=nextrcvr_it) {
	nextrcvr_it=rcvr_it;
	nextrcvr_it++;
	
	if (rcvr_it->expired()) {
	  it->second.erase(rcvr_it);
	}
	
      }
      
    }
    it->second.emplace(rcvr);
    
    _end_atomic__dirtynotifies_update(new__dirtynotifies);
    
  }

  virtual void remove_dirty_notification_receiver(std::weak_ptr<recdirty_notification_receiver> rcvr)
  {
    std::lock_guard<std::mutex> adminlock(admin);
    std::shared_ptr<std::unordered_map<std::string,std::set<std::weak_ptr<recdirty_notification_receiver>,std::owner_less<std::weak_ptr<recdirty_notification_receiver>>>>>  new__dirtynotifies;
    std::shared_ptr<recdirty_notification_receiver> rcvr_strong(rcvr);
    if (rcvr_strong) {
      std::tie(new__dirtynotifies) = _begin_atomic__dirtynotifies_update();


      auto iter = new__dirtynotifies->find(rcvr_strong->recfullname);
      
      assert(iter != new__dirtynotifies->end());
      
      new__dirtynotifies->erase(iter);
    
      _end_atomic__dirtynotifies_update(new__dirtynotifies);
    }
  }

  virtual void mark_as_dirty(std::shared_ptr<mutableinfostore> infostore)
  {
    // should be called while holding infostore's write lock
    // mark that the infostore metadata has changed
    // (changes to underlying data repositories are marked as dirty through
    // their corresponding arraymanager)
    
    std::shared_ptr<std::unordered_map<std::string,std::set<std::weak_ptr<recdirty_notification_receiver>,std::owner_less<std::weak_ptr<recdirty_notification_receiver>>>>> __dirtynotifies=_dirtynotifies();

    auto  it = __dirtynotifies->find(infostore->fullname);
    if (it != __dirtynotifies->end()) {
      for (auto & dirtynotify : it->second) {
	std::shared_ptr<recdirty_notification_receiver> notify=dirtynotify.lock();
	if (notify) { // pointer still exists
	  notify->mark_as_dirty(infostore); 
	}
      }
    }
  }


  rwlock_token_set lock_infostores(std::shared_ptr<lockmanager> lmanager,rwlock_token_set all_locks,std::shared_ptr<mutablerecdb> recdb,std::set<std::string> channels_to_lock,bool write)
{
  std::vector<std::shared_ptr<mutableinfostore>> infostores;
  
  for (auto & channame : channels_to_lock) {
    
    
    std::shared_ptr<mutableinfostore> infostore=lookup(channame);
    assert(infostore); // ***!!! Should probably throw an exception instead
    
    
    assert(infostore->manager->locker==lmanager); // all infostores must share same lock manager
    infostores.push_back(infostore);
  }
  
  return lmanager->lock_lockables(all_locks,infostores,write);
  
}


  rwlock_token_set lock_infostores(std::shared_ptr<lockingprocess> process, std::set<std::string> channels_to_lock,bool write)
  {
    std::vector<std::shared_ptr<mutableinfostore>> infostores;
  
    for (auto & channame : channels_to_lock) {
      
      
      std::shared_ptr<mutableinfostore> infostore=lookup(channame);
      assert(infostore); // ***!!! Should probably throw an exception instead
      
      
      //assert(infostore->manager->locker.get()==this); // all infostores must share same lock manager
      infostores.push_back(infostore);
      
    }
    
    
    return process->lock_lockables<mutableinfostore>(infostores,write);
  }
  
  
};



  // A renderable textured geometry contains three pieces:
  //  * The underlying geometry, represented by snde::component 
  //  * A surface parameterization for that component, represented by snde::parameterization
  //  * Image data corresponding to that surface parameterization, represented by
  //    snde_image in the geometry data structure (and owned by a mutablegeomstore or
  //    other structure)
  //  * How and from where that image data gets composited. 
  
class mutablegeomstore: public mutableinfostore
{
public:
  std::shared_ptr<geometry> geom;  // must not be changed after initialization
  std::shared_ptr<component> _comp; // atomic shared pointer
  
  //std::shared_ptr<paramdictentry> paramdict;
  //snde_index image; // address within uv_images field of geometry structure... represents our ownership of this image

  
  mutablegeomstore(std::string leafname,std::string fullname,const recmetadata &metadata,std::shared_ptr<geometry> geom,std::shared_ptr<component> comp) : //,std::shared_ptr<paramdictentry> paramdict) :
    mutableinfostore(leafname,fullname,metadata,geom->manager),
    geom(geom),
    _comp(comp)
    //paramdict(paramdict)
  {
    
  }

  
  std::shared_ptr<component> comp()
  {
    return std::atomic_load(&_comp);
  }

  // mutableparameterizationstore should be locked when calling this function 
  virtual void update_comp(std::shared_ptr<component> new_comp)
  {
    std::atomic_store(&_comp,new_comp);
  }
  

  virtual std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>
  explore_component_get_instances(std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> &component_set,
				  std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,
				  snde_orientation3 orientation,
				  std::shared_ptr<immutable_metadata> metadata,
				  std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names)> get_uv_imagedata)
  {
    std::shared_ptr<mutablegeomstore> our_ptr=std::dynamic_pointer_cast<mutablegeomstore>(shared_from_this());
    component_set.emplace(our_ptr);


    std::shared_ptr<immutable_metadata> merged_metadata = MergeMetadata(our_ptr->metadata.metadata(),metadata);
    
    std::shared_ptr<component> cur_comp=comp();
    if (cur_comp) {
      return cur_comp->explore_component_get_instances(component_set,recdb_reclist,recdb_context,orientation,
						       merged_metadata,get_uv_imagedata);
    }
    else {
      return std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>();
    }
    
  }

  static std::shared_ptr<mutablegeomstore> from_partlist(std::shared_ptr<mutablerecdb> recdb,std::string recdb_context,std::shared_ptr<geometry> geom,std::string assemname,std::shared_ptr<std::vector<std::string>> partnames)
  // NOTE: Does not add the new assembly mutablegeomstore to recdb -- you need to call recdb->addinfostore()
  // on the returned pointer 
  {
    std::shared_ptr<assembly> newassem = assembly::from_partlist(recdb,recdb_context,partnames);

    return std::make_shared<mutablegeomstore>(assemname,recdb_path_join(recdb_context,assemname),recmetadata(),geom,newassem);
  }

  static std::shared_ptr<mutablegeomstore> from_partlist(std::shared_ptr<mutablerecdb> recdb,std::string recdb_context,std::shared_ptr<geometry> geom,std::string assemname, std::shared_ptr<std::vector<std::shared_ptr<mutableinfostore>>> partlist)
  // NOTE: Does not add the new assembly mutablegeomstore to recdb -- you need to call recdb->addinfostore()
  // on the returned pointer
  // NOTE: Filters out any parameterizations or uv_imagedata in partlist, as they aren't part of the assembly --
  // they are included implicitly
  // so long as there is uv_imagedata referencing them
  {
    std::shared_ptr<std::vector<std::string>> partnames=std::make_shared<std::vector<std::string>>();

    for (auto & infostore: *partlist) {

      if (std::dynamic_pointer_cast<mutablegeomstore>(infostore)) {
	// must be a mutablegeomstore, i.e. referencing a component, to be part of the assembly
	partnames->emplace_back(*recdb_relative_path_to(recdb_context,infostore->fullname));
      }
    }
    
    std::shared_ptr<assembly> newassem = assembly::from_partlist(recdb,recdb_context,partnames);

    return std::make_shared<mutablegeomstore>(assemname,recdb_path_join(recdb_context,assemname),recmetadata(),geom,newassem);
  }

  /*  virtual void obtain_lock(std::shared_ptr<lockingprocess> process, std::shared_ptr<mutablerecdb> recdb=null, std::string recdb_context="",snde_infostore_lock_mask_t readmask=SNDE_INFOSTORE_INFOSTORE, snde_infostore_lock_mask_t writemask=0, snde_infostore_lock_mask_t resizemask=0)

  // ***!!!!! Please note: this does NOT lock the parameterization or any texture channels, or the 
 //    vertexarrays or texvertexarrays, or rgba data !!!*** 
  {
    // call superclass to lock the infostore itself
    mutableinfostore::obtain_lock(process,recdb,recdb_context,readmask,writemask,resizemask);
    
    rwlock_token_set temporary_object_trees_lock;
    
    if (readmask & SNDE_COMPONENT_GEOM_ALL || writemask & SNDE_COMPONENT_GEOM_ALL) {
      // if ANY geometry component is being locked for read OR write...
      assert(readmask & SNDE_INFOSTORE_COMPONENTS || writemask & SNDE_INFOSTORE_COMPONENTS); // we have to lock the components in order to lock the geometry
    }

    if (readmask & SNDE_INFOSTORE_OBJECT_TREES || writemask & SNDE_INFOSTORE_OBJECT_TREES) {
      process->get_locks_lockable_mask(shared_from_this(),SNDE_INFOSTORE_OBJECT_TREES,readmask,writemask);
    } else if (readmask & SNDE_INFOSTORE_COMPONENTS || writemask & SNDE_INFOSTORE_COMPONENTS) {
      // we need at least a temporary object_trees read lock in order to safely explore the components

      temporary_object_trees_lock=process->get_locks_read_lockable_temporary(geom);
    }

    if (readmask & SNDE_INFOSTORE_COMPONENTS || writemask & SNDE_INFOSTORE_COMPONENTS) {

      // lock all components of this geom despite arbitrary locking order
      
      comp->obtain_lock(process,recdb,recdb_context,readmask & (~SNDE_COMPONENT_GEOM_ALL),writemask & (~SNDE_COMPONENT_GEOM_ALL));
    }
    
    if (readmask & SNDE_COMPONENT_GEOM_ALL || writemask & SNDE_COMPONENT_GEOM_ALL) {
      comp->obtain_geom_lock(process,readmask,writemask,resizemask);
    }

    //if (readmask & )
    //  param_ptr->obtain_uv_lock(lockprocess,SNDE_UV_GEOM_UVS|SNDE_UV_GEOM_INPLANE2UVCOORDS|SNDE_UV_GEOM_UVCOORDS2INPLANE);

    */ 

};



class mutableparameterizationstore: public mutableinfostore
{
public:
  std::shared_ptr<geometry> geom;
  std::shared_ptr<parameterization> _param;  // atomic shared pointer
  //std::shared_ptr<paramdictentry> paramdict;
  //snde_index image; // address within uv_images field of geometry structure... represents our ownership of this image

  
  mutableparameterizationstore(std::string leafname,std::string fullname,const recmetadata &metadata,std::shared_ptr<geometry> geom,std::shared_ptr<parameterization> param) : //,std::shared_ptr<paramdictentry> paramdict) :
    mutableinfostore(leafname,fullname,metadata,geom->manager),
    geom(geom),
    _param(param)
    //paramdict(paramdict)
  {
    
  }


  std::shared_ptr<parameterization> param()
  {
    return std::atomic_load(&_param);
  }

  // mutableparameterizationstore should be locked when calling this function 
  virtual void update_param(std::shared_ptr<parameterization> new_param)
  {
    std::atomic_store(&_param,new_param);
  }
  
  virtual std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>
  explore_component_get_instances(std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> &component_set,
				  std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,
				  snde_orientation3 orientation,
				  std::shared_ptr<immutable_metadata> metadata,
				  std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names)> get_uv_imagedata)
  {
    std::shared_ptr<mutableparameterizationstore> our_ptr=std::dynamic_pointer_cast<mutableparameterizationstore>(shared_from_this());
    component_set.emplace(our_ptr);

    std::shared_ptr<parameterization> param_ptr = param();  
    std::shared_ptr<immutable_metadata> merged_metadata = MergeMetadata(our_ptr->metadata.metadata(),metadata);

    if (param_ptr) {
      return param_ptr->explore_component_get_instances(component_set,recdb_reclist,recdb_context,orientation,
							merged_metadata,get_uv_imagedata);
    }
    else {
      return std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>();
    }
    
  }


};

  

  
class mutabledatastore: public mutableinfostore {
public:
  // base class for data element stores. 

  void **basearray;  // *basearray is locked by the whole array lock
  unsigned typenum;  // typenum and elementsize may not be changed after the mutabledatastore is created
  size_t elementsize;

  // Exactly one of these (basearray_holder or basearray_owner)
  // should be non-nullptr. If basearray_holder is non-null, then
  // this class owns the data array and basearray should
  // point to basearray_holder. If basearray_owner is non-null
  // then some other class owns it and basearray should point into
  // that class. basearray_owner will keep the owner in memory as
  // long as this mutabledatastore remains in existance.

  // basearray_holder is locked with the whole array data
  void *basearray_holder;
  std::shared_ptr<void> basearray_owner; // locked with whole array data and may be used by (implicit) destructor

  // These remaining fields locked with metadata
  snde_index startelement;
  snde_index numelements;

  std::vector<snde_index> dimlen; // multidimensional shape...
  std::vector<snde_index> strides; // stride for each dimension... see numpy manual for detailed discussion

  
  mutabledatastore(std::string leafname,std::string fullname,
		   const recmetadata &metadata,
		   std::shared_ptr<arraymanager> manager,
		   void **basearray,unsigned typenum,size_t elementsize,
		   void *basearray_holder,
		   std::shared_ptr<void> basearray_owner,  
		   snde_index startelement,
		   snde_index numelements,
		   std::vector<snde_index> dimlen,
		   std::vector<snde_index> strides) :
    mutableinfostore(leafname,fullname,metadata,manager),
    basearray(basearray),
    typenum(typenum),
    elementsize(elementsize),
    basearray_holder(basearray_holder),
    basearray_owner(basearray_owner),
    startelement(startelement),
    numelements(numelements),
    dimlen(dimlen),
    strides(strides)
  {
    
  }
  virtual ~mutabledatastore() {}

  // Disable copy constructor and copy assignment operators
  mutabledatastore(const mutabledatastore &)=delete; /* copy constructor disabled */
  mutabledatastore& operator=(const mutabledatastore &)=delete; /* assignment disabled */

  virtual double element_double(snde_index idx)=0; // WARNING: array should generally be locked for read when calling this function!

  virtual std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>
  explore_component_get_instances(std::set<std::shared_ptr<lockable_infostore_or_component>,std::owner_less<std::shared_ptr<lockable_infostore_or_component>>> &component_set,
				  std::shared_ptr<iterablerecrefs> recdb_reclist,std::string recdb_context,
				  snde_orientation3 orientation,
				  std::shared_ptr<immutable_metadata> metadata,
				  std::function<std::tuple<std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>(std::shared_ptr<iterablerecrefs> recdb_reclist,std::shared_ptr<part> partdata,std::vector<std::string> uv_imagedata_names)> get_uv_imagedata)
  {
    std::shared_ptr<mutabledatastore> our_ptr=std::dynamic_pointer_cast<mutabledatastore>(shared_from_this());
    component_set.emplace(our_ptr);

    // datastores may have uv_parameterization metadata specifying which parameterization to use... in which case
    // that parameterization needs to be explored
    std::shared_ptr<immutable_metadata> merged_metadata = MergeMetadata(our_ptr->metadata.metadata(),metadata);

    std::string uv_parameterization=merged_metadata->GetMetaDatumStr("uv_parameterization","");
    std::shared_ptr<lockable_infostore_or_component> parameterization_ioc;
    std::shared_ptr<mutableparameterizationstore> parameterization_infostore;

    fprintf(stderr,"mutabledatastore::explore_component_get_instances: uv_parameterization=\"%s\"\n",uv_parameterization.c_str());
    
    if (uv_parameterization.size() > 0
	&& (parameterization_ioc=recdb_reclist->lookup(recdb_path_join(recdb_context,uv_parameterization)))
	&& (parameterization_infostore=std::dynamic_pointer_cast<mutableparameterizationstore>(parameterization_ioc))) {

      fprintf(stderr,"mutabledatastore::explore_component_get_instances: exploring uv_parameterization\n");
      
      return parameterization_infostore->explore_component_get_instances(component_set,recdb_reclist,recdb_context,orientation,
									 merged_metadata,get_uv_imagedata);
    }
    else {
      return std::vector<std::tuple<snde_partinstance,std::shared_ptr<part>,std::shared_ptr<parameterization>,std::map<snde_index,std::tuple<std::shared_ptr<mutabledatastore>,std::shared_ptr<image_data>>>>>();
    }
  }

  
  void *void_dataptr()
  {
    return (void *)(((char *)(*basearray)) + startelement*elementsize);
  }
  

  
};



// #defines for typenum
#define MET_FLOAT 0
#define MET_DOUBLE 1
#define MET_HALFFLOAT 2
#define MET_UINT64 3
#define MET_INT64 4
#define MET_UINT32 5
#define MET_INT32 6
#define MET_UINT16 7
#define MET_INT16 8
#define MET_UINT8 9
#define MET_INT8 10
#define MET_RGBA32 11 /* R stored in lowest address... Like OpenGL with GL_RGBA and GL_UNSIGNED_BYTE, or snde_rgba type */ 
#define MET_COMPLEXFLOAT 12
#define MET_COMPLEXDOUBLE 13

// met_typemap is indexed by typeid(type)
static const std::unordered_map<std::type_index,unsigned> met_typemap({
								       {typeid(float), MET_FLOAT},
								       {typeid(double), MET_DOUBLE},
								       // half-precision not generally
								       // available
								       //{typeid(__fp16), MET_HALFFLOAT},
								       {typeid(uint64_t),MET_UINT64},
								       {typeid(int64_t),MET_INT64},
								       {typeid(uint32_t),MET_UINT32},
								       {typeid(int32_t),MET_INT32},
								       {typeid(uint16_t),MET_UINT16},
								       {typeid(int16_t),MET_INT16},
								       {typeid(uint8_t),MET_UINT8},
								       {typeid(int8_t),MET_INT8},
								       {typeid(snde_rgba),MET_RGBA32},
								       {typeid(std::complex<float>),MET_COMPLEXFLOAT},
								       {typeid(std::complex<double>),MET_COMPLEXDOUBLE},
								       
  });

// met_typesizemap is indexed by MET_xxx
static const std::unordered_map<unsigned,size_t> met_typesizemap({
							   {MET_FLOAT,sizeof(float)},
							   {MET_DOUBLE,sizeof(double)},
							   // half-precision not generally
							   // available
							   //{MET_HALFFLOAT,sizeof(__fp16)},
							   {MET_UINT64,sizeof(uint64_t)},
							   {MET_INT64,sizeof(int64_t)},
							   {MET_UINT32,sizeof(uint32_t)},
							   {MET_INT32,sizeof(int32_t)},
							   {MET_UINT16,sizeof(uint16_t)},
							   {MET_INT16,sizeof(int16_t)},
							   {MET_UINT8,sizeof(uint8_t)},
							   {MET_INT8,sizeof(int8_t)},
							   {MET_RGBA32,sizeof(snde_rgba)},
							   {MET_COMPLEXFLOAT,sizeof(std::complex<float>)},
							   {MET_COMPLEXDOUBLE,sizeof(std::complex<double>)},
  });


static const std::unordered_map<unsigned,std::string> met_ocltypemap({
							   {MET_FLOAT,"float"},
							   {MET_DOUBLE,"double"},
							   // half-precision not generally
							   // available
							   {MET_HALFFLOAT,"half"},
							   {MET_UINT64,"unsigned long"},
							   {MET_INT64,"long"},
							   {MET_UINT32,"unsigned int"},
							   {MET_INT32,"int"},
							   {MET_UINT16,"unsigned short"},
							   {MET_INT16,"short"},
							   {MET_UINT8,"unsigned char"},
							   {MET_INT8,"char"},
							   {MET_RGBA32,"snde_rgba"},
							   {MET_COMPLEXFLOAT,"struct { float real; float imag; }"},
							   {MET_COMPLEXDOUBLE,"struct { double real; double imag; }"},
							   
  });


// see https://stackoverflow.com/questions/38644146/choose-template-based-on-run-time-string-in-c


template <class T>
class mutableelementstore : public mutabledatastore {
public:
  typedef T dtype;

  mutableelementstore(std::string name,const recmetadata &metadata,std::shared_ptr<arraymanager> manager,void **basearray,std::shared_ptr<void> basearray_owner,snde_index startelement,snde_index numelements) :
    /* In this case we assume an already-managed array, with an external owner */
    mutabledatastore(name,metadata,manager,
		     basearray,met_typemap.at(typeid(T)),sizeof(T),
		     nullptr,
		     basearray_owner,
		     startelement,
		     numelements,
		     std::vector<snde_index>(),
		     std::vector<snde_index>())  
  {
    dimlen.push_back(numelements);
    strides.push_back(1);
  }

  mutableelementstore(std::string name,const recmetadata &metadata,std::shared_ptr<arraymanager> manager,void **basearray,std::shared_ptr<void> basearray_owner,snde_index startelement,const std::vector<snde_index> &dimlen,const std::vector<snde_index> &strides) :
    /* In this case we assume an already-managed array, with an external owner */
    mutabledatastore(name,metadata,manager,
		     basearray,met_typemap.at(typeid(T)),sizeof(T),
		     nullptr,
		     basearray_owner,
		     startelement,
		     0,
		     std::vector<snde_index>(),
		     std::vector<snde_index>())
  {
    size_t dimcnt;

    this->dimlen=dimlen;
    this->strides=strides;
    
    numelements=1;
    for (dimcnt=0;dimcnt < dimlen.size();dimcnt++) {
      numelements *= dimlen[dimcnt];
    }
  }
    
  mutableelementstore(std::string name,const recmetadata & metadata, std::shared_ptr<arraymanager> manager,size_t numelements) :
    /* In this case we create a new array of the given size */
    mutabledatastore(name,metadata,manager,
		     &basearray_holder,met_typemap.at(typeid(T)),sizeof(T),
		     calloc(1,numelements*sizeof(T)),
		     nullptr,
		     0,
		     numelements,
		     std::vector<snde_index>(),
		     std::vector<snde_index>())
  {
    dimlen.push_back(numelements);
    strides.push_back(1);
    manager->add_unmanaged_array(basearray,sizeof(T),numelements);
    
  }

  mutableelementstore(std::string leafname,std::string fullname,const recmetadata &metadata,std::shared_ptr<arraymanager> manager,const std::vector<snde_index> &dimlen,const std::vector<snde_index> &strides) :
    /* In this case we create a new array of the given size and shape with given strides*/
    mutabledatastore(leafname,fullname,metadata,manager,
		     &basearray_holder,met_typemap.at(typeid(T)),sizeof(T),
		     calloc(total_numelements(dimlen),sizeof(T)),
		     nullptr,
		     0,
		     total_numelements(dimlen),
		     std::vector<snde_index>(),
		     std::vector<snde_index>())
  {
    this->dimlen=dimlen;
    this->strides=strides;

    manager->add_unmanaged_array(basearray,sizeof(T),total_numelements(dimlen));
    
  }

  
  T *dataptr()
  {
    return ((T *)(*basearray)) + startelement; 
  }

  T element(snde_index idx)
  {
    return dataspan()[idx];
  }

  virtual double element_double(snde_index idx){
    return (double)element(idx);
  }
    
  gsl::span<T> dataspan()
  {
    return gsl::span<T>(dataptr(),numelements);
  }


  ~mutableelementstore()
  {

    if (basearray) {
      manager->remove_unmanaged_array(basearray);
    }
  }
};

}

#endif // SNDE_MUTABLERECSTORE_HPP
