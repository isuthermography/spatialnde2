#include <string>
#include <mutex>
#include <unordered_map>

#ifndef SNDE_METADATA_HPP
#define SNDE_METADATA_HPP

namespace snde {

#define MWS_MDT_INT 0
#define MWS_MDT_STR 1
#define MWS_MDT_DBL 2
#define MWS_MDT_UNSIGNED 3
#define MWS_MDT_NONE 4

class metadatum {
public:
  std::string Name;
  
  int64_t intval;
  uint64_t unsignedval;
  std::string strval;
  double dblval;

  unsigned md_type; /* MWS_MDT_... */

  metadatum() :  // invalid, empty metadatum
    Name(""),
    intval(0),
    md_type(MWS_MDT_NONE)
  {

  }

  metadatum(std::string Name,const metadatum &oldmd) :
    Name(Name),
    intval(oldmd.intval),
    unsignedval(oldmd.unsignedval),
    strval(oldmd.strval),
    dblval(oldmd.dblval),
    md_type(oldmd.md_type)
    // copy from pre-existing metadatum
  {
    
  }

  metadatum(std::string Name,int64_t intval) :
    Name(Name),
    intval(intval),
    md_type(MWS_MDT_INT)
  {
    
  }

  
  metadatum(std::string Name,std::string strval) :
    Name(Name),
    strval(strval),
    md_type(MWS_MDT_STR)
  {
    
  }
  
  metadatum(std::string Name,double dblval) :
    Name(Name),
    dblval(dblval),
    md_type(MWS_MDT_DBL)
  {
    
  }

  metadatum(std::string Name,uint64_t unsignedval) :
    Name(Name),
    unsignedval(unsignedval),
    md_type(MWS_MDT_UNSIGNED)
  {
    
  }

  //#if (sizeof(snde_index) != sizeof(uint64_t))
#if (SIZEOF_SNDE_INDEX != 8)
  metadatum(std::string Name,snde_index indexval) :
    Name(Name),
    md_type(MWS_MDT_UNSIGNED)
  {
    assert(indexval != SNDE_INDEX_INVALID); // have not (yet) defined an invalid value 
    unsignedval=indexval;
  }
  
#endif
  
  int64_t Int(int64_t defaultval)
  {
    if (md_type != MWS_MDT_INT) {
      return defaultval;
    }
    return intval;
  }

  int64_t Unsigned(uint64_t defaultval)
  {
    if (md_type != MWS_MDT_UNSIGNED) {
      return defaultval;
    }
    return unsignedval;
  }

  std::string Str(std::string defaultval)
  {
    if (md_type != MWS_MDT_STR) {
      return defaultval;
    }
    return strval;
  }
  double Dbl(double defaultval)
  {
    if (md_type != MWS_MDT_DBL) {
      return defaultval;
    }
    return dblval;
  }

  double Numeric(double defaultval)
  {
    if (md_type == MWS_MDT_DBL) {
      return dblval;
    } else if (md_type == MWS_MDT_INT) {
      return (double)intval;
    } else if (md_type == MWS_MDT_UNSIGNED) {
      return (double)unsignedval;
    } else {
      return defaultval;
    }
  }

  snde_index Index(snde_index defaultval)
  {
    if (md_type == MWS_MDT_INT) {
      if (intval >= 0) return intval;
      else return SNDE_INDEX_INVALID;
    } else if (md_type == MWS_MDT_UNSIGNED) {
      return unsignedval;
    } else {
      return defaultval;
    }
  }
  
};

class wfmmetadata {
public:
  std::shared_ptr<std::unordered_map<std::string,metadatum>> _metadata; // c++11 atomic shared pointer to metadata map
  std::mutex admin; // must be locked during changes to _wfmlist (replacement of C++11 atomic shared_ptr)
  
  wfmmetadata()
    
  {
    std::shared_ptr<std::unordered_map<std::string,metadatum>> new_metadata;
    new_metadata=std::make_shared<std::unordered_map<std::string,metadatum>>();
    
    _end_atomic_update(new_metadata);
  }

  
  // thread-safe copy constructor and copy assignment operators
  wfmmetadata(const wfmmetadata &orig) /* copy constructor  */
  {
    std::shared_ptr<std::unordered_map<std::string,metadatum>> new_metadata;
    new_metadata=std::make_shared<std::unordered_map<std::string,metadatum>>(*orig.metadata());

    _end_atomic_update(new_metadata);    
  }

  // copy assignment operator
  wfmmetadata& operator=(const wfmmetadata & orig)
  {
    std::lock_guard<std::mutex> adminlock(admin);
    std::shared_ptr<std::unordered_map<std::string,metadatum>> new_metadata=std::make_shared<std::unordered_map<std::string,metadatum>>(*orig.metadata());
    _end_atomic_update(new_metadata);

    return *this;
  }
  
  // accessor method for metadata map
  std::shared_ptr<std::unordered_map<std::string,metadatum>> metadata() const
  {
    // read atomic shared pointer
    return std::atomic_load(&_metadata);
  }

  std::tuple<std::shared_ptr<std::unordered_map<std::string,metadatum>>> _begin_atomic_update()
  // admin must be locked when calling this function...
  // it returns new copies of the atomically-guarded data
  {
    
    // Make copies of atomically-guarded data 
    std::shared_ptr<std::unordered_map<std::string,metadatum>> new_metadata=std::make_shared<std::unordered_map<std::string,metadatum>>(*metadata());
    
    return std::make_tuple(new_metadata);

  }

  void _end_atomic_update(std::shared_ptr<std::unordered_map<std::string,metadatum>> new_metadata)
  {
    std::atomic_store(&_metadata,new_metadata);
  }


  int64_t GetMetaDatumInt(std::string Name,int64_t defaultval)
  {
    std::shared_ptr<std::unordered_map<std::string,metadatum>> md=metadata();
    std::unordered_map<std::string,metadatum>::iterator mditer; 

    mditer = md->find(Name);
    if (mditer == md->end() || mditer->second.md_type != MWS_MDT_INT) {
      return defaultval;
    }
    return (*mditer).second.Int(defaultval);
  }

  std::string GetMetaDatumStr(std::string Name,std::string defaultval)
  {
    std::shared_ptr<std::unordered_map<std::string,metadatum>> md=metadata();
    std::unordered_map<std::string,metadatum>::iterator mditer; 

    mditer = md->find(Name);
    if (mditer == md->end() || mditer->second.md_type != MWS_MDT_STR) {
      return defaultval;
    }
    return (*mditer).second.Str(defaultval);
  }

  double GetMetaDatumDbl(std::string Name,double defaultval)
  {
    std::shared_ptr<std::unordered_map<std::string,metadatum>> md=metadata();
    std::unordered_map<std::string,metadatum>::iterator mditer; 

    mditer = md->find(Name);
    if (mditer == md->end() || mditer->second.md_type != MWS_MDT_DBL) {
      return defaultval;
    }
    return (*mditer).second.Dbl(defaultval);
  }

  void AddMetaDatum(metadatum newdatum)
  // Add or update an entry 
  {
    std::lock_guard<std::mutex> adminlock(admin);
    std::shared_ptr<std::unordered_map<std::string,metadatum>> new_metadata;
    
    std::tie(new_metadata) = _begin_atomic_update();
    
    (*new_metadata)[newdatum.Name]=newdatum;
    
    _end_atomic_update(new_metadata);
    
  }


};


};


#endif // SNDE_METADATA_HPP
