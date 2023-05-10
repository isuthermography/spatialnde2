#ifndef SNDE_METADATA_HPP
#define SNDE_METADATA_HPP

#include <string>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <atomic>

#include "snde/snde_error.hpp"

namespace snde {

#define MWS_MDT_INT 0
#define MWS_MDT_STR 1
#define MWS_MDT_DBL 2
#define MWS_MDT_DBL_UNITS 3
#define MWS_MDT_UNSIGNED 4
#define MWS_MDT_BOOL 5
#define MWS_MDT_NONE 6

#define MWS_UNSIGNED_INVALID (~((uint64_t)0))
  
class metadatum {
public:
  std::string Name;
  
  int64_t intval;
  uint64_t unsignedval;
  std::string strval;
  double dblval;
  std::string dblunits;
  bool boolval;

  unsigned md_type; /* MWS_MDT_... */

  metadatum() :  // invalid, empty metadatum
    Name(""),
    intval(0),
    unsignedval(0),
    strval(""),
    dblval(0.0),
    dblunits(""),
    boolval(false),
    md_type(MWS_MDT_NONE)
  {

  }

  metadatum(std::string Name,const metadatum &oldmd) :
    Name(Name),
    intval(oldmd.intval),
    unsignedval(oldmd.unsignedval),
    strval(oldmd.strval),
    dblval(oldmd.dblval),
    dblunits(oldmd.dblunits),
    boolval(oldmd.boolval),
    md_type(oldmd.md_type)
    // copy from pre-existing metadatum
  {
    
  }

  metadatum(std::string Name,int64_t intval) :
    Name(Name),
    intval(intval),
    unsignedval(0),
    strval(""),
    dblval(0.0),
    dblunits(""),
    boolval(false),
    md_type(MWS_MDT_INT)
  {
    
  }

  
  metadatum(std::string Name,std::string strval) :
    Name(Name),
    intval(0),
    unsignedval(0),
    strval(strval),
    dblval(0.0),
    dblunits(""),
    boolval(false),
    md_type(MWS_MDT_STR)
  {
    
  }
  
  metadatum(std::string Name,double dblval) :
    Name(Name),
    intval(0),
    unsignedval(0),
    strval(""),
    dblval(dblval),
    dblunits(""),
    boolval(false),
    md_type(MWS_MDT_DBL)
  {
    
  }

  metadatum(std::string Name,double dblval,std::string units) :
    Name(Name),
    intval(0),
    unsignedval(0),
    strval(""),
    dblval(dblval),
    dblunits(units),
    boolval(false),
    md_type(MWS_MDT_DBL_UNITS)
  {
    
  }

  
  metadatum(std::string Name,uint64_t unsignedval) :
    Name(Name),
    intval(0),
    unsignedval(unsignedval),
    strval(""),
    dblval(0.0),
    dblunits(""),
    boolval(false),
    md_type(MWS_MDT_UNSIGNED)
  {
    
  }


  metadatum(std::string Name,bool boolval) :
    Name(Name),
    intval(0),
    unsignedval(0),
    strval(""),
    dblval(0.0),
    dblunits(""),
    boolval(boolval),
    md_type(MWS_MDT_BOOL)
  {
    
  }

  //#if (sizeof(snde_index) != sizeof(uint64_t))
#if (SIZEOF_SNDE_INDEX != 8)
  metadatum(std::string Name,snde_index indexval) :
    Name(Name),
    intval(0),
    strval(""),
    dblval(0.0),
    dblunits(""),
    boolval(false),
    md_type(MWS_MDT_UNSIGNED)
  {
    if (indexval==SNDE_INDEX_INVALID) {
      unsignedval=MWS_UNSIGNED_INVALID;
    } else {
      unsignedval=indexval;
    }
  }
  
#endif
  
  int64_t Int(int64_t defaultval) const
  {

    if (md_type == MWS_MDT_INT) {
      return intval;

    } else if (md_type==MWS_MDT_UNSIGNED && unsignedval < (1ull<<63)) {
      return (int64_t)unsignedval;
    } else if (md_type == MWS_MDT_BOOL) {
      if (boolval) {
	return 1;
      } else {
	return 0;
      }
    }
    return defaultval;
  }
  
  uint64_t Unsigned(uint64_t defaultval) const
  {
    if (md_type == MWS_MDT_UNSIGNED) {
      return unsignedval;
    } else if (md_type == MWS_MDT_INT && intval >= 0) {
      return (uint64_t)intval;
    } else if (md_type == MWS_MDT_BOOL) {
      if (boolval) {
	return 1;
      } else {
	return 0;
      }
    }
    return defaultval;
  }

  std::string Str(std::string defaultval) const
  {
    if (md_type != MWS_MDT_STR) {
      return defaultval;
    }
    return strval;
  }
  double Dbl(double defaultval) const
  {
    if (md_type == MWS_MDT_DBL) {
      return dblval;
    } else if (md_type == MWS_MDT_DBL_UNITS) {
      return dblval;
    } 
    return defaultval;
  }
  
  std::pair<double,std::string> DblUnits(double defaultval,std::string defaultunits) const
  {
    if (md_type == MWS_MDT_DBL_UNITS) {
      return std::make_pair(dblval,dblunits);
    } 
    return std::make_pair(defaultval,defaultunits);
  }

  // Should have a method that will return a dblunits value in particular units

  
  bool Bool(bool defaultval) const
  {
    if (md_type == MWS_MDT_BOOL) {
      return boolval;
    } else if (md_type == MWS_MDT_INT) {
      return intval != 0;
    } else if (md_type == MWS_MDT_UNSIGNED) {
      return unsignedval != 0;
    }
    return defaultval;
  }
  
  
  double Numeric(double defaultval) const
  {
    if (md_type == MWS_MDT_DBL) {
      return dblval;
    } else if (md_type == MWS_MDT_INT) {
      return (double)intval;
    } else if (md_type == MWS_MDT_UNSIGNED) {
      return (double)unsignedval;
    } else if (md_type == MWS_MDT_BOOL) {
      if (boolval) {
	return 1.0;
      } else {
	return 0.0;
      }
    } else {
      return defaultval;
    }
  }

  snde_index Index(snde_index defaultval) const
  {
    if (md_type == MWS_MDT_INT) {
      if (intval >= 0) return intval;
      else return SNDE_INDEX_INVALID;
    } else if (md_type == MWS_MDT_UNSIGNED) {
      if (unsignedval == MWS_UNSIGNED_INVALID) {
	return SNDE_INDEX_INVALID;
      }
      return unsignedval;
    } else {
      return defaultval;
    }
  }
  
};

  metadatum metadatum_int(std::string Name,int64_t intval);

  metadatum metadatum_str(std::string Name,std::string strval);

  metadatum metadatum_dbl(std::string Name,double doubleval);
  
  metadatum metadatum_dblunits(std::string Name,double doubleval,std::string units);

  metadatum metadatum_bool(std::string Name,bool boolval);

  metadatum metadatum_unsigned(std::string Name,uint64_t unsignedval);

  metadatum metadatum_index(std::string Name,snde_index indexval);
  
  
  class constructible_metadata {
  public:
    std::unordered_map<std::string,metadatum> metadata;

    constructible_metadata()
    {
      
    }
    

    constructible_metadata(const std::unordered_map<std::string,metadatum> &map) :
      metadata(map)
    {
      
    }

    constructible_metadata(const constructible_metadata &orig) :
      metadata(orig.metadata)
    {
      
    }


    constructible_metadata(std::shared_ptr<const constructible_metadata> orig) :
      metadata(orig->metadata)
    {
      
    }

    
    int64_t GetMetaDatumInt(std::string Name,int64_t defaultval) const
    {
      std::unordered_map<std::string,metadatum>::const_iterator mditer; 
      
      mditer = metadata.find(Name);
      if (mditer == metadata.end()) {
	return defaultval;
      }
      return (*mditer).second.Int(defaultval);
    }
    
  uint64_t GetMetaDatumUnsigned(std::string Name,uint64_t defaultval) const
    {
      std::unordered_map<std::string,metadatum>::const_iterator mditer; 
      
      mditer = metadata.find(Name);
      if (mditer == metadata.end()) {
	return defaultval;
      }
      return (*mditer).second.Unsigned(defaultval);
    }

    snde_bool GetMetaDatumBool(std::string Name,snde_bool defaultval) const
    {
      std::unordered_map<std::string,metadatum>::const_iterator mditer; 
      
      mditer = metadata.find(Name);
      if (mditer == metadata.end()) {
	return defaultval;
      }
      return (bool)((*mditer).second.Bool(defaultval));
    }
    

    
    snde_index GetMetaDatumIdx(std::string Name,snde_index defaultval) const
    // actually stored as unsigned
    {
      std::unordered_map<std::string,metadatum>::const_iterator mditer; 
      
      mditer = metadata.find(Name);
      if (mditer == metadata.end()) {
	return defaultval;
      }
      return (*mditer).second.Index(defaultval);
    }
    
  
    std::string GetMetaDatumStr(std::string Name,std::string defaultval) const
    {
      std::unordered_map<std::string,metadatum>::const_iterator mditer; 
      
      mditer = metadata.find(Name);
      if (mditer == metadata.end()) {
	return defaultval;
      }
      return (*mditer).second.Str(defaultval);
    }
    
    double GetMetaDatumDbl(std::string Name,double defaultval) const
    {
      std::unordered_map<std::string,metadatum>::const_iterator mditer; 
      
      mditer = metadata.find(Name);
      if (mditer == metadata.end()) {
	return defaultval;
      }
      return (*mditer).second.Dbl(defaultval);
    }

    std::pair<double,std::string> GetMetaDatumDblUnits(std::string Name,double defaultval,std::string defaultunits) const
    {
      std::unordered_map<std::string,metadatum>::const_iterator mditer; 
      
      mditer = metadata.find(Name);
      if (mditer == metadata.end()) {
	return std::make_pair(defaultval,defaultunits);
      }
      return (*mditer).second.DblUnits(defaultval,defaultunits);
    }

    const metadatum *FindMetaDatum(std::string Name)
    {
      std::unordered_map<std::string,metadatum>::const_iterator mditer; 
      
      mditer = metadata.find(Name);
      if (mditer == metadata.end()) {
	return nullptr;
      }

      return &(*mditer).second;
    }
    void AddMetaDatum(metadatum newdatum)
    // Add or update an entry 
    {
      
      metadata[newdatum.Name]=newdatum;
      
    }
    
    std::string to_string() const
    {
      // Warning: doesn't scale well
      std::string result;

      for (auto && mdname_mdvalue: metadata) {
	assert(mdname_mdvalue.first == mdname_mdvalue.second.Name);
	
	if (mdname_mdvalue.second.md_type==MWS_MDT_INT) {
	  result += ssprintf("%s: INT %lld\n",mdname_mdvalue.first.c_str(),(long long)mdname_mdvalue.second.intval);
	} else if (mdname_mdvalue.second.md_type==MWS_MDT_UNSIGNED) {
	  result += ssprintf("%s: UNSIGNED %llu\n",mdname_mdvalue.first.c_str(),(unsigned long long)mdname_mdvalue.second.unsignedval);	  
	} else if (mdname_mdvalue.second.md_type==MWS_MDT_STR) {
	  result += ssprintf("%s: STR \"%s\"\n",mdname_mdvalue.first.c_str(),mdname_mdvalue.second.strval.c_str());
	  	  
	} else if (mdname_mdvalue.second.md_type==MWS_MDT_DBL) {
	  result += ssprintf("%s: DBL %g\n",mdname_mdvalue.first.c_str(),mdname_mdvalue.second.dblval);
	} else if (mdname_mdvalue.second.md_type==MWS_MDT_DBL_UNITS) {
	  result += ssprintf("%s: DBLUNITS %g %s\n",mdname_mdvalue.first.c_str(),mdname_mdvalue.second.dblval,mdname_mdvalue.second.dblunits.c_str());
	} else if (mdname_mdvalue.second.md_type==MWS_MDT_BOOL) {
	  result += ssprintf("%s: BOOL %s\n",mdname_mdvalue.first.c_str(),mdname_mdvalue.second.boolval ? "True":"False");	  
	  
	} else {
	  throw snde_error("constructible_metadata::to_string(): Invalid metadatum type %u for %s",(unsigned)mdname_mdvalue.second.md_type,mdname_mdvalue.first.c_str());
	}
      }
      return result;
    }
  };

  static std::shared_ptr<constructible_metadata> MergeMetadata(std::shared_ptr<const constructible_metadata> baseline_md,std::shared_ptr<const constructible_metadata> override_md)
  {
    std::shared_ptr<constructible_metadata> retval=std::make_shared<constructible_metadata>();

    if (!baseline_md) {
      baseline_md = std::make_shared<constructible_metadata>();
    }

    if (!override_md) {
      override_md = std::make_shared<constructible_metadata>();
    }
    
    for (auto & mdname_mdvalue: baseline_md->metadata) {
      
      if (override_md->metadata.find(mdname_mdvalue.first)==override_md->metadata.end()) {
	retval->metadata.emplace(mdname_mdvalue);
      }
    }

    for (auto & mdname_mdvalue: override_md->metadata) {
      retval->metadata.emplace(mdname_mdvalue);
    }

    return retval;
  }


  static std::shared_ptr<constructible_metadata> MergeMetadata3(std::shared_ptr<const constructible_metadata> baseline_md,std::shared_ptr<const constructible_metadata> override_md1,std::shared_ptr<const constructible_metadata> override_md2)
  {
    std::shared_ptr<constructible_metadata> retval=std::make_shared<constructible_metadata>();

    if (!baseline_md) {
      baseline_md = std::make_shared<constructible_metadata>();
    }

    if (!override_md1) {
      override_md1 = std::make_shared<constructible_metadata>();
    }

    if (!override_md2) {
      override_md2 = std::make_shared<constructible_metadata>();
    }


    // emplace is no-op if the element is already there. So we put the highest
    // override (override_md2) first, then the secondary override, then the baseline. 

    for (auto & mdname_mdvalue: override_md2->metadata) {
      retval->metadata.emplace(mdname_mdvalue);
    }

    for (auto & mdname_mdvalue: override_md1->metadata) {
      retval->metadata.emplace(mdname_mdvalue);
    }

    for (auto & mdname_mdvalue: baseline_md->metadata) {
      
      retval->metadata.emplace(mdname_mdvalue);
    }


    return retval;
  }

  

  typedef const constructible_metadata immutable_metadata; 
  
  class recmetadata { // OBSOLETE
public:
  std::shared_ptr<immutable_metadata> _metadata; // c++11 atomic shared pointer to immutable metadata map
  std::mutex admin; // must be locked during changes to _metadata (replacement of C++11 atomic shared_ptr)
  
  recmetadata()
    
  {
    std::shared_ptr<immutable_metadata> new_metadata;
    new_metadata=std::make_shared<immutable_metadata>();
    
    _end_atomic_update(new_metadata);
  }

  
  // thread-safe copy constructor and copy assignment operators
  recmetadata(const recmetadata &orig) /* copy constructor  */
  {
    std::shared_ptr<constructible_metadata> new_metadata;
    new_metadata=std::make_shared<constructible_metadata>(*orig.metadata());

    _end_atomic_update(new_metadata);    
  }


  // copy assignment operator
  recmetadata& operator=(const recmetadata & orig)
  {
    std::lock_guard<std::mutex> adminlock(admin);
    std::shared_ptr<constructible_metadata> new_metadata=std::make_shared<constructible_metadata>(*orig.metadata());
    _end_atomic_update(new_metadata);

    return *this;
  }

  // constructor from a std::unordered_map<string,metadatum>
  recmetadata(const std::unordered_map<std::string,metadatum> & map)
  {
    std::shared_ptr<constructible_metadata> new_metadata=std::make_shared<constructible_metadata>(map);
    _end_atomic_update(new_metadata);    
    
  }

  
  // accessor method for metadata map
  std::shared_ptr<immutable_metadata> metadata() const
  {
    // read atomic shared pointer
    return std::atomic_load(&_metadata);
  }

  std::shared_ptr<constructible_metadata> _begin_atomic_update()
  // admin must be locked when calling this function...
  // it returns new copies of the atomically-guarded data
  {
    
    // Make copies of atomically-guarded data 
    std::shared_ptr<constructible_metadata> new_metadata=std::make_shared<constructible_metadata>(*metadata());
    
    return new_metadata;

  }

  void _end_atomic_update(std::shared_ptr<immutable_metadata> new_metadata)
  {
    std::atomic_store(&_metadata,new_metadata);
  }


  int64_t GetMetaDatumInt(std::string Name,int64_t defaultval)
  {
    std::shared_ptr<immutable_metadata> md=metadata();

    return md->GetMetaDatumInt(Name,defaultval);
  }

  uint64_t GetMetaDatumUnsigned(std::string Name,uint64_t defaultval)
  {
    std::shared_ptr<immutable_metadata> md=metadata();

    return md->GetMetaDatumUnsigned(Name,defaultval);
  }

  snde_index GetMetaDatumIdx(std::string Name,snde_index defaultval)
  // actually stored as unsigned
  {
    std::shared_ptr<immutable_metadata> md=metadata();

    return md->GetMetaDatumIdx(Name,defaultval);
  }

  
  std::string GetMetaDatumStr(std::string Name,std::string defaultval)
  {
    std::shared_ptr<immutable_metadata> md=metadata();

    return md->GetMetaDatumStr(Name,defaultval);
  }

  double GetMetaDatumDbl(std::string Name,double defaultval)
  {
    std::shared_ptr<immutable_metadata> md=metadata();

    return md->GetMetaDatumDbl(Name,defaultval);
  }

  void AddMetaDatum(metadatum newdatum)
  // Add or update an entry 
  {
    std::lock_guard<std::mutex> adminlock(admin);
    std::shared_ptr<constructible_metadata> new_metadata; // not officially immutable until we are done with our update
    
    new_metadata = _begin_atomic_update();
    
    new_metadata->metadata[newdatum.Name]=newdatum;
    
    _end_atomic_update(new_metadata);
    
  }


};


};


#endif // SNDE_METADATA_HPP
