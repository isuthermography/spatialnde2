%shared_ptr(snde::constructible_metadata)
%shared_ptr(snde::recmetadata)
%{

  #include "metadata.hpp"
  
%}
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

  metadatum();
  metadatum(std::string Name,const metadatum &oldmd);
  metadatum(std::string Name,int64_t intval);

  
  metadatum(std::string Name,std::string strval);
  metadatum(std::string Name,double dblval);
  metadatum(std::string Name,double dblval,std::string units);

  metadatum(std::string Name,uint64_t unsignedval);
  metadatum(std::string Name,bool boolval);
  
  int64_t Int(int64_t defaultval);
  uint64_t Unsigned(uint64_t defaultval);

  std::string Str(std::string defaultval);
  double Dbl(double defaultval);
  std::pair<double,std::string> DblUnits(double defaultval,std::string defaultunits);
  bool Bool(bool defaultval);

  double Numeric(double defaultval);
  snde_index Index(snde_index defaultval);
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
    
      //constructible_metadata(const constructible_metadata &orig);
      
    constructible_metadata(std::shared_ptr<const constructible_metadata> orig);


      
    int64_t GetMetaDatumInt(std::string Name,int64_t defaultval) const
    {
      std::unordered_map<std::string,metadatum>::const_iterator mditer; 

      mditer = metadata.find(Name);
      if (mditer == metadata.end() || mditer->second.md_type != MWS_MDT_INT) {
	return defaultval;
      }
      return (*mditer).second.Int(defaultval);
    }
    
  uint64_t GetMetaDatumUnsigned(std::string Name,uint64_t defaultval) const
    {
      std::unordered_map<std::string,metadatum>::const_iterator mditer; 
      
      mditer = metadata.find(Name);
      if (mditer == metadata.end() || mditer->second.md_type != MWS_MDT_UNSIGNED) {
	return defaultval;
      }
      return (*mditer).second.Unsigned(defaultval);
    }

    snde_bool GetMetaDatumBool(std::string Name,snde_bool defaultval);

    snde_index GetMetaDatumIdx(std::string Name,snde_index defaultval) const
    // actually stored as unsigned
    {
      std::unordered_map<std::string,metadatum>::const_iterator mditer; 
      
      mditer = metadata.find(Name);
      if (mditer == metadata.end() || mditer->second.md_type != MWS_MDT_UNSIGNED) {
	return defaultval;
      }
      return (*mditer).second.Index(defaultval);
    }
    
  
    std::string GetMetaDatumStr(std::string Name,std::string defaultval) const
    {
      std::unordered_map<std::string,metadatum>::const_iterator mditer; 
      
      mditer = metadata.find(Name);
      if (mditer == metadata.end() || mditer->second.md_type != MWS_MDT_STR) {
	return defaultval;
      }
      return (*mditer).second.Str(defaultval);
    }
    
    double GetMetaDatumDbl(std::string Name,double defaultval) const
    {
      std::unordered_map<std::string,metadatum>::const_iterator mditer; 
      
      mditer = metadata.find(Name);
      if (mditer == metadata.end() || mditer->second.md_type != MWS_MDT_DBL) {
	return defaultval;
      }
      return (*mditer).second.Dbl(defaultval);
    }

    std::pair<double, std::string> GetMetaDatumDblUnits(std::string Name, double defaultval, std::string defaultunits) const
    {
        std::unordered_map<std::string, metadatum>::const_iterator mditer;

        mditer = metadata.find(Name);
        if (mditer == metadata.end()) {
            return std::make_pair(defaultval, defaultunits);
        }
        return (*mditer).second.DblUnits(defaultval, defaultunits);
    }

    void AddMetaDatum(metadatum newdatum)
    // Add or update an entry 
    {
      
      metadata[newdatum.Name]=newdatum;
      
    }
    
    std::string to_string();

  };

  %extend constructible_metadata {
    std::string __str__()
    {
      return self->to_string();
      
    }

    std::string __repr__()
    {
      return self->to_string();
      
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

  static std::shared_ptr<constructible_metadata> MergeMetadata3(std::shared_ptr<const constructible_metadata> baseline_md,std::shared_ptr<const constructible_metadata> override_md1,std::shared_ptr<const constructible_metadata> override_md2);

  typedef const constructible_metadata immutable_metadata; 

%pythoncode %{

immutable_metadata=constructible_metadata;

%}

  
class recmetadata {
public:
  std::shared_ptr<immutable_metadata> _metadata; // c++11 atomic shared pointer to immutable metadata map
  //std::mutex admin; // must be locked during changes to _metadata (replacement of C++11 atomic shared_ptr)
  
  recmetadata();
  
  // thread-safe copy constructor and copy assignment operators
  recmetadata(const recmetadata &orig); /* copy constructor  */

  // copy assignment operator
  //recmetadata& operator=(const recmetadata & orig);

  // constructor from a std::unordered_map<string,metadatum>
  recmetadata(const std::unordered_map<std::string,metadatum> & map);
  
  // accessor method for metadata map
  std::shared_ptr<immutable_metadata> metadata() const;

  //  std::tuple<std::shared_ptr<constructible_metadata>> _begin_atomic_update();

  //void _end_atomic_update(std::shared_ptr<immutable_metadata> new_metadata);

  int64_t GetMetaDatumInt(std::string Name,int64_t defaultval);

  uint64_t GetMetaDatumUnsigned(std::string Name,uint64_t defaultval);

  snde_index GetMetaDatumIdx(std::string Name,snde_index defaultval); // actually stored as unsigned

  
  std::string GetMetaDatumStr(std::string Name,std::string defaultval);

  double GetMetaDatumDbl(std::string Name,double defaultval);

  void AddMetaDatum(metadatum newdatum);


};


};

