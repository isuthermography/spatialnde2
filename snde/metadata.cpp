
#include "snde/snde_types.h"
#include "snde/metadata.hpp"

namespace snde {

    metadatum metadatum_int(std::string Name,int64_t intval)
  {
    return metadatum(Name,intval);
  }

  metadatum metadatum_str(std::string Name,std::string strval)
  {
    return metadatum(Name,strval);
  }

  metadatum metadatum_dbl(std::string Name,double doubleval)
  {
    return metadatum(Name,doubleval);
  }

  metadatum metadatum_unsigned(std::string Name,uint64_t unsignedval)
  {
    return metadatum(Name,unsignedval);
  }

  metadatum metadatum_index(std::string Name,snde_index indexval)
  {
    return metadatum(Name,indexval);
  }

  
  
  
};
