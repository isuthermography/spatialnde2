#include "snde/mutablerecstore.hpp"

namespace snde {
  
  std::string iterablerecrefs::iterator::get_full_name()
  {
    size_t index;
    std::shared_ptr<iterablerecrefs> thisrefs=refs; 
    std::string full_name="/";
    
    for (index=0; index < pos.size();index++) {
      std::shared_ptr<mutableinfostore> sub_infostore;
      std::shared_ptr<iterablerecrefs> sub_refs;
      std::tie(sub_infostore,sub_refs) = refs->recs[pos[index]];
      if (sub_infostore) {
	assert(index==pos.size()-1);
	full_name += sub_infostore->leafname;
	return full_name;
      } else {
	assert(index < pos.size()-1);
	full_name += sub_refs->leafname + "/";
      }
      
      thisrefs=sub_refs; 
    }
    throw std::runtime_error("Bad iterablerecrefs iterator!");
  }
}
