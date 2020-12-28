#include "mutablewfmstore.hpp"

namespace snde {
  
  std::string iterablewfmrefs::iterator::get_full_name()
  {
    size_t index;
    std::shared_ptr<iterablewfmrefs> thisrefs=refs; 
    std::string full_name="/";
    
    for (index=0; index < pos.size();index++) {
      std::shared_ptr<mutableinfostore> sub_infostore;
      std::shared_ptr<iterablewfmrefs> sub_refs;
      std::tie(sub_infostore,sub_refs) = refs->wfms[pos[index]];
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
    throw std::runtime_error("Bad iterablewfmrefs iterator!");
  }
}
