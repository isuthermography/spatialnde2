
#include "snde/cached_recording.hpp"

namespace snde {

  
  std::string get_cache_name(const std::string &base)
  {
    static std::atomic<unsigned> index;

    unsigned curindex = index++;
    return ssprintf("%s%u",base.c_str(),curindex);
  }


};
