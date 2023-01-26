
%{
#include "recdb_paths.hpp"
%}

namespace snde {
  static inline std::pair<std::string,std::string> recdb_path_split(std::string full_path);
  
  static inline std::string recdb_path_context(std::string full_path);
  
  
    static inline std::string recdb_path_join(std::string context,std::string tojoin);
    static std::string recdb_relative_path_to(const std::string &from,const std::string &to);
}
