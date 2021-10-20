#include "snde/snde_error.hpp"

#ifdef __GNUG__
#include <cstdlib>
#include <cxxabi.h>
#endif // __GNUG__

namespace snde {

  SNDE_API unsigned current_debugflags=0;
  //SNDE_API unsigned current_debugflags=SNDE_DC_RECDB|SNDE_DC_NOTIFY;
  //SNDE_API unsigned current_debugflags=SNDE_DC_ALL;


  
  std::string demangle_type_name(const char *name)
  // demangle type_info names
  {
#ifdef __GNUG__ // catches g++ and clang
    int status=1;
    char *ret;
    std::string retstr;
    // Only g++/clang actually name-mangle type_info.name() strings
    ret = abi::__cxa_demangle(name,nullptr,nullptr,&status);
    if (status==0) {
      retstr = ret;
      free(ret);
    } else {
      retstr = name;
    }

    return retstr;
#else // __GNUG__
    return name; 
#endif // __GNUG__
    
  }
  
  
};
 
