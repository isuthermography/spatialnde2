#include "snde/snde_error.hpp"

#ifdef __GNUG__
#include <cstdlib>
#include <cxxabi.h>
#endif // __GNUG__

namespace snde {

  SNDE_API unsigned initial_debugflags=0;
  //SNDE_API unsigned initial_debugflags=SNDE_DC_COMPUTE_DISPATCH;
  //SNDE_API unsigned initial_debugflags=SNDE_DC_RECDB|SNDE_DC_NOTIFY;
  //SNDE_API unsigned initial_debugflags=SNDE_DC_ALL;

  unsigned check_debugflag(unsigned flag, const char *env_var)
  {
    char *var_val = std::getenv(env_var);

    if (var_val && strlen(var_val) && std::string("0") != var_val) {
      return flag;
    }

    return 0;
  }
  
  unsigned read_debugflags()
  {
    unsigned debugflags = initial_debugflags;

    debugflags |= check_debugflag(SNDE_DC_RECDB,"SNDE_DC_RECDB");
    debugflags |= check_debugflag(SNDE_DC_RECMATH,"SNDE_DC_RECMATH");
    debugflags |= check_debugflag(SNDE_DC_NOTIFY,"SNDE_DC_NOTIFY");
    debugflags |= check_debugflag(SNDE_DC_LOCKING,"SNDE_DC_LOCKING");
    debugflags |= check_debugflag(SNDE_DC_APP,"SNDE_DC_APP");
    debugflags |= check_debugflag(SNDE_DC_COMPUTE_DISPATCH,"SNDE_DC_COMPUTE_DISPATCH");
    debugflags |= check_debugflag(SNDE_DC_RENDERING,"SNDE_DC_RENDERING");
    debugflags |= check_debugflag(SNDE_DC_DISPLAY,"SNDE_DC_DISPLAY");
    debugflags |= check_debugflag(SNDE_DC_EVENT,"SNDE_DC_EVENT");
    debugflags |= check_debugflag(SNDE_DC_VIEWER,"SNDE_DC_VIEWER");
    debugflags |= check_debugflag(SNDE_DC_X3D,"SNDE_DC_X3D");
    debugflags |= check_debugflag(SNDE_DC_OPENCL,"SNDE_DC_OPENCL");
    debugflags |= check_debugflag(SNDE_DC_OPENCL_COMPILATION,"SNDE_DC_OPENCL_COMPILATION");
    debugflags |= check_debugflag(SNDE_DC_PYTHON_SUPPORT, "SNDE_DC_PYTHON_SUPPORT");
    debugflags |= check_debugflag(SNDE_DC_MEMLEAK,"SNDE_DC_MEMLEAK");
    debugflags |= check_debugflag(SNDE_DC_ALL,"SNDE_DC_ALL");

    return debugflags;
  }
  

  unsigned current_debugflags()
  {
    static unsigned flags = read_debugflags(); // thread safe per C++11

    return flags; 
  }

  
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
 
