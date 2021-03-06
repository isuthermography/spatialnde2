#ifndef SNDE_ERROR_HPP
#define SNDE_ERROR_HPP

#include <string>
#include <cstring>
#include <cstdarg>

#include <map>
#include <cstdio>

#if defined(_MSC_VER) && _MSC_VER < 1900
#define snprintf _snprintf
#endif

namespace snde {

  
  static inline std::string ssprintf(const std::string fmt,...)
  {
    char *buf=NULL;
    va_list ap;
    int len;
    int nbytes;
    
    len=4*fmt.size();
    
    do {
      buf=(char *)malloc(len+1);
      va_start(ap,fmt);
      nbytes=vsnprintf(buf,len,fmt.c_str(),ap);
      va_end(ap);
      
      if (nbytes >= len || nbytes < 0) {
	len*=2;
	free(buf);
	buf=NULL;
      }
      
    } while (!buf);

    std::string retval(buf);
    free(buf);
    
    return retval;
    
  }
  

  static inline std::string vssprintf(const std::string fmt,va_list ap)
  {
    char *buf=NULL;
    int len;
    int nbytes;
    
    len=4*fmt.size();
    
    do {
      buf=(char *)malloc(len+1);
      nbytes=vsnprintf(buf,len,fmt.c_str(),ap);
      
      if (nbytes >= len || nbytes < 0) {
	len*=2;
	free(buf);
	buf=NULL;
      }
      
    } while (!buf);

    std::string retval(buf);
    free(buf);
    
    return retval;
    
  }

  template <typename ... Args>
  static inline char *cssprintf(const std::string fmt, Args && ... args)
  {
    std::string result;
    char *cresult;

    result=ssprintf(fmt,std::forward<Args>(args) ...);

    cresult=strdup(result.c_str());

    return cresult; /* should free() cresult */
  }

}
#endif /* SNDE_ERROR_HPP */
