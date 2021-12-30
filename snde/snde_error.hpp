#ifndef SNDE_ERROR_HPP
#define SNDE_ERROR_HPP

// CMake's CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS is set, but some global variables still require 
// explicit import and export flags to compile properly with MSVC and other compilers that
// behave similarly.  Newer versions of GCC shouldn't care about the presence of dllimport
// or dllexport, but it doesn't need it.
#ifdef _WIN32
#ifdef SPATIALNDE2_SHAREDLIB_EXPORT
#define SNDE_API __declspec(dllexport)
#else
#define SNDE_API __declspec(dllimport)
#endif
#else
#define SNDE_API
#endif


#ifdef __GNUG__ // catches g++ and clang see https://www.gnu.org/software/libc/manual/html_node/Backtraces.html

#include <execinfo.h>

#endif // __GNUG__

#include <string>
#include <stdexcept>
#include <cstring>
#include <cstdarg>

#include <map>
#include <cstdio>

#if defined(_MSC_VER) && _MSC_VER < 1900
#define snprintf _snprintf
#endif

namespace snde {


  std::string demangle_type_name(const char *name);
  
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



  class snde_error : public std::runtime_error {
  public:
    std::string whatstr; 
#ifdef __GNUG__ // catches g++ and clang see https://www.gnu.org/software/libc/manual/html_node/Backtraces.html
    void *backtrace_buffer[100];
    char **backtrace_syms;
    int num_syms;

#endif // __GNUG__
    template<typename ... Args>
    snde_error(std::string fmt, Args && ... args) : std::runtime_error(std::string("SNDE runtime error: ")+ssprintf(fmt,std::forward<Args>(args) ...)) { 
      whatstr = std::string(std::runtime_error::what())+"\n";
#ifdef __GNUG__ // catches g++ and clang
      num_syms = backtrace(backtrace_buffer,sizeof(backtrace_buffer)/sizeof(void *));

      backtrace_syms = backtrace_symbols(backtrace_buffer,num_syms);


      int cnt;
      for (cnt=1; cnt < num_syms; cnt++) {
	whatstr += ssprintf("[ %d ]: %s\n",cnt,backtrace_syms[cnt]);
      }
#endif
    }

    snde_error &operator=(const snde_error &) = delete;
    snde_error(const snde_error &orig) :
      runtime_error(orig)
    {
#ifdef __GNUG__ // catches g++ and clang
      num_syms = 0;
      backtrace_syms=nullptr;
      whatstr = orig.whatstr; 
#endif // __GNUG__
    }
    
#ifdef __GNUG__ // catches g++ and clang
    virtual ~snde_error()
    {
      if (backtrace_syms) {	
	free(backtrace_syms);
      }
    }
#endif // __GNUG__

    virtual const char *what() const noexcept
    {
      return whatstr.c_str();
    }
    
    // Alternate constructor with leading int (that is ignored)
    // so we can construct without doing string formatting
    //snde_error(int junk,std::string msg) : std::runtime_error(std::string("SNDE runtime error: ") + msg) {
    //
    //}
  };

  static inline std::string portable_strerror(int errnum)
  {
    char *errstr;
    char *buf=nullptr;

#ifdef _WIN32
    // Win32 strerror() is thread safe per MS docs
    errstr=strerror(errnum);
#else
    {
      int buflen=1; // Make this big once tested
#if (_POSIX_C_SOURCE >= 200112L) && !  _GNU_SOURCE
      int err=0;
      // XSI strerror_r()
      do {
	if (buf) {
	  free(buf);
	}
	
	buf=(char *)malloc(buflen);
	buf[0]=0;
	err=strerror_r(errnum,buf,buflen);
	buf[buflen-1]=0;
	buflen *= 2; 
      } while (err && (err==ERANGE || (err < 0 && errno==ERANGE)));
      errstr=buf;
#else
      // GNU strerror_r()
      do {
	if (buf) {
	  free(buf);
	}
	
	buf=(char *)malloc(buflen);
	buf[0]=0;
	errstr=strerror_r(errnum,buf,buflen);
	buf[buflen-1]=0;
	buflen *= 2; 
      } while (errstr==buf && strlen(errstr)==buflen-1);
      
#endif
    }
#endif
    std::string retval(errstr);
    
    if (buf) {
      free(buf);
    }
    
    return retval;
  }
  
  class posix_error : public std::runtime_error {
  public:
    int _errno;

    template<typename ... Args>
    posix_error(std::string fmt, Args && ... args) : _errno(errno), std::runtime_error(ssprintf("POSIX runtime error %d (%s): %s",_errno,portable_strerror(_errno).c_str(),cssprintf(fmt,std::forward<Args>(args) ...))) { /* cssprintf will leak memory, but that's OK because this is an error and should only happen rarely  */
      //std::string foo=openclerrorstring[clerrnum];
      //std::string bar=openclerrorstring.at(clerrnum);
      //std::string fubar=openclerrorstring.at(-37);
      
    }
  };

  template<typename ... Args>
  void snde_warning(std::string fmt, Args && ... args)
  {
    std::string warnstr = ssprintf(fmt,std::forward<Args>(args) ...);
    fprintf(stderr,"SNDE WARNING: %s\n",warnstr.c_str());
  }

  SNDE_API extern unsigned initial_debugflags;
  unsigned current_debugflags();

  
  template<typename ... Args>
  void snde_debug(unsigned dbgclass,std::string fmt, Args && ... args)
  {
    
    if (dbgclass & current_debugflags()) {
      std::string warnstr = ssprintf(fmt,std::forward<Args>(args) ...);
      fprintf(stderr,"SNDE DEBUG: %s\n",warnstr.c_str());
    }
  }
    // defines for dbgclass/current_debugflags
    // !!!*** SEE ALSO CHECKFLAG ENTRIES IN SNDE_ERROR.CPP ***!!! 
#define SNDE_DC_RECDB (1<<0)
#define SNDE_DC_RECMATH (1<<1)
#define SNDE_DC_NOTIFY (1<<2)
#define SNDE_DC_LOCKING (1<<3)
#define SNDE_DC_APP (1<<4)
#define SNDE_DC_COMPUTE_DISPATCH (1<<5)
#define SNDE_DC_RENDERING (1<<6)
#define SNDE_DC_DISPLAY (1<<7)
#define SNDE_DC_EVENT (1<<8)
#define SNDE_DC_VIEWER (1<<9)
#define SNDE_DC_ALL ((1<<10)-1)

   
}
#endif /* SNDE_ERROR_HPP */
