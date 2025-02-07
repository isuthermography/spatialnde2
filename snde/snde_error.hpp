#ifndef SNDE_ERROR_HPP
#define SNDE_ERROR_HPP


#ifdef __GNUG__ // catches g++ and clang see https://www.gnu.org/software/libc/manual/html_node/Backtraces.html

#include <execinfo.h>

#endif // __GNUG__

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif

#include <string>
#include <stdexcept>
#include <cstring>
#include <cstdarg>
#include <ctime>
#include <iomanip>

#include <map>
#include <cstdio>
#include <cstring>
#include "snde/snde_types.h"

#if defined(_MSC_VER) && _MSC_VER < 1900
#define snprintf _snprintf
#endif

namespace snde {


  std::string demangle_type_name(const char *name);
  
  static inline std::string ssprintf(const std::string fmt,...)
  {
    char *buf=NULL;
    va_list ap;
    size_t len;
    unsigned long long nbytes;
    
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
    size_t len;
    unsigned long long nbytes;
    
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
    std::string shortwhatstr; 
    std::string whatstr; 
#ifdef __GNUG__ // catches g++ and clang see https://www.gnu.org/software/libc/manual/html_node/Backtraces.html
    void *backtrace_buffer[100];
    char **backtrace_syms;
    int num_syms;

#endif // __GNUG__
    template<typename ... Args>
    snde_error(std::string fmt, Args && ... args) : std::runtime_error(std::string("SNDE runtime error: ")+ssprintf(fmt,std::forward<Args>(args) ...)) { 
      shortwhatstr = std::string(std::runtime_error::what());
      whatstr = shortwhatstr+"\n";
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
    
    virtual ~snde_error()
    {
#ifdef __GNUG__ // catches g++ and clang
      if (backtrace_syms) {	
	free(backtrace_syms);
      }
#endif // __GNUG__
    }

    virtual const char *what() const noexcept
    {
      return whatstr.c_str();
    }

    virtual const char *shortwhat() const noexcept
    {
      return shortwhatstr.c_str();
    }

    // Alternate constructor with leading int (that is ignored)
    // so we can construct without doing string formatting
    //snde_error(int junk,std::string msg) : std::runtime_error(std::string("SNDE runtime error: ") + msg) {
    //
    //}
  };

  class snde_indexerror: public snde_error {
  public:
    //index error so we can throw python IndexError from C++ code
    // note that this class is not wrapped for python; instead
    // there is a separate spatialnde2.snde_indexerror class
    // defined in spatialnde2.i
    template<typename ... Args>
    snde_indexerror(std::string fmt, Args && ... args) : snde_error(fmt,std::forward<Args>(args) ...) { 
      
    }

    snde_indexerror &operator=(const snde_indexerror &) = delete;
    snde_indexerror(const snde_indexerror &orig) :
      snde_error(orig)
    {

    }
    
    virtual ~snde_indexerror()
    {

    }

  };

  class snde_stopiteration: public snde_error {
  public:
    //index error so we can throw python IndexError from C++ code
    // note that this class is not wrapped for python; instead
    // there is a separate spatialnde2.snde_indexerror class
    // defined in spatialnde2.i
    template<typename ... Args>
    snde_stopiteration(std::string fmt, Args && ... args) : snde_error(fmt,std::forward<Args>(args) ...) { 
      
    }

    snde_stopiteration &operator=(const snde_stopiteration &) = delete;
    snde_stopiteration(const snde_stopiteration &orig) :
      snde_error(orig)
    {

    }
    
    virtual ~snde_stopiteration()
    {

    }

  };

#ifdef _WIN32
  static inline std::string GetWin32ErrorAsString(DWORD err)
  {
      if (err == 0) {
          return std::string();
      }

      LPSTR buf = nullptr;
      size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
          NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, NULL);

      std::string message(buf, size);
      message.erase(message.find_last_not_of(" \t\n\r\f\v") + 1);

      LocalFree(buf);
      return message;
  }
#else
  static inline std::string GetWin32ErrorAsString(unsigned long err)
  {
    assert(0);
    return "";
  }
#endif

  static inline std::string portable_strerror(int errnum)
  {
    char *buf=nullptr;

#ifdef _WIN32
    // Win32 strerror() is thread safe per MS docs
    char errstr[95];
    strerror_s(errstr, 95, errnum);
#else
    char* errstr;
    {
      int buflen=1; // Make this big once tested
#if ((_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600) && !_GNU_SOURCE) || __APPLE__
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
  
  class posix_error : public snde_error {
  public:
    int _myerrno;

    template<typename ... Args>
    posix_error(std::string fmt, Args && ... args) : 
        _myerrno(errno), 
        snde_error("%s",ssprintf("POSIX runtime error %d (%s): %s", _myerrno,portable_strerror(_myerrno).c_str(),ssprintf(fmt,std::forward<Args>(args) ...).c_str()).c_str())
    { 
      //std::string foo=openclerrorstring[clerrnum];
      //std::string bar=openclerrorstring.at(clerrnum);
      //std::string fubar=openclerrorstring.at(-37);
      
    }
  };

  // This class is only for use with Windows API calls found in Windows.h

#ifdef _WIN32
  class win32_error : public snde_error {
  public:
      DWORD _myerrno;
      std::string _errstr;

      template<typename ... Args>
      win32_error(std::string fmt, Args&& ... args) :
          _myerrno(GetLastError()),
          _errstr(GetWin32ErrorAsString(_myerrno)),
          snde_error("%s", ssprintf("Win32 runtime error 0x%lx (%s): %s", _myerrno, _errstr.c_str(), ssprintf(fmt, std::forward<Args>(args) ...).c_str()).c_str())
      {
          //std::string foo=openclerrorstring[clerrnum];
          //std::string bar=openclerrorstring.at(clerrnum);
          //std::string fubar=openclerrorstring.at(-37);

      }
  };
#endif // _WIN32

  
  template<typename ... Args>
  void snde_warning(std::string fmt, Args && ... args)
  {
    std::string warnstr = ssprintf(fmt,std::forward<Args>(args) ...);
    std::time_t time_now = std::time(nullptr);
    std::tm* tm = std::localtime(&time_now);
    char timebuf[80];
    strftime(timebuf, sizeof(timebuf), "%Y-%M-%d %H:%M:%S", tm);
    fprintf(stderr,"[%s] SNDE WARNING: %s\n", timebuf ,warnstr.c_str());
  }

  SNDE_API extern unsigned initial_debugflags;
  unsigned current_debugflags();

  
  template<typename ... Args>
  void snde_debug(unsigned dbgclass,std::string fmt, Args && ... args)
  {
    
    if (dbgclass & current_debugflags()) {
      std::string warnstr = ssprintf(fmt,std::forward<Args>(args) ...);
      std::time_t time_now = std::time(nullptr);
      std::tm* tm = std::localtime(&time_now);
      char timebuf[80];
      strftime(timebuf, sizeof(timebuf), "%Y-%M-%d %H:%M:%S", tm);
      fprintf(stderr,"[%s] SNDE DEBUG: %s\n", timebuf, warnstr.c_str());
    }
  }
    // defines for dbgclass/current_debugflags
    // !!!*** SEE ALSO CHECKFLAG ENTRIES IN SNDE_ERROR.CPP AND ENTRIES IN snde_error.i ***!!! 
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
#define SNDE_DC_X3D (1<<10)
#define SNDE_DC_OPENCL (1<<11)
#define SNDE_DC_OPENCL_COMPILATION (1<<12)
#define SNDE_DC_PYTHON_SUPPORT (1<<13)
#define SNDE_DC_MEMLEAK (1<<14)
#define SNDE_DC_ALL ((1<<15)-1)

   
}
#endif /* SNDE_ERROR_HPP */
