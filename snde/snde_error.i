//%shared_ptr(snde::recording_base);
//snde_rawaccessible(snde::recording_base);


%{
  #include "snde/snde_error.hpp"
%}


namespace snde {


  std::string demangle_type_name(const char *name);
  

  class snde_error : public std::runtime_error {
  public:
    std::string whatstr; 
    snde_error(std::string msg);  // We pretend to swig that the constructor just takes a single string argument

    snde_error &operator=(const snde_error &) = delete;
    snde_error(const snde_error &orig);

    virtual ~snde_error();

    virtual const char *what() const noexcept;
    
    // Alternate constructor with leading int (that is ignored)
    // so we can construct without doing string formatting
    //snde_error(int junk,std::string msg) : std::runtime_error(std::string("SNDE runtime error: ") + msg) {
    //
    //}
  };

  
  class posix_error : public snde_error {
  public:
    int _myerrno;
    
    posix_error(std::string); // pretend it only takes one argument for Python
  };

#ifdef _WIN32

  class win32_error : public snde_error {
  public:
    unsigned long _myerrno;
    std::string _errstr;
    
    win32_error(std::string); // pretend it only takes one argument for Python
  };
#endif // _WIN32

  void snde_warning(std::string fmt); // pretend it only takes one argument for Python

  extern unsigned initial_debugflags;
  unsigned current_debugflags();

  
  void snde_debug(unsigned dbgclass,std::string fmt); // ignore varargs for Python
  
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
#define SNDE_DC_X3D (1<<10)
#define SNDE_DC_OPENCL (1<<11)
#define SNDE_DC_OPENCL_COMPILATION (1<<12)
#define SNDE_DC_MEMLEAK (1<<13)
#define SNDE_DC_ALL ((1<<14)-1)

   
}
