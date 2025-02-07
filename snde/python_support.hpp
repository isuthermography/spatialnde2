#ifndef SNDE_PYTHON_SUPPORT_HPP
#define SNDE_PYTHON_SUPPORT_HPP

#ifdef SNDE_PYTHON_SUPPORT_ENABLED

#include "snde/snde_error.hpp"

#pragma push_macro("slots")
#undef slots
#include <Python.h>
#pragma pop_macro("slots")

/*
There is a potential for deadlocks when intermixing C++ code that could be called by Qt.  Some Python Qt bindings do not 
necessarily explicitly release the GIL.  This code provides an optional capability built on an RAII model that behaves 
similarly to Py_BEGIN_ALLOW_THREADS and Py_END_ALLOW_THREADS macros.  For any function in which we should explicitly drop 
the GIL, simply add a SNDE_BeginDropPythonGILBlock to the beginning of the function and a SNDE_EndDropPythonGILBlock to the end.  
These macros do include curly braces to ensure the GIL state is restored before the function returns.  They will do 
nothing if the SNDE_PYTHON_SUPPORT_ENABLED flag is not set or if the thread has not already called Py_Initialize.
*/

#define SNDE_BeginDropPythonGILBlock { DropPythonGIL dropGIL(__FUNCTION__, __FILE__, __LINE__);
#define SNDE_EndDropPythonGILBlock }

namespace snde {
  // On WIN32 we link spatialnde2 with python and always
  // drop the GIL if it is held in appropriate blocks.
  
  // On Mac and Linux we dynamically look up whether
  // python symbols exist in our running binary, and
  // if they do, we drop the GIL if it is held in
  // appropriate blocks.
  // This is necessary because on Mac, there is a difference
  // between linking to a library and loading it as a
  // module. If you both link to it and load it with
  // different locations, you get a crash.

  // The Mac/Linux behavior should probably be ported over
  // to Windows, but there doesn't seem to be a simple
  // Windows call for getting the address of a symbol
  // without knowing the name of the dll it should be
  // coming from. That name will change with python
  // versions, making things messy.
  
#ifndef _WIN32
  extern std::atomic<bool> python_syms_initialized;
  extern std::mutex python_syms_write_lock; // Last in locking order
  extern int (*python_Py_IsInitialized)(void);
  extern PyGILState_STATE (*python_PyGILState_Check)(void);
  extern PyThreadState * (*python_PyEval_SaveThread)(void);
  extern void (*python_PyEval_RestoreThread)(PyThreadState *);
  int python_getsyms();
#endif // _WIN32
  
  class DropPythonGIL {
  public:

    DropPythonGIL(std::string caller, std::string file, int line);

    virtual ~DropPythonGIL();
    

    DropPythonGIL(const DropPythonGIL&) = delete;
    DropPythonGIL& operator=(const DropPythonGIL&) = delete;

  private:
    PyThreadState* _state;    
    std::string _caller;
    std::string _file;
    int _line;

  };  
  

}

#else // SNDE_PYTHON_SUPPORT_ENABLED

#define SNDE_BeginDropPythonGILBlock
#define SNDE_EndDropPythonGILBlock

#endif // SNDE_PYTHON_SUPPORT_ENABLED

#endif // SNDE_PYTHON_SUPPORT_HPP
