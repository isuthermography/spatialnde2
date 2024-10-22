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

  class DropPythonGIL {
  public:

    DropPythonGIL(std::string caller, std::string file, int line) : 
      _state(nullptr),
      _caller(caller),
      _file(file),
      _line(line)
    {
      if (Py_IsInitialized() && PyGILState_Check()) {
	snde_debug(SNDE_DC_PYTHON_SUPPORT, "Dropping GIL by %s in %s:%d", _caller.c_str(), _file.c_str(), _line);
	_state = PyEval_SaveThread();
      }
    }

    virtual ~DropPythonGIL() {
      if (_state) {
	snde_debug(SNDE_DC_PYTHON_SUPPORT, "Restoring GIL for %s in %s:%d", _caller.c_str(), _file.c_str(), _line);
	PyEval_RestoreThread(_state);
      }
    }

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
