#ifndef _WIN32
#include <dlfcn.h>
#endif
#include <atomic>
#include <mutex>
#include "python_support.hpp"

namespace snde {
#ifdef SNDE_PYTHON_SUPPORT_ENABLED
#ifndef _WIN32
  std::atomic<bool> python_syms_initialized;
  std::mutex python_syms_write_lock; // Last in locking order
  int (*python_Py_IsInitialized)(void);
  PyGILState_STATE (*python_PyGILState_Check)(void);
  PyThreadState * (*python_PyEval_SaveThread)(void);
  void (*python_PyEval_RestoreThread)(PyThreadState *);

  static std::atomic<unsigned short> symattempt_counter;


  
  int python_getsyms()
  {
    while (!python_syms_initialized) {
      std::lock_guard<std::mutex> syms_write(python_syms_write_lock);
      if (python_syms_initialized) {
        break;
      }
      python_Py_IsInitialized=(int (*)(void))dlsym(RTLD_DEFAULT,"Py_IsInitialized");
      if (!python_Py_IsInitialized) {
        //snde_warning("python_support: python library not loaded");
        python_syms_initialized=true;
        break;
      }
      python_PyGILState_Check=(PyGILState_STATE (*)(void))dlsym(RTLD_DEFAULT,"PyGILState_Check");
      python_PyEval_SaveThread=(PyThreadState * (*)(void))dlsym(RTLD_DEFAULT,"PyEval_SaveThread");
      python_PyEval_RestoreThread=(void (*)(PyThreadState*))dlsym(RTLD_DEFAULT,"PyEval_RestoreThread");
      python_syms_initialized = true;
    }
    if (!python_Py_IsInitialized) {
      symattempt_counter++;
      if (!(symattempt_counter %1000)) {
        // Sporadically attempt dlsym() --not every time because on
        // MacOS dlsym is documented as being slow -- to see if
        // python has since been loaded. Generate an error if it
        // has because python should be loaded prior to spatialnde2.
        if (dlsym(RTLD_DEFAULT,"Py_IsInitialized")) {
          throw snde_error("spatialnde2 python_support: python found loaded when previously it hadn't been. Must load python prior to spatialnde2.");
        }
      }
    }
    return (bool) python_Py_IsInitialized;
  }

#endif // _WIN32
  DropPythonGIL::DropPythonGIL(std::string caller, std::string file, int line) : 
      _state(nullptr),
      _caller(caller),
      _file(file),
      _line(line)
    {
      if (
#ifdef _WIN32
          Py_IsInitialized() && PyGILState_Check()
#else 
          python_getsyms() &&
          python_Py_IsInitialized() && python_PyGILState_Check()
#endif // _WIN32
          ) {
	snde_debug(SNDE_DC_PYTHON_SUPPORT, "Dropping GIL by %s in %s:%d", _caller.c_str(), _file.c_str(), _line);
#ifdef _WIN32
	_state = PyEval_SaveThread();
#else
        _state = python_PyEval_SaveThread();
#endif // _WIN32
      }
    }
  
  DropPythonGIL::~DropPythonGIL()
  {
    if (_state) {
      snde_debug(SNDE_DC_PYTHON_SUPPORT, "Restoring GIL for %s in %s:%d", _caller.c_str(), _file.c_str(), _line);
#ifdef _WIN32
      PyEval_RestoreThread(_state);
#else
      python_PyEval_RestoreThread(_state);
#endif
    }
  }
#endif // SNDE_PYTHON_SUPPORT_ENABLED
};
