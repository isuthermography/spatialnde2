//%shared_ptr(snde::memallocator);
//%shared_ptr(snde::cmemallocator);

%{
  
#include "pywrapper.hpp"
%}


namespace snde {
  class CountedPyObject {
  public:
    PyObject *_obj;
    inline CountedPyObject();

    inline CountedPyObject(PyObject *obj);

    inline CountedPyObject(const CountedPyObject &orig); 

    //inline CountedPyObject & operator=(const CountedPyObject &orig);

    inline PyObject *value();
    
    inline ~CountedPyObject();
  };
}

