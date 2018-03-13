
namespace snde {
  

  class CountedPyObject {
  public:
    PyObject *_obj;
    inline CountedPyObject() : _obj(NULL) {}
    
    inline CountedPyObject(PyObject *obj) : _obj(obj)
    {
      Py_INCREF(obj);
    }

    inline CountedPyObject(const CountedPyObject &orig) 
    {
      _obj=orig._obj;
      if (_obj) {
	Py_INCREF(_obj);
      }
    }

    inline CountedPyObject & operator=(const CountedPyObject &orig)
    {
      if (_obj) {
	Py_DECREF(_obj);
      }
      _obj=orig._obj;
      if (_obj) {
	Py_INCREF(_obj);
      }
      return *this;
    }

    inline PyObject *value() {
      Py_INCREF(_obj); /* return a new reference */
      return _obj;
    }
    
    inline
    ~CountedPyObject() {
      if (_obj) {
	Py_DECREF(_obj);
      }
    }
  };

}
