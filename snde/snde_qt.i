  // ****!!!!!! IMPORTANT:
  // QT Ownership semantics
  // ----------------------
  // The QT convention is that widgets should be owned by their parent
  // e.g. a MainWindow, etc.
  // So here on contruction we require a parent QWidget for the wrapped
  // object (unlike in C++ we make the argument non-optional).
  // That QWidget (passed from PySide) then has ownership. The
  // new object (e.g. QTRecViewer) will survive so long as its parent lives.
  // if Python wrappers, such as the wrapped QTRecViewer, still exist
  // after that, they should not be used, as they point to freed memory.
  // (but it should be OK for them to go out of scope.

  // The wrapped QTRecViewer has .QWidget() and .QObject() Python methods
  // to obtain PySide - wrapped versions of it. Those should be used for
  // any base QObject and/or QWidget methods. Custom methods of QTRecViewer
  // should be SWIG-wrapped and callable directly.

  // QWidget objects must be called from the main thread.  Failure to do so
  // can result in unpredictable crashes.  The .QWidget() method is detected
  // by dataguzzler-python QtWrapper because only QWidget-derived spatialnde2
  // objects need to be Qt-wrapped to run in the main thread.

  // THEREFORE IT IS CRITICAL TO MARK ALL QWIDGET DERIVED CLASSES WITH THE
  // snde_qwidget_inheritor() MACRO DEFINED BELOW!

  // The output typemap for classes marked as snde_qobject_inheritor
  // prevents swig from taking ownership of the new object, so that QT can instead. 

  // How all of this interfaces with signals and slots remains to be determined...


  // general usage:
  // mark classes that inherit from QObject as:
  //   snde_qobject_inheritor(ClassName);
  // mark classes that inherit from QWidget as:
  //   snde_qwidget_inheritor(ClassName); // also implicitly performs snde_qobject_inheritor() magic





// General purpose SWIG stuff for QT

// Dummy QWidget for SWIG to be aware of so we can call setParent method
class QWidget {
  void setParent(QWidget *Parent);
};

class QOpenGLWidget: public QWidget {

};

class QOpenGLContext;
class QOffscreenSurface;




namespace snde {

   // Output typemap for returning QObjects with a pyside wrapper instead of swig
  %typemap(out) QObject *(PyObject *bindings_name=nullptr,PyObject *bindings=nullptr,bool using_pyside=true,PyObject *wrapInstance=nullptr,PyObject *qtcore=nullptr,PyObject *qobject=nullptr) {
    // Try Shiboken first to see if it is already loaded
#ifdef SNDE_ENABLE_QT5
  bindings_name=PyUnicode_FromString("shiboken2");
#endif
#ifdef SNDE_ENABLE_QT6
  bindings_name=PyUnicode_FromString("shiboken6");
#endif
  bindings = PyImport_GetModule(bindings_name);
  if (!bindings) {
    using_pyside=false;
    Py_DECREF(bindings_name);
    // Now try PyQt to see if it is already loaded
#ifdef SNDE_ENABLE_QT5
    bindings_name=PyUnicode_FromString("PyQt5.sip");
#endif
#ifdef SNDE_ENABLE_QT6
    bindings_name=PyUnicode_FromString("PyQt6.sip");
#endif
    bindings = PyImport_GetModule(bindings_name);
  }
  if (!bindings){
    PyErr_SetString(PyExc_ImportError,"Neither PySide nor PyQt bindings are loaded");
    SWIG_fail;// raise exception up
  }
  if (using_pyside){
    wrapInstance=PyObject_GetAttrString(bindings,"wrapInstance");
  } else {
    wrapInstance=PyObject_GetAttrString(bindings,"wrapinstance");
  }
  if (!wrapInstance) SWIG_fail; // raise exception up 

  if (using_pyside) {
#ifdef SNDE_ENABLE_QT5
    qtcore = PyImport_ImportModule("PySide2.QtCore");
#endif
#ifdef SNDE_ENABLE_QT6
    qtcore = PyImport_ImportModule("PySide6.QtCore");
#endif
  } else {
#ifdef SNDE_ENABLE_QT5
    qtcore = PyImport_ImportModule("PyQt5.QtCore");
#endif
#ifdef SNDE_ENABLE_QT6
    qtcore = PyImport_ImportModule("PyQt6.QtCore");
#endif
  }
  if (!qtcore) SWIG_fail; // raise exception up 
  qobject=PyObject_GetAttrString(qtcore,"QObject");
  if (!qobject) SWIG_fail; // raise exception up 
  
  //$result = PyTuple_New(2);
  //PyTuple_SetItem($result,0,PyObject_CallFunction(shib2_wrapInstance,(char *)"KO",(unsigned long long)((uintptr_t)($1)),qobject));
  //PyTuple_SetItem($result,1,
  //SWIG_NewPointerObj(SWIG_as_voidptr($1),$descriptor(QObject *),0));
  $result = PyObject_CallFunction(wrapInstance,(char *)"KO",(unsigned long long)((uintptr_t)($1)),qobject);
  

  
  Py_XDECREF(qobject);
  Py_XDECREF(qtcore);
  Py_XDECREF(wrapInstance);
  Py_XDECREF(bindings);
  Py_XDECREF(bindings_name);
}



  // Output typemap for returning QWidgets with pyside wrapper instead of swig
    %typemap(out) QWidget *(PyObject *bindings_name=nullptr,PyObject *bindings=nullptr,bool using_pyside=true,PyObject *wrapInstance=nullptr,PyObject *qtwidgets=nullptr,PyObject *qwidget=nullptr) {
    // Try Shiboken first to see if it is already loaded
#ifdef SNDE_ENABLE_QT5
  bindings_name = PyUnicode_FromString("shiboken2");
#endif
#ifdef SNDE_ENABLE_QT6
  bindings_name = PyUnicode_FromString("shiboken6");
#endif
  bindings = PyImport_GetModule(bindings_name);
  if (!bindings) {
    using_pyside=false;
    Py_DECREF(bindings_name);
    // Now try PyQt to see if it is already loaded
      #ifdef SNDE_ENABLE_QT5
    bindings_name=PyUnicode_FromString("PyQt5.sip");
#endif
#ifdef SNDE_ENABLE_QT6
    bindings_name=PyUnicode_FromString("PyQt6.sip");
#endif
    bindings = PyImport_GetModule(bindings_name);
  }
  if (!bindings){
    PyErr_SetString(PyExc_ImportError,"Neither PySide nor PyQt bindings are loaded");
    SWIG_fail;// raise exception up
  }
  if (using_pyside){
    wrapInstance=PyObject_GetAttrString(bindings,"wrapInstance");
  } else {
    wrapInstance=PyObject_GetAttrString(bindings,"wrapinstance");
  }
  if (!wrapInstance) SWIG_fail; // raise exception up
  if (using_pyside) {
#ifdef SNDE_ENABLE_QT5
    qtwidgets = PyImport_ImportModule("PySide2.QtWidgets");
#endif
#ifdef SNDE_ENABLE_QT6
    qtwidgets = PyImport_ImportModule("PySide6.QtWidgets");
#endif
  } else {
#ifdef SNDE_ENABLE_QT5
    qtwidgets = PyImport_ImportModule("PyQt5.QtWidgets");
#endif
#ifdef SNDE_ENABLE_QT6
    qtwidgets = PyImport_ImportModule("PyQt6.QtWidgets");
#endif
  }
  if (!qtwidgets) SWIG_fail; // raise exception up 
  qwidget=PyObject_GetAttrString(qtwidgets,"QWidget");
  if (!qwidget) SWIG_fail; // raise exception up 
  
  //$result = PyTuple_New(2);
  //PyTuple_SetItem($result,0,PyObject_CallFunction(shib2_wrapInstance,(char *)"KO",(unsigned long long)((uintptr_t)($1)),qobject));
  //PyTuple_SetItem($result,1,
  //SWIG_NewPointerObj(SWIG_as_voidptr($1),$descriptor(QObject *),0));
  $result = PyObject_CallFunction(wrapInstance,(char *)"KO",(unsigned long long)((uintptr_t)($1)),qwidget);
  

  
  Py_XDECREF(qwidget);
  Py_XDECREF(qtwidgets);
  Py_XDECREF(wrapInstance);
  Py_XDECREF(bindings);
  Py_XDECREF(bindings_name);
}


  

%define snde_qobject_inheritor(qobject_subclass)
   // make the constructor return a Python object that does NOT
   // own the underlying C++ object. This is so that the C++ object
   // can be owned by it's parent widget, following the QT convention
   // We do this with an output typemap
   
  // Output typemap for returning the object from a qobject subclass
  // constructor WITHOUT ownership (so that ownership can go to the
  // parent widget 
%typemap(out) qobject_subclass *qobject_subclass {
  
    $result = SWIG_NewPointerObj(SWIG_as_voidptr($1),$descriptor(qobject_subclass *),0); // it is the zero flags argument here that creates the wrapper without ownership
  
}
   

   // give the qobject subclass a .QObject() method that will get us the pyside-wrapped QObject
%extend qobject_subclass {
    QObject *QObject() {
      return self; 
    }    
};
  
%enddef 


  
%define snde_qwidget_inheritor(qwidget_subclass)
   // extension with QWidget() method that will get us the pyside-wrapped QObject
   // also SWIG wrapper doesn't own the object because our parent will (see snde_qobject_inheritor)
   // and we have a .QObject() method too.
   
%extend qwidget_subclass {
    QWidget *QWidget() {
      return self; 
    }
    
};

  snde_qobject_inheritor(qwidget_subclass); // also apply qobject characteristics to the qwidget. This also means the SWIG wrapper won't own the Python object so the QT parent can handle ownership
%enddef 

  // input typemap for QWidget
   %typemap(in) QWidget * (PyObject *bindings_in_name=nullptr,PyObject *bindings_in=nullptr,bool using_pyside_in=true,PyObject *getPointer=nullptr,PyObject *PointerResult=nullptr,QWidget *SwigWidget=nullptr) {
    // already a swig pointer
    if (!SWIG_ConvertPtr($input,(void **)&SwigWidget,$descriptor(QWidget *),0)) {
      $1 = SwigWidget;
    } else {
      // Try Shiboken first to see if it is already loaded
#ifdef SNDE_ENABLE_QT5
      bindings_in_name=PyUnicode_FromString("shiboken2");
#endif
#ifdef SNDE_ENABLE_QT6
      bindings_in_name=PyUnicode_FromString("shiboken6");
#endif
      bindings_in = PyImport_GetModule(bindings_in_name);
      if (!bindings_in) {
	using_pyside_in=false;
	Py_DECREF(bindings_in_name);
	// Now try PyQt to see if it is already loaded
#ifdef SNDE_ENABLE_QT5
	bindings_in_name=PyUnicode_FromString("PyQt5.sip");
#endif
#ifdef SNDE_ENABLE_QT6
	bindings_in_name=PyUnicode_FromString("PyQt6.sip");
#endif
	bindings_in = PyImport_GetModule(bindings_in_name);
      }
      if (!bindings_in){
	PyErr_SetString(PyExc_ImportError,"Neither PySide nor PyQt bindings are loaded");
	SWIG_fail;// raise exception up
      }

      if (using_pyside_in) {
	
	getPointer=PyObject_GetAttrString(bindings_in,"getCppPointer");
      } else {
	getPointer=PyObject_GetAttrString(bindings_in,"unwrapinstance");
      }
      
      PointerResult = PyObject_CallFunction(getPointer,(char *)"O",$input);
      if (!PointerResult) SWIG_fail;

      if (using_pyside_in) {
	// PySide getCppPointer() returns a tuple, the first element of which is the pointer as an integer
	if (!PyTuple_Check(PointerResult)) SWIG_fail;
	$1 = static_cast<QWidget *>(PyLong_AsVoidPtr(PyTuple_GetItem(PointerResult,0)));
      } else {
	$1 = static_cast<QWidget *>(PyLong_AsVoidPtr(PointerResult));
      }
      if (PyErr_Occurred()) SWIG_fail;
    }

    Py_XDECREF(PointerResult);
    Py_XDECREF(getPointer);
    Py_XDECREF(bindings_in);
    Py_XDECREF(bindings_in_name);
    
  }

}; // end namespace snde

