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
%typemap(out) QObject *(PyObject *shibok=nullptr,PyObject *shib_wrapInstance=nullptr,PyObject *pyside_qtcore=nullptr,PyObject *qobject=nullptr) {
#ifdef SNDE_ENABLE_QT5
  shibok = PyImport_ImportModule("shiboken2");
#endif
#ifdef SNDE_ENABLE_QT6
  shibok = PyImport_ImportModule("shiboken6");
#endif
  if (!shibok) SWIG_fail; // raise exception up 
  shib_wrapInstance=PyObject_GetAttrString(shibok,"wrapInstance");
  if (!shib_wrapInstance) SWIG_fail; // raise exception up 

#ifdef SNDE_ENABLE_QT5
  pyside_qtcore = PyImport_ImportModule("PySide2.QtCore");
#endif
#ifdef SNDE_ENABLE_QT6
  pyside_qtcore = PyImport_ImportModule("PySide6.QtCore");
#endif
  if (!pyside_qtcore) SWIG_fail; // raise exception up 
  qobject=PyObject_GetAttrString(pyside_qtcore,"QObject");
  if (!qobject) SWIG_fail; // raise exception up 
  
  //$result = PyTuple_New(2);
  //PyTuple_SetItem($result,0,PyObject_CallFunction(shib2_wrapInstance,(char *)"KO",(unsigned long long)((uintptr_t)($1)),qobject));
  //PyTuple_SetItem($result,1,
  //SWIG_NewPointerObj(SWIG_as_voidptr($1),$descriptor(QObject *),0));
  $result = PyObject_CallFunction(shib_wrapInstance,(char *)"KO",(unsigned long long)((uintptr_t)($1)),qobject);
  

  
  Py_XDECREF(qobject);
  Py_XDECREF(pyside_qtcore);
  Py_XDECREF(shib_wrapInstance);
  Py_XDECREF(shibok);
}



  // Output typemap for returning QWidgets with pyside wrapper instead of swig
%typemap(out) QWidget *(PyObject *shibok=nullptr,PyObject *shib_wrapInstance=nullptr,PyObject *pyside_qtwidgets=nullptr,PyObject *qwidget=nullptr) {
#ifdef SNDE_ENABLE_QT5
  shibok = PyImport_ImportModule("shiboken2");
#endif
#ifdef SNDE_ENABLE_QT6
  shibok = PyImport_ImportModule("shiboken6");
#endif
  if (!shibok) SWIG_fail; // raise exception up 
  shib_wrapInstance=PyObject_GetAttrString(shibok,"wrapInstance");
  if (!shib_wrapInstance) SWIG_fail; // raise exception up 

#ifdef SNDE_ENABLE_QT5
  pyside_qtwidgets = PyImport_ImportModule("PySide2.QtWidgets");
#endif
#ifdef SNDE_ENABLE_QT6
  pyside_qtwidgets = PyImport_ImportModule("PySide6.QtWidgets");
#endif
  if (!pyside_qtwidgets) SWIG_fail; // raise exception up 
  qwidget=PyObject_GetAttrString(pyside_qtwidgets,"QWidget");
  if (!qwidget) SWIG_fail; // raise exception up 
  
  //$result = PyTuple_New(2);
  //PyTuple_SetItem($result,0,PyObject_CallFunction(shib2_wrapInstance,(char *)"KO",(unsigned long long)((uintptr_t)($1)),qobject));
  //PyTuple_SetItem($result,1,
  //SWIG_NewPointerObj(SWIG_as_voidptr($1),$descriptor(QObject *),0));
  $result = PyObject_CallFunction(shib_wrapInstance,(char *)"KO",(unsigned long long)((uintptr_t)($1)),qwidget);
  

  
  Py_XDECREF(qwidget);
  Py_XDECREF(pyside_qtwidgets);
  Py_XDECREF(shib_wrapInstance);
  Py_XDECREF(shibok);
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
  %typemap(in) QWidget * (PyObject *shibok_two=nullptr,PyObject *shib_getCppPointer=nullptr,PyObject *PointerTuple=nullptr,QWidget *SwigWidget=nullptr) {
    // already a swig pointer
    if (!SWIG_ConvertPtr($input,(void **)&SwigWidget,$descriptor(QWidget *),0)) {
      $1 = SwigWidget;
    } else {
#ifdef SNDE_ENABLE_QT5
      // try for a shiboken2 pyside pointer
      shibok_two = PyImport_ImportModule("shiboken2");
#endif
#ifdef SNDE_ENABLE_QT6
      // try for a shiboken6 pyside pointer
      shibok_two = PyImport_ImportModule("shiboken6");
#endif
      if (!shibok_two) SWIG_fail; /* raise exception up */
      shib_getCppPointer=PyObject_GetAttrString(shibok_two,"getCppPointer");
      
      PointerTuple = PyObject_CallFunction(shib_getCppPointer,(char *)"O",$input);
      if (!PointerTuple) SWIG_fail;
      
      if (!PyTuple_Check(PointerTuple)) SWIG_fail;
      $1 = static_cast<QWidget *>(PyLong_AsVoidPtr(PyTuple_GetItem(PointerTuple,0)));
      if (PyErr_Occurred()) SWIG_fail;
    }

    Py_XDECREF(PointerTuple);
    Py_XDECREF(shib_getCppPointer);
    Py_XDECREF(shibok_two);
    
  }

}; // end namespace snde

