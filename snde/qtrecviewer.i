%{
#include "snde/qtrecviewer.hpp"
%}

// info on shiboken wrapping:


class QHboxLayout;
class QVBoxLayout;
class QLineEdit;

// https://lists.qt-project.org/pipermail/pyside/2012-September/000647.html
namespace snde {

  class qt_osg_compositor; // qt_osg_compositor.hpp
  class QTRecSelector; // qtrecviewer_support.hpp
  class qtrec_position_manager; // qtrecviewer_support.hpp
  class recdatabase; // recstore.hpp
  class QTRecViewer;


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

  // How all of this interfaces with signals and slots remains to be determined...
   



   // Output typemap for returning QObjects with a pyside wrapper instead of swig
%typemap(out) QObject *(PyObject *shibok2=nullptr,PyObject *shib2_wrapInstance=nullptr,PyObject *pyside2_qtcore=nullptr,PyObject *qobject=nullptr) {
  shibok2 = PyImport_ImportModule("shiboken2");
  if (!shibok2) SWIG_fail; // raise exception up 
  shib2_wrapInstance=PyObject_GetAttrString(shibok2,"wrapInstance");
  if (!shib2_wrapInstance) SWIG_fail; // raise exception up 

  
  pyside2_qtcore = PyImport_ImportModule("PySide2.QtCore");
  if (!pyside2_qtcore) SWIG_fail; // raise exception up 
  qobject=PyObject_GetAttrString(pyside2_qtcore,"QObject");
  if (!qobject) SWIG_fail; // raise exception up 
  
  //$result = PyTuple_New(2);
  //PyTuple_SetItem($result,0,PyObject_CallFunction(shib2_wrapInstance,(char *)"KO",(unsigned long long)((uintptr_t)($1)),qobject));
  //PyTuple_SetItem($result,1,
  //SWIG_NewPointerObj(SWIG_as_voidptr($1),$descriptor(QObject *),0));
  $result = PyObject_CallFunction(shib2_wrapInstance,(char *)"KO",(unsigned long long)((uintptr_t)($1)),qobject);
  

  
  Py_XDECREF(qobject);
  Py_XDECREF(pyside2_qtcore);
  Py_XDECREF(shib2_wrapInstance);
  Py_XDECREF(shibok2);
}



  // Output typemap for returning QWidgets with pyside wrapper instead of swig
%typemap(out) QWidget *(PyObject *shibok2=nullptr,PyObject *shib2_wrapInstance=nullptr,PyObject *pyside2_qtwidgets=nullptr,PyObject *qwidget=nullptr) {
  shibok2 = PyImport_ImportModule("shiboken2");
  if (!shibok2) SWIG_fail; // raise exception up 
  shib2_wrapInstance=PyObject_GetAttrString(shibok2,"wrapInstance");
  if (!shib2_wrapInstance) SWIG_fail; // raise exception up 

  
  pyside2_qtwidgets = PyImport_ImportModule("PySide2.QtWidgets");
  if (!pyside2_qtwidgets) SWIG_fail; // raise exception up 
  qwidget=PyObject_GetAttrString(pyside2_qtwidgets,"QWidget");
  if (!qwidget) SWIG_fail; // raise exception up 
  
  //$result = PyTuple_New(2);
  //PyTuple_SetItem($result,0,PyObject_CallFunction(shib2_wrapInstance,(char *)"KO",(unsigned long long)((uintptr_t)($1)),qobject));
  //PyTuple_SetItem($result,1,
  //SWIG_NewPointerObj(SWIG_as_voidptr($1),$descriptor(QObject *),0));
  $result = PyObject_CallFunction(shib2_wrapInstance,(char *)"KO",(unsigned long long)((uintptr_t)($1)),qwidget);
  

  
  Py_XDECREF(qwidget);
  Py_XDECREF(pyside2_qtwidgets);
  Py_XDECREF(shib2_wrapInstance);
  Py_XDECREF(shibok2);
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
%extend qwidget_subclass {
    QWidget *QWidget() {
      return self; 
    }
    
};

  snde_qobject_inheritor(qwidget_subclass); // also apply qobject characteristics to the qwidget 
%enddef 

 snde_qwidget_inheritor(QTRecViewer);

  // input typemap for QWidget
  %typemap(in) QWidget * (PyObject *shibok_two=nullptr,PyObject *shib2_getCppPointer=nullptr,PyObject *PointerTuple=nullptr,QWidget *SwigWidget=nullptr) {
    // already a swig pointer
    if (!SWIG_ConvertPtr($input,(void **)&SwigWidget,$descriptor(QWidget *),0)) {
      $1 = SwigWidget;
    } else {
      // try for a shiboken2 pyside pointer
      shibok_two = PyImport_ImportModule("shiboken2");
      if (!shibok_two) SWIG_fail; /* raise exception up */
      shib2_getCppPointer=PyObject_GetAttrString(shibok_two,"getCppPointer");
      
      PointerTuple = PyObject_CallFunction(shib2_getCppPointer,(char *)"O",$input);
      if (!PointerTuple) SWIG_fail;
      
      if (!PyTuple_Check(PointerTuple)) SWIG_fail;
      $1 = static_cast<QWidget *>(PyLong_AsVoidPtr(PyTuple_GetItem(PointerTuple,0)));
      if (PyErr_Occurred()) SWIG_fail;
    }

    Py_XDECREF(PointerTuple);
    Py_XDECREF(shib2_getCppPointer);
    Py_XDECREF(shibok_two);
    
  }

  
  class QTRecViewer: public QWidget {
    // Q_OBJECT
  public:
    QSharedPointer<qt_osg_compositor> OSGWidget; // OSGWidget is NOT parented; instead it is a QSharedPointer with the QObject::deleteLater deleter. This is so we can reference it from other threads, e.g. to pull out the pose e.g. for qt_osg_compositor_view_tracking_pose_recording. See https://stackoverflow.com/questions/12623690/qsharedpointer-and-qobjectdeletelater
    std::weak_ptr<recdatabase> recdb;
    std::shared_ptr<display_info> display;
    std::string selected; // name of selected channel

    
    std::unordered_map<std::string,QTRecSelector *> Selectors; // indexed by FullName
    
    QHBoxLayout *layout;
    QWidget *DesignerTree;
    QWidget *RecListScrollAreaContent;
    QVBoxLayout *RecListScrollAreaLayout;
    //   QSpacerItem *RecListScrollAreaBottomSpace;
    QLineEdit *ViewerStatus;
    qtrec_position_manager *posmgr; 

    //std::shared_ptr<std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<globalrevision>)>> ready_globalrev_quicknotify;
    
    
    QTRecViewer(std::shared_ptr<recdatabase> recdb,QWidget *parent);
    QTRecViewer(const QTRecViewer &) = delete;
    QTRecViewer & operator=(const QTRecViewer &) = delete;
    virtual ~QTRecViewer();
    
    std::shared_ptr<display_channel> FindDisplayChan(QTRecSelector *Selector);
    
    void set_selected(QTRecSelector *Selector);
    void deselect_other_selectors(QTRecSelector *Selected);
							  
    //public slots:
    void update_rec_list();
    
    void UpdateViewerStatus();
    void SelectorClicked(bool checked);
    void Darken(bool checked);
    void ResetIntensity(bool checked);
    void Brighten(bool checked);
    void LessContrast(bool checked);
    void MoreContrast(bool checked);

    //signals:
    //void NeedRedraw();    

  };
  
  

}
