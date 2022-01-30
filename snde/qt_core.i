// General purpose SWIG stuff for QT

// Dummy QWidget for SWIG to be aware of so we can call setParent method
class QWidget {
  void setParent(QWidget *Parent);
};

class QOpenGLWidget: public QWidget {

};

class QOpenGLContext;
class QOffscreenSurface;
