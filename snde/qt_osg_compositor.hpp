#ifndef SNDE_QT_OSG_COMPOSITOR_HPP
#define SNDE_QT_OSG_COMPOSITOR_HPP

#include <QOpenGLWidget>
#include <QOpenGLContext>
#include <QThread>
#include <QPointer>
#include <QOffscreenSurface>
#include <QOpenGLWidget>
#include <QOpenGLContext>

#include "snde/openscenegraph_compositor.hpp"

namespace snde {

  class qt_osg_compositor;
  class QTRecViewer; // qtrecviewer.hpp
  
  class qt_osg_worker_thread: public QThread {
    Q_OBJECT
  public:
    
    qt_osg_compositor *comp; 
    
    qt_osg_worker_thread(qt_osg_compositor *comp,QObject *parent);
    virtual ~qt_osg_worker_thread()=default;
    
    virtual void run();
    
    virtual void emit_need_update();
    
  signals:
    void compositor_need_update();
  };

  class qt_osg_compositor: public QOpenGLWidget, public osg_compositor  {
    Q_OBJECT
  public:

    // This class should be instantiated from the QT main loop thread
    // Since this inherits from a QT class it should use QT ownership
    // semantics: Provide a parent QT object to take ownership, and just
    // store the class pointer in that object. Other references should
    // be via QPointers, which act vaguely like std::weak_ptr.
    //
    // It should be instantiated from the QT main loop thread.

    // Child QObjects
    QOpenGLContext *RenderContext; // used by the renderers within the compositor
    qt_osg_worker_thread *qt_worker_thread; // replaces osg_compositor worker_thread, owned by this object as parent
    QOffscreenSurface *DummyOffscreenSurface; // offscreen surface; we never actually render this but it provdes an OpenGL context into which we can allocate our own framebuffers

    // Weak QObject references
    QPointer<QTRecViewer> Parent_Viewer; // weak pointer connection used for event forwarding; nullptr OK

    qt_osg_compositor(std::shared_ptr<recdatabase> recdb,
		      std::shared_ptr<display_info> display,
		      osg::ref_ptr<osgViewer::Viewer> Viewer,osg::ref_ptr<osgViewer::GraphicsWindow> GraphicsWindow,
		      bool threaded,bool enable_threaded_opengl,QPointer<QTRecViewer> Parent_Viewer,QWidget *parent=nullptr);
    
    qt_osg_compositor(const qt_osg_compositor &) = delete;
    qt_osg_compositor & operator=(const qt_osg_compositor &) = delete;
    ~qt_osg_compositor();
    
    virtual void initializeGL();

    virtual void worker_code();
    virtual void _start_worker_thread();
    virtual void _join_worker_thread();
    virtual void perform_ondemand_calcs();
    virtual void perform_layer_rendering();
    
    virtual void paintGL();

  
  public slots:
    void update();

  };
  

};


#endif // SNDE_QT_OSG_COMPOSITOR_HPP
