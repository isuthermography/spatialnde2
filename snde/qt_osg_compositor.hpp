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
    qt_osg_worker_thread *qt_worker_thread; // replaces osg_compositor worker_thread, owned by this object as parent
    // QTimer *AnimTimer;
    // osg::ref_ptr<osg_picker> picker;
    

    // Weak QObject references
    QPointer<QTRecViewer> Parent_QTRViewer; // weak pointer connection used for event forwarding; nullptr OK

    // non QObject-managed owned objects -- must be manually freed via destructor etc.
    // (these cannot be owned because they have to be moved to another thread) 
    QOpenGLContext *RenderContext; // used by the renderers within the compositor
    QOffscreenSurface *DummyOffscreenSurface; // offscreen surface; we never actually render this but it provdes an OpenGL context into which we can allocate our own framebuffers


    qt_osg_compositor(std::shared_ptr<recdatabase> recdb,
		      std::shared_ptr<display_info> display,
		      osg::ref_ptr<osgViewer::Viewer> Viewer,
		      bool threaded,bool enable_threaded_opengl,bool enable_shaders,QPointer<QTRecViewer> Parent_QTRViewer,QWidget *parent=nullptr);
    
    qt_osg_compositor(const qt_osg_compositor &) = delete;
    qt_osg_compositor & operator=(const qt_osg_compositor &) = delete;
    virtual ~qt_osg_compositor();
    
    virtual void trigger_rerender();
    virtual void initializeGL();

    virtual void worker_code();
    virtual void _start_worker_thread(std::unique_lock<std::mutex> *adminlock);
    virtual void _join_worker_thread();
    virtual void perform_ondemand_calcs(std::unique_lock<std::mutex> *adminlock);
    virtual void perform_layer_rendering(std::unique_lock<std::mutex> *adminlock);
    virtual void perform_compositing(std::unique_lock<std::mutex> *adminlock);
    virtual void wake_up_ondemand_locked(std::unique_lock<std::mutex> *adminlock);
    virtual void wake_up_renderer_locked(std::unique_lock<std::mutex> *adminlock);
    virtual void wake_up_compositor_locked(std::unique_lock<std::mutex> *adminlock);

    virtual void clean_up_renderer_locked(std::unique_lock<std::mutex> *adminlock);

    
    virtual void paintGL();
    virtual void resizeGL(int width,int height); 


    virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent(QMouseEvent *event);
    virtual void wheelEvent(QWheelEvent *event);
    virtual bool event(QEvent *event);

    //virtual void ClearPickedOrientation(); 
  
  public slots:
    void rerender();
    void update();

  };


  
  class qt_osg_compositor_view_tracking_pose_recording: public tracking_pose_recording {
  public:
    QWeakPointer<qt_osg_compositor> compositor;
    virtual snde_orientation3 get_channel_to_reorient_pose(std::shared_ptr<recording_set_state> rss) const;
    // channel to reorient is the channel that should appear fixed in this
    // view, with the same view as in the channel to reorient
    
    qt_osg_compositor_view_tracking_pose_recording(struct recording_params params,size_t info_structsize,std::string channel_to_reorient,std::string component_name,QSharedPointer<qt_osg_compositor> compositor);
    
  };
  


};


#endif // SNDE_QT_OSG_COMPOSITOR_HPP
