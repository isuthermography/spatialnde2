#ifndef SNDE_QTRECVIEWER_HPP
#define SNDE_QTRECVIEWER_HPP

#include <ios>
#include <cmath>

#include <QString>
#include <QWidget>
#include <QOpenGLWidget>
#include <QFrame>
#include <QToolButton>
#include <QPushButton>
#include <QAbstractSlider>
#include <QLineEdit>
#include <QSlider>
#include <QLayout>
#include <QtUiTools/QUiLoader>
#include <QTimer>

#include <osgViewer/Viewer>

#include "snde/rec_display.hpp"


namespace snde {

  class qt_osg_compositor; // qt_osg_compositor.hpp
  class QTRecSelector; // qtrecviewer_support.hpp
  class qtrec_position_manager; // qtrecviewer_support.hpp
  class recdatabase; // recstore.hpp
  
  // See https://vicrucann.github.io/tutorials/
  // See https://gist.github.com/vicrucann/874ec3c0a7ba4a814bd84756447bc798 "OpenSceneGraph + QOpenGLWidget - minimal example"
  // and http://forum.openscenegraph.org/viewtopic.php?t=16549 "QOpenGLWidget in osgQt"
  // and http://forum.openscenegraph.org/viewtopic.php?t=15097 "OSG 3.2.1 and Qt5 Widget integration"

    
  class QTRecViewer : public QWidget {
    Q_OBJECT
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

    std::shared_ptr<std::function<void(std::shared_ptr<recdatabase> recdb,std::shared_ptr<globalrevision>)>> ready_globalrev_quicknotify;
    
    
    QTRecViewer(std::shared_ptr<recdatabase> recdb,QWidget *parent=nullptr);

    QTRecViewer(const QTRecViewer &) = delete;
    QTRecViewer & operator=(const QTRecViewer &) = delete;
    virtual ~QTRecViewer();
    
    std::shared_ptr<display_channel> FindDisplayChan(QTRecSelector *Selector);
    
    void set_selected(QTRecSelector *Selector);
    void deselect_other_selectors(QTRecSelector *Selected);
							  
  public slots:
    void update_rec_list(); // Triggered from a hint from the recording database that there was a new globalrev
    
    void UpdateViewerStatus();
    void SelectorClicked(bool checked);
    void Darken(bool checked);
    void ResetIntensity(bool checked);
    void Brighten(bool checked);
    void LessContrast(bool checked);
    void MoreContrast(bool checked);
    void RotateColormap(bool checked);

  signals:
    void NeedRedraw();
    void NewGlobalRev(); // emitted after QtRecViewer updates its recording list upon detection of a globalrev update by update_rec_list() and once update_rec_list has updated the recording lists.

  };
  
  

  }
#endif // SNDE_QTRECVIEWER_HPP
