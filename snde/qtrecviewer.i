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


  // see snde_qt.i for information on QObject/QWidget semantics, shiboken2/PySide2 interoperability, etc. 
  snde_qwidget_inheritor(QTRecViewer); // also implicitly performs snde_qobject_inheritor() magic


  
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


    snde_orientation3 get_camera_pose(std::string channelpath);			    void set_camera_pose(std::string channelpath,const snde_orientation3 &newpose);

    snde_coord get_rotation_center_dist(std::string channelpath);
    void set_rotation_center_dist(std::string channelpath,snde_coord newcenterdist);

				  

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
