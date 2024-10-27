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
  // The following line is critical for all QWidget-derived classes.  See snde_qt.i for more info.
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
    
    //std::shared_ptr<display_channel> FindDisplayChan(std::string channame);
    //std::shared_ptr<display_channel> FindDisplayChan(QTRecSelector *Selector);
    
    void set_selected(QTRecSelector *Selector);
    void set_selected(std::string channame);
    void deselect_other_selectors(QTRecSelector *Selected);


    snde_orientation3 get_camera_pose(std::string channelpath);			    void set_camera_pose(std::string channelpath,const snde_orientation3 &newpose);

    snde_coord get_rotation_center_dist(std::string channelpath);
    void set_rotation_center_dist(std::string channelpath,snde_coord newcenterdist);
    float GetChannelContrast(std::string channelpath);
    void SetChannelContrast(std::string channelpath, float contrast);
    float GetChannelBrightness(std::string channelpath);
    void SetChannelBrightness(std::string channelpath, float brightness);
    void EnableChannel(std::string chanpath);
    void DisableChannel(std::string chanpath);
				  

    //public slots:
    void update_rec_list();
    
    void UpdateViewerStatus();
    void SelectorClicked(bool checked);
    void Darken(bool checked);
    void ResetIntensity(bool checked);
    void SetOffsetToMean(bool checked);
    void Brighten(bool checked);
    void NextFrame(bool checked);
    void PreviousFrame(bool checked);
    void LessContrast(bool checked);
    void MoreContrast(bool checked);

    //signals:
    //void NeedRedraw();    

  };
  
  

}
