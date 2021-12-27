#ifndef SNDE_QTRECVIEWER_SUPPORT_HPP
#define SNDE_QTRECVIEWER_SUPPORT_HPP

#include <QFrame>
#include <QRadioButton>
#include <QAbstractSlider>
#include <QVBoxLayout>
#include <QEvent>

#include "snde/snde_types.h"
#include "snde/geometry_types.h"
#include "snde/colormap.h"
#include "snde/rec_display.hpp"

namespace snde {

  class QTRecViewer; // forward declaration



    
  class QTRecSelector : public QFrame {
  public:
    std::string Name;
    QTRecViewer *Viewer;
    QRadioButton *RadioButton;
    bool touched_during_update; // has this QTRecSelector been touched during the latest update pass
    QPalette basepalette;
    QPalette selectedpalette;
    bool selected;
    RecColor reccolor; 
    
    QTRecSelector(QTRecViewer *Viewer,std::string Name,RecColor reccolor,QWidget *parent=0);
    
    void setcolor(RecColor newcolor);      
    void setselected(bool newselected);

    
    bool eventFilter(QObject *object,QEvent *event); // in qtrecviewer.cpp
  };


  std::tuple<int,double> round_to_zoom_digit(double val); // Rounds an integer 0-10 to the nearest valid zoom digit (1,2, or 5);  returns (index,rounded) where index is 0, 1, or 2

  
  class qtrec_position_manager: public QObject
  {
    Q_OBJECT

  public:    
    std::shared_ptr<display_info> display;
    std::shared_ptr<display_channel> selected_channel;
    QTRecViewer *Parent_Viewer; // our parent
    QAbstractSlider *HorizSlider; // owned by our parent
    QAbstractSlider *VertSlider; // owned by our parent

    QAbstractSlider *HorizZoom; // owned by our parent
    //QToolButton *HorizZoomInButton; // owned by our parent
    //QToolButton *HorizZoomOutButton; // owned by our parent
    QAbstractSlider *VertZoom; // owned by our parent
    //QToolButton *VertZoomInButton; // owned by our parent
    //QToolButton *VertZoomOutButton; // owned by our parent
    
    double power; /* determines nature of curve mapping between slider position and motion */
    int nsteps;
    int nzoomsteps;
    // zoom values go e.g. for nzoomsteps=7, in REVERSE ORDER units per division of 
    // 1e-1, 2e-1, 5e-1, 1e0, 2e0, 5e0, 1e1

    
    qtrec_position_manager(std::shared_ptr<display_info> display,QAbstractSlider *HorizSlider,QAbstractSlider *VertSlider,QAbstractSlider *HorizZoom,
			   //QToolButton *HorizZoomInButton, QToolButton *HorizZoomOutButton,
			   QAbstractSlider *VertZoom,
			   //QToolButton *VertZoomInButton,QToolButton *VertZoomOutButton
			   QTRecViewer *Parent_Viewer,
			   QObject *Parent);
    ~qtrec_position_manager()=default;


    std::tuple<double,bool> GetHorizScale();
    std::tuple<double,bool> GetVertScale();
    void SetHorizScale(double horizscale,bool horizpixelflag);
    void SetVertScale(double vertscale,bool vertpixelflag);

    double GetScaleFromZoomPos(int zoompos,bool pixelflag);
    int GetZoomPosFromScale(double scale, bool pixelflag);
    
    std::tuple<double,double,double> GetHorizEdges();
    std::tuple<double,double,double> GetVertEdges();  
    
    void trigger();
    
    void set_selected(std::shared_ptr<display_channel> chan);
  
  public slots:

    void HorizSliderActionTriggered(int action);

    void VertSliderActionTriggered(int action);
    void HorizZoomActionTriggered(int action);
    void VertZoomActionTriggered(int action);
    
    void VertZoomIn(bool);
    void VertZoomOut(bool);
    void HorizZoomIn(bool);
    void HorizZoomOut(bool);
    
    signals:
    void NewPosition();
    //void NewHorizSliderPosition(int value);
    //void NewVertSliderPosition(int value);
  };
    


}

#endif // SNDE_QTRECVIEWER_SUPPORT_HPP
