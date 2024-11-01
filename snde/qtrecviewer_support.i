%{
#include "snde/qtrecviewer_support.hpp"
%}


namespace snde {

  class QTRecViewer; // forward declaration



    
  class QTRecSelector;

  class qtrec_position_manager;


  // see snde_qt.i for information on QObject/QWidget semantics, shiboken2/PySide2 interoperability, etc. 
  snde_qobject_inheritor(qtrec_position_manager); 

  
  class qtrec_position_manager: public QObject
  {
   //Q_OBJECT

  public:    
    std::shared_ptr<display_info> display;
    std::shared_ptr<display_channel> selected_channel;
    QTRecViewer *Parent_Viewer; // our parent
    
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
  
  //public slots:

    void HorizSliderActionTriggered(int action);

    void VertSliderActionTriggered(int action);
    void HorizZoomActionTriggered(int action);
    void VertZoomActionTriggered(int action);
    
    void VertZoomIn(bool);
    void VertZoomOut(bool);
    void HorizZoomIn(bool);
    void HorizZoomOut(bool);
    
    //signals:
    //void NewPosition();
    //void NewHorizSliderPosition(int value);
    //void NewVertSliderPosition(int value);
  };
    


}



