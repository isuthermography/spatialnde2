#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>

#include "openscenegraph_renderer.hpp"

#ifndef SNDE_OPENSCENEGRAPH_PICKER_HPP
#define SNDE_OPENSCENEGRAPH_PICKER_HPP

namespace snde {
  
  class osg_picker : public osgGA::GUIEventHandler {
    // Implement point picker.
    // Adds self to View with View->addEventHandler(osg_picker)

  public:
    osg_renderer *renderer; // NOTE: Creator's responsibility
    // to ensure that renderer lasts at least as long as picker
    // (usually by making the picker an osg::ref_ptr within
    // the osg_renderer subclass 
    
    osg_picker(osg_renderer *renderer) :
		   renderer(renderer)
    {
      renderer->Viewer->addEventHandler(this);
      
    }
    
    virtual bool handle(const osgGA::GUIEventAdapter& ea,osgGA::GUIActionAdapter& aa, osg::Object*, osg::NodeVisitor*)
    {
      osgGA::GUIEventAdapter::EventType ev = ea.getEventType();

      osgViewer::View *view = dynamic_cast<osgViewer::View *>(&aa);

      if (view && renderer && ev==osgGA::GUIEventAdapter::PUSH) {
	// clicked a point
	if (ea.getButton()==1) {
	  // left (selector) mouse button
	  
	  osgUtil::LineSegmentIntersector::Intersections intersections;
	  
	  if (view->computeIntersections(ea,intersections)) {
	    for (auto & intersection: intersections) {
	      if (intersection.drawable.valid()) {
		auto geom_userdata = dynamic_cast<osg_instancecacheentry::geom_userdata *>(intersection.drawable->getUserData());
		if (geom_userdata) {
		  std::shared_ptr<osg_instancecacheentry> cacheentry = geom_userdata->cacheentry.lock();
		  if (cacheentry) {
		    osg::Vec3d coords = intersection.getLocalIntersectPoint();
		    //osg::Vec3d normal = intersection.getLocalIntersectNormal();
		    unsigned int trinum = intersection.primitiveIndex;
		    fprintf(stderr,"Got picked point: %f,%f,%f on triangle #%d\n",coords.x(),coords.y(),coords.z(),trinum);
		  }
		}
	      }
	      
	    }
	  }
	}
      }

      return false;
    }

  };



}

#endif // SNDE_OPENSCENEGRAPH_PICKER_HPP
