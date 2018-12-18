#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osgViewer/Viewer>

#include "snde2_osg.hpp"
#include "revision_manager.hpp"


/* inop and abandoned... see x3d_viewer.cpp instead */

int main(int argc, char **argv)
{
  snde::trm revman(-1);
  osg_instancecache geomcache;
  
  
  // construct the viewer.
  osgViewer::Viewer viewer;
  osg::ref_ptr<snde::OSGComponent> OSGComp(new snde::OSGComponent(geom,geomcache,comp,revman));
  
  // add model to viewer.
  viewer.setSceneData( OSGComp );
  
  // Run the viewer
    return viewer.run();
}

