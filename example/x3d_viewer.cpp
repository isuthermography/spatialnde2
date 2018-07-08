
#include <GL/glut.h>
#include <osgViewer/Viewer>
#include <osgGA/TrackballManipulator>

#include "x3d.hpp"

osg::ref_ptr<osgViewer::Viewer> viewer;

osg::observer_ptr<osgViewer::GraphicsWindow> window;

bool mousepressed=false;


void x3d_viewer_display()
{
  /* perform rendering into back buffer */
  if (viewer.valid()) viewer->frame();

  // swap front and back buffers
  glutSwapBuffers();

  /* Should we glutPostRedisplay() here or only if there is motion? */
}

void x3d_viewer_reshape(int width, int height)
{
  if (window.valid()) {
    window->resized(window->getTraits()->x,window->getTraits()->y,width,height);
    window->getEventQueue()->windowResize(window->GetTraits()->x,window->getTraits()->y,width,height);

  }
}

void x3d_viewer_mouse(int button, int state, int x, int y)
{
  if (window.valid()) {
    if (state==0) {
      window->getEventQueue()->mouseButtonPress(x,y,botton+1);
      mousepressed=true;
    } else {
      window->getEventQueue()->mouseButtonRelease(x,y,botton+1);
      mousepressed=false;
    }
  }
}

void x3d_viewer_motion(int x, int y)
{
  if (window.valid()) {
    window->getEventQueue()->mouseMotion(x,y);
    if (mousepressed) {
      glutPostRedisplay();
    }
  }
  
}

void x3d_viewer_kbd(unsigned char key, int x, int y)
{
  switch(key) {
  case 'q':
  case 'Q':
    if (viewer.valid()) viewer=0;
    glutDestroyWindow(glutGetWindow());
    break;


  default:
    if (window.valid()) {
      window->getEventQueue()->keyPress((osgGA::GuiEventAdapter::KeySymbol)key);
      window->getEventQueue()->keyRelease((osgGA::GuiEventAdapter::KeySymbol)key);
    } 
  }
}


int main(int argc, char **argv)
{
  cl_context context;
  cl_device_id device;
  std::string clmsgs;

  glutInit(&argc, argv);


  if (argc < 2) {
    fprintf(stderr,"USAGE: %s <x3d_file.x3d>\n", argv[0]);
    exit(1);
  }

  std::tie(context,device,clmsgs) = get_opencl_context("::",true,NULL,NULL);

  fprintf(stderr,"%s",clmsgs.c_str());


  

  std::shared_ptr<memallocator> lowlevel_alloc;
  std::shared_ptr<allocator_alignment> alignment_requirements;
  std::shared_ptr<arraymanager> manager;
  std::shared_ptr<geometry> geom;
  
  // lowlevel_alloc performs the actual host-side memory allocations
  lowlevel_alloc=std::make_shared<cmemallocator>();


  // alignment requirements specify constraints on allocation
  // block sizes
  alignment_requirements=std::make_shared<allocator_alignment>();
  // Each OpenCL device can impose an alignment requirement...
  add_opencl_alignment_requirement(alignment_requirements,device);
  
  // the arraymanager handles multiple arrays, including
  //   * Allocating space, reallocating when needed
  //   * Locking (implemented by manager.locker)
  //   * On-demand caching of array data to GPUs 
  manager=std::make_shared<arraymanager>(lowlevel_alloc,alignment_requirements);

  // geom is a C++ wrapper around a C data structure that
  // contains multiple arrays to be managed by the
  // arraymanager. These arrays are managed in
  // groups. All arrays in a group are presumed
  // to have parallel content, and are allocated,
  // freed, and locked in parallel.

  // Note that this initialization (adding arrays to
  // the arraymanager) is presumed to occur in a single-
  // threaded environment, whereas execution can be
  // freely done from multiple threads (with appropriate
  // locking of resources) 
  geom=std::make_shared<geometry>(1e-6,manager);
  


  std::shared_ptr<std::vector<std::shared_ptr<meshedpart>>> parts = x3d_load_geometry(geom,argv[1],false,false); // !!!*** Try enable vertex reindexing !!!***


  std::shared_ptr<assembly> assem = assembly::from_partlist("part_assembly",parts);
  std::unordered_map<std::string,paramdictentry> emptyparamdict;
  
  std::vector<snde_partinstance> instances = get_instances(snde_null_orientation3(),emptyparamdict)

  glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA|GLUT_DEPTH);
  glutInitWindowSize(1024,768);
  glutCreateWindow(argv[0]);

  glutDisplayFunc(&x3d_viewer_display);
  glutReshapeFunc(&x3d_viewer_reshape);

  glutMouseFunc(&x3d_viewer_mouse);
  glutMotionFunc(&x3d_viewer_motion);

  glutKeyboardFunc(&x3d_viewer_kbd);

  viewer = new osgViewer::Viewer;
  window = viewer->setUpViewerAsEmbeddedInWindow();
  viewer->setSceneData(!!!);
  viewer->setCameraManipulator(new osgGA::TrackballManipulator);

  viewer->realize();
  glutMainLoop();

  exit(0);

}
